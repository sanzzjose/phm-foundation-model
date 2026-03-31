"""Test sintetico del flujo de diagnostico determinista de padding parcial.

`_run_partial_padding_diagnostic` opera sobre shards reales de Drive y no
se puede ejercitar en CI sin esos shards. Aqui aislamos su nucleo logico:

1. Construir una ventana sintetica con padding al final.
2. Derivar `partial_patch_mask = vsm.any(-1) & ~vsm.all(-1)`.
3. Forzar `ssl_mask` exactamente sobre esos patches parciales.
4. Pasar por el modelo `PatchTSTPhm.tiny()`.
5. Verificar que la loss reportada por
   `compute_masked_reconstruction_loss_with_metrics` cumple:
     - `loss` finita;
     - `n_loss_elements > 0`;
     - `padding_ignored_elements > 0`.

Estos asserts garantizan que el contrato del modelo + loss + masking
funciona end-to-end sin necesidad de shards reales.
"""

from __future__ import annotations

import torch

from models.patchtst_phm import PatchTSTPhm
from training.ssl.loss import compute_masked_reconstruction_loss_with_metrics
from training.ssl.masking import canonicalize_valid_patch_mask


# Constantes del contrato MVP
N = 32
P = 16
W = N * P  # 512


def _make_partial_window(C: int, valid_len: int):
    """Crea (x, vtm, vpm) sinteticos con padding fino al final.

    Mas concretamente:
      - `valid_len`: numero de timesteps reales (resto = padding).
      - Si valid_len no es multiplo de P, el ultimo patch valido es parcial.
    """
    x = torch.randn(1, C, N, P)
    vtm = torch.zeros(1, W, dtype=torch.bool)
    vtm[0, :valid_len] = True
    # valid_patch_mask: True si el patch tiene al menos un timestep real
    vsm = vtm.reshape(1, N, P)
    vpm_per_patch = vsm.any(dim=-1)  # (1, N)
    vpm = vpm_per_patch.unsqueeze(1).expand(1, C, N).contiguous()
    return x, vtm, vpm


def test_partial_padding_end_to_end():
    """Flujo end-to-end con padding parcial: loss finita y pad ignorado > 0."""
    torch.manual_seed(0)
    C = 4
    # valid_len = 500 → ultimos 12 timesteps son padding. El ultimo patch
    # (timesteps 496..511) tiene 4 reales y 12 padding → patch parcial.
    x, vtm, vpm = _make_partial_window(C, valid_len=500)
    B, _, _, _ = x.shape
    vpm_canon = canonicalize_valid_patch_mask(vpm, B, C, N)

    # partial_patch_mask: patches con algun real Y algun padding
    vsm = vtm.reshape(B, N, P)
    partial = vsm.any(dim=-1) & ~vsm.all(dim=-1)  # (B, N)
    assert partial.any().item(), "No hay patches parciales en la ventana sintetica"

    # ssl_mask forzado a True exactamente sobre los patches parciales,
    # repetido para todos los canales (channel-independent).
    ssl_mask = partial.unsqueeze(1).expand(B, C, N).contiguous() & vpm_canon
    assert ssl_mask.any().item(), "ssl_mask vacio: no se enmascaro ningun patch parcial"

    model = PatchTSTPhm.tiny()
    model.eval()
    with torch.no_grad():
        out = model(x, vtm, vpm_canon, ssl_mask)
        metrics = compute_masked_reconstruction_loss_with_metrics(
            pred=out["reconstruction"],
            target=x,
            ssl_mask=ssl_mask,
            valid_time_mask=vtm,
            valid_patch_mask=vpm_canon,
            loss_fn="mse",
        )

    loss = metrics["loss"]
    assert torch.isfinite(loss).item(), f"loss no finita: {loss}"
    n_loss = int(metrics["n_loss_elements"].item())
    assert n_loss > 0, f"n_loss_elements={n_loss}: no contribuyo ningun timestep"
    pad_ig = int(metrics["padding_ignored_elements"].item())
    assert pad_ig > 0, (
        f"padding_ignored_elements={pad_ig}: la mascara fina no ignoro padding"
    )


def test_fully_valid_window_no_padding_ignored():
    """Caso de control: si la ventana no tiene padding, padding_ignored == 0."""
    torch.manual_seed(1)
    C = 4
    x, vtm, vpm = _make_partial_window(C, valid_len=W)  # todo real
    B, _, _, _ = x.shape
    vpm_canon = canonicalize_valid_patch_mask(vpm, B, C, N)
    # Enmascarar el ultimo patch (totalmente real)
    ssl_mask = torch.zeros(B, C, N, dtype=torch.bool)
    ssl_mask[:, :, -1] = True

    model = PatchTSTPhm.tiny()
    model.eval()
    with torch.no_grad():
        out = model(x, vtm, vpm_canon, ssl_mask)
        metrics = compute_masked_reconstruction_loss_with_metrics(
            pred=out["reconstruction"],
            target=x,
            ssl_mask=ssl_mask,
            valid_time_mask=vtm,
            valid_patch_mask=vpm_canon,
        )
    assert int(metrics["padding_ignored_elements"].item()) == 0
    assert torch.isfinite(metrics["loss"]).item()
