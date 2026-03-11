"""Tests de shape y channel-independence del backbone PatchTSTPhm.

El objetivo es blindar el contrato channel-independent:

- La misma instancia del modelo acepta C=1, C=24 y C=317 en llamadas
  distintas, sin reinicializar parametros.
- `reconstruction.shape == (B, C, N, P)`.
- `tokens.shape == (B, C, N, d_model)`.
- Ninguna capa inicial depende de C.
- `src_key_padding_mask` no marca como invalidos patches enmascarados.
"""

from __future__ import annotations

import pytest
import torch

from models.patchtst_phm import PatchTSTPhm


# Constantes del contrato MVP
P = 16
N = 32
W = N * P  # 512


def _build_inputs(B: int, C: int, fill_valid: bool = True):
    x = torch.randn(B, C, N, P)
    if fill_valid:
        vtm = torch.ones(B, W, dtype=torch.bool)
        vpm = torch.ones(B, C, N, dtype=torch.bool)
    else:
        vtm = torch.zeros(B, W, dtype=torch.bool)
        vpm = torch.zeros(B, C, N, dtype=torch.bool)
    return x, vtm, vpm


def _model_tiny():
    return PatchTSTPhm.tiny()


# ----------------------------------------------------------------------
# 1. Cada C valido funciona
# ----------------------------------------------------------------------


@pytest.mark.parametrize("C", [1, 24, 317])
def test_channel_independence_single_c(C):
    model = _model_tiny()
    B = 2
    x, vtm, vpm = _build_inputs(B, C)
    out = model(x, vtm, vpm)
    assert out["reconstruction"].shape == (B, C, N, P)
    assert out["tokens"].shape == (B, C, N, model.d_model)
    assert out["pooled"].shape == (B, C, model.d_model)


# ----------------------------------------------------------------------
# 2. La misma instancia acepta distintos C en llamadas diferentes
# ----------------------------------------------------------------------


def test_same_instance_accepts_different_c():
    model = _model_tiny()
    # Snapshot de parametros antes
    n_params_before = sum(p.numel() for p in model.parameters())
    for C in (1, 7, 24, 317):
        x, vtm, vpm = _build_inputs(2, C)
        out = model(x, vtm, vpm)
        assert out["reconstruction"].shape == (2, C, N, P)
    n_params_after = sum(p.numel() for p in model.parameters())
    assert n_params_before == n_params_after, (
        "El numero de parametros cambio entre llamadas: alguna capa depende de C"
    )


# ----------------------------------------------------------------------
# 3. La capa inicial no depende de C (Linear(P, d_model))
# ----------------------------------------------------------------------


def test_initial_layer_independent_of_c():
    model = _model_tiny()
    # patch_embedding: Linear(P, d_model). Su entrada es P, no P*C.
    assert model.patch_embedding.in_features == P
    assert model.patch_embedding.out_features == model.d_model


# ----------------------------------------------------------------------
# 4. Con ssl_mask, los patches enmascarados-pero-validos SI participan en
#    atencion (no entran en key_padding_mask)
# ----------------------------------------------------------------------


def test_ssl_mask_no_excludes_from_attention():
    """Patches enmascarados deben seguir formando parte del flujo.

    Verificacion indirecta: si forzamos un patch entero como mask-pero-valido,
    su token de salida no es exactamente igual al de un patch invalido
    (que si estaria en padding).
    """
    model = _model_tiny()
    model.eval()
    B, C = 1, 2
    x, vtm, vpm = _build_inputs(B, C)
    ssl_mask = torch.zeros(B, C, N, dtype=torch.bool)
    ssl_mask[:, :, 5] = True  # patch 5 enmascarado pero valido

    out = model(x, vtm, vpm, ssl_mask=ssl_mask)
    tok_masked = out["tokens"][:, :, 5, :]

    # Caso de referencia: el mismo patch marcado como invalido (key padding).
    vpm_bad = vpm.clone()
    vpm_bad[:, :, 5] = False
    out_invalid = model(x, vtm, vpm_bad, ssl_mask=ssl_mask)
    tok_invalid = out_invalid["tokens"][:, :, 5, :]

    # Si los dos casos produjeran el mismo token, querria decir que el patch
    # masked-pero-valido no contribuye a la atencion. Verificamos que
    # difieren significativamente.
    diff = (tok_masked - tok_invalid).abs().sum().item()
    assert diff > 1e-3, f"diff={diff}: el masked-valid se comporta como invalido"


# ----------------------------------------------------------------------
# 5. Ninguna fila con todos los patches invalidos rompe el forward
# ----------------------------------------------------------------------


def test_row_all_invalid_does_not_break_forward():
    """El edge case 'todos los patches invalidos' no debe producir NaN."""
    model = _model_tiny()
    model.eval()
    B, C = 2, 3
    x, vtm, vpm = _build_inputs(B, C)
    vpm[0, 0, :] = False  # fila (0,0) sin ningun patch valido
    out = model(x, vtm, vpm)
    assert not torch.isnan(out["reconstruction"]).any()
    assert not torch.isnan(out["tokens"]).any()
    # Pooled de la fila degenerada debe ser exactamente cero
    assert torch.allclose(out["pooled"][0, 0], torch.zeros_like(out["pooled"][0, 0]))


# ----------------------------------------------------------------------
# 6. Backward funciona y los gradientes llegan al patch_embedding y mask_token
# ----------------------------------------------------------------------


def test_backward_reaches_patch_embedding_and_mask_token():
    model = _model_tiny()
    B, C = 2, 4
    x, vtm, vpm = _build_inputs(B, C)
    ssl_mask = torch.zeros(B, C, N, dtype=torch.bool)
    ssl_mask[:, :, ::4] = True  # cada 4 patches

    out = model(x, vtm, vpm, ssl_mask=ssl_mask)
    loss = out["reconstruction"].mean()
    loss.backward()

    assert model.patch_embedding.weight.grad is not None
    assert model.patch_embedding.weight.grad.abs().sum().item() > 0
    assert model.mask_token.grad is not None
    assert model.mask_token.grad.abs().sum().item() > 0


# ----------------------------------------------------------------------
# 7. ssl_mask=None: el modelo funciona en modo "sin masking"
# ----------------------------------------------------------------------


def test_no_ssl_mask_inference_mode():
    model = _model_tiny()
    model.eval()
    B, C = 1, 5
    x, vtm, vpm = _build_inputs(B, C)
    out = model(x, vtm, vpm, ssl_mask=None)
    assert out["reconstruction"].shape == (B, C, N, P)


# ----------------------------------------------------------------------
# 8. Build desde dict de config (YAML)
# ----------------------------------------------------------------------


def test_build_from_config():
    from models.patchtst_phm import build_patchtst_phm
    cfg = {
        "patch_size": 16, "n_patches": 32, "d_model": 64,
        "n_layers": 2, "n_heads": 4, "d_ff": 256, "dropout": 0.1,
    }
    m = build_patchtst_phm(cfg)
    assert isinstance(m, PatchTSTPhm)
    assert m.d_model == 64
