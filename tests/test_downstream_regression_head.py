"""Tests del RegressionHead y RegressionDownstreamModel.

Sinteticos, no requieren shards reales. Validan:

- shape y dtype de la prediccion (`(B,)` por defecto, `(B, 1)` con
  `keep_last_dim=True`);
- backward sin NaN ni Inf;
- variantes lineal vs MLP de 2 capas (activacion + dropout opcionales);
- el wrapper acepta los inputs canonicos del contrato `(B, C, N, P)`
  con `valid_time_mask (B, W)` y `valid_patch_mask (B, C, N)`;
- channel-independence: el mismo modelo acepta C=1 y C=24;
- freeze_backbone activa congelacion de gradiente solo en backbone;
- compatibilidad con un sample sintetico estilo CMAPSS_RUL
  (C=24, N=32, P=16, W=512), incluyendo padding parcial.

Skip defensivo en Windows local: nuestra instalacion local de torch
tiene un ABI incompat con NumPy 2 al importar
`torch._subclasses.functional_tensor`. En Colab/Linux corre sin
problemas.
"""

from __future__ import annotations

import sys

import pytest


if sys.platform == "win32":
    pytest.skip(
        "Windows local: torch + numpy2 ABI incompat. Estos tests corren "
        "PASS en Colab/Linux.",
        allow_module_level=True,
    )


import torch

from models.patchtst_phm import PatchTSTPhm
from training.downstream.heads import (
    RegressionDownstreamModel,
    RegressionHead,
)


# Contrato CMAPSS RUL canonico:
N_PATCHES = 32
PATCH_SIZE = 16
W = N_PATCHES * PATCH_SIZE  # 512
C_DEFAULT = 24


def _inputs(B: int, C: int, W: int = W, N: int = N_PATCHES, P: int = PATCH_SIZE):
    """Construye un batch sintetico bien formado para el contrato del backbone."""
    x = torch.randn(B, C, N, P)
    vtm = torch.ones(B, W, dtype=torch.bool)
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    return x, vtm, vpm


def _tiny_backbone(d_model: int = 32):
    """PatchTSTPhm tiny para tests rapidos (1 layer, 2 heads, d_model=32)."""
    return PatchTSTPhm(
        patch_size=PATCH_SIZE,
        n_patches=N_PATCHES,
        d_model=d_model,
        n_layers=1,
        n_heads=2,
        d_ff=64,
        dropout=0.0,
    )


# ----------------------------------------------------------------------
# RegressionHead aislada
# ----------------------------------------------------------------------


def test_regression_head_shape_default():
    """Default (`keep_last_dim=False`): salida `(B,)`."""
    head = RegressionHead(d_model=64)
    x = torch.randn(8, 64)
    y = head(x)
    assert y.shape == (8,)
    assert y.dtype == torch.float32


def test_regression_head_shape_keep_last_dim():
    """`keep_last_dim=True`: salida `(B, 1)`."""
    head = RegressionHead(d_model=64, keep_last_dim=True)
    x = torch.randn(5, 64)
    y = head(x)
    assert y.shape == (5, 1)


def test_regression_head_mlp_dos_capas():
    """Con `hidden_dim`, usa MLP de 2 capas. Salida sigue siendo `(B,)`."""
    head = RegressionHead(
        d_model=32, hidden_dim=64, dropout=0.1, activation="gelu",
    )
    x = torch.randn(4, 32)
    y = head(x)
    assert y.shape == (4,)
    # Verifica que internamente se construyo el MLP (no la version lineal).
    assert head.is_mlp is True


def test_regression_head_lineal_simple():
    """Sin `hidden_dim`, usa la cabeza lineal minima."""
    head = RegressionHead(d_model=32)
    assert head.is_mlp is False
    x = torch.randn(4, 32)
    y = head(x)
    assert y.shape == (4,)


def test_regression_head_backward_no_nan():
    """Backward llena de gradientes finitos las dos cabezas (lineal y MLP)."""
    for hidden_dim in (None, 64):
        head = RegressionHead(
            d_model=32, hidden_dim=hidden_dim, activation="relu",
        )
        x = torch.randn(8, 32, requires_grad=True)
        y = head(x)
        # MSE contra un target arbitrario.
        target = torch.randn(8)
        loss = torch.nn.functional.mse_loss(y, target)
        assert torch.isfinite(loss)
        loss.backward()
        for p in head.parameters():
            assert p.grad is not None
            assert torch.isfinite(p.grad).all(), (
                f"grad no finito en {p.shape}, hidden_dim={hidden_dim}"
            )


def test_regression_head_acepta_activations_validas():
    """Las activaciones soportadas no lanzan."""
    for act in (None, "none", "relu", "gelu", "tanh"):
        head = RegressionHead(d_model=16, hidden_dim=32, activation=act)
        y = head(torch.randn(2, 16))
        assert y.shape == (2,)


def test_regression_head_rechaza_inputs_invalidos():
    """Validaciones del constructor."""
    with pytest.raises(ValueError, match="d_model"):
        RegressionHead(d_model=0)
    with pytest.raises(ValueError, match="dropout"):
        RegressionHead(d_model=16, dropout=1.0)
    with pytest.raises(ValueError, match="dropout"):
        RegressionHead(d_model=16, dropout=-0.1)
    with pytest.raises(ValueError, match="hidden_dim"):
        RegressionHead(d_model=16, hidden_dim=0)
    with pytest.raises(ValueError, match="activation"):
        RegressionHead(d_model=16, hidden_dim=32, activation="swish")


def test_regression_head_rechaza_forward_invalidos():
    """forward valida shape y d_model."""
    head = RegressionHead(d_model=16)
    with pytest.raises(ValueError, match="B, d_model"):
        head(torch.randn(4, 8, 16))  # 3D
    with pytest.raises(ValueError, match="d_model"):
        head(torch.randn(4, 32))  # dimension distinta a la configurada


# ----------------------------------------------------------------------
# RegressionDownstreamModel (wrapper end-to-end)
# ----------------------------------------------------------------------


def test_regression_wrapper_forward_C24_N32_P16():
    """Forward end-to-end con el contrato canonico CMAPSS RUL (C=24)."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    x, vtm, vpm = _inputs(B=2, C=C_DEFAULT)
    out = model(x, valid_time_mask=vtm, valid_patch_mask=vpm)
    assert set(out.keys()) >= {"prediction", "pooled", "tokens"}
    assert out["prediction"].shape == (2,)
    assert out["pooled"].shape == (2, 32)
    assert out["tokens"].shape == (2, C_DEFAULT, N_PATCHES, 32)
    assert torch.isfinite(out["prediction"]).all()


def test_regression_wrapper_channel_independence_C1_vs_C24():
    """El mismo modelo acepta cualquier C: contrato channel-independent."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    for C in (1, 5, 24, 48):
        x, vtm, vpm = _inputs(B=2, C=C)
        out = model(x, vtm, vpm)
        assert out["prediction"].shape == (2,)
        assert out["tokens"].shape == (2, C, N_PATCHES, 32)


def test_regression_wrapper_con_padding_parcial():
    """vtm con padding causal por la izquierda + vpm coherente con vtm.

    Simula una unidad CMAPSS corta (t_idx=200 < W=512), patron real del
    builder. Comprueba que la prediccion sigue siendo finita.
    """
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    B, C = 2, C_DEFAULT

    # 312 ciclos reales, 200 padded por la izquierda.
    n_real = 312
    vtm = torch.zeros(B, W, dtype=torch.bool)
    vtm[:, W - n_real:] = True
    # vpm: True donde al menos un timestep del patch es valido.
    vtm_n_p = vtm.reshape(B, N_PATCHES, PATCH_SIZE)
    patch_any_valid = vtm_n_p.any(dim=-1)         # (B, N)
    vpm = patch_any_valid.unsqueeze(1).expand(B, C, N_PATCHES).contiguous()

    x = torch.randn(B, C, N_PATCHES, PATCH_SIZE)
    out = model(x, valid_time_mask=vtm, valid_patch_mask=vpm)
    assert out["prediction"].shape == (B,)
    assert torch.isfinite(out["prediction"]).all()


def test_regression_wrapper_canales_constantes_mask():
    """`canales_constantes_mask` se respeta en el pooling."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    x, vtm, vpm = _inputs(B=2, C=C_DEFAULT)
    # Marcamos los primeros 3 canales como constantes en ambos samples.
    cmask = torch.zeros(2, C_DEFAULT, dtype=torch.bool)
    cmask[:, :3] = True
    out = model(x, valid_time_mask=vtm, valid_patch_mask=vpm,
                canales_constantes_mask=cmask)
    assert out["prediction"].shape == (2,)
    assert torch.isfinite(out["prediction"]).all()


def test_regression_wrapper_freeze_backbone_solo_cabeza_entrena():
    """Con freeze_backbone=True, solo la cabeza recibe gradiente."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone, freeze_backbone=True)
    # Sanity: requires_grad = False en todos los params del backbone.
    for p in model.backbone.parameters():
        assert p.requires_grad is False
    # Cabeza: requires_grad = True.
    for p in model.head.parameters():
        assert p.requires_grad is True

    x, vtm, vpm = _inputs(B=2, C=C_DEFAULT)
    target = torch.randn(2)
    out = model(x, vtm, vpm)
    loss = torch.nn.functional.mse_loss(out["prediction"], target)
    loss.backward()
    # Cabeza: tiene grads finitos.
    for p in model.head.parameters():
        assert p.grad is not None
        assert torch.isfinite(p.grad).all()
    # Backbone: ningun grad asignado (p.grad puede ser None).
    for p in model.backbone.parameters():
        assert p.grad is None or torch.all(p.grad == 0)


def test_regression_wrapper_unfrozen_backbone_recibe_grad():
    """Sin freeze, todos los parametros del backbone reciben grad."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone, freeze_backbone=False)
    x, vtm, vpm = _inputs(B=2, C=C_DEFAULT)
    target = torch.randn(2)
    loss = torch.nn.functional.mse_loss(
        model(x, vtm, vpm)["prediction"], target,
    )
    loss.backward()
    # Backbone debe haber recibido gradiente en al menos un parametro.
    n_finite_grads = 0
    for p in model.backbone.parameters():
        if p.grad is not None and torch.isfinite(p.grad).all():
            n_finite_grads += 1
    assert n_finite_grads > 0


def test_regression_wrapper_param_groups_linear_probing():
    """En freeze_backbone, devuelve un solo grupo (la cabeza)."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone, freeze_backbone=True)
    groups = model.trainable_parameter_groups(lr_head=1e-3)
    assert len(groups) == 1
    assert groups[0]["lr"] == 1e-3
    # Solo parametros de la cabeza.
    n_head = sum(p.numel() for p in model.head.parameters())
    n_grp = sum(p.numel() for p in groups[0]["params"])
    assert n_grp == n_head


def test_regression_wrapper_param_groups_two_lrs():
    """Sin freeze y con lr_backbone distinto a lr_head, dos grupos."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    groups = model.trainable_parameter_groups(lr_head=1e-3, lr_backbone=1e-5)
    assert len(groups) == 2
    lrs = sorted(g["lr"] for g in groups)
    assert lrs == [1e-5, 1e-3]


def test_regression_wrapper_param_groups_un_solo_lr():
    """Si lr_backbone es None o igual a lr_head, un solo grupo."""
    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    g1 = model.trainable_parameter_groups(lr_head=1e-3)
    assert len(g1) == 1
    g2 = model.trainable_parameter_groups(lr_head=1e-3, lr_backbone=1e-3)
    assert len(g2) == 1


def test_regression_wrapper_compatible_con_sample_tar_sintetico():
    """Smoke de compatibilidad con un sample estilo CMAPSS_RUL:

    - patches (C, N, P) = (24, 32, 16) float32
    - valid_time_mask (W,) = (512,) bool
    - valid_patch_mask (C, N) = (24, 32) bool

    Construye batch B=1 con None-broadcast (igual que haria el
    DataLoader) y lanza forward. Espera prediction (1,) finita.
    """
    import numpy as np
    C, N, P = C_DEFAULT, N_PATCHES, PATCH_SIZE
    # Simulacion del payload de un sample real.
    patches_np = np.random.randn(C, N, P).astype(np.float32)
    vtm_np = np.ones(N * P, dtype=bool)
    vtm_np[:200] = False  # 200 padded por la izquierda (200 < W)
    vpm_np = vtm_np.reshape(N, P).any(axis=1)  # (N,)
    vpm_np = np.broadcast_to(vpm_np, (C, N)).copy()  # (C, N)

    backbone = _tiny_backbone(d_model=32)
    model = RegressionDownstreamModel(backbone=backbone)
    x = torch.from_numpy(patches_np[None, ...])
    vtm = torch.from_numpy(vtm_np[None, ...])
    vpm = torch.from_numpy(vpm_np[None, ...])
    out = model(x, valid_time_mask=vtm, valid_patch_mask=vpm)
    assert out["prediction"].shape == (1,)
    assert torch.isfinite(out["prediction"]).all()
