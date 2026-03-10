"""Tests unitarios de `training.ssl.loss.masked_reconstruction_loss`.

Estos tests blindan el comportamiento de la loss SSL en presencia de patches
parcialmente validos. Son obligatorios antes de lanzar pretraining (sec 17 de
`CLAUDE.md`) porque con `tail_policy='pad'` global, los patches parciales
son sistematicos: cualquier trayectoria con `T > W` y resto > 0 produce una
ventana parcial cuyo ultimo patch valido es mitad real / mitad padding. Sin
la mascara fina a nivel de timestep, la cabeza SSL aprenderia a "predecir"
padding como si fuese senal real.

Convenciones de los tests:

- Trabajamos con un mini-contrato `(B, C, N, P)` reducido (`C=2, N=4, P=4`
  → W=16) para que las maticulas sean inspeccionables a mano.
- Las predicciones se construyen como copias del target mas perturbaciones
  controladas en regiones especificas (real, padding, masked, no masked) para
  comprobar que solo las regiones (masked AND real) contribuyen.
- Todos los tests son CPU, deterministicos (`torch.manual_seed(0)`).

Ejecucion:

    pytest tests/test_ssl_loss_padding.py -v
"""

from __future__ import annotations

import math

import pytest
import torch

from training.ssl.loss import (
    derivar_valid_sample_mask,
    masked_reconstruction_loss,
)


# Mini-contrato (B, C, N, P) para inspeccionabilidad
B, C, N, P = 1, 2, 4, 4
W = N * P  # = 16


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def _zeros_pred_target():
    """Devuelve pred=target=zeros con la forma canonica."""
    return torch.zeros(B, C, N, P), torch.zeros(B, C, N, P)


def _all_valid_time_mask():
    """Ventana completa (sin padding)."""
    return torch.ones(B, W, dtype=torch.bool)


def _partial_valid_time_mask(valid_len: int):
    """Ventana parcial: solo los primeros `valid_len` timesteps son reales."""
    m = torch.zeros(B, W, dtype=torch.bool)
    m[:, :valid_len] = True
    return m


def _ssl_mask_only_last_patch():
    """Enmascara solo el ultimo patch (N-1=3) en todos los canales."""
    m = torch.zeros(B, C, N, dtype=torch.bool)
    m[:, :, N - 1] = True
    return m


def _ssl_mask_all_patches():
    """Enmascara todos los patches en todos los canales."""
    return torch.ones(B, C, N, dtype=torch.bool)


# ----------------------------------------------------------------------
# Test 1: Sanity check sobre ventana completa sin padding
# ----------------------------------------------------------------------


def test_loss_zero_si_pred_igual_target_y_ventana_completa():
    """Caso trivial: prediccion perfecta → loss = 0."""
    pred, target = _zeros_pred_target()
    ssl_mask = _ssl_mask_all_patches()
    valid_time = _all_valid_time_mask()
    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


def test_loss_mse_estandar_si_no_hay_padding():
    """Sin padding, sin ssl_mask selectivo, la loss debe coincidir con MSE."""
    torch.manual_seed(0)
    pred = torch.randn(B, C, N, P)
    target = torch.randn(B, C, N, P)
    ssl_mask = _ssl_mask_all_patches()
    valid_time = _all_valid_time_mask()
    loss_ours = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    loss_ref = ((pred - target).pow(2)).mean()
    assert loss_ours.item() == pytest.approx(loss_ref.item(), rel=1e-6)


# ----------------------------------------------------------------------
# Test 2: Patch parcial — el caso critico
# ----------------------------------------------------------------------


def test_padding_dentro_del_ultimo_patch_no_contribuye():
    """Patch parcial: solo los timesteps reales del ultimo patch contribuyen.

    Setup: ventana de W=16 con valid_len=14 → ultimo patch (timesteps 12..15)
    tiene 2 reales (12, 13) y 2 padding (14, 15). Si todos los patches estan
    enmascarados, la loss debe contar 3 patches completos (4 timesteps cada
    uno) + 2 timesteps del cuarto = 14 elementos contribuyentes por canal.
    """
    pred = torch.zeros(B, C, N, P)
    target = torch.zeros(B, C, N, P)

    # Perturbamos los timesteps de PADDING del ultimo patch: indices p=2, p=3.
    # Si la mascara funciona, esto NO debe afectar a la loss.
    pred[:, :, N - 1, 2] = 999.0
    pred[:, :, N - 1, 3] = -999.0

    valid_time = _partial_valid_time_mask(valid_len=14)
    ssl_mask = _ssl_mask_all_patches()

    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    # Esperado: 0. El padding no contribuye.
    assert loss.item() == pytest.approx(0.0, abs=1e-7), (
        f"La loss capturo perturbaciones de padding: loss={loss.item()}"
    )


def test_real_dentro_del_ultimo_patch_si_contribuye():
    """Simetrico al anterior: perturbar los timesteps REALES del ultimo patch
    si debe disparar la loss."""
    pred = torch.zeros(B, C, N, P)
    target = torch.zeros(B, C, N, P)

    # Perturbamos los timesteps REALES del ultimo patch: indices p=0, p=1
    # (timesteps absolutos 12 y 13). Padding son p=2, p=3.
    pred[:, :, N - 1, 0] = 2.0
    pred[:, :, N - 1, 1] = -2.0

    valid_time = _partial_valid_time_mask(valid_len=14)
    ssl_mask = _ssl_mask_all_patches()

    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)

    # Esperado: suma de errores cuadraticos / numero de contribuyentes.
    # Contribuyen: 3 patches completos (12 timesteps) + 2 reales del ultimo = 14.
    # Por canal: 14. Por batch: 14 * C * B = 28 timesteps.
    # Errores: solo los 4 perturbados (2 canales x 2 timesteps).
    # Cada uno tiene error |2|^2 = 4. Total errores no nulos: 4 * 4 = 16.
    # Loss = 16 / (14 * C * B) = 16 / 28 ≈ 0.5714
    esperado = (4 * (2.0**2)) / (14 * C * B)
    assert loss.item() == pytest.approx(esperado, rel=1e-5), (
        f"loss={loss.item()}, esperado={esperado}"
    )


# ----------------------------------------------------------------------
# Test 3: ssl_mask selectivo — patches no enmascarados no deben contribuir
# ----------------------------------------------------------------------


def test_patches_no_enmascarados_no_contribuyen():
    """Solo los patches en ssl_mask contribuyen, aunque haya error en otros."""
    pred = torch.zeros(B, C, N, P)
    target = torch.zeros(B, C, N, P)

    # Error en patches NO enmascarados (n=0, n=1, n=2) — debe ignorarse.
    pred[:, :, 0, :] = 5.0
    pred[:, :, 1, :] = 5.0
    pred[:, :, 2, :] = 5.0
    # Y en el patch enmascarado (n=3), pred = target → contribucion 0.

    valid_time = _all_valid_time_mask()
    ssl_mask = _ssl_mask_only_last_patch()

    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    assert loss.item() == pytest.approx(0.0, abs=1e-7), (
        f"La loss capturo error en patches no enmascarados: loss={loss.item()}"
    )


# ----------------------------------------------------------------------
# Test 4: Reduction='sum' y reduction='none'
# ----------------------------------------------------------------------


def test_reduction_sum_coincide_con_suma_de_errores_validos():
    """Suma absoluta de errores enmascarados-y-validos."""
    pred = torch.zeros(B, C, N, P)
    target = torch.zeros(B, C, N, P)
    pred[:, 0, N - 1, 0] = 3.0  # canal 0, ultimo patch, primer timestep
    pred[:, 1, N - 1, 1] = -3.0  # canal 1, ultimo patch, segundo timestep

    valid_time = _partial_valid_time_mask(valid_len=14)  # reales: 0..13
    ssl_mask = _ssl_mask_all_patches()

    loss_sum = masked_reconstruction_loss(
        pred, target, ssl_mask, valid_time, reduction="sum"
    )
    # Esperado: dos errores |3|^2 = 9 cada uno, total 18.
    assert loss_sum.item() == pytest.approx(18.0, rel=1e-6)


def test_reduction_none_devuelve_tensor_elementwise_enmascarado():
    """reduction='none' devuelve el tensor de errores ya enmascarado."""
    pred = torch.ones(B, C, N, P)
    target = torch.zeros(B, C, N, P)

    valid_time = _partial_valid_time_mask(valid_len=14)
    ssl_mask = _ssl_mask_only_last_patch()

    out = masked_reconstruction_loss(
        pred, target, ssl_mask, valid_time, reduction="none"
    )

    assert out.shape == (B, C, N, P)
    # Patches 0..2 (no enmascarados) deben estar a 0
    assert out[:, :, :3, :].sum().item() == 0.0
    # Patch 3 (enmascarado): timesteps 0,1 son reales (error = 1^2 = 1),
    # timesteps 2,3 son padding (error = 0 por mascara).
    assert out[:, :, 3, 0].sum().item() == pytest.approx(1.0 * B * C)
    assert out[:, :, 3, 1].sum().item() == pytest.approx(1.0 * B * C)
    assert out[:, :, 3, 2].sum().item() == 0.0
    assert out[:, :, 3, 3].sum().item() == 0.0


# ----------------------------------------------------------------------
# Test 5: loss_fn = 'mae'
# ----------------------------------------------------------------------


def test_mae_coincide_con_l1_estandar_en_ventana_completa():
    torch.manual_seed(1)
    pred = torch.randn(B, C, N, P)
    target = torch.randn(B, C, N, P)
    valid_time = _all_valid_time_mask()
    ssl_mask = _ssl_mask_all_patches()
    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time, loss_fn="mae")
    ref = (pred - target).abs().mean()
    assert loss.item() == pytest.approx(ref.item(), rel=1e-6)


# ----------------------------------------------------------------------
# Test 6: Caso degenerado — ningun timestep contribuye
# ----------------------------------------------------------------------


def test_devuelve_cero_sin_nan_si_no_hay_contribuyentes():
    """Si ssl_mask es todo False o valid_time_mask es todo False, no debe nan."""
    pred = torch.randn(B, C, N, P)
    target = torch.randn(B, C, N, P)

    # Caso A: ssl_mask vacio
    ssl_vacio = torch.zeros(B, C, N, dtype=torch.bool)
    valid_time = _all_valid_time_mask()
    loss_a = masked_reconstruction_loss(pred, target, ssl_vacio, valid_time)
    assert not math.isnan(loss_a.item())
    assert loss_a.item() == pytest.approx(0.0, abs=1e-7)

    # Caso B: valid_time_mask vacio
    ssl_full = _ssl_mask_all_patches()
    valid_vacio = torch.zeros(B, W, dtype=torch.bool)
    loss_b = masked_reconstruction_loss(pred, target, ssl_full, valid_vacio)
    assert not math.isnan(loss_b.item())
    assert loss_b.item() == pytest.approx(0.0, abs=1e-7)


def test_patch_totalmente_padding_aunque_enmascarado_no_contribuye():
    """Caso patologico: alguien enmascara un patch totalmente padding.

    El protocolo normal (ssl_mask es subset de valid_patch_mask) no deberia
    permitir esto, pero la funcion debe ser robusta: contribucion = 0.
    """
    pred = torch.zeros(B, C, N, P)
    target = torch.zeros(B, C, N, P)
    pred[:, :, N - 1, :] = 999.0  # patch totalmente padding, lleno de basura

    # Patch n=3 totalmente padding (timesteps 12..15 son padding).
    valid_time = _partial_valid_time_mask(valid_len=12)
    ssl_mask = _ssl_mask_only_last_patch()  # erroneamente enmascarado

    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    assert loss.item() == pytest.approx(0.0, abs=1e-7)


# ----------------------------------------------------------------------
# Test 7: Gradiente fluye solo por contribuyentes
# ----------------------------------------------------------------------


def test_gradiente_fluye_solo_por_timesteps_contribuyentes():
    """Backward: solo los pred[b,c,n,p] con (ssl & valid) deben tener grad ≠ 0."""
    pred = torch.zeros(B, C, N, P, requires_grad=True)
    target = torch.ones(B, C, N, P)  # forzar error no trivial

    valid_time = _partial_valid_time_mask(valid_len=14)
    ssl_mask = _ssl_mask_only_last_patch()

    loss = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    loss.backward()

    g = pred.grad
    assert g is not None
    # Patches no enmascarados: grad debe ser 0
    assert g[:, :, :3, :].abs().sum().item() == 0.0
    # Patch 3 (enmascarado), padding (p=2, p=3): grad debe ser 0
    assert g[:, :, 3, 2].abs().sum().item() == 0.0
    assert g[:, :, 3, 3].abs().sum().item() == 0.0
    # Patch 3 (enmascarado), real (p=0, p=1): grad debe ser distinto de 0
    assert g[:, :, 3, 0].abs().sum().item() > 0.0
    assert g[:, :, 3, 1].abs().sum().item() > 0.0


# ----------------------------------------------------------------------
# Test 8: Validacion de shapes
# ----------------------------------------------------------------------


def test_rechaza_shapes_inconsistentes():
    pred = torch.zeros(B, C, N, P)
    target = torch.zeros(B, C, N, P)
    ssl_mask = _ssl_mask_all_patches()
    valid_time = _all_valid_time_mask()

    # pred ≠ target shape
    with pytest.raises(ValueError, match="pred.shape"):
        masked_reconstruction_loss(torch.zeros(B, C, N, P + 1), target, ssl_mask, valid_time)

    # ssl_mask shape erronea
    with pytest.raises(ValueError, match="ssl_mask.shape"):
        masked_reconstruction_loss(
            pred, target, torch.zeros(B, C, N + 1, dtype=torch.bool), valid_time
        )

    # valid_time_mask shape erronea
    with pytest.raises(ValueError, match="valid_time_mask.shape"):
        masked_reconstruction_loss(
            pred, target, ssl_mask, torch.zeros(B, W + 1, dtype=torch.bool)
        )

    # ssl_mask no bool
    with pytest.raises(ValueError, match="ssl_mask debe ser bool"):
        masked_reconstruction_loss(pred, target, torch.zeros(B, C, N), valid_time)

    # valid_time_mask no bool
    with pytest.raises(ValueError, match="valid_time_mask debe ser bool"):
        masked_reconstruction_loss(pred, target, ssl_mask, torch.zeros(B, W))

    # loss_fn invalido
    with pytest.raises(ValueError, match="loss_fn"):
        masked_reconstruction_loss(pred, target, ssl_mask, valid_time, loss_fn="huber")

    # reduction invalido
    with pytest.raises(ValueError, match="reduction"):
        masked_reconstruction_loss(pred, target, ssl_mask, valid_time, reduction="avg")


# ----------------------------------------------------------------------
# Test 9: Batch mixto — ventanas completas y parciales juntas
# ----------------------------------------------------------------------


def test_batch_mixto_completa_y_parcial():
    """Un batch con (sample 0) ventana completa y (sample 1) ventana parcial.

    Verifica que la mascara se aplica correctamente por batch element, no
    se contamina entre samples.
    """
    B2 = 2
    pred = torch.zeros(B2, C, N, P)
    target = torch.zeros(B2, C, N, P)

    # Sample 0 (ventana completa): perturbacion en patch 3, todos los timesteps
    # son reales → todos contribuyen.
    pred[0, :, N - 1, :] = 1.0

    # Sample 1 (ventana parcial, valid_len=14): misma perturbacion, pero solo
    # 2 de los 4 timesteps son reales → solo 2 contribuyen.
    pred[1, :, N - 1, :] = 1.0

    valid_time = torch.zeros(B2, W, dtype=torch.bool)
    valid_time[0, :] = True  # sample 0: completa
    valid_time[1, :14] = True  # sample 1: parcial

    ssl_mask = torch.zeros(B2, C, N, dtype=torch.bool)
    ssl_mask[:, :, N - 1] = True

    loss_sum = masked_reconstruction_loss(pred, target, ssl_mask, valid_time, reduction="sum")
    # Sample 0: 4 timesteps * C canales * (1.0)^2 = 4*C = 8
    # Sample 1: 2 timesteps * C canales * (1.0)^2 = 2*C = 4
    # Total: 12
    assert loss_sum.item() == pytest.approx(12.0, rel=1e-6)

    loss_mean = masked_reconstruction_loss(pred, target, ssl_mask, valid_time)
    # Contribuyentes: 4*C en sample 0 + 2*C en sample 1 = 6*C = 12
    # Sin embargo, todo el error es 1.0 por timestep contribuyente → mean = 1.0
    assert loss_mean.item() == pytest.approx(1.0, rel=1e-6)


# ----------------------------------------------------------------------
# Test 10: Helper derivar_valid_sample_mask
# ----------------------------------------------------------------------


def test_helper_reshape_correctamente():
    """derivar_valid_sample_mask debe convertir (B, W) a (B, N, P) preservando
    el orden temporal."""
    valid_time = torch.zeros(B, W, dtype=torch.bool)
    valid_time[0, :10] = True  # timesteps 0..9 reales (10), 10..15 padding (6)

    out = derivar_valid_sample_mask(valid_time, n_patches=N, patch_size=P)
    assert out.shape == (B, N, P)

    # Patch 0 (timesteps 0..3): todo real
    assert out[0, 0, :].all().item()
    # Patch 1 (timesteps 4..7): todo real
    assert out[0, 1, :].all().item()
    # Patch 2 (timesteps 8..11): 0..9 reales, 10..11 padding
    assert out[0, 2, 0].item() and out[0, 2, 1].item()
    assert not out[0, 2, 2].item() and not out[0, 2, 3].item()
    # Patch 3 (timesteps 12..15): todo padding
    assert not out[0, 3, :].any().item()


def test_helper_rechaza_dimensiones_mal():
    with pytest.raises(ValueError, match="W="):
        derivar_valid_sample_mask(
            torch.zeros(B, W + 1, dtype=torch.bool), n_patches=N, patch_size=P
        )
    with pytest.raises(ValueError, match="bool"):
        derivar_valid_sample_mask(
            torch.zeros(B, W).float(), n_patches=N, patch_size=P
        )
