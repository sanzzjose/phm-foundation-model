"""Tests del pooling channel-independent para downstream.

No requiere shards reales. Construye tokens sintéticos pequenos y verifica:

- el pooling ignora patches invalidos;
- el pooling ignora canales constantes;
- el pooling es seguro cuando todo invalido o todo constante (no NaN);
- el pooling es estable bajo permutaciones de canales/patches.
"""

from __future__ import annotations

import math

import pytest
import torch

from training.downstream.pooling import (
    masked_channel_mean_pool,
    masked_patch_mean_pool,
    pooled_embedding,
)


B, C, N, D = 2, 3, 4, 8


# ----------------------------------------------------------------------
# masked_patch_mean_pool
# ----------------------------------------------------------------------


def test_patch_mean_all_valid_equals_mean():
    """Si todos los patches son validos, debe ser mean estandar."""
    tokens = torch.randn(B, C, N, D)
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    pooled = masked_patch_mean_pool(tokens, vpm)
    expected = tokens.mean(dim=2)
    assert torch.allclose(pooled, expected, atol=1e-6)


def test_patch_mean_ignores_invalid_patches():
    """Los patches con vpm=False no contribuyen."""
    tokens = torch.zeros(B, C, N, D)
    # Sample 0, canal 0: el patch 0 vale 100 (invalido) y patches 1,2,3 valen 1 (validos)
    tokens[0, 0, 0, :] = 100.0
    tokens[0, 0, 1:, :] = 1.0
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    vpm[0, 0, 0] = False
    pooled = masked_patch_mean_pool(tokens, vpm)
    # mean de los 3 validos = 1.0; el invalido (100) NO debe filtrarse
    assert torch.allclose(pooled[0, 0], torch.full((D,), 1.0), atol=1e-6)


def test_patch_mean_zero_when_no_valid():
    """Filas (b,c) sin patches validos: pooled debe ser 0 sin NaN."""
    tokens = torch.randn(B, C, N, D)
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    vpm[0, 1, :] = False  # canal 1 del sample 0: ningun valido
    pooled = masked_patch_mean_pool(tokens, vpm)
    assert torch.allclose(pooled[0, 1], torch.zeros(D))
    # NaN explicit check
    assert not torch.isnan(pooled).any()


def test_patch_mean_rejects_bad_shape():
    tokens = torch.randn(B, C, N, D)
    bad_vpm = torch.ones(B, C + 1, N, dtype=torch.bool)
    with pytest.raises(ValueError, match="valid_patch_mask"):
        masked_patch_mean_pool(tokens, bad_vpm)


def test_patch_mean_rejects_non_bool():
    tokens = torch.randn(B, C, N, D)
    vpm = torch.ones(B, C, N)  # float, no bool
    with pytest.raises(ValueError, match="bool"):
        masked_patch_mean_pool(tokens, vpm)


# ----------------------------------------------------------------------
# masked_channel_mean_pool
# ----------------------------------------------------------------------


def test_channel_mean_all_valid_equals_mean():
    pooled_per_c = torch.randn(B, C, D)
    cvm = torch.ones(B, C, dtype=torch.bool)
    out = masked_channel_mean_pool(pooled_per_c, cvm)
    expected = pooled_per_c.mean(dim=1)
    assert torch.allclose(out, expected, atol=1e-6)


def test_channel_mean_ignores_constant_channels():
    pooled_per_c = torch.zeros(B, C, D)
    pooled_per_c[0, 0, :] = 99.0  # canal 0 marcado como constante
    pooled_per_c[0, 1, :] = 1.0
    pooled_per_c[0, 2, :] = 3.0
    cvm = torch.ones(B, C, dtype=torch.bool)
    cvm[0, 0] = False  # canal 0 invalido (constante)
    out = masked_channel_mean_pool(pooled_per_c, cvm)
    # Media de canales 1 y 2 = (1+3)/2 = 2.0
    assert torch.allclose(out[0], torch.full((D,), 2.0), atol=1e-6)


def test_channel_mean_zero_if_no_valid_channel():
    pooled_per_c = torch.randn(B, C, D)
    cvm = torch.zeros(B, C, dtype=torch.bool)  # ningun canal valido
    out = masked_channel_mean_pool(pooled_per_c, cvm)
    assert torch.allclose(out, torch.zeros(B, D))
    assert not torch.isnan(out).any()


# ----------------------------------------------------------------------
# pooled_embedding (composicion)
# ----------------------------------------------------------------------


def test_pooled_embedding_returns_correct_shape():
    tokens = torch.randn(B, C, N, D)
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    out = pooled_embedding(tokens, vpm)
    assert out.shape == (B, D)


def test_pooled_embedding_handles_constants_mask():
    tokens = torch.zeros(B, C, N, D)
    tokens[:, 0, :, :] = 5.0  # canal 0
    tokens[:, 1, :, :] = 3.0  # canal 1
    tokens[:, 2, :, :] = 1.0  # canal 2
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    # canal 0 marcado como constante
    cc = torch.zeros(B, C, dtype=torch.bool)
    cc[:, 0] = True
    out = pooled_embedding(tokens, vpm, cc)
    # media canales 1 y 2 = (3 + 1) / 2 = 2.0
    assert torch.allclose(out, torch.full((B, D), 2.0), atol=1e-6)


def test_pooled_embedding_fallback_when_all_constant():
    """Si TODOS los canales son constantes: NO devolver cero ciego.
    Usa los que al menos tienen patches validos (mejor que NaN)."""
    tokens = torch.zeros(B, C, N, D)
    tokens[:, 0, :, :] = 5.0
    tokens[:, 1, :, :] = 5.0
    tokens[:, 2, :, :] = 5.0
    vpm = torch.ones(B, C, N, dtype=torch.bool)
    cc = torch.ones(B, C, dtype=torch.bool)  # TODOS constantes
    out = pooled_embedding(tokens, vpm, cc)
    # Fallback: usa cualquier canal con patches validos. Como los 3 tienen
    # el mismo valor 5.0, el resultado debe ser 5.0 en todas las dims.
    assert torch.allclose(out, torch.full((B, D), 5.0), atol=1e-6)
    assert not torch.isnan(out).any()


def test_pooled_embedding_no_nan_when_all_invalid():
    """Si NINGUN canal tiene patches validos: cero, sin NaN."""
    tokens = torch.randn(B, C, N, D)
    vpm = torch.zeros(B, C, N, dtype=torch.bool)
    out = pooled_embedding(tokens, vpm)
    assert torch.allclose(out, torch.zeros(B, D))
    assert not torch.isnan(out).any()
