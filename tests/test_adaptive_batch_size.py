"""Tests de la politica adaptativa de batch_size por canales.

Motivacion: el contrato channel-independent significa que el coste real
del transformer es `B * C * N`. Si C varia mucho entre datasets (DUS20
con C=1, PHM14 con C=317), un `batch_size` fijo desequilibra VRAM y
compute. La politica `adaptive_by_channels` limita `B * C <=
max_channel_batch` salvo cuando se llega al piso `min_batch_size`.
"""

from __future__ import annotations

import pytest

from training.sampling import compute_adaptive_batch_size


# ----------------------------------------------------------------------
# Casos canonicos del prompt
# ----------------------------------------------------------------------


def test_c_1_devuelve_batch_size_completo():
    # max_channel_batch=512, C=1 → 512 patches por step, pero capeado a 32
    assert compute_adaptive_batch_size(1, 32, 512) == 32


def test_c_8_devuelve_batch_size_completo():
    # 512 // 8 = 64, capeado a 32
    assert compute_adaptive_batch_size(8, 32, 512) == 32


def test_c_24_respeta_el_calculo():
    # 512 // 24 = 21 (entero)
    assert compute_adaptive_batch_size(24, 32, 512) == 21


def test_c_317_cae_a_1():
    # 512 // 317 = 1
    assert compute_adaptive_batch_size(317, 32, 512) == 1


def test_min_batch_size_floor():
    # Con C enorme, sin min_batch_size se quedaria a 0 (no permitido).
    # max_channel_batch=10, C=100 → 10 // 100 = 0 → floor a 1.
    assert compute_adaptive_batch_size(100, 32, 10) == 1


def test_min_batch_size_personalizado():
    # min_batch_size=2: con C=100 y max_channel_batch=10 → max(2, 0) = 2,
    # capeado a 32 → 2.
    assert compute_adaptive_batch_size(100, 32, 10, min_batch_size=2) == 2


def test_nunca_devuelve_cero():
    # Para cualquier C razonable, el resultado >= min_batch_size >= 1.
    for c in (1, 8, 24, 60, 317, 1024):
        assert compute_adaptive_batch_size(c, 32, 512) >= 1


def test_no_supera_batch_size():
    # En ningun caso B_eff supera el batch_size pasado como argumento.
    for c in (1, 2, 4, 8, 24, 60, 317):
        b_eff = compute_adaptive_batch_size(c, 32, 4096)
        assert b_eff <= 32


def test_max_channel_batch_none_equivale_a_fixed():
    # Sin max_channel_batch, la politica es equivalente a batch fijo.
    for c in (1, 8, 24, 317):
        assert compute_adaptive_batch_size(c, 32, None) == 32
        assert compute_adaptive_batch_size(c, 32, 0) == 32


# ----------------------------------------------------------------------
# Errores
# ----------------------------------------------------------------------


def test_rechaza_n_channels_no_positivo():
    with pytest.raises(ValueError, match="n_channels"):
        compute_adaptive_batch_size(0, 32, 512)
    with pytest.raises(ValueError, match="n_channels"):
        compute_adaptive_batch_size(-1, 32, 512)


def test_rechaza_batch_size_no_positivo():
    with pytest.raises(ValueError, match="batch_size"):
        compute_adaptive_batch_size(8, 0, 512)


def test_rechaza_min_batch_size_invalido():
    with pytest.raises(ValueError, match="min_batch_size"):
        compute_adaptive_batch_size(8, 32, 512, min_batch_size=0)
