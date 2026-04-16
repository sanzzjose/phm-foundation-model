"""Tests del server momentum (FedAvgM, Fase 4d).

Bloque 1 — funciones puras `state_dict_l2_norm`, `compute_state_dict_delta`,
`apply_server_momentum`:

- compute_state_dict_delta calcula `sd_new - sd_old`.
- apply_server_momentum con beta=0 equivale a aplicar delta puro (FedAvg
  estandar) bit-a-bit.
- apply_server_momentum con beta=0.9 acumula velocity correctamente entre
  rondas: v_t = beta*v_{t-1} + delta_t; w_t = w_{t-1} + v_t.
- Nesterov aplica `delta + beta*v_t` en lugar de v_t.

Bloque 2 — `_validate_federated_config` acepta `algorithm=fedavgm` con
  `fedprox_mu=null` y rechaza ambiguedades.

Bloque 3 — `resolve_fedprox_config` con `algorithm=fedavgm` devuelve
  `fedprox_enabled=False`.
"""

from __future__ import annotations

import pytest
import torch

from training.fl.aggregation import (
    apply_server_momentum,
    compute_state_dict_delta,
    state_dict_l2_norm,
)


# ----------------------------------------------------------------------
# Bloque 1: funciones puras
# ----------------------------------------------------------------------


def test_state_dict_l2_norm_zeros():
    sd = {"a": torch.zeros(3), "b": torch.zeros(5)}
    assert state_dict_l2_norm(sd) == 0.0


def test_state_dict_l2_norm_unit():
    sd = {"a": torch.tensor([1.0, 0.0, 0.0]), "b": torch.tensor([0.0, 1.0])}
    # ||a||^2 + ||b||^2 = 1 + 1 = 2 -> sqrt(2)
    assert state_dict_l2_norm(sd) == pytest.approx(2.0 ** 0.5)


def test_state_dict_l2_norm_skips_non_float():
    sd = {"a": torch.tensor([1.0, 0.0]), "n": torch.tensor([3, 4], dtype=torch.int32)}
    # solo cuenta "a", ||a|| = 1.0
    assert state_dict_l2_norm(sd) == pytest.approx(1.0)


def test_compute_state_dict_delta():
    a = {"w": torch.tensor([2.0, 3.0])}
    b = {"w": torch.tensor([1.0, 1.0])}
    d = compute_state_dict_delta(a, b)
    assert torch.allclose(d["w"], torch.tensor([1.0, 2.0]))


def test_compute_state_dict_delta_skips_shape_mismatch():
    a = {"w": torch.tensor([1.0, 2.0, 3.0])}
    b = {"w": torch.tensor([1.0, 2.0])}
    d = compute_state_dict_delta(a, b)
    # shape mismatch -> se omite ese key
    assert "w" not in d


def test_compute_state_dict_delta_skips_int_tensors():
    a = {"w": torch.tensor([1.0, 2.0]), "n": torch.tensor([3, 4], dtype=torch.int32)}
    b = {"w": torch.tensor([0.0, 1.0]), "n": torch.tensor([3, 4], dtype=torch.int32)}
    d = compute_state_dict_delta(a, b)
    assert "w" in d
    assert "n" not in d


# ----------------------------------------------------------------------
# apply_server_momentum — backward compat
# ----------------------------------------------------------------------


def test_apply_server_momentum_beta0_equivale_fedavg():
    """beta=0 debe reproducir bit-a-bit `w_new = w_old + delta`."""
    w = {"a": torch.tensor([1.0, 2.0]), "b": torch.tensor([0.0])}
    delta = {"a": torch.tensor([0.5, -0.5]), "b": torch.tensor([1.0])}
    sd_new, v_new = apply_server_momentum(w, None, delta, beta=0.0)
    assert torch.allclose(sd_new["a"], torch.tensor([1.5, 1.5]))
    assert torch.allclose(sd_new["b"], torch.tensor([1.0]))
    # velocity[k] = beta*0 + delta = delta cuando beta=0
    assert torch.allclose(v_new["a"], delta["a"])


def test_apply_server_momentum_beta09_round1():
    """En la primera ronda, v_1 = 0.9*0 + delta = delta, w_new = w + delta.
    El efecto de beta>0 solo aparece en rondas posteriores."""
    w = {"a": torch.tensor([10.0])}
    delta = {"a": torch.tensor([1.0])}
    sd_new, v_new = apply_server_momentum(w, None, delta, beta=0.9)
    assert torch.allclose(v_new["a"], torch.tensor([1.0]))
    assert torch.allclose(sd_new["a"], torch.tensor([11.0]))


def test_apply_server_momentum_beta09_acumula_round2():
    """En la 2a ronda con beta=0.9 y deltas iguales [1.0]:

    Ronda 1: v=1.0, w=10+1=11
    Ronda 2: v=0.9*1.0+1.0=1.9, w=11+1.9=12.9

    """
    w = {"a": torch.tensor([10.0])}
    delta1 = {"a": torch.tensor([1.0])}
    sd1, v1 = apply_server_momentum(w, None, delta1, beta=0.9)
    assert torch.allclose(sd1["a"], torch.tensor([11.0]))
    assert torch.allclose(v1["a"], torch.tensor([1.0]))

    delta2 = {"a": torch.tensor([1.0])}
    sd2, v2 = apply_server_momentum(sd1, v1, delta2, beta=0.9)
    assert torch.allclose(v2["a"], torch.tensor([1.9]), atol=1e-6)
    assert torch.allclose(sd2["a"], torch.tensor([12.9]), atol=1e-6)


def test_apply_server_momentum_nesterov():
    """Nesterov: update = delta + beta*v, no solo v."""
    w = {"a": torch.tensor([0.0])}
    delta = {"a": torch.tensor([1.0])}
    sd_new, v_new = apply_server_momentum(
        w, None, delta, beta=0.5, nesterov=True
    )
    # v_1 = 0.5*0 + 1 = 1
    # update = delta + beta*v = 1 + 0.5*1 = 1.5
    # w_new = 0 + 1.5 = 1.5
    assert torch.allclose(v_new["a"], torch.tensor([1.0]))
    assert torch.allclose(sd_new["a"], torch.tensor([1.5]))


def test_apply_server_momentum_beta_out_of_range_aborta():
    w = {"a": torch.tensor([0.0])}
    delta = {"a": torch.tensor([1.0])}
    with pytest.raises(ValueError, match="beta"):
        apply_server_momentum(w, None, delta, beta=1.5)
    with pytest.raises(ValueError, match="beta"):
        apply_server_momentum(w, None, delta, beta=-0.1)


def test_apply_server_momentum_preserva_buffers_no_float():
    """Buffers no-float (counters, masks integer) se copian intactos."""
    w = {
        "a": torch.tensor([10.0]),
        "counter": torch.tensor([5], dtype=torch.int32),
    }
    delta = {"a": torch.tensor([1.0])}  # NO incluye 'counter'
    sd_new, _ = apply_server_momentum(w, None, delta, beta=0.9)
    assert "counter" in sd_new
    assert torch.equal(sd_new["counter"], torch.tensor([5], dtype=torch.int32))


def test_apply_server_momentum_3_rondas_equivale_acumulado():
    """Test conjunto: 3 rondas, deltas [1, 2, 0.5], beta=0.5.

    Ronda 1: v=1, w=0+1=1
    Ronda 2: v=0.5*1+2=2.5, w=1+2.5=3.5
    Ronda 3: v=0.5*2.5+0.5=1.75, w=3.5+1.75=5.25
    """
    w = {"a": torch.tensor([0.0])}
    deltas = [torch.tensor([1.0]), torch.tensor([2.0]), torch.tensor([0.5])]
    expected_v = [1.0, 2.5, 1.75]
    expected_w = [1.0, 3.5, 5.25]
    v = None
    for i, d in enumerate(deltas):
        w, v = apply_server_momentum(w, v, {"a": d}, beta=0.5)
        assert torch.allclose(v["a"], torch.tensor([expected_v[i]]), atol=1e-6), \
            f"v en ronda {i+1}"
        assert torch.allclose(w["a"], torch.tensor([expected_w[i]]), atol=1e-6), \
            f"w en ronda {i+1}"


# ----------------------------------------------------------------------
# Bloque 2: validacion de algorithm=fedavgm en _validate_federated_config
# ----------------------------------------------------------------------


from training.train_ssl_federated import _validate_federated_config


def test_validate_federated_fedavgm_basico():
    cfg = {"federated": {"algorithm": "fedavgm"}}
    out = _validate_federated_config(cfg)
    assert out["algorithm"] == "fedavgm"
    assert out["fedprox_enabled"] is False
    assert out["fedprox_mu"] is None


def test_validate_federated_fedavgm_con_mu_no_cero_aborta():
    """fedavgm + fedprox_mu > 0 es ambiguo (mezcla momentum y proximal)."""
    cfg = {"federated": {"algorithm": "fedavgm", "fedprox_mu": 0.01}}
    with pytest.raises(ValueError, match="ambiguo|fedprox"):
        _validate_federated_config(cfg)


def test_validate_federated_fedavgm_con_mu_cero_ok():
    """fedavgm + fedprox_mu=0 explicito es OK (== null)."""
    cfg = {"federated": {"algorithm": "fedavgm", "fedprox_mu": 0.0}}
    out = _validate_federated_config(cfg)
    assert out["algorithm"] == "fedavgm"
    assert out["fedprox_enabled"] is False


def test_validate_federated_fedavg_sin_cambios():
    """Regresion: fedavg sigue funcionando bit-a-bit."""
    cfg = {"federated": {"algorithm": "fedavg"}}
    out = _validate_federated_config(cfg)
    assert out["algorithm"] == "fedavg"
    assert out["fedprox_enabled"] is False


def test_validate_federated_fedprox_sin_cambios():
    """Regresion: fedprox sigue funcionando bit-a-bit."""
    cfg = {"federated": {"algorithm": "fedprox", "fedprox_mu": 0.01}}
    out = _validate_federated_config(cfg)
    assert out["algorithm"] == "fedprox"
    assert out["fedprox_enabled"] is True


def test_validate_federated_algorithm_desconocido_aborta():
    cfg = {"federated": {"algorithm": "scaffold"}}
    with pytest.raises(ValueError, match="desconocido"):
        _validate_federated_config(cfg)


# ----------------------------------------------------------------------
# Bloque 3: resolve_fedprox_config en cliente
# ----------------------------------------------------------------------


from training.fl.client import resolve_fedprox_config


def test_resolve_fedprox_fedavgm_es_fedprox_inactivo():
    """El cliente bajo fedavgm debe comportarse como FedAvg (sin proximal)."""
    out = resolve_fedprox_config({"algorithm": "fedavgm"})
    assert out["algorithm"] == "fedavgm"
    assert out["fedprox_enabled"] is False
    assert out["fedprox_mu"] == 0.0
