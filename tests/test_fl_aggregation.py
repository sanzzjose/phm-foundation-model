"""Tests de FedAvg aggregation.

Cubre:
- fedavg_state_dict con tensores conocidos: media ponderada exacta.
- weights uniform vs ponderado.
- rechazos: keys mismatch, shapes mismatch, weights negativos/cero.
- tensores no-float: copia del primer cliente si todos coinciden, error si no.
- estimate_communication_mb.
- compute_drift_l2.
"""

from __future__ import annotations

import pytest


def _t(*args, **kwargs):
    """Import torch dentro de tests para que la coleccion no falle si torch
    no esta disponible (lo cual seria raro pero defensivo)."""
    import torch
    return torch.tensor(*args, **kwargs)


# ----------------------------------------------------------------------
# fedavg_state_dict
# ----------------------------------------------------------------------


def test_fedavg_uniform_de_dos_clientes():
    import torch
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([1.0, 2.0, 3.0]), "b": _t([0.0])}
    b = {"w": _t([3.0, 4.0, 5.0]), "b": _t([2.0])}
    out = fedavg_state_dict([a, b])
    assert torch.allclose(out["w"], _t([2.0, 3.0, 4.0]))
    assert torch.allclose(out["b"], _t([1.0]))


def test_fedavg_ponderado():
    import torch
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([10.0])}
    b = {"w": _t([0.0])}
    out = fedavg_state_dict([a, b], weights=[3.0, 1.0])
    # (3*10 + 1*0) / 4 = 7.5
    assert torch.allclose(out["w"], _t([7.5]))


def test_fedavg_rechaza_keys_mismatch():
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([1.0])}
    b = {"v": _t([1.0])}
    with pytest.raises(ValueError, match="keys mismatch"):
        fedavg_state_dict([a, b])


def test_fedavg_rechaza_shapes_mismatch():
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([1.0, 2.0])}
    b = {"w": _t([1.0, 2.0, 3.0])}
    with pytest.raises(ValueError, match="shape mismatch"):
        fedavg_state_dict([a, b])


def test_fedavg_rechaza_pesos_negativos():
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([1.0])}
    b = {"w": _t([2.0])}
    with pytest.raises(ValueError, match="invalido"):
        fedavg_state_dict([a, b], weights=[1.0, -1.0])


def test_fedavg_rechaza_suma_cero():
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([1.0])}
    b = {"w": _t([2.0])}
    with pytest.raises(ValueError, match="no positivo"):
        fedavg_state_dict([a, b], weights=[0.0, 0.0])


def test_fedavg_state_dicts_vacio():
    from training.fl.aggregation import fedavg_state_dict
    with pytest.raises(ValueError, match="vacio"):
        fedavg_state_dict([])


def test_fedavg_tensor_no_float_iguales_se_copian():
    import torch
    from training.fl.aggregation import fedavg_state_dict
    a = {"w": _t([1.0]), "step": torch.tensor([3], dtype=torch.int64)}
    b = {"w": _t([3.0]), "step": torch.tensor([3], dtype=torch.int64)}
    out = fedavg_state_dict([a, b])
    assert torch.equal(out["step"], torch.tensor([3], dtype=torch.int64))
    assert torch.allclose(out["w"], _t([2.0]))


def test_fedavg_tensor_no_float_diferentes_rechazo():
    import torch
    from training.fl.aggregation import fedavg_state_dict
    a = {"step": torch.tensor([3], dtype=torch.int64)}
    b = {"step": torch.tensor([5], dtype=torch.int64)}
    with pytest.raises(ValueError, match="no-floating"):
        fedavg_state_dict([a, b])


# ----------------------------------------------------------------------
# estimate_communication_mb
# ----------------------------------------------------------------------


def test_estimate_communication_mb():
    from training.fl.aggregation import estimate_communication_mb
    # 800k params * 4 bytes * 2 (bidir) * 10 clientes = 64 MB
    mb = estimate_communication_mb(param_count=800_000, n_clients_participated=10)
    assert mb == pytest.approx(64.0 / 1.024 / 1.024, rel=0.01)


def test_estimate_communication_zero_si_inputs_invalidos():
    from training.fl.aggregation import estimate_communication_mb
    assert estimate_communication_mb(0, 10) == 0.0
    assert estimate_communication_mb(100, 0) == 0.0


# ----------------------------------------------------------------------
# compute_drift_l2
# ----------------------------------------------------------------------


def test_drift_l2_zero_si_iguales():
    import torch
    from training.fl.aggregation import compute_drift_l2
    a = {"w": _t([1.0, 2.0]), "b": _t([0.0])}
    assert compute_drift_l2(a, a) == 0.0


def test_drift_l2_distancia_conocida():
    import torch
    from training.fl.aggregation import compute_drift_l2
    a = {"w": _t([0.0, 0.0])}
    b = {"w": _t([3.0, 4.0])}
    # sqrt(9 + 16) = 5
    assert compute_drift_l2(a, b) == pytest.approx(5.0)


def test_drift_l2_ignora_no_float():
    import torch
    from training.fl.aggregation import compute_drift_l2
    a = {"step": torch.tensor([1], dtype=torch.int64)}
    b = {"step": torch.tensor([100], dtype=torch.int64)}
    # Solo tensores floating cuentan; aqui no hay ninguno -> 0
    assert compute_drift_l2(a, b) == 0.0
