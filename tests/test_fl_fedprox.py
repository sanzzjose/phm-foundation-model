"""Tests del soporte FedProx en el cliente FL y en la validacion del config.

Cubren:

- `compute_fedprox_penalty(model, snapshot)` y `snapshot_global_params(...)`:
    - penalty == 0 cuando el modelo local == snapshot global;
    - penalty con valor cerrado conocido cuando hay desplazamiento;
    - no se incluyen buffers no-float ni parametros sin grad;
    - sin parametros activos / sin snapshot: devuelve 0 sin romper.

- `resolve_fedprox_config(cfg["federated"])`:
    - defaults FedAvg no activan FedProx;
    - FedProx con mu invalido (null / 0 / negativo) devuelve inactivo;
    - FedProx con mu > 0 devuelve activo.

- `_validate_federated_config(cfg)` del trainer:
    - algorithm desconocido falla;
    - FedAvg + mu > 0 falla (config ambiguo);
    - FedProx + mu null falla;
    - FedProx + mu <= 0 falla;
    - FedAvg + mu null/0 pasa;
    - FedProx + mu > 0 pasa.

Sigue el patron del repo: torch se importa dentro de cada test que lo
necesita (los demas no lo requieren).
"""

from __future__ import annotations

import pytest


# ----------------------------------------------------------------------
# resolve_fedprox_config: capa pura sin torch
# ----------------------------------------------------------------------


def test_resolve_fedprox_defaults_fedavg_no_activa():
    from training.fl.client import resolve_fedprox_config
    # Caso 1: dict vacio -> FedAvg implicito.
    r = resolve_fedprox_config({})
    assert r["algorithm"] == "fedavg"
    assert r["fedprox_enabled"] is False
    assert r["fedprox_mu"] == 0.0
    # Caso 2: algorithm=fedavg explicito, mu null.
    r = resolve_fedprox_config({"algorithm": "fedavg", "fedprox_mu": None})
    assert r["fedprox_enabled"] is False
    # Caso 3: algorithm=fedavg, mu=0 explicito.
    r = resolve_fedprox_config({"algorithm": "fedavg", "fedprox_mu": 0})
    assert r["fedprox_enabled"] is False


def test_resolve_fedprox_mu_invalido_deja_inactivo():
    """En la capa de resolucion, mu invalido bajo fedprox **no rompe**;
    deja FedProx inactivo. La validacion estricta esta en el trainer."""
    from training.fl.client import resolve_fedprox_config
    for mu in (None, 0, 0.0, -0.01, "wat"):
        r = resolve_fedprox_config({"algorithm": "fedprox", "fedprox_mu": mu})
        assert r["fedprox_enabled"] is False, mu
        assert r["fedprox_mu"] == 0.0, mu


def test_resolve_fedprox_mu_positivo_activa():
    from training.fl.client import resolve_fedprox_config
    r = resolve_fedprox_config({"algorithm": "fedprox", "fedprox_mu": 0.01})
    assert r["algorithm"] == "fedprox"
    assert r["fedprox_enabled"] is True
    assert r["fedprox_mu"] == pytest.approx(0.01)
    # Tambien acepta string convertible a float (defensivo).
    r = resolve_fedprox_config({"algorithm": "FedProx", "fedprox_mu": "0.05"})
    assert r["algorithm"] == "fedprox"
    assert r["fedprox_enabled"] is True
    assert r["fedprox_mu"] == pytest.approx(0.05)


# ----------------------------------------------------------------------
# compute_fedprox_penalty y snapshot_global_params
# ----------------------------------------------------------------------


def test_fedprox_penalty_zero_si_modelo_igual_global():
    """Penalty == 0 cuando theta_local == theta_global (apertura de ronda).

    Construimos un Linear y un snapshot identico (la copia del state_dict
    al inicio de la ronda). La penalty debe ser exactamente 0.
    """
    import torch
    from training.fl.client import snapshot_global_params, compute_fedprox_penalty
    torch.manual_seed(0)
    model = torch.nn.Linear(4, 3, bias=True)
    # Snapshot = state_dict actual (apertura de ronda).
    sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
    snap = snapshot_global_params(model, sd, device=torch.device("cpu"))
    # Snapshot cubre 2 params float entrenables (weight + bias).
    assert set(snap.keys()) == {"weight", "bias"}
    pen = compute_fedprox_penalty(model, snap)
    assert isinstance(pen, torch.Tensor)
    assert pen.dim() == 0
    assert float(pen) == pytest.approx(0.0)


def test_fedprox_penalty_valor_conocido_linear():
    """Linear pequeno con weight=[1,2,3] y global snapshot=[0,0,0]:
    penalty = 1+4+9 = 14 (sin sesgo, sin escalar por mu)."""
    import torch
    from training.fl.client import compute_fedprox_penalty
    model = torch.nn.Linear(3, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, 2.0, 3.0]]))
    # Snapshot global = 0 elementwise.
    snap = {"weight": torch.zeros_like(model.weight)}
    pen = compute_fedprox_penalty(model, snap)
    assert float(pen) == pytest.approx(14.0)


def test_fedprox_penalty_bias_y_weight():
    """Linear con bias: penalty cubre weight + bias y suma cada cuadrado."""
    import torch
    from training.fl.client import compute_fedprox_penalty
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[3.0, 4.0]]))  # 9 + 16 = 25
        model.bias.copy_(torch.tensor([1.0]))           # 1
    snap = {
        "weight": torch.zeros_like(model.weight),
        "bias": torch.zeros_like(model.bias),
    }
    pen = compute_fedprox_penalty(model, snap)
    assert float(pen) == pytest.approx(26.0)


def test_fedprox_no_incluye_buffers_no_float():
    """Buffers integer/bool NO deben contar para la penalty ni provocar fallo.

    Construimos un modulo con un buffer int (e.g. step counter). Tras
    llamar snapshot_global_params + compute_fedprox_penalty, el modulo
    no debe romper y los buffers no-float no deben aparecer en el snapshot
    ni contribuir a la penalty.
    """
    import torch
    from training.fl.client import snapshot_global_params, compute_fedprox_penalty

    class TinyModule(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = torch.nn.Linear(2, 1, bias=False)
            self.register_buffer("step_count", torch.tensor(7, dtype=torch.int64))
            self.register_buffer("flag", torch.tensor(True, dtype=torch.bool))

    model = TinyModule()
    with torch.no_grad():
        model.lin.weight.copy_(torch.tensor([[2.0, 0.0]]))
    sd_full = {
        # weight es lo unico float; los buffers no deben entrar en el snap.
        "lin.weight": torch.zeros_like(model.lin.weight),
        "step_count": torch.tensor(99, dtype=torch.int64),
        "flag": torch.tensor(False, dtype=torch.bool),
    }
    snap = snapshot_global_params(model, sd_full)
    assert set(snap.keys()) == {"lin.weight"}, snap.keys()
    pen = compute_fedprox_penalty(model, snap)
    # ||[2,0] - [0,0]||^2 = 4. Buffers no contribuyen.
    assert float(pen) == pytest.approx(4.0)


def test_fedprox_ignora_params_sin_grad():
    """Parametros con `requires_grad=False` no entran al snapshot ni a la
    penalty. Esto evita que la regularizacion atraiga capas congeladas
    fuera del rango global."""
    import torch
    from training.fl.client import snapshot_global_params, compute_fedprox_penalty
    model = torch.nn.Linear(2, 1, bias=True)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[5.0, 5.0]]))
        model.bias.copy_(torch.tensor([5.0]))
    model.weight.requires_grad_(False)  # congelado
    sd = {
        "weight": torch.zeros_like(model.weight),
        "bias": torch.zeros_like(model.bias),
    }
    snap = snapshot_global_params(model, sd)
    # Solo bias deberia estar en el snapshot.
    assert set(snap.keys()) == {"bias"}, snap.keys()
    pen = compute_fedprox_penalty(model, snap)
    # Solo bias = 5, ||5||^2 = 25.
    assert float(pen) == pytest.approx(25.0)


def test_fedprox_penalty_cero_si_snapshot_vacio():
    """Sin snapshot (o con keys ajenas al modelo), la penalty es 0 y no
    rompe. Util como red de seguridad si por accidente algun cliente se
    queda sin global_state_dict."""
    import torch
    from training.fl.client import compute_fedprox_penalty
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[9.0, 9.0]]))
    pen = compute_fedprox_penalty(model, {})
    assert float(pen) == pytest.approx(0.0)


def test_fedprox_penalty_gradiente_fluye():
    """Confirmamos que el escalar resultante mantiene grad_fn de los
    parametros del modelo, asi `.backward()` ajustara el modelo segun el
    termino proximal."""
    import torch
    from training.fl.client import compute_fedprox_penalty
    model = torch.nn.Linear(2, 1, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.tensor([[1.0, -1.0]]))
    snap = {"weight": torch.zeros_like(model.weight)}
    pen = compute_fedprox_penalty(model, snap)
    assert pen.requires_grad
    pen.backward()
    # dpenalty/dtheta = 2*(theta - 0) = 2*theta.
    assert model.weight.grad is not None
    assert torch.allclose(
        model.weight.grad,
        2.0 * torch.tensor([[1.0, -1.0]]),
    )


# ----------------------------------------------------------------------
# _validate_federated_config: validacion estricta del trainer
# ----------------------------------------------------------------------


def test_validacion_config_fedavg_defaults_pasa():
    """`algorithm=fedavg` + `fedprox_mu=null` (los defaults del repo) pasa
    sin error y devuelve fedprox_enabled=False."""
    from training.train_ssl_federated import _validate_federated_config
    r = _validate_federated_config(
        {"federated": {"algorithm": "fedavg", "fedprox_mu": None}}
    )
    assert r["algorithm"] == "fedavg"
    assert r["fedprox_enabled"] is False
    assert r["fedprox_mu"] is None
    # FedAvg + mu = 0 explicito tambien pasa.
    r2 = _validate_federated_config(
        {"federated": {"algorithm": "fedavg", "fedprox_mu": 0}}
    )
    assert r2["fedprox_enabled"] is False


def test_validacion_config_rechaza_fedavg_con_mu_positivo():
    """FedAvg con `fedprox_mu > 0` es config ambiguo: debe fallar duro."""
    from training.train_ssl_federated import _validate_federated_config
    with pytest.raises(ValueError, match="fedavg.*fedprox_mu"):
        _validate_federated_config(
            {"federated": {"algorithm": "fedavg", "fedprox_mu": 0.01}}
        )


def test_validacion_config_rechaza_fedprox_mu_null():
    """FedProx sin `fedprox_mu` definido debe fallar duro."""
    from training.train_ssl_federated import _validate_federated_config
    with pytest.raises(ValueError, match="fedprox.*fedprox_mu"):
        _validate_federated_config(
            {"federated": {"algorithm": "fedprox", "fedprox_mu": None}}
        )


def test_validacion_config_rechaza_fedprox_mu_no_positivo():
    """FedProx con `fedprox_mu <= 0` debe fallar duro."""
    from training.train_ssl_federated import _validate_federated_config
    for mu in (0, 0.0, -0.01):
        with pytest.raises(ValueError, match="fedprox_mu"):
            _validate_federated_config(
                {"federated": {"algorithm": "fedprox", "fedprox_mu": mu}}
            )


def test_validacion_config_rechaza_algorithm_desconocido():
    from training.train_ssl_federated import _validate_federated_config
    with pytest.raises(ValueError, match="algorithm"):
        _validate_federated_config(
            {"federated": {"algorithm": "scaffold", "fedprox_mu": 0.01}}
        )


def test_validacion_config_fedprox_mu_positivo_pasa():
    from training.train_ssl_federated import _validate_federated_config
    r = _validate_federated_config(
        {"federated": {"algorithm": "fedprox", "fedprox_mu": 0.01}}
    )
    assert r["algorithm"] == "fedprox"
    assert r["fedprox_enabled"] is True
    assert r["fedprox_mu"] == pytest.approx(0.01)
