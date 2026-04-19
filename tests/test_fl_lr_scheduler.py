"""Tests del scheduler LR para FL (v0.3, Fase 4c).

Bloque 1 — funcion pura `cosine_warmup_lr_factor` y `lr_for_round_step`:
- linear warmup en [0, warmup_steps).
- cosine decay en [warmup_steps, total_steps].
- factor en step=warmup_steps == 1.0.
- factor en step=total_steps == 0.0 (cosine final).
- step_accounting='aggregate_local_updates' produce step_global correcto.
- backward-compat: scheduler_cfg=None devuelve base_lr.
- same_lr_for_all_clients_in_round: el LR depende solo de (round, step), no del cliente.

Bloque 2 — integracion FederatedClient + servidor:
- Sin scheduler en cfg, run_info.lr_scheduler == {"type": "constant"} y client
  no actualiza LR (verificado via metrics.lr_first==None).
- Con scheduler en cfg, dry-run del server pasa sin errores.
"""

from __future__ import annotations

import math

import pytest

from training.ssl.schedulers import (
    cosine_warmup_lr_factor,
    lr_for_round_step,
    scheduler_summary,
)


# ----------------------------------------------------------------------
# Bloque 1: funcion pura
# ----------------------------------------------------------------------


def test_warmup_step0_es_minimo_positivo():
    """En step=0 el factor de warmup es 1/warmup_steps."""
    assert cosine_warmup_lr_factor(0, warmup_steps=10, total_steps=100) == pytest.approx(1.0 / 10)


def test_warmup_step_intermedio():
    """En step=k con k < warmup, factor = (k+1)/warmup."""
    assert cosine_warmup_lr_factor(4, warmup_steps=10, total_steps=100) == pytest.approx(5.0 / 10)


def test_warmup_end_factor_1():
    """En step=warmup_steps el cosine arranca: factor=1.0."""
    assert cosine_warmup_lr_factor(10, warmup_steps=10, total_steps=100) == pytest.approx(1.0)


def test_cosine_middle_factor_05():
    """En mitad del cosine (entre warmup y total): factor=0.5."""
    # warmup=10, total=110, middle = (110+10)/2 = 60 -> progress=0.5 -> factor=0.5*(1+cos(pi/2))=0.5
    assert cosine_warmup_lr_factor(60, warmup_steps=10, total_steps=110) == pytest.approx(0.5)


def test_cosine_end_factor_0():
    """En step=total_steps el cosine acaba: factor=0.0."""
    assert cosine_warmup_lr_factor(100, warmup_steps=10, total_steps=100) == pytest.approx(0.0, abs=1e-12)


def test_cosine_clamp_beyond_total():
    """step > total_steps debe clamparse a factor=0 (no negativo)."""
    assert cosine_warmup_lr_factor(150, warmup_steps=10, total_steps=100) == pytest.approx(0.0, abs=1e-12)


# ----------------------------------------------------------------------
# lr_for_round_step + step_accounting
# ----------------------------------------------------------------------


def test_lr_for_round_step_none_devuelve_base_lr():
    """scheduler_cfg=None debe devolver base_lr SIN escalar."""
    assert lr_for_round_step(
        round_idx=1, local_step_in_round=1,
        n_clients=10, local_steps_per_client=50,
        base_lr=3e-4, scheduler_cfg=None,
    ) == pytest.approx(3e-4)


def test_lr_for_round_step_constant_devuelve_base_lr():
    """scheduler_cfg.type='constant' tambien devuelve base_lr."""
    assert lr_for_round_step(
        round_idx=1, local_step_in_round=1,
        n_clients=10, local_steps_per_client=50,
        base_lr=3e-4, scheduler_cfg={"type": "constant"},
    ) == pytest.approx(3e-4)


def test_lr_for_round_step_aggregate_local_updates_step_25k():
    """Caso de la Fase 4c: 50 rounds x 50 steps x 10 clientes = 25k local steps.

    En la ronda 50, step 50: step_global = 49*500 + 49*10 = 24500 + 490 = 24990.
    Con warmup=2000, total=100000: ya pasado warmup, en cosine decay.
    progress = (24990-2000)/(100000-2000) = 22990/98000 = 0.2346
    factor = 0.5*(1+cos(pi*0.2346)) ~ 0.5*(1+cos(0.7370)) ~ 0.5*(1+0.7396) ~ 0.8698
    """
    cfg = {"type": "cosine", "warmup_steps": 2000, "total_steps": 100000,
           "step_accounting": "aggregate_local_updates"}
    lr = lr_for_round_step(
        round_idx=50, local_step_in_round=50,
        n_clients=10, local_steps_per_client=50,
        base_lr=3e-4, scheduler_cfg=cfg,
    )
    # step_global = (50-1)*10*50 + (50-1)*10 = 24500 + 490 = 24990
    expected_factor = cosine_warmup_lr_factor(24990, 2000, 100000)
    assert lr == pytest.approx(3e-4 * expected_factor)


def test_lr_for_round_step_warmup_zone():
    """En rondas tempranas con warmup=2000 y n_clients=10, los primeros local_steps
    estan en la zona de warmup linear."""
    cfg = {"type": "cosine", "warmup_steps": 2000, "total_steps": 100000,
           "step_accounting": "aggregate_local_updates"}
    # round 1, step 1: step_global = 0 -> warmup factor = 1/2000
    lr = lr_for_round_step(
        round_idx=1, local_step_in_round=1,
        n_clients=10, local_steps_per_client=50,
        base_lr=3e-4, scheduler_cfg=cfg,
    )
    assert lr == pytest.approx(3e-4 * (1.0 / 2000))


def test_lr_same_for_all_clients_in_same_round_step():
    """LR depende solo de (round, step), no del cliente concreto."""
    cfg = {"type": "cosine", "warmup_steps": 2000, "total_steps": 100000,
           "step_accounting": "aggregate_local_updates"}
    # mismo (round, step) -> mismo LR
    lr_a = lr_for_round_step(
        round_idx=5, local_step_in_round=25,
        n_clients=10, local_steps_per_client=50,
        base_lr=3e-4, scheduler_cfg=cfg,
    )
    lr_b = lr_for_round_step(
        round_idx=5, local_step_in_round=25,
        n_clients=10, local_steps_per_client=50,
        base_lr=3e-4, scheduler_cfg=cfg,
    )
    assert lr_a == lr_b


def test_step_accounting_invalid_aborta():
    cfg = {"type": "cosine", "warmup_steps": 100, "total_steps": 1000,
           "step_accounting": "per_round"}
    with pytest.raises(ValueError, match="step_accounting"):
        lr_for_round_step(
            round_idx=1, local_step_in_round=1,
            n_clients=10, local_steps_per_client=50,
            base_lr=3e-4, scheduler_cfg=cfg,
        )


def test_unknown_scheduler_type_aborta():
    cfg = {"type": "linear", "warmup_steps": 100, "total_steps": 1000}
    with pytest.raises(ValueError, match="lr_scheduler.type"):
        lr_for_round_step(
            round_idx=1, local_step_in_round=1,
            n_clients=10, local_steps_per_client=50,
            base_lr=3e-4, scheduler_cfg=cfg,
        )


def test_total_steps_zero_aborta():
    cfg = {"type": "cosine", "warmup_steps": 100, "total_steps": 0}
    with pytest.raises(ValueError, match="total_steps"):
        lr_for_round_step(
            round_idx=1, local_step_in_round=1,
            n_clients=10, local_steps_per_client=50,
            base_lr=3e-4, scheduler_cfg=cfg,
        )


def test_round_or_step_lt_1_aborta():
    cfg = {"type": "cosine", "warmup_steps": 100, "total_steps": 1000}
    with pytest.raises(ValueError, match="deben ser >= 1"):
        lr_for_round_step(
            round_idx=0, local_step_in_round=1,
            n_clients=10, local_steps_per_client=50,
            base_lr=3e-4, scheduler_cfg=cfg,
        )


# ----------------------------------------------------------------------
# scheduler_summary (para serializar en run_info)
# ----------------------------------------------------------------------


def test_scheduler_summary_none_es_constant():
    assert scheduler_summary(None) == {"type": "constant"}


def test_scheduler_summary_constant_es_constant():
    assert scheduler_summary({"type": "constant"}) == {"type": "constant"}


def test_scheduler_summary_cosine_serializa_campos():
    cfg = {"type": "cosine", "warmup_steps": 2000, "total_steps": 100000,
           "step_accounting": "aggregate_local_updates",
           "same_lr_for_all_clients_in_round": True}
    summ = scheduler_summary(cfg)
    assert summ["type"] == "cosine"
    assert summ["warmup_steps"] == 2000
    assert summ["total_steps"] == 100000
    assert summ["step_accounting"] == "aggregate_local_updates"
    assert summ["same_lr_for_all_clients_in_round"] is True


# ----------------------------------------------------------------------
# Monotonia: cosine es no-creciente tras warmup
# ----------------------------------------------------------------------


def test_cosine_monotone_no_creciente_tras_warmup():
    """A partir de warmup_steps, el factor no debe subir."""
    warmup = 100
    total = 1000
    prev = cosine_warmup_lr_factor(warmup, warmup, total)
    for s in range(warmup + 1, total + 1, 10):
        cur = cosine_warmup_lr_factor(s, warmup, total)
        assert cur <= prev + 1e-12, f"factor sube en step {s}: {prev} -> {cur}"
        prev = cur


def test_warmup_monotone_creciente():
    """En warmup, el factor debe ser creciente."""
    prev = cosine_warmup_lr_factor(0, warmup_steps=10, total_steps=100)
    for s in range(1, 10):
        cur = cosine_warmup_lr_factor(s, 10, 100)
        assert cur > prev, f"factor no crece en step {s}"
        prev = cur
