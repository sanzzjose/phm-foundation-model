"""Tests preflight del config `ssl_federated_pilot.yaml`.

Validan que el config pilot del FL queda congelado bit-a-bit con la
decision del bloque FL pilot (commit 7). Cualquier cambio accidental
en el YAML rompe estos tests y fuerza una revision explicita antes de
relanzar pilot/full.

Independientes de torch (solo yaml + asserts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_PATH = _REPO_ROOT / "training" / "configs" / "ssl_federated_pilot.yaml"


@pytest.fixture(scope="module")
def pilot_cfg() -> dict:
    assert _CFG_PATH.is_file(), f"Falta config pilot FL: {_CFG_PATH}"
    return yaml.safe_load(_CFG_PATH.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Identidad del run
# ----------------------------------------------------------------------


def test_pilot_run_name_y_stage(pilot_cfg):
    assert pilot_cfg["run_name"] == "ssl_federated_pilot_patchtst_phm"
    assert pilot_cfg["stage"] == "pilot"


# ----------------------------------------------------------------------
# Backbone
# ----------------------------------------------------------------------


def test_pilot_model_block(pilot_cfg):
    m = pilot_cfg["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["n_layers"] == 4
    assert m["n_heads"] == 4
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


# ----------------------------------------------------------------------
# SSL config
# ----------------------------------------------------------------------


def test_pilot_ssl_block(pilot_cfg):
    s = pilot_cfg["ssl"]
    assert float(s["mask_ratio"]) == 0.3
    assert s["loss"] == "mse"


# ----------------------------------------------------------------------
# Data + sampling
# ----------------------------------------------------------------------


def test_pilot_data_block(pilot_cfg):
    d = pilot_cfg["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["split"] == "train"
    assert int(d["batch_size"]) == 32
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert int(d["min_batch_size"]) == 1
    assert d["client_source"] == "results/audit/audit_groups.json"
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


# ----------------------------------------------------------------------
# Federated config
# ----------------------------------------------------------------------


def test_pilot_federated_block(pilot_cfg):
    f = pilot_cfg["federated"]
    assert f["algorithm"] == "fedavg"
    assert f["fedprox_mu"] is None
    assert int(f["n_rounds"]) == 10
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    assert int(f["ckpt_every_rounds"]) == 5


# ----------------------------------------------------------------------
# Training (lr, weight_decay, grad_clip, amp)
# ----------------------------------------------------------------------


def test_pilot_training_block(pilot_cfg):
    t = pilot_cfg["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert float(t["grad_clip_norm"]) == 1.0
    assert t["amp"] == "auto"


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------


def test_pilot_paths_block(pilot_cfg):
    p = pilot_cfg["paths"]
    assert p["processed_root"] == "/content/drive/MyDrive/fm_fl_phmd/processed"
    assert p["log_dir"] == \
        "/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated"
    assert p["checkpoint_dir"] == \
        "/content/drive/MyDrive/fm_fl_phmd/checkpoints/ssl_federated_pilot"


# ----------------------------------------------------------------------
# Budget global del stage pilot
# ----------------------------------------------------------------------


def test_pilot_budget_total_local_steps(pilot_cfg):
    """n_rounds * local_steps * n_clients == 5000 y <= 20000 (cap stage=pilot).

    Los 10 clientes vienen del audit v2.3 (10 dominios FL cerrados,
    sec 7.bis CLAUDE.md). El cap 20000 es la condicion duro stage=pilot
    que separa pilot de full en el trainer FL.
    """
    f = pilot_cfg["federated"]
    n_clients = 10  # topologia FL cerrada en audit v2.3
    n_total_local_steps = (
        int(f["n_rounds"]) * int(f["local_steps"]) * n_clients
    )
    assert n_total_local_steps == 5000, n_total_local_steps
    assert n_total_local_steps <= 20000, (
        f"n_total_local_steps={n_total_local_steps} excede el cap del "
        "stage=pilot (20000). Pasar a stage=full requiere autorizacion "
        "explicita."
    )
