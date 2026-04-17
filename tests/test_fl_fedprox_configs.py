"""Tests preflight de los configs FedProx (smoke + pilot, v0.2).

Validan que los configs FedProx queden congelados bit-a-bit con la
politica acordada antes de lanzar un smoke o un pilot real en Colab.
Cualquier drift accidental en el YAML rompe el test y fuerza revision.

Independientes de torch (solo yaml + asserts).

Replicamos las invariantes de `test_fl_pilot_config.py` y anyadimos:

- algorithm == fedprox.
- fedprox_mu == 0.01.
- run_name distinto del FedAvg smoke/pilot.
- checkpoint_dir distinto del FedAvg smoke/pilot (no debe pisar el
  ckpt FedAvg en Drive).
- log_dir compartido con FedAvg (es el mismo arbol; el run_name discrimina).
- budgets respetan los caps stage del trainer FL.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_SMOKE_FP = _CFG_DIR / "ssl_federated_smoke_fedprox_mu0_01.yaml"
_PILOT_FP = _CFG_DIR / "ssl_federated_pilot_fedprox_mu0_01.yaml"
_SMOKE_FA = _CFG_DIR / "ssl_federated_smoke.yaml"
_PILOT_FA = _CFG_DIR / "ssl_federated_pilot.yaml"

# Caps duros del trainer FL (training/train_ssl_federated.STAGE_MAX_LOCAL_STEPS).
_CAP_SMOKE = 1_000
_CAP_PILOT = 20_000


@pytest.fixture(scope="module")
def smoke_fp_cfg() -> dict:
    assert _SMOKE_FP.is_file(), f"Falta config smoke FedProx: {_SMOKE_FP}"
    return yaml.safe_load(_SMOKE_FP.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pilot_fp_cfg() -> dict:
    assert _PILOT_FP.is_file(), f"Falta config pilot FedProx: {_PILOT_FP}"
    return yaml.safe_load(_PILOT_FP.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def smoke_fa_cfg() -> dict:
    assert _SMOKE_FA.is_file(), f"Falta config smoke FedAvg: {_SMOKE_FA}"
    return yaml.safe_load(_SMOKE_FA.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def pilot_fa_cfg() -> dict:
    assert _PILOT_FA.is_file(), f"Falta config pilot FedAvg: {_PILOT_FA}"
    return yaml.safe_load(_PILOT_FA.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# SMOKE FedProx
# ----------------------------------------------------------------------


def test_smoke_fp_existe_y_es_yaml(smoke_fp_cfg):
    assert isinstance(smoke_fp_cfg, dict)


def test_smoke_fp_run_name_y_stage(smoke_fp_cfg):
    assert smoke_fp_cfg["run_name"] == "ssl_federated_smoke_fedprox_mu0_01_patchtst_phm"
    assert smoke_fp_cfg["stage"] == "smoke"


def test_smoke_fp_federated_block(smoke_fp_cfg):
    f = smoke_fp_cfg["federated"]
    assert str(f["algorithm"]).lower() == "fedprox"
    assert float(f["fedprox_mu"]) == 0.01
    assert int(f["n_rounds"]) == 2
    assert int(f["local_steps"]) == 5
    assert f["clients_per_round"] == "all"


def test_smoke_fp_model_block(smoke_fp_cfg):
    m = smoke_fp_cfg["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["n_layers"] == 4
    assert m["n_heads"] == 4
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


def test_smoke_fp_data_block(smoke_fp_cfg):
    d = smoke_fp_cfg["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["split"] == "train"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert int(d["min_batch_size"]) == 1
    assert d["client_source"] == "results/audit/audit_groups.json"
    assert d["aggregation_weight_policy"] == "final_client_weight"


def test_smoke_fp_paths(smoke_fp_cfg):
    p = smoke_fp_cfg["paths"]
    assert p["log_dir"] == (
        "/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated"
    )
    # checkpoint_dir debe diferenciarse del FedAvg para no pisar.
    assert "fedprox_mu0_01" in p["checkpoint_dir"]


def test_smoke_fp_budget_dentro_del_cap(smoke_fp_cfg):
    f = smoke_fp_cfg["federated"]
    n_clients = 10
    total = int(f["n_rounds"]) * int(f["local_steps"]) * n_clients
    assert total == 100
    assert total <= _CAP_SMOKE, total


# ----------------------------------------------------------------------
# PILOT FedProx
# ----------------------------------------------------------------------


def test_pilot_fp_existe_y_es_yaml(pilot_fp_cfg):
    assert isinstance(pilot_fp_cfg, dict)


def test_pilot_fp_run_name_y_stage(pilot_fp_cfg):
    assert pilot_fp_cfg["run_name"] == "ssl_federated_pilot_fedprox_mu0_01_patchtst_phm"
    assert pilot_fp_cfg["stage"] == "pilot"


def test_pilot_fp_federated_block(pilot_fp_cfg):
    f = pilot_fp_cfg["federated"]
    assert str(f["algorithm"]).lower() == "fedprox"
    assert float(f["fedprox_mu"]) == 0.01
    assert int(f["n_rounds"]) == 10
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    assert int(f["ckpt_every_rounds"]) == 5


def test_pilot_fp_model_block(pilot_fp_cfg):
    m = pilot_fp_cfg["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["n_layers"] == 4
    assert m["n_heads"] == 4
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


def test_pilot_fp_data_block(pilot_fp_cfg):
    d = pilot_fp_cfg["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["split"] == "train"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert int(d["min_batch_size"]) == 1
    assert d["aggregation_weight_policy"] == "final_client_weight"


def test_pilot_fp_training_block(pilot_fp_cfg):
    t = pilot_fp_cfg["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert float(t["grad_clip_norm"]) == 1.0
    assert t["amp"] == "auto"


def test_pilot_fp_paths(pilot_fp_cfg):
    p = pilot_fp_cfg["paths"]
    assert p["processed_root"] == (
        "/content/drive/MyDrive/fm_fl_phmd/processed"
    )
    assert p["log_dir"] == (
        "/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated"
    )
    assert "fedprox_mu0_01" in p["checkpoint_dir"]


def test_pilot_fp_budget_dentro_del_cap(pilot_fp_cfg):
    f = pilot_fp_cfg["federated"]
    n_clients = 10
    total = int(f["n_rounds"]) * int(f["local_steps"]) * n_clients
    assert total == 5_000
    assert total <= _CAP_PILOT, total


# ----------------------------------------------------------------------
# AISLAMIENTO frente al FedAvg pilot/smoke (no debe pisar resultados)
# ----------------------------------------------------------------------


def test_fp_run_name_distinto_de_fedavg(
    smoke_fp_cfg, smoke_fa_cfg, pilot_fp_cfg, pilot_fa_cfg
):
    assert smoke_fp_cfg["run_name"] != smoke_fa_cfg["run_name"]
    assert pilot_fp_cfg["run_name"] != pilot_fa_cfg["run_name"]


def test_fp_checkpoint_dir_distinto_de_fedavg(
    smoke_fp_cfg, smoke_fa_cfg, pilot_fp_cfg, pilot_fa_cfg
):
    assert smoke_fp_cfg["paths"]["checkpoint_dir"] != \
        smoke_fa_cfg["paths"]["checkpoint_dir"]
    assert pilot_fp_cfg["paths"]["checkpoint_dir"] != \
        pilot_fa_cfg["paths"]["checkpoint_dir"]


def test_fp_resto_de_hyperparams_identicos_al_fedavg(
    smoke_fp_cfg, smoke_fa_cfg, pilot_fp_cfg, pilot_fa_cfg
):
    """Excluyendo `run_name`, `federated.algorithm`, `federated.fedprox_mu`
    y `paths.checkpoint_dir`, el resto del YAML debe coincidir con el
    config FedAvg homologo. Esto blinda que la unica variable cambiada
    en la ablacion es el algoritmo + mu.
    """
    def _strip_diffs(cfg: dict) -> dict:
        c = {k: v for k, v in cfg.items() if k != "run_name"}
        c["federated"] = {
            k: v for k, v in c.get("federated", {}).items()
            if k not in ("algorithm", "fedprox_mu")
        }
        c["paths"] = {
            k: v for k, v in c.get("paths", {}).items()
            if k != "checkpoint_dir"
        }
        return c

    assert _strip_diffs(smoke_fp_cfg) == _strip_diffs(smoke_fa_cfg)
    assert _strip_diffs(pilot_fp_cfg) == _strip_diffs(pilot_fa_cfg)
