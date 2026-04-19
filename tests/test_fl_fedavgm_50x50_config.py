"""Tests preflight del config FedAvgM 50x50 m=0.9 (Fase 4d).

Blinda la receta bit-a-bit:
- arquitectura base + 36 PS + 10 clientes + caps.
- algorithm=fedavgm, fedprox_mu=null.
- server_momentum.beta=0.9, nesterov=false, initialize=zeros.
- n_rounds=50, local_steps=50, stage=full.
- Salvo algorithm + server_momentum + run_name + checkpoint_dir, el resto
  coincide con ssl_federated_fedavg_50x50.yaml (la unica variable de la
  ablacion es el momentum del servidor).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_CFG_MOM = _CFG_DIR / "ssl_federated_fedavgm_50x50_m0_9.yaml"
_CFG_BASE = _CFG_DIR / "ssl_federated_fedavg_50x50.yaml"


@pytest.fixture(scope="module")
def cfg_mom() -> dict:
    assert _CFG_MOM.is_file(), f"Falta config: {_CFG_MOM}"
    return yaml.safe_load(_CFG_MOM.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg_base() -> dict:
    assert _CFG_BASE.is_file(), f"Falta config base: {_CFG_BASE}"
    return yaml.safe_load(_CFG_BASE.read_text(encoding="utf-8"))


def test_existe_y_es_yaml(cfg_mom):
    assert isinstance(cfg_mom, dict)


def test_run_name_y_stage(cfg_mom):
    assert cfg_mom["run_name"] == "ssl_federated_fedavgm_50x50_m0_9_patchtst_phm"
    assert cfg_mom["stage"] == "full"


def test_federated_algorithm_fedavgm(cfg_mom):
    f = cfg_mom["federated"]
    assert str(f["algorithm"]).lower() == "fedavgm"
    assert f["fedprox_mu"] is None
    assert int(f["n_rounds"]) == 50
    assert int(f["local_steps"]) == 50


def test_server_momentum_block(cfg_mom):
    sm = cfg_mom["server_momentum"]
    assert float(sm["beta"]) == 0.9
    assert sm["nesterov"] is False
    assert str(sm["initialize"]).lower() == "zeros"


def test_model_block_base(cfg_mom):
    m = cfg_mom["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


def test_data_block(cfg_mom):
    d = cfg_mom["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


def test_training_block(cfg_mom):
    t = cfg_mom["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert t["amp"] == "auto"


def test_no_scheduler_block(cfg_mom):
    """FedAvgM 50x50 m0_9 NO debe tener lr_scheduler (queremos aislar
    el efecto momentum vs scheduler; este config compara contra el
    fedavg_50x50 constant, no contra el cosine)."""
    assert "lr_scheduler" not in cfg_mom


def test_run_name_y_ckpt_dir_unicos(cfg_mom, cfg_base):
    assert cfg_mom["run_name"] != cfg_base["run_name"]
    assert cfg_mom["paths"]["checkpoint_dir"] != cfg_base["paths"]["checkpoint_dir"]
    assert "fedavgm" in cfg_mom["paths"]["checkpoint_dir"]


def test_solo_cambia_algorithm_momentum_runname_y_ckpt(cfg_mom, cfg_base):
    """La ablacion solo introduce algorithm, server_momentum, run_name y
    checkpoint_dir; el resto del YAML coincide con fedavg_50x50."""
    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k not in ("run_name", "server_momentum")}
        cc["federated"] = {
            k: v for k, v in c.get("federated", {}).items() if k != "algorithm"
        }
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items() if k != "checkpoint_dir"
        }
        return cc
    assert _strip(cfg_mom) == _strip(cfg_base)


def test_budget_25k(cfg_mom):
    f = cfg_mom["federated"]
    assert int(f["n_rounds"]) * int(f["local_steps"]) * 10 == 25_000
