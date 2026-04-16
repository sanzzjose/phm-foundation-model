"""Tests preflight del config FedAvg 50x50 + cosine warmup (Fase 4c).

Blinda la receta del config nuevo bit-a-bit:
- arquitectura base + 36 PS + 10 clientes + same caps.
- algorithm=fedavg, fedprox_mu=null.
- n_rounds=50, local_steps=50 -> 25.000 local steps.
- stage=full (cap 500k).
- lr_scheduler con cosine + warmup_steps=2000 + total_steps=100000.

Independiente de torch (solo yaml + asserts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_CFG_COSINE = _CFG_DIR / "ssl_federated_fedavg_50x50_cosine_warmup.yaml"
_CFG_CONST = _CFG_DIR / "ssl_federated_fedavg_50x50.yaml"


@pytest.fixture(scope="module")
def cfg_cos() -> dict:
    assert _CFG_COSINE.is_file(), f"Falta config: {_CFG_COSINE}"
    return yaml.safe_load(_CFG_COSINE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg_const() -> dict:
    assert _CFG_CONST.is_file(), f"Falta config base: {_CFG_CONST}"
    return yaml.safe_load(_CFG_CONST.read_text(encoding="utf-8"))


def test_existe_y_es_yaml(cfg_cos):
    assert isinstance(cfg_cos, dict)


def test_run_name_y_stage(cfg_cos):
    assert cfg_cos["run_name"] == "ssl_federated_fedavg_50x50_cosine_warmup_patchtst_phm"
    assert cfg_cos["stage"] == "full"


def test_federated_budget(cfg_cos):
    f = cfg_cos["federated"]
    assert int(f["n_rounds"]) == 50
    assert int(f["local_steps"]) == 50
    assert str(f["algorithm"]).lower() == "fedavg"
    assert f["fedprox_mu"] is None
    total = int(f["n_rounds"]) * int(f["local_steps"]) * 10
    assert total == 25_000


def test_lr_scheduler_cosine_warmup(cfg_cos):
    sch = cfg_cos["lr_scheduler"]
    assert str(sch["type"]).lower() == "cosine"
    assert int(sch["warmup_steps"]) == 2000
    assert int(sch["total_steps"]) == 100_000
    assert sch["step_accounting"] == "aggregate_local_updates"
    assert sch["same_lr_for_all_clients_in_round"] is True


def test_model_base(cfg_cos):
    m = cfg_cos["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


def test_data_block(cfg_cos):
    d = cfg_cos["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


def test_training_block(cfg_cos):
    t = cfg_cos["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert t["amp"] == "auto"


def test_paths_distintos_del_constant(cfg_cos, cfg_const):
    assert cfg_cos["run_name"] != cfg_const["run_name"]
    assert cfg_cos["paths"]["checkpoint_dir"] != cfg_const["paths"]["checkpoint_dir"]
    assert "cosine_warmup" in cfg_cos["paths"]["checkpoint_dir"]


def test_constant_config_no_tiene_scheduler(cfg_const):
    """El config 50x50 constante NO debe tener bloque lr_scheduler
    (backward-compat: ausente = LR constante)."""
    assert "lr_scheduler" not in cfg_const


def test_solo_cambia_scheduler_run_name_y_ckpt_dir(cfg_cos, cfg_const):
    """Salvo run_name, paths.checkpoint_dir y lr_scheduler (presente solo
    en el config cosine), el resto debe ser identico."""
    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k not in ("run_name", "lr_scheduler")}
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items()
            if k != "checkpoint_dir"
        }
        return cc
    assert _strip(cfg_cos) == _strip(cfg_const)
