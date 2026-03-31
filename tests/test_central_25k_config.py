"""Tests preflight del config central budget-matched 25k (Fase 4b).

Valida que el config quede congelado bit-a-bit con la receta central full
salvo los hiperparams que dependen del numero de steps (max_steps,
warmup_steps, checkpoint_every, distribution_log_every, log_every) y los
paths.

Comparacion budget-matched principal:
  central_25k vs fedavg_50x50 (ambos 25.000 local steps).

Independiente de torch (solo yaml + asserts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_CFG_25K = _CFG_DIR / "ssl_central_25k_budget_matched.yaml"
_CFG_FULL = _CFG_DIR / "ssl_central_full.yaml"

# Caps duros (training/train_ssl_central._validate_train_config).
_CAP_FULL_STAGE = 500_000


@pytest.fixture(scope="module")
def cfg25k() -> dict:
    assert _CFG_25K.is_file(), f"Falta config: {_CFG_25K}"
    return yaml.safe_load(_CFG_25K.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg_full() -> dict:
    assert _CFG_FULL.is_file(), f"Falta config: {_CFG_FULL}"
    return yaml.safe_load(_CFG_FULL.read_text(encoding="utf-8"))


def test_existe_y_es_yaml(cfg25k):
    assert isinstance(cfg25k, dict)


def test_stage_full_para_25k(cfg25k):
    """25.000 > caps de coverage/pilot (10.000 cada uno) -> exige stage=full."""
    assert cfg25k["stage"] == "full"


def test_max_steps_25000(cfg25k):
    assert int(cfg25k["training"]["max_steps"]) == 25_000
    assert int(cfg25k["training"]["max_steps"]) <= _CAP_FULL_STAGE


def test_run_name(cfg25k):
    assert cfg25k["run_name"] == "ssl_central_25k_budget_matched_patchtst_phm"


def test_model_block_base(cfg25k):
    m = cfg25k["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["n_layers"] == 4
    assert m["n_heads"] == 4
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


def test_ssl_block(cfg25k):
    s = cfg25k["ssl"]
    assert s["objective"] == "masked_patch_prediction"
    assert float(s["mask_ratio"]) == 0.3
    assert s["dynamic_masking"] is True
    assert s["loss"] == "mse"


def test_data_block(cfg25k):
    d = cfg25k["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["split"] == "train"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert int(d["min_batch_size"]) == 1
    assert d["sampling_strategy"] == "weighted"
    assert float(d["min_dataset_presence"]) == 0.001


def test_training_block(cfg25k):
    t = cfg25k["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert float(t["grad_clip_norm"]) == 1.0
    assert int(t["warmup_steps"]) == 500  # 2% de 25.000
    assert t["schedule"] == "cosine"
    assert t["amp"] == "auto"


def test_seed_42(cfg25k):
    assert int(cfg25k["seed"]) == 42


def test_checkpoint_dir_distinto_del_full(cfg25k, cfg_full):
    assert cfg25k["paths"]["checkpoint_dir"] != cfg_full["paths"]["checkpoint_dir"]
    assert "25k" in cfg25k["paths"]["checkpoint_dir"]


def test_solo_cambian_steps_y_paths_vs_full(cfg25k, cfg_full):
    """Salvo run_name, training.{max_steps, warmup_steps, checkpoint_every,
    distribution_log_every, log_every} y paths.checkpoint_dir, todo lo demas
    debe coincidir con ssl_central_full (misma receta arquitectura/data)."""
    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k != "run_name"}
        cc["training"] = {
            k: v for k, v in c.get("training", {}).items()
            if k not in ("max_steps", "warmup_steps", "checkpoint_every",
                         "distribution_log_every", "log_every")
        }
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items()
            if k != "checkpoint_dir"
        }
        return cc
    assert _strip(cfg25k) == _strip(cfg_full)
