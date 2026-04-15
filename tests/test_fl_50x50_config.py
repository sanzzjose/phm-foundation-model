"""Tests preflight del config federado FedAvg 50x50 (Fase 4b).

Valida que el config quede congelado bit-a-bit con la politica acordada antes
de lanzar el run real en Colab.

Invariantes compartidas con fedavg_25x50:
- modelo PatchTSTPhm base (d_model=128, 4 capas, 4 heads, P=16, N=32);
- data PRETRAIN_SOURCE / split train / adaptive_by_channels / cap 512 /
  final_client_weight con caps audit v2.3;
- algorithm=fedavg, fedprox_mu=null.

Especifico de 50x50:
- n_rounds=50, local_steps=50, ckpt_every_rounds=10;
- 50*50*10 = 25.000 local steps > 20.000 (cap pilot) -> exige stage=full
  (cap 500.000);
- run_name y checkpoint_dir DIFERENTES del 25x50.

Independiente de torch (solo yaml + asserts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_CFG_50 = _CFG_DIR / "ssl_federated_fedavg_50x50.yaml"
_CFG_25 = _CFG_DIR / "ssl_federated_fedavg_25x50.yaml"

# Caps duros (training/train_ssl_federated.STAGE_MAX_LOCAL_STEPS).
_CAP_PILOT = 20_000
_CAP_FULL = 500_000


@pytest.fixture(scope="module")
def cfg50() -> dict:
    assert _CFG_50.is_file(), f"Falta config 50x50: {_CFG_50}"
    return yaml.safe_load(_CFG_50.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg25() -> dict:
    assert _CFG_25.is_file(), f"Falta config 25x50: {_CFG_25}"
    return yaml.safe_load(_CFG_25.read_text(encoding="utf-8"))


def test_existe_y_es_yaml(cfg50):
    assert isinstance(cfg50, dict)


def test_stage_full(cfg50):
    """50x50 = 25.000 > cap pilot 20.000 -> debe usar stage=full."""
    assert cfg50["stage"] == "full"


def test_run_name(cfg50):
    assert cfg50["run_name"] == "ssl_federated_fedavg_50x50_patchtst_phm"


def test_federated_budget_50x50(cfg50):
    f = cfg50["federated"]
    assert int(f["n_rounds"]) == 50
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    assert int(f["ckpt_every_rounds"]) == 10
    total = int(f["n_rounds"]) * int(f["local_steps"]) * 10
    assert total == 25_000
    assert total > _CAP_PILOT, "50x50 supera el cap pilot, exige stage=full"
    assert total <= _CAP_FULL, total


def test_algorithm_fedavg(cfg50):
    f = cfg50["federated"]
    assert str(f["algorithm"]).lower() == "fedavg"
    assert f["fedprox_mu"] is None


def test_model_block_base(cfg50):
    m = cfg50["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["n_layers"] == 4
    assert m["n_heads"] == 4
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


def test_data_block(cfg50):
    d = cfg50["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["split"] == "train"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert int(d["min_batch_size"]) == 1
    assert d["client_source"] == "results/audit/audit_groups.json"
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


def test_training_block(cfg50):
    t = cfg50["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert float(t["grad_clip_norm"]) == 1.0
    assert t["amp"] == "auto"


def test_seed_42(cfg50):
    assert int(cfg50["seed"]) == 42


def test_run_name_distinto_del_25x50(cfg50, cfg25):
    assert cfg50["run_name"] != cfg25["run_name"]


def test_checkpoint_dir_distinto_del_25x50(cfg50, cfg25):
    assert cfg50["paths"]["checkpoint_dir"] != cfg25["paths"]["checkpoint_dir"]


def test_log_dir_compartido_con_25x50(cfg50, cfg25):
    # mismo arbol; run_name discrimina.
    assert cfg50["paths"]["log_dir"] == cfg25["paths"]["log_dir"]


def test_solo_cambia_rounds_stage_y_paths(cfg50, cfg25):
    """Salvo n_rounds (25->50), stage (pilot->full), run_name y
    checkpoint_dir (y ckpt_every_rounds que se ajusta a 10 con 50 rondas),
    todo lo demas debe coincidir con fedavg_25x50."""
    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k not in ("run_name", "stage")}
        cc["federated"] = {
            k: v for k, v in c.get("federated", {}).items()
            if k not in ("n_rounds", "ckpt_every_rounds")
        }
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items()
            if k != "checkpoint_dir"
        }
        return cc
    assert _strip(cfg50) == _strip(cfg25)
