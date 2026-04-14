"""Tests preflight del barrido β de FedAvgM (Fase 4e-lite).

Blinda los configs `ssl_federated_fedavgm_50x50_m0_{3,5,7}.yaml`
bit-a-bit. La única variable de la ablación entre estos configs y el
β=0.9 anterior es `server_momentum.beta` (+ run_name + checkpoint_dir).

Estructura paralela al test β=0.9 pero parametrizada por (fichero, beta).
Mantengo `test_fl_fedavgm_50x50_config.py` separado para no romper el
test inicial.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

# (fichero, beta_esperado, slug_run_name)
_BETA_CONFIGS = [
    ("ssl_federated_fedavgm_50x50_m0_3.yaml", 0.3, "m0_3"),
    ("ssl_federated_fedavgm_50x50_m0_5.yaml", 0.5, "m0_5"),
    ("ssl_federated_fedavgm_50x50_m0_7.yaml", 0.7, "m0_7"),
]

_CFG_BASE = _CFG_DIR / "ssl_federated_fedavg_50x50.yaml"


def _load(name: str) -> dict:
    p = _CFG_DIR / name
    assert p.is_file(), f"Falta config: {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cfg_base() -> dict:
    assert _CFG_BASE.is_file(), f"Falta config base: {_CFG_BASE}"
    return yaml.safe_load(_CFG_BASE.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Invariantes comunes a las 3 configs β del barrido
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_existe_y_es_yaml(name, beta, slug):
    assert isinstance(_load(name), dict)


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_run_name_codifica_slug(name, beta, slug):
    cfg = _load(name)
    assert cfg["run_name"] == f"ssl_federated_fedavgm_50x50_{slug}_patchtst_phm"


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_stage_full_y_budget(name, beta, slug):
    cfg = _load(name)
    assert cfg["stage"] == "full"
    f = cfg["federated"]
    assert int(f["n_rounds"]) == 50
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    total = int(f["n_rounds"]) * int(f["local_steps"]) * 10
    assert total == 25_000


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_algorithm_fedavgm(name, beta, slug):
    cfg = _load(name)
    f = cfg["federated"]
    assert str(f["algorithm"]).lower() == "fedavgm"
    assert f["fedprox_mu"] is None


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_server_momentum_beta_correcto(name, beta, slug):
    cfg = _load(name)
    sm = cfg["server_momentum"]
    assert float(sm["beta"]) == beta
    assert sm["nesterov"] is False
    assert str(sm["initialize"]).lower() == "zeros"


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_model_block_base(name, beta, slug):
    m = _load(name)["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_data_block(name, beta, slug):
    d = _load(name)["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_training_block(name, beta, slug):
    t = _load(name)["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert t["amp"] == "auto"


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_no_scheduler_block(name, beta, slug):
    """El barrido β NO debe tener scheduler (aislamos efecto momentum
    vs cosine; el run principal compara contra FedAvg constant)."""
    assert "lr_scheduler" not in _load(name)


# ----------------------------------------------------------------------
# Aislamiento: run_names y checkpoint_dirs únicos entre las 3 configs
# ----------------------------------------------------------------------


def test_run_names_unicos():
    names = [_load(c[0])["run_name"] for c in _BETA_CONFIGS]
    assert len(set(names)) == len(names), names


def test_checkpoint_dirs_unicos():
    dirs = [_load(c[0])["paths"]["checkpoint_dir"] for c in _BETA_CONFIGS]
    assert len(set(dirs)) == len(dirs), dirs


# ----------------------------------------------------------------------
# Comparación bit-a-bit contra FedAvg 50x50 constant
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_solo_cambia_algorithm_momentum_runname_y_ckpt(name, beta, slug, cfg_base):
    """La única variable entre cada config β y FedAvg constant es:
    algorithm + server_momentum + run_name + paths.checkpoint_dir.
    """
    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k not in ("run_name", "server_momentum")}
        cc["federated"] = {
            k: v for k, v in c.get("federated", {}).items() if k != "algorithm"
        }
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items() if k != "checkpoint_dir"
        }
        return cc
    assert _strip(_load(name)) == _strip(cfg_base)


# ----------------------------------------------------------------------
# Coherencia contra el config β=0.9 ya existente
# ----------------------------------------------------------------------


_CFG_M09 = _CFG_DIR / "ssl_federated_fedavgm_50x50_m0_9.yaml"


@pytest.mark.parametrize("name,beta,slug", _BETA_CONFIGS)
def test_solo_cambia_beta_runname_y_ckpt_vs_m09(name, beta, slug):
    """Entre el barrido β y el β=0.9, la única diferencia es
    server_momentum.beta + run_name + paths.checkpoint_dir.
    """
    if not _CFG_M09.is_file():
        pytest.skip(f"No existe ref m09: {_CFG_M09}")
    cfg_new = _load(name)
    cfg_m09 = yaml.safe_load(_CFG_M09.read_text(encoding="utf-8"))

    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k != "run_name"}
        cc["server_momentum"] = {
            k: v for k, v in c.get("server_momentum", {}).items() if k != "beta"
        }
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items() if k != "checkpoint_dir"
        }
        return cc
    assert _strip(cfg_new) == _strip(cfg_m09)
