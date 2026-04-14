"""Tests preflight de los configs federados escalados 25x50 (Fase 4a).

Validan que los configs del primer escalado federado (FedAvg 25x50 +
barrido FedProx mu en {0.001, 0.01, 0.05, 0.1}) queden congelados
bit-a-bit con la politica acordada antes de lanzar runs reales en Colab.

Invariantes compartidas con el pilot 10x50 (mismo modelo, mismos 36 PS,
misma topologia de 10 clientes, mismo sampler con caps audit v2.3):

- modelo PatchTSTPhm base (d_model=128, 4 capas, 4 heads, P=16, N=32);
- data PRETRAIN_SOURCE / split train / adaptive_by_channels / cap 512 /
  final_client_weight;
- n_rounds=25, local_steps=50, clients_per_round=all, ckpt_every_rounds=5;
- stage=pilot y budget 25*50*10 = 12 500 <= cap pilot 20 000.

Especifico del escalado:

- FedAvg 25x50: algorithm=fedavg, fedprox_mu=null.
- FedProx 25x50: algorithm=fedprox y fedprox_mu == el del nombre.
- run_name y checkpoint_dir UNICOS por config (no se pisan en Drive).
- salvo run_name, algorithm, fedprox_mu y checkpoint_dir, el resto del
  YAML coincide con el FedAvg 25x50 (la unica variable de la ablacion es
  el algoritmo + mu).

Independientes de torch (solo yaml + asserts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_CAP_PILOT = 20_000

# (fichero, mu_esperado). mu None == FedAvg.
_FEDPROX_CONFIGS = {
    "ssl_federated_fedprox_mu0_001_25x50.yaml": 0.001,
    "ssl_federated_fedprox_mu0_01_25x50.yaml": 0.01,
    "ssl_federated_fedprox_mu0_05_25x50.yaml": 0.05,
    "ssl_federated_fedprox_mu0_1_25x50.yaml": 0.1,
}
_FEDAVG_CONFIG = "ssl_federated_fedavg_25x50.yaml"
_ALL_CONFIGS = [_FEDAVG_CONFIG, *_FEDPROX_CONFIGS.keys()]


def _load(name: str) -> dict:
    p = _CFG_DIR / name
    assert p.is_file(), f"Falta config 25x50: {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fedavg_cfg() -> dict:
    return _load(_FEDAVG_CONFIG)


# ----------------------------------------------------------------------
# Invariantes comunes a TODOS los configs 25x50
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_existe_y_es_yaml(name):
    assert isinstance(_load(name), dict)


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_stage_pilot(name):
    assert _load(name)["stage"] == "pilot"


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_federated_budget_25x50(name):
    f = _load(name)["federated"]
    assert int(f["n_rounds"]) == 25
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    assert int(f["ckpt_every_rounds"]) == 5
    total = int(f["n_rounds"]) * int(f["local_steps"]) * 10
    assert total == 12_500
    assert total <= _CAP_PILOT, total


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_model_block_base(name):
    m = _load(name)["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["n_layers"] == 4
    assert m["n_heads"] == 4
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_data_block(name):
    d = _load(name)["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["split"] == "train"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert int(d["min_batch_size"]) == 1
    assert d["client_source"] == "results/audit/audit_groups.json"
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_training_block(name):
    t = _load(name)["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert float(t["grad_clip_norm"]) == 1.0
    assert t["amp"] == "auto"


@pytest.mark.parametrize("name", _ALL_CONFIGS)
def test_seed_42(name):
    assert int(_load(name)["seed"]) == 42


# ----------------------------------------------------------------------
# Especifico FedAvg vs FedProx
# ----------------------------------------------------------------------


def test_fedavg_algorithm(fedavg_cfg):
    f = fedavg_cfg["federated"]
    assert str(f["algorithm"]).lower() == "fedavg"
    assert f["fedprox_mu"] is None
    assert fedavg_cfg["run_name"] == "ssl_federated_fedavg_25x50_patchtst_phm"


@pytest.mark.parametrize("name,mu", list(_FEDPROX_CONFIGS.items()))
def test_fedprox_algorithm_y_mu(name, mu):
    cfg = _load(name)
    f = cfg["federated"]
    assert str(f["algorithm"]).lower() == "fedprox"
    assert float(f["fedprox_mu"]) == mu
    # el run_name debe codificar el mu para no colisionar
    assert "fedprox" in cfg["run_name"]
    assert "25x50" in cfg["run_name"]


# ----------------------------------------------------------------------
# Aislamiento: run_name y checkpoint_dir unicos entre los 5 configs
# ----------------------------------------------------------------------


def test_run_names_unicos():
    names = [_load(c)["run_name"] for c in _ALL_CONFIGS]
    assert len(set(names)) == len(names), names


def test_checkpoint_dirs_unicos():
    dirs = [_load(c)["paths"]["checkpoint_dir"] for c in _ALL_CONFIGS]
    assert len(set(dirs)) == len(dirs), dirs


def test_log_dir_compartido():
    # El log_dir es el mismo arbol; el run_name discrimina cada run.
    log_dirs = {_load(c)["paths"]["log_dir"] for c in _ALL_CONFIGS}
    assert log_dirs == {
        "/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated"
    }


# ----------------------------------------------------------------------
# La unica variable de la ablacion: algorithm + mu (+ run_name/ckpt_dir)
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_FEDPROX_CONFIGS.keys()))
def test_resto_identico_al_fedavg(name, fedavg_cfg):
    """Excluyendo run_name, federated.algorithm, federated.fedprox_mu y
    paths.checkpoint_dir, el resto del YAML debe coincidir con el FedAvg
    25x50 homologo."""
    def _strip(cfg: dict) -> dict:
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

    assert _strip(_load(name)) == _strip(fedavg_cfg)
