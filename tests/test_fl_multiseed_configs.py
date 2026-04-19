"""Tests preflight de los runs multi-seed (Fase 4f).

Blinda los 4 configs multi-seed:
  - ssl_federated_fedavg_50x50_seed{43,44}.yaml
  - ssl_federated_fedavgm_50x50_m0_7_seed{43,44}.yaml

La unica variable entre cada multi-seed y su base (seed=42) es:
  seed + run_name + paths.checkpoint_dir.

Asi nos aseguramos de que el barrido de seeds NO introduce cambios
silenciosos en la receta (algorithm, server_momentum, modelo, sampler,
etc.).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

# (fichero, base, seed_esperado, slug_run_name, algorithm_esperado, beta_esperado)
# Si no hay momentum (FedAvg), beta_esperado=None.
_MULTISEED_CONFIGS = [
    (
        "ssl_federated_fedavg_50x50_seed43.yaml",
        "ssl_federated_fedavg_50x50.yaml",
        43, "seed43", "fedavg", None,
    ),
    (
        "ssl_federated_fedavg_50x50_seed44.yaml",
        "ssl_federated_fedavg_50x50.yaml",
        44, "seed44", "fedavg", None,
    ),
    (
        "ssl_federated_fedavgm_50x50_m0_7_seed43.yaml",
        "ssl_federated_fedavgm_50x50_m0_7.yaml",
        43, "m0_7_seed43", "fedavgm", 0.7,
    ),
    (
        "ssl_federated_fedavgm_50x50_m0_7_seed44.yaml",
        "ssl_federated_fedavgm_50x50_m0_7.yaml",
        44, "m0_7_seed44", "fedavgm", 0.7,
    ),
]


def _load(name: str) -> dict:
    p = _CFG_DIR / name
    assert p.is_file(), f"Falta config: {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Invariantes basicos de cada config multi-seed
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_existe_y_es_yaml(name, base, seed, slug, algo, beta):
    assert isinstance(_load(name), dict)


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_seed_correcto(name, base, seed, slug, algo, beta):
    assert int(_load(name)["seed"]) == seed


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_stage_full_y_budget(name, base, seed, slug, algo, beta):
    cfg = _load(name)
    assert cfg["stage"] == "full"
    f = cfg["federated"]
    assert int(f["n_rounds"]) == 50
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    total = int(f["n_rounds"]) * int(f["local_steps"]) * 10
    assert total == 25_000


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_algorithm_correcto(name, base, seed, slug, algo, beta):
    f = _load(name)["federated"]
    assert str(f["algorithm"]).lower() == algo
    assert f["fedprox_mu"] is None


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_server_momentum_coherente(name, base, seed, slug, algo, beta):
    cfg = _load(name)
    if beta is None:
        assert "server_momentum" not in cfg
    else:
        sm = cfg["server_momentum"]
        assert float(sm["beta"]) == beta
        assert sm["nesterov"] is False
        assert str(sm["initialize"]).lower() == "zeros"


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_run_name_codifica_slug_y_seed(name, base, seed, slug, algo, beta):
    cfg = _load(name)
    assert cfg["run_name"].endswith("_patchtst_phm")
    assert slug in cfg["run_name"]
    assert f"seed{seed}" in cfg["run_name"]


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_checkpoint_dir_dedicado(name, base, seed, slug, algo, beta):
    cfg = _load(name)
    ckpt = cfg["paths"]["checkpoint_dir"]
    assert f"seed{seed}" in ckpt, (
        f"checkpoint_dir debe contener 'seed{seed}' para no pisar la"
        f" version seed=42: {ckpt}"
    )


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_model_block_base(name, base, seed, slug, algo, beta):
    m = _load(name)["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_data_block(name, base, seed, slug, algo, beta):
    d = _load(name)["data"]
    assert d["role"] == "PRETRAIN_SOURCE"
    assert d["batch_size_policy"] == "adaptive_by_channels"
    assert int(d["max_channel_batch"]) == 512
    assert d["client_sampling_strategy"] == "weighted"
    assert d["aggregation_weight_policy"] == "final_client_weight"


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_training_block(name, base, seed, slug, algo, beta):
    t = _load(name)["training"]
    assert float(t["lr"]) == 0.0003
    assert float(t["weight_decay"]) == 0.05
    assert t["amp"] == "auto"


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_no_scheduler_block(name, base, seed, slug, algo, beta):
    """Ni FedAvg constant ni FedAvgM beta=0.7 usan scheduler aqui."""
    assert "lr_scheduler" not in _load(name)


# ----------------------------------------------------------------------
# Aislamiento: run_names y checkpoint_dirs unicos entre las 4 configs
# ----------------------------------------------------------------------


def test_run_names_unicos():
    names = [_load(c[0])["run_name"] for c in _MULTISEED_CONFIGS]
    assert len(set(names)) == len(names), names


def test_checkpoint_dirs_unicos():
    dirs = [_load(c[0])["paths"]["checkpoint_dir"] for c in _MULTISEED_CONFIGS]
    assert len(set(dirs)) == len(dirs), dirs


# ----------------------------------------------------------------------
# Comparacion bit-a-bit contra el config base correspondiente
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,base,seed,slug,algo,beta", _MULTISEED_CONFIGS)
def test_solo_cambia_seed_runname_y_ckpt_vs_base(
    name, base, seed, slug, algo, beta,
):
    """La unica variable entre cada multi-seed y su base es:
    seed + run_name + paths.checkpoint_dir. Todo lo demas igual."""
    cfg_new = _load(name)
    cfg_base = _load(base)

    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k not in ("run_name", "seed")}
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items() if k != "checkpoint_dir"
        }
        return cc

    assert _strip(cfg_new) == _strip(cfg_base)
