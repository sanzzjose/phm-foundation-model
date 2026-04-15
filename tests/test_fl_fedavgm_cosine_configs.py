"""Tests preflight de los configs FedAvgM beta=0.7 + cosine (Fase 4f.bis).

Blinda los 3 configs cosine multi-seed:
  - ssl_federated_fedavgm_50x50_m0_7_cosine.yaml        (seed=42)
  - ssl_federated_fedavgm_50x50_m0_7_cosine_seed43.yaml (seed=43)
  - ssl_federated_fedavgm_50x50_m0_7_cosine_seed44.yaml (seed=44)

Dos invariantes criticos:
  1. Los 3 cosine son identicos entre si salvo seed + run_name +
     paths.checkpoint_dir (mismo patron que el multi-seed de Fase 4f).
  2. El cosine seed=42 es identico al beta=0.7 SIN scheduler
     (ssl_federated_fedavgm_50x50_m0_7.yaml) salvo run_name +
     paths.checkpoint_dir + el bloque lr_scheduler anadido. Es decir: la
     UNICA diferencia de receta entre 4f.bis y 4f-beta0.7 es el cosine.

Independiente de torch (solo yaml + asserts).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

# Base sin scheduler (beta=0.7 constante, Fase 4e-lite / 4f seed=42).
_BASE_NOSCHED = "ssl_federated_fedavgm_50x50_m0_7.yaml"

# (fichero, seed_esperado, slug_en_run_name)
_COSINE_CONFIGS = [
    ("ssl_federated_fedavgm_50x50_m0_7_cosine.yaml", 42, "m0_7_cosine"),
    ("ssl_federated_fedavgm_50x50_m0_7_cosine_seed43.yaml", 43, "m0_7_cosine_seed43"),
    ("ssl_federated_fedavgm_50x50_m0_7_cosine_seed44.yaml", 44, "m0_7_cosine_seed44"),
]


def _load(name: str) -> dict:
    p = _CFG_DIR / name
    assert p.is_file(), f"Falta config: {p}"
    return yaml.safe_load(p.read_text(encoding="utf-8"))


# ----------------------------------------------------------------------
# Invariantes basicos de cada config cosine
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_existe_y_es_yaml(name, seed, slug):
    assert isinstance(_load(name), dict)


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_seed_correcto(name, seed, slug):
    assert int(_load(name)["seed"]) == seed


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_stage_full_y_budget(name, seed, slug):
    cfg = _load(name)
    assert cfg["stage"] == "full"
    f = cfg["federated"]
    assert int(f["n_rounds"]) == 50
    assert int(f["local_steps"]) == 50
    assert f["clients_per_round"] == "all"
    assert int(f["n_rounds"]) * int(f["local_steps"]) * 10 == 25_000


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_algorithm_fedavgm(name, seed, slug):
    f = _load(name)["federated"]
    assert str(f["algorithm"]).lower() == "fedavgm"
    assert f["fedprox_mu"] is None


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_server_momentum_beta07(name, seed, slug):
    sm = _load(name)["server_momentum"]
    assert float(sm["beta"]) == 0.7
    assert sm["nesterov"] is False
    assert str(sm["initialize"]).lower() == "zeros"


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_lr_scheduler_cosine_warmup(name, seed, slug):
    sch = _load(name)["lr_scheduler"]
    assert str(sch["type"]).lower() == "cosine"
    assert int(sch["warmup_steps"]) == 2000
    assert int(sch["total_steps"]) == 100_000
    assert sch["step_accounting"] == "aggregate_local_updates"
    assert sch["same_lr_for_all_clients_in_round"] is True


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_run_name_codifica_slug_y_seed(name, seed, slug):
    cfg = _load(name)
    assert cfg["run_name"].endswith("_patchtst_phm")
    assert slug in cfg["run_name"]
    if seed != 42:
        assert f"seed{seed}" in cfg["run_name"]


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_checkpoint_dir_dedicado(name, seed, slug):
    ckpt = _load(name)["paths"]["checkpoint_dir"]
    assert "m0_7_cosine" in ckpt
    if seed != 42:
        assert f"seed{seed}" in ckpt


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS)
def test_model_block_base(name, seed, slug):
    m = _load(name)["model"]
    assert m["name"] == "patchtst_phm_base"
    assert m["d_model"] == 128
    assert m["patch_size"] == 16
    assert m["n_patches"] == 32


# ----------------------------------------------------------------------
# Aislamiento: run_names y checkpoint_dirs unicos entre los 3 cosine
# ----------------------------------------------------------------------


def test_run_names_unicos():
    names = [_load(c[0])["run_name"] for c in _COSINE_CONFIGS]
    assert len(set(names)) == len(names), names


def test_checkpoint_dirs_unicos():
    dirs = [_load(c[0])["paths"]["checkpoint_dir"] for c in _COSINE_CONFIGS]
    assert len(set(dirs)) == len(dirs), dirs


# ----------------------------------------------------------------------
# Invariante 1: los 3 cosine solo cambian seed + run_name + ckpt_dir
# ----------------------------------------------------------------------


@pytest.mark.parametrize("name,seed,slug", _COSINE_CONFIGS[1:])
def test_seed43_44_solo_cambian_seed_runname_ckpt_vs_seed42(name, seed, slug):
    cfg_new = _load(name)
    cfg_base = _load(_COSINE_CONFIGS[0][0])  # cosine seed=42

    def _strip(c: dict) -> dict:
        cc = {k: v for k, v in c.items() if k not in ("run_name", "seed")}
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items() if k != "checkpoint_dir"
        }
        return cc

    assert _strip(cfg_new) == _strip(cfg_base)


# ----------------------------------------------------------------------
# Invariante 2: cosine seed=42 == beta=0.7 sin scheduler + SOLO el cosine
# ----------------------------------------------------------------------


def test_cosine_seed42_solo_anade_scheduler_vs_base_noschd():
    """La unica diferencia de receta entre el cosine seed=42 y el
    beta=0.7 constante (ambos seed=42) es el bloque lr_scheduler
    anadido. run_name y checkpoint_dir tambien cambian (obvio)."""
    cfg_cos = _load(_COSINE_CONFIGS[0][0])
    cfg_base = _load(_BASE_NOSCHED)

    # El base NO debe tener scheduler (backward-compat).
    assert "lr_scheduler" not in cfg_base
    # El cosine SI.
    assert "lr_scheduler" in cfg_cos

    def _strip(c: dict) -> dict:
        cc = {
            k: v for k, v in c.items()
            if k not in ("run_name", "lr_scheduler")
        }
        cc["paths"] = {
            k: v for k, v in c.get("paths", {}).items() if k != "checkpoint_dir"
        }
        return cc

    assert _strip(cfg_cos) == _strip(cfg_base)
