"""Tests de los 4 configs oficiales del downstream federado pilot
(commit FL-DOWN-pilot, Fase 4 del bloque federado).

Valida sin torch:

- existen los 4 YAMLs;
- `run_name` unico entre los 4;
- 2 datasets exactamente (CWRU y HSG18, 2 modos cada uno);
- backbone PatchTST base identico en los 4;
- batching policy adaptive_by_channels con cap 512;
- metric_for_best = macro_f1_val, head_dropout = 0.1, lr_head = 1e-3;
- lr_backbone canonico segun modo:
    linear_probing      -> 1e-4 (canonico aunque el backbone este
                            congelado en linear; el trainer hace
                            float(.get("lr_backbone", 0.0)) y no acepta
                            null);
    full_finetuning     -> 1e-5 (sweet spot validado en CWRU + HSG18
                            con SSL central);
- paths.log_dir y paths.checkpoint_dir contienen
  `downstream_federated_pilot` (carpeta separada del downstream central
  para no pisar runs previos).

Este test es la red de seguridad ante cambios accidentales antes de
lanzar las 4 corridas oficiales (~3.5 h en A100).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_DIR = REPO_ROOT / "training" / "configs"

CFGS = {
    "cwru_linear_probing": (
        CFG_DIR / "downstream_cwru_fedavg_pilot_linear_probing.yaml"
    ),
    "cwru_full_finetuning_lr1e-5": (
        CFG_DIR / "downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5.yaml"
    ),
    "hsg18_linear_probing": (
        CFG_DIR / "downstream_hsg18_fedavg_pilot_linear_probing.yaml"
    ),
    "hsg18_full_finetuning_lr1e-5": (
        CFG_DIR / "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5.yaml"
    ),
}


def _load_all():
    out = {}
    for name, p in CFGS.items():
        assert p.is_file(), f"Falta config oficial federada: {p}"
        out[name] = yaml.safe_load(p.read_text(encoding="utf-8"))
    return out


def test_existen_los_4_configs():
    """Pre-check: los 4 YAMLs estan en disco."""
    for name, p in CFGS.items():
        assert p.is_file(), f"Falta {p}"


def test_run_name_unico_en_los_4():
    """Cada modo necesita run_name distinto para que el trainer no
    pise logs ni checkpoints entre corridas."""
    cfgs = _load_all()
    run_names = [c["run_name"] for c in cfgs.values()]
    assert len(set(run_names)) == 4, f"run_name duplicado: {run_names}"
    # Naming canonico:
    assert cfgs["cwru_linear_probing"]["run_name"] == \
        "downstream_cwru_fedavg_pilot_linear_probing"
    assert cfgs["cwru_full_finetuning_lr1e-5"]["run_name"] == \
        "downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5"
    assert cfgs["hsg18_linear_probing"]["run_name"] == \
        "downstream_hsg18_fedavg_pilot_linear_probing"
    assert cfgs["hsg18_full_finetuning_lr1e-5"]["run_name"] == \
        "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5"


def test_datasets_cwru_hsg18_exactos():
    """2 configs CWRU, 2 configs HSG18, en ambos sitios (data.dataset y
    nivel top-level del YAML)."""
    cfgs = _load_all()
    cwru = [c for k, c in cfgs.items() if k.startswith("cwru_")]
    hsg = [c for k, c in cfgs.items() if k.startswith("hsg18_")]
    assert len(cwru) == 2
    assert len(hsg) == 2
    for c in cwru:
        assert c["dataset"] == "CWRU"
        assert c["data"]["dataset"] == "CWRU"
    for c in hsg:
        assert c["dataset"] == "HSG18"
        assert c["data"]["dataset"] == "HSG18"


def test_task_classification_multiclass_en_los_4():
    for c in _load_all().values():
        assert c["task"] == "classification_multiclass"


def test_modelo_patchtst_base_identico_en_los_4():
    """Los 4 configs deben usar el MISMO backbone PatchTSTPhm base; lo
    unico que cambia entre corridas es la cabeza, el dataset y los LRs
    del optimizer.
    """
    cfgs = _load_all()
    expected_model = {
        "name": "patchtst_phm_base",
        "d_model": 128,
        "n_layers": 4,
        "n_heads": 4,
        "d_ff": 512,
        "dropout": 0.1,
        "patch_size": 16,
        "n_patches": 32,
    }
    for name, c in cfgs.items():
        for k, v in expected_model.items():
            assert c["model"][k] == v, (
                f"{name}: model.{k}={c['model'].get(k)} != esperado {v}"
            )


def test_data_policy_y_caps():
    """batch_size_policy=adaptive_by_channels, max_channel_batch=512,
    min_batch_size=1 en los 4 (mismo que la version central).
    """
    cfgs = _load_all()
    for name, c in cfgs.items():
        assert c["data"]["batch_size_policy"] == "adaptive_by_channels", name
        assert int(c["data"]["max_channel_batch"]) == 512, name
        assert int(c["data"]["min_batch_size"]) == 1, name
        assert int(c["data"]["batch_size"]) == 64, name
        # processed_root canonico (no toca processed_downstream).
        assert c["data"]["processed_root"] == \
            "/content/drive/MyDrive/fm_fl_phmd/processed"


def test_training_metric_dropout_lr_head():
    """metric_for_best=macro_f1_val (mayor = mejor), head_dropout=0.1,
    lr_head=1e-3 en los 4.
    """
    cfgs = _load_all()
    for name, c in cfgs.items():
        t = c["training"]
        assert t["metric_for_best"] == "macro_f1_val", name
        assert float(t["head_dropout"]) == 0.1, name
        assert float(t["lr_head"]) == 1e-3, name
        assert int(t["max_epochs"]) == 20, name
        assert t["amp"] == "auto", name
        assert float(t["weight_decay"]) == 0.01, name
        assert float(t["grad_clip_norm"]) == 1.0, name


def test_lr_backbone_por_modo():
    """lr_backbone canonico por modo:
      linear_probing      -> 1e-4 (backbone congelado, lr no se aplica
                              pero el trainer no admite null aqui).
      full_finetuning     -> 1e-5 (sweet spot CWRU + HSG18).
    """
    cfgs = _load_all()
    for k in ("cwru_linear_probing", "hsg18_linear_probing"):
        lrb = cfgs[k]["training"]["lr_backbone"]
        assert lrb is not None, k
        assert float(lrb) == 1e-4, (k, lrb)
    for k in ("cwru_full_finetuning_lr1e-5", "hsg18_full_finetuning_lr1e-5"):
        lrb = cfgs[k]["training"]["lr_backbone"]
        assert lrb is not None, k
        assert float(lrb) == 1e-5, (k, lrb)


def test_paths_contienen_downstream_federated_pilot():
    """Las salidas van a `downstream_federated_pilot/<dataset>/` (carpeta
    separada del downstream central para no mezclar artefactos).
    """
    cfgs = _load_all()
    for name, c in cfgs.items():
        log_dir = c["paths"]["log_dir"]
        ckpt_dir = c["paths"]["checkpoint_dir"]
        assert "downstream_federated_pilot" in log_dir, (name, log_dir)
        assert "downstream_federated_pilot" in ckpt_dir, (name, ckpt_dir)
        # cwru -> .../cwru ; hsg18 -> .../hsg18
        ds = c["dataset"].lower()
        assert log_dir.endswith(f"/{ds}"), (name, log_dir, ds)
        assert ckpt_dir.endswith(f"/{ds}"), (name, ckpt_dir, ds)


def test_seed_42_y_dataset_consistente_top_vs_data():
    """seed=42 en los 4 (consistente con central) y dataset coincide
    entre top-level y data."""
    cfgs = _load_all()
    for name, c in cfgs.items():
        assert int(c["seed"]) == 42, name
        assert c["dataset"] == c["data"]["dataset"], (
            name, c["dataset"], c["data"]["dataset"],
        )


def test_ablacion_hsg18_lr1e4_consistente_con_base():
    """Config de ablacion diagnostica HSG18 full_finetuning con
    lr_backbone=1e-4 (10x mas alto que el sweet spot central de 1e-5).

    Ablacion barata para distinguir:
      A) problema de adaptacion downstream por LR demasiado conservador
         sobre el ckpt FL pilot (menos informativo que el central);
      B) problema estructural del embedding FL en dominio HDD.

    El test blinda que el YAML es identico al hsg18 full_ft canonico
    salvo en `run_name` y `lr_backbone`; el resto de hiperparametros y
    paths quedan igual.
    """
    import yaml
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    p_abl  = repo_root / "training" / "configs" / "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-4.yaml"
    p_base = repo_root / "training" / "configs" / "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5.yaml"
    assert p_abl.is_file(), f"Falta config ablacion: {p_abl}"
    assert p_base.is_file(), f"Falta config base: {p_base}"
    c_abl  = yaml.safe_load(p_abl.read_text(encoding="utf-8"))
    c_base = yaml.safe_load(p_base.read_text(encoding="utf-8"))

    # 1. run_name distinto del base y de los 4 oficiales.
    assert c_abl["run_name"] == "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-4"
    assert c_abl["run_name"] != c_base["run_name"]
    canonicos_4 = {
        "downstream_cwru_fedavg_pilot_linear_probing",
        "downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5",
        "downstream_hsg18_fedavg_pilot_linear_probing",
        "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5",
    }
    assert c_abl["run_name"] not in canonicos_4

    # 2. dataset y task identicos.
    assert c_abl["dataset"] == "HSG18"
    assert c_abl["data"]["dataset"] == "HSG18"
    assert c_abl["task"] == "classification_multiclass"

    # 3. Modelo identico al canonico.
    for k in ("name", "d_model", "n_layers", "n_heads", "d_ff",
              "dropout", "patch_size", "n_patches"):
        assert c_abl["model"][k] == c_base["model"][k], k
    assert c_abl["model"]["name"] == "patchtst_phm_base"

    # 4. Data identico al canonico.
    for k in ("processed_root", "dataset", "batch_size",
              "batch_size_policy", "max_channel_batch",
              "min_batch_size", "num_workers"):
        assert c_abl["data"][k] == c_base["data"][k], k
    assert c_abl["data"]["batch_size_policy"] == "adaptive_by_channels"
    assert int(c_abl["data"]["max_channel_batch"]) == 512

    # 5. Training identico salvo lr_backbone.
    for k in ("max_epochs", "lr_head", "weight_decay", "amp",
              "grad_clip_norm", "log_every", "eval_every_epochs",
              "metric_for_best", "head_dropout",
              "max_train_batches_per_epoch", "max_val_batches",
              "max_test_batches"):
        assert c_abl["training"][k] == c_base["training"][k], k
    assert c_abl["training"]["metric_for_best"] == "macro_f1_val"
    assert float(c_abl["training"]["head_dropout"]) == 0.1
    assert float(c_abl["training"]["lr_head"]) == 0.001
    assert int(c_abl["training"]["max_epochs"]) == 20

    # 6. lr_backbone es lo unico distinto: 1e-4 vs 1e-5 del base.
    assert float(c_abl["training"]["lr_backbone"]) == 1e-4
    assert float(c_base["training"]["lr_backbone"]) == 1e-5

    # 7. Paths identicos (mismo log_dir y checkpoint_dir; el run_name
    # distinto evita colision de subcarpetas).
    assert c_abl["paths"]["log_dir"] == c_base["paths"]["log_dir"]
    assert c_abl["paths"]["checkpoint_dir"] == c_base["paths"]["checkpoint_dir"]
    assert "downstream_federated_pilot/hsg18" in c_abl["paths"]["log_dir"]
    assert "downstream_federated_pilot/hsg18" in c_abl["paths"]["checkpoint_dir"]

    # 8. seed = 42, identico.
    assert int(c_abl["seed"]) == 42 == int(c_base["seed"])


def test_no_pisamos_logs_central():
    """Las paths.log_dir y paths.checkpoint_dir de los 4 configs no
    coinciden con las del downstream central (cwru sin '_federated_pilot').
    """
    cfgs = _load_all()
    for name, c in cfgs.items():
        log_dir = c["paths"]["log_dir"]
        ckpt_dir = c["paths"]["checkpoint_dir"]
        assert not log_dir.endswith("/downstream/cwru"), name
        assert not log_dir.endswith("/downstream/hsg18"), name
        assert not ckpt_dir.endswith("/downstream/cwru"), name
        assert not ckpt_dir.endswith("/downstream/hsg18"), name
