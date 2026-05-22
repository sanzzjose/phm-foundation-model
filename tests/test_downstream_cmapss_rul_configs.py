"""Tests preflight de los 3 configs CMAPSS_RUL por modo.

Validan que los configs `downstream_cmapss_rul_*` queden congelados
con la politica acordada antes de lanzar las 3 corridas reales.
Cualquier drift accidental en el LR rompe el test.

Motivacion: el trainer `train_downstream_rul.py` normaliza
`lr_backbone=None` a "un solo grupo de optimizer con lr=lr_head". Si
los 3 modos compartiesen un unico YAML con `lr_backbone=null`, entonces
`from_scratch` y `full_finetuning` recibirian implicitamente
`lr_backbone=lr_head=1e-3`, lo cual NO es el sweet spot validado en
CWRU/HSG18 (catastrophic forgetting con 1e-4 sobre central; 1e-5 es
el sweet spot full_ft). Por tanto, configs separados con
`lr_backbone` explicito por modo.

Independientes de torch (solo yaml + asserts).
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

_FROM_SCRATCH = _CFG_DIR / "downstream_cmapss_rul_from_scratch.yaml"
_LINEAR_PROBING = _CFG_DIR / "downstream_cmapss_rul_linear_probing.yaml"
_FULL_FT_LR1E5 = _CFG_DIR / "downstream_cmapss_rul_full_finetuning_lr1e-5.yaml"

_ALL = {
    "from_scratch": _FROM_SCRATCH,
    "linear_probing": _LINEAR_PROBING,
    "full_finetuning_lr1e-5": _FULL_FT_LR1E5,
}


# ----------------------------------------------------------------------
# 1. Existencia
# ----------------------------------------------------------------------


def test_los_3_configs_rul_existen():
    for mode, p in _ALL.items():
        assert p.is_file(), f"falta config CMAPSS_RUL {mode}: {p}"


@pytest.fixture(scope="module")
def cfgs() -> dict:
    return {mode: yaml.safe_load(p.read_text(encoding="utf-8")) for mode, p in _ALL.items()}


# ----------------------------------------------------------------------
# 2. Identidad del run + dataset
# ----------------------------------------------------------------------


def test_dataset_es_CMAPSS_RUL(cfgs):
    for mode, c in cfgs.items():
        assert c["dataset"] == "CMAPSS_RUL", (mode, c["dataset"])
        assert c["task"] == "regression_rul", (mode, c.get("task"))


def test_run_name_contiene_el_modo(cfgs):
    """Permite trazar logs/checkpoints por modo. El trainer escribe en
    paths.log_dir/<run_name>, asi que con run_names distintos los 3
    modos NO se pisan aunque compartan log_dir/checkpoint_dir."""
    assert cfgs["from_scratch"]["run_name"] == "downstream_cmapss_rul_from_scratch"
    assert cfgs["linear_probing"]["run_name"] == "downstream_cmapss_rul_linear_probing"
    assert cfgs["full_finetuning_lr1e-5"]["run_name"] == \
        "downstream_cmapss_rul_full_finetuning_lr1e-5"
    # Los 3 run_name son distintos.
    rns = {c["run_name"] for c in cfgs.values()}
    assert len(rns) == 3, rns


# ----------------------------------------------------------------------
# 3. target_key canonico
# ----------------------------------------------------------------------


def test_target_key_es_rul_capped_125(cfgs):
    """rul_capped_125 es el target canonico (Heimes 2008) tras el builder
    CMAPSS_RUL. No mezclar con rul_physical sin decision explicita."""
    for mode, c in cfgs.items():
        assert c["data"]["target_key"] == "rul_capped_125", (
            f"{mode}: target_key={c['data']['target_key']!r}, esperado rul_capped_125"
        )


# ----------------------------------------------------------------------
# 4. LR por modo
# ----------------------------------------------------------------------


def test_lr_head_es_1e_3_en_los_3_modos(cfgs):
    for mode, c in cfgs.items():
        assert math.isclose(float(c["training"]["lr_head"]), 1e-3, abs_tol=1e-12), (
            f"{mode}: lr_head={c['training']['lr_head']}, esperado 1e-3"
        )


def test_from_scratch_lr_backbone_es_1e_4(cfgs):
    """from_scratch: backbone entrenable desde init random; lr_backbone
    explicito (no null) para evitar que el trainer caiga al default
    de un solo grupo con lr=lr_head=1e-3."""
    c = cfgs["from_scratch"]
    lrb = c["training"].get("lr_backbone")
    assert lrb is not None, "from_scratch.lr_backbone NO debe ser null"
    assert math.isclose(float(lrb), 1e-4, abs_tol=1e-12), (
        f"from_scratch.lr_backbone={lrb}, esperado 1e-4"
    )


def test_linear_probing_lr_backbone_es_null(cfgs):
    """linear_probing: backbone congelado por --mode linear_probing;
    lr_backbone se ignora en el trainer. Por convencion en este config
    se deja null para hacer evidente que no se aplica."""
    c = cfgs["linear_probing"]
    lrb = c["training"].get("lr_backbone")
    assert lrb is None, f"linear_probing.lr_backbone={lrb!r}, esperado null"


def test_full_finetuning_lr_backbone_es_1e_5(cfgs):
    """full_finetuning: sweet spot validado en CWRU/HSG18 = 1e-5.
    lr=1e-4 produjo catastrophic forgetting del encoder SSL; 1e-5 afina
    sin destruir el backbone."""
    c = cfgs["full_finetuning_lr1e-5"]
    lrb = c["training"].get("lr_backbone")
    assert lrb is not None, "full_finetuning.lr_backbone NO debe ser null"
    assert math.isclose(float(lrb), 1e-5, abs_tol=1e-12), (
        f"full_finetuning.lr_backbone={lrb}, esperado 1e-5"
    )


# ----------------------------------------------------------------------
# 5. Batch adaptativo
# ----------------------------------------------------------------------


def test_batch_adaptativo_b_x_c_dentro_del_cap_para_c24(cfgs):
    """CMAPSS_RUL tiene C=24 canales (sensores NASA). El batch adaptativo
    debe garantizar `B*C <= 512` con C=24 -> B_effective = min(B_requested,
    512 // 24) = min(B, 21). Con B_requested=32 -> B_effective=21 -> B*C=504.
    Verificamos que los campos esten correctamente configurados."""
    C_CMAPSS_RUL = 24
    for mode, c in cfgs.items():
        d = c["data"]
        assert d["batch_size_policy"] == "adaptive_by_channels", (
            f"{mode}: batch_size_policy={d['batch_size_policy']!r}"
        )
        max_cb = int(d["max_channel_batch"])
        min_bs = int(d["min_batch_size"])
        b_req = int(d["batch_size"])
        assert max_cb == 512, (mode, max_cb)
        assert min_bs == 1, (mode, min_bs)
        assert b_req == 32, (mode, b_req)
        # Simulamos el adaptive batch size: B*C debe quedar <= max_cb (con
        # piso min_bs=1).
        b_eff = max(min_bs, min(b_req, max_cb // C_CMAPSS_RUL))
        bc = b_eff * C_CMAPSS_RUL
        assert bc <= max_cb, (
            f"{mode}: B_eff*C={bc} > cap {max_cb}; revisar batch_size_policy"
        )


# ----------------------------------------------------------------------
# 6. n_channels_fallback consistente
# ----------------------------------------------------------------------


def test_n_channels_fallback_24(cfgs):
    """n_channels_fallback solo se usa si el manifest no es accesible.
    CMAPSS_RUL tiene 24 canales."""
    for mode, c in cfgs.items():
        assert int(c["data"]["n_channels_fallback"]) == 24, (
            f"{mode}: n_channels_fallback={c['data']['n_channels_fallback']}"
        )


# ----------------------------------------------------------------------
# 7. metric_for_best + lower_is_better
# ----------------------------------------------------------------------


def test_metric_for_best_rmse_val_lower_is_better(cfgs):
    for mode, c in cfgs.items():
        assert c["training"]["metric_for_best"] == "rmse_val", (
            f"{mode}: metric_for_best={c['training']['metric_for_best']!r}"
        )
        assert bool(c["training"]["lower_is_better"]) is True, (
            f"{mode}: lower_is_better debe ser true para rmse"
        )


# ----------------------------------------------------------------------
# 8. save_predictions default false
# ----------------------------------------------------------------------


def test_save_predictions_false_por_defecto(cfgs):
    for mode, c in cfgs.items():
        assert bool(c["evaluation"]["save_predictions"]) is False, (
            f"{mode}: save_predictions debe ser false por defecto"
        )


# ----------------------------------------------------------------------
# 9. Paths: log_dir + checkpoint_dir consistentes
# ----------------------------------------------------------------------


def test_paths_log_y_checkpoint_consistentes(cfgs):
    """Los 3 configs comparten paths.log_dir y paths.checkpoint_dir; el
    trainer agrega `run_name` antes de escribir, por lo que los 3 modos
    NO se pisan (run_names distintos)."""
    log_dirs = {c["paths"]["log_dir"] for c in cfgs.values()}
    ckpt_dirs = {c["paths"]["checkpoint_dir"] for c in cfgs.values()}
    assert log_dirs == {
        "/content/drive/MyDrive/fm_fl_phmd/logs/downstream/cmapss_rul"
    }, log_dirs
    assert ckpt_dirs == {
        "/content/drive/MyDrive/fm_fl_phmd/checkpoints/downstream/cmapss_rul"
    }, ckpt_dirs


def test_paths_finales_por_modo_no_se_pisan(cfgs):
    """Combinando log_dir + run_name, los 3 modos escriben en
    directorios distintos."""
    finales_log = {
        Path(c["paths"]["log_dir"]) / c["run_name"] for c in cfgs.values()
    }
    finales_ckpt = {
        Path(c["paths"]["checkpoint_dir"]) / c["run_name"] for c in cfgs.values()
    }
    assert len(finales_log) == 3, finales_log
    assert len(finales_ckpt) == 3, finales_ckpt


# ----------------------------------------------------------------------
# 10. processed_downstream_root apunta al builder, NO a processed/
# ----------------------------------------------------------------------


def test_processed_downstream_root(cfgs):
    """CMAPSS_RUL NO se entrena sobre processed/CMAPSS/ (harmonization
    v0.5; target ambiguo entre splits). El builder dedicado escribe en
    processed_downstream/CMAPSS_RUL/. Ver
    results/downstream/cmapss_rul_decision/decision.md."""
    for mode, c in cfgs.items():
        root = c["data"]["processed_downstream_root"]
        assert root == "/content/drive/MyDrive/fm_fl_phmd/processed_downstream", (
            f"{mode}: processed_downstream_root={root!r}; debe apuntar al builder"
        )
        # NO debe ser processed/ a secas (eso seria la harmonization v0.5).
        assert root != "/content/drive/MyDrive/fm_fl_phmd/processed", (
            f"{mode}: confusion entre processed/ y processed_downstream/"
        )


# ----------------------------------------------------------------------
# 11. Equivalencia bit-a-bit fuera de los campos LR/run_name
# ----------------------------------------------------------------------


def _strip_diffs(cfg: dict) -> dict:
    """Devuelve una copia del cfg sin los campos que DEBEN diferir entre
    modos: run_name + training.lr_backbone. El resto del YAML debe
    coincidir bit-a-bit entre los 3 configs."""
    c = {k: v for k, v in cfg.items() if k != "run_name"}
    c["training"] = {
        k: v for k, v in c.get("training", {}).items()
        if k != "lr_backbone"
    }
    return c


def test_los_3_configs_son_identicos_salvo_run_name_y_lr_backbone(cfgs):
    """Garantia clave del bloque: la unica variable cambiada entre los
    3 modos es run_name + lr_backbone. Esto blinda que la comparacion
    inter-modos sea justa (mismo backbone, mismo head, mismo target,
    mismo batch adaptativo, mismo dataset, mismo seed)."""
    a = _strip_diffs(cfgs["from_scratch"])
    b = _strip_diffs(cfgs["linear_probing"])
    c = _strip_diffs(cfgs["full_finetuning_lr1e-5"])
    assert a == b, (
        "from_scratch vs linear_probing difieren fuera de "
        "run_name + lr_backbone; revisar drift accidental"
    )
    assert b == c, (
        "linear_probing vs full_finetuning_lr1e-5 difieren fuera de "
        "run_name + lr_backbone; revisar drift accidental"
    )
