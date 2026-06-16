"""Tests unitarios de `training.experiments`.

Cubren las primitivas mínimas que cualquier corrida del proyecto debe
producir: hash de config, serialización JSON estricta, escritura de
``result_row``, manifest de *checkpoint* y registro de experimentos.

Los tests son rápidos y autocontenidos (no requieren GPU, Drive ni
shards reales). Usan ``tmp_path`` de pytest para los ficheros de salida.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from training.experiments import (
    CheckpointManifest,
    ExperimentRegistry,
    JsonlLogger,
    REQUIRED_RESULT_ROW_FIELDS,
    REQUIRED_RUN_INFO_FIELDS,
    ResultRow,
    RunManifest,
    build_run_manifest,
    config_hash,
    json_safe,
    new_experiment_id,
    read_result_rows,
    write_checkpoint_manifest,
    write_result_row,
)
from training.experiments.metrics_schema import (
    ALLOWED_PHASES,
    ALLOWED_ROLES,
    ALLOWED_STATUS,
    ALLOWED_TASK_TYPES,
    assert_categorical,
    metric_direction,
    validate_required_fields,
)


# ---------------------------------------------------------------------------
# config_hash
# ---------------------------------------------------------------------------

def test_config_hash_is_deterministic():
    cfg = {"a": 1, "b": [2, 3]}
    assert config_hash(cfg) == config_hash(cfg)


def test_config_hash_is_order_insensitive():
    cfg_a = {"a": 1, "b": 2}
    cfg_b = {"b": 2, "a": 1}
    assert config_hash(cfg_a) == config_hash(cfg_b)


def test_config_hash_length_is_16():
    h = config_hash({"x": 1})
    assert len(h) == 16
    assert all(c in "0123456789abcdef" for c in h)


def test_config_hash_changes_with_content():
    h1 = config_hash({"a": 1})
    h2 = config_hash({"a": 2})
    assert h1 != h2


# ---------------------------------------------------------------------------
# json_safe + JsonlLogger
# ---------------------------------------------------------------------------

def test_json_safe_handles_nan_inf():
    assert json_safe(float("nan")) is None
    assert json_safe(float("inf")) is None
    assert json_safe(float("-inf")) is None
    assert json_safe(0.5) == 0.5


def test_json_safe_recursive_dict_list():
    rec = {"a": [float("nan"), 1, "x"], "b": {"c": float("inf")}}
    safe = json_safe(rec)
    assert safe == {"a": [None, 1, "x"], "b": {"c": None}}


def test_json_safe_path_to_str(tmp_path):
    safe = json_safe(tmp_path)
    assert isinstance(safe, str)
    assert str(tmp_path) == safe


def test_jsonl_logger_writes_valid_jsonl(tmp_path):
    log_path = tmp_path / "metrics.jsonl"
    with JsonlLogger(log_path) as logger:
        logger.log({"step": 1, "loss": 0.5})
        logger.log({"step": 2, "loss": float("inf")})  # debe quedar como null
        logger.log({"step": 3, "loss": float("nan"), "grad_norm": 1.0})

    # Releemos: cada línea debe ser JSON válido sin literales NaN/Infinity.
    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3
    for line in lines:
        parsed = json.loads(line)
        assert "step" in parsed
        for k, v in parsed.items():
            if isinstance(v, float):
                assert math.isfinite(v), f"valor no finito en {k}: {v}"

    # Específicamente: step 2 -> loss=None, step 3 -> loss=None.
    assert json.loads(lines[1])["loss"] is None
    assert json.loads(lines[2])["loss"] is None


def test_jsonl_logger_rejects_truly_unserializable(tmp_path):
    """Si el record contiene un objeto no serializable y no caza el
    fallback ``str()``, debería igualmente serializarse vía str."""
    log_path = tmp_path / "metrics.jsonl"
    with JsonlLogger(log_path) as logger:
        # set y bytes no son JSON; json_safe los convierte vía str.
        logger.log({"step": 1, "weird": {1, 2, 3}})
    line = log_path.read_text(encoding="utf-8").strip()
    parsed = json.loads(line)
    assert isinstance(parsed["weird"], str)  # fallback aplicado


# ---------------------------------------------------------------------------
# Validación de schema
# ---------------------------------------------------------------------------

def test_validate_required_fields_returns_missing():
    rec = {"a": 1, "b": None}
    missing = validate_required_fields(rec, ("a", "b", "c"), "test")
    assert "b" in missing
    assert "c" in missing
    assert "a" not in missing


def test_assert_categorical_raises_on_invalid():
    with pytest.raises(ValueError):
        assert_categorical("bad", ALLOWED_PHASES, "phase")


def test_assert_categorical_accepts_valid():
    assert_categorical("ssl_central", ALLOWED_PHASES, "phase")


def test_metric_direction_known_metrics():
    assert metric_direction("macro_f1") == "higher"
    assert metric_direction("rmse") == "lower"


def test_metric_direction_unknown_raises():
    with pytest.raises(KeyError):
        metric_direction("not_a_metric")


@pytest.mark.parametrize("name,direction", [
    # Clasificación
    ("macro_f1", "higher"),
    ("macro_f1_val", "higher"),
    ("balanced_accuracy", "higher"),
    ("accuracy", "higher"),
    ("auroc", "higher"),
    ("auprc", "higher"),
    # Regresión / RUL / SOH
    ("rmse", "lower"),
    ("rmse_val", "lower"),
    ("mae", "lower"),
    ("r2", "higher"),
    ("cmapss_score", "lower"),
    # SSL
    ("loss", "lower"),
    ("ssl_loss", "lower"),
    ("ssl_train_loss", "lower"),
    ("ssl_val_loss_weighted", "lower"),
    # Federado y operativos
    ("communication_mb", "lower"),
    ("elapsed_seconds", "lower"),
    ("nonfinite_count", "lower"),
    ("client_drift_norm", "lower"),
    ("coverage_dataset_count", "higher"),
    ("coverage_client_count", "higher"),
])
def test_metric_direction_complete(name, direction):
    """Las 21 métricas exigidas por el hardening tienen dirección declarada."""
    assert metric_direction(name) == direction


# ---------------------------------------------------------------------------
# RunManifest + build_run_manifest
# ---------------------------------------------------------------------------

def test_build_run_manifest_basic():
    m = build_run_manifest(
        experiment_id="exp1",
        phase="downstream_probe",
        dataset="CWRU",
        role="TRANSFER_TARGET",
        task_type="classification",
        model_name="PatchTSTPhm_base",
        checkpoint_origin="central",
        seed=42,
        config={"a": 1},
    )
    assert m.experiment_id == "exp1"
    assert m.config_hash is not None
    assert len(m.config_hash) == 16
    d = m.to_dict()
    for k in REQUIRED_RUN_INFO_FIELDS:
        assert k in d, f"falta {k} en run_info dict"


def test_build_run_manifest_rejects_invalid_phase():
    with pytest.raises(ValueError):
        build_run_manifest(
            experiment_id="exp",
            phase="banana_phase",
            dataset="x",
            role="TRANSFER_TARGET",
            task_type="classification",
            model_name="m",
            checkpoint_origin="central",
            seed=0,
        )


# ---------------------------------------------------------------------------
# ResultRow + write_result_row + read_result_rows
# ---------------------------------------------------------------------------

def _make_row():
    return ResultRow(
        experiment_id="exp1",
        phase="downstream_probe",
        dataset="CWRU",
        role="TRANSFER_TARGET",
        task_type="classification",
        model_name="PatchTSTPhm_base",
        checkpoint_origin="central",
        seed=42,
        primary_metric_name="macro_f1",
        primary_metric_value=0.829,
        status="ok",
        created_at="2026-06-05T10:00:00",
    )


def test_write_result_row_writes_json_and_csv(tmp_path):
    row = _make_row()
    json_path = write_result_row(row, tmp_path)
    assert json_path.exists()
    csv_path = tmp_path / "result_row.csv"
    assert csv_path.exists()

    # JSON debe ser estricto y parseable.
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["primary_metric_value"] == 0.829
    assert payload["dataset"] == "CWRU"

    # CSV debe tener una fila con la misma información.
    content = csv_path.read_text(encoding="utf-8")
    assert "primary_metric_value" in content
    assert "0.829" in content


def test_write_result_row_rejects_missing_field(tmp_path):
    row = _make_row()
    row.primary_metric_name = None  # campo requerido (el NOMBRE no puede ser None)
    with pytest.raises(ValueError):
        write_result_row(row, tmp_path)


def test_write_result_row_allows_none_primary_value_for_partial(tmp_path):
    """`primary_metric_value=None` es legítimo en corridas no-ok (partial).

    Una corrida partial/skipped/failed puede no tener métrica primaria
    válida; el result_row debe poder escribirse igualmente para
    trazabilidad. Solo `primary_metric_value` es nullable cuando presente.
    """
    row = _make_row()
    row.primary_metric_value = None
    row.status = "partial"
    json_path = write_result_row(row, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["primary_metric_value"] is None
    assert payload["status"] == "partial"


def test_read_result_rows_reads_multiple(tmp_path):
    for i, ds in enumerate(["CWRU", "HSG18", "CMAPSS_RUL"]):
        row = _make_row()
        row.experiment_id = f"exp{i}"
        row.dataset = ds
        write_result_row(row, tmp_path / ds)
    rows = read_result_rows(tmp_path)
    assert len(rows) == 3
    datasets = {r["dataset"] for r in rows}
    assert datasets == {"CWRU", "HSG18", "CMAPSS_RUL"}


def test_write_result_row_handles_nan_in_secondary(tmp_path):
    row = _make_row()
    row.secondary_metric_name = "auroc"
    row.secondary_metric_value = float("nan")
    json_path = write_result_row(row, tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["secondary_metric_value"] is None


# ---------------------------------------------------------------------------
# CheckpointManifest
# ---------------------------------------------------------------------------

def test_write_checkpoint_manifest(tmp_path):
    m = CheckpointManifest(
        checkpoint_id="central_full_100k",
        origin="central",
        model_name="PatchTSTPhm_base",
        model_config={"d_model": 128, "n_layers": 4},
        data_config={"window_size": 512, "patch_size": 16},
        optimizer_steps=99961,
        param_count=801808,
        seed=42,
        elapsed_seconds=12875.8,
        config_hash="9ed84508a6820265",
        artifact_path_or_id="drive://checkpoints/central_full/ckpt_step100000.pt",
        known_caveats=["39 AMP overflows tolerated by GradScaler"],
    )
    path = write_checkpoint_manifest(m, tmp_path)
    assert path.exists()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["checkpoint_id"] == "central_full_100k"
    assert payload["optimizer_steps"] == 99961
    assert payload["known_caveats"] == ["39 AMP overflows tolerated by GradScaler"]


# ---------------------------------------------------------------------------
# ExperimentRegistry
# ---------------------------------------------------------------------------

def test_experiment_registry_records_and_reads(tmp_path):
    registry = ExperimentRegistry(tmp_path / "experiment_index.jsonl")
    registry.record({"experiment_id": "e1", "dataset": "CWRU", "status": "ok"})
    registry.record({"experiment_id": "e2", "dataset": "HSG18", "status": "ok"})
    entries = list(registry.read_all())
    assert len(entries) == 2
    ids = {e["experiment_id"] for e in entries}
    assert ids == {"e1", "e2"}
    # ts auto añadido.
    for e in entries:
        assert "ts" in e


def test_experiment_registry_tolerates_nan(tmp_path):
    registry = ExperimentRegistry(tmp_path / "idx.jsonl")
    registry.record({"experiment_id": "e1", "loss": float("nan")})
    entries = list(registry.read_all())
    assert entries[0]["loss"] is None


def test_new_experiment_id_is_legible():
    eid = new_experiment_id(
        phase="ssl_central",
        dataset="all_PS",
        model_name="PatchTSTPhm_base",
        checkpoint_origin="scratch",
        seed=42,
        suffix="100k",
    )
    assert "ssl_central" in eid
    assert "all_PS" in eid
    assert "seed42" in eid
    assert "100k" in eid


def test_new_experiment_id_no_suffix():
    eid = new_experiment_id(
        phase="downstream_probe",
        dataset="CWRU",
        model_name="m",
        checkpoint_origin="central",
        seed=0,
    )
    assert eid.endswith("seed0")
