"""Tests para `training.downstream.task_registry` y los builders adaptadores."""
from __future__ import annotations

import pytest

# Importar el paquete `builders` para forzar el registro de las 7 tareas.
import training.downstream.builders  # noqa: F401

from training.downstream.task_registry import (
    TaskSpec,
    get_task,
    list_tasks,
    register_task,
    reset_registry_for_tests,
)


def test_seven_builders_are_registered():
    """Importar `builders` debe registrar las 7 tareas iniciales."""
    expected = {
        "CWRU",
        "HSG18",
        "CMAPSS_RUL",
        "CALCE_CS2",
        "PHMAP23",
        "PBCP16",
        "PHM18",
    }
    registered = {t.dataset for t in list_tasks()}
    assert expected.issubset(registered), (
        f"Faltan tareas: {expected - registered}"
    )


def test_cwru_is_ready_with_classification():
    spec = get_task("CWRU")
    assert spec.task_type == "classification"
    assert spec.role == "TRANSFER_TARGET"
    assert spec.primary_metric == "macro_f1"
    assert spec.status == "ready"


def test_cmapss_rul_is_ready_with_rul():
    spec = get_task("CMAPSS_RUL")
    assert spec.task_type == "rul"
    assert spec.primary_metric == "rmse"
    assert spec.status == "ready"
    assert "no_confirmada" in (spec.caveat or "").lower() or "no_confirmada" in (spec.caveat or "")


def test_calce_cs2_is_needs_semantic_review():
    spec = get_task("CALCE_CS2")
    assert spec.status == "needs_semantic_review"
    # NO debe inventar target.
    assert spec.target_definition is None
    # Corrección 2026-06-05: el target armonizado es `rul`, no `soh`.
    assert spec.task_type == "rul"


def test_phmap23_is_ready_with_classification():
    """PHMAP23 pasó a ready tras verificar los labels reales de los shards.

    Lectura de los 3 splits (708 ventanas): fault discreto 0..24 (25
    clases), constante por trayectoria, train cubre las clases de
    val/test. Semántica de clasificación confirmada.
    """
    spec = get_task("PHMAP23")
    assert spec.status == "ready"
    assert spec.task_type == "classification"
    assert spec.primary_metric == "macro_f1"
    assert spec.target_policy == "unit_label"
    assert spec.target_definition is not None
    # El caveat debe seguir advirtiendo del desbalance fuerte.
    assert "59.3" in (spec.caveat or "")


def test_phm18_carries_historical_caveat():
    spec = get_task("PHM18")
    assert spec.status == "ready"
    assert spec.caveat is not None
    assert "historical_uncapped_batch_v0_1" in spec.caveat


def test_register_task_validates_categoricals():
    """Registrar un TaskSpec con campos inválidos debe lanzar ValueError."""
    with pytest.raises(ValueError):
        register_task(TaskSpec(
            dataset="BAD",
            role="NOT_A_ROLE",
            task_type="classification",
            primary_metric="macro_f1",
        ))


def test_list_tasks_filter_by_status():
    needs = list_tasks(status="needs_semantic_review")
    datasets = {t.dataset for t in needs}
    assert "CALCE_CS2" in datasets
    # PHMAP23 pasó a ready tras verificar los shards reales (2026-06-05).
    assert "PHMAP23" not in datasets
    # CWRU es ready, no debe aparecer en needs.
    assert "CWRU" not in datasets


def test_list_tasks_filter_by_task_type():
    rul = list_tasks(task_type="rul")
    assert any(t.dataset == "CMAPSS_RUL" for t in rul)
    classif = list_tasks(task_type="classification")
    assert any(t.dataset == "CWRU" for t in classif)


def test_get_task_unknown_raises():
    with pytest.raises(KeyError):
        get_task("DOES_NOT_EXIST")


def test_to_dict_serializable():
    spec = get_task("CWRU")
    d = spec.to_dict()
    assert d["dataset"] == "CWRU"
    assert "primary_metric" in d
    assert "supports_linear_probing" in d


# ---------------------------------------------------------------------------
# Bloqueo real de needs_semantic_review (Tarea 7 del hardening)
# ---------------------------------------------------------------------------

from training.downstream.task_registry import is_runnable, require_runnable


def test_is_runnable_ready_returns_true():
    assert is_runnable(get_task("CWRU")) is True
    assert is_runnable(get_task("HSG18")) is True
    assert is_runnable(get_task("CMAPSS_RUL")) is True
    # PHMAP23 pasó a ready tras verificar los shards reales.
    assert is_runnable(get_task("PHMAP23")) is True


def test_is_runnable_needs_semantic_review_returns_false():
    assert is_runnable(get_task("CALCE_CS2")) is False


def test_require_runnable_passes_for_ready():
    spec = require_runnable("CWRU")
    assert spec.status == "ready"


def test_require_runnable_blocks_needs_semantic_review():
    """needs_semantic_review NO puede ejecutar entrenamiento real."""
    with pytest.raises(RuntimeError) as exc:
        require_runnable("CALCE_CS2")
    assert "CALCE_CS2" in str(exc.value)
    assert "needs_semantic_review" in str(exc.value)


def test_require_runnable_passes_for_phmap23():
    """PHMAP23 pasó a ready tras verificar los shards reales (2026-06-05)."""
    spec = require_runnable("PHMAP23")
    assert spec.status == "ready"


def test_require_runnable_propagates_keyerror_for_unknown():
    with pytest.raises(KeyError):
        require_runnable("DOES_NOT_EXIST")


def test_probe_suite_imports_runnable_helpers():
    """probe_suite debe poder importar `is_runnable` y `require_runnable`."""
    from training.downstream import probe_suite as ps
    assert ps.is_runnable is is_runnable
    assert ps.require_runnable is require_runnable
