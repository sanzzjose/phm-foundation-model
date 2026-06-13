"""Esquema canónico de métricas y campos para los manifests del proyecto.

Este módulo define:

* ``REQUIRED_RUN_INFO_FIELDS``: lista de campos mínimos que debe contener
  todo ``run_info.json`` para considerarse comparable en la tabla maestra;
* ``REQUIRED_RESULT_ROW_FIELDS``: lista de campos mínimos de una fila
  ``result_row.json`` integrable en ``master_*_runs.csv``;
* ``METRIC_DIRECTION``: dirección de mejora (``higher_is_better``,
  ``lower_is_better``) para cada métrica canónica del proyecto;
* ``ALLOWED_PHASES``, ``ALLOWED_ROLES``, ``ALLOWED_TASK_TYPES``,
  ``ALLOWED_STATUS``: conjuntos cerrados de valores categóricos.

Las funciones de validación que viven aquí son intencionalmente
ligeras: comprueban presencia de claves y tipos básicos, sin verificar
contenido semántico. La validación profunda es responsabilidad de cada
módulo consumidor (por ejemplo ``ssl_eval_suite`` o ``probe_suite``).
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple


# ---------------------------------------------------------------------------
# Valores categóricos permitidos
# ---------------------------------------------------------------------------

ALLOWED_PHASES = (
    "ssl_central",
    "ssl_federated",
    "ssl_eval",
    "downstream_probe",
    "downstream_full",
    "baseline",
    "anomaly",
)

ALLOWED_ROLES = (
    "PRETRAIN_SOURCE",
    "TRANSFER_TARGET",
    "EXTERNAL_TARGET",
    "MIXED",
)

ALLOWED_TASK_TYPES = (
    "ssl",
    "classification",
    "rul",
    "regression",
    "soh",
    "anomaly",
)

ALLOWED_STATUS = (
    "ok",
    "failed",
    "partial",
    "skipped",
    "dry_run",
)


# ---------------------------------------------------------------------------
# Campos mínimos por tipo de manifest
# ---------------------------------------------------------------------------

REQUIRED_RUN_INFO_FIELDS: Tuple[str, ...] = (
    "experiment_id",
    "phase",
    "dataset",
    "role",
    "task_type",
    "model_name",
    "checkpoint_origin",
    "seed",
    "status",
    "created_at",
)

REQUIRED_RESULT_ROW_FIELDS: Tuple[str, ...] = (
    "experiment_id",
    "phase",
    "dataset",
    "role",
    "task_type",
    "model_name",
    "checkpoint_origin",
    "seed",
    "primary_metric_name",
    "primary_metric_value",
    "status",
    "created_at",
)


# ---------------------------------------------------------------------------
# Dirección de mejora por métrica canónica
# ---------------------------------------------------------------------------

#: Dirección de mejora para las métricas habituales del proyecto.
#:
#: ``"higher"`` significa que **mayor es mejor** (clasificación, AUROC,
#: etc.); ``"lower"`` significa que **menor es mejor** (RMSE, MAE, loss,
#: etc.). Esta tabla la consume el reporting al construir las
#: comparaciones para saber qué dirección apunta una mejora.
METRIC_DIRECTION: Dict[str, str] = {
    # Clasificación
    "macro_f1": "higher",
    "macro_f1_val": "higher",
    "balanced_accuracy": "higher",
    "accuracy": "higher",
    "auroc": "higher",
    "auprc": "higher",
    # Regresión / RUL / SOH
    "rmse": "lower",
    "rmse_val": "lower",
    "mae": "lower",
    "r2": "higher",
    "cmapss_score": "lower",
    # SSL
    "loss": "lower",
    "ssl_loss": "lower",
    "ssl_train_loss": "lower",
    "ssl_val_loss_weighted": "lower",
    # Federado y operativos
    "communication_mb": "lower",
    "elapsed_seconds": "lower",
    "nonfinite_count": "lower",
    "client_drift_norm": "lower",
    "coverage_dataset_count": "higher",
    "coverage_client_count": "higher",
}


# ---------------------------------------------------------------------------
# Helpers de validación
# ---------------------------------------------------------------------------

def assert_categorical(value: str, allowed: Tuple[str, ...], field_name: str) -> None:
    """Valida que ``value`` esté en ``allowed``. Lanza ``ValueError`` si no."""
    if value not in allowed:
        raise ValueError(
            f"Valor invalido para `{field_name}`: {value!r}. "
            f"Permitidos: {sorted(allowed)!r}."
        )


def validate_required_fields(
    record: Dict[str, Any],
    required: Tuple[str, ...],
    record_label: str = "record",
) -> List[str]:
    """Devuelve la lista de campos requeridos ausentes en ``record``.

    Lista vacía indica validación correcta. No lanza excepción para que
    el llamante pueda decidir si abortar o solo avisar.
    """
    missing = [k for k in required if k not in record or record[k] is None]
    return missing


def metric_direction(metric_name: str) -> str:
    """Devuelve la dirección de mejora de una métrica.

    Lanza ``KeyError`` si la métrica no está registrada; el caller debe
    registrarla en :data:`METRIC_DIRECTION` antes de usarla.
    """
    if metric_name not in METRIC_DIRECTION:
        raise KeyError(
            f"Metrica {metric_name!r} no registrada. "
            f"Anadela a METRIC_DIRECTION antes de usarla."
        )
    return METRIC_DIRECTION[metric_name]
