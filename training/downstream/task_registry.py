"""Registro central de tareas downstream del proyecto.

Define formalmente qué tarea está disponible sobre cada TRANSFER_TARGET
o EXTERNAL_TARGET, qué métrica primaria la mide y qué modos de
adaptación soporta. Los builders concretos (``cwru_classification``,
``hsg18_classification``, etc.) registran sus :class:`TaskSpec` en este
registro al importarse.

Reglas:

* Ningún PRETRAIN_SOURCE entra como tarea downstream.
* Las tareas cuya semántica no está clara (CALCE_CS2, PHMAP23,
  IEEE14, ...) se registran con ``status="needs_semantic_review"`` y
  ``target_definition=None``: el código consumidor debe abortar antes
  de generar un target inventado.
* ``status="ready"`` significa que el builder y el target están
  validados sobre datos reales y pueden lanzar entrenamiento.
* ``status="not_implemented"`` indica que el builder aún no existe.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from training.experiments.metrics_schema import (
    ALLOWED_ROLES,
    ALLOWED_TASK_TYPES,
    assert_categorical,
)


VALID_STATUS = (
    "ready",
    "needs_semantic_review",
    "not_implemented",
)


@dataclass
class TaskSpec:
    """Especificación canónica de una tarea downstream.

    Atributos
    ---------
    dataset : str
        Identificador canónico del dataset (ej. ``"CWRU"``, ``"CMAPSS_RUL"``).
    role : str
        ``TRANSFER_TARGET`` o ``EXTERNAL_TARGET``.
    task_type : str
        ``classification``, ``rul``, ``regression``, ``soh``, ``anomaly``.
    primary_metric : str
        Nombre de la métrica primaria (ej. ``macro_f1``, ``rmse``).
    secondary_metrics : list[str]
        Métricas auxiliares reportadas en el ``run_info``.
    supports_linear_probing : bool
    supports_full_finetuning : bool
    supports_anomaly : bool
    split_policy : str
        Política de split (``by_unit``, ``by_trajectory``, ``benchmark_official``,
        ``by_cell``).
    target_definition : Optional[str]
        Descripción breve de cómo se construye el target. ``None`` cuando
        la semántica no está clara aún.
    target_policy : Optional[str]
        Política para targets de ventana (``last_valid``, ``unit_label``,
        ``majority``, ``event``).
    caveat : Optional[str]
        Advertencia breve a propagar a los manifests.
    source_artifacts : list[str]
        Rutas o identificadores esperados de artefactos fuente
        (manifests del builder armonizado, shards, etc.). Solo
        descriptivo; el código consumidor decide cómo resolverlos.
    status : str
        ``ready`` / ``needs_semantic_review`` / ``not_implemented``.
    """

    dataset: str
    role: str
    task_type: str
    primary_metric: str
    secondary_metrics: List[str] = field(default_factory=list)
    supports_linear_probing: bool = True
    supports_full_finetuning: bool = True
    supports_anomaly: bool = False
    split_policy: str = "by_unit"
    target_definition: Optional[str] = None
    target_policy: Optional[str] = None
    caveat: Optional[str] = None
    source_artifacts: List[str] = field(default_factory=list)
    status: str = "not_implemented"

    def validate(self) -> None:
        assert_categorical(self.role, ALLOWED_ROLES, "role")
        assert_categorical(self.task_type, ALLOWED_TASK_TYPES, "task_type")
        if self.status not in VALID_STATUS:
            raise ValueError(
                f"TaskSpec({self.dataset}).status={self.status!r} no permitido. "
                f"Valores: {list(VALID_STATUS)}"
            )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# El registro vive como dict global. Es seguro porque los builders se
# importan una vez por proceso (idempotencia).
_REGISTRY: Dict[str, TaskSpec] = {}


def register_task(spec: TaskSpec) -> TaskSpec:
    """Registra ``spec`` en el catálogo. Reemplaza si ya existía."""
    spec.validate()
    _REGISTRY[spec.dataset] = spec
    return spec


def get_task(dataset: str) -> TaskSpec:
    """Devuelve la TaskSpec para ``dataset`` o lanza ``KeyError``."""
    if dataset not in _REGISTRY:
        raise KeyError(
            f"Tarea no registrada: {dataset!r}. "
            f"Importa el builder correspondiente para registrarla."
        )
    return _REGISTRY[dataset]


def list_tasks(
    *,
    role: Optional[str] = None,
    task_type: Optional[str] = None,
    status: Optional[str] = None,
) -> List[TaskSpec]:
    """Lista las tareas registradas filtradas por rol/tipo/estado."""
    out = list(_REGISTRY.values())
    if role is not None:
        out = [t for t in out if t.role == role]
    if task_type is not None:
        out = [t for t in out if t.task_type == task_type]
    if status is not None:
        out = [t for t in out if t.status == status]
    return out


def reset_registry_for_tests() -> None:
    """Limpia el registro. Solo para uso en tests."""
    _REGISTRY.clear()


# ---------------------------------------------------------------------------
# Helpers de runnability
# ---------------------------------------------------------------------------

def is_runnable(spec: TaskSpec) -> bool:
    """Indica si una :class:`TaskSpec` puede entrar a entrenamiento real.

    Una tarea es *runnable* solo si su ``status`` es ``"ready"``. Las
    tareas con ``status="needs_semantic_review"`` o ``"not_implemented"``
    NO pueden ejecutar entrenamiento ni producir métrica válida; el
    código consumidor debe abortar antes de generar un target inventado.
    """
    return spec.status == "ready"


def require_runnable(dataset: str) -> TaskSpec:
    """Devuelve la TaskSpec si es runnable; si no, lanza ``RuntimeError``.

    Pensado para usar como guard al inicio de cualquier código que vaya
    a lanzar entrenamiento real sobre un dataset del registry::

        spec = require_runnable("CWRU")
        # ahora podemos asumir spec.status == "ready"

    Si la tarea no está registrada, propaga el ``KeyError`` de
    :func:`get_task` sin envolver.
    """
    spec = get_task(dataset)
    if not is_runnable(spec):
        raise RuntimeError(
            f"La tarea {dataset!r} no es runnable (status={spec.status!r}). "
            f"Caveat: {spec.caveat or 'sin glosa'}. "
            "No se puede ejecutar entrenamiento real sobre una tarea con "
            "status distinto de 'ready'."
        )
    return spec
