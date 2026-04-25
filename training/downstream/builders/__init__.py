"""Builders adaptadores para las tareas downstream del proyecto.

Cada submódulo registra una :class:`~training.downstream.task_registry.TaskSpec`
al ser importado. La importación del paquete fuerza el registro de
todas las tareas conocidas; los consumidores pueden entonces consultar
``training.downstream.task_registry.list_tasks()`` sin preocuparse del
orden de importación.

Convención: los builders SON adaptadores delgados. La lógica de
entrenamiento vive en ``training.train_downstream_classification`` y
``training.train_downstream_rul`` (ya implementados). El builder se
encarga de declarar metadatos canónicos y de exponer una función
``build_task_inputs`` reutilizable por ``probe_suite``.

Para datasets cuya semántica del target no está confirmada
(``CALCE_CS2``, ``PHMAP23``, ``IEEE14``, ``PHME20``, ``CBM14``), el
builder los registra con ``status="needs_semantic_review"`` y devuelve
explícitamente ``None`` al pedir un target, en lugar de inventarlo.
"""

from . import (
    cwru_classification,
    hsg18_classification,
    cmapss_rul,
    calce_cs2_rul_soh,
    phmap23_classification,
    pbcp16_classification,
    phm18_classification,
)

__all__ = [
    "cwru_classification",
    "hsg18_classification",
    "cmapss_rul",
    "calce_cs2_rul_soh",
    "phmap23_classification",
    "pbcp16_classification",
    "phm18_classification",
]
