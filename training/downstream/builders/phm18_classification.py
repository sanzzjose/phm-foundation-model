"""Builder adaptador para PHM18 (clasificación de wind).

PHM18 es TRANSFER_TARGET primary de clasificación, dominio eólica
(22 canales, 3 clases). Las corridas históricas están etiquetadas
como ``historical_uncapped_batch_v0_1``: corrieron con un ``batch``
no capeado por ``compute_adaptive_batch_size`` debido a la versión
inicial del trainer. La corrida queda como ``ready`` para re-ejecutar
bajo la política de batch adaptativo posterior al fix.
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="PHM18",
    role="TRANSFER_TARGET",
    task_type="classification",
    primary_metric="macro_f1",
    secondary_metrics=["balanced_accuracy", "accuracy"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=False,
    split_policy="by_trajectory",
    target_definition="multiclass label (3 clases)",
    target_policy="unit_label",
    caveat=(
        "Las corridas v0.1 ejecutaron con batch sin tope adaptativo "
        "(historical_uncapped_batch_v0_1). El trainer actual ya corrige el "
        "ajuste B·C ≤ 512; futuras corridas usar política nueva."
    ),
    source_artifacts=["processed/PHM18"],
    status="ready",
))
