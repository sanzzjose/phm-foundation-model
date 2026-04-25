"""Builder adaptador para PBCP16 (clasificación).

PBCP16 es TRANSFER_TARGET primary de clasificación. Ya tiene corridas
validadas en el bloque central (3 modos × 1 seed). El baseline
from_scratch lidera y el SSL no aporta en este dataset; el builder lo
documenta como caveat.
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="PBCP16",
    role="TRANSFER_TARGET",
    task_type="classification",
    primary_metric="macro_f1",
    secondary_metrics=["balanced_accuracy", "accuracy"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=False,
    split_policy="by_trajectory",
    target_definition="multiclass fault label (5 clases naturalmente balanceadas)",
    target_policy="unit_label",
    caveat=(
        "Baseline alto: el régimen from_scratch obtiene el mejor macro-F1 "
        "del bloque central. SSL no aporta en este dataset bajo el setup actual."
    ),
    source_artifacts=["processed/PBCP16"],
    status="ready",
))
