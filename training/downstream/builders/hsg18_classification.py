"""Builder adaptador para HSG18 classification.

HSG18 (Hard Disk Drives) es TRANSFER_TARGET primary con clasificación
binaria (healthy / fail). Ya tiene corridas validadas en central y FL.
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="HSG18",
    role="TRANSFER_TARGET",
    task_type="classification",
    primary_metric="macro_f1",
    secondary_metrics=["balanced_accuracy", "accuracy", "auroc"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=True,
    split_policy="by_trajectory",
    target_definition="binary failure label (healthy / fail)",
    target_policy="unit_label",
    caveat=(
        "El cliente FL hdd contiene un solo PRETRAIN_SOURCE (HSF15); "
        "el régimen federado no captura diversidad intra-dominio."
    ),
    source_artifacts=["processed/HSG18"],
    status="ready",
))
