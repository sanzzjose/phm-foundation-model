"""Builder adaptador para la tarea de clasificación sobre CWRU.

CWRU es uno de los TRANSFER_TARGET *primary* del proyecto y un
benchmark reconocido en bearings. Su semántica ya está validada por
las corridas previas: 4 clases, target=``fault``, ``label_coverage=1.0``,
split por ``trajectory_id``. El builder se limita a registrar la
:class:`TaskSpec` canónica; la lógica de entrenamiento real vive en
``training.train_downstream_classification``.
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="CWRU",
    role="TRANSFER_TARGET",
    task_type="classification",
    primary_metric="macro_f1",
    secondary_metrics=["balanced_accuracy", "accuracy"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=True,
    split_policy="by_trajectory",
    target_definition="fault label per trajectory (4 clases)",
    target_policy="unit_label",
    caveat=None,
    source_artifacts=["processed/CWRU"],
    status="ready",
))
