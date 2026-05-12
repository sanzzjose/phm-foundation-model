"""Builder adaptador para CMAPSS_RUL (regresión RUL).

CMAPSS_RUL es el TRANSFER_TARGET de regresión RUL del proyecto. Su
constitución usa el builder dedicado ``training.build_cmapss_rul_downstream``
con ``pipeline_config_hash=8317ba2a1bc87e20``. Esta TaskSpec NO hace
clamp automático de RUL; el target ya viene definido por el builder
(``rul_capped_125`` o ``rul_physical`` según la corrida).
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="CMAPSS_RUL",
    role="TRANSFER_TARGET",
    task_type="rul",
    primary_metric="rmse",
    secondary_metrics=["mae", "r2", "cmapss_score"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=False,
    split_policy="benchmark_official",
    target_definition=(
        "RUL físico reconstruido por el builder dedicado: "
        "train/val rul = max_cycle − cycle; test rul = last_cycle − cycle + official_RUL. "
        "Cap canónico en 125 ciclos (rul_capped_125)."
    ),
    target_policy="last_valid",
    caveat=(
        "Bloque cerrado con veredicto NO_CONFIRMADA: el baseline from_scratch "
        "tiene el menor RMSE; transferencia SSL no aporta bajo el setup actual. "
        "Pendiente ablación con head MLP, attention pooling y más épocas."
    ),
    source_artifacts=["processed_downstream/CMAPSS_RUL"],
    status="ready",
))
