"""Paquete `training.experiments`: infraestructura común de experimentos.

Esta capa fija un contrato mínimo para que cualquier corrida (SSL central,
SSL federado, downstream classification, downstream RUL, baseline,
anomalía) produzca artefactos comparables: ``run_info.json``,
``metrics.jsonl`` JSON estricto sin NaN/Infinity, ``summary.json``,
``config_effective.yaml``, ``checkpoint_manifest.json`` y una fila
integrable en la tabla maestra (``result_row.json`` / ``result_row.csv``).

Los módulos exponen primitivas reutilizables. Los entrenadores actuales
(``training.train_ssl_central``, ``training.train_downstream_classification``,
``training.train_downstream_rul``) implementan internamente piezas
equivalentes; este paquete las consolida y añade las que faltan
(``run_manifest``, ``result_writer``, ``checkpoint_manifest``,
``experiment_registry``). La migración de los entrenadores se planifica
para una iteración posterior y NO se hace en esta sesión para no romper
los outputs históricos.
"""

from .config_hash import (
    config_hash,
    get_git_info,
    load_config,
    now_ts,
)
from .jsonl_logger import (
    JsonlLogger,
    json_safe,
)
from .metrics_schema import (
    REQUIRED_RUN_INFO_FIELDS,
    REQUIRED_RESULT_ROW_FIELDS,
    METRIC_DIRECTION,
)
from .run_manifest import (
    RunManifest,
    build_run_manifest,
)
from .result_writer import (
    ResultRow,
    write_result_row,
    read_result_rows,
)
from .checkpoint_manifest import (
    CheckpointManifest,
    write_checkpoint_manifest,
)
from .experiment_registry import (
    ExperimentRegistry,
    new_experiment_id,
)

__all__ = [
    "config_hash",
    "get_git_info",
    "load_config",
    "now_ts",
    "JsonlLogger",
    "json_safe",
    "REQUIRED_RUN_INFO_FIELDS",
    "REQUIRED_RESULT_ROW_FIELDS",
    "METRIC_DIRECTION",
    "RunManifest",
    "build_run_manifest",
    "ResultRow",
    "write_result_row",
    "read_result_rows",
    "CheckpointManifest",
    "write_checkpoint_manifest",
    "ExperimentRegistry",
    "new_experiment_id",
]
