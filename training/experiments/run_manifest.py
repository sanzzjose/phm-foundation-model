"""Manifest unificado de una corrida (run).

Una :class:`RunManifest` describe los metadatos canónicos de un
experimento: identificador, fase, dataset, rol, tarea, modelo, origen
del *checkpoint*, semilla, métrica primaria, métricas de validación y
test, estado, *caveat* y timestamp. La función :func:`build_run_manifest`
construye un manifest a partir de la config efectiva y de un puñado de
argumentos, rellenando defaults razonables y validando los campos
categóricos.

El manifest se serializa a ``run_info.json`` mediante :meth:`to_dict` y
:func:`json.dumps` con ``allow_nan=False``. La validación profunda
(consistencia entre campos, presencia de métricas obligatorias por fase,
etc.) es responsabilidad del consumidor.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Optional

from .config_hash import config_hash, get_git_info, now_ts
from .metrics_schema import (
    ALLOWED_PHASES,
    ALLOWED_ROLES,
    ALLOWED_STATUS,
    ALLOWED_TASK_TYPES,
    assert_categorical,
)


@dataclass
class RunManifest:
    """Manifest canónico de una corrida del proyecto.

    Atributos
    ---------
    experiment_id : str
        Identificador legible, único por sesión. Convención sugerida:
        ``<phase>__<dataset>__<model>__<checkpoint_origin>__<seed>``.
    phase : str
        Fase del experimento (ver :data:`ALLOWED_PHASES`).
    dataset : str
        Dataset principal o etiqueta del conjunto evaluado (por ejemplo
        ``"all_PS"`` para evaluación SSL agregada).
    role : str
        Rol del dataset (ver :data:`ALLOWED_ROLES`).
    task_type : str
        Tipo de tarea (ver :data:`ALLOWED_TASK_TYPES`).
    model_name : str
        Nombre del modelo (por ejemplo ``PatchTSTPhm_base`` o
        ``PatchTSTPhm_base_plus``).
    model_config : dict
        Subdict con los hiperparámetros del modelo realmente usados.
    data_config : dict
        Subdict con la configuración de datos (ventana, stride, patch,
        normalización, política de cola).
    checkpoint_origin : str
        Etiqueta del origen del *checkpoint*: ``scratch``, ``central``,
        ``FedAvg``, ``FedProx``, ``SCAFFOLD``, ``FedAvgM``, ``unknown``.
    checkpoint_path_or_id : Optional[str]
        Identificador interno del *checkpoint* (no necesariamente ruta).
    seed : int
        Semilla aleatoria principal.
    primary_metric : Optional[str]
        Nombre de la métrica primaria (por ejemplo ``macro_f1`` o
        ``rmse``). Puede ser None mientras la corrida no haya terminado.
    val_metric : Optional[float]
        Valor de la métrica primaria en el conjunto de validación.
    test_metric : Optional[float]
        Valor de la métrica primaria en el conjunto de test.
    status : str
        Estado final del run (ver :data:`ALLOWED_STATUS`).
    caveat : Optional[str]
        Etiqueta breve de condiciones especiales (por ejemplo
        ``"historical_uncapped_batch_v0_1"``).
    created_at : str
        Timestamp ISO 8601 local. Se inicializa con :func:`now_ts`.
    code_version : Optional[str]
        Git hash del repo en el momento de la corrida.
    git_dirty : Optional[bool]
        True si el working tree no estaba limpio.
    config_hash : Optional[str]
        Hash SHA-256 (16 hex) de la config efectiva.
    """

    experiment_id: str
    phase: str
    dataset: str
    role: str
    task_type: str
    model_name: str
    checkpoint_origin: str
    seed: int
    status: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    checkpoint_path_or_id: Optional[str] = None
    primary_metric: Optional[str] = None
    val_metric: Optional[float] = None
    test_metric: Optional[float] = None
    caveat: Optional[str] = None
    created_at: str = field(default_factory=now_ts)
    code_version: Optional[str] = None
    git_dirty: Optional[bool] = None
    config_hash: Optional[str] = None

    def validate(self) -> None:
        """Valida los campos categóricos y devuelve nada.

        Lanza ``ValueError`` si algún valor no está en su conjunto
        permitido. No valida tipos primitivos ni semántica.
        """
        assert_categorical(self.phase, ALLOWED_PHASES, "phase")
        assert_categorical(self.role, ALLOWED_ROLES, "role")
        assert_categorical(self.task_type, ALLOWED_TASK_TYPES, "task_type")
        assert_categorical(self.status, ALLOWED_STATUS, "status")

    def to_dict(self) -> Dict[str, Any]:
        """Devuelve un ``dict`` serializable (claves ordenadas estables)."""
        return asdict(self)


def build_run_manifest(
    *,
    experiment_id: str,
    phase: str,
    dataset: str,
    role: str,
    task_type: str,
    model_name: str,
    checkpoint_origin: str,
    seed: int,
    status: str = "ok",
    config: Optional[Dict[str, Any]] = None,
    repo_root: Optional[Any] = None,
    **kwargs: Any,
) -> RunManifest:
    """Construye un :class:`RunManifest` rellenando ``config_hash`` y git.

    ``config`` se serializa para calcular ``config_hash`` y ``repo_root``
    se usa para obtener ``git_hash`` y ``git_dirty``. Cualquier campo
    adicional pasa por ``**kwargs`` (debe ser un atributo de la dataclass).

    Esta función no escribe nada en disco; eso es responsabilidad del
    ``result_writer``.
    """
    cfg_hash = config_hash(config) if config is not None else None
    git_info = get_git_info(repo_root) if repo_root is not None else {
        "git_hash": "unknown",
        "git_dirty": False,
    }

    manifest = RunManifest(
        experiment_id=experiment_id,
        phase=phase,
        dataset=dataset,
        role=role,
        task_type=task_type,
        model_name=model_name,
        checkpoint_origin=checkpoint_origin,
        seed=seed,
        status=status,
        config_hash=cfg_hash,
        code_version=git_info.get("git_hash"),
        git_dirty=git_info.get("git_dirty"),
        **kwargs,
    )
    manifest.validate()
    return manifest
