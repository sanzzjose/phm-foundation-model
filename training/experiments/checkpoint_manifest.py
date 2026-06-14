"""Manifest canónico para un *checkpoint*.

Un :class:`CheckpointManifest` describe los metadatos asociados a un
*checkpoint* del banco: identificador, origen, modelo, configuración,
pasos efectivos, parámetros, *config_hash*, fecha, ruta o id en el
banco y un campo libre ``known_caveats`` para advertencias específicas
(por ejemplo "tolerated AMP overflows: 39").

El manifest es independiente del fichero binario del *checkpoint*: el
``.pt`` puede vivir en Drive o en otro almacenamiento externo; el
manifest siempre se versiona en repo porque pesa pocos KB.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config_hash import now_ts
from .jsonl_logger import json_safe


@dataclass
class CheckpointManifest:
    """Manifest de un *checkpoint* del banco de modelos.

    Atributos
    ---------
    checkpoint_id : str
        Identificador legible único en el banco (por ejemplo
        ``central_full_100k`` o ``fedprox_pilot_mu0_01``).
    origin : str
        Etiqueta del origen (``scratch``, ``central``, ``FedAvg``,
        ``FedProx``, ``SCAFFOLD``, ``FedAvgM``).
    model_name : str
        Nombre del modelo (por ejemplo ``PatchTSTPhm_base``).
    model_config : dict
        Hiperparámetros del modelo realmente entrenado.
    data_config : dict
        Configuración de datos efectiva durante el entrenamiento.
    optimizer_steps : Optional[int]
        Número de actualizaciones del optimizador efectivas.
    n_rounds : Optional[int]
        Número de rondas federadas (solo régimen federado).
    n_clients : Optional[int]
        Número de clientes federados (solo régimen federado).
    param_count : Optional[int]
        Parámetros entrenables del modelo.
    seed : Optional[int]
        Semilla principal del entrenamiento.
    elapsed_seconds : Optional[float]
        Duración total del entrenamiento.
    config_hash : Optional[str]
        Hash de la config efectiva.
    code_version : Optional[str]
        Git hash del repo en el momento del entrenamiento.
    artifact_path_or_id : Optional[str]
        Ruta o id del fichero binario del *checkpoint* (puede vivir
        fuera del repo).
    known_caveats : list[str]
        Advertencias breves específicas del *checkpoint*.
    created_at : str
        Timestamp ISO 8601 local.
    """

    checkpoint_id: str
    origin: str
    model_name: str
    model_config: Dict[str, Any] = field(default_factory=dict)
    data_config: Dict[str, Any] = field(default_factory=dict)
    optimizer_steps: Optional[int] = None
    n_rounds: Optional[int] = None
    n_clients: Optional[int] = None
    param_count: Optional[int] = None
    seed: Optional[int] = None
    elapsed_seconds: Optional[float] = None
    config_hash: Optional[str] = None
    code_version: Optional[str] = None
    artifact_path_or_id: Optional[str] = None
    known_caveats: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=now_ts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def write_checkpoint_manifest(
    manifest: CheckpointManifest,
    out_dir: Path,
    *,
    json_name: str = "checkpoint_manifest.json",
) -> Path:
    """Serializa ``manifest`` a ``out_dir/json_name`` en JSON estricto."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / json_name
    safe = json_safe(manifest.to_dict())
    with path.open("w", encoding="utf-8") as f:
        json.dump(safe, f, ensure_ascii=False, allow_nan=False, indent=2)
    return path
