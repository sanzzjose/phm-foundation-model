"""Registro central de experimentos.

Esta capa fina permite generar identificadores legibles y únicos para
cada corrida (``experiment_id``) y mantener un índice persistente en
disco (``experiment_index.jsonl``) con la lista de experimentos creados.
El índice no sustituye a los manifests ``run_info.json`` individuales:
solo facilita auditar el catálogo de runs.

El módulo es deliberadamente delgado. La búsqueda eficiente y el
filtrado avanzado se delegan al ``build_master_table`` que vive en
``results/reporting/``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from .config_hash import now_ts
from .jsonl_logger import json_safe


def new_experiment_id(
    phase: str,
    dataset: str,
    model_name: str,
    checkpoint_origin: str,
    seed: int,
    *,
    suffix: Optional[str] = None,
) -> str:
    """Construye un identificador legible y razonablemente único.

    Convención: ``<phase>__<dataset>__<model>__<origin>__seed<seed>``
    con un sufijo opcional para distinguir variantes (por ejemplo
    ``lr1e-5``). El identificador no incorpora timestamp; si se necesita
    desambiguar runs idénticos, añadir un sufijo explícito.
    """
    base = f"{phase}__{dataset}__{model_name}__{checkpoint_origin}__seed{seed}"
    if suffix:
        base = f"{base}__{suffix}"
    return base


class ExperimentRegistry:
    """Índice append-only de experimentos creados en una raíz de runs.

    El índice se guarda como ``experiment_index.jsonl`` (una entrada por
    línea, JSON estricto). Cada entrada contiene los campos básicos del
    experimento: ``experiment_id``, ``phase``, ``dataset``, ``role``,
    ``task_type``, ``checkpoint_origin``, ``seed``, ``status``,
    ``created_at`` y opcionalmente ``run_info_path``.

    La clase es segura para uso concurrente solo en escritura simple
    (cada llamada a :meth:`record` abre el fichero en *append* y lo
    cierra). No proporciona locks de fichero.
    """

    def __init__(self, registry_path: Path) -> None:
        registry_path = Path(registry_path)
        registry_path.parent.mkdir(parents=True, exist_ok=True)
        self.path = registry_path

    def record(self, entry: Dict[str, Any]) -> None:
        """Añade una entrada al índice tras pasarla por :func:`json_safe`.

        La entrada debe ser un dict serializable; ``ts`` se añade si no
        está presente.
        """
        entry = dict(entry)
        entry.setdefault("ts", now_ts())
        safe = json_safe(entry)
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n")

    def read_all(self) -> Iterable[Dict[str, Any]]:
        """Itera todas las entradas del índice.

        Las entradas corruptas (líneas que no son JSON válido) se
        devuelven como ``{"_error": "...", "_raw": "..."}`` para no
        abortar la iteración. El consumidor decide cómo manejarlas.
        """
        if not self.path.is_file():
            return
        with self.path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception as exc:
                    yield {"_error": str(exc), "_line": i, "_raw": line[:200]}
