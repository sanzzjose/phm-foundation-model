"""Escritura de filas ``result_row.json`` y ``result_row.csv``.

Una *result row* es la versión condensada de un manifest pensada para
integrarse en la tabla maestra. Cada fila contiene los campos mínimos
definidos por :data:`metrics_schema.REQUIRED_RESULT_ROW_FIELDS` más
opcionalmente un *delta* respecto a un *baseline* y un veredicto
cualitativo.

El módulo expone:

* :class:`ResultRow`: dataclass con los campos canónicos;
* :func:`write_result_row`: escribe la fila a un par
  ``result_row.json`` / ``result_row.csv`` en un directorio destino;
* :func:`read_result_rows`: lee todas las ``result_row.json`` bajo un
  directorio raíz y devuelve una lista de dicts, lista para integrar en
  la tabla maestra.
"""
from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from .jsonl_logger import json_safe
from .metrics_schema import (
    REQUIRED_RESULT_ROW_FIELDS,
    validate_required_fields,
)


@dataclass
class ResultRow:
    """Fila canónica para integrar en la tabla maestra.

    Los campos numéricos son ``Optional[float]`` porque algunos *result
    rows* corresponden a corridas en curso o canceladas. La validación
    profunda (presencia obligatoria de ``primary_metric_value`` cuando
    ``status == "ok"``) se delega al consumidor del CSV maestro.
    """

    experiment_id: str
    phase: str
    dataset: str
    role: str
    task_type: str
    model_name: str
    checkpoint_origin: str
    seed: int
    primary_metric_name: str
    primary_metric_value: Optional[float]
    status: str
    created_at: str
    secondary_metric_name: Optional[str] = None
    secondary_metric_value: Optional[float] = None
    delta_vs_baseline: Optional[float] = None
    delta_vs_baseline_rel_pct: Optional[float] = None
    verdict: Optional[str] = None
    caveat: Optional[str] = None
    config_hash: Optional[str] = None
    code_version: Optional[str] = None
    notes: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # `extra` queda como dict anidado en JSON, pero se aplana en CSV
        # mediante prefijo `extra__<key>` desde write_result_row.
        return d


def _flatten_extra_for_csv(row_dict: Dict[str, Any]) -> Dict[str, Any]:
    """Aplana ``extra`` a columnas ``extra__<k>`` para el CSV."""
    out = dict(row_dict)
    extra = out.pop("extra", None) or {}
    for k, v in extra.items():
        out[f"extra__{k}"] = v
    return out


def write_result_row(
    row: ResultRow,
    out_dir: Path,
    *,
    json_name: str = "result_row.json",
    csv_name: str = "result_row.csv",
) -> Path:
    """Escribe ``row`` en ``out_dir/{json_name, csv_name}``.

    Devuelve la ruta del JSON escrito. El JSON se serializa con
    ``allow_nan=False`` tras pasar por :func:`json_safe`, garantizando
    cumplimiento estricto de RFC 8259. El CSV se escribe siempre con
    cabecera (una fila por fichero) para que sea autoexplicativo.

    Valida que los campos requeridos estén presentes y no sean ``None``
    (lanza ``ValueError`` si falta alguno).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    payload = row.to_dict()
    # `primary_metric_value` puede ser None de forma legítima cuando la
    # corrida no es "ok" (status partial / skipped / failed): no hubo
    # métrica primaria válida. Exigimos que la CLAVE esté presente, pero
    # permitimos que su valor sea None. El resto de campos requeridos no
    # pueden ser None.
    nullable_when_present = {"primary_metric_value"}
    missing = []
    for k in REQUIRED_RESULT_ROW_FIELDS:
        if k not in payload:
            missing.append(k)
        elif payload[k] is None and k not in nullable_when_present:
            missing.append(k)
    if missing:
        raise ValueError(
            f"Faltan campos requeridos en ResultRow: {missing}. "
            f"Esperados: {list(REQUIRED_RESULT_ROW_FIELDS)}"
        )

    json_path = out_dir / json_name
    safe_payload = json_safe(payload)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(safe_payload, f, ensure_ascii=False, allow_nan=False, indent=2)

    # CSV aplanado con columnas extra__<k>.
    flat = _flatten_extra_for_csv(payload)
    csv_path = out_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(flat.keys()))
        writer.writeheader()
        writer.writerow({k: ("" if v is None else v) for k, v in flat.items()})

    return json_path


def read_result_rows(root: Path, glob: str = "**/result_row.json") -> List[Dict[str, Any]]:
    """Lee todas las ``result_row.json`` bajo ``root`` y devuelve dicts.

    Útil para construir la tabla maestra agregada desde los runs ya
    persistidos. No filtra por ``status``; el consumidor decide qué
    hacer con los runs no completados.
    """
    root = Path(root)
    out: List[Dict[str, Any]] = []
    for p in sorted(root.glob(glob)):
        try:
            with p.open("r", encoding="utf-8") as f:
                out.append(json.load(f))
        except Exception as exc:
            # Robusto: no abortamos por una fila corrupta; el caller
            # puede inspeccionar el log.
            out.append({"_error": f"failed to parse {p}: {exc}"})
    return out
