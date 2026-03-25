"""Logger JSONL estricto y helper de saneamiento para JSON.

El proyecto exige que ``metrics.jsonl`` sea JSON estándar parseable por
cualquier consumidor estricto (sin literales ``NaN``, ``Infinity`` o
``-Infinity``). Esa restricción es importante porque bajo *mixed
precision* (AMP) el ``grad_norm`` puede ser infinito en algunos pasos
toleros por el ``GradScaler``; dejar el literal ``Infinity`` en el JSONL
violaría la RFC 8259, aunque Python lo tolere.

``json_safe`` convierte los valores a una forma JSON estricta antes de
serializar, sustituyendo cualquier ``float`` no finito por ``None``. La
clase ``JsonlLogger`` envuelve un fichero JSONL y aplica ``json_safe`` a
cada registro antes de ``json.dumps(..., allow_nan=False)``.
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Dict


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def json_safe(obj: Any) -> Any:
    """Convierte ``obj`` a una forma JSON estricta (sin NaN / ±Inf).

    Reglas:

    * ``float`` ``NaN`` / ``±Inf`` -> ``None`` (JSON ``null``);
    * ``numpy`` floats / ints -> tipos Python nativos; si no son finitos,
      ``None``;
    * tensores ``torch`` escalares -> idem;
    * ``Path`` -> ``str``;
    * ``dict`` / ``list`` / ``tuple`` -> recursivo;
    * cualquier otro tipo no serializable estricto -> ``str(obj)``
      como fallback defensivo.

    ``numpy`` y ``torch`` se importan de forma perezosa para que el
    módulo no fuerce dependencias innecesarias.
    """
    if isinstance(obj, Path):
        return str(obj)
    # bool antes que int (bool es subclase de int en Python).
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    try:
        import numpy as _np
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, _np.ndarray):
            return [json_safe(x) for x in obj.tolist()]
    except Exception:
        pass
    try:
        import torch as _torch
        if isinstance(obj, _torch.Tensor):
            if obj.ndim == 0:
                v = obj.item()
                if isinstance(v, float):
                    return v if math.isfinite(v) else None
                return v
            return [json_safe(x) for x in obj.tolist()]
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


class JsonlLogger:
    """Logger JSONL estricto: cada línea es JSON válido sin NaN/Infinity.

    Uso típico::

        logger = JsonlLogger(Path("metrics.jsonl"))
        logger.log({"step": 1, "loss": 0.5})
        logger.close()

    El método :meth:`log` aplica :func:`json_safe` al registro antes de
    serializar con ``allow_nan=False``. Si tras la limpieza algún valor
    sigue siendo no serializable, ``json.dumps`` lanzará ``TypeError``,
    lo cual es lo deseable: forzamos detectar inputs malformados.

    El logger crea el directorio padre si no existe y abre el fichero en
    modo *append*, de modo que sea seguro reanudar un run sin perder los
    registros previos.
    """

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = path.open("a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", _ts())
        safe = json_safe(record)
        self.f.write(
            json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass

    def __enter__(self) -> "JsonlLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
