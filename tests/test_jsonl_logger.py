"""Tests del JsonlLogger estricto y del helper `_json_safe`.

Motivacion: bajo AMP el grad_norm puede ser `inf` ocasionalmente
(comportamiento normal del GradScaler). Si se persiste literal en
`metrics.jsonl`, Python lo tolera pero el fichero NO es JSON estricto
(viola RFC 8259). Los parsers conservadores y servicios remotos lo
rechazan.

Aqui se verifica que:
  - `_json_safe` convierte float NaN/inf/-inf a None.
  - `JsonlLogger.log` escribe lineas JSON estandar (sin "NaN"/"Infinity"
    literal), parseables por `json.loads` sin parametros especiales.
  - `allow_nan=False` esta activo al serializar.
"""

from __future__ import annotations

import json
import math

import pytest

from training.train_ssl_central import JsonlLogger, _json_safe


# ----------------------------------------------------------------------
# _json_safe
# ----------------------------------------------------------------------


def test_json_safe_inf_to_none():
    assert _json_safe(float("inf")) is None
    assert _json_safe(float("-inf")) is None
    assert _json_safe(float("nan")) is None


def test_json_safe_finite_passes_through():
    assert _json_safe(0.0) == 0.0
    assert _json_safe(1.5) == 1.5
    assert _json_safe(-3.14) == -3.14
    assert _json_safe(0) == 0
    assert _json_safe(True) is True
    assert _json_safe(False) is False
    assert _json_safe("hola") == "hola"
    assert _json_safe(None) is None


def test_json_safe_dict_recursive():
    out = _json_safe({
        "a": float("inf"),
        "b": [float("nan"), 1.0, float("-inf")],
        "c": {"d": float("nan"), "e": 42},
    })
    assert out == {
        "a": None,
        "b": [None, 1.0, None],
        "c": {"d": None, "e": 42},
    }


def test_json_safe_list_with_mixed_nonfinites():
    out = _json_safe([float("inf"), 2.5, float("nan"), float("-inf"), 7])
    assert out == [None, 2.5, None, None, 7]


def test_json_safe_pathlib_to_str():
    from pathlib import Path
    p = Path("/tmp/x/y")
    out = _json_safe(p)
    assert isinstance(out, str)
    assert "x" in out


# ----------------------------------------------------------------------
# JsonlLogger
# ----------------------------------------------------------------------


def test_jsonl_logger_writes_no_infinity_literal(tmp_path):
    logger = JsonlLogger(tmp_path / "m.jsonl")
    logger.log({
        "step": 1,
        "x": float("inf"),
        "y": float("nan"),
        "z": -float("inf"),
        "loss": 0.42,
    })
    logger.close()
    text = (tmp_path / "m.jsonl").read_text(encoding="utf-8")
    # No debe aparecer ninguno de los tokens no-estandar.
    assert "NaN" not in text, f"Aparece NaN literal: {text!r}"
    assert "Infinity" not in text, f"Aparece Infinity literal: {text!r}"
    assert "-Infinity" not in text


def test_jsonl_logger_lines_are_strict_json(tmp_path):
    logger = JsonlLogger(tmp_path / "m.jsonl")
    logger.log({"step": 1, "x": float("inf"), "y": float("nan")})
    logger.log({"step": 2, "loss": 0.5, "grad_norm": None})
    logger.close()
    # json.loads sin parametros: debe parsear cada linea como JSON estricto.
    for line in (tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        for k, v in rec.items():
            if isinstance(v, float):
                # Cualquier float debe ser finito (los inf/nan se mapearon a None).
                assert math.isfinite(v), f"Float no finito en {k}: {v}"


def test_jsonl_logger_nonfinites_become_null(tmp_path):
    logger = JsonlLogger(tmp_path / "m.jsonl")
    logger.log({"grad_norm": float("inf"), "step": 100})
    logger.close()
    rec = json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["grad_norm"] is None
    assert rec["step"] == 100


def test_jsonl_logger_preserves_normal_records(tmp_path):
    """Records 'limpios' (sin no-finitos) se serializan tal cual."""
    logger = JsonlLogger(tmp_path / "m.jsonl")
    logger.log({"step": 5, "dataset": "CMAPSS", "loss": 0.123, "grad_norm": 0.45})
    logger.close()
    rec = json.loads((tmp_path / "m.jsonl").read_text(encoding="utf-8").splitlines()[0])
    assert rec["step"] == 5
    assert rec["dataset"] == "CMAPSS"
    assert rec["loss"] == 0.123
    assert rec["grad_norm"] == 0.45


def test_allow_nan_false_is_actually_active(tmp_path, monkeypatch):
    """Si por algun camino un NaN se cuela al `json.dumps`, debe fallar
    porque `allow_nan=False` esta activado. Forzamos el escenario:
    parcheamos `_json_safe` para que NO limpie y verificamos que
    `JsonlLogger.log` lanza ValueError."""
    import training.train_ssl_central as mod

    def passthrough(obj):
        # No limpia; deja el NaN tal cual para forzar fallo en dumps.
        return obj

    monkeypatch.setattr(mod, "_json_safe", passthrough)
    logger = JsonlLogger(tmp_path / "m.jsonl")
    with pytest.raises(ValueError):
        logger.log({"x": float("nan")})
    logger.close()
