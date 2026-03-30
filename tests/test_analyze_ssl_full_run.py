"""Tests del script de analisis post-hoc del SSL central full.

Cubre:

- Parser robusto de JSONL que contiene literal `Infinity`/`NaN`
  (logs historicos pre-patch).
- Conteo correcto de `nonfinite_grad_steps_total` por tipo.
- Deteccion de outliers de loss finita por encima del umbral.
- Salida `posthoc_analysis.json` es JSON ESTRICTO (parseable con
  `json.loads` sin parametros, sin NaN/Infinity literales).
- Sin requerir torch ni checkpoint.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

from training.analyze_ssl_full_run import (
    _robust_loads,
    _safe,
    _stats,
    analyze,
    main,
)


# ----------------------------------------------------------------------
# Helpers de fixture
# ----------------------------------------------------------------------


def _write_sample_run(tmp: Path) -> Path:
    """Escribe run_info.json + metrics.jsonl sinteticos en tmp."""
    log_dir = tmp / "ssl_central_full_patchtst_phm"
    log_dir.mkdir(parents=True)

    run_info = {
        "run_name":                  "ssl_central_full_patchtst_phm",
        "git_hash":                  "deadbeef1234",
        "git_dirty":                 True,
        "config_hash":               "fakecfg00000000",
        "param_count":               801808,
        "stage":                     "full",
        "optimizer_steps":           97,
        "skipped_steps":             0,
        "amp_overflow_steps":        2,
        "amp_nonfinite_grad_steps":  2,
        "datasets_seen":             {"FAKE1": 50, "FAKE2": 47},
        "clients_seen":              {"clientA": 97},
        "max_effective_bc":          510,
        "elapsed_seconds":           1234.5,
        "model":                     {},
        "ssl":                       {},
        "data":                      {},
        "training":                  {},
    }
    (log_dir / "run_info.json").write_text(
        json.dumps(run_info, indent=2), encoding="utf-8"
    )

    # metrics.jsonl con mezcla controlada:
    #  - step normal con loss razonable
    #  - step con grad_norm Infinity (literal historico)
    #  - step con grad_norm NaN (literal historico)
    #  - step con loss enorme finita (outlier)
    lines = [
        json.dumps({"step": 1, "dataset": "FAKE1", "loss": 0.5,
                    "grad_norm": 0.42, "optimizer_applied": True,
                    "amp_nonfinite_grad": False}),
        # Infinity literal (logs pre-patch). json.dumps no produce esto a menos
        # que se desactive allow_nan, asi que lo escribimos a mano.
        '{"step": 2, "dataset": "FAKE1", "loss": 0.6, "grad_norm": Infinity, "optimizer_applied": false, "amp_nonfinite_grad": true}',
        '{"step": 3, "dataset": "FAKE2", "loss": 0.7, "grad_norm": NaN, "optimizer_applied": false, "amp_nonfinite_grad": true}',
        # Step normal con loss enorme finita
        json.dumps({"step": 4, "dataset": "FAKE2", "loss": 5000.0,
                    "grad_norm": 0.31, "optimizer_applied": True,
                    "amp_nonfinite_grad": False}),
        # Step de distribution
        json.dumps({"step": 5, "kind": "distribution",
                    "datasets_observed": {"FAKE1": 0.5, "FAKE2": 0.5},
                    "datasets_expected": {"FAKE1": 0.5, "FAKE2": 0.5}}),
    ]
    (log_dir / "metrics.jsonl").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    return log_dir


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_robust_loads_detects_infinity_and_nan():
    rec, kinds = _robust_loads('{"x": Infinity, "y": NaN, "z": -Infinity}')
    assert rec == {"x": None, "y": None, "z": None}
    assert kinds == {"inf": 1, "nan": 1, "neg_inf": 1}


def test_robust_loads_normal_line():
    rec, kinds = _robust_loads('{"step": 1, "loss": 0.42}')
    assert rec == {"step": 1, "loss": 0.42}
    assert kinds == {}


def test_safe_drops_nonfinites():
    out = _safe({"a": float("inf"), "b": [float("nan"), 1.5], "c": "ok"})
    assert out == {"a": None, "b": [None, 1.5], "c": "ok"}


def test_stats_empty_and_single():
    s = _stats([])
    assert s["count"] == 0
    assert s["mean"] is None
    s = _stats([2.0])
    assert s["count"] == 1
    assert s["mean"] == 2.0


def test_analyze_counts_nonfinites_and_huge_loss(tmp_path):
    log_dir = _write_sample_run(tmp_path)
    result = analyze(log_dir=log_dir, checkpoint=None,
                     huge_loss_threshold=1000.0)
    # 2 no-finitos en total (Infinity + NaN)
    assert result["nonfinite_grad_steps_total"] == 2
    assert result["nonfinite_grad_by_kind"].get("inf", 0) == 1
    assert result["nonfinite_grad_by_kind"].get("nan", 0) == 1
    # huge_finite_loss_steps contiene step 4
    assert len(result["huge_finite_loss_steps"]) >= 1
    assert any(r["step"] == 4 for r in result["huge_finite_loss_steps"])
    # 4 step records (1,2,3,4) + 1 distribution
    assert result["total_step_records"] == 4
    assert result["total_distribution_records"] == 1
    # Sin --checkpoint
    assert result["checkpoint_state_dict_all_finite"] is None


def test_main_writes_strict_json(tmp_path):
    log_dir = _write_sample_run(tmp_path)
    rc = main([
        "--log-dir", str(log_dir),
    ])
    assert rc == 0
    out_json = log_dir / "posthoc_analysis.json"
    out_md = log_dir / "posthoc_analysis.md"
    assert out_json.is_file()
    assert out_md.is_file()
    # Parse estricto (sin parametros)
    text = out_json.read_text(encoding="utf-8")
    assert "NaN" not in text
    assert "Infinity" not in text
    data = json.loads(text)
    assert data["nonfinite_grad_steps_total"] == 2
    assert isinstance(data["loss_buckets"], list)


def test_analyze_handles_missing_metrics_lines(tmp_path):
    """Lineas vacias o malformadas no rompen el script."""
    log_dir = tmp_path / "run"
    log_dir.mkdir()
    (log_dir / "run_info.json").write_text(
        json.dumps({
            "run_name": "test", "git_hash": "x", "config_hash": "y",
            "param_count": 100, "stage": "full",
            "optimizer_steps": 0, "skipped_steps": 0,
            "datasets_seen": {}, "clients_seen": {},
        }), encoding="utf-8",
    )
    (log_dir / "metrics.jsonl").write_text(
        '\n'
        '{"step": 1, "loss": 0.5}\n'
        '\n'
        '{ corrupted not json\n'  # linea invalida
        '{"step": 2, "loss": 0.4}\n',
        encoding="utf-8",
    )
    res = analyze(log_dir=log_dir)
    # Las dos lineas validas se contaron; la corrupta se ignoro.
    assert res["total_step_records"] == 2
