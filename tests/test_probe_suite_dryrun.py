"""Tests del Probe Suite v1 en modo dry-run y list.

No lanzan entrenamiento real. Verifican el plan, los modos por tarea y
el comportamiento del modo ``run`` esqueleto (status=skipped).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.downstream import probe_suite as suite


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "training/configs/probe_suite_v1.yaml"


def test_load_default_config():
    cfg = suite.load_config(CFG_PATH)
    assert "datasets" in cfg
    expected = {"CWRU", "HSG18", "CALCE_CS2", "PHMAP23", "CMAPSS_RUL"}
    assert expected.issubset(set(cfg["datasets"]))


def test_list_mode_runs():
    rc = suite.cmd_list(REPO_ROOT)
    assert rc == 0


def test_dry_run_succeeds_with_default_cfg():
    cfg = suite.load_config(CFG_PATH)
    rc = suite.cmd_dry_run(cfg, REPO_ROOT)
    assert rc == 0


def test_dry_run_marks_needs_review_as_skipped(capsys):
    """En dry-run, CALCE_CS2 sigue como skipped (needs_semantic_review).

    PHMAP23 pasó a ready tras verificar los shards reales (2026-06-05):
    aparece en el plan pero como runnable, no bloqueado.
    """
    cfg = suite.load_config(CFG_PATH)
    suite.cmd_dry_run(cfg, REPO_ROOT)
    out = capsys.readouterr().out
    assert "CALCE_CS2" in out
    assert "needs_semantic_review" in out
    assert "PHMAP23" in out


def test_dry_run_raises_when_datasets_empty():
    cfg = {"checkpoint": {"id": "x"}}
    with pytest.raises(ValueError):
        suite.cmd_dry_run(cfg, REPO_ROOT)


def test_plan_expands_modes_and_skips_calce(tmp_path):
    """El plan expande modos para runnable y deja CALCE_CS2 como skipped.

    La corrida real (con mocks) se prueba en test_probe_suite_run.py.
    """
    cfg = suite.load_config(CFG_PATH)
    plan = suite.plan_probe_tasks(cfg)
    runnable = [e for e in plan if e.runnable]
    skipped = [e for e in plan if not e.runnable]
    # CALCE_CS2 es la única skipped (needs_semantic_review).
    assert {e.dataset for e in skipped} == {"CALCE_CS2"}
    # Los 4 datasets runnable aparecen en el plan ejecutable.
    assert {"CWRU", "HSG18", "PHMAP23", "CMAPSS_RUL"} == {e.dataset for e in runnable}
