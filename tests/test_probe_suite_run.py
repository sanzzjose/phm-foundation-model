"""Tests de la corrida real del Probe Suite v1 (`cmd_run`) con mocks.

No entrenan: sustituyen la costura `_invoke_trainer` por un fake que
escribe un `run_info.json` realista. Así se valida la orquestación (plan,
skip, agregados, paths, selección por validación) sin torch ni datos.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.downstream import probe_suite as suite
from training.downstream import builders  # noqa: F401 (registra tasks)


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "training/configs/probe_suite_v1.yaml"


def _make_ctx(tmp_path, **kw):
    defaults = dict(
        checkpoint_path=Path("/fake/ckpt.pt"),
        checkpoint_id="central_full_100k",
        checkpoint_origin="central",
        output_dir=tmp_path,
        device="cpu",
        seed=42,
        repo_root=REPO_ROOT,
    )
    defaults.update(kw)
    return suite.ProbeContext(**defaults)


def _fake_trainer_factory(best_value=0.7, metric="macro_f1_val"):
    """Devuelve un _invoke_trainer fake que escribe run_info.json."""
    def _fake(trainer_kind, cfg, trainer_mode, checkpoint, repo_root):
        log_dir = Path(cfg["paths"]["log_dir"]) / cfg["run_name"]
        log_dir.mkdir(parents=True, exist_ok=True)
        run_info = {
            "mode": trainer_mode,
            "run_name": cfg["run_name"],
            "dataset": cfg["dataset"],
            "metric_for_best": cfg["training"].get("metric_for_best", metric),
            "best_value": best_value,
            "best_epoch": 3,
            "elapsed_seconds": 1.0,
            "config_hash": "deadbeef",
            "git_hash": "abc123",
            "n_classes": 25,
            "test_metrics": {"macro_f1": best_value - 0.05},
        }
        (log_dir / "run_info.json").write_text(json.dumps(run_info), encoding="utf-8")
        return 0
    return _fake


# ---------------------------------------------------------------------------
# Plan / skip
# ---------------------------------------------------------------------------

def test_run_skips_calce_cs2_needs_semantic_review(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path)
    rc = suite.cmd_run(cfg, ctx)
    assert rc == 0
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    skipped = [t for t in summary["tasks"] if t["status"] == "skipped"]
    assert {t["dataset"] for t in skipped} == {"CALCE_CS2"}
    assert "CALCE_CS2" in summary["datasets_skipped"]


def test_run_only_dataset_plans_only_that(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="PHMAP23", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    executed = summary["datasets_executed"]
    assert executed == ["PHMAP23"]
    assert summary["n_tasks_ok"] == 1


def test_run_max_tasks_limits_runnable(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, max_tasks=1)
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    # Solo 1 tarea ejecutable (ok/partial/failed), aunque CALCE_CS2 sigue skipped.
    n_exec = summary["n_tasks_ok"] + summary["n_tasks_partial"] + summary["n_tasks_failed"]
    assert n_exec == 1


# ---------------------------------------------------------------------------
# Outputs y agregados
# ---------------------------------------------------------------------------

def test_run_writes_per_task_result_rows(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="PHMAP23", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx)
    task_dir = tmp_path / "per_task" / "PHMAP23__linear_probing"
    assert (task_dir / "result_row.json").is_file()
    assert (task_dir / "summary.json").is_file()
    row = json.loads((task_dir / "result_row.json").read_text())
    assert row["dataset"] == "PHMAP23"
    assert row["phase"] == "downstream_probe"
    assert row["status"] == "ok"
    assert row["primary_metric_value"] == pytest.approx(0.7)


def test_run_summary_counts(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path)
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    # 7 ejecutables + 1 skipped = 8 entradas.
    assert summary["n_tasks_total"] == 8
    assert summary["n_tasks_ok"] == 7
    assert summary["n_tasks_skipped"] == 1
    assert summary["n_tasks_failed"] == 0
    assert summary["checkpoint_id"] == "central_full_100k"
    assert summary["checkpoint_origin"] == "central"
    assert summary["selection_split"] == "val"


def test_run_rankings_exclude_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path)
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    # per_task_val_metric NO incluye skipped.
    datasets_in_metric = {r["dataset"] for r in summary["per_task_val_metric"]}
    assert "CALCE_CS2" not in datasets_in_metric


def test_run_uses_val_for_selection(tmp_path, monkeypatch):
    """El valor primario reportado proviene de validación (best_value)."""
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(best_value=0.83))
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="CWRU", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    entry = summary["per_task_val_metric"][0]
    assert entry["primary_val_metric"] == pytest.approx(0.83)
    assert entry["status"] == "ok"
    assert summary["selection_split"] == "val"


# ---------------------------------------------------------------------------
# Robustez ante fallos
# ---------------------------------------------------------------------------

def test_run_task_failure_is_isolated(tmp_path, monkeypatch):
    """Si un trainer lanza, esa tarea es failed pero la suite continúa."""
    def _raising(trainer_kind, cfg, trainer_mode, checkpoint, repo_root):
        raise RuntimeError("boom en el trainer")

    monkeypatch.setattr(suite, "_invoke_trainer", _raising)
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="PHMAP23", only_mode="linear_probing")
    rc = suite.cmd_run(cfg, ctx)
    assert rc == 0  # la suite no aborta
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    assert summary["n_tasks_failed"] == 1
    failed = [t for t in summary["tasks"] if t["status"] == "failed"][0]
    assert "boom" in failed["error"]
    # result_row.json se escribe igualmente con status failed.
    task_dir = tmp_path / "per_task" / "PHMAP23__linear_probing"
    row = json.loads((task_dir / "result_row.json").read_text())
    assert row["status"] == "failed"
    assert row["primary_metric_value"] is None


def test_run_missing_checkpoint_marks_failed(tmp_path, monkeypatch):
    """Sin checkpoint, un probe que lo requiere queda failed (no entrena)."""
    called = {"n": 0}

    def _should_not_run(*a, **k):
        called["n"] += 1
        return 0

    monkeypatch.setattr(suite, "_invoke_trainer", _should_not_run)
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, checkpoint_path=None,
                    only_dataset="CWRU", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    assert summary["n_tasks_failed"] == 1
    assert called["n"] == 0  # nunca se invoca el trainer sin checkpoint


# ---------------------------------------------------------------------------
# build_trainer_config (puro)
# ---------------------------------------------------------------------------

def test_build_trainer_config_classification_full_ft():
    from training.downstream.task_registry import get_task
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"))
    built = suite.build_trainer_config(
        "CWRU", get_task("CWRU"), "full_finetuning_short", ctx, cfg)
    assert built["trainer_kind"] == "classification"
    assert built["trainer_mode"] == "full_finetuning"
    # full_finetuning_short usa lr_backbone conservador.
    assert built["cfg"]["training"]["lr_backbone"] == pytest.approx(1e-5)
    assert built["cfg"]["model"]["d_model"] == 128


def test_build_trainer_config_rul_mlp_head():
    from training.downstream.task_registry import get_task
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"))
    built = suite.build_trainer_config(
        "CMAPSS_RUL", get_task("CMAPSS_RUL"), "mlp_2layer", ctx, cfg)
    assert built["trainer_kind"] == "rul"
    assert built["trainer_mode"] == "linear_probing"
    assert built["cfg"]["head"]["hidden_dim"] == 256
    assert built["cfg"]["data"]["target_key"] == "rul_capped_125"


def test_build_trainer_config_rul_linear_head():
    from training.downstream.task_registry import get_task
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"))
    built = suite.build_trainer_config(
        "CMAPSS_RUL", get_task("CMAPSS_RUL"), "linear", ctx, cfg)
    assert built["cfg"]["head"]["hidden_dim"] is None


# ---------------------------------------------------------------------------
# --skip-existing: reanudar suite parcial
# ---------------------------------------------------------------------------

def test_skip_existing_reuses_completed_task(tmp_path, monkeypatch):
    """Si ya hay run_info.json, --skip-existing no re-invoca el trainer."""
    # 1) Primera corrida (mock) genera run_info.json para PHMAP23.
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(best_value=0.5))
    cfg = suite.load_config(CFG_PATH)
    ctx1 = _make_ctx(tmp_path, only_dataset="PHMAP23", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx1)

    # 2) Segunda corrida con skip_existing: el trainer NO debe llamarse.
    called = {"n": 0}

    def _should_not_run(*a, **k):
        called["n"] += 1
        return 0

    monkeypatch.setattr(suite, "_invoke_trainer", _should_not_run)
    ctx2 = _make_ctx(tmp_path, only_dataset="PHMAP23", only_mode="linear_probing",
                     skip_existing=True)
    suite.cmd_run(cfg, ctx2)
    assert called["n"] == 0
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    reused = [t for t in summary["tasks"] if t.get("reused_existing")]
    assert len(reused) == 1
    assert reused[0]["primary_metric_value"] == pytest.approx(0.5)


def test_skip_existing_runs_missing_task(tmp_path, monkeypatch):
    """skip_existing ejecuta los que faltan (sin run_info previo)."""
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(best_value=0.6))
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="HSG18", only_mode="linear_probing",
                    skip_existing=True)
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "probe_suite_v1_summary.json").read_text())
    assert summary["n_tasks_ok"] == 1
    # No reusado: se ejecuto porque no existia run_info previo.
    task = [t for t in summary["tasks"] if t["dataset"] == "HSG18"][0]
    assert not task.get("reused_existing")
