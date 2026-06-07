"""Tests del Downstream Full v1 (`downstream_full_suite`) con mocks.

No entrenan: sustituyen la costura `_invoke_trainer` por un fake que
escribe un `run_info.json` realista. Validan la orquestación (matriz
checkpoint × dataset × modo, dedup de from_scratch, skip de CALCE_CS2,
guard de PRETRAIN_SOURCE, filtros, selección por validación, paths) sin
torch ni datos.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from training.downstream import downstream_full_suite as suite
from training.downstream import builders  # noqa: F401 (registra tasks)
from training.downstream.task_registry import TaskSpec, get_task, register_task


REPO_ROOT = Path(__file__).resolve().parents[1]
CFG_PATH = REPO_ROOT / "training/configs/downstream_full_v1.yaml"


def _ckpts(n=2):
    base = [
        suite.CheckpointRef("central_25k", "central", "/fake/c25k.pt"),
        suite.CheckpointRef("fedavgm_50x50_m0_7_canonical", "fedavgm", "/fake/fedavgm.pt"),
        suite.CheckpointRef("fedavg_50x50", "fedavg", "/fake/fedavg.pt"),
    ]
    return base[:n]


def _make_ctx(tmp_path, checkpoints=None, **kw):
    defaults = dict(
        checkpoints=checkpoints if checkpoints is not None else _ckpts(2),
        output_dir=tmp_path,
        device="cpu",
        seed=42,
        repo_root=REPO_ROOT,
    )
    defaults.update(kw)
    return suite.FullContext(**defaults)


def _fake_trainer_factory(best_value=0.7, metric="macro_f1_val", calls=None):
    def _fake(trainer_kind, cfg, trainer_mode, checkpoint, repo_root):
        if calls is not None:
            calls.append({"mode": trainer_mode, "checkpoint": checkpoint,
                          "run_name": cfg["run_name"]})
        log_dir = Path(cfg["paths"]["log_dir"]) / cfg["run_name"]
        log_dir.mkdir(parents=True, exist_ok=True)
        run_info = {
            "mode": trainer_mode,
            "run_name": cfg["run_name"],
            "dataset": cfg["dataset"],
            "metric_for_best": cfg["training"].get("metric_for_best", metric),
            "best_value": best_value,
            "best_epoch": 5,
            "elapsed_seconds": 2.0,
            "config_hash": "deadbeef",
            "git_hash": "abc123",
            "n_classes": 4,
            "test_metrics": {"macro_f1": best_value - 0.05},
        }
        (log_dir / "run_info.json").write_text(json.dumps(run_info), encoding="utf-8")
        return 0
    return _fake


# ---------------------------------------------------------------------------
# Matriz / plan (puro, sin trainer)
# ---------------------------------------------------------------------------

def test_dry_run_succeeds():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), checkpoints=suite.checkpoints_from_config(cfg))
    assert suite.cmd_dry_run(cfg, ctx) == 0


def test_matrix_skips_calce_cs2():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"))
    matrix = suite.build_run_matrix(cfg, ctx)
    skipped = [e for e in matrix if not e.runnable]
    assert "CALCE_CS2" in {e.dataset for e in skipped}
    calce = [e for e in skipped if e.dataset == "CALCE_CS2"]
    assert len(calce) == 1
    assert calce[0].reason == "needs_semantic_review"


def test_from_scratch_deduped_once_per_dataset():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), checkpoints=_ckpts(3))
    matrix = suite.build_run_matrix(cfg, ctx)
    cwru_fs = [e for e in matrix if e.dataset == "CWRU" and e.mode == "from_scratch"]
    assert len(cwru_fs) == 1
    assert cwru_fs[0].checkpoint_id == suite.FROM_SCRATCH_ID
    assert cwru_fs[0].checkpoint_path is None


def test_checkpoint_dependent_modes_per_checkpoint():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), checkpoints=_ckpts(3))
    matrix = suite.build_run_matrix(cfg, ctx)
    cwru_lin = [e for e in matrix if e.dataset == "CWRU" and e.mode == "linear_probing"]
    assert len(cwru_lin) == 3  # una por checkpoint
    assert {e.checkpoint_id for e in cwru_lin} == {c.id for c in _ckpts(3)}


def test_only_dataset_filters():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), only_dataset="CWRU")
    matrix = suite.build_run_matrix(cfg, ctx)
    assert {e.dataset for e in matrix} == {"CWRU"}


def test_only_checkpoint_excludes_from_scratch():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), only_checkpoint="central_25k")
    matrix = suite.build_run_matrix(cfg, ctx)
    runnable = [e for e in matrix if e.runnable]
    # Solo el checkpoint pedido; from_scratch (otro checkpoint_id) excluido.
    assert {e.checkpoint_id for e in runnable} == {"central_25k"}
    assert all(e.mode != "from_scratch" for e in runnable)


def test_only_checkpoint_from_scratch_runs_only_from_scratch():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), only_checkpoint=suite.FROM_SCRATCH_ID)
    matrix = suite.build_run_matrix(cfg, ctx)
    runnable = [e for e in matrix if e.runnable]
    assert runnable, "debe haber tareas from_scratch"
    assert all(e.mode == "from_scratch" for e in runnable)


def test_max_tasks_limits_runnable():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/tmp/x"), max_tasks=2)
    matrix = suite.build_run_matrix(cfg, ctx)
    assert len([e for e in matrix if e.runnable]) == 2


def test_pretrain_source_is_guarded():
    """Una tarea con role=PRETRAIN_SOURCE nunca es runnable (guard duro)."""
    register_task(TaskSpec(
        dataset="FAKE_PS", role="PRETRAIN_SOURCE", task_type="classification",
        primary_metric="macro_f1", status="ready"))
    cfg = {
        "datasets": ["FAKE_PS"],
        "modes_per_task": {"FAKE_PS": ["linear_probing"]},
        "checkpoints": [{"id": "x", "origin": "central", "path": "/fake/x.pt"}],
    }
    ctx = _make_ctx(Path("/tmp/x"),
                    checkpoints=[suite.CheckpointRef("x", "central", "/fake/x.pt")])
    matrix = suite.build_run_matrix(cfg, ctx)
    fake = [e for e in matrix if e.dataset == "FAKE_PS"]
    assert len(fake) == 1
    assert not fake[0].runnable
    assert fake[0].reason == "role_not_downstream:PRETRAIN_SOURCE"


# ---------------------------------------------------------------------------
# Construcción de config (puro)
# ---------------------------------------------------------------------------

def test_build_trainer_config_layout_and_full_ft():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/out"))
    entry = suite.RunMatrixEntry(
        checkpoint_id="central_25k", checkpoint_origin="central",
        checkpoint_path="/fake/c25k.pt", dataset="CWRU", mode="full_finetuning",
        runnable=True, task_status="ready", task_type="classification",
        role="TRANSFER_TARGET", primary_metric="macro_f1")
    built = suite.build_trainer_config(entry, ctx, cfg)
    assert built["trainer_kind"] == "classification"
    assert built["trainer_mode"] == "full_finetuning"
    assert built["cfg"]["training"]["lr_backbone"] == pytest.approx(1e-5)
    # Layout <ckpt>/<dataset>/<mode>.
    assert built["task_dir"] == Path("/out/central_25k/CWRU/full_finetuning")
    assert built["cfg"]["run_name"] == "full_finetuning"


def test_build_trainer_config_rul_mlp_head():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/out"))
    entry = suite.RunMatrixEntry(
        checkpoint_id="fedavgm_50x50_m0_7_canonical", checkpoint_origin="fedavgm",
        checkpoint_path="/fake/f.pt", dataset="CMAPSS_RUL", mode="mlp_2layer",
        runnable=True, task_status="ready", task_type="rul",
        role="TRANSFER_TARGET", primary_metric="rmse")
    built = suite.build_trainer_config(entry, ctx, cfg)
    assert built["trainer_kind"] == "rul"
    assert built["trainer_mode"] == "linear_probing"
    assert built["cfg"]["head"]["hidden_dim"] == 256
    assert built["cfg"]["data"]["target_key"] == "rul_capped_125"


# ---------------------------------------------------------------------------
# Corrida real (con mocks)
# ---------------------------------------------------------------------------

def test_run_writes_result_rows_and_global_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory())
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="CWRU", only_mode="linear_probing")
    rc = suite.cmd_run(cfg, ctx)
    assert rc == 0
    # 2 checkpoints x CWRU x linear_probing = 2 result_rows en su layout.
    for cid in ("central_25k", "fedavgm_50x50_m0_7_canonical"):
        task_dir = tmp_path / cid / "CWRU" / "linear_probing"
        assert (task_dir / "result_row.json").is_file()
        assert (task_dir / "summary.json").is_file()
        row = json.loads((task_dir / "result_row.json").read_text())
        assert row["phase"] == "downstream_full"
        assert row["status"] == "ok"
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    assert summary["n_tasks_ok"] == 2
    assert summary["selection_split"] == "val"


def test_run_uses_val_for_selection(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(best_value=0.83))
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, checkpoints=_ckpts(1),
                    only_dataset="CWRU", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    entry = summary["per_task_val_metric"][0]
    assert entry["primary_val_metric"] == pytest.approx(0.83)
    assert entry["status"] == "ok"
    assert summary["selection_split"] == "val"


def test_from_scratch_runs_without_checkpoint(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(calls=calls))
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, only_dataset="CWRU", only_mode="from_scratch")
    suite.cmd_run(cfg, ctx)
    # from_scratch se ejecuta una vez, con checkpoint=None.
    assert len(calls) == 1
    assert calls[0]["checkpoint"] is None
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    assert summary["n_tasks_ok"] == 1


def test_checkpoint_dependent_missing_path_fails(tmp_path, monkeypatch):
    called = {"n": 0}

    def _should_not_run(*a, **k):
        called["n"] += 1
        return 0

    monkeypatch.setattr(suite, "_invoke_trainer", _should_not_run)
    cfg = suite.load_config(CFG_PATH)
    # Checkpoint sin path: linear_probing lo requiere -> failed sin invocar trainer.
    ctx = _make_ctx(tmp_path,
                    checkpoints=[suite.CheckpointRef("nopath", "central", None)],
                    only_dataset="CWRU", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx)
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    assert summary["n_tasks_failed"] == 1
    assert called["n"] == 0


def test_run_task_failure_is_isolated(tmp_path, monkeypatch):
    def _raising(*a, **k):
        raise RuntimeError("boom downstream")

    monkeypatch.setattr(suite, "_invoke_trainer", _raising)
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, checkpoints=_ckpts(1),
                    only_dataset="CWRU", only_mode="linear_probing")
    rc = suite.cmd_run(cfg, ctx)
    assert rc == 0
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    assert summary["n_tasks_failed"] == 1
    failed = [t for t in summary["tasks"] if t["status"] == "failed"][0]
    assert "boom" in failed["error"]


def test_skip_existing_reuses_completed(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(best_value=0.5))
    cfg = suite.load_config(CFG_PATH)
    ctx1 = _make_ctx(tmp_path, checkpoints=_ckpts(1),
                     only_dataset="CWRU", only_mode="linear_probing")
    suite.cmd_run(cfg, ctx1)

    called = {"n": 0}

    def _should_not_run(*a, **k):
        called["n"] += 1
        return 0

    monkeypatch.setattr(suite, "_invoke_trainer", _should_not_run)
    ctx2 = _make_ctx(tmp_path, checkpoints=_ckpts(1),
                     only_dataset="CWRU", only_mode="linear_probing",
                     skip_existing=True)
    suite.cmd_run(cfg, ctx2)
    assert called["n"] == 0
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    reused = [t for t in summary["tasks"] if t.get("reused_existing")]
    assert len(reused) == 1
    assert reused[0]["primary_metric_value"] == pytest.approx(0.5)


def test_checkpoints_from_config_reads_principal_set():
    cfg = suite.load_config(CFG_PATH)
    ckpts = suite.checkpoints_from_config(cfg)
    ids = {c.id for c in ckpts}
    assert "fedavgm_50x50_m0_7_canonical" in ids
    assert "central_25k" in ids
    assert all(c.path for c in ckpts)


# ---------------------------------------------------------------------------
# Overrides de LR / AMP (ablaciones cabos 6a)
# ---------------------------------------------------------------------------

def _entry(ds, mode, cid="central_25k", origin="central", path="/fake/c.pt",
           ttype="classification", metric="macro_f1"):
    return suite.RunMatrixEntry(
        checkpoint_id=cid, checkpoint_origin=origin, checkpoint_path=path,
        dataset=ds, mode=mode, runnable=True, task_status="ready",
        task_type=ttype, role="TRANSFER_TARGET", primary_metric=metric)


def test_default_config_keeps_canonical_lrs_and_amp():
    """Sin overrides: full_ft lr_backbone=1e-5, amp=auto, from_scratch lr_head=1e-3."""
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/out"))
    ft = suite.build_trainer_config(_entry("CWRU", "full_finetuning"), ctx, cfg)
    assert ft["cfg"]["training"]["lr_backbone"] == pytest.approx(1e-5)
    assert ft["cfg"]["training"]["amp"] == "auto"
    fs = suite.build_trainer_config(
        _entry("CWRU", "from_scratch", cid="from_scratch", origin="none", path=None),
        ctx, cfg)
    assert fs["cfg"]["training"]["lr_head"] == pytest.approx(1e-3)
    assert fs["cfg"]["training"]["lr_backbone"] is None  # un solo grupo


def test_lr_head_from_scratch_override():
    cfg = suite.load_config(
        REPO_ROOT / "training/configs/downstream_full_v1_cwru_fromscratch_lowlr.yaml")
    ctx = _make_ctx(Path("/out"))
    fs = suite.build_trainer_config(
        _entry("CWRU", "from_scratch", cid="from_scratch", origin="none", path=None),
        ctx, cfg)
    assert fs["cfg"]["training"]["lr_head"] == pytest.approx(3e-4)
    assert fs["cfg"]["training"]["amp"] == "off"


def test_lr_backbone_full_finetuning_override():
    for name, expected in [
        ("downstream_full_v1_cwru_ft_lr1e-6.yaml", 1e-6),
        ("downstream_full_v1_cwru_ft_lr5e-5.yaml", 5e-5),
    ]:
        cfg = suite.load_config(REPO_ROOT / "training/configs" / name)
        ctx = _make_ctx(Path("/out"))
        ft = suite.build_trainer_config(
            _entry("CWRU", "full_finetuning", cid="fedavgm_50x50_m0_7_canonical",
                   origin="fedavgm", path="/fake/f.pt"), ctx, cfg)
        assert ft["cfg"]["training"]["lr_backbone"] == pytest.approx(expected)


def test_override_does_not_leak_across_modes():
    """lr_head_from_scratch NO afecta a linear_probing; lr_backbone_full_finetuning
    NO afecta a from_scratch."""
    cfg = {"classification": {"lr_head": 1e-3, "lr_head_from_scratch": 3e-4,
                              "lr_backbone_full_finetuning": 1e-6}}
    ctx = _make_ctx(Path("/out"))
    lin = suite.build_trainer_config(_entry("CWRU", "linear_probing"), ctx, cfg)
    assert lin["cfg"]["training"]["lr_head"] == pytest.approx(1e-3)  # no baja
    assert lin["cfg"]["training"]["lr_backbone"] is None             # frozen
    fs = suite.build_trainer_config(
        _entry("CWRU", "from_scratch", cid="from_scratch", origin="none", path=None),
        ctx, cfg)
    assert fs["cfg"]["training"]["lr_head"] == pytest.approx(3e-4)
    assert fs["cfg"]["training"]["lr_backbone"] is None  # el ft override no aplica


def test_ablation_configs_build_matrix():
    """Las 3 configs de ablación cargan y producen una matriz coherente."""
    for name, mode in [
        ("downstream_full_v1_cwru_fromscratch_lowlr.yaml", "from_scratch"),
        ("downstream_full_v1_cwru_ft_lr1e-6.yaml", "full_finetuning"),
        ("downstream_full_v1_cwru_ft_lr5e-5.yaml", "full_finetuning"),
    ]:
        cfg = suite.load_config(REPO_ROOT / "training/configs" / name)
        ctx = _make_ctx(Path("/out"), checkpoints=suite.checkpoints_from_config(cfg))
        matrix = suite.build_run_matrix(cfg, ctx)
        runnable = [e for e in matrix if e.runnable]
        assert runnable and all(e.dataset == "CWRU" and e.mode == mode for e in runnable)


# ---------------------------------------------------------------------------
# Overrides CLI de calibración LR + tag (Fase 7a)
# ---------------------------------------------------------------------------

def test_cli_override_lr_backbone_full_ft():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/out"), override_lr_backbone=3e-6)
    ft = suite.build_trainer_config(_entry("CWRU", "full_finetuning"), ctx, cfg)
    assert ft["cfg"]["training"]["lr_backbone"] == pytest.approx(3e-6)
    assert ft["lr_backbone"] == pytest.approx(3e-6)


def test_cli_override_lr_backbone_ignored_for_linear_probing():
    """El override de lr_backbone NO des-congela linear_probing."""
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/out"), override_lr_backbone=3e-6)
    lin = suite.build_trainer_config(_entry("CWRU", "linear_probing"), ctx, cfg)
    assert lin["cfg"]["training"]["lr_backbone"] is None


def test_cli_override_lr_head_and_amp():
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(Path("/out"), override_lr_head=5e-4, override_amp="off")
    ft = suite.build_trainer_config(_entry("CWRU", "full_finetuning"), ctx, cfg)
    assert ft["cfg"]["training"]["lr_head"] == pytest.approx(5e-4)
    assert ft["cfg"]["training"]["amp"] == "off"


def test_tag_layout_nested_and_no_collision():
    cfg = suite.load_config(CFG_PATH)
    base = _entry("CWRU", "full_finetuning", cid="fedavgm_50x50_m0_7_canonical",
                  origin="fedavgm", path="/fake/f.pt")
    # sin tag -> .../full_finetuning
    no_tag = suite.build_trainer_config(base, _make_ctx(Path("/out")), cfg)
    assert no_tag["task_dir"] == Path("/out/fedavgm_50x50_m0_7_canonical/CWRU/full_finetuning")
    assert no_tag["run_name"] == "full_finetuning"
    # con tag -> .../full_finetuning/<tag> (no colisiona con el canónico)
    tagged = suite.build_trainer_config(base, _make_ctx(Path("/out"), tag="lr_1e-6"), cfg)
    assert tagged["task_dir"] == Path(
        "/out/fedavgm_50x50_m0_7_canonical/CWRU/full_finetuning/lr_1e-6")
    assert tagged["run_name"] == "lr_1e-6"
    # el trainer escribe en log_dir/run_name = .../full_finetuning/lr_1e-6 = task_dir
    assert Path(tagged["cfg"]["paths"]["log_dir"]) == Path(
        "/out/fedavgm_50x50_m0_7_canonical/CWRU/full_finetuning")


def test_run_with_tag_records_lr_and_tag(tmp_path, monkeypatch):
    monkeypatch.setattr(suite, "_invoke_trainer", _fake_trainer_factory(best_value=0.62))
    cfg = suite.load_config(CFG_PATH)
    ctx = _make_ctx(tmp_path, checkpoints=_ckpts(1),
                    only_dataset="CWRU", only_mode="full_finetuning",
                    override_lr_backbone=1e-6, tag="lr_1e-6")
    suite.cmd_run(cfg, ctx)
    # result_row en el sub-dir con tag
    task_dir = tmp_path / "central_25k" / "CWRU" / "full_finetuning" / "lr_1e-6"
    assert (task_dir / "result_row.json").is_file()
    summary = json.loads((tmp_path / "downstream_full_v1_summary.json").read_text())
    row = summary["per_task_val_metric"][0]
    assert row["tag"] == "lr_1e-6"
    assert row["lr_backbone"] == pytest.approx(1e-6)
    assert summary["selection_split"] == "val"
