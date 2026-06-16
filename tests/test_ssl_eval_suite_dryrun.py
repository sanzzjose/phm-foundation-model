"""Tests del dry-run de la SSL validation suite.

No ejecutan la evaluación real: validan que el módulo lee la config,
detecta los 36 PRETRAIN_SOURCE en ``results/processed_summary.csv`` y
rechaza configs que mencionen explícitamente TRANSFER_TARGET.
"""
from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from training.ssl import ssl_eval_suite as suite


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_load_default_config():
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    assert isinstance(cfg, dict)
    assert cfg["data"]["role_filter"] == "PRETRAIN_SOURCE"


def test_dry_run_detects_36_ps():
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    rc = suite.cmd_dry_run(cfg, REPO_ROOT)
    assert rc == 0


def test_dry_run_rejects_explicit_tt(tmp_path):
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    # Si la config menciona explícitamente un TRANSFER_TARGET, debe abortar.
    cfg.setdefault("data", {})["datasets"] = ["CWRU"]
    with pytest.raises(ValueError) as exc:
        suite.cmd_dry_run(cfg, REPO_ROOT)
    assert "TRANSFER_TARGET" in str(exc.value)


def test_eval_requires_checkpoint(tmp_path):
    """`cmd_eval` sin checkpoint debe abortar con ValueError claro."""
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    with pytest.raises(ValueError) as exc:
        suite.cmd_eval(cfg, REPO_ROOT, checkpoint_path=None, output_dir=tmp_path)
    assert "checkpoint" in str(exc.value).lower()


def test_eval_checkpoint_not_found(tmp_path):
    """`cmd_eval` con ruta inexistente debe fallar con FileNotFoundError."""
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    fake = tmp_path / "no_existe.pt"
    with pytest.raises(FileNotFoundError):
        suite.cmd_eval(cfg, REPO_ROOT, checkpoint_path=fake, output_dir=tmp_path)


def test_assert_no_tt_via_include_datasets(tmp_path):
    """`include_datasets` con un TT también debe abortar."""
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    cfg.setdefault("data", {})["include_datasets"] = ["CWRU", "HSG18"]
    with pytest.raises(ValueError) as exc:
        suite.cmd_dry_run(cfg, REPO_ROOT)
    assert "TRANSFER_TARGET" in str(exc.value)


def test_assert_no_tt_allows_all_keyword():
    """`include_datasets: all` NO debe abortar (no enumera TT)."""
    cfg = suite.load_config(REPO_ROOT / "training/configs/ssl_eval_suite.yaml")
    # La config por defecto trae include_datasets: all -> dry-run pasa.
    rc = suite.cmd_dry_run(cfg, REPO_ROOT)
    assert rc == 0


# ---------------------------------------------------------------------------
# checkpoint_origin / checkpoint_id (CLI > config > default)
# ---------------------------------------------------------------------------

def test_resolve_checkpoint_origin_cli_priority():
    """El CLI gana sobre la config."""
    assert suite.resolve_checkpoint_origin("fedavg", {"origin": "central"}) == "fedavg"


def test_resolve_checkpoint_origin_config_fallback():
    """Sin CLI, usa config.checkpoint.origin."""
    assert suite.resolve_checkpoint_origin(None, {"origin": "central"}) == "central"


def test_resolve_checkpoint_origin_default_unknown():
    """Sin CLI ni config, devuelve 'unknown' (compatibilidad)."""
    assert suite.resolve_checkpoint_origin(None, {}) == "unknown"


def test_resolve_checkpoint_id_cli_priority():
    assert suite.resolve_checkpoint_id("central_5k", {"id": "cfgid"}, "stem") == "central_5k"


def test_resolve_checkpoint_id_config_then_stem():
    assert suite.resolve_checkpoint_id(None, {"id": "cfgid"}, "stem") == "cfgid"
    assert suite.resolve_checkpoint_id(None, {}, "stem") == "stem"


def test_cli_parses_checkpoint_origin_and_id():
    """`--checkpoint-origin` / `--checkpoint-id` se parsean correctamente."""
    args = suite.parse_args([
        "--mode", "eval", "--checkpoint-origin", "fedprox",
        "--checkpoint-id", "central_5k",
    ])
    assert args.checkpoint_origin == "fedprox"
    assert args.checkpoint_id == "central_5k"


def test_cli_defaults_are_none_for_origin_and_id():
    """Sin flags, ambos son None (default compatible con el comportamiento previo)."""
    args = suite.parse_args([])
    assert args.checkpoint_origin is None
    assert args.checkpoint_id is None


def test_cli_rejects_invalid_origin():
    """`--checkpoint-origin` con valor fuera del conjunto cerrado aborta."""
    with pytest.raises(SystemExit):
        suite.parse_args(["--checkpoint-origin", "no_such_origin"])
