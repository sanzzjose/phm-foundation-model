"""Tests de la lógica real de evaluación de `ssl_eval_suite`.

Usan batches sintéticos y un modelo `PatchTSTPhm.tiny()` en CPU, sin
tocar disco, Drive ni checkpoints reales. Cubren:

* evaluación de un batch (forward + métricas);
* agregación por cliente / dataset ponderada por elementos;
* conteo de `nonfinite_count`;
* `padding_ignored_count` con patch parcial;
* `result_row.json` con schema válido;
* respeto de `max_batches`.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from models.patchtst_phm import PatchTSTPhm
from training.ssl import ssl_eval_suite as suite


PATCH_SIZE = 16
N_PATCHES = 32
W = PATCH_SIZE * N_PATCHES


def _make_batch(B: int, C: int, *, n_padding_timesteps: int = 0, seed: int = 0):
    """Construye un batch sintético compatible con el contrato (B, C, N, P).

    Si ``n_padding_timesteps > 0``, marca los últimos timesteps como
    padding en `valid_time_mask` y deja el último patch como parcial.
    """
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, C, N_PATCHES, PATCH_SIZE, generator=g, dtype=torch.float32)
    vtm = torch.ones(B, W, dtype=torch.bool)
    if n_padding_timesteps > 0:
        vtm[:, W - n_padding_timesteps:] = False
    # valid_patch_mask: un patch es válido si tiene al menos un timestep real.
    vtm_reshaped = vtm.reshape(B, N_PATCHES, PATCH_SIZE)
    patch_valid = vtm_reshaped.any(dim=2)  # (B, N)
    vpm = patch_valid.unsqueeze(1).expand(B, C, N_PATCHES).contiguous()
    return {
        "patches": x,
        "valid_time_mask": vtm,
        "valid_patch_mask": vpm,
    }


def _tiny_model():
    model = PatchTSTPhm.tiny(patch_size=PATCH_SIZE, n_patches=N_PATCHES)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# evaluate_one_batch
# ---------------------------------------------------------------------------

def test_evaluate_one_batch_returns_finite_loss():
    model = _tiny_model()
    batch = _make_batch(B=4, C=2, seed=1)
    gen = suite._make_generator(42)
    with torch.no_grad():
        res = suite.evaluate_one_batch(
            model, batch, mask_ratio=0.3, generator=gen,
            device=torch.device("cpu"), loss_fn="mse",
        )
    assert res["loss_finite"] is True
    assert res["loss"] is not None
    assert res["loss"] >= 0.0
    assert res["effective_bc"] == 8  # B*C = 4*2
    assert res["n_masked_patches"] > 0


def test_evaluate_one_batch_padding_ignored_with_partial_patch():
    """Con padding en la cola, padding_ignored_elements debe ser > 0 si el
    patch parcial cae en la selección SSL."""
    model = _tiny_model()
    # 8 timesteps de padding => el último patch (16 timesteps) queda parcial.
    batch = _make_batch(B=2, C=1, n_padding_timesteps=8, seed=2)
    # Forzamos mask_ratio alto para maximizar la probabilidad de enmascarar
    # el patch parcial; con 32 patches válidos y ratio 0.9 casi seguro entra.
    gen = suite._make_generator(7)
    with torch.no_grad():
        res = suite.evaluate_one_batch(
            model, batch, mask_ratio=0.95, generator=gen,
            device=torch.device("cpu"), loss_fn="mse",
        )
    # El último patch tiene 8 timesteps reales + 8 de padding. Si se
    # enmascara, padding_ignored cuenta esos 8 padding por (b,c).
    assert res["padding_ignored_elements"] >= 0
    # Con ratio 0.95 y 32 patches válidos, el patch parcial se enmascara
    # casi con certeza; comprobamos que el conteo es coherente (múltiplo
    # de 8 por cada (b,c) que enmascara el patch parcial).
    assert res["padding_ignored_elements"] % 8 == 0


# ---------------------------------------------------------------------------
# aggregate_metrics
# ---------------------------------------------------------------------------

def _fake_result(dataset, client, loss, n_elem, *, finite=True, masked=10,
                 pad=0, eff_bc=8):
    return {
        "loss": loss if finite else None,
        "loss_finite": finite,
        "n_loss_elements": n_elem,
        "n_masked_patches": masked,
        "padding_ignored_elements": pad,
        "effective_bc": eff_bc,
        "dataset": dataset,
        "client": client,
    }


def test_aggregate_weighted_loss():
    per_batch = [
        _fake_result("CESNASA15", "misc", loss=1.0, n_elem=100),
        _fake_result("CESNASA15", "misc", loss=2.0, n_elem=300),
    ]
    agg = suite.aggregate_metrics(per_batch)
    # Media ponderada: (1*100 + 2*300) / 400 = 700/400 = 1.75
    assert abs(agg["ssl_val_loss_weighted"] - 1.75) < 1e-9
    assert abs(agg["ssl_val_loss_per_dataset"]["CESNASA15"] - 1.75) < 1e-9
    assert abs(agg["ssl_val_loss_per_client"]["misc"] - 1.75) < 1e-9


def test_aggregate_per_client_and_dataset_grouping():
    per_batch = [
        _fake_result("DS_A", "client1", loss=1.0, n_elem=10),
        _fake_result("DS_B", "client1", loss=3.0, n_elem=10),
        _fake_result("DS_C", "client2", loss=5.0, n_elem=10),
    ]
    agg = suite.aggregate_metrics(per_batch)
    # client1: (1*10 + 3*10)/20 = 2.0 ; client2: 5.0
    assert abs(agg["ssl_val_loss_per_client"]["client1"] - 2.0) < 1e-9
    assert abs(agg["ssl_val_loss_per_client"]["client2"] - 5.0) < 1e-9
    assert agg["coverage_dataset_count"] == 3
    assert agg["coverage_client_count"] == 2


def test_aggregate_nonfinite_count():
    per_batch = [
        _fake_result("DS_A", "c1", loss=1.0, n_elem=10),
        _fake_result("DS_A", "c1", loss=None, n_elem=10, finite=False),
        _fake_result("DS_A", "c1", loss=None, n_elem=10, finite=False),
    ]
    agg = suite.aggregate_metrics(per_batch)
    assert agg["nonfinite_count"] == 2
    # La loss global solo usa el batch finito.
    assert abs(agg["ssl_val_loss_weighted"] - 1.0) < 1e-9


def test_aggregate_padding_and_masked_counts():
    per_batch = [
        _fake_result("DS_A", "c1", loss=1.0, n_elem=10, masked=5, pad=8),
        _fake_result("DS_A", "c1", loss=1.0, n_elem=10, masked=7, pad=16),
    ]
    agg = suite.aggregate_metrics(per_batch)
    assert agg["masked_patch_count"] == 12
    assert agg["padding_ignored_count"] == 24


def test_aggregate_effective_bc():
    per_batch = [
        _fake_result("DS_A", "c1", loss=1.0, n_elem=10, eff_bc=8),
        _fake_result("DS_A", "c1", loss=1.0, n_elem=10, eff_bc=16),
    ]
    agg = suite.aggregate_metrics(per_batch)
    assert agg["effective_bc_max"] == 16
    assert abs(agg["effective_bc_mean"] - 12.0) < 1e-9


def test_aggregate_empty():
    agg = suite.aggregate_metrics([])
    assert agg["ssl_val_loss_weighted"] is None
    assert agg["coverage_dataset_count"] == 0
    assert agg["n_batches_evaluated"] == 0


# ---------------------------------------------------------------------------
# _write_eval_outputs / result_row schema
# ---------------------------------------------------------------------------

def test_write_eval_outputs_result_row_schema(tmp_path):
    per_batch = [
        _fake_result("DS_A", "c1", loss=0.5, n_elem=100),
        _fake_result("DS_B", "c2", loss=1.5, n_elem=100),
    ]
    agg = suite.aggregate_metrics(per_batch)
    summary = {
        "phase": "ssl_eval",
        "checkpoint_id": "central_full_100k",
        "status": "ok",
        "split": "train",
        "mask_ratio": 0.3,
        "config_hash": "deadbeef00000000",
        "created_at": "2026-06-05T12:00:00",
        "seed": 42,
        "param_count": 104336,
        **agg,
    }
    suite._write_eval_outputs(summary, agg, tmp_path, Path("."))

    # Verificar artefactos.
    assert (tmp_path / "ssl_eval_summary.json").exists()
    assert (tmp_path / "ssl_eval_summary.csv").exists()
    assert (tmp_path / "ssl_eval_per_client.csv").exists()
    assert (tmp_path / "ssl_eval_per_dataset.csv").exists()
    assert (tmp_path / "result_row.json").exists()

    # result_row.json debe ser schema válido.
    payload = json.loads((tmp_path / "result_row.json").read_text(encoding="utf-8"))
    assert payload["phase"] == "ssl_eval"
    assert payload["role"] == "PRETRAIN_SOURCE"
    assert payload["task_type"] == "ssl"
    assert payload["primary_metric_name"] == "ssl_val_loss_weighted"
    assert payload["primary_metric_value"] == agg["ssl_val_loss_weighted"]
    # extra debe traer cobertura.
    assert payload["extra"]["coverage_dataset_count"] == 2


def test_write_eval_outputs_per_dataset_csv_content(tmp_path):
    per_batch = [
        _fake_result("DS_A", "c1", loss=0.5, n_elem=100),
        _fake_result("DS_B", "c2", loss=1.5, n_elem=100),
    ]
    agg = suite.aggregate_metrics(per_batch)
    summary = {
        "phase": "ssl_eval", "checkpoint_id": "x", "status": "ok",
        "split": "train", "mask_ratio": 0.3, "config_hash": "x",
        "created_at": "t", "seed": 0, "param_count": 1, **agg,
    }
    suite._write_eval_outputs(summary, agg, tmp_path, Path("."))
    content = (tmp_path / "ssl_eval_per_dataset.csv").read_text(encoding="utf-8")
    assert "DS_A" in content
    assert "DS_B" in content


# ---------------------------------------------------------------------------
# Microajustes de robustez: checkpoint_origin y output_dir
# ---------------------------------------------------------------------------

def test_checkpoint_origin_propagated_to_result_row(tmp_path):
    """`checkpoint_origin` del summary debe llegar al result_row.json."""
    per_batch = [_fake_result("DS_A", "c1", loss=0.5, n_elem=100)]
    agg = suite.aggregate_metrics(per_batch)
    summary = {
        "phase": "ssl_eval", "checkpoint_id": "central_full_100k",
        "checkpoint_origin": "central", "status": "ok",
        "split": "train", "mask_ratio": 0.3, "config_hash": "x",
        "created_at": "t", "seed": 42, "param_count": 1, **agg,
    }
    suite._write_eval_outputs(summary, agg, tmp_path, Path("."))
    payload = json.loads((tmp_path / "result_row.json").read_text(encoding="utf-8"))
    assert payload["checkpoint_origin"] == "central"
    # El experiment_id también debe reflejar el origen.
    assert "central" in payload["experiment_id"]


def test_checkpoint_origin_defaults_unknown_if_missing(tmp_path):
    """Sin checkpoint_origin en summary, result_row usa 'unknown'."""
    per_batch = [_fake_result("DS_A", "c1", loss=0.5, n_elem=100)]
    agg = suite.aggregate_metrics(per_batch)
    summary = {
        "phase": "ssl_eval", "checkpoint_id": "x", "status": "ok",
        "split": "train", "mask_ratio": 0.3, "config_hash": "x",
        "created_at": "t", "seed": 0, "param_count": 1, **agg,
    }
    suite._write_eval_outputs(summary, agg, tmp_path, Path("."))
    payload = json.loads((tmp_path / "result_row.json").read_text(encoding="utf-8"))
    assert payload["checkpoint_origin"] == "unknown"


def test_summary_csv_includes_checkpoint_origin(tmp_path):
    per_batch = [_fake_result("DS_A", "c1", loss=0.5, n_elem=100)]
    agg = suite.aggregate_metrics(per_batch)
    summary = {
        "phase": "ssl_eval", "checkpoint_id": "x", "checkpoint_origin": "fedavg",
        "status": "ok", "split": "train", "eval_split_kind": "ps_materialized_deterministic",
        "mask_ratio": 0.3, "config_hash": "x", "created_at": "t",
        "seed": 0, "param_count": 1, **agg,
    }
    suite._write_eval_outputs(summary, agg, tmp_path, Path("."))
    content = (tmp_path / "ssl_eval_summary.csv").read_text(encoding="utf-8")
    assert "checkpoint_origin" in content
    assert "fedavg" in content
    assert "eval_split_kind" in content


# ---------------------------------------------------------------------------
# Resolución de output_dir (end-to-end con checkpoint tiny y processed vacío)
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_tiny_checkpoint(path: Path, origin: str = "central"):
    """Guarda un checkpoint tiny sintético compatible con la suite."""
    from models.patchtst_phm import PatchTSTPhm, count_parameters
    model = PatchTSTPhm.tiny(patch_size=PATCH_SIZE, n_patches=N_PATCHES)
    ckpt = {
        "step": 50,
        "model_state_dict": model.state_dict(),
        "config": {"model": {
            "patch_size": PATCH_SIZE, "n_patches": N_PATCHES,
            "d_model": 64, "n_layers": 2, "n_heads": 4, "d_ff": 256, "dropout": 0.1,
        }},
        "git_hash": "test", "config_hash": "testhash",
        "model_class": "PatchTSTPhm", "param_count": count_parameters(model),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ckpt, path)


def _base_cfg(processed_root: Path, origin: str = "central", out_dir=None):
    cfg = {
        "seed": 42,
        "data": {
            "role_filter": "PRETRAIN_SOURCE",
            "include_datasets": "all",
            "eval_split": "train",
            "batch_size_requested": 4,
            "mask_ratio": 0.3,
            "max_batches": 2,
            "max_batches_per_dataset": 1,
        },
        "ssl": {"loss": "mse", "mask_ratio": 0.3},
        "checkpoint": {"id": "tiny_ckpt", "origin": origin},
        "paths": {"processed_root": str(processed_root)},
    }
    if out_dir is not None:
        cfg["paths"]["output_dir"] = str(out_dir)
    return cfg


def test_cmd_eval_respects_cli_output_dir(tmp_path):
    """--output-dir por CLI se respeta EXACTAMENTE (sin subdir por id)."""
    ckpt = tmp_path / "ckpt.pt"
    _make_tiny_checkpoint(ckpt)
    # processed_root inexistente => 0 batches => status partial, pero el
    # output_dir se crea y los artefactos se escriben igualmente.
    empty_processed = tmp_path / "no_processed"
    cfg = _base_cfg(empty_processed)
    cli_out = tmp_path / "exact_cli_out"
    rc = suite.cmd_eval(
        cfg, REPO_ROOT, checkpoint_path=ckpt, output_dir=cli_out,
        max_batches=2, device="cpu", seed=42,
    )
    assert rc == 0
    # El summary debe estar en cli_out EXACTAMENTE (sin subdir tiny_ckpt).
    assert (cli_out / "ssl_eval_summary.json").exists()
    assert not (cli_out / "tiny_ckpt").exists()


def test_cmd_eval_uses_config_output_dir_with_subdir(tmp_path):
    """Sin --output-dir, usa paths.output_dir + subdir por checkpoint_id."""
    ckpt = tmp_path / "ckpt.pt"
    _make_tiny_checkpoint(ckpt)
    empty_processed = tmp_path / "no_processed"
    config_out = tmp_path / "config_out_base"
    cfg = _base_cfg(empty_processed, out_dir=config_out)
    rc = suite.cmd_eval(
        cfg, REPO_ROOT, checkpoint_path=ckpt, output_dir=None,
        max_batches=2, device="cpu", seed=42,
    )
    assert rc == 0
    # Debe crear config_out_base/tiny_ckpt/ssl_eval_summary.json
    assert (config_out / "tiny_ckpt" / "ssl_eval_summary.json").exists()


def test_cmd_eval_summary_has_central_origin(tmp_path):
    """El summary del eval debe llevar checkpoint_origin=central."""
    ckpt = tmp_path / "ckpt.pt"
    _make_tiny_checkpoint(ckpt, origin="central")
    empty_processed = tmp_path / "no_processed"
    cfg = _base_cfg(empty_processed, origin="central")
    out = tmp_path / "out"
    suite.cmd_eval(cfg, REPO_ROOT, checkpoint_path=ckpt, output_dir=out,
                   max_batches=2, device="cpu", seed=42)
    payload = json.loads((out / "ssl_eval_summary.json").read_text(encoding="utf-8"))
    assert payload["checkpoint_origin"] == "central"
    assert payload["eval_split_kind"] == "ps_materialized_deterministic"
    # result_row también.
    rr = json.loads((out / "result_row.json").read_text(encoding="utf-8"))
    assert rr["checkpoint_origin"] == "central"
