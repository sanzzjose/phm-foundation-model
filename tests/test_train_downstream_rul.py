"""Tests del trainer downstream RUL (`training/train_downstream_rul.py`).

Cubren:

- `find_shards_rul`: localizacion segun el patron del builder.
- `_extract_target`: validacion de target_key y fallback meta -> npy.
- `_collate`: apila targets float y patches.
- Dry-run via CLI sobre un directorio sintetico con manifest y shards
  fake (sin entrenar).
- 1 step end-to-end con sample sintetico (skip Windows local por torch).

Para los tests que requieren torch/PatchTSTPhm se aplica el mismo skip
defensivo por plataforma que en los otros tests downstream.
"""

from __future__ import annotations

import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

from training.train_downstream_rul import (
    ALLOWED_TARGET_KEYS,
    RUL_DATASET_NAME,
    _collate,
    _extract_target,
    find_shards_rul,
    resolve_rul_batch_size,
)


# ----------------------------------------------------------------------
# Helpers para construir un dataset sintetico estilo CMAPSS_RUL.
# ----------------------------------------------------------------------


def _build_synthetic_rul_corpus(
    root: Path,
    n_train: int = 8,
    n_val: int = 4,
    n_test: int = 6,
    C: int = 24,
    N: int = 4,
    P: int = 2,
):
    """Crea un dataset CMAPSS_RUL sintetico minimo en `root`.

    Estructura:
        root/CMAPSS_RUL/manifest.json
        root/CMAPSS_RUL/train/shard_0000.tar  (n_train samples)
        root/CMAPSS_RUL/val/shard_0000.tar    (n_val samples)
        root/CMAPSS_RUL/test/shard_0000.tar   (n_test samples)

    Dimensiones reducidas (N=4, P=2) para que los tests vayan rapidos.
    """
    W = N * P
    ds_root = Path(root) / RUL_DATASET_NAME
    ds_root.mkdir(parents=True, exist_ok=True)

    # Manifest minimo con los campos que lee el trainer.
    manifest = {
        "dataset": RUL_DATASET_NAME,
        "role": "TRANSFER_TARGET",
        "client": "aero_engines",
        "window_size": W,
        "patch_size": P,
        "n_patches": N,
        "n_channels": C,
        "target_policy": "rul_at_prediction_cycle",
        "target_candidates": list(ALLOWED_TARGET_KEYS),
        "stride": 5,
        "include_last_per_unit": True,
        "min_valid_timesteps": 2,
    }
    (ds_root / "manifest.json").write_text(json.dumps(manifest, indent=2))

    splits = {"train": n_train, "val": n_val, "test": n_test}
    rng = np.random.default_rng(0)
    for split, n in splits.items():
        split_dir = ds_root / split
        split_dir.mkdir(parents=True, exist_ok=True)
        tar_path = split_dir / "shard_0000.tar"
        with tarfile.open(tar_path, "w") as tf:
            for i in range(n):
                key = f"cmapss_synth_{split}_unit{i}_w{i:06d}"
                patches = rng.normal(size=(C, N, P)).astype(np.float32)
                vtm = np.ones(W, dtype=bool)
                vpm = np.ones((C, N), dtype=bool)
                cc = np.zeros(C, dtype=bool)
                mean_c = patches.mean(axis=(1, 2)).astype(np.float32)
                std_c = (patches.std(axis=(1, 2)) + 1e-6).astype(np.float32)
                rul_phys = np.float32(50.0 + i)
                rul_capped = np.float32(min(float(rul_phys), 125.0))
                meta = {
                    "fd_subset": "FD001",
                    "split": split,
                    "source_split": "train_orig" if split != "test" else "test_orig",
                    "unit_id": i,
                    "unit_global_id": f"CMAPSS_FD001_{'train_orig' if split != 'test' else 'test_orig'}_unit{i}",
                    "t_idx_in_unit": i,
                    "cycle": i + 1,
                    "valid_timesteps": W,
                    "selected_by_last_override": False,
                    "below_min_valid_because_last": False,
                    "window_size": W,
                    "patch_size": P,
                    "n_patches": N,
                    "n_channels": C,
                    "target_rul_physical": float(rul_phys),
                    "target_rul_capped_125": float(rul_capped),
                }

                def _npy(arr):
                    buf = io.BytesIO()
                    np.save(buf, arr, allow_pickle=False)
                    return buf.getvalue()

                blobs = {
                    "patches.npy": _npy(patches),
                    "valid_time_mask.npy": _npy(vtm),
                    "valid_patch_mask.npy": _npy(vpm),
                    "canales_constantes_mask.npy": _npy(cc),
                    "mean.npy": _npy(mean_c),
                    "std_used.npy": _npy(std_c),
                    "rul_physical.npy": _npy(rul_phys),
                    "rul_capped_125.npy": _npy(rul_capped),
                    "meta.json": json.dumps(meta, sort_keys=True).encode("utf-8"),
                }
                for ext in sorted(blobs.keys()):
                    data = blobs[ext]
                    info = tarfile.TarInfo(name=f"{key}.{ext}")
                    info.size = len(data)
                    info.mtime = 0
                    tf.addfile(info, io.BytesIO(data))
    return ds_root


# ----------------------------------------------------------------------
# find_shards_rul
# ----------------------------------------------------------------------


def test_find_shards_rul_localiza_los_3_splits(tmp_path):
    _build_synthetic_rul_corpus(tmp_path)
    for split, expected_n in [("train", 1), ("val", 1), ("test", 1)]:
        shards = find_shards_rul(tmp_path, split)
        assert len(shards) == expected_n, (split, shards)
        assert all(s.name.startswith("shard_") and s.suffix == ".tar"
                   for s in shards)


def test_find_shards_rul_vacio_si_no_existe(tmp_path):
    assert find_shards_rul(tmp_path / "no_existe", "train") == []
    # Directorio que existe pero sin shards.
    (tmp_path / RUL_DATASET_NAME / "train").mkdir(parents=True)
    assert find_shards_rul(tmp_path, "train") == []


# ----------------------------------------------------------------------
# _extract_target
# ----------------------------------------------------------------------


def test_extract_target_prefiere_meta():
    sample = {
        "__key__": "k",
        "meta": {"target_rul_physical": 42.0, "target_rul_capped_125": 42.0},
    }
    assert _extract_target(sample, "rul_physical") == 42.0
    assert _extract_target(sample, "rul_capped_125") == 42.0


def test_extract_target_fallback_npy():
    """Si meta no tiene la clave, usa el escalar .npy."""
    sample = {
        "__key__": "k",
        "meta": {},
        "rul_physical": np.float32(7.0),
    }
    assert _extract_target(sample, "rul_physical") == 7.0


def test_extract_target_rechaza_key_desconocida():
    sample = {"__key__": "k", "meta": {"target_rul_physical": 1.0}}
    with pytest.raises(ValueError, match="target_key desconocido"):
        _extract_target(sample, "rul_unknown")


def test_extract_target_falla_si_no_hay_target():
    sample = {"__key__": "k", "meta": {}}
    with pytest.raises(ValueError, match="no contiene"):
        _extract_target(sample, "rul_physical")


# ----------------------------------------------------------------------
# resolve_rul_batch_size
# ----------------------------------------------------------------------


def test_resolve_rul_batch_size_manifest_24_canales(tmp_path):
    """Manifest con n_channels=24: el cap B*C<=512 da B<=21."""
    _build_synthetic_rul_corpus(tmp_path, C=24)
    bs = resolve_rul_batch_size(
        tmp_path, {"batch_size": 64, "max_channel_batch": 512},
    )
    assert bs["n_channels"] == 24
    assert bs["n_channels_source"] == "manifest"
    # 64 * 24 = 1536 > 512 -> B_eff <= 512/24 = 21
    assert bs["batch_size_effective"] <= 21
    assert bs["effective_bc"] <= 512


def test_resolve_rul_batch_size_fallback_config(tmp_path):
    """Sin manifest, usa n_channels_fallback de data_cfg."""
    # Sin construir corpus.
    bs = resolve_rul_batch_size(
        tmp_path, {"batch_size": 32, "n_channels_fallback": 24},
    )
    assert bs["n_channels"] == 24
    assert bs["n_channels_source"] == "config_fallback"
    assert bs["warnings"]  # debe advertir


def test_resolve_rul_batch_size_falla_sin_manifest_sin_fallback(tmp_path):
    """Sin manifest y sin n_channels_fallback, aborta duro."""
    with pytest.raises(RuntimeError, match="No se pudo inferir n_channels"):
        resolve_rul_batch_size(tmp_path, {"batch_size": 32})


# ----------------------------------------------------------------------
# _collate
# ----------------------------------------------------------------------


def test_collate_apila_targets_float(tmp_path):
    """Verifica que _collate produce targets float32 (B,)."""
    torch = pytest.importorskip("torch")
    samples = []
    for v in (10.0, 20.0, 30.0):
        samples.append({
            "patches": torch.zeros(2, 4, 2),
            "valid_time_mask": torch.ones(8, dtype=torch.bool),
            "valid_patch_mask": torch.ones(2, 4, dtype=torch.bool),
            "canales_constantes_mask": torch.zeros(2, dtype=torch.bool),
            "target": v,
            "__key__": f"k{v}",
        })
    batch = _collate(samples)
    assert batch["targets"].shape == (3,)
    assert batch["targets"].dtype == torch.float32
    assert batch["targets"].tolist() == [10.0, 20.0, 30.0]
    assert batch["patches"].shape == (3, 2, 4, 2)
    assert batch["valid_time_mask"].shape == (3, 8)
    assert batch["valid_patch_mask"].shape == (3, 2, 4)


def test_collate_rechaza_patches_heterogeneos():
    torch = pytest.importorskip("torch")
    s1 = {
        "patches": torch.zeros(2, 4, 2),
        "valid_time_mask": torch.ones(8, dtype=torch.bool),
        "valid_patch_mask": torch.ones(2, 4, dtype=torch.bool),
        "canales_constantes_mask": torch.zeros(2, dtype=torch.bool),
        "target": 10.0,
        "__key__": "k1",
    }
    s2 = dict(s1)
    s2["patches"] = torch.zeros(3, 4, 2)  # C distinto
    with pytest.raises(ValueError, match="heterogeneo"):
        _collate([s1, s2])


# ----------------------------------------------------------------------
# CLI dry-run (no entrena)
# ----------------------------------------------------------------------


def test_cli_dry_run_sin_drive(tmp_path):
    """`--dry-run` con processed_downstream_root inexistente no aborta:
    cuenta 0 shards y reporta. Necesita n_channels_fallback para que
    resolve_rul_batch_size no falle.
    """
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    from training.train_downstream_rul import main
    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    cfg_path.write_text(
        f"""
run_name: rul_dryrun_test
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: null
  dropout: 0.0
  activation: null
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}/no_existe
  target_key: rul_capped_125
  n_channels_fallback: 24
  batch_size: 4
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.001
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )
    rc = main([
        "--config", str(cfg_path),
        "--mode", "from_scratch",
        "--dry-run",
    ])
    assert rc == 0


def test_cli_dry_run_con_corpus_sintetico(tmp_path):
    """`--dry-run` con corpus sintetico: cuenta 1 shard por split, lee
    manifest, hace forward sintetico, sale rc=0."""
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    from training.train_downstream_rul import main
    _build_synthetic_rul_corpus(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    cfg_path.write_text(
        f"""
run_name: rul_dryrun_synth
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: 16
  dropout: 0.0
  activation: gelu
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}
  target_key: rul_capped_125
  n_channels_fallback: 24
  batch_size: 2
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.001
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )
    rc = main([
        "--config", str(cfg_path),
        "--mode", "from_scratch",
        "--dry-run",
    ])
    assert rc == 0


# ----------------------------------------------------------------------
# 1 step end-to-end sintetico (entrena UNA epoca con max_train_batches_per_epoch=2)
# ----------------------------------------------------------------------


def test_train_e2e_sintetico_una_epoca(tmp_path):
    """Corre un training real corto sobre el corpus sintetico:
    1 epoca con cap 2 batches/epoca, modo from_scratch. Verifica que
    se generan los artefactos esperados (run_info.json, metrics.jsonl,
    best.pt) y que `best_value` y `best_epoch` quedan registrados.
    """
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    from training.train_downstream_rul import main
    _build_synthetic_rul_corpus(tmp_path, n_train=8, n_val=4, n_test=4)
    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    cfg_path.write_text(
        f"""
run_name: rul_train_synth_e2e
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: null
  dropout: 0.0
  activation: null
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}
  target_key: rul_capped_125
  batch_size: 2
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.01
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
  max_train_batches_per_epoch: 2
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )
    rc = main([
        "--config", str(cfg_path),
        "--mode", "from_scratch",
    ])
    assert rc == 0
    # Artefactos esperados.
    run_info_path = log_dir / "rul_train_synth_e2e" / "run_info.json"
    metrics_path = log_dir / "rul_train_synth_e2e" / "metrics.jsonl"
    ckpt_path = ckpt_dir / "rul_train_synth_e2e" / "best.pt"
    assert run_info_path.is_file()
    assert metrics_path.is_file()
    assert ckpt_path.is_file()
    ri = json.loads(run_info_path.read_text())
    assert ri["dataset"] == "CMAPSS_RUL"
    assert ri["target_key"] == "rul_capped_125"
    assert ri["mode"] == "from_scratch"
    assert ri["best_epoch"] == 1
    assert ri["metric_for_best"] == "rmse_val"
    assert ri["lower_is_better"] is True
    # best_value debe ser un float (rmse computado en val).
    assert ri["best_value"] is not None and ri["best_value"] >= 0.0
    # test_metrics presente (porque hicimos eval test con best ckpt).
    assert ri["test_metrics"] is not None
    for k in ("mae", "rmse", "r2", "cmapss_score"):
        assert k in ri["test_metrics"]


def test_train_acepta_lr_backbone_null_en_yaml(tmp_path):
    """`lr_backbone: null` en el YAML (caso linear_probing por defecto)
    debe interpretarse como "un solo grupo" sin reventar con
    `float(None)`. Regresion del bug del commit 5 inicial.
    """
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    from training.train_downstream_rul import main
    _build_synthetic_rul_corpus(tmp_path, n_train=4, n_val=2, n_test=2)
    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    cfg_path.write_text(
        f"""
run_name: rul_train_lr_backbone_null
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: null
  dropout: 0.0
  activation: null
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}
  target_key: rul_capped_125
  batch_size: 2
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.001
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
  max_train_batches_per_epoch: 1
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )
    rc = main([
        "--config", str(cfg_path),
        "--mode", "from_scratch",
    ])
    assert rc == 0


def test_cmd_dry_run_pasa_mode_y_checkpoint_reales_a_load_regressor(
    tmp_path, monkeypatch
):
    """Regresion: cmd_dry_run debe llamar a load_regressor con el mode y
    checkpoint reales (no hardcoded 'from_scratch'/None).

    Antes del fix, el dry-run en modo linear_probing o full_finetuning
    NO ejercia la carga del ckpt SSL: invocaba load_regressor con
    ("from_scratch", None) hardcoded, asi que decia "OK" sin haber
    intentado abrir el .pt en disco. Si el ckpt era invalido o estaba
    mal nombrado, el dry-run no lo detectaba.

    Este test parchea load_regressor con un fake que captura mode y
    checkpoint recibidos, lanza el dry-run con --mode linear_probing
    --checkpoint <dummy>, y verifica que el fake recibe esos mismos
    valores (no los hardcoded del bug).
    """
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    import training.train_downstream_rul as mod
    torch = pytest.importorskip("torch")

    captured = {"calls": []}

    class _FakeBackbone:
        d_model = 8
        def parameters(self):
            return iter([])

    class _FakeModel(torch.nn.Module):
        d_model = 8
        def __init__(self):
            super().__init__()
            self.backbone = _FakeBackbone()
        def forward(self, x, valid_time_mask=None, valid_patch_mask=None,
                    canales_constantes_mask=None):
            B = x.shape[0]
            return {
                "prediction": torch.zeros(B),
                "pooled": torch.zeros(B, self.d_model),
                "tokens": torch.zeros(B, x.shape[1], x.shape[2], self.d_model),
            }

    def _fake_load_regressor(model_cfg, mode, checkpoint, head_cfg):
        captured["calls"].append({
            "mode": mode,
            "checkpoint": str(checkpoint) if checkpoint is not None else None,
        })
        return _FakeModel(), {"n_trainable": 1, "n_total": 1}

    monkeypatch.setattr(mod, "load_regressor", _fake_load_regressor)

    # Config minimo. No necesita shards reales (dry-run cuenta y reporta
    # pero no entrena). n_channels_fallback evita que
    # resolve_rul_batch_size aborte sin manifest.
    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    dummy_ckpt = tmp_path / "fake_ssl.pt"
    dummy_ckpt.write_bytes(b"")  # Existe en disco; load_regressor esta mockeado
    cfg_path.write_text(
        f"""
run_name: rul_dryrun_mode_passthrough
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: null
  dropout: 0.0
  activation: null
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}/no_existe
  target_key: rul_capped_125
  n_channels_fallback: 24
  batch_size: 2
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.001
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )

    rc = mod.main([
        "--config", str(cfg_path),
        "--mode", "linear_probing",
        "--checkpoint", str(dummy_ckpt),
        "--dry-run",
    ])
    assert rc == 0
    assert len(captured["calls"]) == 1, (
        f"load_regressor deberia llamarse exactamente una vez en el dry-run; "
        f"calls={captured['calls']}"
    )
    call = captured["calls"][0]
    assert call["mode"] == "linear_probing", (
        f"El dry-run paso mode={call['mode']!r} a load_regressor; "
        f"esperado 'linear_probing'. Regresion del bug pre-fix."
    )
    assert call["checkpoint"] == str(dummy_ckpt), (
        f"El dry-run paso checkpoint={call['checkpoint']!r}; "
        f"esperado {str(dummy_ckpt)!r}. Regresion del bug pre-fix."
    )


def test_cmd_dry_run_from_scratch_pasa_checkpoint_none(tmp_path, monkeypatch):
    """Complemento del test anterior: en --mode from_scratch sin
    --checkpoint, load_regressor debe recibir checkpoint=None tambien
    desde el dry-run.
    """
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    import training.train_downstream_rul as mod
    torch = pytest.importorskip("torch")

    captured = {"calls": []}

    class _FakeModel(torch.nn.Module):
        d_model = 8
        def forward(self, x, valid_time_mask=None, valid_patch_mask=None,
                    canales_constantes_mask=None):
            B = x.shape[0]
            return {
                "prediction": torch.zeros(B),
                "pooled": torch.zeros(B, self.d_model),
                "tokens": torch.zeros(B, x.shape[1], x.shape[2], self.d_model),
            }

    def _fake_load_regressor(model_cfg, mode, checkpoint, head_cfg):
        captured["calls"].append({
            "mode": mode,
            "checkpoint": checkpoint,
        })
        return _FakeModel(), {"n_trainable": 1, "n_total": 1}

    monkeypatch.setattr(mod, "load_regressor", _fake_load_regressor)

    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    cfg_path.write_text(
        f"""
run_name: rul_dryrun_from_scratch_no_ckpt
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: null
  dropout: 0.0
  activation: null
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}/no_existe
  target_key: rul_capped_125
  n_channels_fallback: 24
  batch_size: 2
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.001
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )
    rc = mod.main([
        "--config", str(cfg_path),
        "--mode", "from_scratch",
        "--dry-run",
    ])
    assert rc == 0
    assert len(captured["calls"]) == 1
    call = captured["calls"][0]
    assert call["mode"] == "from_scratch"
    assert call["checkpoint"] is None


def test_configs_oficiales_commit6_consistentes():
    """Los 3 YAMLs de las corridas oficiales (commit 6 del bloque RUL)
    deben tener:

    - run_name distinto entre los tres (para que el trainer no pise
      logs/ckpts entre modos);
    - misma base: dataset, task, target_key, batch_size_policy,
      metric_for_best, lower_is_better;
    - lr_head identico (1e-3);
    - lr_backbone especifico por modo:
        from_scratch     -> 1e-4
        linear_probing   -> None  (backbone congelado)
        full_finetuning  -> 1e-5  (sweet spot CWRU/HSG18).

    Este test es la red de seguridad ante cambios accidentales en los
    YAMLs antes de lanzar las 3 corridas reales. No requiere torch.
    """
    import yaml
    from pathlib import Path
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "training" / "configs"
    cfg_paths = {
        "from_scratch": cfg_dir / "downstream_cmapss_rul_from_scratch.yaml",
        "linear_probing": cfg_dir / "downstream_cmapss_rul_linear_probing.yaml",
        "full_finetuning_lr1e-5":
            cfg_dir / "downstream_cmapss_rul_full_finetuning_lr1e-5.yaml",
    }
    cfgs = {}
    for name, p in cfg_paths.items():
        assert p.is_file(), f"Falta config oficial: {p}"
        cfgs[name] = yaml.safe_load(p.read_text(encoding="utf-8"))

    # 1. run_name distinto en los 3.
    run_names = {n: c["run_name"] for n, c in cfgs.items()}
    assert len(set(run_names.values())) == 3, (
        f"run_name debe ser distinto en los 3 configs; recibido: {run_names}"
    )
    assert run_names["from_scratch"] == "downstream_cmapss_rul_from_scratch"
    assert run_names["linear_probing"] == "downstream_cmapss_rul_linear_probing"
    assert run_names["full_finetuning_lr1e-5"] == \
        "downstream_cmapss_rul_full_finetuning_lr1e-5"

    # 2. Base comun en los 3.
    for name, c in cfgs.items():
        assert c["dataset"] == "CMAPSS_RUL", name
        assert c["task"] == "regression_rul", name
        assert c["data"]["target_key"] == "rul_capped_125", name
        assert c["data"]["batch_size_policy"] == "adaptive_by_channels", name
        assert c["training"]["metric_for_best"] == "rmse_val", name
        assert c["training"]["lower_is_better"] is True, name
        # Cabeza minima lineal.
        assert c["head"]["hidden_dim"] is None, name
        assert float(c["head"]["dropout"]) == 0.0, name
        # lr_head identico.
        assert float(c["training"]["lr_head"]) == 0.001, (
            name, c["training"]["lr_head"],
        )

    # 3. lr_backbone especifico por modo.
    fs_lrb = cfgs["from_scratch"]["training"]["lr_backbone"]
    assert fs_lrb is not None and float(fs_lrb) == 1e-4, fs_lrb
    lp_lrb = cfgs["linear_probing"]["training"]["lr_backbone"]
    assert lp_lrb is None, lp_lrb
    ft_lrb = cfgs["full_finetuning_lr1e-5"]["training"]["lr_backbone"]
    assert ft_lrb is not None and float(ft_lrb) == 1e-5, ft_lrb


def test_train_aborta_si_target_key_invalido(tmp_path):
    """`target_key` fuera de ALLOWED_TARGET_KEYS hace abortar el train."""
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    from training.train_downstream_rul import main
    _build_synthetic_rul_corpus(tmp_path)
    cfg_path = tmp_path / "cfg.yaml"
    log_dir = tmp_path / "logs"
    ckpt_dir = tmp_path / "ckpts"
    cfg_path.write_text(
        f"""
run_name: rul_train_bad_target
seed: 42
dataset: CMAPSS_RUL
task: regression_rul
model:
  patch_size: 2
  n_patches: 4
  d_model: 8
  n_layers: 1
  n_heads: 2
  d_ff: 16
  dropout: 0.0
head:
  hidden_dim: null
  dropout: 0.0
  activation: null
  keep_last_dim: false
data:
  processed_downstream_root: {tmp_path}
  target_key: rul_super_secret
  batch_size: 2
  batch_size_policy: adaptive_by_channels
  max_channel_batch: 512
training:
  max_epochs: 1
  lr_head: 0.001
  lr_backbone: null
  weight_decay: 0.01
  amp: false
  grad_clip_norm: 1.0
  log_every: 1
  eval_every_epochs: 1
  metric_for_best: rmse_val
  lower_is_better: true
paths:
  log_dir: {log_dir}
  checkpoint_dir: {ckpt_dir}
"""
    )
    with pytest.raises(ValueError, match="target_key invalido"):
        main([
            "--config", str(cfg_path),
            "--mode", "from_scratch",
        ])
