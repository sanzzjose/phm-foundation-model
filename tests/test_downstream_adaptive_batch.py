"""Tests del batch adaptativo por canales en downstream classification.

Cubre:

- compute_adaptive_batch_size (re-exportado desde training.sampling):
  casos C=1, 2, 22, 48 con cap=512 y verificacion de min_batch_size.
- resolve_downstream_batch_size:
  - politica fixed devuelve batch_size sin tocar.
  - politica adaptive_by_channels usa el cap.
  - inferencia de n_channels desde manifest > primer sample > fallback.
- helper de inferencia _infer_n_channels: manifest existe / no existe.

No requiere Drive ni torch (las funciones son numpy + json puros).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _import_sampling():
    # Import diferido: torch puede no estar disponible en algunos entornos.
    from training.sampling import compute_adaptive_batch_size
    return compute_adaptive_batch_size


# ----------------------------------------------------------------------
# compute_adaptive_batch_size: casos clave
# ----------------------------------------------------------------------


def test_adaptive_batch_c1_no_cambia():
    """C=1, batch=64, cap=512 -> 64 (cap nunca afecta a C=1 con cap>=64)."""
    f = _import_sampling()
    assert f(n_channels=1, batch_size=64, max_channel_batch=512) == 64


def test_adaptive_batch_c2_no_cambia():
    """C=2, batch=64, cap=512 -> 64 (B*C=128 <= 512)."""
    f = _import_sampling()
    assert f(n_channels=2, batch_size=64, max_channel_batch=512) == 64


def test_adaptive_batch_c22_capeado():
    """C=22, batch=64, cap=512 -> floor(512/22)=23 (cap activo)."""
    f = _import_sampling()
    assert f(n_channels=22, batch_size=64, max_channel_batch=512) == 23


def test_adaptive_batch_c48_capeado():
    """C=48, batch=64, cap=512 -> floor(512/48)=10 (cap activo)."""
    f = _import_sampling()
    assert f(n_channels=48, batch_size=64, max_channel_batch=512) == 10


def test_adaptive_batch_min_batch_respetado():
    """Si C es enorme, el cap // C puede ser 0; min_batch_size=1 garantiza >=1."""
    f = _import_sampling()
    # C=317 (PHM14), cap=512 -> floor(512/317)=1
    assert f(n_channels=317, batch_size=64, max_channel_batch=512) == 1
    # C=600 con cap=512: 512//600=0, debe subir a min_batch_size=1
    assert f(n_channels=600, batch_size=64, max_channel_batch=512) == 1


def test_adaptive_batch_sin_cap_devuelve_batch():
    """Si max_channel_batch es None o 0, la politica es fixed (devuelve batch)."""
    f = _import_sampling()
    assert f(n_channels=22, batch_size=64, max_channel_batch=None) == 64
    assert f(n_channels=22, batch_size=64, max_channel_batch=0) == 64


def test_adaptive_batch_rechaza_inputs_invalidos():
    f = _import_sampling()
    with pytest.raises(ValueError):
        f(n_channels=0, batch_size=64, max_channel_batch=512)
    with pytest.raises(ValueError):
        f(n_channels=-1, batch_size=64, max_channel_batch=512)
    with pytest.raises(ValueError):
        f(n_channels=2, batch_size=0, max_channel_batch=512)
    with pytest.raises(ValueError):
        f(n_channels=2, batch_size=64, max_channel_batch=512, min_batch_size=0)


# ----------------------------------------------------------------------
# resolve_downstream_batch_size: manifest, fallback, politicas
# ----------------------------------------------------------------------


@pytest.fixture
def fake_processed_root(tmp_path: Path) -> Path:
    """Crea estructura processed/<DATASET>/manifest.json sin shards reales."""
    root = tmp_path / "processed"
    root.mkdir()
    return root


def _write_manifest(root: Path, ds: str, manifest: dict) -> Path:
    p = root / ds
    p.mkdir(parents=True, exist_ok=True)
    (p / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    return p


def test_resolve_fixed_devuelve_batch_size(fake_processed_root):
    from training.train_downstream_classification import resolve_downstream_batch_size
    _write_manifest(fake_processed_root, "FAKE", {"n_channels": 22})
    info = resolve_downstream_batch_size(
        fake_processed_root, "FAKE",
        {"batch_size": 64, "batch_size_policy": "fixed"},
    )
    assert info["batch_size_policy"] == "fixed"
    assert info["batch_size_effective"] == 64
    assert info["n_channels"] == 22
    assert info["n_channels_source"] == "manifest"
    assert info["effective_bc"] == 64 * 22
    assert info["warnings"] == []


def test_resolve_adaptive_c22_capeado(fake_processed_root):
    from training.train_downstream_classification import resolve_downstream_batch_size
    _write_manifest(fake_processed_root, "FAKE", {"n_channels": 22})
    info = resolve_downstream_batch_size(
        fake_processed_root, "FAKE",
        {
            "batch_size": 64,
            "batch_size_policy": "adaptive_by_channels",
            "max_channel_batch": 512,
            "min_batch_size": 1,
        },
    )
    assert info["batch_size_policy"] == "adaptive_by_channels"
    assert info["batch_size_effective"] == 23
    assert info["effective_bc"] == 23 * 22
    assert info["n_channels"] == 22


def test_resolve_adaptive_c2_no_cambia(fake_processed_root):
    from training.train_downstream_classification import resolve_downstream_batch_size
    _write_manifest(fake_processed_root, "CWRU", {"n_channels": 2})
    info = resolve_downstream_batch_size(
        fake_processed_root, "CWRU",
        {
            "batch_size": 64,
            "batch_size_policy": "adaptive_by_channels",
            "max_channel_batch": 512,
            "min_batch_size": 1,
        },
    )
    assert info["batch_size_effective"] == 64
    assert info["effective_bc"] == 128


def test_resolve_manifest_ausente_usa_fallback(fake_processed_root):
    """Sin manifest y sin shards, usa n_channels_fallback con warning."""
    from training.train_downstream_classification import resolve_downstream_batch_size
    # NO escribimos manifest ni shards.
    (fake_processed_root / "GHOST").mkdir()
    info = resolve_downstream_batch_size(
        fake_processed_root, "GHOST",
        {
            "batch_size": 32,
            "batch_size_policy": "adaptive_by_channels",
            "max_channel_batch": 512,
            "min_batch_size": 1,
            "n_channels_fallback": 5,
        },
    )
    assert info["n_channels"] == 5
    assert info["n_channels_source"] == "fallback_config"
    assert info["batch_size_effective"] == 32
    assert any("fallback" in w.lower() for w in info["warnings"])


def test_resolve_sin_nada_usa_default_2_con_warning(fake_processed_root):
    """Sin manifest, sin shards, sin fallback -> n_channels=2 con warning fuerte."""
    from training.train_downstream_classification import resolve_downstream_batch_size
    (fake_processed_root / "GHOST").mkdir()
    info = resolve_downstream_batch_size(
        fake_processed_root, "GHOST",
        {"batch_size": 64, "batch_size_policy": "fixed"},
    )
    assert info["n_channels"] == 2
    assert info["n_channels_source"] == "fallback_default"
    assert info["batch_size_effective"] == 64
    # Debe haber warning indicando que es solo dry-run local
    assert any("dry-run" in w.lower() or "default" in w.lower() for w in info["warnings"])


def test_resolve_politica_desconocida_falla(fake_processed_root):
    from training.train_downstream_classification import resolve_downstream_batch_size
    _write_manifest(fake_processed_root, "FAKE", {"n_channels": 2})
    with pytest.raises(ValueError, match="batch_size_policy"):
        resolve_downstream_batch_size(
            fake_processed_root, "FAKE",
            {"batch_size": 64, "batch_size_policy": "wat"},
        )


# ---------------------------------------------------------------------------
# resolve_lr_backbone: tolerancia a lr_backbone null (regresion del probe)
# ---------------------------------------------------------------------------

def test_resolve_lr_backbone_none_when_null():
    """lr_backbone: null -> None (no TypeError). Regresion del Probe Suite."""
    from training.train_downstream_classification import resolve_lr_backbone
    assert resolve_lr_backbone({"lr_backbone": None}) is None


def test_resolve_lr_backbone_none_when_missing():
    from training.train_downstream_classification import resolve_lr_backbone
    assert resolve_lr_backbone({}) is None


def test_resolve_lr_backbone_zero_is_none():
    from training.train_downstream_classification import resolve_lr_backbone
    assert resolve_lr_backbone({"lr_backbone": 0.0}) is None


def test_resolve_lr_backbone_positive_passthrough():
    from training.train_downstream_classification import resolve_lr_backbone
    assert resolve_lr_backbone({"lr_backbone": 1e-5}) == 1e-5
