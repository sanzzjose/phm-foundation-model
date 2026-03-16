"""Tests del parser TAR robusto en `training.phm_webdataset`.

El `__key__` de cada sample puede contener puntos (e.g. `unit.001_w000001`).
La version inicial usaba `member.name.split('.', 1)[0]`, lo que rompia esos
nombres. Aqui validamos que el nuevo parser por stripping de sufijos
conocidos:

- Recupera correctamente `__key__` aunque contenga puntos.
- Agrupa los miembros del sample bajo la misma clave.
- Detecta y reporta miembros con sufijo desconocido.
- Falla con `RuntimeError` si faltan claves obligatorias en strict=True.

Estos tests construyen `.tar` sinteticos pequenos en `tmp_path`, sin
depender de Drive ni de shards reales.
"""

from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import numpy as np
import pytest

# Importamos del lector puro (sin torch) para que estos tests no requieran
# torch en CI/local cuando solo se quiera ejercitar el parser de claves.
from training.phm_tar_reader import iter_samples_from_tar


# ----------------------------------------------------------------------
# Helpers para construir tars sinteticos
# ----------------------------------------------------------------------


def _np_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _add(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _full_sample(key: str) -> dict:
    """Devuelve dict {member_name: bytes} para un sample completo."""
    C, N, P, W = 2, 4, 8, 32
    return {
        f"{key}.patches.npy": _np_bytes(np.zeros((C, N, P), dtype=np.float32)),
        f"{key}.valid_time_mask.npy": _np_bytes(np.ones(W, dtype=bool)),
        f"{key}.valid_patch_mask.npy": _np_bytes(np.ones((C, N), dtype=bool)),
        f"{key}.mean.npy": _np_bytes(np.zeros(C, dtype=np.float32)),
        f"{key}.std_used.npy": _np_bytes(np.ones(C, dtype=np.float32)),
        f"{key}.canales_constantes_mask.npy": _np_bytes(np.zeros(C, dtype=bool)),
        f"{key}.target.json": json.dumps({"target_window": 1.0}).encode("utf-8"),
        f"{key}.meta.json": json.dumps({
            "dataset": "FAKE", "role": "PRETRAIN_SOURCE",
            "client": "misc", "patch_size": P, "window_size": W,
        }).encode("utf-8"),
    }


def _write_tar(tar_path: Path, entries: dict) -> None:
    with tarfile.open(tar_path, "w") as tar:
        for name, data in entries.items():
            _add(tar, name, data)


# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------


def test_key_with_dots_preserved(tmp_path):
    """Clave con puntos: 'unit.001_w000001' debe quedar intacta."""
    key = "unit.001_w000001"
    tar_path = tmp_path / "ds-train-000000.tar"
    _write_tar(tar_path, _full_sample(key))

    samples = list(iter_samples_from_tar(tar_path, strict=True))
    assert len(samples) == 1
    s = samples[0]
    assert s["__key__"] == key, f"clave esperada={key!r}, obtenida={s['__key__']!r}"
    # Todas las claves obligatorias presentes
    for k in ("patches", "valid_time_mask", "valid_patch_mask", "meta"):
        assert k in s, f"falta {k}"
    assert s["meta"]["dataset"] == "FAKE"
    assert s["target"]["target_window"] == 1.0


def test_multiple_samples_with_dotted_keys(tmp_path):
    """Varios samples con puntos en la clave no se confunden entre si."""
    keys = ["a.b.c_w000001", "x.y_w000002", "plain_key_w000003"]
    entries = {}
    for k in keys:
        entries.update(_full_sample(k))
    tar_path = tmp_path / "ds-train-000001.tar"
    _write_tar(tar_path, entries)

    samples = list(iter_samples_from_tar(tar_path, strict=True))
    assert sorted(s["__key__"] for s in samples) == sorted(keys)


def test_unknown_member_reported_not_raised(tmp_path):
    """Un sufijo desconocido se ignora silenciosamente o se reporta como
    __unknown_members__, pero NO debe romper si las claves obligatorias estan."""
    key = "trajectory.x_w000001"
    entries = _full_sample(key)
    # Anadimos un miembro con sufijo no canónico
    entries[f"{key}.extra_diag.bin"] = b"\x00\x01\x02"
    tar_path = tmp_path / "ds-train-000002.tar"
    _write_tar(tar_path, entries)

    samples = list(iter_samples_from_tar(tar_path, strict=True))
    assert len(samples) == 1
    # El miembro desconocido no tiene por que aparecer en el sample
    # (se agrupa como miembro global desconocido); lo importante es que
    # no haya levantado excepcion.


def test_strict_raises_on_missing_required_key(tmp_path):
    """Si falta patches/valid_time_mask/valid_patch_mask/meta y strict=True,
    levanta RuntimeError."""
    key = "broken_w000001"
    entries = _full_sample(key)
    # Quitamos patches.npy del sample
    del entries[f"{key}.patches.npy"]
    tar_path = tmp_path / "ds-train-broken.tar"
    _write_tar(tar_path, entries)

    with pytest.raises(RuntimeError, match="patches"):
        list(iter_samples_from_tar(tar_path, strict=True))


def test_non_strict_tolerates_missing_keys(tmp_path):
    """Con strict=False, se entrega el sample tal cual y el caller decide."""
    key = "broken_w000002"
    entries = _full_sample(key)
    del entries[f"{key}.meta.json"]
    tar_path = tmp_path / "ds-train-broken2.tar"
    _write_tar(tar_path, entries)

    samples = list(iter_samples_from_tar(tar_path, strict=False))
    assert len(samples) == 1
    assert "meta" not in samples[0]


def test_split_key_helper_directly():
    """Test directo del helper de stripping."""
    from training.phm_tar_reader import _split_key_and_field

    assert _split_key_and_field("unit.001_w000001.patches.npy") == (
        "unit.001_w000001", "patches.npy"
    )
    assert _split_key_and_field("CMAPSS__subset-1_w000005.meta.json") == (
        "CMAPSS__subset-1_w000005", "meta.json"
    )
    assert _split_key_and_field("k.valid_time_mask.npy") == (
        "k", "valid_time_mask.npy"
    )
    # Sufijo desconocido
    assert _split_key_and_field("x.extra.bin") is None
