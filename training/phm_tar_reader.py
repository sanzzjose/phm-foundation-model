"""Lector puro (numpy + tarfile) para shards en formato WebDataset.

Modulo independiente de `torch`. Lo usan:

- `training.phm_webdataset`, que envuelve el iterador como `torch.utils.data`.
- Tests unitarios que validan el parser de claves sin requerir torch.
- Scripts ad-hoc de inspeccion en entornos sin GPU.

Contrato de los shards: ver `training/phm_webdataset.py`.

IMPORTANTE: el `__key__` de cada sample puede contener puntos (e.g.
`unit.001_w000001`). Por eso NO se usa `split('.', 1)[0]`. La separacion
clave/campo se hace con stripping de sufijos conocidos.
"""

from __future__ import annotations

import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np


_NPY_KEYS = (
    "patches",
    "valid_time_mask",
    "valid_patch_mask",
    "mean",
    "std_used",
    "canales_constantes_mask",
    # Targets escalares del builder CMAPSS RUL (commits 3/fd0cec9). Los
    # parseamos como np.ndarray escalares; el trainer downstream RUL los
    # lee normalmente desde `sample["meta"][target_rul_*]` (mas robusto
    # ante cambios de naming), pero permitir leer el .npy escalar tambien
    # es util para diagnostico/inspeccion.
    "rul_physical",
    "rul_capped_125",
)

# Sufijos esperados, con el punto inicial. Orden: del mas largo al mas
# corto para que ".valid_time_mask.npy" se detecte antes que ".npy".
_KNOWN_SUFFIXES: Tuple[str, ...] = (
    ".patches.npy",
    ".valid_time_mask.npy",
    ".valid_patch_mask.npy",
    ".canales_constantes_mask.npy",
    ".std_used.npy",
    ".mean.npy",
    ".rul_physical.npy",
    ".rul_capped_125.npy",
    ".target.json",
    ".meta.json",
)

# Claves obligatorias por sample (sec 9 + sec 12 de CLAUDE.md)
_REQUIRED_KEYS = ("patches", "valid_time_mask", "valid_patch_mask", "meta")


def _split_key_and_field(name: str) -> Optional[Tuple[str, str]]:
    """Separa un nombre de miembro en (`__key__`, `field`).

    Probamos los sufijos conocidos de mayor a menor longitud. Si ninguno
    casa, devolvemos None y el caller decide.
    """
    for sfx in _KNOWN_SUFFIXES:
        if name.endswith(sfx):
            key = name[: -len(sfx)]
            field = sfx[1:]  # sin el punto inicial
            return key, field
    return None


def find_shards(
    processed_root: Path, dataset: str, split: str = "train"
) -> List[Path]:
    """Localiza los `.tar` de un dataset/split en el corpus harmonizado.

    Convencion del full v0.5:
        <processed_root>/<dataset>/<split>/<dataset>-<split>-NNNNNN.tar
    """
    processed_root = Path(processed_root)
    split_dir = processed_root / dataset / split
    if not split_dir.is_dir():
        return []
    pattern = f"{dataset}-{split}-*.tar"
    return sorted(split_dir.glob(pattern))


def iter_samples_from_tar(
    tar_path: Path,
    strict: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Itera samples de un `.tar` WDS leyendo con `tarfile` puro.

    Args:
        tar_path: ruta al `.tar`.
        strict: si True (default) y un sample no contiene las claves
            obligatorias (`patches`, `valid_time_mask`, `valid_patch_mask`,
            `meta`), levanta `RuntimeError`. Si False, yield el sample tal
            cual.

    Yields:
        Dicts con keys:
            - patches, valid_time_mask, valid_patch_mask, mean, std_used,
              canales_constantes_mask (np.ndarray, los disponibles)
            - target (dict)
            - meta (dict)
            - __key__ (str)
            - __unknown_members__ (list[str], opcional)
    """
    tar_path = Path(tar_path)
    with tarfile.open(tar_path, "r") as tar:
        members_by_key: Dict[str, List[Tuple[str, tarfile.TarInfo]]] = defaultdict(list)
        unknown_global: List[str] = []
        for member in tar.getmembers():
            if not member.isfile():
                continue
            parsed = _split_key_and_field(member.name)
            if parsed is None:
                unknown_global.append(member.name)
                continue
            key, field = parsed
            members_by_key[key].append((field, member))

        for key in sorted(members_by_key.keys()):
            sample: Dict[str, Any] = {"__key__": key}
            unknown_local: List[str] = []
            for field, member in members_by_key[key]:
                f = tar.extractfile(member)
                if f is None:
                    continue
                data = f.read()
                if field.endswith(".npy"):
                    arr_key = field[: -len(".npy")]
                    if arr_key in _NPY_KEYS:
                        sample[arr_key] = np.load(
                            io.BytesIO(data), allow_pickle=False
                        )
                    else:
                        unknown_local.append(field)
                elif field == "target.json":
                    sample["target"] = json.loads(data.decode("utf-8"))
                elif field == "meta.json":
                    sample["meta"] = json.loads(data.decode("utf-8"))
                else:
                    unknown_local.append(field)
            if unknown_local:
                sample["__unknown_members__"] = unknown_local

            if strict:
                missing = [k for k in _REQUIRED_KEYS if k not in sample]
                if missing:
                    raise RuntimeError(
                        f"Sample '{key}' en {tar_path.name} le falta keys "
                        f"obligatorias: {missing}. Otros miembros desconocidos "
                        f"en el tar (no asignados a sample): {unknown_global[:10]}"
                    )
            yield sample


def inspect_first_shard(
    dataset_name: str,
    processed_root: Path,
    split: str = "train",
    n_samples: int = 1,
) -> Dict[str, Any]:
    """Inspecciona el primer shard de un dataset sin cargar el corpus entero.

    Devuelve dict con: shard_path, n_members_total, members_first_sample,
    shapes_dtypes, meta_summary, has_partial_patch, samples_inspected,
    n_samples_with_partial_patch.
    """
    processed_root = Path(processed_root)
    shards = find_shards(processed_root, dataset_name, split)
    if not shards:
        raise FileNotFoundError(
            f"No hay shards para {dataset_name}/{split} en {processed_root}"
        )
    shard_path = shards[0]

    out: Dict[str, Any] = {
        "shard_path": str(shard_path),
        "dataset": dataset_name,
        "split": split,
    }

    with tarfile.open(shard_path, "r") as tar:
        all_members = [m.name for m in tar.getmembers() if m.isfile()]
    out["n_members_total"] = len(all_members)

    shapes_dtypes: Dict[str, Dict[str, Any]] = {}
    meta_summary = None
    has_partial = False
    n_with_partial = 0
    members_first: Optional[List[str]] = None
    read = 0
    for sample in iter_samples_from_tar(shard_path, strict=True):
        if members_first is None:
            members_first = [
                f"{sample['__key__']}.{k}"
                for k in ("patches.npy", "valid_time_mask.npy",
                          "valid_patch_mask.npy", "mean.npy", "std_used.npy",
                          "canales_constantes_mask.npy", "target.json",
                          "meta.json")
            ]
        for k in _NPY_KEYS:
            if k in sample:
                arr = sample[k]
                shapes_dtypes[k] = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
        m = sample.get("meta", {})
        if meta_summary is None:
            meta_summary = {
                "dataset": m.get("dataset"),
                "role": m.get("role"),
                "client": m.get("client"),
                "dominio": m.get("dominio"),
                "idx_window": m.get("idx_window"),
                "trajectory_id": m.get("trajectory_id"),
                "window_size": m.get("window_size"),
                "patch_size": m.get("patch_size"),
            }
        vtm = sample.get("valid_time_mask")
        if vtm is not None:
            W = int(vtm.shape[0])
            p_size = int(m.get("patch_size", 16))
            n_patches_local = W // p_size
            vsm = vtm.reshape(n_patches_local, p_size)
            partial = vsm.any(axis=-1) & ~vsm.all(axis=-1)
            if bool(partial.any()):
                n_with_partial += 1
                if not has_partial:
                    has_partial = True
        read += 1
        if read >= n_samples:
            break

    out["members_first_sample"] = members_first or []
    out["shapes_dtypes"] = shapes_dtypes
    out["meta_summary"] = meta_summary or {}
    out["has_partial_patch"] = has_partial
    out["samples_inspected"] = read
    out["n_samples_with_partial_patch"] = n_with_partial
    return out
