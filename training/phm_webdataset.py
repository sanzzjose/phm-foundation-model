"""DataLoader (torch) sobre los shards harmonizados v0.5.

Este modulo construye los iteradores torch encima del lector puro de
`training/phm_tar_reader.py` (numpy + tarfile). La separacion permite que
el parser TAR sea testeable sin torch.

Politica MVP (sec 9 de `CLAUDE.md`):

- Un batch contiene muestras de un solo dataset (`batching_policy='por_dataset'`).
- En centralizado se rotan datasets entre batches con un sampler (weighted,
  round_robin o uniform); en smoke se prefiere round_robin para garantizar
  cobertura uniforme.
- Solo se leen datasets con `role == 'PRETRAIN_SOURCE'` para SSL pretraining.
- Por defecto se usa el split `train`.

API publica:

- `iter_samples_from_tar`, `inspect_first_shard`, `find_shards`,
  `_split_key_and_field`, `_NPY_KEYS`, `_KNOWN_SUFFIXES`, `_REQUIRED_KEYS`:
  re-exportados desde `phm_tar_reader` para no romper imports existentes.
- `iter_dataset_batches`: itera batches de un solo dataset (torch).
- `phm_collate`: collate custom que apila tensores y conserva meta/target
  como listas.
- `build_centralized_loader(plan, root, ..., strategy='weighted')`:
  itera batches segun una estrategia de rotacion entre datasets.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence

import numpy as np
import torch

from training.phm_tar_reader import (
    _KNOWN_SUFFIXES,
    _NPY_KEYS,
    _REQUIRED_KEYS,
    _split_key_and_field,
    find_shards,
    inspect_first_shard,
    iter_samples_from_tar,
)


# Re-exportamos compute_adaptive_batch_size desde training.sampling, donde
# vive sin dependencia de torch (asi los tests pueden ejercitarlo en CI sin
# instalar torch).
from training.sampling import compute_adaptive_batch_size  # noqa: E402

__all__ = [
    "iter_samples_from_tar",
    "inspect_first_shard",
    "find_shards",
    "_split_key_and_field",
    "_NPY_KEYS",
    "_KNOWN_SUFFIXES",
    "_REQUIRED_KEYS",
    "iter_dataset_batches",
    "build_centralized_loader",
    "phm_collate",
    "compute_adaptive_batch_size",
]


# ----------------------------------------------------------------------
# IterableDataset por dataset
# ----------------------------------------------------------------------


@dataclass
class _DatasetSpec:
    name: str
    role: Optional[str]
    client: Optional[str]
    shards: Sequence[Path]
    split: str


class _SingleDatasetIterable(torch.utils.data.IterableDataset):
    """Itera samples *de un solo dataset* desde sus shards."""

    def __init__(
        self,
        spec: _DatasetSpec,
        shuffle: bool = True,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.spec = spec
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        shards = list(self.spec.shards)
        if self.shuffle:
            rng.shuffle(shards)
        for shard_path in shards:
            for s in iter_samples_from_tar(shard_path, strict=True):
                yield _sample_to_tensors(s)


def _sample_to_tensors(s: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte arrays numpy a tensores torch."""
    out = {
        "patches": torch.from_numpy(s["patches"]).contiguous(),
        "valid_time_mask": torch.from_numpy(s["valid_time_mask"]).contiguous(),
        "valid_patch_mask": torch.from_numpy(s["valid_patch_mask"]).contiguous(),
        "mean": torch.from_numpy(s["mean"]).contiguous(),
        "std_used": torch.from_numpy(s["std_used"]).contiguous(),
        "canales_constantes_mask": torch.from_numpy(
            s["canales_constantes_mask"]
        ).contiguous(),
        "target": s.get("target", {}),
        "meta": s.get("meta", {}),
        "__key__": s.get("__key__"),
    }
    return out


def phm_collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate custom: apila tensores y mantiene `meta`/`target` como listas.

    Verifica que el batch sea homogeneo (mismo C, N, P, W). Es responsabilidad
    del sampler garantizar batches por dataset.
    """
    if not batch:
        raise ValueError("phm_collate recibio batch vacio")
    p0 = batch[0]["patches"].shape
    for i, item in enumerate(batch[1:], 1):
        if item["patches"].shape != p0:
            raise ValueError(
                f"batch heterogeneo: sample[0].patches={p0} vs "
                f"sample[{i}].patches={item['patches'].shape}. El sampler "
                "debe garantizar batches por dataset."
            )

    out = {
        "patches": torch.stack([b["patches"] for b in batch], dim=0),
        "valid_time_mask": torch.stack(
            [b["valid_time_mask"] for b in batch], dim=0
        ),
        "valid_patch_mask": torch.stack(
            [b["valid_patch_mask"] for b in batch], dim=0
        ),
        "mean": torch.stack([b["mean"] for b in batch], dim=0),
        "std_used": torch.stack([b["std_used"] for b in batch], dim=0),
        "canales_constantes_mask": torch.stack(
            [b["canales_constantes_mask"] for b in batch], dim=0
        ),
        "targets": [b["target"] for b in batch],
        "metas": [b["meta"] for b in batch],
        "keys": [b["__key__"] for b in batch],
    }

    B, C, N, P = out["patches"].shape
    W = N * P
    assert out["patches"].dtype == torch.float32, f"patches dtype {out['patches'].dtype}"
    assert out["valid_time_mask"].shape == (B, W), out["valid_time_mask"].shape
    assert out["valid_time_mask"].dtype == torch.bool
    assert out["valid_patch_mask"].shape == (B, C, N), out["valid_patch_mask"].shape
    assert out["valid_patch_mask"].dtype == torch.bool
    assert "target" not in out, "target no debe colatearse como tensor"
    return out


def iter_dataset_batches(
    dataset_name: str,
    processed_root: Path,
    split: str = "train",
    batch_size: int = 8,
    shuffle: bool = True,
    seed: int = 0,
    role: Optional[str] = None,
    client: Optional[str] = None,
    drop_last: bool = True,
) -> Iterator[Dict[str, Any]]:
    """Itera batches de un solo dataset."""
    shards = find_shards(processed_root, dataset_name, split)
    if not shards:
        raise FileNotFoundError(
            f"No shards encontrados: {processed_root}/{dataset_name}/{split}/*.tar"
        )
    spec = _DatasetSpec(
        name=dataset_name, role=role, client=client, shards=shards, split=split
    )
    iterable = _SingleDatasetIterable(spec, shuffle=shuffle, seed=seed)
    buf: List[Dict[str, Any]] = []
    for sample in iterable:
        buf.append(sample)
        if len(buf) == batch_size:
            yield phm_collate(buf)
            buf = []
    if buf and not drop_last:
        yield phm_collate(buf)


def build_centralized_loader(
    plan: Sequence[Dict[str, Any]],
    processed_root: Path,
    batch_size: int = 8,
    split: str = "train",
    seed: int = 0,
    max_steps: Optional[int] = None,
    strategy: str = "weighted",
    batch_size_policy: str = "fixed",
    max_channel_batch: Optional[int] = None,
    min_batch_size: int = 1,
) -> Iterator[Dict[str, Any]]:
    """Iterador centralizado que rota datasets segun una estrategia.

    Estrategias de muestreo soportadas:

    - **weighted** (default productivo): proporcional a `final_dataset_weight`.
    - **round_robin**: alterna los datasets en orden fijo, ciclico, sin pesos.
    - **uniform**: muestreo uniforme entre los datasets del plan.

    Politicas de batch_size:

    - **fixed** (default): todos los datasets usan `batch_size`.
    - **adaptive_by_channels**: cada dataset usa
      `compute_adaptive_batch_size(n_channels, batch_size, max_channel_batch,
      min_batch_size)`. Asegura que B*C no supere `max_channel_batch` salvo
      cuando ya esta en `min_batch_size`. Requiere que cada `plan[i]` tenga
      una columna `n_channels`.

    Cada batch yielded incluye los metadatos:
        __dataset__, __client__, __n_channels__, __batch_size_effective__,
        __effective_bc__ = batch_size_effective * n_channels.
    """
    if strategy not in ("weighted", "round_robin", "uniform"):
        raise ValueError(
            f"strategy desconocida: {strategy!r}. "
            "Acepta: 'weighted', 'round_robin', 'uniform'."
        )
    if batch_size_policy not in ("fixed", "adaptive_by_channels"):
        raise ValueError(
            f"batch_size_policy desconocida: {batch_size_policy!r}. "
            "Acepta: 'fixed', 'adaptive_by_channels'."
        )
    rng = np.random.default_rng(seed)
    datasets = [str(row["dataset"]) for row in plan]
    n_ds = len(datasets)
    if n_ds == 0:
        raise ValueError("Plan vacio")

    if strategy == "weighted":
        weights = np.array(
            [float(row["final_dataset_weight"]) for row in plan], dtype=np.float64
        )
        if weights.sum() <= 0:
            raise ValueError("Sampling plan sin pesos positivos")
        weights = weights / weights.sum()
    else:
        weights = None

    # Pre-calculamos el batch_size efectivo por dataset segun la politica.
    bs_eff_by_ds: Dict[str, int] = {}
    nc_by_ds: Dict[str, int] = {}
    for row in plan:
        ds = str(row["dataset"])
        nc = int(row.get("n_channels", 0) or 0)
        nc_by_ds[ds] = nc
        if batch_size_policy == "adaptive_by_channels":
            if nc <= 0:
                raise ValueError(
                    f"adaptive_by_channels requiere n_channels > 0 en el plan "
                    f"para todos los datasets; {ds} tiene {nc}."
                )
            bs_eff_by_ds[ds] = compute_adaptive_batch_size(
                n_channels=nc,
                batch_size=batch_size,
                max_channel_batch=max_channel_batch,
                min_batch_size=min_batch_size,
            )
        else:
            bs_eff_by_ds[ds] = int(batch_size)

    open_iters: Dict[str, Iterator[Dict[str, Any]]] = {}

    def _open(name: str) -> Iterator[Dict[str, Any]]:
        return iter_dataset_batches(
            dataset_name=name,
            processed_root=processed_root,
            split=split,
            batch_size=bs_eff_by_ds[name],
            shuffle=True,
            seed=int(rng.integers(0, 2**31 - 1)),
        )

    step = 0
    rr_idx = 0
    while True:
        if max_steps is not None and step >= max_steps:
            return
        if strategy == "weighted":
            idx = int(rng.choice(n_ds, p=weights))
        elif strategy == "uniform":
            idx = int(rng.integers(0, n_ds))
        else:
            idx = rr_idx % n_ds
            rr_idx += 1
        name = datasets[idx]
        if name not in open_iters:
            open_iters[name] = _open(name)
        try:
            batch = next(open_iters[name])
        except StopIteration:
            open_iters[name] = _open(name)
            try:
                batch = next(open_iters[name])
            except StopIteration:
                continue
        batch["__dataset__"] = name
        if "client" in plan[idx]:
            batch["__client__"] = plan[idx]["client"]
        # Metadatos del batch para logging y validacion VRAM
        b_eff_actual = int(batch["patches"].shape[0])
        c_actual = int(batch["patches"].shape[1])
        batch["__n_channels__"] = c_actual
        batch["__batch_size_effective__"] = b_eff_actual
        batch["__effective_bc__"] = b_eff_actual * c_actual
        step += 1
        yield batch
