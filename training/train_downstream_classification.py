"""Trainer downstream de clasificacion sobre TRANSFER_TARGETs limpios.

Implementacion para el primer downstream (CWRU classification_multiclass)
con tres modos mutuamente exclusivos:

    --mode from_scratch        backbone random + cabeza random
    --mode linear_probing      encoder SSL congelado + cabeza entrenable
    --mode full_finetuning     encoder SSL + cabeza, ambos entrenables

Para los modos basados en SSL hay que pasar `--checkpoint <ruta>`
apuntando al ckpt del SSL central full (`ckpt_step100000.pt`).

Lee shards de Drive en `processed/<DATASET>/{train,val,test}/`. Usa
`phm_tar_reader.iter_samples_from_tar` (puro numpy + tarfile, sin
asumir orden de columnas en `target.json`).

Trazabilidad por run en `paths.log_dir/<run_name>/`:

    - `config.yaml`            : copia del YAML cargado.
    - `run_info.json`          : git_hash, config_hash, mode, n_classes,
                                  label_mapping, mejor metrica, etc.
    - `metrics.jsonl`          : JSON estricto (sin NaN/Infinity), una
                                  linea por batch de train y por evaluacion.
    - `label_mapping.json`     : dict {label_str: class_id} aprendido
                                  desde train.

Checkpoint final en `paths.checkpoint_dir/<run_name>/best.pt` con
`model_state_dict`, `optimizer_state_dict`, `config`, `epoch`, `step`,
`metric_for_best`, `best_value`, `label_mapping`, `git_hash`.

Modo `--dry-run` no entrena: lee manifest, cuenta shards, inspecciona
las primeras N muestras, construye modelo + cabeza, hace un forward
sintetico (o con un batch real si Drive existe), reporta `n_classes` y
`label_mapping`, y sale.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# JsonlLogger + _json_safe se reutilizan del trainer SSL central
from training.train_ssl_central import (
    JsonlLogger,
    _json_safe,
    config_hash,
    get_git_info,
    load_config,
)
from training.downstream.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall_f1,
    support_per_class,
)
from training.phm_tar_reader import find_shards, iter_samples_from_tar
from training.sampling import compute_adaptive_batch_size


# ----------------------------------------------------------------------
# Helpers comunes
# ----------------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def resolve_lr_backbone(training_cfg: Dict[str, Any]) -> Optional[float]:
    """Resuelve el LR del backbone tolerando ``lr_backbone: null``.

    Reglas (mismas que el trainer RUL, para consistencia):

    * clave ausente o ``None`` -> ``None`` (backbone sin grupo de
      parametros, p.ej. ``linear_probing`` con encoder congelado);
    * numerico ``<= 0`` -> ``None`` (equivalente a congelado);
    * numerico ``> 0`` -> ese LR.

    Antes, ``float(training_cfg.get("lr_backbone", 0.0))`` reventaba con
    ``TypeError`` cuando el YAML traia ``lr_backbone: null`` explicito
    (el default 0.0 solo aplica si la CLAVE falta, no si su valor es
    None). Esto rompia el Probe Suite en modo ``linear_probing``.
    """
    raw = training_cfg.get("lr_backbone")
    if raw is None:
        return None
    val = float(raw)
    return val if val > 0 else None


def _safe_json_dump(obj: Any, path: Path) -> None:
    """Escritura JSON estricta (allow_nan=False)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(obj), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _normalize_label(v: Any) -> str:
    """Convierte el target a una etiqueta string estable.

    El target en `target.json` puede venir como int, float (si la
    harmonization fuerza `float(target_win)`), string, etc. Devolvemos
    siempre una string canonica.
    """
    if isinstance(v, bool):
        return str(int(v))
    if isinstance(v, float):
        # Si es entero, conservamos forma "1" no "1.0"
        if math.isfinite(v) and v.is_integer():
            return str(int(v))
        return str(v)
    if isinstance(v, int):
        return str(v)
    return str(v)


def _extract_label(sample: Dict[str, Any]) -> Optional[str]:
    """Extrae la etiqueta de un sample desde `target.json`.

    El contrato escrito por la harmonization v0.5 guarda `target_window`
    como valor escalar (politica `ultimo_valor_valido` o
    `etiqueta_unidad`). Para classification_multiclass este escalar es la
    clase.
    """
    tgt = sample.get("target", {})
    if not isinstance(tgt, dict):
        return None
    v = tgt.get("target_window")
    if v is None:
        v = tgt.get("target")
    if v is None:
        return None
    return _normalize_label(v)


# ----------------------------------------------------------------------
# Inferencia de n_channels y politica adaptativa de batch_size
# ----------------------------------------------------------------------


def _infer_n_channels(
    processed_root: Path, dataset: str, manifest_first: bool = True
) -> Tuple[Optional[int], str]:
    """Devuelve `(n_channels, fuente)` o `(None, "missing")` si no se puede inferir.

    Orden de busqueda:
      1. `processed_root/<dataset>/manifest.json` -> campo `n_channels`.
      2. Primera muestra del primer shard de train (lectura barata: el primer
         registro tiene `patches.shape == (C, N, P)`, leemos C).
    """
    if manifest_first:
        mpath = processed_root / dataset / "manifest.json"
        if mpath.is_file():
            try:
                m = json.loads(mpath.read_text(encoding="utf-8"))
                nc = m.get("n_channels", m.get("n_canales"))
                if nc is not None and int(nc) > 0:
                    return int(nc), "manifest"
            except Exception:
                pass

    # Segunda opcion: leer primer sample de train.
    shards = find_shards(processed_root, dataset, "train")
    if shards:
        try:
            for s in iter_samples_from_tar(shards[0], strict=True):
                arr = s.get("patches")
                if arr is not None and hasattr(arr, "shape") and len(arr.shape) == 3:
                    return int(arr.shape[0]), "first_sample_train"
                break  # solo necesitamos el primero
        except Exception:
            pass
    return None, "missing"


def resolve_downstream_batch_size(
    processed_root: Path, dataset: str, data_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Resuelve `batch_size_effective` para downstream segun la politica.

    Lectura de claves en `data_cfg`:
      - `batch_size` (default 64).
      - `batch_size_policy`: "fixed" | "adaptive_by_channels" (default "fixed"
        para compatibilidad historica con runs v0.1).
      - `max_channel_batch` (default 512).
      - `min_batch_size` (default 1).
      - `n_channels_fallback`: usado SOLO si no se puede inferir desde Drive
        (manifest+primera muestra fallan). Util en dry-run local sin Drive.

    Returns:
        dict con:
          - batch_size_requested
          - batch_size_effective
          - batch_size_policy
          - n_channels
          - n_channels_source: "manifest" | "first_sample_train" |
            "fallback_config" | "fallback_default"
          - max_channel_batch
          - min_batch_size
          - effective_bc = batch_size_effective * n_channels
          - warnings: list[str]
    """
    warnings: List[str] = []
    requested = int(data_cfg.get("batch_size", 64))
    policy = str(data_cfg.get("batch_size_policy", "fixed"))
    max_channel_batch = data_cfg.get("max_channel_batch", 512)
    min_batch_size = int(data_cfg.get("min_batch_size", 1))

    n_channels, source = _infer_n_channels(processed_root, dataset)
    if n_channels is None:
        fb = data_cfg.get("n_channels_fallback")
        if fb is not None and int(fb) > 0:
            n_channels = int(fb)
            source = "fallback_config"
            warnings.append(
                "n_channels no inferible desde Drive; usando "
                f"n_channels_fallback={n_channels} (solo dry-run/local)."
            )
        else:
            # Default conservador: 2 (CWRU). El dry-run lo permite con warning.
            n_channels = 2
            source = "fallback_default"
            warnings.append(
                "n_channels no inferible y sin n_channels_fallback; "
                "asumiendo n_channels=2 (solo dry-run local). NO usar en "
                "entrenamiento real."
            )

    if policy == "adaptive_by_channels":
        effective = compute_adaptive_batch_size(
            n_channels=n_channels,
            batch_size=requested,
            max_channel_batch=int(max_channel_batch) if max_channel_batch else None,
            min_batch_size=min_batch_size,
        )
    elif policy == "fixed":
        effective = requested
    else:
        raise ValueError(
            f"batch_size_policy desconocida: {policy!r}. "
            "Acepta: 'fixed', 'adaptive_by_channels'."
        )

    return {
        "batch_size_requested": requested,
        "batch_size_effective": int(effective),
        "batch_size_policy": policy,
        "n_channels": int(n_channels),
        "n_channels_source": source,
        "max_channel_batch": int(max_channel_batch) if max_channel_batch else None,
        "min_batch_size": min_batch_size,
        "effective_bc": int(effective) * int(n_channels),
        "warnings": warnings,
    }


# ----------------------------------------------------------------------
# Construir label_mapping desde train
# ----------------------------------------------------------------------


def build_label_mapping(
    processed_root: Path,
    dataset: str,
    split: str = "train",
    max_samples: Optional[int] = None,
) -> Tuple[Dict[str, int], Counter]:
    """Recorre los shards de un split y construye label -> class_id.

    Devuelve (mapping, counts). El mapping es estable (ordena las
    etiquetas alfabeticamente para que la asignacion no dependa del
    orden de lectura de shards).
    """
    shards = find_shards(processed_root, dataset, split)
    if not shards:
        raise FileNotFoundError(
            f"No hay shards en {processed_root}/{dataset}/{split}"
        )
    counts: Counter = Counter()
    n = 0
    for shard in shards:
        for sample in iter_samples_from_tar(shard, strict=True):
            label = _extract_label(sample)
            if label is None:
                continue
            counts[label] += 1
            n += 1
            if max_samples and n >= max_samples:
                break
        if max_samples and n >= max_samples:
            break

    if not counts:
        raise RuntimeError(
            f"Ningun sample en {dataset}/{split} tiene etiqueta legible."
        )

    labels_sorted = sorted(counts.keys())
    mapping = {lbl: i for i, lbl in enumerate(labels_sorted)}
    return mapping, counts


# ----------------------------------------------------------------------
# Dataset iter para downstream (similar a SSL pero devuelve etiqueta)
# ----------------------------------------------------------------------


def _sample_to_tensors_downstream(
    s: Dict[str, Any], label_mapping: Dict[str, int]
) -> Dict[str, Any]:
    """Convierte un sample numpy a tensores torch + clase int.

    Returns:
        dict con keys: patches, valid_time_mask, valid_patch_mask,
        canales_constantes_mask, y `class_id` (int) si el label es
        conocido; lanza KeyError si el label no esta en el mapping
        (caller decide si abortar o saltar).
    """
    import torch
    label = _extract_label(s)
    if label is None:
        raise ValueError(
            f"Sample {s.get('__key__')} sin etiqueta legible"
        )
    if label not in label_mapping:
        raise KeyError(
            f"Sample {s.get('__key__')} tiene clase desconocida {label!r} "
            f"(no estaba en train)."
        )
    return {
        "patches":                  torch.from_numpy(s["patches"]).contiguous(),
        "valid_time_mask":          torch.from_numpy(s["valid_time_mask"]).contiguous(),
        "valid_patch_mask":         torch.from_numpy(s["valid_patch_mask"]).contiguous(),
        "canales_constantes_mask":  torch.from_numpy(s["canales_constantes_mask"]).contiguous(),
        "class_id":                 int(label_mapping[label]),
        "__key__":                  s.get("__key__"),
    }


def iter_split_batches(
    processed_root: Path,
    dataset: str,
    split: str,
    batch_size: int,
    label_mapping: Dict[str, int],
    shuffle: bool = True,
    seed: int = 0,
    max_batches: Optional[int] = None,
):
    """Itera batches de un split de un dataset downstream."""
    import numpy as np
    import torch

    shards = find_shards(processed_root, dataset, split)
    if not shards:
        raise FileNotFoundError(f"No hay shards en {processed_root}/{dataset}/{split}")

    rng = np.random.default_rng(seed)
    if shuffle:
        shards = list(shards)
        rng.shuffle(shards)

    buf: List[Dict[str, Any]] = []
    n_yielded = 0
    for shard in shards:
        for s in iter_samples_from_tar(shard, strict=True):
            try:
                t = _sample_to_tensors_downstream(s, label_mapping)
            except KeyError:
                # Clase no vista en train: anomalia, propagar para asserts
                raise
            buf.append(t)
            if len(buf) == batch_size:
                yield _collate(buf)
                buf = []
                n_yielded += 1
                if max_batches and n_yielded >= max_batches:
                    return
    if buf:
        yield _collate(buf)


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apila el batch para downstream classification."""
    import torch
    p0 = batch[0]["patches"].shape
    for i, b in enumerate(batch[1:], 1):
        if b["patches"].shape != p0:
            raise ValueError(
                f"batch heterogeneo en patches: ref {p0} vs sample {i} {b['patches'].shape}"
            )
    out = {
        "patches": torch.stack([b["patches"] for b in batch], dim=0),
        "valid_time_mask": torch.stack([b["valid_time_mask"] for b in batch], dim=0),
        "valid_patch_mask": torch.stack([b["valid_patch_mask"] for b in batch], dim=0),
        "canales_constantes_mask": torch.stack(
            [b["canales_constantes_mask"] for b in batch], dim=0
        ),
        "labels": torch.tensor([b["class_id"] for b in batch], dtype=torch.long),
        "keys": [b["__key__"] for b in batch],
    }
    return out


# ----------------------------------------------------------------------
# Carga del backbone con SSL pretraining si procede
# ----------------------------------------------------------------------


def load_classifier(
    model_cfg: Dict[str, Any],
    n_classes: int,
    mode: str,
    checkpoint: Optional[Path],
    head_dropout: float = 0.0,
):
    """Construye y configura el classifier segun el modo."""
    import torch
    from models.patchtst_phm import build_patchtst_phm, count_parameters
    from training.downstream.heads import DownstreamClassifier

    backbone = build_patchtst_phm(model_cfg)
    if mode in ("linear_probing", "full_finetuning"):
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"--mode {mode} requiere --checkpoint valido; recibido {checkpoint}"
            )
        ck = torch.load(str(checkpoint), map_location="cpu")
        sd = ck.get("model_state_dict")
        if sd is None:
            raise RuntimeError(f"checkpoint sin model_state_dict: {checkpoint}")
        missing, unexpected = backbone.load_state_dict(sd, strict=False)
        if missing:
            print(f"  warning: missing keys ({len(missing)}): {missing[:5]}...")
        if unexpected:
            print(f"  warning: unexpected keys ({len(unexpected)}): {unexpected[:5]}...")
    freeze_backbone = (mode == "linear_probing")
    clf = DownstreamClassifier(
        backbone=backbone,
        n_classes=n_classes,
        freeze_backbone=freeze_backbone,
        head_dropout=head_dropout,
    )
    n_trainable = sum(p.numel() for p in clf.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in clf.parameters())
    return clf, {"n_trainable": n_trainable, "n_total": n_total}


# ----------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------


def cmd_dry_run(
    cfg: Dict[str, Any], mode: str, checkpoint: Optional[Path], repo_root: Path
) -> int:
    """Lectura, inspeccion y forward sintetico/real sin entrenar."""
    print(f"[{_ts()}] === DRY-RUN DOWNSTREAM === mode={mode}")
    print(f"  config_hash: {config_hash(cfg)}")

    data_cfg = cfg["data"]
    processed_root = Path(data_cfg["processed_root"])
    dataset = data_cfg["dataset"]

    # Manifest si esta accesible
    manifest_path = processed_root / dataset / "manifest.json"
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"  manifest: target_col={m.get('target_col')}  "
                  f"target_policy={m.get('target_policy')}  "
                  f"target_warning={m.get('target_warning')}  "
                  f"n_units_total={m.get('n_units_total')}")
        except Exception as e:
            print(f"  manifest no se pudo leer: {e}")
    else:
        print(f"  manifest no encontrado en {manifest_path} (esperado si "
              "estas fuera de Colab/Drive)")

    # Contar shards por split
    for split in ("train", "val", "test"):
        shards = find_shards(processed_root, dataset, split)
        print(f"  {split}: {len(shards)} shards")

    # Si hay train shards, inferir n_classes
    label_mapping: Dict[str, int] = {}
    train_counts: Counter = Counter()
    if find_shards(processed_root, dataset, "train"):
        print("\nConstruyendo label_mapping desde train...")
        try:
            label_mapping, train_counts = build_label_mapping(
                processed_root, dataset, split="train",
                max_samples=cfg.get("training", {}).get("max_train_batches_per_epoch"),
            )
            print(f"  n_classes detectadas: {len(label_mapping)}")
            print(f"  label_mapping: {label_mapping}")
            print(f"  conteo por clase (train): {dict(train_counts)}")
        except Exception as e:
            print(f"  fallo al construir label_mapping: {e}")
            label_mapping = {}

    # Si no se pudo inferir, asumimos un fallback razonable para el dry-run
    if not label_mapping:
        n_classes_assumed = 4
        print(f"\n  fallback: n_classes={n_classes_assumed} (no se pudo leer train)")
    else:
        n_classes_assumed = len(label_mapping)

    # Construir classifier
    print(f"\nConstruyendo classifier (mode={mode})...")
    try:
        clf, info = load_classifier(
            cfg["model"], n_classes_assumed, mode, checkpoint,
            head_dropout=float(cfg.get("training", {}).get("head_dropout", 0.0)),
        )
        print(f"  classifier: trainable={info['n_trainable']:,}  "
              f"total={info['n_total']:,}")
    except Exception as e:
        print(f"  fallo al construir classifier: {e}")
        return 2

    # Resolver batch_size adaptativo (T2). El dry-run debe usar el mismo
    # helper que el train real para detectar problemas antes de entrenar.
    print("\nResolviendo batch_size adaptativo...")
    bs_info = resolve_downstream_batch_size(processed_root, dataset, data_cfg)
    for w in bs_info["warnings"]:
        print(f"  WARN: {w}")
    print(
        f"  policy={bs_info['batch_size_policy']}  "
        f"requested={bs_info['batch_size_requested']}  "
        f"effective={bs_info['batch_size_effective']}  "
        f"n_channels={bs_info['n_channels']} ({bs_info['n_channels_source']})  "
        f"effective_bc={bs_info['effective_bc']}"
    )

    # Forward synthetic con C real (no hardcoded a 2).
    print("\nForward sintetico (un batch random):")
    try:
        import torch
        B = max(1, min(2, bs_info["batch_size_effective"]))
        C = bs_info["n_channels"]
        P = int(cfg["model"]["patch_size"])
        N = int(cfg["model"]["n_patches"])
        x = torch.randn(B, C, N, P)
        vtm = torch.ones(B, N * P, dtype=torch.bool)
        vpm = torch.ones(B, C, N, dtype=torch.bool)
        cc = torch.zeros(B, C, dtype=torch.bool)
        clf.eval()
        with torch.no_grad():
            out = clf(x, vtm, vpm, cc)
        print(f"  logits: {tuple(out['logits'].shape)}  "
              f"pooled: {tuple(out['pooled'].shape)}")
        print(f"  primeros logits: {out['logits'][0].tolist()}")
    except Exception as e:
        print(f"  fallo en forward: {e}")
        return 3

    print(f"\n[{_ts()}] === DRY-RUN OK ===")
    return 0


# ----------------------------------------------------------------------
# Entrenamiento + evaluacion
# ----------------------------------------------------------------------


@torch.no_grad() if False else (lambda f: f)  # placeholder para autocompletado
def _evaluate(
    clf,
    processed_root: Path,
    dataset: str,
    split: str,
    label_mapping: Dict[str, int],
    batch_size: int,
    device,
    seed: int,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evalua un split. Devuelve metricas agregadas + per-class.

    Incluye `y_true` y `y_pred` en el dict devuelto para que el caller
    decida si los persiste a fichero separado (segun
    `evaluation.save_predictions`). El `run_info.json` NO debe guardarlos
    por defecto.

    Metricas:
      - n_samples
      - accuracy, balanced_accuracy, macro_f1
      - confusion_matrix (lista de listas, shape K x K)
      - per_class: dict con precision/recall/f1/support_true/support_pred
      - labels_by_class_id: lista length K con los labels string de cada
        class_id, ordenados por id (label_mapping invertido).
      - zero_support_classes: list[str] con los labels que tienen
        support_true=0 en este split (=clases que existen en train pero
        no aparecen aqui).
    """
    import torch
    clf.eval()
    all_pred: List[int] = []
    all_true: List[int] = []
    with torch.no_grad():
        for batch in iter_split_batches(
            processed_root, dataset, split, batch_size,
            label_mapping, shuffle=False, seed=seed, max_batches=max_batches,
        ):
            x = batch["patches"].to(device)
            vtm = batch["valid_time_mask"].to(device)
            vpm = batch["valid_patch_mask"].to(device)
            cc = batch["canales_constantes_mask"].to(device)
            labels = batch["labels"].to(device)
            out = clf(x, vtm, vpm, cc)
            preds = out["logits"].argmax(dim=-1)
            all_pred.extend(preds.detach().cpu().tolist())
            all_true.extend(labels.detach().cpu().tolist())
    n_classes = len(label_mapping)
    per_class = per_class_precision_recall_f1(all_true, all_pred, n_classes=n_classes)
    # Invertir label_mapping {label: id} -> [label_para_id_0, label_para_id_1, ...]
    labels_by_class_id = [None] * n_classes
    for lbl, idx in label_mapping.items():
        if 0 <= idx < n_classes:
            labels_by_class_id[idx] = lbl
    # Clases que no aparecen en este split (support_true == 0):
    zero_support = [
        labels_by_class_id[c] for c in range(n_classes)
        if per_class["support_true"][c] == 0
    ]
    return {
        "n_samples": len(all_true),
        "accuracy": accuracy(all_true, all_pred),
        "balanced_accuracy": balanced_accuracy(all_true, all_pred, n_classes=n_classes),
        "macro_f1": macro_f1(all_true, all_pred, n_classes=n_classes),
        "confusion_matrix": confusion_matrix(all_true, all_pred, n_classes=n_classes).tolist(),
        "per_class": per_class,
        "labels_by_class_id": labels_by_class_id,
        "zero_support_classes": zero_support,
        "y_true": all_true,
        "y_pred": all_pred,
    }


def cmd_train(
    cfg: Dict[str, Any], mode: str, checkpoint: Optional[Path], repo_root: Path
) -> int:
    """Entrena en un modo (`from_scratch` | `linear_probing` | `full_finetuning`).

    Estructura del bucle:
      - construye label_mapping desde train.
      - inicializa classifier segun mode.
      - epoca = pasada completa sobre los shards de train (shuffle por
        shard, batches por dataset).
      - cada `eval_every_epochs` evalua val; guarda best si mejora
        `metric_for_best`.
      - al final del entrenamiento evalua test con el best ckpt.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    print(f"[{_ts()}] === TRAIN DOWNSTREAM === mode={mode}  run={cfg['run_name']}")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    paths = cfg["paths"]
    processed_root = Path(data_cfg["processed_root"])
    dataset = data_cfg["dataset"]

    # Label mapping desde train
    print("Construyendo label_mapping desde train...")
    label_mapping, train_counts = build_label_mapping(processed_root, dataset, "train")
    n_classes = len(label_mapping)
    print(f"  n_classes={n_classes}  mapping={label_mapping}")
    print(f"  counts train: {dict(train_counts)}")

    # Validar que val/test no contienen clases desconocidas
    for s in ("val", "test"):
        if find_shards(processed_root, dataset, s):
            _, counts_s = build_label_mapping(processed_root, dataset, s)
            unknown = set(counts_s.keys()) - set(label_mapping.keys())
            if unknown:
                raise RuntimeError(
                    f"Split {s} contiene clases desconocidas (no vistas en train): {unknown}"
                )

    # Outputs
    log_dir = Path(paths["log_dir"]) / cfg["run_name"]
    ckpt_dir = Path(paths["checkpoint_dir"]) / cfg["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    _safe_json_dump({"label_mapping": label_mapping,
                     "train_counts": dict(train_counts)},
                    log_dir / "label_mapping.json")
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    clf, info = load_classifier(
        cfg["model"], n_classes, mode, checkpoint,
        head_dropout=float(training_cfg.get("head_dropout", 0.0)),
    )
    clf.to(device)
    print(f"  trainable params: {info['n_trainable']:,}  / total {info['n_total']:,}")

    # Optimizer + LR. `lr_backbone` admite `null` explicito en el YAML
    # (caso linear_probing, backbone congelado). Ver resolve_lr_backbone.
    param_groups = clf.trainable_parameter_groups(
        lr_head=float(training_cfg["lr_head"]),
        lr_backbone=resolve_lr_backbone(training_cfg),
    )
    optimizer = AdamW(param_groups, weight_decay=float(training_cfg["weight_decay"]))

    amp_cfg = training_cfg.get("amp", "auto")
    use_amp = (amp_cfg in ("auto", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    git_info = get_git_info(repo_root)
    cfg_hash = config_hash(cfg)
    logger = JsonlLogger(log_dir / "metrics.jsonl")
    grad_clip = float(training_cfg.get("grad_clip_norm", 1.0))
    log_every = int(training_cfg.get("log_every", 50))
    eval_every = int(training_cfg.get("eval_every_epochs", 1))
    metric_for_best = str(training_cfg.get("metric_for_best", "macro_f1_val"))
    max_train_per_epoch = training_cfg.get("max_train_batches_per_epoch")
    max_val_batches = training_cfg.get("max_val_batches")
    max_test_batches = training_cfg.get("max_test_batches")

    best_value = -float("inf")
    best_epoch = -1
    best_ckpt_path: Optional[Path] = None
    history: List[Dict[str, Any]] = []
    global_step = 0
    t0 = time.time()

    loss_fn = torch.nn.CrossEntropyLoss()
    max_epochs = int(training_cfg["max_epochs"])

    # Resolver batch_size adaptativo (T2). Misma logica que dry-run.
    bs_info = resolve_downstream_batch_size(processed_root, dataset, data_cfg)
    for w in bs_info["warnings"]:
        print(f"  WARN batch_size: {w}")
    # V5: en train REAL (no dry-run), abortar si caemos al default 2
    # porque no se pudo inferir n_channels desde manifest ni desde
    # primer sample ni desde n_channels_fallback. Entrenar con C=2
    # asumido sobre un dataset con C real distinto produciria batches
    # mal formateados y resultados invalidos.
    if bs_info["n_channels_source"] == "fallback_default":
        raise RuntimeError(
            f"No se pudo inferir n_channels para {dataset}: ni manifest "
            f"({processed_root}/{dataset}/manifest.json), ni primer sample "
            "train, ni n_channels_fallback en data_cfg. Asumir n_channels=2 "
            "es inseguro en train real. Fix: anadir n_channels al manifest "
            "o pasar n_channels_fallback en el YAML con el valor correcto."
        )
    batch_size = int(bs_info["batch_size_effective"])
    print(
        f"  batch_size: policy={bs_info['batch_size_policy']}  "
        f"requested={bs_info['batch_size_requested']}  "
        f"effective={batch_size}  "
        f"n_channels={bs_info['n_channels']} ({bs_info['n_channels_source']})  "
        f"effective_bc={bs_info['effective_bc']}"
    )

    # Contador AMP overflow (steps con grad_norm no finito). Tolerado bajo AMP
    # (GradScaler omite el step), fatal sin AMP (algo va realmente mal).
    amp_nonfinite_grad_steps = 0

    for epoch in range(1, max_epochs + 1):
        clf.train()
        train_iter = iter_split_batches(
            processed_root, dataset, "train",
            batch_size, label_mapping,
            shuffle=True, seed=seed + epoch,
            max_batches=max_train_per_epoch,
        )
        for batch in train_iter:
            global_step += 1
            x = batch["patches"].to(device, non_blocking=True)
            vtm = batch["valid_time_mask"].to(device, non_blocking=True)
            vpm = batch["valid_patch_mask"].to(device, non_blocking=True)
            cc = batch["canales_constantes_mask"].to(device, non_blocking=True)
            labels = batch["labels"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = clf(x, vtm, vpm, cc)
                    loss = loss_fn(out["logits"], labels)
            else:
                out = clf(x, vtm, vpm, cc)
                loss = loss_fn(out["logits"], labels)

            # Loss no finita: fallo duro (igual con o sin AMP).
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"step {global_step}: loss no finita ({float(loss.detach())}). Abortando."
                )

            # Backward + clip + step. Bajo AMP: el GradScaler puede detectar
            # gradientes no finitos y omitir el step (devuelve grad_norm=inf).
            # Tolerado. Sin AMP: si grad_norm no finito, algo va mal, abortar.
            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in clf.parameters() if p.requires_grad], grad_clip
                )
                # scaler.step internamente comprueba si los grads son finitos.
                # Si no, no aplica el optimizer.step y baja el scale para el
                # proximo step. No abortar.
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in clf.parameters() if p.requires_grad], grad_clip
                )
                optimizer.step()

            # Diagnostico del paso (T4). Compatibilidad con SSL central trainer:
            # mismos campos optimizer_applied/grad_norm_nonfinite_kind/amp_nonfinite_grad.
            gn_finite = bool(torch.isfinite(grad_norm).item())
            if not gn_finite:
                amp_nonfinite_grad_steps += 1
                if not use_amp:
                    # Sin AMP no hay GradScaler; un grad_norm no finito indica
                    # un problema real (NaN en el modelo, datos corruptos, ...).
                    raise RuntimeError(
                        f"step {global_step}: grad_norm no finito sin AMP "
                        f"({float(grad_norm.detach())}). Abortando."
                    )

            if global_step % log_every == 0:
                with torch.no_grad():
                    preds = out["logits"].argmax(dim=-1)
                    acc = (preds == labels).float().mean().item()
                # Detach explicito para no emitir UserWarning con grad enganchado.
                loss_val = float(loss.detach())
                gn_val = float(grad_norm.detach()) if gn_finite else None
                # optimizer_applied: True si el optimizer.step se ejecutó realmente.
                # Sin AMP siempre True (ya abortamos arriba si gn no finito).
                # Con AMP: True solo si los grads eran finitos; el GradScaler
                # internamente omite el step si detecta no-finitos.
                optimizer_applied = bool(gn_finite or not use_amp)
                gn_nonfinite_kind = None
                if not gn_finite:
                    gn_raw = float(grad_norm.detach())
                    gn_nonfinite_kind = "nan" if gn_raw != gn_raw else "inf"
                print(f"  e{epoch:>2d} step {global_step:>5d}  "
                      f"loss={loss_val:.4f}  acc_batch={acc:.4f}  "
                      f"gn={'inf/nan' if not gn_finite else f'{gn_val:.3f}'}")
                logger.log({
                    "kind": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": loss_val,
                    "acc_batch": acc,
                    "grad_norm": gn_val,
                    "grad_norm_is_finite": gn_finite,
                    "grad_norm_nonfinite_kind": gn_nonfinite_kind,
                    "amp_nonfinite_grad": (not gn_finite) and use_amp,
                    "optimizer_applied": optimizer_applied,
                    "batch_size_effective": batch_size,
                    "effective_bc": batch_size * bs_info["n_channels"],
                })

        # Eval val al final de cada epoch (o segun eval_every)
        if epoch % eval_every == 0:
            print(f"  e{epoch:>2d} eval val...")
            val_metrics = _evaluate(
                clf, processed_root, dataset, "val",
                label_mapping, batch_size, device,
                seed=seed, max_batches=max_val_batches,
            )
            print(f"    val: acc={val_metrics['accuracy']:.4f}  "
                  f"bal_acc={val_metrics['balanced_accuracy']:.4f}  "
                  f"macro_f1={val_metrics['macro_f1']:.4f}  "
                  f"n={val_metrics['n_samples']}")
            if val_metrics["zero_support_classes"]:
                print(f"    val WARN: clases sin soporte en val: "
                      f"{val_metrics['zero_support_classes']}")
            logger.log({
                "kind": "val_eval",
                "epoch": epoch,
                "step": global_step,
                "n_samples": val_metrics["n_samples"],
                "accuracy": val_metrics["accuracy"],
                "balanced_accuracy": val_metrics["balanced_accuracy"],
                "macro_f1": val_metrics["macro_f1"],
                "zero_support_classes": val_metrics["zero_support_classes"],
                "support_true": val_metrics["per_class"]["support_true"],
                "support_pred": val_metrics["per_class"]["support_pred"],
            })

            metric_key = metric_for_best.replace("_val", "")
            val_value = val_metrics.get(metric_key, val_metrics["macro_f1"])
            if val_value > best_value:
                best_value = float(val_value)
                best_epoch = epoch
                best_ckpt_path = ckpt_dir / "best.pt"
                ck = {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": clf.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "mode": mode,
                    "metric_for_best": metric_for_best,
                    "best_value": best_value,
                    "label_mapping": label_mapping,
                    "git_hash": git_info["git_hash"],
                    "config_hash": cfg_hash,
                }
                torch.save(ck, best_ckpt_path)
                print(f"    NEW BEST ({metric_for_best}={best_value:.4f}) "
                      f"-> {best_ckpt_path}")

    # Eval final test con best ckpt
    test_metrics: Optional[Dict[str, Any]] = None
    if best_ckpt_path is not None and best_ckpt_path.is_file():
        print(f"\nCargando best ckpt para test: {best_ckpt_path}")
        ck = torch.load(str(best_ckpt_path), map_location=device)
        clf.load_state_dict(ck["model_state_dict"])
        test_metrics = _evaluate(
            clf, processed_root, dataset, "test",
            label_mapping, batch_size, device,
            seed=seed, max_batches=max_test_batches,
        )
        print(f"  test: acc={test_metrics['accuracy']:.4f}  "
              f"bal_acc={test_metrics['balanced_accuracy']:.4f}  "
              f"macro_f1={test_metrics['macro_f1']:.4f}  "
              f"n={test_metrics['n_samples']}")
        if test_metrics["zero_support_classes"]:
            print(f"  test WARN: clases sin soporte en test: "
                  f"{test_metrics['zero_support_classes']}")
        logger.log({
            "kind": "test_eval",
            "from_best_epoch": best_epoch,
            "n_samples": test_metrics["n_samples"],
            "accuracy": test_metrics["accuracy"],
            "balanced_accuracy": test_metrics["balanced_accuracy"],
            "macro_f1": test_metrics["macro_f1"],
            "zero_support_classes": test_metrics["zero_support_classes"],
            "support_true": test_metrics["per_class"]["support_true"],
            "support_pred": test_metrics["per_class"]["support_pred"],
        })

        # Opcional: persistir y_true/y_pred a fichero separado segun config.
        eval_cfg = cfg.get("evaluation") or {}
        if bool(eval_cfg.get("save_predictions", False)):
            preds_path = log_dir / "predictions_test.json"
            _safe_json_dump(
                {
                    "y_true": test_metrics["y_true"],
                    "y_pred": test_metrics["y_pred"],
                    "label_mapping": label_mapping,
                },
                preds_path,
            )
            print(f"  predicciones test guardadas en: {preds_path}")

    elapsed = time.time() - t0
    logger.close()

    # Construir version compacta de test_metrics para run_info (sin
    # y_true/y_pred, que son arrays largos).
    test_metrics_compact = None
    if test_metrics is not None:
        test_metrics_compact = {
            k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred")
        }

    run_info = {
        "ts": _ts(),
        "mode": mode,
        "run_name": cfg["run_name"],
        "dataset": dataset,
        "seed": seed,
        "git_hash": git_info["git_hash"],
        "git_dirty": git_info["git_dirty"],
        "config_hash": cfg_hash,
        "checkpoint_loaded": str(checkpoint) if checkpoint else None,
        "label_mapping": label_mapping,
        "train_counts": dict(train_counts),
        "n_classes": n_classes,
        "n_trainable_params": info["n_trainable"],
        "n_total_params": info["n_total"],
        # Batch info (T2).
        "batch_size_requested":  bs_info["batch_size_requested"],
        "batch_size_effective":  bs_info["batch_size_effective"],
        "batch_size_policy":     bs_info["batch_size_policy"],
        "n_channels":            bs_info["n_channels"],
        "n_channels_source":     bs_info["n_channels_source"],
        "max_channel_batch":     bs_info["max_channel_batch"],
        "min_batch_size":        bs_info["min_batch_size"],
        "effective_bc":          bs_info["effective_bc"],
        # AMP info (T4).
        "amp_used":                  bool(use_amp),
        "amp_nonfinite_grad_steps":  amp_nonfinite_grad_steps,
        # Metricas.
        "best_epoch": best_epoch,
        "best_value": best_value if best_epoch >= 0 else None,
        "metric_for_best": metric_for_best,
        "test_metrics": test_metrics_compact,
        "zero_support_classes_test": (
            test_metrics["zero_support_classes"] if test_metrics is not None else None
        ),
        "elapsed_seconds": round(elapsed, 1),
        "model": cfg["model"],
        "training": cfg["training"],
    }
    _safe_json_dump(run_info, log_dir / "run_info.json")
    print(f"\n[{_ts()}] === TRAIN DOWNSTREAM end ===  best_epoch={best_epoch}  "
          f"best_{metric_for_best}={best_value:.4f}")
    print(f"Logs:  {log_dir}")
    print(f"Ckpts: {ckpt_dir}")
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Downstream classification trainer")
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("from_scratch", "linear_probing", "full_finetuning"),
        required=True,
    )
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="ruta al ckpt del SSL central full (requerido en "
                        "linear_probing y full_finetuning)")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    repo_root = _REPO_ROOT
    if args.dry_run:
        return cmd_dry_run(cfg, args.mode, args.checkpoint, repo_root)
    if args.mode in ("linear_probing", "full_finetuning") and args.checkpoint is None:
        print(f"ERROR: --mode {args.mode} requiere --checkpoint", file=sys.stderr)
        return 2
    return cmd_train(cfg, args.mode, args.checkpoint, repo_root)


if __name__ == "__main__":
    sys.exit(main())
