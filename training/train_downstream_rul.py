"""Trainer downstream de regresion RUL sobre CMAPSS_RUL (commit 5).

Estructuralmente analogo a `train_downstream_classification.py` pero
para regresion escalar. Tres modos mutuamente exclusivos:

    --mode from_scratch        backbone random + cabeza random
    --mode linear_probing      encoder SSL congelado + cabeza entrenable
    --mode full_finetuning     encoder SSL + cabeza, ambos entrenables

Para los modos basados en SSL hay que pasar `--checkpoint <ruta>` al
ckpt del SSL central full (`ckpt_step100000.pt`).

Lee shards del builder CMAPSS RUL en
`<processed_downstream_root>/CMAPSS_RUL/{train,val,test}/shard_*.tar`,
no del corpus harmonizado SSL (`processed/`). Esto refleja la decision
de no usar el target ambiguo de `processed/CMAPSS/` (ver
`results/downstream/cmapss_rul_decision/decision.md`).

Target seleccionable via config:

    target_key: rul_capped_125    # canonico (Heimes 2008), o
    target_key: rul_physical      # sin cap

Metricas: MAE, RMSE, R^2, CMAPSS-Score (Saxena 2008, asimetrica).

Trazabilidad por run (analoga al classification trainer):

    paths.log_dir/<run_name>/{config.yaml, run_info.json, metrics.jsonl}
    paths.checkpoint_dir/<run_name>/best.pt

Modo `--dry-run`: lee manifest CMAPSS_RUL, cuenta shards por split,
inspecciona el primer sample, construye modelo + cabeza, hace forward
sintetico, reporta target_key, n_channels (siempre 24) y sale sin
entrenar.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Reutilizamos los helpers del trainer classification (JsonlLogger,
# _safe_json_dump, _ts, resolve_downstream_batch_size, etc.) y del SSL
# central (config_hash, get_git_info, load_config).
from training.train_ssl_central import (
    JsonlLogger,
    config_hash,
    get_git_info,
    load_config,
)
from training.train_downstream_classification import (
    _safe_json_dump,
    _ts,
)
from training.downstream.metrics import regression_metrics
from training.phm_tar_reader import iter_samples_from_tar
from training.sampling import compute_adaptive_batch_size


# ----------------------------------------------------------------------
# Constantes especificas RUL
# ----------------------------------------------------------------------

# El builder CMAPSS_RUL escribe shards con el patron `shard_NNNN.tar`.
# Distinto del patron `<DATASET>-<split>-NNNN.tar` de la harmonization
# v0.5 SSL; por eso se localizan con un helper propio.
RUL_DATASET_NAME = "CMAPSS_RUL"
RUL_SHARD_GLOB = "shard_*.tar"

# Targets canonicos del manifest del builder.
ALLOWED_TARGET_KEYS = ("rul_physical", "rul_capped_125")


def find_shards_rul(processed_downstream_root: Path, split: str) -> List[Path]:
    """Localiza los shards CMAPSS_RUL de un split.

    Convencion del builder (commits 3/fd0cec9):
        <processed_downstream_root>/CMAPSS_RUL/<split>/shard_NNNN.tar
    """
    split_dir = Path(processed_downstream_root) / RUL_DATASET_NAME / split
    if not split_dir.is_dir():
        return []
    return sorted(split_dir.glob(RUL_SHARD_GLOB))


# ----------------------------------------------------------------------
# Lectura de target escalar desde sample
# ----------------------------------------------------------------------


def _extract_target(sample: Dict[str, Any], target_key: str) -> float:
    """Lee el target escalar de un sample CMAPSS_RUL.

    Estrategia:
      1. Preferir `sample["meta"][f"target_{target_key}"]` (el builder
         enriquece meta.json con `target_rul_physical` y
         `target_rul_capped_125` durante `build_sample_payload`).
      2. Fallback: si el meta no lo tiene (versiones futuras del
         builder), leer del `.npy` escalar `sample[target_key]`
         (`rul_physical` o `rul_capped_125`) que el tar reader carga
         como np.ndarray escalar tras la extension de `_KNOWN_SUFFIXES`.

    Args:
        sample: dict emitido por `iter_samples_from_tar`.
        target_key: nombre canonico en `ALLOWED_TARGET_KEYS`.

    Raises:
        ValueError si `target_key` no es valido o si el sample no
            contiene el target.
    """
    if target_key not in ALLOWED_TARGET_KEYS:
        raise ValueError(
            f"target_key desconocido: {target_key!r}. "
            f"Esperado uno de {ALLOWED_TARGET_KEYS}."
        )
    meta = sample.get("meta") or {}
    meta_key = f"target_{target_key}"
    if meta_key in meta:
        return float(meta[meta_key])
    # Fallback: leer del .npy escalar.
    arr = sample.get(target_key)
    if arr is not None:
        try:
            return float(arr.item()) if hasattr(arr, "item") else float(arr)
        except Exception as exc:
            raise ValueError(
                f"sample {sample.get('__key__')}: target {target_key!r} "
                f"no convertible a float: {exc}"
            )
    raise ValueError(
        f"sample {sample.get('__key__')}: no contiene meta['{meta_key}'] "
        f"ni .{target_key}.npy."
    )


# ----------------------------------------------------------------------
# Batch construction
# ----------------------------------------------------------------------


def _sample_to_tensors(sample: Dict[str, Any], target_key: str) -> Dict[str, Any]:
    """Convierte un sample numpy a tensores torch + target float."""
    import torch
    target = _extract_target(sample, target_key)
    return {
        "patches":                 torch.from_numpy(sample["patches"]).contiguous(),
        "valid_time_mask":         torch.from_numpy(sample["valid_time_mask"]).contiguous(),
        "valid_patch_mask":        torch.from_numpy(sample["valid_patch_mask"]).contiguous(),
        "canales_constantes_mask": torch.from_numpy(sample["canales_constantes_mask"]).contiguous(),
        "target":                  float(target),
        "__key__":                 sample.get("__key__"),
    }


def _collate(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Apila el batch downstream RUL."""
    import torch
    p0 = batch[0]["patches"].shape
    for i, b in enumerate(batch[1:], 1):
        if b["patches"].shape != p0:
            raise ValueError(
                f"batch heterogeneo en patches: ref {p0} vs sample {i} "
                f"{b['patches'].shape}"
            )
    return {
        "patches":                 torch.stack([b["patches"] for b in batch], dim=0),
        "valid_time_mask":         torch.stack([b["valid_time_mask"] for b in batch], dim=0),
        "valid_patch_mask":        torch.stack([b["valid_patch_mask"] for b in batch], dim=0),
        "canales_constantes_mask": torch.stack(
            [b["canales_constantes_mask"] for b in batch], dim=0,
        ),
        "targets": torch.tensor(
            [b["target"] for b in batch], dtype=torch.float32,
        ),
        "keys": [b["__key__"] for b in batch],
    }


def iter_split_batches(
    processed_downstream_root: Path,
    split: str,
    batch_size: int,
    target_key: str,
    shuffle: bool = True,
    seed: int = 0,
    max_batches: Optional[int] = None,
) -> Iterator[Dict[str, Any]]:
    """Itera batches de un split de CMAPSS_RUL."""
    import numpy as np

    shards = find_shards_rul(processed_downstream_root, split)
    if not shards:
        raise FileNotFoundError(
            f"No hay shards en {processed_downstream_root}/"
            f"{RUL_DATASET_NAME}/{split}/{RUL_SHARD_GLOB}"
        )

    rng = np.random.default_rng(seed)
    if shuffle:
        shards = list(shards)
        rng.shuffle(shards)

    buf: List[Dict[str, Any]] = []
    n_yielded = 0
    for shard in shards:
        for s in iter_samples_from_tar(shard, strict=True):
            buf.append(_sample_to_tensors(s, target_key))
            if len(buf) == batch_size:
                yield _collate(buf)
                buf = []
                n_yielded += 1
                if max_batches and n_yielded >= max_batches:
                    return
    if buf:
        yield _collate(buf)


# ----------------------------------------------------------------------
# Batch adaptativo para CMAPSS_RUL (n_channels = 24 garantizado por manifest)
# ----------------------------------------------------------------------


def resolve_rul_batch_size(
    processed_downstream_root: Path, data_cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Calcula el batch_size efectivo CMAPSS_RUL aplicando el cap `B*C`.

    A diferencia del classifier (`resolve_downstream_batch_size`), aqui
    `n_channels` esta garantizado por el manifest del builder
    (`n_channels=24`); si no aparece, fallamos duro porque el manifest
    es canonico y construido por nosotros.
    """
    warnings: List[str] = []
    manifest_path = (
        Path(processed_downstream_root) / RUL_DATASET_NAME / "manifest.json"
    )
    n_channels = None
    n_channels_source = "unknown"
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            n_channels = int(m.get("n_channels", 0))
            if n_channels > 0:
                n_channels_source = "manifest"
        except Exception as exc:
            warnings.append(f"manifest CMAPSS_RUL no parseable: {exc}")
    if not n_channels:
        # Fallback explicito del config (para dry-runs fuera de Drive).
        n_channels_fallback = data_cfg.get("n_channels_fallback")
        if n_channels_fallback:
            n_channels = int(n_channels_fallback)
            n_channels_source = "config_fallback"
            warnings.append(
                f"manifest sin n_channels; usando "
                f"n_channels_fallback={n_channels}"
            )
        else:
            raise RuntimeError(
                f"No se pudo inferir n_channels de "
                f"{manifest_path}. CMAPSS_RUL canonico = 24; el manifest "
                f"deberia tenerlo. Pasa data.n_channels_fallback=24 en el "
                f"YAML solo para dry-runs sin Drive."
            )

    batch_size_requested = int(data_cfg.get("batch_size", 8))
    policy = str(data_cfg.get("batch_size_policy", "adaptive_by_channels"))
    max_channel_batch = int(data_cfg.get("max_channel_batch", 512))
    min_batch_size = int(data_cfg.get("min_batch_size", 1))

    if policy == "adaptive_by_channels":
        bs_eff = compute_adaptive_batch_size(
            n_channels=n_channels,
            batch_size=batch_size_requested,
            max_channel_batch=max_channel_batch,
            min_batch_size=min_batch_size,
        )
    elif policy in ("fixed", "static"):
        bs_eff = max(min_batch_size, batch_size_requested)
    else:
        raise ValueError(
            f"batch_size_policy desconocida: {policy!r}. "
            "Esperado 'adaptive_by_channels' o 'fixed'."
        )

    return {
        "batch_size_requested": batch_size_requested,
        "batch_size_effective": int(bs_eff),
        "batch_size_policy": policy,
        "n_channels": int(n_channels),
        "n_channels_source": n_channels_source,
        "max_channel_batch": int(max_channel_batch),
        "min_batch_size": int(min_batch_size),
        "effective_bc": int(bs_eff) * int(n_channels),
        "warnings": warnings,
    }


# ----------------------------------------------------------------------
# Construccion del modelo RUL
# ----------------------------------------------------------------------


def load_regressor(
    model_cfg: Dict[str, Any],
    mode: str,
    checkpoint: Optional[Path],
    head_cfg: Dict[str, Any],
):
    """Construye y configura el regressor segun el modo.

    Args:
        model_cfg: bloque `model:` del YAML (forwarded a
            `build_patchtst_phm`).
        mode: 'from_scratch' | 'linear_probing' | 'full_finetuning'.
        checkpoint: ruta al ckpt SSL central (obligatorio salvo
            from_scratch).
        head_cfg: bloque `head:` del YAML con `hidden_dim`, `dropout`,
            `activation`, `keep_last_dim`. Todos opcionales.

    Returns:
        (model, info_dict).
    """
    import torch
    from models.patchtst_phm import build_patchtst_phm
    from training.downstream.heads import RegressionDownstreamModel

    backbone = build_patchtst_phm(model_cfg)
    if mode in ("linear_probing", "full_finetuning"):
        if checkpoint is None or not Path(checkpoint).is_file():
            raise FileNotFoundError(
                f"--mode {mode} requiere --checkpoint valido; "
                f"recibido {checkpoint}"
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
    model = RegressionDownstreamModel(
        backbone=backbone,
        freeze_backbone=freeze_backbone,
        head_hidden_dim=head_cfg.get("hidden_dim"),
        head_dropout=float(head_cfg.get("dropout", 0.0)),
        head_activation=head_cfg.get("activation"),
        head_keep_last_dim=bool(head_cfg.get("keep_last_dim", False)),
    )
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return model, {"n_trainable": n_trainable, "n_total": n_total}


# ----------------------------------------------------------------------
# Evaluacion
# ----------------------------------------------------------------------


def _evaluate(
    model,
    processed_downstream_root: Path,
    split: str,
    target_key: str,
    batch_size: int,
    device,
    seed: int,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    """Evalua un split. Devuelve regression_metrics + arrays para
    persistencia opcional.
    """
    import torch
    model.eval()
    y_true: List[float] = []
    y_pred: List[float] = []
    with torch.no_grad():
        for batch in iter_split_batches(
            processed_downstream_root, split, batch_size, target_key,
            shuffle=False, seed=seed, max_batches=max_batches,
        ):
            x = batch["patches"].to(device)
            vtm = batch["valid_time_mask"].to(device)
            vpm = batch["valid_patch_mask"].to(device)
            cc = batch["canales_constantes_mask"].to(device)
            targets = batch["targets"].to(device)
            out = model(x, vtm, vpm, cc)
            pred = out["prediction"]
            # Si el head se configuro keep_last_dim=True, prediction es
            # (B, 1); squeezeamos para comparar con (B,).
            if pred.dim() == 2 and pred.shape[1] == 1:
                pred = pred.squeeze(-1)
            y_true.extend(targets.detach().cpu().tolist())
            y_pred.extend(pred.detach().cpu().tolist())
    metrics = regression_metrics(y_true, y_pred, prefix="", include_cmapss_score=True)
    metrics["n_samples"] = len(y_true)
    metrics["y_true"] = y_true
    metrics["y_pred"] = y_pred
    return metrics


# ----------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------


def cmd_dry_run(
    cfg: Dict[str, Any], mode: str, checkpoint: Optional[Path], repo_root: Path,
) -> int:
    """Inspeccion + forward sintetico sin entrenar."""
    print(f"[{_ts()}] === DRY-RUN DOWNSTREAM RUL === mode={mode}")
    print(f"  config_hash: {config_hash(cfg)}")

    data_cfg = cfg["data"]
    processed_downstream_root = Path(data_cfg["processed_downstream_root"])
    target_key = str(data_cfg.get("target_key", "rul_capped_125"))
    print(f"  target_key: {target_key}")

    if target_key not in ALLOWED_TARGET_KEYS:
        print(f"  ERROR: target_key invalido {target_key!r}; "
              f"esperado {ALLOWED_TARGET_KEYS}")
        return 2

    # Manifest CMAPSS_RUL si accesible.
    manifest_path = processed_downstream_root / RUL_DATASET_NAME / "manifest.json"
    if manifest_path.is_file():
        try:
            m = json.loads(manifest_path.read_text(encoding="utf-8"))
            print(f"  manifest: "
                  f"role={m.get('role')}  "
                  f"client={m.get('client')}  "
                  f"window_size={m.get('window_size')}  "
                  f"n_channels={m.get('n_channels')}  "
                  f"target_policy={m.get('target_policy')}")
            tcands = m.get("target_candidates")
            if tcands:
                if target_key not in tcands:
                    print(f"  WARN: target_key {target_key!r} no esta en "
                          f"target_candidates del manifest: {tcands}")
        except Exception as exc:
            print(f"  manifest no parseable: {exc}")
    else:
        print(f"  manifest no encontrado en {manifest_path} "
              "(esperable fuera de Colab/Drive)")

    # Contar shards por split.
    for split in ("train", "val", "test"):
        shards = find_shards_rul(processed_downstream_root, split)
        print(f"  {split}: {len(shards)} shards")

    # Resolver batch + n_channels (con fallback si no hay manifest).
    try:
        bs_info = resolve_rul_batch_size(processed_downstream_root, data_cfg)
        for w in bs_info["warnings"]:
            print(f"  WARN batch_size: {w}")
        print(
            f"  batch_size: policy={bs_info['batch_size_policy']}  "
            f"requested={bs_info['batch_size_requested']}  "
            f"effective={bs_info['batch_size_effective']}  "
            f"n_channels={bs_info['n_channels']} ({bs_info['n_channels_source']})  "
            f"effective_bc={bs_info['effective_bc']}"
        )
    except Exception as exc:
        print(f"  batch_size: no se pudo resolver ({exc})")

    # Forward sintetico (no toca Drive).
    head_cfg = cfg.get("head") or {}
    print(f"  head: hidden_dim={head_cfg.get('hidden_dim')}  "
          f"dropout={head_cfg.get('dropout', 0.0)}  "
          f"activation={head_cfg.get('activation')}  "
          f"keep_last_dim={head_cfg.get('keep_last_dim', False)}")
    try:
        import torch  # noqa: F401
        # Forward sintetico realista: pasar el `mode` y `checkpoint` reales
        # a load_regressor para que el dry-run ejerza exactamente el mismo
        # path de carga del backbone que cmd_train. Antes se pasaba
        # ("from_scratch", None) hardcoded, lo cual saltaba la carga del
        # ckpt SSL incluso cuando el caller habia pasado --checkpoint con
        # --mode linear_probing o --mode full_finetuning; el dry-run
        # mentia diciendo "OK" sin verificar el ckpt.
        model, info = load_regressor(cfg["model"], mode, checkpoint, head_cfg)
        print(f"  modelo: n_trainable={info['n_trainable']:,}  "
              f"n_total={info['n_total']:,}")
        # Forward sintetico B=2, C=24, N=32, P=16 (contrato canonico).
        import torch as _torch
        B = 2
        C = 24
        N = cfg["model"].get("n_patches", 32)
        P = cfg["model"].get("patch_size", 16)
        W = N * P
        x = _torch.randn(B, C, N, P)
        vtm = _torch.ones(B, W, dtype=_torch.bool)
        vpm = _torch.ones(B, C, N, dtype=_torch.bool)
        out = model(x, valid_time_mask=vtm, valid_patch_mask=vpm)
        print(f"  forward sintetico OK: "
              f"prediction.shape={tuple(out['prediction'].shape)}, "
              f"pooled.shape={tuple(out['pooled'].shape)}")
    except Exception as exc:
        print(f"  WARN forward sintetico fallo: {exc}")

    print(f"[{_ts()}] === DRY-RUN OK ===")
    return 0


# ----------------------------------------------------------------------
# Train
# ----------------------------------------------------------------------


def cmd_train(
    cfg: Dict[str, Any], mode: str, checkpoint: Optional[Path], repo_root: Path,
) -> int:
    """Entrena en un modo (`from_scratch` | `linear_probing` | `full_finetuning`)."""
    import numpy as np
    import torch
    from torch.optim import AdamW

    print(f"[{_ts()}] === TRAIN DOWNSTREAM RUL === mode={mode}  "
          f"run={cfg['run_name']}")
    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    data_cfg = cfg["data"]
    training_cfg = cfg["training"]
    head_cfg = cfg.get("head") or {}
    paths = cfg["paths"]
    processed_downstream_root = Path(data_cfg["processed_downstream_root"])
    target_key = str(data_cfg.get("target_key", "rul_capped_125"))

    if target_key not in ALLOWED_TARGET_KEYS:
        raise ValueError(
            f"target_key invalido en config: {target_key!r}; "
            f"esperado {ALLOWED_TARGET_KEYS}"
        )

    # Sanity de shards.
    train_shards = find_shards_rul(processed_downstream_root, "train")
    val_shards = find_shards_rul(processed_downstream_root, "val")
    test_shards = find_shards_rul(processed_downstream_root, "test")
    if not train_shards:
        raise FileNotFoundError(
            f"No hay shards train CMAPSS_RUL en "
            f"{processed_downstream_root}/{RUL_DATASET_NAME}/train/"
        )

    # Outputs.
    log_dir = Path(paths["log_dir"]) / cfg["run_name"]
    ckpt_dir = Path(paths["checkpoint_dir"]) / cfg["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}  target_key={target_key}")

    model, info = load_regressor(cfg["model"], mode, checkpoint, head_cfg)
    model.to(device)
    print(f"  trainable params: {info['n_trainable']:,}  / total {info['n_total']:,}")

    # Optimizer + LR. A diferencia del classifier, aqui aceptamos
    # `lr_backbone: null` explicito en el YAML (caso linear_probing y
    # el default conservador para CMAPSS RUL). float(None) rompe, asi
    # que se normaliza antes: None / 0 / 0.0 -> None (= un solo grupo).
    lr_backbone_cfg = training_cfg.get("lr_backbone")
    if lr_backbone_cfg is None:
        lr_backbone_arg = None
    else:
        lr_backbone_val = float(lr_backbone_cfg)
        lr_backbone_arg = lr_backbone_val if lr_backbone_val > 0 else None
    param_groups = model.trainable_parameter_groups(
        lr_head=float(training_cfg["lr_head"]),
        lr_backbone=lr_backbone_arg,
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
    metric_for_best = str(training_cfg.get("metric_for_best", "rmse_val"))
    lower_is_better = bool(training_cfg.get("lower_is_better", True))
    max_train_per_epoch = training_cfg.get("max_train_batches_per_epoch")
    max_val_batches = training_cfg.get("max_val_batches")
    max_test_batches = training_cfg.get("max_test_batches")

    # Batch size adaptativo.
    bs_info = resolve_rul_batch_size(processed_downstream_root, data_cfg)
    for w in bs_info["warnings"]:
        print(f"  WARN batch_size: {w}")
    batch_size = int(bs_info["batch_size_effective"])
    print(
        f"  batch_size: policy={bs_info['batch_size_policy']}  "
        f"requested={bs_info['batch_size_requested']}  "
        f"effective={batch_size}  "
        f"n_channels={bs_info['n_channels']} ({bs_info['n_channels_source']})  "
        f"effective_bc={bs_info['effective_bc']}"
    )

    # Inicial: best segun direccion.
    init_best = float("inf") if lower_is_better else -float("inf")
    best_value = init_best
    best_epoch = -1
    best_ckpt_path: Optional[Path] = None
    global_step = 0
    amp_nonfinite_grad_steps = 0
    t0 = time.time()

    loss_fn = torch.nn.MSELoss(reduction="mean")
    max_epochs = int(training_cfg["max_epochs"])

    def _is_better(new: float, current: float) -> bool:
        if lower_is_better:
            return new < current
        return new > current

    for epoch in range(1, max_epochs + 1):
        model.train()
        train_iter = iter_split_batches(
            processed_downstream_root, "train",
            batch_size, target_key,
            shuffle=True, seed=seed + epoch,
            max_batches=max_train_per_epoch,
        )
        for batch in train_iter:
            global_step += 1
            x = batch["patches"].to(device, non_blocking=True)
            vtm = batch["valid_time_mask"].to(device, non_blocking=True)
            vpm = batch["valid_patch_mask"].to(device, non_blocking=True)
            cc = batch["canales_constantes_mask"].to(device, non_blocking=True)
            targets = batch["targets"].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(x, vtm, vpm, cc)
                    pred = out["prediction"]
                    if pred.dim() == 2 and pred.shape[1] == 1:
                        pred = pred.squeeze(-1)
                    loss = loss_fn(pred, targets)
            else:
                out = model(x, vtm, vpm, cc)
                pred = out["prediction"]
                if pred.dim() == 2 and pred.shape[1] == 1:
                    pred = pred.squeeze(-1)
                loss = loss_fn(pred, targets)

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"step {global_step}: loss no finita "
                    f"({float(loss.detach())}). Abortando."
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    grad_clip,
                )
                optimizer.step()

            gn_finite = bool(torch.isfinite(grad_norm).item())
            if not gn_finite:
                amp_nonfinite_grad_steps += 1
                if not use_amp:
                    raise RuntimeError(
                        f"step {global_step}: grad_norm no finito sin AMP "
                        f"({float(grad_norm.detach())}). Abortando."
                    )

            if global_step % log_every == 0:
                with torch.no_grad():
                    mse_batch = float(loss.detach())
                    mae_batch = float(
                        torch.mean(torch.abs(pred.detach() - targets)).item()
                    )
                gn_val = float(grad_norm.detach()) if gn_finite else None
                optimizer_applied = bool(gn_finite or not use_amp)
                gn_nonfinite_kind = None
                if not gn_finite:
                    gn_raw = float(grad_norm.detach())
                    gn_nonfinite_kind = "nan" if gn_raw != gn_raw else "inf"
                print(f"  e{epoch:>2d} step {global_step:>5d}  "
                      f"loss(mse)={mse_batch:.4f}  mae_batch={mae_batch:.4f}  "
                      f"gn={'inf/nan' if not gn_finite else f'{gn_val:.3f}'}")
                logger.log({
                    "kind": "train_step",
                    "epoch": epoch,
                    "step": global_step,
                    "loss": mse_batch,
                    "mae_batch": mae_batch,
                    "grad_norm": gn_val,
                    "grad_norm_is_finite": gn_finite,
                    "grad_norm_nonfinite_kind": gn_nonfinite_kind,
                    "amp_nonfinite_grad": (not gn_finite) and use_amp,
                    "optimizer_applied": optimizer_applied,
                    "batch_size_effective": batch_size,
                    "effective_bc": batch_size * bs_info["n_channels"],
                })

        # Eval val al final de la epoca.
        if epoch % eval_every == 0 and val_shards:
            print(f"  e{epoch:>2d} eval val...")
            val_metrics = _evaluate(
                model, processed_downstream_root, "val",
                target_key, batch_size, device,
                seed=seed, max_batches=max_val_batches,
            )
            print(f"    val: mae={val_metrics['mae']:.4f}  "
                  f"rmse={val_metrics['rmse']:.4f}  "
                  f"r2={val_metrics['r2']:.4f}  "
                  f"cmapss_score={val_metrics['cmapss_score']:.2f}  "
                  f"n={val_metrics['n_samples']}")
            # Log compacto (sin y_true/y_pred).
            logger.log({
                "kind": "val_eval",
                "epoch": epoch,
                "step": global_step,
                "n_samples": val_metrics["n_samples"],
                "mae": val_metrics["mae"],
                "rmse": val_metrics["rmse"],
                "r2": val_metrics["r2"],
                "cmapss_score": val_metrics["cmapss_score"],
            })

            # metric_for_best convencion: `rmse_val` -> mira `val_metrics["rmse"]`.
            # Aceptamos formatos: "<metric>_val" o "<metric>" directamente.
            metric_key = metric_for_best
            if metric_key.endswith("_val"):
                metric_key = metric_key[: -len("_val")]
            if metric_key not in val_metrics:
                raise KeyError(
                    f"metric_for_best={metric_for_best!r} no apunta a una "
                    f"metrica devuelta por _evaluate. Disponibles: "
                    f"{[k for k in val_metrics.keys() if k not in ('y_true', 'y_pred')]}"
                )
            val_value = float(val_metrics[metric_key])

            if _is_better(val_value, best_value):
                best_value = val_value
                best_epoch = epoch
                best_ckpt_path = ckpt_dir / "best.pt"
                ck = {
                    "epoch": epoch,
                    "step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "config": cfg,
                    "mode": mode,
                    "metric_for_best": metric_for_best,
                    "lower_is_better": lower_is_better,
                    "best_value": best_value,
                    "target_key": target_key,
                    "git_hash": git_info["git_hash"],
                    "config_hash": cfg_hash,
                }
                torch.save(ck, best_ckpt_path)
                print(f"    NEW BEST ({metric_for_best}={best_value:.4f}) "
                      f"-> {best_ckpt_path}")

    # Eval final test con best ckpt.
    test_metrics: Optional[Dict[str, Any]] = None
    if best_ckpt_path is not None and best_ckpt_path.is_file() and test_shards:
        print(f"\nCargando best ckpt para test: {best_ckpt_path}")
        ck = torch.load(str(best_ckpt_path), map_location=device)
        model.load_state_dict(ck["model_state_dict"])
        test_metrics = _evaluate(
            model, processed_downstream_root, "test",
            target_key, batch_size, device,
            seed=seed, max_batches=max_test_batches,
        )
        print(f"  test: mae={test_metrics['mae']:.4f}  "
              f"rmse={test_metrics['rmse']:.4f}  "
              f"r2={test_metrics['r2']:.4f}  "
              f"cmapss_score={test_metrics['cmapss_score']:.2f}  "
              f"n={test_metrics['n_samples']}")
        logger.log({
            "kind": "test_eval",
            "from_best_epoch": best_epoch,
            "n_samples": test_metrics["n_samples"],
            "mae": test_metrics["mae"],
            "rmse": test_metrics["rmse"],
            "r2": test_metrics["r2"],
            "cmapss_score": test_metrics["cmapss_score"],
        })

        # Persistir y_true/y_pred si el config lo pide.
        eval_cfg = cfg.get("evaluation") or {}
        if bool(eval_cfg.get("save_predictions", False)):
            preds_path = log_dir / "predictions_test.json"
            _safe_json_dump(
                {
                    "y_true": test_metrics["y_true"],
                    "y_pred": test_metrics["y_pred"],
                    "target_key": target_key,
                },
                preds_path,
            )
            print(f"  predicciones test guardadas en: {preds_path}")

    elapsed = time.time() - t0
    logger.close()

    test_metrics_compact = None
    if test_metrics is not None:
        test_metrics_compact = {
            k: v for k, v in test_metrics.items() if k not in ("y_true", "y_pred")
        }

    run_info = {
        "ts": _ts(),
        "mode": mode,
        "run_name": cfg["run_name"],
        "dataset": RUL_DATASET_NAME,
        "target_key": target_key,
        "seed": seed,
        "git_hash": git_info["git_hash"],
        "git_dirty": git_info["git_dirty"],
        "config_hash": cfg_hash,
        "checkpoint_loaded": str(checkpoint) if checkpoint else None,
        "n_trainable_params": info["n_trainable"],
        "n_total_params": info["n_total"],
        # Batch.
        "batch_size_requested": bs_info["batch_size_requested"],
        "batch_size_effective": bs_info["batch_size_effective"],
        "batch_size_policy":    bs_info["batch_size_policy"],
        "n_channels":           bs_info["n_channels"],
        "n_channels_source":    bs_info["n_channels_source"],
        "max_channel_batch":    bs_info["max_channel_batch"],
        "min_batch_size":       bs_info["min_batch_size"],
        "effective_bc":         bs_info["effective_bc"],
        # AMP.
        "amp_used":                 bool(use_amp),
        "amp_nonfinite_grad_steps": amp_nonfinite_grad_steps,
        # Metricas.
        "best_epoch": best_epoch,
        "best_value": best_value if best_epoch >= 0 else None,
        "metric_for_best": metric_for_best,
        "lower_is_better": lower_is_better,
        "test_metrics": test_metrics_compact,
        "elapsed_seconds": round(elapsed, 1),
        "model": cfg["model"],
        "training": cfg["training"],
        "head": head_cfg,
    }
    _safe_json_dump(run_info, log_dir / "run_info.json")
    print(f"\n[{_ts()}] === TRAIN DOWNSTREAM RUL end ===  "
          f"best_epoch={best_epoch}  best_{metric_for_best}={best_value}")
    print(f"Logs:  {log_dir}")
    print(f"Ckpts: {ckpt_dir}")
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Downstream RUL regression trainer (CMAPSS_RUL)"
    )
    p.add_argument("--config", type=Path, required=True)
    p.add_argument(
        "--mode",
        choices=("from_scratch", "linear_probing", "full_finetuning"),
        required=True,
    )
    p.add_argument(
        "--checkpoint", type=Path, default=None,
        help="ruta al ckpt del SSL central full (requerido en "
             "linear_probing y full_finetuning)",
    )
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
