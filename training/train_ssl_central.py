"""Entrenamiento SSL centralizado: dry-run, smoke y train.

Modos del CLI:

- ``--mode dry-run``: NO carga shards, NO entrena. Lee summaries, construye
  el `sampling_plan`, lista datasets PRETRAIN_SOURCE, verifica rutas de
  shards y escribe `results/pretraining/ssl_sampling_plan.csv`. Si la
  config define `data.batch_size_policy='adaptive_by_channels'`, reporta
  el `B_eff` por dataset. Util para validar la politica sin gastar GPU.

- ``--mode smoke``: usa una lista corta de datasets pequenos
  (`data.smoke_datasets`) y `training.max_steps` reducido. Hace forward +
  backward + optimizer step reales sobre 5 datasets en orden round_robin
  por defecto. Verifica contrato end-to-end + diagnostico determinista
  de padding parcial antes del loop. Guarda checkpoint pequeno en Drive.

- ``--mode train``: pretraining centralizado real, gateado por
  `_validate_train_config`. Soporta tres stages mutuamente exclusivos:
    * ``stage: coverage`` (cap 10k steps): cubre los 36 PS con
      ``sampling_strategy: round_robin`` para validar compatibilidad VRAM.
    * ``stage: pilot``    (cap 10k steps): version corta de la politica
      productiva (``sampling_strategy: weighted``) con scheduler y
      reporte periodico de distribucion empirica.
    * ``stage: full``     (cap 500k steps): pretraining productivo sobre
      los 36 PS con weighted + caps + ``min_dataset_presence`` opcional.
  Cualquier otra stage o `max_steps` por encima del cap se rechaza con
  RuntimeError.

Trazabilidad por run (sec 13 de `CLAUDE.md`), en `paths.log_dir/<run_name>/`:

    - `config.yaml`        : copia del YAML efectivo.
    - `run_info.json`      : git_hash, git_dirty, config_hash, seed,
                             param_count, stage, conteos finales
                             (optimizer_steps, skipped_steps,
                             amp_overflow_steps, amp_nonfinite_grad_steps,
                             max_effective_bc, coverage_pass, etc.).
    - `metrics.jsonl`      : JSON estricto (`allow_nan=False`) con una
                             linea por step. Bajo AMP, si grad_norm fue
                             no finito, se persiste `grad_norm=null` y
                             se etiqueta con `grad_norm_nonfinite_kind`
                             (`'inf'`|`'nan'`), `amp_nonfinite_grad=True`,
                             `optimizer_applied=False`.
    - `sampling_plan.csv`  : copia del plan usado en el run.

Checkpoints: `paths.checkpoint_dir/<run_name>/ckpt_step{N}.pt` con
`model_state_dict`, `optimizer_state_dict`, `scheduler_state_dict`,
`config`, `step`, `git_hash`, `config_hash`, `param_count`,
`model_class`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


# Permitir ejecucion como `python -m training.train_ssl_central` y como
# script directo desde la raiz del repo.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from training.sampling import (
    compute_pretraining_sampling_plan,
    load_sources,
    write_sampling_plan_csv,
)


# ----------------------------------------------------------------------
# Utilidades: git info + cargar config
# ----------------------------------------------------------------------


def get_git_info(repo_root: Path) -> Dict[str, Any]:
    """Devuelve git hash y dirty flag. No falla si no es repo git."""
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=str(repo_root), stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        h = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(status)
    except Exception:
        dirty = False
    return {"git_hash": h, "git_dirty": dirty}


def load_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: Dict[str, Any]) -> str:
    """Hash SHA256 (16 hex) del YAML efectivo serializado de forma estable.

    Util para trazabilidad: dos runs con la misma config producen el mismo
    hash. Usamos `sort_keys=True` para que el orden no afecte.
    """
    blob = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ----------------------------------------------------------------------
# Helpers de logging
# ----------------------------------------------------------------------


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _json_safe(obj):
    """Convierte un valor a una forma JSON estricta (sin NaN/Infinity).

    Reglas:
      - float NaN / inf / -inf -> None (JSON-null).
      - numpy floats / ints -> tipos Python nativos; si no son finitos, None.
      - torch tensors escalares -> idem.
      - Path -> str.
      - dict / list / tuple -> recursivo.
      - cualquier otro tipo no serializable estricto -> str(obj) como fallback
        defensivo (no deberia ocurrir en el logging del training).

    Se usa antes de `json.dumps(..., allow_nan=False)` para que el fichero
    `metrics.jsonl` sea siempre JSON estandar (parseable por cualquier
    consumidor estricto). El motivo: bajo AMP, `grad_norm` puede ser inf
    en algunos steps, y dejar literal "Infinity" en el .jsonl viola el
    estandar (RFC 8259), aunque Python lo tolere.
    """
    import math
    # Path al pasar por isinstance debe convertirse a string.
    if isinstance(obj, Path):
        return str(obj)
    # bool antes que int (bool es subclase de int en Python).
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, (int,)):
        return obj
    if isinstance(obj, float):
        if math.isfinite(obj):
            return obj
        return None
    # numpy: import perezoso para no forzar dependencia si no esta cargado.
    try:
        import numpy as _np  # type: ignore
        if isinstance(obj, _np.bool_):
            return bool(obj)
        if isinstance(obj, _np.integer):
            return int(obj)
        if isinstance(obj, _np.floating):
            v = float(obj)
            return v if math.isfinite(v) else None
        if isinstance(obj, _np.ndarray):
            return [_json_safe(x) for x in obj.tolist()]
    except Exception:
        pass
    # torch (opcional): si el objeto es un tensor escalar, intentamos extraerlo.
    try:
        import torch as _torch  # type: ignore
        if isinstance(obj, _torch.Tensor):
            if obj.ndim == 0:
                v = obj.item()
                if isinstance(v, float):
                    return v if math.isfinite(v) else None
                return v
            return [_json_safe(x) for x in obj.tolist()]
    except Exception:
        pass
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    if obj is None or isinstance(obj, str):
        return obj
    # Fallback defensivo.
    return str(obj)


class JsonlLogger:
    """Logger JSONL estricto: cada linea es JSON valido sin NaN/Infinity.

    Usa `_json_safe` para limpiar el record antes de serializar con
    `allow_nan=False`. Si tras la limpieza algun valor sigue siendo no
    serializable, `json.dumps` lanzara TypeError, lo cual es lo que
    queremos: forzamos detectar inputs malformados.
    """

    def __init__(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self.f = open(path, "a", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        record = dict(record)
        record.setdefault("ts", _ts())
        safe = _json_safe(record)
        self.f.write(
            json.dumps(safe, ensure_ascii=False, allow_nan=False) + "\n"
        )
        self.f.flush()

    def close(self) -> None:
        try:
            self.f.close()
        except Exception:
            pass


# ----------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------


def cmd_dry_run(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Ejecuta el modo dry-run: solo lectura, plan y verificacion de paths."""
    print(f"[{_ts()}] === DRY-RUN === run={cfg.get('run_name')}")
    print(f"  config_hash:       {config_hash(cfg)}")
    paths = cfg.get("paths", {})
    proc, cli, asum = load_sources(
        repo_root / "results/processed_summary.csv",
        repo_root / "results/client_summary.csv",
        repo_root / "results/audit/audit_summary.json",
    )
    print(f"  processed_summary: {len(proc)} filas")
    print(f"  audit_version:     {asum.get('audit_version')}")

    # PRETRAIN_SOURCE filter
    ps = [r for r in proc if r["role"] == "PRETRAIN_SOURCE"]
    tt = [r for r in proc if r["role"] == "TRANSFER_TARGET"]
    print(f"  PS={len(ps)}  TT={len(tt)}")
    assert len(ps) == 36 and len(tt) == 11, (
        f"Esperado 36 PS + 11 TT; obtenido {len(ps)} PS + {len(tt)} TT"
    )

    plan = compute_pretraining_sampling_plan(
        proc, cli, asum,
        steps_per_epoch=cfg.get("training", {}).get("max_steps"),
        min_dataset_presence=float(
            cfg.get("data", {}).get("min_dataset_presence", 0.0)
        ),
    )
    assert len(plan) == 36, f"Plan tiene {len(plan)} datasets, esperado 36"

    # Invariantes del plan
    sum_ds = sum(p["final_dataset_weight"] for p in plan)
    assert abs(sum_ds - 1.0) < 1e-6, f"sum(final_dataset_weight)={sum_ds}"
    max_ds = max(p["final_dataset_weight"] for p in plan)
    assert max_ds <= 0.10 + 1e-6, f"max_ds={max_ds} > cap 0.10"
    # max client
    seen_c = set(); max_cl = 0.0
    for p in plan:
        if p["client"] not in seen_c:
            seen_c.add(p["client"])
            max_cl = max(max_cl, p["final_client_weight"])
    assert max_cl <= 0.25 + 1e-6, f"max_cl={max_cl} > cap 0.25"
    cnc = next(p["final_client_weight"] for p in plan if p["client"] == "cnc_milling")
    # Tolerancia: cuando se activa `min_dataset_presence`, la redistribucion
    # proporcional desde "above" toma una fraccion muy pequena tambien de
    # los datasets en el piso de cliente (NMILL en este corpus), bajando
    # cnc_milling unos pocos puntos por debajo de 0.005. Es un efecto
    # esperado del groupby `final_client_weight = sum(final_dataset_weight)`
    # tras el piso por dataset. Aceptamos hasta 1% de desviacion relativa.
    assert cnc >= 0.005 * 0.99, f"cnc_milling={cnc} < 0.005 (tol 1%)"
    # Ningun TT en el plan
    tt_names = {r["dataset"] for r in tt}
    assert tt_names.isdisjoint({p["dataset"] for p in plan})

    print("\nTop 10 datasets por final_dataset_weight:")
    for p in sorted(plan, key=lambda r: -r["final_dataset_weight"])[:10]:
        flags = []
        if p["capped_dataset"]: flags.append("cap_ds")
        if p["capped_client"]:  flags.append("cap_cl")
        if p["min_presence_applied"]: flags.append("min")
        fs = "[" + ",".join(flags) + "]" if flags else ""
        print(
            f"  {p['dataset']:<13}{p['client']:<18}"
            f"raw={p['raw_dataset_weight']:.4f}  "
            f"final={p['final_dataset_weight']:.4f}  {fs}"
        )

    # Verificar rutas de shards de TODOS los 36 PS
    processed_root = Path(paths.get("processed_root", "/content/drive/MyDrive/fm_fl_phmd/processed"))
    print(f"\nVerificando rutas en {processed_root} (los 36 PS)")
    missing = []
    found = 0
    for p in plan:
        d = processed_root / p["dataset"] / cfg["data"]["split"]
        if d.is_dir():
            tars = list(d.glob(f"{p['dataset']}-{cfg['data']['split']}-*.tar"))
            if tars:
                found += 1
            else:
                missing.append(p["dataset"])
        else:
            missing.append(p["dataset"])
    print(f"  shards encontrados: {found}/36  |  missing: {len(missing)}")
    if missing:
        # Mostrar los primeros 8 para diagnostico
        head = missing[:8]
        print(f"  primeros missing: {head}{' ...' if len(missing) > 8 else ''}")

    out = repo_root / paths.get("sampling_plan_out", "results/pretraining/ssl_sampling_plan.csv")
    write_sampling_plan_csv(plan, out)
    print(f"\nEscrito plan: {out}")

    # Si la config define batch_size_policy='adaptive_by_channels', mostrar
    # el batch_size efectivo por dataset (top-5 mas anchos = mayor riesgo VRAM).
    data_cfg = cfg.get("data", {})
    if data_cfg.get("batch_size_policy") == "adaptive_by_channels":
        from training.sampling import compute_adaptive_batch_size
        batch_size_cfg = int(data_cfg.get("batch_size", 32))
        max_cb = data_cfg.get("max_channel_batch")
        min_bs = int(data_cfg.get("min_batch_size", 1))
        plan_with_bs = []
        for p in plan:
            nc = int(p.get("n_channels", 0))
            if nc <= 0:
                plan_with_bs.append((p["dataset"], nc, batch_size_cfg, 0))
                continue
            b_eff = compute_adaptive_batch_size(
                n_channels=nc, batch_size=batch_size_cfg,
                max_channel_batch=max_cb, min_batch_size=min_bs,
            )
            plan_with_bs.append((p["dataset"], nc, b_eff, b_eff * nc))
        print(
            f"\nBatch_size adaptativo (batch_size={batch_size_cfg}, "
            f"max_channel_batch={max_cb}, min_batch_size={min_bs}):"
        )
        # Top 5 mas anchos (mayor C)
        top_c = sorted(plan_with_bs, key=lambda r: -r[1])[:5]
        print("  Top 5 mas anchos (mayor C):")
        for ds, nc, b_eff, eff_bc in top_c:
            print(f"    {ds:<13} C={nc:>3}  B_eff={b_eff:>3}  B*C={eff_bc}")
        # Top 5 mas estrechos (menor C)
        bot_c = sorted(plan_with_bs, key=lambda r: r[1])[:5]
        print("  Top 5 mas estrechos (menor C):")
        for ds, nc, b_eff, eff_bc in bot_c:
            print(f"    {ds:<13} C={nc:>3}  B_eff={b_eff:>3}  B*C={eff_bc}")
        # max B*C global
        max_bc = max(r[3] for r in plan_with_bs)
        print(f"  max(B*C) observado: {max_bc} (cap: {max_cb})")

    if missing:
        print(
            f"\nAVISO: {len(missing)} datasets sin shards (esperado si estas "
            f"fuera de Colab/Drive)."
        )

    print(f"\n[{_ts()}] === DRY-RUN OK ===")
    return 0


# ----------------------------------------------------------------------
# Smoke (requiere torch + shards reales)
# ----------------------------------------------------------------------


def _run_partial_padding_diagnostic(
    model,
    plan: List[Dict[str, Any]],
    processed_root: Path,
    device,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Diagnostico determinista de padding parcial.

    Busca en los `smoke_datasets` una muestra con `valid_time_mask.sum() < W`
    (ventana parcial), construye `partial_patch_mask` y fuerza un `ssl_mask`
    que enmascare exactamente esos patches parciales. Verifica que la loss
    es finita y que `padding_ignored_elements > 0`.

    Si no encuentra ninguna muestra parcial en los primeros samples
    inspeccionados, lanza `RuntimeError`: el smoke debe fallar antes de
    seguir, porque significa que el subset elegido no ejercita el contrato.
    """
    import numpy as np
    import torch

    from training.phm_tar_reader import find_shards, iter_samples_from_tar
    from training.ssl.loss import compute_masked_reconstruction_loss_with_metrics
    from training.ssl.masking import canonicalize_valid_patch_mask

    split = cfg["data"]["split"]
    # Limites generosos: si tras 200 samples no hay parcial, el subset es raro
    MAX_INSPECTED = 200

    chosen = None
    inspected = 0
    for row in plan:
        ds = str(row["dataset"])
        shards = find_shards(processed_root, ds, split)
        if not shards:
            continue
        for shard_path in shards:
            for sample in iter_samples_from_tar(shard_path, strict=True):
                inspected += 1
                vtm = sample["valid_time_mask"]
                W = int(vtm.shape[0])
                p_size = int(sample["meta"].get("patch_size", 16))
                if int(vtm.sum()) < W:
                    n_local = W // p_size
                    vsm = vtm.reshape(n_local, p_size)
                    partial = vsm.any(axis=-1) & ~vsm.all(axis=-1)
                    if bool(partial.any()):
                        chosen = (ds, sample, partial.astype(bool), n_local, p_size, W)
                        break
                if inspected >= MAX_INSPECTED:
                    break
            if chosen is not None or inspected >= MAX_INSPECTED:
                break
        if chosen is not None or inspected >= MAX_INSPECTED:
            break

    if chosen is None:
        raise RuntimeError(
            f"Diagnostico de padding parcial FALLO: tras inspeccionar "
            f"{inspected} samples en los smoke_datasets no se encontro "
            "ninguno con patch parcial. El smoke no puede validar el "
            "contrato tail_policy=pad. Revisa los datasets escogidos."
        )

    ds_name, sample, partial_np, n_local, p_size, W = chosen
    patches_np = sample["patches"]   # (C, N, P)
    vtm_np = sample["valid_time_mask"]
    vpm_np = sample["valid_patch_mask"]  # (C, N) bool

    # Construir batch B=1
    x = torch.from_numpy(patches_np).unsqueeze(0).to(device)
    vtm = torch.from_numpy(vtm_np).unsqueeze(0).to(device)
    vpm = torch.from_numpy(vpm_np).unsqueeze(0).to(device)
    B, C, N, P = x.shape

    # ssl_mask forzado: True exactamente donde partial es True, repetido por canal
    partial_t = torch.from_numpy(partial_np).to(device)        # (N,)
    ssl_mask = partial_t.view(1, 1, N).expand(B, C, N).contiguous()
    # Solo nos quedamos con los que son ademas valid_patch_mask=True
    vpm_canon = canonicalize_valid_patch_mask(vpm, B, C, N)
    ssl_mask = ssl_mask & vpm_canon

    n_forced = int(ssl_mask.sum().item())
    if n_forced == 0:
        raise RuntimeError(
            f"Diagnostico: el sample '{sample['__key__']}' tiene patches "
            "parciales pero ninguno es valido segun valid_patch_mask. "
            "Esto es contradictorio con la harmonization v0.5."
        )

    model.eval()
    with torch.no_grad():
        out = model(x, vtm, vpm_canon, ssl_mask)
        metrics = compute_masked_reconstruction_loss_with_metrics(
            pred=out["reconstruction"],
            target=x,
            ssl_mask=ssl_mask,
            valid_time_mask=vtm,
            valid_patch_mask=vpm_canon,
            loss_fn=cfg["ssl"].get("loss", "mse"),
        )
    model.train()

    loss_v = float(metrics["loss"].item())
    n_loss = int(metrics["n_loss_elements"].item())
    pad_ig = int(metrics["padding_ignored_elements"].item())
    ok = (
        torch.isfinite(metrics["loss"]).item()
        and n_loss > 0
        and pad_ig > 0
    )
    return {
        "result": "pass" if ok else "fail",
        "dataset": ds_name,
        "sample_key": sample["__key__"],
        "n_inspected": inspected,
        "n_partial_patches_in_sample": int(partial_np.sum()),
        "n_forced_masked": n_forced,
        "loss": loss_v,
        "n_loss_elements": n_loss,
        "padding_ignored_elements": pad_ig,
    }


def cmd_smoke(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Smoke training estricto: pocos steps, datasets pequenos, fallo duro
    si algo no cumple el contrato."""
    import numpy as np
    import torch
    from torch.optim import AdamW

    from models.patchtst_phm import build_patchtst_phm, count_parameters
    from training.phm_webdataset import build_centralized_loader
    from training.ssl.loss import compute_masked_reconstruction_loss_with_metrics
    from training.ssl.masking import canonicalize_valid_patch_mask, generate_ssl_mask

    print(f"[{_ts()}] === SMOKE === run={cfg.get('run_name')}")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    smoke_names = cfg["data"]["smoke_datasets"]
    smoke_strategy = cfg["data"].get("smoke_sampling_strategy", "round_robin")
    if smoke_strategy not in ("round_robin", "uniform", "weighted"):
        raise ValueError(f"smoke_sampling_strategy desconocida: {smoke_strategy}")

    proc, cli, asum = load_sources(
        repo_root / "results/processed_summary.csv",
        repo_root / "results/client_summary.csv",
        repo_root / "results/audit/audit_summary.json",
    )
    plan_full = compute_pretraining_sampling_plan(proc, cli, asum)
    plan = [p for p in plan_full if p["dataset"] in smoke_names]
    if not plan:
        raise RuntimeError(
            f"Ninguno de los smoke_datasets {smoke_names} es PRETRAIN_SOURCE"
        )

    # Verificacion explicita: ningun TT entre los smoke_datasets
    for p in plan:
        ds_row = next(r for r in proc if r["dataset"] == p["dataset"])
        if ds_row["role"] != "PRETRAIN_SOURCE":
            raise RuntimeError(f"{p['dataset']} es {ds_row['role']}, no PS")

    # Renormalizar pesos solo si la estrategia es 'weighted' (informativo)
    s = sum(p["final_dataset_weight"] for p in plan)
    if s > 0:
        for p in plan:
            p["final_dataset_weight"] = p["final_dataset_weight"] / s
    print(f"\nSmoke datasets ({len(plan)}), strategy={smoke_strategy}:")
    for p in plan:
        print(f"  {p['dataset']:<13}{p['client']:<18}w_norm={p['final_dataset_weight']:.4f}")

    paths = cfg["paths"]
    log_dir = Path(paths["log_dir"]) / cfg["run_name"]
    ckpt_dir = Path(paths["checkpoint_dir"]) / cfg["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info(repo_root)
    cfg_hash = config_hash(cfg)
    write_sampling_plan_csv(plan, log_dir / "sampling_plan.csv")
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = build_patchtst_phm(cfg["model"]).to(device)
    n_params = count_parameters(model)
    print(f"Modelo: {cfg['model'].get('name')}  ({n_params:,} parametros)")

    # Diagnostico determinista de padding parcial ANTES del loop. Si falla,
    # detenemos el smoke con error claro.
    print(f"\n[{_ts()}] Diagnostico determinista de padding parcial...")
    processed_root = Path(paths["processed_root"])
    try:
        diag = _run_partial_padding_diagnostic(
            model=model, plan=plan, processed_root=processed_root,
            device=device, cfg=cfg,
        )
    except Exception as e:
        # Persistimos un run_info parcial antes de fallar para trazabilidad
        run_info_fail = {
            "ts": _ts(), "mode": "smoke", "run_name": cfg["run_name"],
            "seed": seed, "git_hash": git_info["git_hash"],
            "git_dirty": git_info["git_dirty"], "config_hash": cfg_hash,
            "param_count": n_params, "smoke_sampling_strategy": smoke_strategy,
            "datasets": [p["dataset"] for p in plan],
            "partial_padding_diagnostic": {"result": "fail", "error": str(e)},
        }
        (log_dir / "run_info.json").write_text(
            json.dumps(run_info_fail, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"[{_ts()}] DIAGNOSTICO PADDING PARCIAL FALLO: {e}")
        return 2
    print(
        f"  -> {diag['result'].upper()}  ds={diag['dataset']}  "
        f"sample={diag['sample_key']}  loss={diag['loss']:.4f}  "
        f"pad_ignored={diag['padding_ignored_elements']}"
    )

    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg["training"]["lr"]),
        weight_decay=float(cfg["training"]["weight_decay"]),
    )
    amp_cfg = cfg["training"].get("amp", "auto")
    use_amp = (amp_cfg in ("auto", True)) and device.type == "cuda"
    # API nueva (PyTorch >=2.0): torch.amp.* en lugar de torch.cuda.amp.*
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP: {use_amp}")

    max_steps = int(cfg["training"]["max_steps"])
    batch_size = int(cfg["data"]["batch_size"])
    loader = build_centralized_loader(
        plan=plan,
        processed_root=processed_root,
        batch_size=batch_size,
        split=cfg["data"]["split"],
        seed=seed,
        max_steps=max_steps,
        strategy=smoke_strategy,
    )

    logger = JsonlLogger(log_dir / "metrics.jsonl")
    log_every = int(cfg["training"]["log_every"])
    ckpt_every = int(cfg["training"]["checkpoint_every"])
    grad_clip = float(cfg["training"]["grad_clip_norm"])
    mask_ratio = float(cfg["ssl"]["mask_ratio"])

    # Contadores
    datasets_seen = Counter()
    clients_seen = Counter()
    optimizer_steps = 0
    skipped_steps = 0
    any_padding_ignored = False

    model.train()
    step = 0
    t0 = time.time()
    for batch in loader:
        step += 1
        x = batch["patches"].to(device, non_blocking=True)
        vtm = batch["valid_time_mask"].to(device, non_blocking=True)
        vpm = batch["valid_patch_mask"].to(device, non_blocking=True)
        B, C, N, P = x.shape

        vpm_canon = canonicalize_valid_patch_mask(vpm, B, C, N)
        ssl_mask = generate_ssl_mask(vpm_canon, mask_ratio=mask_ratio)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(x, vtm, vpm_canon, ssl_mask)
                metrics = compute_masked_reconstruction_loss_with_metrics(
                    pred=out["reconstruction"], target=x, ssl_mask=ssl_mask,
                    valid_time_mask=vtm, valid_patch_mask=vpm_canon,
                    loss_fn=cfg["ssl"].get("loss", "mse"),
                )
                loss = metrics["loss"]
        else:
            out = model(x, vtm, vpm_canon, ssl_mask)
            metrics = compute_masked_reconstruction_loss_with_metrics(
                pred=out["reconstruction"], target=x, ssl_mask=ssl_mask,
                valid_time_mask=vtm, valid_patch_mask=vpm_canon,
                loss_fn=cfg["ssl"].get("loss", "mse"),
            )
            loss = metrics["loss"]

        if not torch.isfinite(loss):
            # Smoke estricto: fallar inmediatamente. No continuamos en silencio.
            raise RuntimeError(
                f"smoke step {step}: loss no finita "
                f"(dataset={batch.get('__dataset__')}). Abortando smoke."
            )
        if int(metrics["n_loss_elements"].item()) == 0:
            skipped_steps += 1
            continue

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        optimizer_steps += 1
        datasets_seen[batch.get("__dataset__")] += 1
        if batch.get("__client__"):
            clients_seen[batch["__client__"]] += 1
        if int(metrics["padding_ignored_elements"].item()) > 0:
            any_padding_ignored = True

        if step % log_every == 0 or step == 1:
            print(
                f"  step {step:>4d}/{max_steps}  ds={batch.get('__dataset__'):<13}"
                f"  loss={loss.item():.4f}  eff_mask={metrics['effective_mask_ratio'].item():.3f}"
                f"  pad_ignored={int(metrics['padding_ignored_elements'].item())}"
                f"  gradn={float(grad_norm):.3f}"
            )
        logger.log({
            "step": step,
            "dataset": batch.get("__dataset__"),
            "client": batch.get("__client__"),
            "loss": float(loss.item()),
            "n_loss_elements": int(metrics["n_loss_elements"].item()),
            "n_masked_patches": int(metrics["n_masked_patches"].item()),
            "n_valid_patches": int(metrics["n_valid_patches"].item()),
            "effective_mask_ratio": float(metrics["effective_mask_ratio"].item()),
            "padding_ignored_elements": int(metrics["padding_ignored_elements"].item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": float(grad_norm),
            "batch_shape": list(x.shape),
        })

        if step % ckpt_every == 0 or step == max_steps:
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": cfg,
                "git_hash": git_info["git_hash"],
                "config_hash": cfg_hash,
                "model_class": "PatchTSTPhm",
                "param_count": n_params,
            }
            ckpt_path = ckpt_dir / f"ckpt_step{step:06d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    elapsed = time.time() - t0
    logger.close()

    # Validaciones finales del smoke
    expected_datasets = set(smoke_names)
    actual_datasets = set(datasets_seen.keys())
    smoke_pass = (
        actual_datasets == expected_datasets
        and optimizer_steps > 0
        and diag["result"] == "pass"
    )

    run_info = {
        "ts": _ts(),
        "mode": "smoke",
        "run_name": cfg["run_name"],
        "seed": seed,
        "git_hash": git_info["git_hash"],
        "git_dirty": git_info["git_dirty"],
        "config_hash": cfg_hash,
        "param_count": n_params,
        "smoke_sampling_strategy": smoke_strategy,
        "datasets": [p["dataset"] for p in plan],
        "clients": sorted({p["client"] for p in plan}),
        "model": cfg["model"],
        "ssl": cfg["ssl"],
        "data": cfg["data"],
        "training": cfg["training"],
        "datasets_seen": dict(datasets_seen),
        "clients_seen": dict(clients_seen),
        "steps_per_dataset": dict(datasets_seen),
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "any_padding_ignored_elements": any_padding_ignored,
        "partial_padding_diagnostic": diag,
        "smoke_pass": smoke_pass,
        "elapsed_seconds": round(elapsed, 1),
    }
    (log_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[{_ts()}] === SMOKE end === {step} steps en {elapsed:.1f}s")
    print(f"  optimizer_steps : {optimizer_steps}")
    print(f"  skipped_steps   : {skipped_steps}")
    print(f"  datasets_seen   : {dict(datasets_seen)}")
    print(f"  clients_seen    : {dict(clients_seen)}")
    print(f"  any_padding_ignored: {any_padding_ignored}")
    print(f"  partial_padding_diag: {diag['result']}")
    print(f"Logs: {log_dir}")
    print(f"Ckpts: {ckpt_dir}")
    if not smoke_pass:
        print(
            f"\nSMOKE FAIL: datasets vistos {actual_datasets} != esperados "
            f"{expected_datasets}, o optimizer_steps={optimizer_steps}, "
            f"o diag={diag['result']}."
        )
        return 3
    print("SMOKE PASS")
    return 0


# ----------------------------------------------------------------------
# Train (no implementado todavia, simbolico)
# ----------------------------------------------------------------------


def _validate_train_config(cfg: Dict[str, Any]) -> None:
    """Guardas duras antes de permitir cmd_train.

    Acepta stages: 'coverage', 'pilot', 'full'.
    Caps de max_steps por stage:
      - coverage / pilot: <= 10 000
      - full: <= 500 000 (suficiente para A100 24h con margen)
    """
    stage = cfg.get("stage")
    if stage not in ("coverage", "pilot", "full"):
        raise RuntimeError(
            f"cmd_train solo acepta configs con stage in "
            f"{{coverage, pilot, full}}; recibido stage={stage!r}."
        )
    max_steps = int(cfg.get("training", {}).get("max_steps", 0))
    if max_steps <= 0:
        raise RuntimeError(f"max_steps={max_steps}: debe ser > 0.")
    cap_by_stage = {"coverage": 10000, "pilot": 10000, "full": 500000}
    cap = cap_by_stage[stage]
    if max_steps > cap:
        raise RuntimeError(
            f"max_steps={max_steps} > cap_{stage}={cap}. Para mas, considerar "
            "particionar el entrenamiento en runs con resume."
        )
    role = cfg.get("data", {}).get("role")
    if role != "PRETRAIN_SOURCE":
        raise RuntimeError(
            f"data.role debe ser 'PRETRAIN_SOURCE'; recibido {role!r}."
        )


def _build_lr_scheduler(optimizer, training_cfg: Dict[str, Any]):
    """Devuelve un scheduler segun training.schedule. None si schedule=constant."""
    import torch
    schedule = str(training_cfg.get("schedule", "constant"))
    warmup_steps = int(training_cfg.get("warmup_steps", 0))
    max_steps = int(training_cfg["max_steps"])
    if schedule == "constant" or warmup_steps <= 0:
        return None

    def lr_lambda(step: int) -> float:
        # linear warmup
        if step < warmup_steps:
            return float(step + 1) / float(max(1, warmup_steps))
        if schedule == "cosine":
            import math
            progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
            progress = max(0.0, min(1.0, progress))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def cmd_train(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Pretraining central. Acepta stages 'coverage', 'pilot' y 'full'.

    Stages soportados (con caps en `_validate_train_config`):
      - 'coverage' (cap 10k): valida compatibilidad VRAM y cobertura
        sobre los 36 PS con `sampling_strategy='round_robin'`.
      - 'pilot' (cap 10k): version corta de la politica productiva
        (`weighted` + caps + scheduler) con reporte periodico de
        distribucion empirica.
      - 'full' (cap 500k): pretraining productivo real sobre los 36 PS
        con weighted + caps + `min_dataset_presence` opcional.

    Diferencias con `cmd_smoke`:
      - Lee todos los 36 PRETRAIN_SOURCE del processed_summary (no un subset).
      - Estrategia de muestreo configurable via `data.sampling_strategy`.
      - Batch size adaptativo por dataset segun `data.batch_size_policy`.
      - Scheduler linear warmup + cosine si `training.schedule='cosine'`.
      - Resume desde checkpoint via `training.resume_from`.
      - Distribucion empirica observada vs esperada cada
        `training.distribution_log_every` steps.
      - Bajo AMP: tolera `grad_norm` no finito (gestionado por GradScaler);
        sin AMP: aborta. Loss no finita: aborta siempre.
      - Logging JSONL estricto (`allow_nan=False`): los `grad_norm` no
        finitos quedan como `null` con `grad_norm_nonfinite_kind` para
        trazabilidad post-hoc.
    """
    import numpy as np
    import torch
    from torch.optim import AdamW

    from models.patchtst_phm import build_patchtst_phm, count_parameters
    from training.phm_webdataset import build_centralized_loader
    from training.ssl.loss import compute_masked_reconstruction_loss_with_metrics
    from training.ssl.masking import canonicalize_valid_patch_mask, generate_ssl_mask

    _validate_train_config(cfg)
    stage = cfg["stage"]
    print(f"[{_ts()}] === TRAIN ({stage.upper()}) === run={cfg.get('run_name')}")

    seed = int(cfg.get("seed", 42))
    torch.manual_seed(seed)
    np.random.seed(seed)

    proc, cli, asum = load_sources(
        repo_root / "results/processed_summary.csv",
        repo_root / "results/client_summary.csv",
        repo_root / "results/audit/audit_summary.json",
    )
    plan = compute_pretraining_sampling_plan(
        proc, cli, asum,
        min_dataset_presence=float(
            cfg.get("data", {}).get("min_dataset_presence", 0.0)
        ),
    )
    assert len(plan) == 36, f"Plan tiene {len(plan)} datasets, esperado 36"
    tt_names = {r["dataset"] for r in proc if r["role"] == "TRANSFER_TARGET"}
    assert tt_names.isdisjoint({p["dataset"] for p in plan})

    print(f"\nDatasets ({len(plan)} PRETRAIN_SOURCE):")
    for p in plan[:5]:
        print(
            f"  {p['dataset']:<13}{p['client']:<18}"
            f"C={p['n_channels']:>3}  w={p['final_dataset_weight']:.4f}"
        )
    print(f"  ... ({len(plan)-5} mas)")

    paths = cfg["paths"]
    log_dir = Path(paths["log_dir"]) / cfg["run_name"]
    ckpt_dir = Path(paths["checkpoint_dir"]) / cfg["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    git_info = get_git_info(repo_root)
    cfg_hash = config_hash(cfg)
    write_sampling_plan_csv(plan, log_dir / "sampling_plan.csv")
    (log_dir / "config.yaml").write_text(
        yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding="utf-8"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    model = build_patchtst_phm(cfg["model"]).to(device)
    n_params = count_parameters(model)
    print(f"Modelo: {cfg['model'].get('name')}  ({n_params:,} parametros)")

    # Optimizer + scheduler
    training_cfg = cfg["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_cfg["lr"]),
        weight_decay=float(training_cfg["weight_decay"]),
    )
    scheduler = _build_lr_scheduler(optimizer, training_cfg)
    amp_cfg = training_cfg.get("amp", "auto")
    use_amp = (amp_cfg in ("auto", True)) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    print(f"AMP: {use_amp}")

    # Resume opcional
    resume_from = training_cfg.get("resume_from")
    start_step = 0
    if resume_from:
        ckpt = torch.load(resume_from, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        print(f"Reanudado desde {resume_from} (step={start_step})")

    # DataLoader con batch adaptativo
    data_cfg = cfg["data"]
    processed_root = Path(paths["processed_root"])
    loader = build_centralized_loader(
        plan=plan,
        processed_root=processed_root,
        batch_size=int(data_cfg["batch_size"]),
        split=data_cfg["split"],
        seed=seed + start_step,
        max_steps=int(training_cfg["max_steps"]) - start_step,
        strategy=str(data_cfg.get("sampling_strategy", "weighted")),
        batch_size_policy=str(data_cfg.get("batch_size_policy", "fixed")),
        max_channel_batch=data_cfg.get("max_channel_batch"),
        min_batch_size=int(data_cfg.get("min_batch_size", 1)),
    )

    logger = JsonlLogger(log_dir / "metrics.jsonl")
    log_every = int(training_cfg["log_every"])
    ckpt_every = int(training_cfg["checkpoint_every"])
    distribution_log_every = int(training_cfg.get("distribution_log_every", 0) or 0)
    grad_clip = float(training_cfg["grad_clip_norm"])
    mask_ratio = float(cfg["ssl"]["mask_ratio"])

    datasets_seen = Counter()
    clients_seen = Counter()
    optimizer_steps = 0
    skipped_steps = 0
    amp_overflow_steps = 0          # alias historico (compat con run_info anterior)
    amp_nonfinite_grad_steps = 0    # nombre preciso: steps con grad_norm no finito bajo AMP
    any_padding_ignored = False
    max_effective_bc = 0
    require_all_seen = bool(data_cfg.get("require_all_datasets_seen", False))

    model.train()
    step = start_step
    t0 = time.time()
    for batch in loader:
        step += 1
        x = batch["patches"].to(device, non_blocking=True)
        vtm = batch["valid_time_mask"].to(device, non_blocking=True)
        vpm = batch["valid_patch_mask"].to(device, non_blocking=True)
        B, C, N, P = x.shape

        vpm_canon = canonicalize_valid_patch_mask(vpm, B, C, N)
        ssl_mask = generate_ssl_mask(vpm_canon, mask_ratio=mask_ratio)

        optimizer.zero_grad(set_to_none=True)
        if use_amp:
            with torch.amp.autocast("cuda"):
                out = model(x, vtm, vpm_canon, ssl_mask)
                metrics = compute_masked_reconstruction_loss_with_metrics(
                    pred=out["reconstruction"], target=x, ssl_mask=ssl_mask,
                    valid_time_mask=vtm, valid_patch_mask=vpm_canon,
                    loss_fn=cfg["ssl"].get("loss", "mse"),
                )
                loss = metrics["loss"]
        else:
            out = model(x, vtm, vpm_canon, ssl_mask)
            metrics = compute_masked_reconstruction_loss_with_metrics(
                pred=out["reconstruction"], target=x, ssl_mask=ssl_mask,
                valid_time_mask=vtm, valid_patch_mask=vpm_canon,
                loss_fn=cfg["ssl"].get("loss", "mse"),
            )
            loss = metrics["loss"]

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"train step {step}: loss no finita "
                f"(dataset={batch.get('__dataset__')}). Abortando."
            )
        if int(metrics["n_loss_elements"].item()) == 0:
            # coverage es estricto; pilot tolera pero registra
            skipped_steps += 1
            if stage == "coverage":
                raise RuntimeError(
                    f"coverage step {step}: 0 elementos validos en "
                    f"dataset={batch.get('__dataset__')}. Abortando."
                )
            continue

        amp_overflow_this_step = False
        grad_norm_nonfinite_kind = None  # 'inf' | 'nan' | None
        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Con AMP, grad_norm=inf es comportamiento normal: significa
            # overflow en fp16. El GradScaler detecta el overflow, NO aplica
            # el optimizer.step() y reduce el scale automaticamente. No es
            # divergencia del modelo. Lo contamos y seguimos.
            if not torch.isfinite(grad_norm):
                amp_overflow_this_step = True
                gv = float(grad_norm)
                # math.isnan vs math.isinf para distinguir el tipo de overflow.
                import math as _math
                if _math.isnan(gv):
                    grad_norm_nonfinite_kind = "nan"
                else:
                    grad_norm_nonfinite_kind = "inf"
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # Sin AMP, grad_norm=inf si es divergencia real. Abortamos.
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"train step {step}: grad_norm no finito ({grad_norm}) sin AMP. Abortando."
                )
            optimizer.step()
        optimizer_applied = not amp_overflow_this_step
        if scheduler is not None:
            # Solo avanzamos el scheduler si el step realmente se aplico
            # (en AMP+overflow, scaler.step() es no-op y no consume tasa).
            if optimizer_applied:
                scheduler.step()
        # Contadores: si hubo overflow AMP, el step no se aplico
        if amp_overflow_this_step:
            amp_overflow_steps += 1
            amp_nonfinite_grad_steps += 1
        else:
            optimizer_steps += 1
            datasets_seen[batch.get("__dataset__")] += 1
            if batch.get("__client__"):
                clients_seen[batch["__client__"]] += 1
        eff_bc = int(batch.get("__effective_bc__", B * C))
        if eff_bc > max_effective_bc:
            max_effective_bc = eff_bc
        if int(metrics["padding_ignored_elements"].item()) > 0:
            any_padding_ignored = True

        if step % log_every == 0 or step == start_step + 1:
            print(
                f"  step {step:>5d}/{training_cfg['max_steps']}  "
                f"ds={batch.get('__dataset__'):<13} C={C:>3} B={B:>2} "
                f"loss={loss.item():.4f}  eff_mask={metrics['effective_mask_ratio'].item():.3f}"
                f"  pad_ig={int(metrics['padding_ignored_elements'].item())}"
                f"  gradn={float(grad_norm):.3f}"
            )
        # JsonlLogger usa _json_safe + allow_nan=False: si grad_norm es no
        # finito, lo pasamos como None explicito y registramos su tipo en
        # `grad_norm_nonfinite_kind` para trazabilidad post-hoc.
        gn_val = float(grad_norm)
        gn_finite = bool(torch.isfinite(grad_norm).item())
        logger.log({
            "step": step,
            "dataset": batch.get("__dataset__"),
            "client": batch.get("__client__"),
            "n_channels": batch.get("__n_channels__"),
            "batch_size_effective": batch.get("__batch_size_effective__"),
            "effective_bc": eff_bc,
            "loss": float(loss.item()),
            "n_loss_elements": int(metrics["n_loss_elements"].item()),
            "n_masked_patches": int(metrics["n_masked_patches"].item()),
            "n_valid_patches": int(metrics["n_valid_patches"].item()),
            "effective_mask_ratio": float(metrics["effective_mask_ratio"].item()),
            "padding_ignored_elements": int(metrics["padding_ignored_elements"].item()),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "grad_norm": gn_val if gn_finite else None,
            "grad_norm_is_finite": gn_finite,
            "grad_norm_nonfinite_kind": grad_norm_nonfinite_kind,
            "amp_nonfinite_grad": bool(amp_overflow_this_step),
            "optimizer_applied": bool(optimizer_applied),
            "batch_shape": list(x.shape),
        })

        # Distribucion empirica vs esperada (solo pilot, normalmente)
        if distribution_log_every > 0 and step % distribution_log_every == 0:
            total = sum(datasets_seen.values())
            if total > 0:
                expected = {p["dataset"]: float(p["final_dataset_weight"]) for p in plan}
                expected_cli = {}
                for p in plan:
                    expected_cli[p["client"]] = expected_cli.get(p["client"], 0.0) + float(p["final_dataset_weight"])
                obs_ds = {k: v / total for k, v in datasets_seen.items()}
                obs_cl = {k: v / total for k, v in clients_seen.items()}
                dist = {
                    "step": step,
                    "kind": "distribution",
                    "datasets_observed": obs_ds,
                    "datasets_expected": expected,
                    "datasets_abs_error": {k: abs(obs_ds.get(k, 0.0) - expected.get(k, 0.0)) for k in expected},
                    "clients_observed": obs_cl,
                    "clients_expected": expected_cli,
                    "clients_abs_error": {k: abs(obs_cl.get(k, 0.0) - expected_cli.get(k, 0.0)) for k in expected_cli},
                }
                logger.log(dist)

        if step % ckpt_every == 0 or step == int(training_cfg["max_steps"]):
            ckpt = {
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
                "config": cfg,
                "git_hash": git_info["git_hash"],
                "config_hash": cfg_hash,
                "model_class": "PatchTSTPhm",
                "param_count": n_params,
            }
            ckpt_path = ckpt_dir / f"ckpt_step{step:06d}.pt"
            torch.save(ckpt, ckpt_path)
            print(f"  checkpoint -> {ckpt_path}")

    elapsed = time.time() - t0
    logger.close()

    expected_full_ps = {p["dataset"] for p in plan}
    coverage_pass = expected_full_ps == set(datasets_seen.keys())
    pilot_pass = (stage == "pilot") and optimizer_steps > 0

    run_info = {
        "ts": _ts(),
        "mode": "train",
        "stage": stage,
        "run_name": cfg["run_name"],
        "seed": seed,
        "git_hash": git_info["git_hash"],
        "git_dirty": git_info["git_dirty"],
        "config_hash": cfg_hash,
        "param_count": n_params,
        "datasets": [p["dataset"] for p in plan],
        "clients": sorted({p["client"] for p in plan}),
        "model": cfg["model"],
        "ssl": cfg["ssl"],
        "data": cfg["data"],
        "training": cfg["training"],
        "datasets_seen": dict(datasets_seen),
        "clients_seen": dict(clients_seen),
        "steps_per_dataset": dict(datasets_seen),
        "optimizer_steps": optimizer_steps,
        "skipped_steps": skipped_steps,
        "amp_overflow_steps": amp_overflow_steps,           # alias historico
        "amp_nonfinite_grad_steps": amp_nonfinite_grad_steps,  # nombre preciso
        "any_padding_ignored_elements": any_padding_ignored,
        "max_effective_bc": max_effective_bc,
        "coverage_pass": coverage_pass,
        "pilot_pass": pilot_pass,
        "elapsed_seconds": round(elapsed, 1),
    }
    (log_dir / "run_info.json").write_text(
        json.dumps(run_info, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n[{_ts()}] === TRAIN ({stage.upper()}) end ===")
    print(f"  optimizer_steps : {optimizer_steps}")
    print(f"  skipped_steps   : {skipped_steps}")
    print(f"  datasets_seen   : {len(datasets_seen)}/{len(expected_full_ps)}")
    print(f"  max_effective_bc: {max_effective_bc}")
    print(f"Logs: {log_dir}")
    print(f"Ckpts: {ckpt_dir}")
    if stage == "coverage" and require_all_seen and not coverage_pass:
        missing = expected_full_ps - set(datasets_seen.keys())
        print(f"\nCOVERAGE FAIL: datasets no vistos: {sorted(missing)}")
        return 4
    if stage == "coverage":
        print("COVERAGE PASS")
    if stage == "pilot":
        print("PILOT end (revisar manualmente metrics.jsonl + distribucion observada)")
    return 0


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SSL pretraining centralizado (PatchTST channel-independent)"
    )
    parser.add_argument(
        "--mode", choices=("dry-run", "smoke", "train"), default="dry-run"
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Ruta al YAML de configuracion (ssl_smoke.yaml o ssl_central_base.yaml)"
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    repo_root = _REPO_ROOT
    if args.mode == "dry-run":
        return cmd_dry_run(cfg, repo_root)
    if args.mode == "smoke":
        return cmd_smoke(cfg, repo_root)
    if args.mode == "train":
        return cmd_train(cfg, repo_root)
    raise ValueError(f"modo desconocido: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
