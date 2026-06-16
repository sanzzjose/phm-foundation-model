"""Downstream Full v1: downstream COMPLETO sobre el checkpoint_bank_v1.

A diferencia del Probe Suite v1 (screening: pocas épocas, un solo
checkpoint, probe seed fija), esta suite ejecuta downstream con
presupuesto completo (más épocas, full fine-tuning real) sobre VARIOS
checkpoints del banco, para producir la comparación central-vs-federado
del TFM.

Diseño (Fase 6a):

* NO duplica trainers: invoca los mismos
  ``train_downstream_classification`` / ``train_downstream_rul`` a través
  de la costura :func:`_invoke_trainer` (idéntica firma a la del Probe
  Suite, lo que permite mockearla en tests sin torch).
* Lee la lista de checkpoints de la config versionada
  (``downstream_full_v1.yaml``) o, opcionalmente, de un
  ``checkpoint_bank_v1.json`` vía ``--checkpoint-bank``.
* Itera (checkpoint × dataset × modo). Los modos checkpoint-dependientes
  (``linear_probing`` / ``full_finetuning`` / ``linear`` / ``mlp_2layer``)
  se ejecutan una vez por checkpoint; ``from_scratch`` es independiente
  del checkpoint y se ejecuta UNA vez por dataset (dedup).
* Salta CALCE_CS2 (``needs_semantic_review``) y cualquier tarea cuyo rol
  no sea TRANSFER_TARGET/EXTERNAL_TARGET (nunca PRETRAIN_SOURCE).
* Selección por VALIDACIÓN; el test solo se reporta para trazabilidad.

Layout de salida::

    results/downstream/full_v1/<checkpoint_id>/<dataset>/<mode>/
        result_row.json
        summary.json
        run_info.json        (lo escribe el trainer)
    results/downstream/full_v1/downstream_full_v1_summary.{json,csv}

Modos del CLI: ``--mode {list,dry-run,run}`` y filtros ``--only-dataset``,
``--only-checkpoint``, ``--only-mode``, ``--max-tasks``, ``--skip-existing``.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from training.downstream import builders  # noqa: F401 (registra tasks)
from training.downstream.task_registry import TaskSpec, get_task, is_runnable
from training.experiments import (
    METRIC_DIRECTION,
    JsonlLogger,
    ResultRow,
    config_hash,
    get_git_info,
    json_safe,
    new_experiment_id,
    now_ts,
    write_result_row,
)


REPO_ROOT_DEFAULT = Path(__file__).resolve().parents[2]

# Backbone PatchTSTPhm base (801 808 params). Debe coincidir con el
# checkpoint SSL evaluado. Fijo para toda la suite.
MODEL_BASE: Dict[str, Any] = {
    "name": "patchtst_phm_base",
    "d_model": 128,
    "n_layers": 4,
    "n_heads": 4,
    "d_ff": 512,
    "dropout": 0.1,
    "patch_size": 16,
    "n_patches": 32,
}

DEFAULT_PROCESSED_ROOT = "/content/drive/MyDrive/fm_fl_phmd/processed"
DEFAULT_PROCESSED_DOWNSTREAM_ROOT = (
    "/content/drive/MyDrive/fm_fl_phmd/processed_downstream"
)

# Pseudo-checkpoint para from_scratch (baseline independiente del ckpt SSL).
FROM_SCRATCH_ID = "from_scratch"
FROM_SCRATCH_ORIGIN = "none"

ALLOWED_DOWNSTREAM_ROLES = ("TRANSFER_TARGET", "EXTERNAL_TARGET")


# ---------------------------------------------------------------------------
# Mapeo de modos del downstream completo -> (trainer_mode, overrides)
# ---------------------------------------------------------------------------

def _full_mode_spec(task_type: str, mode: str) -> Dict[str, Any]:
    """Traduce un modo del downstream completo a la receta del trainer.

    Soporta, además de los modos del probe, ``from_scratch`` y
    ``full_finetuning`` completos para ambos tipos de tarea.

    Lanza ``ValueError`` si el (task_type, mode) no está soportado.
    """
    if task_type == "classification":
        if mode == "from_scratch":
            return {"trainer_kind": "classification", "trainer_mode": "from_scratch",
                    "lr_backbone": None, "head": None, "checkpoint_required": False}
        if mode == "linear_probing":
            return {"trainer_kind": "classification", "trainer_mode": "linear_probing",
                    "lr_backbone": None, "head": None, "checkpoint_required": True}
        if mode == "full_finetuning":
            return {"trainer_kind": "classification", "trainer_mode": "full_finetuning",
                    "lr_backbone": 1e-5, "head": None, "checkpoint_required": True}
        raise ValueError(f"Modo no soportado para classification: {mode!r}")
    if task_type == "rul":
        linear_head = {"hidden_dim": None, "dropout": 0.0, "activation": None,
                       "keep_last_dim": False}
        mlp_head = {"hidden_dim": 256, "dropout": 0.1, "activation": "gelu",
                    "keep_last_dim": False}
        if mode == "from_scratch":
            return {"trainer_kind": "rul", "trainer_mode": "from_scratch",
                    "lr_backbone": None, "head": linear_head, "checkpoint_required": False}
        if mode == "linear":
            return {"trainer_kind": "rul", "trainer_mode": "linear_probing",
                    "lr_backbone": None, "head": linear_head, "checkpoint_required": True}
        if mode == "mlp_2layer":
            return {"trainer_kind": "rul", "trainer_mode": "linear_probing",
                    "lr_backbone": None, "head": mlp_head, "checkpoint_required": True}
        if mode == "full_finetuning":
            return {"trainer_kind": "rul", "trainer_mode": "full_finetuning",
                    "lr_backbone": 1e-5, "head": linear_head, "checkpoint_required": True}
        raise ValueError(f"Modo no soportado para rul: {mode!r}")
    raise ValueError(f"task_type no soportado: {task_type!r}")


def is_checkpoint_dependent(mode: str) -> bool:
    """``from_scratch`` no depende del checkpoint SSL; el resto sí."""
    return mode != "from_scratch"


def _resolve_lrs_and_amp(block: Dict[str, Any], mode: str, mode_spec: Dict[str, Any]):
    """Resuelve (lr_head, lr_backbone, amp) con overrides opcionales de config.

    Por defecto reproduce el comportamiento canónico:
      - lr_head = block.lr_head (1e-3).
      - lr_backbone = el del mode_spec (full_finetuning 1e-5; None en
        linear/from_scratch -> un solo grupo a lr_head).
      - amp = "auto".

    Overrides de ablación (solo si presentes en el bloque del YAML):
      - ``lr_head_from_scratch``: baja el LR de from_scratch (que entrena el
        backbone aleatorio a lr_head; 1e-3 puede divergir).
      - ``lr_backbone_full_finetuning``: ablación del LR del backbone en
        full_finetuning (p.ej. diagnosticar colapsos de transferencia).
      - ``amp``: "auto" | "off" (fp32 es más estable desde cero).
    """
    lr_head = float(block.get("lr_head", 1e-3))
    lr_backbone = mode_spec["lr_backbone"]
    if mode == "from_scratch" and block.get("lr_head_from_scratch") is not None:
        lr_head = float(block["lr_head_from_scratch"])
    if mode == "full_finetuning" and block.get("lr_backbone_full_finetuning") is not None:
        lr_backbone = float(block["lr_backbone_full_finetuning"])
    amp = str(block.get("amp", "auto"))
    return lr_head, lr_backbone, amp


def _apply_ctx_overrides(lr_head, lr_backbone, amp, mode: str, ctx: "FullContext"):
    """Aplica los overrides CLI de Fase 7a (precedencia sobre config).

    - ``override_lr_head`` afecta a cualquier modo (lr de la cabeza).
    - ``override_lr_backbone`` SOLO se aplica en ``full_finetuning`` (el
      backbone está congelado en linear/linear_probing/mlp_2layer, y en
      from_scratch el grupo es único a lr_head). Así un override de grid no
      des-congela un linear probing por accidente.
    - ``override_amp`` afecta a cualquier modo.
    """
    if ctx.override_lr_head is not None:
        lr_head = float(ctx.override_lr_head)
    if ctx.override_lr_backbone is not None and mode == "full_finetuning":
        lr_backbone = float(ctx.override_lr_backbone)
    if ctx.override_amp is not None:
        amp = str(ctx.override_amp)
    return lr_head, lr_backbone, amp


# ---------------------------------------------------------------------------
# Contexto y entradas del plan
# ---------------------------------------------------------------------------

@dataclass
class CheckpointRef:
    id: str
    origin: str
    path: Optional[str]


@dataclass
class FullContext:
    checkpoints: List[CheckpointRef]
    output_dir: Path
    device: str = "auto"
    seed: int = 42
    processed_root: str = DEFAULT_PROCESSED_ROOT
    processed_downstream_root: str = DEFAULT_PROCESSED_DOWNSTREAM_ROOT
    repo_root: Path = REPO_ROOT_DEFAULT
    only_dataset: Optional[str] = None
    only_checkpoint: Optional[str] = None
    only_mode: Optional[str] = None
    max_tasks: Optional[int] = None
    skip_existing: bool = False
    # Overrides de calibración (Fase 7a). None = sin override (comportamiento 6a).
    override_lr_backbone: Optional[float] = None
    override_lr_head: Optional[float] = None
    override_amp: Optional[str] = None
    tag: Optional[str] = None


@dataclass
class RunMatrixEntry:
    checkpoint_id: str
    checkpoint_origin: str
    checkpoint_path: Optional[str]
    dataset: str
    mode: Optional[str]
    runnable: bool
    task_status: str
    reason: Optional[str] = None
    task_type: Optional[str] = None
    role: Optional[str] = None
    primary_metric: Optional[str] = None
    caveat: Optional[str] = None


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def checkpoints_from_config(cfg: Dict[str, Any]) -> List[CheckpointRef]:
    out: List[CheckpointRef] = []
    for c in cfg.get("checkpoints", []) or []:
        out.append(CheckpointRef(
            id=str(c["id"]), origin=str(c.get("origin", "unknown")),
            path=c.get("path"),
        ))
    if not out:
        raise ValueError("La config debe declarar `checkpoints` (o usar --checkpoint-bank).")
    return out


def checkpoints_from_bank(bank_json_path: Path) -> List[CheckpointRef]:
    """Lee los `included` de un checkpoint_bank_v1.json como checkpoints.

    Solo se incluyen entradas con `checkpoint_path` no vacío. Pensado para
    ``--checkpoint-bank``; el banco vive local-only bajo docs/.
    """
    data = json.loads(Path(bank_json_path).read_text(encoding="utf-8"))
    out: List[CheckpointRef] = []
    for e in data.get("included", []):
        path = e.get("checkpoint_path")
        if not path:
            continue
        out.append(CheckpointRef(id=str(e["id"]), origin=str(e.get("origin", "unknown")),
                                 path=path))
    if not out:
        raise ValueError(f"Sin checkpoints `included` con path en {bank_json_path}")
    return out


def _datasets_in_config(cfg: Dict[str, Any]) -> List[str]:
    datasets = cfg.get("datasets") or []
    if not datasets:
        raise ValueError("La config de downstream_full_v1 debe declarar `datasets`.")
    return list(datasets)


# ---------------------------------------------------------------------------
# Matriz de ejecución (checkpoint × dataset × modo), con dedup de from_scratch
# ---------------------------------------------------------------------------

def build_run_matrix(cfg: Dict[str, Any], ctx: FullContext) -> List[RunMatrixEntry]:
    """Expande la matriz de ejecución respetando filtros y dedup.

    Reglas:
      - Datasets ``needs_semantic_review`` / no registrados / no
        TRANSFER_TARGET producen UNA entrada skipped (checkpoint-independiente).
      - ``from_scratch`` se emite UNA vez por dataset (checkpoint pseudo
        ``from_scratch``), no por cada checkpoint.
      - Los modos checkpoint-dependientes se emiten por cada checkpoint.
      - ``only_dataset`` / ``only_checkpoint`` / ``only_mode`` filtran.
      - ``max_tasks`` limita las entradas EJECUTABLES (skipped no cuentan).
    """
    modes_per_task = cfg.get("modes_per_task", {})
    ckpts = ctx.checkpoints
    entries: List[RunMatrixEntry] = []
    n_runnable = 0

    def _cap_reached() -> bool:
        return ctx.max_tasks is not None and n_runnable >= ctx.max_tasks

    for ds in _datasets_in_config(cfg):
        if ctx.only_dataset is not None and ds != ctx.only_dataset:
            continue
        # Resolver la TaskSpec; skipped si no runnable o no es TT.
        try:
            spec: TaskSpec = get_task(ds)
        except KeyError:
            entries.append(RunMatrixEntry(
                checkpoint_id="*", checkpoint_origin="*", checkpoint_path=None,
                dataset=ds, mode=None, runnable=False,
                task_status="not_registered", reason="not_registered"))
            continue
        if not is_runnable(spec):
            entries.append(RunMatrixEntry(
                checkpoint_id="*", checkpoint_origin="*", checkpoint_path=None,
                dataset=ds, mode=None, runnable=False, task_status=spec.status,
                reason=spec.status, task_type=spec.task_type, role=spec.role,
                primary_metric=spec.primary_metric, caveat=spec.caveat))
            continue
        if spec.role not in ALLOWED_DOWNSTREAM_ROLES:
            # Guard duro: nunca entrenar downstream sobre un PRETRAIN_SOURCE.
            entries.append(RunMatrixEntry(
                checkpoint_id="*", checkpoint_origin="*", checkpoint_path=None,
                dataset=ds, mode=None, runnable=False, task_status=spec.status,
                reason=f"role_not_downstream:{spec.role}", task_type=spec.task_type,
                role=spec.role, primary_metric=spec.primary_metric, caveat=spec.caveat))
            continue

        for mode in modes_per_task.get(ds, [spec.task_type]):
            if ctx.only_mode is not None and mode != ctx.only_mode:
                continue
            if is_checkpoint_dependent(mode):
                for ck in ckpts:
                    if ctx.only_checkpoint is not None and ck.id != ctx.only_checkpoint:
                        continue
                    if _cap_reached():
                        continue
                    entries.append(RunMatrixEntry(
                        checkpoint_id=ck.id, checkpoint_origin=ck.origin,
                        checkpoint_path=ck.path, dataset=ds, mode=mode,
                        runnable=True, task_status=spec.status,
                        task_type=spec.task_type, role=spec.role,
                        primary_metric=spec.primary_metric, caveat=spec.caveat))
                    n_runnable += 1
            else:
                # from_scratch: independiente del checkpoint, una sola vez.
                if (ctx.only_checkpoint is not None
                        and ctx.only_checkpoint != FROM_SCRATCH_ID):
                    continue
                if _cap_reached():
                    continue
                entries.append(RunMatrixEntry(
                    checkpoint_id=FROM_SCRATCH_ID, checkpoint_origin=FROM_SCRATCH_ORIGIN,
                    checkpoint_path=None, dataset=ds, mode=mode, runnable=True,
                    task_status=spec.status, task_type=spec.task_type, role=spec.role,
                    primary_metric=spec.primary_metric, caveat=spec.caveat))
                n_runnable += 1
    return entries


# ---------------------------------------------------------------------------
# Construcción de la config del trainer (layout <ckpt>/<dataset>/<mode>)
# ---------------------------------------------------------------------------

def build_trainer_config(
    entry: RunMatrixEntry, ctx: FullContext, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Config dict que consume ``cmd_train`` del trainer. Pura (sin torch).

    El trainer escribe ``run_info.json`` y ``best.pt`` en
    ``log_dir/run_name`` y ``checkpoint_dir/run_name`` respectivamente; con
    ``log_dir = checkpoint_dir = output_dir/<ckpt_id>/<dataset>`` y
    ``run_name = <mode>`` ambos caen en el task_dir deseado.
    """
    spec = get_task(entry.dataset)
    mode_spec = _full_mode_spec(spec.task_type, entry.mode)
    out_dir = Path(ctx.output_dir)
    ds_dir = out_dir / entry.checkpoint_id / entry.dataset
    seed = int(ctx.seed)

    # Layout: sin tag -> <ckpt>/<dataset>/<mode>/ (canónico 6a). Con tag (grid
    # de LR, Fase 7a) -> <ckpt>/<dataset>/<mode>/<tag>/, sin colisionar con el
    # canónico. El trainer escribe en log_dir/run_name, así que:
    #   sin tag: log_dir=ds_dir, run_name=<mode>  -> ds_dir/<mode>
    #   con tag: log_dir=ds_dir/<mode>, run_name=<tag> -> ds_dir/<mode>/<tag>
    if ctx.tag:
        mode_dir = ds_dir / str(entry.mode)
        task_dir = mode_dir / ctx.tag
        io_dir = mode_dir
        run_name = str(ctx.tag)
    else:
        task_dir = ds_dir / str(entry.mode)
        io_dir = ds_dir
        run_name = str(entry.mode)

    base_cfg: Dict[str, Any] = {
        "run_name": run_name,
        "seed": seed,
        "dataset": entry.dataset,
        "model": dict(MODEL_BASE),
        "paths": {"log_dir": str(io_dir), "checkpoint_dir": str(io_dir)},
    }

    if mode_spec["trainer_kind"] == "classification":
        c = cfg.get("classification", {})
        base_cfg["task"] = "classification_multiclass"
        base_cfg["data"] = {
            "processed_root": ctx.processed_root,
            "dataset": entry.dataset,
            "batch_size": 64,
            "batch_size_policy": "adaptive_by_channels",
            "max_channel_batch": 512,
            "min_batch_size": 1,
            "num_workers": 2,
        }
        lr_head, lr_backbone, amp = _resolve_lrs_and_amp(c, entry.mode, mode_spec)
        lr_head, lr_backbone, amp = _apply_ctx_overrides(
            lr_head, lr_backbone, amp, entry.mode, ctx)
        base_cfg["training"] = {
            "max_epochs": int(c.get("max_epochs", 20)),
            "early_stopping_patience": c.get("early_stopping_patience", 8),
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "weight_decay": 0.01,
            "amp": amp,
            "grad_clip_norm": 1.0,
            "log_every": 50,
            "eval_every_epochs": 1,
            "metric_for_best": str(c.get("metric", "macro_f1_val")),
            "head_dropout": 0.1,
        }
    else:  # rul
        r = cfg.get("regression", {})
        base_cfg["task"] = "regression_rul"
        base_cfg["head"] = mode_spec["head"]
        base_cfg["data"] = {
            "processed_downstream_root": ctx.processed_downstream_root,
            "target_key": "rul_capped_125",
            "n_channels_fallback": 24,
            "batch_size": 32,
            "batch_size_policy": "adaptive_by_channels",
            "max_channel_batch": 512,
            "min_batch_size": 1,
            "num_workers": 2,
        }
        lr_head, lr_backbone, amp = _resolve_lrs_and_amp(r, entry.mode, mode_spec)
        lr_head, lr_backbone, amp = _apply_ctx_overrides(
            lr_head, lr_backbone, amp, entry.mode, ctx)
        base_cfg["training"] = {
            "max_epochs": int(r.get("max_epochs", 30)),
            "early_stopping_patience": r.get("early_stopping_patience", 10),
            "lr_head": lr_head,
            "lr_backbone": lr_backbone,
            "weight_decay": 0.01,
            "amp": amp,
            "grad_clip_norm": 1.0,
            "log_every": 50,
            "eval_every_epochs": 1,
            "metric_for_best": str(r.get("metric", "rmse_val")),
            "lower_is_better": True,
        }
        base_cfg["evaluation"] = {"save_predictions": False}

    return {
        "trainer_kind": mode_spec["trainer_kind"],
        "trainer_mode": mode_spec["trainer_mode"],
        "checkpoint_required": mode_spec["checkpoint_required"],
        "run_name": run_name,
        "task_dir": task_dir,
        "tag": ctx.tag,
        "lr_backbone": lr_backbone,
        "lr_head": lr_head,
        "cfg": base_cfg,
    }


# ---------------------------------------------------------------------------
# Costura del trainer (los tests la sustituyen) — firma idéntica al probe
# ---------------------------------------------------------------------------

def _invoke_trainer(
    trainer_kind: str, cfg: Dict[str, Any], trainer_mode: str,
    checkpoint: Optional[Path], repo_root: Path,
) -> int:
    if trainer_kind == "classification":
        from training import train_downstream_classification as trainer
    elif trainer_kind == "rul":
        from training import train_downstream_rul as trainer
    else:
        raise ValueError(f"trainer_kind desconocido: {trainer_kind!r}")
    return trainer.cmd_train(cfg, trainer_mode, checkpoint, repo_root)


def _read_run_info(task_dir: Path) -> Optional[Dict[str, Any]]:
    p = Path(task_dir) / "run_info.json"
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _result_from_run_info(
    run_info: Dict[str, Any], spec: TaskSpec, built: Dict[str, Any],
    entry: RunMatrixEntry, task_dir: Path, *, reused: bool = False,
) -> Dict[str, Any]:
    metric_name = run_info.get("metric_for_best") or spec.primary_metric
    best_value = run_info.get("best_value")
    primary_value = float(best_value) if isinstance(best_value, (int, float)) else None
    status = "ok" if primary_value is not None else "partial"
    return {
        "checkpoint_id": entry.checkpoint_id,
        "checkpoint_origin": entry.checkpoint_origin,
        "dataset": entry.dataset,
        "mode": entry.mode,
        "tag": built.get("tag"),
        "lr_backbone": built.get("lr_backbone"),
        "lr_head": built.get("lr_head"),
        "trainer_mode": built["trainer_mode"],
        "trainer_kind": built["trainer_kind"],
        "status": status,
        "role": spec.role,
        "task_type": spec.task_type,
        "primary_metric_name": metric_name,
        "primary_metric_value": primary_value,
        "selection_split": "val",
        "best_epoch": run_info.get("best_epoch"),
        "elapsed_seconds": run_info.get("elapsed_seconds"),
        "n_classes": run_info.get("n_classes"),
        "test_metrics": run_info.get("test_metrics"),
        "caveat": spec.caveat,
        "task_dir": str(task_dir),
        "config_hash": run_info.get("config_hash"),
        "code_version": run_info.get("git_hash"),
        "reused_existing": bool(reused),
    }


def _failed_result(
    entry: RunMatrixEntry, ctx: FullContext, built: Dict[str, Any], error: str,
) -> Dict[str, Any]:
    spec = get_task(entry.dataset)
    result = {
        "checkpoint_id": entry.checkpoint_id,
        "checkpoint_origin": entry.checkpoint_origin,
        "dataset": entry.dataset,
        "mode": entry.mode,
        "trainer_mode": built.get("trainer_mode"),
        "trainer_kind": built.get("trainer_kind"),
        "status": "failed",
        "role": spec.role,
        "task_type": spec.task_type,
        "primary_metric_name": spec.primary_metric,
        "primary_metric_value": None,
        "selection_split": "val",
        "error": error,
        "caveat": spec.caveat,
        "task_dir": str(built.get("task_dir", "")),
    }
    _persist_task_artifacts(result, entry, ctx, built)
    return result


def _persist_task_artifacts(
    result: Dict[str, Any], entry: RunMatrixEntry, ctx: FullContext,
    built: Dict[str, Any],
) -> None:
    task_dir = Path(built.get("task_dir") or (
        Path(ctx.output_dir) / entry.checkpoint_id / entry.dataset / str(entry.mode)))
    task_dir.mkdir(parents=True, exist_ok=True)

    row = ResultRow(
        experiment_id=new_experiment_id(
            "downstream_full", result["dataset"], MODEL_BASE["name"],
            entry.checkpoint_origin, int(ctx.seed),
            suffix=f"{entry.checkpoint_id}_{result.get('mode') or 'na'}",
        ),
        phase="downstream_full",
        dataset=result["dataset"],
        role=result.get("role") or "TRANSFER_TARGET",
        task_type=result.get("task_type") or "classification",
        model_name=MODEL_BASE["name"],
        checkpoint_origin=entry.checkpoint_origin,
        seed=int(ctx.seed),
        primary_metric_name=result.get("primary_metric_name") or "unknown",
        primary_metric_value=result.get("primary_metric_value"),
        status=result["status"],
        created_at=now_ts(),
        caveat=result.get("caveat"),
        config_hash=result.get("config_hash"),
        code_version=result.get("code_version"),
        extra={
            "checkpoint_id": entry.checkpoint_id,
            "mode": result.get("mode"),
            "trainer_mode": result.get("trainer_mode"),
            "selection_split": result.get("selection_split", "val"),
            "best_epoch": result.get("best_epoch"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "test_metrics": result.get("test_metrics"),
            "error": result.get("error"),
        },
    )
    write_result_row(row, task_dir)
    with (task_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, ensure_ascii=False, allow_nan=False, indent=2)


def run_single_task(
    entry: RunMatrixEntry, ctx: FullContext, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Ejecuta una entrada (checkpoint, dataset, modo) y devuelve su resultado."""
    spec = get_task(entry.dataset)
    built = build_trainer_config(entry, ctx, cfg)
    task_dir = built["task_dir"]
    task_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = Path(entry.checkpoint_path) if entry.checkpoint_path else None
    if built["checkpoint_required"] and checkpoint is None:
        return _failed_result(entry, ctx, built,
                              "checkpoint requerido pero no disponible")

    try:
        rc = _invoke_trainer(built["trainer_kind"], built["cfg"],
                             built["trainer_mode"], checkpoint, ctx.repo_root)
    except Exception as exc:  # noqa: BLE001
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return _failed_result(entry, ctx, built, tb)

    if rc != 0:
        return _failed_result(entry, ctx, built, f"trainer devolvió rc={rc}")

    run_info = _read_run_info(task_dir)
    if run_info is None:
        return _failed_result(entry, ctx, built,
                              "no se encontró run_info.json tras entrenar")

    result = _result_from_run_info(run_info, spec, built, entry, task_dir)
    _persist_task_artifacts(result, entry, ctx, built)
    return result


def load_existing_result(
    entry: RunMatrixEntry, ctx: FullContext, cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Si ya existe run_info.json para la entrada, lo reutiliza (skip-existing)."""
    spec = get_task(entry.dataset)
    built = build_trainer_config(entry, ctx, cfg)
    run_info = _read_run_info(built["task_dir"])
    if run_info is None:
        return None
    result = _result_from_run_info(run_info, spec, built, entry, built["task_dir"],
                                   reused=True)
    _persist_task_artifacts(result, entry, ctx, built)
    return result


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_dry_run(cfg: Dict[str, Any], ctx: FullContext) -> int:
    print(f"[{now_ts()}] === DOWNSTREAM FULL V1 DRY-RUN ===")
    print(f"  config_hash: {config_hash(cfg)}")
    print(f"  git: {get_git_info(ctx.repo_root)}")
    print(f"  checkpoints: {[c.id for c in ctx.checkpoints]}")
    matrix = build_run_matrix(cfg, ctx)
    runnable = [e for e in matrix if e.runnable]
    skipped = [e for e in matrix if not e.runnable]
    print(f"\n  Tareas ejecutables ({len(runnable)}):")
    for e in runnable:
        print(f"    ckpt={e.checkpoint_id:<30} {e.dataset:<12} mode={e.mode}")
    print(f"\n  Tareas omitidas ({len(skipped)}):")
    for e in skipped:
        print(f"    {e.dataset:<12} status={e.task_status} reason={e.reason}")
    print("\n  Dry-run OK. La corrida real requiere --mode run y checkpoints en Drive.")
    return 0


def cmd_run(cfg: Dict[str, Any], ctx: FullContext) -> int:
    print(f"[{now_ts()}] === DOWNSTREAM FULL V1 RUN ===")
    out_dir = Path(ctx.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    matrix = build_run_matrix(cfg, ctx)
    runnable = [e for e in matrix if e.runnable]
    skipped = [e for e in matrix if not e.runnable]
    print(f"  checkpoints={[c.id for c in ctx.checkpoints]}")
    print(f"  ejecutables={len(runnable)} skipped={len(skipped)}")

    logger = JsonlLogger(out_dir / "downstream_full_v1.jsonl")
    logger.log({"event": "suite_started", "n_runnable": len(runnable),
                "n_skipped": len(skipped),
                "checkpoints": [c.id for c in ctx.checkpoints]})

    task_results: List[Dict[str, Any]] = []

    for e in skipped:
        res = {
            "checkpoint_id": e.checkpoint_id, "dataset": e.dataset, "mode": None,
            "status": "skipped", "role": e.role, "task_type": e.task_type,
            "primary_metric_name": e.primary_metric, "primary_metric_value": None,
            "reason": e.reason, "caveat": e.caveat,
        }
        task_results.append(res)
        logger.log({"event": "task_skipped", "dataset": e.dataset, "reason": e.reason})

    for e in runnable:
        print(f"\n  >>> {e.checkpoint_id} / {e.dataset} / {e.mode}")
        if ctx.skip_existing:
            existing = load_existing_result(e, ctx, cfg)
            if existing is not None:
                task_results.append(existing)
                logger.log({"event": "task_reused", "checkpoint_id": e.checkpoint_id,
                            "dataset": e.dataset, "mode": e.mode,
                            "status": existing["status"]})
                print(f"      REUSADO status={existing['status']} "
                      f"{existing.get('primary_metric_name')}="
                      f"{existing.get('primary_metric_value')}")
                continue
        logger.log({"event": "task_started", "checkpoint_id": e.checkpoint_id,
                    "dataset": e.dataset, "mode": e.mode})
        res = run_single_task(e, ctx, cfg)
        task_results.append(res)
        logger.log({"event": "task_finished", "checkpoint_id": e.checkpoint_id,
                    "dataset": e.dataset, "mode": e.mode, "status": res["status"],
                    "primary_metric_value": res.get("primary_metric_value")})
        print(f"      status={res['status']} "
              f"{res.get('primary_metric_name')}={res.get('primary_metric_value')}")

    logger.close()
    summary = _build_global_summary(cfg, ctx, task_results)
    _write_global_summary(out_dir, summary)
    print(f"\n  summary: {out_dir / 'downstream_full_v1_summary.json'}")
    print(f"  ok={summary['n_tasks_ok']} failed={summary['n_tasks_failed']} "
          f"skipped={summary['n_tasks_skipped']}")
    return 0


def _build_global_summary(
    cfg: Dict[str, Any], ctx: FullContext, task_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    ok = [r for r in task_results if r["status"] == "ok"]
    failed = [r for r in task_results if r["status"] == "failed"]
    skipped = [r for r in task_results if r["status"] == "skipped"]
    partial = [r for r in task_results if r["status"] == "partial"]

    per_task = [
        {
            "checkpoint_id": r.get("checkpoint_id"),
            "dataset": r["dataset"],
            "mode": r.get("mode"),
            "tag": r.get("tag"),
            "lr_backbone": r.get("lr_backbone"),
            "status": r["status"],
            "primary_metric_name": r.get("primary_metric_name"),
            "primary_val_metric": r.get("primary_metric_value"),
            "direction": METRIC_DIRECTION.get(str(r.get("primary_metric_name")), "unknown"),
            "test_metrics": r.get("test_metrics"),
        }
        for r in task_results if r["status"] in ("ok", "partial")
    ]

    return {
        "phase": "downstream_full",
        "suite_version": "v1",
        "config_hash": config_hash(cfg),
        "created_at": now_ts(),
        "seed": int(ctx.seed),
        "selection_split": "val",
        "checkpoints": [c.id for c in ctx.checkpoints],
        "status": "ok" if (ok or partial) else ("failed" if failed else "skipped"),
        "n_tasks_total": len(task_results),
        "n_tasks_ok": len(ok),
        "n_tasks_partial": len(partial),
        "n_tasks_failed": len(failed),
        "n_tasks_skipped": len(skipped),
        "per_task_val_metric": per_task,
        "caveats": sorted({f"{r['dataset']}: {r.get('caveat')}"
                           for r in task_results if r.get("caveat")}),
        "tasks": task_results,
    }


def _write_global_summary(out_dir: Path, summary: Dict[str, Any]) -> None:
    import csv

    out_dir = Path(out_dir)
    with (out_dir / "downstream_full_v1_summary.json").open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, allow_nan=False, indent=2)

    csv_path = out_dir / "downstream_full_v1_summary.csv"
    fields = ["checkpoint_id", "dataset", "mode", "tag", "lr_backbone", "status",
              "task_type", "primary_metric_name", "primary_metric_value",
              "selection_split"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary["tasks"]:
            writer.writerow({
                "checkpoint_id": r.get("checkpoint_id"),
                "dataset": r.get("dataset"),
                "mode": r.get("mode"),
                "tag": r.get("tag"),
                "lr_backbone": ("" if r.get("lr_backbone") is None
                                else r.get("lr_backbone")),
                "status": r.get("status"),
                "task_type": r.get("task_type"),
                "primary_metric_name": r.get("primary_metric_name"),
                "primary_metric_value": (
                    "" if r.get("primary_metric_value") is None
                    else r.get("primary_metric_value")),
                "selection_split": r.get("selection_split", "val"),
            })


def cmd_list(repo_root: Path) -> int:
    from training.downstream.task_registry import list_tasks
    print(f"[{now_ts()}] Tareas downstream registradas:")
    for task in list_tasks():
        print(f"  - {task.dataset:<14} role={task.role:<16} type={task.task_type:<14} "
              f"status={task.status}")
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("training/configs/downstream_full_v1.yaml"))
    parser.add_argument("--mode", choices=["list", "dry-run", "run"], default="dry-run")
    parser.add_argument("--checkpoint-bank", type=Path, default=None,
                        help="JSON del banco; usa sus `included` como checkpoints.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--only-dataset", type=str, default=None)
    parser.add_argument("--only-checkpoint", type=str, default=None)
    parser.add_argument("--only-mode", type=str, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--override-lr-backbone", type=float, default=None,
                        help="Override del lr_backbone (solo full_finetuning). Fase 7a.")
    parser.add_argument("--override-lr-head", type=float, default=None,
                        help="Override del lr_head (cualquier modo).")
    parser.add_argument("--override-amp", type=str, default=None,
                        choices=["auto", "off"], help="Override de AMP.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Sub-nivel <mode>/<tag>/ en el output (grid de LR); "
                             "no colisiona con la corrida canónica sin tag.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    return parser.parse_args(argv)


def _ctx_from_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> FullContext:
    if args.checkpoint_bank is not None:
        ckpts = checkpoints_from_bank(args.checkpoint_bank)
    else:
        ckpts = checkpoints_from_config(cfg)
    out_dir = args.output_dir or Path(cfg.get("output_dir", "results/downstream/full_v1"))
    return FullContext(
        checkpoints=ckpts,
        output_dir=Path(out_dir),
        device=args.device,
        seed=args.seed if args.seed is not None else int(cfg.get("seed", 42)),
        processed_root=cfg.get("processed_root", DEFAULT_PROCESSED_ROOT),
        processed_downstream_root=cfg.get(
            "processed_downstream_root", DEFAULT_PROCESSED_DOWNSTREAM_ROOT),
        repo_root=args.repo_root,
        only_dataset=args.only_dataset,
        only_checkpoint=args.only_checkpoint,
        only_mode=args.only_mode,
        max_tasks=args.max_tasks,
        skip_existing=args.skip_existing,
        override_lr_backbone=args.override_lr_backbone,
        override_lr_head=args.override_lr_head,
        override_amp=args.override_amp,
        tag=args.tag,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "list":
        return cmd_list(args.repo_root)
    cfg = load_config(args.config)
    ctx = _ctx_from_args(args, cfg)
    if args.mode == "dry-run":
        return cmd_dry_run(cfg, ctx)
    return cmd_run(cfg, ctx)


if __name__ == "__main__":
    sys.exit(main())
