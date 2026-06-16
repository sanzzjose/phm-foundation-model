"""Probe Suite v1: evaluación rápida de utilidad de un *checkpoint* SSL.

La suite ejecuta una batería de probes baratos sobre los datasets
``CWRU``, ``HSG18``, ``CALCE_CS2``, ``PHMAP23`` y ``CMAPSS_RUL`` y
produce ``probe_suite_v1_summary.{json,csv}`` que permite descartar
*checkpoints* (central / FedAvg / FedProx) manifiestamente malos antes
de escalar el pretraining federado a 50/100 rondas.

Modos disponibles:

* ``--mode list``: lista las tareas registradas y su estado.
* ``--mode dry-run``: verifica la config y el plan sin entrenar.
* ``--mode run``: lanza los probes reales reutilizando los trainers ya
  implementados (``train_downstream_classification`` /
  ``train_downstream_rul``), SOLO sobre tareas ``ready``.

Reglas del Probe Suite v1 (sec 16 de CLAUDE.md):

* Solo usa validación, NUNCA test, para seleccionar/rankear. Los
  trainers reportan test solo como métrica informativa con el best ckpt
  ya elegido por validación.
* Las tareas ``needs_semantic_review`` / ``not_implemented`` quedan como
  ``status="skipped"`` y NO entran en rankings ni agregados.
* La suite NO sustituye al downstream completo: es un screening (pocas
  épocas, sin grids, sin multi-seed).

Integración con los trainers (esta sesión):

Cada probe se traduce a una llamada ``cmd_train(cfg, trainer_mode,
checkpoint, repo_root)`` del trainer correspondiente. Los "modos" del
probe son abstractos y se mapean a ``(trainer_mode, overrides)``:

* clasificación: ``linear_probing`` -> ``linear_probing``;
  ``full_finetuning_short`` -> ``full_finetuning`` con ``lr_backbone=1e-5``.
* RUL: ``linear`` -> ``linear_probing`` + cabeza lineal;
  ``mlp_2layer`` -> ``linear_probing`` + cabeza MLP (``hidden_dim=256``).

El trainer real se invoca a través de :func:`_invoke_trainer`, una
costura fina que los tests sustituyen para validar la orquestación
(plan, skip, agregados, paths) sin entrenar.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from training.downstream import builders  # noqa: F401 (registra tasks)
from training.downstream.task_registry import (
    TaskSpec,
    get_task,
    is_runnable,
    list_tasks,
    require_runnable,
)
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

# Bloque de modelo del backbone PatchTSTPhm base. Debe coincidir con el
# checkpoint SSL que se evalúa (801 808 params). Es fijo para toda la
# suite: no se exploran tamaños aquí (eso es Fase 10).
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

# Raíces por defecto en Drive (sobreescribibles por la config o el ctx).
DEFAULT_PROCESSED_ROOT = "/content/drive/MyDrive/fm_fl_phmd/processed"
DEFAULT_PROCESSED_DOWNSTREAM_ROOT = (
    "/content/drive/MyDrive/fm_fl_phmd/processed_downstream"
)


# ---------------------------------------------------------------------------
# Mapeo de modos abstractos del probe -> (trainer_mode, overrides)
# ---------------------------------------------------------------------------

def _probe_mode_spec(task_type: str, probe_mode: str) -> Dict[str, Any]:
    """Traduce un modo abstracto del probe a la receta del trainer.

    Devuelve un dict con:
      - ``trainer_kind``: "classification" | "rul";
      - ``trainer_mode``: modo aceptado por el trainer
        (``linear_probing`` | ``full_finetuning``);
      - ``lr_backbone``: override (None = backbone congelado);
      - ``head``: override de cabeza para RUL (None si no aplica);
      - ``checkpoint_required``: True (todos los probes parten de un SSL ckpt).

    Lanza ``ValueError`` si el (task_type, probe_mode) no está soportado.
    """
    if task_type == "classification":
        if probe_mode == "linear_probing":
            return {
                "trainer_kind": "classification",
                "trainer_mode": "linear_probing",
                "lr_backbone": None,
                "head": None,
                "checkpoint_required": True,
            }
        if probe_mode == "full_finetuning_short":
            return {
                "trainer_kind": "classification",
                "trainer_mode": "full_finetuning",
                "lr_backbone": 1e-5,
                "head": None,
                "checkpoint_required": True,
            }
        raise ValueError(
            f"Modo de probe no soportado para classification: {probe_mode!r}"
        )
    if task_type == "rul":
        if probe_mode == "linear":
            return {
                "trainer_kind": "rul",
                "trainer_mode": "linear_probing",
                "lr_backbone": None,
                "head": {
                    "hidden_dim": None,
                    "dropout": 0.0,
                    "activation": None,
                    "keep_last_dim": False,
                },
                "checkpoint_required": True,
            }
        if probe_mode == "mlp_2layer":
            return {
                "trainer_kind": "rul",
                "trainer_mode": "linear_probing",
                "lr_backbone": None,
                "head": {
                    "hidden_dim": 256,
                    "dropout": 0.1,
                    "activation": "gelu",
                    "keep_last_dim": False,
                },
                "checkpoint_required": True,
            }
        raise ValueError(f"Modo de probe no soportado para rul: {probe_mode!r}")
    raise ValueError(f"task_type no soportado por el probe suite: {task_type!r}")


# ---------------------------------------------------------------------------
# Contexto de ejecución
# ---------------------------------------------------------------------------

@dataclass
class ProbeContext:
    """Parámetros de una corrida real del Probe Suite."""

    checkpoint_path: Optional[Path]
    checkpoint_id: str
    checkpoint_origin: str
    output_dir: Path
    device: str = "auto"
    seed: int = 42
    processed_root: str = DEFAULT_PROCESSED_ROOT
    processed_downstream_root: str = DEFAULT_PROCESSED_DOWNSTREAM_ROOT
    repo_root: Path = REPO_ROOT_DEFAULT
    only_dataset: Optional[str] = None
    only_mode: Optional[str] = None
    max_tasks: Optional[int] = None
    skip_existing: bool = False


@dataclass
class ProbePlanEntry:
    """Una entrada del plan: (dataset, probe_mode) y si es ejecutable."""

    dataset: str
    probe_mode: Optional[str]
    runnable: bool
    task_status: str
    reason: Optional[str] = None
    task_type: Optional[str] = None
    role: Optional[str] = None
    primary_metric: Optional[str] = None
    caveat: Optional[str] = None


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _datasets_in_config(cfg: Dict[str, Any]) -> List[str]:
    datasets = cfg.get("datasets") or []
    if not datasets:
        raise ValueError("La config del Probe Suite v1 debe declarar `datasets`.")
    return list(datasets)


def _checkpoint_id_from_cfg(cfg: Dict[str, Any]) -> str:
    return cfg.get("checkpoint", {}).get("id", "unknown")


def plan_probe_tasks(
    cfg: Dict[str, Any],
    *,
    only_dataset: Optional[str] = None,
    only_mode: Optional[str] = None,
    max_tasks: Optional[int] = None,
) -> List[ProbePlanEntry]:
    """Expande ``datasets`` x ``modes_per_task`` en un plan de probes.

    - Las tareas ``needs_semantic_review`` / ``not_implemented`` / no
      registradas producen UNA entrada ``runnable=False`` (skipped),
      sin expandir modos.
    - Las tareas ``ready`` producen una entrada por modo declarado en
      ``modes_per_task[dataset]``.
    - ``only_dataset`` / ``only_mode`` filtran el plan; ``max_tasks``
      limita el número de entradas EJECUTABLES (las skipped no cuentan
      contra el límite, siempre se reportan).
    """
    modes_per_task = cfg.get("modes_per_task", {})
    entries: List[ProbePlanEntry] = []
    n_runnable = 0
    for ds in _datasets_in_config(cfg):
        if only_dataset is not None and ds != only_dataset:
            continue
        try:
            spec: TaskSpec = get_task(ds)
        except KeyError:
            entries.append(ProbePlanEntry(
                dataset=ds, probe_mode=None, runnable=False,
                task_status="not_registered", reason="not_registered",
            ))
            continue
        if not is_runnable(spec):
            entries.append(ProbePlanEntry(
                dataset=ds, probe_mode=None, runnable=False,
                task_status=spec.status, reason=spec.status,
                task_type=spec.task_type, role=spec.role,
                primary_metric=spec.primary_metric, caveat=spec.caveat,
            ))
            continue
        modes = modes_per_task.get(ds, [spec.task_type])
        for mode in modes:
            if only_mode is not None and mode != only_mode:
                continue
            if max_tasks is not None and n_runnable >= max_tasks:
                continue
            entries.append(ProbePlanEntry(
                dataset=ds, probe_mode=mode, runnable=True,
                task_status=spec.status, task_type=spec.task_type,
                role=spec.role, primary_metric=spec.primary_metric,
                caveat=spec.caveat,
            ))
            n_runnable += 1
    return entries


# ---------------------------------------------------------------------------
# Construcción de la config del trainer
# ---------------------------------------------------------------------------

def build_trainer_config(
    dataset: str,
    spec: TaskSpec,
    probe_mode: str,
    ctx: ProbeContext,
    cfg: Dict[str, Any],
) -> Dict[str, Any]:
    """Construye la config dict que consume ``cmd_train`` del trainer.

    Pura (no importa torch). Devuelve un dict con keys:
    ``trainer_kind``, ``trainer_mode``, ``checkpoint_required``,
    ``run_name``, ``task_dir`` y ``cfg`` (la config real del trainer).
    """
    mode_spec = _probe_mode_spec(spec.task_type, probe_mode)
    run_name = f"{dataset}__{probe_mode}"
    out_dir = Path(ctx.output_dir)
    task_dir = out_dir / "per_task" / run_name
    ckpt_dir = out_dir / "per_task_ckpts" / run_name

    seed = int(ctx.seed)
    base_cfg: Dict[str, Any] = {
        "run_name": run_name,
        "seed": seed,
        "dataset": dataset,
        "model": dict(MODEL_BASE),
        "paths": {
            "log_dir": str(out_dir / "per_task"),
            "checkpoint_dir": str(out_dir / "per_task_ckpts"),
        },
    }

    if mode_spec["trainer_kind"] == "classification":
        c = cfg.get("classification", {})
        base_cfg["task"] = "classification_multiclass"
        base_cfg["data"] = {
            "processed_root": ctx.processed_root,
            "dataset": dataset,
            "batch_size": 64,
            "batch_size_policy": "adaptive_by_channels",
            "max_channel_batch": 512,
            "min_batch_size": 1,
            "num_workers": 2,
        }
        base_cfg["training"] = {
            "max_epochs": int(c.get("max_epochs", 10)),
            "lr_head": float(c.get("lr_head", 1e-3)),
            "lr_backbone": mode_spec["lr_backbone"],
            "weight_decay": 0.01,
            "amp": "auto",
            "grad_clip_norm": 1.0,
            "log_every": 50,
            "eval_every_epochs": 1,
            "metric_for_best": str(c.get("metric", "macro_f1_val")),
            "head_dropout": 0.1,
            "max_train_batches_per_epoch": c.get("max_train_batches_per_epoch"),
            "max_val_batches": c.get("max_val_batches"),
            "max_test_batches": c.get("max_test_batches"),
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
        base_cfg["training"] = {
            "max_epochs": int(r.get("max_epochs", 20)),
            "lr_head": float(r.get("lr_head", 1e-3)),
            "lr_backbone": mode_spec["lr_backbone"],
            "weight_decay": 0.01,
            "amp": "auto",
            "grad_clip_norm": 1.0,
            "log_every": 50,
            "eval_every_epochs": 1,
            "metric_for_best": str(r.get("metric", "rmse_val")),
            "lower_is_better": True,
            "max_train_batches_per_epoch": r.get("max_train_batches_per_epoch"),
            "max_val_batches": r.get("max_val_batches"),
            "max_test_batches": r.get("max_test_batches"),
        }
        base_cfg["evaluation"] = {"save_predictions": False}

    return {
        "trainer_kind": mode_spec["trainer_kind"],
        "trainer_mode": mode_spec["trainer_mode"],
        "checkpoint_required": mode_spec["checkpoint_required"],
        "run_name": run_name,
        "task_dir": task_dir,
        "ckpt_dir": ckpt_dir,
        "cfg": base_cfg,
    }


# ---------------------------------------------------------------------------
# Costura de invocación del trainer (los tests la sustituyen)
# ---------------------------------------------------------------------------

def _invoke_trainer(
    trainer_kind: str,
    cfg: Dict[str, Any],
    trainer_mode: str,
    checkpoint: Optional[Path],
    repo_root: Path,
) -> int:
    """Importa y llama al trainer real. Aislada para tests (monkeypatch).

    Importación perezosa de torch: así importar ``probe_suite`` no
    arrastra el stack de entrenamiento.
    """
    if trainer_kind == "classification":
        from training import train_downstream_classification as trainer
    elif trainer_kind == "rul":
        from training import train_downstream_rul as trainer
    else:
        raise ValueError(f"trainer_kind desconocido: {trainer_kind!r}")
    return trainer.cmd_train(cfg, trainer_mode, checkpoint, repo_root)


def _read_run_info(task_dir: Path) -> Optional[Dict[str, Any]]:
    """Lee ``run_info.json`` que el trainer escribe en su ``log_dir``."""
    p = Path(task_dir) / "run_info.json"
    if not p.is_file():
        return None
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_single_probe(
    entry: ProbePlanEntry, ctx: ProbeContext, cfg: Dict[str, Any]
) -> Dict[str, Any]:
    """Ejecuta un probe (dataset, modo) y devuelve su resultado.

    Construye la config, invoca el trainer vía :func:`_invoke_trainer`,
    lee ``run_info.json`` y extrae la métrica de validación (best). NO
    usa test para seleccionar: el ``best_value`` del trainer ya proviene
    de la validación (``metric_for_best`` termina en ``_val``).

    Cualquier excepción se captura y devuelve ``status="failed"`` con un
    traceback corto, para que la suite continúe con las demás tareas.
    """
    spec = get_task(entry.dataset)
    built = build_trainer_config(entry.dataset, spec, entry.probe_mode, ctx, cfg)
    task_dir = built["task_dir"]
    task_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = ctx.checkpoint_path
    if built["checkpoint_required"] and checkpoint is None:
        return _failed_result(
            entry, ctx, built,
            "checkpoint requerido pero no se pasó --checkpoint",
        )

    try:
        rc = _invoke_trainer(
            built["trainer_kind"], built["cfg"], built["trainer_mode"],
            checkpoint, ctx.repo_root,
        )
    except Exception as exc:  # noqa: BLE001 (queremos capturar todo)
        tb = "".join(traceback.format_exception_only(type(exc), exc)).strip()
        return _failed_result(entry, ctx, built, tb)

    if rc != 0:
        return _failed_result(
            entry, ctx, built, f"trainer devolvió rc={rc}",
        )

    run_info = _read_run_info(task_dir)
    if run_info is None:
        return _failed_result(
            entry, ctx, built, "no se encontró run_info.json tras entrenar",
        )

    result = _result_from_run_info(run_info, spec, built, entry, task_dir)
    _persist_task_artifacts(result, entry, ctx, built)
    return result


def _result_from_run_info(
    run_info: Dict[str, Any], spec: TaskSpec, built: Dict[str, Any],
    entry: ProbePlanEntry, task_dir: Path, *, reused: bool = False,
) -> Dict[str, Any]:
    """Construye el dict de resultado a partir de un ``run_info.json``.

    La métrica primaria es ``best_value`` (proviene de VALIDACIÓN, ya que
    ``metric_for_best`` termina en ``_val``). ``reused=True`` marca que el
    resultado se recupero de un run previo (``--skip-existing``).
    """
    metric_name = run_info.get("metric_for_best") or spec.primary_metric
    best_value = run_info.get("best_value")
    primary_value = (
        float(best_value) if isinstance(best_value, (int, float)) else None
    )
    status = "ok" if primary_value is not None else "partial"
    return {
        "dataset": entry.dataset,
        "probe_mode": entry.probe_mode,
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


def load_existing_probe_result(
    entry: ProbePlanEntry, ctx: ProbeContext, cfg: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Si ya existe ``run_info.json`` para (dataset, modo), devuelve su
    resultado sin re-entrenar. Si no, devuelve ``None``.

    Pensado para ``--skip-existing``: permite reanudar una suite parcial
    (p.ej. tras un corte de runtime) reutilizando los probes ya
    completados en Drive y solo ejecutando los que faltan.
    """
    spec = get_task(entry.dataset)
    built = build_trainer_config(entry.dataset, spec, entry.probe_mode, ctx, cfg)
    task_dir = built["task_dir"]
    run_info = _read_run_info(task_dir)
    if run_info is None:
        return None
    result = _result_from_run_info(run_info, spec, built, entry, task_dir, reused=True)
    _persist_task_artifacts(result, entry, ctx, built)
    return result


def _failed_result(
    entry: ProbePlanEntry, ctx: ProbeContext, built: Dict[str, Any],
    error: str,
) -> Dict[str, Any]:
    spec = get_task(entry.dataset)
    result = {
        "dataset": entry.dataset,
        "probe_mode": entry.probe_mode,
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
    result: Dict[str, Any], entry: ProbePlanEntry, ctx: ProbeContext,
    built: Dict[str, Any],
) -> None:
    """Escribe ``result_row.json`` + ``summary.json`` por tarea."""
    task_dir = Path(built.get("task_dir") or (ctx.output_dir / "per_task" / f"{entry.dataset}__{entry.probe_mode}"))
    task_dir.mkdir(parents=True, exist_ok=True)

    row = ResultRow(
        experiment_id=new_experiment_id(
            "downstream_probe",
            result["dataset"],
            MODEL_BASE["name"],
            ctx.checkpoint_origin,
            int(ctx.seed),
            suffix=str(result.get("probe_mode") or "na"),
        ),
        phase="downstream_probe",
        dataset=result["dataset"],
        role=result.get("role") or "TRANSFER_TARGET",
        task_type=result.get("task_type") or "classification",
        model_name=MODEL_BASE["name"],
        checkpoint_origin=ctx.checkpoint_origin,
        seed=int(ctx.seed),
        primary_metric_name=result.get("primary_metric_name") or "unknown",
        primary_metric_value=result.get("primary_metric_value"),
        status=result["status"],
        created_at=now_ts(),
        caveat=result.get("caveat"),
        config_hash=result.get("config_hash"),
        code_version=result.get("code_version"),
        extra={
            "probe_mode": result.get("probe_mode"),
            "trainer_mode": result.get("trainer_mode"),
            "checkpoint_id": ctx.checkpoint_id,
            "selection_split": result.get("selection_split", "val"),
            "best_epoch": result.get("best_epoch"),
            "elapsed_seconds": result.get("elapsed_seconds"),
            "error": result.get("error"),
        },
    )
    write_result_row(row, task_dir)

    summary_path = task_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(result), f, ensure_ascii=False, allow_nan=False, indent=2)


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_list(repo_root: Path) -> int:
    """Lista las tareas registradas y su estado."""
    print(f"[{now_ts()}] Tareas registradas en el task registry:")
    for task in list_tasks():
        print(f"  - {task.dataset:<14} role={task.role:<16} type={task.task_type:<14} "
              f"status={task.status:<24} primary={task.primary_metric}")
    return 0


def cmd_dry_run(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Verifica el plan de la suite sin entrenar."""
    print(f"[{now_ts()}] === PROBE SUITE V1 DRY-RUN ===")
    print(f"  config_hash: {config_hash(cfg)}")
    print(f"  git: {get_git_info(repo_root)}")
    print(f"  checkpoint_id: {_checkpoint_id_from_cfg(cfg)}")

    plan = plan_probe_tasks(cfg)
    runnable = [e for e in plan if e.runnable]
    skipped = [e for e in plan if not e.runnable]

    print(f"\n  Probes ejecutables ({len(runnable)}):")
    for e in runnable:
        print(f"    {e.dataset:<12} mode={e.probe_mode:<22} "
              f"type={e.task_type} primary={e.primary_metric}")
    print(f"\n  Probes omitidos ({len(skipped)}):")
    for e in skipped:
        print(f"    {e.dataset:<12} status={e.task_status} "
              f"reason={e.reason} ({e.caveat or 'sin glosa'})")
    print("\n  Dry-run OK. La corrida real requiere --mode run y un checkpoint.")
    return 0


def cmd_run(cfg: Dict[str, Any], ctx: ProbeContext) -> int:
    """Corrida real del Probe Suite: ejecuta los probes ``ready``."""
    print(f"[{now_ts()}] === PROBE SUITE V1 RUN ===")
    out_dir = Path(ctx.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan = plan_probe_tasks(
        cfg, only_dataset=ctx.only_dataset, only_mode=ctx.only_mode,
        max_tasks=ctx.max_tasks,
    )
    runnable = [e for e in plan if e.runnable]
    skipped = [e for e in plan if not e.runnable]

    ckpt_id = ctx.checkpoint_id
    print(f"  checkpoint_id={ckpt_id} origin={ctx.checkpoint_origin}")
    print(f"  ejecutables={len(runnable)} skipped={len(skipped)}")

    logger = JsonlLogger(out_dir / "probe_suite_v1.jsonl")
    logger.log({"event": "probe_suite_started", "checkpoint_id": ckpt_id,
                "checkpoint_origin": ctx.checkpoint_origin,
                "n_runnable": len(runnable), "n_skipped": len(skipped)})

    task_results: List[Dict[str, Any]] = []

    # Skipped: registrar como tales (no entran en rankings).
    for e in skipped:
        res = {
            "dataset": e.dataset,
            "probe_mode": None,
            "status": "skipped",
            "role": e.role,
            "task_type": e.task_type,
            "primary_metric_name": e.primary_metric,
            "primary_metric_value": None,
            "reason": e.reason,
            "caveat": e.caveat,
        }
        task_results.append(res)
        logger.log({"event": "task_skipped", "dataset": e.dataset,
                    "reason": e.reason})

    # Ejecutables: lanzar el trainer real, capturando fallos por tarea.
    for e in runnable:
        print(f"\n  >>> probe {e.dataset} / {e.probe_mode}")
        # --skip-existing: reutilizar un run previo si ya hay run_info.json.
        if ctx.skip_existing:
            existing = load_existing_probe_result(e, ctx, cfg)
            if existing is not None:
                task_results.append(existing)
                logger.log({"event": "task_reused", "dataset": e.dataset,
                            "probe_mode": e.probe_mode,
                            "status": existing["status"]})
                print(f"      REUSADO (run previo) status={existing['status']} "
                      f"{existing.get('primary_metric_name')}="
                      f"{existing.get('primary_metric_value')}")
                continue
        logger.log({"event": "task_started", "dataset": e.dataset,
                    "probe_mode": e.probe_mode})
        res = run_single_probe(e, ctx, cfg)
        task_results.append(res)
        logger.log({"event": "task_finished", "dataset": e.dataset,
                    "probe_mode": e.probe_mode, "status": res["status"],
                    "primary_metric_name": res.get("primary_metric_name"),
                    "primary_metric_value": res.get("primary_metric_value")})
        print(f"      status={res['status']} "
              f"{res.get('primary_metric_name')}="
              f"{res.get('primary_metric_value')}")

    logger.close()

    summary = _build_global_summary(cfg, ctx, task_results)
    _write_global_summary(out_dir, summary)
    print(f"\n  summary: {out_dir / 'probe_suite_v1_summary.json'}")
    print(f"  ok={summary['n_tasks_ok']} failed={summary['n_tasks_failed']} "
          f"skipped={summary['n_tasks_skipped']}")
    # La corrida no falla aunque alguna tarea individual falle: el reporting
    # lo refleja con status por tarea. Solo devolvemos !=0 si NO se ejecutó
    # ninguna tarea ejecutable que estaba planificada.
    return 0


def _build_global_summary(
    cfg: Dict[str, Any], ctx: ProbeContext, task_results: List[Dict[str, Any]]
) -> Dict[str, Any]:
    ok = [r for r in task_results if r["status"] == "ok"]
    failed = [r for r in task_results if r["status"] == "failed"]
    skipped = [r for r in task_results if r["status"] == "skipped"]
    partial = [r for r in task_results if r["status"] == "partial"]

    datasets_executed = sorted({r["dataset"] for r in task_results
                                if r["status"] in ("ok", "partial", "failed")})
    datasets_skipped = sorted({r["dataset"] for r in skipped})

    # Métrica de validación por tarea (solo ok/partial; skipped excluidas).
    per_task_metric = [
        {
            "dataset": r["dataset"],
            "probe_mode": r.get("probe_mode"),
            "status": r["status"],
            "primary_metric_name": r.get("primary_metric_name"),
            "primary_val_metric": r.get("primary_metric_value"),
            "direction": METRIC_DIRECTION.get(
                str(r.get("primary_metric_name")), "unknown"
            ),
        }
        for r in task_results if r["status"] in ("ok", "partial")
    ]

    return {
        "phase": "downstream_probe",
        "suite_version": "v1",
        "checkpoint_id": ctx.checkpoint_id,
        "checkpoint_origin": ctx.checkpoint_origin,
        "config_hash": config_hash(cfg),
        "created_at": now_ts(),
        "seed": int(ctx.seed),
        "selection_split": "val",
        "status": "ok" if (ok or partial) else ("failed" if failed else "skipped"),
        "n_tasks_total": len(task_results),
        "n_tasks_ok": len(ok),
        "n_tasks_partial": len(partial),
        "n_tasks_failed": len(failed),
        "n_tasks_skipped": len(skipped),
        "datasets_executed": datasets_executed,
        "datasets_skipped": datasets_skipped,
        "per_task_val_metric": per_task_metric,
        "caveats": sorted({
            f"{r['dataset']}: {r.get('caveat')}"
            for r in task_results if r.get("caveat")
        }),
        "tasks": task_results,
    }


def _write_global_summary(out_dir: Path, summary: Dict[str, Any]) -> None:
    import csv

    out_dir = Path(out_dir)
    json_path = out_dir / "probe_suite_v1_summary.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, allow_nan=False, indent=2)

    # CSV: una fila por tarea (incluye skipped para trazabilidad, pero el
    # consumidor del ranking debe filtrar status != ok/partial).
    csv_path = out_dir / "probe_suite_v1_summary.csv"
    fields = ["dataset", "probe_mode", "status", "task_type",
              "primary_metric_name", "primary_metric_value", "selection_split"]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in summary["tasks"]:
            writer.writerow({
                "dataset": r.get("dataset"),
                "probe_mode": r.get("probe_mode"),
                "status": r.get("status"),
                "task_type": r.get("task_type"),
                "primary_metric_name": r.get("primary_metric_name"),
                "primary_metric_value": (
                    "" if r.get("primary_metric_value") is None
                    else r.get("primary_metric_value")
                ),
                "selection_split": r.get("selection_split", "val"),
            })


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path,
                        default=Path("training/configs/probe_suite_v1.yaml"))
    parser.add_argument("--mode", choices=["list", "dry-run", "run"],
                        default="list")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--checkpoint-id", type=str, default=None)
    parser.add_argument("--checkpoint-origin", type=str, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-tasks", type=int, default=None)
    parser.add_argument("--only-dataset", type=str, default=None)
    parser.add_argument("--only-mode", type=str, default=None)
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Reutiliza probes ya completados (run_info.json presente) y solo "
             "ejecuta los que faltan. Util para reanudar una suite parcial.",
    )
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT_DEFAULT)
    return parser.parse_args(argv)


def _ctx_from_args(args: argparse.Namespace, cfg: Dict[str, Any]) -> ProbeContext:
    ckpt_cfg = cfg.get("checkpoint", {})
    out_dir = args.output_dir or Path(cfg.get("output_dir", "results/downstream/probes/v1"))
    return ProbeContext(
        checkpoint_path=args.checkpoint,
        checkpoint_id=args.checkpoint_id or ckpt_cfg.get("id", "unknown"),
        checkpoint_origin=args.checkpoint_origin or ckpt_cfg.get("origin", "unknown"),
        output_dir=Path(out_dir),
        device=args.device,
        seed=args.seed if args.seed is not None else int(cfg.get("seed", 42)),
        repo_root=args.repo_root,
        only_dataset=args.only_dataset,
        only_mode=args.only_mode,
        max_tasks=args.max_tasks,
        skip_existing=args.skip_existing,
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.mode == "list":
        return cmd_list(args.repo_root)
    cfg = load_config(args.config)
    if args.mode == "dry-run":
        return cmd_dry_run(cfg, args.repo_root)
    ctx = _ctx_from_args(args, cfg)
    return cmd_run(cfg, ctx)


if __name__ == "__main__":
    sys.exit(main())
