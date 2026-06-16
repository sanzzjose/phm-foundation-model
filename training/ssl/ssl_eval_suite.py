"""SSL validation suite: evaluación SSL fija, determinista y comparable.

Construye un conjunto determinista de *batches* de validación SSL sobre
PRETRAIN_SOURCE y evalúa cualquier *checkpoint* contra ese conjunto. Es
la pieza que permite comparar *encoders* central / FedAvg / FedProx /
SCAFFOLD bajo las mismas condiciones, sin tocar TRANSFER_TARGET y sin
elegir *checkpoints* solo por la *loss* SSL de entrenamiento.

Métricas reportadas:

* ``ssl_val_loss_weighted`` — media de la *loss* por batch ponderada por
  el número de elementos que contribuyen (``n_loss_elements``);
* ``ssl_val_loss_per_client`` — idem agrupado por cliente FL;
* ``ssl_val_loss_per_dataset`` — idem agrupado por dataset;
* ``masked_patch_count`` — total de patches enmascarados evaluados;
* ``padding_ignored_count`` — *timesteps* de padding ignorados por la
  *loss* (relevante para ``tail_policy=pad``);
* ``nonfinite_count`` — número de batches con *loss* no finita;
* ``effective_bc_mean`` / ``effective_bc_max`` — control de coste real
  (``batch_size * n_channels`` por batch);
* ``coverage_dataset_count`` / ``coverage_client_count`` — datasets y
  clientes realmente evaluados.

Reglas obligatorias:

* No usa TRANSFER_TARGET en ningún caso. El plan se construye filtrando
  ``role == PRETRAIN_SOURCE`` y cualquier mención de un TT en la config
  aborta la ejecución.
* No entrena: ``model.eval()`` + ``torch.no_grad()``. No modifica el
  *checkpoint*.
* Semilla fija para que la selección de ``ssl_mask`` sea reproducible.
* Soporta ``max_batches`` (corte global) y ``max_batches_per_dataset``
  para *smoke* barato.

La función pura :func:`evaluate_batches` recibe un iterador de batches ya
preparados (cada uno con sus máscaras) y agrega las métricas. Es testable
con batches sintéticos sin tocar disco ni Drive.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml

from training.experiments import (
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


# ---------------------------------------------------------------------------
# Config y fuentes
# ---------------------------------------------------------------------------

def load_config(path: Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_checkpoint_origin(
    cli_origin: Optional[str], checkpoint_cfg: Dict[str, Any]
) -> str:
    """Resuelve el origen del checkpoint con prioridad CLI > config > 'unknown'.

    Pura y testable. El CLI (``--checkpoint-origin``) gana sobre
    ``config.checkpoint.origin``; si ninguno está, devuelve ``"unknown"``.
    """
    if cli_origin is not None:
        return str(cli_origin)
    return str((checkpoint_cfg or {}).get("origin", "unknown"))


def resolve_checkpoint_id(
    cli_id: Optional[str], checkpoint_cfg: Dict[str, Any], fallback_stem: str
) -> str:
    """Resuelve el id del checkpoint con prioridad CLI > config > stem.

    Pura y testable. El CLI (``--checkpoint-id``) gana sobre
    ``config.checkpoint.id``; si ninguno está, usa el ``stem`` del fichero.
    """
    return str(cli_id or (checkpoint_cfg or {}).get("id") or fallback_stem)


def _read_processed_summary(repo_root: Path) -> List[Dict[str, Any]]:
    """Lee ``results/processed_summary.csv`` y devuelve filas como dicts."""
    p = repo_root / "results" / "processed_summary.csv"
    if not p.is_file():
        raise FileNotFoundError(
            f"No se encuentra processed_summary.csv en {p}. "
            "La SSL eval suite necesita este fichero como inventario."
        )
    with p.open("r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _ps_datasets(processed_rows: List[Dict[str, Any]]) -> List[str]:
    """Devuelve los datasets con ``role == PRETRAIN_SOURCE``."""
    return [r["dataset"] for r in processed_rows if r.get("role") == "PRETRAIN_SOURCE"]


def _tt_datasets(processed_rows: List[Dict[str, Any]]) -> List[str]:
    return [r["dataset"] for r in processed_rows if r.get("role") == "TRANSFER_TARGET"]


def _assert_no_tt(cfg: Dict[str, Any], processed_rows: List[Dict[str, Any]]) -> None:
    """Aborta si la config menciona explícitamente algún TRANSFER_TARGET.

    Inspecciona ``data.datasets`` y ``data.include_datasets`` (si están).
    El valor especial ``"all"`` no enumera datasets y por tanto no puede
    introducir un TT (el plan se filtra a PS aguas abajo).
    """
    tt = set(_tt_datasets(processed_rows))
    declared = set()
    data_cfg = cfg.get("data", {}) or {}
    for key in ("datasets", "include_datasets"):
        val = data_cfg.get(key)
        if isinstance(val, (list, tuple)):
            declared.update(str(v) for v in val)
    leaked = declared & tt
    if leaked:
        raise ValueError(
            "SSL eval suite no puede usar TRANSFER_TARGET. "
            f"Datasets prohibidos en config: {sorted(leaked)}"
        )


def _resolve_ps_plan(repo_root: Path) -> List[Dict[str, Any]]:
    """Construye el plan PS canónico (36 datasets) con su cliente FL.

    Reutiliza ``compute_pretraining_sampling_plan`` para que el cliente de
    cada dataset y el ``n_channels`` salgan de la misma fuente que el
    pretraining. Si las fuentes no están disponibles, propaga el error.
    """
    from training.sampling import compute_pretraining_sampling_plan, load_sources

    proc, cli, asum = load_sources(
        repo_root / "results/processed_summary.csv",
        repo_root / "results/client_summary.csv",
        repo_root / "results/audit/audit_summary.json",
    )
    plan = compute_pretraining_sampling_plan(proc, cli, asum)
    return plan


# ---------------------------------------------------------------------------
# Núcleo de evaluación (puro, testable con batches sintéticos)
# ---------------------------------------------------------------------------

def _make_generator(seed: int):
    """Devuelve un ``torch.Generator`` reproducible para la selección SSL."""
    import torch
    g = torch.Generator()
    g.manual_seed(int(seed))
    return g


def evaluate_one_batch(
    model: Any,
    batch: Dict[str, Any],
    *,
    mask_ratio: float,
    generator: Any,
    device: Any,
    loss_fn: str = "mse",
) -> Dict[str, Any]:
    """Evalúa un único batch SSL y devuelve métricas escalares.

    No hace *backward*. Asume que el modelo ya está en ``eval()`` y que la
    llamada se envuelve en ``torch.no_grad()`` por el caller (o aquí
    mismo). Devuelve un dict con ``loss`` (float o ``None`` si no finita),
    ``n_loss_elements``, ``n_masked_patches``, ``padding_ignored_elements``,
    ``effective_bc`` y banderas de estado.
    """
    import torch

    from training.ssl.loss import compute_masked_reconstruction_loss_with_metrics
    from training.ssl.masking import (
        canonicalize_valid_patch_mask,
        generate_ssl_mask,
    )

    x = batch["patches"].to(device)
    vtm = batch["valid_time_mask"].to(device)
    vpm = batch["valid_patch_mask"].to(device)
    B, C, N, P = x.shape

    vpm_canon = canonicalize_valid_patch_mask(vpm, B, C, N)
    # generate_ssl_mask usa el generator en la CPU; movemos la máscara al
    # device tras generarla para mantener el determinismo del muestreo
    # independientemente del device de cómputo.
    ssl_mask_cpu = generate_ssl_mask(
        vpm_canon.cpu(), mask_ratio=mask_ratio, generator=generator
    )
    ssl_mask = ssl_mask_cpu.to(device)

    out = model(x, vtm, vpm_canon, ssl_mask)
    metrics = compute_masked_reconstruction_loss_with_metrics(
        pred=out["reconstruction"],
        target=x,
        ssl_mask=ssl_mask,
        valid_time_mask=vtm,
        valid_patch_mask=vpm_canon,
        loss_fn=loss_fn,
    )

    loss_t = metrics["loss"]
    loss_finite = bool(torch.isfinite(loss_t).item())
    return {
        "loss": float(loss_t.item()) if loss_finite else None,
        "loss_finite": loss_finite,
        "n_loss_elements": int(metrics["n_loss_elements"].item()),
        "n_masked_patches": int(metrics["n_masked_patches"].item()),
        "padding_ignored_elements": int(metrics["padding_ignored_elements"].item()),
        "effective_bc": int(B * C),
        "batch_shape": [B, C, N, P],
    }


def aggregate_metrics(
    per_batch: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Agrega una lista de resultados de :func:`evaluate_one_batch`.

    Cada entrada de ``per_batch`` debe traer, además de las claves de
    :func:`evaluate_one_batch`, los campos ``dataset`` y ``client`` para
    el agrupamiento. La *loss* global y por grupo se pondera por
    ``n_loss_elements`` (los batches con cero elementos no contribuyen a
    la media pero sí cuentan en cobertura si tienen patches válidos).

    Devuelve un dict con las métricas canónicas de la suite.
    """
    total_weighted_loss = 0.0
    total_elements = 0
    masked_patch_count = 0
    padding_ignored_count = 0
    nonfinite_count = 0
    effective_bcs: List[int] = []

    by_dataset_loss: Dict[str, float] = defaultdict(float)
    by_dataset_elems: Dict[str, int] = defaultdict(int)
    by_client_loss: Dict[str, float] = defaultdict(float)
    by_client_elems: Dict[str, int] = defaultdict(int)

    datasets_seen = set()
    clients_seen = set()

    for r in per_batch:
        ds = r.get("dataset")
        cl = r.get("client")
        if ds is not None:
            datasets_seen.add(ds)
        if cl is not None:
            clients_seen.add(cl)

        masked_patch_count += int(r.get("n_masked_patches", 0))
        padding_ignored_count += int(r.get("padding_ignored_elements", 0))
        effective_bcs.append(int(r.get("effective_bc", 0)))

        if not r.get("loss_finite", False) or r.get("loss") is None:
            nonfinite_count += 1
            continue

        n_elem = int(r.get("n_loss_elements", 0))
        if n_elem <= 0:
            continue
        loss_val = float(r["loss"])
        total_weighted_loss += loss_val * n_elem
        total_elements += n_elem
        if ds is not None:
            by_dataset_loss[ds] += loss_val * n_elem
            by_dataset_elems[ds] += n_elem
        if cl is not None:
            by_client_loss[cl] += loss_val * n_elem
            by_client_elems[cl] += n_elem

    ssl_val_loss_weighted = (
        total_weighted_loss / total_elements if total_elements > 0 else None
    )
    per_dataset = {
        ds: (by_dataset_loss[ds] / by_dataset_elems[ds])
        for ds in by_dataset_loss
        if by_dataset_elems[ds] > 0
    }
    per_client = {
        cl: (by_client_loss[cl] / by_client_elems[cl])
        for cl in by_client_loss
        if by_client_elems[cl] > 0
    }

    eff_mean = (sum(effective_bcs) / len(effective_bcs)) if effective_bcs else None
    eff_max = max(effective_bcs) if effective_bcs else None

    return {
        "ssl_val_loss_weighted": ssl_val_loss_weighted,
        "ssl_val_loss_per_dataset": per_dataset,
        "ssl_val_loss_per_client": per_client,
        "masked_patch_count": masked_patch_count,
        "padding_ignored_count": padding_ignored_count,
        "nonfinite_count": nonfinite_count,
        "effective_bc_mean": eff_mean,
        "effective_bc_max": eff_max,
        "coverage_dataset_count": len(datasets_seen),
        "coverage_client_count": len(clients_seen),
        "n_batches_evaluated": len(per_batch),
    }


# ---------------------------------------------------------------------------
# Carga de modelo y construcción del iterador de batches
# ---------------------------------------------------------------------------

def load_model_from_checkpoint(checkpoint_path: Path, device: Any) -> Tuple[Any, Dict[str, Any]]:
    """Carga ``PatchTSTPhm`` desde un checkpoint y lo deja en ``eval()``.

    El checkpoint debe contener ``model_state_dict`` y ``config`` (formato
    que escribe ``training.train_ssl_central``). Devuelve el modelo y la
    config del checkpoint. NO modifica el fichero.
    """
    import torch

    from models.patchtst_phm import build_patchtst_phm, count_parameters

    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if "model_state_dict" not in ckpt:
        raise ValueError(
            f"Checkpoint {checkpoint_path} no contiene 'model_state_dict'."
        )
    ckpt_cfg = ckpt.get("config", {})
    model_cfg = ckpt_cfg.get("model", {})
    if not model_cfg:
        raise ValueError(
            f"Checkpoint {checkpoint_path} no trae config.model; no se puede "
            "reconstruir la arquitectura."
        )
    model = build_patchtst_phm(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    info = {
        "checkpoint_step": ckpt.get("step"),
        "checkpoint_config_hash": ckpt.get("config_hash"),
        "checkpoint_git_hash": ckpt.get("git_hash"),
        "param_count": count_parameters(model),
        "model_config": model_cfg,
    }
    return model, info


def iter_eval_batches(
    plan: List[Dict[str, Any]],
    processed_root: Path,
    *,
    split: str,
    batch_size: int,
    seed: int,
    max_batches_per_dataset: Optional[int],
    max_batches_total: Optional[int],
) -> Iterator[Dict[str, Any]]:
    """Itera batches deterministas de validación SSL (PS-only).

    Recorre los datasets del plan en orden fijo (determinista) y, para
    cada uno, produce hasta ``max_batches_per_dataset`` batches. El corte
    global ``max_batches_total`` detiene la iteración antes de agotar el
    plan. Cada batch se anota con ``dataset`` y ``client`` para el
    agrupamiento posterior. Los datasets sin shards en disco se saltan con
    un aviso (no abortan la suite).
    """
    from training.phm_webdataset import iter_dataset_batches
    from training.phm_tar_reader import find_shards

    emitted_total = 0
    for row in plan:
        ds = str(row["dataset"])
        client = row.get("client")
        shards = find_shards(processed_root, ds, split)
        if not shards:
            print(f"  [skip] {ds}: sin shards en {processed_root}/{ds}/{split}")
            continue
        emitted_ds = 0
        for batch in iter_dataset_batches(
            dataset_name=ds,
            processed_root=processed_root,
            split=split,
            batch_size=batch_size,
            shuffle=False,           # determinismo: sin shuffle
            seed=seed,
            client=client,
            drop_last=False,
        ):
            batch["__dataset__"] = ds
            batch["__client__"] = client
            yield batch
            emitted_ds += 1
            emitted_total += 1
            if max_batches_per_dataset is not None and emitted_ds >= max_batches_per_dataset:
                break
            if max_batches_total is not None and emitted_total >= max_batches_total:
                return
        if max_batches_total is not None and emitted_total >= max_batches_total:
            return


# ---------------------------------------------------------------------------
# Comandos
# ---------------------------------------------------------------------------

def cmd_dry_run(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Verifica la config, el plan y los paths sin tocar el modelo."""
    print(f"[{now_ts()}] === SSL EVAL DRY-RUN ===")
    print(f"  config_hash: {config_hash(cfg)}")
    print(f"  git: {get_git_info(repo_root)}")

    processed = _read_processed_summary(repo_root)
    ps = _ps_datasets(processed)
    print(f"  PRETRAIN_SOURCE detectados: {len(ps)} (esperados: 36)")
    assert len(ps) == 36, f"Esperados 36 PS; encontrados {len(ps)}"

    _assert_no_tt(cfg, processed)
    print("  OK: la config NO referencia datasets TRANSFER_TARGET.")

    paths = cfg.get("paths", {})
    if "processed_root" in paths:
        proc_root = Path(paths["processed_root"])
        print(f"  processed_root: {proc_root}")
        print(f"    existe en disco: {proc_root.is_dir()}")
    if "checkpoint_path" in paths:
        print(f"  checkpoint_path: {paths['checkpoint_path']}")
    print()
    print("Dry-run OK. La evaluacion real requiere `--mode eval` y un checkpoint.")
    return 0


def _resolve_device(requested: Optional[str]) -> Any:
    """Resuelve el device a usar. ``None`` o ``"auto"`` => cuda si hay."""
    import torch

    if requested in (None, "auto"):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(requested)


def cmd_eval(
    cfg: Dict[str, Any],
    repo_root: Path,
    *,
    checkpoint_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    max_batches: Optional[int] = None,
    device: Optional[str] = None,
    seed: Optional[int] = None,
    checkpoint_origin: Optional[str] = None,
    checkpoint_id: Optional[str] = None,
) -> int:
    """Evaluación SSL real de un checkpoint sobre PRETRAIN_SOURCE.

    Carga el modelo, construye los batches deterministas PS-only, calcula
    las métricas y escribe los artefactos (``ssl_eval_summary.json/csv``,
    ``ssl_eval_per_client.csv``, ``ssl_eval_per_dataset.csv``,
    ``ssl_eval.jsonl`` y ``result_row.json``). No entrena, no modifica el
    checkpoint.
    """
    import torch

    if checkpoint_path is None:
        raise ValueError(
            "cmd_eval requiere --checkpoint con la ruta a un checkpoint .pt"
        )
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint no encontrado: {checkpoint_path}")

    # Determinismo + anti-leakage.
    processed = _read_processed_summary(repo_root)
    _assert_no_tt(cfg, processed)

    eff_seed = int(seed if seed is not None else cfg.get("seed", 42))
    torch.manual_seed(eff_seed)

    dev = _resolve_device(device)
    print(f"[{now_ts()}] === SSL EVAL === device={dev} seed={eff_seed}")

    # Config de datos.
    data_cfg = cfg.get("data", {}) or {}
    paths_cfg = cfg.get("paths", {}) or {}
    split = data_cfg.get("eval_split", data_cfg.get("split", "train"))
    batch_size = int(data_cfg.get("batch_size_requested", data_cfg.get("batch_size", 8)))
    mask_ratio = float(data_cfg.get("mask_ratio", cfg.get("ssl", {}).get("mask_ratio", 0.30)))
    loss_fn = cfg.get("ssl", {}).get("loss", "mse")

    eff_max_batches = max_batches if max_batches is not None else data_cfg.get("max_batches")
    max_batches_per_dataset = data_cfg.get("max_batches_per_dataset")

    processed_root = Path(paths_cfg.get("processed_root", "processed"))

    # Origen del checkpoint. Prioridad: CLI (--checkpoint-origin) > config
    # (checkpoint.origin) > "unknown". Permite etiquetar correctamente
    # checkpoints central / fedavg / fedprox / scaffold / fedavgm sin
    # depender de un YAML por régimen.
    checkpoint_cfg = cfg.get("checkpoint", {}) or {}
    checkpoint_origin = resolve_checkpoint_origin(checkpoint_origin, checkpoint_cfg)

    # Salida. Prioridad de resolución del directorio base:
    #   1. --output-dir por CLI (se respeta EXACTAMENTE, sin subdir extra).
    #   2. paths.output_dir de la config -> se le añade subdir por checkpoint_id.
    #   3. default repo: results/pretraining/ssl_eval/<checkpoint_id>.
    # El id sigue la misma prioridad CLI > config > stem del fichero.
    checkpoint_id = resolve_checkpoint_id(
        checkpoint_id, checkpoint_cfg, checkpoint_path.stem
    )
    if output_dir is not None:
        # CLI explícito: ruta tal cual, sin añadir subdirectorio.
        output_dir = Path(output_dir)
    else:
        base = paths_cfg.get("output_dir")
        if base:
            output_dir = Path(base) / checkpoint_id
        else:
            output_dir = repo_root / "results" / "pretraining" / "ssl_eval" / checkpoint_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Cargar modelo.
    model, model_info = load_model_from_checkpoint(checkpoint_path, dev)
    print(f"  modelo cargado: {model_info['param_count']:,} parametros, "
          f"step={model_info['checkpoint_step']}")

    # Plan PS-only.
    plan = _resolve_ps_plan(repo_root)
    print(f"  plan PS: {len(plan)} datasets")

    # Loop de evaluación (sin gradiente).
    logger = JsonlLogger(output_dir / "ssl_eval.jsonl")
    logger.log({"event": "eval_started", "checkpoint_id": checkpoint_id,
                "device": str(dev), "seed": eff_seed})

    generator = _make_generator(eff_seed)
    per_batch: List[Dict[str, Any]] = []
    with torch.no_grad():
        for batch in iter_eval_batches(
            plan, processed_root,
            split=split, batch_size=batch_size, seed=eff_seed,
            max_batches_per_dataset=max_batches_per_dataset,
            max_batches_total=eff_max_batches,
        ):
            res = evaluate_one_batch(
                model, batch,
                mask_ratio=mask_ratio, generator=generator,
                device=dev, loss_fn=loss_fn,
            )
            res["dataset"] = batch.get("__dataset__")
            res["client"] = batch.get("__client__")
            per_batch.append(res)
            logger.log({
                "event": "batch_evaluated",
                "dataset": res["dataset"],
                "client": res["client"],
                "loss": res["loss"],
                "n_loss_elements": res["n_loss_elements"],
                "effective_bc": res["effective_bc"],
            })
    logger.close()

    agg = aggregate_metrics(per_batch)

    status = "ok" if agg["n_batches_evaluated"] > 0 and agg["nonfinite_count"] == 0 else "partial"
    summary = {
        "phase": "ssl_eval",
        "checkpoint_id": checkpoint_id,
        "checkpoint_origin": checkpoint_origin,
        "checkpoint_path": str(checkpoint_path),
        "config_hash": config_hash(cfg),
        "created_at": now_ts(),
        "device": str(dev),
        "seed": eff_seed,
        # La suite evalúa el split PS materializado (actualmente 'train')
        # de forma determinista. NO es un holdout SSL separado: el proyecto
        # no materializa un split de validación SSL distinto del de
        # entrenamiento para PRETRAIN_SOURCE.
        "split": split,
        "eval_split_kind": "ps_materialized_deterministic",
        "mask_ratio": mask_ratio,
        "max_batches": eff_max_batches,
        "max_batches_per_dataset": max_batches_per_dataset,
        "status": status,
        **model_info,
        **agg,
    }

    _write_eval_outputs(summary, agg, output_dir, repo_root)
    print(f"[{now_ts()}] ssl_eval terminado: status={status} "
          f"n_batches={agg['n_batches_evaluated']} "
          f"loss_weighted={agg['ssl_val_loss_weighted']} "
          f"coverage_ds={agg['coverage_dataset_count']} "
          f"coverage_cl={agg['coverage_client_count']}")
    print(f"  artefactos en {output_dir}")
    return 0


def _write_eval_outputs(
    summary: Dict[str, Any],
    agg: Dict[str, Any],
    output_dir: Path,
    repo_root: Path,
) -> None:
    """Escribe summary.json/csv, per_client.csv, per_dataset.csv y result_row."""
    # summary.json
    summary_path = output_dir / "ssl_eval_summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(json_safe(summary), f, ensure_ascii=False, allow_nan=False, indent=2)

    # summary.csv (una fila, campos escalares; los dicts per_* se omiten)
    scalar_keys = [
        "checkpoint_id", "checkpoint_origin", "phase", "status", "split",
        "eval_split_kind", "mask_ratio",
        "ssl_val_loss_weighted", "masked_patch_count", "padding_ignored_count",
        "nonfinite_count", "effective_bc_mean", "effective_bc_max",
        "coverage_dataset_count", "coverage_client_count", "n_batches_evaluated",
        "param_count", "config_hash", "created_at",
    ]
    csv_path = output_dir / "ssl_eval_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar_keys)
        w.writeheader()
        w.writerow({k: ("" if summary.get(k) is None else summary.get(k)) for k in scalar_keys})

    # per_client.csv
    pc_path = output_dir / "ssl_eval_per_client.csv"
    with pc_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["client", "ssl_val_loss"])
        for cl, v in sorted(agg["ssl_val_loss_per_client"].items()):
            w.writerow([cl, v])

    # per_dataset.csv
    pd_path = output_dir / "ssl_eval_per_dataset.csv"
    with pd_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "ssl_val_loss"])
        for ds, v in sorted(agg["ssl_val_loss_per_dataset"].items()):
            w.writerow([ds, v])

    # result_row.json compatible con la tabla maestra. El checkpoint_origin
    # proviene del summary (que lo lee de config.checkpoint.origin), no se
    # hardcodea a "unknown".
    checkpoint_origin = str(summary.get("checkpoint_origin", "unknown"))
    row = ResultRow(
        experiment_id=new_experiment_id(
            phase="ssl_eval",
            dataset="all_PS",
            model_name="PatchTSTPhm",
            checkpoint_origin=checkpoint_origin,
            seed=int(summary.get("seed", 0)),
            suffix=str(summary.get("checkpoint_id", "ckpt")),
        ),
        phase="ssl_eval",
        dataset="all_PS",
        role="PRETRAIN_SOURCE",
        task_type="ssl",
        model_name="PatchTSTPhm",
        checkpoint_origin=checkpoint_origin,
        seed=int(summary.get("seed", 0)),
        primary_metric_name="ssl_val_loss_weighted",
        primary_metric_value=summary.get("ssl_val_loss_weighted"),
        status=summary.get("status", "partial"),
        created_at=summary.get("created_at", now_ts()),
        config_hash=summary.get("config_hash"),
        caveat=None,
        extra={
            "coverage_dataset_count": agg["coverage_dataset_count"],
            "coverage_client_count": agg["coverage_client_count"],
            "nonfinite_count": agg["nonfinite_count"],
            "masked_patch_count": agg["masked_patch_count"],
            "padding_ignored_count": agg["padding_ignored_count"],
            "effective_bc_max": agg["effective_bc_max"],
        },
    )
    write_result_row(row, output_dir)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("training/configs/ssl_eval_suite.yaml"),
        help="YAML con la config de la suite.",
    )
    parser.add_argument(
        "--mode",
        choices=["dry-run", "eval"],
        default="dry-run",
        help="Modo de ejecución.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Ruta a un checkpoint a evaluar (solo en --mode=eval).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directorio donde escribir los artefactos de evaluación.",
    )
    parser.add_argument(
        "--checkpoint-origin",
        type=str,
        default=None,
        choices=["central", "fedavg", "fedprox", "scaffold", "fedavgm", "unknown"],
        help="Origen del checkpoint. Prioridad sobre config.checkpoint.origin. "
             "Si no se pasa, se usa la config o 'unknown'.",
    )
    parser.add_argument(
        "--checkpoint-id",
        type=str,
        default=None,
        help="Identificador legible del checkpoint (override de config y "
             "del stem del fichero).",
    )
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Corte global de batches evaluados (smoke).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="cpu / cuda / auto (por defecto auto).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Semilla (override de la config).",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT_DEFAULT,
        help="Raíz del repo (por defecto detectada).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.mode == "dry-run":
        return cmd_dry_run(cfg, args.repo_root)
    return cmd_eval(
        cfg, args.repo_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        max_batches=args.max_batches,
        device=args.device,
        seed=args.seed,
        checkpoint_origin=args.checkpoint_origin,
        checkpoint_id=args.checkpoint_id,
    )


if __name__ == "__main__":
    sys.exit(main())
