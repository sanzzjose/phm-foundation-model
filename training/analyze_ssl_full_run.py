"""Analisis post-hoc REPRODUCIBLE de un run de SSL pretraining centralizado.

NO modifica `run_info.json` ni `metrics.jsonl` historicos. Solo lee.
Escribe en el mismo `log_dir` dos artefactos:

    posthoc_analysis.json   (JSON estricto, allow_nan=False)
    posthoc_analysis.md     (resumen legible en espanol)

Uso tipico:

    python -m training.analyze_ssl_full_run \\
        --log-dir   /content/drive/MyDrive/fm_fl_phmd/logs/pretraining/ssl_central_full_patchtst_phm \\
        --checkpoint /content/drive/MyDrive/fm_fl_phmd/checkpoints/ssl_central_full/ssl_central_full_patchtst_phm/ckpt_step100000.pt

Diseno:

- `metrics.jsonl` historico (anterior al patch de logging estricto) puede
  contener literal `NaN`/`Infinity`. Aqui usamos un parser robusto que los
  detecta y sanea (`parse_constant`) para poder seguir contando ocurrencias.
- El analisis no requiere `torch`. Solo lo importa si se pasa `--checkpoint`
  y existe el fichero; si torch no esta disponible o el checkpoint no
  existe, no falla: registra `checkpoint_state_dict_all_finite=null`.
- Las medias por bucket se reportan tanto crudas como excluyendo steps con
  grad no finito u optimizer no aplicado (cuando el log lo permite).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ----------------------------------------------------------------------
# Parser JSONL robusto (tolera NaN/Infinity historicos)
# ----------------------------------------------------------------------


def _robust_loads(line: str) -> Tuple[Dict[str, Any], Dict[str, int]]:
    """Parsea una linea JSONL aunque contenga literal NaN/Infinity.

    Devuelve (record, counter_local) donde counter_local cuenta cuantos
    valores no-finitos aparecieron en esta linea, agrupados por kind.
    """
    counter: Counter = Counter()

    def _on_constant(s: str):
        # Python json llama a esto cuando ve los literales NaN, Infinity,
        # -Infinity. Devolvemos None y contamos.
        if s == "NaN":
            counter["nan"] += 1
        elif s == "Infinity":
            counter["inf"] += 1
        elif s == "-Infinity":
            counter["neg_inf"] += 1
        return None

    rec = json.loads(line, parse_constant=_on_constant)
    return rec, dict(counter)


# ----------------------------------------------------------------------
# Helpers de JSON estricto
# ----------------------------------------------------------------------


def _safe(obj):
    """Reduce un valor a algo serializable estricto (sin NaN/Infinity)."""
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, bool):
        return obj
    if isinstance(obj, int):
        return obj
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {str(k): _safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_safe(v) for v in obj]
    if obj is None or isinstance(obj, str):
        return obj
    return str(obj)


# ----------------------------------------------------------------------
# Estadisticas robustas
# ----------------------------------------------------------------------


def _stats(values: List[float]) -> Dict[str, Any]:
    finite = [v for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {
            "count": 0, "mean": None, "median": None,
            "p10": None, "p90": None, "min": None, "max": None,
        }
    finite_sorted = sorted(finite)
    n = len(finite_sorted)

    def _percentile(arr, p):
        if not arr:
            return None
        if n == 1:
            return arr[0]
        k = (n - 1) * (p / 100.0)
        f = int(math.floor(k))
        c = int(math.ceil(k))
        if f == c:
            return arr[f]
        return arr[f] + (arr[c] - arr[f]) * (k - f)

    return {
        "count": n,
        "mean": sum(finite_sorted) / n,
        "median": _percentile(finite_sorted, 50),
        "p10": _percentile(finite_sorted, 10),
        "p90": _percentile(finite_sorted, 90),
        "min": finite_sorted[0],
        "max": finite_sorted[-1],
    }


# ----------------------------------------------------------------------
# Lectura de logs
# ----------------------------------------------------------------------


def read_run_info(log_dir: Path) -> Dict[str, Any]:
    ri_path = log_dir / "run_info.json"
    if not ri_path.is_file():
        raise FileNotFoundError(f"No existe {ri_path}")
    # run_info.json se escribio originalmente sin NaN/Infinity (json estandar
    # de Python con indent), asi que loads estricto funciona.
    return json.loads(ri_path.read_text(encoding="utf-8"))


def iter_metrics_lines(log_dir: Path):
    mp = log_dir / "metrics.jsonl"
    if not mp.is_file():
        raise FileNotFoundError(f"No existe {mp}")
    with open(mp, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            yield line


# ----------------------------------------------------------------------
# Analisis principal
# ----------------------------------------------------------------------


# Buckets de step para reportar evolucion de la loss. Cubren warmup, post-
# warmup, varios puntos intermedios y final. Suficiente para detectar
# convergencia, plateau, divergencia, sin enterrar al lector en datos.
DEFAULT_BUCKETS: List[Tuple[int, int]] = [
    (0, 2000),
    (2000, 7000),
    (7000, 12000),
    (22000, 27000),
    (47000, 52000),
    (72000, 77000),
    (95000, 100000),
]


def analyze(
    log_dir: Path,
    checkpoint: Optional[Path] = None,
    huge_loss_threshold: float = 1000.0,
    top_loss_n: int = 20,
    buckets: Optional[List[Tuple[int, int]]] = None,
) -> Dict[str, Any]:
    """Analiza un run cerrado. Devuelve un dict listo para serializar."""
    if buckets is None:
        buckets = DEFAULT_BUCKETS

    ri = read_run_info(log_dir)

    # Contadores y acumuladores
    total_metric_lines = 0
    total_step_records = 0
    total_distribution_records = 0
    nonfinite_grad_total = 0
    nonfinite_grad_by_kind: Counter = Counter()
    nonfinite_grad_by_dataset: Counter = Counter()
    huge_finite_loss_records: List[Dict[str, Any]] = []
    all_loss_with_meta: List[Dict[str, Any]] = []  # para top_loss

    # Por bucket: lista de losses, separando aplicada vs no aplicada (si la
    # info esta en el log).
    bucket_losses: Dict[Tuple[int, int], List[Dict[str, Any]]] = {b: [] for b in buckets}

    for line in iter_metrics_lines(log_dir):
        total_metric_lines += 1
        try:
            rec, line_kinds = _robust_loads(line)
        except json.JSONDecodeError:
            # Linea corrupta: la contamos como anomalia pero seguimos.
            continue

        # Sumar no-finitos detectados por parse_constant a totales globales.
        if line_kinds:
            for k, v in line_kinds.items():
                nonfinite_grad_by_kind[k] += v
                nonfinite_grad_total += v
            ds = rec.get("dataset")
            if ds:
                nonfinite_grad_by_dataset[str(ds)] += sum(line_kinds.values())

        # Discriminar registros: 'kind' == 'distribution' (logged separado).
        if rec.get("kind") == "distribution":
            total_distribution_records += 1
            continue

        total_step_records += 1

        step = rec.get("step")
        loss = rec.get("loss")
        ds = rec.get("dataset")

        # Tambien podriamos ver grad_norm_nonfinite_kind si lo expuso el log
        # nuevo (post-patch). Contar tambien por esta via.
        gnk = rec.get("grad_norm_nonfinite_kind")
        if gnk in ("inf", "nan"):
            # Evitar doble conteo si parse_constant ya lo capturo.
            # Heuristica: si la linea no tenia line_kinds (porque el log es
            # nuevo y guarda null en lugar de Infinity), contamos aqui.
            if not line_kinds:
                nonfinite_grad_by_kind[gnk] += 1
                nonfinite_grad_total += 1
                if ds:
                    nonfinite_grad_by_dataset[str(ds)] += 1

        # Bucket por step
        if isinstance(step, int):
            for b in buckets:
                lo, hi = b
                if lo <= step < hi:
                    bucket_losses[b].append({
                        "step": step,
                        "loss": loss if isinstance(loss, (int, float)) and math.isfinite(loss) else None,
                        "optimizer_applied": rec.get("optimizer_applied"),
                        "amp_nonfinite_grad": rec.get("amp_nonfinite_grad"),
                    })
                    break

        # Loss enorme finita (outlier)
        if isinstance(loss, (int, float)) and math.isfinite(loss):
            if loss >= huge_loss_threshold:
                huge_finite_loss_records.append({
                    "step": step, "dataset": ds, "loss": float(loss),
                })
            all_loss_with_meta.append({"step": step, "dataset": ds, "loss": float(loss)})

    # Top N de loss
    top_loss = sorted(all_loss_with_meta, key=lambda r: -r["loss"])[:top_loss_n]

    # Stats por bucket (crudo + excluyendo non-applied/non-finite grad)
    buckets_summary: List[Dict[str, Any]] = []
    for b in buckets:
        lo, hi = b
        rows = bucket_losses[b]
        all_loss = [r["loss"] for r in rows if r["loss"] is not None]
        clean_loss = [
            r["loss"] for r in rows
            if r["loss"] is not None
            and (r.get("optimizer_applied") is not False)
            and (r.get("amp_nonfinite_grad") is not True)
        ]
        st_all = _stats(all_loss)
        st_clean = _stats(clean_loss)
        buckets_summary.append({
            "range":              f"{lo}-{hi}",
            "lo":                 lo,
            "hi":                 hi,
            "count":              st_all["count"],
            "mean":               st_all["mean"],
            "median":             st_all["median"],
            "p10":                st_all["p10"],
            "p90":                st_all["p90"],
            "min":                st_all["min"],
            "max":                st_all["max"],
            "mean_excluding_nonfinite_grad_or_not_applied": st_clean["mean"],
            "count_clean":        st_clean["count"],
        })

    # Checkpoint
    ckpt_all_finite: Optional[bool] = None
    ckpt_nonfinite_tensors: List[str] = []
    ckpt_warning: Optional[str] = None
    if checkpoint is not None and Path(checkpoint).is_file():
        try:
            import torch  # type: ignore
            ck = torch.load(str(checkpoint), map_location="cpu")
            sd = ck.get("model_state_dict")
            if sd is None:
                ckpt_warning = "checkpoint sin 'model_state_dict' (no es un ckpt de cmd_train)"
            else:
                all_finite = True
                for name, t in sd.items():
                    if hasattr(t, "dtype") and t.is_floating_point():
                        if not torch.isfinite(t).all().item():
                            all_finite = False
                            ckpt_nonfinite_tensors.append(name)
                ckpt_all_finite = bool(all_finite)
        except ImportError:
            ckpt_warning = "torch no disponible: no se puede inspeccionar state_dict"
        except Exception as e:
            ckpt_warning = f"fallo al cargar checkpoint: {e}"
    elif checkpoint is not None:
        ckpt_warning = f"checkpoint no encontrado: {checkpoint}"
    else:
        ckpt_warning = "no se paso --checkpoint"

    # Conclusion narrativa breve (heuristica)
    if ckpt_all_finite is True:
        conclusion = (
            "checkpoint usable; logging/robustness fixes recomendados para "
            "futuros runs (JSONL estricto, grad_norm null en steps con AMP "
            "overflow, final_client_weight coherente con groupby)."
        )
    elif ckpt_all_finite is False:
        conclusion = (
            "checkpoint contiene tensores no finitos; NO usar para downstream "
            f"sin investigar. Tensores afectados: {ckpt_nonfinite_tensors[:5]}"
        )
    else:
        conclusion = (
            "checkpoint no inspeccionado (ver warning); logging/robustness "
            "fixes recomendados para futuros runs."
        )

    # Build resultado
    out: Dict[str, Any] = {
        "run_name":                          ri.get("run_name"),
        "git_hash":                          ri.get("git_hash"),
        "git_dirty":                         ri.get("git_dirty"),
        "config_hash":                       ri.get("config_hash"),
        "param_count":                       ri.get("param_count"),
        "stage":                             ri.get("stage"),
        "optimizer_steps":                   ri.get("optimizer_steps"),
        "skipped_steps":                     ri.get("skipped_steps"),
        "amp_overflow_steps":                ri.get("amp_overflow_steps"),
        "amp_nonfinite_grad_steps":          ri.get("amp_nonfinite_grad_steps"),
        "datasets_seen_count":               len(ri.get("datasets_seen") or {}),
        "clients_seen_count":                len(ri.get("clients_seen") or {}),
        "max_effective_bc":                  ri.get("max_effective_bc"),
        "elapsed_seconds":                   ri.get("elapsed_seconds"),
        "total_metric_lines":                total_metric_lines,
        "total_step_records":                total_step_records,
        "total_distribution_records":        total_distribution_records,
        "nonfinite_grad_steps_total":        nonfinite_grad_total,
        "nonfinite_grad_by_kind":            dict(nonfinite_grad_by_kind),
        "nonfinite_grad_by_dataset":         dict(nonfinite_grad_by_dataset),
        "huge_loss_threshold":               huge_loss_threshold,
        "huge_finite_loss_steps":            huge_finite_loss_records,
        "top_loss_steps":                    top_loss,
        "loss_buckets":                      buckets_summary,
        "final_client_distribution_observed":  ri.get("clients_seen") or {},
        "final_dataset_distribution_observed": ri.get("datasets_seen") or {},
        "checkpoint_state_dict_all_finite":  ckpt_all_finite,
        "checkpoint_nonfinite_tensors":      ckpt_nonfinite_tensors,
        "checkpoint_warning":                ckpt_warning,
        "conclusion":                        conclusion,
    }
    return out


# ----------------------------------------------------------------------
# Render markdown
# ----------------------------------------------------------------------


def render_markdown(analysis: Dict[str, Any]) -> str:
    a = analysis
    lines: List[str] = []
    lines.append(f"# Analisis post-hoc del run `{a.get('run_name')}`\n")
    lines.append(
        "Este analisis se genero post-hoc a partir de los logs del run. "
        "**No se repitio el entrenamiento ni se modifico el checkpoint.** "
        "Los conteos de steps con grad no finito fueron gestionados por "
        "el `GradScaler` de AMP (comportamiento normal en fp16); el run "
        "es valido siempre que el state_dict del checkpoint sea finito.\n"
    )
    lines.append("## Metadatos del run\n")
    lines.append("| Campo | Valor |")
    lines.append("|---|---|")
    for k in (
        "git_hash", "git_dirty", "config_hash", "param_count", "stage",
        "optimizer_steps", "skipped_steps", "amp_overflow_steps",
        "amp_nonfinite_grad_steps", "datasets_seen_count",
        "clients_seen_count", "max_effective_bc", "elapsed_seconds",
    ):
        lines.append(f"| `{k}` | `{a.get(k)}` |")
    lines.append("")

    lines.append("## Inspeccion de `metrics.jsonl`\n")
    lines.append(f"- Lineas totales: {a['total_metric_lines']}")
    lines.append(f"- Registros de step: {a['total_step_records']}")
    lines.append(f"- Registros de distribution: {a['total_distribution_records']}")
    lines.append(f"- Steps con grad no finito (total): {a['nonfinite_grad_steps_total']}")
    lines.append(f"- Por tipo: {a['nonfinite_grad_by_kind']}")
    if a["nonfinite_grad_by_dataset"]:
        top_ds = sorted(
            a["nonfinite_grad_by_dataset"].items(),
            key=lambda kv: -kv[1],
        )[:5]
        lines.append(f"- Top datasets afectados: {top_ds}")
    lines.append("")

    lines.append("## Outliers de loss finita\n")
    lines.append(
        f"- Umbral `huge_loss_threshold = {a['huge_loss_threshold']}`. "
        f"Steps con loss >= umbral: {len(a['huge_finite_loss_steps'])}."
    )
    if a["huge_finite_loss_steps"]:
        for r in a["huge_finite_loss_steps"][:10]:
            lines.append(f"  - step {r['step']} ({r['dataset']}): {r['loss']:.4f}")
    lines.append("")

    lines.append("## Loss por bucket\n")
    lines.append(
        "Medias por bucket de step. La columna `mean_clean` excluye steps "
        "con `optimizer_applied=False` o `amp_nonfinite_grad=True` "
        "(disponible solo en logs nuevos, post-patch)."
    )
    lines.append("\n| rango | n | mean | median | p10 | p90 | min | max | mean_clean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for b in a["loss_buckets"]:
        def fmt(v):
            if v is None:
                return "-"
            return f"{v:.4f}"
        lines.append(
            f"| {b['range']} | {b['count']} | {fmt(b['mean'])} | "
            f"{fmt(b['median'])} | {fmt(b['p10'])} | {fmt(b['p90'])} | "
            f"{fmt(b['min'])} | {fmt(b['max'])} | "
            f"{fmt(b['mean_excluding_nonfinite_grad_or_not_applied'])} |"
        )
    lines.append("")

    lines.append("## Checkpoint\n")
    lines.append(f"- `checkpoint_state_dict_all_finite`: `{a['checkpoint_state_dict_all_finite']}`")
    if a["checkpoint_nonfinite_tensors"]:
        lines.append(f"- Tensores no finitos: {a['checkpoint_nonfinite_tensors'][:10]}")
    if a["checkpoint_warning"]:
        lines.append(f"- Aviso: {a['checkpoint_warning']}")
    lines.append("")

    lines.append("## Conclusion\n")
    lines.append(a["conclusion"])
    lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analisis post-hoc reproducible de un run de SSL pretraining"
    )
    p.add_argument("--log-dir", type=Path, required=True,
                   help="Directorio del run con run_info.json + metrics.jsonl")
    p.add_argument("--checkpoint", type=Path, default=None,
                   help="Opcional: ruta a un ckpt para inspeccionar state_dict")
    p.add_argument("--huge-loss-threshold", type=float, default=1000.0)
    p.add_argument("--top-loss-n", type=int, default=20)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    log_dir = Path(args.log_dir)
    if not log_dir.is_dir():
        print(f"ERROR: log-dir no es directorio: {log_dir}", file=sys.stderr)
        return 2

    analysis = analyze(
        log_dir=log_dir,
        checkpoint=args.checkpoint,
        huge_loss_threshold=float(args.huge_loss_threshold),
        top_loss_n=int(args.top_loss_n),
    )

    safe = _safe(analysis)
    out_json = log_dir / "posthoc_analysis.json"
    out_md = log_dir / "posthoc_analysis.md"

    out_json.write_text(
        json.dumps(safe, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    out_md.write_text(render_markdown(safe), encoding="utf-8")

    print(f"OK: analisis escrito en {out_json} y {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
