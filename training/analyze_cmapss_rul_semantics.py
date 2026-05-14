"""Analisis preparatorio de la semantica RUL de CMAPSS (no decide nada).

Objetivo:

- Inspeccionar los shards harmonizados de CMAPSS sin entrenar nada.
- Recolectar estadisticas del target (`target_window` o `target` segun el
  contrato del shard) para evaluar mas adelante si requiere transformacion
  (clamp, horizonte maximo, inversion de signo).
- Detectar si la harmonization procesada basta por si sola para inferir
  la semantica temporal del RUL, o si hay que volver al loader original
  de PHMD para hacerlo bien.

**Reglas estrictas**:

- NO transformar targets.
- NO decidir clamp ni horizonte maximo.
- NO entrenar nada.
- Salida en `--out-dir`: dos ficheros (`*.json` estricto y `*.md`).

CLI:

    python -m training.analyze_cmapss_rul_semantics \\
        --processed-root /content/drive/MyDrive/fm_fl_phmd/processed \\
        --dataset CMAPSS \\
        --out-dir results/downstream/cmapss_rul_semantics
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from training.phm_tar_reader import (
    find_shards,
    iter_samples_from_tar,
)


# ----------------------------------------------------------------------
# Helpers comunes
# ----------------------------------------------------------------------


def _stats_target(values: List[float]) -> Dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(v)]
    if not finite:
        return {
            "count": 0, "min": None, "max": None,
            "mean": None, "median": None,
            "n_negative": 0, "frac_negative": None,
        }
    n = len(finite)
    n_neg = sum(1 for v in finite if v < 0)
    return {
        "count":         n,
        "min":           min(finite),
        "max":           max(finite),
        "mean":          sum(finite) / n,
        "median":        statistics.median(finite),
        "n_negative":    n_neg,
        "frac_negative": n_neg / n if n else None,
    }


def _safe(obj):
    """Reduce a JSON estricto, igual que en analyze_ssl_full_run."""
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
# Extraccion del target por sample
# ----------------------------------------------------------------------


def _extract_target_from_sample(sample: Dict[str, Any]) -> Optional[float]:
    """Obtiene el target escalar del sample.

    El contrato escrito por la harmonization v0.5 guarda en
    `target.json` la clave `target_window` (politica 'ultimo_valor_valido').
    Si no esta, intentamos `target` como fallback.
    """
    tgt = sample.get("target", {})
    if not isinstance(tgt, dict):
        return None
    v = tgt.get("target_window")
    if v is None:
        v = tgt.get("target")
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _padding_ratio_window(sample: Dict[str, Any]) -> Optional[float]:
    """Fraccion de timesteps de padding en la ventana del sample.

    Devuelve None si no podemos calcularlo (vtm ausente).
    """
    vtm = sample.get("valid_time_mask")
    if vtm is None:
        return None
    try:
        W = int(vtm.shape[0])
        n_valid = int(vtm.sum())
        if W <= 0:
            return None
        return (W - n_valid) / W
    except Exception:
        return None


# ----------------------------------------------------------------------
# Recoleccion por split y por trayectoria
# ----------------------------------------------------------------------


def collect_from_shards(
    processed_root: Path, dataset: str, splits: Tuple[str, ...] = ("train", "val", "test"),
    max_samples_per_split: Optional[int] = None,
) -> Dict[str, Any]:
    """Recorre los shards y agrupa estadisticas por split / trajectory_id."""
    per_split_targets: Dict[str, List[float]] = defaultdict(list)
    per_split_padding: Dict[str, List[float]] = defaultdict(list)
    per_split_trajectories: Dict[str, Dict[str, List[Tuple[Optional[int], Optional[float]]]]] = {
        s: defaultdict(list) for s in splits
    }
    n_total_samples = 0
    n_samples_with_target = 0
    n_samples_without_target = 0
    splits_found: List[str] = []
    shards_used_by_split: Dict[str, List[str]] = {}

    for split in splits:
        shards = find_shards(processed_root, dataset, split)
        if not shards:
            continue
        splits_found.append(split)
        shards_used_by_split[split] = [str(s) for s in shards]
        n_read = 0
        for shard in shards:
            for sample in iter_samples_from_tar(shard, strict=True):
                n_total_samples += 1
                meta = sample.get("meta", {})
                traj_id = str(meta.get("trajectory_id") or "")
                idx_window = meta.get("idx_window")
                if isinstance(idx_window, (int, float)) and math.isfinite(idx_window):
                    idx_window = int(idx_window)
                else:
                    idx_window = None
                target = _extract_target_from_sample(sample)
                if target is not None:
                    n_samples_with_target += 1
                    per_split_targets[split].append(target)
                else:
                    n_samples_without_target += 1
                pad = _padding_ratio_window(sample)
                if pad is not None:
                    per_split_padding[split].append(pad)
                if traj_id:
                    per_split_trajectories[split][traj_id].append((idx_window, target))
                n_read += 1
                if max_samples_per_split and n_read >= max_samples_per_split:
                    break
            if max_samples_per_split and n_read >= max_samples_per_split:
                break

    # Estadisticas por split
    stats_by_split: Dict[str, Dict[str, Any]] = {}
    traj_stats_by_split: Dict[str, Dict[str, Any]] = {}
    for split in splits_found:
        stats_by_split[split] = _stats_target(per_split_targets[split])
        pad_vals = per_split_padding[split]
        if pad_vals:
            stats_by_split[split]["padding_ratio_window_mean"] = sum(pad_vals) / len(pad_vals)
            stats_by_split[split]["padding_ratio_window_max"] = max(pad_vals)
        else:
            stats_by_split[split]["padding_ratio_window_mean"] = None
            stats_by_split[split]["padding_ratio_window_max"] = None

        # Por trayectoria
        traj_dict = per_split_trajectories[split]
        n_traj = len(traj_dict)
        n_traj_multi_window = sum(1 for v in traj_dict.values() if len(v) > 1)
        traj_stats_by_split[split] = {
            "n_trajectories":           n_traj,
            "n_trajectories_multi_window": n_traj_multi_window,
            "frac_multi_window":        (n_traj_multi_window / n_traj) if n_traj else 0.0,
        }

        # Correlacion idx_window vs target dentro de cada trayectoria con
        # >1 ventana y idx_window valido. Devolvemos sign of slope a nivel
        # agregado: cuantas trayectorias muestran target decreciente con
        # idx_window (= RUL fisico decreciente con el tiempo, esperado en
        # CMAPSS), cuantas creciente, y cuantas planas.
        n_dec = 0
        n_inc = 0
        n_flat = 0
        n_eval = 0
        for tid, rows in traj_dict.items():
            with_idx = [(i, t) for i, t in rows if i is not None and t is not None]
            if len(with_idx) < 2:
                continue
            with_idx.sort(key=lambda r: r[0])
            i_first, t_first = with_idx[0]
            i_last, t_last = with_idx[-1]
            if i_last == i_first:
                continue
            delta = t_last - t_first
            n_eval += 1
            if abs(delta) < 1e-9:
                n_flat += 1
            elif delta < 0:
                n_dec += 1
            else:
                n_inc += 1
        traj_stats_by_split[split]["trajectories_evaluated_for_trend"] = n_eval
        traj_stats_by_split[split]["target_trend_decreasing"] = n_dec
        traj_stats_by_split[split]["target_trend_increasing"] = n_inc
        traj_stats_by_split[split]["target_trend_flat"] = n_flat

    return {
        "n_total_samples":         n_total_samples,
        "n_samples_with_target":   n_samples_with_target,
        "n_samples_without_target": n_samples_without_target,
        "splits_found":            splits_found,
        "shards_used_by_split":    shards_used_by_split,
        "stats_by_split":          stats_by_split,
        "trajectory_stats":        traj_stats_by_split,
    }


# ----------------------------------------------------------------------
# Diagnostico y narrativa
# ----------------------------------------------------------------------


def _build_diagnostic(stats: Dict[str, Any]) -> Dict[str, Any]:
    """Resumen ejecutivo para decidir si la harmonization basta."""
    enough_data_for_temporal_inference = False
    splits_with_multi_window: List[str] = []
    for split, ts in stats.get("trajectory_stats", {}).items():
        if ts.get("n_trajectories_multi_window", 0) >= 5:
            splits_with_multi_window.append(split)
            enough_data_for_temporal_inference = True

    has_negative_targets = False
    for split, st in stats.get("stats_by_split", {}).items():
        if st.get("n_negative", 0) > 0:
            has_negative_targets = True
            break

    return {
        "enough_data_for_temporal_inference": enough_data_for_temporal_inference,
        "splits_with_multi_window":           splits_with_multi_window,
        "has_negative_targets":               has_negative_targets,
        "notes": [
            "Si enough_data_for_temporal_inference es False, no hay ventanas "
            "suficientes por trayectoria en los shards procesados como para "
            "inferir la semantica temporal del RUL. Habria que volver al "
            "loader original de PHMD para esta decision.",
            "PROHIBIDO clamp automatico. La decision sobre clamp / horizonte "
            "maximo / inversion de signo requiere inspeccion adicional fuera "
            "de los shards y debe documentarse antes de cualquier downstream.",
        ],
    }


# ----------------------------------------------------------------------
# Render markdown
# ----------------------------------------------------------------------


def render_markdown(analysis: Dict[str, Any]) -> str:
    a = analysis
    lines: List[str] = []
    lines.append(f"# Analisis preparatorio: semantica RUL de `{a['dataset']}`\n")
    lines.append(
        "Este analisis se generó leyendo los shards harmonizados de "
        f"`{a['dataset']}` desde `{a['processed_root']}`. "
        "**No transforma targets ni decide ninguna politica.** Su objetivo "
        "es preparar la decision metodologica futura (clamp / horizonte "
        "maximo / inversion de signo del RUL), no resolverla.\n"
    )
    lines.append("## Manifest del dataset\n")
    if a.get("manifest"):
        m = a["manifest"]
        rel = ["dataset", "role", "evaluation_tier", "target_col",
               "target_policy", "target_warning", "padding_ratio",
               "n_units_total", "n_windows_total", "tail_policy",
               "audit_version", "pipeline_version"]
        for k in rel:
            if k in m:
                lines.append(f"- `{k}`: `{m[k]}`")
    else:
        lines.append("- (manifest no disponible en `processed_root`)")
    lines.append("")
    lines.append("## Conteo de samples leidos\n")
    lines.append(f"- Splits encontrados: `{a['stats']['splits_found']}`")
    lines.append(f"- Total samples: `{a['stats']['n_total_samples']}`")
    lines.append(f"- Con target legible: `{a['stats']['n_samples_with_target']}`")
    lines.append(f"- Sin target legible: `{a['stats']['n_samples_without_target']}`")
    lines.append("")
    lines.append("## Estadisticas del target por split\n")
    lines.append("| split | count | min | max | mean | median | n_neg | frac_neg | pad_mean | pad_max |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    def _fmt(v):
        if v is None:
            return "-"
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    for split, st in a["stats"]["stats_by_split"].items():
        lines.append(
            f"| {split} | {st['count']} | {_fmt(st['min'])} | {_fmt(st['max'])} | "
            f"{_fmt(st['mean'])} | {_fmt(st['median'])} | {st['n_negative']} | "
            f"{_fmt(st['frac_negative'])} | {_fmt(st['padding_ratio_window_mean'])} | "
            f"{_fmt(st['padding_ratio_window_max'])} |"
        )
    lines.append("")
    lines.append("## Estructura temporal por trayectoria\n")
    lines.append("Para cada split, cuantas trayectorias tienen >1 ventana y "
                 "como cambia el target entre la primera y la ultima ventana "
                 "ordenadas por `idx_window`.\n")
    lines.append("| split | n_traj | multi_window | frac_multi | trend_decr | trend_incr | trend_flat |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for split, ts in a["stats"]["trajectory_stats"].items():
        lines.append(
            f"| {split} | {ts['n_trajectories']} | "
            f"{ts['n_trajectories_multi_window']} | "
            f"{_fmt(ts['frac_multi_window'])} | "
            f"{ts['target_trend_decreasing']} | "
            f"{ts['target_trend_increasing']} | "
            f"{ts['target_trend_flat']} |"
        )
    lines.append("")
    lines.append("## Diagnostico\n")
    diag = a["diagnostic"]
    lines.append(f"- `enough_data_for_temporal_inference`: `{diag['enough_data_for_temporal_inference']}`")
    lines.append(f"- `splits_with_multi_window`: `{diag['splits_with_multi_window']}`")
    lines.append(f"- `has_negative_targets`: `{diag['has_negative_targets']}`")
    for n in diag.get("notes", []):
        lines.append(f"- {n}")
    lines.append("")
    if not diag["enough_data_for_temporal_inference"]:
        lines.append(
            "**Conclusion**: la harmonization procesada no basta para "
            "inferir la semantica RUL; se requiere inspeccionar raw PHMD "
            "o el loader original. **La decision RUL queda pendiente.**\n"
        )
    else:
        lines.append(
            "**Conclusion**: hay datos temporales suficientes en los shards. "
            "Aun asi, no se decide nada en este script; la transformacion "
            "RUL queda pendiente y debe documentarse explicitamente en "
            "`pending_downstream_and_sampling.md` antes del downstream.\n"
        )
    return "\n".join(lines)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analisis preparatorio de la semantica RUL de un dataset"
    )
    p.add_argument("--processed-root", type=Path, required=True,
                   help="Raiz de Drive con los datasets procesados (v0.5)")
    p.add_argument("--dataset", type=str, default="CMAPSS")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/downstream/cmapss_rul_semantics"))
    p.add_argument("--max-samples-per-split", type=int, default=None,
                   help="Para tests/diagnostico; limita lectura por split.")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    processed_root = Path(args.processed_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Manifest opcional
    manifest_path = processed_root / args.dataset / "manifest.json"
    manifest_dict: Optional[Dict[str, Any]] = None
    if manifest_path.is_file():
        try:
            manifest_dict = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as e:
            manifest_dict = {"error": f"no se pudo leer: {e}"}

    # Anti-leakage opcional (no lo procesamos pero lo registramos como referencia)
    anti_leakage_path = processed_root / args.dataset / "anti_leakage_report.json"
    anti_leakage_dict: Optional[Dict[str, Any]] = None
    if anti_leakage_path.is_file():
        try:
            anti_leakage_dict = json.loads(anti_leakage_path.read_text(encoding="utf-8"))
        except Exception as e:
            anti_leakage_dict = {"error": f"no se pudo leer: {e}"}

    # Recoleccion principal
    stats = collect_from_shards(
        processed_root=processed_root,
        dataset=args.dataset,
        max_samples_per_split=args.max_samples_per_split,
    )
    diagnostic = _build_diagnostic(stats)

    analysis: Dict[str, Any] = {
        "dataset":            args.dataset,
        "processed_root":     str(processed_root),
        "manifest":           manifest_dict,
        "anti_leakage":       anti_leakage_dict,
        "stats":              stats,
        "diagnostic":         diagnostic,
        "no_transformacion":  True,
        "decision_rul":       "PENDIENTE",
    }

    out_json = out_dir / "cmapss_rul_semantics.json"
    out_md   = out_dir / "cmapss_rul_semantics.md"
    out_json.write_text(
        json.dumps(_safe(analysis), ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    out_md.write_text(render_markdown(analysis), encoding="utf-8")
    print(f"OK: escrito {out_json} y {out_md}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
