"""Agregado unico de resultados downstream classification primary.

Lee `results/downstream/<dataset>/<mode>/run_info.json` para todos los
TT primary classification del MVP (CWRU, HSG18, PBCP16, PHM18) y los
modos disponibles, y produce dos artefactos consolidados:

- `results/downstream/summary_classification_primary.json`
- `results/downstream/summary_classification_primary.csv`

Incluye:

- metricas test agregadas y per-class si estan en el run_info;
- batch_size_effective y effective_bc cuando el run los tiene; null si
  el run es historico (anterior al fix de batch adaptativo);
- zero_support_classes_test si esta;
- best_mode_by_dataset calculado por macro_f1 maximo;
- notes automaticas:
    * "historical_uncapped_batch_v0_1" si el run carece de
      batch_size_effective y el dataset tiene n_channels conocido > 2.
    * "catastrophic_forgetting" si modo es full_finetuning y macro_f1
      esta MUY por debajo del baseline from_scratch del mismo dataset.

Modo CLI:

    python -m training.summarize_downstream_classification \\
        [--results-root results/downstream] \\
        [--datasets CWRU,HSG18,PBCP16,PHM18]

Sin args usa los defaults. Salida atomica via write+rename para no
corromper si falla a mitad.

Sin emojis, sin firmas IA.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


# Defaults sobre el MVP. Si en el futuro se anaden TT primary, ampliar aqui.
DEFAULT_DATASETS = ("CWRU", "HSG18", "PBCP16", "PHM18")
DEFAULT_MODES = ("from_scratch", "linear_probing", "full_finetuning", "full_finetuning_lr1e-5")
DEFAULT_RESULTS_ROOT = Path("results/downstream")


# ----------------------------------------------------------------------
# Helpers internos
# ----------------------------------------------------------------------


def _json_safe(o: Any) -> Any:
    """Convierte tipos no serializables (Path, set, numpy) a primitivos."""
    if o is None or isinstance(o, (bool, int, float, str)):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, set):
        return [_json_safe(v) for v in sorted(o)]
    if isinstance(o, Path):
        return str(o)
    try:
        return _json_safe(o.item())  # numpy scalar
    except Exception:
        return repr(o)


def _atomic_write_text(path: Path, text: str) -> None:
    """Escritura atomica: write to .tmp + rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


# ----------------------------------------------------------------------
# Carga de un run_info y normalizacion
# ----------------------------------------------------------------------


def load_run_info(run_info_path: Path) -> Optional[Dict[str, Any]]:
    """Lee `run_info.json` y devuelve dict; None si no existe o ilegible."""
    if not run_info_path.is_file():
        return None
    try:
        return json.loads(run_info_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def extract_row(
    dataset: str, mode: str, run_info: Dict[str, Any]
) -> Dict[str, Any]:
    """Construye una fila del agregado a partir de un run_info."""
    tm = run_info.get("test_metrics") or {}
    per_class = tm.get("per_class") if isinstance(tm, dict) else None

    row = {
        "dataset":               dataset,
        "mode":                  mode,
        "run_name":              run_info.get("run_name"),
        "config_hash":           run_info.get("config_hash"),
        "git_hash":              run_info.get("git_hash"),
        "n_classes":             run_info.get("n_classes"),
        "n_samples_test":        tm.get("n_samples") if isinstance(tm, dict) else None,
        "accuracy":              tm.get("accuracy") if isinstance(tm, dict) else None,
        "balanced_accuracy":     tm.get("balanced_accuracy") if isinstance(tm, dict) else None,
        "macro_f1":              tm.get("macro_f1") if isinstance(tm, dict) else None,
        "best_epoch":            run_info.get("best_epoch"),
        "best_value":            run_info.get("best_value"),
        "batch_size_effective":  run_info.get("batch_size_effective"),  # null en runs historicos
        "effective_bc":          run_info.get("effective_bc"),
        "zero_support_classes_test": run_info.get("zero_support_classes_test"),
        "per_class":             per_class,
        "labels_by_class_id":    tm.get("labels_by_class_id") if isinstance(tm, dict) else None,
        "elapsed_seconds":       run_info.get("elapsed_seconds"),
        "notes":                 [],
    }
    return row


def annotate_rows(rows: List[Dict[str, Any]]) -> None:
    """Anota notas automaticas en cada fila (in-place).

    - `historical_uncapped_batch_v0_1`: si `batch_size_effective` es None
      (run historico) Y `n_channels`, si lo conocemos, da `B*C > 512`
      con `batch_size=64`. Como n_channels en runs historicos no esta
      en el run_info, lo inferimos con la regla: PHM18 sabemos que es 22.
      Si no podemos inferir, omitimos la nota.
    - `catastrophic_forgetting`: si `mode == "full_finetuning"` y su
      `macro_f1` es < `from_scratch.macro_f1 - 0.10` del mismo dataset.
    """
    # Indexar por (dataset, mode) -> row
    by_key = {(r["dataset"], r["mode"]): r for r in rows}

    # n_channels conocidos para PHM18 (22) por documentacion. CWRU=2,
    # HSG18=1, PBCP16=1. Si en el futuro hay un run historico con un
    # dataset desconocido, esta tabla habria que extenderla.
    KNOWN_C = {"CWRU": 2, "HSG18": 1, "PBCP16": 1, "PHM18": 22}
    BATCH_DEFAULT_V01 = 64
    CAP_DEFAULT = 512

    for r in rows:
        # 1) historical_uncapped_batch_v0_1
        if r["batch_size_effective"] is None:
            ds = r["dataset"]
            nc = KNOWN_C.get(ds)
            if nc is not None and BATCH_DEFAULT_V01 * nc > CAP_DEFAULT:
                r["notes"].append("historical_uncapped_batch_v0_1")

        # 2) catastrophic_forgetting
        if r["mode"] == "full_finetuning":
            fs = by_key.get((r["dataset"], "from_scratch"))
            if (fs is not None and fs.get("macro_f1") is not None
                    and r.get("macro_f1") is not None
                    and r["macro_f1"] < fs["macro_f1"] - 0.10):
                r["notes"].append("catastrophic_forgetting")


def best_mode_by_dataset(rows: Sequence[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Devuelve `{dataset: {mode, macro_f1}}` con el mejor por macro_f1."""
    best: Dict[str, Dict[str, Any]] = {}
    for r in rows:
        mf1 = r.get("macro_f1")
        if mf1 is None:
            continue
        ds = r["dataset"]
        cur = best.get(ds)
        if cur is None or mf1 > cur["macro_f1"]:
            best[ds] = {
                "mode": r["mode"],
                "macro_f1": mf1,
                "balanced_accuracy": r.get("balanced_accuracy"),
                "accuracy": r.get("accuracy"),
                "best_epoch": r.get("best_epoch"),
                "config_hash": r.get("config_hash"),
                "run_name": r.get("run_name"),
            }
    return best


# ----------------------------------------------------------------------
# Construccion del agregado
# ----------------------------------------------------------------------


def collect_rows(
    results_root: Path, datasets: Sequence[str], modes: Sequence[str]
) -> List[Dict[str, Any]]:
    """Recorre `results_root/<ds>/<mode>/run_info.json` y devuelve filas."""
    rows: List[Dict[str, Any]] = []
    for ds in datasets:
        ds_dir = results_root / ds.lower()
        if not ds_dir.is_dir():
            continue
        for mode in modes:
            ri = load_run_info(ds_dir / mode / "run_info.json")
            if ri is None:
                continue
            rows.append(extract_row(ds, mode, ri))
    annotate_rows(rows)
    return rows


# ----------------------------------------------------------------------
# Serializacion
# ----------------------------------------------------------------------


CSV_FIELDS = [
    "dataset",
    "mode",
    "run_name",
    "config_hash",
    "git_hash",
    "n_classes",
    "n_samples_test",
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "best_epoch",
    "best_value",
    "batch_size_effective",
    "effective_bc",
    "zero_support_classes_test",
    "elapsed_seconds",
    "notes",
]


def write_csv(rows: Sequence[Dict[str, Any]], out_path: Path) -> None:
    """CSV plano con columnas escalares. Listas se serializan como JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for r in rows:
            payload = {}
            for k in CSV_FIELDS:
                v = r.get(k)
                if isinstance(v, (list, dict)):
                    payload[k] = json.dumps(_json_safe(v), ensure_ascii=False)
                else:
                    payload[k] = v
            writer.writerow(payload)
    os.replace(tmp, out_path)


def write_json(
    rows: Sequence[Dict[str, Any]],
    best: Dict[str, Dict[str, Any]],
    out_path: Path,
    extras: Optional[Dict[str, Any]] = None,
) -> None:
    payload = {
        "version": 1,
        "results": [_json_safe(r) for r in rows],
        "best_mode_by_dataset": _json_safe(best),
    }
    if extras:
        payload["extras"] = _json_safe(extras)
    _atomic_write_text(
        out_path,
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False),
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregator downstream classification primary")
    p.add_argument(
        "--results-root", type=Path, default=DEFAULT_RESULTS_ROOT,
        help=f"raiz de results/downstream/ (default {DEFAULT_RESULTS_ROOT})",
    )
    p.add_argument(
        "--datasets", type=str, default=",".join(DEFAULT_DATASETS),
        help="lista coma-separada de TT primary (default CWRU,HSG18,PBCP16,PHM18)",
    )
    p.add_argument(
        "--modes", type=str, default=",".join(DEFAULT_MODES),
        help="modos a buscar (default los 4 conocidos)",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    rows = collect_rows(args.results_root, datasets, modes)
    if not rows:
        print(f"No se encontraron runs en {args.results_root}", file=sys.stderr)
        return 1
    best = best_mode_by_dataset(rows)
    out_json = args.results_root / "summary_classification_primary.json"
    out_csv = args.results_root / "summary_classification_primary.csv"
    write_json(rows, best, out_json)
    write_csv(rows, out_csv)
    print(f"OK: {len(rows)} filas")
    print(f"  -> {out_json}")
    print(f"  -> {out_csv}")
    print(f"  best_mode_by_dataset:")
    for ds, b in sorted(best.items()):
        print(f"    {ds:<12} {b['mode']:<20} macro_f1={b['macro_f1']:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
