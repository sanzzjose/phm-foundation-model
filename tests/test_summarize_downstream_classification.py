"""Tests del aggregator de resultados downstream classification primary.

Construye un layout sintetico `results/downstream/<ds>/<mode>/run_info.json`
y verifica:

- `best_mode_by_dataset` calcula bien el maximo por macro_f1.
- las notas automaticas se aplican (historical_uncapped_batch_v0_1 y
  catastrophic_forgetting).
- el JSON producido es estricto (allow_nan=False).
- el CSV tiene las columnas esperadas y serializa listas como JSON.
- runs ausentes se omiten sin romper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture
def fake_results(tmp_path: Path) -> Path:
    """Crea results/downstream/<ds>/<mode>/run_info.json sintetico."""
    root = tmp_path / "results" / "downstream"

    # CWRU: 3 modos completos + reintento full_ft. Post-fix (batch_size_effective presente).
    def _ri(macro_f1, mode, ds="CWRU", n_classes=4, bs_eff=64, nc=2):
        return {
            "run_name": f"downstream_{ds.lower()}_patchtst_base_{mode}",
            "config_hash": f"hash_{ds}_{mode}",
            "git_hash": "abc1234",
            "n_classes": n_classes,
            "best_epoch": 12,
            "best_value": macro_f1 + 0.01,
            "batch_size_effective": bs_eff,
            "effective_bc": (bs_eff * nc) if bs_eff is not None else None,
            "test_metrics": {
                "n_samples": 1000,
                "accuracy": macro_f1 + 0.02,
                "balanced_accuracy": macro_f1 + 0.01,
                "macro_f1": macro_f1,
                "labels_by_class_id": [str(i) for i in range(n_classes)],
            },
            "elapsed_seconds": 3000.0,
            "zero_support_classes_test": [],
        }

    for ds, modes in [
        ("CWRU", [("from_scratch", 0.3503), ("linear_probing", 0.7046),
                  ("full_finetuning", 0.8292)]),
        ("HSG18", [("from_scratch", 0.5693), ("linear_probing", 0.9056),
                   ("full_finetuning", 0.9504)]),
    ]:
        for mode, mf1 in modes:
            d = root / ds.lower() / mode
            d.mkdir(parents=True, exist_ok=True)
            (d / "run_info.json").write_text(
                json.dumps(_ri(mf1, mode, ds=ds)), encoding="utf-8"
            )

    # PHM18 historico: SIN batch_size_effective (run v0.1).
    for mode, mf1 in [("from_scratch", 0.3655), ("linear_probing", 0.2739),
                     ("full_finetuning", 0.3406)]:
        d = root / "phm18" / mode
        d.mkdir(parents=True, exist_ok=True)
        ri = _ri(mf1, mode, ds="PHM18", n_classes=3, bs_eff=None, nc=22)
        ri["batch_size_effective"] = None
        ri["effective_bc"] = None
        (d / "run_info.json").write_text(json.dumps(ri), encoding="utf-8")

    # PBCP16: full_ft mucho peor que from_scratch -> deberia disparar
    # catastrophic_forgetting (delta > 0.10).
    for mode, mf1 in [("from_scratch", 0.9074), ("full_finetuning", 0.5000)]:
        d = root / "pbcp16" / mode
        d.mkdir(parents=True, exist_ok=True)
        (d / "run_info.json").write_text(
            json.dumps(_ri(mf1, mode, ds="PBCP16", n_classes=5)),
            encoding="utf-8",
        )

    return root


# ----------------------------------------------------------------------


def test_collect_y_best(fake_results):
    from training.summarize_downstream_classification import (
        collect_rows, best_mode_by_dataset,
    )
    rows = collect_rows(
        fake_results,
        datasets=["CWRU", "HSG18", "PHM18", "PBCP16", "INEXISTENTE"],
        modes=["from_scratch", "linear_probing", "full_finetuning"],
    )
    # 3 (CWRU) + 3 (HSG18) + 3 (PHM18) + 2 (PBCP16) = 11
    assert len(rows) == 11

    best = best_mode_by_dataset(rows)
    assert best["CWRU"]["mode"] == "full_finetuning"
    assert best["CWRU"]["macro_f1"] == pytest.approx(0.8292)
    assert best["HSG18"]["mode"] == "full_finetuning"
    assert best["PHM18"]["mode"] == "from_scratch"
    assert best["PBCP16"]["mode"] == "from_scratch"


def test_notas_automaticas(fake_results):
    from training.summarize_downstream_classification import collect_rows
    rows = collect_rows(
        fake_results,
        datasets=["PHM18", "PBCP16", "CWRU"],
        modes=["from_scratch", "linear_probing", "full_finetuning"],
    )
    by_key = {(r["dataset"], r["mode"]): r for r in rows}

    # PHM18: batch_size_effective None y C=22 (default 64*22 > 512) -> nota.
    for mode in ("from_scratch", "linear_probing", "full_finetuning"):
        assert "historical_uncapped_batch_v0_1" in by_key[("PHM18", mode)]["notes"], \
            f"PHM18/{mode} deberia tener historical_uncapped_batch_v0_1"

    # CWRU: batch_size_effective presente -> no debe tener la nota.
    assert "historical_uncapped_batch_v0_1" not in by_key[("CWRU", "from_scratch")]["notes"]

    # PBCP16/full_finetuning con macro_f1=0.5 vs from_scratch=0.9074 -> delta > 0.10
    # -> catastrophic_forgetting.
    assert "catastrophic_forgetting" in by_key[("PBCP16", "full_finetuning")]["notes"]

    # CWRU/full_finetuning con macro_f1=0.8292 > from_scratch=0.3503 -> NO.
    assert "catastrophic_forgetting" not in by_key[("CWRU", "full_finetuning")]["notes"]


def test_escritura_json_csv_atomica(fake_results, tmp_path):
    from training.summarize_downstream_classification import (
        collect_rows, best_mode_by_dataset, write_json, write_csv,
    )
    rows = collect_rows(
        fake_results,
        datasets=["CWRU", "HSG18"],
        modes=["from_scratch", "linear_probing", "full_finetuning"],
    )
    best = best_mode_by_dataset(rows)
    out_json = tmp_path / "out.json"
    out_csv = tmp_path / "out.csv"
    write_json(rows, best, out_json)
    write_csv(rows, out_csv)
    assert out_json.is_file()
    assert out_csv.is_file()
    # JSON estricto: allow_nan=False (sin NaN/Infinity).
    payload = json.loads(out_json.read_text(encoding="utf-8"))
    assert payload["version"] == 1
    assert "results" in payload
    assert "best_mode_by_dataset" in payload
    # No debe haber NaN serializado como string.
    raw = out_json.read_text(encoding="utf-8")
    assert "NaN" not in raw
    assert "Infinity" not in raw
    # CSV: cabecera con todas las columnas.
    head = out_csv.read_text(encoding="utf-8").splitlines()[0]
    for col in ("dataset", "mode", "macro_f1", "batch_size_effective", "notes"):
        assert col in head


def test_cli_smoke(fake_results, tmp_path, capsys):
    from training.summarize_downstream_classification import main
    rc = main([
        "--results-root", str(fake_results),
        "--datasets", "CWRU,HSG18,PHM18,PBCP16",
        "--modes", "from_scratch,linear_probing,full_finetuning",
    ])
    assert rc == 0
    captured = capsys.readouterr()
    assert "filas" in captured.out
    assert "best_mode_by_dataset" in captured.out


def test_runs_ausentes_no_rompen(tmp_path):
    """Si results_root no existe, main devuelve 1 sin excepcion."""
    from training.summarize_downstream_classification import main
    rc = main([
        "--results-root", str(tmp_path / "no_existe"),
        "--datasets", "CWRU",
        "--modes", "from_scratch",
    ])
    assert rc == 1
