"""Tests de la politica de sampling (`training.sampling`).

Comprueba contra los datos reales de `results/processed_summary.csv` y
`results/audit/audit_summary.json` que:

- Solo se usan PRETRAIN_SOURCE.
- 36 datasets, 10 clientes.
- Sumas de pesos finales = 1 (por dataset y por cliente).
- Ningun dataset supera el cap 0.10.
- Ningun cliente supera el cap 0.25 (salvo imposibilidad matematica
  explicita).
- cnc_milling alcanza el min_client_presence = 0.005.
- PHM10 (dominante: 22.6% raw) queda en el cap 0.10.
- TRANSFER_TARGET ausentes del plan.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from training.sampling import (
    CAP_MAX_CLIENT_WEIGHT,
    CAP_MAX_DATASET_WEIGHT,
    MIN_CLIENT_PRESENCE,
    compute_pretraining_sampling_plan,
    load_sources,
)


_REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def sources():
    proc, cli, asum = load_sources(
        _REPO_ROOT / "results/processed_summary.csv",
        _REPO_ROOT / "results/client_summary.csv",
        _REPO_ROOT / "results/audit/audit_summary.json",
    )
    return proc, cli, asum


@pytest.fixture(scope="module")
def plan(sources):
    proc, cli, asum = sources
    return compute_pretraining_sampling_plan(proc, cli, asum)


# ----------------------------------------------------------------------
# Plan: tamano, roles, clientes
# ----------------------------------------------------------------------


def test_plan_only_pretrain_source(sources, plan):
    proc, _, _ = sources
    role_by_ds = {r["dataset"]: r["role"] for r in proc}
    for row in plan:
        assert role_by_ds[row["dataset"]] == "PRETRAIN_SOURCE", (
            f"{row['dataset']} no es PRETRAIN_SOURCE"
        )


def test_plan_has_36_datasets(plan):
    assert len(plan) == 36


def test_plan_has_10_clients(plan):
    clients = sorted({row["client"] for row in plan})
    assert len(clients) == 10
    # Topologia cerrada del audit v2.3 (sec 7.bis CLAUDE.md)
    expected = {
        "aero_engines", "batteries", "bearings", "cnc_milling", "gearboxes",
        "hdd", "misc", "misc_industrial", "phm_challenges", "wind",
    }
    assert set(clients) == expected


def test_no_transfer_target_in_plan(sources, plan):
    proc, _, _ = sources
    tt_names = {r["dataset"] for r in proc if r["role"] == "TRANSFER_TARGET"}
    plan_names = {row["dataset"] for row in plan}
    assert tt_names.isdisjoint(plan_names), "Hay TRANSFER_TARGET en el plan"


# ----------------------------------------------------------------------
# Sumas
# ----------------------------------------------------------------------


def test_dataset_weights_sum_to_one(plan):
    s = sum(row["final_dataset_weight"] for row in plan)
    assert abs(s - 1.0) < 1e-6, f"Suma pesos dataset = {s}"


def test_client_weights_sum_to_one(plan):
    seen = set()
    s = 0.0
    for row in plan:
        c = row["client"]
        if c not in seen:
            seen.add(c)
            s += row["final_client_weight"]
    assert abs(s - 1.0) < 1e-6, f"Suma pesos cliente = {s}"


# ----------------------------------------------------------------------
# Caps
# ----------------------------------------------------------------------


def test_no_dataset_exceeds_cap(plan):
    max_ds = max(row["final_dataset_weight"] for row in plan)
    assert max_ds <= CAP_MAX_DATASET_WEIGHT + 1e-6, (
        f"max_ds={max_ds} > cap={CAP_MAX_DATASET_WEIGHT}"
    )


def test_no_client_exceeds_cap(plan):
    seen = set()
    weights = []
    for row in plan:
        c = row["client"]
        if c not in seen:
            seen.add(c)
            weights.append(row["final_client_weight"])
    max_cl = max(weights)
    assert max_cl <= CAP_MAX_CLIENT_WEIGHT + 1e-6, (
        f"max_cl={max_cl} > cap={CAP_MAX_CLIENT_WEIGHT}"
    )


def test_phm10_capped(plan):
    """PHM10 tiene 22.6% raw → debe quedar en el cap 0.10."""
    phm10 = next(row for row in plan if row["dataset"] == "PHM10")
    assert phm10["final_dataset_weight"] <= CAP_MAX_DATASET_WEIGHT + 1e-6
    assert phm10["capped_dataset"] is True


# ----------------------------------------------------------------------
# min_client_presence
# ----------------------------------------------------------------------


def test_cnc_milling_min_presence(plan):
    cnc = next(row for row in plan if row["client"] == "cnc_milling")
    assert cnc["final_client_weight"] >= MIN_CLIENT_PRESENCE - 1e-9, (
        f"cnc_milling final_client_weight={cnc['final_client_weight']} "
        f"< min={MIN_CLIENT_PRESENCE}"
    )
    assert cnc["min_presence_applied"] is True


# ----------------------------------------------------------------------
# Pesos por dataset proporcionales a n_channel_patches dentro de cada cliente
# ----------------------------------------------------------------------


def test_intra_client_proportional_to_channel_patches(plan):
    """Dentro de un cliente sin caps, los datasets se reparten en proporcion
    a su n_channel_patches.

    Caso testeable: cnc_milling tiene 1 solo dataset (NMILL), por tanto su
    peso es exactamente el del cliente. Verifico ese ratio.
    """
    cnc_rows = [r for r in plan if r["client"] == "cnc_milling"]
    assert len(cnc_rows) == 1
    assert cnc_rows[0]["dataset"] == "NMILL"
    assert (
        abs(cnc_rows[0]["final_dataset_weight"] - cnc_rows[0]["final_client_weight"])
        < 1e-6
    )


# ----------------------------------------------------------------------
# Flags
# ----------------------------------------------------------------------


def test_flags_present(plan):
    """Cada row debe tener los flags booleanos especificados."""
    for row in plan:
        assert isinstance(row["capped_dataset"], bool)
        assert isinstance(row["capped_client"], bool)
        assert isinstance(row["min_presence_applied"], bool)


# ----------------------------------------------------------------------
# min_dataset_presence (nuevo en full)
# ----------------------------------------------------------------------


def test_min_dataset_presence_lifts_small_datasets(sources):
    """Activado a 0.001, todos los datasets con peso < 0.001 suben al piso."""
    proc, cli, asum = sources
    plan = compute_pretraining_sampling_plan(
        proc, cli, asum, min_dataset_presence=0.001
    )
    # Ningun dataset queda por debajo del piso (modulo tolerancia FP).
    min_w = min(row["final_dataset_weight"] for row in plan)
    assert min_w >= 0.001 - 1e-9, f"min_w={min_w} < 0.001"

    # Los que subieron tienen el flag min_dataset_presence_applied=True.
    floored = [r for r in plan if r["min_dataset_presence_applied"]]
    assert len(floored) > 0, "Ningun dataset floored, pero hay algunos < 0.001 sin piso"

    # Suma sigue siendo 1.
    s = sum(row["final_dataset_weight"] for row in plan)
    assert abs(s - 1.0) < 1e-6

    # Cap por dataset sigue respetado (puede bajar un poco por la
    # redistribucion pero no superar el cap).
    max_w = max(row["final_dataset_weight"] for row in plan)
    assert max_w <= 0.10 + 1e-6


def test_min_dataset_presence_default_is_zero(plan):
    """Sin pasar min_dataset_presence, no se aplica el piso."""
    for row in plan:
        assert row["min_dataset_presence_applied"] is False


def test_min_dataset_presence_rechaza_imposible(sources):
    """Si min_dataset_presence * n_datasets > 1, debe levantar ValueError."""
    proc, cli, asum = sources
    # 36 datasets * 0.03 = 1.08 > 1.0
    with pytest.raises(ValueError, match="imposible"):
        compute_pretraining_sampling_plan(
            proc, cli, asum, min_dataset_presence=0.03
        )


# ----------------------------------------------------------------------
# Consistencia final_client_weight tras min_dataset_presence
# ----------------------------------------------------------------------


def test_final_client_weight_matches_grouped_dataset_weights_with_min_dataset_presence(sources):
    """Tras aplicar el piso por dataset, final_client_weight debe ser
    exactamente la suma de final_dataset_weight de los datasets del
    cliente. Antes del fix, conservaba un valor pre-piso y podia
    divergir hasta ~0.007 en este corpus.
    """
    proc, cli, asum = sources
    plan = compute_pretraining_sampling_plan(
        proc, cli, asum, min_dataset_presence=0.001
    )
    # Agrupar por cliente sumando final_dataset_weight
    sum_by_client = {}
    fcw_by_client = {}
    for row in plan:
        c = row["client"]
        sum_by_client[c] = sum_by_client.get(c, 0.0) + row["final_dataset_weight"]
        if c not in fcw_by_client:
            fcw_by_client[c] = row["final_client_weight"]
        else:
            # Mismo cliente debe reportar el MISMO final_client_weight en
            # todas sus filas (no fluctua dataset a dataset).
            assert abs(row["final_client_weight"] - fcw_by_client[c]) < 1e-9

    # final_client_weight == groupby(final_dataset_weight) bit-a-bit (modulo FP).
    for c, expected in sum_by_client.items():
        observed = fcw_by_client[c]
        assert abs(observed - expected) < 1e-9, (
            f"Cliente {c}: groupby={expected:.9f} vs "
            f"final_client_weight={observed:.9f}"
        )


def test_final_client_weight_consistent_without_piso(sources, plan):
    """Sin piso (default 0.0), la igualdad final_client_weight ==
    sum(final_dataset_weight por cliente) tambien debe cumplirse."""
    sum_by_client = {}
    fcw_by_client = {}
    for row in plan:
        c = row["client"]
        sum_by_client[c] = sum_by_client.get(c, 0.0) + row["final_dataset_weight"]
        fcw_by_client[c] = row["final_client_weight"]
    for c, expected in sum_by_client.items():
        observed = fcw_by_client[c]
        assert abs(observed - expected) < 1e-9, (
            f"Cliente {c}: groupby={expected:.9f} vs "
            f"final_client_weight={observed:.9f} (sin piso)"
        )
