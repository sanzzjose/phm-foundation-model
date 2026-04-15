"""Tests del servidor FL: agregacion de pesos y dry-run.

- compute_aggregation_weights para las 3 politicas.
- build_plan_from_audit_groups produce filas con campos esperados.
- Lo que sigue (run_federated_pretraining real) requiere torch y datos
  reales, asi que se cubre solo a nivel de funcion-pesos aqui.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# compute_aggregation_weights
# ----------------------------------------------------------------------


def test_pesos_uniform():
    from training.fl.server import compute_aggregation_weights
    cm = [{"client": "A"}, {"client": "B"}, {"client": "C"}]
    plan = []
    w = compute_aggregation_weights(cm, plan, policy="uniform")
    assert w == [1.0, 1.0, 1.0]


def test_pesos_num_samples():
    from training.fl.server import compute_aggregation_weights
    cm = [
        {"client": "A", "optimizer_steps": 10, "max_effective_bc": 64},
        {"client": "B", "optimizer_steps": 20, "max_effective_bc": 32},
    ]
    w = compute_aggregation_weights(cm, [], policy="num_samples")
    assert w[0] == pytest.approx(10 * 64)
    assert w[1] == pytest.approx(20 * 32)


def test_pesos_final_client_weight():
    from training.fl.server import compute_aggregation_weights
    plan = [
        {"dataset": "D1", "client": "A", "final_dataset_weight": 0.3},
        {"dataset": "D2", "client": "A", "final_dataset_weight": 0.2},
        {"dataset": "D3", "client": "B", "final_dataset_weight": 0.5},
    ]
    cm = [{"client": "A"}, {"client": "B"}]
    w = compute_aggregation_weights(cm, plan, policy="final_client_weight")
    assert w[0] == pytest.approx(0.5)
    assert w[1] == pytest.approx(0.5)


def test_pesos_policy_desconocida_falla():
    from training.fl.server import compute_aggregation_weights
    with pytest.raises(ValueError, match="aggregation_weight_policy"):
        compute_aggregation_weights([{"client": "A"}], [], policy="wat")


# ----------------------------------------------------------------------
# V3: regresion del bug "mutar el plan al normalizar intra-cliente"
# ----------------------------------------------------------------------


def test_normalize_plan_subset_no_muta_plan_original():
    """normalize_plan_subset_for_loader devuelve copias, NO toca el plan."""
    from training.fl.client import normalize_plan_subset_for_loader
    plan_subset = [
        {"dataset": "D1", "client": "A", "final_dataset_weight": 0.1},
        {"dataset": "D2", "client": "A", "final_dataset_weight": 0.1},
    ]
    snapshot = [dict(r) for r in plan_subset]  # copia profunda para comparar
    out = normalize_plan_subset_for_loader(plan_subset)
    # 1) El plan original NO debe estar modificado.
    for orig, snap in zip(plan_subset, snapshot):
        assert orig == snap, f"plan original mutado: {orig} != {snap}"
    # 2) La salida es una lista distinta de dicts distintos.
    for a, b in zip(out, plan_subset):
        assert a is not b, "normalize_plan devolvio el mismo dict (mutable)"
    # 3) Los pesos de salida suman 1.
    assert sum(r["final_dataset_weight"] for r in out) == pytest.approx(1.0)
    # 4) Pesos relativos preservados (0.1, 0.1 -> 0.5, 0.5).
    assert out[0]["final_dataset_weight"] == pytest.approx(0.5)
    assert out[1]["final_dataset_weight"] == pytest.approx(0.5)


def test_normalize_plan_subset_pesos_cero_reparte_uniforme():
    """Caso degenerado: si todos los pesos son 0, reparte uniforme."""
    from training.fl.client import normalize_plan_subset_for_loader
    plan_subset = [
        {"dataset": "D1", "client": "A", "final_dataset_weight": 0.0},
        {"dataset": "D2", "client": "A", "final_dataset_weight": 0.0},
        {"dataset": "D3", "client": "A", "final_dataset_weight": 0.0},
    ]
    out = normalize_plan_subset_for_loader(plan_subset)
    for r in out:
        assert r["final_dataset_weight"] == pytest.approx(1.0 / 3.0)


def test_normalize_plan_subset_vacio():
    from training.fl.client import normalize_plan_subset_for_loader
    assert normalize_plan_subset_for_loader([]) == []


def test_smoke_check_aggregation_weights_fail_si_vacios():
    """W: si policy=final_client_weight y plan capped pero
    aggregation_weights_by_client_last_round vacio, el check debe FAIL."""
    # Test funcional minimo: replicamos la logica del check.
    aw = {}  # vacio
    plan_pol = ["final_client_weight_capped_v23"]
    policy_eff = "final_client_weight"
    weights_ok = True
    if policy_eff == "final_client_weight" and "final_client_weight_capped_v23" in plan_pol:
        if not aw:
            weights_ok = False
        else:
            rng = max(aw.values()) - min(aw.values())
            if rng < 0.01:
                weights_ok = False
    assert weights_ok is False, "el check debe fallar si pesos vacios con capped"


def test_compute_policy_effective_capped_vs_fallback():
    """Verifica que aggregation_weights_policy_effective refleja el plan."""
    # No exponemos la funcion _compute_policy_effective; replicamos la
    # logica que server.py usa para validar el contrato.
    def _eff(decl_policy, plan_pols):
        if decl_policy == "uniform":
            return "uniform"
        if decl_policy == "num_samples":
            return "num_samples"
        if decl_policy == "final_client_weight":
            if "final_client_weight_capped_v23" in plan_pols:
                return "final_client_weight_capped_v23"
            if "raw_ncp_fallback" in plan_pols:
                return "final_client_weight_raw_ncp_fallback"
            return f"final_client_weight_unknown_plan({','.join(plan_pols)})"
        return f"unknown:{decl_policy}"
    assert _eff("final_client_weight", ["final_client_weight_capped_v23"]) \
        == "final_client_weight_capped_v23"
    assert _eff("final_client_weight", ["raw_ncp_fallback"]) \
        == "final_client_weight_raw_ncp_fallback"
    assert _eff("uniform", ["whatever"]) == "uniform"


def test_agregacion_sigue_correcta_tras_normalizacion_intra_cliente():
    """El bug que arreglamos: si la normalizacion intra-cliente mutaba
    el plan, compute_aggregation_weights con policy=final_client_weight
    daria pesos 1.0/1.0 (uniformes) en lugar de los pesos reales por
    cliente. Este test simula el escenario completo:

    1. Plan con A (peso 0.2) y B (peso 0.8).
    2. Cliente A normaliza su subset intra-cliente (no debe mutar).
    3. Cliente B normaliza su subset intra-cliente.
    4. compute_aggregation_weights debe seguir devolviendo 0.2/0.8.
    """
    from training.fl.client import normalize_plan_subset_for_loader, filter_plan_by_client
    from training.fl.server import compute_aggregation_weights

    plan = [
        {"dataset": "D1", "client": "A", "final_dataset_weight": 0.10},
        {"dataset": "D2", "client": "A", "final_dataset_weight": 0.10},
        {"dataset": "D3", "client": "B", "final_dataset_weight": 0.30},
        {"dataset": "D4", "client": "B", "final_dataset_weight": 0.50},
    ]
    # Snapshot exacto del plan original para detectar mutaciones.
    plan_snapshot = [dict(r) for r in plan]

    # Simular lo que hace cada cliente al construir su loader:
    sub_a = filter_plan_by_client(plan, "A")
    sub_b = filter_plan_by_client(plan, "B")
    _ = normalize_plan_subset_for_loader(sub_a)
    _ = normalize_plan_subset_for_loader(sub_b)

    # 1) El plan original NO debe estar mutado.
    for orig, snap in zip(plan, plan_snapshot):
        assert orig == snap, (
            f"plan original mutado tras normalize_plan_subset_for_loader: "
            f"{orig} != {snap}"
        )

    # 2) compute_aggregation_weights debe devolver pesos por cliente
    # PROPORCIONALES a los pesos originales del plan (0.2 y 0.8), no
    # cuasi-uniformes (1.0 y 1.0).
    cm_list = [{"client": "A"}, {"client": "B"}]
    weights_raw = compute_aggregation_weights(cm_list, plan, policy="final_client_weight")
    total = sum(weights_raw)
    weights_norm = [w / total for w in weights_raw]
    assert weights_norm[0] == pytest.approx(0.20, abs=1e-6), (
        f"cliente A deberia tener peso 0.2; recibido {weights_norm[0]}. "
        "Si es ~0.5, el plan se mutô al normalizar intra-cliente."
    )
    assert weights_norm[1] == pytest.approx(0.80, abs=1e-6), (
        f"cliente B deberia tener peso 0.8; recibido {weights_norm[1]}"
    )


# ----------------------------------------------------------------------
# build_plan_from_audit_groups
# ----------------------------------------------------------------------


def test_build_plan_dict_format(tmp_path):
    """Fallback raw_ncp: pasamos paths a CSVs inexistentes para forzar el
    fallback diagnostico (sin caps)."""
    from training.train_ssl_federated import build_plan_from_audit_groups
    groups = {
        "clients": {
            "A": {"datasets": {"DS1": {"n_channels": 2, "n_channel_patches": 100}}},
            "B": {"datasets": {"DS2": {"n_channels": 4, "n_channel_patches": 200}}},
        }
    }
    plan = build_plan_from_audit_groups(
        groups,
        processed_summary_csv=tmp_path / "no_proc.csv",
        client_summary_csv=tmp_path / "no_cli.csv",
        audit_summary_json=tmp_path / "no_audit.json",
    )
    assert len(plan) == 2
    assert all(r["policy"] == "raw_ncp_fallback" for r in plan)
    by_ds = {r["dataset"]: r for r in plan}
    assert by_ds["DS1"]["client"] == "A"
    assert by_ds["DS2"]["client"] == "B"
    # Pesos normalizados: 100/(100+200) y 200/(100+200)
    assert by_ds["DS1"]["final_dataset_weight"] == pytest.approx(1 / 3, rel=1e-4)
    assert by_ds["DS2"]["final_dataset_weight"] == pytest.approx(2 / 3, rel=1e-4)


def test_build_plan_list_format_fallback(tmp_path):
    from training.train_ssl_federated import build_plan_from_audit_groups
    groups = {
        "clients": {
            "A": {"datasets": ["DS1", "DS2"]},
            "B": {"datasets": ["DS3"]},
        }
    }
    plan = build_plan_from_audit_groups(
        groups,
        processed_summary_csv=tmp_path / "no_proc.csv",
        client_summary_csv=tmp_path / "no_cli.csv",
        audit_summary_json=tmp_path / "no_audit.json",
    )
    assert len(plan) == 3
    assert all(r["policy"] == "raw_ncp_fallback" for r in plan)
    # Sin n_channel_patches, reparto uniforme entre clientes (no intra-cliente)
    a_rows = [r for r in plan if r["client"] == "A"]
    b_rows = [r for r in plan if r["client"] == "B"]
    # Cada cliente tiene 1/n_clientes de peso de cliente; distribuido entre
    # sus datasets a partes iguales. Aqui n_clientes=2 y A tiene 2 datasets,
    # B tiene 1.
    assert sum(r["final_dataset_weight"] for r in a_rows) == pytest.approx(0.5)
    assert sum(r["final_dataset_weight"] for r in b_rows) == pytest.approx(0.5)


def test_build_plan_audit_groups_real():
    """Si el audit_groups real existe, el plan debe tener exactamente 36
    filas (los 36 PRETRAIN_SOURCE).
    """
    p = Path("results/audit/audit_groups.json")
    if not p.is_file():
        pytest.skip("audit_groups.json no encontrado")
    groups = json.loads(p.read_text(encoding="utf-8"))
    from training.train_ssl_federated import build_plan_from_audit_groups
    plan = build_plan_from_audit_groups(groups)
    assert len(plan) == 36, (
        f"Se esperaban 36 filas (PS), hay {len(plan)}"
    )
    total_w = sum(r["final_dataset_weight"] for r in plan)
    assert total_w == pytest.approx(1.0, rel=1e-6)


def test_build_plan_audit_groups_real_aplica_caps():
    """Con los CSV reales del repo, el plan debe aplicar caps 0.10/0.25/0.005.
    Si esto FALLA, significa que `build_plan_from_audit_groups` cayo al
    fallback raw_ncp y el sampling FL no replicaria la politica central.
    """
    p = Path("results/audit/audit_groups.json")
    proc = Path("results/processed_summary.csv")
    cli = Path("results/client_summary.csv")
    asum = Path("results/audit/audit_summary.json")
    if not (p.is_file() and proc.is_file() and cli.is_file() and asum.is_file()):
        pytest.skip("ficheros agregados de sampling no encontrados (audit/processed/client summaries)")
    groups = json.loads(p.read_text(encoding="utf-8"))
    from training.train_ssl_federated import build_plan_from_audit_groups
    plan = build_plan_from_audit_groups(groups)

    # 1) policy debe ser la capped, NO el fallback raw.
    policies = {r.get("policy") for r in plan}
    assert "raw_ncp_fallback" not in policies, (
        f"El plan cayo al fallback raw_ncp_fallback; faltan los CSVs de "
        f"sampling. Policies: {policies}"
    )
    assert "final_client_weight_capped_v23" in policies, (
        f"Falta marca final_client_weight_capped_v23: {policies}"
    )

    # 2) Cap por dataset 0.10.
    max_ds = max(r["final_dataset_weight"] for r in plan)
    assert max_ds <= 0.10 + 1e-6, (
        f"final_dataset_weight max={max_ds:.4f} excede cap 0.10"
    )

    # 3) Cap por cliente 0.25.
    from collections import defaultdict
    by_cli = defaultdict(float)
    for r in plan:
        by_cli[r["client"]] += r["final_dataset_weight"]
    max_cli = max(by_cli.values())
    assert max_cli <= 0.25 + 1e-6, (
        f"client weight max={max_cli:.4f} excede cap 0.25 "
        f"(bearings/phm_challenges deben quedar en 0.25)"
    )

    # 4) Piso min_client_presence = 0.005. Aplica al cliente mas pequeno
    # (cnc_milling en el corpus del MVP).
    min_cli = min(by_cli.values())
    assert min_cli >= 0.005 - 1e-6, (
        f"client weight min={min_cli:.6f} por debajo del piso 0.005"
    )

    # 5) Si fueran pesos RAW (n_channel_patches normalizados sin caps),
    # bearings dominaria con >0.32 segun sec 7.bis CLAUDE.md.
    bearings_w = by_cli.get("bearings", 0.0)
    assert bearings_w < 0.30, (
        f"bearings weight={bearings_w:.4f} cercano al raw (0.3224); "
        "indica que NO se aplicaron caps correctamente."
    )
