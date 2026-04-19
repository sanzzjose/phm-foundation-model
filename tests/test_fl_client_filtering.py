"""Tests del filtrado de plan por cliente.

No requiere torch (solo `filter_plan_by_client` y verificacion logica).
Para `build_clients_from_audit_groups` necesitariamos torch (instancia
PatchTSTPhm); se cubre en test_fl_server_dryrun.

Verificamos:
- cada cliente solo ve sus PRETRAIN_SOURCE;
- union de datasets cubre los 36 PS sin duplicados;
- ningun TRANSFER_TARGET cuela.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _audit_groups_real():
    """Carga el audit_groups real del repo (si existe)."""
    p = Path("results/audit/audit_groups.json")
    if not p.is_file():
        pytest.skip(f"audit_groups.json no encontrado en {p}")
    return json.loads(p.read_text(encoding="utf-8"))


def _build_plan_from_groups(groups):
    """Replica simplificada de build_plan_from_audit_groups (formato real v2.3)."""
    rows = []
    for client, info in groups.get("clients", {}).items():
        datasets = info.get("datasets", [])
        if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
            for meta in datasets:
                rows.append({
                    "dataset": str(meta.get("dataset", "")),
                    "client": client,
                    "n_channels": int(meta.get("n_channels", 0)),
                    "n_channel_patches": int(meta.get("n_channel_patches", 0)),
                    "final_dataset_weight": 0.0,
                })
        elif isinstance(datasets, dict):
            for ds, meta in datasets.items():
                rows.append({
                    "dataset": ds, "client": client,
                    "n_channels": int(meta.get("n_channels", 0)) if isinstance(meta, dict) else 0,
                    "n_channel_patches": int(meta.get("n_channel_patches", 0)) if isinstance(meta, dict) else 0,
                    "final_dataset_weight": 0.0,
                })
        elif isinstance(datasets, list):
            for ds in datasets:
                rows.append({
                    "dataset": str(ds), "client": client,
                    "n_channels": 0, "n_channel_patches": 0,
                    "final_dataset_weight": 0.0,
                })
    return rows


def test_filter_plan_by_client_sintetico():
    from training.fl.client import filter_plan_by_client
    plan = [
        {"dataset": "DS1", "client": "A"},
        {"dataset": "DS2", "client": "A"},
        {"dataset": "DS3", "client": "B"},
    ]
    a = filter_plan_by_client(plan, "A")
    b = filter_plan_by_client(plan, "B")
    c = filter_plan_by_client(plan, "C")
    assert {r["dataset"] for r in a} == {"DS1", "DS2"}
    assert {r["dataset"] for r in b} == {"DS3"}
    assert c == []


def test_filter_plan_solo_pretrain_source():
    """Garantia clave del FL: no TT en el plan filtrado por cliente.
    En nuestro caso `audit_groups.json` ya contiene SOLO los clientes con
    sus PS; los TT no aparecen como datasets dentro de clients_meta.
    El test verifica que el plan generado a partir de audit_groups
    realmente solo cubre PS."""
    groups = _audit_groups_real()
    plan = _build_plan_from_groups(groups)
    ds_plan = sorted({r["dataset"] for r in plan})

    # Lista de TT conocidos del MVP (sec 4.bis CLAUDE.md):
    KNOWN_TT = {
        "CMAPSS", "CALCE_CS2", "CWRU", "CNCMILL18", "PHMAP23", "HSG18",
        "PBCP16", "IEEE14", "CBM14", "PHME20", "PHM18",
    }
    for ds in ds_plan:
        assert ds not in KNOWN_TT, (
            f"TRANSFER_TARGET '{ds}' aparece en el plan FL, lo cual viola "
            "sec 3-4 del CLAUDE.md."
        )


def test_audit_groups_tiene_10_clientes():
    groups = _audit_groups_real()
    clients = groups.get("clients", {})
    assert len(clients) == 10, (
        f"Se esperaban 10 clientes (sec 7.bis), hay {len(clients)}: "
        f"{sorted(clients.keys())}"
    )


def _client_dataset_names(info):
    """Extrae nombres de dataset del info de un cliente, soportando los 3
    formatos posibles."""
    datasets = info.get("datasets", [])
    if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
        return [str(d.get("dataset", "")) for d in datasets]
    if isinstance(datasets, dict):
        return list(datasets.keys())
    if isinstance(datasets, list):
        return [str(d) for d in datasets]
    return []


def test_audit_groups_cubre_36_pretrain_sources():
    """La union de datasets sobre todos los clientes debe ser 36 PS."""
    groups = _audit_groups_real()
    union = set()
    for c, info in groups.get("clients", {}).items():
        union.update(_client_dataset_names(info))
    assert len(union) == 36, (
        f"Se esperaban 36 PS unicos en union (sec 4.bis), hay {len(union)}"
    )


def test_audit_groups_sin_datasets_duplicados_entre_clientes():
    groups = _audit_groups_real()
    seen: dict = {}
    for c, info in groups.get("clients", {}).items():
        for ds in _client_dataset_names(info):
            assert ds not in seen, (
                f"Dataset {ds!r} aparece en cliente {seen[ds]} y en {c}. "
                "No debe haber duplicados entre clientes."
            )
            seen[ds] = c
