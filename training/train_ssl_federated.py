"""Trainer SSL federado: dry-run / smoke / train.

CLI:

    python -m training.train_ssl_federated \\
        --mode dry-run|smoke|train \\
        --config training/configs/ssl_federated_smoke.yaml

Modos:

- **dry-run**: no entrena. Lee audit_groups, construye plan por cliente,
  verifica 10 clientes / 36 PS / 0 TT, construye modelo. Si Drive existe,
  hace forward sintetico de un batch. Escribe un report.

- **smoke**: entrena POCO (default 2 rondas x 5 steps x 10 clientes = 100
  steps locales). Criterio PASS:
    - loss finita en al menos un cliente por ronda;
    - global state_dict cambia tras agregacion (norma global delta != 0);
    - no TT usados (verificado por construccion en build_clients_*);
    - max_effective_bc <= 512 (o respeta min_batch_size explicito).

- **train**: usa stage segun cfg (smoke <= 1k, pilot <= 20k, full <= 500k
  steps locales totales). Gating duro para no lanzar mas alla del stage.

Logging:

- `metrics.jsonl`: una linea por ronda con `kind=round` (ver server.py).
- `run_info.json`: agregado final + smoke_pass / pilot_pass / coverage_pass.

Reusa `JsonlLogger`, `_json_safe`, `config_hash`, `get_git_info`,
`load_config` de `training.train_ssl_central`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


from training.train_ssl_central import (
    JsonlLogger,
    _json_safe,
    config_hash,
    get_git_info,
    load_config,
)


# ----------------------------------------------------------------------
# Constantes de stage
# ----------------------------------------------------------------------

STAGE_MAX_LOCAL_STEPS = {
    "smoke": 1_000,
    "pilot": 20_000,
    "full": 500_000,
}


# ----------------------------------------------------------------------
# Validacion del bloque cfg["federated"] (FedProx v0.2)
# ----------------------------------------------------------------------


_VALID_ALGORITHMS = ("fedavg", "fedprox", "fedavgm")


def _validate_federated_config(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Valida `cfg["federated"]` para FedAvg/FedProx/FedAvgM.

    Reglas:

    - `algorithm` debe estar en {"fedavg", "fedprox", "fedavgm"} (case-insensitive).
    - Si `algorithm == "fedavg"`, `fedprox_mu` debe ser None o == 0.
      Cualquier mu > 0 es config ambiguo (lo que se entrenaria no seria
      FedAvg). Falla duro para evitar runs invisibles tipo "fedavg pero
      con penalty".
    - Si `algorithm == "fedprox"`, `fedprox_mu` debe ser float > 0.
      mu null/<=0 es invalido (FedProx con mu=0 es FedAvg disfrazado).
    - Si `algorithm == "fedavgm"`, el cliente NO usa termino proximal
      (fedprox_mu debe ser None o 0). El servidor mantiene momentum
      con `cfg["server_momentum"]` (validado por separado en el server).

    Returns:
        dict { "algorithm", "fedprox_mu", "fedprox_enabled" } con los
        valores efectivos resueltos.

    Raises:
        ValueError con un mensaje claro y accionable.
    """
    fl_cfg = cfg.get("federated", {}) or {}
    algo_raw = fl_cfg.get("algorithm", "fedavg")
    algorithm = str(algo_raw).strip().lower()
    if algorithm not in _VALID_ALGORITHMS:
        raise ValueError(
            f"federated.algorithm desconocido: {algo_raw!r}. "
            f"Acepta: {' | '.join(_VALID_ALGORITHMS)}."
        )
    mu_raw = fl_cfg.get("fedprox_mu", None)
    if algorithm in ("fedavg", "fedavgm"):
        # En FedAvg y FedAvgM el cliente no usa termino proximal. fedprox_mu
        # debe estar ausente o ser 0 para evitar ambiguedad.
        if mu_raw is not None:
            try:
                mu_f = float(mu_raw)
            except (TypeError, ValueError):
                raise ValueError(
                    f"federated.fedprox_mu no es float ni null: {mu_raw!r}. "
                    f"Bajo algorithm={algorithm} debe ser null o 0."
                )
            if mu_f != 0.0:
                raise ValueError(
                    f"federated.algorithm={algorithm} pero fedprox_mu={mu_f} "
                    "(debe ser null o 0). Config ambiguo: usa algorithm="
                    "fedprox para activar el termino proximal."
                )
        return {"algorithm": algorithm, "fedprox_mu": None, "fedprox_enabled": False}
    # algorithm == "fedprox"
    if mu_raw is None:
        raise ValueError(
            "federated.algorithm=fedprox requiere fedprox_mu (float > 0); "
            "actualmente es null."
        )
    try:
        mu_f = float(mu_raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"federated.fedprox_mu no es float: {mu_raw!r}."
        )
    if mu_f <= 0:
        raise ValueError(
            f"federated.algorithm=fedprox requiere fedprox_mu > 0; "
            f"recibido {mu_f}."
        )
    return {"algorithm": "fedprox", "fedprox_mu": mu_f, "fedprox_enabled": True}


def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def _atomic_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(obj), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


# ----------------------------------------------------------------------
# Construccion del plan filtrado por rol PRETRAIN_SOURCE
# ----------------------------------------------------------------------


def _flatten_audit_groups(audit_groups: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Aplana `audit_groups['clients']` a una lista plana de
    `{dataset, client, n_channels, n_channel_patches}`.

    Soporta tres formatos historicos:
      1. lista de dicts con `dataset`, `n_channel_patches`, ... (audit v2.3 real).
      2. dict `{ds_name: {n_channels, n_channel_patches}}`.
      3. lista de strings.
    """
    clients = audit_groups.get("clients", {})
    rows: List[Dict[str, Any]] = []
    for client, info in clients.items():
        datasets = info.get("datasets", [])
        if isinstance(datasets, list) and datasets and isinstance(datasets[0], dict):
            for meta in datasets:
                rows.append({
                    "dataset": str(meta.get("dataset", "")),
                    "client": client,
                    "n_channels": int(meta.get("n_channels", 0)),
                    "n_channel_patches": int(meta.get("n_channel_patches", 0)),
                })
        elif isinstance(datasets, dict):
            for ds, meta in datasets.items():
                if isinstance(meta, dict):
                    rows.append({
                        "dataset": ds, "client": client,
                        "n_channels": int(meta.get("n_channels", 0)),
                        "n_channel_patches": int(meta.get("n_channel_patches", 0)),
                    })
                else:
                    rows.append({"dataset": ds, "client": client,
                                 "n_channels": 0, "n_channel_patches": 0})
        elif isinstance(datasets, list):
            for ds in datasets:
                rows.append({"dataset": str(ds), "client": client,
                             "n_channels": 0, "n_channel_patches": 0})
    return rows


def build_plan_from_audit_groups(
    audit_groups: Dict[str, Any],
    processed_summary_csv: Optional[Path] = None,
    client_summary_csv: Optional[Path] = None,
    audit_summary_json: Optional[Path] = None,
) -> List[Dict[str, Any]]:
    """Construye el plan FL con pesos finales CAPPED por cliente y dataset.

    Politica:

    - **Preferido (final_client_weight productivo)**: si los CSV/JSON
      cuyas rutas se pasan existen, usa `compute_pretraining_sampling_plan`
      del SSL central, que aplica los caps cerrados de audit v2.3
      (`0.10 / 0.25 / 0.005`). Esto replica exactamente la politica
      central, para comparacion justa.
    - **Fallback diagnostico (solo si los CSVs no estan)**: emite un plan
      con pesos `n_channel_patches` raw normalizados. Marca cada fila con
      `policy="raw_ncp_fallback"` y registra un warning. Este fallback
      NO debe usarse en pretraining real; sirve solo para dry-run en
      entornos sin acceso a Drive o sin los CSVs.

    Returns:
        Lista de dicts. Cada fila incluye `policy` para que el caller
        pueda assertar si se usaron caps.
    """
    # Path por defecto en el repo.
    if processed_summary_csv is None:
        processed_summary_csv = Path("results/processed_summary.csv")
    if client_summary_csv is None:
        client_summary_csv = Path("results/client_summary.csv")
    if audit_summary_json is None:
        audit_summary_json = Path("results/audit/audit_summary.json")

    # 1. Preferido: sampling plan completo con caps.
    if (processed_summary_csv.is_file()
            and client_summary_csv.is_file()
            and audit_summary_json.is_file()):
        from training.sampling import (
            load_sources, compute_pretraining_sampling_plan,
        )
        proc, cli, asum = load_sources(
            processed_summary_csv, client_summary_csv, audit_summary_json,
        )
        plan = compute_pretraining_sampling_plan(proc, cli, asum)
        for r in plan:
            r["policy"] = "final_client_weight_capped_v23"
        return plan

    # 2. Fallback diagnostico: raw n_channel_patches normalizados.
    print(
        "  WARN sampling: no encuentro processed_summary/client_summary/"
        "audit_summary; usando pesos RAW n_channel_patches sin caps. "
        "NO usar para entrenamiento real."
    )
    rows = _flatten_audit_groups(audit_groups)
    total = float(sum(r["n_channel_patches"] for r in rows))
    if total <= 0:
        # Sin n_channel_patches: reparto uniforme intra-cliente.
        from collections import Counter
        per_c = Counter(r["client"] for r in rows)
        for r in rows:
            r["final_dataset_weight"] = 1.0 / max(1, per_c[r["client"]] * len(per_c))
            r["final_client_weight"] = 1.0 / max(1, len(per_c))
    else:
        for r in rows:
            r["final_dataset_weight"] = r["n_channel_patches"] / total
        client_sum: Dict[str, float] = {}
        for r in rows:
            client_sum[r["client"]] = client_sum.get(r["client"], 0.0) + r["final_dataset_weight"]
        for r in rows:
            r["final_client_weight"] = client_sum[r["client"]]
    for r in rows:
        r["policy"] = "raw_ncp_fallback"
    return rows


# ----------------------------------------------------------------------
# Dry-run
# ----------------------------------------------------------------------


def _verify_no_tt_in_plan(
    plan: List[Dict[str, Any]],
    processed_summary_csv: Path = Path("results/processed_summary.csv"),
) -> Dict[str, Any]:
    """Verifica que NINGUN dataset del plan FL tiene role != PRETRAIN_SOURCE.

    Cruza los nombres del plan contra `processed_summary.csv['role']`. Si
    el CSV no existe, usa una lista hardcoded de TT conocidos (sec 4.bis
    CLAUDE.md) como red de seguridad.

    Returns:
        {"ok": bool, "violations": [datasets que no son PS],
         "source": "processed_summary_csv" | "hardcoded_tt_list",
         "checked_via": "role" | "name"}
    """
    plan_ds = {str(r["dataset"]) for r in plan}
    violations: List[str] = []
    if processed_summary_csv.is_file():
        import csv as _csv
        with open(processed_summary_csv, "r", newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                ds = str(row.get("dataset", ""))
                if ds in plan_ds and str(row.get("role", "")) != "PRETRAIN_SOURCE":
                    violations.append(ds)
        return {"ok": not violations, "violations": violations,
                "source": "processed_summary_csv", "checked_via": "role"}
    # Fallback: lista hardcoded de TT del MVP (sec 4.bis CLAUDE.md).
    KNOWN_TT = {
        "CMAPSS", "CALCE_CS2", "CWRU", "CNCMILL18", "PHMAP23", "HSG18",
        "PBCP16", "IEEE14", "CBM14", "PHME20", "PHM18",
    }
    for ds in plan_ds:
        if ds in KNOWN_TT:
            violations.append(ds)
    return {"ok": not violations, "violations": violations,
            "source": "hardcoded_tt_list", "checked_via": "name"}


def cmd_dry_run(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Construye plan + clientes, NO entrena. Aplica asserts duros.

    Codigos de retorno:
        0: OK (10 clientes, 36 PS, 0 TT, modelo OK, forward sintetico OK).
        2: audit_groups no encontrado.
        3: fallo construyendo clientes.
        4: numero de clientes != 10.
        5: numero de datasets unicos != 36.
        6: violacion TT (algun dataset del plan tiene role != PRETRAIN_SOURCE).
        7: fallo en forward sintetico.
        8: config federated invalido (algorithm/fedprox_mu inconsistentes).

    En todos los casos != 0, escribe un report JSON con el diagnostico.
    """
    print(f"[{_ts()}] === DRY-RUN FL === stage={cfg.get('stage', 'smoke')}")
    print(f"  config_hash: {config_hash(cfg)}")
    # Validacion FedProx v0.2: rechaza configs ambiguos antes de tocar
    # nada en disco. Si esto falla, no se construye el report dir.
    try:
        fprox_eff = _validate_federated_config(cfg)
        print(
            f"  algorithm: {fprox_eff['algorithm']}  "
            f"fedprox_mu: {fprox_eff['fedprox_mu']}  "
            f"fedprox_enabled: {fprox_eff['fedprox_enabled']}"
        )
    except ValueError as e:
        print(f"  ERROR config federated: {e}")
        return 8

    data_cfg = cfg.get("data", {})
    audit_path = Path(data_cfg.get("client_source", "results/audit/audit_groups.json"))
    # Directorio para report (preferimos log_dir si existe en cfg, en
    # otro caso results/pretraining_federated/).
    paths_cfg = cfg.get("paths", {})
    report_dir = Path(
        paths_cfg.get("log_dir", "results/pretraining_federated")
    ) / cfg.get("run_name", "dry_run_unnamed")
    report: Dict[str, Any] = {
        "ts": _ts(),
        "stage": cfg.get("stage", "smoke"),
        "run_name": cfg.get("run_name", "dry_run_unnamed"),
        "config_hash": config_hash(cfg),
        "audit_path": str(audit_path),
        "checks": {},
    }

    def _write_report(rc: int):
        report["return_code"] = rc
        report["ok"] = (rc == 0)
        try:
            _atomic_json_dump(report, report_dir / "dry_run_report.json")
            print(f"  report escrito en: {report_dir / 'dry_run_report.json'}")
        except Exception as e:
            print(f"  WARN no se pudo escribir report: {e}")
        return rc

    if not audit_path.is_file():
        report["error"] = f"audit_groups no encontrado en {audit_path}"
        print(f"  ERROR: {report['error']}")
        return _write_report(2)
    audit_groups = json.loads(audit_path.read_text(encoding="utf-8"))
    plan = build_plan_from_audit_groups(audit_groups)
    report["plan_rows"] = len(plan)
    report["plan_policy"] = list({r.get("policy", "?") for r in plan})
    print(f"  plan: {len(plan)} filas (PRETRAIN_SOURCE)")
    print(f"  plan policy: {report['plan_policy']}")

    # Verificacion clientes
    clients_in_plan = sorted({str(r["client"]) for r in plan})
    print(f"  clients ({len(clients_in_plan)}): {clients_in_plan}")
    report["checks"]["n_clients"] = {"value": len(clients_in_plan), "expected": 10, "ok": len(clients_in_plan) == 10}
    if len(clients_in_plan) != 10:
        report["error"] = f"se esperaban 10 clientes, hay {len(clients_in_plan)}"
        print(f"  ERROR: {report['error']}")
        return _write_report(4)

    n_datasets = len({str(r["dataset"]) for r in plan})
    print(f"  datasets unicos: {n_datasets}  (esperado 36)")
    report["checks"]["n_datasets"] = {"value": n_datasets, "expected": 36, "ok": n_datasets == 36}
    if n_datasets != 36:
        report["error"] = f"se esperaban 36 datasets unicos, hay {n_datasets}"
        print(f"  ERROR: {report['error']}")
        return _write_report(5)

    # Verificacion 0 TT usando processed_summary.role (no solo audit_groups).
    tt_check = _verify_no_tt_in_plan(plan)
    report["checks"]["no_tt_in_plan"] = tt_check
    print(f"  TT check via {tt_check['source']}/{tt_check['checked_via']}: "
          f"violations={tt_check['violations']}")
    if not tt_check["ok"]:
        report["error"] = f"TT en el plan FL: {tt_check['violations']}"
        print(f"  ERROR: {report['error']}")
        return _write_report(6)

    # Construir clientes sin entrenar
    from training.fl.client import build_clients_from_audit_groups
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    except Exception:
        device = None

    try:
        clients = build_clients_from_audit_groups(
            audit_groups=audit_groups,
            plan=plan,
            processed_root=Path(paths_cfg.get(
                "processed_root", "/content/drive/MyDrive/fm_fl_phmd/processed",
            )),
            model_cfg=cfg["model"],
            ssl_cfg=cfg.get("ssl", {}),
            training_cfg=cfg.get("training", {}),
            data_cfg=data_cfg,
            device=device,
            seed=int(cfg.get("seed", 42)),
        )
        print(f"  clients construidos: {len(clients)}")
        clients_meta = []
        for c in clients:
            print(f"    - {c.name:<20} -> {len(c.plan_subset)} datasets")
            clients_meta.append({"name": c.name, "n_datasets": len(c.plan_subset)})
        report["clients"] = clients_meta
    except Exception as e:
        report["error"] = f"fallo construyendo clientes: {e}"
        print(f"  ERROR: {report['error']}")
        return _write_report(3)

    # Forward sintetico para validar modelo
    try:
        import torch
        from models.patchtst_phm import build_patchtst_phm
        model = build_patchtst_phm(cfg["model"])
        B, C = 2, 2
        P = int(cfg["model"]["patch_size"])
        N = int(cfg["model"]["n_patches"])
        x = torch.randn(B, C, N, P)
        vtm = torch.ones(B, N * P, dtype=torch.bool)
        vpm = torch.ones(B, C, N, dtype=torch.bool)
        from training.ssl.masking import generate_ssl_mask
        gen = torch.Generator(); gen.manual_seed(42)
        sm = generate_ssl_mask(vpm, mask_ratio=0.3, generator=gen)
        out = model(x, vtm, vpm, ssl_mask=sm)
        assert "reconstruction" in out, "el modelo debe devolver 'reconstruction'"
        recon_shape = tuple(out["reconstruction"].shape)
        print(f"  forward sintetico OK: reconstruction shape {recon_shape}")
        report["checks"]["forward_synthetic"] = {"ok": True, "reconstruction_shape": list(recon_shape)}
    except Exception as e:
        report["checks"]["forward_synthetic"] = {"ok": False, "error": str(e)}
        report["error"] = f"forward sintetico fallo: {e}"
        print(f"  ERROR: {report['error']}")
        return _write_report(7)

    print(f"[{_ts()}] === DRY-RUN FL OK ===")
    return _write_report(0)


# ----------------------------------------------------------------------
# Smoke
# ----------------------------------------------------------------------


def cmd_smoke(
    cfg: Dict[str, Any], repo_root: Path, stage_label: Optional[str] = None
) -> int:
    """Entrena pocas rondas. Nucleo compartido con cmd_train (pilot/full).

    Args:
        cfg: config FL.
        repo_root: raiz del repo (para git_hash).
        stage_label: si se pasa (pilot|full), los campos run_info se
            escriben como pilot_pass/pilot_checks o full_pass/full_checks
            en lugar de smoke_pass/smoke_checks. Por defecto usa el
            stage del cfg.

    Criterios PASS (idénticos en smoke/pilot/full):
      - todos los clientes participantes tienen al menos 1 step exitoso;
      - global state_dict cambia tras al menos una ronda;
      - no TT usados (verificado contra processed_summary.role);
      - max_effective_bc <= 512 salvo override min_batch_size;
      - pesos de agregacion reflejan los caps del plan (V2).
    """
    stage = str(cfg.get("stage", "smoke"))
    fl_cfg = cfg.get("federated", {})
    # Validacion FedProx v0.2: misma logica que en cmd_dry_run. Rechazar
    # configs ambiguos antes de crear log_dir / ckpt_dir.
    try:
        fprox_eff = _validate_federated_config(cfg)
    except ValueError as e:
        print(f"ERROR config federated: {e}")
        return 8
    # data_cfg se usa tanto para audit_path como para los smoke_checks
    # finales (max_channel_batch, min_batch_size). Lo capturamos una sola
    # vez aqui para evitar NameError en el bloque de checks.
    data_cfg = cfg.get("data", {})
    total = int(fl_cfg.get("n_rounds", 2)) * int(fl_cfg.get("local_steps", 5))
    max_total = STAGE_MAX_LOCAL_STEPS.get(stage, 1000)
    n_clients_estimate = 10
    if total * n_clients_estimate > max_total:
        print(
            f"ERROR: stage={stage} permite max {max_total} local steps; "
            f"config pide ~{total * n_clients_estimate}. Reduce n_rounds/local_steps."
        )
        return 4

    paths = cfg.get("paths", {})
    log_dir = Path(paths.get("log_dir", "/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated")) / cfg["run_name"]
    ckpt_dir = Path(paths.get("checkpoint_dir", "/content/drive/MyDrive/fm_fl_phmd/checkpoints/ssl_federated_smoke")) / cfg["run_name"]
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    audit_path = Path(data_cfg.get(
        "client_source", "results/audit/audit_groups.json"
    ))
    if not audit_path.is_file():
        print(f"ERROR audit_groups no encontrado: {audit_path}")
        return 2
    audit_groups = json.loads(audit_path.read_text(encoding="utf-8"))
    plan = build_plan_from_audit_groups(audit_groups)
    processed_root = Path(paths.get("processed_root", "/content/drive/MyDrive/fm_fl_phmd/processed"))

    cfg_hash = config_hash(cfg)
    git_info = get_git_info(repo_root)
    logger = JsonlLogger(log_dir / "metrics.jsonl")
    # Etiqueta efectiva del stage (pilot/full sobreescriben):
    stage_eff = str(stage_label) if stage_label else stage
    labels = STAGE_LABELS.get(stage_eff, STAGE_LABELS["smoke"])
    banner = labels["banner"]
    pass_key = labels["pass_key"]
    checks_key = labels["checks_key"]
    print(f"[{_ts()}] === {banner} FL === stage={stage_eff}  run={cfg['run_name']}")
    print(f"  config_hash: {cfg_hash}")
    print(
        f"  algorithm: {fprox_eff['algorithm']}  "
        f"fedprox_mu: {fprox_eff['fedprox_mu']}  "
        f"fedprox_enabled: {fprox_eff['fedprox_enabled']}"
    )

    from training.fl.server import run_federated_pretraining
    run_info = run_federated_pretraining(
        cfg=cfg,
        audit_groups=audit_groups,
        plan=plan,
        processed_root=processed_root,
        log_dir=log_dir,
        ckpt_dir=ckpt_dir,
        logger=logger,
    )

    # ------------------------------------------------------------------
    # Criterios smoke_pass reforzados (U4). Persistimos cada check para
    # diagnostico en run_info.
    # ------------------------------------------------------------------
    clients_participated = list(run_info.get("opt_steps_per_client_total", {}).keys())
    checks: Dict[str, Any] = {}

    # (1) loss finita en TODOS los clientes participantes (en al menos una
    # ronda cada uno).
    loss_finite_per_client = run_info.get("loss_finite_per_client_round", {})
    clients_with_any_finite_loss = [
        c for c, rounds in loss_finite_per_client.items() if any(rounds)
    ]
    checks["all_clients_finite_loss"] = {
        "ok": (set(clients_with_any_finite_loss) == set(clients_participated)
               and len(clients_participated) > 0),
        "clients_participated": clients_participated,
        "clients_with_any_finite_loss": clients_with_any_finite_loss,
    }

    # (2) total_local_optimizer_steps > 0 POR cliente.
    opt_steps = run_info.get("opt_steps_per_client_total", {})
    clients_without_opt_steps = [c for c in clients_participated if opt_steps.get(c, 0) <= 0]
    checks["all_clients_opt_steps_gt0"] = {
        "ok": (len(clients_without_opt_steps) == 0 and len(clients_participated) > 0),
        "clients_without_opt_steps": clients_without_opt_steps,
        "opt_steps_per_client": opt_steps,
    }

    # (3) global_state_norm cambia en AL MENOS una ronda.
    checks["global_state_changes"] = {
        "ok": bool(run_info.get("state_norm_changed_in_any_round", False)),
    }

    # (4) max_effective_bc <= max_channel_batch (salvo min_batch_size>1 override).
    max_cb = int(data_cfg.get("max_channel_batch", 512) or 512)
    min_bs = int(data_cfg.get("min_batch_size", 1))
    max_bc_seen = int(run_info.get("max_effective_bc_global", 0))
    # Si min_batch_size>1 podriamos exceder el cap legitimamente (caso PHM14
    # con C=317 y min_batch_size=1 ya da B*C=317<=512). Solo es violacion
    # si supera el cap Y no es por override explicito min_batch_size.
    bc_ok = (max_bc_seen <= max_cb) or (min_bs > 1)
    checks["max_effective_bc_within_cap"] = {
        "ok": bool(bc_ok),
        "max_effective_bc_global": max_bc_seen,
        "max_channel_batch": max_cb,
        "min_batch_size": min_bs,
    }

    # (5) 0 TT usados: reutilizamos el verificador del dry-run.
    tt_check = _verify_no_tt_in_plan(plan)
    checks["no_tt_in_plan"] = tt_check

    # (6) V2/W: si policy=final_client_weight y plan capped v23, los pesos
    # de agregacion por cliente NO deben ser todos iguales (salvo que
    # realmente lo sean en el plan, lo cual no ocurre con audit v2.3).
    # Si todos los pesos son aprox 1/n_clientes, es sintoma del bug de
    # mutacion del plan o del fallback raw que perdio los caps.
    # W tambien: si los pesos estan VACIOS bajo policy=final_client_weight
    # con plan capped, es fallo duro (no podemos verificar nada).
    aw = run_info.get("aggregation_weights_by_client_last_round", {}) or {}
    plan_pol = run_info.get("plan_policy_unique", []) or []
    policy_eff = run_info.get("aggregation_weight_policy", "")
    weights_ok = True
    weights_diag = {
        "aggregation_weight_policy": policy_eff,
        "plan_policy_unique": plan_pol,
        "weights": aw,
    }
    if policy_eff == "final_client_weight" and "final_client_weight_capped_v23" in plan_pol:
        if not aw:
            # W: pesos vacios bajo capped + final_client_weight es fallo duro.
            weights_ok = False
            weights_diag["warning"] = (
                "aggregation_weights_by_client_last_round vacio bajo "
                "policy=final_client_weight + plan capped v23. No es posible "
                "verificar que los caps se aplicaron."
            )
        else:
            mn = min(aw.values()); mx = max(aw.values())
            # En el corpus real con caps, el rango es ~0.005..0.25 (50x).
            if (mx - mn) < 0.01:
                weights_ok = False
                weights_diag["warning"] = (
                    f"pesos por cliente cuasi-uniformes (max-min={mx - mn:.4f}); "
                    "posible mutacion del plan o perdida de caps."
                )
    checks["aggregation_weights_reflect_caps"] = {"ok": weights_ok, **weights_diag}

    all_pass = all(c["ok"] for c in checks.values())

    run_info.update({
        "kind": "run_info",
        "ts": _ts(),
        "stage": stage_eff,
        "run_name": cfg["run_name"],
        "config_hash": cfg_hash,
        "git_hash": git_info["git_hash"],
        "git_dirty": git_info["git_dirty"],
        pass_key: all_pass,
        checks_key: checks,
    })
    logger.close()
    _atomic_json_dump(run_info, log_dir / "run_info.json")
    print(f"[{_ts()}] === {banner} FL end ===  {pass_key}={all_pass}")
    for name, ck in checks.items():
        ok_str = "OK" if ck["ok"] else "FAIL"
        print(f"  [{ok_str}] {name}")
    print(f"Logs: {log_dir}  Ckpts: {ckpt_dir}")
    return 0 if all_pass else 5


# ----------------------------------------------------------------------
# Train (pilot / full). Mismo nucleo que smoke pero con etiquetas y
# campos run_info adaptados al stage. NO se ejecuta sin autorizacion.
# ----------------------------------------------------------------------


# Mapeo stage -> (pass_key, checks_key, banner).
# El nucleo de cmd_smoke escribe en run_info los campos cuyo nombre
# corresponde al stage. Esto evita que un pilot quede etiquetado como
# "smoke_pass" en su run_info, lo cual seria confuso en analisis futuros.
STAGE_LABELS = {
    "smoke": {"pass_key": "smoke_pass", "checks_key": "smoke_checks", "banner": "SMOKE"},
    "pilot": {"pass_key": "pilot_pass", "checks_key": "pilot_checks", "banner": "PILOT"},
    "full":  {"pass_key": "full_pass",  "checks_key": "full_checks",  "banner": "FULL"},
}


def cmd_train(cfg: Dict[str, Any], repo_root: Path) -> int:
    """Train federado productivo. Stage debe ser pilot o full.

    smoke se dispara con --mode smoke (ver `cmd_smoke`). Aqui solo se
    procesan stages que cambian la semantica del run_info:
      - pilot escribe pilot_pass y pilot_checks.
      - full escribe full_pass y full_checks.
      - El gating duro por STAGE_MAX_LOCAL_STEPS aplica igual que en
        cmd_smoke.

    NOTA: este comando NO debe lanzarse sin autorizacion explicita.
    """
    stage = str(cfg.get("stage", "smoke"))
    if stage not in STAGE_MAX_LOCAL_STEPS:
        print(f"ERROR stage desconocido: {stage}")
        return 4
    if stage == "smoke":
        # smoke pasa por su comando dedicado.
        return cmd_smoke(cfg, repo_root)
    # pilot / full: reusamos cmd_smoke como motor (los checks y el
    # logging son los mismos), pero le pedimos que escriba con el label
    # del stage actual.
    return cmd_smoke(cfg, repo_root, stage_label=stage)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SSL federated trainer (cross-silo simulado)")
    p.add_argument(
        "--mode", choices=("dry-run", "smoke", "train"), required=True,
    )
    p.add_argument("--config", type=Path, required=True)
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    cfg = load_config(args.config)
    if args.mode == "dry-run":
        return cmd_dry_run(cfg, _REPO_ROOT)
    if args.mode == "smoke":
        return cmd_smoke(cfg, _REPO_ROOT)
    return cmd_train(cfg, _REPO_ROOT)


if __name__ == "__main__":
    sys.exit(main())
