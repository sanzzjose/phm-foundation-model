"""Servidor FL: orquestador de rondas FedAvg cross-silo simulado.

API principal:

    run_federated_pretraining(cfg, logger) -> final_state_dict

Por ronda:
1. Envia `global_state_dict` a cada cliente seleccionado.
2. Cada cliente entrena `n_local_steps`.
3. Agrega con FedAvg (peso por cliente segun `aggregation_weight_policy`).
4. Loggea metricas y opcionalmente guarda checkpoint.

Politicas de pesos de agregacion:
- `final_client_weight` (default productivo): usa los pesos finales del
  sampling plan central, agrupados por cliente (replica la politica
  central con caps 0.10/0.25/0.005).
- `uniform`: 1/n_clientes.
- `num_samples`: ponderado por #samples efectivos en la ronda
  (proxy = optimizer_steps * batch_size_effective_medio).

No implementa networking real. Para FL real, envolver cada FederatedClient
en un wrapper Flower (no incluido todavia).
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


def compute_aggregation_weights(
    client_metrics: Sequence[Dict[str, Any]],
    plan: Sequence[Dict[str, Any]],
    policy: str = "final_client_weight",
) -> List[float]:
    """Devuelve pesos por cliente para FedAvg.

    Args:
        client_metrics: lista de dicts devueltos por `FederatedClient.local_train`.
            Debe contener al menos `client` (nombre).
        plan: sampling plan completo (con `client` y `final_client_weight` o
            `final_dataset_weight`).
        policy: 'final_client_weight' | 'uniform' | 'num_samples'.

    Returns:
        Lista de floats no normalizada (el agregador FedAvg ya normaliza).
    """
    if not client_metrics:
        return []

    if policy == "uniform":
        return [1.0] * len(client_metrics)

    if policy == "num_samples":
        out: List[float] = []
        for cm in client_metrics:
            ops = int(cm.get("optimizer_steps", 0) or 0)
            bc = int(cm.get("max_effective_bc", 0) or 0)
            out.append(max(1.0, float(ops * max(1, bc))))
        return out

    if policy == "final_client_weight":
        # Suma por cliente del final_dataset_weight.
        by_client: Dict[str, float] = {}
        for r in plan:
            c = str(r.get("client"))
            w = float(r.get("final_dataset_weight", 0.0))
            by_client[c] = by_client.get(c, 0.0) + w
        out = []
        for cm in client_metrics:
            c = str(cm.get("client"))
            out.append(max(1e-12, by_client.get(c, 0.0)))
        return out

    raise ValueError(
        f"aggregation_weight_policy desconocida: {policy!r}. "
        "Acepta: final_client_weight | uniform | num_samples."
    )


def run_federated_pretraining(
    cfg: Dict[str, Any],
    audit_groups: Dict[str, Any],
    plan: Sequence[Dict[str, Any]],
    processed_root: Path,
    log_dir: Path,
    ckpt_dir: Path,
    logger: Any,
) -> Dict[str, Any]:
    """Ejecuta el bucle federado y devuelve `run_info` final.

    Args:
        cfg: config FL completa.
        audit_groups: contenido de `audit_groups.json`.
        plan: sampling plan (lista de dicts).
        processed_root: ruta a `processed/`.
        log_dir, ckpt_dir: directorios de salida.
        logger: instancia tipo `JsonlLogger` con `.log(dict)` y `.close()`.

    Returns:
        run_info con `n_rounds`, `total_local_optimizer_steps`,
        `clients_seen`, `datasets_seen_by_client`, `param_count`,
        `final_loss_mean_weighted`, `cumulative_communication_mb`, etc.
    """
    import torch
    from models.patchtst_phm import build_patchtst_phm, count_parameters
    from training.fl.aggregation import (
        fedavg_state_dict, estimate_communication_mb,
    )
    from training.fl.client import build_clients_from_audit_groups

    t0 = time.time()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    fl_cfg = cfg.get("federated", {})
    n_rounds = int(fl_cfg.get("n_rounds", 2))
    n_local_steps = int(fl_cfg.get("local_steps", 5))
    clients_per_round_cfg = fl_cfg.get("clients_per_round", "all")
    aggregation_policy = str(cfg.get("data", {}).get(
        "aggregation_weight_policy", "final_client_weight"
    ))

    # Modelo de referencia para inicializar y agregar.
    model_cfg = cfg["model"]
    global_model = build_patchtst_phm(model_cfg).to(device)
    # count_parameters acepta `trainable_only` (no `only_trainable`).
    param_count = count_parameters(global_model, trainable_only=True)
    global_sd = {k: v.detach().cpu() for k, v in global_model.state_dict().items()}

    # FedProx v0.2: pasar el bloque cfg["federated"] a cada cliente para
    # que `local_train` lea algorithm/fedprox_mu. Default FedAvg si no
    # esta presente.
    clients = build_clients_from_audit_groups(
        audit_groups=audit_groups,
        plan=plan,
        processed_root=processed_root,
        model_cfg=model_cfg,
        ssl_cfg=cfg.get("ssl", {}),
        training_cfg=cfg.get("training", {}),
        data_cfg=cfg.get("data", {}),
        federated_cfg=fl_cfg,
        device=device,
        seed=int(cfg.get("seed", 42)),
    )
    if not clients:
        raise RuntimeError(
            "Sin clientes FL: audit_groups.clients vacio o plan sin matches."
        )

    # LR scheduler opcional (v0.3, Fase 4c). Si esta presente, se propaga
    # a cada cliente UNA SOLA VEZ (los attrs no cambian entre rondas).
    # Si esta ausente, el comportamiento es LR constante (historico).
    from training.ssl.schedulers import scheduler_summary
    lr_scheduler_cfg = cfg.get("lr_scheduler")
    scheduler_eff = scheduler_summary(lr_scheduler_cfg)
    for c in clients:
        c.lr_scheduler_cfg = lr_scheduler_cfg
        c.n_clients_in_round = len(clients)
        c.local_steps_per_client = n_local_steps
    print(f"  lr_scheduler effective: {scheduler_eff}")

    cumulative_comm_mb = 0.0
    total_local_optimizer_steps = 0
    datasets_seen_by_client: Dict[str, set] = {}

    # Estado agregado para los criterios smoke_pass (U4):
    state_norm_changed_in_any_round = False
    opt_steps_per_client_total: Dict[str, int] = {}
    loss_finite_per_client_round: Dict[str, List[bool]] = {}
    amp_nonfinite_grad_steps_total: Dict[str, int] = {}
    max_effective_bc_global = 0
    last_aggregation_weights_by_client: Dict[str, float] = {}

    # FedProx v0.2: estado efectivo del algoritmo y agregados last-round
    # para run_info.
    from training.fl.client import resolve_fedprox_config
    fprox_eff = resolve_fedprox_config(fl_cfg)
    algorithm_effective = str(fprox_eff["algorithm"])
    fedprox_mu_effective = (
        float(fprox_eff["fedprox_mu"]) if fprox_eff["fedprox_enabled"] else None
    )
    fedprox_enabled = bool(fprox_eff["fedprox_enabled"])

    # FedAvgM (Fase 4d) — server momentum. Default = sin momentum (FedAvg).
    # Si `algorithm=fedavgm`, leemos cfg["server_momentum"]; si no, beta=0 y
    # el comportamiento es bit-a-bit igual al historico.
    server_mom_cfg = cfg.get("server_momentum") or {}
    server_momentum_enabled = (algorithm_effective == "fedavgm")
    server_momentum_beta = (
        float(server_mom_cfg.get("beta", 0.9)) if server_momentum_enabled else 0.0
    )
    server_momentum_nesterov = bool(server_mom_cfg.get("nesterov", False))
    server_momentum_init = str(server_mom_cfg.get("initialize", "zeros")).lower()
    if server_momentum_enabled and not (0.0 <= server_momentum_beta <= 1.0):
        raise ValueError(
            f"server_momentum.beta={server_momentum_beta} fuera de [0,1]"
        )
    if server_momentum_enabled and server_momentum_init not in ("zeros",):
        raise ValueError(
            f"server_momentum.initialize='{server_momentum_init}' no soportado "
            "(solo 'zeros' por ahora)"
        )
    server_velocity = None  # se inicializa lazily a zeros con el shape de delta
    if server_momentum_enabled:
        print(
            f"  server_momentum effective: beta={server_momentum_beta}, "
            f"nesterov={server_momentum_nesterov}, initialize={server_momentum_init}"
        )
    last_loss_mean_weighted: Optional[float] = None
    last_reconstruction_loss_mean_weighted: Optional[float] = None
    last_fedprox_loss_mean_weighted: Optional[float] = None
    last_fedprox_penalty_mean_weighted: Optional[float] = None

    # plan_policy_unique: V2. Detecta si el plan FL fue construido con
    # caps cerrados (final_client_weight_capped_v23) o cayo al fallback
    # raw_ncp_fallback. Util para el smoke check: si policy=
    # final_client_weight pero plan cayo a raw, NO deberiamos confiar
    # en que los pesos por cliente sean los del audit v2.3.
    plan_policy_unique = sorted({str(r.get("policy", "?")) for r in plan})

    # aggregation_weights_policy_effective (W): describe la politica de
    # agregacion REAL combinando el policy del config con el origen del
    # plan. Util en analisis posterior para no confundir lo declarado
    # con lo aplicado.
    def _compute_policy_effective(decl_policy: str, plan_pols: List[str]) -> str:
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

    aggregation_policy_effective = _compute_policy_effective(
        aggregation_policy, plan_policy_unique
    )

    for round_idx in range(1, n_rounds + 1):
        # Seleccion de clientes (MVP: all).
        if clients_per_round_cfg == "all":
            selected = clients
        else:
            n_sel = min(int(clients_per_round_cfg), len(clients))
            selected = clients[:n_sel]

        cm_list: List[Dict[str, Any]] = []
        loss_per_client: Dict[str, Any] = {}  # puede ser None si cliente no aporto loss valida
        drift_per_client: Dict[str, float] = {}
        for c in selected:
            metrics = c.local_train(global_sd, round_idx, n_local_steps)
            cm_list.append(metrics)
            loss_per_client[c.name] = metrics["loss_mean"]  # None si todos los steps skipped
            drift_per_client[c.name] = float(metrics["drift_l2_norm"])
            total_local_optimizer_steps += int(metrics["optimizer_steps"])
            for d in metrics["datasets_seen"]:
                datasets_seen_by_client.setdefault(c.name, set()).add(d)
            # Estado agregado para smoke_pass (U4):
            opt_steps_per_client_total[c.name] = (
                opt_steps_per_client_total.get(c.name, 0) + int(metrics["optimizer_steps"])
            )
            loss_finite_per_client_round.setdefault(c.name, []).append(
                bool(metrics.get("loss_finite_in_any_step", False))
            )
            amp_nonfinite_grad_steps_total[c.name] = (
                amp_nonfinite_grad_steps_total.get(c.name, 0)
                + int(metrics.get("amp_nonfinite_grad_steps", 0))
            )

        # Agregacion. `weights` viene SIN normalizar; lo normalizamos
        # aqui para logging y validacion, ademas de pasarlo crudo a
        # `fedavg_state_dict` (que tambien lo normaliza internamente).
        weights = compute_aggregation_weights(cm_list, plan, aggregation_policy)
        weights_sum = float(sum(weights)) if weights else 0.0
        weights_norm = (
            [float(w) / weights_sum for w in weights] if weights_sum > 0
            else [0.0] * len(weights)
        )
        aggregation_weights_by_client = {
            cm["client"]: w for cm, w in zip(cm_list, weights_norm)
        }
        new_sd = fedavg_state_dict(
            [cm["local_state_dict"] for cm in cm_list], weights,
        )

        # FedAvgM (Fase 4d) — aplicar momentum del servidor sobre la
        # agregacion. Si algorithm != fedavgm, el bloque queda inactivo y
        # `new_sd` se usa tal cual (comportamiento historico).
        from training.fl.aggregation import (
            compute_state_dict_delta, apply_server_momentum, state_dict_l2_norm,
        )
        if server_momentum_enabled:
            server_delta = compute_state_dict_delta(new_sd, global_sd)
            server_delta_norm_v = state_dict_l2_norm(server_delta)
            new_sd, server_velocity = apply_server_momentum(
                sd_global=global_sd,
                velocity_prev=server_velocity,
                delta=server_delta,
                beta=server_momentum_beta,
                nesterov=server_momentum_nesterov,
            )
            server_velocity_norm_v = state_dict_l2_norm(server_velocity)
        else:
            server_delta_norm_v = None
            server_velocity_norm_v = None

        # Verificar que el state_dict global cambia.
        # (Sanity check de smoke: comparamos suma de norms.)
        old_norm = sum(
            float(v.float().norm().item())
            for v in global_sd.values()
            if hasattr(v, "norm")
        )
        new_norm = sum(
            float(v.float().norm().item())
            for v in new_sd.values()
            if hasattr(v, "norm")
        )
        # Marca: el state_dict global cambia tras agregacion en esta ronda?
        if abs(new_norm - old_norm) > 1e-12:
            state_norm_changed_in_any_round = True
        global_sd = new_sd
        last_aggregation_weights_by_client = dict(aggregation_weights_by_client)

        # Track max_effective_bc agregado
        round_max_bc = max(int(cm["max_effective_bc"]) for cm in cm_list) if cm_list else 0
        max_effective_bc_global = max(max_effective_bc_global, round_max_bc)

        # Loss agregada ponderada (ignorando clientes con loss_mean None).
        # Calculamos en paralelo la media ponderada de las 3 metricas
        # FedProx (en FedAvg: reconstruction == loss, penalty == 0,
        # prox_loss == 0). El divisor es el mismo: suma de pesos con
        # cliente que aporto loss valida.
        total_w_valid = 0.0
        loss_sum_w = 0.0
        recon_sum_w = 0.0
        fprox_loss_sum_w = 0.0
        fprox_penalty_sum_w = 0.0

        def _isfinite(x: Any) -> bool:
            try:
                xf = float(x)
            except (TypeError, ValueError):
                return False
            return xf == xf and xf not in (float("inf"), float("-inf"))

        for w, cm in zip(weights, cm_list):
            lm = cm.get("loss_mean")
            if lm is None or not _isfinite(lm):
                continue
            wf = float(w)
            loss_sum_w += wf * float(lm)
            total_w_valid += wf
            # reconstruction_loss_mean: presente desde v0.2; si por algun
            # caso no lo esta (cliente antiguo), cae al lm como fallback.
            rl = cm.get("reconstruction_loss_mean")
            recon_sum_w += wf * (float(rl) if _isfinite(rl) else float(lm))
            fl_v = cm.get("fedprox_loss_mean", 0.0)
            fprox_loss_sum_w += wf * (float(fl_v) if _isfinite(fl_v) else 0.0)
            fp_v = cm.get("fedprox_penalty_mean", 0.0)
            fprox_penalty_sum_w += wf * (float(fp_v) if _isfinite(fp_v) else 0.0)

        loss_mean_weighted = (loss_sum_w / total_w_valid) if total_w_valid > 0 else None
        reconstruction_loss_mean_weighted = (
            (recon_sum_w / total_w_valid) if total_w_valid > 0 else None
        )
        fedprox_loss_mean_weighted = (
            (fprox_loss_sum_w / total_w_valid) if total_w_valid > 0 else None
        )
        fedprox_penalty_mean_weighted = (
            (fprox_penalty_sum_w / total_w_valid) if total_w_valid > 0 else None
        )

        last_loss_mean_weighted = loss_mean_weighted
        last_reconstruction_loss_mean_weighted = reconstruction_loss_mean_weighted
        last_fedprox_loss_mean_weighted = fedprox_loss_mean_weighted
        last_fedprox_penalty_mean_weighted = fedprox_penalty_mean_weighted

        comm_mb = estimate_communication_mb(param_count, len(selected))
        cumulative_comm_mb += comm_mb

        # Client update norm stats (Fase 4d). Aprovechamos drift_per_client
        # que ya almacena ||local_i - global||_2 por cliente. Mean/std
        # diagnostican heterogeneidad inter-cliente.
        _drift_vals = [float(d) for d in drift_per_client.values() if d is not None]
        if _drift_vals:
            client_update_norm_mean = sum(_drift_vals) / len(_drift_vals)
            _var = sum((d - client_update_norm_mean) ** 2 for d in _drift_vals) / len(_drift_vals)
            client_update_norm_std = _var ** 0.5
        else:
            client_update_norm_mean = None
            client_update_norm_std = None

        # LR stats agregadas por ronda (None si scheduler off). Como
        # `same_lr_for_all_clients_in_round=True` por construccion, basta
        # con tomar los del primer cliente con LR observado.
        lr_first_round = None
        lr_last_round = None
        lr_mean_round = None
        lr_min_round = None
        lr_max_round = None
        for cm in cm_list:
            if cm.get("lr_first") is not None:
                lr_first_round = cm["lr_first"]
                lr_last_round = cm["lr_last"]
                lr_mean_round = cm["lr_mean"]
                lr_min_round = cm["lr_min"]
                lr_max_round = cm["lr_max"]
                break

        round_log = {
            "kind": "round",
            "round": round_idx,
            "clients_participated": [c.name for c in selected],
            "loss_mean_weighted": loss_mean_weighted,
            # FedProx v0.2: agregados ponderados de la loss SSL pura y
            # del aporte proximal. En FedAvg, reconstruction == loss y
            # fedprox_* = 0. En FedProx, loss = reconstruction + prox.
            "reconstruction_loss_mean_weighted": reconstruction_loss_mean_weighted,
            "fedprox_loss_mean_weighted": fedprox_loss_mean_weighted,
            "fedprox_penalty_mean_weighted": fedprox_penalty_mean_weighted,
            "algorithm": algorithm_effective,
            "fedprox_mu": fedprox_mu_effective,
            "fedprox_enabled": fedprox_enabled,
            "loss_by_client": loss_per_client,
            "drift_by_client": drift_per_client,
            "aggregation_weight_policy": aggregation_policy,
            # Politica efectiva (W): combina policy declarado + origen
            # del plan. final_client_weight_capped_v23 si el plan tiene
            # caps; final_client_weight_raw_ncp_fallback si cayo al
            # fallback raw. Util para no confundir lo declarado con lo
            # aplicado en analisis posterior.
            "aggregation_weights_policy_effective": aggregation_policy_effective,
            "aggregation_weights_by_client": aggregation_weights_by_client,
            "aggregation_weights_sum": weights_sum,
            "communication_mb_estimated": comm_mb,
            "cumulative_communication_mb": cumulative_comm_mb,
            "max_effective_bc": max(
                int(cm["max_effective_bc"]) for cm in cm_list
            ) if cm_list else 0,
            "global_state_norm_before": old_norm,
            "global_state_norm_after": new_norm,
            # LR scheduler (v0.3, Fase 4c). None si scheduler off.
            "lr_first": lr_first_round,
            "lr_last": lr_last_round,
            "lr_mean": lr_mean_round,
            "lr_min": lr_min_round,
            "lr_max": lr_max_round,
            # Server momentum stats (Fase 4d). None si fedavgm off.
            "server_delta_norm": server_delta_norm_v,
            "server_velocity_norm": server_velocity_norm_v,
            # Heterogeneidad inter-cliente (siempre disponible).
            "client_update_norm_mean": client_update_norm_mean,
            "client_update_norm_std": client_update_norm_std,
        }
        logger.log(round_log)
        print(
            f"  round {round_idx}/{n_rounds}: "
            f"loss_mean_weighted={loss_mean_weighted:.4f}  "
            f"comm_mb={comm_mb:.1f}  "
            f"global_norm_delta={new_norm - old_norm:+.3f}"
        )

        # Checkpoint opcional
        ckpt_every = int(fl_cfg.get("ckpt_every_rounds", 0))
        if ckpt_every > 0 and round_idx % ckpt_every == 0:
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            torch.save(
                {"model_state_dict": global_sd, "round": round_idx, "config": cfg},
                ckpt_dir / f"ckpt_round{round_idx:03d}.pt",
            )

    # Checkpoint final
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    final_path = ckpt_dir / "ckpt_final.pt"
    torch.save({"model_state_dict": global_sd, "round": n_rounds, "config": cfg}, final_path)

    elapsed = time.time() - t0
    return {
        "n_rounds": n_rounds,
        "local_steps": n_local_steps,
        "total_local_optimizer_steps": total_local_optimizer_steps,
        "clients_seen": [c.name for c in clients],
        "datasets_seen_by_client": {
            k: sorted(v) for k, v in datasets_seen_by_client.items()
        },
        "param_count": param_count,
        "cumulative_communication_mb": cumulative_comm_mb,
        "aggregation_weight_policy": aggregation_policy,
        "checkpoint_final": str(final_path),
        "elapsed_seconds": round(elapsed, 2),
        # Estado agregado para smoke_pass criteria (U4):
        "state_norm_changed_in_any_round": state_norm_changed_in_any_round,
        "opt_steps_per_client_total": opt_steps_per_client_total,
        "loss_finite_per_client_round": loss_finite_per_client_round,
        "amp_nonfinite_grad_steps_total": amp_nonfinite_grad_steps_total,
        "max_effective_bc_global": max_effective_bc_global,
        # Logging de agregacion (V2/W):
        "aggregation_weights_by_client_last_round": last_aggregation_weights_by_client,
        "plan_policy_unique": plan_policy_unique,
        "aggregation_weights_policy_effective": aggregation_policy_effective,
        # FedProx v0.2: estado efectivo + last-round agregados ponderados.
        # En FedAvg, fedprox_enabled=False, fedprox_mu=None,
        # final_fedprox_loss_mean_weighted=0 y
        # final_reconstruction_loss_mean_weighted == final_loss_mean_weighted.
        "algorithm": algorithm_effective,
        "fedprox_mu": fedprox_mu_effective,
        "fedprox_enabled": fedprox_enabled,
        "final_loss_mean_weighted": last_loss_mean_weighted,
        "final_reconstruction_loss_mean_weighted": last_reconstruction_loss_mean_weighted,
        "final_fedprox_loss_mean_weighted": last_fedprox_loss_mean_weighted,
        "final_fedprox_penalty_mean_weighted": last_fedprox_penalty_mean_weighted,
        # LR scheduler v0.3 (Fase 4c). {"type": "constant"} si scheduler off.
        "lr_scheduler": scheduler_eff,
        # Server momentum v0.4 (Fase 4d). {"enabled": False} si fedavgm off.
        "server_momentum": (
            {
                "enabled": True,
                "beta": server_momentum_beta,
                "nesterov": server_momentum_nesterov,
                "initialize": server_momentum_init,
            }
            if server_momentum_enabled
            else {"enabled": False}
        ),
    }
