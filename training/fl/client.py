"""FederatedClient: entrena local sobre sus PRETRAIN_SOURCE.

Bug metodologico evitado: la normalizacion intra-cliente de pesos
**NO debe mutar el plan original** (`self.plan_subset` o el `plan`
global del servidor). Si se mutara, llamadas posteriores a
`compute_aggregation_weights(plan, ...)` con policy
`final_client_weight` calcularian pesos incorrectos basandose en los
valores normalizados intra-cliente (todos sumarian 1 dentro del
cliente, dando la falsa apariencia de pesos uniformes entre clientes).

La funcion `normalize_plan_subset_for_loader` devuelve **copias**
shallow-isoladas (un dict nuevo por fila) con el campo
`final_dataset_weight` renormalizado a sumar 1 intra-cliente, sin
tocar el plan original ni `self.plan_subset`.

El cliente reusa al maximo el pipeline central:

- modelo: `models.patchtst_phm.PatchTSTPhm`.
- masking: `training.ssl.masking.generate_ssl_mask`.
- loss: `training.ssl.loss.compute_masked_reconstruction_loss_with_metrics`.
- dataloader: `training.phm_webdataset.build_centralized_loader` aplicado a
  un subset filtrado del sampling plan (solo los datasets del cliente).
- AMP y logging robusto: misma logica que `train_ssl_central.py`.

API:

    client = FederatedClient(name, plan_subset, processed_root, model_cfg,
                             ssl_cfg, training_cfg, data_cfg,
                             federated_cfg=cfg.get("federated", {}))
    metrics = client.local_train(global_state_dict, round_idx, n_local_steps)

Devuelve `(local_state_dict, metrics_dict)` donde metrics_dict incluye:
loss_mean, n_steps_attempted, optimizer_steps, amp_nonfinite_grad_steps,
max_effective_bc, datasets_seen, drift_l2_norm, elapsed_seconds,
algorithm, fedprox_mu, reconstruction_loss_mean, fedprox_penalty_mean,
fedprox_loss_mean.

FedProx (v0.2): si `federated_cfg["algorithm"] == "fedprox"` y
`federated_cfg["fedprox_mu"]` es float > 0, cada cliente anyade al
objetivo SSL el termino proximal:

    loss_total = loss_ssl + 0.5 * mu * sum_l ||theta_l - theta_global_l||^2

donde la suma corre sobre los parametros entrenables float del modelo
local y `theta_global_l` es un snapshot CLONADO/DETACHED del
state_dict global recibido al inicio de la ronda (no se muta).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence


def resolve_fedprox_config(
    federated_cfg: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Devuelve el estado FedProx efectivo a partir del bloque cfg["federated"].

    Centraliza la regla de activacion para que tanto el cliente como los
    tests reusen la misma logica:

    - FedAvg explicito (`algorithm=="fedavg"`): FedProx inactivo, mu=0.0.
    - FedAvg implicito (`algorithm` ausente o vacio): FedProx inactivo.
    - FedProx con `fedprox_mu` None / 0 / <0: FedProx **inactivo**, mu=0.0
      (intentar usar FedProx sin mu valido no debe rompler en runtime;
      la validacion estricta del config se hace aparte en
      `training.train_ssl_federated._validate_federated_config`).
    - FedProx con `fedprox_mu > 0`: FedProx activo con ese mu.

    Returns:
        {
          "algorithm": "fedavg" | "fedprox",
          "fedprox_mu": float (0.0 si inactivo, mu si activo),
          "fedprox_enabled": bool,
        }
    """
    fc = federated_cfg or {}
    algorithm = str(fc.get("algorithm", "fedavg")).strip().lower() or "fedavg"
    mu_raw = fc.get("fedprox_mu", None)
    fedprox_enabled = False
    fedprox_mu = 0.0
    if algorithm == "fedprox":
        try:
            mu_val = float(mu_raw) if mu_raw is not None else 0.0
        except (TypeError, ValueError):
            mu_val = 0.0
        if mu_val > 0:
            fedprox_enabled = True
            fedprox_mu = mu_val
    return {
        "algorithm": algorithm,
        "fedprox_mu": fedprox_mu,
        "fedprox_enabled": fedprox_enabled,
    }


def snapshot_global_params(
    model: Any,
    global_state_dict: Mapping[str, Any],
    device: Any = None,
) -> Dict[str, Any]:
    """Devuelve un dict {name -> tensor} con la copia DETACHED/CLONADA de
    los parametros entrenables float del modelo, leida del state_dict
    global recibido al inicio de la ronda.

    Reglas:

    - Solo se incluyen parametros con `requires_grad=True` y dtype float.
    - Buffers no-float (counters, masks integer) se ignoran.
    - El tensor se clona y mueve al `device` del modelo con el dtype del
      parametro local correspondiente (asi diff es elementwise sin casts).
    - NO muta `global_state_dict`.

    Util como snapshot para el termino proximal de FedProx. El termino
    proximal debe medir la distancia entre theta_local actual y
    theta_global del INICIO de la ronda (no del state_dict global vivo).
    """
    import torch
    snap: Dict[str, Any] = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not p.is_floating_point():
            continue
        if name not in global_state_dict:
            continue
        g = global_state_dict[name]
        if not isinstance(g, torch.Tensor) or not g.is_floating_point():
            continue
        gp = g.detach().clone()
        if device is not None:
            gp = gp.to(device=device, dtype=p.dtype)
        else:
            gp = gp.to(dtype=p.dtype)
        snap[name] = gp
    return snap


def compute_fedprox_penalty(
    model: Any,
    global_snapshot: Mapping[str, Any],
) -> Any:
    """Devuelve la suma `sum_l ||theta_l - theta_global_l||^2` como
    tensor escalar (NO escalado por mu).

    El caller multiplica por `0.5 * mu` antes de sumarlo a la loss SSL.

    Solo se recorren parametros entrenables float; los buffers no-float
    y los parametros no presentes en el snapshot se ignoran sin error
    (el snapshot ya filtra por contrato).

    Devuelve siempre un tensor escalar para que `.backward()` fluya. Si
    no hay parametros que sumar, devuelve un cero diferenciable
    `torch.zeros(())` en el device del primer parametro disponible (o
    CPU si el modelo esta vacio).
    """
    import torch
    # Resolver device de salida desde algun parametro del modelo (si
    # existe) para no perder grad-tracking.
    out_device = torch.device("cpu")
    for p in model.parameters():
        out_device = p.device
        break
    penalty = torch.zeros((), device=out_device)
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if not p.is_floating_point():
            continue
        g = global_snapshot.get(name, None)
        if g is None:
            continue
        diff = p - g
        penalty = penalty + (diff * diff).sum()
    return penalty


def normalize_plan_subset_for_loader(
    plan_subset: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Devuelve copias shallow del subset con `final_dataset_weight`
    renormalizado a sumar 1 INTRA-cliente.

    Garantia: ni el `plan_subset` recibido ni los dicts que contiene
    quedan modificados. El caller puede usar libremente el plan global
    despues de invocar esta funcion.

    Si la suma de pesos es 0 (caso degenerado), reparte uniforme entre
    las filas del subset.
    """
    if not plan_subset:
        return []
    copies = [dict(r) for r in plan_subset]
    total = float(sum(float(r.get("final_dataset_weight", 0.0)) for r in copies))
    if total > 0:
        for r in copies:
            r["final_dataset_weight"] = float(r.get("final_dataset_weight", 0.0)) / total
    else:
        n = max(1, len(copies))
        for r in copies:
            r["final_dataset_weight"] = 1.0 / n
    return copies


@dataclass
class FederatedClient:
    """Cliente FL en proceso (sin networking).

    `federated_cfg` se anade en v0.2 para soportar FedProx sin cambiar
    la signatura `local_train(global_state_dict, round_idx, n_local_steps)`.
    Default = `{}` mantiene el comportamiento FedAvg (FedProx desactivado),
    asi los clientes y tests existentes pre-v0.2 siguen funcionando sin
    cambios.
    """
    name: str
    plan_subset: List[Dict[str, Any]]            # subset del sampling plan
    processed_root: Path
    model_cfg: Dict[str, Any]
    ssl_cfg: Dict[str, Any]
    training_cfg: Dict[str, Any]
    data_cfg: Dict[str, Any]
    federated_cfg: Dict[str, Any] = field(default_factory=dict)
    # Scheduler LR opcional (v0.3, Fase 4c). Si None se mantiene el
    # comportamiento historico (LR constante = training.lr). Si tiene valor,
    # el server lo pone antes de cada round con:
    #   {"type": "cosine", "warmup_steps": int, "total_steps": int,
    #    "step_accounting": "aggregate_local_updates",
    #    "same_lr_for_all_clients_in_round": True}
    # y debe ir acompanado de _scheduler_n_clients y _scheduler_local_steps.
    lr_scheduler_cfg: Optional[Dict[str, Any]] = None
    # Numero de clientes en la ronda (para el step_accounting agregado).
    # Default = 10 = topologia FL del MVP; el server lo actualiza si cambia.
    n_clients_in_round: int = 10
    # Steps locales planeados por cliente y por ronda (mismo en todos los
    # clientes en este MVP, federated.local_steps).
    local_steps_per_client: int = 0
    device: Optional[Any] = None                  # torch.device
    seed: int = 42

    # Estado interno (no serializable, se reconstruye por ronda):
    _model: Any = field(default=None, repr=False)
    _datasets_seen: set = field(default_factory=set, repr=False)

    def datasets(self) -> List[str]:
        return [str(r["dataset"]) for r in self.plan_subset]

    def _build_model(self):
        from models.patchtst_phm import build_patchtst_phm
        if self._model is None:
            self._model = build_patchtst_phm(self.model_cfg)
            if self.device is not None:
                self._model.to(self.device)
        return self._model

    def local_train(
        self,
        global_state_dict: Dict[str, Any],
        round_idx: int,
        n_local_steps: int,
    ) -> Dict[str, Any]:
        """Entrena `n_local_steps` localmente y devuelve metrics + state_dict local."""
        import torch
        from training.ssl.masking import generate_ssl_mask
        from training.ssl.loss import compute_masked_reconstruction_loss_with_metrics
        from training.phm_webdataset import build_centralized_loader
        from training.fl.aggregation import compute_drift_l2

        t0 = time.time()
        device = self.device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        model = self._build_model()
        # Cargar pesos globales (los del servidor).
        model.load_state_dict(global_state_dict)
        model.train()

        # Optimizer fresco por ronda (FedAvg estandar).
        lr = float(self.training_cfg.get("lr", 3e-4))
        wd = float(self.training_cfg.get("weight_decay", 0.05))
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr, weight_decay=wd,
        )
        # LR scheduler opcional (v0.3, Fase 4c). Si esta activo, el LR del
        # optimizer se sobreescribe en cada local_step segun la receta
        # `lr_for_round_step` con step_accounting='aggregate_local_updates'.
        # Si esta None, comportamiento historico = LR constante = lr base.
        from training.ssl.schedulers import lr_for_round_step
        _scheduler_enabled = (
            self.lr_scheduler_cfg is not None
            and str(self.lr_scheduler_cfg.get("type", "constant")).lower() != "constant"
        )
        lr_observed: list = []  # LR aplicado en cada local_step (para metrics)

        amp_cfg = self.training_cfg.get("amp", "auto")
        use_amp = (amp_cfg in ("auto", True)) and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        grad_clip = float(self.training_cfg.get("grad_clip_norm", 1.0))

        # FedProx v0.2: leer estado efectivo desde federated_cfg. Default =
        # FedAvg (fedprox_enabled=False, mu=0.0).
        fprox = resolve_fedprox_config(self.federated_cfg)
        fedprox_enabled = bool(fprox["fedprox_enabled"])
        fedprox_mu = float(fprox["fedprox_mu"])
        algorithm = str(fprox["algorithm"])

        # Snapshot global por ronda: copia clonada/detached de los
        # parametros entrenables float, en device/dtype del modelo local.
        # Solo se construye si FedProx esta activo, para no pagar memoria
        # extra en FedAvg.
        global_snapshot: Dict[str, Any] = {}
        if fedprox_enabled:
            global_snapshot = snapshot_global_params(
                model, global_state_dict, device=device,
            )

        # Dataloader filtrado: solo los datasets del cliente. Usamos el
        # helper puro `normalize_plan_subset_for_loader` que devuelve
        # COPIAS con pesos renormalizados a sumar 1 intra-cliente, SIN
        # mutar el plan original ni self.plan_subset. Esto es critico
        # porque el servidor reutiliza el plan global mas tarde con
        # `compute_aggregation_weights(plan, policy="final_client_weight")`
        # y necesita los pesos sin normalizar intra-cliente.
        sub = normalize_plan_subset_for_loader(self.plan_subset)

        loader = build_centralized_loader(
            plan=sub,
            processed_root=self.processed_root,
            batch_size=int(self.data_cfg.get("batch_size", 32)),
            split=str(self.data_cfg.get("split", "train")),
            seed=self.seed + round_idx,
            max_steps=n_local_steps,
            strategy=str(self.data_cfg.get("client_sampling_strategy", "weighted")),
            batch_size_policy=str(self.data_cfg.get("batch_size_policy", "fixed")),
            max_channel_batch=self.data_cfg.get("max_channel_batch"),
            min_batch_size=int(self.data_cfg.get("min_batch_size", 1)),
        )

        # SSL params
        mask_ratio = float(self.ssl_cfg.get("mask_ratio", 0.3))
        loss_kind = str(self.ssl_cfg.get("loss", "mse"))

        loss_total = 0.0                     # suma de la loss usada para optimizar
        reconstruction_loss_total = 0.0      # suma de la loss SSL pura (sin proximal)
        fedprox_penalty_total = 0.0          # suma de ||theta-theta_global||^2 (sin 0.5*mu)
        fedprox_loss_total = 0.0             # suma de 0.5 * mu * penalty (lo que se suma a SSL)
        loss_terms_count = 0
        n_steps_attempted = 0
        optimizer_steps = 0
        amp_nonfinite_grad_steps = 0
        max_effective_bc = 0
        skipped_steps_no_loss_elements = 0
        last_nonfinite_grad_kind: list = []  # "inf" / "nan"

        for batch in loader:
            n_steps_attempted += 1
            # Si scheduler activo, aplicar LR antes del forward/backward.
            # El step counter es n_steps_attempted (1-indexed); avanza
            # incluso si el step se omite (sin loss_elements o grad nonfinite),
            # igual que el central avanza el scheduler por step intentado.
            if _scheduler_enabled:
                lr_step = lr_for_round_step(
                    round_idx=int(round_idx),
                    local_step_in_round=int(n_steps_attempted),
                    n_clients=int(self.n_clients_in_round),
                    local_steps_per_client=int(self.local_steps_per_client),
                    base_lr=float(lr),
                    scheduler_cfg=self.lr_scheduler_cfg,
                )
                for pg in optimizer.param_groups:
                    pg["lr"] = lr_step
                lr_observed.append(float(lr_step))
            x = batch["patches"].to(device, non_blocking=True)
            vtm = batch["valid_time_mask"].to(device, non_blocking=True)
            vpm = batch["valid_patch_mask"].to(device, non_blocking=True)
            self._datasets_seen.add(batch.get("__dataset__"))
            max_effective_bc = max(max_effective_bc, int(batch["__effective_bc__"]))

            gen = torch.Generator(device="cpu")
            gen.manual_seed(int(self.seed) + int(round_idx) * 10_000 + int(n_steps_attempted))
            ssl_mask = generate_ssl_mask(vpm.cpu(), mask_ratio=mask_ratio, generator=gen).to(vpm.device)

            optimizer.zero_grad(set_to_none=True)
            # NOTA: compute_masked_reconstruction_loss_with_metrics devuelve
            # un DICT (no tupla). Pasamos valid_patch_mask explicitamente
            # como red de seguridad. Recuperamos loss y n_loss_elements.
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(x, vtm, vpm, ssl_mask=ssl_mask)
                    loss_metrics = compute_masked_reconstruction_loss_with_metrics(
                        out["reconstruction"], x, ssl_mask, vtm,
                        valid_patch_mask=vpm, loss_fn=loss_kind,
                    )
                    loss_ssl = loss_metrics["loss"]
                    # FedProx: anyadir 0.5*mu*||theta-theta_global||^2 si
                    # esta activo. La penalty se calcula dentro del autocast
                    # context para coherencia de precision con la loss SSL.
                    if fedprox_enabled:
                        prox_penalty = compute_fedprox_penalty(
                            model, global_snapshot,
                        )
                        prox_loss = 0.5 * fedprox_mu * prox_penalty
                        loss = loss_ssl + prox_loss
                    else:
                        prox_penalty = None
                        prox_loss = None
                        loss = loss_ssl
            else:
                out = model(x, vtm, vpm, ssl_mask=ssl_mask)
                loss_metrics = compute_masked_reconstruction_loss_with_metrics(
                    out["reconstruction"], x, ssl_mask, vtm,
                    valid_patch_mask=vpm, loss_fn=loss_kind,
                )
                loss_ssl = loss_metrics["loss"]
                if fedprox_enabled:
                    prox_penalty = compute_fedprox_penalty(
                        model, global_snapshot,
                    )
                    prox_loss = 0.5 * fedprox_mu * prox_penalty
                    loss = loss_ssl + prox_loss
                else:
                    prox_penalty = None
                    prox_loss = None
                    loss = loss_ssl
            n_loss_elements = int(loss_metrics["n_loss_elements"])

            # Si el batch no aporta ningun elemento valido (e.g. todos los
            # patches enmascarados caen en padding), saltamos sin contar
            # como optimizer_step. NO es fallo: es contrato del wrapper
            # de loss (sec 14 CLAUDE.md).
            if n_loss_elements == 0:
                skipped_steps_no_loss_elements += 1
                continue

            # Loss no finita = fallo duro (igual que en train_ssl_central).
            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"client={self.name} round={round_idx} step={n_steps_attempted}: "
                    f"loss no finita ({float(loss.detach())}). Abortando."
                )

            if use_amp:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
                # scaler.step internamente comprueba finitud; si no, omite
                # el optimizer.step y reduce el scale.
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], grad_clip,
                )
                # Sin AMP, grad_norm no finito es problema real (NaN en el
                # modelo o datos corruptos): abortar para no entrenar basura.
                if not torch.isfinite(grad_norm):
                    raise RuntimeError(
                        f"client={self.name} round={round_idx} step={n_steps_attempted}: "
                        f"grad_norm no finito sin AMP ({float(grad_norm.detach())}). "
                        "Abortando."
                    )
                optimizer.step()

            gn_finite = bool(torch.isfinite(grad_norm).item())
            optimizer_applied = bool(gn_finite or not use_amp)
            if not gn_finite:
                amp_nonfinite_grad_steps += 1
                gn_raw = float(grad_norm.detach())
                last_nonfinite_grad_kind.append("nan" if gn_raw != gn_raw else "inf")
            if optimizer_applied:
                optimizer_steps += 1
                loss_total += float(loss.detach())
                reconstruction_loss_total += float(loss_ssl.detach())
                if fedprox_enabled:
                    fedprox_penalty_total += float(prox_penalty.detach())
                    fedprox_loss_total += float(prox_loss.detach())
                loss_terms_count += 1

        local_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        drift = compute_drift_l2(
            local_sd, {k: v.cpu() for k, v in global_state_dict.items()}
        )

        # Medias por step optimizado. Si el cliente no aporto loss valida
        # (loss_terms_count == 0), todos quedan None / 0 segun semantica.
        def _mean(total: float) -> Optional[float]:
            return (total / loss_terms_count) if loss_terms_count > 0 else None

        # LR stats por ronda (None si scheduler off). Util para diagnosticar
        # el efecto del cosine+warmup vs constante.
        if lr_observed:
            lr_first = lr_observed[0]
            lr_last = lr_observed[-1]
            lr_mean_v = sum(lr_observed) / len(lr_observed)
            lr_min = min(lr_observed)
            lr_max = max(lr_observed)
        else:
            lr_first = lr_last = lr_mean_v = lr_min = lr_max = None

        return {
            "client": self.name,
            "round": round_idx,
            "n_local_steps_requested": n_local_steps,
            "n_steps_attempted": n_steps_attempted,
            "optimizer_steps": optimizer_steps,
            "amp_nonfinite_grad_steps": amp_nonfinite_grad_steps,
            "skipped_steps_no_loss_elements": skipped_steps_no_loss_elements,
            "grad_norm_nonfinite_kinds": list(last_nonfinite_grad_kind),
            # LR stats por ronda (None si scheduler off / LR constante).
            "lr_first": lr_first,
            "lr_last": lr_last,
            "lr_mean": lr_mean_v,
            "lr_min": lr_min,
            "lr_max": lr_max,
            # Loss total optimizada. Para FedAvg coincide con la
            # reconstruction; para FedProx incluye el termino proximal.
            "loss_mean": _mean(loss_total),
            # SSL reconstruction loss pura (siempre disponible, util para
            # comparar convergencia entre FedAvg y FedProx en igualdad de
            # condiciones).
            "reconstruction_loss_mean": _mean(reconstruction_loss_total),
            # Penalty `sum ||theta-theta_global||^2` sin escalar por
            # 0.5*mu. Diagnostico de drift intra-paso. En FedAvg es 0.0.
            "fedprox_penalty_mean": (
                _mean(fedprox_penalty_total) if fedprox_enabled else 0.0
            ),
            # 0.5 * mu * penalty: aporte exacto sumado a la loss SSL en
            # FedProx. En FedAvg es 0.0.
            "fedprox_loss_mean": (
                _mean(fedprox_loss_total) if fedprox_enabled else 0.0
            ),
            "fedprox_mu": (fedprox_mu if fedprox_enabled else None),
            "algorithm": algorithm,
            "fedprox_enabled": fedprox_enabled,
            "loss_finite_in_any_step": (loss_terms_count > 0),
            "max_effective_bc": max_effective_bc,
            "datasets_seen": sorted(d for d in self._datasets_seen if d is not None),
            "drift_l2_norm": float(drift),
            "elapsed_seconds": round(time.time() - t0, 2),
            "local_state_dict": local_sd,
        }


def filter_plan_by_client(
    plan: Sequence[Dict[str, Any]], client_name: str
) -> List[Dict[str, Any]]:
    """Devuelve solo las filas del sampling plan que pertenecen al cliente."""
    return [r for r in plan if str(r.get("client")) == str(client_name)]


def build_clients_from_audit_groups(
    audit_groups: Dict[str, Any],
    plan: Sequence[Dict[str, Any]],
    processed_root: Path,
    model_cfg: Dict[str, Any],
    ssl_cfg: Dict[str, Any],
    training_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    federated_cfg: Optional[Dict[str, Any]] = None,
    device: Optional[Any] = None,
    seed: int = 42,
) -> List[FederatedClient]:
    """Construye un `FederatedClient` por entry en `audit_groups['clients']`.

    `federated_cfg` (v0.2) se pasa a cada cliente para que `local_train`
    pueda activar FedProx cuando proceda. Default `None` -> `{}` mantiene
    el comportamiento FedAvg historico (FedProx desactivado).
    """
    if federated_cfg is None:
        federated_cfg = {}
    clients_meta = audit_groups.get("clients", {})
    out: List[FederatedClient] = []
    for client_name in sorted(clients_meta.keys()):
        sub = filter_plan_by_client(plan, client_name)
        if not sub:
            # Cliente sin PS en el plan: omitir (no deberia pasar para corpus PS).
            continue
        out.append(FederatedClient(
            name=client_name,
            plan_subset=sub,
            processed_root=processed_root,
            model_cfg=model_cfg,
            ssl_cfg=ssl_cfg,
            training_cfg=training_cfg,
            data_cfg=data_cfg,
            federated_cfg=dict(federated_cfg),
            device=device,
            seed=seed,
        ))
    return out
