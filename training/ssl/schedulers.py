"""Schedulers de learning rate compatibles entre central y federado.

Diseño:

- Funciones PURAS (sin torch) para que los tests sean rápidos y precisos.
- `cosine_warmup_lr_factor(step, warmup_steps, total_steps)` reproduce
  bit-a-bit la receta del trainer central (`training/train_ssl_central._build_lr_scheduler`):
    * linear warmup en `[0, warmup_steps)`: factor = (step+1)/warmup_steps.
    * cosine decay en `[warmup_steps, total_steps]`: factor = 0.5 * (1 + cos(pi * progress)).
- `lr_for_round_step(...)` calcula el LR a aplicar en un step local concreto
  del trainer federado, respetando la política
  `step_accounting='aggregate_local_updates'` del plan:
    * `step_global = (round_idx - 1) * n_clients * local_steps_per_client
                     + (local_step_in_round - 1) * n_clients`.
    * todos los clientes en el mismo `local_step_in_round` de la misma ronda
      ven exactamente el mismo LR (depende solo de `step_global`,
      no del orden de iteración del cliente).
- Si `scheduler_cfg` es `None` o `type` ∈ {`constant`, ``""``}, devuelve
  `base_lr` sin tocar nada (backward-compatible con configs históricos).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional


def cosine_warmup_lr_factor(
    step: int, warmup_steps: int, total_steps: int
) -> float:
    """Factor multiplicativo (0..1) para el LR base.

    Compatibilidad con `training.train_ssl_central._build_lr_scheduler` cuando
    `schedule='cosine'` y `warmup_steps>0`.

    - `step < warmup_steps`         -> `(step + 1) / max(1, warmup_steps)`.
    - `warmup_steps <= step <= total_steps` -> cosine decay.
    - `step > total_steps`          -> clamp al final del cosine (factor=0).
    """
    s = int(step)
    w = int(warmup_steps)
    t = int(total_steps)
    if w < 0 or t < 0:
        raise ValueError(f"warmup_steps={w} y total_steps={t} deben ser >= 0")
    if s < w:
        return float(s + 1) / float(max(1, w))
    progress = (s - w) / max(1, t - w)
    progress = max(0.0, min(1.0, progress))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


def lr_for_round_step(
    *,
    round_idx: int,
    local_step_in_round: int,
    n_clients: int,
    local_steps_per_client: int,
    base_lr: float,
    scheduler_cfg: Optional[Dict[str, Any]],
) -> float:
    """Calcula el LR a aplicar en el step local indicado.

    Parameters
    ----------
    round_idx : int (1-indexed)
        Numero de ronda federada (la primera es 1).
    local_step_in_round : int (1-indexed)
        Numero de local step dentro de la ronda (el primero es 1).
    n_clients : int
        Numero total de clientes federados que participan en la ronda.
    local_steps_per_client : int
        Steps locales planeados por cliente y por ronda.
    base_lr : float
        LR base (el del config `training.lr`).
    scheduler_cfg : dict or None
        Config del scheduler. Claves esperadas:
        - `type`: "cosine" | "constant" (o ausencia => constant).
        - `warmup_steps`: int >= 0.
        - `total_steps`: int > 0.
        - `step_accounting`: "aggregate_local_updates" (unica soportada).

    Devuelve
    --------
    float
        El LR a poner en `optimizer.param_groups[*]['lr']` para ese step.
        Si scheduler_cfg es None o type=constant, devuelve `base_lr`.
    """
    if scheduler_cfg is None:
        return float(base_lr)
    stype = str(scheduler_cfg.get("type", "constant")).lower()
    if stype in ("", "constant"):
        return float(base_lr)
    if stype != "cosine":
        raise ValueError(f"lr_scheduler.type='{stype}' no soportado (solo 'cosine' o 'constant')")
    accounting = str(
        scheduler_cfg.get("step_accounting", "aggregate_local_updates")
    )
    if accounting != "aggregate_local_updates":
        raise ValueError(
            f"lr_scheduler.step_accounting='{accounting}' no soportado "
            "(solo 'aggregate_local_updates')"
        )

    warmup_steps = int(scheduler_cfg.get("warmup_steps", 0))
    total_steps = int(scheduler_cfg.get("total_steps", 0))
    if total_steps <= 0:
        raise ValueError("lr_scheduler.total_steps debe ser > 0 si type=cosine")

    if round_idx < 1 or local_step_in_round < 1:
        raise ValueError(
            f"round_idx ({round_idx}) y local_step_in_round "
            f"({local_step_in_round}) deben ser >= 1"
        )

    round_start = (int(round_idx) - 1) * int(n_clients) * int(local_steps_per_client)
    step_global = round_start + (int(local_step_in_round) - 1) * int(n_clients)
    factor = cosine_warmup_lr_factor(step_global, warmup_steps, total_steps)
    return float(base_lr) * factor


def scheduler_summary(scheduler_cfg: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Devuelve una vista normalizada del scheduler para serializar en run_info.

    Si scheduler_cfg es None o type=constant, devuelve `{"type": "constant"}`.
    """
    if scheduler_cfg is None:
        return {"type": "constant"}
    stype = str(scheduler_cfg.get("type", "constant")).lower()
    if stype in ("", "constant"):
        return {"type": "constant"}
    return {
        "type": stype,
        "warmup_steps": int(scheduler_cfg.get("warmup_steps", 0)),
        "total_steps": int(scheduler_cfg.get("total_steps", 0)),
        "step_accounting": str(
            scheduler_cfg.get("step_accounting", "aggregate_local_updates")
        ),
        "same_lr_for_all_clients_in_round": bool(
            scheduler_cfg.get("same_lr_for_all_clients_in_round", True)
        ),
    }
