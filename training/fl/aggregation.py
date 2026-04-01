"""Agregacion FedAvg de state_dicts.

Contrato:

    fedavg_state_dict(state_dicts, weights) -> aggregated_state_dict

donde `state_dicts` es una secuencia de mapping `name -> Tensor` y
`weights` es una secuencia de floats positivos (pesos de cliente para
la media). Si `weights` no se pasa, se asume uniform.

Reglas:

- Todos los state_dicts deben tener exactamente las mismas keys y
  shapes; si no, `ValueError`.
- Solo se promedian tensores en punto flotante. Para buffers no-float
  (counters, masks integer, etc.) se devuelve el valor del primer
  cliente, y se verifica con assert que todos coinciden.
- Si la suma de pesos es 0 o cualquier peso es negativo, `ValueError`.

El modulo NO importa torch al cargar para que los tests basicos de
validacion de pesos funcionen sin torch. La logica real de promedio
necesita torch y se importa lazy dentro de la funcion.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def _validate_weights(weights: Sequence[float], n_clients: int) -> list:
    if len(weights) != n_clients:
        raise ValueError(
            f"len(weights)={len(weights)} != n_clients={n_clients}"
        )
    if any((w is None) or (w < 0) or (w != w) for w in weights):
        raise ValueError(
            f"weights con valor invalido (None, negativo o NaN): {weights}"
        )
    s = float(sum(weights))
    if s <= 0:
        raise ValueError(f"sum(weights)={s} no positivo")
    return [float(w) / s for w in weights]


def _validate_keys_and_shapes(state_dicts: Sequence[Mapping[str, Any]]) -> None:
    if not state_dicts:
        raise ValueError("state_dicts vacio")
    keys0 = set(state_dicts[0].keys())
    if not keys0:
        raise ValueError("state_dict[0] sin keys")
    for i, sd in enumerate(state_dicts[1:], 1):
        keys = set(sd.keys())
        if keys != keys0:
            missing = keys0 - keys
            extra = keys - keys0
            raise ValueError(
                f"keys mismatch en state_dict[{i}]: missing={sorted(missing)[:5]}, "
                f"extra={sorted(extra)[:5]}"
            )
    # Shapes (solo si los valores son tensores con .shape; si no, lo dejamos pasar
    # y verificamos en la pasada de mean).
    ref = state_dicts[0]
    for i, sd in enumerate(state_dicts[1:], 1):
        for k in keys0:
            r = ref[k]
            v = sd[k]
            if hasattr(r, "shape") and hasattr(v, "shape"):
                if tuple(r.shape) != tuple(v.shape):
                    raise ValueError(
                        f"shape mismatch en key {k!r}: client0={tuple(r.shape)} "
                        f"vs client{i}={tuple(v.shape)}"
                    )


def fedavg_state_dict(
    state_dicts: Sequence[Mapping[str, Any]],
    weights: Sequence[float] = None,
) -> Mapping[str, Any]:
    """FedAvg ponderada: agg[k] = sum(w_i * sd_i[k]) / sum(w_i).

    Args:
        state_dicts: secuencia de state_dicts (mismas keys y shapes).
        weights: pesos por cliente (positivos). Si None, uniform.

    Returns:
        dict con las mismas keys, valores promediados.

    Raises:
        ValueError si validaciones fallan.
    """
    if not state_dicts:
        raise ValueError("state_dicts vacio")
    n = len(state_dicts)
    if weights is None:
        w_norm = [1.0 / n] * n
    else:
        w_norm = _validate_weights(weights, n)

    _validate_keys_and_shapes(state_dicts)

    # Import lazy de torch: solo necesario si los valores son tensores.
    import torch

    aggregated = {}
    ref = state_dicts[0]
    for k in ref.keys():
        ref_v = ref[k]
        # Tensor floating-point: promediamos ponderado.
        if isinstance(ref_v, torch.Tensor) and ref_v.is_floating_point():
            acc = torch.zeros_like(ref_v)
            for sd, w in zip(state_dicts, w_norm):
                v = sd[k]
                if not isinstance(v, torch.Tensor):
                    raise TypeError(
                        f"key {k!r}: cliente con valor no-tensor donde se espera "
                        f"torch.Tensor: type={type(v)}"
                    )
                acc = acc + (v.to(acc.dtype) * float(w))
            aggregated[k] = acc
        elif isinstance(ref_v, torch.Tensor):
            # Tensor no-float (int, bool, ...): no promediamos; verificamos
            # que todos los clientes coinciden, y copiamos el valor de
            # referencia.
            for i, sd in enumerate(state_dicts[1:], 1):
                v = sd[k]
                if not torch.equal(ref_v, v):
                    raise ValueError(
                        f"key {k!r} es tensor no-floating y los clientes "
                        f"divergen (client0 vs client{i}). FedAvg no puede "
                        "promediar este tipo."
                    )
            aggregated[k] = ref_v.clone()
        else:
            # Valor no-tensor: igual, verificamos coincidencia y copiamos.
            for i, sd in enumerate(state_dicts[1:], 1):
                v = sd[k]
                if v != ref_v:
                    raise ValueError(
                        f"key {k!r} es no-tensor y los clientes divergen "
                        f"(client0={ref_v} vs client{i}={v})."
                    )
            aggregated[k] = ref_v

    return aggregated


def estimate_communication_mb(
    param_count: int, n_clients_participated: int, bytes_per_param: int = 4
) -> float:
    """Estima coste de comunicacion bidireccional en MB para una ronda.

    Modelo simple:
      - servidor envia el state_dict global a cada cliente: param_count * bytes_per_param
      - cliente devuelve su state_dict actualizado: idem
      - total = 2 * param_count * bytes_per_param * n_clients_participated

    Args:
        param_count: numero de parametros del modelo.
        n_clients_participated: clientes activos en la ronda.
        bytes_per_param: 4 para float32, 2 para float16/bfloat16, 1 para int8.

    Returns:
        Estimacion en MB (float).
    """
    if param_count <= 0 or n_clients_participated <= 0 or bytes_per_param <= 0:
        return 0.0
    total_bytes = 2 * param_count * bytes_per_param * n_clients_participated
    return total_bytes / (1024.0 * 1024.0)


# ----------------------------------------------------------------------
# Server momentum (FedAvgM, Hsu et al. 2019) — Fase 4d
# ----------------------------------------------------------------------


def state_dict_l2_norm(sd: Mapping[str, Any]) -> float:
    """L2 norm global sobre todos los tensores floating-point del state_dict.

    Util como diagnostico de tamaño del delta agregado o de la velocity
    del servidor. Devuelve 0.0 si no hay tensores floating.
    """
    import torch
    if not sd:
        return 0.0
    sq_sum = 0.0
    for v in sd.values():
        if isinstance(v, torch.Tensor) and v.is_floating_point():
            d = v.detach().float()
            sq_sum += float((d * d).sum().item())
    return float(sq_sum) ** 0.5


def compute_state_dict_delta(
    sd_new: Mapping[str, Any], sd_old: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Devuelve un state_dict con `delta[k] = sd_new[k] - sd_old[k]`.

    Solo procesa tensores floating-point con shapes compatibles entre ambos
    state_dicts. Keys no-tensor o int/bool se omiten (no se restan).
    """
    import torch
    if not sd_new or not sd_old:
        return {}
    keys = set(sd_new.keys()) & set(sd_old.keys())
    delta: Dict[str, Any] = {}
    for k in keys:
        a = sd_new[k]
        b = sd_old[k]
        if (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.is_floating_point() and b.is_floating_point()
                and tuple(a.shape) == tuple(b.shape)):
            delta[k] = a.detach() - b.detach()
    return delta


def apply_server_momentum(
    sd_global: Mapping[str, Any],
    velocity_prev: Mapping[str, Any] | None,
    delta: Mapping[str, Any],
    beta: float,
    nesterov: bool = False,
) -> tuple:
    """Aplica FedAvgM (Hsu et al. 2019) al modelo global.

    Convencion clasica (`lr_server=1` implicito):

        v_t = beta * v_{t-1} + delta_t
        w_{t+1} = w_t + v_t              (sin Nesterov)
        w_{t+1} = w_t + (delta_t + beta * v_t)   (con Nesterov)

    Cuando `beta == 0`, `v_t == delta_t` y `w_{t+1} == w_t + delta_t`,
    que es equivalente a FedAvg estandar. Esto garantiza
    **backward-compat**: pasar beta=0 reproduce el comportamiento
    historico.

    Args:
        sd_global: state_dict del servidor antes de la ronda.
        velocity_prev: velocity acumulada (mismo shape que delta). Si None,
            se inicializa a zeros para cada key presente en delta.
        delta: state_dict diferencial `delta = aggregated - sd_global`.
        beta: factor de momentum (0 = sin momentum).
        nesterov: si True, aplica Nesterov-momentum.

    Returns:
        (sd_new, velocity_new): nuevo state_dict global y nueva velocity.
        Solo los keys que estan en `delta` se actualizan; el resto se copia
        de `sd_global` tal cual (preserva buffers no-float, masks integer,
        etc.).
    """
    import torch
    if not (0.0 <= float(beta) <= 1.0):
        raise ValueError(f"beta={beta} fuera de rango [0, 1]")

    if velocity_prev is None:
        velocity_prev = {k: torch.zeros_like(v) for k, v in delta.items()}

    velocity_new: Dict[str, Any] = {}
    sd_new: Dict[str, Any] = {}
    for k, v in sd_global.items():
        if k in delta:
            d = delta[k]
            v_old = velocity_prev.get(k)
            if v_old is None:
                v_old = torch.zeros_like(d)
            v_new = float(beta) * v_old + d
            velocity_new[k] = v_new
            update = (d + float(beta) * v_new) if nesterov else v_new
            sd_new[k] = v + update
        else:
            # Keys no-float o no presentes en delta: copiar tal cual.
            sd_new[k] = v.clone() if hasattr(v, "clone") else v
    # Asegurar que velocity_new tiene un entry por cada delta key (incluso
    # si sd_global no tenia ese key, caso raro pero correcto).
    for k in delta:
        if k not in velocity_new:
            v_old = velocity_prev.get(k)
            if v_old is None:
                v_old = torch.zeros_like(delta[k])
            velocity_new[k] = float(beta) * v_old + delta[k]
    return sd_new, velocity_new


def compute_drift_l2(
    sd_local: Mapping[str, Any], sd_global: Mapping[str, Any]
) -> float:
    """L2 norm de (sd_local - sd_global) sobre tensores floating-point.

    Util como diagnostico de no-IID: cuanto mayor el drift por cliente,
    mayor heterogeneidad.

    Returns:
        norma L2 total (sqrt(sum of squares)), 0.0 si no hay tensores
        floating o si los state_dicts no son comparables.
    """
    import torch
    if not sd_local or not sd_global:
        return 0.0
    keys = set(sd_local.keys()) & set(sd_global.keys())
    sq_sum = 0.0
    for k in keys:
        a = sd_local[k]
        b = sd_global[k]
        if (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)
                and a.is_floating_point() and b.is_floating_point()
                and tuple(a.shape) == tuple(b.shape)):
            diff = (a.detach() - b.detach()).flatten()
            sq_sum += float((diff * diff).sum().item())
    return float(sq_sum) ** 0.5
