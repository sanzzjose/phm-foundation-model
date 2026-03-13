"""Politica de sampling para el SSL pretraining.

Sec 7 y sec 7.bis-sampling de `CLAUDE.md` definen la politica cerrada tras
el audit v2.3:

    base_weight:            n_channel_patches
    cap_max_dataset_weight: 0.10
    cap_max_client_weight:  0.25
    min_client_presence:    0.005

Esta politica es comun a centralizado y federado. La diferencia entre
ambos regimenes esta en como se aplica (rotacion de datasets en
centralizado, rondas por cliente en federado), no en los pesos base.

La funcion canonica `compute_pretraining_sampling_plan` produce una tabla
con un peso final por dataset y por cliente que respeta los caps. Se
escribe a `results/pretraining/ssl_sampling_plan.csv` en dry-run.

Aspectos clave del algoritmo:

1. **Solo PRETRAIN_SOURCE**: TRANSFER_TARGET nunca participa en pretraining.
2. **Pesos base por dataset**: proporcionales a `n_channel_patches`
   (numero de patches validos canal-vez).
3. **Caps con redistribucion**: cuando un dataset supera 0.10, se trunca y
   el exceso se redistribuye proporcionalmente entre los demas. Lo mismo
   para clientes con cap 0.25. El proceso es iterativo hasta convergencia.
4. **`min_client_presence` como piso**: el cliente con menor peso (en este
   corpus, `cnc_milling`) recibe al menos 0.005 para garantizar que sus
   datasets se vean en cada epoca.
5. **Consistencia**: el peso final por cliente es la suma de los pesos
   finales por dataset que pertenecen a ese cliente. Se valida con asserts.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# Constantes canonicas (sec 7.bis-sampling de CLAUDE.md)
CAP_MAX_DATASET_WEIGHT = 0.10
CAP_MAX_CLIENT_WEIGHT = 0.25
MIN_CLIENT_PRESENCE = 0.005
# Piso por dataset opcional. Default 0.0 (sin piso, como en el pilot weighted).
# Se activa en el full real para garantizar cobertura minima de datasets pequenos
# (SSPSNASA15, DUS20, CESNASA15) que con weighted puro pueden no aparecer.
MIN_DATASET_PRESENCE_DEFAULT = 0.0


def compute_adaptive_batch_size(
    n_channels: int,
    batch_size: int,
    max_channel_batch: Optional[int] = None,
    min_batch_size: int = 1,
) -> int:
    """Calcula el batch_size efectivo por dataset segun su numero de canales.

    Para el contrato channel-independent, el coste real del transformer es
    proporcional a `B * C * N`. Si `C` varia entre datasets (DUS20 con C=1,
    PHM14 con C=317), un `batch_size` fijo desequilibra el coste compute y
    el uso de VRAM. La politica adaptativa limita `B * C <= max_channel_batch`:

        B_eff = min(batch_size, max(min_batch_size, max_channel_batch // C))

    Args:
        n_channels: C real del dataset.
        batch_size: techo absoluto (batch_size del config).
        max_channel_batch: presupuesto B*C. Si None o <= 0, devuelve
            simplemente `batch_size` (politica `fixed`).
        min_batch_size: piso (default 1: incluso PHM14 con C=317 lee >=1).

    Returns:
        Batch size efectivo `B_eff` en `[min_batch_size, batch_size]`.

    Raises:
        ValueError si n_channels <= 0 o batch_size <= 0 o min_batch_size < 1.
    """
    if n_channels <= 0:
        raise ValueError(f"n_channels debe ser >0, recibido {n_channels}")
    if batch_size <= 0:
        raise ValueError(f"batch_size debe ser >0, recibido {batch_size}")
    if min_batch_size < 1:
        raise ValueError(f"min_batch_size debe ser >=1, recibido {min_batch_size}")
    if max_channel_batch is None or max_channel_batch <= 0:
        return int(batch_size)
    by_cap = max_channel_batch // n_channels
    by_cap = max(min_batch_size, int(by_cap))
    return int(min(batch_size, by_cap))


@dataclass
class SamplingPlanRow:
    """Schema documental de una fila del sampling plan.

    Las filas reales se construyen como `dict` (mas flexible al expandir
    campos opcionales). Este dataclass se mantiene como referencia exacta
    de las columnas que el CSV y la API publica garantizan. Actualizado
    tras `min_dataset_presence` y la recalculacion de `final_client_weight`
    como groupby de `final_dataset_weight`.
    """
    dataset: str
    client: str
    n_channels: int
    n_channel_patches: int
    raw_dataset_weight: float
    capped_dataset_weight: float
    final_dataset_weight: float
    raw_client_weight: float
    final_client_weight: float
    expected_batches_per_epoch: Optional[float]
    capped_dataset: bool
    capped_client: bool
    min_presence_applied: bool
    min_dataset_presence_applied: bool


def _cap_with_redistribution(weights: np.ndarray, cap: float, max_iter: int = 100) -> np.ndarray:
    """Aplica `cap` redistribuyendo el exceso proporcionalmente.

    Algoritmo:
      1. Si ningun peso supera `cap`, devuelve `weights` tal cual.
      2. Si alguno supera, lo recorta a `cap`, suma el exceso total y lo
         redistribuye entre los demas en proporcion a sus pesos actuales.
      3. Repite hasta convergencia o `max_iter`.

    Si `cap * len(weights) < 1`, el cap es matematicamente imposible y se
    devuelve la distribucion uniforme `1/len(weights)` con un warning.
    Lo gestiona el caller via flags.
    """
    w = weights.copy().astype(np.float64)
    n = len(w)
    if cap * n < 1.0 - 1e-9:
        # Imposible matematicamente. Devolvemos uniforme.
        return np.full(n, 1.0 / n)

    for _ in range(max_iter):
        excess_idx = w > cap + 1e-12
        if not excess_idx.any():
            break
        excess = (w[excess_idx] - cap).sum()
        w[excess_idx] = cap
        free_idx = ~excess_idx
        if free_idx.sum() == 0:
            # Todos en el cap → no se puede redistribuir
            break
        free_total = w[free_idx].sum()
        if free_total <= 0:
            # Reparto uniforme entre libres
            w[free_idx] = excess / free_idx.sum()
        else:
            w[free_idx] += excess * (w[free_idx] / free_total)

    # Normalizacion final por estabilidad numerica
    s = w.sum()
    if s > 0:
        w = w / s
    return w


def _apply_min_client_presence(
    client_weights: Dict[str, float], min_presence: float
) -> Tuple[Dict[str, float], List[str]]:
    """Garantiza que ningun cliente tenga peso < `min_presence`.

    Si alguno esta por debajo, se sube a `min_presence` y se descuenta
    proporcionalmente de los demas. Devuelve los nuevos pesos normalizados
    y la lista de clientes a los que se aplico el piso.
    """
    w = dict(client_weights)
    floored: List[str] = []
    if min_presence <= 0:
        return w, floored

    for _ in range(100):
        below = {k: v for k, v in w.items() if v < min_presence}
        if not below:
            break
        delta = sum(min_presence - v for v in below.values())
        above = {k: v for k, v in w.items() if v >= min_presence and k not in floored}
        if not above:
            break
        total_above = sum(above.values())
        if total_above <= delta:
            break  # imposible
        for k in below:
            w[k] = min_presence
            if k not in floored:
                floored.append(k)
        for k, v in above.items():
            w[k] = v - delta * (v / total_above)

    s = sum(w.values())
    if s > 0:
        w = {k: v / s for k, v in w.items()}
    return w, floored


def compute_pretraining_sampling_plan(
    processed_summary: Sequence[Dict[str, Any]],
    client_summary: Sequence[Dict[str, Any]],
    audit_summary: Dict[str, Any],
    steps_per_epoch: Optional[int] = None,
    cap_max_dataset_weight: float = CAP_MAX_DATASET_WEIGHT,
    cap_max_client_weight: float = CAP_MAX_CLIENT_WEIGHT,
    min_client_presence: float = MIN_CLIENT_PRESENCE,
    min_dataset_presence: float = MIN_DATASET_PRESENCE_DEFAULT,
) -> List[Dict[str, Any]]:
    """Calcula el plan de sampling para SSL pretraining.

    Args:
        processed_summary: filas de `results/processed_summary.csv` (dict).
        client_summary: filas de `results/client_summary.csv` (dict).
        audit_summary: contenido de `results/audit/audit_summary.json`.
        steps_per_epoch: si se pasa, se calcula `expected_batches_per_epoch`.
        cap_max_dataset_weight: cap por dataset (default 0.10).
        cap_max_client_weight: cap por cliente (default 0.25).
        min_client_presence: piso por cliente (default 0.005).
        min_dataset_presence: piso por dataset (default 0.0 = sin piso).
            Cuando > 0, garantiza que ningun dataset tenga
            `final_dataset_weight < min_dataset_presence`. Util para el
            full real, donde con weighted puro datasets muy pequenos
            (SSPSNASA15, DUS20) pueden no aparecer en N steps.
            La distribucion intra-cliente se renormaliza tras aplicar el
            piso y los caps por cliente/dataset se siguen respetando.

    Returns:
        Lista de dicts (1 por dataset PRETRAIN_SOURCE) con todas las columnas
        especificadas en sec 6 del prompt de backbone.
    """
    # 1. Filtrar solo PRETRAIN_SOURCE
    ps_rows = [r for r in processed_summary if r.get("role") == "PRETRAIN_SOURCE"]
    if not ps_rows:
        raise ValueError("processed_summary no contiene PRETRAIN_SOURCE")

    # 2. Pesos raw por dataset (base = n_channel_patches)
    # Recogemos tambien n_channels por dataset para que el plan exponga
    # esa columna y permita batch_size adaptativo aguas abajo.
    n_ch_by_ds: Dict[str, int] = {}
    client_by_ds: Dict[str, str] = {}
    n_channels_by_ds: Dict[str, int] = {}
    for r in ps_rows:
        ds = str(r["dataset"])
        n_ch_by_ds[ds] = int(r["n_channel_patches"])
        client_by_ds[ds] = str(r["client"])
        # `n_channels` viene del processed_summary.csv (verificado en CSV
        # generado por el full v0.5; columna obligatoria sec 13 CLAUDE.md).
        n_channels_by_ds[ds] = int(r["n_channels"]) if "n_channels" in r and r["n_channels"] not in ("", None) else 0
    datasets = sorted(n_ch_by_ds.keys())
    raw_w = np.array([n_ch_by_ds[d] for d in datasets], dtype=np.float64)
    total = raw_w.sum()
    if total <= 0:
        raise ValueError("Suma de n_channel_patches no positiva")
    raw_w_norm = raw_w / total

    # 3. Cap por dataset con redistribucion (PHM10 que supera 0.10 se trunca)
    capped_w = _cap_with_redistribution(raw_w_norm, cap=cap_max_dataset_weight)
    capped_dataset_flag = capped_w < raw_w_norm - 1e-9

    # 4. Peso raw por cliente (antes de cualquier cap)
    raw_client: Dict[str, float] = {}
    for ds, w in zip(datasets, raw_w_norm):
        c = client_by_ds[ds]
        raw_client[c] = raw_client.get(c, 0.0) + float(w)

    # Peso por cliente tras cap-dataset (suma de los datasets ya capeados)
    capped_client_dict: Dict[str, float] = {}
    for ds, w in zip(datasets, capped_w):
        c = client_by_ds[ds]
        capped_client_dict[c] = capped_client_dict.get(c, 0.0) + float(w)

    # 5. Cap efectivo por cliente: min(cap_client, cap_dataset * n_ds_in_client).
    #    Si un cliente tiene 1 solo dataset, no puede recibir mas de
    #    cap_dataset; de lo contrario violariamos el cap por dataset al
    #    distribuir intra-cliente.
    clients_sorted = sorted(capped_client_dict.keys())
    n_ds_in_client = {
        c: sum(1 for d in datasets if client_by_ds[d] == c)
        for c in clients_sorted
    }
    cap_client_effective = {
        c: min(cap_max_client_weight, cap_max_dataset_weight * n_ds_in_client[c])
        for c in clients_sorted
    }

    # Aplicar caps individuales por cliente con redistribucion. Como los
    # caps efectivos no son uniformes entre clientes, usamos un water-filling
    # iterativo: capear los que excedan su cap_efectivo, redistribuir el
    # exceso proporcional a los que aun pueden crecer.
    w_cli = np.array(
        [capped_client_dict[c] for c in clients_sorted], dtype=np.float64
    )
    caps_eff = np.array(
        [cap_client_effective[c] for c in clients_sorted], dtype=np.float64
    )
    for _ in range(200):
        over = w_cli > caps_eff + 1e-12
        if not over.any():
            break
        excess = float((w_cli[over] - caps_eff[over]).sum())
        w_cli[over] = caps_eff[over]
        free = (~over) & (w_cli < caps_eff - 1e-12)
        if not free.any():
            break
        free_total = float(w_cli[free].sum())
        if free_total <= 0:
            # Reparto uniforme entre libres
            w_cli[free] += excess / float(free.sum())
        else:
            w_cli[free] += excess * (w_cli[free] / free_total)
    w_cli = w_cli / w_cli.sum()
    capped_client_flag_arr = w_cli < np.array(
        [capped_client_dict[c] for c in clients_sorted]
    ) - 1e-9
    capped_client_flag = dict(zip(clients_sorted, capped_client_flag_arr.tolist()))

    # 6. min_client_presence
    client_w_after_min, floored_clients = _apply_min_client_presence(
        dict(zip(clients_sorted, w_cli.tolist())),
        min_presence=min_client_presence,
    )

    # 7. Distribuir el peso de cliente entre sus datasets, en proporcion al
    #    capped_dataset_weight intra-cliente, **capeando intra-cliente** a
    #    cap_max_dataset_weight (water-filling local). Como el cap efectivo
    #    de cliente es <= cap_dataset * n_ds_in_client, esto siempre tiene
    #    solucion factible.
    final_ds: Dict[str, float] = {}
    for c in clients_sorted:
        ds_of_c = [d for d in datasets if client_by_ds[d] == c]
        if not ds_of_c:
            continue
        weights_intra = np.array(
            [capped_w[datasets.index(d)] for d in ds_of_c], dtype=np.float64
        )
        s_intra = weights_intra.sum()
        if s_intra <= 0:
            weights_intra = np.full(len(ds_of_c), 1.0 / len(ds_of_c))
        else:
            weights_intra = weights_intra / s_intra
        # Reparto inicial proporcional al peso intra-cliente
        ds_w = weights_intra * client_w_after_min[c]
        # Water-filling local: capear a cap_max_dataset_weight
        for _ in range(100):
            over = ds_w > cap_max_dataset_weight + 1e-12
            if not over.any():
                break
            excess = float((ds_w[over] - cap_max_dataset_weight).sum())
            ds_w[over] = cap_max_dataset_weight
            free = ~over
            if not free.any():
                break
            free_total = float(ds_w[free].sum())
            if free_total <= 0:
                ds_w[free] += excess / float(free.sum())
            else:
                ds_w[free] += excess * (ds_w[free] / free_total)
        for d, w in zip(ds_of_c, ds_w):
            final_ds[d] = float(w)

    # Normalizacion final por estabilidad numerica
    s = sum(final_ds.values())
    if s > 0:
        final_ds = {d: v / s for d, v in final_ds.items()}

    # 7.bis Piso por dataset (opcional, default 0.0). Sube cualquier dataset
    # por debajo de `min_dataset_presence` al piso y resta proporcionalmente
    # del resto. Itera hasta convergencia. Solo factible si
    # min_dataset_presence * n_datasets <= 1.
    floored_datasets: List[str] = []
    if min_dataset_presence > 0:
        if min_dataset_presence * len(final_ds) > 1.0 + 1e-9:
            raise ValueError(
                f"min_dataset_presence={min_dataset_presence} * "
                f"n_datasets={len(final_ds)} > 1; imposible matematicamente"
            )
        for _ in range(100):
            below = {k: v for k, v in final_ds.items() if v < min_dataset_presence}
            if not below:
                break
            delta = sum(min_dataset_presence - v for v in below.values())
            above = {
                k: v for k, v in final_ds.items()
                if v >= min_dataset_presence and k not in floored_datasets
            }
            if not above:
                break
            total_above = sum(above.values())
            if total_above <= delta:
                break
            for k in below:
                final_ds[k] = min_dataset_presence
                if k not in floored_datasets:
                    floored_datasets.append(k)
            for k, v in above.items():
                final_ds[k] = v - delta * (v / total_above)
        s2 = sum(final_ds.values())
        if s2 > 0:
            final_ds = {d: v / s2 for d, v in final_ds.items()}

    # 7.ter Recalcular el peso final por cliente como AGRUPACION de los
    # pesos finales por dataset. Esto es lo unico semanticamente correcto:
    # `client_w_after_min` (paso 6) se calculo ANTES de aplicar el piso por
    # dataset (paso 7.bis), y por tanto puede estar desfasado cuando hay
    # datasets levantados al piso. La suma agrupada por cliente cumple
    # automaticamente que sum_clientes == sum_datasets == 1.
    final_client_from_ds: Dict[str, float] = {}
    for d, w in final_ds.items():
        c = client_by_ds[d]
        final_client_from_ds[c] = final_client_from_ds.get(c, 0.0) + float(w)

    # 8. Construir tabla
    plan: List[Dict[str, Any]] = []
    for d in datasets:
        c = client_by_ds[d]
        row = {
            "dataset":                    d,
            "client":                     c,
            "n_channels":                 n_channels_by_ds[d],
            "n_channel_patches":          n_ch_by_ds[d],
            "raw_dataset_weight":         float(raw_w_norm[datasets.index(d)]),
            "capped_dataset_weight":      float(capped_w[datasets.index(d)]),
            "final_dataset_weight":       float(final_ds[d]),
            "raw_client_weight":          float(raw_client[c]),
            "final_client_weight":        float(final_client_from_ds[c]),
            "capped_dataset":             bool(capped_dataset_flag[datasets.index(d)]),
            "capped_client":              bool(capped_client_flag.get(c, False)),
            "min_presence_applied":       bool(c in floored_clients),
            "min_dataset_presence_applied": bool(d in floored_datasets),
        }
        if steps_per_epoch is not None:
            row["expected_batches_per_epoch"] = float(
                row["final_dataset_weight"] * steps_per_epoch
            )
        else:
            row["expected_batches_per_epoch"] = None
        plan.append(row)

    # 9. Validaciones de consistencia (asserts internos)
    sum_ds = sum(r["final_dataset_weight"] for r in plan)
    assert abs(sum_ds - 1.0) < 1e-6, f"Suma de pesos final por dataset = {sum_ds}"
    sum_cl = sum(final_client_from_ds.values())
    assert abs(sum_cl - 1.0) < 1e-6, (
        f"Suma de pesos final por cliente (groupby) = {sum_cl}"
    )
    # Consistencia: para cada cliente, final_client_weight == suma de
    # final_dataset_weight de sus datasets.
    sum_per_client_check: Dict[str, float] = {}
    for r in plan:
        sum_per_client_check[r["client"]] = (
            sum_per_client_check.get(r["client"], 0.0) + r["final_dataset_weight"]
        )
    for c, expected in final_client_from_ds.items():
        observed = sum_per_client_check.get(c, 0.0)
        assert abs(observed - expected) < 1e-9, (
            f"Cliente {c}: groupby={observed} != final_client_from_ds={expected}"
        )

    return plan


def write_sampling_plan_csv(
    plan: Sequence[Dict[str, Any]], out_path: Path
) -> None:
    """Escribe el plan a CSV, sobreescribiendo si existe."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "dataset",
        "client",
        "n_channels",
        "n_channel_patches",
        "raw_dataset_weight",
        "capped_dataset_weight",
        "final_dataset_weight",
        "raw_client_weight",
        "final_client_weight",
        "expected_batches_per_epoch",
        "capped_dataset",
        "capped_client",
        "min_presence_applied",
        "min_dataset_presence_applied",
    ]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in plan:
            writer.writerow({k: row.get(k) for k in fields})


def load_sources(
    processed_summary_csv: Path,
    client_summary_csv: Path,
    audit_summary_json: Path,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Lee las tres fuentes que necesita `compute_pretraining_sampling_plan`."""
    with open(processed_summary_csv, "r", newline="", encoding="utf-8") as f:
        proc = list(csv.DictReader(f))
    with open(client_summary_csv, "r", newline="", encoding="utf-8") as f:
        cli = list(csv.DictReader(f))
    asum = json.loads(Path(audit_summary_json).read_text(encoding="utf-8"))
    return proc, cli, asum
