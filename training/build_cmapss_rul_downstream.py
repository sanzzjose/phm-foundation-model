"""Builder de CMAPSS RUL downstream desde raw (commit 1: parsers + preview).

Resuelve la decisión `CMAPSS_RUL_DECISION` (ver
`results/downstream/cmapss_rul_decision/decision.md`):

- No usa `processed/CMAPSS/` (target ambiguo, mezcla train/val/test).
- Reconstruye RUL físico desde raw PHMD:
    - train/val: `rul_physical = max_cycle_by(FD, unit) - cycle`.
    - test:      `rul_physical = last_observed_cycle_by(FD, unit) - cycle
                                  + official_RUL_FD[unit]`.
- Variante: `rul_capped_125 = min(rul_physical, R_max)`. Solo se
  aplica si `--rul-cap` > 0 Y después de tener RUL físico ≥ 0.

Decisiones de ventaneo/split (sec del prompt X):

- `window_size = 512` (compatibilidad con SSL central).
- `window_mode = rolling_causal`: una ventana por ciclo `t` que mira
  los últimos W ciclos hasta `t` con padding causal por la izquierda
  si `t < W`. **NO una sola ventana por trayectoria**. Esto aumenta
  drásticamente el número de muestras y permite RUL por ciclo.
- `target_policy = rul_at_prediction_cycle`: el target de la ventana
  cuyo último ciclo es `t` es `rul_physical[t]` (no agregado).
- `split_policy = unit_holdout_by_fd`, `val_frac = 0.2`, `seed = 42`:
  20% de unidades de train por FD se reservan como val (sin solapar
  con test, que ya está separado por CMAPSS NASA).
- `normalization_policy = instance_norm_per_window_channel_ignore_padding`
  (sec 10 CLAUDE.md, consistente con SSL central).
- `include_op_settings = True`: 21 sensores + 3 op_settings = **24
  canales** por ventana.

CLI por defecto en `--dry-run`: parsea raw, reconstruye RUL físico
por unidad, estima ventanas y tamaño en disco. NO escribe shards.

Funciones puras testeables:

- `parse_cmapss_txt_filelike(fp, fd_subset, split)`
- `parse_official_rul_filelike(fp)`
- `load_cmapss_raw_from_split_zips(raw_root)`
- `compute_train_rul(cycles, max_cycle)`
- `compute_test_rul(cycles, last_observed_cycle, official_rul)`
- `cap_rul(rul, cap)`
- `build_train_val_test_rul(parsed, val_frac, seed, rul_cap)`
- `preview_summary(parsed_with_rul, window_size)`

No descarga ni reentrena nada.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np


# Constantes del formato CMAPSS NASA estandar
CMAPSS_N_COLUMNS = 26          # unit_id, cycle, op1..op3, s1..s21
CMAPSS_N_OP_SETTINGS = 3
CMAPSS_N_SENSORS = 21
CMAPSS_N_CHANNELS_WITH_OP = CMAPSS_N_OP_SETTINGS + CMAPSS_N_SENSORS  # 24
CMAPSS_FD_SUBSETS = ("FD001", "FD002", "FD003", "FD004")

# Constantes del writer (commit 3)
MANIFEST_VERSION = "v0.1"
SHARD_PREFIX = "shard"
DEFAULT_SHARD_SIZE = 1024
DONE_FLAG_NAME = "done.flag"
MANIFEST_NAME = "manifest.json"
# Stride canonico de la decision (sec "Politica de seleccion de t_idx"
# de results/downstream/cmapss_rul_decision/decision.md). El writer
# se niega a escribir shards con stride distinto a este salvo
# --allow-noncanonical, para evitar que una ablation se cuele como
# version canonica.
CANONICAL_STRIDE = 5


# ----------------------------------------------------------------------
# Funciones puras (testeables, sin deps pesadas)
# ----------------------------------------------------------------------


def compute_train_rul(cycles: Sequence[float], max_cycle: float) -> List[float]:
    """RUL físico para train/val (run-to-failure).

    Args:
        cycles: ciclos observados por una trayectoria, ordenados ascendente.
        max_cycle: ciclo de fallo de esa trayectoria (= max de cycles).

    Returns:
        Lista del mismo length: `max_cycle - cycle` por cada elemento.
        Garantizado `>= 0` cuando `cycles` es monotonicamente creciente
        y `max_cycle == max(cycles)`.

    Raises:
        ValueError si max_cycle < max(cycles).
    """
    if cycles is None:
        return []
    cs = [float(c) for c in cycles]
    if not cs:
        return []
    mx = float(max_cycle)
    if mx + 1e-9 < max(cs):
        raise ValueError(
            f"max_cycle={mx} < max(cycles)={max(cs)}; pasar el ciclo "
            "real de fallo (run-to-failure)."
        )
    return [mx - c for c in cs]


def compute_test_rul(
    cycles: Sequence[float],
    last_observed_cycle: float,
    official_rul: float,
) -> List[float]:
    """RUL físico para test (interrumpido antes del fallo).

    Args:
        cycles: ciclos observados por la trayectoria de test.
        last_observed_cycle: último ciclo observado (= max de cycles).
        official_rul: RUL oficial publicado por CMAPSS para esa unidad
            en `RUL_FDxxx.txt` (lo que falta para fallo desde el último
            ciclo observado).

    Returns:
        `(last_observed_cycle - cycle) + official_rul` para cada elemento.
        Garantizado `>= 0` cuando `cycles` es creciente y
        `last_observed_cycle == max(cycles)` y `official_rul >= 0`.

    Raises:
        ValueError si official_rul < 0 o last_observed_cycle < max(cycles).
    """
    if cycles is None:
        return []
    cs = [float(c) for c in cycles]
    if not cs:
        return []
    last = float(last_observed_cycle)
    off = float(official_rul)
    if last + 1e-9 < max(cs):
        raise ValueError(
            f"last_observed_cycle={last} < max(cycles)={max(cs)}; debe ser "
            "el ultimo ciclo observado de la trayectoria de test."
        )
    if off < 0:
        raise ValueError(
            f"official_rul={off} debe ser >= 0 (CMAPSS publica RUL no negativo)."
        )
    return [(last - c) + off for c in cs]


def cap_rul(rul: Sequence[float], cap: Optional[float]) -> List[float]:
    """Aplica `min(rul, cap)` si cap es un float > 0; en otro caso devuelve la lista tal cual.

    No transforma negativos: si la entrada tiene negativos, el caller
    NO ha reconstruido RUL físico todavía y NO debe usar este cap (por
    eso el contrato manda: primero reconstruir, después capar).
    """
    if rul is None:
        return []
    rs = [float(r) for r in rul]
    if cap is None or cap <= 0:
        return rs
    cap_f = float(cap)
    return [min(r, cap_f) for r in rs]


# ----------------------------------------------------------------------
# Parsers CMAPSS NASA (formato whitespace, 26 columnas)
# ----------------------------------------------------------------------


def parse_cmapss_txt_filelike(
    fp: Any, fd_subset: str, split: str
) -> Dict[str, np.ndarray]:
    """Parsea un fichero `train_FDxxx.txt` o `test_FDxxx.txt` del formato
    NASA estandar (whitespace, sin header, 26 columnas).

    Estructura de columnas (NASA):
        0 = unit_id (int >0)
        1 = cycle (int >0)
        2..4 = op_setting_1, op_setting_2, op_setting_3 (float)
        5..25 = sensor_01..sensor_21 (float)

    Args:
        fp: file-like en modo binario o texto. Acepta tanto bytes como
            str porque viene de `zipfile.open()` (binario) o de tests
            con `io.StringIO`.
        fd_subset: una de CMAPSS_FD_SUBSETS, para etiquetar las filas.
        split: "train" o "test".

    Returns:
        dict con keys:
          - unit_id: np.int64 (N,)
          - cycle: np.int64 (N,)
          - op_settings: np.float32 (N, 3)
          - sensors: np.float32 (N, 21)
          - fd_subset: str (constante, repetido en meta)
          - split: str ("train" o "test")
          - n_rows: int
          - n_units: int

    Raises:
        ValueError si el numero de columnas != 26, si unit_id o cycle no
        son enteros positivos, o si cycle no es monotono creciente por
        unidad tras ordenar.
    """
    if fd_subset not in CMAPSS_FD_SUBSETS:
        raise ValueError(
            f"fd_subset desconocido: {fd_subset!r}; esperado {CMAPSS_FD_SUBSETS}"
        )
    if split not in ("train", "test"):
        raise ValueError(f"split debe ser 'train' o 'test', recibido {split!r}")

    # Normalizamos el file-like a texto. Si viene en bytes (zipfile.open
    # devuelve binario), decodificamos como UTF-8 (CMAPSS NASA es ASCII puro).
    data = fp.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="strict")
    if not data.strip():
        raise ValueError(f"fichero {fd_subset}/{split} esta vacio")

    # np.loadtxt sobre un StringIO. Garantiza 26 columnas o lanza.
    arr = np.loadtxt(io.StringIO(data), dtype=np.float64)
    if arr.ndim == 1:
        # Caso degenerado: una sola fila.
        arr = arr.reshape(1, -1)
    if arr.shape[1] != CMAPSS_N_COLUMNS:
        raise ValueError(
            f"{fd_subset}/{split}: se esperaban {CMAPSS_N_COLUMNS} columnas "
            f"(formato NASA: unit_id, cycle, op1..3, s1..21), encontradas "
            f"{arr.shape[1]}"
        )

    unit_id = arr[:, 0].astype(np.int64)
    cycle = arr[:, 1].astype(np.int64)
    op_settings = arr[:, 2:5].astype(np.float32)
    sensors = arr[:, 5:].astype(np.float32)

    # Validaciones duras.
    if (unit_id <= 0).any():
        raise ValueError(
            f"{fd_subset}/{split}: unit_id contiene valores <=0 (min="
            f"{int(unit_id.min())})."
        )
    if (cycle <= 0).any():
        raise ValueError(
            f"{fd_subset}/{split}: cycle contiene valores <=0 (min="
            f"{int(cycle.min())})."
        )
    # Comprobacion de tipos: los floats originales deben coincidir con int.
    if not np.allclose(unit_id, arr[:, 0]):
        raise ValueError(f"{fd_subset}/{split}: unit_id no es entero puro.")
    if not np.allclose(cycle, arr[:, 1]):
        raise ValueError(f"{fd_subset}/{split}: cycle no es entero puro.")

    # Monotonicidad de cycle por unidad: tras ordenar (unit_id, cycle)
    # el cycle dentro de cada unidad debe ser 1, 2, 3, ... estrictamente
    # creciente. Lo validamos.
    order = np.lexsort((cycle, unit_id))
    unit_sorted = unit_id[order]
    cycle_sorted = cycle[order]
    # Para cada unidad, los ciclos deben ser monotonos crecientes.
    for u in np.unique(unit_sorted):
        mask = unit_sorted == u
        cyc_u = cycle_sorted[mask]
        if not np.all(np.diff(cyc_u) > 0):
            raise ValueError(
                f"{fd_subset}/{split}: cycle no monotono creciente para "
                f"unit_id={int(u)} tras ordenar. cycles={cyc_u[:10].tolist()}"
            )

    # Reordenamos la salida tambien por (unit_id, cycle) para deja la
    # tabla canonica.
    return {
        "unit_id":     unit_id[order],
        "cycle":       cycle[order],
        "op_settings": op_settings[order],
        "sensors":     sensors[order],
        "fd_subset":   fd_subset,
        "split":       split,
        "n_rows":      int(arr.shape[0]),
        "n_units":     int(np.unique(unit_id).size),
    }


def parse_official_rul_filelike(fp: Any) -> Dict[int, int]:
    """Parsea `RUL_FDxxx.txt` de CMAPSS NASA (una columna, una linea por
    unidad de test).

    Convencion NASA: la **linea i (1-indexed)** del fichero contiene el
    RUL oficial para `unit_id = i` de test del mismo FD. Por tanto la
    longitud del fichero debe coincidir con el numero de unidades de
    test de ese FD.

    Args:
        fp: file-like binario o texto.

    Returns:
        dict `{unit_id: int}` mapeando unit_id (1-indexed) a su RUL
        oficial publicado por CMAPSS.

    Raises:
        ValueError si el fichero esta vacio, contiene valores no
        enteros, o contiene valores negativos.
    """
    data = fp.read()
    if isinstance(data, bytes):
        data = data.decode("utf-8", errors="strict")
    if not data.strip():
        raise ValueError("RUL_FDxxx.txt esta vacio")
    arr = np.loadtxt(io.StringIO(data), dtype=np.float64)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    if (arr < 0).any():
        raise ValueError(
            f"RUL oficial contiene valores negativos (min={float(arr.min())}); "
            "CMAPSS publica RUL no negativo."
        )
    # Comprobacion de tipos: debe ser entero puro.
    arr_int = arr.astype(np.int64)
    if not np.allclose(arr_int, arr):
        raise ValueError("RUL_FDxxx.txt contiene valores no enteros.")
    return {int(i + 1): int(v) for i, v in enumerate(arr_int)}


# ----------------------------------------------------------------------
# Loader: lee los 2 zips PHMD sin extraer, devuelve datos por FD
# ----------------------------------------------------------------------


def load_cmapss_raw_from_split_zips(
    raw_root: Path,
    fd_subsets: Sequence[str] = CMAPSS_FD_SUBSETS,
) -> Dict[str, Dict[str, Any]]:
    """Carga raw CMAPSS desde `raw_root/CMAPSS_train.zip` +
    `raw_root/CMAPSS_test.zip` sin extraer.

    Layout esperado dentro de cada zip:
      `CMAPSS_train.zip` -> `CMAPSS/train/train_FDxxx.txt`
      `CMAPSS_test.zip`  -> `CMAPSS/test/test_FDxxx.txt`
                            `CMAPSS/test/RUL_FDxxx.txt`

    Args:
        raw_root: directorio que contiene los dos zips.
        fd_subsets: subsets a cargar (por defecto los 4).

    Returns:
        dict `{fd_subset: {"train": <parsed>, "test": <parsed>,
        "official_rul": {unit_id: rul}}}`.

    Raises:
        FileNotFoundError si falta alguno de los dos zips.
        ValueError si algun fichero esperado falta dentro del zip o si
            `len(RUL_FDxxx) != n_units_test`.
    """
    import zipfile
    raw_root = Path(raw_root)
    train_zip_p = raw_root / "CMAPSS_train.zip"
    test_zip_p = raw_root / "CMAPSS_test.zip"
    if not train_zip_p.is_file():
        raise FileNotFoundError(f"No existe {train_zip_p}")
    if not test_zip_p.is_file():
        raise FileNotFoundError(f"No existe {test_zip_p}")

    out: Dict[str, Dict[str, Any]] = {}
    with zipfile.ZipFile(train_zip_p) as zf_tr, zipfile.ZipFile(test_zip_p) as zf_te:
        tr_names = set(zf_tr.namelist())
        te_names = set(zf_te.namelist())
        for fd in fd_subsets:
            train_entry = f"CMAPSS/train/train_{fd}.txt"
            test_entry = f"CMAPSS/test/test_{fd}.txt"
            rul_entry = f"CMAPSS/test/RUL_{fd}.txt"
            if train_entry not in tr_names:
                raise ValueError(
                    f"Falta {train_entry} en {train_zip_p.name}. "
                    f"Entradas disponibles (primeras 10): "
                    f"{sorted(tr_names)[:10]}"
                )
            if test_entry not in te_names:
                raise ValueError(f"Falta {test_entry} en {test_zip_p.name}")
            if rul_entry not in te_names:
                raise ValueError(f"Falta {rul_entry} en {test_zip_p.name}")

            with zf_tr.open(train_entry) as f:
                train_data = parse_cmapss_txt_filelike(f, fd, "train")
            with zf_te.open(test_entry) as f:
                test_data = parse_cmapss_txt_filelike(f, fd, "test")
            with zf_te.open(rul_entry) as f:
                official_rul = parse_official_rul_filelike(f)

            # Validacion cruzada: len(RUL_FDxxx) debe = n_units_test.
            if len(official_rul) != test_data["n_units"]:
                raise ValueError(
                    f"{fd}: len(RUL_{fd})={len(official_rul)} != "
                    f"n_units_test={test_data['n_units']}. El fichero "
                    "RUL publica un valor por linea (1-indexed) y debe "
                    "coincidir con el numero de unidades de test."
                )
            # Validacion: los unit_id de test deben ser exactamente
            # 1..n_units_test (CMAPSS NASA siempre los emite asi).
            unit_ids_test = sorted(set(test_data["unit_id"].tolist()))
            expected = list(range(1, test_data["n_units"] + 1))
            if unit_ids_test != expected:
                raise ValueError(
                    f"{fd}: unit_id de test no son 1..{test_data['n_units']}. "
                    f"Primeros: {unit_ids_test[:10]}"
                )

            out[fd] = {
                "train": train_data,
                "test": test_data,
                "official_rul": official_rul,
            }
    return out


# ----------------------------------------------------------------------
# Reconstruccion de RUL fisico por unidad + split val + cap
# ----------------------------------------------------------------------


def _fd_seed_offset(fd: str) -> int:
    """Offset determinista para componer la semilla del split val por FD.

    BUG HISTORICO: hasta commit 3 el codigo usaba `hash(fd)` directamente,
    pero el builtin `hash(str)` en Python depende de `PYTHONHASHSEED`, que
    se randomiza por defecto en cada arranque del interprete. Esto
    produjo splits val no reproducibles entre procesos (confirmado en
    Colab con 3 corridas dando 3 splits distintos sobre los mismos datos).

    Este helper devuelve un offset estable:
      - Si `fd` esta en `CMAPSS_FD_SUBSETS`: su indice (0..3).
      - Si no: derivado de `sha256(fd.encode())` (estable entre procesos).

    Returns:
        int >= 0 que se suma al seed global para inicializar el RNG por FD.
    """
    if fd in CMAPSS_FD_SUBSETS:
        return CMAPSS_FD_SUBSETS.index(fd)
    # Fallback estable: primeros 2 bytes del sha256.
    return int.from_bytes(
        hashlib.sha256(fd.encode("utf-8")).digest()[:2], "big"
    )


def build_train_val_test_rul(
    parsed: Dict[str, Dict[str, Any]],
    val_frac: float = 0.2,
    seed: int = 42,
    rul_cap: Optional[float] = 125.0,
) -> Dict[str, Dict[str, Any]]:
    """Reconstruye RUL fisico por unidad y aplica split val/train por FD.

    Para cada FD:
      - Train: agrupa por unit_id, calcula `rul_physical[i] = max_cycle - cycle[i]`.
      - Val: subconjunto de unidades de train (val_frac, estratificado por
        FD, seed reproducible). Las unidades de val NO aparecen en train.
      - Test: agrupa por unit_id, calcula
        `rul_physical[i] = (last_cycle - cycle[i]) + official_rul[unit_id]`.

    Args:
        parsed: salida de `load_cmapss_raw_from_split_zips`.
        val_frac: fraccion de unidades de train que pasan a val
            (default 0.2).
        seed: semilla del muestreo de val (default 42).
        rul_cap: cap opcional sobre rul_physical (default 125, Heimes).
            Si <=0 o None, no se aplica cap.

    Returns:
        dict con la misma estructura `{fd: {"train": ..., "val": ...,
        "test": ..., "unit_split": {"train_units": [...], "val_units":
        [...], "test_units": [...]}}}`. Cada split contiene los arrays
        originales (unit_id, cycle, op_settings, sensors) MAS las
        columnas:
          - rul_physical: np.float32 (N,)
          - rul_capped_125: np.float32 (N,) (= rul_physical si rul_cap<=0)

    Raises:
        AssertionError si min(rul_physical) < 0 en cualquier split.
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"val_frac debe estar en (0,1), recibido {val_frac}")

    rng = np.random.default_rng(seed)
    out: Dict[str, Dict[str, Any]] = {}

    for fd, fd_data in parsed.items():
        train = fd_data["train"]
        test = fd_data["test"]
        official_rul = fd_data["official_rul"]

        # 1) RUL fisico para train
        train_rul = _compute_rul_for_split(
            train, official_rul=None, is_test=False,
        )

        # 2) RUL fisico para test
        test_rul = _compute_rul_for_split(
            test, official_rul=official_rul, is_test=True,
        )

        # 3) Asserts de no negatividad
        assert train_rul.min() >= 0, (
            f"{fd}/train: rul_physical min = {train_rul.min()} < 0"
        )
        assert test_rul.min() >= 0, (
            f"{fd}/test: rul_physical min = {test_rul.min()} < 0"
        )

        # 4) Cap opcional
        if rul_cap is not None and rul_cap > 0:
            train_capped = np.minimum(train_rul, float(rul_cap)).astype(np.float32)
            test_capped = np.minimum(test_rul, float(rul_cap)).astype(np.float32)
        else:
            train_capped = train_rul.copy()
            test_capped = test_rul.copy()

        # 5) Split val por unidades (estratificado por FD = solo este FD)
        unique_units_train = np.unique(train["unit_id"])
        n_val = max(1, int(round(len(unique_units_train) * val_frac)))
        # Seed derivada del seed global + offset determinista por FD.
        # Antes usabamos `hash(fd)`, pero `hash(str)` depende de
        # PYTHONHASHSEED y produce splits no reproducibles entre procesos.
        # `_fd_seed_offset` garantiza el mismo offset para el mismo FD en
        # cualquier python y cualquier maquina.
        rng_fd = np.random.default_rng(seed + _fd_seed_offset(fd))
        val_units = sorted(rng_fd.choice(unique_units_train, size=n_val, replace=False).tolist())
        val_units_set = set(val_units)
        train_units = sorted([int(u) for u in unique_units_train if int(u) not in val_units_set])

        # 6) Construir train_split y val_split a partir de train.
        train_mask = ~np.isin(train["unit_id"], list(val_units_set))
        val_mask = np.isin(train["unit_id"], list(val_units_set))

        def _subset(data, rul, capped, mask):
            return {
                "unit_id":       data["unit_id"][mask],
                "cycle":         data["cycle"][mask],
                "op_settings":   data["op_settings"][mask],
                "sensors":       data["sensors"][mask],
                "rul_physical":  rul[mask].astype(np.float32),
                "rul_capped_125": capped[mask].astype(np.float32),
                "fd_subset":     fd,
                "n_rows":        int(mask.sum()),
                "n_units":       int(np.unique(data["unit_id"][mask]).size),
            }

        train_split = _subset(train, train_rul, train_capped, train_mask)
        train_split["split"] = "train"
        val_split = _subset(train, train_rul, train_capped, val_mask)
        val_split["split"] = "val"
        test_split = {
            "unit_id":       test["unit_id"],
            "cycle":         test["cycle"],
            "op_settings":   test["op_settings"],
            "sensors":       test["sensors"],
            "rul_physical":  test_rul.astype(np.float32),
            "rul_capped_125": test_capped.astype(np.float32),
            "fd_subset":     fd,
            "n_rows":        int(test["n_rows"]),
            "n_units":       int(test["n_units"]),
            "split":         "test",
        }

        out[fd] = {
            "train": train_split,
            "val":   val_split,
            "test":  test_split,
            "unit_split": {
                "train_units": train_units,
                "val_units":   val_units,
                "test_units":  sorted([int(u) for u in np.unique(test["unit_id"]).tolist()]),
            },
        }

    return out


def _compute_rul_for_split(
    data: Dict[str, np.ndarray],
    official_rul: Optional[Dict[int, int]],
    is_test: bool,
) -> np.ndarray:
    """Helper interno: aplica compute_train_rul o compute_test_rul por
    unidad y devuelve un array (N,) con el RUL fisico de cada fila.

    El orden de salida es el mismo que el de `data` (que ya viene
    ordenado por (unit_id, cycle) tras `parse_cmapss_txt_filelike`).
    """
    n = data["n_rows"]
    out_rul = np.zeros(n, dtype=np.float64)
    for u in np.unique(data["unit_id"]):
        mask = data["unit_id"] == u
        cycles_u = data["cycle"][mask].tolist()
        last_cycle = max(cycles_u)
        if is_test:
            rul_u = compute_test_rul(
                cycles_u,
                last_observed_cycle=last_cycle,
                official_rul=official_rul[int(u)],
            )
        else:
            rul_u = compute_train_rul(cycles_u, max_cycle=last_cycle)
        out_rul[mask] = np.array(rul_u, dtype=np.float64)
    return out_rul


# ----------------------------------------------------------------------
# Preview rolling_causal: estima ventanas y tamano en disco
# ----------------------------------------------------------------------


def preview_summary(
    parsed_with_rul: Dict[str, Dict[str, Any]],
    window_size: int = 512,
    patch_size: int = 16,
    n_channels: int = CMAPSS_N_CHANNELS_WITH_OP,
    bytes_per_float: int = 4,
    stride: int = 1,
    min_valid_timesteps: Optional[int] = None,
    include_last_per_unit: bool = True,
) -> Dict[str, Any]:
    """Estima ventanas rolling_causal POST-FILTRO y tamano en disco.

    Con `stride=1` y sin filtros, `n_windows == n_rows`. Cuando se
    activan `stride > 1`, `min_valid_timesteps` o `include_last_per_unit`,
    la relacion cambia: solo se emiten los t_idx que devuelve
    `selected_t_indices`. Este preview computa la seleccion exacta sin
    materializar las ventanas (rapido y barato).

    Returns:
        dict con:
          - by_fd: {fd: {split: {...}}}
          - totals: idem agregado
          - decisions: eco de la politica aplicada (W, P, stride,
            min_valid, include_last).
    """
    by_fd: Dict[str, Any] = {}
    total_train_w = total_val_w = total_test_w = 0
    total_full_w = 0
    total_padded_w = 0
    total_valid_timesteps = 0
    total_timesteps = 0
    total_dropped_min_valid = 0
    total_added_last = 0

    for fd, fd_data in parsed_with_rul.items():
        fd_summary: Dict[str, Any] = {}
        for split_name in ("train", "val", "test"):
            sd = fd_data[split_name]
            cycles_per_unit = []
            for u in np.unique(sd["unit_id"]):
                cycles_per_unit.append(int((sd["unit_id"] == u).sum()))
            cycles_per_unit_arr = np.array(cycles_per_unit, dtype=np.int64) if cycles_per_unit else np.array([0])

            # POST-FILTRO: para cada unidad calculamos los t_idx
            # seleccionados con la politica activa.
            n_windows_selected = 0
            n_full_split = 0
            n_padded_split = 0
            valid_ts_split = 0
            tot_ts_split = 0
            n_dropped_split = 0
            n_added_last_split = 0
            n_units_with_window = 0
            n_units_only_last = 0
            # RUL stats sobre las ventanas seleccionadas (no sobre n_rows).
            sel_ruls: List[float] = []
            sel_caps: List[float] = []

            for u in np.unique(sd["unit_id"]):
                mask = sd["unit_id"] == u
                Tu = int(mask.sum())
                if Tu == 0:
                    continue
                indices_u, dropped_u, added_u = selected_t_indices(
                    T=Tu,
                    stride=stride,
                    min_valid_timesteps=min_valid_timesteps,
                    include_last_per_unit=include_last_per_unit,
                )
                if not indices_u:
                    continue
                n_units_with_window += 1
                # Unidad que solo entra por last_override (T < min_valid).
                mv = int(min_valid_timesteps) if min_valid_timesteps else 0
                if mv > 0 and Tu < mv and indices_u == [Tu - 1] and include_last_per_unit:
                    n_units_only_last += 1
                n_windows_selected += len(indices_u)
                n_dropped_split += dropped_u
                n_added_last_split += added_u

                # Para cada t_idx seleccionado, contamos full/padded y
                # timesteps validos.
                unit_rul = sd["rul_physical"][mask]
                unit_cap = sd["rul_capped_125"][mask]
                for t_idx in indices_u:
                    n_valid = min(t_idx + 1, window_size)
                    if n_valid == window_size:
                        n_full_split += 1
                    else:
                        n_padded_split += 1
                    valid_ts_split += n_valid
                    tot_ts_split += window_size
                    sel_ruls.append(float(unit_rul[t_idx]))
                    sel_caps.append(float(unit_cap[t_idx]))

            frac_padded = (n_padded_split / max(1, n_padded_split + n_full_split))
            frac_valid_ts = (valid_ts_split / max(1, tot_ts_split))

            sel_ruls_arr = np.array(sel_ruls, dtype=np.float32) if sel_ruls else np.array([0.0])
            sel_caps_arr = np.array(sel_caps, dtype=np.float32) if sel_caps else np.array([0.0])

            fd_summary[split_name] = {
                "n_units_total":       sd["n_units"],
                "n_units_with_at_least_one_window": n_units_with_window,
                "n_units_only_last_override":       n_units_only_last,
                "n_rows_original":     sd["n_rows"],
                "n_windows_selected":  n_windows_selected,
                "n_windows_dropped_by_min_valid":   n_dropped_split,
                "n_windows_added_by_last_override": n_added_last_split,
                "cycles_per_unit_min":    int(cycles_per_unit_arr.min()),
                "cycles_per_unit_median": float(np.median(cycles_per_unit_arr)),
                "cycles_per_unit_max":    int(cycles_per_unit_arr.max()),
                # RUL stats post-seleccion (no sobre n_rows).
                "rul_physical_min":    float(sel_ruls_arr.min()),
                "rul_physical_median": float(np.median(sel_ruls_arr)),
                "rul_physical_max":    float(sel_ruls_arr.max()),
                "rul_capped_125_min":    float(sel_caps_arr.min()),
                "rul_capped_125_median": float(np.median(sel_caps_arr)),
                "rul_capped_125_max":    float(sel_caps_arr.max()),
                "n_channels": n_channels,
                "n_windows_full":      n_full_split,
                "n_windows_padded":    n_padded_split,
                "frac_windows_padded": round(frac_padded, 4),
                "frac_timesteps_valid_avg": round(frac_valid_ts, 4),
            }
            total_full_w += n_full_split
            total_padded_w += n_padded_split
            total_valid_timesteps += valid_ts_split
            total_timesteps += tot_ts_split
            total_dropped_min_valid += n_dropped_split
            total_added_last += n_added_last_split

        official_rul_count = len(fd_data.get("unit_split", {}).get("test_units", []))
        fd_summary["official_rul_count"] = official_rul_count
        fd_summary["unit_split"] = fd_data["unit_split"]
        by_fd[fd] = fd_summary
        total_train_w += fd_summary["train"]["n_windows_selected"]
        total_val_w   += fd_summary["val"]["n_windows_selected"]
        total_test_w  += fd_summary["test"]["n_windows_selected"]

    # Tamano por ventana (float32). Aproximacion:
    # patches: n_channels * window_size * bytes (los N*P timesteps).
    bytes_per_window = n_channels * window_size * bytes_per_float + 1024  # +1 KB meta/masks
    total_w = total_train_w + total_val_w + total_test_w
    estimated_size_gb = (total_w * bytes_per_window) / (1024 ** 3)

    return {
        "by_fd": by_fd,
        "totals": {
            "n_windows_train": total_train_w,
            "n_windows_val":   total_val_w,
            "n_windows_test":  total_test_w,
            "n_windows_total": total_w,
            "bytes_per_window_estimate": int(bytes_per_window),
            "estimated_size_gb_float32": round(estimated_size_gb, 2),
            "n_windows_full":      total_full_w,
            "n_windows_padded":    total_padded_w,
            "frac_windows_padded": round(
                total_padded_w / max(1, total_full_w + total_padded_w), 4
            ),
            "frac_timesteps_valid_avg": round(
                total_valid_timesteps / max(1, total_timesteps), 4
            ),
            # Z3: stats post-filtro globales.
            "n_windows_dropped_by_min_valid":   total_dropped_min_valid,
            "n_windows_added_by_last_override": total_added_last,
        },
        "decisions": {
            "window_size":      window_size,
            "patch_size":       patch_size,
            "n_patches":        window_size // patch_size,
            "n_channels":       n_channels,
            "window_mode":      "rolling_causal",
            "target_policy":    "rul_at_prediction_cycle",
            "split_policy":     "unit_holdout_by_fd",
            "normalization_policy": "instance_norm_per_window_channel_ignore_padding",
            # Z: politica de seleccion post-filtro.
            "stride":                   int(stride),
            "min_valid_timesteps":      int(min_valid_timesteps) if min_valid_timesteps else None,
            "include_last_per_unit":    bool(include_last_per_unit),
        },
    }


# ----------------------------------------------------------------------
# Ventaneo rolling_causal + instance normalization + patching (numpy)
# ----------------------------------------------------------------------
#
# Pipeline single-sample (sin torch, sin batch):
#   build_rolling_causal_window(unit_arr, t_idx, W)
#       -> (window (W, C), valid_time_mask (W,) bool)
#   instance_normalize_window(window, valid_time_mask, eps=1e-6)
#       -> (normalized (W, C), mean (C,), std_used (C,),
#           canales_constantes_mask (C,))
#   patch_window(normalized, valid_time_mask, P)
#       -> (patches (C, N, P), valid_patch_mask (C, N))
#   iter_unit_windows(unit_id, cycles, channels, rul_phys, rul_capped,
#                     fd, W, P, stride=1)
#       -> generator de samples {patches, valid_time_mask,
#           valid_patch_mask, canales_constantes_mask, mean, std_used,
#           target, meta}
#   iter_split_windows(split_dict, W, P, stride=1)
#       -> generator agregado por unidad.
#
# Convencion shapes (alineada con sec 9 CLAUDE.md):
#   serie por unidad: (T, C)
#   ventana          : (W, C)
#   patches          : (C, N, P)  con N = W // P
#   valid_time_mask  : (W,) bool
#   valid_patch_mask : (C, N) bool
#   canales_constantes_mask: (C,) bool
#
# Padding causal por la izquierda: si en el ciclo t la unidad no tiene
# todavia W ciclos observados, las posiciones [0, W-1-t_idx) se rellenan
# con ceros y `valid_time_mask` queda False ahi. El encoder PatchTST ve
# esos ceros pero la loss / pooling downstream los ignoran via las
# masks (sec 14 CLAUDE.md).


def build_rolling_causal_window(
    unit_channels: np.ndarray,
    t_idx: int,
    window_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Construye una ventana causal `[t_idx-W+1, t_idx]` con padding
    izquierdo si la unidad aun no tiene W ciclos.

    Args:
        unit_channels: array (T, C) con los canales de UNA unidad
            ordenados por cycle ascendente.
        t_idx: indice del ciclo objetivo (0-indexed dentro de la unidad).
            La ventana es `unit_channels[t_idx - W + 1 : t_idx + 1]`.
        window_size: W.

    Returns:
        (window, valid_time_mask):
          - window: (W, C) float32. Ceros en padding.
          - valid_time_mask: (W,) bool. True en posiciones reales.

    Raises:
        ValueError si t_idx < 0 o t_idx >= T o W <= 0.
    """
    if unit_channels.ndim != 2:
        raise ValueError(
            f"unit_channels debe ser (T, C), recibido shape {unit_channels.shape}"
        )
    T, C = unit_channels.shape
    if t_idx < 0 or t_idx >= T:
        raise ValueError(
            f"t_idx debe estar en [0, T={T}), recibido {t_idx}"
        )
    if window_size <= 0:
        raise ValueError(f"window_size debe ser > 0, recibido {window_size}")

    W = int(window_size)
    window = np.zeros((W, C), dtype=np.float32)
    valid_time_mask = np.zeros(W, dtype=bool)

    # n_real: cuantos ciclos reales caben en la ventana terminando en t_idx.
    # Como minimo 1 (el propio t_idx), como maximo W.
    n_real = min(W, t_idx + 1)
    # Indice de inicio en la trayectoria de la unidad.
    src_start = t_idx - n_real + 1
    src_end = t_idx + 1
    # Las posiciones reales van al FINAL de la ventana (causal: el ciclo
    # mas reciente esta en la ultima posicion W-1).
    dst_start = W - n_real
    window[dst_start:W] = unit_channels[src_start:src_end].astype(np.float32)
    valid_time_mask[dst_start:W] = True

    return window, valid_time_mask


def instance_normalize_window(
    window: np.ndarray,
    valid_time_mask: np.ndarray,
    eps: float = 1e-6,
    std_threshold: float = 1e-4,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Instance normalization por canal, ignorando padding.

    Para cada canal `c`:
      - Si hay >= 2 timesteps validos y std > std_threshold:
        normalized[:, c] = (window[:, c] - mean_c) / std_c en posiciones
        validas; 0 en padding.
      - Si std <= std_threshold (canal cuasi-constante) o solo hay 1
        timestep valido: canales_constantes_mask[c] = True y
        normalized[:, c] = 0 en todas las posiciones. mean = valor de
        la unica observacion (o 0 si ninguna).

    Args:
        window: (W, C) float32.
        valid_time_mask: (W,) bool.
        eps: clamp del denominador en la division (evita 1/0).
        std_threshold: umbral de varianza por debajo del cual el canal
            se considera constante.

    Returns:
        (normalized, mean, std_used, canales_constantes_mask)
          - normalized: (W, C) float32.
          - mean: (C,) float32.
          - std_used: (C,) float32 (el divisor real usado, eps si
            constante).
          - canales_constantes_mask: (C,) bool.
    """
    if window.ndim != 2:
        raise ValueError(f"window debe ser (W, C); recibido {window.shape}")
    if valid_time_mask.shape != (window.shape[0],):
        raise ValueError(
            f"valid_time_mask shape {valid_time_mask.shape} != "
            f"(W={window.shape[0]},)"
        )
    if valid_time_mask.dtype != bool:
        raise ValueError(
            f"valid_time_mask debe ser bool, recibido {valid_time_mask.dtype}"
        )

    W, C = window.shape
    normalized = np.zeros_like(window, dtype=np.float32)
    mean = np.zeros(C, dtype=np.float32)
    std_used = np.full(C, float(eps), dtype=np.float32)
    constant_mask = np.zeros(C, dtype=bool)

    n_valid = int(valid_time_mask.sum())
    if n_valid == 0:
        # Ventana entera de padding: todos los canales constantes en 0.
        constant_mask[:] = True
        return normalized, mean, std_used, constant_mask

    valid_window = window[valid_time_mask]  # (n_valid, C)
    means_c = valid_window.mean(axis=0)
    # ddof=0 (poblacional) para que con n_valid=1 no dividamos por 0;
    # de todas formas marcamos constante en ese caso.
    stds_c = valid_window.std(axis=0, ddof=0)

    for c in range(C):
        mean[c] = float(means_c[c])
        if n_valid < 2 or stds_c[c] <= std_threshold:
            constant_mask[c] = True
            std_used[c] = float(eps)
            # normalized[:, c] queda en 0
        else:
            s = float(stds_c[c])
            std_used[c] = s
            # Restamos mean solo en posiciones validas y dividimos por std;
            # padding queda en 0.
            normalized[valid_time_mask, c] = (
                (valid_window[:, c] - means_c[c]) / s
            ).astype(np.float32)

    return normalized, mean, std_used, constant_mask


def patch_window(
    normalized: np.ndarray,
    valid_time_mask: np.ndarray,
    patch_size: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reordena `(W, C)` normalizado a patches `(C, N, P)` y construye
    `valid_patch_mask (C, N)`.

    El reordering es el del contrato channel-independent (sec 9 CLAUDE.md):
    cada canal se procesa por el encoder de forma independiente, asi que
    la dimension C va primero, despues N patches de longitud P.

    Args:
        normalized: (W, C) float32. Salida de `instance_normalize_window`.
        valid_time_mask: (W,) bool.
        patch_size: P (debe dividir a W exactamente).

    Returns:
        (patches, valid_patch_mask):
          - patches: (C, N, P) float32 con N = W // P.
          - valid_patch_mask: (C, N) bool. True si al menos un timestep
            del patch es valido (mismo por canal porque
            valid_time_mask es solo temporal).

    Raises:
        ValueError si W % P != 0.
    """
    if normalized.ndim != 2:
        raise ValueError(f"normalized debe ser (W, C); recibido {normalized.shape}")
    W, C = normalized.shape
    if W % patch_size != 0:
        raise ValueError(
            f"W={W} debe ser divisible por P={patch_size}"
        )
    N = W // patch_size
    P = patch_size
    # (W, C) -> (C, W) -> (C, N, P)
    patches = normalized.T.reshape(C, N, P).astype(np.float32)

    # valid_patch_mask: para cada patch, True si al menos un timestep es
    # valido. Mismo valor para todos los canales (la mascara temporal no
    # depende del canal en CMAPSS).
    vtm_n_p = valid_time_mask.reshape(N, P)
    patch_any_valid = vtm_n_p.any(axis=1)  # (N,)
    valid_patch_mask = np.broadcast_to(patch_any_valid, (C, N)).copy()

    return patches, valid_patch_mask


def selected_t_indices(
    T: int,
    stride: int = 1,
    min_valid_timesteps: Optional[int] = None,
    include_last_per_unit: bool = True,
) -> Tuple[List[int], int, int]:
    """Devuelve los `t_idx` a emitir por una unidad de longitud T bajo
    la politica rolling_causal con stride + min_valid + last_override.

    Reglas:

    - `min_valid_timesteps` (si no None y >0):
      el primer t_idx regular es `min_valid_timesteps - 1`. Justifica:
      una ventana causal terminando en `t_idx` tiene `t_idx + 1`
      timesteps reales (siempre que `t_idx + 1 <= W`). Para garantizar
      al menos `min_valid` timesteps reales, exigimos `t_idx >=
      min_valid - 1`.
    - `stride >= 1`: tras el primer t_idx, avanza de `stride` en `stride`.
    - `include_last_per_unit=True`: si `T-1` no cae en la rejilla, se
      anade al final. Esto asegura que **el ultimo ciclo observado**
      (que en train es el ciclo de fallo con `rul_physical=0`) siempre
      se emita.
    - Si `T < min_valid_timesteps`:
        - con `include_last=True`: se emite solo `T-1` (anotado como
          `below_min_valid_because_last=True` para trazabilidad).
        - con `include_last=False`: no se emite nada.

    Args:
        T: longitud de la trayectoria (n_ciclos de la unidad).
        stride: paso entre indices regulares.
        min_valid_timesteps: piso de timesteps reales en la ventana, o None.
        include_last_per_unit: si True, garantizar emision de t=T-1.

    Returns:
        Tupla `(indices, n_dropped_by_min_valid, n_added_by_last_override)`:
          - `indices`: lista de t_idx en orden ascendente, sin duplicados.
          - `n_dropped_by_min_valid`: cuantos t_idx descartamos por el
            piso `min_valid_timesteps` (comparado con stride sin filtro).
          - `n_added_by_last_override`: 1 si `include_last` anadio el
            T-1 que no estaba ya en la rejilla; 0 si ya estaba o si
            include_last=False; **incluso si T < min_valid y solo emitimos
            T-1, se cuenta como 1** (semantica: el last_override fue
            quien permitio emitir ese sample).

    Raises:
        ValueError si T < 0, stride < 1, min_valid_timesteps <= 0.
    """
    if T < 0:
        raise ValueError(f"T debe ser >=0, recibido {T}")
    if stride < 1:
        raise ValueError(f"stride debe ser >=1, recibido {stride}")
    if min_valid_timesteps is not None and min_valid_timesteps <= 0:
        raise ValueError(
            f"min_valid_timesteps debe ser None o >0, recibido {min_valid_timesteps}"
        )
    if T == 0:
        return [], 0, 0

    mv = int(min_valid_timesteps) if min_valid_timesteps else 0
    # Caso especial: T < mv.
    if mv > 0 and T < mv:
        if include_last_per_unit:
            # Solo emitimos T-1 marcado como below_min.
            return [T - 1], 0, 1
        else:
            # Sin override y por debajo del piso: nada.
            return [], T, 0

    # Indice de inicio regular: max(0, mv-1).
    start = max(0, mv - 1)
    # n_dropped_by_min_valid: los t_idx < start que stride habria emitido sin filtro.
    # Sin filtro: range(0, T, stride). El primero >= start es
    # `((start - 0 + stride - 1) // stride) * stride` si start>0. Mas sencillo:
    # contamos cuantos t_idx en [0, start) caian en la rejilla stride.
    dropped = 0
    if start > 0:
        # Indices que stride emitiria pero descartamos: 0, stride, 2*stride, ... < start.
        # Numero: ceil(start / stride) = (start + stride - 1) // stride.
        dropped = (start + stride - 1) // stride

    indices = list(range(start, T, stride))
    added_by_last = 0
    if include_last_per_unit and (T - 1) not in indices:
        # T - 1 puede ser < start (no deberia, porque ya tratamos T<mv arriba)
        # o > start pero no en la rejilla.
        if T - 1 >= 0:
            indices.append(T - 1)
            indices.sort()
            added_by_last = 1
    return indices, dropped, added_by_last


def iter_unit_windows(
    unit_id: int,
    unit_cycles: np.ndarray,
    unit_channels: np.ndarray,
    unit_rul_physical: np.ndarray,
    unit_rul_capped: np.ndarray,
    fd_subset: str,
    split: str,
    window_size: int = 512,
    patch_size: int = 16,
    stride: int = 1,
    min_valid_timesteps: Optional[int] = None,
    include_last_per_unit: bool = True,
):
    """Itera ventanas rolling_causal sobre los ciclos de una unidad.

    Usa `selected_t_indices(...)` para decidir que t_idx emitir bajo la
    politica `(stride, min_valid_timesteps, include_last_per_unit)`.

    Args:
        unit_id: int.
        unit_cycles: (T,) int64. Cycles ordenados ascendentemente.
        unit_channels: (T, C) float32.
        unit_rul_physical: (T,) float32.
        unit_rul_capped: (T,) float32.
        fd_subset: 'FD001'..'FD004'.
        split: 'train' | 'val' | 'test'.
        window_size, patch_size, stride.
        min_valid_timesteps: piso de timesteps reales (sec del prompt Z).
        include_last_per_unit: incluir t=T-1 aunque no caiga en rejilla.

    Yields:
        dict con keys:
          patches: (C, N, P) float32
          valid_time_mask: (W,) bool
          valid_patch_mask: (C, N) bool
          canales_constantes_mask: (C,) bool
          mean: (C,) float32
          std_used: (C,) float32
          target: {'rul_physical', 'rul_capped_125', 'cycle'}
          meta: {fd_subset, split, unit_id, cycle, t_idx_in_unit,
                 valid_timesteps, below_min_valid_because_last,
                 selected_by_last_override, min_valid_timesteps, stride}
    """
    T = int(unit_channels.shape[0])
    indices, _dropped, added_by_last = selected_t_indices(
        T=T,
        stride=stride,
        min_valid_timesteps=min_valid_timesteps,
        include_last_per_unit=include_last_per_unit,
    )
    if not indices:
        return

    # Set para detectar cuales fueron anadidos por el last override (no
    # estaban en la rejilla regular). Es el unico t_idx > max_regular.
    # added_by_last=1 implica que T-1 NO estaba en rejilla (se anadio).
    # Pero solo es last_override si efectivamente T-1 no aparecia en
    # range(start, T, stride). Caso T<mv: tambien marca selected_by_last.
    mv = int(min_valid_timesteps) if min_valid_timesteps else 0
    below_min = (mv > 0 and T < mv)

    # Recalculamos los "regular" indices para distinguir last override.
    if below_min:
        regular_set = set()
    else:
        start = max(0, mv - 1)
        regular_set = set(range(start, T, stride))

    for t_idx in indices:
        window, vtm = build_rolling_causal_window(
            unit_channels, t_idx, window_size,
        )
        norm, mean, std_used, const_mask = instance_normalize_window(window, vtm)
        patches, vpm = patch_window(norm, vtm, patch_size)
        cycle = int(unit_cycles[t_idx])
        valid_timesteps = int(vtm.sum())
        is_last_override = (t_idx not in regular_set)
        meta = {
            "fd_subset": fd_subset,
            "split": split,
            "unit_id": int(unit_id),
            "cycle": cycle,
            "t_idx_in_unit": int(t_idx),
            "valid_timesteps": valid_timesteps,
            "min_valid_timesteps": int(mv) if mv > 0 else None,
            "stride": int(stride),
            "selected_by_last_override": bool(is_last_override),
            "below_min_valid_because_last": bool(below_min and is_last_override),
        }
        yield {
            "patches": patches,
            "valid_time_mask": vtm,
            "valid_patch_mask": vpm,
            "canales_constantes_mask": const_mask,
            "mean": mean,
            "std_used": std_used,
            "target": {
                "rul_physical": float(unit_rul_physical[t_idx]),
                "rul_capped_125": float(unit_rul_capped[t_idx]),
                "cycle": cycle,
            },
            "meta": meta,
        }


def iter_split_windows(
    split_dict: Dict[str, Any],
    window_size: int = 512,
    patch_size: int = 16,
    stride: int = 1,
    min_valid_timesteps: Optional[int] = None,
    include_last_per_unit: bool = True,
):
    """Itera ventanas de todas las unidades de un split.

    Args:
        split_dict: salida de `build_train_val_test_rul`[fd][split].
        window_size, patch_size, stride, min_valid_timesteps,
        include_last_per_unit: ver `iter_unit_windows`.

    Yields:
        Mismos samples que `iter_unit_windows`.
    """
    fd = split_dict["fd_subset"]
    split = split_dict["split"]
    unit_id = split_dict["unit_id"]
    cycle = split_dict["cycle"]
    op = split_dict["op_settings"]
    se = split_dict["sensors"]
    channels = np.concatenate([op, se], axis=1).astype(np.float32)
    rul_phys = split_dict["rul_physical"]
    rul_capped = split_dict["rul_capped_125"]

    for u in np.unique(unit_id):
        mask = unit_id == u
        yield from iter_unit_windows(
            unit_id=int(u),
            unit_cycles=cycle[mask],
            unit_channels=channels[mask],
            unit_rul_physical=rul_phys[mask],
            unit_rul_capped=rul_capped[mask],
            fd_subset=fd,
            split=split,
            window_size=window_size,
            patch_size=patch_size,
            stride=stride,
            min_valid_timesteps=min_valid_timesteps,
            include_last_per_unit=include_last_per_unit,
        )


# ----------------------------------------------------------------------
# Writer TAR + manifest (commit 3) — NO ejecuta nada sin --write-shards.
# ----------------------------------------------------------------------

# Política de mapping split del builder -> split original NASA. CMAPSS train
# y val del builder vienen ambos del split train original; el split test del
# builder es el test original tal cual. Ver sec "Contratos técnicos" en
# results/downstream/cmapss_rul_decision/decision.md.
_BUILDER_SPLIT_TO_SOURCE_SPLIT = {
    "train": "train_orig",
    "val": "train_orig",
    "test": "test_orig",
}


def split_to_source_split(split_builder: str) -> str:
    """Devuelve el `source_split` correspondiente al split del builder.

    train | val -> 'train_orig'
    test        -> 'test_orig'

    Raises:
        ValueError si `split_builder` no es uno de los 3 conocidos.
    """
    if split_builder not in _BUILDER_SPLIT_TO_SOURCE_SPLIT:
        raise ValueError(
            f"split_builder desconocido: {split_builder!r}. "
            f"Esperado uno de {list(_BUILDER_SPLIT_TO_SOURCE_SPLIT.keys())}."
        )
    return _BUILDER_SPLIT_TO_SOURCE_SPLIT[split_builder]


def make_unit_global_id(fd_subset: str, source_split: str, unit_id: int) -> str:
    """Devuelve el `unit_global_id` canonico:

        CMAPSS_<fd>_<source_split>_unit<unit_id>

    donde `source_split in {train_orig, test_orig}` segun NASA, no el split
    del builder. Esto evita falsos positivos al comparar unit_id numericos
    entre train_orig y test_orig (son motores distintos).
    """
    return f"CMAPSS_{fd_subset}_{source_split}_unit{int(unit_id)}"


def make_sample_key(
    fd_subset: str,
    split_builder: str,
    source_split: str,
    unit_id: int,
    t_idx: int,
) -> str:
    """Devuelve la `__key__` webdataset del sample.

    Formato: cmapss_<fd>_<split_builder>_<source_split>_unit<id>_w<t_idx:06d>
    """
    return (
        f"cmapss_{fd_subset}_{split_builder}_{source_split}_"
        f"unit{int(unit_id)}_w{int(t_idx):06d}"
    )


def expand_valid_time_mask_cw(vtm_w: np.ndarray, n_channels: int) -> np.ndarray:
    """Expande `valid_time_mask` de `(W,)` a `(C, W)` por broadcast.

    En CMAPSS la mascara temporal no depende del canal (padding causal
    uniforme), pero el contrato channel-independent del SSL central exige
    persistir la mascara con dimension C. Asi el DataLoader trata
    homogeneamente todos los datasets sin reshapes ad-hoc.
    """
    if vtm_w.ndim != 1:
        raise ValueError(f"vtm_w debe ser (W,); recibido {vtm_w.shape}")
    if vtm_w.dtype != bool:
        raise ValueError(f"vtm_w debe ser bool, recibido {vtm_w.dtype}")
    if n_channels <= 0:
        raise ValueError(f"n_channels debe ser > 0, recibido {n_channels}")
    return np.broadcast_to(vtm_w[None, :], (int(n_channels), vtm_w.shape[0])).copy()


def _npy_bytes(arr: np.ndarray) -> bytes:
    """Serializa un numpy array a bytes formato .npy (allow_pickle=False)."""
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def build_sample_payload(sample: Dict[str, Any]) -> Tuple[str, Dict[str, bytes]]:
    """Construye el payload webdataset-style de UN sample.

    Args:
        sample: dict emitido por `iter_unit_windows` (`patches`,
            `valid_time_mask (W,)`, `valid_patch_mask (C,N)`,
            `canales_constantes_mask (C,)`, `mean (C,)`, `std_used (C,)`,
            `target {rul_physical, rul_capped_125, cycle}`, `meta`).

    Returns:
        (key, blobs_dict):
          - key: `__key__` webdataset (sin extensiones).
          - blobs_dict: dict {ext: bytes} con las entries que iran al TAR
            (cada una se escribe como `{key}.{ext}`).

    El payload incluye:
      patches.npy                       (C, N, P)   float32
      valid_time_mask.npy               (W,)        bool
      valid_patch_mask.npy              (C, N)      bool
      canales_constantes_mask.npy       (C,)        bool
      mean.npy                          (C,)        float32
      std_used.npy                      (C,)        float32
      rul_physical.npy                  scalar      float32
      rul_capped_125.npy                scalar      float32
      meta.json                         dict        utf-8

    Contrato canonico de masks (compatible con PatchTSTPhm.forward):
      - `valid_time_mask` se guarda como `(W,)` para que al batchear
        sea `(B, W)`, exactamente lo que espera el encoder. La
        expansion `(C, W)` queda disponible via
        `expand_valid_time_mask_cw` para consumidores que la
        necesiten, pero NO se persiste en disco.
      - `valid_patch_mask` se guarda como `(C, N)` para que al
        batchear sea `(B, C, N)` (canonicalizable por el encoder).

    Y enriquece `meta` con: `source_split`, `unit_global_id`,
    `window_size`, `patch_size`, `n_patches`, `n_channels`,
    `target_rul_physical`, `target_rul_capped_125`.
    """
    meta_in = sample["meta"]
    fd = str(meta_in["fd_subset"])
    split_b = str(meta_in["split"])
    unit_id = int(meta_in["unit_id"])
    t_idx = int(meta_in["t_idx_in_unit"])
    source_split = split_to_source_split(split_b)
    unit_global_id = make_unit_global_id(fd, source_split, unit_id)
    key = make_sample_key(fd, split_b, source_split, unit_id, t_idx)

    patches = sample["patches"]              # (C, N, P)
    if patches.ndim != 3:
        raise ValueError(f"patches debe ser (C,N,P); recibido {patches.shape}")
    C, N, P = patches.shape

    vtm_w = sample["valid_time_mask"]        # (W,)
    if vtm_w.ndim != 1:
        raise ValueError(
            f"valid_time_mask debe ser (W,); recibido {vtm_w.shape}"
        )
    W = vtm_w.shape[0]
    if W != N * P:
        raise ValueError(
            f"W={W} no coincide con N*P={N*P}"
        )
    # `valid_time_mask` se persiste como (W,) por sample. Al batchear
    # da (B, W) que es exactamente lo que PatchTSTPhm.forward espera.
    # Si algun consumidor necesita la version expandida (C, W), debe
    # llamar a `expand_valid_time_mask_cw` en el DataLoader/collate.

    vpm = sample["valid_patch_mask"]         # (C, N)
    if vpm.shape != (C, N):
        raise ValueError(
            f"valid_patch_mask debe ser (C,N)=({C},{N}); "
            f"recibido {vpm.shape}"
        )

    const_mask = sample["canales_constantes_mask"]
    if const_mask.shape != (C,):
        raise ValueError(
            f"canales_constantes_mask debe ser (C,)=({C},); "
            f"recibido {const_mask.shape}"
        )

    mean_c = sample["mean"]
    std_c = sample["std_used"]

    rul_phys = np.float32(sample["target"]["rul_physical"])
    rul_capped = np.float32(sample["target"]["rul_capped_125"])

    meta_out = dict(meta_in)
    meta_out["source_split"] = source_split
    meta_out["unit_global_id"] = unit_global_id
    meta_out["window_size"] = int(W)
    meta_out["patch_size"] = int(P)
    meta_out["n_patches"] = int(N)
    meta_out["n_channels"] = int(C)
    meta_out["target_rul_physical"] = float(rul_phys)
    meta_out["target_rul_capped_125"] = float(rul_capped)

    blobs = {
        "patches.npy": _npy_bytes(patches.astype(np.float32, copy=False)),
        "valid_time_mask.npy": _npy_bytes(vtm_w),       # (W,) canonico
        "valid_patch_mask.npy": _npy_bytes(vpm),         # (C, N)
        "canales_constantes_mask.npy": _npy_bytes(const_mask),
        "mean.npy": _npy_bytes(mean_c.astype(np.float32, copy=False)),
        "std_used.npy": _npy_bytes(std_c.astype(np.float32, copy=False)),
        "rul_physical.npy": _npy_bytes(rul_phys),
        "rul_capped_125.npy": _npy_bytes(rul_capped),
        "meta.json": json.dumps(meta_out, sort_keys=True).encode("utf-8"),
    }
    return key, blobs


def compute_pipeline_config_hash(decisions: Dict[str, Any]) -> str:
    """Hash determinista (hex 16 chars) del bloque `decisions` de la
    politica del builder. Si dos corridas declaran misma politica deben
    obtener el mismo hash bit-a-bit.
    """
    payload = json.dumps(decisions, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def compute_pipeline_code_version() -> str:
    """Devuelve el git HEAD del repo si esta disponible; si no, 'unknown'.

    No falla si no es un repo git ni si `git` no esta en PATH.
    """
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, check=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception:
        return "unknown"


def _write_shard_tar(
    out_dir: Path,
    split_builder: str,
    shard_idx: int,
    samples_payloads: List[Tuple[str, Dict[str, bytes]]],
) -> Path:
    """Escribe N samples a un fichero .tar (webdataset-style).

    Cada sample contribuye varios entries con el patron `{key}.{ext}`.
    Los entries dentro del mismo sample son consecutivos en el TAR, asi
    el reader puede agruparlos por prefijo.

    Args:
        out_dir: directorio raiz; el shard se escribe en
            `out_dir/<split_builder>/<SHARD_PREFIX>_<NNNN>.tar`.
        split_builder: 'train' | 'val' | 'test'.
        shard_idx: indice de shard (0-based).
        samples_payloads: lista de (key, blobs_dict) producida por
            `build_sample_payload`.

    Returns:
        Path al .tar escrito.
    """
    split_dir = out_dir / split_builder
    split_dir.mkdir(parents=True, exist_ok=True)
    tar_path = split_dir / f"{SHARD_PREFIX}_{shard_idx:04d}.tar"
    with tarfile.open(tar_path, "w") as tf:
        for key, blobs in samples_payloads:
            # Orden estable de entries: alfabetico por extension para que
            # dos corridas con misma data produzcan TARs identicos bit-a-bit
            # salvo timestamps (que tampoco controlamos en tarfile).
            for ext in sorted(blobs.keys()):
                data = blobs[ext]
                info = tarfile.TarInfo(name=f"{key}.{ext}")
                info.size = len(data)
                info.mtime = 0  # determinista
                tf.addfile(info, io.BytesIO(data))
    return tar_path


def write_split_shards(
    parsed_with_rul: Dict[str, Any],
    split_builder: str,
    decisions: Dict[str, Any],
    out_dir: Path,
    shard_size: int = DEFAULT_SHARD_SIZE,
) -> Dict[str, Any]:
    """Itera ventanas del split builder sobre los 4 FD y escribe shards
    TAR en `out_dir/<split_builder>/`.

    No aplica filtros adicionales: usa la misma politica
    (`stride`, `min_valid_timesteps`, `include_last_per_unit`) del
    bloque `decisions`.

    Args:
        parsed_with_rul: salida de `build_train_val_test_rul`.
        split_builder: 'train' | 'val' | 'test'.
        decisions: dict con `window_size, patch_size, stride,
            min_valid_timesteps, include_last_per_unit`.
        out_dir: raiz donde se escriben los shards (sin nombre de split).
        shard_size: numero de samples por shard.

    Returns:
        dict con `n_samples`, `n_shards`, `n_units`, `unit_global_ids`
        (list[str]), `rul_physical_min`, `rul_physical_max`,
        `rul_capped_125_max`, `shard_paths`.
    """
    if shard_size <= 0:
        raise ValueError(f"shard_size debe ser > 0, recibido {shard_size}")
    W = int(decisions["window_size"])
    P = int(decisions["patch_size"])
    stride = int(decisions["stride"])
    mv = decisions.get("min_valid_timesteps", None)
    mv_int = int(mv) if (mv is not None and int(mv) > 0) else None
    incl = bool(decisions.get("include_last_per_unit", True))

    n_samples = 0
    n_shards = 0
    unit_global_ids: List[str] = []
    rul_phys_vals: List[float] = []
    rul_capped_vals: List[float] = []
    shard_paths: List[str] = []
    buffer: List[Tuple[str, Dict[str, bytes]]] = []

    for fd in CMAPSS_FD_SUBSETS:
        fd_data = parsed_with_rul.get(fd)
        if fd_data is None:
            continue
        split_dict = fd_data.get(split_builder)
        if split_dict is None:
            continue
        for sample in iter_split_windows(
            split_dict,
            window_size=W,
            patch_size=P,
            stride=stride,
            min_valid_timesteps=mv_int,
            include_last_per_unit=incl,
        ):
            key, blobs = build_sample_payload(sample)
            meta = json.loads(blobs["meta.json"].decode("utf-8"))
            unit_global_ids.append(meta["unit_global_id"])
            rul_phys_vals.append(float(meta["target_rul_physical"]))
            rul_capped_vals.append(float(meta["target_rul_capped_125"]))
            buffer.append((key, blobs))
            n_samples += 1
            if len(buffer) >= shard_size:
                tar_path = _write_shard_tar(out_dir, split_builder, n_shards, buffer)
                shard_paths.append(str(tar_path))
                n_shards += 1
                buffer = []

    if buffer:
        tar_path = _write_shard_tar(out_dir, split_builder, n_shards, buffer)
        shard_paths.append(str(tar_path))
        n_shards += 1
        buffer = []

    return {
        "split_builder": split_builder,
        "n_samples": n_samples,
        "n_shards": n_shards,
        "n_units": int(len(set(unit_global_ids))),
        "unit_global_ids": unit_global_ids,
        "rul_physical_min": float(min(rul_phys_vals)) if rul_phys_vals else 0.0,
        "rul_physical_max": float(max(rul_phys_vals)) if rul_phys_vals else 0.0,
        "rul_capped_125_max": float(max(rul_capped_vals)) if rul_capped_vals else 0.0,
        "shard_paths": shard_paths,
    }


def validate_anti_leakage(
    parsed_with_rul: Dict[str, Any],
    split_stats: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """Verifica que no haya solape de `unit_global_id` entre splits del
    builder y que el split val_orig se construye desde train_orig de
    cada FD.

    Args:
        parsed_with_rul: salida de `build_train_val_test_rul` (contiene
            `unit_split` por FD).
        split_stats: dict {split_builder: write_split_shards_result}.

    Returns:
        dict con resultado de cada check (todos True si todo OK).

    Raises:
        AssertionError si algun check falla.
    """
    train_ids = set(split_stats["train"]["unit_global_ids"])
    val_ids = set(split_stats["val"]["unit_global_ids"])
    test_ids = set(split_stats["test"]["unit_global_ids"])

    overlap_train_val = train_ids & val_ids
    overlap_train_test = train_ids & test_ids
    overlap_val_test = val_ids & test_ids
    assert not overlap_train_val, (
        f"Anti-leakage: train y val comparten {len(overlap_train_val)} "
        f"unit_global_ids: {sorted(overlap_train_val)[:5]}..."
    )
    assert not overlap_train_test, (
        f"Anti-leakage: train y test comparten {len(overlap_train_test)} "
        f"unit_global_ids: {sorted(overlap_train_test)[:5]}..."
    )
    assert not overlap_val_test, (
        f"Anti-leakage: val y test comparten {len(overlap_val_test)} "
        f"unit_global_ids: {sorted(overlap_val_test)[:5]}..."
    )

    # train_orig: todos los ids de train y val del builder deben tener
    # source_split=train_orig; los de test, test_orig.
    def _all_source_split(ids: set, expected: str) -> bool:
        for gid in ids:
            # gid = "CMAPSS_<fd>_<source>_unit<id>"
            parts = gid.split("_")
            # parts[0]='CMAPSS', parts[1]=fd, parts[2]=source_a, parts[3]=source_b
            # source_split tiene un guion bajo dentro: train_orig / test_orig
            source = f"{parts[2]}_{parts[3]}"
            if source != expected:
                return False
        return True

    train_all_train_orig = _all_source_split(train_ids, "train_orig")
    val_all_train_orig = _all_source_split(val_ids, "train_orig")
    test_all_test_orig = _all_source_split(test_ids, "test_orig")
    assert train_all_train_orig, (
        "Anti-leakage: split builder train contiene ids con source_split "
        "distinto de train_orig"
    )
    assert val_all_train_orig, (
        "Anti-leakage: split builder val contiene ids con source_split "
        "distinto de train_orig"
    )
    assert test_all_test_orig, (
        "Anti-leakage: split builder test contiene ids con source_split "
        "distinto de test_orig"
    )

    # Consistencia con parsed_with_rul[fd]['unit_split']:
    # train_ids del builder ∪ val_ids del builder == train_orig de cada FD.
    by_fd_checks: Dict[str, Dict[str, Any]] = {}
    for fd in CMAPSS_FD_SUBSETS:
        if fd not in parsed_with_rul:
            continue
        us = parsed_with_rul[fd]["unit_split"]
        expected_train_orig = {
            make_unit_global_id(fd, "train_orig", int(u))
            for u in us["train_units"]
        }
        expected_val_from_train_orig = {
            make_unit_global_id(fd, "train_orig", int(u))
            for u in us["val_units"]
        }
        expected_test_orig = {
            make_unit_global_id(fd, "test_orig", int(u))
            for u in us["test_units"]
        }
        actual_train_fd = {gid for gid in train_ids if f"_{fd}_" in gid}
        actual_val_fd = {gid for gid in val_ids if f"_{fd}_" in gid}
        actual_test_fd = {gid for gid in test_ids if f"_{fd}_" in gid}
        # Permitimos que algunos units no aparezcan (porque min_valid o
        # selected_t_indices puede no emitir nada). El check duro es que
        # los ids observados sean subset de los esperados.
        sub_train = actual_train_fd.issubset(expected_train_orig)
        sub_val = actual_val_fd.issubset(expected_val_from_train_orig)
        sub_test = actual_test_fd.issubset(expected_test_orig)
        assert sub_train, (
            f"Anti-leakage {fd}: train del builder no es subset de "
            f"train_orig"
        )
        assert sub_val, (
            f"Anti-leakage {fd}: val del builder no es subset de "
            f"train_orig (val proviene del train original)"
        )
        assert sub_test, (
            f"Anti-leakage {fd}: test del builder no es subset de "
            f"test_orig"
        )
        by_fd_checks[fd] = {
            "n_train_orig_expected": len(expected_train_orig),
            "n_val_from_train_orig_expected": len(expected_val_from_train_orig),
            "n_test_orig_expected": len(expected_test_orig),
            "n_train_observed": len(actual_train_fd),
            "n_val_observed": len(actual_val_fd),
            "n_test_observed": len(actual_test_fd),
            "train_subset_of_train_orig": True,
            "val_subset_of_train_orig": True,
            "test_subset_of_test_orig": True,
        }

    return {
        "no_overlap_train_val": True,
        "no_overlap_train_test": True,
        "no_overlap_val_test": True,
        "train_source_split_is_train_orig": True,
        "val_source_split_is_train_orig": True,
        "test_source_split_is_test_orig": True,
        "train_val_drawn_from_train_orig": True,
        "test_is_test_orig": True,
        "by_fd": by_fd_checks,
    }


def build_manifest(
    decisions: Dict[str, Any],
    split_stats: Dict[str, Dict[str, Any]],
    anti_leakage: Dict[str, Any],
    pipeline_config_hash: str,
    pipeline_code_version: str,
    parsed_with_rul: Dict[str, Any],
    shard_size: int,
    formula_train_val: str,
    formula_test: str,
) -> Dict[str, Any]:
    """Construye el `manifest.json` canonico del dataset CMAPSS_RUL.

    Sigue el contrato de la sec 13 de CLAUDE.md adaptado a RUL.
    """
    W = int(decisions["window_size"])
    P = int(decisions["patch_size"])
    N = W // P
    C = int(decisions.get("n_channels", CMAPSS_N_CHANNELS_WITH_OP))

    n_windows = {
        s: int(split_stats[s]["n_samples"]) for s in ("train", "val", "test")
    }
    n_units = {
        s: int(split_stats[s]["n_units"]) for s in ("train", "val", "test")
    }
    n_shards = {
        s: int(split_stats[s]["n_shards"]) for s in ("train", "val", "test")
    }

    # Temporal / channel patches: cada ventana aporta N patches temporales
    # validos (asumimos valid_patch_mask se cuenta a parte de manera mas
    # precisa, pero aqui usamos el numero maximo posible por ventana
    # consistente con el contrato de SSL central; el writer ya guarda
    # valid_patch_mask exacto por sample).
    n_temporal_patches = {s: n_windows[s] * N for s in ("train", "val", "test")}
    n_channel_patches = {s: n_windows[s] * N * C for s in ("train", "val", "test")}

    rul_phys_min = {
        s: float(split_stats[s]["rul_physical_min"]) for s in ("train", "val", "test")
    }
    rul_phys_max = {
        s: float(split_stats[s]["rul_physical_max"]) for s in ("train", "val", "test")
    }
    rul_capped_max = {
        s: float(split_stats[s]["rul_capped_125_max"]) for s in ("train", "val", "test")
    }

    # n_units originales (de NASA, antes del split val).
    n_units_original = {}
    for fd in CMAPSS_FD_SUBSETS:
        if fd in parsed_with_rul:
            us = parsed_with_rul[fd]["unit_split"]
            n_units_original[fd] = {
                "train_orig": len(us["train_units"]) + len(us["val_units"]),
                "test_orig": len(us["test_units"]),
            }

    return {
        "dataset": "CMAPSS_RUL",
        "manifest_version": MANIFEST_VERSION,
        "role": "TRANSFER_TARGET",
        "evaluation_tier": "primary",
        "client": "aero_engines",
        "pipeline_code_version": pipeline_code_version,
        "pipeline_config_hash": pipeline_config_hash,
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "window_size": W,
        "patch_size": P,
        "n_patches": N,
        "n_channels": C,
        "stride": int(decisions["stride"]),
        "min_valid_timesteps": decisions.get("min_valid_timesteps"),
        "include_last_per_unit": bool(decisions.get("include_last_per_unit", True)),
        "window_mode": "rolling_causal",
        "target_policy": "rul_at_prediction_cycle",
        "target_candidates": ["rul_physical", "rul_capped_125"],
        "target_warnings": [],
        "split_policy": "unit_holdout_by_fd",
        "val_frac": 0.2,
        "split_seed": 42,
        "normalization_policy": "instance_norm_per_window_channel_ignore_padding",
        "normalization_stats_saved": True,
        "batching_policy": "por_dataset",
        "shard_size": int(shard_size),
        "formula_train_val": formula_train_val,
        "formula_test": formula_test,
        "rul_cap": float(decisions.get("rul_cap", 125.0)),
        "n_units_por_split": n_units,
        "n_units_original_por_fd": n_units_original,
        "n_windows_por_split": n_windows,
        "n_shards_por_split": n_shards,
        "n_temporal_patches_por_split": n_temporal_patches,
        "n_channel_patches_por_split": n_channel_patches,
        "rul_physical_min_por_split": rul_phys_min,
        "rul_physical_max_por_split": rul_phys_max,
        "rul_capped_125_max_por_split": rul_capped_max,
        "unit_global_id_policy": "CMAPSS_<fd>_<source_split>_unit<unit_id>",
        "anti_leakage_checks": anti_leakage,
        "warnings": [],
    }


def assert_writer_hard_constraints(
    split_stats: Dict[str, Dict[str, Any]],
    preview_totals: Optional[Dict[str, Any]],
    decisions: Dict[str, Any],
) -> None:
    """Asserts duros pre-escritura.

    - `rul_physical_min == 0` en train y val (por `include_last_per_unit`).
    - `rul_physical_min >= 0` en test (oficial RUL >= 0).
    - `rul_capped_125_max <= rul_cap` en los 3 splits.
    - Si se pasa `preview_totals`, los conteos deben coincidir bit-a-bit.
    """
    cap = float(decisions.get("rul_cap", 125.0))
    incl = bool(decisions.get("include_last_per_unit", True))

    for s in ("train", "val", "test"):
        rp_min = float(split_stats[s]["rul_physical_min"])
        assert rp_min >= 0.0, (
            f"writer assert: {s} rul_physical_min={rp_min} < 0"
        )

    if incl:
        for s in ("train", "val"):
            rp_min = float(split_stats[s]["rul_physical_min"])
            assert rp_min == 0.0, (
                f"writer assert: con include_last_per_unit=True, "
                f"{s} rul_physical_min debe ser 0, observado {rp_min}"
            )

    if cap > 0:
        for s in ("train", "val", "test"):
            rc_max = float(split_stats[s]["rul_capped_125_max"])
            assert rc_max <= cap + 1e-6, (
                f"writer assert: {s} rul_capped_125_max={rc_max} > cap={cap}"
            )

    if preview_totals is not None:
        expected = {
            "train": int(preview_totals["n_windows_train"]),
            "val": int(preview_totals["n_windows_val"]),
            "test": int(preview_totals["n_windows_test"]),
        }
        for s in ("train", "val", "test"):
            got = int(split_stats[s]["n_samples"])
            exp = expected[s]
            assert got == exp, (
                f"writer assert: n_samples[{s}]={got} != preview={exp}. "
                f"Posible cambio de politica entre preview y writer."
            )


def write_shards_main(
    args: argparse.Namespace,
    parsed_with_rul: Dict[str, Any],
    preview: Dict[str, Any],
    formula_train_val: str,
    formula_test: str,
) -> int:
    """Orquestador del writer: itera los 3 splits, escribe TARs, valida
    anti-leakage, escribe `manifest.json` y `done.flag`.

    Reentrancia: si `done.flag` existe y el `pipeline_config_hash` del
    manifest coincide con la corrida actual, no reescribe nada (a menos
    que se pase `--force-overwrite`).

    Returns:
        0 si exito; 1 si error en asserts duros.
    """
    # Guard: la version canonica usa stride=CANONICAL_STRIDE. Cualquier
    # otro stride solo se permite con --allow-noncanonical, para evitar
    # que una ablation se cuele como version oficial del downstream RUL.
    if int(args.stride) != CANONICAL_STRIDE and not args.allow_noncanonical:
        print(
            f"\n[writer] ABORT: stride={args.stride} != canonico="
            f"{CANONICAL_STRIDE}. Si esto es deliberado (ablation),"
            f" pasa --allow-noncanonical y un --out-dir distinto del"
            f" canonico. No se escribe nada."
        )
        return 1

    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    done_flag = out_dir / DONE_FLAG_NAME
    manifest_path = out_dir / MANIFEST_NAME

    mv = args.min_valid_timesteps if args.min_valid_timesteps > 0 else None
    decisions = {
        "window_size": int(args.window_size),
        "patch_size": int(args.patch_size),
        "n_patches": int(args.window_size) // int(args.patch_size),
        "n_channels": CMAPSS_N_CHANNELS_WITH_OP,
        "window_mode": "rolling_causal",
        "target_policy": "rul_at_prediction_cycle",
        "split_policy": "unit_holdout_by_fd",
        "normalization_policy":
            "instance_norm_per_window_channel_ignore_padding",
        "stride": int(args.stride),
        "min_valid_timesteps": int(mv) if mv is not None else None,
        "include_last_per_unit": bool(args.include_last_per_unit),
        "rul_cap": float(args.rul_cap),
        "val_frac": 0.2,
        "split_seed": 42,
    }
    pipeline_config_hash = compute_pipeline_config_hash(decisions)
    pipeline_code_version = compute_pipeline_code_version()

    # Reentrancia.
    if done_flag.exists() and not args.force_overwrite:
        if manifest_path.exists():
            try:
                existing = json.loads(manifest_path.read_text())
                existing_hash = existing.get("pipeline_config_hash")
                if existing_hash == pipeline_config_hash:
                    print(
                        f"\n[writer] done.flag existente con "
                        f"pipeline_config_hash={pipeline_config_hash} "
                        f"coincide. Saltando escritura. Pasa "
                        f"--force-overwrite para reescribir."
                    )
                    return 0
                else:
                    print(
                        f"\n[writer] done.flag existente con hash distinto "
                        f"(disk={existing_hash}, current={pipeline_config_hash}). "
                        f"Aborto por seguridad. Borra el directorio o pasa "
                        f"--force-overwrite."
                    )
                    return 1
            except Exception as e:
                print(f"\n[writer] error leyendo manifest existente: {e}")
                return 1
        else:
            print(
                "\n[writer] done.flag existente pero manifest.json ausente. "
                "Aborto por consistencia."
            )
            return 1

    print(f"\n[writer] escritura autorizada.")
    print(f"[writer] pipeline_config_hash = {pipeline_config_hash}")
    print(f"[writer] pipeline_code_version = {pipeline_code_version}")
    print(f"[writer] shard_size = {args.shard_size}")

    split_stats: Dict[str, Dict[str, Any]] = {}
    for split_builder in ("train", "val", "test"):
        print(f"[writer] split={split_builder}: escribiendo shards ...")
        stats = write_split_shards(
            parsed_with_rul=parsed_with_rul,
            split_builder=split_builder,
            decisions=decisions,
            out_dir=out_dir,
            shard_size=int(args.shard_size),
        )
        print(
            f"[writer]   n_samples={stats['n_samples']} "
            f"n_shards={stats['n_shards']} "
            f"n_units={stats['n_units']} "
            f"rul_phys=[{stats['rul_physical_min']:.0f},"
            f"{stats['rul_physical_max']:.0f}] "
            f"cap_max={stats['rul_capped_125_max']:.1f}"
        )
        split_stats[split_builder] = stats

    # Asserts duros y anti-leakage.
    try:
        preview_totals = (preview or {}).get("totals") if preview else None
        assert_writer_hard_constraints(split_stats, preview_totals, decisions)
        anti_leakage = validate_anti_leakage(parsed_with_rul, split_stats)
    except AssertionError as e:
        print(f"\n[writer] ASSERT DURA FALLO: {e}")
        return 1

    # Manifest.
    manifest = build_manifest(
        decisions=decisions,
        split_stats=split_stats,
        anti_leakage=anti_leakage,
        pipeline_config_hash=pipeline_config_hash,
        pipeline_code_version=pipeline_code_version,
        parsed_with_rul=parsed_with_rul,
        shard_size=int(args.shard_size),
        formula_train_val=formula_train_val,
        formula_test=formula_test,
    )
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    done_flag.write_text(
        f"pipeline_config_hash={pipeline_config_hash}\n"
        f"pipeline_code_version={pipeline_code_version}\n"
        f"generated_at={manifest['generated_at']}\n",
        encoding="utf-8",
    )
    print(f"\n[writer] manifest escrito en {manifest_path}")
    print(f"[writer] done.flag escrito en {done_flag}")
    return 0


# ----------------------------------------------------------------------
# Inspección de raw CMAPSS (no descarga)
# ----------------------------------------------------------------------


def _detect_fd_files(filenames: Sequence[str]) -> Dict[str, Any]:
    """Inspecciona una lista de nombres de fichero y agrupa por subset FD.

    CMAPSS estandar (NASA) usa nombres como `train_FD001.txt`,
    `test_FD001.txt`, `RUL_FD001.txt` para cada uno de los 4 subsets
    (FD001..FD004). PHMD puede entregar un layout distinto; este detector
    soporta el formato estandar y reporta gaps si los hay.

    Returns:
        dict con:
          - fd_subsets: lista de FD identificados (e.g. ['FD001','FD002']).
          - missing_train: list[str] FDs sin train_FDxxx.
          - missing_test:  list[str] FDs sin test_FDxxx.
          - missing_rul:   list[str] FDs sin RUL_FDxxx.
          - unrecognized:  list[str] ficheros que no encajan en el patron.
    """
    import re
    fd_re = re.compile(r"^(train|test|RUL)_FD(\d{3})\.txt$", re.IGNORECASE)
    has: Dict[str, set] = {"train": set(), "test": set(), "RUL": set()}
    unrecognized: List[str] = []
    for fn in filenames:
        m = fd_re.match(fn)
        if m is None:
            unrecognized.append(fn)
            continue
        kind = m.group(1).lower()
        fd = f"FD{m.group(2)}"
        if kind == "rul":
            has["RUL"].add(fd)
        else:
            has[kind].add(fd)

    all_fds = sorted(has["train"] | has["test"] | has["RUL"])
    missing_train = sorted([fd for fd in all_fds if fd not in has["train"]])
    missing_test = sorted([fd for fd in all_fds if fd not in has["test"]])
    missing_rul = sorted([fd for fd in all_fds if fd not in has["RUL"]])
    return {
        "fd_subsets": all_fds,
        "missing_train": missing_train,
        "missing_test": missing_test,
        "missing_rul": missing_rul,
        "unrecognized": unrecognized,
    }


def inspect_raw_cmapss(raw_root: Path) -> Dict[str, Any]:
    """Detecta el raw CMAPSS en cualquiera de los layouts conocidos.

    Layouts soportados, en orden de preferencia:

      1. **Directorio expandido**: `raw_root/CMAPSS/` (con subdirs
         train/ y test/ o ficheros sueltos). Listado recursivo.
      2. **Zip unico estilo NASA**: `raw_root/CMAPSS.zip`. Inspeccion
         con `zipfile` sin extraer.
      3. **Layout PHMD (zips splitados)**: `raw_root/CMAPSS_train.zip`
         + `raw_root/CMAPSS_test.zip`. Cada zip contiene `CMAPSS/train/`
         o `CMAPSS/test/` con `train_FDxxx.txt`, `test_FDxxx.txt`,
         `RUL_FDxxx.txt`. Inspeccion conjunta sin extraer.

    En cualquier caso, comprueba la presencia de los tripletes
    train/test/RUL por FD subset y reporta missing por kind.

    Returns:
        dict con:
          - raw_root
          - raw_missing: bool
          - raw_layout: str ("dir" | "zip_unico" | "zips_split_phmd" | None)
          - raw_zip_path: ruta al .zip unico si existe (layout 2)
          - raw_zip_train_path / raw_zip_test_path: rutas a los zips
            PHMD splitados (layout 3)
          - raw_dir_path: ruta al directorio expandido (layout 1)
          - candidate_files: lista de nombres relevantes
          - fd_subsets, missing_train, missing_test, missing_rul,
            unrecognized: ver `_detect_fd_files`
          - notes: list[str]
    """
    import zipfile
    notes: List[str] = []
    raw_root = Path(raw_root)
    out = {
        "raw_root": str(raw_root),
        "raw_missing": True,
        "raw_layout": None,
        "raw_zip_path": None,
        "raw_zip_train_path": None,
        "raw_zip_test_path": None,
        "raw_dir_path": None,
        "candidate_files": [],
        "fd_subsets": [],
        "missing_train": [],
        "missing_test": [],
        "missing_rul": [],
        "unrecognized": [],
        "notes": notes,
    }
    if not raw_root.exists():
        notes.append(f"raw_root {raw_root} no existe")
        return out

    dir_p = raw_root / "CMAPSS"
    zip_p = raw_root / "CMAPSS.zip"
    zip_train_p = raw_root / "CMAPSS_train.zip"
    zip_test_p = raw_root / "CMAPSS_test.zip"
    files: List[str] = []

    if dir_p.is_dir():
        # Layout 1: directorio expandido.
        out["raw_missing"] = False
        out["raw_layout"] = "dir"
        out["raw_dir_path"] = str(dir_p)
        files = sorted([
            p.name for p in dir_p.rglob("*") if p.is_file()
        ])
        out["candidate_files"] = files
        notes.append(f"layout=dir; encontrado {dir_p} con {len(files)} ficheros")
    elif zip_p.is_file():
        # Layout 2: zip unico estilo NASA.
        out["raw_missing"] = False
        out["raw_layout"] = "zip_unico"
        out["raw_zip_path"] = str(zip_p)
        try:
            with zipfile.ZipFile(zip_p, "r") as zf:
                # basename para que el detector de FD funcione con paths
                # tipo "CMAPSS/train/train_FD001.txt" o "train_FD001.txt".
                files = sorted([Path(n).name for n in zf.namelist() if not n.endswith("/")])
                out["candidate_files"] = files
                notes.append(
                    f"layout=zip_unico; encontrado {zip_p.name} con "
                    f"{len(files)} entradas; inspeccionado sin extraer."
                )
        except Exception as e:
            notes.append(f"WARN no se pudo abrir zip {zip_p}: {e}")
            out["raw_missing"] = True
    elif zip_train_p.is_file() and zip_test_p.is_file():
        # Layout 3: zips splitados PHMD. Tienen que estar AMBOS para
        # tener train + test + RUL.
        out["raw_missing"] = False
        out["raw_layout"] = "zips_split_phmd"
        out["raw_zip_train_path"] = str(zip_train_p)
        out["raw_zip_test_path"] = str(zip_test_p)
        n_train = n_test = 0
        try:
            with zipfile.ZipFile(zip_train_p, "r") as zf_tr:
                tr_names = [n for n in zf_tr.namelist() if not n.endswith("/")]
                n_train = len(tr_names)
            with zipfile.ZipFile(zip_test_p, "r") as zf_te:
                te_names = [n for n in zf_te.namelist() if not n.endswith("/")]
                n_test = len(te_names)
            # Union de basenames.
            files = sorted([Path(n).name for n in (tr_names + te_names)])
            out["candidate_files"] = files
            notes.append(
                f"layout=zips_split_phmd; encontrados {zip_train_p.name} "
                f"({n_train} entradas) y {zip_test_p.name} ({n_test} entradas); "
                "inspeccionados sin extraer."
            )
        except Exception as e:
            notes.append(f"WARN no se pudo abrir zips PHMD: {e}")
            out["raw_missing"] = True
    elif zip_train_p.is_file() or zip_test_p.is_file():
        # Solo uno de los dos zips splitados: incompleto.
        present = zip_train_p.name if zip_train_p.is_file() else zip_test_p.name
        missing = zip_test_p.name if zip_train_p.is_file() else zip_train_p.name
        notes.append(
            f"layout PHMD parcial: encontrado {present} pero falta {missing}. "
            "El builder necesita AMBOS zips para construir RUL fisico."
        )
        # raw_missing sigue True; el caller decide.

    if not out["raw_missing"] and files:
        fd_info = _detect_fd_files(files)
        out.update(fd_info)
        if not fd_info["fd_subsets"]:
            notes.append(
                "WARN: no se identifico ningun fichero train/test/RUL_FDxxx.txt. "
                "El layout del raw entregado por PHMD puede diferir del CMAPSS "
                "NASA estandar; revisar candidate_files manualmente."
            )
        else:
            notes.append(
                f"FD subsets detectados: {fd_info['fd_subsets']}"
            )
            for kind in ("train", "test", "rul"):
                missing = fd_info[f"missing_{kind}"]
                if missing:
                    notes.append(f"FALTAN ficheros {kind}_FDxxx para: {missing}")
            if fd_info["unrecognized"]:
                notes.append(
                    f"ficheros no reconocidos por el patron FDxxx: "
                    f"{fd_info['unrecognized'][:10]}"
                )

    if out["raw_missing"]:
        notes.append(
            "raw CMAPSS no encontrado. Comandos esperados (en Colab):\n"
            "  ls /content/drive/MyDrive/fm_fl_phmd/raw/datasets/ | grep -i cmapss\n"
            "Si tampoco esta, hay que descargarlo via el notebook 00_download_datasets.\n"
            "No descargamos automaticamente."
        )
    return out


# ----------------------------------------------------------------------
# Dry-run report
# ----------------------------------------------------------------------


def _json_safe(o: Any) -> Any:
    if o is None or isinstance(o, (bool, int, float, str)):
        if isinstance(o, float) and not math.isfinite(o):
            return None
        return o
    if isinstance(o, dict):
        return {str(k): _json_safe(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_json_safe(v) for v in o]
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_dry_run_report(report: Dict[str, Any], out_dir: Path) -> None:
    """Escribe `dry_run_report.{json,md}` en `out_dir` (crea si no existe)."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    j = out_dir / "dry_run_report.json"
    j.write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    md_lines = [
        "# CMAPSS RUL builder — dry-run report",
        "",
        f"Timestamp: {report.get('timestamp')}",
        f"raw_root: `{report.get('raw_root')}`",
        f"raw_missing: **{report.get('raw_missing')}**",
        f"raw_layout: **{report.get('raw_layout')}**",
        "",
    ]
    if report.get("raw_zip_path"):
        md_lines.append(f"- raw_zip_path: `{report['raw_zip_path']}`")
    if report.get("raw_zip_train_path"):
        md_lines.append(f"- raw_zip_train_path: `{report['raw_zip_train_path']}`")
    if report.get("raw_zip_test_path"):
        md_lines.append(f"- raw_zip_test_path: `{report['raw_zip_test_path']}`")
    if report.get("raw_dir_path"):
        md_lines.append(f"- raw_dir_path: `{report['raw_dir_path']}`")
    fd_subs = report.get("fd_subsets", [])
    if fd_subs:
        md_lines.append("")
        md_lines.append("## FD subsets detectados")
        md_lines.append("")
        md_lines.append(f"- subsets: {fd_subs}")
        for kind, key in [
            ("train", "missing_train"),
            ("test", "missing_test"),
            ("RUL", "missing_rul"),
        ]:
            missing = report.get(key, [])
            estado = "OK" if not missing else f"FALTAN: {missing}"
            md_lines.append(f"- {kind}_FDxxx: {estado}")
        if report.get("unrecognized"):
            md_lines.append(
                f"- no reconocidos por patron FDxxx (primeros 10): "
                f"{report['unrecognized'][:10]}"
            )

    if report.get("candidate_files"):
        files = report.get("candidate_files", [])
        md_lines.append("")
        md_lines.append("## Ficheros encontrados (primeros 80)")
        md_lines.append("")
        for f in files[:80]:
            md_lines.append(f"- `{f}`")
    md_lines.append("")
    md_lines.append("## Notas")
    md_lines.append("")
    for n in report.get("notes", []):
        md_lines.append(f"- {n}")
    md_lines.append("")
    # Politica de seleccion post-filtro (commit 2b)
    preview = report.get("preview") or {}
    dec = preview.get("decisions") or {}
    totals = preview.get("totals") or {}
    if dec:
        md_lines.append("")
        md_lines.append("## Politica de seleccion rolling_causal (post-filtro)")
        md_lines.append("")
        md_lines.append(f"- window_size: `{dec.get('window_size')}`")
        md_lines.append(f"- patch_size: `{dec.get('patch_size')}`")
        md_lines.append(f"- n_patches: `{dec.get('n_patches')}`")
        md_lines.append(f"- n_channels: `{dec.get('n_channels')}`")
        md_lines.append(f"- window_mode: `{dec.get('window_mode')}`")
        md_lines.append(f"- target_policy: `{dec.get('target_policy')}`")
        md_lines.append(f"- split_policy: `{dec.get('split_policy')}`")
        md_lines.append(f"- normalization_policy: `{dec.get('normalization_policy')}`")
        md_lines.append(f"- **stride**: `{dec.get('stride')}`")
        md_lines.append(f"- **min_valid_timesteps**: `{dec.get('min_valid_timesteps')}`")
        md_lines.append(f"- **include_last_per_unit**: `{dec.get('include_last_per_unit')}`")
        if totals:
            md_lines.append("")
            md_lines.append("## Totales post-filtro")
            md_lines.append("")
            md_lines.append(f"- n_windows_train: **{totals.get('n_windows_train')}**")
            md_lines.append(f"- n_windows_val: **{totals.get('n_windows_val')}**")
            md_lines.append(f"- n_windows_test: **{totals.get('n_windows_test')}**")
            md_lines.append(f"- n_windows_dropped_by_min_valid: {totals.get('n_windows_dropped_by_min_valid')}")
            md_lines.append(f"- n_windows_added_by_last_override: {totals.get('n_windows_added_by_last_override')}")
            md_lines.append(f"- estimated_size_gb_float32: **{totals.get('estimated_size_gb_float32')} GB**")
            md_lines.append(f"- frac_windows_padded: {totals.get('frac_windows_padded')}")
            md_lines.append(f"- frac_timesteps_valid_avg: {totals.get('frac_timesteps_valid_avg')}")
    md_lines.append("")
    md_lines.append("## Decision aplicable")
    md_lines.append("")
    md_lines.append(
        "Reconstruir RUL fisico desde raw (formula por split). Politica "
        "stride/min_valid/last_override confirmada. Ver "
        "`results/downstream/cmapss_rul_decision/decision.md`."
    )
    (out_dir / "dry_run_report.md").write_text(
        "\n".join(md_lines) + "\n", encoding="utf-8"
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="CMAPSS RUL downstream builder (esqueleto, dry-run por defecto)"
    )
    p.add_argument(
        "--raw-root", type=Path,
        default=Path("/content/drive/MyDrive/fm_fl_phmd/raw/datasets"),
        help="raiz de raw/datasets donde se busca CMAPSS",
    )
    p.add_argument(
        "--out-dir", type=Path,
        default=Path("/content/drive/MyDrive/fm_fl_phmd/processed_downstream/CMAPSS_RUL"),
        help="directorio destino para shards (no se escribe sin --write-shards)",
    )
    p.add_argument(
        "--results-dir", type=Path,
        default=Path("results/downstream/cmapss_rul_decision"),
        help="directorio del repo donde escribir dry_run_report",
    )
    p.add_argument(
        "--window-size", type=int, default=512,
        help="W ventana en muestras (default 512 = mismo que SSL)",
    )
    p.add_argument(
        "--stride", type=int, default=CANONICAL_STRIDE,
        help=f"stride entre ventanas (default canonico {CANONICAL_STRIDE}). "
             f"Con --write-shards, stride != {CANONICAL_STRIDE} requiere "
             f"--allow-noncanonical.",
    )
    p.add_argument(
        "--patch-size", type=int, default=16,
        help="P patch (default 16)",
    )
    p.add_argument(
        "--rul-cap", type=float, default=125.0,
        help="cap opcional sobre rul_physical (Heimes 2008 = 125). "
             "Solo aplica DESPUES de reconstruccion. <=0 = sin cap.",
    )
    p.add_argument(
        "--min-valid-timesteps", type=int, default=128,
        help="piso de timesteps reales en cada ventana (default 128). "
             "Descarta los primeros min_valid-1 ciclos por unidad "
             "(donde el padding es extremo). Pasar 0 para deshabilitar.",
    )
    p.add_argument(
        "--include-last-per-unit", dest="include_last_per_unit",
        action="store_true", default=True,
        help="(default) garantizar emision del ciclo T-1 de cada unidad "
             "(ciclo final, RUL=0 en train).",
    )
    p.add_argument(
        "--no-include-last-per-unit", dest="include_last_per_unit",
        action="store_false",
        help="desactivar el override del ultimo ciclo.",
    )
    p.add_argument(
        "--dry-run", action="store_true", default=True,
        help="por defecto True: solo inspecciona raw, no escribe shards",
    )
    p.add_argument(
        "--write-shards", action="store_true", default=False,
        help="autoriza la escritura de shards reales (default False)",
    )
    p.add_argument(
        "--shard-size", type=int, default=DEFAULT_SHARD_SIZE,
        help=f"numero de samples por shard TAR (default {DEFAULT_SHARD_SIZE}).",
    )
    p.add_argument(
        "--force-overwrite", action="store_true", default=False,
        help="reescribir shards aunque exista done.flag con mismo hash.",
    )
    p.add_argument(
        "--allow-noncanonical", action="store_true", default=False,
        help=f"autoriza --write-shards con stride distinto del canonico "
             f"({CANONICAL_STRIDE}). Solo para ablations; aparta el output "
             f"de la version canonica del downstream RUL.",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    print(f"=== build_cmapss_rul_downstream === ts={time.strftime('%Y-%m-%dT%H:%M:%S')}")
    print(f"  raw-root:     {args.raw_root}")
    print(f"  out-dir:      {args.out_dir}")
    print(f"  results-dir:  {args.results_dir}")
    print(f"  W={args.window_size}  stride={args.stride}  P={args.patch_size}")
    print(f"  rul-cap:                {args.rul_cap}  (<=0 = sin cap)")
    mv = args.min_valid_timesteps if args.min_valid_timesteps > 0 else None
    print(f"  min-valid-timesteps:    {mv}")
    print(f"  include-last-per-unit:  {args.include_last_per_unit}")
    print(f"  dry-run:      {args.dry_run}  write-shards: {args.write_shards}")

    # Dry-run real: inspecciona raw y escribe report.
    report = inspect_raw_cmapss(args.raw_root)
    report["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    report["window_size"] = args.window_size
    report["stride"] = args.stride
    report["patch_size"] = args.patch_size
    report["rul_cap"] = args.rul_cap
    report["formula_train_val"] = "max_cycle_by(FD, unit) - cycle"
    report["formula_test"] = (
        "last_observed_cycle_by(FD, unit) - cycle + official_RUL_FD[unit]"
    )
    if report["raw_missing"]:
        write_dry_run_report(report, args.results_dir)
        print(f"\nraw CMAPSS NO encontrado en {args.raw_root}")
        print(f"Report escrito en: {args.results_dir}")
        return 0  # no es error: es un report informativo

    # COMMIT 1: si el raw esta presente y es zips_split_phmd, parseamos
    # y generamos el preview rolling_causal. Sin escritura de shards.
    parsed_with_rul: Optional[Dict[str, Any]] = None
    if report.get("raw_layout") == "zips_split_phmd":
        try:
            # Solo cargamos los FD realmente presentes en los zips para no
            # forzar FD001..FD004 en tests sinteticos. En real CMAPSS PHMD
            # son los 4 igual.
            fd_present = tuple(report.get("fd_subsets") or CMAPSS_FD_SUBSETS)
            parsed = load_cmapss_raw_from_split_zips(
                args.raw_root, fd_subsets=fd_present,
            )
            parsed_with_rul = build_train_val_test_rul(
                parsed,
                val_frac=0.2,
                seed=42,
                rul_cap=args.rul_cap,
            )
            preview = preview_summary(
                parsed_with_rul,
                window_size=args.window_size,
                patch_size=args.patch_size,
                n_channels=CMAPSS_N_CHANNELS_WITH_OP,
                stride=args.stride,
                min_valid_timesteps=mv,
                include_last_per_unit=args.include_last_per_unit,
            )
            report["preview"] = preview
            print("\n=== Preview rolling_causal (commit 2b: stride + min_valid + last_override) ===")
            print(f"  politica: stride={preview['decisions']['stride']}  "
                  f"min_valid_timesteps={preview['decisions']['min_valid_timesteps']}  "
                  f"include_last_per_unit={preview['decisions']['include_last_per_unit']}")
            for fd, fd_sum in preview["by_fd"].items():
                for spl in ("train", "val", "test"):
                    s = fd_sum[spl]
                    extras = ""
                    if spl == "test":
                        extras = f"  official_rul_count={fd_sum['official_rul_count']}"
                    print(
                        f"  {fd}/{spl:<5} units={s['n_units_total']:>3} "
                        f"(with_window={s['n_units_with_at_least_one_window']:>3}, "
                        f"only_last={s['n_units_only_last_override']:>2})  "
                        f"n_rows={s['n_rows_original']:>6}  "
                        f"n_sel={s['n_windows_selected']:>6}  "
                        f"dropped={s['n_windows_dropped_by_min_valid']:>5}  "
                        f"last_added={s['n_windows_added_by_last_override']:>3}  "
                        f"rul_phys[{s['rul_physical_min']:.0f},{s['rul_physical_max']:.0f}]"
                        f"{extras}"
                    )
            tot = preview["totals"]
            print(f"\n  TOTALES post-filtro:")
            print(f"    n_windows_selected: train={tot['n_windows_train']}  "
                  f"val={tot['n_windows_val']}  test={tot['n_windows_test']}  "
                  f"TOTAL={tot['n_windows_total']}")
            print(f"    dropped_by_min_valid: {tot['n_windows_dropped_by_min_valid']}  "
                  f"added_by_last_override: {tot['n_windows_added_by_last_override']}")
            print(f"    estimated_size_gb_float32={tot['estimated_size_gb_float32']} GB "
                  f"(~{tot['bytes_per_window_estimate']/1024:.1f} KB/ventana)")
            print(f"    ventanas SIN padding: {tot['n_windows_full']}; "
                  f"CON padding: {tot['n_windows_padded']} "
                  f"({tot['frac_windows_padded']*100:.1f}%)")
            print(f"    frac_timesteps_valid_avg (post-filtro): "
                  f"{tot['frac_timesteps_valid_avg']:.4f}")
            if tot["estimated_size_gb_float32"] > 100:
                print("  WARN: estimated_size > 100 GB; considerar stride > 1 o min_valid mayor "
                      "antes de --write-shards.")
        except Exception as e:
            print(f"\nWARN parseo/preview fallo: {e}")
            report["preview_error"] = str(e)

    write_dry_run_report(report, args.results_dir)

    print(f"\nraw CMAPSS encontrado. Report escrito en: {args.results_dir}")

    # Si se pidio --write-shards, ahora entra el writer (commit 3).
    if args.write_shards and parsed_with_rul is not None \
            and "preview" in report:
        rc = write_shards_main(
            args=args,
            parsed_with_rul=parsed_with_rul,
            preview=report["preview"],
            formula_train_val=report["formula_train_val"],
            formula_test=report["formula_test"],
        )
        return rc
    elif args.write_shards:
        print(
            "\n[writer] --write-shards solicitado pero no hay parsed_with_rul "
            "ni preview disponibles (raw missing o preview fallo). No se "
            "escribe nada."
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
