"""Metricas downstream: clasificacion + regresion.

Implementaciones en numpy/torch puros, sin dependencia obligatoria de
sklearn. Si sklearn esta disponible se usa para verificacion cruzada en
los tests; en runtime las funciones devuelven resultados identicos
modulo precision floating.

Clasificacion (convencion):

- `y_true`: array/tensor 1D con clase verdadera por sample (int en
  `[0, n_classes)`).
- `y_pred`: idem, prediccion arg-max.

Regresion (convencion):

- `y_true`: array/tensor 1D float con target continuo (e.g. RUL ciclos).
- `y_pred`: idem, prediccion escalar del modelo.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np


ArrayLike = Union[np.ndarray, "torch.Tensor", list, tuple]

# Intentamos detectar torch UNA SOLA VEZ al cargar el modulo. Si torch no
# esta disponible o falla al cargar (clasico en Windows con DLL fbgemm),
# capturamos cualquier excepcion y simplemente no exponemos el soporte
# torch en `_to_numpy`. Las metricas siguen funcionando con numpy/listas.
try:
    import torch as _torch  # type: ignore
    _HAS_TORCH = True
except Exception:
    _torch = None  # type: ignore
    _HAS_TORCH = False


def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convierte tensores torch o listas a np.int64."""
    if _HAS_TORCH and isinstance(x, _torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x)
    if arr.dtype.kind == "f":
        arr = arr.astype(np.int64)
    return arr


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Fraccion de aciertos. Si `y_true` esta vacio, devuelve 0."""
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.size == 0:
        return 0.0
    if yt.shape != yp.shape:
        raise ValueError(f"shapes distintas: y_true {yt.shape} vs y_pred {yp.shape}")
    return float((yt == yp).mean())


def confusion_matrix(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: Optional[int] = None
) -> np.ndarray:
    """Matriz de confusion (rows=true, cols=pred), shape (K, K) int64.

    Si `n_classes` es None, se infiere como `max(y_true ∪ y_pred) + 1`.
    """
    yt = _to_numpy(y_true)
    yp = _to_numpy(y_pred)
    if yt.shape != yp.shape:
        raise ValueError(f"shapes distintas: y_true {yt.shape} vs y_pred {yp.shape}")
    if n_classes is None:
        if yt.size == 0 and yp.size == 0:
            n_classes = 1
        else:
            n_classes = int(max(int(yt.max(initial=-1)), int(yp.max(initial=-1))) + 1)
            if n_classes < 1:
                n_classes = 1
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    if yt.size == 0:
        return cm
    # Validacion de rango
    if (yt < 0).any() or (yt >= n_classes).any():
        raise ValueError(f"y_true contiene clases fuera de [0, {n_classes})")
    if (yp < 0).any() or (yp >= n_classes).any():
        raise ValueError(f"y_pred contiene clases fuera de [0, {n_classes})")
    for t, p in zip(yt.ravel(), yp.ravel()):
        cm[int(t), int(p)] += 1
    return cm


def balanced_accuracy(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: Optional[int] = None
) -> float:
    """Media del recall por clase: `mean_c (TP_c / (TP_c + FN_c))`.

    Si una clase no aparece en `y_true`, su recall se considera no
    definido y se ignora en la media (no contribuye con 0). Si ninguna
    clase tiene soporte en `y_true`, devuelve 0.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    row_sums = cm.sum(axis=1)  # TP + FN por clase
    recalls = []
    for c in range(cm.shape[0]):
        if row_sums[c] > 0:
            recalls.append(cm[c, c] / row_sums[c])
    if not recalls:
        return 0.0
    return float(np.mean(recalls))


def macro_f1(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: Optional[int] = None
) -> float:
    """F1 macro: media de F1 por clase con soporte > 0.

    Para cada clase c: `precision_c = TP_c / col_sum_c`,
    `recall_c = TP_c / row_sum_c`, `F1_c = 2 PR / (P+R)` con guardas
    contra division por cero.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    col_sums = cm.sum(axis=0)
    row_sums = cm.sum(axis=1)
    f1s = []
    for c in range(cm.shape[0]):
        tp = cm[c, c]
        prec = tp / col_sums[c] if col_sums[c] > 0 else 0.0
        rec = tp / row_sums[c] if row_sums[c] > 0 else 0.0
        if row_sums[c] == 0:
            # Sin soporte real en y_true; no entra en la media
            continue
        if prec + rec == 0:
            f1 = 0.0
        else:
            f1 = 2.0 * prec * rec / (prec + rec)
        f1s.append(f1)
    if not f1s:
        return 0.0
    return float(np.mean(f1s))


def support_per_class(y_true: ArrayLike, n_classes: Optional[int] = None) -> np.ndarray:
    """Cuenta de samples por clase en `y_true`. shape (K,) int64."""
    yt = _to_numpy(y_true)
    if n_classes is None:
        n_classes = int(yt.max(initial=-1) + 1) if yt.size > 0 else 0
    counts = np.zeros(n_classes, dtype=np.int64)
    for t in yt.ravel():
        if 0 <= int(t) < n_classes:
            counts[int(t)] += 1
    return counts


def per_class_precision_recall_f1(
    y_true: ArrayLike, y_pred: ArrayLike, n_classes: int
) -> dict:
    """Precision/recall/F1 por clase + soporte verdadero y predicho.

    Notas semanticas:

    - `support_true[c] = #{i: y_true[i] == c}`. Si es 0, la clase no aparece
      en y_true y `recall[c]` queda `None` (no definido). Esto es la misma
      regla que sklearn con `zero_division=None`.
    - `support_pred[c] = #{i: y_pred[i] == c}`. Si es 0, la clase no se
      predijo nunca y `precision[c]` queda `None`.
    - Si soporte true es 0, F1 tambien queda `None`.
    - Si soporte true>0 pero soporte pred==0 (la clase existe pero nunca se
      predijo correctamente), precision=0 cuando recall>0 es imposible (tp=0
      necesariamente); F1=0. Cubrimos eso explicitamente.

    Args:
        y_true: array/tensor 1D con clase verdadera (int en [0, n_classes)).
        y_pred: idem prediccion.
        n_classes: K (fuerza el rango; util cuando no todas las clases
            aparecen en y_true ni en y_pred).

    Returns:
        dict con keys:
          - precision: list[float|None], len K.
          - recall: list[float|None], len K.
          - f1: list[float|None], len K.
          - support_true: list[int], len K.
          - support_pred: list[int], len K.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes=n_classes)
    support_true = cm.sum(axis=1)  # TP+FN por clase
    support_pred = cm.sum(axis=0)  # TP+FP por clase
    precision: list = []
    recall: list = []
    f1: list = []
    for c in range(n_classes):
        tp = int(cm[c, c])
        st = int(support_true[c])
        sp = int(support_pred[c])

        # Recall:
        #   - support_true == 0 -> recall no definido (None).
        #   - support_true > 0  -> tp/st.
        if st == 0:
            rec_val = None
        else:
            rec_val = tp / st

        # Precision:
        #   - support_pred == 0 -> precision no definida (None).
        #   - support_pred > 0  -> tp/sp.
        if sp == 0:
            prec_val = None
        else:
            prec_val = tp / sp

        # F1:
        #   - si support_true == 0 -> None (no tiene sentido sin recall).
        #   - si support_pred == 0 y support_true > 0 -> 0.0 (la clase
        #     existe pero el modelo nunca la predice; F1 colapsa a 0).
        #   - resto: formula estandar 2PR/(P+R), 0 si denominador 0.
        if st == 0:
            f1_val = None
        elif sp == 0:
            f1_val = 0.0
        else:
            denom = prec_val + rec_val
            f1_val = (2.0 * prec_val * rec_val / denom) if denom > 0 else 0.0

        precision.append(float(prec_val) if prec_val is not None else None)
        recall.append(float(rec_val) if rec_val is not None else None)
        f1.append(float(f1_val) if f1_val is not None else None)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support_true": [int(x) for x in support_true.tolist()],
        "support_pred": [int(x) for x in support_pred.tolist()],
    }


# ----------------------------------------------------------------------
# Metricas de regresion (RUL CMAPSS y futuros downstream continuos)
# ----------------------------------------------------------------------


def _to_numpy_float(x: ArrayLike) -> np.ndarray:
    """Convierte tensores/listas a np.float64 manteniendo el valor.

    A diferencia de `_to_numpy`, NO castea a int. Necesario para regresion.
    """
    if _HAS_TORCH and isinstance(x, _torch.Tensor):
        x = x.detach().cpu().numpy()
    arr = np.asarray(x, dtype=np.float64)
    return arr


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Mean Absolute Error: `mean(|y_true - y_pred|)`.

    Si los inputs estan vacios, devuelve 0.0 (semantica "no error
    computable" tratada como cero para no contaminar la metrica
    agregada).
    """
    yt = _to_numpy_float(y_true)
    yp = _to_numpy_float(y_pred)
    if yt.size == 0:
        return 0.0
    if yt.shape != yp.shape:
        raise ValueError(
            f"shapes distintas: y_true {yt.shape} vs y_pred {yp.shape}"
        )
    return float(np.mean(np.abs(yt - yp)))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Error: `sqrt(mean((y_true - y_pred)^2))`."""
    yt = _to_numpy_float(y_true)
    yp = _to_numpy_float(y_pred)
    if yt.size == 0:
        return 0.0
    if yt.shape != yp.shape:
        raise ValueError(
            f"shapes distintas: y_true {yt.shape} vs y_pred {yp.shape}"
        )
    return float(np.sqrt(np.mean((yt - yp) ** 2)))


def r2(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Coeficiente de determinacion `R^2 = 1 - SS_res / SS_tot`.

    Devuelve 0.0 si `y_true` esta vacio. Si `var(y_true) == 0` (todos
    los targets iguales), `SS_tot=0`; en ese caso devolvemos:
      - 1.0 si `SS_res == 0` (prediccion perfecta de la constante);
      - 0.0 en cualquier otro caso (para no inventar valores).
    """
    yt = _to_numpy_float(y_true)
    yp = _to_numpy_float(y_pred)
    if yt.size == 0:
        return 0.0
    if yt.shape != yp.shape:
        raise ValueError(
            f"shapes distintas: y_true {yt.shape} vs y_pred {yp.shape}"
        )
    ss_res = float(np.sum((yt - yp) ** 2))
    ss_tot = float(np.sum((yt - yt.mean()) ** 2))
    if ss_tot == 0.0:
        return 1.0 if ss_res == 0.0 else 0.0
    return 1.0 - ss_res / ss_tot


def cmapss_score(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    alpha_under: float = 13.0,
    alpha_over: float = 10.0,
) -> float:
    """Funcion de score asimetrica de Saxena 2008 para RUL CMAPSS.

    Penaliza mas la sobreestimacion del RUL (predecir vida que no queda
    es operacionalmente peligroso) que la subestimacion (predecir
    mantenimiento antes de tiempo, conservador).

    Formula por sample con `d = y_pred - y_true`:
      - si `d < 0` (subestimacion):  `exp(-d / alpha_under) - 1`,
        con `alpha_under = 13` por convencion (Saxena 2008).
      - si `d >= 0` (sobreestimacion): `exp(d / alpha_over) - 1`,
        con `alpha_over = 10` por convencion.

    El score TOTAL es la **suma** sobre todos los samples (Saxena lo
    define asi). **Menor es mejor**. Devuelve 0.0 si los inputs estan
    vacios.

    Refs:
      Saxena, Goebel, Simon, Eklund. "Damage propagation modeling for
      aircraft engine run-to-failure simulation." PHM 2008.
    """
    if alpha_under <= 0 or alpha_over <= 0:
        raise ValueError(
            f"alpha_under y alpha_over deben ser > 0; "
            f"recibido under={alpha_under}, over={alpha_over}"
        )
    yt = _to_numpy_float(y_true)
    yp = _to_numpy_float(y_pred)
    if yt.size == 0:
        return 0.0
    if yt.shape != yp.shape:
        raise ValueError(
            f"shapes distintas: y_true {yt.shape} vs y_pred {yp.shape}"
        )
    d = yp - yt
    # Para d < 0 (under): exp(-d / alpha_under) - 1 = exp(|d|/alpha_under) - 1
    # Para d >= 0 (over): exp( d / alpha_over)  - 1
    score_under = np.exp(-d[d < 0] / float(alpha_under)) - 1.0
    score_over = np.exp(d[d >= 0] / float(alpha_over)) - 1.0
    return float(score_under.sum() + score_over.sum())


def regression_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    prefix: str = "",
    include_cmapss_score: bool = True,
) -> dict:
    """Agregador: devuelve dict con `mae`, `rmse`, `r2` y opcionalmente
    `cmapss_score`. Todas las claves se prefijan con `prefix` (e.g.
    `"val_"` o `"test_"`) para encajar limpiamente en el JSONL de logs.

    Args:
        y_true, y_pred: 1D arrays/tensores float.
        prefix: string prepended a cada clave del dict de salida.
        include_cmapss_score: si True, incluye `cmapss_score` con los
            alphas Saxena 2008 (13/10). Util desactivarlo en tareas de
            regresion que no son RUL.
    """
    out = {
        f"{prefix}mae": mae(y_true, y_pred),
        f"{prefix}rmse": rmse(y_true, y_pred),
        f"{prefix}r2": r2(y_true, y_pred),
    }
    if include_cmapss_score:
        out[f"{prefix}cmapss_score"] = cmapss_score(y_true, y_pred)
    return out
