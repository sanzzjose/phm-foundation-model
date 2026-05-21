"""Tests de las metricas de regresion en `training.downstream.metrics`.

Cubren:

- `mae`, `rmse`, `r2` con casos perfectos, vacios y todos-iguales.
- `cmapss_score` Saxena 2008 asimetrico (penaliza mas sobreestimar).
- `regression_metrics` agregador con prefijo y flag `include_cmapss_score`.
- Validacion de shapes y argumentos invalidos.

Independientes de torch (las metricas son numpy puro).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from training.downstream.metrics import (
    cmapss_score,
    mae,
    r2,
    regression_metrics,
    rmse,
)


# ----------------------------------------------------------------------
# mae
# ----------------------------------------------------------------------


def test_mae_perfecto():
    """Si y_pred == y_true, mae == 0."""
    yt = np.array([0.0, 10.0, 50.0, 125.0])
    yp = yt.copy()
    assert mae(yt, yp) == 0.0


def test_mae_diferencia_constante():
    """Error constante de +5: mae == 5."""
    yt = np.array([0.0, 10.0, 50.0, 125.0])
    yp = yt + 5.0
    assert mae(yt, yp) == 5.0


def test_mae_simetrico_signo():
    """mae usa |.|: mae(+5) == mae(-5)."""
    yt = np.array([10.0, 20.0])
    assert mae(yt, yt + 5.0) == mae(yt, yt - 5.0) == 5.0


def test_mae_vacio_devuelve_cero():
    assert mae([], []) == 0.0


def test_mae_rechaza_shapes_distintas():
    with pytest.raises(ValueError, match="shapes distintas"):
        mae([1.0, 2.0], [1.0, 2.0, 3.0])


# ----------------------------------------------------------------------
# rmse
# ----------------------------------------------------------------------


def test_rmse_perfecto():
    yt = np.array([0.0, 10.0, 50.0, 125.0])
    assert rmse(yt, yt) == 0.0


def test_rmse_caso_conocido():
    """Errores [3, 4]: rmse = sqrt((9+16)/2) = sqrt(12.5)."""
    yt = np.array([0.0, 0.0])
    yp = np.array([3.0, 4.0])
    assert math.isclose(rmse(yt, yp), math.sqrt(12.5), rel_tol=1e-9)


def test_rmse_mayor_o_igual_que_mae():
    """RMSE >= MAE siempre (desigualdad de potencias)."""
    rng = np.random.default_rng(42)
    yt = rng.normal(size=100)
    yp = rng.normal(size=100)
    assert rmse(yt, yp) >= mae(yt, yp) - 1e-9


def test_rmse_vacio_devuelve_cero():
    assert rmse([], []) == 0.0


# ----------------------------------------------------------------------
# r2
# ----------------------------------------------------------------------


def test_r2_perfecto():
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    assert r2(yt, yt) == 1.0


def test_r2_prediccion_media_es_cero():
    """Si y_pred es la media de y_true, R2 == 0 (SS_res == SS_tot)."""
    yt = np.array([1.0, 2.0, 3.0, 4.0])
    yp = np.full_like(yt, yt.mean())
    # SS_res = SS_tot -> 1 - 1 = 0
    assert math.isclose(r2(yt, yp), 0.0, abs_tol=1e-12)


def test_r2_target_constante_pred_constante_correcta():
    """y_true constante, y_pred = misma constante -> R2 = 1.0."""
    yt = np.full(5, 7.0)
    yp = np.full(5, 7.0)
    assert r2(yt, yp) == 1.0


def test_r2_target_constante_pred_distinta_devuelve_cero():
    """y_true constante pero y_pred != y_true -> R2 = 0.0 (semantica
    explicita; sklearn devuelve nan o -inf segun version)."""
    yt = np.full(5, 7.0)
    yp = np.full(5, 8.0)
    assert r2(yt, yp) == 0.0


def test_r2_vacio_devuelve_cero():
    assert r2([], []) == 0.0


# ----------------------------------------------------------------------
# cmapss_score (Saxena 2008)
# ----------------------------------------------------------------------


def test_cmapss_score_perfecto_es_cero():
    yt = np.array([0.0, 50.0, 100.0])
    assert cmapss_score(yt, yt) == 0.0


def test_cmapss_score_penaliza_mas_sobreestimar():
    """Para el mismo |error|, sobreestimar (d>0) penaliza mas que
    subestimar (d<0). Saxena 2008 alpha_over=10 < alpha_under=13.

    Para d=+5: exp(5/10) - 1 ≈ 0.6487.
    Para d=-5: exp(5/13) - 1 ≈ 0.4685.
    El primero es mayor: el modelo recibe mas penalty por exagerar
    cuanta vida queda (operacionalmente peligroso).
    """
    yt = np.array([50.0])
    yp_over = np.array([55.0])   # d=+5 (sobreestima vida util)
    yp_under = np.array([45.0])  # d=-5 (subestima)
    s_over = cmapss_score(yt, yp_over)
    s_under = cmapss_score(yt, yp_under)
    assert s_over > s_under > 0.0
    # Coincidencia bit-a-bit con la formula analitica.
    expected_over = math.exp(5.0 / 10.0) - 1.0
    expected_under = math.exp(5.0 / 13.0) - 1.0
    assert math.isclose(s_over, expected_over, rel_tol=1e-9)
    assert math.isclose(s_under, expected_under, rel_tol=1e-9)


def test_cmapss_score_es_suma_no_media():
    """El score total Saxena 2008 es la SUMA sobre samples, no la media.

    Dos samples con d=+5 cada uno deben dar 2 * (exp(0.5) - 1).
    """
    yt = np.array([50.0, 70.0])
    yp = np.array([55.0, 75.0])
    s = cmapss_score(yt, yp)
    expected = 2.0 * (math.exp(0.5) - 1.0)
    assert math.isclose(s, expected, rel_tol=1e-9)


def test_cmapss_score_alphas_custom():
    """Permite alphas distintos del default Saxena 2008."""
    yt = np.array([50.0])
    yp = np.array([55.0])  # d=+5
    # alpha_over=20 (penaliza menos): score menor.
    s_loose = cmapss_score(yt, yp, alpha_under=13.0, alpha_over=20.0)
    s_default = cmapss_score(yt, yp)
    assert s_loose < s_default


def test_cmapss_score_vacio_devuelve_cero():
    assert cmapss_score([], []) == 0.0


def test_cmapss_score_rechaza_alphas_no_positivos():
    yt = np.array([50.0])
    yp = np.array([55.0])
    with pytest.raises(ValueError, match="alpha"):
        cmapss_score(yt, yp, alpha_under=0.0)
    with pytest.raises(ValueError, match="alpha"):
        cmapss_score(yt, yp, alpha_over=-1.0)


# ----------------------------------------------------------------------
# regression_metrics (agregador)
# ----------------------------------------------------------------------


def test_regression_metrics_keys_default():
    """Por defecto incluye mae, rmse, r2 y cmapss_score sin prefijo."""
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(yt, yp)
    assert set(m.keys()) == {"mae", "rmse", "r2", "cmapss_score"}
    assert m["mae"] == 0.0
    assert m["rmse"] == 0.0
    assert m["r2"] == 1.0
    assert m["cmapss_score"] == 0.0


def test_regression_metrics_con_prefix():
    """Con prefix, todas las claves vienen prefijadas."""
    yt = np.array([1.0, 2.0, 3.0])
    yp = np.array([1.0, 2.0, 3.0])
    m = regression_metrics(yt, yp, prefix="val_")
    assert set(m.keys()) == {"val_mae", "val_rmse", "val_r2", "val_cmapss_score"}


def test_regression_metrics_sin_cmapss_score():
    """`include_cmapss_score=False` deja solo las metricas generales."""
    yt = np.array([1.0, 2.0])
    yp = np.array([1.5, 2.5])
    m = regression_metrics(yt, yp, include_cmapss_score=False)
    assert set(m.keys()) == {"mae", "rmse", "r2"}
    assert "cmapss_score" not in m


def test_regression_metrics_acepta_torch_si_disponible():
    """Si torch esta instalado, las metricas aceptan tensores.

    Skip defensivo por plataforma: en Windows local nuestra instalacion
    de torch tiene ABI incompat con NumPy 2 (al construir tensores
    .numpy() en `_to_numpy_float`). En Colab Linux corre PASS.
    """
    import sys
    if sys.platform == "win32":
        pytest.skip("Skip Windows local: torch+numpy2 ABI; PASS en Colab.")
    try:
        import torch
    except Exception as exc:
        pytest.skip(f"torch no disponible aqui: {exc}")
    yt = torch.tensor([0.0, 10.0, 50.0], dtype=torch.float32)
    yp = torch.tensor([1.0, 9.0, 51.0], dtype=torch.float32)
    m = regression_metrics(yt, yp)
    # mae = (1 + 1 + 1) / 3 = 1.0
    assert math.isclose(m["mae"], 1.0, rel_tol=1e-6)
