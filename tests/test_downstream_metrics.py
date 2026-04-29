"""Tests de las metricas de clasificacion downstream (puro numpy).

No requieren torch. Sklearn opcional; cuando esta disponible, se usa para
verificar valores cruzados contra `sklearn.metrics`.
"""

from __future__ import annotations

import numpy as np
import pytest

from training.downstream.metrics import (
    accuracy,
    balanced_accuracy,
    confusion_matrix,
    macro_f1,
    per_class_precision_recall_f1,
    support_per_class,
)


# ----------------------------------------------------------------------
# accuracy
# ----------------------------------------------------------------------


def test_accuracy_perfect():
    assert accuracy([0, 1, 2], [0, 1, 2]) == 1.0


def test_accuracy_zero():
    assert accuracy([0, 1, 2], [1, 2, 0]) == 0.0


def test_accuracy_half():
    assert accuracy([0, 1, 2, 3], [0, 1, 9, 9]) == pytest.approx(0.5)


def test_accuracy_empty():
    assert accuracy([], []) == 0.0


# ----------------------------------------------------------------------
# confusion_matrix
# ----------------------------------------------------------------------


def test_confusion_matrix_basic():
    cm = confusion_matrix([0, 0, 1, 1, 2], [0, 1, 1, 1, 2], n_classes=3)
    assert cm.shape == (3, 3)
    assert cm.tolist() == [
        [1, 1, 0],  # true=0: 1 acertado, 1 mal a clase 1
        [0, 2, 0],  # true=1: 2 acertados
        [0, 0, 1],  # true=2: 1 acertado
    ]


def test_confusion_matrix_inferred_n_classes():
    cm = confusion_matrix([0, 1, 2], [0, 1, 2])
    assert cm.shape == (3, 3)


def test_confusion_matrix_rejects_out_of_range():
    with pytest.raises(ValueError):
        confusion_matrix([0, 1, 5], [0, 1, 0], n_classes=3)


# ----------------------------------------------------------------------
# balanced_accuracy
# ----------------------------------------------------------------------


def test_balanced_accuracy_perfect():
    assert balanced_accuracy([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == 1.0


def test_balanced_accuracy_handles_imbalance():
    # Imbalance heavy: 9 de clase 0, 1 de clase 1. Predice todo clase 0.
    # accuracy = 0.9, balanced_accuracy = 0.5 (recall_0=1, recall_1=0).
    y_true = [0] * 9 + [1]
    y_pred = [0] * 10
    assert accuracy(y_true, y_pred) == pytest.approx(0.9)
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(0.5)


def test_balanced_accuracy_empty():
    assert balanced_accuracy([], []) == 0.0


# ----------------------------------------------------------------------
# macro_f1
# ----------------------------------------------------------------------


def test_macro_f1_perfect():
    assert macro_f1([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]) == 1.0


def test_macro_f1_zero():
    # Todas predicciones equivocadas a clase 0
    assert macro_f1([1, 1, 2, 2], [0, 0, 0, 0]) == 0.0


def test_macro_f1_caso_conocido():
    # 4 clases, distribucion balanceada. Predice perfecto excepto 1 error.
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 0]  # ultimo 3 predicho como 0
    # Confusion matrix por clase:
    # clase 0: tp=2 (true=0 y pred=0), col_0=3 (pred=0 en pos 0,4,7), row_0=2
    #          P=2/3, R=2/2=1, F1=2*(2/3)*1/(2/3+1) = 4/5 = 0.8
    # clase 1: tp=2, P=1, R=1, F1=1
    # clase 2: tp=2, P=1, R=1, F1=1
    # clase 3: tp=1 (pos 3), fn=1 (pos 7 -> pred 0), row_3=2, col_3=1
    #          P=1/1=1, R=1/2=0.5, F1=2*1*0.5/1.5 = 2/3 = 0.6667
    # macro = (0.8 + 1 + 1 + 0.6667) / 4 = 0.8667
    expected = (0.8 + 1.0 + 1.0 + 2.0 / 3.0) / 4
    assert macro_f1(y_true, y_pred) == pytest.approx(expected, rel=1e-4)


# ----------------------------------------------------------------------
# Soporte por clase
# ----------------------------------------------------------------------


def test_support_per_class():
    counts = support_per_class([0, 0, 1, 2, 2, 2], n_classes=4)
    assert counts.tolist() == [2, 1, 3, 0]


# ----------------------------------------------------------------------
# per_class_precision_recall_f1
# ----------------------------------------------------------------------


def test_per_class_perfecto():
    """Prediccion perfecta: precision=recall=F1=1 para todas las clases."""
    out = per_class_precision_recall_f1([0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2], n_classes=3)
    assert out["precision"] == [1.0, 1.0, 1.0]
    assert out["recall"] == [1.0, 1.0, 1.0]
    assert out["f1"] == [1.0, 1.0, 1.0]
    assert out["support_true"] == [2, 2, 2]
    assert out["support_pred"] == [2, 2, 2]


def test_per_class_clase_ausente_en_y_true():
    """Clase con support_true=0: recall=None, F1=None, precision=0 si se predijo, None si no."""
    # n_classes=4, y_true solo tiene clases 0,1,2. Clase 3 ausente.
    # y_pred ALGUNAS veces predice clase 3 (FP).
    y_true = [0, 1, 2, 0, 1]
    y_pred = [0, 1, 3, 0, 1]
    out = per_class_precision_recall_f1(y_true, y_pred, n_classes=4)
    # Clase 3: support_true=0, support_pred=1 (un FP).
    assert out["support_true"][3] == 0
    assert out["support_pred"][3] == 1
    assert out["recall"][3] is None
    assert out["f1"][3] is None
    assert out["precision"][3] == 0.0  # se predijo, todos FP


def test_per_class_clase_nunca_predicha():
    """Clase con support_pred=0 pero support_true>0: precision=None, F1=0, recall=0."""
    # y_true contiene clase 2; y_pred nunca la predice.
    y_true = [0, 1, 2, 2]
    y_pred = [0, 1, 0, 1]
    out = per_class_precision_recall_f1(y_true, y_pred, n_classes=3)
    assert out["support_true"][2] == 2
    assert out["support_pred"][2] == 0
    assert out["recall"][2] == 0.0
    assert out["precision"][2] is None  # nunca predicha
    assert out["f1"][2] == 0.0  # F1 colapsa a 0


def test_per_class_clase_completamente_ausente():
    """Clase con support_true=0 Y support_pred=0: todo None."""
    y_true = [0, 1, 0, 1]
    y_pred = [0, 1, 0, 1]
    out = per_class_precision_recall_f1(y_true, y_pred, n_classes=4)
    # Clases 2 y 3 ausentes en y_true y nunca predichas
    for c in (2, 3):
        assert out["support_true"][c] == 0
        assert out["support_pred"][c] == 0
        assert out["precision"][c] is None
        assert out["recall"][c] is None
        assert out["f1"][c] is None


def test_per_class_empty():
    """Entradas vacias: todo None y soportes 0."""
    out = per_class_precision_recall_f1([], [], n_classes=3)
    assert out["support_true"] == [0, 0, 0]
    assert out["support_pred"] == [0, 0, 0]
    assert out["precision"] == [None, None, None]
    assert out["recall"] == [None, None, None]
    assert out["f1"] == [None, None, None]


def test_per_class_caso_conocido():
    """Mismo escenario que test_macro_f1_caso_conocido."""
    y_true = [0, 1, 2, 3, 0, 1, 2, 3]
    y_pred = [0, 1, 2, 3, 0, 1, 2, 0]
    out = per_class_precision_recall_f1(y_true, y_pred, n_classes=4)
    # clase 0: tp=2, sp=3 (idx 0,4,7), st=2 -> P=2/3, R=1, F1=0.8
    assert out["precision"][0] == pytest.approx(2 / 3)
    assert out["recall"][0] == 1.0
    assert out["f1"][0] == pytest.approx(0.8)
    # clase 1: perfecto
    assert out["precision"][1] == 1.0
    assert out["f1"][1] == 1.0
    # clase 2: perfecto
    assert out["precision"][2] == 1.0
    assert out["f1"][2] == 1.0
    # clase 3: tp=1, sp=1, st=2 -> P=1, R=0.5, F1=2/3
    assert out["precision"][3] == 1.0
    assert out["recall"][3] == 0.5
    assert out["f1"][3] == pytest.approx(2.0 / 3.0, rel=1e-4)


# ----------------------------------------------------------------------
# Verificacion cruzada con sklearn (si disponible)
# ----------------------------------------------------------------------


def test_cross_check_with_sklearn():
    skl = pytest.importorskip("sklearn.metrics")
    np.random.seed(0)
    y_true = np.random.randint(0, 5, size=200)
    y_pred = np.random.randint(0, 5, size=200)
    assert accuracy(y_true, y_pred) == pytest.approx(
        skl.accuracy_score(y_true, y_pred), rel=1e-6
    )
    assert balanced_accuracy(y_true, y_pred) == pytest.approx(
        skl.balanced_accuracy_score(y_true, y_pred), rel=1e-6
    )
    assert macro_f1(y_true, y_pred) == pytest.approx(
        skl.f1_score(y_true, y_pred, average="macro", zero_division=0),
        rel=1e-6,
    )
