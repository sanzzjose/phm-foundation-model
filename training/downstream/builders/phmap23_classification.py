"""Builder adaptador para PHMAP23 (clasificación multiclase de fallo).

PHMAP23 es un TRANSFER_TARGET secundario, dominio asignado `gearboxes`
en el audit (las señales reales `P1..P7 + spacecraft` apuntan al reto de
propulsión PHM Asia-Pacific 2023). Interesa al Probe Suite v1 por
ampliar el conjunto a un dominio mecánico no explotado.

**Estado: ready (verificado contra shards reales, 2026-06-05).**

Recorrido de la decisión (tres pasadas):

1. audit-based: promovido a `ready` (audit decía etiqueta_unidad, 25
   clases). 2. manifest-based: REVERTIDO a `needs_semantic_review` porque
   el manifest declaraba `target_policy=ultimo_valor_valido` y no exponía
   clases por split. 3. shards-based (esta): se leyeron los **labels
   reales** de los 3 splits con `training.phm_tar_reader` (708 ventanas
   totales) y la semántica queda confirmada.

Hallazgos de la lectura de shards (ver
`docs/experiments/TARGET_SEMANTIC_REVIEW_PHMAP23_SHARDS.md`):

* el target real es ``target['target_window']``, derivado de la columna
  ``fault`` con ``ultimo_valor_valido``: valores **discretos 0..24** (25
  clases). Candidato único;
* **constante por trayectoria**: 0/177 unidades con target no constante.
  Por tanto ``ultimo_valor_valido`` es equivalente a una etiqueta
  estática de unidad (``unit_label``);
* **cobertura de clases**: ``train`` cubre TODAS las clases presentes en
  ``val`` y ``test`` (``val_minus_train=[]``, ``test_minus_train=[]``,
  unión = 25 clases);
* anti-leakage en `pass` (sin trajectory_id compartido entre splits).

El mapeo string/float -> int se deriva en tiempo de entrenamiento
ordenando las clases observadas en ``train`` (idéntico a CWRU/HSG18, ya
`ready`). No se inventa target.

**Caveats de DATOS (no de semántica), para la memoria:**

* **desbalance fuerte**: la clase 0 es el 59.3% (140/236 ventanas por
  split); las clases 1..24 tienen 4 ventanas cada una en cada split (en
  ``train``, exactamente 1 trayectoria por clase minoritaria). Reportar
  ``macro_f1`` y ``balanced_accuracy``, NUNCA accuracy;
* **bajo volumen**: 236 ventanas por split, 25 clases;
* ``padding_ratio=0.1636`` (warning ``tail_policy_pad_padding_moderado``,
  aceptado por el contrato de máscaras);
* las tres particiones tienen distribución de clases idéntica (cada una
  con la misma composición de fallos), un caso favorable para cobertura
  pero exigente por el desbalance.
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="PHMAP23",
    role="TRANSFER_TARGET",
    task_type="classification",
    primary_metric="macro_f1",
    secondary_metrics=["balanced_accuracy", "accuracy"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=False,
    split_policy="by_trajectory",
    target_definition=(
        "Clasificación multiclase de modo de fallo. Target = "
        "target['target_window'] (último valor válido de la columna "
        "`fault`): 25 clases discretas 0..24, candidato único. Verificado "
        "en shards: constante por trayectoria (equivale a etiqueta estática "
        "de unidad). Mapeo float->int por orden de las clases de train."
    ),
    target_policy="unit_label",
    caveat=(
        "Verificado leyendo labels reales de los 3 splits (708 ventanas) con "
        "phm_tar_reader: fault discreto 0..24 (25 clases), constante por "
        "trayectoria, train cubre todas las clases de val/test, anti_leakage "
        "pass. La política del manifest es ultimo_valor_valido pero se "
        "verifica equivalente a unit_label. DESBALANCE FUERTE: clase 0 = "
        "59.3% (140/236 por split), clases 1..24 = 4 ventanas cada una (1 "
        "trayectoria/clase en train). Reportar macro_f1 y balanced_accuracy, "
        "no accuracy. Bajo volumen (236 ventanas/split), padding_ratio 0.1636."
    ),
    source_artifacts=["processed/PHMAP23"],
    status="ready",
))
