"""Builder adaptador para CALCE_CS2 (RUL en baterías).

CALCE_CS2 es un TRANSFER_TARGET importante para el Probe Suite v1
porque introduce un segundo objetivo de regresión distinto de CMAPSS,
en dominio baterías (11 canales, 6 571 ventanas según el audit v2.3).

**Estado actual: needs_semantic_review.**

La revisión semántica
(``docs/experiments/TARGET_SEMANTIC_REVIEW_CALCE_PHMAP23.md``), verificada
contra el **manifest real** de ``processed/CALCE_CS2/manifest.json`` (run
``20260521T084107Z``, ``pipeline_config_hash=1b63d9b7b912e7f3``), corrige
un error de tipado del builder anterior: la armonización v0.5 **no
produce ningún target SOH/capacity**. El único candidato de target es
``rul``:

* ``target_col='rul'``, ``target_candidates=['rul']``, tipo ``rul``;
* política ``ultimo_valor_valido``; ``target_warning=null``;
* ``Charge_Capacity(Ah)``, ``Discharge_Capacity(Ah)`` y ``soc`` están en
  ``signal_cols`` (son FEATURES, no targets): no hay candidato SOH;
* rango ``[0.0, 1003.0]`` según el audit (sin negativos, a diferencia de
  CMAPSS).

Por eso el ``task_type`` pasa de ``soh`` a ``rul``: es la realidad de lo
armonizado. **El status sigue bloqueado** porque la semántica *física*
del RUL no está confirmada. El manifest real lo refuerza:

* ``time_col=null`` y ``order_info.ordered_by_unit_and_cycle=false``: NO
  hay columna de ciclo/tiempo, el orden es por ``_source_row_order``. Sin
  ciclo no se puede confirmar monotonía del RUL ni la convención de EOL;
* ¿``rul`` es ciclos-a-EOL por celda? ¿qué fija el cero? ¿el ``1003`` es
  cap o el ciclo real de la celda más longeva?;
* solo 6 celdas (``n_units_por_split = {train:4, val:1, test:1}``): 1
  celda en val y 1 en test, base muy fina para evaluación.

La regla del proyecto (CLAUDE.md sec. 11) es explícita: no construir RUL
si falta ciclo/EOL claro y no hacer clamp ni transformación silenciosa.
Mientras esas preguntas no se respondan (manifest insuficiente: haría
falta el raw con su columna de ciclo), este builder NO genera target y
deja ``target_definition=None``. El probe suite NO debe lanzar
entrenamiento sobre esta tarea mientras el status sea
``needs_semantic_review``.
"""
from __future__ import annotations

from training.downstream.task_registry import TaskSpec, register_task


SPEC = register_task(TaskSpec(
    dataset="CALCE_CS2",
    role="TRANSFER_TARGET",
    task_type="rul",
    primary_metric="rmse",
    secondary_metrics=["mae", "r2"],
    supports_linear_probing=True,
    supports_full_finetuning=True,
    supports_anomaly=True,
    split_policy="by_cell",
    target_definition=None,
    target_policy=None,
    caveat=(
        "Verificado contra manifest real (2026-06-05): target_col=rul "
        "(candidato único), policy ultimo_valor_valido, target_warning=null. "
        "Capacity/SOH son signal_cols (features), no target. time_col=null y "
        "ordered_by_unit_and_cycle=false: sin columna de ciclo no se puede "
        "confirmar monotonía/EOL del RUL. Solo 6 celdas (split 4/1/1). "
        "Pendiente raw con ciclo antes de generar target. No clamp ni "
        "transformación silenciosa."
    ),
    source_artifacts=["processed/CALCE_CS2"],
    status="needs_semantic_review",
))
