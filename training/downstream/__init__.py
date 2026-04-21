"""Modulos para evaluacion downstream del encoder SSL.

Contiene:

- `pooling`:  reducciones tokens (B, C, N, d) -> (B, d) respetando
  `valid_patch_mask` y `canales_constantes_mask` (channel-independent).
- `heads`:    cabezas downstream (clasificacion por ahora; RUL/anomalia
  vendran cuando se aborde).
- `metrics`:  metricas estandar de clasificacion (accuracy,
  balanced_accuracy, macro_f1, confusion_matrix). En numpy puro sin
  requerir sklearn, con fallback opcional.

IMPORTANTE sobre los imports:

- Los modulos `pooling` y `heads` requieren `torch`.
- El modulo `metrics` solo requiere `numpy`.

Para que los tests de `metrics` puedan ejecutarse en entornos sin torch
(CI/local sin GPU), NO importamos pooling/heads al cargar este paquete.
Los consumidores deben importar explicitamente desde el submodulo, p.ej.:

    from training.downstream.metrics import accuracy
    from training.downstream.pooling import pooled_embedding      # requiere torch
    from training.downstream.heads import DownstreamClassifier    # requiere torch
"""
