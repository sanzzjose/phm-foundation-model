# Resultados de downstream classification

Resumen agregado de las evaluaciones downstream supervised sobre los
`TRANSFER_TARGET` primary del MVP, usando como backbone:

- el encoder SSL **central full** (`ckpt_step100000.pt`, 100k steps,
  36 `PRETRAIN_SOURCE`); y
- el encoder SSL **federado pilot** (`ssl_federated_pilot/ckpt_final.pt`,
  10 rondas × 50 local steps × 10 clientes), evaluación parcial en
  CWRU y HSG18; veredicto **CONDITIONAL**. Detalle completo en
  `results/downstream/fl_pilot_vs_central/README.md`.

## Tabla resumen

Cada celda es `test_macro_f1` (mejor por dataset en **negrita**).

| dataset | dominio | C | n_windows | n_clases | from_scratch | linear_probing | full_ft (lr 1e-5) | mejor |
|---------|---------|--:|----------:|---------:|-------------:|---------------:|------------------:|-------|
| CWRU    | bearings | 2 | 140 134 | 4 | 0.3503 | 0.7046 | **0.8292** | full_ft |
| HSG18   | hdd      | 1 |  38 896 | 2 | 0.5693 | 0.9056 | **0.9504** | full_ft |
| PBCP16  | small    | 1 |   1 975 | 5 | **0.9074** | 0.8287 | 0.8891 | from_scratch |
| PHM18   | wind     | 22 |  3 621 | 3 | **0.3655** | 0.2739 | 0.3406 | from_scratch (todos ≈0.35) |

Métricas complementarias (`balanced_accuracy` y `accuracy`) en
`<dataset>/<mode>/run_info.json`. Curvas completas en
`MyDrive/fm_fl_phmd/logs/downstream/<dataset>/<mode>/metrics.jsonl`.

## Lectura

### CWRU y HSG18: hipótesis principal del TFM confirmada

Patrón idéntico en ambos:

```
from_scratch  <<  linear_probing  <  full_finetuning (lr_backbone=1e-5)
```

- CWRU: `from_scratch=0.35` → `linear=0.70` (+35 pp) → `full_ft=0.83` (+12 pp).
- HSG18: `from_scratch=0.57` → `linear=0.91` (+34 pp) → `full_ft=0.95` (+5 pp).

Que el patrón se replique en dos dominios físicos distintos (bearings
de CWRU vs hdd de HSG18) refuerza la narrativa: el SSL multi-dataset
**aprende representaciones temporales reutilizables**, no solo memoriza
patrones de un dominio concreto.

El sweet spot `lr_backbone = lr_head/100 = 1e-5` se valida también: en
HSG18 supera al linear probing (+5 pp), igual que en CWRU.

### PBCP16: el SSL no transfiere

`from_scratch (0.9074) > full_ft (0.8891) > linear (0.8287)`. En este
setting, **from_scratch gana**; el SSL es ligeramente inferior, no
disruptivamente.

Caracterización: dataset muy pequeño (2 k ventanas total, ~474 train),
5 clases perfectamente balanceadas (237 cada una), 1 canal. No
disponemos de las curvas train/val episode-a-episode en este informe
para afirmar con certeza que `from_scratch` sobreajusta el training; lo
que sí observamos es que su `best_epoch=20/20` (último), sugiriendo que
quedaba margen sin saturar el `metric_for_best`.

**Hallazgo a documentar honestamente**: el SSL no garantiza mejora
universal. En datasets pequeños con tarea bien definida y baja
diversidad de dominios, el sesgo aprendido durante el pretraining
puede no aportar (o restar muy poco). Es exactamente el tipo de matiz
que conviene discutir en la memoria sin sobreinterpretar.

### PHM18: todos los modos por debajo de 0.40

`from_scratch=0.366`, `full_ft=0.341`, `linear=0.274`. `bal_acc < 0.4`
en todos.

**Diagnóstico v0.1, no conclusión fuerte**. Tres observaciones:

1. **Tamaño efectivo bajo**: PHM18 tiene 1 207 unidades pero solo
   3 621 ventanas totales (~3 ventanas por unidad). Audit v2.3 emite
   `tail_policy_pad_padding_moderado` para este dataset.
2. **Batch no capeado en v0.1**: las configs declaraban
   `batch_size_policy=adaptive_by_channels`, pero el trainer v0.1 lo
   ignoraba. PHM18 con C=22 corrió con `batch=64` (`B*C=1408 > cap=512`),
   lo cual NO degrada la métrica pero introduce un punto de inconsistencia
   respecto a CWRU/HSG18 que sí cumplían el cap. **Tras el fix del
   trainer (2026-05-24)**, una nueva corrida usaría `B_eff=23`.
3. **Posible problema de formulación**: target `fault` con
   distribución 1596/708/297 (3 clases). No hemos validado que ese
   target sea el adecuado para esta tarea.

Con la formulación y `W=512` actuales, **todos los modos son bajos**;
no podemos atribuir el problema únicamente al SSL ni al dataset. Vías
de análisis pendientes:

- Rerunear con el batch adaptativo activo y comparar métricas finales.
- Ablation con `W=128` o `W=256` para aumentar ventanas/unidad (sec
  12 del `CLAUDE.md`).
- Revisar candidatos de target.

Mientras tanto, PHM18 v0.1 queda marcado como
`historical_uncapped_batch_v0_1` en el agregado de resultados.

### Cautela sobre clases ausentes en test

Cuando una clase está en `label_mapping` (vista en train) pero no
aparece en test, las métricas agregadas (`balanced_accuracy`,
`macro_f1`) ignoran esa clase silenciosamente (regla equivalente a
`zero_division=None` en sklearn). Es una decisión metodológica
defendible pero hay que reportarla. Desde el fix de 2026-05-24, cada
`run_info.json` registra `zero_support_classes_test`. Para los 4 TT
primary del MVP, las clases ausentes en test se listarán en el
agregado `summary_classification_primary.json`.

## Configuración común

- Backbone: `patchtst_phm_base` (d_model=128, 4 layers, 4 heads,
  d_ff=512). **801 808 parámetros**.
- Optimizer: AdamW, weight_decay=0.01.
- Schedule: 20 épocas, batch=64.
- `metric_for_best = macro_f1_val`. Best ckpt cargado para evaluar
  test.
- AMP auto en CUDA. GradScaler tolerante a overflow fp16.
- `lr_head = 1e-3` en todos los modos.
- `lr_backbone = 1e-4` en `from_scratch` (config base CWRU).
- `lr_backbone = 1e-5` en `full_finetuning` (sweet spot detectado en
  CWRU; ratio `lr_head/100`).
- `linear_probing`: backbone congelado, `lr_backbone` no aplica.

### Política de batch_size: histórica vs post-fix

Las 4 corridas de CWRU y las 3 nuevas (HSG18, PBCP16, PHM18) **v0.1
se ejecutaron con `batch_size=64` fijo**, no adaptativo. Hasta C=2
(CWRU, PBCP16, HSG18) eso da `B*C ≤ 128` y no hay riesgo de VRAM. Pero
PHM18 tiene C=22, por lo que `B*C = 1408 > max_channel_batch=512`. Los
manifests de PHM18 v0.1 lo registran como `batch_size_effective=64`
porque el trainer ignoraba `batch_size_policy=adaptive_by_channels`.

**Fix aplicado el 2026-05-24** (commit posterior a `c32ce23`):
`train_downstream_classification.py` ahora respeta
`batch_size_policy=adaptive_by_channels` via `compute_adaptive_batch_size`
y registra `batch_size_effective`, `effective_bc`, `n_channels`,
`n_channels_source` en `run_info.json` y `metrics.jsonl`.

Si se va a **citar PHM18 como evidencia metodológica** del TFM, hay
que reproducirlo tras el fix (PHM18 con `B_eff = 512 // 22 = 23`).
Mientras tanto la fila de PHM18 en la tabla queda etiquetada como
`historical_uncapped_batch_v0_1` en `summary_classification_primary.json`.

## Reproducibilidad

Cada `<dataset>/<mode>/run_info.json` registra:

- `config_hash` (hash del YAML completo).
- `git_hash` (commit del trainer).
- `label_mapping` (cómo se mapean clases string → int desde train).
- `n_classes`, `n_trainable_params`.
- `best_epoch`, `best_value` (mejor `macro_f1_val`).
- `test_metrics` (acc, balanced_accuracy, macro_f1, confusion_matrix
  y, desde el fix de 2026-05-24, `per_class` con precision/recall/f1
  y `support_true`/`support_pred` por clase).
- `zero_support_classes_test` (lista de etiquetas que no aparecen en
  test; útil para detectar splits donde alguna clase global no se
  evalúa).
- `batch_size_requested`, `batch_size_effective`, `batch_size_policy`,
  `n_channels`, `effective_bc` (desde el fix).
- `amp_used`, `amp_nonfinite_grad_steps` (desde el fix).
- `elapsed_seconds`.

`run_info.json` **no** guarda `y_true`/`y_pred` por defecto. Si se
quiere persistir las predicciones para análisis posterior, hay que
añadir `evaluation: { save_predictions: true }` en el YAML; entonces
se escribe `predictions_test.json` aparte.

Para reejecutar:

```bash
# Drive ya tiene el SSL ckpt; el config esta versionado en repo
python -m training.train_downstream_classification \
  --config <copiar config.yaml de results/downstream/<dataset>/<mode>/> \
  --mode <mode> \
  --checkpoint /content/drive/MyDrive/fm_fl_phmd/checkpoints/ssl_central_full/ssl_central_full_patchtst_phm/ckpt_step100000.pt
```

## Pendientes

- **CALCE_CS2 e IEEE14**: TT primary pero target RUL (`rul`, `rul35`).
  Esperan a tener cabeza RUL en `training/downstream/heads.py`.
- **CMAPSS**: TT primary RUL pero bloqueado además por la semántica
  pendiente del target (ver `docs/decisions/pending_downstream_and_sampling.md`
  sec 1).
- **TT secondary**: CNCMILL18, PHMAP23, CBM14, PHME20. Pendientes para
  ampliación. PHMAP23 tiene `target_col=fault` (classification);
  CNCMILL18, CBM14 y PHME20 requieren verificar tipo de target.
- **Pretraining federado**: pendiente, plan en sec 4 del pending doc.
- **Investigación PHM18**: por qué el dataset es intrínsecamente
  difícil; posible ablation con `W` reducido (128 o 256).

## Timeline

| fecha | commit | hito |
|-------|--------|------|
| 2026-05-23 | b143244 | feat downstream: módulo + trainer + tests |
| 2026-05-23 | 158dee0 | feat config reintento full_ft con `lr_backbone=1e-5` |
| 2026-05-23 | 7fe07c7 | results CWRU 3 modos originales |
| 2026-05-23 | eb52434 | fix UserWarning de `float(loss)` |
| 2026-05-23 | c223a5f | results CWRU reintento `full_ft lr1e-5` |
| 2026-05-23 | c0d2636 | feat notebook `run_downstream_5tt` |
| 2026-05-23 | 06de36e | fix pre-check robusto + git config |
| 2026-05-24 | 3131584 | results 3 TT primary (HSG18, PBCP16, PHM18) |
