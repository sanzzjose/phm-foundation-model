# Downstream FL pilot vs Central — CWRU + HSG18

Evaluación downstream del ckpt FL pilot (5 000 optimizer steps,
`ssl_federated_pilot_patchtst_phm/ckpt_final.pt`,
`pipeline=final_client_weight_capped_v23`) sobre los 2 TT primary
classification donde el SSL central confirmó transferencia clara.

**Decisión post-corrida**: **CONDITIONAL** (ver criterios al final).
**Decisión post-ablación HSG18 lr1e-4 (2026-05-25)**: **CONDITIONAL reforzado a NO-GO para HSG18**; hipótesis B (estructural) confirmada; ver sección "Ablación lr_backbone=1e-4 confirma hipótesis B".

- 4 corridas en Colab Pro+ A100-SXM4-80GB (2 datasets × 2 modos SSL).
- Tiempo total: ~2 h 14 min (CWRU ~1 h 47 min; HSG18 ~28 min).
- 1 corrida adicional de ablación HSG18 `full_finetuning lr_backbone=1e-4` (~14 min A100, commit `250868f`).
- Trainer `e9fb202` (4 corridas base) + `250868f` (ablación lr1e-4). Configs `c70179f` + `250868f` (versionados).

## Tabla central vs federado (macro_f1 test)

| dataset | from_scratch | central_linear | central_full_1e-5 | **fed_linear** | **fed_full_1e-5** | ratio fed/central full |
|---|---:|---:|---:|---:|---:|---:|
| CWRU  | 0.3503 | 0.7046 | **0.8292** | **0.4456** | **0.6635** | **80.0 %** |
| HSG18 | 0.5693 | 0.9056 | **0.9504** | **0.6080** | **0.5547** | **58.4 %** |

### Tabla HSG18 expandida con ablación lr1e-4 (3 modos FL)

| modo FL | macro_f1 | bal_acc | acc | best_epoch | recall clase 0 | recall clase 1 | elapsed |
|---|---:|---:|---:|---:|---:|---:|---:|
| `fed_linear`        | **0.6080** | 0.6370 | 0.6370 | 7/20  | 36.5 % | 90.9 % | 777.1 s |
| `fed_full_lr1e-5`   | 0.5547     | 0.6176 | 0.6176 | 10/20 | 24.2 % | 99.3 % | 873.3 s |
| `fed_full_lr1e-4`   | **0.3333** (colapso) | 0.5000 | 0.5000 | **1/20** | **0.0 %** | **100.0 %** | 852.7 s |

- `fed_linear` es el **mejor de los tres** en HSG18: backbone congelado obliga a la cabeza a explotar las pocas dimensiones discriminativas del embedding.
- `fed_full_lr1e-5` empeora respecto a linear y colapsa a la mayoritaria (recall clase 0 = 24 %).
- `fed_full_lr1e-4` **colapsa degeneradamente** en la primera época: predice siempre clase 1. Más LR no rescata, lo acelera al colapso. Ver siguiente sección.

## Deltas

### fed vs from_scratch (¿transfiere el FL?)

| dataset | linear | full_lr1e-5 | conclusión |
|---|---:|---:|---|
| CWRU  | +0.0953 | **+0.3132** | FL transfiere; full duplica el delta linear |
| HSG18 | +0.0387 | **−0.0146** | linear marginal; **full destruye señal** |

### fed vs central (¿cuánto pierde el FL?)

| dataset | linear | full_lr1e-5 |
|---|---:|---:|
| CWRU  | −0.2590 | −0.1657 |
| HSG18 | −0.2976 | **−0.3957** |

## Diagnóstico crítico — HSG18 full_ft colapsa a la clase mayoritaria

Lectura clave del `run_info.json` de `hsg18/full_finetuning_lr1e-5`:

```
confusion_matrix:    [[553, 1735],
                      [ 15, 2273]]
recall  clase 0: 24.2 %    (predice solo 553 / 2 288)
recall  clase 1: 99.3 %    (predice 2 273 / 2 288)
```

El modelo aprende a **predecir clase 1 casi siempre**. Con
`lr_backbone=1e-5` el backbone FL apenas se mueve (recordemos: la
norma del global state FL durante el pilot solo varió un 0.06 %), y
la cabeza absorbe el desbalance natural del train (clase 1 = 18 304
vs clase 0 = 9 152, ratio 2:1).

En cambio, **`linear_probing` HSG18 sí balancea** (recall clase 0:
36.5 %; clase 1: 90.9 %): el embedding FL congelado obliga a la
cabeza lineal a explotar las pocas dimensiones discriminativas que
tenga, en lugar de colapsar.

**Implicación práctica**: el SSL FL pilot **no proporciona** un
embedding suficientemente discriminativo para HSG18, ni siquiera
para que un full_ft con LR conservador lo aproveche.

## Ablación `lr_backbone=1e-4` confirma hipótesis B (estructural)

Para distinguir entre dos hipótesis sobre por qué HSG18 `full_ft_lr1e-5`
colapsa, ejecutamos una **ablación diagnóstica** subiendo
`lr_backbone` de `1e-5` a `1e-4` (10× más alto, el mismo valor que
provocó *catastrophic forgetting* sobre el ckpt central en CWRU).

- **Hipótesis A (adaptación)**: el ckpt FL es menos informativo que el
  central, así que el backbone necesita más holgura (LR mayor) para
  escapar del colapso. Si A → `macro_f1_lr1e-4 > macro_f1_lr1e-5` y
  recall clase 0 mejora.
- **Hipótesis B (estructural)**: el embedding FL HDD no es
  discriminativo (el cliente `hdd` es mono-dataset HSF15); ningún LR
  razonable lo arregla. Si B → `macro_f1_lr1e-4 ≤ fed_linear` o sigue
  cerca de `from_scratch` o colapsa de otra manera.

### Resultado experimental (commit `250868f`, 14 min A100, `config_hash=83cb85a84b582b92`)

```
test_metrics:
  n_samples         : 4576
  accuracy          : 0.5000
  balanced_accuracy : 0.5000
  macro_f1          : 0.3333333333333333
  confusion_matrix  : [[   0, 2288],
                       [   0, 2288]]
  per_class.recall  : [0.0, 1.0]
  per_class.f1      : [0.0, 0.6666666666666666]
best_epoch          : 1 / 20   <- mejor val ya en la primera epoca; despues solo empeora
amp_nonfinite_grad_steps : 6
elapsed_seconds     : 852.7
```

**El modelo predice clase 1 para las 4 576 muestras del test**. Es el
colapso degenerado más extremo posible: 0 muestras clasificadas como
clase 0, recall clase 0 = 0 %. El `best_epoch=1` indica que la
val_macro_f1 nunca volvió a superar la primera época; el backbone se
destruyó casi inmediatamente.

### Deltas

| comparación | Δ macro_f1 |
|---|---:|
| `lr1e-4` vs `lr1e-5` | **−0.2213** (peor) |
| `lr1e-4` vs `fed_linear` | **−0.2747** (peor) |
| `lr1e-4` vs `from_scratch HSG18` | **−0.2360** (peor que random init) |
| ratio `lr1e-4` / `central_full_lr1e-5` | **35.1 %** |

Ambos criterios del runbook se cumplen para favorecer B:

- `macro_f1_lr1e-4 (0.3333) ≤ macro_f1_lr1e-5 (0.5547)` → cumplido.
- `macro_f1_lr1e-4 (0.3333) ≤ fed_linear (0.6080)` → cumplido.
- `macro_f1_lr1e-4 (0.3333) ≤ from_scratch (0.5693)` → cumplido también.
- Colapso degenerado (recall clase 0 = 0 %) → cumplido (criterio
  bonus que excluye categóricamente A).

### Lectura

**Hipótesis B confirmada**. El embedding FL pilot para HDD no contiene
información discriminativa útil. Más LR no rescata; solo acelera la
pérdida del backbone. El cliente FL `hdd` con un solo dataset
(HSF15, peso de agregación 0.048) **no produce un encoder transferible
al dominio HDD**.

Esto es un **hallazgo metodológico citable**: la transferencia FL no
solo es dominio-dependiente, sino que en clientes **mono-dataset** el
embedding es **estructuralmente insuficiente** para downstream en otro
dataset del mismo dominio físico. No es un problema de hiperparámetros
de adaptación; es un problema de cobertura intra-cliente del corpus FL.

### Implicaciones para la recomendación (C)

La recomendación (C) original ("ablar `lr_backbone=1e-4` en HSG18
full_ft para evitar el colapso a la clase mayoritaria") **queda
descartada por sí sola**. La ablación demuestra que la salida del
colapso de `lr1e-5` no se encuentra subiendo el LR; se encuentra (si
existe) mejorando la diversidad intra-cliente del FL.

Las opciones (A) FedProx con μ pequeño y (B) subir
`min_client_presence` siguen siendo razonables porque atacan la causa
estructural (cliente mono-dataset infraponderado), no el síntoma
(colapso a mayoritaria). Pero ninguna de las dos es trivialmente
suficiente; la hipótesis B sugiere que **clientes mono-dataset**
(`hdd`, `wind`, `cnc_milling`, `aero_engines`) **pueden ser un
límite estructural del FL cross-silo simulado** con este corpus.

## Diagnóstico — CWRU sí aprovecha el SSL FL

Lectura del confusion matrix CWRU full_ft FL:

```
clase 0: recall 30.4 %  (147/477)   precision 38.3 %    F1 0.339
clase 1: recall 91.4 % (2173/2378)  precision 83.7 %    F1 0.874
clase 2: recall 68.1 %  (324/476)   precision 90.8 %    F1 0.778
```

3 clases reales con señal aprovechable. Comparado con el central full
(macro_f1 0.8292), el FL pierde sobre todo en la clase minoritaria 0
(0.339 FL vs ~0.7 central estimado). El embedding FL captura las
señales fáciles (clases 1 y 2) y pierde la difícil (0).

## Por qué este patrón es esperable

La topología FL del pilot (sec 7.bis de `CLAUDE.md`) cuenta cuántos
datasets distintos hay por cliente:

| cliente | n_datasets_plan | datasets_sampled_pilot | dominio |
|---|---:|---:|---|
| `bearings` | 12 | 9 | rodamientos (cliente más grande) |
| `phm_challenges` | 4 | 4 | desafíos heterogéneos |
| `misc` | 4 | 3 | misc industrial |
| `misc_industrial` | 5 | 4 | compressor + drills + transformers + ... |
| `batteries` | 5 | 5 | baterías |
| `gearboxes` | 2 | 2 | engranajes |
| `aero_engines` | 1 | 1 | NCMAPSS |
| `cnc_milling` | 1 | 1 | NMILL |
| **`hdd`** | **1** | **1** | **HSF15 — el único dataset HDD** |
| **`wind`** | **1** | **1** | **PHM14 — viento** |

**HSG18 es un dataset HDD** y el cliente `hdd` del FL tiene **un solo
dataset** (HSF15) representando todo el dominio. Cuando FedAvg promedia
los gradientes de 10 clientes con sus respectivos `final_client_weight_capped_v23`
(`hdd: 0.048`), la representación HDD que el ckpt FL produce está
**diluida** por las representaciones de los otros 9 dominios.

CWRU, en cambio, es un dataset de **rodamientos**, y el cliente
`bearings` (peso efectivo `0.249`, el cap superior) contiene **12
datasets distintos del mismo dominio físico** en su plan (IMS,
JNUB, KAUG17, LGB20, PRONOSTIA, SEUGB17, UPM20, UPM23, XJTU-SY +
otros sampleados). Aunque cnetral cubre todos a la vez, FedAvg
extrae al menos las señales bearing comunes que se transfieren a
CWRU.

**Hallazgo metodológico citable**: la transferencia FL es
**dominio-dependiente** y proporcional a la **diversidad intra-cliente**
del dominio en cuestión.

## Criterio de decisión

| criterio | objetivo | CWRU | HSG18 | global |
|---|---|:-:|:-:|:-:|
| `fed_linear > from_scratch` | señal FL | ✓ (+9.5 pp) | ✓ (+3.9 pp) | ✓ |
| `fed_full ≥ 0.9 × central_full` | competitividad | ✗ (80 %) | ✗ (58 %) | ✗ |
| ablación `lr1e-4` salva HSG18 (criterio C) | rescate por LR | n/a | ✗ (colapso 0/4 576 clase 0) | ✗ |

- **GO**: requeriría ambos criterios ✓ → **NO se cumple**.
- **NO-GO**: requeriría `fed_linear ≤ from_scratch` en algún dataset
  → **NO se cumple** (marginal en HSG18 linear, pero positivo).
- **CONDITIONAL**: ambos criterios mixtos → **se cumple** para CWRU; en
  HSG18 el FL se acepta como **NO-GO local** tras la ablación (hipótesis
  B estructural confirmada).

**Veredicto**: **CONDITIONAL global, NO-GO local en HSG18**. El FL
transfiere parcialmente a CWRU (cliente `bearings` multi-dataset) y no
transfiere al dominio HDD (cliente `hdd` mono-dataset HSF15). La
ablación `lr_backbone=1e-4` confirma que el límite en HSG18 **no es de
adaptación, es estructural** del corpus federado.

## Recomendaciones antes de un full FL FedAvg

Cualquiera de las 3 (o combinación) es razonable; cada una con su
propio run_name distinto y sin pisar el FL pilot:

### A. FedProx con μ ≈ 0.01–0.1

Sec 15 `CLAUDE.md` ya prevé FedProx como variante principal vs no-IID.
Reduce client drift agregando una penalización proximal
`μ·||θ_local − θ_global||²` en cada cliente. Esperable: mejora en
clientes pequeños como `hdd` (1 dataset, peso 0.048) sin penalizar
los grandes. Coste: implementación + ablación de μ.

### B. Subir `min_client_presence` de 0.005 a 0.05 para clientes mono-dataset

Da más peso de agregación a `hdd`, `wind`, `cnc_milling`, `aero_engines`.
Coste mínimo: cambio en `sampling_policy` del audit. Riesgo:
desbalancear contra `bearings` y `phm_challenges` que sí están bien
representados; habría que ablar el efecto en CWRU.

### C. Ablación de `lr_backbone` en HSG18 full_ft — EJECUTADA, descartada

`lr_backbone=1e-4` (no 1e-5) podría sacar al modelo del colapso a la
clase mayoritaria. Es lo que vimos en CWRU central: 1e-4 fue
catastrófico, 1e-5 fue el sweet spot. Pero el ckpt FL es **menos
informativo** que el central, así que quizá necesita más holgura
para adaptarse.

**Resultado real (commit `250868f`, 14 min A100)**: la ablación
**colapsa degeneradamente** (macro_f1 = 0.3333, recall clase 0 = 0 %,
predice clase 1 para las 4 576 muestras del test, `best_epoch=1/20`).
Peor que `fed_linear`, peor que `fed_full_lr1e-5`, peor incluso que
`from_scratch`. Confirma hipótesis B: el problema **no es de
adaptación por LR**, es **estructural del embedding FL HDD**. Detalle
en la sección "Ablación `lr_backbone=1e-4` confirma hipótesis B" más
arriba y en
`results/downstream/fl_pilot_vs_central/hsg18/full_finetuning_lr1e-4/run_info.json`.

### D. Aceptar el resultado y lanzar full FL FedAvg

Honesto, citable como evidencia de los límites del FL canónico. Los
~6 h de A100 darían un boost en CWRU (de 80 % a quizá 90 %) pero
**no resolverían HSG18** porque el problema es estructural
(diversidad intra-cliente baja). La ablación lr1e-4 lo refuerza
empíricamente.

**Mi recomendación actualizada** tras la ablación: la opción **C
queda descartada** como rescate por sí sola. Combinar **A** (FedProx
μ pequeño) y/o **B** (`min_client_presence=0.05` para mono-dataset) en
un nuevo pilot v0.2, **antes** del full. Si la mejora es clara en
HSG18, entonces full FL. Si no, aceptar **D** y reportar honestamente:
el FL cross-silo simulado con corpus PHM tiene un límite estructural
en clientes mono-dataset que no se resuelve subiendo `lr_backbone`.

## Comparativa con la hipótesis principal del TFM (estado consolidado)

| dataset | from_scratch | central_linear | central_full_1e-5 | **fed_linear** | **fed_full_1e-5** | central confirma | FL confirma |
|---|---:|---:|---:|---:|---:|:-:|:-:|
| CWRU (bearings, 4 cls) | 0.3503 | 0.7046 | **0.8292** | 0.4456 | 0.6635 | **SÍ** | **parcial** |
| HSG18 (hdd, 2 cls) | 0.5693 | 0.9056 | **0.9504** | 0.6080 | 0.5547 | **SÍ** | **NO fiable** |
| PBCP16 (small, 5 cls) | **0.9074** | 0.8287 | 0.8891 | n/a | n/a | NO | n/a |
| PHM18 (wind, 3 cls) | **0.3655** | 0.2739 | 0.3406 | n/a | n/a | NO | n/a |
| CMAPSS_RUL (aero, reg) | r2 **−0.40** | r2 −1.13 | r2 −0.44 | n/a | n/a | NO | n/a |

**Hipótesis principal del TFM, estado tras este bloque**:

- **Central**: confirmada en 2/4 classification primary (CWRU + HSG18),
  no aporta en PBCP16/PHM18. **NO_CONFIRMADA en CMAPSS_RUL**
  (refutada por bloque RUL cerrado 2026-05-26, ver
  `results/downstream/cmapss_rul/`): ningún modo SSL mejora vs
  from_scratch (linear catastrófico +23 % RMSE; full marginal +1.5 % peor).
- **Federado**: confirmada parcialmente en CWRU (80 % del central);
  no confirmada en HSG18 (full_ft colapsa). **CMAPSS_RUL no se evaluó
  con checkpoints federados en este bloque**: aunque el cliente FL
  `aero_engines` contiene NCMAPSS como fuente relacionada, el central
  100k ya fue NO_CONFIRMADA en CMAPSS_RUL y los checkpoints FL
  disponibles son pilotos de budget reducido; por tanto, no se prioriza
  una evaluación FL-RUL adicional para el MVP. **No se afirma que
  FL-RUL esté refutado**: simplemente no se evaluó. PBCP16/PHM18
  tampoco se evaluaron con FL.

## Artefactos

### Versionados en el repo

- `results/downstream/fl_pilot_vs_central/summary.json` — agregado
  citable de las 4 corridas FL + las 6 central + deltas + ratios +
  decisión.
- `results/downstream/fl_pilot_vs_central/README.md` — este documento.
- `results/downstream/fl_pilot_vs_central/{cwru,hsg18}/{linear_probing,full_finetuning_lr1e-5}/run_info.json`
  — copia **bit-a-bit** de los 4 run_info reales de Drive.
- `results/downstream/fl_pilot_vs_central/hsg18/full_finetuning_lr1e-4/run_info.json`
  — copia **bit-a-bit** del run_info real de la ablación diagnóstica
  (commit `250868f`).
- `training/configs/downstream_{cwru,hsg18}_fedavg_pilot_*.yaml` —
  los 4 configs base (commit `c70179f`) + `downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-4.yaml`
  (commit `250868f`, ablación).
- `notebooks/downstream/run_fl_downstream_pilot_cwru_hsg18.ipynb` —
  notebook ejecutado en Colab (4 corridas base).
- `notebooks/downstream/run_hsg18_fed_pilot_lr_ablation.ipynb` —
  notebook ejecutado en Colab (ablación lr1e-4, commit `250868f`).

### Pesados en Drive (NO versionados)

```
/content/drive/MyDrive/fm_fl_phmd/
  logs/downstream_federated_pilot/cwru/
      downstream_cwru_fedavg_pilot_linear_probing/run_info.json + metrics.jsonl + config.yaml
      downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5/...
      _stdout/*.stdout.log
  logs/downstream_federated_pilot/hsg18/
      ...
  checkpoints/downstream_federated_pilot/cwru/.../best.pt
  checkpoints/downstream_federated_pilot/hsg18/.../best.pt
```

## Estado del bloque federado tras esta evaluación

- Pilot FL SSL: **CERRADO / PASS**.
- Downstream FL pilot CWRU + HSG18: **CERRADO / CONDITIONAL global, NO-GO local HSG18**.
- Ablación HSG18 `lr_backbone=1e-4`: **CERRADO / hipótesis B confirmada**
  (estructural; LR alto no rescata el embedding FL mono-dataset).
- Full FL: **PENDIENTE — NO autorizado hasta diagnóstico estructural**.
  La opción (C) (subir `lr_backbone`) queda descartada por la ablación.
  Vías abiertas: (A) FedProx, (B) `min_client_presence` ajustada,
  o (D) aceptar y reportar el límite estructural.
- FedProx, SCAFFOLD: pendientes según prioridades del TFM.
