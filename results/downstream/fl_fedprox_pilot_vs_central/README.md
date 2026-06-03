# Downstream FedProx pilot vs FedAvg pilot vs Central — CWRU + HSG18

Evaluacion downstream del ckpt **FL FedProx pilot mu=0.01** (commit
`bb38367`, `pilot_pass=true`) sobre los 2 TT primary classification.

**Decision post-corrida**: **NO-GO full FedProx vanilla** por dos motivos
(ver `summary.json` y la seccion "Veredicto" al final).

- 4 corridas en Colab Pro+ A100-SXM4-80GB (2 datasets x 2 modos SSL).
- Tiempo total: ~2 h 4 min (CWRU ~1 h 47 min; HSG18 ~28 min).
- Trainer `5fb90bc`. Configs FedProx commiteados con el mismo `5fb90bc`.

## Aviso importante sobre el notebook

La celda 15 del notebook `run_fl_fedprox_downstream_pilot_cwru_hsg18.ipynb`
tenia un **bug** en la dict comprehension que duplicaba los valores HSG18
sobre la columna CWRU de la tabla comparativa:

```python
# BUG: el for-ds interno sombrea el ds externo
fedprox = {ds: {mode: rows[(ds, mode)]['macro_f1']
                for ds in ['CWRU', 'HSG18']   # <-- shadow
                for mode in ['linear_probing', 'full_finetuning_lr1e-5']
                if (ds, mode) in rows}
           for ds in ['CWRU', 'HSG18']}
```

Resultado: la tabla comparativa **mostraba 0.7242 para CWRU linear** y
**0.5628 para CWRU full**, valores que en realidad pertenecen a HSG18.
**Los `run_info.json` reales (versionados en este directorio) son la
fuente canonica**. El bug fue corregido en el cierre del bloque
downstream FedProx (commit `25cdd81`); los run_info.json versionados
son la fuente canonica. La tabla per-run de la celda 14 si era
correcta.

## Tabla central vs FedAvg pilot vs FedProx pilot (macro_f1 test, REAL)

| dataset | modo | from_scratch | central | FedAvg pilot | **FedProx pilot** | **Δ FP - FA** |
|---|---|---:|---:|---:|---:|---:|
| CWRU  | linear_probing         | 0.3503 | 0.7046 | 0.4456 | **0.4587** | **+0.0131** |
| CWRU  | full_finetuning_lr1e-5 | 0.3503 | 0.8292 | 0.6635 | **0.4889** | **−0.1746** |
| HSG18 | linear_probing         | 0.5693 | 0.9056 | 0.6080 | **0.7242** | **+0.1162** |
| HSG18 | full_finetuning_lr1e-5 | 0.5693 | 0.9504 | 0.5547 | **0.5628** | **+0.0082** |

## Resumen cuantitativo

Solo **1 de 4** corridas mejora claramente con FedProx; **1 empeora**
significativamente; **2 son marginales** (dentro del ruido entre runs):

| caso | Δ FP − FA | lectura |
|---|---:|---|
| HSG18 linear | **+11.62 pp** | mejora clara; el termino proximal ayuda al embedding del cliente hdd mono-dataset en su forma congelada |
| CWRU linear | +1.31 pp | marginal; ruido |
| HSG18 full | +0.82 pp | marginal; sigue por debajo del umbral de colapso |
| **CWRU full** | **−17.46 pp** | **empeora claramente**; FedProx anclado + lr_backbone=1e-5 reduce plasticidad |

## Comparacion vs Central

| dataset | modo | central | **FedProx pilot** | ratio FP / central |
|---|---|---:|---:|---:|
| CWRU  | linear         | 0.7046 | 0.4587 | **65.1 %** |
| CWRU  | full_lr1e-5    | 0.8292 | 0.4889 | **59.0 %** |
| HSG18 | linear         | 0.9056 | 0.7242 | **80.0 %** |
| HSG18 | full_lr1e-5    | 0.9504 | 0.5628 | **59.2 %** |

El **mejor ratio FP/central es HSG18 linear (80 %)**. Es el unico caso
donde FedProx se acerca razonablemente al central. CWRU full FedProx
queda al **59 %** del central; gap muy grande.

## Diagnostico HSG18 full_finetuning_lr1e-5

```
confusion_matrix (FedProx full):
[[ 589, 1699],
 [  38, 2250]]

recall clase 0 (FedProx): 0.2574    (umbral runbook 0.30, NO se alcanza)
recall clase 1 (FedProx): 0.9834

referencia FedAvg full:
recall clase 0 (FedAvg):  0.2417    (umbral 0.30, tampoco)
recall clase 1 (FedAvg):  0.9934
```

El FedProx full HSG18 colapsa **muy similar al FedAvg full**. Mejora
muy ligera en recall clase 0 (+1.57 pp absolutos), pero sigue **por
debajo del umbral**. La hipotesis B estructural (cliente FL `hdd`
mono-dataset HSF15 no produce un encoder transferible al dominio HDD
bajo full_finetuning) **se reconfirma con FedProx**.

## Diagnostico CWRU full_finetuning_lr1e-5 (empeoramiento)

```
confusion_matrix (FedProx full CWRU):
[[ 214,  159, 104, 0],
 [ 577, 1793,   8, 0],
 [ 219,  122, 135, 0],
 [   0,    0,   0, 0]]    <- clase 3 = zero_support_test (no aparece en test)

recall  clase 0 (FedProx): 0.4486    (FedAvg full era 0.304 segun confusion previa)
recall  clase 1 (FedProx): 0.7540    (FedAvg full era 0.914)
recall  clase 2 (FedProx): 0.2836    (FedAvg full era 0.681)
recall  clase 3: indefinido (sin soporte en test)
```

FedProx CWRU full **pierde plasticidad en clases 1 y 2** respecto a
FedAvg full. La clase mayoritaria (1) sigue dominando la prediccion
pero el modelo se confunde mas con la clase 2 (de 0.681 a 0.284 recall).
Es coherente con la hipotesis de que **el termino proximal durante
SSL reduce la flexibilidad del backbone bajo lr_backbone=1e-5**: el
encoder esta mas anclado y necesita LR de adaptacion mas conservador
para evitar regresar a la zona del modelo global.

## Comparacion con FedAvg pilot (FL transfiere)

Recordatorio del bloque FedAvg downstream cerrado en commit `e9fb202`:

| dataset | modo | FedAvg pilot | central | ratio FA/central |
|---|---|---:|---:|---:|
| CWRU  | linear      | 0.4456 | 0.7046 | 63.2 % |
| CWRU  | full_lr1e-5 | 0.6635 | 0.8292 | 80.0 % |
| HSG18 | linear      | 0.6080 | 0.9056 | 67.1 % |
| HSG18 | full_lr1e-5 | 0.5547 | 0.9504 | 58.4 % |

**Comparando los ratios FP/central vs FA/central** (cuanto se acerca
cada FL al SSL central):

| caso | FA/central | FP/central | mejor |
|---|---:|---:|---|
| CWRU linear      | 63.2 % | **65.1 %** | FedProx (+1.9 pp) |
| CWRU full_lr1e-5 | **80.0 %** | 59.0 % | **FedAvg** (+21.0 pp) |
| HSG18 linear     | 67.1 % | **80.0 %** | FedProx (+12.9 pp) |
| HSG18 full_lr1e-5 | 58.4 % | **59.2 %** | FedProx (+0.8 pp) |

**FedAvg gana claramente en CWRU full** (80.0 % vs 59.0 %). FedProx
gana claramente en HSG18 linear (80.0 % vs 67.1 %) y marginalmente en
los otros dos casos.

## Veredicto: NO-GO full FedProx vanilla

| criterio | umbral | FedProx | resultado |
|---|---|---:|:-:|
| `HSG18 full recall clase 0` | ≥ 0.30 | 0.2574 | ✗ (colapsa) |
| `HSG18 full Δ FP-FA macro_f1` | ≥ +0.05 | +0.0082 | ✗ (marginal) |
| `CWRU full no empeora > 5 pp` | Δ ≥ −0.05 | **−0.1746** | ✗ (empeora 17.5 pp) |

**Tres criterios formales NO se cumplen**. NO autorizar full FedProx
vanilla. Hipotesis B estructural reconfirmada en HSG18 + nueva senal
de **incompatibilidad LR=1e-5 con FedProx pretrain** en CWRU full.

## Hallazgo metodologico citable para el TFM

1. **FedProx mu=0.01 mejora la convergencia SSL** (loss agregada baja
   11.57 % vs 7.55 % en FedAvg pilot en mismo budget; ver
   `results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/`).
2. **Esa mejora SSL NO se traduce en mejor transferencia downstream
   general**. Solo 1 de 4 corridas mejora claramente (HSG18 linear).
3. **CWRU full empeora 17.5 pp** con FedProx vs FedAvg en mismo
   `lr_backbone=1e-5`. Senal de que el sweet spot LR de adaptacion
   difiere entre central/FedAvg y FedProx (mas conservador para
   FedProx?).
4. **HSG18 sigue colapsando** en full_finetuning (recall clase 0 < 0.30).
   El termino proximal no rescata el embedding del cliente FL `hdd`
   mono-dataset. **Hipotesis B estructural se reconfirma**.

## Vias abiertas (NO autorizadas en v0.2)

| via | descripcion | coste estimado |
|---|---|---|
| **E nueva** | Ablar `lr_backbone=1e-6` en CWRU full FedProx para verificar si el sweet spot LR difiere | ~50 min A100 |
| **B** | Subir `min_client_presence` 0.005 → 0.05 para mono-dataset (hdd, wind, cnc_milling, aero_engines) y rehacer pilot+eval | ~3 h A100 (pilot + 4 downstream) |
| **D** | Aceptar y reportar honestamente que FedProx mu=0.01 mejora el SSL pero no la transferencia downstream general | 0 h |

**Mi recomendacion**: **D + un E corto**. La opcion D ya es citable
y honesta: el TFM puede defender:

- FedAvg pilot transfiere parcialmente (80 % en CWRU full, ~60 % en
  HSG18; CONDITIONAL global, NO-GO local HSG18).
- FedProx mu=0.01 mejora SSL pero **no mejora downstream salvo en HSG18
  linear**; en CWRU full empeora. Hallazgo metodologico negativo
  citable.
- Hipotesis B estructural (HSG18 cliente mono-dataset) reconfirmada.

Una ablacion **E corta** (`lr_backbone=1e-6` en CWRU full FedProx, 1
corrida ~50 min) permitiria sostener si el empeoramiento es por sweet
spot LR o por la naturaleza FedProx. Eso fortaleceria la narrativa
sin coste alto.

## Comparativa con la hipotesis principal del TFM (estado consolidado)

| dataset | from_scratch | central_linear | central_full_1e-5 | **fed_linear** | **fed_full_1e-5** | **fp_linear** | **fp_full_1e-5** |
|---|---:|---:|---:|---:|---:|---:|---:|
| CWRU  | 0.3503 | 0.7046 | **0.8292** | 0.4456 | 0.6635 | 0.4587 | 0.4889 |
| HSG18 | 0.5693 | 0.9056 | **0.9504** | 0.6080 | 0.5547 | **0.7242** | 0.5628 |

| via FL | CWRU linear | CWRU full | HSG18 linear | HSG18 full |
|---|:-:|:-:|:-:|:-:|
| FedAvg pilot | parcial (63 %) | **parcial (80 %)** | parcial (67 %) | colapso (58 %) |
| FedProx pilot mu=0.01 | parcial (65 %) | regresion (59 %) | **mejor que FA (80 %)** | colapso (59 %) |

**Hipotesis principal del TFM**, estado tras este bloque:

- **Central**: confirmada en 2/4 classification primary (CWRU + HSG18),
  no aporta en PBCP16/PHM18. **NO_CONFIRMADA en CMAPSS_RUL**
  (refutada por bloque RUL cerrado 2026-05-26, ver
  `results/downstream/cmapss_rul/`): ningun modo SSL mejora vs
  from_scratch (linear catastrofico +23 % RMSE; full marginal +1.5 % peor).
- **Federado FedAvg**: confirmada parcialmente en CWRU full (80 %); no
  confirmada en HSG18 full (colapso).
- **Federado FedProx mu=0.01**: confirmada parcialmente en HSG18
  linear (80 % del central, mejor que FedAvg); NO confirmada en CWRU
  full (regresion respecto a FedAvg).
- **CMAPSS_RUL no se evaluó con checkpoints federados en este bloque**.
  Aunque el cliente FL `aero_engines` contiene NCMAPSS como fuente
  relacionada, el central 100k ya fue NO_CONFIRMADA en CMAPSS_RUL y los
  checkpoints FL disponibles son pilotos de budget reducido; por tanto,
  no se prioriza una evaluación FL-RUL adicional para el MVP. CMAPSS_RUL
  sigue siendo un TRANSFER_TARGET; CMAPSS estricto no entra en
  pretraining. **No se afirma que FL-RUL esté refutado**: simplemente
  no se evaluó ni priorizó.
- **Conclusion**: la transferencia SSL es **task-dependent** (funciona
  en classification, no en regresion RUL) **y** algoritmo-dependent en
  FL (FedAvg vs FedProx con tradeoffs distintos por dominio). No hay
  una variante FL canonica que domine en todos los casos. Hallazgo
  metodologico citable.

## Artefactos

### Versionados en el repo

- `results/downstream/fl_fedprox_pilot_vs_central/summary.json` -
  agregado citable de las 4 corridas FedProx + central + FedAvg +
  deltas + ratios + decision.
- `results/downstream/fl_fedprox_pilot_vs_central/README.md` - este
  documento.
- `results/downstream/fl_fedprox_pilot_vs_central/{cwru,hsg18}/{linear_probing,full_finetuning_lr1e-5}/run_info.json`
  - copia **bit-a-bit** de los 4 run_info reales de Drive.
- `training/configs/downstream_{cwru,hsg18}_fedprox_pilot_mu0_01_*.yaml`
  - los 4 configs usados (commit `5fb90bc`).
- `notebooks/downstream/run_fl_fedprox_downstream_pilot_cwru_hsg18.ipynb`
  - El notebook nacio en `5fb90bc`, pero la celda comparativa fue
  corregida posteriormente en `25cdd81` (fix del bug del shadowing
  de `ds` en la dict comprehension).

### Pesados en Drive (NO versionados)

```
/content/drive/MyDrive/fm_fl_phmd/
  logs/downstream_federated_pilot_fedprox_mu0_01/
      cwru/
          downstream_cwru_fedprox_pilot_mu0_01_linear_probing/{run_info.json, metrics.jsonl, config.yaml, label_mapping.json}
          downstream_cwru_fedprox_pilot_mu0_01_full_finetuning_lr1e-5/...
          _stdout/*.stdout.log
      hsg18/
          downstream_hsg18_fedprox_pilot_mu0_01_linear_probing/...
          downstream_hsg18_fedprox_pilot_mu0_01_full_finetuning_lr1e-5/...
          _stdout/*.stdout.log
  checkpoints/downstream_federated_pilot_fedprox_mu0_01/
      cwru/.../best.pt
      hsg18/.../best.pt
```

## Estado del bloque federado tras esta evaluacion

- Pilot FL FedAvg v0.1: CERRADO / PASS.
- Downstream FL FedAvg pilot CWRU/HSG18: CERRADO / CONDITIONAL global, NO-GO local HSG18.
- Ablacion HSG18 lr1e-4 (FedAvg): CERRADO, hipotesis B estructural confirmada.
- Pilot FL FedProx mu=0.01 v0.2: CERRADO / PASS (loss SSL mejor que FedAvg, +4.0 pp).
- **Downstream FL FedProx pilot CWRU/HSG18: CERRADO / NO-GO full FedProx vanilla** (HSG18 full colapsa + CWRU full empeora 17 pp).
- Full FedProx: **NO autorizado**. Vias abiertas: E nueva (ablar lr_backbone), B (min_client_presence), o D (aceptar y reportar).
- FedProx ablacion mu o SCAFFOLD: pendientes segun prioridades del TFM.
