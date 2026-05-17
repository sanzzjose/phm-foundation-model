# CMAPSS_RUL_DECISION (2026-05-24)

> Documento citable y versionado en el repo. El análisis interno y la
> deliberación se encuentran en `docs/decisions/pending_downstream_and_sampling.md`
> sec 1 (gitignored).

## Decisión

**Reconstruir RUL físico desde raw / loader original de PHMD; no usar el
target actual de `processed/CMAPSS/`.**

## Contexto resumido

El análisis ejecutado en Colab sobre los shards de `processed/CMAPSS/`
(script `training/analyze_cmapss_rul_semantics.py`) reveló tres
distribuciones del campo `rul` por split:

| split | count | min  | max | median | n_negative | frac_negative |
|------:|------:|-----:|----:|-------:|-----------:|--------------:|
| train | 569   | 0    | 31  | 0      | 0          | 0.00          |
| val   | 142   | 0    | 0   | 0      | 0          | 0.00          |
| test  | 707   | -466 | 170 | -72    | 490        | 0.69          |

La asimetría confirma que el loader de PHMD entrega `rul` con
convenciones distintas entre splits originales:

- **train/val**: ya es "ciclos restantes hasta fallo", decreciente
  hasta 0 cuando ocurre el fallo (run-to-failure).
- **test**: parece codificado como `cycle − failure_cycle` (la mayoría
  negativos = observaciones anteriores al fallo).

La harmonization v0.5 aplicó `target_policy='ultimo_valor_valido'` y
`target_warning='rul_negative_values'`, pero **no resuelve la mezcla
semántica**: cualquier modelo entrenado sobre este `target_window`
aprendería un mapeo inconsistente entre splits.

## Plan correcto

Volver al raw y reconstruir RUL físico con la fórmula apropiada por
split:

```
# train / val (run-to-failure)
rul_physical[i] = max_cycle_by(FD, unit_id) - cycle[i]

# test (interrumpido antes del fallo, con RUL oficial publicado)
rul_physical[i] = last_observed_cycle_by(FD, unit_id) - cycle[i] + official_RUL_FD[unit_id]
```

Con esto, `rul_physical >= 0` por construcción en ambos splits. Sobre
esa base puede derivarse opcionalmente:

```
rul_capped_125 = min(rul_physical, 125)
```

como variante secundaria (estándar literatura desde Heimes 2008). El
cap **no es un parche** sobre negativos; solo aplica una vez que ya
hay RUL físico bien definido.

## Lo que NO se hace

- No se entrena RUL supervisado con `target_window` actual de
  `processed/CMAPSS/`.
- No se conserva `rul` original "tal cual" como métrica reportable.
- No se invierte signo de forma global.
- No se aplica clamp automático sin haber reconstruido RUL físico
  primero.
- No se modifica `processed/CMAPSS/` existente (los shards SSL siguen
  válidos: el SSL no usa el target).
- No se reharmonizan los 47 datasets.
- No se redescarga raw.

## Lo que SÍ se hace

1. Builder dedicado `training/build_cmapss_rul_downstream.py` con:
   - CLI `--dry-run` por defecto (no escribe shards).
   - Funciones puras `compute_train_rul`, `compute_test_rul`, `cap_rul`.
   - Lectura del raw existente en Drive (no descarga).
2. Salida separada en `processed_downstream/CMAPSS_RUL/` (no toca
   `processed/`).
3. Tests sintéticos en `tests/test_build_cmapss_rul_downstream.py`.
4. Manifest con la fórmula aplicada por split + cap + hash del código.
5. Posteriormente, cabeza `RegressionHead` y trainer RUL análogos al
   pipeline classification ya validado.

## Estado

- Decisión: **DECIDIDA** el 2026-05-24.
- Layout raw real PHMD = `zips_split_phmd`:
  `raw/datasets/CMAPSS_train.zip` + `raw/datasets/CMAPSS_test.zip`.
  Estructura interna NASA estándar: `CMAPSS/train/train_FDxxx.txt`
  y `CMAPSS/test/{test,RUL}_FDxxx.txt` para FD001..FD004 (los 4
  subsets completos). `inspect_raw_cmapss` extendido para reconocer
  este layout sin extraer.

### Commits del builder

- **Commit 1 (parsers + preview básico, sin escritura de shards)**:
  **CERRADO** el 2026-05-24 con 14 tests nuevos PASS. Añade:
  - `parse_cmapss_txt_filelike` y `parse_official_rul_filelike` (formato
    NASA whitespace 26 columnas; validación de columnas, enteros, monotonía).
  - `load_cmapss_raw_from_split_zips`: lee los 2 zips PHMD sin extraer,
    valida que `len(RUL_FDxxx) == n_units_test`.
  - `build_train_val_test_rul`: reconstruye RUL físico por unidad con
    `compute_train_rul`/`compute_test_rul`, asserts duros `min >= 0`,
    aplica `cap_125` SOLO después de tener `rul_physical >= 0`, split val
    20% por unidades estratificado por FD con seed=42.
  - `preview_summary` inicial con `window_mode=rolling_causal`, `n_windows
    = n_rows`, estimación de tamaño en disco.
- **Commit 2 (ventaneo `rolling_causal` + patching + normalización,
  preview enriquecido)**: **CERRADO** el 2026-05-24. Añade:
  - `iter_unit_windows` / `iter_split_windows` en memoria, con
    `valid_time_mask`, `valid_patch_mask`, `instance_norm` por
    ventana/canal ignorando padding.
  - Patching `(C, N, P) = (24, 32, 16)` en `float32`.
  - Preview con estadísticas de padding y tamaño estimado.
- **Commit 2b (política `stride / min_valid_timesteps /
  include_last_per_unit` + preview filtrado)**: **CERRADO** el 2026-05-25
  con 67/67 PASS local y 256 PASS + 1 SKIP en la suite completa. Añade:
  - Helper puro `selected_t_indices(T, stride, min_valid_timesteps,
    include_last_per_unit)` con 6 tests exhaustivos.
  - `iter_unit_windows` / `iter_split_windows` / `preview_summary`
    extendidos para aceptar la política; meta por sample con
    `valid_timesteps`, `below_min_valid_because_last`,
    `selected_by_last_override`, `min_valid_timesteps`, `stride`.
  - CLI `--min-valid-timesteps`, `--include-last-per-unit /
    --no-include-last-per-unit`.
  - Stdout y `dry_run_report.md` con métricas post-filtro.
- **Dry-run real filtrado en Colab (commit 2b)**: **EJECUTADO / PASS**
  el 2026-05-25. Artefactos versionados:
  - `results/downstream/cmapss_rul_decision/dry_run_report.md`
  - `results/downstream/cmapss_rul_decision/dry_run_report.json`

### Política canónica confirmada por el dry-run real

```
W                       = 512
P                       = 16
N                       = 32   (= W / P)
C                       = 24   (3 op_settings + 21 sensores)
window_mode             = rolling_causal
stride                  = 5
min_valid_timesteps     = 128
include_last_per_unit   = True
normalization_policy    = instance_norm_per_window_channel_ignore_padding
split_policy            = unit_holdout_by_fd
val_frac                = 0.2
split_seed              = 42
target_policy           = rul_at_prediction_cycle
rul_cap                 = 125  (Heimes 2008, aplicado SOLO sobre RUL físico ≥ 0)
```

### Totales reales del dry-run filtrado (FD001..FD004, train+val+test)

Post-fix `_fd_seed_offset` (commit `8854851`), confirmados bit-a-bit
en Colab con dos corridas idénticas:

```
n_windows_train                   = 11 795
n_windows_val                     =  3 131
n_windows_test                    =  6 610
n_windows_total                   = 21 536
n_windows_dropped_by_min_valid    = 28 938
n_windows_added_by_last_override  =  1 200
estimated_size_gb_float32         = 1.01 GB
frac_timesteps_valid_avg          = 0.3807
frac_windows_padded               = 0.9995  (solo 11 ventanas con T ≥ W)
```

> **Fix `_fd_seed_offset` (2026-05-25): CERRADO**. La partición
> concreta `n_windows_train` / `n_windows_val` registrada inicialmente
> en `dry_run_report.{md,json}` del commit 2c (`11931 / 2995`) **no era
> reproducible entre procesos** porque dependía de `hash(str)` (afectado
> por `PYTHONHASHSEED`, randomizado por defecto). Tres corridas dieron
> tres splits distintos en la misma VM: `11931/2995`, `11995/2931`,
> `11851/3075`. El fix sustituye `hash(fd)` por `_fd_seed_offset(fd)`
> con offset canónico `FD001..FD004 → 0..3` (y fallback `sha256` para
> FDs no canónicos). Los totales arriba son los definitivos
> post-fix.

`rul_physical_min` es **0** en train y val de los 4 FD (gracias a
`include_last_per_unit`, el ciclo de fallo se preserva siempre); en
test es ≥ 6 (consistente con `last_cycle − cycle + official_RUL`,
donde `T-1` está antes del fallo). `rul_capped_125_max = 125` exacto
en los 12 (FD × split). El conteo `only_last` en test es 303 unidades
con `T < 128` que aportan **una** ventana por unidad via override,
marcadas en meta con `below_min_valid_because_last=True`.

### Commit 3 (writer TAR + manifest + asserts duros): CERRADO / PASS (2026-05-25)

`--write-shards` ejecutado en Colab con `pipeline_code_version =
fd0cec90604900791de15beca4a79eb9fbe22adc`,
`pipeline_config_hash = 8317ba2a1bc87e20` (16 hex chars, determinista,
mismo en futuras corridas con misma política).

**Resultados en disco** (`/MyDrive/fm_fl_phmd/processed_downstream/CMAPSS_RUL/`):

```
manifest.json    -> versionado en results/downstream/cmapss_rul_decision/manifest_real.json
done.flag        -> marcador de cierre
train/  shard_0000.tar .. shard_0011.tar    (12 shards, 11795 samples)
val/    shard_0000.tar .. shard_0003.tar    ( 4 shards,  3131 samples)
test/   shard_0000.tar .. shard_0006.tar    ( 7 shards,  6610 samples)
TOTAL: ~1.2 GB en Drive (dry-run estimaba 1.01 GB, +20% por overhead TAR).
```

| split | n_units | n_windows | n_shards | rul_phys_min | rul_phys_max | cap_max | n_temp_patches | n_chan_patches |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 567 | 11 795 | 12 | 0.0 | 366 | 125 | 377 440 | 9 058 560 |
| val   | 142 |  3 131 |  4 | 0.0 | 415 | 125 | 100 192 | 2 404 608 |
| test  | 707 |  6 610 |  7 | 6.0 | 426 | 125 | 211 520 | 5 076 480 |
| **total** | **1416** | **21 536** | **23** | — | — | — | **689 152** | **16 539 648** |

**Cuadre vs dry-run versionado** (`fd0cec9`): bit-a-bit. `n_windows` train/val/test
exactos. `train + val = 709 = train_orig` (FD001:100 + FD002:260 + FD003:100 +
FD004:249); `test = 707 = test_orig` (FD001:100 + FD002:259 + FD003:100 +
FD004:248). Sin pérdida ni duplicación de unidades.

**Asserts duros pre-escritura** (PASS):
- `rul_physical_min == 0` en train y val de los 4 FD (gracias a
  `include_last_per_unit=True`, el ciclo de fallo entra siempre).
- `rul_physical_min >= 0` en test (>= 6 por construcción, consistente
  con `last_cycle - cycle + official_RUL`).
- `rul_capped_125_max == 125` exacto en los 3 splits.
- `n_samples` coincide con preview bit-a-bit en los 3 splits.

**Anti-leakage** (12 checks PASS, 3 por FD × 4 FD + globales):
- `no_overlap_{train_val, train_test, val_test} = True`.
- `train_source_split_is_train_orig = True`, `val_source_split_is_train_orig
  = True`, `test_source_split_is_test_orig = True`.
- `train_val_drawn_from_train_orig = True`, `test_is_test_orig = True`.
- Por FD: `n_train_observed == n_train_orig_expected`,
  `n_val_observed == n_val_from_train_orig_expected`,
  `n_test_observed == n_test_orig_expected` (todos exactos).

**Roundtrip sanity sobre `train/shard_0000.tar`**:
```
sample key:        cmapss_FD001_train_train_orig_unit1_w000127
patches:           (24, 32, 16) float32
valid_time_mask:   (512,) bool         <- contrato (W,) cumplido
valid_patch_mask:  (24, 32) bool       <- contrato (C, N) cumplido
source_split:      train_orig
unit_global_id:    CMAPSS_FD001_train_orig_unit1
target_rul_physical:   64.0    (cycle 128 / max 192 -> RUL = 64)
target_rul_capped_125: 64.0
```

Tests Colab post-pull: **93/93 PASS** (90 anteriores + 2 CLI guards +
1 PatchTST compat que en Colab Linux corre y pasa, en Windows local se
salta por incompat torch/numpy2).

### Próximos pasos (commits separados)

- **Commit 4**: `RegressionHead = Linear(d_model, 1)` en
  `training/downstream/heads.py` + `RegressionDownstreamModel` análogo
  a `DownstreamClassifier`.
- **Commit 5**: `training/train_downstream_rul.py` + config YAML
  (`training/configs/downstream_cmapss_rul.yaml`), reusando
  `JsonlLogger`, `pooled_embedding`, `compute_adaptive_batch_size`.
- **Commit 6**: corridas downstream RUL en Colab (`from_scratch`,
  `linear_probing`, `full_finetuning` con `lr_backbone=1e-5`).
  Targets: `rul_capped_125` por defecto (estándar literatura);
  métricas: RMSE, MAE, R², CMAPSS-Score asimétrico (penaliza más sobrestimación
  que subestimación).

## Decisiones de ventaneo (cerradas el 2026-05-24, ampliadas el 2026-05-25)

### Fijas (consistencia con SSL central)

- **`window_size = 512`** por compatibilidad con SSL central full.
  W=128 y W=256 quedan como **ablation futura**, no canónica.
- **`patch_size = 16`**, **`n_patches = 32`**.
- **`window_mode = rolling_causal`**: una ventana por ciclo `t` que
  mira los últimos W ciclos hasta `t`. Padding causal por la izquierda
  cuando `t < W`. **NO una sola ventana por trayectoria**.
- **`target_policy = rul_at_prediction_cycle`**: el target de la
  ventana cuyo último ciclo es `t` es `rul_physical[t]` (NO agregado).
- **`split_policy = unit_holdout_by_fd`**, **`val_frac = 0.2`**,
  **`split_seed = 42`**.
- **`normalization_policy = instance_norm_per_window_channel_ignore_padding`**.
- **`include_op_settings = True`** → 24 canales (3 op + 21 sensores).

### Política de selección de t_idx (cerrada el 2026-05-25 tras preview real)

Tras observar en el dry-run real (`stride=1`, sin filtro):

- 265 256 ventanas totales (train+val+test sumados sobre los 4 FD).
- **99.98 % de ventanas con padding** (solo 46 ventanas con historial
  completo de W=512).
- `frac_timesteps_valid_avg = 0.2193`: cada ventana tiene en promedio
  solo el 22 % de timesteps reales.

Las trayectorias CMAPSS son cortas comparadas con W=512 (max 553 en
FD004; max 362 en FD001). Con `stride=1` se generan ventanas casi
idénticas, redundantes y con padding extremo en los primeros ciclos.

**Política canónica adoptada**:

- **`stride = 5`**: una ventana cada 5 ciclos. Reduce 5× la redundancia
  manteniendo solapamiento 507/512 entre ventanas consecutivas.
- **`min_valid_timesteps = 128`** (≥25 % de W): descarta las ventanas
  donde el modelo predeciría RUL con < 128 ciclos de historial real
  (caso poco informativo).
- **`include_last_per_unit = True`**: garantiza que el **ciclo final
  de cada unidad** se emita siempre, aunque no caiga en la rejilla
  stride o quede por debajo del piso. En train/val esto preserva la
  observación de **fallo (`rul_physical = 0`)** por unidad; en test
  preserva el último ciclo observado con su `official_rul` exacto.

Implicaciones de esta política:

- El número de ventanas baja de ~265k (stride=1 sin filtro) a una
  fracción defendible.
- `frac_timesteps_valid_avg` sube (por descartar los primeros ciclos
  con padding extremo).
- Las unidades con `T < 128` solo aportan **una** ventana (la del
  ciclo final, via `last_override`). Se anotan en
  `meta.below_min_valid_because_last=True` para trazabilidad.

### Ablations futuras (NO canónicas, no se ejecutan ahora)

- W reducido a 256 o 128 (rompería compatibilidad con SSL central;
  requeriría re-pretraining).
- `stride = 1` (densidad máxima, redundancia máxima).
- `min_valid_timesteps = 256` (50 % de W, más restrictivo).
- `include_last_per_unit = False` (sin override, descarta unidades
  cortas).

Cualquier ablation se ejecutaría con su propio `--out-dir` separado y
manifest aparte, sin reemplazar la versión canónica.

## Contratos técnicos para el commit 3 (writer)

Las siguientes reglas deben respetarse cuando se active `--write-shards`.
Quedan registradas aquí para que el writer no improvise.

### 1. `valid_patch_mask` debe persistirse como `(C, N)`, no `(N,)`

Aunque dentro de una misma ventana todos los canales comparten el
mismo `valid_time_mask` por construcción (el padding causal por la
izquierda es uniforme en C), el contrato canónico de los shards SSL
(`processed/<dataset>/`) y el contrato channel-independent del
encoder reciben máscaras con forma `(C, N)`. Para mantener
compatibilidad bit-a-bit con el DataLoader existente y con
`masked_reconstruction_loss`, el builder debe expandir
`valid_patch_mask` a `(C, N)` antes de guardarlo (broadcast trivial
del `(N,)` por canal). Lo mismo aplica a `valid_time_mask`: forma
`(C, W)` en disco aunque la información sea redundante. Esto evita
que el trainer downstream tenga que hacer reshapes ad-hoc en el
collate y mantiene la interfaz idéntica a la harmonization v0.5.

### 2. Anti-leakage train/val se comprueba **dentro del train original**

CMAPSS tiene dos `unit_id` numéricos completamente independientes:
los de `train_FDxxx.txt` (1..N_train_FD) y los de `test_FDxxx.txt`
(1..N_test_FD). El val del builder se construye **muestreando 20 %
de las unidades del train original** estratificado por FD con
`seed=42`. Por tanto:

- El check `set(train_units_FDx) ∩ set(val_units_FDx) == ∅` se
  evalúa sobre las unidades originales del split train de cada FD.
- El test queda intacto: todas sus unidades pasan a `split=test`
  sin estar en train ni en val.
- **No** hay reparto adicional entre train y test: el split test
  viene tal cual del benchmark NASA.

### 3. Usar `source_split` en `unit_global_id`, no comparar `unit_id` numérico

`unit_id=3` en `FD001/train` y `unit_id=3` en `FD001/test` son
**unidades físicamente distintas** (diferentes motores, sin
relación). Cualquier check anti-leakage que compare `unit_id`
puro reportaría falsos positivos. El identificador canónico para
trazabilidad y anti-leakage es:

```
unit_global_id = f"CMAPSS_{fd}_{source_split}_unit{unit_id}"
```

donde `source_split ∈ {train_orig, test_orig}` refleja la
procedencia en el benchmark NASA original, no el split del builder
(`train | val | test` del builder). Esto cumple la regla general de
la sec 6 de `CLAUDE.md` (unit_global_id incluye procedencia, no
solo unit_id).

El `manifest.json` del commit 3 debe registrar explícitamente
`unit_global_id_policy = "CMAPSS_<fd>_<source_split>_unit<unit_id>"`
y `anti_leakage_checks = { units_unique_per_split: true,
no_overlap_train_val: true, train_val_drawn_from_train_orig: true,
test_is_test_orig: true }`.

## Contrato final del payload TAR (cerrado el 2026-05-25)

Tras el commit `8854851` (fix determinismo) y la corrección del
contrato de máscaras post-commit 3, cada sample del shard `.tar` se
serializa como nueve blobs con shapes definitivas:

```
patches.npy                       (C, N, P)   float32
valid_time_mask.npy               (W,)        bool
valid_patch_mask.npy              (C, N)      bool
canales_constantes_mask.npy       (C,)        bool
mean.npy                          (C,)        float32
std_used.npy                      (C,)        float32
rul_physical.npy                  scalar      float32
rul_capped_125.npy                scalar      float32
meta.json                         dict        utf-8
```

donde `C = 24` (3 op_settings + 21 sensores), `W = 512`, `P = 16`,
`N = 32`.

### Justificación de las shapes

- **`patches`** `(C, N, P)`: contrato channel-independent del SSL
  central (sec 9 CLAUDE.md). Al batchear: `(B, C, N, P)`.
- **`valid_time_mask`** `(W,)`: el padding causal es uniforme entre
  canales, por lo que persistir `(C, W)` sería redundante. Al
  batchear: `(B, W)`, exactamente lo que espera
  `PatchTSTPhm.forward`. Si algún consumidor necesita la expansión
  `(C, W)` (p.ej. para máscaras por canal en otros datasets), debe
  llamarse a `expand_valid_time_mask_cw` en el DataLoader/collate.
- **`valid_patch_mask`** `(C, N)`: contrato canónico del encoder; al
  batchear da `(B, C, N)`, canonicalizable.
- **`mean`, `std_used`** `(C,)`: estadísticas de normalización por
  canal ignorando padding. Permiten al downstream recuperar el valor
  absoluto si lo necesita (sec 10 CLAUDE.md).
- **`rul_*`** escalar `float32`: target por ventana, no agregado.

### Test de compatibilidad

`tests/test_build_cmapss_rul_downstream.py::test_writer_payload_compatible_con_patchtst_phm`
escribe un TAR sintético, lee un sample y verifica que
`PatchTSTPhm.forward(patches[None,...], vtm[None,...], vpm[None,...])`
es válido sin reshapes ad-hoc. Skip en Windows local por incompat
torch+numpy2 del entorno, corre PASS en Colab.

### Guard CLI de stride canónico

A partir del commit `fd0cec9`, la CLI tiene:

- `--stride` con default `5` (canónico).
- `--write-shards` aborta con rc=1 si `stride != 5` y NO se pasa
  `--allow-noncanonical`.
- `--allow-noncanonical` queda reservado para ablations: el
  pipeline_config_hash incluye el stride, por tanto el manifest del
  output ablation no coincidiría con el canónico y queda claramente
  separado.

## Referencias

- Heimes 2008, "Recurrent Neural Networks for Remaining Useful Life
  Estimation", PHM Conference (cap de 125 ciclos).
- CMAPSS dataset documentation en NASA Prognostics Data Repository.
- `docs/decisions/pending_downstream_and_sampling.md` sec 1 para
  histórico completo de la deliberación.
