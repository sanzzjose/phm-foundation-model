# Downstream RUL — CMAPSS_RUL (TRANSFER_TARGET primary)

Resultados de las **3 corridas oficiales** del bloque RUL del TFM,
ejecutadas en Colab Pro+ A100 sobre el dataset construido por el
builder dedicado (`processed_downstream/CMAPSS_RUL/`,
`pipeline_config_hash=8317ba2a1bc87e20`).

- Dataset: **CMAPSS_RUL** (4 FD, 567 / 142 / 707 unidades train / val / test;
  11 795 / 3 131 / 6 610 ventanas; 23 shards = ~1.2 GB en Drive).
- Tarea: **regresión RUL** (`target_key=rul_capped_125`, cap Heimes 2008).
- Backbone: `PatchTSTPhm` base (d_model=128, 4 layers, 4 heads, d_ff=512,
  patch_size=16, n_patches=32, 24 canales = 3 op_settings + 21 sensores).
- Cabeza: `RegressionHead` lineal mínima `Linear(d_model=128, 1)` (129 params).
- SSL checkpoint: `ssl_central_full/ssl_central_full_patchtst_phm/ckpt_step100000.pt`
  (9.3 MB, 100 000 optimizer steps en 36 PRETRAIN_SOURCE, loss 47.7 %
  de reducción).
- Trainer: `training/train_downstream_rul.py` con configs versionados
  en `training/configs/downstream_cmapss_rul_*.yaml` (commit `baef887`).
- Configs blindados por `tests/test_downstream_cmapss_rul_configs.py` (16/16 PASS, commit `8696bc2`).
- Notebook ejecutado: `notebooks/downstream/run_downstream_cmapss_rul.ipynb` (commit `86c7369`).
- `git_hash` real de las 3 corridas: `86c73695721e044fc971366ac81ef5d6ee205848` (= `86c7369`).

## Resultados agregados

| mode | best_epoch | rmse_val | mae_test | rmse_test | r2_test | cmapss_score | elapsed | n_trainable |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **from_scratch** | **20 / 20** | 39.08 | **37.01** | **43.67** | **−0.40** | 754 754 | 571.5 s | 801 937 |
| linear_probing | 20 / 20 | 43.21 | 47.21 | 53.86 | **−1.13** | 1 002 485 | 386.6 s | 129 |
| full_finetuning (lr_backbone 1e-5) | 15 / 20 | 39.54 | 38.21 | 44.33 | **−0.44** | 645 389 | 486.1 s | 801 937 |

Tiempo total: ~24 min en A100 (corpus pequeño; A100 muy infrautilizada).

## Lectura honesta

1. **`r2_test < 0` en los 3 modos**. Esto significa que predecir
   la media de RUL en test sería **mejor** que cualquiera de los 3
   modelos. Es el hallazgo principal de esta corrida.

2. **`linear_probing` es claramente el peor** (RMSE_test 53.86 vs 43.67
   de from_scratch; r2 −1.13 vs −0.40). Solo 129 parámetros entrenables
   sobre el embedding del SSL central no son suficientes para mapear a
   RUL.

3. **`from_scratch` y `full_finetuning_lr1e-5` quedan prácticamente
   empatados** en r2_test. La diferencia (RMSE 43.67 vs 44.33;
   cmapss_score 754k vs 645k) está dentro del ruido entre seeds; no hay
   ganancia clara del SSL en full_ft.

4. **Con esta configuración, el SSL central full NO transfiere a
   CMAPSS RUL**.

5. **Esto NO invalida los resultados de CWRU + HSG18**, donde la
   hipótesis principal del TFM **sí queda confirmada** en classification
   (linear_probing >> from_scratch; full_ft con lr_backbone=1e-5 mejor
   aún). La transferencia del SSL es **task-dependent**: ayuda en
   clases que comparten estructura con el masked patch prediction
   (texturas locales discriminativas) y no necesariamente en regresión
   continua con shift entre train y test.

## Diagnóstico metodológico

Factores que explican plausiblemente el resultado:

- **El masked patch reconstruction aprende textura local**, no la
  dinámica monotónica de degradación. RUL pide *proyectar hacia
  adelante* cuánta vida queda, una tarea de agregación temporal
  monótona que el SSL no incentivó explícitamente.
- **`W=512` con `frac_timesteps_valid_avg ≈ 0.38`**: el 62 % de cada
  ventana es padding causal por la izquierda. El encoder pasa la mayor
  parte del cómputo viendo ceros. Esta amenaza a la validez está
  registrada desde el audit v2.3 y la decisión de mantener `W=512`
  por compatibilidad con el SSL central.
- **Shift train / test**: train+val tienen `rul_min=0` por construcción
  (`include_last_per_unit=True` garantiza la observación del fallo);
  test tiene `rul_min ≥ 6` porque NASA interrumpe las trayectorias
  antes del fallo. La región `RUL=0` que el modelo aprende a predecir
  en train **nunca aparece** en test. Mismatch sistemático que ningún
  modo de adaptación cubre.
- **20 épocas pueden ser pocas**: `from_scratch` y `linear_probing`
  ambos tienen `best_epoch=20/20`, es decir, mejoraban en la última
  época y no han convergido.
- **`lr_head=1e-3` posiblemente alto** para regresión con MSE en rango
  inicial `loss ≈ 10⁴`: los gradientes iniciales del head son grandes
  (smoke `gn ≈ 1500`), el optimizer puede oscilar.
- **stdout logs no guardados**: el directorio `_stdout/` no existía al
  lanzar las celdas con `tee`; los `run_info.json`, `metrics.jsonl` y
  `best.pt` sí están en Drive. No se vuelve a entrenar para recuperar
  el stdout — los artefactos canónicos del trainer cubren la
  trazabilidad.

## Estado de la hipótesis principal del TFM

Recordatorio del patrón confirmado en classification:

| dataset | from_scratch | linear_probing | full_ft (lr 1e-5) | mejor | hipótesis |
|---|---:|---:|---:|---|---|
| CWRU (4 cls) | macro_f1 0.35 | 0.70 | **0.83** | full_ft | confirmada |
| HSG18 (2 cls) | macro_f1 0.57 | 0.91 | **0.95** | full_ft | confirmada |
| PBCP16 (5 cls) | **0.91** | 0.83 | 0.89 | from_scratch | no aporta (small) |
| PHM18 (3 cls) | **0.37** | 0.27 | 0.34 | from_scratch | no aporta (hard) |
| **CMAPSS_RUL** | **r2 −0.40** | r2 −1.13 | r2 −0.44 | from_scratch | **no aporta (regresión)** |

La hipótesis principal queda confirmada en 2 / 4 classification
primary (CWRU y HSG18, dominios físicamente distintos: bearings y hdd)
y refutada en 1 / 1 regresión RUL. El TFM debe presentarlo así: el
SSL central full **transfiere bien a clasificación de fallos en
dominios distintos del de pretraining**, y **no transfiere a regresión
RUL con esta configuración canónica**.

## Ablaciones futuras (NO ejecutar ahora)

Líneas de mejora para una segunda iteración del bloque RUL si el
calendario del TFM lo permite. Cada una iría con su `run_name` distinto
para no pisar las corridas oficiales.

| Ablación | Hipótesis |
|---|---|
| `W=128` ó `W=256` + re-pretraining SSL coherente | Reducir `frac_timesteps_valid_avg` de 0.38 hacia 0.7-0.9 puede liberar capacidad del encoder para representar la trayectoria. |
| `head_hidden_dim=64` ó `128` con `activation=gelu` | Cabeza no lineal sobre el embedding pre-trained puede capturar la relación `embedding → RUL` que la lineal no extrae. |
| `lr_head=1e-4`, `lr_backbone=1e-6` | Atenuar los gradientes iniciales (loss inicial ~10⁴ + MSE) y permitir aprendizaje más estable. |
| 50–100 épocas con early stopping por `rmse_val` | `best_epoch=20/20` en 2/3 modos indica que el plateau no se alcanzó. |
| Loss Huber / L1 / asymmetric en vez de MSE | Robustez frente al cap `rul_capped_125=125` y a outliers del rango alto. |
| `target_key=rul_physical` sin cap | Diagnóstico: ¿el cap esconde estructura aprovechable o ya tiene la relación que el modelo necesita? |
| Aumentar `lr_backbone` específicamente para el head warmup | Calentar el head con backbone congelado N épocas, luego liberar. |

Ninguna de estas ablaciones se ejecuta como parte del bloque oficial.

## Artefactos

### Versionados en el repo

- `results/downstream/cmapss_rul/summary_rul_cmapss.json` — agregado
  citable de las 3 corridas (este directorio), con deltas, ratios y
  verdict NO_CONFIRMADA + matices.
- `results/downstream/cmapss_rul/{from_scratch,linear_probing,full_finetuning_lr1e-5}/run_info.json`
  — copia **bit-a-bit** de los 3 run_info reales de Drive.
- `results/downstream/cmapss_rul/README.md` — este documento.
- `training/train_downstream_rul.py` — trainer.
- `training/downstream/heads.py` — `RegressionHead` +
  `RegressionDownstreamModel`.
- `training/configs/downstream_cmapss_rul_{from_scratch,linear_probing,full_finetuning_lr1e-5}.yaml`
  — los 3 configs oficiales con `config_hash` distinto.
- `tests/test_train_downstream_rul.py`, `tests/test_downstream_regression_head.py`,
  `tests/test_downstream_regression_metrics.py` — tests.
- `results/downstream/cmapss_rul_decision/{decision.md, dry_run_report.{md,json}, manifest_real.json}`
  — decisión metodológica + manifest del builder.

### Pesados en Drive (NO versionados)

```
/content/drive/MyDrive/fm_fl_phmd/logs/downstream/cmapss_rul/
    downstream_cmapss_rul_from_scratch/
        run_info.json
        metrics.jsonl
        config.yaml
    downstream_cmapss_rul_linear_probing/
        run_info.json
        metrics.jsonl
        config.yaml
    downstream_cmapss_rul_full_finetuning_lr1e-5/
        run_info.json
        metrics.jsonl
        config.yaml

/content/drive/MyDrive/fm_fl_phmd/checkpoints/downstream/cmapss_rul/
    downstream_cmapss_rul_from_scratch/best.pt
    downstream_cmapss_rul_linear_probing/best.pt
    downstream_cmapss_rul_full_finetuning_lr1e-5/best.pt
```

> **Cierre operativo**: los 3 `run_info.json` ya están copiados
> bit-a-bit desde Drive a `results/downstream/cmapss_rul/<mode>/run_info.json`
> en el commit que cierra este bloque. Son la fuente canónica versionada.
