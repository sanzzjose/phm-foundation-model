# Analisis post-hoc del run `ssl_central_full_patchtst_phm`

Este analisis se genero post-hoc a partir de los logs del run. **No se repitio el entrenamiento ni se modifico el checkpoint.** Los conteos de steps con grad no finito fueron gestionados por el `GradScaler` de AMP (comportamiento normal en fp16); el run es valido siempre que el state_dict del checkpoint sea finito.

## Metadatos del run

| Campo | Valor |
|---|---|
| `git_hash` | `b3767b9268fdec22489129fe428eff820382b98e` |
| `git_dirty` | `True` |
| `config_hash` | `9ed84508a6820265` |
| `param_count` | `801808` |
| `stage` | `full` |
| `optimizer_steps` | `99961` |
| `skipped_steps` | `0` |
| `amp_overflow_steps` | `39` |
| `amp_nonfinite_grad_steps` | `None` |
| `datasets_seen_count` | `36` |
| `clients_seen_count` | `10` |
| `max_effective_bc` | `510` |
| `elapsed_seconds` | `12875.8` |

## Inspeccion de `metrics.jsonl`

- Lineas totales: 100040
- Registros de step: 100000
- Registros de distribution: 40
- Steps con grad no finito (total): 39
- Por tipo: {'inf': 37, 'nan': 2}
- Top datasets afectados: [('NB14', 7), ('PHM14', 7), ('AC16', 6), ('UNIBO21', 5), ('HSF15', 3)]

## Outliers de loss finita

- Umbral `huge_loss_threshold = 1000.0`. Steps con loss >= umbral: 2.
  - step 28003 (PHM14): 3147966.0000
  - step 78176 (PHM14): 3147966.0000

## Loss por bucket

Medias por bucket de step. La columna `mean_clean` excluye steps con `optimizer_applied=False` o `amp_nonfinite_grad=True` (disponible solo en logs nuevos, post-patch).

| rango | n | mean | median | p10 | p90 | min | max | mean_clean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0-2000 | 1999 | 0.7974 | 0.8305 | 0.5582 | 1.0099 | 0.0041 | 1.3485 | 0.7974 |
| 2000-7000 | 5000 | 0.6504 | 0.7098 | 0.1733 | 0.9947 | 0.0013 | 1.1657 | 0.6504 |
| 7000-12000 | 5000 | 0.5760 | 0.5695 | 0.0881 | 0.9820 | 0.0015 | 1.1404 | 0.5760 |
| 22000-27000 | 5000 | 0.4628 | 0.4445 | 0.0591 | 0.8226 | 0.0007 | 1.0913 | 0.4628 |
| 47000-52000 | 5000 | 0.4115 | 0.4548 | 0.0461 | 0.7315 | 0.0004 | 1.0062 | 0.4115 |
| 72000-77000 | 5000 | 0.4190 | 0.4399 | 0.0470 | 0.7372 | 0.0003 | 1.1403 | 0.4190 |
| 95000-100000 | 5000 | 0.4167 | 0.4391 | 0.0433 | 0.7434 | 0.0003 | 1.0175 | 0.4167 |

## Checkpoint

- `checkpoint_state_dict_all_finite`: `True`

## Conclusion

checkpoint usable; logging/robustness fixes recomendados para futuros runs (JSONL estricto, grad_norm null en steps con AMP overflow, final_client_weight coherente con groupby).
