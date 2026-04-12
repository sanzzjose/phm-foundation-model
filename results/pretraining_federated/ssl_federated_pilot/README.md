# Pilot FL — CERRADO / PASS (2026-05-25, commit `9b6c9fb`)

Primera ejecución productiva del pretraining federado SSL con FedAvg
cross-silo simulado sobre los 36 PRETRAIN_SOURCE. Budget: 10 rondas ×
50 local steps × 10 clientes = **5 000 optimizer steps locales**, el
mismo presupuesto que el pilot weighted central.

Run en Colab Pro+ A100-SXM4-80GB.

## Resumen

| campo | valor |
|---|---|
| `run_name` | `ssl_federated_pilot_patchtst_phm` |
| `stage` | `pilot` |
| `config_hash` | `082ca64313105f05` |
| `git_hash` | `9b6c9fb1ec447f01b3833e9947b855a267e695a1` |
| `git_dirty` | `false` |
| `pilot_pass` | **`true`** |
| `n_rounds` | 10 |
| `local_steps` (por cliente por ronda) | 50 |
| `requested_local_optimizer_steps` | 5 000 |
| `total_local_optimizer_steps` | **4 989** |
| `param_count` | 801 808 |
| `elapsed_seconds` | 1 138.05 (18.97 min) |
| `cumulative_communication_mb` | 611.73 |
| `max_effective_bc_global` | 510 (≤ cap 512) |
| `aggregation_weight_policy` declarada | `final_client_weight` |
| `aggregation_weights_policy_effective` aplicada | **`final_client_weight_capped_v23`** |
| `checkpoint_final` (Drive) | `.../checkpoints/ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt` |

## 6/6 pilot_checks PASS

| check | resultado |
|---|---|
| `all_clients_finite_loss` | OK (10/10 clientes con loss finita) |
| `all_clients_opt_steps_gt0` | OK (todos ≥ 493 steps) |
| `global_state_changes` | OK (agregación efectiva por ronda) |
| `max_effective_bc_within_cap` | OK (510 ≤ 512, PHM14 C=317 sin OOM) |
| `no_tt_in_plan` | OK (0 TRANSFER_TARGET en el plan; verificado vía `processed_summary.role`) |
| `aggregation_weights_reflect_caps` | OK (caps cerrados del audit v2.3 aplicados) |

## Optimizer steps + AMP overflow por cliente

| cliente | opt_steps | amp_nonfinite_grad_steps | aggregation_weight_last_round |
|---|---:|---:|---:|
| bearings | 500 | 0 | **0.2492** (cap 0.25) |
| phm_challenges | 500 | 0 | **0.2441** |
| misc | 497 | 3 | 0.1254 |
| misc_industrial | 500 | 0 | 0.1131 |
| aero_engines | 493 | 7 | 0.0997 |
| batteries | 499 | 1 | 0.0627 |
| hdd | 500 | 0 | 0.0481 |
| wind | 500 | 0 | 0.0300 |
| gearboxes | 500 | 0 | 0.0228 |
| cnc_milling | 500 | 0 | **0.0050** (piso `min_client_presence`) |

- **11 steps perdidos a AMP overflow** (7 + 1 + 3) sobre 5 000 = 0.22 %.
  Comportamiento esperado del `GradScaler` fp16; los step se omiten,
  no abortan. Mismo orden de magnitud que el central full (0.04 % en
  100k steps).
- Distribución de pesos efectivos **bit-a-bit con la sec 7.bis de
  `CLAUDE.md`** (caps cerrados audit v2.3, ver tabla central full
  para comparación).

## Convergencia por ronda

| round | loss_mean_weighted | global_norm_delta_signed | cumulative_comm_mb |
|---:|---:|---:|---:|
|  1 | 0.8296 | +0.148 | 61.2 |
|  2 | 0.8141 | −0.022 | 122.3 |
|  3 | 0.7982 | −0.071 | 183.5 |
|  4 | 0.7920 | −0.039 | 244.7 |
|  5 | 0.7845 | −0.023 | 305.9 |
|  6 | 0.7908 | −0.001 | 367.0 |
|  7 | 0.7977 | −0.014 | 428.2 |
|  8 | 0.7817 | +0.007 | 489.4 |
|  9 | 0.7788 | +0.005 | 550.6 |
| 10 | **0.7670** | +0.044 | 611.7 |

- **`loss_delta_pct = −7.55 %`** entre round 1 y round 10.
- Mínimo histórico en round 10 (0.7670), sin meseta clara aún.
- Pequeño rebote rondas 5→7 (0.7845 → 0.7977) seguido de recuperación
  rondas 8→10. Patrón consistente con FedAvg cross-silo: la
  agregación absorbe parte del progreso local cada ronda, lo cual
  produce micro-oscilaciones.

## Dinámica del global state norm

| ronda | norm_before | norm_after | Δ | observación |
|---:|---:|---:|---:|---|
|  1 | 280.0047 | 280.1532 | +0.148 | init random + primera agregación |
|  2 | 280.1532 | 280.1308 | −0.022 | |
|  3 | 280.1308 | 280.0601 | −0.071 | mayor contracción del bloque |
|  4 | 280.0601 | 280.0216 | −0.039 | |
|  5 | 280.0216 | 279.9988 | −0.023 | norm baja por debajo de 280 |
|  6 | 279.9988 | 279.9978 | −0.001 | quasi-meseta |
|  7 | 279.9978 | 279.9833 | −0.014 | mínimo de norm (279.98) |
|  8 | 279.9833 | 279.9905 | +0.007 | norm rebota leve |
|  9 | 279.9905 | 279.9954 | +0.005 | |
| 10 | 279.9954 | 280.0393 | +0.044 | regresa cerca del valor inicial |

Rango total `[279.9833, 280.1532]` = **0.17 unidades** de norma sobre
~280 (variación relativa del **0.06 %**). El backbone se mueve, pero
en una región muy estrecha del espacio de pesos. Esto es señal
cualitativa de que la agregación FedAvg con caps capped_v23 mantiene
estabilidad numérica; no una señal de que el modelo aprenda mucho.
La métrica clave de progreso sigue siendo `loss_mean_weighted`.

## Loss por cliente en round 10

Loss SSL del último cliente en round 10 (menor = mejor reconstrucción)
y `drift_by_client` (norma `||θ_local_post − θ_global_pre||`, distancia
recorrida durante los 50 local steps):

| cliente | loss round 10 | drift round 10 | n_datasets | comentario |
|---|---:|---:|---:|---|
| `wind` | **0.066** | 1.62 | 1 (PHM14) | mejor reconstrucción; C=317 quizá facilita |
| `batteries` | 0.480 | 1.39 | 5 | señal limpia |
| `misc` | 0.596 | 1.56 | 3 | |
| `aero_engines` | 0.732 | 1.84 | 1 (NCMAPSS) | trayectorias largas |
| `phm_challenges` | 0.741 | 1.18 | 4 | dominio heterogéneo |
| `hdd` | 0.827 | 2.11 | 1 (HSF15) | drift más alto |
| `bearings` | 0.920 | 1.14 | 12 | cliente más grande; drift bajo |
| `cnc_milling` | 0.950 | 2.04 | 1 (NMILL) | señal mecánica fina |
| `misc_industrial` | 0.971 | 1.66 | 5 | |
| `gearboxes` | 0.990 | 1.31 | 2 | |

Valores de drift en `[1.14, 2.11]` indican drift moderado y
comparable entre clientes; **ningún cliente diverge significativamente**
del global. Dinámica esperable de FedAvg con `lr=3e-4` y 50 steps
locales.

## Interpretación

1. **El pipeline pilot federado queda VALIDADO**: 6/6 checks PASS,
   pesos efectivos coinciden bit-a-bit con la política cerrada,
   convergencia estable, ningún cliente queda sin participar, batch
   adaptativo opera correctamente bajo carga FL real, AMP overflow
   tolerado.

2. **La loss SSL baja −7.55 %** en el mismo budget en el que el
   **pilot central bajó −25.2 %**. Esto es esperable: FedAvg
   cross-silo sobre 36 PRETRAIN_SOURCE heterogéneos converge más
   lento que central por:
   - **Client drift**: cada cliente optimiza sobre un subset
     dataset-heterogéneo y la agregación promedia gradientes
     potencialmente conflictivos entre dominios.
   - **Comunicación menos eficiente que SGD continuo**: cada ronda
     resetea efectivamente los optimizers locales (FedAvg sin
     momentum coordinated).
   - **Caps capped_v23 redistribuyen pesos**: bearings cap 0.25 y
     phm_challenges cap 0.244 dan menos peso al dataset dominante
     (PHM10 está dentro de phm_challenges con peso intra-cliente
     limitado adicionalmente).

3. **La loss SSL es métrica intrínseca**. No declara nada todavía
   sobre transferencia. **La evidencia principal del bloque FL será
   la evaluación downstream del ckpt federado en CWRU y HSG18**
   (los 2 TT donde el SSL central confirmó transferencia clara).
   Hasta ese punto, NO se afirma que el federado transfiera.

4. **Comparación cualitativa pilot central vs pilot federado** sobre
   el mismo budget de 5 000 optimizer steps:

   | dimensión | pilot central | pilot FL |
   |---|---|---|
   | Δloss | −25.2 % | −7.55 % |
   | elapsed | 691.6 s | 1 138.05 s |
   | overhead | n/a | +64 % por comunicación entre clientes |
   | comm acumulada | 0 MB | 611.73 MB |
   | **plan coverage (sampling plan)** | **36/36** | **36/36** |
   | **datasets sampled durante el run** | **34/36** (2 pequeños fuera por probabilidad weighted) | **31/36** (5 datasets pequeños fuera dentro de clientes grandes) |
   | clientes participantes | n/a | 10/10 |
   | max abs_err caps | 0.0070 | ≤ 0.0001 (caps bit-a-bit) |

   El central converge más rápido en menos tiempo. El FL ve **menos
   datasets distintos en 5 000 steps** que el central (31 vs 34): no
   es un fallo del FL, es un efecto estadístico del sampling weighted
   dentro de cada cliente, agravado en clientes grandes con datasets
   pequeños (p.ej. `bearings` tiene 12 datasets en el plan, pero
   varios de los más pequeños no aparecen en 500 steps locales × 10
   rondas). Un full o un pilot más largo aumentaría esta cobertura.
   Lo que **sí** garantiza el FL es: 10/10 clientes participan en
   cada ronda, 0 TT en el plan, caps `capped_v23` aplicados con
   tolerancia ≤ 1e-4, y B*C ≤ 512 en todos los pasos.

## Decisión post-pilot

`pilot_pass=true` con loss decreciente, todos los checks duros OK y
pesos efectivos bit-a-bit con la política cerrada. **Recomendación:
CONDITIONAL → evaluar downstream del ckpt pilot en CWRU y HSG18
antes de lanzar el full FL.**

Si el downstream con el ckpt pilot ya muestra señal de transferencia
(`linear_probing > from_scratch` en CWRU/HSG18, similar al patrón
central), el full FL es **GO** con alta confianza.

Si el downstream con el ckpt pilot no muestra señal o muestra una
señal claramente inferior al central pilot, el full FL queda
**CONDITIONAL** y conviene revisar antes:

- la duración total efectiva (¿100k steps son suficientes para FL?
  Quizá necesitamos 150k+);
- el algoritmo (¿probar FedProx con `μ` pequeño antes de meter más
  rondas en FedAvg?);
- la política de sampling intra-cliente.

**NO lanzar full FL hasta tener este diagnóstico downstream.**

## Artefactos

### Versionados en el repo

- `results/pretraining_federated/ssl_federated_pilot/run_info.json`
  — copia **bit-a-bit** del `run_info.json` real de Drive
  (este directorio).
- `results/pretraining_federated/ssl_federated_pilot/dry_run_report.json`
  — copia bit-a-bit del dry-run pre-train.
- `results/pretraining_federated/ssl_federated_pilot/metrics_round_summary.json`
  — por-ronda parseado del `metrics.jsonl` real (incluye
  `loss_by_client`, `drift_by_client`,
  `global_state_norm_before/after`, `ts` por ronda).
- `results/pretraining_federated/ssl_federated_pilot/README.md` —
  este documento.
- `training/configs/ssl_federated_pilot.yaml` — config del run
  (`config_hash=082ca64313105f05`).
- `notebooks/runbooks/fl_pilot_commit7_colab.md` — runbook ejecutado.

### Pesados en Drive (NO versionados)

```
/content/drive/MyDrive/fm_fl_phmd/
  logs/pretraining_federated/ssl_federated_pilot_patchtst_phm/
      run_info.json          (idéntico al versionado en el repo)
      metrics.jsonl          (curvas por ronda + train_step JSONL)
      dry_run_report.json    (idéntico al versionado en el repo)
  logs/pretraining_federated/_stdout/
      ssl_federated_pilot_patchtst_phm.stdout.log
      ssl_federated_pilot_dryrun.stdout.log
  checkpoints/ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/
      ckpt_round005.pt
      ckpt_round010.pt
      ckpt_final.pt          (=> ckpt_round010.pt, ~9.3 MB)
```

> **Resuelto**: `dry_run_report.json` ya está versionado en el repo
> (copia bit-a-bit desde Drive). Solo el `metrics.jsonl` completo y
> los `.pt` quedan exclusivamente en Drive por tamaño.

## Siguiente paso

Evaluar el ckpt federado en CWRU y HSG18 (los 2 TT primary
classification con SSL central confirmado). Si la transferencia se
sostiene, el commit posterior arrancará el full FL (100k steps);
si no, diagnóstico antes de gastar las ~6 h estimadas de A100.
