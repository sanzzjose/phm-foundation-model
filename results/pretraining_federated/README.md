# Pretraining federado SSL — resultados

> Resumen agregado del bloque federado FedAvg + FedProx cross-silo
> simulado. El análisis interno y deliberación están en
> `docs/decisions/pending_downstream_and_sampling.md` sec 4 (gitignored).

## Pilot FL FedProx mu=0.01 v0.2 — CERRADO / PASS (2026-05-26, commit `bb38367`)

Primera corrida productiva del pretraining federado SSL con **FedProx**
(`algorithm=fedprox`, `fedprox_mu=0.01`) sobre los mismos 36
PRETRAIN_SOURCE, mismos 10 clientes y mismos caps `capped_v23` que el
pilot FedAvg v0.1. Ejecutado en Colab Pro+ A100-SXM4-80GB tras el smoke
PASS (mismo notebook, 2 fases). Comparable bit-a-bit con el pilot FedAvg.

| campo | valor |
|---|---|
| `run_name` | `ssl_federated_pilot_fedprox_mu0_01_patchtst_phm` |
| `stage` | pilot |
| `algorithm` | **fedprox** |
| `fedprox_mu` | **0.01** |
| `fedprox_enabled` | true |
| `config_hash` | `678df3d8feb46f82` |
| `git_hash` | `bb383679ca1cc886762578d7ef7befdd8a3b87ed` (`bb38367`) |
| `git_dirty` | false |
| **`pilot_pass`** | **`true`** |
| `n_rounds` | 10 |
| `local_steps` | 50 (por cliente por ronda) |
| `total_local_optimizer_steps` | **4 989** (de 5 000; 11 omitidos por AMP overflow) |
| `param_count` | 801 808 |
| `cumulative_communication_mb` | 611.73 |
| `max_effective_bc_global` | 510 (≤ cap 512) |
| `aggregation_weights_policy_effective` | `final_client_weight_capped_v23` |
| `elapsed_seconds` | 1 061.57 (17.69 min en A100-SXM4-80GB) |

### Trayectoria de loss vs FedAvg pilot v0.1

| ronda | loss (FedAvg) | loss (FedProx) | recon (FedProx) | prox (FedProx) |
|---:|---:|---:|---:|---:|
| 1 | 0.8296 | 0.8395 | 0.8337 | 0.0058 |
| 10 | 0.7670 | **0.7423** | **0.7401** | 0.0022 |
| **Δ %** | **−7.55 %** | **−11.57 %** | **−11.23 %** | decrece (anclaje suave) |

**FedProx converge mejor que FedAvg en el mismo budget**: la loss SSL
agregada baja un 11.6 % vs el 7.6 % de FedAvg, sin destruir nada
(`reconstruction` y `prox` ambos finitos en todas las rondas). El
término proximal (`final_fedprox_loss_mean_weighted = 0.0022`) es
pequeño en magnitud pero estabilizador: la `fedprox_penalty_mean_weighted`
final es 0.4473, lo que indica un drift acotado y consistente.

### 6/6 pilot_checks PASS

- `all_clients_finite_loss` (10/10 clientes).
- `all_clients_opt_steps_gt0` (aero_engines 492, batteries 497,
  resto exactamente 500).
- `global_state_changes` (agregación efectiva por ronda).
- `max_effective_bc_within_cap` (510 ≤ 512; PHM14 C=317 sin OOM).
- `no_tt_in_plan` (0 TRANSFER_TARGET en el plan, verificado vs
  `processed_summary.role`).
- `aggregation_weights_reflect_caps` (pesos efectivos bit-a-bit con
  sec 7.bis CLAUDE.md: bearings 0.2492 cap 0.25, phm_challenges
  0.2441, cnc_milling 0.0050 piso, ...).

### Pesos de agregación (idénticos a FedAvg pilot v0.1)

| cliente | peso agregación FL | nota |
|---|---:|---|
| bearings | 0.2492 | cap por cliente 0.25 |
| phm_challenges | 0.2441 | cap por cliente 0.25 |
| misc | 0.1254 | |
| misc_industrial | 0.1131 | |
| aero_engines | 0.0997 | |
| batteries | 0.0627 | |
| hdd | 0.0481 | |
| wind | 0.0300 | |
| gearboxes | 0.0228 | |
| cnc_milling | 0.0050 | piso `min_client_presence` |

Idéntico bit-a-bit al pilot FedAvg v0.1 porque la política
`final_client_weight_capped_v23` no depende del algoritmo: el plan
sigue siendo el mismo (`audit_groups.json` v2.3 + caps cerrados).

### Cobertura de datasets durante los 5 000 optimizer steps

**Plan coverage 36/36** PRETRAIN_SOURCE. **Datasets sampled durante el
run 31/36**, idéntico al FedAvg pilot v0.1 (la cobertura depende del
sampler weighted, no del algoritmo FL):

| cliente | datasets sampled | total plan |
|---|---|---:|
| aero_engines | NCMAPSS | 1/1 |
| batteries | CALCE_CX2, FCLB19, NB1, NB14, UNIBO21 | 5/5 |
| bearings | IMS, JNUB, KAUG17, LGB20, PRONOSTIA, SEUGB17, UPM20, UPM23, XJTU-SY | 9/12 |
| cnc_milling | NMILL | 1/1 |
| gearboxes | ARAMIS20, PHMAP21 | 2/2 |
| hdd | HSF15 | 1/1 |
| misc | HIRFNASA15, OBDD17, SSPSNASA15 | 3/4 |
| misc_industrial | AC16, CBMv3, DFD15, PTRB19 | 4/5 |
| phm_challenges | PHM10, PHM15, PHME24, PPD18 | 4/4 |
| wind | PHM14 | 1/1 |

### Comparación FedAvg v0.1 vs FedProx v0.2

| métrica | FedAvg pilot v0.1 | FedProx pilot v0.2 | delta |
|---|---:|---:|---|
| pilot_pass | true | **true** | ✓ |
| total_local_optimizer_steps | 4 989 / 5 000 | **4 989 / 5 000** | idéntico |
| loss r1 → r10 | 0.8296 → 0.7670 | **0.8395 → 0.7423** | mejor reducción |
| **loss_delta_pct** | **−7.55 %** | **−11.57 %** | **+4.0 pp** |
| reconstruction_delta_pct | n/a (métrica nueva v0.2) | **−11.23 %** | — |
| prox term r10 | n/a | 0.0022 (~0.3 % de loss) | anclaje pequeño |
| elapsed_seconds | 1 138.05 | 1 061.57 | −6.7 % (más rápido) |
| cumulative_communication_mb | 611.73 | 611.73 | idéntico |
| pesos agregación | capped_v23 | capped_v23 | idénticos |
| datasets vistos | 31/36 | 31/36 | idéntico |

### Decisión post-pilot

Pilot FedProx v0.2 **CERRADO / PASS**. La loss SSL converge **mejor**
que en FedAvg con el mismo budget, el término proximal es estable y
los 6 checks PASS. Esto validó el pipeline FedProx end-to-end y
autorizó la evaluación downstream del ckpt FedProx pilot en CWRU +
HSG18, **ejecutada y cerrada en commit `25cdd81`: NO-GO full FedProx
vanilla**.

Detalle de la eval downstream FedProx (commit `25cdd81`):
`results/downstream/fl_fedprox_pilot_vs_central/`. Resumen:
- CWRU linear FedProx +1.3 pp vs FedAvg (marginal).
- CWRU full FedProx **−17.5 pp vs FedAvg** (empeora, posible sweet
  spot LR distinto para FedProx).
- HSG18 linear FedProx +11.6 pp vs FedAvg (mejora clara).
- HSG18 full FedProx +0.8 pp vs FedAvg pero sigue colapsando
  (recall clase 0 = 0.2574 < 0.30 umbral runbook).

**Veredicto formal**: **NO-GO full FedProx vanilla** (hipótesis B
estructural reconfirmada + regresión nueva en CWRU full). El problema
HSG18 sigue siendo la diversidad intra-cliente (cliente `hdd`
mono-dataset), no el algoritmo FL.

### Outputs

- En Drive (pesados, NO versionados):
  - `checkpoints/ssl_federated_pilot_fedprox_mu0_01/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/ckpt_final.pt`.
  - `logs/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/{run_info.json, metrics.jsonl, dry_run_report.json}`.
  - `logs/pretraining_federated/_stdout/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm.stdout.log`.
- En el repo (versionados bit-a-bit en el commit `ec2bf2f`):
  `results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/{run_info.json, dry_run_report.json, metrics_round_summary.json, README.md}`
  y `results/pretraining_federated/ssl_federated_smoke_fedprox_mu0_01/{run_info.json, dry_run_report.json}`
  (mismo patrón que el pilot FedAvg v0.1).

## Smoke FL FedProx mu=0.01 v0.2 — CERRADO / PASS (2026-05-26, commit `bb38367`)

Pre-validación del pipeline FedProx antes del pilot. 100 local steps
(2 rondas × 5 steps × 10 clientes) en A100, ~3.6 min.

| campo | valor |
|---|---|
| `run_name` | `ssl_federated_smoke_fedprox_mu0_01_patchtst_phm` |
| `stage` | smoke |
| `config_hash` | `e89d1661836518eb` |
| `git_hash` | `bb38367` |
| `total_local_optimizer_steps` | 100 |
| `final_loss_mean_weighted` | 0.8842 |
| `final_reconstruction_loss_mean_weighted` | 0.8837 |
| `final_fedprox_loss_mean_weighted` | 0.0005 (≪ loss SSL) |
| `final_fedprox_penalty_mean_weighted` | 0.0988 |
| `max_effective_bc_global` | 510 (≤ 512) |
| `aggregation_weights_policy_effective` | `final_client_weight_capped_v23` |
| `elapsed_seconds` | 215.15 |
| **`smoke_pass`** | **`true`** |

**6/6 smoke_checks PASS** (los mismos del FedAvg smoke v0.2 +
`aggregation_weights_reflect_caps`). Loss baja un 4.3 % entre rondas
(0.9241 → 0.8842), confirmando que el pipeline FedProx aprende sin
romperse en el budget mínimo. Habilitó el pilot.

## Ablación HSG18 lr1e-4 — CERRADA, hipótesis B confirmada (2026-05-25, commit `250868f`)

Ablación diagnóstica del downstream FL pilot HSG18 para discriminar entre:
- **A (adaptación)**: ckpt FL menos informativo → el backbone necesita
  más holgura (LR mayor) que el central para escapar del colapso a la
  clase mayoritaria.
- **B (estructural)**: el embedding FL HDD es insuficiente; cliente
  `hdd` mono-dataset (HSF15) no produce un encoder transferible.

1 corrida `full_finetuning lr_backbone=1e-4` sobre el ckpt
`ssl_federated_pilot_patchtst_phm/ckpt_final.pt`. Resultado:

```
test macro_f1            : 0.3333 (vs 0.5547 con lr1e-5, vs 0.6080 fed_linear)
confusion_matrix         : [[   0, 2288],
                            [   0, 2288]]
recall clase 0           : 0.0 %     (0 de 2 288 predicciones)
recall clase 1           : 100.0 %   (predice clase 1 a todas las 4 576 muestras)
best_epoch               : 1 / 20    (el modelo nunca supera la primera epoca)
```

**Colapso degenerado total**. Peor que `fed_linear` (−0.2747 macro_f1),
peor que `fed_full_lr1e-5` (−0.2213), peor que `from_scratch` (−0.2360).
**Hipótesis B confirmada**: más LR no rescata el embedding FL HDD, lo
acelera al colapso. El problema no es de adaptación; es estructural del
corpus FL en dominios mono-dataset.

**Implicación para full FL**: la opción (C) (subir `lr_backbone`) queda
descartada por sí sola. La causa raíz está en la **diversidad
intra-cliente** del corpus FL, no en hiperparámetros de adaptación. Vías
abiertas: (A) FedProx, (B) `min_client_presence` ajustada, o (D)
aceptar y reportar el límite estructural como hallazgo metodológico.

Detalle completo y diff vs lr1e-5 en
`results/downstream/fl_pilot_vs_central/README.md` (sección "Ablación
`lr_backbone=1e-4` confirma hipótesis B"). `run_info.json` real
versionado bit-a-bit en
`results/downstream/fl_pilot_vs_central/hsg18/full_finetuning_lr1e-4/run_info.json`.

## Eval downstream FL pilot — CWRU + HSG18 — CONDITIONAL (2026-05-25, commit `e9fb202`)

4 corridas de downstream classification sobre el ckpt
`ssl_federated_pilot_patchtst_phm/ckpt_final.pt`, 2 datasets × 2 modos
(`linear_probing` + `full_finetuning_lr1e-5`). El `from_scratch` se
mantiene del bloque downstream central anterior (mismo seed, backbone
random, mismo resultado esperado).

| dataset | from_scratch | central_linear | central_full_1e-5 | **fed_linear** | **fed_full_1e-5** | ratio fed/central full |
|---|---:|---:|---:|---:|---:|---:|
| CWRU  | 0.3503 | 0.7046 | **0.8292** | **0.4456** | **0.6635** | **80.0 %** |
| HSG18 | 0.5693 | 0.9056 | **0.9504** | **0.6080** | **0.5547** | **58.4 %** |

**Veredicto**: **CONDITIONAL global, NO-GO local en HSG18** (tras
ablación lr1e-4, ver sección anterior). El FL transfiere parcialmente
a CWRU (cliente `bearings` con 12 datasets en el plan, 9 sampled →
embedding generaliza a otro dataset bearing). En HSG18 el FL es
marginal en linear (+3.9 pp sobre baseline) y **destruye señal en
full_ft** (−1.5 pp); el `full_ft_lr1e-5` colapsa a la clase
mayoritaria (recall clase 0 = 24 %, clase 1 = 99 %). Causa
estructural: el cliente FL `hdd` tiene **1 solo dataset** (HSF15)
representando todo el dominio HDD; el embedding FL no captura
suficiente diversidad para HSG18.

**Hallazgo metodológico citable**: la transferencia FL es
**dominio-dependiente** y proporcional a la **diversidad
intra-cliente** del dominio. Donde hay varios datasets del mismo
dominio en un cliente, el FL transfiere; donde el cliente es
mono-dataset, no, y subir `lr_backbone` no lo rescata (la ablación
lr1e-4 lo confirma empíricamente con colapso degenerado).

**Decisión**: **NO autorizar full FL FedAvg vanilla todavía**.
Antes considerar:
- **A**: FedProx con `μ ≈ 0.01–0.1` (sec 15 CLAUDE.md, reduce drift).
- **B**: subir `min_client_presence` de 0.005 a 0.05 para clientes
  mono-dataset (hdd, wind, cnc_milling, aero_engines).
- ~~**C**: ablar `lr_backbone=1e-4` en HSG18 full_ft~~ — **DESCARTADA
  tras ablación** (colapso degenerado, hipótesis B confirmada).
- **D**: aceptar el resultado y lanzar full FL FedAvg sabiendo que
  HSG18 será débil. Honesto y citable, pero no exhausto.

Detalle completo, deltas/ratios numéricos, confusion matrices,
diagnóstico por dataset y comparativa con la hipótesis principal del
TFM en
`results/downstream/fl_pilot_vs_central/README.md`. Los 4
`run_info.json` reales versionados bit-a-bit + `summary.json` +
configs (commit `c70179f`) + notebook ejecutado (commit `e9fb202`) +
ablación lr1e-4 versionada bit-a-bit (commit `250868f`).

## Pilot FL v0.1 — CERRADO / PASS (2026-05-25, commit `9b6c9fb`)

Primera ejecución productiva del pretraining federado SSL con FedAvg
cross-silo simulado sobre los 36 PRETRAIN_SOURCE. Budget: **10 rondas
× 50 local steps × 10 clientes = 5 000 optimizer steps**, mismo que
el pilot central weighted.

| campo | valor |
|---|---|
| `run_name` | `ssl_federated_pilot_patchtst_phm` |
| `stage` | pilot |
| `config_hash` | `082ca64313105f05` |
| `git_hash` | `9b6c9fb1ec447f01b3833e9947b855a267e695a1` |
| **`pilot_pass`** | **`true`** |
| `n_rounds` | 10 |
| `local_steps` | 50 (por cliente por ronda) |
| `total_local_optimizer_steps` | **4 989** (de 5 000; 11 omitidos por AMP overflow) |
| `param_count` | 801 808 |
| `cumulative_communication_mb` | 611.73 |
| `max_effective_bc_global` | 510 (≤ cap 512) |
| `aggregation_weights_policy_effective` | `final_client_weight_capped_v23` |
| `elapsed_seconds` | 1 138.05 (18.97 min en A100-SXM4-80GB) |

**Loss SSL agregada**: `0.8296` (round 1) → `0.7670` (round 10) =
**−7.55 %**. Mínimo en round 10, sin meseta clara. Para comparar,
el pilot central weighted bajó **−25.2 %** en el mismo budget; FL
converge más lento por client drift, comunicación discreta y los
caps capped_v23 que redistribuyen pesos lejos del dominio dominante
(`bearings → 0.2492` cap 0.25, `phm_challenges → 0.2441`).

**6/6 pilot_checks PASS**:
- `all_clients_finite_loss` (10/10 clientes).
- `all_clients_opt_steps_gt0` (todos ≥ 493).
- `global_state_changes` (agregación efectiva por ronda).
- `max_effective_bc_within_cap` (510 ≤ 512; PHM14 C=317 sin OOM).
- `no_tt_in_plan` (0 TRANSFER_TARGET en el plan, verificado vs
  `processed_summary.role`).
- `aggregation_weights_reflect_caps` (pesos efectivos bit-a-bit con
  sec 7.bis CLAUDE.md: bearings 0.2492 cap 0.25, phm_challenges
  0.2441, cnc_milling 0.0050 piso, …).

**11 / 5 000 steps perdidos a AMP overflow** (aero_engines 7 +
batteries 1 + misc 3 = 0.22 %). Tolerados por `GradScaler` fp16,
no abortan. Mismo orden de magnitud que el central full (0.04 % en
100k steps).

**Decisión**: el pipeline pilot FL queda validado. La loss SSL es
métrica intrínseca; la evidencia principal del bloque FL será la
evaluación downstream del ckpt federado en CWRU y HSG18. Hasta ese
punto NO se afirma transferencia. Sin lanzar full FL todavía:
**CONDITIONAL → diagnóstico downstream antes de las ~6 h de A100
que cuesta el full**.

**Cobertura de datasets** durante los 5 000 optimizer steps:
**plan coverage 36/36** PRETRAIN_SOURCE (declarado por
`audit_groups.json` y verificado por el dry-run); **datasets sampled
durante el run 31/36** según `datasets_seen_by_client` del
`run_info.json`. Los 5 datasets pequeños fuera son efecto
estadístico del sampling weighted con caps capped_v23 en clientes
grandes (e.g. `bearings` con 12 datasets en el plan ve solo los más
ponderados en 500 local steps × 10 rondas). Un budget mayor (full
FL ~100k steps) debería incrementar la cobertura; no es señal de
fallo. Para referencia, el pilot central weighted con el mismo
budget vio **34/36** datasets (mismo efecto estadístico, distinto
patrón por la ausencia de clientes en el central).

Detalle completo y tabla por cliente / por ronda en
`results/pretraining_federated/ssl_federated_pilot/README.md`.
Artefactos versionados: `ssl_federated_pilot/run_info.json`,
`dry_run_report.json` y `metrics_round_summary.json`. Ckpts pesados
en Drive
(`checkpoints/ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt`).

## Smoke FL v0.2 — CERRADO / PASS, pesos capped reales (2026-05-24, commit `6415038`)

Re-ejecución del smoke tras el hotfix metodológico (commit `6415038`)
que corrigió la mutación del plan global durante la normalización
intra-cliente. Esta versión usa los pesos `final_client_weight_capped_v23`
reales del audit v2.3, no la uniformidad implícita que tuvo el v0.1.

| campo | valor |
|---|---|
| run_name | `ssl_federated_smoke_patchtst_phm` |
| stage | smoke |
| local steps solicitados | 2 × 5 × 10 = 100 |
| `total_local_optimizer_steps` (efectivos aplicados) | **99** |
| `param_count` (backbone) | 801 808 |
| `aggregation_weight_policy` | `final_client_weight` |
| `plan_policy_unique` | `["final_client_weight_capped_v23"]` |
| `config_hash` | `f0abda1a47ed272e` |
| `git_hash` | `6415038` (hotfix V1-V5) |
| `elapsed_seconds` | ~93 |
| `max_effective_bc_global` | 510 (≤ cap 512) |
| `amp_nonfinite_grad_steps_total` | **batteries=1, resto=0** |
| **`smoke_pass`** | **true** |

### opt_steps efectivos por cliente

100 local steps solicitados, **99 optimizer steps aplicados**. El step
omitido corresponde a un grad no finito bajo AMP en `batteries`; el
GradScaler lo saltó (comportamiento normal del scaler fp16), no aborta.

| cliente | opt_steps efectivos | amp_nonfinite_grad_steps |
|---|---:|---:|
| aero_engines | 10 | 0 |
| batteries | **9** | **1** |
| bearings | 10 | 0 |
| cnc_milling | 10 | 0 |
| gearboxes | 10 | 0 |
| hdd | 10 | 0 |
| misc | 10 | 0 |
| misc_industrial | 10 | 0 |
| phm_challenges | 10 | 0 |
| wind | 10 | 0 |

### Pesos de agregación por cliente (última ronda)

Coinciden bit-a-bit con los caps cerrados del audit v2.3 (sec 7.bis
CLAUDE.md):

| cliente | peso agregación FL | nota |
|---|---:|---|
| bearings | 0.2492 | cap por cliente 0.25 |
| phm_challenges | 0.2441 | cap por cliente 0.25 |
| misc | 0.1254 | |
| misc_industrial | 0.1131 | |
| aero_engines | 0.0997 | |
| batteries | 0.0627 | |
| hdd | 0.0481 | |
| wind | 0.0300 | |
| gearboxes | 0.0228 | |
| cnc_milling | 0.0050 | piso `min_client_presence` |

### Trayectoria de loss

```
round 1/2: loss_mean_weighted = 0.9867   global_norm_delta = +0.103
round 2/2: loss_mean_weighted = 0.8887   global_norm_delta = +0.026
```

Reducción ~10% en 2 rondas. **No es comparable numéricamente con el
v0.1** porque allí los pesos efectivos eran uniformes (la loss agregada
era la media simple, no la ponderada por caps). En v0.2 los dominios
`bearings` y `phm_challenges` dominan (~0.49 del total combinado), lo
cual desplaza la loss agregada hacia los valores que esos dos clientes
producen sobre sus datasets ruidosos.

### 6 smoke_checks (todos OK)

1. `all_clients_finite_loss` — los 10 clientes con al menos una loss
   finita.
2. `all_clients_opt_steps_gt0` — los 10 con `optimizer_steps > 0`
   (batteries = 9 por el step AMP saltado, resto = 10). El check
   exige `> 0`, no exactamente 10.
3. `global_state_changes` — `state_dict` cambia tras agregación en
   al menos una ronda.
4. `max_effective_bc_within_cap` — 510 ≤ 512.
5. `no_tt_in_plan` — 0 violaciones via `processed_summary.role`.
6. **`aggregation_weights_reflect_caps`** (nuevo en v0.2) — `max-min`
   de los pesos por cliente ≈ 0.244 (rango 0.005..0.249), muy por
   encima del umbral 0.01. Confirma que los caps se aplican
   correctamente y descarta el bug de mutación del plan.

> **Nota sobre el skip AMP**: el step omitido en `batteries` por grad
> no finito **no invalida el smoke**. El criterio PASS exige
> `optimizer_steps > 0` por cliente, loss finita, state change tras
> agregación, `B*C ≤ cap`, 0 TT en el plan y pesos capped reflejando
> el audit v2.3. Los seis se cumplen.

### Outputs v0.2

- `results/pretraining_federated/ssl_federated_smoke_v0_2/run_info.json`
  (citable, este commit). **Este es el run_info que se debe usar para
  cualquier análisis cuantitativo del smoke FL**.
- Drive: `checkpoints/ssl_federated_smoke/.../ckpt_final.pt` y
  `logs/pretraining_federated/.../{metrics.jsonl, run_info.json}`
  (sobreescritos respecto al v0.1 al relanzar; el ckpt v0.1 en Drive
  ya no es recuperable porque la corrida v0.2 escribe en el mismo
  path. Esto es intencional: el ckpt v0.1 reflejaba pesos uniformes
  implícitos y no debería usarse).

---

## Smoke FL v0.1 — HISTÓRICO con caveat metodológico (2026-05-24, commit `20ebe5a`)

> **Estado: histórico con caveat**. Pipeline FL end-to-end validado,
> pero los **pesos de agregación efectivos fueron uniformes implícitos**
> (todos los clientes pesaron 0.1), no los `final_client_weight_capped_v23`
> que el `run_info.json` reportaba.

### Bug descubierto y corregido

`FederatedClient.local_train` normalizaba el `final_dataset_weight`
intra-cliente **mutando los dicts del plan global** in-place. Tras
los 10 clientes de la primera ronda, todos los pesos del plan sumaban
1 dentro de cada cliente. Cuando el server llamaba a
`compute_aggregation_weights(plan, policy="final_client_weight")`,
obtenía 1.0 por cliente y normalizaba a 1/10 cada uno → equivalente
a `policy=uniform`.

**Impacto**:
- Pipeline FedAvg end-to-end ✓ válido (FedAvg se ejecutó correctamente).
- Cobertura, latencia, comunicación, batch adaptativo ✓ válidos.
- `aggregation_weight_policy="final_client_weight"` reportado en
  `run_info.json` ✗ **engañoso** — la política efectiva fue uniform.
- `loss_mean_weighted` calculada con pesos uniformes, no con caps.

**Fix**: helper puro `normalize_plan_subset_for_loader` (commit
`6415038`) que devuelve copias shallow sin mutar el plan, más un
smoke check `aggregation_weights_reflect_caps` que detecta el síntoma
si reapareciera.

### Outputs v0.1 (preservados como evidencia)

- `results/pretraining_federated/ssl_federated_smoke/run_info.json`
  (commit `29a6c64`, `git_hash=20ebe5a`, pre-hotfix): contiene los 5
  smoke_checks originales y `smoke_pass=true`. **No contiene los
  campos `aggregation_weights_by_client_last_round` ni
  `plan_policy_unique` ni el sexto check
  `aggregation_weights_reflect_caps`** (esos campos se introdujeron
  en el hotfix `6415038`). El bug NO es visible inspeccionando este
  run_info: se demuestra analíticamente leyendo el código pre-hotfix
  de `FederatedClient.local_train` y verificando que mutaba
  `final_dataset_weight` in-place, lo cual hacía que
  `compute_aggregation_weights` viera el plan ya normalizado a sumar
  1 intra-cliente y devolviera 0.1 por cliente (= uniform).

**No usar el ckpt v0.1 ni las métricas v0.1 para narrativa cuantitativa
del TFM**. Sirve solo como evidencia documentada de que el pipeline
end-to-end funcionaba antes de detectar el bug de agregación.

---

## Detalles técnicos comunes a ambos smokes

> Esta sección NO contiene métricas. Para los números del v0.2 ver la
> primera sección. Para los del v0.1, ver el `run_info.json` v0.1.

Características compartidas (no varían entre v0.1 y v0.2):

- **Setup**: 10 clientes (audit v2.3 sec 7.bis), 2 rondas × 5 steps
  locales × 10 clientes = **100 local steps solicitados**. Los
  `optimizer_steps` efectivos pueden ser menores si AMP omite algún
  step con grad no finito (comportamiento normal del GradScaler fp16).
- **Backbone**: `patchtst_phm_base`, 801 808 parámetros entrenables.
- **Política declarada**: `aggregation_weight_policy=final_client_weight`
  (la efectiva difiere: v0.1 fue uniform implícito por bug, v0.2 es
  capped real).
- **Batch adaptativo**: `B*C ≤ max_channel_batch=512`, `min_batch_size=1`.
  Cubre PHM14 (C=317, B=1) y HSF15/PHME24 (C=17, B=30, B*C=510).
- **Política de checks comunes** (los 5 originales):
  1. `all_clients_finite_loss`.
  2. `all_clients_opt_steps_gt0`.
  3. `global_state_changes` (state_dict varía tras agregación).
  4. `max_effective_bc_within_cap`.
  5. `no_tt_in_plan` (verificado contra `processed_summary.role`).
- **Check exclusivo del v0.2**: `aggregation_weights_reflect_caps`
  (introducido en commit `6415038`; v0.1 no lo tiene).

### Datasets vistos por cliente (no varían smoke v0.1 ↔ v0.2)

| cliente | datasets vistos | total disponibles |
|---|---|---:|
| aero_engines | NCMAPSS | 1/1 |
| batteries | CALCE_CX2, FCLB19, NB1, NB14, UNIBO21 | 5/5 |
| bearings | IMS, PRONOSTIA, UPM23, XJTU-SY | 4/12 |
| cnc_milling | NMILL | 1/1 |
| gearboxes | ARAMIS20, PHMAP21 | 2/2 |
| hdd | HSF15 | 1/1 |
| misc | HIRFNASA15, OBDD17 | 2/4 |
| misc_industrial | AC16, DFD15, PTRB19 | 3/5 |
| phm_challenges | PHM10, PHME24 | 2/4 |
| wind | PHM14 | 1/1 |

Los clientes con pocos datasets (aero_engines, hdd, wind, cnc_milling)
ven todos los suyos en 10 steps. Los grandes (bearings, misc,
misc_industrial, phm_challenges) ven un subconjunto coherente con su
peso intra-cliente; cobertura completa esperada en pilot/full.

### Lo que el smoke valida (citable para el TFM, solo v0.2)

1. Pipeline FL cross-silo simulado funciona end-to-end con los 36
   `PRETRAIN_SOURCE` reales y los 10 clientes audit v2.3.
2. Agregación FedAvg ponderada por `final_client_weight` con caps
   cerrados (0.10 / 0.25 / 0.005) **efectivamente aplicada** (max-min
   de pesos por cliente = 0.244, no uniforme).
3. Batch adaptativo gestiona PHM14 (C=317) y HSF15/PHME24 (C=17, B=30)
   sin OOM ni excesos del cap.
4. Logging robusto (loss no finita = abort, AMP scaler tolerado,
   per-step `optimizer_applied`) idéntico al SSL central.
5. Verificación de "0 TRANSFER_TARGET" via `processed_summary.role`.

### Lo que el smoke NO valida

- Convergencia productiva (100 steps totales es demasiado poco).
- Comparación cuantitativa central vs federado sobre `TRANSFER_TARGET`.
- Coste de comunicación real (sin networking, solo estimación).

Esos puntos se cubren en pilot/full y en la evaluación downstream con
el ckpt federado.

## Próximos pasos

1. **Pilot FL** (config `training/configs/ssl_federated_pilot.yaml`,
   10 rondas × 50 steps × 10 clientes = 5 000 totales, gateado por
   stage=pilot). **NO ejecutar sin autorización explícita** — primer
   entrenamiento federado significativo.
2. **Comparación contra central**: tras pilot, usar el ckpt federado
   como backbone en linear_probing / full_finetuning sobre CWRU /
   HSG18 (los TT donde el central confirmó transferencia) y comparar
   con los resultados del ckpt central full.
3. **Full FL** solo si pilot pasa criterios análogos al pilot central
   (loss converge, sin colapso, coverage adecuada). Presupuesto
   equivalente al central full: 100 rondas × 100 steps × 10 clientes =
   100 k.

## Timeline

| fecha | commit | hito |
|---|---|---|
| 2026-05-24 | 2698ae8 | fix downstream hardening + CMAPSS RUL decisión |
| 2026-05-24 | 1dd3f0e | feat FL FedAvg skeleton + tests + smoke config |
| 2026-05-24 | 20ebe5a | fix data_cfg no definido en cmd_smoke |
| 2026-05-24 | f0d7800 | docs(fl): README v0.1 del smoke FL CERRADO / PASS |
| 2026-05-24 | 29a6c64 | results(fl) v0.1: run_info.json del smoke FL pre-hotfix |
| 2026-05-24 | 6415038 | **fix(fl) hotfix V1-V5**: no mutar plan + logging agg + pilot semántica |
| 2026-05-24 | 4c02f37 | docs(fl): README v0.2 con caveat del smoke v0.1 |
| 2026-05-24 | fdc2224 | (sobreescribió run_info v0.1 con v0.2 en mismo path; reorganizado luego) |
| 2026-05-24 | (este commit W) | results: reorganizar v0.2 a path separado + README corregido |
