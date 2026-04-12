# Pilot FL SSL FedProx mu=0.01 v0.2 — CERRADO / PASS (2026-05-26)

Primera corrida productiva del pretraining federado SSL con **FedProx**
(`algorithm=fedprox`, `fedprox_mu=0.01`) sobre los mismos 36
PRETRAIN_SOURCE, mismos 10 clientes y mismos caps `capped_v23` que el
pilot FedAvg v0.1. Ejecutado en Colab Pro+ A100-SXM4-80GB tras el smoke
PASS (mismo notebook, 2 fases). **Comparable bit-a-bit con el pilot FedAvg**:
solo cambia el algoritmo + termino proximal; plan, semillas, batch
adaptativo, pesos de agregacion y volumen de comunicacion son idénticos.

## Cabecera del run

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
| `local_steps` | 50 |
| `clients_per_round` | all (10) |
| `total_local_optimizer_steps` | **4 989** (de 5 000; 11 omitidos por AMP overflow) |
| `param_count` | 801 808 |
| `cumulative_communication_mb` | 611.73 |
| `max_effective_bc_global` | 510 (≤ cap 512) |
| `aggregation_weights_policy_effective` | `final_client_weight_capped_v23` |
| `elapsed_seconds` | 1 061.57 (17.69 min en A100-SXM4-80GB) |

## 6/6 pilot_checks PASS

| check | resultado | observacion |
|---|---|---|
| `all_clients_finite_loss` | OK | 10/10 clientes con loss finita en todas las rondas |
| `all_clients_opt_steps_gt0` | OK | aero_engines 492, batteries 497, resto exactamente 500 |
| `global_state_changes` | OK | el `state_dict` global cambia tras cada agregacion |
| `max_effective_bc_within_cap` | OK | 510 ≤ 512; PHM14 C=317 sin OOM |
| `no_tt_in_plan` | OK | 0 TRANSFER_TARGET en el plan, via `processed_summary.role` |
| `aggregation_weights_reflect_caps` | OK | pesos efectivos bit-a-bit con audit v2.3 (max-min = 0.244, ≫ 0.01) |

## Trayectoria por ronda (loss / reconstruction / proximal)

| ronda | loss | reconstruction | fedprox_loss | penalty | global_norm Δ |
|---:|---:|---:|---:|---:|---:|
| 1 | 0.8395 | 0.8337 | 0.0058 | 1.1568 | +0.090 |
| 2 | 0.8189 | 0.8165 | 0.0025 | 0.4955 | −0.043 |
| 3 | 0.8080 | 0.8061 | 0.0019 | 0.3801 | −0.053 |
| 4 | 0.7957 | 0.7942 | 0.0016 | 0.3109 | −0.052 |
| 5 | 0.7921 | 0.7903 | 0.0017 | 0.3470 | −0.056 |
| 6 | 0.7889 | 0.7870 | 0.0019 | 0.3709 | −0.043 |
| 7 | 0.7856 | 0.7835 | 0.0021 | 0.4206 | −0.028 |
| 8 | 0.7609 | 0.7585 | 0.0024 | 0.4828 | −0.024 |
| 9 | 0.7562 | 0.7536 | 0.0026 | 0.5162 | −0.042 |
| **10** | **0.7423** | **0.7401** | **0.0022** | **0.4473** | **−0.021** |

- **loss_delta_pct = −11.57 %** (r1 → r10).
- **reconstruction_delta_pct = −11.23 %** (r1 → r10).
- El término proximal arranca alto en r1 (drift grande respecto al
  init random) y baja drásticamente en r2 (modelo local ya cerca del
  global tras la primera ronda). De r2 en adelante oscila en
  [0.31, 0.52], con peso `0.5 * mu * penalty = 0.0015..0.0026`, es
  decir **≪ 0.3 % de la loss SSL**. Anclaje suave y estable.
- La `global_norm` baja monótonamente desde r2 hasta r10, lo cual
  refleja convergencia ordenada del state_dict agregado.

## Comparacion vs Pilot FedAvg v0.1 (commit `9b6c9fb`)

| metrica | FedAvg v0.1 | **FedProx v0.2** | delta |
|---|---:|---:|---|
| pilot_pass | true | **true** | ✓ |
| total_local_optimizer_steps | 4 989 / 5 000 | **4 989 / 5 000** | idéntico |
| loss r1 → r10 | 0.8296 → 0.7670 | **0.8395 → 0.7423** | mejor reducción |
| **loss_delta_pct** | **−7.55 %** | **−11.57 %** | **+4.0 pp** |
| reconstruction_delta_pct | n/a (métrica nueva v0.2) | **−11.23 %** | — |
| prox term r10 | n/a | 0.0022 (~0.3 % de loss) | anclaje pequeno |
| elapsed_seconds | 1 138.05 | 1 061.57 | −6.7 % (más rápido) |
| cumulative_communication_mb | 611.73 | 611.73 | idéntico |
| pesos agregacion | capped_v23 | capped_v23 | idénticos |
| datasets vistos | 31/36 | 31/36 | idéntico |

**Conclusión cuantitativa**: FedProx con mu=0.01 **converge mejor que
FedAvg en el mismo budget** (+4.0 pp de reducción de loss). La única
variable que cambia es el término proximal, así que la mejora es
atribuible al cambio algorítmico.

## Cobertura de datasets durante los 5 000 optimizer steps

**Plan coverage 36/36** PRETRAIN_SOURCE. **Datasets sampled durante el
run 31/36**, idéntico al FedAvg pilot v0.1:

| cliente | datasets sampled | total plan | nota |
|---|---|---:|---|
| aero_engines | NCMAPSS | 1/1 | |
| batteries | CALCE_CX2, FCLB19, NB1, NB14, UNIBO21 | 5/5 | |
| bearings | IMS, JNUB, KAUG17, LGB20, PRONOSTIA, SEUGB17, UPM20, UPM23, XJTU-SY | **9/12** | MFPT, CWRU (TT), CESNASA15, UOC18 quedan fuera; expected en 5 k steps weighted |
| cnc_milling | NMILL | 1/1 | |
| gearboxes | ARAMIS20, PHMAP21 | 2/2 | |
| hdd | HSF15 | 1/1 | |
| misc | HIRFNASA15, OBDD17, SSPSNASA15 | 3/4 | DUS20 queda fuera |
| misc_industrial | AC16, CBMv3, DFD15, PTRB19 | 4/5 | |
| phm_challenges | PHM10, PHM15, PHME24, PPD18 | 4/4 | |
| wind | PHM14 | 1/1 | |

## Pesos de agregacion (idénticos a FedAvg pilot v0.1 y a smoke FedProx v0.2)

| cliente | peso agregacion | nota |
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

`max−min = 0.2442` ≫ umbral 0.01 del check
`aggregation_weights_reflect_caps`. Constantes a lo largo de las 10
rondas (el plan FL no se reordena entre rondas).

## AMP overflow por cliente (11 / 5 000 steps perdidos, 0.22 %)

| cliente | amp_nonfinite_grad_steps | obs |
|---|---:|---|
| aero_engines | 8 | NCMAPSS con C=24 y trayectorias largas, prone a overflow fp16 |
| batteries | 3 | reparto entre los 5 datasets |
| resto (8 clientes) | 0 | sin overflow |

Mismos órdenes de magnitud que el central full (0.04 % en 100 k
steps). Los 11 steps perdidos no abortan: el `GradScaler` los omite
(comportamiento normal del scaler fp16). El **total de optimizer
steps coincide bit-a-bit con FedAvg pilot v0.1** (4 989 / 5 000),
lo cual refuerza que la comparación es justa.

## Outputs

- En Drive (pesados, NO versionados):
  - `checkpoints/ssl_federated_pilot_fedprox_mu0_01/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/ckpt_final.pt`.
  - `logs/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/metrics.jsonl` (10 lineas de `kind=round`).
  - `logs/pretraining_federated/_stdout/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm.stdout.log`.
- En el repo (versionados bit-a-bit con este commit):
  - `results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/run_info.json` (verbatim de Drive).
  - `results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/dry_run_report.json` (verbatim de Drive).
  - `results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/metrics_round_summary.json` (derivado de metrics.jsonl, mismo formato que el FedAvg pilot v0.1 + campos FedProx v0.2).
  - este `README.md`.

## Lectura cualitativa

1. **FedProx mu=0.01 mejora la convergencia SSL** de forma clara y
   reproducible. +4 pp de reduccion de loss en 5 000 steps es relevante
   y atribuible al término proximal (resto de variables fijado).
2. El termino proximal es **pequeno en valor absoluto** (0.0022 al
   final, 0.3 % de la loss SSL), pero **estabilizador**: ancla los
   updates locales lo suficiente para que la agregacion FedAvg
   produzca un descenso más consistente. No sobre-regulariza.
3. La cobertura de datasets es idéntica a FedAvg, lo cual descarta
   que la mejora venga de "ver mas datos diversos".
4. Los pesos de agregacion son idénticos bit-a-bit, lo cual descarta
   que la mejora venga de cambios en la politica de muestreo o cap.
5. La unica variable que diferencia ambos pilots es el **termino
   proximal**, asi que la mejora de loss es atribuible al cambio
   algoritmico. Hallazgo metodologico citable.

## Siguiente paso autorizado

Evaluación downstream del ckpt FedProx pilot en CWRU + HSG18 (4 corridas
linear_probing + full_finetuning_lr1e-5), mismo patrón que el bloque
`results/downstream/fl_pilot_vs_central/` que cerramos para FedAvg.
Esa evaluación decidira:

- **GO** full FedProx si HSG18 mejora ≥ +5 pp macro_f1 sin destruir CWRU.
- **CONDITIONAL** si HSG18 mejora 1–4 pp.
- **NO-GO** full FedProx vanilla si HSG18 sigue colapsando (hipotesis
  B estructural confirmada por la ablacion lr1e-4); pasar a opcion B
  (`min_client_presence` ajustada) o aceptar opcion D y reportar el
  limite estructural.

Hasta entonces: full FedProx **NO autorizado**.
