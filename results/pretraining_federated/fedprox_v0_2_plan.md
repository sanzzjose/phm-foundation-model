# Plan FedProx FL v0.2 — EJECUTADO / PASS (2026-05-26)

> **Estado**: smoke + pilot FedProx mu=0.01 ejecutados en Colab A100
> el 2026-05-26 (commit `bb38367`). **Ambos PASS**. Ckpt final
> disponible en Drive. La sección de cierre con números reales esta
> en `Estado tras ejecucion` al final de este documento; la
> deliberacion previa se mantiene tal cual.

## Motivacion

El pilot FedAvg v0.1 (commit `9b6c9fb`) cerro PASS pipeline pero la
evaluacion downstream del ckpt federado revelo dos comportamientos
opuestos (`results/downstream/fl_pilot_vs_central/`):

| TT primary | dominio | cliente FL | n_datasets cliente | fed_full_1e-5 / central_full | veredicto |
|---|---|---|---:|---:|---|
| CWRU  | bearings | `bearings`        | 12 (9 sampled) | 80.0 % | transfiere parcialmente |
| HSG18 | hdd      | `hdd` (HSF15) | 1              | 58.4 % | no transfiere; colapso a clase mayoritaria |

La ablacion HSG18 `lr_backbone=1e-4` (commit `250868f` + `2028943`)
descarto la hipotesis de adaptacion: subir el LR colapsa
degeneradamente (`macro_f1=0.3333`, recall clase 0 = 0 %), confirmando
que el embedding FL HDD es **estructuralmente insuficiente** y que el
problema esta en la diversidad intra-cliente del corpus federado.

FedProx es la siguiente variante federada minima prevista por el TFM
(sec 15 CLAUDE.md). Anyade un termino proximal a la loss SSL local:

```
loss_total = loss_ssl + 0.5 * mu * sum_l ||theta_l - theta_global_l||^2
```

donde `theta_global_l` es el snapshot del parametro `l` recibido al
inicio de la ronda. Esto **reduce client drift**: cada cliente puede
moverse menos lejos del estado global por ronda, lo cual deberia
beneficiar especialmente a clientes con datasets pocos / muy
heterogeneos (e.g. `hdd`, `wind`, `cnc_milling`, `aero_engines`).

## Que cambia respecto al pilot FedAvg

| campo | FedAvg v0.1 | FedProx v0.2 |
|---|---|---|
| `federated.algorithm` | `fedavg` | **`fedprox`** |
| `federated.fedprox_mu` | `null` | **`0.01`** |
| `run_name` | `ssl_federated_pilot_patchtst_phm` | `ssl_federated_pilot_fedprox_mu0_01_patchtst_phm` |
| `paths.checkpoint_dir` | `.../checkpoints/ssl_federated_pilot` | `.../checkpoints/ssl_federated_pilot_fedprox_mu0_01` |
| termino proximal en loss local | no | si (`0.5 * mu * drift_l2_sq`) |
| metricas separadas en log | no | si: `reconstruction_loss_mean_weighted`, `fedprox_loss_mean_weighted`, `fedprox_penalty_mean_weighted` |
| nuevos campos `run_info` | — | `algorithm`, `fedprox_mu`, `fedprox_enabled`, `final_*_mean_weighted` |

`mu=0.01` es deliberadamente conservador (cifra inicial estandar en la
literatura FedProx). Si el pilot pasa, se podra ablar `mu in {0.001,
0.01, 0.1}` para localizar el sweet spot, pero esa ablacion no entra
en v0.2.

## Que NO cambia

| invariante | valor |
|---|---|
| Datasets del corpus FL | mismos 36 PRETRAIN_SOURCE (audit v2.3) |
| Topologia FL | mismos 10 clientes |
| Politica de pesos | `final_client_weight` con caps cerrados `capped_v23` (0.10 / 0.25 / 0.005) |
| Batch adaptativo | `B*C <= max_channel_batch=512`, `min_batch_size=1` |
| Backbone | `patchtst_phm_base` (801 808 params) |
| Hiperparametros training | `lr=3e-4`, `weight_decay=0.05`, `grad_clip=1.0`, `amp=auto` |
| Presupuesto pilot | 10 rondas x 50 steps x 10 clientes = 5 000 local steps |
| Stage cap | `pilot <= 20 000` local steps |
| `min_client_presence` | sigue en 0.005 (no se toca en v0.2 para aislar el efecto de FedProx) |
| `sampling_policy` | sin cambios |

Tampoco se modifica nada de FedAvg v0.1: el codigo conserva el camino
FedAvg bit-a-bit (`algorithm=fedavg` o `fedprox_mu=null/0` -> FedProx
inactivo, sin cambios de comportamiento). Los tests existentes siguen
pasando.

## Que se ha preparado

| artefacto | path | proposito |
|---|---|---|
| codigo | `training/fl/client.py` | `resolve_fedprox_config`, `snapshot_global_params`, `compute_fedprox_penalty`, `FederatedClient.federated_cfg`, loop con termino proximal opcional |
| codigo | `training/fl/server.py` | pasa `cfg["federated"]` a clientes, agrega 3 metricas FedProx por ronda, anyade `algorithm/fedprox_mu/fedprox_enabled/final_*` a `run_info` |
| codigo | `training/train_ssl_federated.py` | `_validate_federated_config(cfg)` rechaza combinaciones ambiguas |
| config | `training/configs/ssl_federated_smoke_fedprox_mu0_01.yaml` | smoke 2x5x10 |
| config | `training/configs/ssl_federated_pilot_fedprox_mu0_01.yaml` | pilot 10x50x10 |
| tests | `tests/test_fl_fedprox.py` | 16 tests: penalty + resolve + validacion |
| tests | `tests/test_fl_fedprox_configs.py` | 18 tests: blindaje bit-a-bit de los 2 configs vs FedAvg homologos |
| notebook | `notebooks/pretraining/run_fl_fedprox_mu0_01_smoke_pilot.ipynb` | 12 celdas de ejecucion + summary |

## Criterios post-smoke (decision automatica)

Ejecutar el smoke en Colab A100 con el notebook. Resultado esperado:

- `smoke_pass == true` con los 6 checks OK (mismos que FedAvg smoke v0.2):
  loss finita en todos los clientes, `optimizer_steps > 0` por cliente,
  state cambia tras agregacion, `max_effective_bc <= 512`, 0 TT en plan,
  pesos de agregacion reflejan caps (max-min > 0.01).
- `run_info.algorithm == "fedprox"` y `fedprox_mu == 0.01`.
- `final_fedprox_loss_mean_weighted > 0` (el termino proximal contribuye).
- `final_reconstruction_loss_mean_weighted` finito (la loss SSL pura
  sigue calculandose en paralelo).

| resultado smoke | decision |
|---|---|
| smoke_pass false (cualquier check FAIL) | **NO-GO**. Diagnosticar y decidir si reintentar o pasar a opcion D del veredicto FL. |
| smoke_pass true | autorizado el pilot. |

## Criterios post-pilot (decision narrativa)

Ejecutar el pilot en Colab A100 con el notebook (segunda mitad).
Resultado esperado:

- `pilot_pass == true` con los 6 pilot_checks OK.
- `final_reconstruction_loss_mean_weighted < initial_reconstruction_loss_mean_weighted`
  (la loss SSL converge igual o mejor que FedAvg).
- No explosion del termino proximal (`final_fedprox_penalty_mean_weighted`
  estable o decreciendo tras las primeras rondas).
- Ckpt final en `checkpoints/ssl_federated_pilot_fedprox_mu0_01/.../ckpt_final.pt`.

| resultado pilot | decision |
|---|---|
| pilot_pass false | NO-GO. No usar ckpt para downstream. |
| pilot_pass true pero reconstruction_loss no mejora vs FedAvg | CONDITIONAL. Evaluar downstream igual; si downstream tampoco mejora, opcion D. |
| pilot_pass true y trayectoria estable | autorizada eval downstream HSG18 + CWRU. |

## Criterios post-downstream FedProx (decision sobre full FL)

Tras el pilot exitoso, se replicara el bloque `fl_pilot_vs_central`
(4 corridas en Colab Pro+ A100, ~2 h en total) pero usando el ckpt
FedProx pilot en lugar del FedAvg pilot:

- HSG18 `linear_probing` + `full_finetuning_lr1e-5`.
- CWRU `linear_probing` + `full_finetuning_lr1e-5`.

Esta evaluacion la hara un notebook aparte (no incluido en v0.2)
analogo a `notebooks/downstream/run_fl_downstream_pilot_cwru_hsg18.ipynb`,
con el ckpt FedProx cargado por path explicito.

| resultado downstream FedProx | decision |
|---|---|
| HSG18 mejora claramente (`macro_f1_full_fp > macro_f1_full_fa` por ≥ 5 pp) sin destruir CWRU | **GO** full FedProx (100 k steps, ~6 h A100). |
| HSG18 sigue colapsando o solo mejora 1-2 pp | mantener CONDITIONAL. Considerar opcion B (`min_client_presence=0.05`) en una v0.3 o aceptar opcion D. |
| HSG18 mejora pero CWRU empeora > 5 pp | mu = 0.01 sobre-regulariza. Considerar `mu = 0.001` en v0.3. |
| HSG18 colapsa y `fedprox_penalty_mean_weighted` apenas existe | mu = 0.01 demasiado pequeno para anclar; o problema realmente estructural. Opcion D. |

## Lo que NO se aborda en v0.2

- Ablacion de `mu` (queda para v0.3 si v0.2 es CONDITIONAL).
- Subir `min_client_presence` (opcion B; queda separada para aislar el
  efecto FedProx puro).
- Full FL (siempre requiere autorizacion explicita; el codigo respeta
  el gating `stage` y los configs FedProx solo son smoke + pilot).
- SCAFFOLD u otros algoritmos FL (no previstos para el MVP).
- Cambios en el corpus FL, audit_groups o sampling policy.

## Referencias

- CLAUDE.md sec 15: "FedProx como variante principal frente a no-IID".
- CLAUDE.md sec 7.bis: topologia FL cerrada de 10 clientes con pesos
  `capped_v23`.
- `results/pretraining_federated/README.md`: estado del bloque
  federado, incluida la ablacion HSG18 lr1e-4.
- `results/downstream/fl_pilot_vs_central/README.md`: 4 corridas
  FedAvg pilot + ablacion lr1e-4. Veredicto CONDITIONAL global, NO-GO
  local HSG18.
- `docs/decisions/pending_downstream_and_sampling.md` (gitignored):
  deliberacion interna detallada.

## Estado

| fase | estado |
|---|---|
| codigo FedProx en `training/fl/` | implementado (commit `bb38367`) |
| validacion config en trainer | implementada |
| 2 configs FedProx (smoke + pilot) | versionados |
| tests unitarios (penalty + resolve + validacion) | 16/16 PASS local |
| tests de configs (blindaje vs FedAvg) | 18/18 PASS local |
| pytest FL completo en Colab | **87/87 PASS** (75 locales + 12 adaptive batch en Colab) |
| notebook Colab smoke + pilot | ejecutado end-to-end 2026-05-26 |
| **smoke FedProx real en A100** | **CERRADO / PASS** (`smoke_pass=true`, 6/6 checks) |
| **pilot FedProx real en A100** | **CERRADO / PASS** (`pilot_pass=true`, 6/6 checks) |
| eval downstream FedProx (CWRU+HSG18) | **CERRADA en commit `25cdd81`: NO-GO full FedProx vanilla** (HSG18 full colapsa + CWRU full empeora 17 pp vs FedAvg) |
| full FedProx | **NO autorizado** (verdict NO-GO de la eval downstream confirma hipotesis B estructural + sweet spot LR distinto para FedProx) |

## Estado tras ejecucion (2026-05-26, commit `bb38367`)

### Smoke FedProx — PASS

- `smoke_pass=true`, 6/6 smoke_checks OK.
- `config_hash=e89d1661836518eb`, `git_hash=bb38367`.
- `total_local_optimizer_steps=100/100` (sin AMP overflow).
- Loss r1 → r2: 0.9241 → 0.8842 (−4.3 %).
- `final_fedprox_loss_mean_weighted=0.0005` (anclaje muy pequeno en
  smoke; esperable: solo 2 rondas de drift posible).
- `final_fedprox_penalty_mean_weighted=0.0988`.
- `max_effective_bc_global=510 ≤ 512`.
- `aggregation_weights_policy_effective=final_client_weight_capped_v23`.
- `elapsed_seconds=215.15` (~3.6 min en A100).
- Ckpt smoke en Drive (no se usa para downstream; descarte tras
  validacion del pipeline).

### Pilot FedProx — PASS

- `pilot_pass=true`, 6/6 pilot_checks OK.
- `config_hash=678df3d8feb46f82`, `git_hash=bb38367`, `git_dirty=false`.
- `total_local_optimizer_steps=4 989/5 000` (11 omitidos por AMP
  overflow; **mismo numero exacto que FedAvg pilot v0.1**, lo cual
  refuerza que la comparacion es justa).
- Loss r1 → r10: 0.8395 → 0.7423 (**−11.57 %**, vs −7.55 % del FedAvg
  pilot v0.1; **+4.0 pp** mejor en el mismo budget).
- Reconstruction r1 → r10: 0.8337 → 0.7401 (**−11.23 %**).
- Prox term r1 → r10: 0.0058 → 0.0022 (decrece: el modelo local se
  acerca al global a medida que ambos convergen, lo cual es el
  comportamiento esperado de FedProx con mu pequeno).
- `final_fedprox_penalty_mean_weighted=0.4473` (drift cuadratico
  agregado acotado, sin explosion).
- `max_effective_bc_global=510 ≤ 512`.
- `cumulative_communication_mb=611.73` (idem FedAvg, esperable: solo
  cambia la loss interna, no el state_dict comunicado).
- `aggregation_weights_policy_effective=final_client_weight_capped_v23`.
- Pesos agregacion idem FedAvg pilot v0.1: bearings 0.2492 (cap),
  phm_challenges 0.2441 (cap), cnc_milling 0.0050 (piso), ...
- Cobertura datasets: 31/36 sampled (mismo que FedAvg pilot v0.1; la
  cobertura depende del sampler weighted, no del algoritmo FL).
- `elapsed_seconds=1 061.57` (~17.7 min en A100, −6.7 % vs FedAvg
  pilot v0.1 a misma carga; probablemente ruido de scheduling, no
  efecto del algoritmo).
- Ckpt en Drive:
  `checkpoints/ssl_federated_pilot_fedprox_mu0_01/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/ckpt_final.pt`.

### Lectura cualitativa

1. **FedProx mu=0.01 mejora la convergencia SSL** de forma clara y
   reproducible (mismo seed, mismo plan, mismo budget). +4 pp de
   reduccion de loss en 5 000 steps es relevante.
2. El termino proximal es **pequeno en valor absoluto** (0.0022 al
   final, 0.3 % de la loss SSL), pero **estabilizador**: ancla los
   updates locales lo suficiente para que la agregacion FedAvg
   produzca un descenso mas consistente. No sobre-regulariza.
3. La cobertura de datasets es idéntica a FedAvg, lo cual descarta
   que la mejora venga de "ver mas datos diversos".
4. Los pesos de agregacion son idénticos bit-a-bit, lo cual descarta
   que la mejora venga de cambios en la politica de muestreo o cap.
5. La unica variable que diferencia ambos pilots es el **termino
   proximal**, asi que la mejora de loss es atribuible al cambio
   algoritmico. Hallazgo metodologico citable.

### Eval downstream FedProx CWRU + HSG18 — CERRADA (commit `25cdd81`)

Eval downstream FedProx CWRU/HSG18 ejecutada y cerrada en `25cdd81`:
**NO-GO full FedProx vanilla**. 4 corridas en Colab A100 (~2 h),
ckpt evaluado = `ssl_federated_pilot_fedprox_mu0_01/ckpt_final.pt`.
Comparada bit-a-bit contra:

- baseline `from_scratch`,
- central (`ssl_central_full/ckpt_step100000.pt`),
- **FedAvg pilot (`ssl_federated_pilot/ckpt_final.pt`)**.

Resultados macro_f1 test (verbatim de los 4 `run_info.json` versionados
en `results/downstream/fl_fedprox_pilot_vs_central/`):

| dataset | modo | FedAvg | FedProx | Δ FP−FA |
|---|---|---:|---:|---:|
| CWRU  | linear_probing         | 0.4456 | 0.4587 | +0.0131 marginal |
| CWRU  | full_finetuning_lr1e-5 | 0.6635 | **0.4889** | **−0.1746 empeora** |
| HSG18 | linear_probing         | 0.6080 | **0.7242** | **+0.1162 mejora** |
| HSG18 | full_finetuning_lr1e-5 | 0.5547 | 0.5628 | +0.0082 marginal |

### Criterio para autorizar full FedProx (CERRADO con NO-GO)

| resultado downstream FedProx | decision sobre full FedProx | aplicacion al run real |
|---|---|---|
| HSG18 mejora claramente (`macro_f1_full_fp − macro_f1_full_fa ≥ +5 pp`) sin destruir CWRU | **GO** full FedProx (~6 h A100, 100 k steps) | NO se cumple (delta HSG18 full = +0.82 pp) |
| HSG18 mejora 1–4 pp o pierde el colapso a clase mayoritaria | CONDITIONAL: documentar ganancia parcial, decidir caso por caso si justifica full | NO se cumple (HSG18 full sigue colapsando con recall clase 0 = 0.2574 < 0.30) |
| HSG18 sigue colapsando (recall clase 0 < 30 %) | **NO-GO** full FedProx vanilla. El problema es estructural del corpus FL (hdd mono-dataset), no del algoritmo. Considerar opcion B (`min_client_presence=0.05` en v0.3) o aceptar opcion D y reportar el limite. | **SE CUMPLE: HSG18 full colapsa, recall clase 0 = 0.2574** |
| CWRU empeora > 5 pp (mu=0.01 sobre-regulariza) | considerar `mu=0.001` en v0.3 | SE CUMPLE TAMBIEN: CWRU full empeora 17.5 pp |

### Outputs FedProx v0.2 en Drive

```
/content/drive/MyDrive/fm_fl_phmd/
├── checkpoints/
│   ├── ssl_federated_smoke_fedprox_mu0_01/...ckpt_final.pt
│   └── ssl_federated_pilot_fedprox_mu0_01/...ckpt_final.pt   <- usar para downstream
└── logs/pretraining_federated/
    ├── ssl_federated_smoke_fedprox_mu0_01_patchtst_phm/{run_info.json, metrics.jsonl, dry_run_report.json}
    ├── ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/{run_info.json, metrics.jsonl, dry_run_report.json}
    └── _stdout/*.stdout.log
```

### Outputs FedProx v0.2 en el repo

| versionado | path |
|---|---|
| commit `d06ad26` | `results/pretraining_federated/README.md` (seccion FedProx) |
| commit `d06ad26` | `results/pretraining_federated/fedprox_v0_2_plan.md` (este documento, con `Estado tras ejecucion`) |
| commit `ec2bf2f` | `results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/{run_info.json, dry_run_report.json, metrics_round_summary.json, README.md}` (bit-a-bit verbatim desde Drive, mismo patron que `ssl_federated_pilot/` para FedAvg) |
| commit `ec2bf2f` | `results/pretraining_federated/ssl_federated_smoke_fedprox_mu0_01/{run_info.json, dry_run_report.json}` (evidencia preflight) |

