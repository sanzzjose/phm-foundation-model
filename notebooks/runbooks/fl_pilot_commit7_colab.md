# Runbook — FL pilot (commit 7) en Colab A100

Procedimiento canónico para ejecutar el **pilot federado** del bloque
FL del TFM. Lectura previa obligatoria:

- `training/configs/ssl_federated_pilot.yaml` — config congelado.
- `tests/test_fl_pilot_config.py` — assertions sobre el config.
- `results/pretraining_federated/README.md` — historial (smoke v0.1, v0.2).
- `CLAUDE.md` sec 7 / 7.bis / 15 — política FL, caps, métricas.

**Restricciones operativas**:

- No lanzar full FL en esta corrida (solo pilot).
- No tocar `processed/` ni `processed_downstream/`.
- No modificar checkpoints/logs históricos.
- Solo escribir outputs nuevos del pilot en
  `logs/pretraining_federated/ssl_federated_pilot_patchtst_phm/` y
  `checkpoints/ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/`.

---

## Celda 1 — Mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Celda 2 — Init repo

```python
!bash /content/drive/MyDrive/fm_fl_phmd/colab_init.sh
```

## Celda 3 — Pull al HEAD del bloque FL pilot

```python
%cd /content/fm_fl_phmd
!git pull --ff-only
!git log -1 --oneline
!git status --porcelain
```

El HEAD debe contener el commit del bloque FL pilot (tests + runbook +
config validado).

## Celda 4 — Verificar GPU A100

```python
!nvidia-smi | head -15
```

**Decisión dura**: si la GPU **no** es A100 (e.g. sale `Tesla T4` o
`V100`), **no continuar**. T4/V100 no son objetivo de esta corrida; el
pilot bajo otra GPU no es comparable con el central full (que también
se hizo en A100). Cierra la sesión y vuelve cuando haya A100 disponible.

## Celda 5 — Tests FL preflight

```python
!python -m pytest tests/test_fl_aggregation.py \
                  tests/test_fl_client_filtering.py \
                  tests/test_fl_server_dryrun.py \
                  tests/test_adaptive_batch_size.py \
                  tests/test_fl_pilot_config.py -q
```

Esperado: **53 PASS** sin SKIP (en Linux Colab corren todos). Si algún
test falla, **parar y reportar**.

## Celda 6 — Dry-run pilot (sin entrenar)

```python
# Asegurar que el directorio de stdout existe.
!mkdir -p /content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated/_stdout

!python -m training.train_ssl_federated \
  --mode dry-run \
  --config training/configs/ssl_federated_pilot.yaml \
  2>&1 | tee /content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated/_stdout/ssl_federated_pilot_dryrun.stdout.log
```

**Esperado del dry-run**:

- 10 clientes.
- 36 datasets únicos (todos los `PRETRAIN_SOURCE`).
- 0 `TRANSFER_TARGET`.
- `plan_policy` contiene `final_client_weight_capped_v23`.
- Forward sintético OK.
- `dry_run_report.json` escrito en `logs/pretraining_federated/.../`.

Si alguno de estos asserts falla, **parar**. No relanzar tampoco
celda 7 hasta arreglarlo.

## Celda 7 — Pilot real

Solo correr si **A100 confirmada + tests PASS + dry-run OK**.

```python
!mkdir -p /content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated/_stdout

!python -m training.train_ssl_federated \
  --mode train \
  --config training/configs/ssl_federated_pilot.yaml \
  2>&1 | tee /content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated/_stdout/ssl_federated_pilot_patchtst_phm.stdout.log
```

**Tiempo estimado**: ~15-25 min en A100 (extrapolando del smoke v0.2
que hizo 100 steps en 93 s; 5000 steps ≈ 4650 s = 78 min teórico,
pero con menor overhead por ronda el wall clock real estará entre 15
y 25 min).

**Esperado al terminar**:

- `stage=pilot`.
- `run_name=ssl_federated_pilot_patchtst_phm`.
- 10 rondas completadas.
- En `run_info.json`: `pilot_pass: true`.
- Logs en
  `/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated/ssl_federated_pilot_patchtst_phm/`.
- Checkpoint final
  `/content/drive/MyDrive/fm_fl_phmd/checkpoints/ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt`.
- Checkpoints intermedios (por `ckpt_every_rounds: 5`):
  - `ckpt_round005.pt`
  - `ckpt_round010.pt`

Si `pilot_pass: false`, **NO continuar** a downstream con el ckpt
federado. Reportar y diagnosticar.

## Celda 8 — Resumen del pilot

```python
import json
from pathlib import Path

log_dir = Path("/content/drive/MyDrive/fm_fl_phmd/logs/pretraining_federated/ssl_federated_pilot_patchtst_phm")
ckpt   = Path("/content/drive/MyDrive/fm_fl_phmd/checkpoints/ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt")
ri_path      = log_dir / "run_info.json"
metrics_path = log_dir / "metrics.jsonl"

print("run_info exists:", ri_path.is_file(), ri_path)
print("metrics exists:", metrics_path.is_file(), metrics_path)
print("ckpt exists:", ckpt.is_file(), ckpt)

ri = json.loads(ri_path.read_text())
print(json.dumps({
    "stage": ri.get("stage"),
    "run_name": ri.get("run_name"),
    "config_hash": ri.get("config_hash"),
    "git_hash": ri.get("git_hash"),
    "git_dirty": ri.get("git_dirty"),
    "pilot_pass": ri.get("pilot_pass"),
    "n_rounds": ri.get("n_rounds"),
    "local_steps": ri.get("local_steps"),
    "total_local_optimizer_steps": ri.get("total_local_optimizer_steps"),
    "param_count": ri.get("param_count"),
    "cumulative_communication_mb": ri.get("cumulative_communication_mb"),
    "max_effective_bc_global": ri.get("max_effective_bc_global"),
    "aggregation_weight_policy": ri.get("aggregation_weight_policy"),
    "aggregation_weights_policy_effective": ri.get("aggregation_weights_policy_effective"),
    "plan_policy_unique": ri.get("plan_policy_unique"),
    "checkpoint_final": ri.get("checkpoint_final"),
    "elapsed_seconds": ri.get("elapsed_seconds"),
    "amp_nonfinite_grad_steps_total": ri.get("amp_nonfinite_grad_steps_total"),
    "opt_steps_per_client_total": ri.get("opt_steps_per_client_total"),
    "aggregation_weights_by_client_last_round": ri.get("aggregation_weights_by_client_last_round"),
}, indent=2))

print("\nchecks:")
for k, v in (ri.get("pilot_checks") or {}).items():
    print(k, "=>", v.get("ok"), v)

rounds = []
for line in metrics_path.read_text().splitlines():
    if not line.strip():
        continue
    rec = json.loads(line)
    if rec.get("kind") == "round":
        rounds.append(rec)

print("\nround loss:")
for r in rounds:
    print(r.get("round"),
          r.get("loss_mean_weighted"),
          r.get("max_effective_bc"),
          r.get("cumulative_communication_mb"))

if rounds:
    first = rounds[0].get("loss_mean_weighted")
    last  = rounds[-1].get("loss_mean_weighted")
    if first and last:
        print("loss_delta_pct:", 100.0 * (last - first) / first)
```

Pega el output completo de esta celda. Con eso pasamos a FASE 3 y
versionamos los artefactos ligeros del pilot.

---

## NO ejecutar (queda fuera del runbook)

- `training/configs/ssl_federated_full.yaml` — no existe ni se crea aún.
- Full FL — bloqueado hasta validar pilot.
- FedProx — opcional, fuera de este runbook.
- Downstream FL con el ckpt federado — fase 4-5, runbook aparte.

## Criterios de decisión post-pilot

Tras pegar el output de Celda 8, decidiremos:

- **GO** para full FL si `pilot_pass=true` + loss baja + sin oscilaciones extremas.
- **CONDITIONAL** si pasa los checks pero la loss apenas baja (validar con
  evaluación downstream en CWRU/HSG18 con el ckpt pilot antes del full).
- **NO-GO** si `pilot_pass=false` o la loss explota.

La decisión queda documentada en
`results/pretraining_federated/ssl_federated_pilot/README.md`.
