# Runbook Colab — Downstream federado pilot (CWRU + HSG18)

Evalúa si el ckpt FL pilot (`ssl_federated_pilot_patchtst_phm/ckpt_final.pt`)
produce representaciones útiles en los 2 TT primary classification donde
el SSL central confirmó transferencia: **CWRU** y **HSG18**.

**Decisión post-corrida**:
- **GO full FL** si `macro_f1(fed_linear) > macro_f1(from_scratch)` en
  ambos datasets y `macro_f1(fed_full) ≥ 0.9 × macro_f1(central_full)`.
- **CONDITIONAL** si `linear_fed > from_scratch` pero `full_fed < 0.9 ×
  full_central`: requiere diagnóstico antes de gastar las ~6 h del full FL.
- **NO-GO** si `linear_fed ≤ from_scratch`: el SSL FL no transfiere bajo
  esta política y hay que revisar antes de seguir.

**Coste estimado en A100**: 4 corridas × ~50–65 min = **~3.5 h totales**.

**Restricciones aplicables**:
- NO lanzar full FL.
- NO tocar `processed/` ni `processed_downstream/`.
- NO usar `ckpt_round005.pt` ni `ckpt_round010.pt` (solo `ckpt_final.pt`).
- NO ejecutar las 4 corridas en paralelo (serial, predecible).
- NO modificar el ckpt FL ni los logs históricos.

---

## Celda 1 — mount Drive

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Celda 2 — colab_init

```python
!bash /content/drive/MyDrive/fm_fl_phmd/colab_init.sh
```

## Celda 3 — pull al HEAD del bloque FL downstream pilot

```python
%cd /content/fm_fl_phmd
!git pull --ff-only
!git log -1 --oneline
!git status --porcelain
```

Debe imprimir el último commit del bloque downstream federado (run con
`git log` al final). `git status --porcelain` debe estar vacío.

## Celda 4 — verificar GPU

```python
!nvidia-smi | head -15
```

**Si NO es A100, parar aquí.** T4/V100 no son objetivo: las 4 corridas
tardarían el doble y consumirían tu cuota de Colab innecesariamente.

## Celda 5 — pytest preflight

```python
!python -m pytest \
  tests/test_downstream_federated_pilot_configs.py \
  tests/test_downstream_metrics.py \
  tests/test_downstream_pooling.py \
  tests/test_downstream_adaptive_batch.py -q
```

Esperado: **todos PASS, sin SKIP** (en Colab Linux con torch sano).

## Celda 6 — verificar checkpoint federado

```python
import torch
from pathlib import Path

CKPT_FL = Path(
    "/content/drive/MyDrive/fm_fl_phmd/checkpoints/"
    "ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt"
)

# ls del checkpoint
!ls -lh "{CKPT_FL}"

# Cargar y validar contrato minimo
ck = torch.load(str(CKPT_FL), map_location="cpu", weights_only=False)
print("\nkeys top-level:", list(ck.keys()))
assert "model_state_dict" in ck, "ckpt FL no tiene model_state_dict"
sd = ck["model_state_dict"]
print(f"model_state_dict: {len(sd)} tensores, primer key: {next(iter(sd))}")
for opt_key in ("config_hash", "git_hash", "param_count", "stage",
                "run_name", "round", "epoch"):
    if opt_key in ck:
        print(f"{opt_key}: {ck[opt_key]}")
```

Esperado: `model_state_dict` con muchos tensores; `config_hash =
082ca64313105f05`; `git_hash` comienza por `9b6c9fb`; `stage = pilot`.

## Celda 7 — preparar _stdout

```python
!mkdir -p /content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot/cwru/_stdout
!mkdir -p /content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot/hsg18/_stdout
```

## Celda 8 — 4 dry-runs (sanity rápido, no entrena)

```python
CKPT_FL = (
    "/content/drive/MyDrive/fm_fl_phmd/checkpoints/"
    "ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt"
)

import subprocess

configs = [
    ("training/configs/downstream_cwru_fedavg_pilot_linear_probing.yaml",
     "linear_probing"),
    ("training/configs/downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5.yaml",
     "full_finetuning"),
    ("training/configs/downstream_hsg18_fedavg_pilot_linear_probing.yaml",
     "linear_probing"),
    ("training/configs/downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5.yaml",
     "full_finetuning"),
]
for cfg, mode in configs:
    print(f"\n=== DRY-RUN {cfg} --mode {mode} ===")
    !python -m training.train_downstream_classification \
        --config "{cfg}" \
        --mode "{mode}" \
        --checkpoint "{CKPT_FL}" \
        --dry-run 2>&1 | tail -15
```

Cada dry-run debe terminar con `config_hash`, `n_classes` detectado y
`label_mapping`. Si alguno aborta por shards inexistentes, revisar
`processed/<DATASET>/` antes de lanzar las corridas reales.

## Celda 9 — corrida 1/4: CWRU linear_probing

```python
import time

CKPT_FL = (
    "/content/drive/MyDrive/fm_fl_phmd/checkpoints/"
    "ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt"
)

t0 = time.time()
!python -m training.train_downstream_classification \
  --config training/configs/downstream_cwru_fedavg_pilot_linear_probing.yaml \
  --mode linear_probing \
  --checkpoint "{CKPT_FL}" \
  2>&1 | tee /content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot/cwru/_stdout/downstream_cwru_fedavg_pilot_linear_probing.stdout.log
print(f"\n[total] CWRU linear_probing elapsed = {time.time() - t0:.1f} s")
```

## Celda 10 — corrida 2/4: CWRU full_finetuning_lr1e-5

```python
t0 = time.time()
!python -m training.train_downstream_classification \
  --config training/configs/downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5.yaml \
  --mode full_finetuning \
  --checkpoint "{CKPT_FL}" \
  2>&1 | tee /content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot/cwru/_stdout/downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5.stdout.log
print(f"\n[total] CWRU full_ft_lr1e-5 elapsed = {time.time() - t0:.1f} s")
```

## Celda 11 — corrida 3/4: HSG18 linear_probing

```python
t0 = time.time()
!python -m training.train_downstream_classification \
  --config training/configs/downstream_hsg18_fedavg_pilot_linear_probing.yaml \
  --mode linear_probing \
  --checkpoint "{CKPT_FL}" \
  2>&1 | tee /content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot/hsg18/_stdout/downstream_hsg18_fedavg_pilot_linear_probing.stdout.log
print(f"\n[total] HSG18 linear_probing elapsed = {time.time() - t0:.1f} s")
```

## Celda 12 — corrida 4/4: HSG18 full_finetuning_lr1e-5

```python
t0 = time.time()
!python -m training.train_downstream_classification \
  --config training/configs/downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5.yaml \
  --mode full_finetuning \
  --checkpoint "{CKPT_FL}" \
  2>&1 | tee /content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot/hsg18/_stdout/downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5.stdout.log
print(f"\n[total] HSG18 full_ft_lr1e-5 elapsed = {time.time() - t0:.1f} s")
```

## Celda 13 — summary federado y comparación con central

```python
import json
from pathlib import Path

fed_base = Path("/content/drive/MyDrive/fm_fl_phmd/logs/downstream_federated_pilot")
runs_fed = [
    ("CWRU",  "linear_probing",
     fed_base / "cwru"  / "downstream_cwru_fedavg_pilot_linear_probing"             / "run_info.json"),
    ("CWRU",  "full_finetuning_lr1e-5",
     fed_base / "cwru"  / "downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5"     / "run_info.json"),
    ("HSG18", "linear_probing",
     fed_base / "hsg18" / "downstream_hsg18_fedavg_pilot_linear_probing"            / "run_info.json"),
    ("HSG18", "full_finetuning_lr1e-5",
     fed_base / "hsg18" / "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5"    / "run_info.json"),
]

def _pick(d, *keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default

print(f"{'dataset':<8s} {'mode':<26s} {'macro_f1':>10s} {'bal_acc':>10s} {'acc':>8s} {'best_ep':>8s} {'elapsed':>10s} {'n_train':>10s} {'config_hash':>20s}")
print("-" * 130)
fed_rows = {}
for ds, mode, p in runs_fed:
    if not p.is_file():
        print(f"{ds:<8s} {mode:<26s} (NO run_info en {p})")
        continue
    ri = json.loads(p.read_text())
    tm = ri.get("test_metrics") or {}
    row = {
        "macro_f1": _pick(tm, "macro_f1", default=0.0),
        "balanced_accuracy": _pick(tm, "balanced_accuracy", default=0.0),
        "accuracy": _pick(tm, "accuracy", default=0.0),
        "best_epoch": ri.get("best_epoch", -1),
        "elapsed_seconds": ri.get("elapsed_seconds", 0.0),
        "n_trainable_params": ri.get("n_trainable_params", 0),
        "config_hash": ri.get("config_hash", ""),
        "checkpoint_loaded": ri.get("checkpoint_loaded", ""),
    }
    fed_rows[(ds, mode)] = row
    print(f"{ds:<8s} {mode:<26s} {row['macro_f1']:>10.4f} {row['balanced_accuracy']:>10.4f} {row['accuracy']:>8.4f} {row['best_epoch']:>8d} {row['elapsed_seconds']:>10.1f}s {row['n_trainable_params']:>10d} {row['config_hash']:>20s}")

# Comparativa central vs federado: lee summary central versionado en repo.
print("\n=== Comparativa central vs federado (macro_f1 test) ===")
central_summary_path = Path("/content/fm_fl_phmd/results/downstream/summary_classification_primary.json")
if not central_summary_path.is_file():
    print(f"WARN no encontrado {central_summary_path}; busca el equivalente.")
else:
    central = json.loads(central_summary_path.read_text())
    # Espera estructura compatible; ajusta si difiere.
    print(json.dumps(central, indent=2)[:2000])

# Tabla comparativa final central vs federado.
print("\n=== Tabla central vs federado ===")
print(f"{'dataset':<8s} | {'from_scratch':>13s} | {'central_linear':>15s} | {'central_full_1e-5':>18s} | {'fed_linear':>11s} | {'fed_full_1e-5':>14s}")
print("-" * 100)
# Estos numeros son los de los runs centrales ya cerrados en el repo.
central_known = {
    "CWRU":  {"from_scratch": 0.3503, "linear": 0.7046, "full_1e-5": 0.8292},
    "HSG18": {"from_scratch": 0.5693, "linear": 0.9056, "full_1e-5": 0.9504},
}
for ds in ("CWRU", "HSG18"):
    c = central_known[ds]
    fl_lin = fed_rows.get((ds, "linear_probing"),         {}).get("macro_f1")
    fl_ft  = fed_rows.get((ds, "full_finetuning_lr1e-5"), {}).get("macro_f1")
    fl_lin_str = f"{fl_lin:.4f}" if fl_lin is not None else "n/a"
    fl_ft_str  = f"{fl_ft:.4f}"  if fl_ft  is not None else "n/a"
    print(f"{ds:<8s} | {c['from_scratch']:>13.4f} | {c['linear']:>15.4f} | {c['full_1e-5']:>18.4f} | {fl_lin_str:>11s} | {fl_ft_str:>14s}")

# Deltas
print("\n=== Deltas (federado vs central) ===")
for ds in ("CWRU", "HSG18"):
    c = central_known[ds]
    fl_lin = fed_rows.get((ds, "linear_probing"),         {}).get("macro_f1")
    fl_ft  = fed_rows.get((ds, "full_finetuning_lr1e-5"), {}).get("macro_f1")
    if fl_lin is not None:
        d_lin_vs_fs = fl_lin - c["from_scratch"]
        d_lin_vs_central = fl_lin - c["linear"]
        print(f"{ds:<8s} linear:  fed - from_scratch = {d_lin_vs_fs:+.4f} | fed - central_linear = {d_lin_vs_central:+.4f}")
    if fl_ft is not None:
        d_ft_vs_fs = fl_ft - c["from_scratch"]
        d_ft_vs_central = fl_ft - c["full_1e-5"]
        print(f"{ds:<8s} full:    fed - from_scratch = {d_ft_vs_fs:+.4f} | fed - central_full  = {d_ft_vs_central:+.4f}")

print("\nPega el output completo en el chat para que se cierre la Fase 4.")
```

---

## Tras pegar el output al chat

Cierre del bloque downstream federado pilot:

1. Asistente crea `results/downstream/fl_pilot_vs_central/`:
   - `summary.json` (las 4 corridas FL + las 6 central).
   - `README.md` (tabla central vs federado + decisión GO/CONDITIONAL/NO-GO).
2. Copia los 4 `run_info.json` reales de Drive (igual que el patrón CWRU/HSG18 central).
3. Actualiza `results/downstream/README.md` y `results/pretraining_federated/README.md` con el cierre Fase 4.
4. Commit + push tras autorización.

**Pendiente operativo, no bloquea**: si por timing tuvieras que parar
después de 1 o 2 corridas, los `best.pt` ya escritos en Drive se
mantienen y puedes reanudar en otra sesión retomando desde la celda
que faltó.
