# Pretraining central SSL — resultados

> Resumen agregado del bloque central SSL (PatchTSTPhm,
> channel-independent, masked patch prediction). El analisis
> interno y deliberacion estan en
> `docs/decisions/pending_downstream_and_sampling.md` (gitignored).

Este bloque cubre las 4 etapas del SSL central, en orden:

1. **ssl_central_smoke** — smoke de tubo y diagnostico padding parcial.
2. **ssl_central_coverage** — verificacion de cobertura 36/36 PS con `sampling_strategy=round_robin`.
3. **ssl_central_pilot** — piloto productivo `sampling_strategy=weighted` en 5 000 steps.
4. **ssl_central_full** — pretraining real en 100 000 steps. Checkpoint usado downstream.

Todos los stages comparten arquitectura (`PatchTSTPhm`), corpus
(36 PRETRAIN_SOURCE) y politica de caps `capped_v23`
(`cap_max_dataset_weight=0.10`, `cap_max_client_weight=0.25`,
`min_client_presence=0.005`).

## Resumen comparativo

| stage | pass | config_hash | optimizer_steps | datasets_seen | clients_seen | param_count | amp_overflow |
|---|---|---|---:|---:|---:|---:|---:|
| `ssl_central_smoke` | **True** | `46628aedb05becd6` | 50 | 5 | 4 | 104 336 | 0 |
| `ssl_central_coverage` | **True** | `e5cfd3b0684c7918` | 72 | 36 | 10 | 801 808 | 0 |
| `ssl_central_pilot` | **True** | `e4970c173c9dc244` | 4 999 | 34 | 10 | 801 808 | 1 |
| `ssl_central_full` | **True** | `9ed84508a6820265` | 99 961 | 36 | 10 | 801 808 | 39 |

## ssl_central_smoke

| campo | valor |
|---|---|
| `run_name` | `ssl_smoke_patchtst_phm` |
| `config_hash` | `46628aedb05becd6` |
| `git_hash` | `3de700cdf573535a9a19856a1999d24ecaf8e020` |
| `smoke_pass` | `True` |
| `optimizer_steps` | `50` |
| `param_count` | `104 336` |
| `elapsed_seconds` | `11.6` |
| `amp_nonfinite_grad_steps` | `0` |

## ssl_central_coverage

| campo | valor |
|---|---|
| `run_name` | `ssl_central_coverage_patchtst_phm` |
| `stage` | `coverage` |
| `config_hash` | `e5cfd3b0684c7918` |
| `git_hash` | `13b940489d5bde15d378f31a2ae862b32d801f87` |
| `coverage_pass` | `True` |
| `optimizer_steps` | `72` |
| `param_count` | `801 808` |
| `elapsed_seconds` | `159.9` |
| `max_effective_bc` | `510` |
| `amp_nonfinite_grad_steps` | `0` |

## ssl_central_pilot

| campo | valor |
|---|---|
| `run_name` | `ssl_central_pilot_patchtst_phm` |
| `stage` | `pilot` |
| `config_hash` | `e4970c173c9dc244` |
| `git_hash` | `8315bddd81b90aa9fce52962e5f4a6f364e72a33` |
| `pilot_pass` | `True` |
| `optimizer_steps` | `4 999` |
| `param_count` | `801 808` |
| `elapsed_seconds` | `691.6` |
| `max_effective_bc` | `510` |
| `amp_nonfinite_grad_steps` | `1` |

## ssl_central_full

| campo | valor |
|---|---|
| `run_name` | `ssl_central_full_patchtst_phm` |
| `stage` | `full` |
| `config_hash` | `9ed84508a6820265` |
| `git_hash` | `b3767b9268fdec22489129fe428eff820382b98e` |
| `coverage_pass` | `True` |
| `optimizer_steps` | `99 961` |
| `param_count` | `801 808` |
| `elapsed_seconds` | `12875.8` |
| `max_effective_bc` | `510` |
| `amp_nonfinite_grad_steps` | `39` |

## Trazabilidad con Drive

Los artefactos pesados (`metrics.jsonl` por step, checkpoints `.pt`)
viven solo en Drive bajo `MyDrive/fm_fl_phmd/logs/pretraining/<run_name>/`
y `MyDrive/fm_fl_phmd/checkpoints/<run_name>/`. Este README y los
`run_info.json` aqui versionados son el contrato citable.

Generado automaticamente por `notebooks/utils/recover_drive_artifacts.py`.
