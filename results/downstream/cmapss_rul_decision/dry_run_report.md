# CMAPSS RUL builder — dry-run report

Timestamp: 2026-05-25T10:55:23
raw_root: `/content/drive/MyDrive/fm_fl_phmd/raw/datasets`
raw_missing: **False**
raw_layout: **zips_split_phmd**

- raw_zip_train_path: `/content/drive/MyDrive/fm_fl_phmd/raw/datasets/CMAPSS_train.zip`
- raw_zip_test_path: `/content/drive/MyDrive/fm_fl_phmd/raw/datasets/CMAPSS_test.zip`

## FD subsets detectados

- subsets: ['FD001', 'FD002', 'FD003', 'FD004']
- train_FDxxx: OK
- test_FDxxx: OK
- RUL_FDxxx: OK

## Ficheros encontrados (primeros 80)

- `RUL_FD001.txt`
- `RUL_FD002.txt`
- `RUL_FD003.txt`
- `RUL_FD004.txt`
- `test_FD001.txt`
- `test_FD002.txt`
- `test_FD003.txt`
- `test_FD004.txt`
- `train_FD001.txt`
- `train_FD002.txt`
- `train_FD003.txt`
- `train_FD004.txt`

## Notas

- layout=zips_split_phmd; encontrados CMAPSS_train.zip (4 entradas) y CMAPSS_test.zip (8 entradas); inspeccionados sin extraer.
- FD subsets detectados: ['FD001', 'FD002', 'FD003', 'FD004']


## Politica de seleccion rolling_causal (post-filtro)

- window_size: `512`
- patch_size: `16`
- n_patches: `32`
- n_channels: `24`
- window_mode: `rolling_causal`
- target_policy: `rul_at_prediction_cycle`
- split_policy: `unit_holdout_by_fd`
- normalization_policy: `instance_norm_per_window_channel_ignore_padding`
- **stride**: `5`
- **min_valid_timesteps**: `128`
- **include_last_per_unit**: `True`

## Totales post-filtro

- n_windows_train: **11795**
- n_windows_val: **3131**
- n_windows_test: **6610**
- n_windows_dropped_by_min_valid: 28938
- n_windows_added_by_last_override: 1200
- estimated_size_gb_float32: **1.01 GB**
- frac_windows_padded: 0.9995
- frac_timesteps_valid_avg: 0.3807

## Decision aplicable

Reconstruir RUL fisico desde raw (formula por split). Politica stride/min_valid/last_override confirmada. Ver `results/downstream/cmapss_rul_decision/decision.md`.
