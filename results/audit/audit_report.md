# Auditoria de datasets PHMD (v2.3)

Generado por `notebooks/exploration/02_dataset_audit.ipynb`. Las decisiones son heuristicas y deben revisarse manualmente para los casos limite.

## Cambios v2.3 frente a v2.2

- `tail_policy='pad'` adoptado como politica operacional global. La ventana extra parcial con padding por la derecha sustituye al descarte de cola del modo `drop`. La justificacion esta en `results/audit/tail_policy_comparison.json` (comparativa sobre los 7 datasets con `tail_drop_ratio > 0.05` en v2.2): PHM18 como TT primary perdia el 23.2% de sus filas crudas con `drop`. Las mascaras `valid_time_mask` y `valid_patch_mask` ya forman parte del contrato y absorben el padding anadido. El diff agregado v2.2 -> v2.3 esta en `results/audit/tail_policy_diff_v22_v23.csv`.
- Conteo exacto de patches validos por ventana (sin derivar de `padding_ratio`).
- Metricas densas nuevas: `n_dense_temporal_patches`, `n_dense_channel_patches`, `invalid_patch_ratio`, `dense_vs_valid_ratio`. Permiten distinguir senal valida del coste denso real del encoder channel-independent.
- `audit_summary.json` incluye `tail_policy` top-level y un bloque `tail_policy_decision` autocontenido con la justificacion.
- `sampling_policy` con valores iniciales numericos: `cap_max_dataset_weight=0.10`, `cap_max_client_weight=0.25`, `min_client_presence=0.005`.
- Warning nuevo `tail_policy_pad_padding_moderado` para TT con padding entre 0.15 y 0.5 (caso esperado: PHM18 y PHMAP23 tras pad).
- Asserts de cierre tras agregar: roles, CMAPSS=TT primary, lista TT exacta y ausencia de padding extremo en PS/TT. La generacion del CSV falla si algo cambia silenciosamente.

### Heredado de v2.2

- `decidir_role` respeta `TRANSFER_TARGETS_PROPUESTOS` antes de aplicar los DROP por longitud o numero de ventanas (las reglas duras `nan_pct_max` y `padding_ratio > 0.8` siguen aplicando a todos).
- `estimated_n_shards = suma de techos por split` (alineado con la escritura real).
- `audit_groups.json` con estructura `{audit_version, timestamp, window_size, stride, patch_size, tail_policy, clients}`.
- Overrides manuales: `SUBSET_ID_OVERRIDE={CMAPSS:FD}`, `TARGET_COL_OVERRIDE={UNIBO21:soc}`. Trayectorias agrupadas por `split + subset_id + unit_col`.
- `sampling_rate_info` agregado en `audit_summary.json` (no warning por dataset).

## Nota sobre patches temporales, por canal y densos

- `n_temporal_patches` = patches validos por ventana, agregado.
- `n_channel_patches`  = `n_temporal_patches * n_canales`. Base para coste/dominancia.
- `n_dense_temporal_patches` = `n_windows * N_PATCHES` (sin descontar padding).
- `n_dense_channel_patches`  = `n_dense_temporal_patches * n_canales`.
- `invalid_patch_ratio`      = `1 - n_channel_patches / n_dense_channel_patches`.
- `dense_vs_valid_ratio`     = `n_dense_channel_patches / n_channel_patches`.

Con `tail_policy=pad`, el coste denso puede subir mas que los patches validos. La loss SSL debe enmascarar tanto el patch invalido (`valid_patch_mask`) como el padding dentro de un patch parcialmente valido (`valid_time_mask.reshape(N, P)`).

## Resumen

- Datasets en el reporte: **53**
- Por role:
  - `DROP`: 4
  - `EXCLUDED`: 2
  - `PRETRAIN_SOURCE`: 36
  - `TRANSFER_TARGET`: 11
- Por dominio:
  - `aero_engines`: 2
  - `batteries`: 6
  - `bearings`: 13
  - `building_hvac`: 1
  - `capacitors`: 1
  - `cnc_milling`: 2
  - `compressor`: 1
  - `drills`: 1
  - `gearboxes`: 3
  - `hdd`: 2
  - `learning_curves`: 1
  - `misc`: 7
  - `mosfets_power`: 2
  - `naval`: 2
  - `phm_challenges`: 6
  - `transformers`: 1
  - `wind`: 2

## Tabla resumen

| dataset | dominio | role | evaluation_tier | subset_id_col_detected | tail_policy | n_unidades_total | n_canales | len_mediana | n_windows | padding_ratio | n_channel_patches | n_dense_channel_patches | invalid_patch_ratio | tipo_target |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| AHU21 | building_hvac | DROP |  |  | pad | 169.0 | 23.0 | 109.0 | 1266.0 | 0.1102 | 830576.0 | 931776.0 | 0.1086 | classification_binary |
| GENDEM18 | misc | DROP |  |  | pad | 9.0 | 23.0 | 2227.0 | 84.0 | 0.0738 | 57362.0 | 61824.0 | 0.0722 | classification_multiclass |
| PBHP16 | misc | DROP |  |  | pad | 18.0 | 1.0 | 665.0 | 41.0 | 0.2317 | 1014.0 | 1312.0 | 0.2271 | classification_multiclass |
| MOSFET11 | mosfets_power | DROP |  |  | pad | 5854.0 | 9.0 | 1.0 | 5854.0 | 0.998 | 52686.0 | 1685952.0 | 0.9688 | rul |
| CURVES | learning_curves | EXCLUDED |  |  |  |  |  |  |  |  |  |  |  |  |
| PHM19 | phm_challenges | EXCLUDED |  |  |  |  |  |  |  |  |  |  |  |  |
| NCMAPSS | aero_engines | PRETRAIN_SOURCE |  |  | pad | 60.0 | 20.0 | 735972.0 | 172869.0 | 0.0003 | 110608120.0 | 110636160.0 | 0.0003 | rul |
| CALCE_CX2 | batteries | PRETRAIN_SOURCE |  |  | pad | 6.0 | 11.0 | 470227.0 | 11523.0 | 0.0004 | 4054413.0 | 4056096.0 | 0.0004 | rul |
| FCLB19 | batteries | PRETRAIN_SOURCE |  |  | pad | 138.0 | 7.0 | 10444.0 | 5530.0 | 0.0201 | 1214136.0 | 1238720.0 | 0.0198 | rul |
| NB1 | batteries | PRETRAIN_SOURCE |  |  | pad | 34.0 | 18.0 | 133545.0 | 28795.0 | 0.0008 | 16572582.0 | 16585920.0 | 0.0008 | rul |
| NB14 | batteries | PRETRAIN_SOURCE |  |  | pad | 22.0 | 3.0 | 1395314.0 | 122014.0 | 0.0001 | 11711802.0 | 11713344.0 | 0.0001 | rul |
| UNIBO21 | batteries | PRETRAIN_SOURCE |  |  | pad | 34.0 | 3.0 | 78549.0 | 12228.0 | 0.0021 | 1171437.0 | 1173888.0 | 0.0021 | regression |
| IMS | bearings | PRETRAIN_SOURCE |  |  | pad | 3.0 | 5.0 | 88309760.0 | 929597.0 | 0.0 | 148735520.0 | 148735520.0 | 0.0 | rul |
| JNUB | bearings | PRETRAIN_SOURCE |  |  | pad | 12.0 | 2.0 | 500500.0 | 35190.0 | 0.0003 | 2251452.0 | 2252160.0 | 0.0003 | classification_multiclass |
| KAUG17 | bearings | PRETRAIN_SOURCE |  |  | pad | 31.0 | 1.0 | 200000.0 | 24211.0 | 0.0011 | 773884.0 | 774752.0 | 0.0011 | classification_multiclass |
| LGB20 | bearings | PRETRAIN_SOURCE |  |  | pad | 39.0 | 4.0 | 62025.0 | 9580.0 | 0.003 | 1222676.0 | 1226240.0 | 0.0029 | regression |
| MFPT | bearings | PRETRAIN_SOURCE |  |  | pad | 17.0 | 1.0 | 146484.0 | 14872.0 | 0.001 | 475455.0 | 475904.0 | 0.0009 | classification_multiclass |
| PRONOSTIA | bearings | PRETRAIN_SOURCE |  |  | pad | 17.0 | 2.0 | 2915840.0 | 214913.0 | 0.0 | 13754432.0 | 13754432.0 | 0.0 | rul |
| RRB23 | bearings | PRETRAIN_SOURCE |  |  | pad | 15.0 | 5.0 | 12273.0 | 1057.0 | 0.0103 | 167430.0 | 169120.0 | 0.01 | rul |
| SEUGB17 | bearings | PRETRAIN_SOURCE |  |  | pad | 60.0 | 8.0 | 314568.0 | 81880.0 | 0.0005 | 20951040.0 | 20961280.0 | 0.0005 | classification_multiclass |
| UOC18 | bearings | PRETRAIN_SOURCE |  |  | pad | 936.0 | 1.0 | 3600.0 | 13104.0 | 0.0692 | 390312.0 | 419328.0 | 0.0692 | classification_multiclass |
| UPM20 | bearings | PRETRAIN_SOURCE |  |  | pad | 135.0 | 3.0 | 360000.0 | 210870.0 | 0.0005 | 20233800.0 | 20243520.0 | 0.0005 | classification_multiclass |
| UPM23 | bearings | PRETRAIN_SOURCE |  |  | pad | 15.0 | 3.0 | 2320000.0 | 135930.0 | 0.0001 | 13048200.0 | 13049280.0 | 0.0001 | classification_multiclass |
| XJTU-SY | bearings | PRETRAIN_SOURCE |  |  | pad | 15.0 | 2.0 | 5275648.0 | 1179633.0 | 0.0 | 75496512.0 | 75496512.0 | 0.0 | rul |
| CESNASA15 | capacitors | PRETRAIN_SOURCE |  |  | pad | 24.0 | 9.0 | 3366.0 | 312.0 | 0.0712 | 83592.0 | 89856.0 | 0.0697 | classification_multiclass |
| NMILL | cnc_milling | PRETRAIN_SOURCE |  |  | pad | 31.0 | 6.0 | 37124.0 | 4836.0 | 0.0048 | 924138.0 | 928512.0 | 0.0047 | classification_multiclass |
| AC16 | compressor | PRETRAIN_SOURCE |  |  | pad | 1800.0 | 1.0 | 50000.0 | 351000.0 | 0.0043 | 11183400.0 | 11232000.0 | 0.0043 | classification_multiclass |
| DFD15 | drills | PRETRAIN_SOURCE |  |  | pad | 119.0 | 1.0 | 262144.0 | 121737.0 | 0.0 | 3895584.0 | 3895584.0 | 0.0 | classification_multiclass |
| ARAMIS20 | gearboxes | PRETRAIN_SOURCE |  |  | pad | 200.0 | 11.0 | 4000.0 | 2992.0 | 0.0461 | 1004696.0 | 1053184.0 | 0.046 | classification_binary |
| PHMAP21 | gearboxes | PRETRAIN_SOURCE |  |  | pad | 27.0 | 2.0 | 1461399.0 | 181887.0 | 0.0001 | 11639440.0 | 11640768.0 | 0.0001 | classification_multiclass |
| HSF15 | hdd | PRETRAIN_SOURCE |  |  | pad | 2204.0 | 17.0 | 6000.0 | 50692.0 | 0.034 | 26639748.0 | 27576448.0 | 0.034 | classification_multiclass |
| DUS20 | misc | PRETRAIN_SOURCE |  |  | pad | 100.0 | 1.0 | 2051.0 | 710.0 | 0.1047 | 20382.0 | 22720.0 | 0.1029 | classification_multiclass |
| HIRFNASA15 | misc | PRETRAIN_SOURCE |  |  | pad | 159.0 | 7.0 | 499694.0 | 314788.0 | 0.0004 | 70486822.0 | 70512512.0 | 0.0004 | rul |
| OBDD17 | misc | PRETRAIN_SOURCE |  |  | pad | 12.0 | 4.0 | 1133285.0 | 59618.0 | 0.0001 | 7630064.0 | 7631104.0 | 0.0001 | rul |
| SSPSNASA15 | misc | PRETRAIN_SOURCE |  |  | pad | 6.0 | 5.0 | 12547.0 | 387.0 | 0.0108 | 61265.0 | 61920.0 | 0.0106 | rul |
| CBMv3 | naval | PRETRAIN_SOURCE |  |  | pad | 3.0 | 25.0 | 176766.0 | 2301.0 | 0.0012 | 1838675.0 | 1840800.0 | 0.0012 | regression |
| PHM10 | phm_challenges | PRETRAIN_SOURCE |  |  | pad | 3.0 | 8.0 | 69200145.0 | 815073.0 | 0.0 | 208658088.0 | 208658688.0 | 0.0 | regression_or_rul |
| PHM15 | phm_challenges | PRETRAIN_SOURCE |  |  | pad | 3.0 | 13.0 | 1873716.0 | 27454.0 | 0.0001 | 11419785.0 | 11420864.0 | 0.0001 | classification_multiclass |
| PHME24 | phm_challenges | PRETRAIN_SOURCE |  |  | pad | 48.0 | 17.0 | 624375.0 | 112858.0 | 0.0003 | 61375338.0 | 61394752.0 | 0.0003 | rul |
| PPD18 | phm_challenges | PRETRAIN_SOURCE |  |  | pad | 10.0 | 25.0 | 20385.0 | 888.0 | 0.0089 | 704225.0 | 710400.0 | 0.0087 | rul |
| PTRB19 | transformers | PRETRAIN_SOURCE |  |  | pad | 119.0 | 6.0 | 512000.0 | 237881.0 | 0.0 | 45673152.0 | 45673152.0 | 0.0 | classification_multiclass |
| PHM14 | wind | PRETRAIN_SOURCE |  |  | pad | 1702.0 | 317.0 | 538.0 | 2664.0 | 0.3951 | 16593682.0 | 27023616.0 | 0.386 | classification_binary |
| CMAPSS | aero_engines | TRANSFER_TARGET | primary | FD | pad | 1416.0 | 24.0 | 184.0 | 1418.0 | 0.6346 | 414504.0 | 1089024.0 | 0.6194 | rul |
| CALCE_CS2 | batteries | TRANSFER_TARGET | primary |  | pad | 6.0 | 11.0 | 281261.0 | 6571.0 | 0.0007 | 2311342.0 | 2312992.0 | 0.0007 | rul |
| CWRU | bearings | TRANSFER_TARGET | primary |  | pad | 153.0 | 2.0 | 122136.0 | 140134.0 | 0.0008 | 8961316.0 | 8968576.0 | 0.0008 | classification_multiclass |
| CNCMILL18 | cnc_milling | TRANSFER_TARGET | secondary |  | pad | 18.0 | 48.0 | 1341.0 | 91.0 | 0.1496 | 119280.0 | 139776.0 | 0.1466 | classification_binary |
| PHMAP23 | gearboxes | TRANSFER_TARGET | secondary |  | pad | 177.0 | 8.0 | 1201.0 | 708.0 | 0.1636 | 152928.0 | 181248.0 | 0.1562 | classification_multiclass |
| HSG18 | hdd | TRANSFER_TARGET | primary |  | pad | 17.0 | 1.0 | 585936.0 | 38896.0 | 0.0003 | 1244349.0 | 1244672.0 | 0.0003 | classification_binary |
| PBCP16 | misc | TRANSFER_TARGET | primary |  | pad | 25.0 | 1.0 | 20480.0 | 1975.0 | 0.0 | 63200.0 | 63200.0 | 0.0 | classification_multiclass |
| IEEE14 | mosfets_power | TRANSFER_TARGET | primary |  | pad | 3.0 | 24.0 | 95908.0 | 1058.0 | 0.0021 | 810864.0 | 812544.0 | 0.0021 | regression_or_rul |
| CBM14 | naval | TRANSFER_TARGET | secondary |  | pad | 3.0 | 16.0 | 3580.0 | 44.0 | 0.0384 | 21680.0 | 22528.0 | 0.0376 | regression |
| PHME20 | phm_challenges | TRANSFER_TARGET | secondary |  | pad | 30.0 | 5.0 | 2395.0 | 277.0 | 0.0739 | 41115.0 | 44320.0 | 0.0723 | rul |
| PHM18 | wind | TRANSFER_TARGET | primary |  | pad | 1207.0 | 22.0 | 1000.0 | 3621.0 | 0.1823 | 2097766.0 | 2549184.0 | 0.1771 | classification_multiclass |

## Diff v2.2 (drop) -> v2.3 (pad) - top 10 deltas

Esta es la lectura agregada de los cambios en metricas tras pasar a `tail_policy='pad'`. La tabla detallada esta en `results/audit/tail_policy_diff_v22_v23.csv` y la justificacion (sobre los 7 datasets con `tail_drop_ratio_alto`) en `results/audit/tail_policy_comparison.json`.

| dataset | role | n_windows_v22 | n_windows_v23 | delta_windows | padding_v22 | padding_v23 | n_channel_patches_v22 | n_channel_patches_v23 | delta_channel_patches_pct |
|---|---|---|---|---|---|---|---|---|---|
| PHM18 | TRANSFER_TARGET | 2414.0 | 3621.0 | 1207.0 | 0.0 | 0.1823 | 1699456.0 | 2097766.0 | 23.44 |
| PHM14 | PRETRAIN_SOURCE | 1704.0 | 2664.0 | 960.0 | 0.1677 | 0.3951 | 14491972.0 | 16593682.0 | 14.5 |
| PHMAP23 | TRANSFER_TARGET | 531.0 | 708.0 | 177.0 | 0.0 | 0.1636 | 135936.0 | 152928.0 | 12.5 |
| CNCMILL18 | TRANSFER_TARGET | 74.0 | 91.0 | 17.0 | 0.0013 | 0.1496 | 113520.0 | 119280.0 | 5.07 |
| DUS20 | PRETRAIN_SOURCE | 611.0 | 710.0 | 99.0 | 0.0 | 0.1047 | 19552.0 | 20382.0 | 4.25 |
| PHME20 | TRANSFER_TARGET | 247.0 | 277.0 | 30.0 | 0.0 | 0.0739 | 39520.0 | 41115.0 | 4.04 |
| CBM14 | TRANSFER_TARGET | 41.0 | 44.0 | 3.0 | 0.0 | 0.0384 | 20992.0 | 21680.0 | 3.28 |
| ARAMIS20 | PRETRAIN_SOURCE | 2793.0 | 2992.0 | 199.0 | 0.0 | 0.0461 | 983136.0 | 1004696.0 | 2.19 |
| HSF15 | PRETRAIN_SOURCE | 48488.0 | 50692.0 | 2204.0 | 0.0 | 0.034 | 26377472.0 | 26639748.0 | 0.99 |
| CESNASA15 | PRETRAIN_SOURCE | 288.0 | 312.0 | 24.0 | 0.0 | 0.0712 | 82944.0 | 83592.0 | 0.78 |

## PRETRAIN_SOURCE (36 datasets)

### NCMAPSS  ·  dominio: `aero_engines`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 60.0 trayectorias, 20.0 canales, longitud mediana 735972.0
- Ventaneo (W=512, tail=pad): 172869.0 ventanas, padding_ratio=0.0003, ~5530406.0 temporal patches, ~110608120.0 channel patches validos, ~110636160.0 densos (invalid_patch_ratio=0.0003), ~7080.71 MB, ~35.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 99.0]
- Peso en corpus PS (channel patches): 11.99%
- Plot: `audit/plots/NCMAPSS.png`

### CALCE_CX2  ·  dominio: `batteries`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 6.0 trayectorias, 11.0 canales, longitud mediana 470227.0
- Ventaneo (W=512, tail=pad): 11523.0 ventanas, padding_ratio=0.0004, ~368583.0 temporal patches, ~4054413.0 channel patches validos, ~4056096.0 densos (invalid_patch_ratio=0.0004), ~259.59 MB, ~4.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 1960.0]
- Peso en corpus PS (channel patches): 0.44%
- Plot: `audit/plots/CALCE_CX2.png`

### FCLB19  ·  dominio: `batteries`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 138.0 trayectorias, 7.0 canales, longitud mediana 10444.0
- Ventaneo (W=512, tail=pad): 5530.0 ventanas, padding_ratio=0.0201, ~173448.0 temporal patches, ~1214136.0 channel patches validos, ~1238720.0 densos (invalid_patch_ratio=0.0198), ~79.28 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [49.0, 1935.0]
- Peso en corpus PS (channel patches): 0.13%
- Plot: `audit/plots/FCLB19.png`

### NB1  ·  dominio: `batteries`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 34.0 trayectorias, 18.0 canales, longitud mediana 133545.0
- Ventaneo (W=512, tail=pad): 28795.0 ventanas, padding_ratio=0.0008, ~920699.0 temporal patches, ~16572582.0 channel patches validos, ~16585920.0 densos (invalid_patch_ratio=0.0008), ~1061.5 MB, ~7.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 615.0]
- Peso en corpus PS (channel patches): 1.80%
- Plot: `audit/plots/NB1.png`

### NB14  ·  dominio: `batteries`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 22.0 trayectorias, 3.0 canales, longitud mediana 1395314.0
- Ventaneo (W=512, tail=pad): 122014.0 ventanas, padding_ratio=0.0001, ~3903934.0 temporal patches, ~11711802.0 channel patches validos, ~11713344.0 densos (invalid_patch_ratio=0.0001), ~749.65 MB, ~27.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 1289.0]
- Peso en corpus PS (channel patches): 1.27%
- Plot: `audit/plots/NB14.png`

### UNIBO21  ·  dominio: `batteries`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 34.0 trayectorias, 3.0 canales, longitud mediana 78549.0
- Ventaneo (W=512, tail=pad): 12228.0 ventanas, padding_ratio=0.0021, ~390479.0 temporal patches, ~1171437.0 channel patches validos, ~1173888.0 densos (invalid_patch_ratio=0.0021), ~75.13 MB, ~4.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `soc` (regression, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `soc`
- Rango target: [-1.2494, 1.0]
- Peso en corpus PS (channel patches): 0.13%
- Plot: `audit/plots/UNIBO21.png`

### IMS  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 3.0 trayectorias, 5.0 canales, longitud mediana 88309760.0
- Ventaneo (W=512, tail=pad): 929597.0 ventanas, padding_ratio=0.0, ~29747104.0 temporal patches, ~148735520.0 channel patches validos, ~148735520.0 densos (invalid_patch_ratio=0.0), ~9519.07 MB, ~187.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 1073.2525]
- Peso en corpus PS (channel patches): 16.12%
- Plot: `audit/plots/IMS.png`

### JNUB  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 12.0 trayectorias, 2.0 canales, longitud mediana 500500.0
- Ventaneo (W=512, tail=pad): 35190.0 ventanas, padding_ratio=0.0003, ~1125726.0 temporal patches, ~2251452.0 channel patches validos, ~2252160.0 densos (invalid_patch_ratio=0.0003), ~144.14 MB, ~9.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 4.0, balance 0.3333
- Peso en corpus PS (channel patches): 0.24%
- Plot: `audit/plots/JNUB.png`

### KAUG17  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 31.0 trayectorias, 1.0 canales, longitud mediana 200000.0
- Ventaneo (W=512, tail=pad): 24211.0 ventanas, padding_ratio=0.0011, ~773884.0 temporal patches, ~773884.0 channel patches validos, ~774752.0 densos (invalid_patch_ratio=0.0011), ~49.58 MB, ~6.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 3.0, balance 0.9091
- Peso en corpus PS (channel patches): 0.08%
- Plot: `audit/plots/KAUG17.png`

### LGB20  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 39.0 trayectorias, 4.0 canales, longitud mediana 62025.0
- Ventaneo (W=512, tail=pad): 9580.0 ventanas, padding_ratio=0.003, ~305669.0 temporal patches, ~1222676.0 channel patches validos, ~1226240.0 densos (invalid_patch_ratio=0.0029), ~78.48 MB, ~4.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `soc` (regression, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `soc`
- Rango target: [0.0, 0.8639]
- Peso en corpus PS (channel patches): 0.13%
- Plot: `audit/plots/LGB20.png`

### MFPT  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 17.0 trayectorias, 1.0 canales, longitud mediana 146484.0
- Ventaneo (W=512, tail=pad): 14872.0 ventanas, padding_ratio=0.001, ~475455.0 temporal patches, ~475455.0 channel patches validos, ~475904.0 densos (invalid_patch_ratio=0.0009), ~30.46 MB, ~5.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 3.0, balance 0.5833
- Peso en corpus PS (channel patches): 0.05%
- Plot: `audit/plots/MFPT.png`

### PRONOSTIA  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 17.0 trayectorias, 2.0 canales, longitud mediana 2915840.0
- Ventaneo (W=512, tail=pad): 214913.0 ventanas, padding_ratio=0.0, ~6877216.0 temporal patches, ~13754432.0 channel patches validos, ~13754432.0 densos (invalid_patch_ratio=0.0), ~880.28 MB, ~44.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 28029.0]
- Peso en corpus PS (channel patches): 1.49%
- Plot: `audit/plots/PRONOSTIA.png`

### RRB23  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 15.0 trayectorias, 5.0 canales, longitud mediana 12273.0
- Ventaneo (W=512, tail=pad): 1057.0 ventanas, padding_ratio=0.0103, ~33486.0 temporal patches, ~167430.0 channel patches validos, ~169120.0 densos (invalid_patch_ratio=0.01), ~10.82 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 1779.0]
- Peso en corpus PS (channel patches): 0.02%
- Plot: `audit/plots/RRB23.png`

### SEUGB17  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 60.0 trayectorias, 8.0 canales, longitud mediana 314568.0
- Ventaneo (W=512, tail=pad): 81880.0 ventanas, padding_ratio=0.0005, ~2618880.0 temporal patches, ~20951040.0 channel patches validos, ~20961280.0 densos (invalid_patch_ratio=0.0005), ~1341.52 MB, ~17.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 10.0, balance 1.0
- Peso en corpus PS (channel patches): 2.27%
- Plot: `audit/plots/SEUGB17.png`

### UOC18  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 936.0 trayectorias, 1.0 canales, longitud mediana 3600.0
- Ventaneo (W=512, tail=pad): 13104.0 ventanas, padding_ratio=0.0692, ~390312.0 temporal patches, ~390312.0 channel patches validos, ~419328.0 densos (invalid_patch_ratio=0.0692), ~26.84 MB, ~4.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 9.0, balance 1.0
- Peso en corpus PS (channel patches): 0.04%
- Plot: `audit/plots/UOC18.png`

### UPM20  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 135.0 trayectorias, 3.0 canales, longitud mediana 360000.0
- Ventaneo (W=512, tail=pad): 210870.0 ventanas, padding_ratio=0.0005, ~6744600.0 temporal patches, ~20233800.0 channel patches validos, ~20243520.0 densos (invalid_patch_ratio=0.0005), ~1295.59 MB, ~43.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 5.0, balance 1.0
- Peso en corpus PS (channel patches): 2.19%
- Plot: `audit/plots/UPM20.png`

### UPM23  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 15.0 trayectorias, 3.0 canales, longitud mediana 2320000.0
- Ventaneo (W=512, tail=pad): 135930.0 ventanas, padding_ratio=0.0001, ~4349400.0 temporal patches, ~13048200.0 channel patches validos, ~13049280.0 densos (invalid_patch_ratio=0.0001), ~835.15 MB, ~29.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault_component` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault_component`
- Clases: 4.0, balance 0.75
- Peso en corpus PS (channel patches): 1.41%
- Plot: `audit/plots/UPM23.png`

### XJTU-SY  ·  dominio: `bearings`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 15.0 trayectorias, 2.0 canales, longitud mediana 5275648.0
- Ventaneo (W=512, tail=pad): 1179633.0 ventanas, padding_ratio=0.0, ~37748256.0 temporal patches, ~75496512.0 channel patches validos, ~75496512.0 densos (invalid_patch_ratio=0.0), ~4831.78 MB, ~237.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 2537.0]
- Peso en corpus PS (channel patches): 8.18%
- Plot: `audit/plots/XJTU-SY.png`

### CESNASA15  ·  dominio: `capacitors`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 24.0 trayectorias, 9.0 canales, longitud mediana 3366.0
- Ventaneo (W=512, tail=pad): 312.0 ventanas, padding_ratio=0.0712, ~9288.0 temporal patches, ~83592.0 channel patches validos, ~89856.0 densos (invalid_patch_ratio=0.0697), ~5.75 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul1` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `rul1`
- Clases: 59.0, balance 0.0038
- Peso en corpus PS (channel patches): 0.01%
- Plot: `audit/plots/CESNASA15.png`

### NMILL  ·  dominio: `cnc_milling`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 31.0 trayectorias, 6.0 canales, longitud mediana 37124.0
- Ventaneo (W=512, tail=pad): 4836.0 ventanas, padding_ratio=0.0048, ~154023.0 temporal patches, ~924138.0 channel patches validos, ~928512.0 densos (invalid_patch_ratio=0.0047), ~59.42 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `CBM` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `CBM`
- Clases: 3.0, balance 0.1061
- Peso en corpus PS (channel patches): 0.10%
- Plot: `audit/plots/NMILL.png`

### AC16  ·  dominio: `compressor`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 1800.0 trayectorias, 1.0 canales, longitud mediana 50000.0
- Ventaneo (W=512, tail=pad): 351000.0 ventanas, padding_ratio=0.0043, ~11183400.0 temporal patches, ~11183400.0 channel patches validos, ~11232000.0 densos (invalid_patch_ratio=0.0043), ~718.85 MB, ~71.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 8.0, balance 1.0
- Peso en corpus PS (channel patches): 1.21%
- Plot: `audit/plots/AC16.png`

### DFD15  ·  dominio: `drills`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 119.0 trayectorias, 1.0 canales, longitud mediana 262144.0
- Ventaneo (W=512, tail=pad): 121737.0 ventanas, padding_ratio=0.0, ~3895584.0 temporal patches, ~3895584.0 channel patches validos, ~3895584.0 densos (invalid_patch_ratio=0.0), ~249.32 MB, ~26.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 4.0, balance 0.9667
- Peso en corpus PS (channel patches): 0.42%
- Plot: `audit/plots/DFD15.png`

### ARAMIS20  ·  dominio: `gearboxes`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 200.0 trayectorias, 11.0 canales, longitud mediana 4000.0
- Ventaneo (W=512, tail=pad): 2992.0 ventanas, padding_ratio=0.0461, ~91336.0 temporal patches, ~1004696.0 channel patches validos, ~1053184.0 densos (invalid_patch_ratio=0.046), ~67.4 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_binary, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 2.0, balance 0.0419
- Peso en corpus PS (channel patches): 0.11%
- Plot: `audit/plots/ARAMIS20.png`

### PHMAP21  ·  dominio: `gearboxes`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 27.0 trayectorias, 2.0 canales, longitud mediana 1461399.0
- Ventaneo (W=512, tail=pad): 181887.0 ventanas, padding_ratio=0.0001, ~5819720.0 temporal patches, ~11639440.0 channel patches validos, ~11640768.0 densos (invalid_patch_ratio=0.0001), ~745.01 MB, ~37.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 4.0, balance 0.5657
- Peso en corpus PS (channel patches): 1.26%
- Plot: `audit/plots/PHMAP21.png`

### HSF15  ·  dominio: `hdd`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 2204.0 trayectorias, 17.0 canales, longitud mediana 6000.0
- Ventaneo (W=512, tail=pad): 50692.0 ventanas, padding_ratio=0.034, ~1567044.0 temporal patches, ~26639748.0 channel patches validos, ~27576448.0 densos (invalid_patch_ratio=0.034), ~1764.89 MB, ~12.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `cooler` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `cooler`
- Clases: 3.0, balance 0.9865
- Peso en corpus PS (channel patches): 2.89%
- Plot: `audit/plots/HSF15.png`

### DUS20  ·  dominio: `misc`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 100.0 trayectorias, 1.0 canales, longitud mediana 2051.0
- Ventaneo (W=512, tail=pad): 710.0 ventanas, padding_ratio=0.1047, ~20382.0 temporal patches, ~20382.0 channel patches validos, ~22720.0 densos (invalid_patch_ratio=0.1029), ~1.45 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `unbalance` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `unbalance`
- Clases: 5.0, balance 0.7024
- Peso en corpus PS (channel patches): 0.00%
- Plot: `audit/plots/DUS20.png`

### HIRFNASA15  ·  dominio: `misc`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 159.0 trayectorias, 7.0 canales, longitud mediana 499694.0
- Ventaneo (W=512, tail=pad): 314788.0 ventanas, padding_ratio=0.0004, ~10069546.0 temporal patches, ~70486822.0 channel patches validos, ~70512512.0 densos (invalid_patch_ratio=0.0004), ~4512.8 MB, ~65.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 2538.0]
- Peso en corpus PS (channel patches): 7.64%
- Plot: `audit/plots/HIRFNASA15.png`

### OBDD17  ·  dominio: `misc`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 12.0 trayectorias, 4.0 canales, longitud mediana 1133285.0
- Ventaneo (W=512, tail=pad): 59618.0 ventanas, padding_ratio=0.0001, ~1907516.0 temporal patches, ~7630064.0 channel patches validos, ~7631104.0 densos (invalid_patch_ratio=0.0001), ~488.39 MB, ~13.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul;state`
- Rango target: [0.0, 77.0]
- Peso en corpus PS (channel patches): 0.83%
- Plot: `audit/plots/OBDD17.png`

### SSPSNASA15  ·  dominio: `misc`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 6.0 trayectorias, 5.0 canales, longitud mediana 12547.0
- Ventaneo (W=512, tail=pad): 387.0 ventanas, padding_ratio=0.0108, ~12253.0 temporal patches, ~61265.0 channel patches validos, ~61920.0 densos (invalid_patch_ratio=0.0106), ~3.96 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 38.0]
- Peso en corpus PS (channel patches): 0.01%
- Plot: `audit/plots/SSPSNASA15.png`

### CBMv3  ·  dominio: `naval`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 3.0 trayectorias, 25.0 canales, longitud mediana 176766.0
- Ventaneo (W=512, tail=pad): 2301.0 ventanas, padding_ratio=0.0012, ~73547.0 temporal patches, ~1838675.0 channel patches validos, ~1840800.0 densos (invalid_patch_ratio=0.0012), ~117.81 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `ptdsc_port` (regression, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `ptdsc_port`
- Rango target: [0.9, 1.0]
- Peso en corpus PS (channel patches): 0.20%
- Plot: `audit/plots/CBMv3.png`

### PHM10  ·  dominio: `phm_challenges`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 3.0 trayectorias, 8.0 canales, longitud mediana 69200145.0
- Ventaneo (W=512, tail=pad): 815073.0 ventanas, padding_ratio=0.0, ~26082261.0 temporal patches, ~208658088.0 channel patches validos, ~208658688.0 densos (invalid_patch_ratio=0.0), ~13354.16 MB, ~165.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `wear` (regression_or_rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `wear`
- Rango target: [24.216, 215.9422]
- Peso en corpus PS (channel patches): 22.61%
- Plot: `audit/plots/PHM10.png`

### PHM15  ·  dominio: `phm_challenges`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 3.0 trayectorias, 13.0 canales, longitud mediana 1873716.0
- Ventaneo (W=512, tail=pad): 27454.0 ventanas, padding_ratio=0.0001, ~878445.0 temporal patches, ~11419785.0 channel patches validos, ~11420864.0 densos (invalid_patch_ratio=0.0001), ~730.94 MB, ~7.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fail` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fail`
- Clases: 5.0, balance 0.1681
- Peso en corpus PS (channel patches): 1.24%
- Plot: `audit/plots/PHM15.png`

### PHME24  ·  dominio: `phm_challenges`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 48.0 trayectorias, 17.0 canales, longitud mediana 624375.0
- Ventaneo (W=512, tail=pad): 112858.0 ventanas, padding_ratio=0.0003, ~3610314.0 temporal patches, ~61375338.0 channel patches validos, ~61394752.0 densos (invalid_patch_ratio=0.0003), ~3929.26 MB, ~24.0 shards
- Temporal: sampling_rate=500.0 Hz (inferido), window_time_seconds=1.024
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 4028.0]
- Peso en corpus PS (channel patches): 6.65%
- Plot: `audit/plots/PHME24.png`

### PPD18  ·  dominio: `phm_challenges`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 10.0 trayectorias, 25.0 canales, longitud mediana 20385.0
- Ventaneo (W=512, tail=pad): 888.0 ventanas, padding_ratio=0.0089, ~28169.0 temporal patches, ~704225.0 channel patches validos, ~710400.0 densos (invalid_patch_ratio=0.0087), ~45.47 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 34697.0]
- Peso en corpus PS (channel patches): 0.08%
- Plot: `audit/plots/PPD18.png`

### PTRB19  ·  dominio: `transformers`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 119.0 trayectorias, 6.0 canales, longitud mediana 512000.0
- Ventaneo (W=512, tail=pad): 237881.0 ventanas, padding_ratio=0.0, ~7612192.0 temporal patches, ~45673152.0 channel patches validos, ~45673152.0 densos (invalid_patch_ratio=0.0), ~2923.08 MB, ~49.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 3.0, balance 0.3333
- Peso en corpus PS (channel patches): 4.95%
- Plot: `audit/plots/PTRB19.png`

### PHM14  ·  dominio: `wind`
- Razon: pasa filtros minimos de calidad y volumen
- Forma: 1702.0 trayectorias, 317.0 canales, longitud mediana 538.0
- Ventaneo (W=512, tail=pad): 2664.0 ventanas, padding_ratio=0.3951, ~52346.0 temporal patches, ~16593682.0 channel patches validos, ~27023616.0 densos (invalid_patch_ratio=0.386), ~1729.51 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_binary, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 2.0, balance 0.0305
- Peso en corpus PS (channel patches): 1.80%
- Plot: `audit/plots/PHM14.png`

## TRANSFER_TARGET (11 datasets)

### CMAPSS  ·  dominio: `aero_engines`
- Razon: benchmark estandar de RUL en literatura PHM
- Evaluation tier: `primary`
- Subset id col detectado: `FD` (excluido de canales)
- Forma: 1416.0 trayectorias, 24.0 canales, longitud mediana 184.0
- Ventaneo (W=512, tail=pad): 1418.0 ventanas, padding_ratio=0.6346, ~17271.0 temporal patches, ~414504.0 channel patches validos, ~1089024.0 densos (invalid_patch_ratio=0.6194), ~69.7 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [-466.0, 542.0]
- Plot: `audit/plots/CMAPSS.png`

### CALCE_CS2  ·  dominio: `batteries`
- Razon: benchmark de RUL en baterias
- Evaluation tier: `primary`
- Forma: 6.0 trayectorias, 11.0 canales, longitud mediana 281261.0
- Ventaneo (W=512, tail=pad): 6571.0 ventanas, padding_ratio=0.0007, ~210122.0 temporal patches, ~2311342.0 channel patches validos, ~2312992.0 densos (invalid_patch_ratio=0.0007), ~148.03 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 1003.0]
- Plot: `audit/plots/CALCE_CS2.png`

### CWRU  ·  dominio: `bearings`
- Razon: benchmark clasico de clasificacion de fallos en rodamientos
- Evaluation tier: `primary`
- Forma: 153.0 trayectorias, 2.0 canales, longitud mediana 122136.0
- Ventaneo (W=512, tail=pad): 140134.0 ventanas, padding_ratio=0.0008, ~4480658.0 temporal patches, ~8961316.0 channel patches validos, ~8968576.0 densos (invalid_patch_ratio=0.0008), ~573.99 MB, ~30.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 4.0, balance 0.1252
- Plot: `audit/plots/CWRU.png`

### CNCMILL18  ·  dominio: `cnc_milling`
- Razon: unico cnc_milling con target wear claro
- Evaluation tier: `secondary`
- Forma: 18.0 trayectorias, 48.0 canales, longitud mediana 1341.0
- Ventaneo (W=512, tail=pad): 91.0 ventanas, padding_ratio=0.1496, ~2485.0 temporal patches, ~119280.0 channel patches validos, ~139776.0 densos (invalid_patch_ratio=0.1466), ~8.95 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `wear` (classification_binary, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `wear`
- Clases: 2.0, balance 0.9001
- Plot: `audit/plots/CNCMILL18.png`

### PHMAP23  ·  dominio: `gearboxes`
- Razon: clasificacion de 25 fallos en gearboxes, alta diversidad
- Evaluation tier: `secondary`
- Forma: 177.0 trayectorias, 8.0 canales, longitud mediana 1201.0
- Ventaneo (W=512, tail=pad): 708.0 ventanas, padding_ratio=0.1636, ~19116.0 temporal patches, ~152928.0 channel patches validos, ~181248.0 densos (invalid_patch_ratio=0.1562), ~11.6 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 25.0, balance 0.0286
- Plot: `audit/plots/PHMAP23.png`

### HSG18  ·  dominio: `hdd`
- Razon: unico hdd con target binario y balance razonable
- Evaluation tier: `primary`
- Forma: 17.0 trayectorias, 1.0 canales, longitud mediana 585936.0
- Ventaneo (W=512, tail=pad): 38896.0 ventanas, padding_ratio=0.0003, ~1244349.0 temporal patches, ~1244349.0 channel patches validos, ~1244672.0 densos (invalid_patch_ratio=0.0003), ~79.66 MB, ~9.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_binary, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 2.0, balance 0.5455
- Plot: `audit/plots/HSG18.png`

### PBCP16  ·  dominio: `misc`
- Razon: clasificacion 5 clases balanceada, misc bien etiquetado
- Evaluation tier: `primary`
- Forma: 25.0 trayectorias, 1.0 canales, longitud mediana 20480.0
- Ventaneo (W=512, tail=pad): 1975.0 ventanas, padding_ratio=0.0, ~63200.0 temporal patches, ~63200.0 channel patches validos, ~63200.0 densos (invalid_patch_ratio=0.0), ~4.04 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 5.0, balance 1.0
- Plot: `audit/plots/PBCP16.png`

### IEEE14  ·  dominio: `mosfets_power`
- Razon: unico mosfets_power con target RUL
- Evaluation tier: `primary`
- Forma: 3.0 trayectorias, 24.0 canales, longitud mediana 95908.0
- Ventaneo (W=512, tail=pad): 1058.0 ventanas, padding_ratio=0.0021, ~33786.0 temporal patches, ~810864.0 channel patches validos, ~812544.0 densos (invalid_patch_ratio=0.0021), ~52.0 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul35` (regression_or_rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul35`
- Rango target: [0.0, 810.3844]
- Plot: `audit/plots/IEEE14.png`

### CBM14  ·  dominio: `naval`
- Razon: unico naval con target accesible
- Evaluation tier: `secondary`
- Forma: 3.0 trayectorias, 16.0 canales, longitud mediana 3580.0
- Ventaneo (W=512, tail=pad): 44.0 ventanas, padding_ratio=0.0384, ~1355.0 temporal patches, ~21680.0 channel patches validos, ~22528.0 densos (invalid_patch_ratio=0.0376), ~1.44 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `gcdsc` (regression, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `gcdsc`
- Rango target: [0.95, 1.0]
- Plot: `audit/plots/CBM14.png`

### PHME20  ·  dominio: `phm_challenges`
- Razon: RUL pequeno y limpio, reto PHM clasico
- Evaluation tier: `secondary`
- Forma: 30.0 trayectorias, 5.0 canales, longitud mediana 2395.0
- Ventaneo (W=512, tail=pad): 277.0 ventanas, padding_ratio=0.0739, ~8223.0 temporal patches, ~41115.0 channel patches validos, ~44320.0 densos (invalid_patch_ratio=0.0723), ~2.84 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 353.9]
- Plot: `audit/plots/PHME20.png`

### PHM18  ·  dominio: `wind`
- Razon: unico wind con target multiclase claro
- Evaluation tier: `primary`
- Forma: 1207.0 trayectorias, 22.0 canales, longitud mediana 1000.0
- Ventaneo (W=512, tail=pad): 3621.0 ventanas, padding_ratio=0.1823, ~95353.0 temporal patches, ~2097766.0 channel patches validos, ~2549184.0 densos (invalid_patch_ratio=0.1771), ~163.15 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 3.0, balance 0.1849
- Plot: `audit/plots/PHM18.png`

## DROP (4 datasets)

### AHU21  ·  dominio: `building_hvac`
- Razon: mediana de longitud 109 muestras: insuficiente para W=512
- Forma: 169.0 trayectorias, 23.0 canales, longitud mediana 109.0
- Ventaneo (W=512, tail=pad): 1266.0 ventanas, padding_ratio=0.1102, ~36112.0 temporal patches, ~830576.0 channel patches validos, ~931776.0 densos (invalid_patch_ratio=0.1086), ~59.63 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_binary, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 2.0, balance 0.0063
- Plot: `audit/plots/AHU21.png`

### GENDEM18  ·  dominio: `misc`
- Razon: solo 84 ventanas estimadas (umbral 100)
- Forma: 9.0 trayectorias, 23.0 canales, longitud mediana 2227.0
- Ventaneo (W=512, tail=pad): 84.0 ventanas, padding_ratio=0.0738, ~2494.0 temporal patches, ~57362.0 channel patches validos, ~61824.0 densos (invalid_patch_ratio=0.0722), ~3.96 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 3.0, balance 0.8306
- Plot: `audit/plots/GENDEM18.png`

### PBHP16  ·  dominio: `misc`
- Razon: solo 41 ventanas estimadas (umbral 100)
- Forma: 18.0 trayectorias, 1.0 canales, longitud mediana 665.0
- Ventaneo (W=512, tail=pad): 41.0 ventanas, padding_ratio=0.2317, ~1014.0 temporal patches, ~1014.0 channel patches validos, ~1312.0 densos (invalid_patch_ratio=0.2271), ~0.08 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `fault` (classification_multiclass, policy=etiqueta_unidad); cobertura 100.0%
- Target candidates: `fault`
- Clases: 3.0, balance 0.5
- Plot: `audit/plots/PBHP16.png`

### MOSFET11  ·  dominio: `mosfets_power`
- Razon: mediana de longitud 1 muestra: dato tabular, no es serie temporal
- Forma: 5854.0 trayectorias, 9.0 canales, longitud mediana 1.0
- Ventaneo (W=512, tail=pad): 5854.0 ventanas, padding_ratio=0.998, ~5854.0 temporal patches, ~52686.0 channel patches validos, ~1685952.0 densos (invalid_patch_ratio=0.9688), ~107.9 MB, ~3.0 shards
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
- Target: `rul` (rul, policy=ultimo_valor_valido); cobertura 100.0%
- Target candidates: `rul`
- Rango target: [0.0, 250.0]
- Plot: `audit/plots/MOSFET11.png`

## EXCLUDED (2 datasets)

### CURVES  ·  dominio: `learning_curves`
- Razon: AssertionError interno en phmd al cargar el primer task
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.

### PHM19  ·  dominio: `phm_challenges`
- Razon: phmd usa DataFrame.append() (removido en pandas 2.0+)
- Temporal: sampling_rate desconocido; W=512 son muestras, no duracion fisica.
