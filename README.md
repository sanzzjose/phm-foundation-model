# Foundation Model for Predictive Maintenance

TFM MULCIA 2025/2026 — diseño, implementación y evaluación de un modelo fundacional de dominio para series temporales industriales de **Prognostics and Health Management (PHM)**, comparando preentrenamiento **centralizado** frente a **federado**.

Autor: José Sanz Durán · Tutor: David Solís Martín · Universidad de Sevilla.

## Idea general

El proyecto explora la aplicación de **Foundation Models** (modelos preentrenados de forma auto-supervisada sobre series temporales) al ámbito del PHM y el mantenimiento predictivo. La motivación es doble: los datos relevantes están dispersos entre empresas y equipos (lo que sugiere aprendizaje federado), y los avances recientes en *time-series foundation models* hacen viable construir representaciones reutilizables en muchos dominios.

El objeto transferible es un **encoder temporal** preentrenado sin etiquetas; su utilidad no se mide por la pérdida de preentrenamiento, sino por **transferencia a datasets que no participan en el preentrenamiento**. Se comparan tres regímenes:

1. **Desde cero** sobre cada dataset destino (línea base).
2. **Centralizado multi-dataset**, con preentrenamiento auto-supervisado sobre todos los datasets fuente (cota superior práctica).
3. **Federado cross-silo simulado**, donde cada dataset (o grupo coherente de datasets) actúa como un cliente.

El trabajo se articula en tres hipótesis: que el preentrenamiento auto-supervisado multi-dataset aprende representaciones reutilizables (H1), que el régimen federado puede aproximarse al centralizado bajo heterogeneidad no-IID (H2) y que FedProx mejora la optimización federada frente a FedAvg (H3).

## Datos y partición

Se usa la librería **PHMD** (Solís-Martín et al., 2025), que da acceso unificado a benchmarks públicos de PHM (CMAPSS, NCMAPSS, CWRU, XJTU-SY, CALCE, IMS, etc.). Tras auditar el catálogo (53 datasets descargados correctamente de 55 candidatas temporales), se fija una partición cerrada por roles:

- **36 PRETRAIN\_SOURCE** — solo preentrenamiento auto-supervisado.
- **11 TRANSFER\_TARGET** — reservados solo para evaluación downstream.
- **4 DROP** y **2 EXCLUDED** — descartados por el contrato o no procesables.

La separación fuente/destino es estricta (los datasets de evaluación nunca entran en el preentrenamiento), el *split* se hace por unidad antes de ventanear y todo el corpus pasa controles anti-leakage. La armonización convierte cada dataset a un contrato tensorial común `(B, C, N, P)` con ventana `W=512`, `patch=16` y `N=32` patches, con máscaras de validez para el padding.

## Modelo

**PatchTSTPhm**: un encoder temporal tipo Transformer inspirado en PatchTST y MOMENT, **channel-independent** (los mismos pesos para datasets de 1 a varios cientos de canales) y compacto (**801.808 parámetros**). El objetivo de preentrenamiento es **masked patch prediction**: se ocultan patches y el modelo los reconstruye a partir del contexto; la pérdida solo cuenta patches enmascarados y timesteps reales. La implementación está en `models/patchtst_phm.py`.

## Estructura del repositorio

```
fm_fl_phmd/
├── setup/                     Setup de Colab y dependencias (ver setup/README.md)
│   ├── colab_bootstrap.ipynb
│   ├── colab_init.sh
│   └── requirements.txt
├── notebooks/                 Notebooks de Colab
│   ├── 00_download_datasets.ipynb
│   ├── exploration/           Auditoría del corpus y análisis previo
│   ├── pretraining/           Preentrenamiento SSL
│   ├── downstream/            Evaluación downstream
│   ├── runbooks/              Guías de ejecución por fase
│   └── utils/                 Utilidades (recuperación de artefactos, etc.)
├── models/                    Arquitectura
│   └── patchtst_phm.py        Encoder PatchTSTPhm channel-independent
├── training/                  Pipeline de entrenamiento y evaluación
│   ├── configs/               Configuraciones YAML (SSL, FL y downstream)
│   ├── ssl/                   Objetivo auto-supervisado (loss, masking)
│   ├── fl/                    Cliente/servidor federado (FedAvg, FedProx, FedAvgM)
│   ├── downstream/            Cabezas, pooling y métricas downstream
│   ├── experiments/           Registro de experimentos y suites de evaluación
│   ├── train_ssl_central.py   Preentrenamiento SSL centralizado
│   ├── train_ssl_federated.py Preentrenamiento SSL federado
│   ├── train_downstream_classification.py
│   ├── train_downstream_rul.py
│   ├── build_cmapss_rul_downstream.py  Builder dedicado de CMAPSS RUL
│   ├── phm_tar_reader.py / phm_webdataset.py  Lectura de shards y DataLoader
│   └── sampling.py            Muestreo ponderado con caps
├── tests/                     Tests unitarios (pytest)
├── results/                   Artefactos versionables (ligeros)
│   ├── audit/                 Resúmenes de la auditoría del corpus
│   ├── pretraining/           run_info y métricas del SSL central
│   ├── pretraining_federated/ run_info y métricas del SSL federado
│   ├── downstream/            Resultados de clasificación y RUL
│   └── plots/                 Figuras para la memoria
├── README.md
└── docs/                      Documentación personal y memoria (no trackeada)
```

Los datasets de PHMD y los artefactos pesados (preprocesados, checkpoints, logs completos) viven en Google Drive (`MyDrive/fm_fl_phmd/`), no en el repositorio. En `results/` solo se versionan los resúmenes ligeros (CSV, `run_info.json`, configuraciones) necesarios para trazar cada resultado.

## Cómo ejecutar

Cada notebook de Colab arranca montando Drive y ejecutando `setup/colab_init.sh` (clona el repo y prepara el entorno). En local:

```bash
pip install -r setup/requirements.txt   # dependencias
pytest tests/                           # batería de tests unitarios
```

El entrenamiento y la evaluación son *config-driven* (YAML en `training/configs/`):

```bash
python training/train_ssl_central.py    --mode train --config training/configs/ssl_central_full.yaml
python training/train_ssl_federated.py  --config training/configs/ssl_federated_fedavg_25x50.yaml
python training/train_downstream_classification.py --config training/configs/downstream_cwru_classification.yaml
python training/train_downstream_rul.py --config training/configs/downstream_cmapss_rul.yaml
```

Cada ejecución registra `config_hash`, hash de commit, semillas e hiperparámetros efectivos para reproducibilidad.

## Dependencias

Listadas en `setup/requirements.txt`. En Colab se instalan automáticamente al ejecutar `colab_init.sh` (ver `setup/README.md`).

## Resultados principales

La evidencia final, sobre datasets no vistos y con tres semillas, es **dependiente de la tarea**:

- **Clasificación de fallos**: el preentrenamiento centralizado transfiere con claridad (macro-F1 en torno a **0,95** en HSG18 y **0,81** en CWRU), muy por encima de entrenar desde cero y del federado a igual presupuesto.
- **RUL (CMAPSS)**: con cabeza congelada la lectura se invierte y los checkpoints federados igualan o superan al centralizado en RMSE relativo (con la cautela de un R² de test negativo).
- Dos hallazgos metodológicos: la pérdida auto-supervisada **no** predice la utilidad downstream, y el learning rate del backbone gobierna el fine-tuning (calibrarlo por validación es decisivo).

## Estado

Fases experimentales cerradas: auditoría del corpus, armonización (47 datasets), preentrenamiento SSL centralizado y federado (FedAvg, FedProx, FedAvgM), y evaluación downstream de clasificación y RUL, con resultados certificados y trazables. La memoria del TFM se mantiene fuera del repositorio (en `docs/`, no trackeada).
