# Setup del proyecto en Google Colab + Drive

Esta carpeta contiene lo necesario para trabajar el TFM desde Google Colab Pro+ usando Google Drive como único almacén persistente. El editor local sigue siendo VSCode contra el repo de GitHub; los notebooks corren en Colab.

## Cómo se relacionan los tres entornos

```
LOCAL (VSCode + Claude Code)
        │ git push / pull
        ▼
GITHUB  (origen único del código)
        │ git pull (lo hace colab_init.sh)
        ▼
COLAB PRO+ (VM con GPU, efímera)
        │ Drive montado en /content/drive
        ▼
GOOGLE DRIVE  (almacén persistente)
  MyDrive/fm_fl_phmd/
    raw/          datasets de phmd en .zip
    processed/    ventanas y patches generados
    checkpoints/  modelos guardados
    logs/         logs de entrenamiento y descarga
    .ssh/         clave SSH del proyecto para Colab → GitHub
    colab_init.sh script que arranca cada sesión de Colab
```

## Ficheros de esta carpeta

- `colab_bootstrap.ipynb` — notebook que se ejecuta **una sola vez por cuenta o máquina nueva**. Crea la estructura de carpetas en Drive, genera la clave SSH, pide al usuario que la registre como Deploy Key en GitHub, clona el repo y deja `colab_init.sh` en Drive para uso futuro.
- `colab_init.sh` — script de arranque rápido. Lo invoca cada notebook al principio de cada sesión. Restaura la clave SSH en la VM, hace `git pull` del repo e instala las dependencias.
- `requirements.txt` — lista canónica de dependencias Python del proyecto.

## Flujo de trabajo

### La primera vez (por usuario o máquina nueva)

1. Abrir `setup/colab_bootstrap.ipynb` en Colab (File → Open notebook → pestaña GitHub).
2. Runtime → Change runtime type → GPU.
3. Ejecutar todas las celdas. En la celda que imprime la clave pública, copiarla y añadirla como Deploy Key con permiso de escritura en:
   https://github.com/sanzzjose/phm-foundation-model/settings/keys/new

### En cualquier sesión posterior

Cada notebook (`00_download_datasets.ipynb`, `02_dataset_audit.ipynb`, etc.) empieza con solo dos líneas que se encargan del setup:

```python
from google.colab import drive; drive.mount('/content/drive')
!bash /content/drive/MyDrive/fm_fl_phmd/colab_init.sh
```

Tras ejecutarlas: Drive montado, repo en `/content/fm_fl_phmd`, deps instaladas y un alias `/content/work` apuntando a `MyDrive/fm_fl_phmd/`.

### Para entrenamientos largos

Activar **Runtime → Manage sessions → Background execution**. Es una funcionalidad de Colab Pro+ que permite cerrar el navegador sin matar la sesión.

## Por qué no usamos un túnel SSH hacia la VM

Aunque técnicamente podríamos exponer la VM como un servidor SSH (mediante cloudflared u otros) y conectarnos desde VSCode local, los términos de servicio de Colab prohíben este tipo de uso. Por eso editamos código en VSCode local y lo sincronizamos vía git con la VM, en vez de tener un cursor directo sobre la VM.

## Notas operativas

- La VM de Colab tiene un disco local rápido (`/content`, unos 200 GB). Drive es lento por API. Para bucles de entrenamiento conviene copiar el conjunto activo a `/content/` y leer desde ahí.
- La carpeta `.ssh/` dentro de Drive contiene una clave usada exclusivamente para Colab → GitHub. Es independiente de la clave SSH del portátil del usuario.
