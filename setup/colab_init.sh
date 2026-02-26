#!/bin/bash
# colab_init.sh
#
# Script que prepara la máquina virtual de Google Colab para trabajar
# con este proyecto. Lo invoca cada notebook al principio de cada sesión.
#
# Lo que hace, en orden:
#   1. Recupera la clave SSH del proyecto desde Google Drive y la deja
#      en ~/.ssh con los permisos correctos.
#   2. Confía en github.com (añade su huella a known_hosts).
#   3. Clona el repositorio (o hace pull si ya existía) en /content/fm_fl_phmd.
#   4. Instala las dependencias de Python listadas en setup/requirements.txt.
#   5. Crea un enlace simbólico /content/work apuntando a la carpeta del
#      proyecto en Drive, para escribir rutas más cortas en los notebooks.
#
# Requisitos previos (los hace setup/colab_bootstrap.ipynb la primera vez):
#   - Google Drive montado en /content/drive
#   - Clave SSH del proyecto en /content/drive/MyDrive/fm_fl_phmd/.ssh/
#   - Clave pública añadida como Deploy Key en GitHub (con write access)

set -e  # si cualquier comando falla, abortamos para no avanzar con errores ocultos

# Rutas que se usan más abajo. Si en algún momento cambia el nombre del repo,
# de la rama, o de la carpeta en Drive, basta con tocarlo aquí.
DRIVE_ROOT=/content/drive/MyDrive/fm_fl_phmd
REPO=/content/fm_fl_phmd
BRANCH=feature/exploration_phmd
GIT_URL=git@github.com:sanzzjose/phm-foundation-model.git


# ---------------------------------------------------------------------------
# 1) Clave SSH: la traemos de Drive a ~/.ssh con permisos restrictivos
#    (si los permisos no son 600, ssh la rechaza por seguridad)
# ---------------------------------------------------------------------------
mkdir -p ~/.ssh
chmod 700 ~/.ssh
cp "$DRIVE_ROOT/.ssh/id_ed25519_colab"     ~/.ssh/id_ed25519
cp "$DRIVE_ROOT/.ssh/id_ed25519_colab.pub" ~/.ssh/id_ed25519.pub
chmod 600 ~/.ssh/id_ed25519
chmod 644 ~/.ssh/id_ed25519.pub


# ---------------------------------------------------------------------------
# 2) Confiar en github.com: evitamos el prompt interactivo "Are you sure
#    you want to continue connecting?" añadiendo su huella a known_hosts
# ---------------------------------------------------------------------------
ssh-keyscan -t ed25519,rsa github.com >> ~/.ssh/known_hosts 2>/dev/null


# ---------------------------------------------------------------------------
# 3) Repositorio:
#    - Si ya existe en /content/fm_fl_phmd: hacemos pull para actualizarlo.
#    - Si no existe: clonamos desde GitHub.
# ---------------------------------------------------------------------------
if [ -d "$REPO/.git" ]; then
    git -C "$REPO" fetch origin
    git -C "$REPO" checkout "$BRANCH"
    git -C "$REPO" pull --ff-only
else
    git clone -b "$BRANCH" "$GIT_URL" "$REPO"
fi


# ---------------------------------------------------------------------------
# 4) Dependencias de Python (instalación silenciosa con -q)
# ---------------------------------------------------------------------------
pip install -q -r "$REPO/setup/requirements.txt"


# ---------------------------------------------------------------------------
# 5) Atajo: /content/work apunta a la carpeta del proyecto en Drive
#    Permite escribir rutas como /content/work/raw en lugar de
#    /content/drive/MyDrive/fm_fl_phmd/raw. Solo cosmético.
# ---------------------------------------------------------------------------
ln -sfn "$DRIVE_ROOT" /content/work

echo "Setup OK"
echo "  Repo:  $REPO"
echo "  Datos: $DRIVE_ROOT (alias /content/work)"
