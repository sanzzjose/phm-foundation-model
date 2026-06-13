"""Utilidades de hashing estable de configuraciones y metadatos git.

El objetivo es que dos corridas con la misma config efectiva produzcan el
mismo ``config_hash`` y, por tanto, sean fácilmente agrupables en la
tabla maestra. El hash es un SHA-256 de 16 dígitos hex calculado sobre
la serialización YAML estable de la config (claves ordenadas, Unicode).

Las funciones de este módulo se diseñan para ser idempotentes y no tener
efectos secundarios. No abren ficheros para escritura ni modifican el
estado global del proceso.
"""
from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def now_ts() -> str:
    """Devuelve un timestamp ISO 8601 local con resolución de segundo.

    Se usa para campos ``ts`` y ``created_at`` de los manifests. Si en el
    futuro se requiere UTC, basta con cambiar la implementación aquí; las
    llamadas no deben asumir zona horaria.
    """
    return time.strftime("%Y-%m-%dT%H:%M:%S")


def load_config(path: Path) -> Dict[str, Any]:
    """Carga una config YAML como ``dict``.

    Lanza ``FileNotFoundError`` si la ruta no existe y propaga errores de
    sintaxis YAML. No realiza ninguna validación semántica.
    """
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def config_hash(cfg: Dict[str, Any]) -> str:
    """Hash SHA-256 (16 caracteres hex) de una config serializada.

    La serialización se hace con ``yaml.safe_dump(..., sort_keys=True,
    allow_unicode=True)``. Esto garantiza que el hash es determinista
    respecto a:

    * el orden de las claves (se ordenan alfabéticamente);
    * la presencia o ausencia de espacios accidentales (PyYAML normaliza);
    * los caracteres no ASCII (se preservan tal cual).

    El hash NO captura la versión de PyYAML; en la práctica los releases
    suceden con cambios menores y este riesgo se acepta a cambio de la
    simplicidad de la implementación.
    """
    blob = yaml.safe_dump(cfg, sort_keys=True, allow_unicode=True).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def get_git_info(repo_root: Optional[Path] = None) -> Dict[str, Any]:
    """Devuelve un dict ``{git_hash, git_dirty}`` para el repo indicado.

    Si ``repo_root`` no es un repo git, si el binario ``git`` no está
    disponible o si las llamadas fallan por cualquier motivo, devuelve un
    dict con ``git_hash="unknown"`` y ``git_dirty=False`` y NO propaga la
    excepción. Esta tolerancia es deliberada: el objetivo es enriquecer
    los manifests, no abortar el run si git falla.

    ``repo_root=None`` usa la cwd como raíz.
    """
    root = Path(repo_root) if repo_root is not None else Path.cwd()
    try:
        h = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        h = "unknown"
    try:
        status = subprocess.check_output(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        dirty = bool(status)
    except Exception:
        dirty = False
    return {"git_hash": h, "git_dirty": dirty}
