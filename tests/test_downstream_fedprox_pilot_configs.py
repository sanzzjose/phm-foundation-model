"""Tests preflight de los 4 configs downstream FedProx pilot mu=0.01.

Validan que los configs FedProx downstream:

- existen y son YAML validos;
- tienen run_name correctos (`fedprox_pilot_mu0_01` en el nombre);
- usan paths de salida distintos del bloque FedAvg downstream
  (`downstream_federated_pilot_fedprox_mu0_01` en log/checkpoint dirs)
  para no pisar resultados FedAvg previos;
- cubren exactamente los 2 datasets CWRU y HSG18 y los 2 modos
  linear_probing y full_finetuning;
- `lr_backbone == 1e-5` en full_finetuning;
- batch adaptativo conservado (mismos valores que FedAvg homologo);
- son **bit-a-bit equivalentes** al homologo FedAvg salvo run_name,
  paths de salida y comentarios documentales (ckpt referenciado).
- el ckpt FedProx en los comentarios de uso es el correcto y NO
  aparece el path del ckpt FedAvg en los comandos de FedProx.

Independientes de torch (solo yaml + asserts + lectura de fichero).
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[1]
_CFG_DIR = _REPO_ROOT / "training" / "configs"

# Mapeo (dataset_lower, mode_label) -> (cfg_fedprox_path, cfg_fedavg_path)
_PAIRS = {
    ("cwru", "linear_probing"): (
        _CFG_DIR / "downstream_cwru_fedprox_pilot_mu0_01_linear_probing.yaml",
        _CFG_DIR / "downstream_cwru_fedavg_pilot_linear_probing.yaml",
    ),
    ("cwru", "full_finetuning_lr1e-5"): (
        _CFG_DIR / "downstream_cwru_fedprox_pilot_mu0_01_full_finetuning_lr1e-5.yaml",
        _CFG_DIR / "downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5.yaml",
    ),
    ("hsg18", "linear_probing"): (
        _CFG_DIR / "downstream_hsg18_fedprox_pilot_mu0_01_linear_probing.yaml",
        _CFG_DIR / "downstream_hsg18_fedavg_pilot_linear_probing.yaml",
    ),
    ("hsg18", "full_finetuning_lr1e-5"): (
        _CFG_DIR / "downstream_hsg18_fedprox_pilot_mu0_01_full_finetuning_lr1e-5.yaml",
        _CFG_DIR / "downstream_hsg18_fedavg_pilot_full_finetuning_lr1e-5.yaml",
    ),
}

_FEDPROX_CKPT = (
    "/content/drive/MyDrive/fm_fl_phmd/checkpoints/"
    "ssl_federated_pilot_fedprox_mu0_01/"
    "ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/"
    "ckpt_final.pt"
)
_FEDAVG_CKPT = (
    "/content/drive/MyDrive/fm_fl_phmd/checkpoints/"
    "ssl_federated_pilot/ssl_federated_pilot_patchtst_phm/ckpt_final.pt"
)


@pytest.fixture(scope="module")
def cfg_pairs() -> dict:
    """Carga los 4 pares (FedProx, FedAvg) en memoria una sola vez."""
    out = {}
    for key, (fp_path, fa_path) in _PAIRS.items():
        assert fp_path.is_file(), f"Falta config FedProx: {fp_path}"
        assert fa_path.is_file(), f"Falta config FedAvg homologo: {fa_path}"
        out[key] = {
            "fedprox": yaml.safe_load(fp_path.read_text(encoding="utf-8")),
            "fedavg": yaml.safe_load(fa_path.read_text(encoding="utf-8")),
            "fp_path": fp_path,
            "fa_path": fa_path,
            "fp_text": fp_path.read_text(encoding="utf-8"),
        }
    return out


# ----------------------------------------------------------------------
# 1. Existencia y carga
# ----------------------------------------------------------------------


def test_los_4_configs_fedprox_existen_y_son_yaml(cfg_pairs):
    assert len(cfg_pairs) == 4
    for key, p in cfg_pairs.items():
        assert isinstance(p["fedprox"], dict), key
        assert isinstance(p["fedavg"], dict), key


# ----------------------------------------------------------------------
# 2. run_name contiene fedprox_pilot_mu0_01
# ----------------------------------------------------------------------


def test_run_name_contiene_fedprox_pilot_mu0_01(cfg_pairs):
    for key, p in cfg_pairs.items():
        rn = p["fedprox"]["run_name"]
        assert "fedprox_pilot_mu0_01" in rn, (key, rn)
        # Y NO contiene fedavg, para evitar copias incompletas.
        assert "fedavg" not in rn, (key, rn)


# ----------------------------------------------------------------------
# 3. Paths NO pisan el bloque FedAvg
# 4. Paths nuevos contienen downstream_federated_pilot_fedprox_mu0_01
# ----------------------------------------------------------------------


def test_paths_no_pisan_fedavg_y_son_fedprox(cfg_pairs):
    for key, p in cfg_pairs.items():
        log_dir = str(p["fedprox"]["paths"]["log_dir"])
        ckpt_dir = str(p["fedprox"]["paths"]["checkpoint_dir"])

        # 3) NO debe contener el path FedAvg downstream
        # (downstream_federated_pilot/ con slash inmediato despues, sin
        # _fedprox_).
        for path in (log_dir, ckpt_dir):
            assert "downstream_federated_pilot_fedprox_mu0_01" in path, (key, path)
            # El path FedAvg downstream es exactamente
            # `.../downstream_federated_pilot/{cwru,hsg18}` (sin sufijo
            # _fedprox_mu0_01). Verificamos que NO esta presente como
            # segmento aislado.
            assert "/downstream_federated_pilot/" not in path, (
                f"{key} path '{path}' pisa el bloque FedAvg downstream"
            )

        # Tambien comprobamos que difieren del FedAvg homologo.
        assert log_dir != p["fedavg"]["paths"]["log_dir"], key
        assert ckpt_dir != p["fedavg"]["paths"]["checkpoint_dir"], key


# ----------------------------------------------------------------------
# 5. Datasets cubiertos: solo CWRU y HSG18
# ----------------------------------------------------------------------


def test_datasets_cubiertos_son_solo_cwru_y_hsg18(cfg_pairs):
    datasets = sorted({p["fedprox"]["dataset"] for p in cfg_pairs.values()})
    assert datasets == ["CWRU", "HSG18"], datasets
    # Y data.dataset coincide con el top-level.
    for key, p in cfg_pairs.items():
        assert p["fedprox"]["dataset"] == p["fedprox"]["data"]["dataset"]


# ----------------------------------------------------------------------
# 6. Modos cubiertos: linear_probing y full_finetuning
# ----------------------------------------------------------------------


def test_modos_cubiertos_segun_run_name(cfg_pairs):
    """Cada config debe ir destinado a linear_probing o full_finetuning_lr1e-5.

    Lo deducimos del run_name (no hay campo `mode` en los YAML; el modo
    se pasa como flag CLI).
    """
    modes_observed = set()
    for (_, mode_label), p in cfg_pairs.items():
        rn = p["fedprox"]["run_name"]
        if mode_label == "linear_probing":
            assert rn.endswith("_linear_probing"), rn
        elif mode_label == "full_finetuning_lr1e-5":
            assert rn.endswith("_full_finetuning_lr1e-5"), rn
        modes_observed.add(mode_label)
    assert modes_observed == {"linear_probing", "full_finetuning_lr1e-5"}


# ----------------------------------------------------------------------
# 7. full_finetuning con lr_backbone == 1e-5
# ----------------------------------------------------------------------


def test_full_finetuning_lr_backbone_1e_5(cfg_pairs):
    for (_, mode_label), p in cfg_pairs.items():
        if mode_label != "full_finetuning_lr1e-5":
            continue
        lr_bb = float(p["fedprox"]["training"]["lr_backbone"])
        assert lr_bb == 1e-5, (mode_label, lr_bb)


# ----------------------------------------------------------------------
# 8. Batch adaptativo identico al FedAvg homologo
# ----------------------------------------------------------------------


def test_batch_adaptativo_conservado(cfg_pairs):
    for key, p in cfg_pairs.items():
        d = p["fedprox"]["data"]
        assert d["batch_size_policy"] == "adaptive_by_channels", key
        assert int(d["max_channel_batch"]) == 512, key
        assert int(d["min_batch_size"]) == 1, key
        assert int(d["batch_size"]) == 64, key


# ----------------------------------------------------------------------
# 9. Equivalencia bit-a-bit con el FedAvg homologo salvo run_name,
#    paths de salida y comentarios documentales (que viven fuera del YAML).
# ----------------------------------------------------------------------


def _strip_run_name_and_output_paths(cfg: dict) -> dict:
    """Devuelve una copia del cfg sin run_name ni paths.{log_dir,checkpoint_dir}.

    Estos son los unicos campos donde se permite divergencia FedProx
    vs FedAvg. Los comentarios (incluyendo el ckpt referenciado en
    `Uso:`) viven fuera del YAML parseado.
    """
    c = {k: v for k, v in cfg.items() if k != "run_name"}
    paths = dict(c.get("paths", {}))
    paths.pop("log_dir", None)
    paths.pop("checkpoint_dir", None)
    c["paths"] = paths
    return c


def test_fedprox_idem_fedavg_salvo_run_name_y_paths(cfg_pairs):
    for key, p in cfg_pairs.items():
        sa = _strip_run_name_and_output_paths(p["fedprox"])
        sb = _strip_run_name_and_output_paths(p["fedavg"])
        assert sa == sb, (
            f"FedProx config {key} difiere del FedAvg homologo fuera de "
            f"los campos permitidos (run_name + paths.{{log_dir,"
            f"checkpoint_dir}}). diff keys: "
            f"{sorted(set(sa.keys()) ^ set(sb.keys()))}"
        )


# ----------------------------------------------------------------------
# 10. El ckpt FedProx aparece en el config y el ckpt FedAvg NO aparece
# ----------------------------------------------------------------------


def test_ckpt_fedprox_referenciado_y_fedavg_ausente(cfg_pairs):
    """Los comentarios del YAML (`Uso:`) deben referenciar el ckpt
    FedProx exacto. Por seguridad, **el ckpt FedAvg pilot NO debe
    aparecer** en ningun comentario de los configs FedProx (para
    evitar copy-paste accidental al lanzar las corridas).
    """
    for key, p in cfg_pairs.items():
        text = p["fp_text"]
        assert _FEDPROX_CKPT in text, (
            f"{key}: el ckpt FedProx exacto NO aparece en el YAML.\n"
            f"esperado: {_FEDPROX_CKPT}"
        )
        # El path FedAvg debe estar AUSENTE del config FedProx.
        # Verificamos contra el path canonico documentado en la sec 17
        # CLAUDE.md / commit `9b6c9fb`.
        assert _FEDAVG_CKPT not in text, (
            f"{key}: el path FedAvg pilot aparece en el config FedProx.\n"
            f"path FedAvg que NO debe estar: {_FEDAVG_CKPT}"
        )
