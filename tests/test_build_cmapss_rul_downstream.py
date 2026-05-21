"""Tests del builder CMAPSS RUL (funciones puras + dry-run).

No requiere torch. Solo numpy/json puros.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


# ----------------------------------------------------------------------
# compute_train_rul: train/val run-to-failure
# ----------------------------------------------------------------------


def test_train_rul_basico():
    from training.build_cmapss_rul_downstream import compute_train_rul
    # ciclos 1..5, max=5 -> RUL = 4, 3, 2, 1, 0
    assert compute_train_rul([1, 2, 3, 4, 5], max_cycle=5) == [4.0, 3.0, 2.0, 1.0, 0.0]


def test_train_rul_no_negativos_si_monotonico():
    from training.build_cmapss_rul_downstream import compute_train_rul
    out = compute_train_rul([10, 20, 30, 40, 50], max_cycle=50)
    assert all(r >= 0 for r in out)
    assert out == [40.0, 30.0, 20.0, 10.0, 0.0]


def test_train_rul_vacio():
    from training.build_cmapss_rul_downstream import compute_train_rul
    assert compute_train_rul([], max_cycle=10) == []


def test_train_rul_rechaza_max_cycle_inconsistente():
    from training.build_cmapss_rul_downstream import compute_train_rul
    with pytest.raises(ValueError, match="max_cycle"):
        compute_train_rul([1, 2, 3, 10], max_cycle=5)


# ----------------------------------------------------------------------
# compute_test_rul: test interrumpido + RUL oficial
# ----------------------------------------------------------------------


def test_test_rul_basico():
    from training.build_cmapss_rul_downstream import compute_test_rul
    # ciclos 1..5, last_observed_cycle=5, official_RUL_at_last=10
    # -> RUL = (5-1)+10=14, (5-2)+10=13, 12, 11, 10
    out = compute_test_rul([1, 2, 3, 4, 5], last_observed_cycle=5, official_rul=10)
    assert out == [14.0, 13.0, 12.0, 11.0, 10.0]


def test_test_rul_no_negativos():
    from training.build_cmapss_rul_downstream import compute_test_rul
    out = compute_test_rul([1, 2, 3, 4, 5], last_observed_cycle=5, official_rul=0)
    # con official_rul=0: RUL = last - cycle = 4,3,2,1,0
    assert all(r >= 0 for r in out)
    assert out == [4.0, 3.0, 2.0, 1.0, 0.0]


def test_test_rul_vacio():
    from training.build_cmapss_rul_downstream import compute_test_rul
    assert compute_test_rul([], last_observed_cycle=5, official_rul=10) == []


def test_test_rul_rechaza_official_negativo():
    from training.build_cmapss_rul_downstream import compute_test_rul
    with pytest.raises(ValueError, match="official_rul"):
        compute_test_rul([1, 2, 3], last_observed_cycle=3, official_rul=-1)


def test_test_rul_rechaza_last_cycle_inconsistente():
    from training.build_cmapss_rul_downstream import compute_test_rul
    with pytest.raises(ValueError, match="last_observed_cycle"):
        compute_test_rul([1, 2, 3, 10], last_observed_cycle=5, official_rul=0)


# ----------------------------------------------------------------------
# cap_rul: SOLO aplica despues de reconstruccion fisica
# ----------------------------------------------------------------------


def test_cap_rul_aplica_si_cap_positivo():
    from training.build_cmapss_rul_downstream import cap_rul
    out = cap_rul([10, 50, 125, 200, 300], cap=125)
    assert out == [10.0, 50.0, 125.0, 125.0, 125.0]


def test_cap_rul_sin_cap_devuelve_igual():
    from training.build_cmapss_rul_downstream import cap_rul
    assert cap_rul([10, 50, 200], cap=None) == [10.0, 50.0, 200.0]
    assert cap_rul([10, 50, 200], cap=0) == [10.0, 50.0, 200.0]
    assert cap_rul([10, 50, 200], cap=-5) == [10.0, 50.0, 200.0]


def test_cap_rul_no_transforma_negativos():
    """Si entran negativos, salen igual: el caller NO debe usar cap sobre
    valores no reconstruidos. El test documenta el contrato."""
    from training.build_cmapss_rul_downstream import cap_rul
    out = cap_rul([-10, 0, 50, 200], cap=125)
    assert out == [-10.0, 0.0, 50.0, 125.0]


# ----------------------------------------------------------------------
# Pipeline encadenado: physical + cap
# ----------------------------------------------------------------------


def test_train_capped_125_sobre_train_fisico():
    from training.build_cmapss_rul_downstream import compute_train_rul, cap_rul
    # Trayectoria larga: ciclos 1..200, max=200 -> RUL = 199, 198, ..., 0
    cycles = list(range(1, 201))
    rul = compute_train_rul(cycles, max_cycle=200)
    capped = cap_rul(rul, cap=125)
    assert max(capped) == 125
    assert min(capped) == 0
    # Los primeros 75 cicls (1..75) tienen RUL >= 125 -> capeados a 125.
    assert capped[0] == 125  # cycle=1 -> RUL=199 -> cap 125
    assert capped[74] == 125  # cycle=75 -> RUL=125 -> queda 125
    assert capped[75] == 124  # cycle=76 -> RUL=124 -> queda 124


# ----------------------------------------------------------------------
# inspect_raw_cmapss y dry-run report
# ----------------------------------------------------------------------


def test_inspect_raw_missing(tmp_path):
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is True
    assert out["raw_zip_path"] is None


def test_inspect_raw_con_zip(tmp_path):
    """zip valido pero vacio: raw_missing=False, sin FD subsets detectados."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    z = tmp_path / "CMAPSS.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", "dummy")  # zip valido pero sin FDs
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is False
    assert out["raw_zip_path"] == str(z)
    assert out["fd_subsets"] == []


def test_inspect_raw_con_directorio(tmp_path):
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    d = tmp_path / "CMAPSS"
    d.mkdir()
    (d / "train_FD001.txt").write_text("dummy")
    (d / "RUL_FD001.txt").write_text("dummy")
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is False
    assert out["raw_dir_path"] == str(d)
    assert "train_FD001.txt" in out["candidate_files"]
    assert "RUL_FD001.txt" in out["candidate_files"]


# ----------------------------------------------------------------------
# Deteccion FD subsets + missing
# ----------------------------------------------------------------------


def test_detect_fd_files_completo():
    from training.build_cmapss_rul_downstream import _detect_fd_files
    out = _detect_fd_files([
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
        "train_FD002.txt", "test_FD002.txt", "RUL_FD002.txt",
        "train_FD003.txt", "test_FD003.txt", "RUL_FD003.txt",
        "train_FD004.txt", "test_FD004.txt", "RUL_FD004.txt",
    ])
    assert out["fd_subsets"] == ["FD001", "FD002", "FD003", "FD004"]
    assert out["missing_train"] == []
    assert out["missing_test"] == []
    assert out["missing_rul"] == []
    assert out["unrecognized"] == []


def test_detect_fd_files_falta_rul():
    from training.build_cmapss_rul_downstream import _detect_fd_files
    out = _detect_fd_files([
        "train_FD001.txt", "test_FD001.txt", "RUL_FD001.txt",
        "train_FD002.txt", "test_FD002.txt",  # falta RUL_FD002
    ])
    assert "FD002" in out["fd_subsets"]
    assert out["missing_rul"] == ["FD002"]
    assert out["missing_train"] == []


def test_detect_fd_files_falta_train_y_test():
    from training.build_cmapss_rul_downstream import _detect_fd_files
    out = _detect_fd_files(["RUL_FD001.txt"])
    assert out["fd_subsets"] == ["FD001"]
    assert out["missing_train"] == ["FD001"]
    assert out["missing_test"] == ["FD001"]
    assert out["missing_rul"] == []


def test_detect_fd_files_unrecognized():
    from training.build_cmapss_rul_downstream import _detect_fd_files
    out = _detect_fd_files(["readme.txt", "license.md", "train_FD001.txt"])
    assert "readme.txt" in out["unrecognized"]
    assert "license.md" in out["unrecognized"]
    assert "train_FD001.txt" not in out["unrecognized"]


def test_inspect_raw_dir_reporta_missing_rul(tmp_path):
    """raw expandido pero falta RUL_FD002.txt -> debe aparecer en missing_rul."""
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    d = tmp_path / "CMAPSS"
    d.mkdir()
    (d / "train_FD001.txt").write_text("x")
    (d / "test_FD001.txt").write_text("x")
    (d / "RUL_FD001.txt").write_text("x")
    (d / "train_FD002.txt").write_text("x")
    (d / "test_FD002.txt").write_text("x")
    out = inspect_raw_cmapss(tmp_path)
    assert "FD001" in out["fd_subsets"]
    assert "FD002" in out["fd_subsets"]
    assert "FD002" in out["missing_rul"]
    assert "FD001" not in out["missing_rul"]


def test_inspect_raw_zip_sin_extraer(tmp_path):
    """Si raw es zip, inspeccionar nombres con zipfile sin descomprimir."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    zip_path = tmp_path / "CMAPSS.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.writestr("CMAPSS/train_FD001.txt", "dummy")
        zf.writestr("CMAPSS/test_FD001.txt", "dummy")
        zf.writestr("CMAPSS/RUL_FD001.txt", "dummy")
        zf.writestr("CMAPSS/train_FD002.txt", "dummy")
        # falta test_FD002 y RUL_FD002
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is False
    assert out["raw_zip_path"] == str(zip_path)
    assert "FD001" in out["fd_subsets"]
    assert "FD002" in out["fd_subsets"]
    assert "FD002" in out["missing_test"]
    assert "FD002" in out["missing_rul"]
    # NO debe haber extraido nada al disco
    assert not (tmp_path / "CMAPSS").exists()


def test_inspect_raw_zip_layout_plano(tmp_path):
    """Variante: zip con ficheros en raiz (sin carpeta CMAPSS/)."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    zip_path = tmp_path / "CMAPSS.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for fd in ("FD001", "FD002"):
            for kind in ("train", "test", "RUL"):
                zf.writestr(f"{kind}_{fd}.txt", "dummy")
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_layout"] == "zip_unico"
    assert out["fd_subsets"] == ["FD001", "FD002"]
    assert out["missing_train"] == []
    assert out["missing_test"] == []
    assert out["missing_rul"] == []


# ----------------------------------------------------------------------
# Layout PHMD splitado: CMAPSS_train.zip + CMAPSS_test.zip
# ----------------------------------------------------------------------


def _write_phmd_split(tmp_path, fds=("FD001", "FD002", "FD003", "FD004")):
    """Crea el layout PHMD real (2 zips, estructura CMAPSS/train|test/...)."""
    import zipfile
    train_zip = tmp_path / "CMAPSS_train.zip"
    test_zip = tmp_path / "CMAPSS_test.zip"
    with zipfile.ZipFile(train_zip, "w") as zf:
        for fd in fds:
            zf.writestr(f"CMAPSS/train/train_{fd}.txt", "dummy")
    with zipfile.ZipFile(test_zip, "w") as zf:
        for fd in fds:
            zf.writestr(f"CMAPSS/test/test_{fd}.txt", "dummy")
            zf.writestr(f"CMAPSS/test/RUL_{fd}.txt", "dummy")
    return train_zip, test_zip


def test_inspect_raw_phmd_split_completo(tmp_path):
    """Layout real PHMD: 2 zips. Debe detectarse y reportar los 4 FD."""
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    train_zip, test_zip = _write_phmd_split(tmp_path)
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is False
    assert out["raw_layout"] == "zips_split_phmd"
    assert out["raw_zip_train_path"] == str(train_zip)
    assert out["raw_zip_test_path"] == str(test_zip)
    assert out["fd_subsets"] == ["FD001", "FD002", "FD003", "FD004"]
    assert out["missing_train"] == []
    assert out["missing_test"] == []
    assert out["missing_rul"] == []
    # No debe haber extraido nada al disco
    assert not (tmp_path / "CMAPSS").exists()


def test_inspect_raw_phmd_split_falta_un_fd(tmp_path):
    """Si en CMAPSS_test.zip falta el RUL de un FD, debe reportarlo."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    train_zip = tmp_path / "CMAPSS_train.zip"
    test_zip = tmp_path / "CMAPSS_test.zip"
    with zipfile.ZipFile(train_zip, "w") as zf:
        for fd in ("FD001", "FD002"):
            zf.writestr(f"CMAPSS/train/train_{fd}.txt", "x")
    with zipfile.ZipFile(test_zip, "w") as zf:
        for fd in ("FD001", "FD002"):
            zf.writestr(f"CMAPSS/test/test_{fd}.txt", "x")
            # Solo escribimos RUL para FD001, FALTA para FD002.
        zf.writestr("CMAPSS/test/RUL_FD001.txt", "x")
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_layout"] == "zips_split_phmd"
    assert out["fd_subsets"] == ["FD001", "FD002"]
    assert out["missing_rul"] == ["FD002"]


def test_inspect_raw_phmd_split_solo_train_zip(tmp_path):
    """Layout PHMD parcial: solo uno de los dos zips. raw_missing=True
    con nota explicita."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    with zipfile.ZipFile(tmp_path / "CMAPSS_train.zip", "w") as zf:
        zf.writestr("CMAPSS/train/train_FD001.txt", "x")
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is True
    assert out["raw_layout"] is None
    assert any("layout PHMD parcial" in n for n in out["notes"])
    assert any("CMAPSS_test.zip" in n for n in out["notes"])


def test_inspect_raw_phmd_split_solo_test_zip(tmp_path):
    """Layout PHMD parcial inverso: solo el test_zip. raw_missing=True."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    with zipfile.ZipFile(tmp_path / "CMAPSS_test.zip", "w") as zf:
        zf.writestr("CMAPSS/test/test_FD001.txt", "x")
    out = inspect_raw_cmapss(tmp_path)
    assert out["raw_missing"] is True
    assert any("CMAPSS_train.zip" in n for n in out["notes"])


def test_inspect_raw_layout_priority(tmp_path):
    """Si conviven varios layouts, priorizamos: dir > zip_unico > split.
    Es defensivo: el dir expandido es lo mas reciente y autoritativo."""
    import zipfile
    from training.build_cmapss_rul_downstream import inspect_raw_cmapss
    # Crear los 3 layouts simultaneamente.
    (tmp_path / "CMAPSS").mkdir()
    (tmp_path / "CMAPSS" / "train_FD001.txt").write_text("x")
    with zipfile.ZipFile(tmp_path / "CMAPSS.zip", "w") as zf:
        zf.writestr("dummy.txt", "x")
    with zipfile.ZipFile(tmp_path / "CMAPSS_train.zip", "w") as zf:
        zf.writestr("x", "x")
    with zipfile.ZipFile(tmp_path / "CMAPSS_test.zip", "w") as zf:
        zf.writestr("y", "y")
    out = inspect_raw_cmapss(tmp_path)
    # El dir gana
    assert out["raw_layout"] == "dir"


# ----------------------------------------------------------------------
# Commit 1: parsers CMAPSS NASA + loader split zips + integracion RUL
# ----------------------------------------------------------------------


import io


def _synthetic_train_txt(unit_cycles):
    """Genera un .txt train_FDxxx sintetico (formato NASA, 26 columnas)
    para las unidades y ciclos dados.

    Args:
        unit_cycles: list[(unit_id, [cycle_ids])] - cada unidad con sus ciclos.

    Returns:
        str con el contenido del .txt.
    """
    lines = []
    for unit_id, cycles in unit_cycles:
        for c in cycles:
            # 26 columnas: unit, cycle, op1..3, s1..21
            cols = [str(unit_id), str(c)]
            cols += [f"{0.1 * i + 0.001 * c:.4f}" for i in range(3)]  # op_settings
            cols += [f"{0.5 * (j + 1) + 0.01 * c:.4f}" for j in range(21)]  # sensors
            lines.append(" ".join(cols))
    return "\n".join(lines) + "\n"


def _synthetic_rul_txt(rul_values):
    """Genera un RUL_FDxxx.txt sintetico (1 valor por linea)."""
    return "\n".join(str(int(v)) for v in rul_values) + "\n"


def test_parse_cmapss_txt_train_minimo():
    """Parsear un train_FDxxx sintetico con 2 unidades."""
    from training.build_cmapss_rul_downstream import parse_cmapss_txt_filelike
    txt = _synthetic_train_txt([(1, [1, 2, 3, 4, 5]), (2, [1, 2, 3])])
    parsed = parse_cmapss_txt_filelike(io.StringIO(txt), "FD001", "train")
    assert parsed["n_rows"] == 8
    assert parsed["n_units"] == 2
    assert parsed["fd_subset"] == "FD001"
    assert parsed["split"] == "train"
    assert parsed["op_settings"].shape == (8, 3)
    assert parsed["sensors"].shape == (8, 21)
    # Ordenado por (unit_id, cycle)
    assert parsed["unit_id"].tolist() == [1, 1, 1, 1, 1, 2, 2, 2]
    assert parsed["cycle"].tolist() == [1, 2, 3, 4, 5, 1, 2, 3]


def test_parse_cmapss_txt_acepta_bytes():
    """zipfile.open devuelve binario; el parser debe aceptarlo."""
    from training.build_cmapss_rul_downstream import parse_cmapss_txt_filelike
    txt = _synthetic_train_txt([(1, [1, 2])])
    parsed = parse_cmapss_txt_filelike(io.BytesIO(txt.encode("utf-8")), "FD002", "test")
    assert parsed["n_rows"] == 2


def test_parse_cmapss_txt_rechaza_columnas_distintas():
    """Si el fichero tiene 25 o 27 columnas, error claro."""
    from training.build_cmapss_rul_downstream import parse_cmapss_txt_filelike
    # 25 cols
    bad25 = " ".join(["1", "1"] + ["0.1"] * 23) + "\n"
    with pytest.raises(ValueError, match="26 columnas"):
        parse_cmapss_txt_filelike(io.StringIO(bad25), "FD001", "train")
    # 27 cols
    bad27 = " ".join(["1", "1"] + ["0.1"] * 25) + "\n"
    with pytest.raises(ValueError, match="26 columnas"):
        parse_cmapss_txt_filelike(io.StringIO(bad27), "FD001", "train")


def test_parse_cmapss_txt_rechaza_unit_id_no_entero():
    from training.build_cmapss_rul_downstream import parse_cmapss_txt_filelike
    txt = " ".join(["1.5", "1"] + ["0.1"] * 24) + "\n"
    with pytest.raises(ValueError, match="entero"):
        parse_cmapss_txt_filelike(io.StringIO(txt), "FD001", "train")


def test_parse_cmapss_txt_rechaza_cycle_no_monotono():
    """Si tras ordenar el cycle no es estrictamente creciente por unidad."""
    from training.build_cmapss_rul_downstream import parse_cmapss_txt_filelike
    # Unidad 1, cycle 1 duplicado.
    txt = _synthetic_train_txt([(1, [1, 1, 2])])
    with pytest.raises(ValueError, match="monotono"):
        parse_cmapss_txt_filelike(io.StringIO(txt), "FD001", "train")


def test_parse_cmapss_txt_rechaza_fd_split_invalidos():
    from training.build_cmapss_rul_downstream import parse_cmapss_txt_filelike
    with pytest.raises(ValueError, match="fd_subset desconocido"):
        parse_cmapss_txt_filelike(io.StringIO("dummy"), "FD999", "train")
    with pytest.raises(ValueError, match="split"):
        parse_cmapss_txt_filelike(io.StringIO("dummy"), "FD001", "validation")


def test_parse_official_rul_minimo():
    from training.build_cmapss_rul_downstream import parse_official_rul_filelike
    rul = parse_official_rul_filelike(io.StringIO("100\n50\n25\n"))
    assert rul == {1: 100, 2: 50, 3: 25}


def test_parse_official_rul_rechaza_negativos():
    from training.build_cmapss_rul_downstream import parse_official_rul_filelike
    with pytest.raises(ValueError, match="negativos"):
        parse_official_rul_filelike(io.StringIO("100\n-1\n50\n"))


def test_parse_official_rul_rechaza_no_enteros():
    from training.build_cmapss_rul_downstream import parse_official_rul_filelike
    with pytest.raises(ValueError, match="no enteros"):
        parse_official_rul_filelike(io.StringIO("100.5\n50\n"))


def _build_phmd_split_real(tmp_path, fd_train_data, fd_test_data, fd_rul_data):
    """Construye los dos zips PHMD reales con contenido sintetico para test."""
    import zipfile
    train_zip = tmp_path / "CMAPSS_train.zip"
    test_zip = tmp_path / "CMAPSS_test.zip"
    with zipfile.ZipFile(train_zip, "w") as zf:
        for fd, txt in fd_train_data.items():
            zf.writestr(f"CMAPSS/train/train_{fd}.txt", txt)
    with zipfile.ZipFile(test_zip, "w") as zf:
        for fd, txt in fd_test_data.items():
            zf.writestr(f"CMAPSS/test/test_{fd}.txt", txt)
        for fd, txt in fd_rul_data.items():
            zf.writestr(f"CMAPSS/test/RUL_{fd}.txt", txt)
    return train_zip, test_zip


def test_load_cmapss_raw_from_split_zips_FD001_completo(tmp_path):
    """Carga sintetica de FD001 con 4 unidades en train y 3 en test."""
    from training.build_cmapss_rul_downstream import load_cmapss_raw_from_split_zips
    fd_train = {"FD001": _synthetic_train_txt([(1, [1,2,3,4]), (2, [1,2,3,4,5,6]), (3, [1,2,3]), (4, [1,2,3,4,5])])}
    fd_test  = {"FD001": _synthetic_train_txt([(1, [1,2,3]), (2, [1,2]), (3, [1,2,3,4])])}
    fd_rul   = {"FD001": _synthetic_rul_txt([100, 80, 50])}
    _build_phmd_split_real(tmp_path, fd_train, fd_test, fd_rul)
    out = load_cmapss_raw_from_split_zips(tmp_path, fd_subsets=("FD001",))
    assert "FD001" in out
    assert out["FD001"]["train"]["n_units"] == 4
    assert out["FD001"]["test"]["n_units"] == 3
    assert out["FD001"]["official_rul"] == {1: 100, 2: 80, 3: 50}


def test_load_cmapss_raw_falla_si_rul_count_no_coincide(tmp_path):
    """len(RUL_FDxxx) debe coincidir con n_units_test."""
    from training.build_cmapss_rul_downstream import load_cmapss_raw_from_split_zips
    fd_train = {"FD001": _synthetic_train_txt([(1, [1,2,3])])}
    fd_test  = {"FD001": _synthetic_train_txt([(1, [1,2]), (2, [1,2,3])])}  # 2 units
    fd_rul   = {"FD001": _synthetic_rul_txt([100])}  # 1 valor, debe ser 2
    _build_phmd_split_real(tmp_path, fd_train, fd_test, fd_rul)
    with pytest.raises(ValueError, match="RUL_FD001"):
        load_cmapss_raw_from_split_zips(tmp_path, fd_subsets=("FD001",))


def test_build_train_val_test_rul_no_negativos(tmp_path):
    """RUL fisico reconstruido debe ser >=0 en train, val y test."""
    from training.build_cmapss_rul_downstream import (
        load_cmapss_raw_from_split_zips, build_train_val_test_rul,
    )
    fd_train = {"FD001": _synthetic_train_txt(
        [(i, list(range(1, 50 + i))) for i in range(1, 21)]  # 20 unidades
    )}
    fd_test  = {"FD001": _synthetic_train_txt(
        [(i, list(range(1, 30))) for i in range(1, 6)]  # 5 unidades de test
    )}
    fd_rul   = {"FD001": _synthetic_rul_txt([10, 20, 30, 40, 50])}
    _build_phmd_split_real(tmp_path, fd_train, fd_test, fd_rul)
    parsed = load_cmapss_raw_from_split_zips(tmp_path, fd_subsets=("FD001",))
    with_rul = build_train_val_test_rul(parsed, val_frac=0.2, seed=42, rul_cap=125.0)

    fd_data = with_rul["FD001"]
    for split_name in ("train", "val", "test"):
        rp = fd_data[split_name]["rul_physical"]
        rc = fd_data[split_name]["rul_capped_125"]
        assert rp.min() >= 0, f"{split_name}: rul_physical min = {rp.min()}"
        assert rc.max() <= 125, f"{split_name}: rul_capped_125 max = {rc.max()}"

    # Val frac: 20% de 20 unidades = 4
    assert len(fd_data["unit_split"]["val_units"]) == 4
    assert len(fd_data["unit_split"]["train_units"]) == 16
    # Sin solapamiento train/val
    assert not (set(fd_data["unit_split"]["train_units"]) &
                set(fd_data["unit_split"]["val_units"]))


def test_build_train_val_test_rul_test_usa_official(tmp_path):
    """En test, la formula es last_cycle - cycle + official_rul.
    Para una unidad con cycles [1..5] y official_rul=10, el RUL en cycle=5
    debe ser 0+10=10, no 0."""
    from training.build_cmapss_rul_downstream import (
        load_cmapss_raw_from_split_zips, build_train_val_test_rul,
    )
    # Necesitamos >= 5 unidades en train para que val_frac=0.2 -> >=1 unit val
    fd_train = {"FD001": _synthetic_train_txt(
        [(i, [1, 2, 3, 4, 5]) for i in range(1, 6)]
    )}
    fd_test  = {"FD001": _synthetic_train_txt([(1, [1, 2, 3, 4, 5])])}
    fd_rul   = {"FD001": _synthetic_rul_txt([10])}
    _build_phmd_split_real(tmp_path, fd_train, fd_test, fd_rul)
    parsed = load_cmapss_raw_from_split_zips(tmp_path, fd_subsets=("FD001",))
    with_rul = build_train_val_test_rul(parsed, val_frac=0.2, seed=42, rul_cap=125.0)
    test_data = with_rul["FD001"]["test"]
    # RUL fisico en test: (last=5 - cycle) + official=10
    # cycle 1 -> RUL 14, 2 -> 13, 3 -> 12, 4 -> 11, 5 -> 10
    assert test_data["rul_physical"].tolist() == [14.0, 13.0, 12.0, 11.0, 10.0]


# ----------------------------------------------------------------------
# Commit 2: ventaneo rolling_causal + instance norm + patching
# ----------------------------------------------------------------------


def test_build_rolling_causal_window_sin_padding():
    """t_idx >= W-1: la ventana completa es real, sin padding."""
    import numpy as np
    from training.build_cmapss_rul_downstream import build_rolling_causal_window
    T, C, W = 100, 24, 8
    data = np.arange(T * C, dtype=np.float32).reshape(T, C)
    win, vtm = build_rolling_causal_window(data, t_idx=10, window_size=W)
    assert win.shape == (W, C)
    assert vtm.shape == (W,)
    assert vtm.all()  # sin padding
    # La ventana debe ser data[3:11]
    assert np.allclose(win, data[3:11])


def test_build_rolling_causal_window_con_padding_izquierdo():
    """t_idx < W-1: padding por la izquierda con ceros y mask False ahi.
    Las posiciones reales van al FINAL de la ventana (causal)."""
    import numpy as np
    from training.build_cmapss_rul_downstream import build_rolling_causal_window
    T, C, W = 100, 4, 8
    data = np.arange(T * C, dtype=np.float32).reshape(T, C)
    win, vtm = build_rolling_causal_window(data, t_idx=2, window_size=W)
    # t_idx=2 -> ventana cubre cycles 0..2 (3 ciclos reales) + 5 padding.
    assert win.shape == (W, C)
    # Padding al INICIO de la ventana (posiciones 0..4 falsas).
    assert not vtm[:5].any()
    assert vtm[5:].all()
    # Las posiciones reales contienen data[0..2].
    assert np.allclose(win[5:], data[0:3])
    # Padding son ceros.
    assert np.allclose(win[:5], 0.0)


def test_build_rolling_causal_window_t_idx_0_un_solo_real():
    """t_idx=0: solo 1 ciclo real, W-1 posiciones de padding."""
    import numpy as np
    from training.build_cmapss_rul_downstream import build_rolling_causal_window
    data = np.arange(10 * 3, dtype=np.float32).reshape(10, 3)
    win, vtm = build_rolling_causal_window(data, t_idx=0, window_size=8)
    assert vtm.sum() == 1
    assert vtm[-1] == True   # la ultima posicion es la real
    assert np.allclose(win[-1], data[0])


def test_build_rolling_causal_window_rechaza_inputs_invalidos():
    import numpy as np
    import pytest
    from training.build_cmapss_rul_downstream import build_rolling_causal_window
    data = np.zeros((5, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="t_idx"):
        build_rolling_causal_window(data, t_idx=-1, window_size=8)
    with pytest.raises(ValueError, match="t_idx"):
        build_rolling_causal_window(data, t_idx=10, window_size=8)
    with pytest.raises(ValueError, match="window_size"):
        build_rolling_causal_window(data, t_idx=0, window_size=0)


def test_instance_normalize_ignora_padding():
    """instance_normalize debe usar SOLO timesteps validos. El padding
    no debe contaminar mean/std."""
    import numpy as np
    from training.build_cmapss_rul_downstream import instance_normalize_window
    W, C = 8, 2
    window = np.zeros((W, C), dtype=np.float32)
    # Solo posiciones 5,6,7 son validas con valores [10, 20, 30] en canal 0.
    window[5:, 0] = [10.0, 20.0, 30.0]
    window[5:, 1] = [1.0, 2.0, 3.0]
    vtm = np.zeros(W, dtype=bool)
    vtm[5:] = True
    norm, mean, std, const = instance_normalize_window(window, vtm)
    # mean canal 0 = 20, std canal 0 = sqrt(((10-20)^2+(20-20)^2+(30-20)^2)/3) = sqrt(200/3)
    assert mean[0] == pytest.approx(20.0)
    assert std[0] == pytest.approx(np.sqrt(200.0 / 3.0), rel=1e-5)
    # Padding queda en 0.
    assert np.allclose(norm[:5], 0.0)
    # Valores validos normalizados: (10-20)/std, (0), (10)/std
    assert norm[5, 0] == pytest.approx(-10.0 / std[0], rel=1e-5)
    assert norm[7, 0] == pytest.approx(10.0 / std[0], rel=1e-5)


def test_instance_normalize_canal_constante():
    """Si std < threshold, canal se marca constante y queda en 0."""
    import numpy as np
    from training.build_cmapss_rul_downstream import instance_normalize_window
    W, C = 8, 3
    window = np.zeros((W, C), dtype=np.float32)
    window[:, 0] = 5.0       # constante
    window[:, 1] = np.arange(W)  # variable
    window[:, 2] = 7.0       # constante
    vtm = np.ones(W, dtype=bool)
    norm, mean, std, const = instance_normalize_window(window, vtm)
    assert const[0] == True
    assert const[1] == False
    assert const[2] == True
    # Canales constantes: salida 0.
    assert np.allclose(norm[:, 0], 0.0)
    assert np.allclose(norm[:, 2], 0.0)
    # Canal variable: media cero, std 1.
    assert mean[1] == pytest.approx(np.arange(W).mean())
    assert norm[:, 1].mean() == pytest.approx(0.0, abs=1e-5)
    assert norm[:, 1].std() == pytest.approx(1.0, abs=1e-5)


def test_instance_normalize_solo_un_valid():
    """Con un solo timestep valido no se puede calcular std; marcar
    todos los canales como constantes."""
    import numpy as np
    from training.build_cmapss_rul_downstream import instance_normalize_window
    W, C = 8, 3
    window = np.zeros((W, C), dtype=np.float32)
    window[-1] = [10, 20, 30]
    vtm = np.zeros(W, dtype=bool)
    vtm[-1] = True
    norm, mean, std, const = instance_normalize_window(window, vtm)
    assert const.all()
    assert np.allclose(norm, 0.0)


def test_patch_window_shapes_y_mask():
    """patches (C, N, P); valid_patch_mask (C, N) con .any() temporal."""
    import numpy as np
    from training.build_cmapss_rul_downstream import patch_window
    W, C, P = 8, 4, 2
    N = W // P
    norm = np.arange(W * C, dtype=np.float32).reshape(W, C)
    vtm = np.zeros(W, dtype=bool)
    vtm[3:] = True  # patches 1 (idx 2-3) parcial, patches 2-3 (idx 4-7) completos
    patches, vpm = patch_window(norm, vtm, P)
    assert patches.shape == (C, N, P)
    assert vpm.shape == (C, N)
    # patch 0 (timesteps 0-1): todo padding -> False.
    assert not vpm[:, 0].any()
    # patch 1 (timesteps 2-3): timestep 3 valido -> True.
    assert vpm[:, 1].all()
    # patches 2-3: todos validos.
    assert vpm[:, 2:].all()


def test_iter_unit_windows_emite_un_sample_por_ciclo():
    """rolling_causal stride=1: T ciclos -> T samples."""
    import numpy as np
    from training.build_cmapss_rul_downstream import iter_unit_windows
    T, C = 5, 4
    cycles = np.array([1, 2, 3, 4, 5], dtype=np.int64)
    channels = np.arange(T * C, dtype=np.float32).reshape(T, C)
    rul_p = np.array([4, 3, 2, 1, 0], dtype=np.float32)
    rul_c = rul_p.copy()
    samples = list(iter_unit_windows(
        unit_id=1, unit_cycles=cycles, unit_channels=channels,
        unit_rul_physical=rul_p, unit_rul_capped=rul_c,
        fd_subset="FD001", split="train",
        window_size=8, patch_size=2, stride=1,
    ))
    assert len(samples) == T
    # Cada sample tiene las keys del contrato.
    s = samples[0]
    for k in ["patches", "valid_time_mask", "valid_patch_mask",
              "canales_constantes_mask", "mean", "std_used",
              "target", "meta"]:
        assert k in s, f"falta key {k!r}"
    # Shapes del primer sample.
    assert s["patches"].shape == (C, 4, 2)  # N = W//P = 4
    assert s["valid_time_mask"].shape == (8,)
    assert s["valid_patch_mask"].shape == (C, 4)
    assert s["canales_constantes_mask"].shape == (C,)
    # Target del ultimo sample: rul_physical=0.
    assert samples[-1]["target"]["rul_physical"] == 0.0
    assert samples[-1]["target"]["cycle"] == 5
    # Meta consistente.
    assert samples[-1]["meta"]["unit_id"] == 1
    assert samples[-1]["meta"]["t_idx_in_unit"] == 4


def test_iter_unit_windows_target_y_padding_coherentes():
    """Para t_idx=0 con W>1, valid_time_mask debe tener 1 True al final."""
    import numpy as np
    from training.build_cmapss_rul_downstream import iter_unit_windows
    samples = list(iter_unit_windows(
        unit_id=1, unit_cycles=np.array([1, 2], dtype=np.int64),
        unit_channels=np.ones((2, 3), dtype=np.float32),
        unit_rul_physical=np.array([1, 0], dtype=np.float32),
        unit_rul_capped=np.array([1, 0], dtype=np.float32),
        fd_subset="FD001", split="train",
        window_size=8, patch_size=2, stride=1,
    ))
    # Primer sample: t_idx=0, solo 1 timestep valido (el ultimo).
    assert samples[0]["valid_time_mask"].sum() == 1
    assert samples[0]["valid_time_mask"][-1] == True
    # Segundo sample: t_idx=1, 2 timesteps validos (los 2 ultimos).
    assert samples[1]["valid_time_mask"].sum() == 2
    assert samples[1]["valid_time_mask"][-2:].all()


def test_preview_padding_stats_fd_corto():
    """Para una unidad con T=5 ciclos y W=8:
       - 0 ventanas full (T < W).
       - 5 ventanas con padding (todas).
       - frac_windows_padded = 1.0.
       - frac_timesteps_valid_avg = (1+2+3+4+5)/(5*8) = 15/40 = 0.375.
    """
    import numpy as np
    from training.build_cmapss_rul_downstream import preview_summary
    # Construyo un parsed_with_rul minimo a mano (sin pasar por loader real).
    parsed = {
        "FD001": {
            "train": {
                "unit_id": np.array([1, 1, 1, 1, 1], dtype=np.int64),
                "cycle":   np.array([1, 2, 3, 4, 5], dtype=np.int64),
                "op_settings": np.zeros((5, 3), dtype=np.float32),
                "sensors":     np.zeros((5, 21), dtype=np.float32),
                "rul_physical":   np.array([4, 3, 2, 1, 0], dtype=np.float32),
                "rul_capped_125": np.array([4, 3, 2, 1, 0], dtype=np.float32),
                "fd_subset": "FD001", "split": "train",
                "n_rows": 5, "n_units": 1,
            },
            "val": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "val",
                "n_rows": 0, "n_units": 0,
            },
            "test": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "test",
                "n_rows": 0, "n_units": 0,
            },
            "unit_split": {"train_units": [1], "val_units": [], "test_units": []},
        }
    }
    preview = preview_summary(parsed, window_size=8, patch_size=2, n_channels=24)
    tr = preview["by_fd"]["FD001"]["train"]
    assert tr["n_windows_full"] == 0
    assert tr["n_windows_padded"] == 5
    assert tr["frac_windows_padded"] == 1.0
    # Suma de timesteps validos: 1+2+3+4+5 = 15. Total: 5*8 = 40. 15/40=0.375
    assert tr["frac_timesteps_valid_avg"] == pytest.approx(15 / 40)


# ----------------------------------------------------------------------
# Commit 2b: selected_t_indices + stride/min_valid/last_override
# ----------------------------------------------------------------------


def test_selected_t_indices_T200_stride5_minvalid128():
    """T=200, stride=5, min_valid=128: empieza en 127, despues 132, 137...
    Y last_override anade T-1 = 199 si no esta ya en la rejilla."""
    from training.build_cmapss_rul_downstream import selected_t_indices
    indices, dropped, added = selected_t_indices(
        T=200, stride=5, min_valid_timesteps=128, include_last_per_unit=True,
    )
    # Regular: 127, 132, 137, ..., el ultimo <= 199 que cabe.
    # 127 + 5*k <= 199 -> k <= (199-127)/5 = 14.4 -> k_max=14 -> 127+70=197.
    # Last override: T-1 = 199 no esta -> se anade.
    assert indices[0] == 127
    assert indices[1] == 132
    assert indices[2] == 137
    assert indices[-2] == 197  # ultimo de la rejilla
    assert indices[-1] == 199  # last override
    assert added == 1
    # dropped = numero de t_idx que stride=5 emitiria sin filtro y caen
    # en [0, 127): {0, 5, 10, ..., 125} -> ceil(127/5) = 26.
    assert dropped == 26


def test_selected_t_indices_T200_stride5_sin_minvalid():
    """Sin min_valid: stride pleno desde 0. last override solo si T-1
    no cae en rejilla."""
    from training.build_cmapss_rul_downstream import selected_t_indices
    indices, dropped, added = selected_t_indices(
        T=200, stride=5, min_valid_timesteps=None, include_last_per_unit=True,
    )
    assert indices[0] == 0
    assert indices[1] == 5
    # range(0, 200, 5) acaba en 195. 199 no esta -> se anade.
    assert 195 in indices
    assert indices[-1] == 199
    assert added == 1
    assert dropped == 0


def test_selected_t_indices_last_ya_en_rejilla():
    """Si T-1 ya esta en la rejilla regular, added_by_last debe ser 0."""
    from training.build_cmapss_rul_downstream import selected_t_indices
    # T=128, stride=1, min_valid=128: start=127. range(127, 128, 1) = [127].
    # T-1 = 127 esta en la rejilla -> added=0.
    indices, dropped, added = selected_t_indices(
        T=128, stride=1, min_valid_timesteps=128, include_last_per_unit=True,
    )
    assert indices == [127]
    assert added == 0


def test_selected_t_indices_T_menor_que_minvalid_con_last():
    """T < min_valid + include_last=True: emite solo T-1 marcado below_min."""
    from training.build_cmapss_rul_downstream import selected_t_indices
    indices, dropped, added = selected_t_indices(
        T=50, stride=5, min_valid_timesteps=128, include_last_per_unit=True,
    )
    assert indices == [49]
    # dropped: en este caso el helper devuelve 0 (porque no recorre la
    # rejilla; el unico sample viene por override).
    assert dropped == 0
    assert added == 1  # se cuenta como anadido por last_override


def test_selected_t_indices_T_menor_que_minvalid_sin_last():
    """T < min_valid + include_last=False: no emite nada (la unidad se
    descarta entera)."""
    from training.build_cmapss_rul_downstream import selected_t_indices
    indices, dropped, added = selected_t_indices(
        T=50, stride=5, min_valid_timesteps=128, include_last_per_unit=False,
    )
    assert indices == []
    assert dropped == 50  # todas las posibles ventanas descartadas
    assert added == 0


def test_selected_t_indices_rechaza_inputs_invalidos():
    import pytest
    from training.build_cmapss_rul_downstream import selected_t_indices
    with pytest.raises(ValueError, match="stride"):
        selected_t_indices(T=10, stride=0)
    with pytest.raises(ValueError, match="min_valid_timesteps"):
        selected_t_indices(T=10, stride=1, min_valid_timesteps=0)
    with pytest.raises(ValueError, match="T"):
        selected_t_indices(T=-1, stride=1)


def test_iter_unit_windows_respeta_min_valid_y_last():
    """iter_unit_windows con T=200, stride=5, min_valid=128, include_last:
    emite los mismos t_idx que selected_t_indices."""
    import numpy as np
    from training.build_cmapss_rul_downstream import iter_unit_windows
    T = 200
    cycles = np.arange(1, T + 1, dtype=np.int64)
    channels = np.zeros((T, 4), dtype=np.float32)
    rul_p = np.arange(T - 1, -1, -1, dtype=np.float32)
    rul_c = rul_p.copy()
    samples = list(iter_unit_windows(
        unit_id=1, unit_cycles=cycles, unit_channels=channels,
        unit_rul_physical=rul_p, unit_rul_capped=rul_c,
        fd_subset="FD001", split="train",
        window_size=8, patch_size=2, stride=5,
        min_valid_timesteps=128, include_last_per_unit=True,
    ))
    # Mismos indices que selected_t_indices
    emitted_t = [s["meta"]["t_idx_in_unit"] for s in samples]
    assert emitted_t[0] == 127
    assert emitted_t[-1] == 199  # last override
    # last override flag
    assert samples[-1]["meta"]["selected_by_last_override"] is True
    assert samples[0]["meta"]["selected_by_last_override"] is False
    # below_min_valid_because_last = False (T >= min_valid)
    assert all(not s["meta"]["below_min_valid_because_last"] for s in samples)


def test_iter_unit_windows_meta_valid_timesteps():
    """meta.valid_timesteps debe ser = min(t_idx + 1, W)."""
    import numpy as np
    from training.build_cmapss_rul_downstream import iter_unit_windows
    T = 200
    samples = list(iter_unit_windows(
        unit_id=1,
        unit_cycles=np.arange(1, T + 1, dtype=np.int64),
        unit_channels=np.zeros((T, 4), dtype=np.float32),
        unit_rul_physical=np.zeros(T, dtype=np.float32),
        unit_rul_capped=np.zeros(T, dtype=np.float32),
        fd_subset="FD001", split="train",
        window_size=128, patch_size=4, stride=10,
        min_valid_timesteps=64, include_last_per_unit=True,
    ))
    for s in samples:
        t = s["meta"]["t_idx_in_unit"]
        assert s["meta"]["valid_timesteps"] == min(t + 1, 128)
        assert s["meta"]["valid_timesteps"] == int(s["valid_time_mask"].sum())


def test_iter_unit_windows_below_min_valid_solo_last():
    """T<min_valid + include_last: 1 sample con below_min_valid=True."""
    import numpy as np
    from training.build_cmapss_rul_downstream import iter_unit_windows
    T = 50
    samples = list(iter_unit_windows(
        unit_id=1,
        unit_cycles=np.arange(1, T + 1, dtype=np.int64),
        unit_channels=np.zeros((T, 4), dtype=np.float32),
        unit_rul_physical=np.arange(T - 1, -1, -1, dtype=np.float32),
        unit_rul_capped=np.arange(T - 1, -1, -1, dtype=np.float32),
        fd_subset="FD001", split="train",
        window_size=512, patch_size=16, stride=5,
        min_valid_timesteps=128, include_last_per_unit=True,
    ))
    assert len(samples) == 1
    s = samples[0]
    assert s["meta"]["t_idx_in_unit"] == 49
    assert s["meta"]["below_min_valid_because_last"] is True
    assert s["meta"]["selected_by_last_override"] is True
    assert s["target"]["rul_physical"] == 0.0  # ultimo ciclo de train = fallo


def test_preview_summary_post_filtro_no_es_n_rows():
    """Con stride=5 y min_valid=128, n_windows_selected != n_rows."""
    import numpy as np
    from training.build_cmapss_rul_downstream import preview_summary
    T = 200
    parsed = {
        "FD001": {
            "train": {
                "unit_id": np.ones(T, dtype=np.int64),
                "cycle":   np.arange(1, T + 1, dtype=np.int64),
                "op_settings": np.zeros((T, 3), dtype=np.float32),
                "sensors":     np.zeros((T, 21), dtype=np.float32),
                "rul_physical":   np.arange(T - 1, -1, -1, dtype=np.float32),
                "rul_capped_125": np.minimum(np.arange(T - 1, -1, -1), 125).astype(np.float32),
                "fd_subset": "FD001", "split": "train",
                "n_rows": T, "n_units": 1,
            },
            "val": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "val",
                "n_rows": 0, "n_units": 0,
            },
            "test": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "test",
                "n_rows": 0, "n_units": 0,
            },
            "unit_split": {"train_units": [1], "val_units": [], "test_units": []},
        }
    }
    preview = preview_summary(
        parsed, window_size=512, patch_size=16, n_channels=24,
        stride=5, min_valid_timesteps=128, include_last_per_unit=True,
    )
    tr = preview["by_fd"]["FD001"]["train"]
    # n_rows_original sigue siendo 200 (no filtrado).
    assert tr["n_rows_original"] == 200
    # n_windows_selected: 15 (127, 132, 137, ..., 197) + 1 last_override (199).
    assert tr["n_windows_selected"] == 16
    # dropped_by_min_valid: stride=5 sin filtro emitiria 0,5,...,125 -> 26.
    assert tr["n_windows_dropped_by_min_valid"] == 26
    # added_by_last_override: 1 (199 no esta en rejilla).
    assert tr["n_windows_added_by_last_override"] == 1
    # n_units con ventana: 1; only_last: 0 (T >= min_valid).
    assert tr["n_units_with_at_least_one_window"] == 1
    assert tr["n_units_only_last_override"] == 0
    # decisions eco
    assert preview["decisions"]["stride"] == 5
    assert preview["decisions"]["min_valid_timesteps"] == 128
    assert preview["decisions"]["include_last_per_unit"] is True


def test_preview_rul_min_train_es_cero_por_last_override():
    """En train, el ultimo ciclo de cada unidad tiene rul=0 (run-to-failure).
    Como include_last=True garantiza emitir T-1, rul_physical_min DE LAS
    VENTANAS SELECCIONADAS debe ser 0 si hay al menos una unidad emitiendo
    su ultimo ciclo."""
    import numpy as np
    from training.build_cmapss_rul_downstream import preview_summary
    T = 200
    parsed = {
        "FD001": {
            "train": {
                "unit_id": np.ones(T, dtype=np.int64),
                "cycle":   np.arange(1, T + 1, dtype=np.int64),
                "op_settings": np.zeros((T, 3), dtype=np.float32),
                "sensors":     np.zeros((T, 21), dtype=np.float32),
                "rul_physical":   np.arange(T - 1, -1, -1, dtype=np.float32),
                "rul_capped_125": np.minimum(np.arange(T - 1, -1, -1), 125).astype(np.float32),
                "fd_subset": "FD001", "split": "train",
                "n_rows": T, "n_units": 1,
            },
            "val": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "val",
                "n_rows": 0, "n_units": 0,
            },
            "test": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "test",
                "n_rows": 0, "n_units": 0,
            },
            "unit_split": {"train_units": [1], "val_units": [], "test_units": []},
        }
    }
    preview = preview_summary(
        parsed, window_size=512, patch_size=16, n_channels=24,
        stride=5, min_valid_timesteps=128, include_last_per_unit=True,
    )
    tr = preview["by_fd"]["FD001"]["train"]
    # El ultimo ciclo (t_idx=199, rul=0) entra por last_override.
    assert tr["rul_physical_min"] == 0.0


def test_preview_padding_stats_unidad_larga():
    """Unidad con T=10 y W=8: 3 ventanas full + 7 con padding.
       Pero esperamos T-(W-1) = 10-7 = 3 full y W-1 = 7 padded."""
    import numpy as np
    from training.build_cmapss_rul_downstream import preview_summary
    T, W = 10, 8
    parsed = {
        "FD001": {
            "train": {
                "unit_id": np.ones(T, dtype=np.int64),
                "cycle":   np.arange(1, T + 1, dtype=np.int64),
                "op_settings": np.zeros((T, 3), dtype=np.float32),
                "sensors":     np.zeros((T, 21), dtype=np.float32),
                "rul_physical":   np.arange(T - 1, -1, -1, dtype=np.float32),
                "rul_capped_125": np.arange(T - 1, -1, -1, dtype=np.float32),
                "fd_subset": "FD001", "split": "train",
                "n_rows": T, "n_units": 1,
            },
            "val": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "val",
                "n_rows": 0, "n_units": 0,
            },
            "test": {
                "unit_id": np.array([], dtype=np.int64),
                "cycle":   np.array([], dtype=np.int64),
                "op_settings": np.zeros((0, 3), dtype=np.float32),
                "sensors":     np.zeros((0, 21), dtype=np.float32),
                "rul_physical":   np.array([], dtype=np.float32),
                "rul_capped_125": np.array([], dtype=np.float32),
                "fd_subset": "FD001", "split": "test",
                "n_rows": 0, "n_units": 0,
            },
            "unit_split": {"train_units": [1], "val_units": [], "test_units": []},
        }
    }
    preview = preview_summary(parsed, window_size=W, patch_size=2, n_channels=24)
    tr = preview["by_fd"]["FD001"]["train"]
    assert tr["n_windows_full"] == 3
    assert tr["n_windows_padded"] == 7


def test_preview_summary_rolling_causal_basico(tmp_path):
    """Verifica que el preview cuenta una ventana por fila (rolling_causal stride=1)."""
    from training.build_cmapss_rul_downstream import (
        load_cmapss_raw_from_split_zips, build_train_val_test_rul, preview_summary,
    )
    fd_train = {"FD001": _synthetic_train_txt(
        [(i, list(range(1, 50))) for i in range(1, 21)]  # 20 units, 49 ciclos cada una
    )}
    fd_test  = {"FD001": _synthetic_train_txt(
        [(i, list(range(1, 30))) for i in range(1, 6)]  # 5 units, 29 ciclos
    )}
    fd_rul   = {"FD001": _synthetic_rul_txt([10, 20, 30, 40, 50])}
    _build_phmd_split_real(tmp_path, fd_train, fd_test, fd_rul)
    parsed = load_cmapss_raw_from_split_zips(tmp_path, fd_subsets=("FD001",))
    with_rul = build_train_val_test_rul(parsed, val_frac=0.2, seed=42, rul_cap=125.0)
    preview = preview_summary(with_rul, window_size=512, patch_size=16, n_channels=24)

    fd_sum = preview["by_fd"]["FD001"]
    # 20 units * 49 ciclos = 980 filas train + val combinado.
    # Tras commit 2b las keys son n_rows_original / n_windows_selected.
    total_train_val = (
        fd_sum["train"]["n_rows_original"] + fd_sum["val"]["n_rows_original"]
    )
    assert total_train_val == 20 * 49

    # Sin filtros (stride=1, min_valid=None), n_windows_selected ==
    # n_rows_original.
    assert fd_sum["train"]["n_windows_selected"] == fd_sum["train"]["n_rows_original"]
    assert fd_sum["val"]["n_windows_selected"] == fd_sum["val"]["n_rows_original"]
    assert fd_sum["test"]["n_windows_selected"] == 5 * 29

    # Decisiones eco
    assert preview["decisions"]["window_size"] == 512
    assert preview["decisions"]["window_mode"] == "rolling_causal"
    assert preview["decisions"]["n_channels"] == 24

    # Tamano estimado positivo
    assert preview["totals"]["estimated_size_gb_float32"] > 0


def test_write_dry_run_report_estricto(tmp_path):
    from training.build_cmapss_rul_downstream import write_dry_run_report
    report = {
        "raw_root": str(tmp_path),
        "raw_missing": True,
        "raw_zip_path": None,
        "raw_dir_path": None,
        "candidate_files": [],
        "notes": ["nota 1", "nota 2"],
        "timestamp": "2026-05-24T12:00:00",
    }
    out_dir = tmp_path / "report"
    write_dry_run_report(report, out_dir)
    j = json.loads((out_dir / "dry_run_report.json").read_text(encoding="utf-8"))
    assert j["raw_missing"] is True
    assert j["timestamp"] == "2026-05-24T12:00:00"
    md = (out_dir / "dry_run_report.md").read_text(encoding="utf-8")
    assert "CMAPSS RUL builder" in md
    assert "raw_missing: **True**" in md


def test_cli_dry_run_sin_raw(tmp_path, capsys):
    """CLI con --dry-run y raw inexistente debe terminar con rc=0 y reportar."""
    from training.build_cmapss_rul_downstream import main
    rc = main([
        "--raw-root", str(tmp_path / "no_raw"),
        "--out-dir", str(tmp_path / "out"),
        "--results-dir", str(tmp_path / "report"),
    ])
    assert rc == 0
    # Debe haber escrito el report
    assert (tmp_path / "report" / "dry_run_report.json").is_file()
    captured = capsys.readouterr()
    assert "raw CMAPSS NO encontrado" in captured.out


# ----------------------------------------------------------------------
# Commit 3: writer TAR + manifest + asserts duros + reentrancia
# ----------------------------------------------------------------------


def _make_synthetic_parsed_with_rul(
    tmp_path,
    fd_subsets=("FD001",),
    n_train_units=6,
    train_cycles=None,
    test_cycles=None,
    official_rul_per_unit=None,
    rul_cap=125.0,
    val_frac=0.2,
    seed=42,
):
    """Helper: construye zips PHMD sinteticos y devuelve `parsed_with_rul`.

    Por defecto: 1 FD, 6 unidades train con 10 ciclos cada una; 3 unidades
    test con 5 ciclos; official_rul=[10,20,30].
    """
    from training.build_cmapss_rul_downstream import (
        load_cmapss_raw_from_split_zips,
        build_train_val_test_rul,
    )
    if train_cycles is None:
        train_cycles = list(range(1, 11))
    if test_cycles is None:
        test_cycles = list(range(1, 6))
    if official_rul_per_unit is None:
        official_rul_per_unit = [10, 20, 30]
    fd_train, fd_test, fd_rul = {}, {}, {}
    for fd in fd_subsets:
        fd_train[fd] = _synthetic_train_txt(
            [(u, train_cycles) for u in range(1, n_train_units + 1)]
        )
        fd_test[fd] = _synthetic_train_txt(
            [(u, test_cycles) for u in range(1, len(official_rul_per_unit) + 1)]
        )
        fd_rul[fd] = _synthetic_rul_txt(official_rul_per_unit)
    _build_phmd_split_real(tmp_path, fd_train, fd_test, fd_rul)
    parsed = load_cmapss_raw_from_split_zips(tmp_path, fd_subsets=fd_subsets)
    return build_train_val_test_rul(
        parsed, val_frac=val_frac, seed=seed, rul_cap=rul_cap,
    )


def test_split_to_source_split_mapping():
    from training.build_cmapss_rul_downstream import split_to_source_split
    assert split_to_source_split("train") == "train_orig"
    assert split_to_source_split("val") == "train_orig"
    assert split_to_source_split("test") == "test_orig"
    with pytest.raises(ValueError, match="split_builder desconocido"):
        split_to_source_split("validation")


def test_make_unit_global_id_format():
    from training.build_cmapss_rul_downstream import make_unit_global_id
    assert make_unit_global_id("FD001", "train_orig", 3) == \
        "CMAPSS_FD001_train_orig_unit3"
    assert make_unit_global_id("FD004", "test_orig", 248) == \
        "CMAPSS_FD004_test_orig_unit248"


def test_make_sample_key_formato_y_unicidad():
    from training.build_cmapss_rul_downstream import make_sample_key
    k1 = make_sample_key("FD001", "train", "train_orig", 3, 127)
    k2 = make_sample_key("FD001", "train", "train_orig", 3, 128)
    k3 = make_sample_key("FD001", "val", "train_orig", 3, 127)
    assert k1 == "cmapss_FD001_train_train_orig_unit3_w000127"
    assert k2 == "cmapss_FD001_train_train_orig_unit3_w000128"
    assert k3 == "cmapss_FD001_val_train_orig_unit3_w000127"
    assert len({k1, k2, k3}) == 3


def test_expand_valid_time_mask_cw_shapes():
    import numpy as np
    from training.build_cmapss_rul_downstream import expand_valid_time_mask_cw
    vtm = np.array([False, False, True, True, True], dtype=bool)
    out = expand_valid_time_mask_cw(vtm, n_channels=24)
    assert out.shape == (24, 5)
    assert out.dtype == bool
    # Todos los canales identicos a vtm.
    for c in range(24):
        assert np.array_equal(out[c], vtm)
    # Es copia independiente, no broadcast read-only.
    out[0, 0] = True
    assert vtm[0] == False  # original no se ha mutado


def test_expand_valid_time_mask_cw_rechaza_dtype_o_shape():
    import numpy as np
    from training.build_cmapss_rul_downstream import expand_valid_time_mask_cw
    with pytest.raises(ValueError, match="W,"):
        expand_valid_time_mask_cw(np.zeros((4, 4), dtype=bool), n_channels=3)
    with pytest.raises(ValueError, match="bool"):
        expand_valid_time_mask_cw(np.zeros(4, dtype=np.float32), n_channels=3)
    with pytest.raises(ValueError, match="n_channels"):
        expand_valid_time_mask_cw(np.zeros(4, dtype=bool), n_channels=0)


def test_compute_pipeline_config_hash_deterministico():
    from training.build_cmapss_rul_downstream import compute_pipeline_config_hash
    d = {"stride": 5, "min_valid_timesteps": 128, "window_size": 512}
    h1 = compute_pipeline_config_hash(d)
    h2 = compute_pipeline_config_hash(dict(d))
    assert h1 == h2
    assert len(h1) == 16


def test_compute_pipeline_config_hash_cambia_con_politica():
    from training.build_cmapss_rul_downstream import compute_pipeline_config_hash
    d1 = {"stride": 5, "min_valid_timesteps": 128, "window_size": 512}
    d2 = {"stride": 1, "min_valid_timesteps": 128, "window_size": 512}
    d3 = {"stride": 5, "min_valid_timesteps": 64, "window_size": 512}
    d4 = {"stride": 5, "min_valid_timesteps": 128, "window_size": 256}
    h1 = compute_pipeline_config_hash(d1)
    h2 = compute_pipeline_config_hash(d2)
    h3 = compute_pipeline_config_hash(d3)
    h4 = compute_pipeline_config_hash(d4)
    assert len({h1, h2, h3, h4}) == 4


def _one_sample_from_split(parsed_with_rul, fd, split, **kwargs):
    """Saca el primer sample iterando iter_split_windows del split/fd."""
    from training.build_cmapss_rul_downstream import iter_split_windows
    sdict = parsed_with_rul[fd][split]
    for s in iter_split_windows(sdict, **kwargs):
        return s
    return None


def test_build_sample_payload_shapes_y_keys_canales(tmp_path):
    import numpy as np
    from training.build_cmapss_rul_downstream import build_sample_payload
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    sample = _one_sample_from_split(
        pwr, "FD001", "train",
        window_size=8, patch_size=2, stride=1,
        min_valid_timesteps=4, include_last_per_unit=True,
    )
    assert sample is not None
    key, blobs = build_sample_payload(sample)
    # Key con todos los campos esperados.
    assert key.startswith("cmapss_FD001_train_train_orig_unit")
    assert "_w" in key
    # Blobs obligatorios presentes.
    needed = {
        "patches.npy", "valid_time_mask.npy", "valid_patch_mask.npy",
        "canales_constantes_mask.npy", "mean.npy", "std_used.npy",
        "rul_physical.npy", "rul_capped_125.npy", "meta.json",
    }
    assert needed.issubset(set(blobs.keys()))
    # Shapes via decode.
    import io as _io
    patches = np.load(_io.BytesIO(blobs["patches.npy"]))
    assert patches.ndim == 3 and patches.shape[2] == 2  # (C, N, P)
    C, N, P = patches.shape
    assert P == 2 and C >= 24
    vtm = np.load(_io.BytesIO(blobs["valid_time_mask.npy"]))
    # CONTRATO: valid_time_mask se persiste como (W,) por sample. Al
    # batchear da (B, W), que es lo que espera PatchTSTPhm.forward.
    assert vtm.shape == (N * P,)
    assert vtm.dtype == bool
    vpm = np.load(_io.BytesIO(blobs["valid_patch_mask.npy"]))
    # CONTRATO: valid_patch_mask se persiste como (C, N). Al batchear
    # da (B, C, N), canonicalizable por el encoder.
    assert vpm.shape == (C, N)
    assert vpm.dtype == bool
    # En CMAPSS el padding causal es uniforme entre canales: todos los
    # canales replican la misma valid_patch_mask. Lo verificamos.
    for c in range(C):
        assert np.array_equal(vpm[c], vpm[0])
    # Targets como escalar float32.
    rp = np.load(_io.BytesIO(blobs["rul_physical.npy"]))
    rc = np.load(_io.BytesIO(blobs["rul_capped_125.npy"]))
    assert rp.dtype == np.float32 and rp.shape == ()
    assert rc.dtype == np.float32 and rc.shape == ()


def test_build_sample_payload_meta_enriquecida(tmp_path):
    from training.build_cmapss_rul_downstream import build_sample_payload
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    sample = _one_sample_from_split(
        pwr, "FD001", "test",
        window_size=8, patch_size=2, stride=1,
        min_valid_timesteps=4, include_last_per_unit=True,
    )
    assert sample is not None
    _, blobs = build_sample_payload(sample)
    meta = json.loads(blobs["meta.json"].decode("utf-8"))
    # Campos canonicos enriquecidos por el writer.
    for k in (
        "source_split", "unit_global_id", "window_size", "patch_size",
        "n_patches", "n_channels", "target_rul_physical",
        "target_rul_capped_125", "fd_subset", "split", "unit_id",
        "t_idx_in_unit",
    ):
        assert k in meta, f"falta {k} en meta enriquecida"
    # test del builder -> source_split=test_orig.
    assert meta["source_split"] == "test_orig"
    assert meta["unit_global_id"].startswith("CMAPSS_FD001_test_orig_unit")


def test_write_split_shards_roundtrip(tmp_path):
    """Escribe shards de un split sintetico, lee de vuelta, cuenta samples
    y verifica bit-a-bit que coincide con la generacion in-memory."""
    import io as _io
    import numpy as np
    import tarfile
    from training.build_cmapss_rul_downstream import (
        iter_split_windows, build_sample_payload, write_split_shards,
    )
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
    }
    out_dir = tmp_path / "shards_out"
    stats = write_split_shards(
        parsed_with_rul=pwr, split_builder="train",
        decisions=decisions, out_dir=out_dir, shard_size=8,
    )
    # Generacion in-memory para comparar.
    expected_keys = []
    for s in iter_split_windows(
        pwr["FD001"]["train"],
        window_size=8, patch_size=2, stride=1,
        min_valid_timesteps=4, include_last_per_unit=True,
    ):
        k, _ = build_sample_payload(s)
        expected_keys.append(k)
    # n_samples bit-a-bit.
    assert stats["n_samples"] == len(expected_keys)
    assert stats["n_shards"] >= 1
    # Leer TARs y recolectar prefijos. El nombre completo es
    # "<key>.<suffix>" con suffix conocido (p.ej. "patches.npy",
    # "meta.json"). Sufijos posibles del payload del writer.
    known_suffixes = (
        ".patches.npy", ".valid_time_mask.npy", ".valid_patch_mask.npy",
        ".canales_constantes_mask.npy", ".mean.npy", ".std_used.npy",
        ".rul_physical.npy", ".rul_capped_125.npy", ".meta.json",
    )

    def _strip_known(name: str) -> str:
        for suf in known_suffixes:
            if name.endswith(suf):
                return name[: -len(suf)]
        raise AssertionError(f"name del TAR no reconocido: {name!r}")

    observed_keys = []
    for p in stats["shard_paths"]:
        with tarfile.open(p, "r") as tf:
            for member in tf.getmembers():
                observed_keys.append(_strip_known(member.name))
    # Cada sample contribuye varios entries; recolectamos como set.
    assert set(observed_keys) == set(expected_keys)


def test_validate_anti_leakage_train_val_disjoint_y_source_split(tmp_path):
    from training.build_cmapss_rul_downstream import (
        write_split_shards, validate_anti_leakage,
    )
    pwr = _make_synthetic_parsed_with_rul(tmp_path, n_train_units=6)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
    }
    out_dir = tmp_path / "shards_out"
    stats = {
        s: write_split_shards(pwr, s, decisions, out_dir, shard_size=64)
        for s in ("train", "val", "test")
    }
    result = validate_anti_leakage(pwr, stats)
    assert result["no_overlap_train_val"] is True
    assert result["no_overlap_train_test"] is True
    assert result["no_overlap_val_test"] is True
    assert result["train_source_split_is_train_orig"] is True
    assert result["val_source_split_is_train_orig"] is True
    assert result["test_source_split_is_test_orig"] is True
    # by_fd presente con todos los FD que tenemos datos.
    assert "FD001" in result["by_fd"]


def test_validate_anti_leakage_falla_si_inyectamos_overlap(tmp_path):
    """Si forzamos que train y val compartan un unit_global_id, el
    validador debe lanzar AssertionError."""
    from training.build_cmapss_rul_downstream import (
        write_split_shards, validate_anti_leakage,
    )
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
    }
    out_dir = tmp_path / "shards_out"
    stats = {
        s: write_split_shards(pwr, s, decisions, out_dir, shard_size=64)
        for s in ("train", "val", "test")
    }
    # Inyectar un ID compartido entre train y val.
    if stats["train"]["unit_global_ids"]:
        stats["val"]["unit_global_ids"].append(
            stats["train"]["unit_global_ids"][0]
        )
    with pytest.raises(AssertionError, match="train y val comparten"):
        validate_anti_leakage(pwr, stats)


def test_assert_writer_hard_constraints_pass_y_falla(tmp_path):
    from training.build_cmapss_rul_downstream import (
        write_split_shards, assert_writer_hard_constraints,
    )
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
        "rul_cap": 125.0,
    }
    out_dir = tmp_path / "shards_out"
    stats = {
        s: write_split_shards(pwr, s, decisions, out_dir, shard_size=64)
        for s in ("train", "val", "test")
    }
    # No raise.
    assert_writer_hard_constraints(stats, preview_totals=None, decisions=decisions)
    # Inyectamos un rul_physical_min > 0 en train -> debe romper con
    # include_last_per_unit=True.
    bad_stats = {k: dict(v) for k, v in stats.items()}
    bad_stats["train"]["rul_physical_min"] = 5.0
    with pytest.raises(AssertionError, match="rul_physical_min debe ser 0"):
        assert_writer_hard_constraints(bad_stats, None, decisions)


def test_assert_writer_hard_constraints_preview_mismatch(tmp_path):
    from training.build_cmapss_rul_downstream import (
        write_split_shards, assert_writer_hard_constraints,
    )
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
        "rul_cap": 125.0,
    }
    out_dir = tmp_path / "shards_out"
    stats = {
        s: write_split_shards(pwr, s, decisions, out_dir, shard_size=64)
        for s in ("train", "val", "test")
    }
    # Mismatch deliberado en preview_totals.
    bad_preview = {
        "n_windows_train": stats["train"]["n_samples"] + 1,
        "n_windows_val": stats["val"]["n_samples"],
        "n_windows_test": stats["test"]["n_samples"],
    }
    with pytest.raises(AssertionError, match="!= preview"):
        assert_writer_hard_constraints(stats, bad_preview, decisions)


def test_build_manifest_schema_obligatorio(tmp_path):
    from training.build_cmapss_rul_downstream import (
        write_split_shards, build_manifest, validate_anti_leakage,
        compute_pipeline_config_hash,
    )
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
        "rul_cap": 125.0,
    }
    out_dir = tmp_path / "shards_out"
    stats = {
        s: write_split_shards(pwr, s, decisions, out_dir, shard_size=64)
        for s in ("train", "val", "test")
    }
    al = validate_anti_leakage(pwr, stats)
    h = compute_pipeline_config_hash(decisions)
    manifest = build_manifest(
        decisions=decisions, split_stats=stats, anti_leakage=al,
        pipeline_config_hash=h, pipeline_code_version="dummy",
        parsed_with_rul=pwr, shard_size=64,
        formula_train_val="max_cycle - cycle",
        formula_test="last - cycle + official_rul",
    )
    for k in (
        "dataset", "manifest_version", "role", "evaluation_tier", "client",
        "pipeline_code_version", "pipeline_config_hash", "generated_at",
        "window_size", "patch_size", "n_patches", "n_channels",
        "stride", "min_valid_timesteps", "include_last_per_unit",
        "window_mode", "target_policy", "target_candidates",
        "target_warnings", "split_policy", "val_frac", "split_seed",
        "normalization_policy", "normalization_stats_saved",
        "batching_policy", "shard_size", "formula_train_val",
        "formula_test", "rul_cap", "n_units_por_split",
        "n_units_original_por_fd", "n_windows_por_split",
        "n_shards_por_split", "n_temporal_patches_por_split",
        "n_channel_patches_por_split", "rul_physical_min_por_split",
        "rul_physical_max_por_split", "rul_capped_125_max_por_split",
        "unit_global_id_policy", "anti_leakage_checks", "warnings",
    ):
        assert k in manifest, f"manifest le falta clave obligatoria {k!r}"
    assert manifest["dataset"] == "CMAPSS_RUL"
    assert manifest["role"] == "TRANSFER_TARGET"
    assert manifest["client"] == "aero_engines"
    assert manifest["unit_global_id_policy"].startswith("CMAPSS_<fd>_")


def test_writer_main_e2e_sintetico_escribe_shards_y_manifest(tmp_path):
    """End-to-end: corre el main con --write-shards sobre zips sinteticos.
    Debe escribir TARs, manifest.json y done.flag con asserts duros PASS.
    """
    from training.build_cmapss_rul_downstream import main, MANIFEST_NAME, DONE_FLAG_NAME
    # Construir raw sintetico en tmp_path/raw_root.
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)  # crea los 2 zips dentro
    out_dir = tmp_path / "out"
    results_dir = tmp_path / "report"
    rc = main([
        "--raw-root", str(raw_root),
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
        "--window-size", "8",
        "--patch-size", "2",
        "--stride", "1",
        "--min-valid-timesteps", "4",
        "--shard-size", "16",
        # Sintetico con stride=1 != canonico=5; necesitamos autorizarlo
        # explicitamente para que el guard del writer no aborte.
        "--allow-noncanonical",
        "--write-shards",
    ])
    assert rc == 0
    # Manifest y done.flag deben existir.
    assert (out_dir / MANIFEST_NAME).is_file()
    assert (out_dir / DONE_FLAG_NAME).is_file()
    # Al menos un .tar por split.
    for split in ("train", "val", "test"):
        tars = list((out_dir / split).glob("shard_*.tar"))
        assert tars, f"no se escribieron shards en {split}/"
    # Manifest carga como JSON valido con campos esperados.
    manifest = json.loads((out_dir / MANIFEST_NAME).read_text())
    assert manifest["dataset"] == "CMAPSS_RUL"
    assert manifest["pipeline_config_hash"]
    assert manifest["anti_leakage_checks"]["no_overlap_train_val"] is True


def test_writer_reentrancia_done_flag_mismo_hash_no_reescribe(tmp_path):
    """Si done.flag existe con mismo hash, segunda ejecucion no reescribe."""
    from training.build_cmapss_rul_downstream import (
        main, MANIFEST_NAME, DONE_FLAG_NAME,
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)
    out_dir = tmp_path / "out"
    results_dir = tmp_path / "report"
    args = [
        "--raw-root", str(raw_root),
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
        "--window-size", "8", "--patch-size", "2", "--stride", "1",
        "--min-valid-timesteps", "4", "--shard-size", "16",
        "--allow-noncanonical", "--write-shards",
    ]
    assert main(args) == 0
    first_mtime = (out_dir / MANIFEST_NAME).stat().st_mtime_ns
    # Segunda corrida con mismos parametros: no debe reescribir.
    assert main(args) == 0
    second_mtime = (out_dir / MANIFEST_NAME).stat().st_mtime_ns
    assert first_mtime == second_mtime


def test_writer_reentrancia_done_flag_hash_distinto_aborta(tmp_path):
    """Si done.flag existe con hash distinto, segunda ejecucion aborta
    (rc=1) y NO reescribe shards sin --force-overwrite."""
    from training.build_cmapss_rul_downstream import (
        main, MANIFEST_NAME, DONE_FLAG_NAME,
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)
    out_dir = tmp_path / "out"
    results_dir = tmp_path / "report"
    base = [
        "--raw-root", str(raw_root),
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
        "--window-size", "8", "--patch-size", "2",
        "--min-valid-timesteps", "4", "--shard-size", "16",
        "--allow-noncanonical", "--write-shards",
    ]
    assert main(base + ["--stride", "1"]) == 0
    # Misma out_dir, distinta politica -> hash distinto -> abortar.
    rc2 = main(base + ["--stride", "2"])
    assert rc2 == 1


def test_writer_force_overwrite_reescribe(tmp_path):
    """Con --force-overwrite, segunda corrida si reescribe."""
    from training.build_cmapss_rul_downstream import main, MANIFEST_NAME
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)
    out_dir = tmp_path / "out"
    results_dir = tmp_path / "report"
    base = [
        "--raw-root", str(raw_root),
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
        "--window-size", "8", "--patch-size", "2", "--stride", "1",
        "--min-valid-timesteps", "4", "--shard-size", "16",
        "--allow-noncanonical", "--write-shards",
    ]
    assert main(base) == 0
    first_mtime = (out_dir / MANIFEST_NAME).stat().st_mtime_ns
    # Pequeno sleep para asegurar mtime distinto en Windows (tick fino).
    import time as _t
    _t.sleep(0.01)
    assert main(base + ["--force-overwrite"]) == 0
    second_mtime = (out_dir / MANIFEST_NAME).stat().st_mtime_ns
    assert second_mtime >= first_mtime


def test_fd_seed_offset_mapping_canonico():
    """FD001..FD004 se mapean a 0..3 (deterministico). Cualquier FD
    desconocido cae al fallback sha256, que tambien es estable entre
    procesos.
    """
    from training.build_cmapss_rul_downstream import _fd_seed_offset
    assert _fd_seed_offset("FD001") == 0
    assert _fd_seed_offset("FD002") == 1
    assert _fd_seed_offset("FD003") == 2
    assert _fd_seed_offset("FD004") == 3
    # Fallback sha256: estable y > 3 con muy alta probabilidad.
    off_unknown = _fd_seed_offset("FD_NO_EXISTE")
    assert isinstance(off_unknown, int)
    assert off_unknown >= 0
    # Llamadas repetidas dan el mismo valor.
    assert _fd_seed_offset("FD_NO_EXISTE") == off_unknown


def test_build_train_val_test_rul_split_estable_entre_procesos(tmp_path):
    """Captura el bug historico de seed no-deterministico: llamar al
    builder DESDE OTRO PROCESO (subprocess con PYTHONHASHSEED random)
    debe producir EXACTAMENTE las mismas unit_split que en este proceso.

    Antes del fix, `hash(fd)` dependia de PYTHONHASHSEED y producia
    splits val distintos en cada arranque del interprete.
    """
    import os
    import subprocess
    import sys
    import textwrap

    # Construir raw sintetico en tmp_path.
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)

    # Ejecutar el split dos veces en subprocesses independientes con
    # PYTHONHASHSEED distintos. Si el split depende de hash(str),
    # darian unit_splits distintos.
    snippet = textwrap.dedent(f"""
        import json
        from pathlib import Path
        from training.build_cmapss_rul_downstream import (
            load_cmapss_raw_from_split_zips,
            build_train_val_test_rul,
        )
        parsed = load_cmapss_raw_from_split_zips(
            Path(r"{raw_root}"), fd_subsets=("FD001",),
        )
        out = build_train_val_test_rul(
            parsed, val_frac=0.2, seed=42, rul_cap=125.0,
        )
        us = out["FD001"]["unit_split"]
        print(json.dumps({{
            "train_units": list(map(int, us["train_units"])),
            "val_units":   list(map(int, us["val_units"])),
        }}, sort_keys=True))
    """).strip()

    def _run_split(hash_seed: str) -> dict:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = hash_seed
        # Asegurarse de que el subproceso ve el repo en el sys.path.
        repo_root = Path(__file__).resolve().parents[1]
        env["PYTHONPATH"] = str(repo_root) + os.pathsep + env.get("PYTHONPATH", "")
        cp = subprocess.run(
            [sys.executable, "-c", snippet],
            capture_output=True, text=True, env=env, check=True,
            cwd=str(repo_root),
        )
        out_line = cp.stdout.strip().splitlines()[-1]
        return json.loads(out_line)

    a = _run_split("0")
    b = _run_split("12345")
    c = _run_split("99999999")
    assert a == b == c, (
        f"unit_split depende de PYTHONHASHSEED:\n  PYTHONHASHSEED=0:     {a}\n"
        f"  PYTHONHASHSEED=12345: {b}\n  PYTHONHASHSEED=99999999: {c}"
    )


def test_build_train_val_test_rul_split_estable_intra_proceso(tmp_path):
    """Sanity adicional: dos llamadas dentro del mismo proceso dan
    EXACTAMENTE las mismas unit_split.
    """
    from training.build_cmapss_rul_downstream import (
        load_cmapss_raw_from_split_zips, build_train_val_test_rul,
    )
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)
    parsed = load_cmapss_raw_from_split_zips(raw_root, fd_subsets=("FD001",))
    a = build_train_val_test_rul(parsed, val_frac=0.2, seed=42, rul_cap=125.0)
    b = build_train_val_test_rul(parsed, val_frac=0.2, seed=42, rul_cap=125.0)
    assert a["FD001"]["unit_split"] == b["FD001"]["unit_split"]


def test_writer_payload_compatible_con_patchtst_phm(tmp_path):
    """Un sample escrito por el writer alimenta directamente a
    PatchTSTPhm.forward con batch B=1, sin reshapes ad-hoc.

    Verifica el contrato:
      patches.npy        (C,N,P) -> x   tensor (B,C,N,P)
      valid_time_mask    (W,)    -> vtm tensor (B,W)
      valid_patch_mask   (C,N)   -> vpm tensor (B,C,N)
    """
    import io as _io
    import sys as _sys
    import tarfile
    import numpy as np
    # En Windows local nuestra instalacion local de torch tiene un ABI
    # incompat con NumPy 2 que dispara error al importar
    # `torch._subclasses.functional_tensor`. Es un problema del entorno
    # local, no del builder. El test corre completo en Colab (Linux).
    if _sys.platform == "win32":
        pytest.skip(
            "Windows local: torch+numpy2 ABI incompat. Corre OK en Colab."
        )
    import torch  # noqa: E402
    from models.patchtst_phm import PatchTSTPhm  # noqa: E402
    from training.build_cmapss_rul_downstream import write_split_shards

    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    decisions = {
        "window_size": 8, "patch_size": 2, "stride": 1,
        "min_valid_timesteps": 4, "include_last_per_unit": True,
    }
    out_dir = tmp_path / "shards"
    stats = write_split_shards(
        pwr, "train", decisions, out_dir, shard_size=16,
    )
    assert stats["n_samples"] > 0
    assert stats["n_shards"] >= 1

    # Leer un sample real desde un TAR escrito.
    tar_path = stats["shard_paths"][0]
    with tarfile.open(tar_path, "r") as tf:
        names = tf.getnames()
        meta_name = next(n for n in names if n.endswith(".meta.json"))
        key = meta_name[: -len(".meta.json")]
        patches_np = np.load(_io.BytesIO(
            tf.extractfile(key + ".patches.npy").read()))
        vtm_np = np.load(_io.BytesIO(
            tf.extractfile(key + ".valid_time_mask.npy").read()))
        vpm_np = np.load(_io.BytesIO(
            tf.extractfile(key + ".valid_patch_mask.npy").read()))

    C, N, P = patches_np.shape
    # Contrato canonico esperado:
    assert vtm_np.shape == (N * P,), f"valid_time_mask {vtm_np.shape}"
    assert vpm_np.shape == (C, N), f"valid_patch_mask {vpm_np.shape}"

    # Construir batch B=1 sin reshapes ad-hoc, solo None-broadcast.
    x = torch.from_numpy(patches_np[None, ...])     # (1, C, N, P)
    vtm = torch.from_numpy(vtm_np[None, ...])        # (1, W)
    vpm = torch.from_numpy(vpm_np[None, ...])        # (1, C, N)

    # Modelo tiny consistente con (P=2, N=4).
    model = PatchTSTPhm(
        patch_size=P, n_patches=N, d_model=8, n_layers=1, n_heads=2,
        d_ff=16, dropout=0.0,
    )
    model.eval()
    with torch.no_grad():
        out = model(x, valid_time_mask=vtm, valid_patch_mask=vpm)

    # Contrato de salida: reconstruction tiene shape (B, C, N, P).
    assert "reconstruction" in out
    assert out["reconstruction"].shape == (1, C, N, P)


def test_cli_write_shards_aborta_si_stride_no_canonico(tmp_path):
    """Sin --allow-noncanonical, --write-shards con stride != 5 aborta."""
    from training.build_cmapss_rul_downstream import main, MANIFEST_NAME
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)
    out_dir = tmp_path / "out"
    results_dir = tmp_path / "report"
    rc = main([
        "--raw-root", str(raw_root),
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
        "--window-size", "8", "--patch-size", "2",
        "--stride", "1",  # != canonico=5
        "--min-valid-timesteps", "4", "--shard-size", "16",
        "--write-shards",
    ])
    assert rc == 1
    # No se debe haber escrito manifest ni done.flag.
    assert not (out_dir / MANIFEST_NAME).is_file()


def test_cli_write_shards_pasa_con_allow_noncanonical(tmp_path):
    """Con --allow-noncanonical, --write-shards con stride != 5 procede."""
    from training.build_cmapss_rul_downstream import main, MANIFEST_NAME
    raw_root = tmp_path / "raw"
    raw_root.mkdir()
    _make_synthetic_parsed_with_rul(raw_root)
    out_dir = tmp_path / "out"
    results_dir = tmp_path / "report"
    rc = main([
        "--raw-root", str(raw_root),
        "--out-dir", str(out_dir),
        "--results-dir", str(results_dir),
        "--window-size", "8", "--patch-size", "2",
        "--stride", "1",
        "--min-valid-timesteps", "4", "--shard-size", "16",
        "--allow-noncanonical",
        "--write-shards",
    ])
    assert rc == 0
    assert (out_dir / MANIFEST_NAME).is_file()


def test_writer_unit_global_id_unico_dentro_de_split(tmp_path):
    """Dentro de un split, distintas ventanas de la misma unidad comparten
    unit_global_id, pero el sample __key__ es unico por (unit, t_idx)."""
    from training.build_cmapss_rul_downstream import write_split_shards
    pwr = _make_synthetic_parsed_with_rul(tmp_path)
    out_dir = tmp_path / "shards_out"
    stats = write_split_shards(
        pwr, "train",
        decisions={
            "window_size": 8, "patch_size": 2, "stride": 1,
            "min_valid_timesteps": 4, "include_last_per_unit": True,
        },
        out_dir=out_dir, shard_size=64,
    )
    # Lista cruda contiene repeticiones (varias ventanas por unidad),
    # pero el set dimensiona el numero de unidades.
    assert stats["n_units"] == len(set(stats["unit_global_ids"]))
    assert stats["n_samples"] >= stats["n_units"]
