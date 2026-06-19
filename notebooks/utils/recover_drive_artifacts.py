"""recover_drive_artifacts.py

Recupera artefactos primarios de Drive al repo para cerrar la asimetria
de trazabilidad entre el SSL central y el resto del proyecto.

Idempotente y reentrante. NO commitea, solo copia ficheros y genera el
README de results/pretraining/. Imprime una tabla final de qué se copio,
qué se salto (mismo contenido) y qué falto (no estaba en Drive).

Ejecutar en Colab tras mount Drive y cd al repo:

    !python notebooks/utils/recover_drive_artifacts.py

Opciones (todas con default sensato):
    --drive-root <path>    Default: /content/drive/MyDrive/fm_fl_phmd
    --repo-root  <path>    Default: cwd
    --max-log-mb <float>   Default: 5.0. Cap de tamano para Bloque F.
    --skip-block <letra>   Repetible. Para excluir A/B/C/D/E/F si interesa.
    --dry-run              No copia nada, solo imprime el plan.

Bloques:
    A) SSL central        : 4 stages (smoke, coverage, pilot, full)
                            + posthoc del full + README generado.
    B) SSL federated      : dry_run_report / config / metrics_round_summary
                            que faltaban en algunas subcarpetas.
    C) Downstream clasif  : metrics.jsonl por (dataset, modo) para los 4 TT.
    D) Downstream CMAPSS  : config.yaml + metrics.jsonl por los 3 modos.
    E) Downstream FL      : config + label_mapping + metrics para FedAvg/FedProx
                            vs central en CWRU/HSG18.
    F) Logs textuales     : audit.log, harmonization full log, write_shards log
                            con cap de tamano.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Optional


# ============================================================
# Configuracion: mapeo Drive -> repo
# ============================================================

# Bloque A: SSL central. Cada tupla es (drive_subdir, repo_subdir).
# El drive_subdir es relativo a logs/pretraining/.
# El repo_subdir es relativo a results/pretraining/.
BLOCK_A_RUNS = [
    ("ssl_central_full_patchtst_phm", "ssl_central_full"),
    ("ssl_central_pilot_patchtst_phm", "ssl_central_pilot"),
    ("ssl_central_coverage_patchtst_phm", "ssl_central_coverage"),
    ("ssl_smoke_patchtst_phm", "ssl_central_smoke"),
]
BLOCK_A_COMMON_FILES = ["run_info.json", "config.yaml", "sampling_plan.csv"]
BLOCK_A_EXTRA_FULL = ["posthoc_analysis.json", "posthoc_analysis.md"]

# Bloque B: SSL federated. Lista de (drive_rel, repo_rel) explicita
# porque los emparejamientos son irregulares.
BLOCK_B_FILES = [
    (
        "logs/pretraining_federated/ssl_federated_smoke_patchtst_phm/dry_run_report.json",
        "results/pretraining_federated/ssl_federated_smoke_v0_2/dry_run_report.json",
    ),
    (
        "logs/pretraining_federated/ssl_federated_smoke_fedprox_mu0_01_patchtst_phm/metrics_round_summary.json",
        "results/pretraining_federated/ssl_federated_smoke_fedprox_mu0_01/metrics_round_summary.json",
    ),
    (
        "logs/pretraining_federated/ssl_federated_pilot_patchtst_phm/config.yaml",
        "results/pretraining_federated/ssl_federated_pilot/config.yaml",
    ),
    (
        "logs/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01_patchtst_phm/config.yaml",
        "results/pretraining_federated/ssl_federated_pilot_fedprox_mu0_01/config.yaml",
    ),
]

# Bloque C: Downstream classification metrics.jsonl
BLOCK_C_DATASETS = {
    "cwru": ["from_scratch", "linear_probing", "full_finetuning", "full_finetuning_lr1e-5"],
    "hsg18": ["from_scratch", "linear_probing", "full_finetuning"],
    "pbcp16": ["from_scratch", "linear_probing", "full_finetuning"],
    "phm18": ["from_scratch", "linear_probing", "full_finetuning"],
}

# Bloque D: Downstream CMAPSS RUL. Drive usa prefijo distinto al repo.
BLOCK_D_MODES = {
    "downstream_cmapss_rul_from_scratch": "from_scratch",
    "downstream_cmapss_rul_linear_probing": "linear_probing",
    "downstream_cmapss_rul_full_finetuning_lr1e-5": "full_finetuning_lr1e-5",
}
BLOCK_D_FILES = ["config.yaml", "metrics.jsonl"]

# Bloque E: Downstream federated. (drive_root, repo_root) -> {dataset: [modes]}
BLOCK_E_RUNS = [
    (
        "downstream_federated_pilot",
        "fl_pilot_vs_central",
        {
            "cwru": ["linear_probing", "full_finetuning_lr1e-5"],
            "hsg18": ["linear_probing", "full_finetuning_lr1e-5", "full_finetuning_lr1e-4"],
        },
    ),
    (
        "downstream_federated_pilot_fedprox_mu0_01",
        "fl_fedprox_pilot_vs_central",
        {
            "cwru": ["linear_probing", "full_finetuning_lr1e-5"],
            "hsg18": ["linear_probing", "full_finetuning_lr1e-5"],
        },
    ),
]
BLOCK_E_FILES = ["config.yaml", "label_mapping.json", "metrics.jsonl"]

# Bloque F: logs textuales historicos (con cap de tamano).
BLOCK_F_FILES = [
    ("logs/audit.log", "results/audit/run_log.log"),
    (
        "logs/harmonization_full/full_20260521T141247Z.log",
        "results/processed_run_logs/full_20260521T141247Z.log",
    ),
    (
        "logs/cmapss_rul_write_shards.log",
        "results/downstream/cmapss_rul_decision/write_shards.log",
    ),
]

# Sanity checks por fichero copiado.
SANITY = {
    "results/pretraining/ssl_central_full/run_info.json": {
        "coverage_pass": True,
        "config_hash": "9ed84508a6820265",
        "param_count": 801808,
    },
    "results/pretraining/ssl_central_pilot/run_info.json": {
        "pilot_pass": True,
        "config_hash": "e4970c173c9dc244",
        "param_count": 801808,
    },
    "results/pretraining/ssl_central_coverage/run_info.json": {
        "coverage_pass": True,
        "config_hash": "e5cfd3b0684c7918",
        "param_count": 801808,
    },
    "results/pretraining/ssl_central_smoke/run_info.json": {
        "smoke_pass": True,
        "config_hash": "46628aedb05becd6",
        "param_count": 104336,
    },
}


# ============================================================
# Utilidades de IO
# ============================================================

def sha256_of(path: Path) -> str:
    """Hash SHA-256 hex de un fichero (chunked)."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def copy_file(src: Path, dst: Path, dry_run: bool, max_bytes: Optional[int] = None) -> tuple[str, int]:
    """Copia src -> dst. Reentrante.

    Devuelve (status, size_bytes) donde status es:
      - "copied"        : copia efectuada en este run
      - "identical"     : destino existia con mismo SHA-256, no se hace nada
      - "missing_src"   : el fichero no existe en Drive
      - "too_large"     : sobrepasa max_bytes y se omite
      - "dry"           : dry-run, se habria copiado
    """
    if not src.is_file():
        return ("missing_src", 0)
    size = src.stat().st_size
    if max_bytes is not None and size > max_bytes:
        return ("too_large", size)
    if dst.is_file() and sha256_of(src) == sha256_of(dst):
        return ("identical", size)
    if dry_run:
        return ("dry", size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return ("copied", size)


# ============================================================
# Construccion del plan
# ============================================================

def build_plan(
    drive_root: Path,
    repo_root: Path,
    skip_blocks: set[str],
    max_log_bytes: int,
) -> list[dict]:
    """Devuelve una lista de items {block, src, dst, max_bytes}.

    Los items mantienen el orden A -> F. Los bloques en skip_blocks se omiten.
    """
    plan: list[dict] = []

    # ---------- Bloque A: SSL central ----------
    if "A" not in skip_blocks:
        for drive_sub, repo_sub in BLOCK_A_RUNS:
            drive_dir = drive_root / "logs" / "pretraining" / drive_sub
            repo_dir = repo_root / "results" / "pretraining" / repo_sub
            files = list(BLOCK_A_COMMON_FILES)
            if drive_sub == "ssl_central_full_patchtst_phm":
                files += BLOCK_A_EXTRA_FULL
            for fname in files:
                plan.append(
                    {
                        "block": "A",
                        "label": f"central:{repo_sub}/{fname}",
                        "src": drive_dir / fname,
                        "dst": repo_dir / fname,
                        "max_bytes": None,
                    }
                )

    # ---------- Bloque B: SSL federated ----------
    if "B" not in skip_blocks:
        for drive_rel, repo_rel in BLOCK_B_FILES:
            plan.append(
                {
                    "block": "B",
                    "label": f"federated:{Path(repo_rel).parent.name}/{Path(repo_rel).name}",
                    "src": drive_root / drive_rel,
                    "dst": repo_root / repo_rel,
                    "max_bytes": None,
                }
            )

    # ---------- Bloque C: Downstream classification metrics ----------
    if "C" not in skip_blocks:
        for ds, modes in BLOCK_C_DATASETS.items():
            for mode in modes:
                src = drive_root / "logs" / "downstream" / ds / mode / "metrics.jsonl"
                dst = repo_root / "results" / "downstream" / ds / mode / "metrics.jsonl"
                plan.append(
                    {
                        "block": "C",
                        "label": f"downstream:{ds}/{mode}/metrics.jsonl",
                        "src": src,
                        "dst": dst,
                        "max_bytes": None,
                    }
                )

    # ---------- Bloque D: Downstream CMAPSS RUL ----------
    if "D" not in skip_blocks:
        for drive_sub, repo_sub in BLOCK_D_MODES.items():
            for fname in BLOCK_D_FILES:
                src = drive_root / "logs" / "downstream" / "cmapss_rul" / drive_sub / fname
                dst = repo_root / "results" / "downstream" / "cmapss_rul" / repo_sub / fname
                plan.append(
                    {
                        "block": "D",
                        "label": f"cmapss_rul:{repo_sub}/{fname}",
                        "src": src,
                        "dst": dst,
                        "max_bytes": None,
                    }
                )

    # ---------- Bloque E: Downstream federated ----------
    # Drive usa subdirs con run_name completo bajo el dataset, p.ej.:
    #   logs/downstream_federated_pilot/cwru/
    #     downstream_cwru_fedavg_pilot_linear_probing/{run_info.json, ...}
    #     downstream_cwru_fedavg_pilot_full_finetuning_lr1e-5/{...}
    # Detectamos el subdir cuyo nombre termina en "_{mode}" para ser
    # robustos al prefijo del run (fedavg/fedprox/etc).
    if "E" not in skip_blocks:
        for drive_root_sub, repo_root_sub, by_ds in BLOCK_E_RUNS:
            for ds, modes in by_ds.items():
                parent = drive_root / "logs" / drive_root_sub / ds
                # Cache de subdirs disponibles para no leer Drive N veces.
                available_subdirs: list[Path] = []
                if parent.is_dir():
                    available_subdirs = [p for p in parent.iterdir() if p.is_dir()]
                for mode in modes:
                    # Match: subdir cuyo nombre termina en "_<mode>" o es exactamente "<mode>".
                    found: Optional[Path] = None
                    for sub in available_subdirs:
                        if sub.name == mode or sub.name.endswith(f"_{mode}"):
                            found = sub
                            break
                    for fname in BLOCK_E_FILES:
                        # Si no encontramos subdir, dejamos el path teorico para
                        # que se reporte como missing_src de forma clara.
                        src = (found if found is not None else (parent / mode)) / fname
                        dst = (
                            repo_root
                            / "results"
                            / "downstream"
                            / repo_root_sub
                            / ds
                            / mode
                            / fname
                        )
                        plan.append(
                            {
                                "block": "E",
                                "label": f"fl:{repo_root_sub}/{ds}/{mode}/{fname}",
                                "src": src,
                                "dst": dst,
                                "max_bytes": None,
                            }
                        )

    # ---------- Bloque F: logs textuales (con cap) ----------
    if "F" not in skip_blocks:
        for drive_rel, repo_rel in BLOCK_F_FILES:
            plan.append(
                {
                    "block": "F",
                    "label": f"log:{Path(repo_rel).name}",
                    "src": drive_root / drive_rel,
                    "dst": repo_root / repo_rel,
                    "max_bytes": max_log_bytes,
                }
            )

    return plan


# ============================================================
# Ejecucion del plan
# ============================================================

def run_plan(plan: list[dict], dry_run: bool) -> list[dict]:
    """Ejecuta el plan y devuelve la lista de resultados con status y size."""
    results: list[dict] = []
    for item in plan:
        status, size = copy_file(
            src=item["src"],
            dst=item["dst"],
            dry_run=dry_run,
            max_bytes=item["max_bytes"],
        )
        results.append(
            {
                "block": item["block"],
                "label": item["label"],
                "status": status,
                "size_kb": round(size / 1024, 1),
            }
        )
    return results


# ============================================================
# Sanity checks sobre fichados copiados
# ============================================================

def run_sanity_checks(repo_root: Path) -> list[str]:
    """Valida campos clave de los run_info.json copiados. Devuelve lista de errores."""
    errors: list[str] = []
    for rel, expected in SANITY.items():
        path = repo_root / rel
        if not path.is_file():
            # Si no se copio, no se valida (probablemente faltaba en Drive).
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            errors.append(f"[SANITY] no se pudo leer {rel}: {e}")
            continue
        for key, exp in expected.items():
            got = data.get(key)
            if got != exp:
                errors.append(f"[SANITY] {rel}: {key}={got!r} != esperado {exp!r}")
    return errors


# ============================================================
# Generacion del README de results/pretraining/
# ============================================================

def _fmt_int(x) -> str:
    if isinstance(x, (int, float)):
        return f"{int(x):,}".replace(",", " ")
    return str(x)


def _fmt_pct(x) -> str:
    if isinstance(x, (int, float)):
        return f"{x*100:.2f} %"
    return str(x)


def _load_run_info(repo_root: Path, repo_sub: str) -> Optional[dict]:
    path = repo_root / "results" / "pretraining" / repo_sub / "run_info.json"
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_central_readme(repo_root: Path) -> str:
    """Construye el README dinamicamente desde los run_info.json copiados."""
    lines: list[str] = []
    lines.append("# Pretraining central SSL — resultados")
    lines.append("")
    lines.append("> Resumen agregado del bloque central SSL (PatchTSTPhm,")
    lines.append("> channel-independent, masked patch prediction). El analisis")
    lines.append("> interno y deliberacion estan en")
    lines.append("> `docs/decisions/pending_downstream_and_sampling.md` (gitignored).")
    lines.append("")
    lines.append("Este bloque cubre las 4 etapas del SSL central, en orden:")
    lines.append("")
    lines.append("1. **ssl_central_smoke** — smoke de tubo y diagnostico padding parcial.")
    lines.append("2. **ssl_central_coverage** — verificacion de cobertura 36/36 PS con `sampling_strategy=round_robin`.")
    lines.append("3. **ssl_central_pilot** — piloto productivo `sampling_strategy=weighted` en 5 000 steps.")
    lines.append("4. **ssl_central_full** — pretraining real en 100 000 steps. Checkpoint usado downstream.")
    lines.append("")
    lines.append("Todos los stages comparten arquitectura (`PatchTSTPhm`), corpus")
    lines.append("(36 PRETRAIN_SOURCE) y politica de caps `capped_v23`")
    lines.append("(`cap_max_dataset_weight=0.10`, `cap_max_client_weight=0.25`,")
    lines.append("`min_client_presence=0.005`).")
    lines.append("")

    # Tabla resumen
    lines.append("## Resumen comparativo")
    lines.append("")
    lines.append("| stage | pass | config_hash | optimizer_steps | datasets_seen | clients_seen | param_count | amp_overflow |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|")

    stage_specs = [
        ("ssl_central_smoke", "smoke_pass"),
        ("ssl_central_coverage", "coverage_pass"),
        ("ssl_central_pilot", "pilot_pass"),
        ("ssl_central_full", "coverage_pass"),
    ]
    for repo_sub, pass_key in stage_specs:
        info = _load_run_info(repo_root, repo_sub)
        if info is None:
            lines.append(f"| `{repo_sub}` | — | — | — | — | — | — | — |")
            continue
        # datasets_seen / clients_seen pueden ser dict, list o int
        ds = info.get("datasets_seen", info.get("datasets_seen_count"))
        if isinstance(ds, dict):
            ds_n = len(ds)
        elif isinstance(ds, list):
            ds_n = len(ds)
        else:
            ds_n = ds
        cl = info.get("clients_seen")
        if isinstance(cl, dict):
            cl_n = len(cl)
        elif isinstance(cl, list):
            cl_n = len(cl)
        else:
            cl_n = cl
        amp = info.get(
            "amp_nonfinite_grad_steps",
            info.get("amp_nonfinite_grad_steps_total", info.get("amp_overflow_steps", 0)),
        )
        lines.append(
            "| `{name}` | **{p}** | `{ch}` | {ost} | {ds} | {cl} | {pc} | {amp} |".format(
                name=repo_sub,
                p=info.get(pass_key, "—"),
                ch=info.get("config_hash", "—"),
                ost=_fmt_int(info.get("optimizer_steps", info.get("optimizer_steps_total", "—"))),
                ds=_fmt_int(ds_n) if ds_n is not None else "—",
                cl=_fmt_int(cl_n) if cl_n is not None else "—",
                pc=_fmt_int(info.get("param_count", "—")),
                amp=_fmt_int(amp) if amp is not None else "0",
            )
        )
    lines.append("")

    # Detalle por stage
    for repo_sub, pass_key in stage_specs:
        info = _load_run_info(repo_root, repo_sub)
        if info is None:
            continue
        lines.append(f"## {repo_sub}")
        lines.append("")
        items = [
            ("run_name", info.get("run_name")),
            ("stage", info.get("stage")),
            ("config_hash", info.get("config_hash")),
            ("git_hash", info.get("git_hash")),
            (pass_key, info.get(pass_key)),
            (
                "optimizer_steps",
                _fmt_int(info.get("optimizer_steps", info.get("optimizer_steps_total"))),
            ),
            ("param_count", _fmt_int(info.get("param_count"))),
            ("elapsed_seconds", info.get("elapsed_seconds")),
            ("max_effective_bc", info.get("max_effective_bc")),
            (
                "amp_nonfinite_grad_steps",
                info.get(
                    "amp_nonfinite_grad_steps",
                    info.get("amp_nonfinite_grad_steps_total", info.get("amp_overflow_steps", 0)),
                ),
            ),
        ]
        lines.append("| campo | valor |")
        lines.append("|---|---|")
        for k, v in items:
            if v is None:
                continue
            lines.append(f"| `{k}` | `{v}` |")
        lines.append("")

    lines.append("## Trazabilidad con Drive")
    lines.append("")
    lines.append("Los artefactos pesados (`metrics.jsonl` por step, checkpoints `.pt`)")
    lines.append("viven solo en Drive bajo `MyDrive/fm_fl_phmd/logs/pretraining/<run_name>/`")
    lines.append("y `MyDrive/fm_fl_phmd/checkpoints/<run_name>/`. Este README y los")
    lines.append("`run_info.json` aqui versionados son el contrato citable.")
    lines.append("")
    lines.append("Generado automaticamente por `notebooks/utils/recover_drive_artifacts.py`.")
    lines.append("")
    return "\n".join(lines)


def write_central_readme(repo_root: Path, dry_run: bool) -> tuple[str, int]:
    content = build_central_readme(repo_root)
    dst = repo_root / "results" / "pretraining" / "README.md"
    size = len(content.encode("utf-8"))
    if dst.is_file() and dst.read_text(encoding="utf-8") == content:
        return ("identical", size)
    if dry_run:
        return ("dry", size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(content, encoding="utf-8")
    return ("copied", size)


# ============================================================
# Resumen + main
# ============================================================

def summarize(results: list[dict]) -> None:
    """Imprime tabla resumen agrupada por bloque."""
    by_block: dict[str, list[dict]] = {}
    for r in results:
        by_block.setdefault(r["block"], []).append(r)

    print()
    print("=" * 72)
    print("RESUMEN POR BLOQUE")
    print("=" * 72)
    for block in sorted(by_block.keys()):
        items = by_block[block]
        counts = {}
        total_kb = 0.0
        for r in items:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
            total_kb += r["size_kb"]
        statuses = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"  [{block}] {len(items):3d} items, {total_kb:8.1f} KB total. {statuses}")

    # Detalle de los problemas
    print()
    print("Items con status != copied/identical:")
    any_issue = False
    for r in results:
        if r["status"] in ("copied", "identical"):
            continue
        any_issue = True
        print(f"  [{r['block']}] {r['status']:12s} {r['label']}  ({r['size_kb']} KB)")
    if not any_issue:
        print("  (ninguno)")
    print()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", default="/content/drive/MyDrive/fm_fl_phmd")
    parser.add_argument("--repo-root", default=".")
    parser.add_argument("--max-log-mb", type=float, default=5.0)
    parser.add_argument("--skip-block", action="append", default=[], choices=list("ABCDEF"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    drive_root = Path(args.drive_root).resolve()
    repo_root = Path(args.repo_root).resolve()
    max_log_bytes = int(args.max_log_mb * 1024 * 1024)
    skip_blocks = set(args.skip_block)

    print(f"Drive root : {drive_root}")
    print(f"Repo  root : {repo_root}")
    print(f"Max log MB : {args.max_log_mb}")
    print(f"Skip blocks: {sorted(skip_blocks) or '[]'}")
    print(f"Dry-run    : {args.dry_run}")
    print()

    if not drive_root.is_dir():
        print(f"[ERROR] Drive root no existe: {drive_root}", file=sys.stderr)
        return 1
    # Guard de raiz del repo: pedimos .git/ porque CLAUDE.md esta gitignored
    # y no aparece en clones recientes (p.ej. Colab tras git pull).
    if not (repo_root / ".git").exists():
        print(
            f"[ERROR] repo-root no parece la raiz del repo (sin .git/): {repo_root}",
            file=sys.stderr,
        )
        return 1

    plan = build_plan(drive_root, repo_root, skip_blocks, max_log_bytes)
    print(f"Plan: {len(plan)} items.")
    print()

    results = run_plan(plan, dry_run=args.dry_run)
    summarize(results)

    # README (solo si el Bloque A no se salto, sino no tiene sentido)
    if "A" not in skip_blocks:
        readme_status, readme_size = write_central_readme(repo_root, dry_run=args.dry_run)
        print(
            f"README results/pretraining/README.md: {readme_status} ({readme_size/1024:.1f} KB)"
        )
        print()

    # Sanity checks solo si no es dry-run (en dry-run no hay ficheros nuevos)
    if not args.dry_run:
        errors = run_sanity_checks(repo_root)
        if errors:
            print("[SANITY] Errores detectados:")
            for e in errors:
                print(f"  - {e}")
            print()
            print("Resultado: SANITY FAIL. Revisa los run_info.json antes de commitear.")
            return 2
        else:
            print("[SANITY] OK: todos los run_info.json copiados pasan los checks.")
            print()

    # Comandos git sugeridos
    print("=" * 72)
    print("PROXIMOS PASOS (manual, no se ejecutan)")
    print("=" * 72)
    print()
    print("Inspecciona el diff:")
    print("    git status")
    print("    git diff --stat results/")
    print()
    print("Si todo cuadra, commitea sin push automatico:")
    print("    git add results/pretraining/ results/pretraining_federated/ \\")
    print("            results/downstream/ results/audit/ results/processed_run_logs/")
    print("    git commit -m 'chore(results): traer artefactos primarios de Drive (central + completos)'")
    print()
    print("Push solo si tu lo decides:")
    print("    git push origin main")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
