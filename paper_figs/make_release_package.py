# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Assemble the final golgi-paper release package under
Fenics_tests/golgi_paper_package/:

  manuscript_plos/        self-contained PLOS .tex (+ figures/ with paths
                          rewritten), .bib, .bst, S1 table, compiled .pdf
  manuscript_softwarex/   SoftwareX companion .tex + figures + .pdf
  golgi_code/             the golgi platform source + figure-repro scripts
                          (no out/, results, caches)
  reproduction_bundles/   the fig4-8 hashed study bundles + checksums
  README.md               how the package fits together + how to reproduce

Re-runnable: wipes and rebuilds golgi_paper_package/ each run. Bundles are
COPIED (working deposit in paper_figs/out/study_bundles is left intact).

Usage:  python paper_figs/make_release_package.py
"""
from __future__ import annotations

import datetime as _dt
import json
import re
import shutil
import zipfile
from pathlib import Path

ROOT = Path(
    "/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/"
    "Fenics_tests"
)
REL = ROOT / "golgi_paper_package"
# Methods paper (the version being submitted to PLOS Comp Biol).
PLOS = ROOT / "PLOS_latex_template"
SWX = ROOT / "SoftwareX"
BUNDLES = ROOT / "paper_figs" / "out" / "study_bundles"

# Build cruft never copied.
_LATEX_CRUFT = {".aux", ".log", ".out", ".fls", ".fdb_latexmk", ".bbl",
                ".blg", ".synctex.gz", ".spl", ".toc"}
_CODE_IGNORE = shutil.ignore_patterns(
    "__pycache__", "*.pyc", "*.pyo", ".DS_Store", ".ipynb_checkpoints",
)

# Curated, reproducible paper_figs set: the canonical scripts behind the
# published figures + the studies, their shared helpers, the data/FEM
# pipeline builders, the panel renders, and the release tooling. Everything
# else in paper_figs/ (scratchpads, superseded figure versions, dead AP
# scripts, one-off geometry fixes, timing probes, exploratory trajectory
# figures) is intentionally left out. The source paper_figs/ is never
# modified — this is a copy allowlist.
KEEP_PAPERFIGS = {
    # shared output layout
    "io_paths.py",
    # final figure plotters (→ published figures)
    "fig3_setup.py",                 # Fig 2 modeling setup
    "fig3_render.py",                # Fig 3 renders
    "fig_validation_full.py",        # Fig 4 (assembles a-f)
    "fig_bucksot.py", "fig_nrv.py", "validate_fig.py",   # Fig 4 panels (c-f) + S8 foundations
    "comsol_validation_fig.py",      # Fig 4 a/b (COMSOL cross-check) + S15 detail
    "_m1_golgi_cylinder.py",         # golgi M1 monopole solve (feeds Fig 4a / S15a)
    "fig_species.py",                # Fig 5 (swine) + Fig 6 (human)
    "fig5_population.py", "fig06_selectivity.py",         # Fig 5/6 helpers
    "rabbit_selectivity_fig.py",     # Fig 7
    "new_human_selectivity_fig.py",  # Fig 8
    "fig_supp_cohort.py", "cohort_table.py", "gen_s1_table.py",  # supp + S1
    "reseed_target_fascicle.py",
    # validation studies + digitized refs + setup renders (Fig 4 + S foundations)
    "validate_dogvns_native.py", "render_dogvns.py",
    "validate_nrv_fem.py", "validate_nrv_recruit.py", "validate_nrv.py",
    "save_nrv_ref.py", "render_nrv.py",
    "validate_bucksot.py", "save_bucksot_ref.py", "render_bucksot_setup.py",
    "validate_fem_analytic.py", "validate_fiber.py", "validate_mrg.py",
    # swine + human cervical vagus pipeline (Fig 5/6)
    "fig5_thresholds.py", "human_bundle_mesh.py", "human_bundle_fem.py",
    "render_popnerve.py",
    # rabbit branching pipeline (Fig 7)
    "rabbit_prep.py", "rabbit_pipeline.py", "rabbit_fem.py",
    "rabbit_sweep_fem.py", "rabbit_pop_sweep.py",
    "rabbit_tripole_sweep_build.py", "rabbit_xsec_contours.py",
    "rabbit_repop_resweep.py", "render_rabbit.py", "render_rabbit_sweep.py",
    "run_rabbit_tripole_sweep_thr.sh",
    # human SCB branching pipeline (Fig 8)
    "new_human3d_prep.py", "new_human3d_traj.py", "new_human_mesh.py",
    "new_human_fem.py", "new_human_sweep.py",
    "new_human_tripole_sweep_build.py", "new_human_tripole_analyze.py",
    "new_human_steer_opt.py", "new_human_xsec_contours.py",
    "new_human_render.py", "render_new_human_sweep.py",
    "run_tripole_sweep_thr.sh",
    # extra polished figures (electrode gallery, pulse types)
    "render_electrodes.py", "pulse_figure.py",
    # release tooling
    "make_study_bundles.py", "make_release_package.py",
}


def _fresh(d: Path) -> None:
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True, exist_ok=True)


def _copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


# --- 1. PLOS manuscript: self-contained (figures copied, paths rewritten) ---
def build_manuscript_plos() -> int:
    out = REL / "manuscript_plos"
    figs = out / "figures"
    _fresh(out)
    figs.mkdir()
    tex = (PLOS / "golgi_manuscript.tex").read_text()
    refs = re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]*)\}", tex)
    copied = {}
    for ref in refs:
        src = (PLOS / ref).resolve()
        base = Path(ref).name
        if not src.is_file():
            print(f"  ! missing figure: {ref}")
            continue
        # basename collision guard (none expected here)
        if base in copied and copied[base] != src:
            base = f"{Path(ref).parent.name}_{base}"
        copied[base] = src
        _copy(src, figs / base)
    # rewrite every includegraphics path to figures/<base>
    def _rw(m):
        opts = m.group(1) or ""
        ref = m.group(2)
        return f"\\includegraphics{opts}{{figures/{Path(ref).name}}}"
    tex2 = re.sub(r"\\includegraphics(\[[^\]]*\])?\{([^}]*)\}", _rw, tex)
    (out / "golgi_manuscript.tex").write_text(tex2)
    # support files
    for name in ("golgi.bib", "plos2025.bst", "s1_table_body.tex",
                 "golgi_manuscript.pdf"):
        p = PLOS / name
        if p.is_file():
            _copy(p, out / name)
    return len(copied)


# --- 2. SoftwareX companion ---
def build_manuscript_softwarex() -> None:
    out = REL / "manuscript_softwarex"
    _fresh(out)
    keep = ("golgi_softwarex.tex", "golgi_softwarex.pdf",
            "golgi_architecture.png", "golgi_bundle.png",
            "gui_overviews.png", "make_softwarex_figs.py")
    for name in keep:
        p = SWX / name
        if p.is_file():
            _copy(p, out / name)


# --- 3. golgi platform code + figure-repro scripts ---
def _check_paperfigs_closure(pf_out: Path, src: Path) -> list[str]:
    """Warn if any copied paper_figs script imports a *local* paper_figs
    module that wasn't copied (would break the script in the release)."""
    src_mods = {p.stem for p in src.glob("*.py")}
    copied = {p.stem for p in pf_out.glob("*.py")}
    missing = set()
    pat = re.compile(r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)",
                     re.MULTILINE)
    for p in pf_out.glob("*.py"):
        for mod in pat.findall(p.read_text()):
            if mod in src_mods and mod not in copied:
                missing.add(f"{p.name} → {mod}")
    return sorted(missing)


def build_golgi_code() -> list[str]:
    out = REL / "golgi_code"
    _fresh(out)
    # the package (core only — already free of scratch/notebooks)
    shutil.copytree(ROOT / "golgi", out / "golgi", ignore=_CODE_IGNORE)
    # top-level entry + project files (incl. license/readme for the public repo)
    for name in ("golgi.py", "cuff_designer.py", "pyproject.toml",
                 "requirements-frozen.txt", "FEATURES.md",
                 "LICENSE", "THIRD_PARTY_LICENSES.md", "README.md"):
        p = ROOT / name
        if p.is_file():
            _copy(p, out / name)
    # tests kept (small headless-API + cable suite); docs/examples/scripts
    # dropped for the public release.
    s = ROOT / "tests"
    if s.is_dir():
        shutil.copytree(s, out / "tests", ignore=_CODE_IGNORE)
    # paper_figs: only the curated reproducible set (NEVER the multi-GB out/)
    pf_out = out / "paper_figs"
    pf_out.mkdir()
    src = ROOT / "paper_figs"
    for name in sorted(KEEP_PAPERFIGS):
        p = src / name
        if p.is_file():
            _copy(p, pf_out / name)
        else:
            print(f"  ! KEEP_PAPERFIGS missing from source: {name}")
    return _check_paperfigs_closure(pf_out, src)


# --- 4. reproduction bundles ---
def build_bundles() -> list[str]:
    out = REL / "reproduction_bundles"
    _fresh(out)
    names = []
    if BUNDLES.is_dir():
        for p in sorted(BUNDLES.iterdir()):
            if p.is_file() and (
                p.suffix == ".zip" or p.name in (
                    "README.md", "CHECKSUMS.sha256", "BUNDLES.json")
            ):
                _copy(p, out / p.name)
                names.append(p.name)
    # revision-control analyses archive (convergence, matched perineurium,
    # extrusion-vs-3D, reconstructed-nerve galleries): a separate hashed,
    # self-contained deposit (scripts + data + figures + CHECKSUMS).
    rc = ROOT / "paper_figs" / "out" / "revision_controls.zip"
    if rc.is_file():
        _copy(rc, out / rc.name)
        names.append(rc.name)
    # fig4b (NRV LIFE) — idealized synthetic study whose mesh was not
    # retained and is not re-meshable on the current pipeline (the
    # multi-domain mesher ignores the local sizing field for this thin-
    # cylinder-in-muscle geometry → runaway / stall). Its FIGURE is fully
    # reproducible from cached results, shipped here in lieu of a study bundle.
    data = ROOT / "paper_figs" / "out" / "data"
    rend = ROOT / "paper_figs" / "out" / "renders"
    f4b = out / "fig04b_nrv_life_figure_data"
    srcs = [data / "validate_nrv.json", data / "validate_nrv_thr.npz",
            data / "nrv_reference.json", rend / "nrv_setup.png"]
    if any(s.is_file() for s in srcs):
        f4b.mkdir(parents=True, exist_ok=True)
        for s in srcs:
            if s.is_file():
                _copy(s, f4b / s.name)
        (f4b / "README.md").write_text(
            "# Fig 4b — NRV LIFE validation (figure data)\n\n"
            "The NRV LIFE benchmark is an idealized monofascicular cylinder "
            "with an intrafascicular electrode. Its golgi study mesh was not "
            "retained and is not re-meshable on the current pipeline (the "
            "multi-domain mesher does not honor the local sizing field for "
            "this geometry), so — unlike the other figures — a "
            "self-contained study bundle is not provided. The figure is fully "
            "reproducible from these cached results via "
            "`paper_figs/fig_nrv.py`:\n\n"
            "- `validate_nrv.json` — golgi recruitment + thresholds "
            "(160 fibers, 20/50 µs; strength–duration rate "
            "ratio 2.0)\n"
            "- `validate_nrv_thr.npz` — per-fiber threshold matrix\n"
            "- `nrv_reference.json` — digitized in-vivo (Nannini & Horch) "
            "and NRV (Couppey et al.) references\n"
            "- `nrv_setup.png` — setup render\n")
    return [n for n in names if n.endswith(".zip")]


# --- 5. top-level README ---
def write_readme(n_figs: int, bundle_names: list[str]) -> None:
    bl = "\n".join(f"  - {n}" for n in bundle_names) or "  (none found)"
    (REL / "README.md").write_text(f"""# golgi — paper release package

Everything needed to read, rebuild, and reproduce the golgi peripheral-nerve
stimulation modeling paper(s).

## Layout

- `manuscript_plos/` — the PLOS Computational Biology manuscript. Self-contained
  LaTeX: `golgi_manuscript.tex` (+ `figures/`, {n_figs} figures with paths
  rewritten to `figures/`), `golgi.bib`, `plos2025.bst`, `s1_table_body.tex`,
  and the compiled `golgi_manuscript.pdf`. Build with `latexmk -pdf
  golgi_manuscript.tex` (PLOS document class from TeX Live).
- `manuscript_softwarex/` — the SoftwareX companion paper (GUI + headless API):
  `golgi_softwarex.tex`, its figures, the compiled PDF, and the figure script.
- `golgi_code/` — the golgi platform source (`golgi/` package + `golgi.py`
  CLI/app entry, `pyproject.toml`, `requirements-frozen.txt`), the docs/tests/
  examples, and `paper_figs/` (the scripts that build every figure and study;
  the multi-GB working outputs under `paper_figs/out/` are intentionally
  excluded).
- `reproduction_bundles/` — integrity-hashed, self-contained golgi **study
  bundles** for Figs 4–8 (one `*.golgi.zip` each), plus `CHECKSUMS.sha256`,
  `BUNDLES.json`, and import/verify instructions in its own `README.md`.

## Reproduce a figure's simulation

```bash
pip install -e golgi_code            # or: pip install -r golgi_code/requirements-frozen.txt
python - <<'PY'
import golgi
proj = golgi.Study.import_bundle(
    "reproduction_bundles/fig07_rabbit_branching.golgi.zip", "/tmp/golgi_repro")
s = golgi.Study.open(proj)           # reopen geometry + mesh + FEM + fibers
PY
```

Verify a bundle byte-for-byte:

```bash
python -c "from golgi.projects.replay import replay_study as r; print(r('reproduction_bundles/fig07_rabbit_branching.golgi.zip').ok)"
# or:  shasum -a 256 -c reproduction_bundles/CHECKSUMS.sha256
```

## Bundles
{bl}
""")


def main() -> None:
    _fresh(REL)
    print("[1/4] PLOS manuscript …")
    nf = build_manuscript_plos()
    print(f"      {nf} figures copied, paths rewritten")
    print("[2/4] SoftwareX companion …")
    build_manuscript_softwarex()
    print("[3/4] golgi code …")
    missing = build_golgi_code()
    if missing:
        print("  ! closure WARNING — copied scripts import non-copied "
              "local modules (add to KEEP_PAPERFIGS):")
        for m in missing:
            print(f"      {m}")
    print("[4/4] reproduction bundles …")
    names = build_bundles()
    write_readme(nf, names)

    # report sizes
    def _sz(p: Path) -> float:
        return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6
    print("\n=== PACKAGE ===", REL)
    for d in sorted(REL.iterdir()):
        if d.is_dir():
            print(f"  {_sz(d):8.1f} MB  {d.name}/")
    print(f"  TOTAL {_sz(REL):.1f} MB · {len(names)} bundles")


if __name__ == "__main__":
    main()
