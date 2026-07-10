# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Reproduce a paper figure by running its generation script(s).

This is the thin driver over the per-figure scripts — the authoritative
figure→script map (matches the numbers in the published paper). Each
script reads the FULL intermediate dataset (`paper_figs/out/_intermediate`,
`out/data`) and writes the multi-panel figure to
`paper_figs/out/figures/{png,pdf,svg}/` via `io_paths.save_fig`. It needs
the source data present and a configured `ROOT` (see io_paths.py) — this
is NOT driven by a study bundle (a bundle carries one config's slice; the
`golgi figure` CLI renders that quick-look, not the composite figure).

Usage:
    python paper_figs/make_figures.py            # all figures
    python paper_figs/make_figures.py fig07      # one figure
    python paper_figs/make_figures.py fig05 fig08
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
# Where the working dataset lives — same resolution as io_paths.ROOT so the
# guard checks exactly where the figure scripts will read from.
_ROOT = Path(os.environ.get("GOLGI_PAPER_ROOT") or HERE.parent)
_OUT = _ROOT / "paper_figs" / "out"

# --- reproducibility guard ------------------------------------------------
# These composite figures are regenerated from the FULL working dataset
# (paper_figs/out/_intermediate + out/data): raw meshes, FEM fields, fiber
# sweeps and cel-shaded renders. That dataset is NOT the Zenodo deposit — the
# Zenodo record ships replay-verified *study bundles* (.golgi.zip), which drive
# `golgi replay` (integrity check) and `golgi figure` (quick-look panels), not
# this script. If the raw dataset is absent, fail early with directions rather
# than deep inside a figure script with a cryptic FileNotFoundError.
BUNDLE_DOI = "10.5281/zenodo.21000094"     # study bundles (concept DOI)
DATA_DOI = ""                              # working-dataset archive — set once deposited


def _dataset_present() -> bool:
    """The composite figures read three trees under ROOT: paper_figs/out/data,
    paper_figs/out/_intermediate, and results_golgi/duke_meshes. Require all
    three so a partial download fails here with directions, not mid-figure."""
    inter, data = _OUT / "_intermediate", _OUT / "data"
    duke = _ROOT / "results_golgi" / "duke_meshes"
    return (data.is_dir() and any(data.glob("*.npz"))
            and inter.is_dir() and any(inter.glob("*"))
            and duke.is_dir() and any(duke.glob("*")))


def _dataset_help() -> str:
    data_line = (f"  https://doi.org/{DATA_DOI}\n" if DATA_DOI else
                 "  (see the wiki: Reproducing the Paper — 'Regenerate the composite figures')\n")
    return (
        "\nThe composite figures are rebuilt from the full working dataset under\n"
        f"  {_OUT}/data,  {_OUT}/_intermediate,  and\n"
        f"  {_ROOT / 'results_golgi' / 'duke_meshes'}\n"
        "(~27 GB of raw meshes, FEM fields, fiber sweeps, renders).\n\n"
        "To regenerate them, download the working-dataset archive from Zenodo and\n"
        "extract it at the repo root (so paper_figs/out/... and results_golgi/...\n"
        "land in place), then re-run this script:\n"
        f"{data_line}"
        "  tar xzf golgi_paper_dataset.tar.gz          # run from the repo root\n"
        "  python paper_figs/make_figures.py\n"
        "Dataset kept elsewhere? point ROOT at the tree that CONTAINS paper_figs/out:\n"
        "  GOLGI_PAPER_ROOT=/path/to/tree python paper_figs/make_figures.py\n\n"
        "Only need to verify or quick-look a study (not rebuild the figures)? Use the\n"
        f"lighter *study bundles* (https://doi.org/{BUNDLE_DOI}) instead:\n"
        "  python paper_figs/fetch_bundles.py\n"
        "  golgi replay <bundle.golgi.zip>   |   golgi figure <bundle.golgi.zip> --out ./figs\n"
    )


# Figure number → the script(s) that produce it. fig05_06_species.py
# produces BOTH Fig 5 (swine) and Fig 6 (human) in one run, so fig05 and
# fig06 map to the same script.
FIGS: dict[str, list[str]] = {
    "fig02": ["fig02_setup.py", "fig02_render.py"],
    "fig04": ["fig04_validation.py"],
    "fig05": ["fig05_06_species.py"],
    "fig06": ["fig05_06_species.py"],
    "fig07": ["fig07_rabbit_selectivity.py"],
    "fig08": ["fig08_human_selectivity.py"],
}


def _scripts_for(want: list[str]) -> list[str]:
    if not want or want == ["all"]:
        want = list(FIGS)
    out: list[str] = []
    for w in want:
        key = w if w.startswith("fig") else f"fig{int(w):02d}"
        if key not in FIGS:
            raise SystemExit(
                f"unknown figure '{w}'; choose from "
                f"{sorted(FIGS)} or 'all'"
            )
        for s in FIGS[key]:
            if s not in out:                     # dedupe (5/6 share a script)
                out.append(s)
    return out


def main(argv: list[str]) -> int:
    scripts = _scripts_for(argv)
    if not _dataset_present():
        print("✗ paper dataset not found — cannot regenerate the composite "
              "figures.", file=sys.stderr)
        print(_dataset_help(), file=sys.stderr)
        return 2
    rc = 0
    for s in scripts:
        script = HERE / s
        if not script.is_file():
            print(f"  ✗ missing: {s}", file=sys.stderr)
            rc = 1
            continue
        print(f"── {s} ──", flush=True)
        r = subprocess.run([sys.executable, str(script)])
        if r.returncode != 0:
            print(f"  ✗ {s} exited {r.returncode}", file=sys.stderr)
            rc = 1
    print("\ndone — figures in paper_figs/out/figures/{png,pdf,svg}/",
          flush=True)
    return rc


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
