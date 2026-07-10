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
DOI = "10.5281/zenodo.21300037"


def _dataset_present() -> bool:
    inter, data = _OUT / "_intermediate", _OUT / "data"
    return inter.is_dir() and data.is_dir() and any(inter.glob("*")) and any(data.glob("*.npz"))


def _dataset_help() -> str:
    return (
        "\nThe paper's composite figures need the full working dataset in\n"
        f"  {_OUT}/_intermediate  and  {_OUT}/data\n"
        "(raw meshes, FEM fields, fiber sweeps, renders). This dataset is the\n"
        "authors' regeneration tree and is not part of the public repo.\n\n"
        "For third-party reproduction use the Zenodo *study bundles* instead —\n"
        f"  https://doi.org/{DOI}\n"
        "  python paper_figs/fetch_bundles.py        # download the bundles\n"
        "  golgi replay <bundle.golgi.zip>           # verify byte-for-byte\n"
        "  golgi figure <bundle.golgi.zip>           # render quick-look panels\n\n"
        "If you DO have the dataset elsewhere, point ROOT at it:\n"
        "  GOLGI_PAPER_ROOT=/path/to/tree python paper_figs/make_figures.py\n"
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
