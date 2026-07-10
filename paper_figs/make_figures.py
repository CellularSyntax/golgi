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

import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

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
