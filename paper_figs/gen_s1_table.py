# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Generate the S1 Table LaTeX longtable (per-sample cohort) from the parsed
cohort CSV. Writes PLOS_latex_template/s1_table_body.tex (a longtable fragment
\\input by the manuscript after the S1 Table caption).
"""
from __future__ import annotations
import csv
from pathlib import Path

ROOT = Path(__file__).parent.parent
CSV = ROOT / "paper_figs/out/tables/cohort_table.csv"
OUT = ROOT / "PLOS_latex_template/s1_table_body.tex"


def short(s):
    """Compact sample id (the Species column keeps the human/swine distinction):
    sub-10_sam-1 -> 10/1, human_sub-46_sam-2 -> 46/2."""
    s = s.replace("human_", "").replace("sub-", "").replace("sam-", "")
    return s.replace("_", "/")


def main():
    rows = list(csv.DictReader(CSV.open()))
    rows.sort(key=lambda r: (r["species"], r["sample"]))
    L = []
    L.append(r"{\small\setlength{\tabcolsep}{5pt}")          # fit text width
    L.append(r"\begin{longtable}{llrrrrrr}")
    L.append(r"\hline")
    L.append(r"Sample & Species & Fascicles & $r_{\max}$ (mm) & Tets & Fibers & "
             r"Perineurium ($\mu$m) & On-mesh \\")
    L.append(r"\hline")
    L.append(r"\endfirsthead")
    L.append(r"\hline")
    L.append(r"Sample & Species & Fascicles & $r_{\max}$ (mm) & Tets & Fibers & "
             r"Perineurium ($\mu$m) & On-mesh \\")
    L.append(r"\hline")
    L.append(r"\endhead")
    L.append(r"\hline")
    L.append(r"\endfoot")
    tot_fib = 0
    for r in rows:
        tot_fib += int(r["fibers"])
        L.append(
            f"{short(r['sample'])} & {r['species']} & {r['fascicles']} & "
            f"{float(r['r_max_mm']):.2f} & {int(r['tets']):,} & {r['fibers']} & "
            f"{float(r['peri_thk_um']):.1f} & {r['on_mesh_pct']}\\% \\\\")
    n_sw = sum(1 for r in rows if r["species"] == "swine")
    n_hu = sum(1 for r in rows if r["species"] == "human")
    L.append(r"\hline")
    L.append(f"\\textbf{{Total}} & {n_sw} swine, {n_hu} human & & & & "
             f"\\textbf{{{tot_fib:,}}} & & \\\\")
    L.append(r"\hline")
    L.append(r"\end{longtable}")
    L.append(r"}")
    OUT.write_text("\n".join(L) + "\n")
    print(f"wrote {OUT}: {len(rows)} samples ({n_sw} swine, {n_hu} human), "
          f"{tot_fib:,} fibers total")


if __name__ == "__main__":
    main()
