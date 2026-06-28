# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Fig 6 / cohort table: parse results_golgi/duke_meshes/BATCH_SUMMARY.txt
into a clean per-sample CSV + summary stats (species, fascicles, mesh size,
fibers, perineurium). Demonstrates the platform ran a real multi-sample,
two-species cohort end-to-end through one pipeline.
"""
from __future__ import annotations
import re, csv, json
from pathlib import Path
import numpy as np

from io_paths import TABLES, DATA
ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
SRC = ROOT / "results_golgi/duke_meshes/BATCH_SUMMARY.txt"

pat = re.compile(
    r"^(?P<sample>\S+?):\s*OK\s+"
    r"(?P<nodes>[\d,]+)\s*nodes,\s*(?P<tets>[\d,]+)\s*tets,\s*"
    r"(?P<fasc>\d+)\s*fascicles,\s*(?P<contacts>\d+)\s*contacts,\s*"
    r"(?P<fibers>\d+)\s*fibers,.*?(?P<onmesh>\d+)%on-mesh.*?"
    r"r_max=(?P<rmax>[\d.]+)mm.*?peri_thk=(?P<peri>[\d.]+)")


def num(s): return int(s.replace(",", ""))


def main():
    rows, skipped = [], []
    for line in SRC.read_text().splitlines():
        if ": SKIP" in line:
            skipped.append(line.split(":")[0]); continue
        m = pat.search(line)
        if not m:
            continue
        g = m.groupdict()
        sample = g["sample"]
        species = "human" if sample.startswith("human") else "swine"
        rows.append(dict(
            sample=sample, species=species,
            nodes=num(g["nodes"]), tets=num(g["tets"]),
            fascicles=int(g["fasc"]), contacts=int(g["contacts"]),
            fibers=int(g["fibers"]), on_mesh_pct=int(g["onmesh"]),
            r_max_mm=float(g["rmax"]), peri_thk_um=float(g["peri"])))

    out_csv = TABLES / "cohort_table.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    def stats(sel):
        fa = [r["fascicles"] for r in sel]
        te = [r["tets"] for r in sel]
        pk = [r["peri_thk_um"] for r in sel]
        return dict(n=len(sel),
                    fascicles=f"{min(fa)}-{max(fa)} (median {int(np.median(fa))})",
                    tets=f"{min(te):,}-{max(te):,}",
                    total_fibers=sum(r["fibers"] for r in sel),
                    peri_thk_um=f"{min(pk):.1f}-{max(pk):.1f}",
                    mean_on_mesh=f"{np.mean([r['on_mesh_pct'] for r in sel]):.1f}%")

    swine = [r for r in rows if r["species"] == "swine"]
    human = [r for r in rows if r["species"] == "human"]
    summary = dict(total_samples=len(rows), skipped=skipped,
                   swine=stats(swine), human=stats(human), all=stats(rows))
    (DATA / "cohort_summary.json").write_text(
        json.dumps(summary, indent=2))

    print(f"parsed {len(rows)} samples ({len(swine)} swine, {len(human)} human), "
          f"{len(skipped)} skipped")
    print(f"  fascicles: {summary['all']['fascicles']}")
    print(f"  tets:      {summary['all']['tets']}")
    print(f"  total fibers simulated: {summary['all']['total_fibers']:,}")
    print(f"  perineurium thk (um): swine {summary['swine']['peri_thk_um']} | "
          f"human {summary['human']['peri_thk_um']}")
    print(f"  wrote {out_csv}")


if __name__ == "__main__":
    main()
