# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded render of the NRV LIFE validation setup: a monofascicular
cat-tibial-sized nerve (epineurium + endoneurium) with an intrafascicular LIFE
(silver wire + two gold active sites) running through the fascicle, plus a few
straight MRG fibers. Same comic/cel style as the dog-VNS / electrode-gallery /
swine / rabbit renders. Writes paper_figs/out/renders/nrv_setup.png.
"""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "paper_figs"))
from render_electrodes import _flat, _ink, _gold, SIL, INK, BODY, EPI  # noqa: E402
from render_components import STYLE  # noqa: E402

ENDO = STYLE["endo"][1]            # purple endoneurium
FIBER = "#d63a3a"                  # MRG fibers
OUT = ROOT / "paper_figs/out/renders/nrv_setup.png"

# schematic proportions (readable; the real nerve is 0.65 mm x 30 mm — far too thin)
EPI_R, ENDO_R, WIRE_R, ACT_R = 1.0, 0.78, 0.10, 0.15
EPI_Z, ENDO_Z, WIRE_Z = 6.0, 6.7, 7.4        # telescoped half-lengths (each layer protrudes)
ACT_LEN, ROW = 1.3, 2.0                       # two gold active sites at z = +/- ROW


def _cyl(r, zhalf, n=120):
    return pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=r, height=2 * zhalf,
                       resolution=n, capping=True)


def build(p):
    # epineurium (translucent rose) -> endoneurium (translucent purple) -> wire, telescoped
    epi = _cyl(EPI_R, EPI_Z); _flat(p, epi, EPI, 0.18); _ink(p, epi, lw=2.4)
    endo = _cyl(ENDO_R, ENDO_Z); _flat(p, endo, ENDO, 0.26); _ink(p, endo, lw=2.2)
    # a few straight MRG fibers in the fascicle (subtle — the LIFE is the focus)
    rng = np.random.default_rng(3)
    nfib = 6
    rr = ENDO_R * 0.8 * np.sqrt(rng.uniform(0, 1, nfib))
    th = rng.uniform(0, 2 * np.pi, nfib)
    for x, y in zip(rr * np.cos(th), rr * np.sin(th)):
        f = pv.Cylinder(center=(x, y, 0), direction=(0, 0, 1), radius=0.014,
                        height=2 * ENDO_Z * 0.98, resolution=14, capping=False)
        _flat(p, f, FIBER, 0.7, sil=False)
    # LIFE wire (silver) + two gold active sites
    wire = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=WIRE_R,
                       height=2 * WIRE_Z, resolution=48, capping=True)
    _flat(p, wire, BODY, 1.0); _ink(p, wire, lw=1.8)
    for z in (-ROW, ROW):
        c = pv.Cylinder(center=(0, 0, z), direction=(0, 0, 1), radius=ACT_R,
                        height=ACT_LEN, resolution=48, capping=True)
        _gold(p, c); _ink(p, c, lw=1.6)


def main():
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=(2200, 1300))
    p.background_color = "white"
    p.enable_depth_peeling(60, occlusion_ratio=0.0)
    p.remove_all_lights()
    for pos, inten in [((7, 11, 13), 0.62), ((-9, 4, 7), 0.34), ((0, -7, -11), 0.30),
                       ((-4, 9, -6), 0.26)]:
        lt = pv.Light(position=pos, focal_point=(0, 0, 0), color="#ffffff", intensity=inten)
        lt.positional = False
        p.add_light(lt)
    build(p)
    p.enable_anti_aliasing("ssaa")
    p.view_vector((0.45, 0.5, 0.74), viewup=(0, 1, 0))      # slight 3/4 view along the nerve
    p.reset_camera(); p.camera.zoom(1.5)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(OUT), window_size=(2200, 1300))
    p.close()
    from PIL import Image
    im = Image.open(OUT).convert("RGB"); a = np.asarray(im)
    ys, xs = np.where((a < 248).any(2))
    if len(xs):
        pad = 18
        im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(im.width, int(xs.max()) + pad), min(im.height, int(ys.max()) + pad))).save(OUT)
    print(f"wrote {OUT.name}")


if __name__ == "__main__":
    main()
