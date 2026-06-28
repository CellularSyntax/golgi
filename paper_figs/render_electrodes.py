# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cel-shaded gallery of golgi's available electrode designs (same comic style as
the dog-VNS render). Geometry comes from golgi's own generators
(golgi.scene.electrode_patches) for the simple combobox cuffs and from golgi's DUKE
cuff_designer for the LivaNova helical coils.

Nerve = epineurium only (opaque for cuff types; translucent for the intrafascicular
LIFE/TIME so the electrode body shows). Silicone is a distinct blue so it pops off the
rose nerve; contacts are opaque gold. LIFE = smooth capped wire + gold ferrules; TIME =
a transverse ribbon (short axially, wide, poking out the sides) carrying the contacts."""
from __future__ import annotations
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import sys, json
from pathlib import Path
import numpy as np
import pyvista as pv

ROOT = Path("/Users/admin/Desktop/DATA/Uni/Postdoc/2026/Students/Yuting Jia/Fenics_tests")
sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "scripts"))
import golgi.scene.electrode_patches as ep  # noqa: E402
from render_components import STYLE, CONTACT_COLOR  # noqa: E402

OUTDIR = ROOT / "paper_figs/out/figures/png/electrodes"
SIL = dict(color="#121419", line_width=4.0)
INK = "#121419"
EPI = STYLE["epi"][1]                 # rose nerve
SILICONE = "#4f93cf"                  # distinct blue cuff (pops off the rose nerve)
GOLD = "#f1c62f"                      # bright contact gold (survives the silicone tint)
BODY = "#878f9c"                      # silver intrafascicular body
EPI_R, EPI_H = 1.0, 14.5
R_CI, R_CO, LCUFF = 1.2, 2.1, 10.0


def _flat(p, mesh, color, opacity, sil=True):
    p.add_mesh(mesh, color=color, opacity=opacity, smooth_shading=True, ambient=0.46,
               diffuse=0.66, specular=0.0, show_scalar_bar=False,
               silhouette=SIL if sil else None)


def _gold(p, mesh):
    p.add_mesh(mesh, color=GOLD, opacity=1.0, smooth_shading=True, ambient=0.62,
               diffuse=0.55, specular=0.08, show_scalar_bar=False, silhouette=SIL)


def _ink(p, mesh, lw=2.4, fa=18):
    e = mesh.extract_feature_edges(feature_angle=fa, boundary_edges=True,
                                   non_manifold_edges=False, feature_edges=True,
                                   manifold_edges=False)
    if e.n_cells:
        p.add_mesh(e, color=INK, line_width=lw, render_lines_as_tubes=True,
                   show_scalar_bar=False)


def _cyl(r, h, n=120):
    return pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=r, height=h,
                       resolution=n, capping=True)


def _ring(R, thk, z0, z1, n=200):
    d = pv.Disc(center=(0, 0, z0), inner=R, outer=R + thk, normal=(0, 0, 1), r_res=2, c_res=n)
    return d.extrude((0, 0, z1 - z0), capping=True)


def add_nerve(p, intra=False):
    # epineurium only: opaque rose for cuff types; translucent for intrafascicular
    nerve = _cyl(EPI_R, EPI_H)
    _flat(p, nerve, EPI, 0.24 if intra else 1.0)
    _ink(p, nerve, lw=2.4)            # contour the end caps (silhouette misses them)


# --------------------------------------------------------------------------- #
#  scene builders
# --------------------------------------------------------------------------- #
def build_cuff(kind, cfg):
    def build(p):
        add_nerve(p, intra=False)
        sil = _ring(R_CI, R_CO - R_CI, -LCUFF / 2, LCUFF / 2)
        _flat(p, sil, SILICONE, 0.28)
        _ink(p, sil, lw=2.2)
        for pad in ep.build_electrode_patches(LCUFF * 1e-3, R_CI * 1e-3, kind, cfg):
            pad = pad.copy(); pad.points = pad.points * 1000.0
            _gold(p, pad)
    return build


def build_life(cfg):
    def build(p):
        add_nerve(p, intra=True)
        R_w = float(cfg["life_diameter_um"]) * 0.5e-3
        nrows = int(cfg["life_n_rows"]); row_sep = float(cfg["life_row_sep_mm"])
        clen = float(cfg["life_contact_length_mm"])
        wire = pv.Cylinder(center=(0, 0, 0), direction=(0, 0, 1), radius=R_w * 0.5,
                           height=EPI_H * 0.92, resolution=56, capping=True)
        _flat(p, wire, BODY, 0.98)
        for z in (np.arange(nrows) - (nrows - 1) / 2.0) * row_sep:
            c = pv.Cylinder(center=(0, 0, z), direction=(0, 0, 1), radius=R_w,
                            height=clen, resolution=56, capping=True)
            _gold(p, c)
    return build


def build_time(cfg):
    def build(p):
        add_nerve(p, intra=True)
        Wt, Wa, th = 3.8, 0.5, 0.12         # transverse(long) x axial(short) x thin; pokes out both sides
        ribbon = pv.Box(bounds=(-Wt / 2, Wt / 2, -th / 2, th / 2, -Wa / 2, Wa / 2))
        _flat(p, ribbon, BODY, 0.98)
        _ink(p, ribbon, lw=2.0)
        ncols = int(cfg["time_n_cols"]); col_sep = float(cfg["time_col_sep_mm"])
        cw = float(cfg["time_contact_w_mm"])
        for xi in (np.arange(ncols) - (ncols - 1) / 2.0) * col_sep:
            c = pv.Box(bounds=(xi - col_sep * 0.34, xi + col_sep * 0.34,
                               th / 2, th / 2 + 0.07, -cw / 2, cw / 2))
            _gold(p, c)
    return build


def build_duke(preset_fn, r_nerve_mm=1.0):
    def build(p):
        import cuff_designer
        preset = json.loads((ROOT / "resources/cuffs" / preset_fn).read_text())
        ns = {"z_nerve": 0.0, "r_nerve": r_nerve_mm * 1e-3, "r_n": r_nerve_mm * 1e-3}
        add_nerve(p, intra=False)
        for _l, _s, mesh, role in cuff_designer.render_design(preset, ns_extras=ns):
            if mesh.n_points == 0:
                continue
            m = mesh.copy(); m.points = m.points * 1000.0
            if role == "conductor":
                _gold(p, m); _ink(p, m, lw=1.8)
            elif role == "insulator":
                _flat(p, m, SILICONE, 0.34); _ink(p, m, lw=1.4)
    return build


def _scene(build, out_png):
    pv.set_plot_theme("document")
    p = pv.Plotter(off_screen=True, window_size=(1500, 1300))
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
    p.view_vector((0.62, 0.46, 0.86), viewup=(0, 1, 0))
    p.reset_camera(); p.camera.zoom(1.45)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    p.screenshot(str(out_png), window_size=(1500, 1300))
    p.close()
    from PIL import Image
    im = Image.open(out_png).convert("RGB")
    a = np.asarray(im)
    ys, xs = np.where((a < 248).any(2))
    if len(xs):
        pad = 18
        im.crop((max(0, int(xs.min()) - pad), max(0, int(ys.min()) - pad),
                 min(im.width, int(xs.max()) + pad),
                 min(im.height, int(ys.max()) + pad))).save(out_png)
    print(f"  wrote {out_png.name}")


GALLERY = [
    ("Bipolar cuff", "el_bipolar"), ("Tripolar cuff", "el_tripolar"),
    ("Multi-contact cuff", "el_multicontact"), ("LivaNova 2000", "el_livanova2000"),
    ("LivaNova 3000", "el_livanova3000"), ("LIFE", "el_life"), ("TIME", "el_time"),
]


def compose_gallery():
    """Minimal Nature-style labeled grid (4 cuffs on top, 3 below, centered)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    plt.rcParams.update({"font.family": "sans-serif",
                         "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"]})
    fig = plt.figure(figsize=(13.6, 6.9))
    gs = GridSpec(2, 8, figure=fig, left=0.006, right=0.994, top=0.93, bottom=0.01,
                  wspace=0.04, hspace=0.16)
    cells = [gs[0, 0:2], gs[0, 2:4], gs[0, 4:6], gs[0, 6:8],
             gs[1, 1:3], gs[1, 3:5], gs[1, 5:7]]
    for cell, (title, slug) in zip(cells, GALLERY):
        ax = fig.add_subplot(cell); ax.axis("off")
        img = OUTDIR / f"{slug}.png"
        if img.exists():
            ax.imshow(plt.imread(str(img)))
        ax.set_title(title, fontsize=12.5, color="#1a1a1a", pad=5)
    out = ROOT / "paper_figs/out/figures/png/electrode_gallery.png"
    fig.savefig(out, dpi=220, facecolor="white"); plt.close(fig)
    print(f"wrote {out}")


def main():
    base = dict(L_cuff_mm=LCUFF)
    specs = [
        ("el_bipolar", build_cuff("bipolar ring-pair",
         dict(base, bipolar_axial_sep_mm=4.0, bipolar_ring_width_mm=0.75))),
        ("el_tripolar", build_cuff("tripolar (anode-cathode-anode)",
         dict(base, tripolar_axial_sep_mm=2.6, tripolar_ring_width_mm=0.7))),
        ("el_multicontact", build_cuff("ring-array (NxM)",
         dict(base, array_n_rows=3, array_n_cols=4, array_row_sep_mm=2.6,
              array_contact_w_mm=0.9, array_contact_phi_deg=55.0))),
        ("el_livanova2000", build_duke("LivaNova2000_v2.json", 1.0)),
        ("el_livanova3000", build_duke("LivaNova3000_v2.json", 1.0)),
        ("el_life", build_life(dict(base, life_n_rows=3, life_n_cols=1, life_row_sep_mm=2.6,
              life_col_sep_mm=0.5, life_contact_length_mm=0.7, life_diameter_um=360.0,
              life_chord_phi_deg=0.0, life_x_mm=0.0, life_y_mm=0.0))),
        ("el_time", build_time(dict(base, time_n_cols=5, time_col_sep_mm=0.32,
              time_contact_w_mm=0.3, time_chord_phi_deg=0.0, time_x_mm=0.0, time_y_mm=0.0))),
    ]
    OUTDIR.mkdir(parents=True, exist_ok=True)
    for slug, build in specs:
        _scene(build, OUTDIR / f"{slug}.png")
    compose_gallery()
    print("done")


if __name__ == "__main__":
    main()
