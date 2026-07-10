# Figures & Reports

golgi turns any study into **publication-grade figures** and a single multi-page **PDF report**, from
the same registry of plots the GUI shows. Everything is exportable per-panel, in bulk, or as a report.

Source: `golgi/figures/` — `registry.py` (catalog), `export.py` (presets/formats), `report.py`
(PDF), `render3d.py` (off-screen 3-D capture), plus the per-figure builders.

---

## The figure registry

`registry.py` is the single source of truth for "what figures exist in this project right now."
Each entry (`FigureSpec`) has an `id`, `title`, `category`, and a `source` — either a 2-D **plotly**
figure or an off-screen **3-D render variant** — and an availability predicate so only computable
figures are offered. Categories span the pipeline:

| Category | Examples |
|---|---|
| **Mesh** | surface-quality and tet-quality histograms |
| **FEM** | axis-line Vₑ, slice Vₑ, activating function, impedance bars |
| **Single fiber** | pulse, propagation, waterfall, CNAP |
| **Population** | diameter KDE, cross-section recruitment, propagation, CNAP |
| **Conductivity** | Cole–Cole σ(f) |
| **Sweep** | recruitment curve, threshold-vs-diameter, activation heatmap |
| **Selectivity** | Veraart-SI bars |
| **3-D renders** | electrode, mesh (per region), fibers, FEM field, Vₑ on endo/epi/fibers, E-field streamlines, cuff zooms |

The 3-D variants render off-screen with a caller-supplied camera so all variants of a category share
one viewpoint.

---

## Export presets & formats

`export.py` defines export **presets** (e.g. **screen** and **publication** at higher DPI with serif
fonts) and renders to **PNG, PDF, SVG, EPS, JPG** for figures and **CSV** for the underlying data.
Colour-blind-safe palettes are available (cividis-based `viridis-cb`, the `ibm-cb` 5-colour palette,
and a grayscale fallback) so output is accessible by default.

Three ways to export:

- **Per-panel** — an export button beside each plot (format + preset popover) → a single file.
- **Bulk** — the **Export figures** drawer: pick a format + preset, tick figures by category, and
  **Export to ZIP** (figures + a `data/` folder of CSVs).
- **Report** — see below.

---

## The PDF report

**Export ▸ Export report** opens the **Generate Report** dialog. You tick which sections to include;
golgi assembles a single multi-page PDF with:

1. **Cover** — project name, timestamp, author, golgi version, and mesh/FEM hashes.
2. **Table of contents.**
3. **Per-domain sections** (optional) — electrode design, mesh (with quality histogram), fiber
   trajectories (branch summary), FEM (axis/slice/activating-function), single-fiber, population, and
   sweep, each with a 3-D snapshot and the relevant figures + parameter tables.
4. **Auto-included appendices** — the conductivity table + σ(f) plot, a **reproducibility appendix**
   (hashes + frozen requirements + active config files), a **bibliography** (citations from the active
   population preset), and an **[audit](Authentication-and-Audit) excerpt**.

Because the report captures the 3-D viewport once, **position the camera before generating**. All
output respects the chosen export preset.

---

### See also
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) ·
[Reproducible Study Bundles](Reproducible-Study-Bundles) · [GUI Walkthrough](GUI-Walkthrough) ·
[Reproducing the Paper](Reproducing-the-Paper)
