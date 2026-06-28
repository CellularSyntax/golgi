# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Multi-page Generate Report PDF (F2.3.c).

Builds a single PDF combining:

  * Cover page: project name, generated-at, generated-by, golgi
    version, mesh sha256, FEM sha256.
  * Table of contents.
  * Per-domain sections (each optional via the dialog):
      - Electrode design (3D viewport snapshot + electrode table)
      - Mesh results (3D snapshot + tet-quality histogram + table)
      - Fiber trajectories (3D snapshot + branch summary)
      - FEM results (3D snapshot + axis/slice/AF figures)
      - Single-fiber simulation (pulse + propagation + waterfall
        + sim-config table)
      - Population simulation (KDE + xsec + activation + config)
      - Sweep (recruitment + threshold + activation heatmap)
  * Conductivity table (auto-included).
  * Reproducibility appendix (auto, sha256s + frozen requirements
    + active config files).
  * Bibliography (auto, citations from active pop_preset).
  * Audit excerpt (auto, project-scoped audit rows).

Each section is a sequence of mpl PDF pages built via
matplotlib.backends.backend_pdf.PdfPages so we don't pull in any
new dependencies — Plotly figures are rasterised through kaleido
+ embedded via imshow.

`generate_report(ctx, *, sections, out_pdf_path)` is the public
entry point. `ctx` is the same FigureExportContext shape that
golgi.figures.registry uses. `sections` is a dict of bool flags
keyed by the dialog section ids; sections whose prereqs are
missing get a placeholder page explaining what's missing
(per the F2.3 spec — "include anyway, with a placeholder
page")."""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .export import (
    PRESETS,
    apply_preset_to_plotly_fig,
)
from .registry import REGISTRY, FigureExportContext, get


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# US Letter portrait — most journals accept Letter, A4 is a one-liner
# config change for future commits (`PAGE_SIZE_IN = (8.27, 11.69)`).
PAGE_SIZE_IN = (8.5, 11.0)

# Title / body font sizes (matplotlib pt — matplotlib's pt is
# real-pt at the figure's DPI, no conversion needed).
TITLE_PT = 18
SECTION_PT = 14
BODY_PT = 10
SMALL_PT = 8
MONO_PT = 8


# ---------------------------------------------------------------------------
# Page-building primitives
# ---------------------------------------------------------------------------


def _new_page(suptitle: str = "") -> "Any":
    """Return a fresh matplotlib Figure sized to the report page.
    Caller adds axes + savefig via pdf.savefig(fig)."""
    import matplotlib.pyplot as plt
    fig = plt.figure(figsize=PAGE_SIZE_IN, dpi=200)
    if suptitle:
        fig.text(
            0.5, 0.96, suptitle,
            ha="center", va="top",
            fontsize=SECTION_PT, fontweight="bold",
        )
    return fig


def _plotly_to_png_ndarray(
    fig_dict: dict,
    scale: float = 2.0,
) -> "np.ndarray":
    """Plotly dict → PNG bytes via kaleido → ndarray via mpl.image.
    `scale` controls the rasterisation density (kaleido multiplies
    dimensions by `scale`). 2.0 gives crisp embeds without ballooning
    the PDF too far."""
    import plotly.io as pio
    import matplotlib.image as _mpli
    png = pio.to_image(
        fig_dict, format="png", scale=scale,
    )
    return _mpli.imread(io.BytesIO(png))


def _placeholder_page(
    pdf, title: str, message: str,
) -> None:
    """For sections whose prereqs are missing — drop a single page
    explaining what to run before re-generating. Keeps the report
    structure intact so page numbers and the TOC don't shift."""
    fig = _new_page(suptitle=title)
    ax = fig.add_axes([0.1, 0.3, 0.8, 0.5])
    ax.axis("off")
    ax.text(
        0.5, 0.7,
        message,
        ha="center", va="top",
        fontsize=BODY_PT,
        color="#555",
        wrap=True,
    )
    pdf.savefig(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _embed_image_page(
    pdf,
    title: str,
    image_ndarray,
    caption: str = "",
) -> None:
    """One page: title (suptitle) + a large image filling the body
    + an optional small caption at the bottom. Used for the 3D
    viewport snapshots."""
    fig = _new_page(suptitle=title)
    ax = fig.add_axes([0.05, 0.1, 0.9, 0.78])
    ax.imshow(image_ndarray)
    ax.set_axis_off()
    if caption:
        fig.text(
            0.5, 0.05,
            caption,
            ha="center", va="top",
            fontsize=SMALL_PT, color="#555",
            wrap=True,
        )
    pdf.savefig(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _embed_plotly_page(
    pdf,
    title: str,
    fig_dict: dict,
    caption: str = "",
    preset_name: str = "match-ui",
) -> None:
    """One page: title + a Plotly figure rendered via kaleido and
    embedded as an image. Applies the preset to the figure first so
    the embedded PNG has paper-friendly styling."""
    import copy
    if not fig_dict or not fig_dict.get("data"):
        _placeholder_page(
            pdf, title,
            "(figure not available — re-run the relevant "
            "pipeline stage and re-generate the report.)",
        )
        return
    preset = PRESETS.get(preset_name) or PRESETS["match-ui"]
    work = copy.deepcopy(fig_dict)
    apply_preset_to_plotly_fig(work, preset)
    try:
        img = _plotly_to_png_ndarray(work, scale=2.0)
    except Exception as ex:                              # noqa: BLE001
        _placeholder_page(
            pdf, title,
            f"(figure rasterisation failed: "
            f"{type(ex).__name__}: {ex})",
        )
        return
    _embed_image_page(pdf, title, img, caption=caption)


def _text_page(
    pdf,
    title: str,
    blocks: Iterable[tuple[str, str]],
) -> None:
    """Page with a list of (sub-heading, body-text) blocks. Useful
    for the reproducibility / bibliography / audit pages where the
    content is structured text, not a chart."""
    fig = _new_page(suptitle=title)
    y = 0.88
    for sub_heading, body in blocks:
        if sub_heading:
            fig.text(
                0.06, y, sub_heading,
                ha="left", va="top",
                fontsize=BODY_PT, fontweight="bold",
            )
            y -= 0.025
        if body:
            fig.text(
                0.06, y, body,
                ha="left", va="top",
                fontsize=SMALL_PT,
                family="monospace" if "\n" in body else None,
                color="#222",
                wrap=True,
            )
            # Rough vertical advance based on line count — matplotlib
            # text doesn't give us a tight bbox without a renderer,
            # so we estimate.
            n_lines = body.count("\n") + 1
            y -= 0.022 * n_lines + 0.012
        if y < 0.08:
            # Out of room — start a new page.
            pdf.savefig(fig)
            import matplotlib.pyplot as plt
            plt.close(fig)
            fig = _new_page(suptitle=f"{title} (cont.)")
            y = 0.88
    pdf.savefig(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _table_page(
    pdf,
    title: str,
    header: list[str],
    rows: list[list[str]],
    col_widths: list[float] | None = None,
) -> None:
    """One page with a title + a matplotlib table. `col_widths`
    sums should be ~1.0; falls back to equal columns when None.
    The table renders inside the page body via ax.table()."""
    fig = _new_page(suptitle=title)
    ax = fig.add_axes([0.05, 0.08, 0.9, 0.82])
    ax.axis("off")
    if not rows:
        ax.text(
            0.5, 0.5,
            "(no data)",
            ha="center", va="center",
            fontsize=BODY_PT, color="#555",
        )
    else:
        widths = col_widths or [1.0 / len(header)] * len(header)
        tbl = ax.table(
            cellText=rows,
            colLabels=header,
            cellLoc="left",
            colLoc="left",
            colWidths=widths,
            loc="upper center",
        )
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(SMALL_PT)
        tbl.scale(1.0, 1.4)
        # Style the header row.
        for j in range(len(header)):
            cell = tbl[0, j]
            cell.set_text_props(fontweight="bold")
            cell.set_facecolor("#ececef")
    pdf.savefig(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


# ---------------------------------------------------------------------------
# Section builders
# ---------------------------------------------------------------------------


def _golgi_version() -> str:
    """Best-effort golgi version. Falls back to the dirty
    git-shortsha when the package isn't installed (in-tree dev)."""
    try:
        from importlib.metadata import version
        return version("golgi")
    except Exception:                                    # noqa: BLE001
        pass
    try:
        import subprocess
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return f"dev-{out.decode().strip()}"
    except Exception:                                    # noqa: BLE001
        return "dev"


def _sha256_of_file(path: Path) -> str:
    """Hex sha256 of a file, or `""` if missing / unreadable."""
    try:
        h = hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:                                    # noqa: BLE001
        return ""


def _format_iso_ts(ts: str) -> str:
    """ISO 8601 timestamp → "YYYY-MM-DD HH:MM:SS" with no
    timezone clutter. Returns the input unchanged on parse
    failures so partially-formed strings still show."""
    if not ts:
        return ""
    try:
        return _dt.datetime.fromisoformat(ts).strftime(
            "%Y-%m-%d %H:%M:%S",
        )
    except Exception:                                    # noqa: BLE001
        return ts


def _cover(
    pdf,
    *,
    project_name: str,
    user_name: str,
    user_meta: dict | None,
    project_created: str,
    project_modified: str,
    golgi_version: str,
    mesh_sha: str,
    fem_sha: str,
    timestamp: str,
) -> None:
    """Title page. Centred logo block + metadata table.

    `user_meta` carries the optional profile fields (email,
    position, institution, country). Empty entries are dropped
    so we don't show empty rows.
    """
    fig = _new_page()
    ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
    ax.axis("off")
    ax.text(
        0.5, 0.85, "GOLGI",
        ha="center", va="center",
        fontsize=40, fontweight="bold", color="#e24b4a",
        family="DejaVu Sans",
    )
    ax.text(
        0.5, 0.795, "Simulation report",
        ha="center", va="center",
        fontsize=16, color="#1f2024",
    )
    ax.text(
        0.5, 0.72, project_name or "(no active project)",
        ha="center", va="center",
        fontsize=19, fontweight="bold", color="#1f2024",
    )

    # Build the rows in three groups: report, project, user.
    # Each entry is (label, value, group_break_after).
    user_meta = user_meta or {}
    user_rows: list[tuple[str, str]] = []
    if user_meta.get("position"):
        user_rows.append(("Position", user_meta["position"]))
    if user_meta.get("institution"):
        user_rows.append(("Institution", user_meta["institution"]))
    if user_meta.get("country"):
        user_rows.append(("Country", user_meta["country"]))
    if user_meta.get("email"):
        user_rows.append(("Email", user_meta["email"]))

    groups: list[tuple[str, list[tuple[str, str]]]] = [
        ("Report", [
            ("Generated at", timestamp),
            ("golgi version", golgi_version),
        ]),
        ("Project", [
            ("Name", project_name or "(no active project)"),
            ("Created", _format_iso_ts(project_created)
             or "(unknown)"),
            ("Last modified",
             _format_iso_ts(project_modified) or "(unknown)"),
            ("mesh sha256",
             (mesh_sha[:16] + "…") if mesh_sha else "(no mesh)"),
            ("FEM sha256",
             (fem_sha[:16] + "…") if fem_sha else "(no FEM)"),
        ]),
        ("Author", [
            ("Name", user_name or "(anonymous)"),
            *user_rows,
        ]),
    ]

    y = 0.62
    for group_title, rows in groups:
        # Group header (small caps, grey).
        ax.text(
            0.20, y, group_title.upper(),
            ha="left", va="top",
            fontsize=SMALL_PT, color="#888a90",
            fontweight="bold",
        )
        y -= 0.028
        for label, value in rows:
            if not value:
                continue
            ax.text(
                0.30, y, label,
                ha="right", va="top",
                fontsize=BODY_PT, color="#666",
            )
            ax.text(
                0.33, y, value,
                ha="left", va="top",
                fontsize=BODY_PT, color="#1f2024",
            )
            y -= 0.030
        y -= 0.010   # extra gap between groups

    ax.text(
        0.5, 0.04,
        "Auto-generated by GOLGI · "
        "this report bundles project state at the moment "
        "of generation.",
        ha="center", va="center",
        fontsize=SMALL_PT, color="#888a90", style="italic",
    )
    pdf.savefig(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _toc(pdf, entries: list[str]) -> None:
    """Simple TOC page. `entries` is a list of section titles in
    order — no page numbers (we generate sequentially)."""
    fig = _new_page(suptitle="Contents")
    ax = fig.add_axes([0.12, 0.08, 0.76, 0.82])
    ax.axis("off")
    y = 0.95
    for entry in entries:
        ax.text(
            0.0, y, "• " + entry,
            ha="left", va="top",
            fontsize=BODY_PT,
        )
        y -= 0.04
    pdf.savefig(fig)
    import matplotlib.pyplot as plt
    plt.close(fig)


def _conductivity_table(
    pdf, state,
) -> None:
    """Per-tissue σ at the active stim frequency. Reads
    state.tissue_sigma_{tag} or whatever the materials drawer
    populates; falls back to a "no data" page when the dict is
    empty."""
    rows: list[list[str]] = []
    # The Materials drawer holds per-tissue sigma in a dict-ish
    # state structure. Read defensively — the user may not have
    # opened the dialog yet.
    sigma_dict = getattr(state, "tissue_sigma_table", None)
    if isinstance(sigma_dict, dict):
        for tag, entry in sigma_dict.items():
            rows.append([
                str(entry.get("label", tag)),
                f"{float(entry.get('sigma_S_per_m', 0.0)):.4g}",
                str(entry.get("source", "")),
            ])
    if not rows:
        # Fallback — best-effort scan for tissue_sigma_<n> keys.
        for n in range(1, 7):
            v = getattr(state, f"tissue_sigma_{n}", None)
            if v is None:
                continue
            rows.append([f"tag {n}", f"{float(v):.4g}", ""])
    _table_page(
        pdf,
        "Conductivities",
        header=["Tissue", "σ [S/m]", "Source"],
        rows=rows,
        col_widths=[0.38, 0.22, 0.40],
    )


def _figure_section(
    pdf,
    section_title: str,
    fig_ids: list[str],
    ctx: FigureExportContext,
) -> None:
    """One page per fig_id in the section. Resolves each spec from
    the registry and pulls the live state dict. Missing figures
    get a placeholder page so the section structure survives."""
    for fig_id in fig_ids:
        try:
            spec = get(fig_id)
        except KeyError:
            _placeholder_page(
                pdf, f"{section_title}",
                f"(unknown figure id '{fig_id}')",
            )
            continue
        fig_dict = getattr(ctx.state, spec.state_var, None)
        title = f"{section_title} — {spec.title}"
        _embed_plotly_page(pdf, title, fig_dict or {})


def _viewport_section(
    pdf,
    section_title: str,
    viewport_png: bytes | None,
    caption: str = "",
) -> None:
    """LEGACY single-snapshot path. Kept around so the report can
    fall back to the live-viewport capture when an off-screen
    render3d variant isn't available (e.g. no mesh built yet).
    The variant path (`_render3d_section`) is the v2 default."""
    if not viewport_png:
        _placeholder_page(
            pdf, section_title,
            "(no 3D viewport snapshot captured — set up the view "
            "you want in the workspace then re-generate.)",
        )
        return
    from .render3d import _png_to_mpl_image
    img = _png_to_mpl_image(viewport_png)
    _embed_image_page(pdf, section_title, img, caption=caption)


def _render3d_section(
    pdf,
    section_title: str,
    variant_ids: list[str],
    ctx: FigureExportContext,
    fallback_png: bytes | None = None,
) -> None:
    """Per-variant pages built off-screen via render3d.render_variant.
    For each variant id, call materialize() (which routes through
    the render3d source) and embed the resulting PNG as one page.
    Variants whose inputs are missing get a placeholder page; the
    optional `fallback_png` is used when ALL variants in the
    section fail (rare — usually one of them succeeds)."""
    from .render3d import _png_to_mpl_image
    any_rendered = False
    for variant_id in variant_ids:
        try:
            spec = get(variant_id)
        except KeyError:
            _placeholder_page(
                pdf, f"{section_title}",
                f"(unknown render3d variant '{variant_id}')",
            )
            continue
        try:
            png_bytes = ctx.state.__class__  # touch (unused)
        except Exception:                                # noqa: BLE001
            pass
        try:
            from .registry import materialize as _mat
            png_bytes = _mat(ctx, variant_id)
        except Exception as ex:                          # noqa: BLE001
            _placeholder_page(
                pdf,
                f"{section_title} · {spec.title}",
                f"(render failed: "
                f"{type(ex).__name__}: {ex})",
            )
            continue
        if not png_bytes:
            _placeholder_page(
                pdf,
                f"{section_title} · {spec.title}",
                "(figure inputs not available — e.g. mesh not "
                "built, fibers not generated.)",
            )
            continue
        img = _png_to_mpl_image(png_bytes)
        _embed_image_page(
            pdf,
            f"{section_title} · {spec.title}",
            img,
            caption=(
                f"variant: {variant_id} · "
                "off-screen iso render"
            ),
        )
        any_rendered = True
    if not any_rendered and fallback_png is not None:
        from .render3d import _png_to_mpl_image
        img = _png_to_mpl_image(fallback_png)
        _embed_image_page(
            pdf, section_title, img,
            caption=(
                "(live-viewport fallback — off-screen renders "
                "could not be built)"
            ),
        )


def _sim_config_block(
    state, kind: str,
) -> list[tuple[str, str]]:
    """Read the active sim config from state and return blocks
    suitable for `_text_page`. `kind` is "single_fiber" or
    "population"."""
    if kind == "single_fiber":
        cfg = [
            ("Backend", str(getattr(state, "fiber_backend", ""))),
            ("Model", str(getattr(state, "fiber_model", ""))),
            ("Diameter [µm]",
             f"{float(getattr(state, 'fiber_diameter_um', 0.0)):.2f}"),
            ("Pulse type",
             str(getattr(state, "fiber_pulse_type", ""))),
            ("Cathode amplitude [mA]",
             f"{float(getattr(state, 'fiber_mono_amp_mA', 0.0)):.3f}"),
            ("Onset [ms]",
             f"{float(getattr(state, 'fiber_onset_ms', 0.0)):.2f}"),
            ("tstop [ms]",
             f"{float(getattr(state, 'fiber_tstop_ms', 0.0)):.2f}"),
        ]
    else:
        rows: list[tuple[str, str]] = [
            ("Active preset",
             str(getattr(state, "pop_preset", ""))),
            ("Seed",
             str(int(getattr(state, "pop_seed", 0)))),
        ]
        row_meta = getattr(state, "pop_row_meta", {}) or {}
        if row_meta:
            rows.append((
                "Sub-populations",
                "\n".join(
                    f"  {rid}: "
                    f"{m.get('model', '')} "
                    f"(μ={m.get('mean_um', 0.0):.2f} µm, "
                    f"σ={m.get('std_um', 0.0):.2f} µm)"
                    for rid, m in row_meta.items()
                ),
            ))
        cfg = rows
    # Convert to (heading, body) pairs collapsed into one block.
    body = "\n".join(f"{label:32s}  {value}" for label, value in cfg)
    return [(kind.replace("_", " ").title() + " config", body)]


def _branch_summary_block(geom) -> list[tuple[str, str]]:
    """Per-branch fiber count + length stats."""
    if (geom is None or geom.fiber_paths_raw is None
            or geom.fiber_branch_idx is None):
        return [("Fiber trajectories",
                 "(no fibers generated yet)")]
    paths = list(geom.fiber_paths_raw)
    bidx = np.asarray(geom.fiber_branch_idx)
    rows: list[str] = []
    for b in sorted(set(int(x) for x in bidx)):
        mask = bidx == b
        n = int(mask.sum())
        if n == 0:
            continue
        # Length in mm
        lengths_mm: list[float] = []
        for i, p in enumerate(paths):
            if not mask[i]:
                continue
            pa = np.asarray(p)
            ds = np.linalg.norm(np.diff(pa, axis=0), axis=1)
            lengths_mm.append(float(ds.sum() * 1e3))
        if not lengths_mm:
            continue
        arr = np.asarray(lengths_mm)
        rows.append(
            f"  branch {b}: n = {n}, "
            f"length mm = {arr.min():.1f} / "
            f"{np.median(arr):.1f} / {arr.max():.1f} "
            f"(min / median / max)"
        )
    body = "\n".join(rows) if rows else "(no branches)"
    return [("Fiber branch summary", body)]


def _reproducibility_block(
    project_dir: Path | None,
    golgi_version: str,
) -> list[tuple[str, str]]:
    """sha256s of the on-disk config + cache files + the lockfile.
    Pairs with F2.2's `golgi replay` once that lands."""
    out: list[tuple[str, str]] = [
        ("golgi version", golgi_version),
    ]
    if project_dir is None or not Path(project_dir).is_dir():
        out.append((
            "Project artifacts",
            "(no active project — sha256 list unavailable.)",
        ))
        return out
    pdir = Path(project_dir)
    # All project artefacts live at the project root in the
    # current layout (matching `_compute_project_status` in
    # app.py) — earlier drafts used `geometry/` / `fem/` /
    # `sims/` subfolder paths that no project on disk uses, so
    # every row showed "(missing)".
    interesting = [
        "project.json",
        "mesh_config.json",
        "electrode_config.json",
        "nerve_paths_seed_config.json",
        "nerve.msh",
        "nerve_paths_fibers.npz",
        "axis_line.npz",
        "paths_Ve.npz",
        "nerve_surface_Ve.npz",
        "slice_volume.npz",
        "fiber_sim_cache.json",
        "pop_state.json",
        "fiber_sim_results.pkl",
        "pop_state.pkl",
    ]
    lines: list[str] = []
    for rel in interesting:
        p = pdir / rel
        if not p.is_file():
            lines.append(f"  {rel:50s}  (missing)")
            continue
        sha = _sha256_of_file(p)
        size = p.stat().st_size
        lines.append(
            f"  {rel:50s}  {sha[:16]}…  {size / 1024:.1f} KB"
        )
    out.append((
        "Project artifacts (sha256 prefix · size)",
        "\n".join(lines),
    ))
    # Frozen requirements — drop the first ~40 lines so we don't
    # blow up the page.
    req = pdir / "env" / "requirements-frozen.txt"
    if req.is_file():
        try:
            head = "\n".join(
                "  " + L for L in
                req.read_text(encoding="utf-8").splitlines()[:40]
            )
            out.append((
                "Pinned requirements (first 40)",
                head,
            ))
        except Exception:                              # noqa: BLE001
            pass
    return out


def _bibliography_block(state) -> list[tuple[str, str]]:
    """Citations from the active pop_preset + any validation
    dataset metadata (F3.3 future-proofed)."""
    citations: list[str] = []
    pop_preset = str(getattr(state, "pop_preset", "") or "")
    if pop_preset:
        try:
            from golgi.state_defaults import pop_presets
            meta = pop_presets.preset_meta(pop_preset)
            if meta and meta.get("citation"):
                citations.append(
                    f"• {pop_preset}: {meta['citation']}"
                )
        except Exception:                              # noqa: BLE001
            pass
    # Hardcoded conductivity reference (IT'IS / Gabriel) when
    # tissue σ values were derived via Cole-Cole.
    if getattr(state, "cc_plot_figure", None):
        citations.append(
            "• Gabriel et al. 1996, Phys Med Biol — "
            "Cole-Cole tissue dielectric parameters."
        )
        citations.append(
            "• IT'IS Tissue Properties Database — "
            "Hasgall et al., extensible source for nerve "
            "tissue dielectrics."
        )
    if not citations:
        citations.append(
            "(no active citations — pick a population preset "
            "or evaluate a Cole-Cole σ to populate this page.)"
        )
    return [("References", "\n".join(citations))]


def _project_status_page(
    pdf,
    project_dir: Path | None,
    *,
    project_name: str,
) -> None:
    """Render the 8-row stage status overview as the second page
    of the report (right after the cover). Same row data as the
    Welcome view's project detail dialog so the user gets a
    familiar at-a-glance summary of what's done vs pending.

    Imports `_compute_project_status` lazily from golgi.app so
    this module stays free of an app.py import at module load
    time (it would be circular: app.py → figures → app.py)."""
    if project_dir is None:
        _placeholder_page(
            pdf, "Project status",
            "(no active project — status overview unavailable.)",
        )
        return
    try:
        from golgi.app import _compute_project_status
    except Exception as ex:                            # noqa: BLE001
        _placeholder_page(
            pdf, "Project status",
            f"(status lookup failed: "
            f"{type(ex).__name__}: {ex})",
        )
        return
    proj = {"dir": str(project_dir), "name": project_name}
    try:
        rows = _compute_project_status(proj) or []
    except Exception as ex:                            # noqa: BLE001
        _placeholder_page(
            pdf, "Project status",
            f"(status build failed: "
            f"{type(ex).__name__}: {ex})",
        )
        return
    # Convert to table rows (status icon + label + details).
    table_rows: list[list[str]] = []
    for row in rows:
        done = bool(row.get("done"))
        icon = "✓" if done else "·"
        status = "done" if done else "pending"
        label = str(row.get("label", row.get("id", "?")))
        details = str(row.get("details") or "—")
        table_rows.append([icon, label, status, details])
    _table_page(
        pdf,
        "Project status",
        header=["", "Stage", "Status", "Details"],
        rows=table_rows,
        col_widths=[0.06, 0.36, 0.14, 0.44],
    )


def _audit_block(project_dir: Path | None) -> list[tuple[str, str]]:
    """Project-scoped audit rows from the auth DB.

    Joins `_AuditEvent` against `_User` so the report's audit
    excerpt shows the username next to each event — earlier
    versions only logged the user_id, which surfaced as a
    nameless integer column."""
    rows: list[str] = []
    try:
        from golgi.auth.models import (
            _AuditEvent, _User, get_session,
        )
        with get_session() as session:
            # Pre-fetch user_id → username once so the audit loop
            # below doesn't issue N queries.
            user_lookup: dict[int, str] = {
                int(u.id): str(
                    u.username or u.email or f"user_{u.id}"
                )
                for u in session.query(_User).all()
            }
            q = session.query(_AuditEvent)
            if project_dir is not None:
                q = q.filter(
                    _AuditEvent.project_dir == str(project_dir),
                )
            events = (
                q.order_by(_AuditEvent.ts.desc()).limit(40).all()
            )
            for ev in events:
                ts = ev.ts.strftime("%Y-%m-%d %H:%M:%S") if ev.ts else ""
                who = (
                    user_lookup.get(int(ev.user_id), "—")
                    if ev.user_id is not None else "—"
                )
                rows.append(
                    f"  {ts}  "
                    f"{who[:16]:16s}  "
                    f"{(ev.action or '')[:28]:28s}  "
                    f"{(ev.status or 'info')[:8]}"
                )
    except Exception as ex:                            # noqa: BLE001
        return [("Audit log",
                 f"(audit query failed: "
                 f"{type(ex).__name__}: {ex})")]
    if not rows:
        return [("Audit log",
                 "(no audit rows for this project — actions "
                 "performed BEFORE this report runs will appear "
                 "here on the next run.)")]
    header = (
        f"  {'timestamp':19s}  {'user':16s}  "
        f"{'action':28s}  status"
    )
    return [(
        "Audit log (most recent 40 events)",
        header + "\n" + "\n".join(rows),
    )]


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


# Default section ids the dialog exposes as checkboxes. Each maps to
# the section-building logic in `generate_report` below.
DEFAULT_SECTIONS: dict[str, bool] = {
    "electrode_design": True,
    "mesh_results": True,
    "fiber_trajectories": True,
    "fem_results": True,
    "single_fiber_sim": True,
    "population_sim": True,
    "sweep": True,
}


def generate_report(
    ctx: FigureExportContext,
    *,
    sections: dict[str, bool],
    viewport_png: bytes | None = None,
    project_name: str = "",
    user_name: str = "",
    user_meta: dict | None = None,
    project_created: str = "",
    project_modified: str = "",
    project_dir: Path | None = None,
) -> bytes:
    """Build the full multi-page report as PDF bytes. Returns the
    PDF as `bytes` so the caller can stream it back via the
    standard data-URI download flow (same shape as the per-figure
    + bulk export handlers).

    `ctx.render3d_kwargs` must be populated by the caller (the
    do_generate_report action handler does this) — render3d
    variants need DEFAULTS / GOLD_STYLE / BRANCH_PALETTE to
    style the off-screen actors."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_pdf import PdfPages

    state = ctx.state
    geom = ctx.geom

    # Cover metadata.
    timestamp = _dt.datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S",
    )
    version = _golgi_version()
    mesh_sha = ""
    fem_sha = ""
    if project_dir is not None:
        # Project artefacts live at the project root, not under
        # geometry/ or fem/ subfolders — the original v1 path
        # convention was wrong, every project showed "(no mesh)".
        mesh_sha = _sha256_of_file(
            Path(project_dir) / "nerve.msh",
        )
        # F3.1: paths_Ve.npz now lives under fem/<design_id>/.
        # Hash the active design's file when available, falling
        # back to the legacy flat root for pre-F3.1 projects.
        _pd = Path(project_dir)
        _ve_candidates = [_pd / "paths_Ve.npz"]
        _fem_root = _pd / "fem"
        if _fem_root.is_dir():
            for _sub in sorted(_fem_root.iterdir()):
                if _sub.is_dir():
                    _ve_candidates.append(_sub / "paths_Ve.npz")
        for _cand in _ve_candidates:
            if _cand.is_file():
                fem_sha = _sha256_of_file(_cand)
                break

    # Plan the TOC up front so the second page lists what's coming.
    toc_entries: list[str] = ["Project status"]
    section_plan: list[tuple[str, str]] = []  # (title, kind)
    if sections.get("electrode_design", False):
        toc_entries.append("Electrode design")
        section_plan.append(("Electrode design", "electrode"))
    if sections.get("mesh_results", False):
        toc_entries.append("Mesh results")
        section_plan.append(("Mesh results", "mesh"))
    if sections.get("fiber_trajectories", False):
        toc_entries.append("Fiber trajectories")
        section_plan.append(("Fiber trajectories", "fibers"))
    if sections.get("fem_results", False):
        toc_entries.append("FEM results")
        section_plan.append(("FEM results", "fem"))
    if sections.get("single_fiber_sim", False):
        toc_entries.append("Single-fiber simulation")
        section_plan.append(("Single-fiber simulation",
                             "single_fiber"))
    if sections.get("population_sim", False):
        toc_entries.append("Population simulation")
        section_plan.append(("Population simulation", "population"))
    if sections.get("sweep", False):
        toc_entries.append("Sweep")
        section_plan.append(("Sweep", "sweep"))
    toc_entries.extend([
        "Conductivities", "Reproducibility appendix",
        "Bibliography", "Audit excerpt",
    ])

    buf = io.BytesIO()
    with PdfPages(buf) as pdf:
        _cover(
            pdf,
            project_name=project_name,
            user_name=user_name,
            user_meta=user_meta,
            project_created=project_created,
            project_modified=project_modified,
            golgi_version=version,
            mesh_sha=mesh_sha,
            fem_sha=fem_sha,
            timestamp=timestamp,
        )
        _toc(pdf, toc_entries)
        # Project status — second content page, mirrors the
        # Welcome view's project detail dialog so the reader gets
        # an at-a-glance "what's been run" summary before diving
        # into individual sections.
        _project_status_page(
            pdf, project_dir, project_name=project_name,
        )

        for title, kind in section_plan:
            if kind == "electrode":
                _render3d_section(
                    pdf, title,
                    [
                        "render3d.electrode_geom",
                        "render3d.electrode_with_saline",
                        "render3d.electrode_in_nerve",
                        "render3d.electrode_polar",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
            elif kind == "mesh":
                _render3d_section(
                    pdf, title + " (3D)",
                    [
                        "render3d.mesh_all_regions",
                        "render3d.mesh_quality_all",
                        "render3d.mesh_muscle",
                        "render3d.mesh_endo",
                        "render3d.mesh_epi",
                        "render3d.mesh_cuff",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _figure_section(
                    pdf, title, ["mesh.tet_quality_hist"], ctx,
                )
            elif kind == "fibers":
                _render3d_section(
                    pdf, title + " (3D)",
                    [
                        "render3d.fibers_in_nerve",
                        "render3d.fibers_epi",
                        "render3d.geometry_full",
                        "render3d.geometry_no_muscle",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _text_page(
                    pdf, title + " — summary",
                    _branch_summary_block(geom),
                )
            elif kind == "fem":
                _render3d_section(
                    pdf, title + " · visibility combos (3D)",
                    [
                        "render3d.fem_full",
                        "render3d.fem_no_muscle",
                        "render3d.fem_no_epi",
                        "render3d.fem_no_endo",
                        "render3d.cuff_zoom_iso",
                        "render3d.cuff_cross_section",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _render3d_section(
                    pdf, title + " · V_e overlays (3D)",
                    [
                        "render3d.ve_on_endo",
                        "render3d.ve_on_epi",
                        "render3d.ve_on_fibers",
                        "render3d.ve_on_all",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _render3d_section(
                    pdf, title + " · E-field streamlines (3D)",
                    [
                        "render3d.field_streamlines",
                        "render3d.field_streamlines_cuff_zoom",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                # Cuff-zoom Ve + streamlines combos — the
                # workhorse pages of an FEM report. Three Ve
                # targets × three context conditions.
                _render3d_section(
                    pdf,
                    title + " · cuff zoom (V_e + cuff + streamlines)",
                    [
                        "render3d.cuff_zoom_ve_epi"
                        "_with_cuff_streamlines",
                        "render3d.cuff_zoom_ve_endo"
                        "_with_cuff_streamlines",
                        "render3d.cuff_zoom_ve_fibers"
                        "_with_cuff_streamlines",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _render3d_section(
                    pdf,
                    title + " · cuff zoom (V_e + streamlines, no cuff)",
                    [
                        "render3d.cuff_zoom_ve_epi"
                        "_streamlines_no_cuff",
                        "render3d.cuff_zoom_ve_endo"
                        "_streamlines_no_cuff",
                        "render3d.cuff_zoom_ve_fibers"
                        "_streamlines_no_cuff",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _render3d_section(
                    pdf,
                    title + " · cuff zoom (V_e only)",
                    [
                        "render3d.cuff_zoom_ve_epi_only",
                        "render3d.cuff_zoom_ve_endo_only",
                        "render3d.cuff_zoom_ve_fibers_only",
                    ],
                    ctx,
                    fallback_png=viewport_png,
                )
                _figure_section(
                    pdf, title,
                    [
                        "fem.axis_line",
                        "fem.slice_volume",
                        "fem.activation_fn",
                    ],
                    ctx,
                )
            elif kind == "single_fiber":
                _figure_section(
                    pdf, title,
                    [
                        "fiber.pulse",
                        "fiber.propagation",
                        "fiber.waterfall",
                    ],
                    ctx,
                )
                _text_page(
                    pdf, title + " — configuration",
                    _sim_config_block(state, "single_fiber"),
                )
            elif kind == "population":
                _figure_section(
                    pdf, title,
                    [
                        "pop.kde",
                        "pop.xsec_cuff",
                        "pop.xsec_activated",
                        "pop.propagation",
                        "pop.waterfall",
                    ],
                    ctx,
                )
                _text_page(
                    pdf, title + " — configuration",
                    _sim_config_block(state, "population"),
                )
            elif kind == "sweep":
                _figure_section(
                    pdf, title,
                    [
                        "sweep.recruitment",
                        "sweep.threshold_scatter",
                        "sweep.activation_heatmap",
                    ],
                    ctx,
                )

        # Auto-included appendices.
        _conductivity_table(pdf, state)
        _embed_plotly_page(
            pdf, "Conductivity — σ(f) (Cole-Cole)",
            getattr(state, "cc_plot_figure", None) or {},
        )
        _text_page(
            pdf, "Reproducibility appendix",
            _reproducibility_block(project_dir, version),
        )
        _text_page(
            pdf, "Bibliography",
            _bibliography_block(state),
        )
        _text_page(
            pdf, "Audit excerpt",
            _audit_block(project_dir),
        )
    return buf.getvalue()
