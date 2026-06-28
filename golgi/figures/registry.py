# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Figure registry (F2.3.a) — single source of truth for "what
figures exist in this project + how do you export them?".

Every panel that shows a figure registers a `FigureSpec` here. The
per-panel export button (`ui/components/figure_export_btn.py`) and
the bulk Exports tab (F2.3.b) + Generate Report (F2.3.c) all read
from the SAME REGISTRY, so adding a new figure means adding ONE
entry here, not three.

This v1 covers all Plotly figures (they store a `{data, layout}`
dict in a state variable — `builder` just reads + deepcopies it
so the live UI figure isn't mutated by a preset apply). The
matplotlib PNG figures (Cole-Cole σ(f), mesh quality histogram,
cuff design preview) store base64 PNG data URIs in state, so they
need their underlying builder called fresh and aren't covered by
this v1 — they will land in F2.3.a follow-up commits with a
`source` of "callable".

Intentionally zero coupling to the UI or pipeline drivers — the
registry only knows about state variables + which existing builder
to call. The action handler (`golgi/actions/figure_export.py`)
performs the actual render."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


# ---------------------------------------------------------------------------
# Context shim
# ---------------------------------------------------------------------------


@dataclass
class FigureExportContext:
    """Tiny bag carrying just enough state for a registry builder to
    work. Stays UI-agnostic so the registry can be exercised
    headlessly (F4.1 entry point). `state` is the live trame State
    object, `geom` is the workspace geometry namespace, `project_dir`
    is the active project's directory (used for output filenames).

    `render3d_kwargs` carries the styling constants (DEFAULTS dict,
    gold style, branch palette) that the render3d variant builders
    need but that the registry module can't import from app.py
    without creating a circular dep. build_app populates this at
    registration time and threads it through the context here.

    Builders should treat `state` as read-only — they only ever READ
    `state.<figure_var>` and never write back."""
    state: Any
    geom: Any = None
    project_dir: Any = None
    render3d_kwargs: dict | None = None


# ---------------------------------------------------------------------------
# FigureSpec
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FigureSpec:
    """One registered figure.

    `id`           Stable string ID. Used in filenames, manifests,
                   per-panel button bindings. Format: "<category>.<name>",
                   lowercase, no spaces (so it round-trips through
                   filesystems + URLs without escaping).
    `title`        Display name shown in the export popover + the bulk
                   Exports tab's checkbox row.
    `category`     Grouping for the bulk Exports tab. Matches the
                   drawer/tab the figure lives in (Cuff / Mesh / FEM /
                   Fibers / Population / Sweep / Conductivity / 3D).
    `source`       "plotly_state"     → read {data,layout} dict from
                                        state.<state_var>.
                   "render3d_variant" → run an off-screen pyvista
                                        render via
                                        golgi.figures.render3d
                                        keyed by `variant_id`.
    `state_var`    The state variable holding the figure dict (only
                   used when source == "plotly_state").
    `variant_id`   The render3d variant id (only used when
                   source == "render3d_variant").
    `availability` Optional predicate `(ctx) -> bool`. When set, the
                   bulk Exports tab uses it to grey-out unavailable
                   figures. Per-panel buttons just disable themselves
                   when the underlying state var is empty.
    """
    id: str
    title: str
    category: str
    source: str
    state_var: str = ""
    variant_id: str = ""
    availability: Optional[Callable[[FigureExportContext], bool]] = None


def _plotly_dict_is_populated(d: Any) -> bool:
    """Same emptiness check the figure builders use for the
    placeholder pattern. A figure is considered populated iff it has
    at least one trace in `data`. The placeholder dicts shipped by
    `_plotly_placeholder` carry exactly one trace (an annotation
    layer) and an `_empty: True` flag — we honour that flag so the
    bulk Exports tab grey-outs them correctly."""
    if not isinstance(d, dict):
        return False
    if d.get("_empty") is True:
        return False
    data = d.get("data") or []
    return bool(len(data) > 0)


def _plotly_available(state_var: str) -> Callable[[FigureExportContext], bool]:
    """Build an availability predicate for a Plotly state var. Reads
    state.<state_var> and applies `_plotly_dict_is_populated`."""

    def _check(ctx: FigureExportContext) -> bool:
        try:
            d = getattr(ctx.state, state_var, None)
            if d is None:
                # Trame state can be dict-like too; try the item
                # access path as a fallback before declaring the
                # var unavailable.
                try:
                    d = ctx.state[state_var]
                except Exception:                            # noqa: BLE001
                    d = None
        except Exception:                                    # noqa: BLE001
            return False
        return _plotly_dict_is_populated(d)

    return _check


def _render3d_available(variant_id: str) -> Callable[[FigureExportContext], bool]:
    """Build an availability predicate for a render3d variant.

    Gates on every input the variant's config actually needs:
      * regions → at least one of cfg.regions present in
                  geom.region_surfaces[_viz]
      * fibers / ve_on_fibers → geom.fiber_paths_raw populated
      * ve_on_fibers → geom.fiber_paths_Ve populated (same length)
      * ve_on_regions → geom.nerve_surface_Ve populated
      * show_field_lines → geom.field_lines_poly populated

    The render_variant function itself silently skips missing
    sub-inputs (e.g. an Ve variant where the surface Ve isn't
    there falls back to solid styling), but the availability
    predicate is stricter so "Select all available" only picks
    variants whose primary scientific content is renderable."""
    from . import render3d as _r3d

    def _check(ctx: FigureExportContext) -> bool:
        cfg = _r3d.VARIANTS.get(variant_id)
        if cfg is None:
            return False
        geom = ctx.geom
        if geom is None:
            return False
        # Region check — needs at least one of the requested tags.
        if cfg.regions:
            rs = (
                getattr(geom, "region_surfaces_viz", None)
                or getattr(geom, "region_surfaces", None)
                or {}
            )
            if not any(rs.get(t) is not None for t in cfg.regions):
                return False
        # Fibers — needed when cfg.fibers OR cfg.ve_on_fibers.
        if cfg.fibers or cfg.ve_on_fibers:
            fp = getattr(geom, "fiber_paths_raw", None)
            if fp is None or len(fp) == 0:
                return False
        # Per-fiber Ve.
        if cfg.ve_on_fibers:
            fpv = getattr(geom, "fiber_paths_Ve", None)
            if fpv is None or len(fpv) == 0:
                return False
        # Surface Ve (endo / epi).
        if cfg.ve_on_regions:
            if getattr(geom, "nerve_surface_Ve", None) is None:
                return False
        # E-field streamlines.
        if cfg.show_field_lines:
            if getattr(geom, "field_lines_poly", None) is None:
                return False
        # If the variant has NO renderable inputs at all (empty
        # regions, no fibers, no streamlines), refuse — would
        # produce a blank page.
        if (not cfg.regions
                and not cfg.fibers
                and not cfg.ve_on_fibers
                and not cfg.show_field_lines):
            return False
        return True

    return _check


# ---------------------------------------------------------------------------
# REGISTRY
# ---------------------------------------------------------------------------
# Ordered roughly along the user workflow (Cuff → Mesh → FEM →
# Fibers → Population → Sweep) so the bulk Exports tab reads top-to-
# bottom in the same direction as the analysis drawer.


REGISTRY: list[FigureSpec] = [
    # ---- Mesh quality (×2: surface tris on import, tets after build) ----
    FigureSpec(
        id="mesh.surface_quality_hist",
        title="Surface triangle quality histogram",
        category="Mesh",
        source="plotly_state",
        state_var="quality_hist_figure",
        availability=_plotly_available("quality_hist_figure"),
    ),
    FigureSpec(
        id="mesh.tet_quality_hist",
        title="Tet quality histogram",
        category="Mesh",
        source="plotly_state",
        state_var="mesh_quality_hist_figure",
        availability=_plotly_available(
            "mesh_quality_hist_figure",
        ),
    ),
    # ---- FEM ----
    FigureSpec(
        id="fem.axis_line",
        title="V_e / E_z along centerline",
        category="FEM",
        source="plotly_state",
        state_var="fem_axis_figure",
        availability=_plotly_available("fem_axis_figure"),
    ),
    FigureSpec(
        id="fem.slice_volume",
        title="V_e slice heatmap (per z)",
        category="FEM",
        source="plotly_state",
        state_var="fem_slice_figure",
        availability=_plotly_available("fem_slice_figure"),
    ),
    FigureSpec(
        id="fem.activation_fn",
        title="Activation function ∂²V_e/∂s²",
        category="FEM",
        source="plotly_state",
        state_var="fem_af_figure",
        availability=_plotly_available("fem_af_figure"),
    ),
    # ---- Single-fiber sim ----
    FigureSpec(
        id="fiber.pulse",
        title="Stimulus pulse waveform",
        category="Single fiber",
        source="plotly_state",
        state_var="fiber_pulse_figure",
        availability=_plotly_available("fiber_pulse_figure"),
    ),
    FigureSpec(
        id="fiber.propagation",
        title="V_m propagation heatmap",
        category="Single fiber",
        source="plotly_state",
        state_var="fiber_propagation_figure",
        availability=_plotly_available("fiber_propagation_figure"),
    ),
    FigureSpec(
        id="fiber.waterfall",
        title="V_m waterfall (sampled nodes)",
        category="Single fiber",
        source="plotly_state",
        state_var="fiber_waterfall_figure",
        availability=_plotly_available("fiber_waterfall_figure"),
    ),
    FigureSpec(
        id="fiber.cnap",
        title="Single-fiber cNAP contribution",
        category="Single fiber",
        source="plotly_state",
        state_var="fiber_cnap_figure",
        availability=_plotly_available("fiber_cnap_figure"),
    ),
    # ---- Population ----
    FigureSpec(
        id="pop.preset_preview",
        title="Population preset preview",
        category="Population",
        source="plotly_state",
        state_var="pop_preset_preview_figure",
        availability=_plotly_available("pop_preset_preview_figure"),
    ),
    FigureSpec(
        id="pop.xsec_cuff",
        title="Cross-section @ cuff center",
        category="Population",
        source="plotly_state",
        state_var="pop_xsec_cuff_figure",
        availability=_plotly_available("pop_xsec_cuff_figure"),
    ),
    FigureSpec(
        id="pop.kde",
        title="Diameter KDE per branch/row",
        category="Population",
        source="plotly_state",
        state_var="pop_kde_figure",
        availability=_plotly_available("pop_kde_figure"),
    ),
    FigureSpec(
        id="pop.xsec_activated",
        title="Cross-section with activated overlay",
        category="Population",
        source="plotly_state",
        state_var="pop_xsec_figure",
        availability=_plotly_available("pop_xsec_figure"),
    ),
    FigureSpec(
        id="pop.propagation",
        title="V_m propagation heatmap (pop)",
        category="Population",
        source="plotly_state",
        state_var="pop_propagation_figure",
        availability=_plotly_available("pop_propagation_figure"),
    ),
    FigureSpec(
        id="pop.waterfall",
        title="V_m waterfall (pop)",
        category="Population",
        source="plotly_state",
        state_var="pop_waterfall_figure",
        availability=_plotly_available("pop_waterfall_figure"),
    ),
    FigureSpec(
        id="pop.cnap",
        title="Population cNAP",
        category="Population",
        source="plotly_state",
        state_var="pop_cnap_figure",
        availability=_plotly_available("pop_cnap_figure"),
    ),
    # ---- Conductivity ----
    FigureSpec(
        id="conductivity.sigma_f",
        title="σ(f) (Cole-Cole)",
        category="Conductivity",
        source="plotly_state",
        state_var="cc_plot_figure",
        availability=_plotly_available("cc_plot_figure"),
    ),
    # ---- Sweep / threshold / recruitment ----
    FigureSpec(
        id="sweep.recruitment",
        title="Recruitment curve",
        category="Sweep",
        source="plotly_state",
        state_var="sweep_recruitment_figure",
        availability=_plotly_available("sweep_recruitment_figure"),
    ),
    FigureSpec(
        id="sweep.threshold_scatter",
        title="Threshold vs diameter",
        category="Sweep",
        source="plotly_state",
        state_var="sweep_threshold_figure",
        availability=_plotly_available("sweep_threshold_figure"),
    ),
    FigureSpec(
        id="sweep.activation_heatmap",
        title="Activation heatmap (fiber × amplitude)",
        category="Sweep",
        source="plotly_state",
        state_var="sweep_heatmap_figure",
        availability=_plotly_available("sweep_heatmap_figure"),
    ),
    # F3.2 — Selectivity. Veraart SI bar across configs at the
    # user-chosen amplitude; threshold-ratio table separately
    # (HTML, not a Plotly figure — exported as table-only via
    # the registry's HTML lane if/when one lands).
    FigureSpec(
        id="selectivity.bar",
        title="Selectivity (Veraart SI · per config)",
        category="Selectivity",
        source="plotly_state",
        state_var="selectivity_bar_figure",
        availability=_plotly_available("selectivity_bar_figure"),
    ),
    # I1 Phase A — DC impedance. Per-contact + per-pair bars
    # under the FEM category so they show alongside the slice
    # heatmap + axis plot in the Exports tab + Report.
    FigureSpec(
        id="fem.impedance_bar",
        title="Per-contact impedance (DC)",
        category="FEM",
        source="plotly_state",
        state_var="impedance_bar_figure",
        availability=_plotly_available("impedance_bar_figure"),
    ),
    FigureSpec(
        id="fem.impedance_per_pair",
        title="Per-pair impedance (DC)",
        category="FEM",
        source="plotly_state",
        state_var="impedance_per_pair_figure",
        availability=_plotly_available(
            "impedance_per_pair_figure",
        ),
    ),
    # ---- 3D viewport variants (F2.3.c Phase 2) ----
    FigureSpec(
        id="render3d.electrode_geom",
        title="Electrode geometry (3D)",
        category="3D · Electrode",
        source="render3d_variant",
        variant_id="render3d.electrode_geom",
        availability=_render3d_available("render3d.electrode_geom"),
    ),
    FigureSpec(
        id="render3d.electrode_polar",
        title="Electrode · anode/cathode (3D)",
        category="3D · Electrode",
        source="render3d_variant",
        variant_id="render3d.electrode_polar",
        availability=_render3d_available("render3d.electrode_polar"),
    ),
    FigureSpec(
        id="render3d.mesh_all_regions",
        title="Mesh · all regions (3D)",
        category="3D · Mesh",
        source="render3d_variant",
        variant_id="render3d.mesh_all_regions",
        availability=_render3d_available("render3d.mesh_all_regions"),
    ),
    FigureSpec(
        id="render3d.mesh_muscle",
        title="Mesh · muscle only (3D)",
        category="3D · Mesh",
        source="render3d_variant",
        variant_id="render3d.mesh_muscle",
        availability=_render3d_available("render3d.mesh_muscle"),
    ),
    FigureSpec(
        id="render3d.mesh_endo",
        title="Mesh · endoneurium only (3D)",
        category="3D · Mesh",
        source="render3d_variant",
        variant_id="render3d.mesh_endo",
        availability=_render3d_available("render3d.mesh_endo"),
    ),
    FigureSpec(
        id="render3d.mesh_epi",
        title="Mesh · epineurium only (3D)",
        category="3D · Mesh",
        source="render3d_variant",
        variant_id="render3d.mesh_epi",
        availability=_render3d_available("render3d.mesh_epi"),
    ),
    FigureSpec(
        id="render3d.mesh_cuff",
        title="Mesh · cuff only (3D)",
        category="3D · Mesh",
        source="render3d_variant",
        variant_id="render3d.mesh_cuff",
        availability=_render3d_available("render3d.mesh_cuff"),
    ),
    FigureSpec(
        id="render3d.fibers_epi",
        title="Fibers · with epi @ α=0.5 (3D)",
        category="3D · Fibers",
        source="render3d_variant",
        variant_id="render3d.fibers_epi",
        availability=_render3d_available("render3d.fibers_epi"),
    ),
    FigureSpec(
        id="render3d.fem_full",
        title="FEM · all regions visible (3D)",
        category="3D · FEM",
        source="render3d_variant",
        variant_id="render3d.fem_full",
        availability=_render3d_available("render3d.fem_full"),
    ),
    FigureSpec(
        id="render3d.fem_no_muscle",
        title="FEM · muscle hidden (3D)",
        category="3D · FEM",
        source="render3d_variant",
        variant_id="render3d.fem_no_muscle",
        availability=_render3d_available("render3d.fem_no_muscle"),
    ),
    FigureSpec(
        id="render3d.fem_no_epi",
        title="FEM · epineurium hidden (3D)",
        category="3D · FEM",
        source="render3d_variant",
        variant_id="render3d.fem_no_epi",
        availability=_render3d_available("render3d.fem_no_epi"),
    ),
    FigureSpec(
        id="render3d.fem_no_endo",
        title="FEM · endoneurium hidden (3D)",
        category="3D · FEM",
        source="render3d_variant",
        variant_id="render3d.fem_no_endo",
        availability=_render3d_available("render3d.fem_no_endo"),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_iso",
        title="Cuff · iso zoom (3D)",
        category="3D · Zoom",
        source="render3d_variant",
        variant_id="render3d.cuff_zoom_iso",
        availability=_render3d_available("render3d.cuff_zoom_iso"),
    ),
    FigureSpec(
        id="render3d.cuff_cross_section",
        title="Cuff · cross-section (3D)",
        category="3D · Zoom",
        source="render3d_variant",
        variant_id="render3d.cuff_cross_section",
        availability=_render3d_available("render3d.cuff_cross_section"),
    ),
    # ---- Additional electrode variants ----
    FigureSpec(
        id="render3d.electrode_with_saline",
        title="Electrode · with saline infill (3D)",
        category="3D · Electrode",
        source="render3d_variant",
        variant_id="render3d.electrode_with_saline",
        availability=_render3d_available(
            "render3d.electrode_with_saline",
        ),
    ),
    FigureSpec(
        id="render3d.electrode_in_nerve",
        title="Electrode · on epineurium (3D)",
        category="3D · Electrode",
        source="render3d_variant",
        variant_id="render3d.electrode_in_nerve",
        availability=_render3d_available(
            "render3d.electrode_in_nerve",
        ),
    ),
    # ---- Mesh quality coloured ----
    FigureSpec(
        id="render3d.mesh_quality_all",
        title="Mesh · quality (RdYlGn with edges) (3D)",
        category="3D · Mesh",
        source="render3d_variant",
        variant_id="render3d.mesh_quality_all",
        availability=_render3d_available(
            "render3d.mesh_quality_all",
        ),
    ),
    # ---- Geometry overviews ----
    FigureSpec(
        id="render3d.geometry_full",
        title="Geometry · full (3D)",
        category="3D · Geometry",
        source="render3d_variant",
        variant_id="render3d.geometry_full",
        availability=_render3d_available("render3d.geometry_full"),
    ),
    FigureSpec(
        id="render3d.geometry_no_muscle",
        title="Geometry · no muscle (3D)",
        category="3D · Geometry",
        source="render3d_variant",
        variant_id="render3d.geometry_no_muscle",
        availability=_render3d_available(
            "render3d.geometry_no_muscle",
        ),
    ),
    FigureSpec(
        id="render3d.fibers_in_nerve",
        title="Fibers · with endo + epi (3D)",
        category="3D · Fibers",
        source="render3d_variant",
        variant_id="render3d.fibers_in_nerve",
        availability=_render3d_available(
            "render3d.fibers_in_nerve",
        ),
    ),
    # ---- Ve overlays ----
    FigureSpec(
        id="render3d.ve_on_endo",
        title="FEM · V_e on endoneurium (3D)",
        category="3D · FEM Ve",
        source="render3d_variant",
        variant_id="render3d.ve_on_endo",
        availability=_render3d_available("render3d.ve_on_endo"),
    ),
    FigureSpec(
        id="render3d.ve_on_epi",
        title="FEM · V_e on epineurium (3D)",
        category="3D · FEM Ve",
        source="render3d_variant",
        variant_id="render3d.ve_on_epi",
        availability=_render3d_available("render3d.ve_on_epi"),
    ),
    FigureSpec(
        id="render3d.ve_on_fibers",
        title="FEM · V_e on fibers (3D)",
        category="3D · FEM Ve",
        source="render3d_variant",
        variant_id="render3d.ve_on_fibers",
        availability=_render3d_available("render3d.ve_on_fibers"),
    ),
    FigureSpec(
        id="render3d.ve_on_all",
        title="FEM · V_e on endo + epi + fibers (3D)",
        category="3D · FEM Ve",
        source="render3d_variant",
        variant_id="render3d.ve_on_all",
        availability=_render3d_available("render3d.ve_on_all"),
    ),
    # ---- E-field streamlines ----
    FigureSpec(
        id="render3d.field_streamlines",
        title="FEM · E-field streamlines (3D)",
        category="3D · E-field",
        source="render3d_variant",
        variant_id="render3d.field_streamlines",
        availability=_render3d_available(
            "render3d.field_streamlines",
        ),
    ),
    FigureSpec(
        id="render3d.field_streamlines_cuff_zoom",
        title="FEM · E-field streamlines (cuff zoom) (3D)",
        category="3D · E-field",
        source="render3d_variant",
        variant_id="render3d.field_streamlines_cuff_zoom",
        availability=_render3d_available(
            "render3d.field_streamlines_cuff_zoom",
        ),
    ),
    # ---- Cuff-zoom Ve + streamlines combos (9 variants) ----
    # Three Ve targets × three context conditions. Same iso
    # camera framed to the cuff bbox throughout, so a viewer
    # flipping through pages sees the same geometric region with
    # progressively-isolated overlays.
    FigureSpec(
        id="render3d.cuff_zoom_ve_epi_with_cuff_streamlines",
        title="Cuff zoom · V_e on epi + cuff + streamlines",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id=(
            "render3d.cuff_zoom_ve_epi_with_cuff_streamlines"
        ),
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_epi_with_cuff_streamlines",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_endo_with_cuff_streamlines",
        title="Cuff zoom · V_e on endo + cuff + streamlines",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id=(
            "render3d.cuff_zoom_ve_endo_with_cuff_streamlines"
        ),
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_endo_with_cuff_streamlines",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_fibers_with_cuff_streamlines",
        title="Cuff zoom · V_e on fibers + cuff + streamlines",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id=(
            "render3d.cuff_zoom_ve_fibers_with_cuff_streamlines"
        ),
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_fibers_with_cuff_streamlines",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_epi_streamlines_no_cuff",
        title="Cuff zoom · V_e on epi + streamlines (no cuff)",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id=(
            "render3d.cuff_zoom_ve_epi_streamlines_no_cuff"
        ),
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_epi_streamlines_no_cuff",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_endo_streamlines_no_cuff",
        title="Cuff zoom · V_e on endo + streamlines (no cuff)",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id=(
            "render3d.cuff_zoom_ve_endo_streamlines_no_cuff"
        ),
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_endo_streamlines_no_cuff",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_fibers_streamlines_no_cuff",
        title="Cuff zoom · V_e on fibers + streamlines (no cuff)",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id=(
            "render3d.cuff_zoom_ve_fibers_streamlines_no_cuff"
        ),
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_fibers_streamlines_no_cuff",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_epi_only",
        title="Cuff zoom · V_e on epi only (3D)",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id="render3d.cuff_zoom_ve_epi_only",
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_epi_only",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_endo_only",
        title="Cuff zoom · V_e on endo only (3D)",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id="render3d.cuff_zoom_ve_endo_only",
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_endo_only",
        ),
    ),
    FigureSpec(
        id="render3d.cuff_zoom_ve_fibers_only",
        title="Cuff zoom · V_e on fibers only (3D)",
        category="3D · Cuff zoom · V_e",
        source="render3d_variant",
        variant_id="render3d.cuff_zoom_ve_fibers_only",
        availability=_render3d_available(
            "render3d.cuff_zoom_ve_fibers_only",
        ),
    ),
]


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------


_BY_ID: dict[str, FigureSpec] = {spec.id: spec for spec in REGISTRY}


def get(fig_id: str) -> FigureSpec:
    """Look up a FigureSpec by id. Raises KeyError on miss — the
    caller usually has a hard-coded id, so a typo should fail loudly
    rather than silently degrade."""
    return _BY_ID[fig_id]


def by_category() -> dict[str, list[FigureSpec]]:
    """Group the registry by category, preserving in-registry order
    within each category. Used by the bulk Exports tab to render
    one section per category."""
    out: dict[str, list[FigureSpec]] = {}
    for spec in REGISTRY:
        out.setdefault(spec.category, []).append(spec)
    return out


def materialize(ctx: FigureExportContext, fig_id: str) -> Any:
    """Resolve a FigureSpec to an actual figure object — a Plotly
    `{data, layout}` dict, PNG bytes (render3d variant), or
    (in future "callable" sources) a matplotlib Figure.

    For `source == "plotly_state"`, returns a deepcopy of the live
    state dict so the caller can mutate it (e.g. apply a preset)
    without affecting the on-screen figure.

    For `source == "render3d_variant"`, calls
    golgi.figures.render3d.render_variant and returns the PNG
    bytes. The caller is responsible for embedding the bytes
    (e.g. via PIL/matplotlib) into the destination format.

    Raises ValueError on an unknown source kind so future additions
    to the enum fail loudly here rather than producing silently-
    empty exports."""
    spec = get(fig_id)
    if spec.source == "plotly_state":
        fig = getattr(ctx.state, spec.state_var, None)
        if fig is None:
            raise RuntimeError(
                f"state.{spec.state_var} is None — the figure has "
                f"not been built yet."
            )
        return copy.deepcopy(fig)
    if spec.source == "render3d_variant":
        from . import render3d as _r3d
        # The render3d module needs the styling constants
        # (DEFAULTS, GOLD_STYLE, BRANCH_PALETTE) which live in
        # app.py. ctx carries a `render3d_kwargs` dict populated
        # by build_app at registration time; this stays here as
        # the single chokepoint for those parameters.
        kwargs = getattr(ctx, "render3d_kwargs", None) or {}
        return _r3d.render_variant(
            spec.variant_id, ctx.geom,
            **kwargs,
        )
    raise ValueError(
        f"unknown FigureSpec.source {spec.source!r} for id {fig_id!r}"
    )
