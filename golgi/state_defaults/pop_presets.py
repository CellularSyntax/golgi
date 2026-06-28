# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Curated fiber-population presets for the Population tab (F1.1).

A `PopPreset` is a per-nerve, per-species library entry: it carries one
or more `PopBranchTemplate`s (one per anatomical sub-branch of the nerve)
and each template carries a list of `PopRow`s describing one fiber
sub-population (myelinated A-α, B, unmyelinated C, …) with a mean
diameter, std, and population fraction.

The values bundled here are starting points drawn from the cited
literature; they are deliberately editable after apply — the user is
expected to refine them for the specific preparation. Picking a preset
removes the "what diameter for B-fibres again?" lookup that students
otherwise get wrong, and standardises the mixture across studies that
cite the same source.

Row shape matches what the Population-tab UI in `golgi/app.py` builds
client-side at row-add time:
    {id, name, backend, model, mean_um, std_um, frac, color}
so `apply_preset` produces rows the existing JS update-handlers can
edit in place without any schema reconciliation.
"""
from __future__ import annotations

import math
import secrets
from dataclasses import dataclass, field
from typing import Sequence


# Backends + model strings must match the registries defined in
# `golgi/app.py` (MYELINATED_MODELS / UNMYELINATED_MODELS). We do not
# import from app.py to avoid a heavyweight import on the slim state-
# defaults module — the values are checked at apply time and a row
# whose model is not recognised by the backend will be flagged in
# the UI rather than silently failing the population sim.
_MYELINATED = (
    "MRG_INTERPOLATION", "MRG_DISCRETE",
    "SMALL_MRG_INTERPOLATION", "SWEENEY",
)
# Thio autonomic/cutaneous fibers are UNMYELINATED (Thio et al. 2024); keeping
# them here also forces the PyFibers backend (the AxonML MRG surrogate cannot
# simulate them) via the _UNMYELINATED check below.
_UNMYELINATED = (
    "SUNDT", "TIGERHOLM", "RATTAY", "SCHILD94", "SCHILD97",
    "THIO_AUTONOMIC", "THIO_CUTANEOUS",
)


@dataclass(frozen=True)
class PopRow:
    """One fiber sub-population inside a branch template.

    `name` is a free-text label (typically the conventional class —
    'A-alpha', 'B', 'C unmyelinated' — so the user can recognise it
    in the per-row KDE legend). `model` must be one of the strings in
    the app's MYELINATED_MODELS + UNMYELINATED_MODELS registry."""
    name: str
    model: str
    mean_um: float
    std_um: float
    frac: float          # percent within the branch, 0..100
    backend: str = "pyfibers"


@dataclass(frozen=True)
class PopBranchTemplate:
    """Per-anatomical-branch rows. `match_labels` is the set of branch
    labels this template should attach to when the user applies the
    preset; matching is case-insensitive substring against the live
    `state.pop_branches_meta[*].label`. The first template whose
    match_labels matches is picked; if none match, `apply_preset`
    falls back to `default_template` (the one with `match_labels=()`)."""
    rows: tuple[PopRow, ...]
    match_labels: tuple[str, ...] = ()    # () = default fallback


@dataclass(frozen=True)
class PopPreset:
    """One curated preset entry. `templates` is ordered; the first
    matching template wins for each detected branch."""
    name: str             # registry key, kebab/snake case
    label: str            # display name in the dropdown
    species: str
    nerve: str
    citation: str         # short attribution shown under the dropdown
    notes: str            # 1-line caveat / scope note
    templates: tuple[PopBranchTemplate, ...]


# ---------------------------------------------------------------------------
# Curated presets
# ---------------------------------------------------------------------------
# Numbers are starting points from the cited sources. Vagal counts in
# particular vary widely across reports (Soltanpour & Santer 1996 vs
# Hoffman & Schnitzlein 1961 vs Verlinden 2016) — defaults here favour
# the more recent quantitative histology where available. Encourage
# users to cite both the preset AND the source they actually relied on.

POP_PRESETS: dict[str, PopPreset] = {}


def _register(p: PopPreset) -> None:
    POP_PRESETS[p.name] = p


_register(PopPreset(
    name="cervical_vagus_human",
    label="Cervical vagus — human",
    species="Homo sapiens",
    nerve="Cervical vagus nerve",
    citation=(
        "Soltanpour & Santer 1996, J Anat; "
        "Verlinden et al. 2016, Auton Neurosci"
    ),
    notes=(
        "Strongly C-fibre dominated by count (~80 %). MRG-family "
        "models do not cover C; assign SUNDT/Tigerholm for those rows."
    ),
    templates=(
        PopBranchTemplate(
            rows=(
                PopRow("A-alpha (motor/large myelinated)",
                       "MRG_INTERPOLATION", 12.0, 2.0, 3.0),
                PopRow("A-beta (mid myelinated)",
                       "MRG_INTERPOLATION", 8.0, 1.5, 7.0),
                PopRow("A-delta (small myelinated)",
                       "SMALL_MRG_INTERPOLATION", 3.0, 0.8, 5.0),
                PopRow("B (preganglionic)",
                       "SMALL_MRG_INTERPOLATION", 2.0, 0.5, 5.0),
                PopRow("C (unmyelinated)",
                       "SUNDT", 0.8, 0.3, 80.0),
            ),
        ),
    ),
))


_register(PopPreset(
    name="cervical_vagus_pig",
    label="Cervical vagus — pig",
    species="Sus scrofa domesticus",
    nerve="Cervical vagus nerve",
    citation=(
        "Settell et al. 2020, J Neural Eng; "
        "Nicolai et al. 2020, J Neural Eng"
    ),
    notes=(
        "Pig VN has multiple distinct fascicles; the same per-branch "
        "mixture is a coarse starting point. Refine per fascicle if "
        "your geometry resolves them."
    ),
    templates=(
        PopBranchTemplate(
            rows=(
                PopRow("Large myelinated (A-alpha/beta)",
                       "MRG_INTERPOLATION", 8.0, 2.0, 10.0),
                PopRow("Small myelinated (A-delta/B)",
                       "SMALL_MRG_INTERPOLATION", 3.0, 0.8, 15.0),
                PopRow("C (unmyelinated)",
                       "SUNDT", 0.8, 0.2, 75.0),
            ),
        ),
    ),
))


_register(PopPreset(
    name="cervical_vagus_rat",
    label="Cervical vagus — rat",
    species="Rattus norvegicus",
    nerve="Cervical vagus nerve",
    citation=(
        "Prechtl & Powley 1990, Anat Embryol; "
        "Pelot et al. 2020, Front Neurosci"
    ),
    notes=(
        "Rat VN counts dominated by small B + C; A-fibres are sparse. "
        "Diameter ranges scale down vs human."
    ),
    templates=(
        PopBranchTemplate(
            rows=(
                PopRow("A-alpha/beta",
                       "MRG_INTERPOLATION", 5.0, 1.5, 5.0),
                PopRow("A-delta",
                       "SMALL_MRG_INTERPOLATION", 2.0, 0.5, 10.0),
                PopRow("B",
                       "SMALL_MRG_INTERPOLATION", 1.5, 0.3, 10.0),
                PopRow("C (unmyelinated)",
                       "SUNDT", 0.5, 0.2, 75.0),
            ),
        ),
    ),
))


_register(PopPreset(
    name="recurrent_laryngeal_human",
    label="Recurrent laryngeal — human",
    species="Homo sapiens",
    nerve="Recurrent laryngeal nerve",
    citation="Mu & Sanders 2009, Anat Rec",
    notes=(
        "Predominantly large myelinated motor; thyroarytenoid branch "
        "in particular is A-alpha dominated. Adjust if simulating the "
        "sensory subset of RLN."
    ),
    templates=(
        PopBranchTemplate(
            rows=(
                PopRow("A-alpha (motor)",
                       "MRG_INTERPOLATION", 14.0, 2.0, 50.0),
                PopRow("A-beta (sensory)",
                       "MRG_INTERPOLATION", 8.0, 2.0, 30.0),
                PopRow("C (sensory)",
                       "SUNDT", 0.8, 0.2, 20.0),
            ),
        ),
    ),
))


_register(PopPreset(
    name="generic_myelinated_A",
    label="Generic — A-alpha myelinated only",
    species="—",
    nerve="—",
    citation="Standard textbook values",
    notes=(
        "Single A-alpha distribution. Useful as a sanity-check "
        "baseline before reaching for a species-specific preset."
    ),
    templates=(
        PopBranchTemplate(
            rows=(
                PopRow("A-alpha",
                       "MRG_INTERPOLATION", 12.0, 2.0, 100.0),
            ),
        ),
    ),
))


_register(PopPreset(
    name="generic_unmyelinated_C",
    label="Generic — C unmyelinated only",
    species="—",
    nerve="—",
    citation="Standard textbook values",
    notes=(
        "Single C distribution. Drives sympathetic/visceral activation "
        "studies that ignore the myelinated population."
    ),
    templates=(
        PopBranchTemplate(
            rows=(
                PopRow("C (unmyelinated)",
                       "SUNDT", 0.8, 0.3, 100.0),
            ),
        ),
    ),
))


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------


def preset_dropdown_items() -> list[dict]:
    """Items for the Vuetify VSelect — `(value, title)` shape. Includes a
    leading 'none' sentinel so the user can dismiss the selection without
    triggering an apply."""
    items: list[dict] = [
        {"value": "", "title": "— choose a preset —"},
    ]
    for p in POP_PRESETS.values():
        items.append({"value": p.name, "title": p.label})
    return items


def preset_meta(name: str) -> dict:
    """Citation + notes payload for the UI sub-text. Returns empty
    strings for the sentinel '' value so the UI can `v_show` against
    `pop_preset_choice` without a separate has-meta flag."""
    p = POP_PRESETS.get(name)
    if p is None:
        return {"label": "", "species": "", "nerve": "",
                "citation": "", "notes": ""}
    return {
        "label": p.label, "species": p.species, "nerve": p.nerve,
        "citation": p.citation, "notes": p.notes,
    }


# ---------------------------------------------------------------------------
# Apply: preset → state.pop_branch_types
# ---------------------------------------------------------------------------


def _match_template(preset: PopPreset,
                     branch_label: str) -> PopBranchTemplate:
    """Pick the per-branch template whose `match_labels` matches the
    detected branch label (case-insensitive substring). Falls back to
    the first template with empty match_labels, else just the first
    template. Currently most presets ship a single default template;
    the match_labels machinery is in place for future presets that
    distinguish e.g. cardiac vs pulmonary branches."""
    if not preset.templates:
        raise ValueError(
            f"preset {preset.name!r} has no templates",
        )
    label_lc = (branch_label or "").lower()
    # First pass: explicit-label match.
    for t in preset.templates:
        for tag in t.match_labels:
            if tag and tag.lower() in label_lc:
                return t
    # Second pass: default fallback (empty match_labels).
    for t in preset.templates:
        if not t.match_labels:
            return t
    # Last resort: first template.
    return preset.templates[0]


def apply_preset(
    preset_name: str,
    branches_meta: Sequence[dict],
    *,
    tab10_palette: Sequence[str],
) -> dict[str, list[dict]]:
    """Map a preset onto the live `pop_branches_meta` and return a new
    `pop_branch_types` dict in the shape the UI expects. Each row gets
    a fresh `id` and a TAB10 colour assigned by global row index so the
    visual identity is stable across the viewport / KDE / chip lookups.

    Caller is responsible for assigning the result to
    `state.pop_branch_types` and invalidating `pop_generated`."""
    preset = POP_PRESETS.get(preset_name)
    if preset is None:
        return {}
    out: dict[str, list[dict]] = {}
    global_idx = 0
    palette_n = max(1, len(tab10_palette))
    for meta in branches_meta:
        idx = meta.get("idx")
        if idx is None:
            continue
        label = str(meta.get("label", ""))
        template = _match_template(preset, label)
        rows_out: list[dict] = []
        for r in template.rows:
            backend = "pyfibers" if r.model in _UNMYELINATED else r.backend
            rows_out.append({
                "id": secrets.token_hex(4),
                "name": r.name,
                "backend": backend,
                "model": r.model,
                "mean_um": float(r.mean_um),
                "std_um": float(r.std_um),
                "frac": float(r.frac),
                "color": tab10_palette[global_idx % palette_n],
            })
            global_idx += 1
        out[str(int(idx))] = rows_out
    return out


# ---------------------------------------------------------------------------
# Preview KDE figure (Plotly dict)
# ---------------------------------------------------------------------------


def _gaussian_pdf(x: list[float], mu: float, sigma: float) -> list[float]:
    """Vectorless normal PDF — preset previews are tiny (≤ 200 pts) so
    a Python loop is fine and avoids pulling numpy into a slim module
    that may be imported headless."""
    if sigma <= 0.0:
        return [0.0 for _ in x]
    inv_two_var = 1.0 / (2.0 * sigma * sigma)
    norm = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    return [norm * math.exp(-((xi - mu) ** 2) * inv_two_var) for xi in x]


def build_preview_kde_figure(preset_name: str) -> dict:
    """Plotly figure dict — one mixture-curve per branch template, with
    each row's Gaussian (scaled by its fraction) overlaid. Used by the
    'Preview' tile under the dropdown before the user commits via Apply.

    Returns the same empty-shape dict as figures/util._plotly_placeholder
    when the preset is unknown / empty, so the trame widget can render
    it without any conditional logic on the caller side."""
    preset = POP_PRESETS.get(preset_name)
    if preset is None or not preset.templates:
        return {
            "data": [],
            "layout": {
                "annotations": [{
                    "text": ("Choose a preset above to see the "
                             "expected diameter mixture."),
                    "x": 0.5, "y": 0.5,
                    "xref": "paper", "yref": "paper",
                    "showarrow": False,
                    "font": {"size": 12, "color": "#888a90"},
                }],
                "xaxis": {"visible": False},
                "yaxis": {"visible": False},
                "margin": {"l": 30, "r": 20, "t": 20, "b": 30},
                "paper_bgcolor": "rgba(255,255,255,0)",
                "plot_bgcolor": "rgba(255,255,255,0)",
                "height": 180,
            },
        }
    # Sample diameters from 0.1 µm up to the max (mu + 4σ) across all
    # rows so the tail of the largest distribution is visible.
    rows: list[PopRow] = []
    for t in preset.templates:
        rows.extend(t.rows)
    if not rows:
        return build_preview_kde_figure("")
    d_max = max(r.mean_um + 4.0 * r.std_um for r in rows)
    d_max = max(d_max, 1.0)
    n_grid = 200
    step = (d_max - 0.1) / float(n_grid - 1)
    x = [0.1 + i * step for i in range(n_grid)]
    traces: list[dict] = []
    # Per-row trace, scaled by frac/100 so visually the area under
    # each curve reflects its mixture weight. Sum trace overlays.
    sum_y = [0.0] * n_grid
    for r in rows:
        y = _gaussian_pdf(x, r.mean_um, r.std_um)
        w = max(0.0, r.frac) / 100.0
        y_scaled = [yi * w for yi in y]
        for i in range(n_grid):
            sum_y[i] += y_scaled[i]
        traces.append({
            "type": "scatter",
            "x": x, "y": y_scaled,
            "mode": "lines",
            "line": {"width": 1.2},
            "name": f"{r.name}  ({r.frac:.0f} %)",
            "hovertemplate": (
                f"{r.name}<br>d = %{{x:.2f}} µm<extra></extra>"
            ),
        })
    traces.append({
        "type": "scatter",
        "x": x, "y": sum_y,
        "mode": "lines",
        "line": {"color": "#1f2024", "width": 2.0, "dash": "dot"},
        "name": "mixture (sum)",
        "hovertemplate": (
            "mixture<br>d = %{x:.2f} µm<extra></extra>"
        ),
    })
    layout = {
        "title": {
            "text": f"Preview · {preset.label}",
            "font": {"size": 12, "color": "#1f2024"},
            "x": 0.02, "xanchor": "left",
        },
        "xaxis": {
            "title": {
                "text": "fiber diameter  (µm)",
                "font": {"size": 11, "color": "#1f2024"},
            },
            "tickfont": {"size": 10, "color": "#4a4a52"},
            "showgrid": True, "gridcolor": "#e5e5ea",
            "range": [0.0, d_max],
        },
        "yaxis": {
            "title": {
                "text": "fraction-weighted density",
                "font": {"size": 11, "color": "#1f2024"},
            },
            "tickfont": {"size": 10, "color": "#4a4a52"},
            "showgrid": True, "gridcolor": "#e5e5ea",
        },
        "margin": {"l": 56, "r": 10, "t": 28, "b": 42},
        "paper_bgcolor": "white",
        "plot_bgcolor": "rgba(248,249,251,0.6)",
        "legend": {
            "orientation": "h",
            "x": 0.0, "y": -0.28,
            "font": {"size": 10, "color": "#4a4a52"},
        },
        "height": 220,
    }
    return {"data": traces, "layout": layout}
