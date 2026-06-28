# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F3.2 — selectivity Plotly figure builders.

Two figures land here for the Compare panel:

* `build_selectivity_bar_figure(...)`: vertical bar chart, one
  bar per config, height = Veraart SI at the user-chosen
  amplitude. Green ⇒ target-selective (positive), red ⇒
  off-target-selective (negative). Hover shows the per-config
  R_target / R_offtarget numbers + the config label.

* `build_threshold_ratio_table(...)`: small HTML table — one row
  per config, columns are (config name, target threshold,
  off-target threshold, ratio). Rendered via `v_html` in the
  Compare panel; not a Plotly figure but lives here so all the
  selectivity rendering is in one place.

Inputs are pre-computed numpy arrays from
`pipeline/selectivity.py` (compute_veraart_si /
compute_threshold_stats_per_branch). Keeps this module free of
the heavy math + easy to unit-test by building dummy dicts.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .export import apply_preset_to_plotly_fig


__all__ = [
    "build_selectivity_bar_figure",
    "build_threshold_ratio_table",
]


def _maybe_apply_preset(
    fig: dict, preset: Optional[object],
) -> dict:
    if preset is not None:
        apply_preset_to_plotly_fig(fig, preset)
    return fig


def _si_to_colour(si: float) -> str:
    """RdYlGn mapping: SI=−1 red, 0 grey, +1 green. Cheap
    piecewise interpolation — no plotly colourscale lookup
    needed since this is per-bar styling."""
    si = float(max(-1.0, min(1.0, si)))
    if si >= 0:
        # 0 → grey, +1 → green
        r = int(180 + (16 - 180) * si)
        g = int(180 + (155 - 180) * si)
        b = int(180 + (75 - 180) * si)
    else:
        # 0 → grey, −1 → red
        t = -si
        r = int(180 + (220 - 180) * t)
        g = int(180 + (60 - 180) * t)
        b = int(180 + (60 - 180) * t)
    return f"rgb({r}, {g}, {b})"


def build_selectivity_bar_figure(
    per_config_si: dict[str, dict],
    *,
    target_branch_label: str = "target",
    offtarget_label: str = "off-target",
    amplitude_mA: Optional[float] = None,
    preset: Optional[object] = None,
) -> dict:
    """Bar chart of Veraart SI per config at one amplitude.

    `per_config_si` shape:
        {cid: {
            "label":       <human-readable config label>,
            "si":          float SI at `amplitude_mA`,
            "R_target":    float fraction (0..1) at amplitude,
            "R_offtarget": float fraction (0..1) at amplitude,
        }}

    The amplitude is rendered in the figure title so the user
    knows which slice of the per-amplitude SI curve they're
    looking at.
    """
    cids = list(per_config_si.keys())
    labels = [
        str(per_config_si[c].get("label", c)) for c in cids
    ]
    sis = [
        float(per_config_si[c].get("si", 0.0)) for c in cids
    ]
    colours = [_si_to_colour(s) for s in sis]
    hovers = []
    for c in cids:
        e = per_config_si[c]
        Rt = float(e.get("R_target", 0.0))
        Ro = float(e.get("R_offtarget", 0.0))
        si = float(e.get("si", 0.0))
        hovers.append(
            f"<b>{e.get('label', c)}</b><br>"
            f"SI = {si:+.3f}<br>"
            f"R({target_branch_label}) = {Rt*100:.1f}%<br>"
            f"R({offtarget_label}) = {Ro*100:.1f}%"
            "<extra></extra>"
        )
    title_amp = (
        f" at I = {amplitude_mA:.3g} mA"
        if amplitude_mA is not None else ""
    )
    fig: dict = {
        "data": [
            {
                "type": "bar",
                "x": labels,
                "y": sis,
                "marker": {
                    "color": colours,
                    "line": {"color": "#222222", "width": 0.5},
                },
                "hovertemplate": hovers,
                "name": "SI",
            },
        ],
        "layout": {
            "title": {
                "text": (
                    f"Veraart selectivity — target = "
                    f"<b>{target_branch_label}</b>"
                    f"{title_amp}"
                ),
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 13},
            },
            "xaxis": {
                "title": {"text": "configuration",
                          "font": {"size": 11}},
                "tickfont": {"size": 10},
            },
            "yaxis": {
                "title": {
                    "text": (
                        "SI  (target − off) / (target + off)"
                    ),
                    "font": {"size": 11},
                },
                "range": [-1.0, 1.0],
                "zeroline": True,
                "zerolinecolor": "#888888",
                "zerolinewidth": 1.2,
                "tickfont": {"size": 10},
                "tickvals": [-1.0, -0.5, 0.0, 0.5, 1.0],
            },
            "shapes": [
                # Reference line at SI = 0 (no selectivity).
                {
                    "type": "line",
                    "xref": "paper", "yref": "y",
                    "x0": 0.0, "x1": 1.0,
                    "y0": 0.0, "y1": 0.0,
                    "line": {
                        "color": "#888888",
                        "width": 1.2,
                        "dash": "dot",
                    },
                },
            ],
            "annotations": [
                {
                    "xref": "paper", "yref": "paper",
                    "x": 1.0, "y": 1.06,
                    "xanchor": "right", "yanchor": "bottom",
                    "text": (
                        "+1 = target only · "
                        "0 = equal · "
                        "−1 = off-target only"
                    ),
                    "showarrow": False,
                    "font": {
                        "size": 9, "color": "#666666",
                    },
                },
            ],
            "margin": {"l": 60, "r": 16, "t": 56, "b": 56},
            "bargap": 0.25,
            "showlegend": False,
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
        },
    }
    return _maybe_apply_preset(fig, preset)


def build_threshold_ratio_table(
    per_config_thresholds: dict[str, dict],
    *,
    target_branch_label: str = "target",
    offtarget_label: str = "off-target",
) -> str:
    """HTML table of threshold-based selectivity per config.

    `per_config_thresholds` shape:
        {cid: {
            "label":           <human-readable label>,
            "T_target_uA":     float (NaN if none activated),
            "T_offtarget_uA":  float (NaN if none, +inf if avoided),
            "ratio":           float (T_off / T_target),
            "n_target":        int,
            "n_offtarget":     int,
        }}

    Returns a self-contained HTML string. The Compare panel
    binds it via `v_html=("selectivity_table_html",)`.
    """
    def _fmt_uA(v: float) -> str:
        if v is None or not np.isfinite(v):
            if v == float("inf"):
                return "—"
            return "n/a"
        return f"{v:.0f} µA"

    def _fmt_ratio(v: float) -> str:
        if v is None or np.isnan(v):
            return "n/a"
        if not np.isfinite(v):
            return "∞ (no off-target activation)"
        return f"{v:.2f}×"

    rows: list[str] = []
    rows.append(
        "<thead><tr>"
        "<th style='text-align:left;'>config</th>"
        f"<th>T({target_branch_label})</th>"
        f"<th>T({offtarget_label})</th>"
        "<th>ratio T<sub>off</sub> / T<sub>target</sub></th>"
        "<th>n target / off</th>"
        "</tr></thead>"
    )
    body_lines: list[str] = []
    for cid, e in per_config_thresholds.items():
        ratio = float(e.get("ratio", float("nan")))
        # Colour the ratio cell to make the "good > 1" intent
        # obvious without forcing the user to read every number.
        if not np.isfinite(ratio) and ratio == float("inf"):
            ratio_colour = "#16a34a"
        elif np.isnan(ratio):
            ratio_colour = "#888888"
        elif ratio >= 2.0:
            ratio_colour = "#16a34a"
        elif ratio >= 1.0:
            ratio_colour = "#7c7d2c"
        else:
            ratio_colour = "#dc2626"
        body_lines.append(
            "<tr>"
            f"<td style='text-align:left; padding:4px 8px;'>"
            f"{e.get('label', cid)}</td>"
            f"<td style='padding:4px 8px;'>"
            f"{_fmt_uA(e.get('T_target_uA'))}</td>"
            f"<td style='padding:4px 8px;'>"
            f"{_fmt_uA(e.get('T_offtarget_uA'))}</td>"
            f"<td style='padding:4px 8px; color:{ratio_colour}; "
            f"font-weight:600;'>{_fmt_ratio(ratio)}</td>"
            f"<td style='padding:4px 8px; color:#666;'>"
            f"{int(e.get('n_target', 0))} / "
            f"{int(e.get('n_offtarget', 0))}</td>"
            "</tr>"
        )
    table = (
        "<table style='border-collapse:collapse; "
        "font-size:12px; width:100%; "
        "background:#fafafa; "
        "border:1px solid #e6e6e8; "
        "border-radius:6px; overflow:hidden;'>"
        + "".join(rows)
        + "<tbody>"
        + "".join(body_lines)
        + "</tbody></table>"
    )
    return table
