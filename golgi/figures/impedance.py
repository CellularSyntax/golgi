# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""I1 Phase A — DC impedance figures.

Plotly bar charts driven by `state.fem_impedance` (loaded from
`<project>/configs/<cid>/impedance.json` after every FEM solve).
One per-contact bar group, one per-pair group, optionally
overlaid across multiple configs in the Compare panel.

I1 Phase B will add a Bode plot variant (log-frequency × log-|Z|)
once Cole-Cole frequency sweeps land.
"""
from __future__ import annotations

from typing import Any, Optional

import numpy as np

from .export import apply_preset_to_plotly_fig


__all__ = [
    "build_impedance_bar_figure",
    "build_impedance_per_pair_figure",
    "fmt_ohms",
]


def _maybe_apply_preset(
    fig: dict, preset: Optional[Any],
) -> dict:
    if preset is not None:
        apply_preset_to_plotly_fig(fig, preset)
    return fig


def fmt_ohms(z: float) -> str:
    """Compact |Z| formatter — picks Ω / kΩ / MΩ by magnitude.
    Shared by the bar charts (hover labels) and the Solve drawer
    summary chips so the units line up across surfaces."""
    if z is None or not np.isfinite(z):
        return "n/a"
    if z >= 1.0e6:
        return f"{z / 1.0e6:.2f} MΩ"
    if z >= 1.0e3:
        return f"{z / 1.0e3:.2f} kΩ"
    return f"{z:.1f} Ω"


def build_impedance_bar_figure(
    per_config_impedance: dict[str, dict],
    *,
    preset: Optional[Any] = None,
) -> dict:
    """Per-contact impedance bar chart.

    `per_config_impedance` shape:
        {cid: {
            "label":       <human-readable config label>,
            "per_contact": [
                {"id": int, "role": str, "Z_ohm": float,
                 "I_inj_A": float, "area_m2": float}, ...
            ],
        }}

    Renders one bar group per config, x-axis = contact id, y-axis
    = |Z| (Ω, log scale). Hover shows role + current + area.
    """
    cids = list(per_config_impedance.keys())
    # Collect the union of contact ids present across configs.
    all_ids = sorted({
        int(c.get("id", 0))
        for cfg in per_config_impedance.values()
        for c in cfg.get("per_contact", [])
    })
    if not all_ids:
        return {"data": [], "layout": {
            "title": {"text": "No access-impedance data yet — "
                              "solve FEM with 'Compute electrode "
                              "access impedance' on"},
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "margin": {"l": 40, "r": 16, "t": 40, "b": 40},
        }}

    traces: list[dict] = []
    for cid in cids:
        entry = per_config_impedance[cid]
        label = str(entry.get("label", cid))
        by_id = {
            int(c.get("id", 0)): c
            for c in entry.get("per_contact", [])
        }
        y_vals: list[float] = []
        hovers: list[str] = []
        for cid_int in all_ids:
            row = by_id.get(cid_int)
            if row is None:
                y_vals.append(float("nan"))
                hovers.append("n/a<extra></extra>")
                continue
            z = float(row.get("Z_ohm", float("nan")))
            I = float(row.get("I_inj_A", 0.0))
            role = str(row.get("role", "?"))
            y_vals.append(z if np.isfinite(z) else 0.0)
            hovers.append(
                f"<b>{label}</b><br>"
                f"contact {cid_int} ({role})<br>"
                f"Z_access = {fmt_ohms(z)}<br>"
                f"I = {I * 1e6:.3f} µA<extra></extra>"
            )
        traces.append({
            "type": "bar",
            "name": label,
            "x": [str(i) for i in all_ids],
            "y": y_vals,
            "hovertemplate": hovers,
        })

    fig: dict = {
        "data": traces,
        "layout": {
            "title": {
                "text": (
                    "Per-contact access impedance "
                    "(tissue spreading; excludes interface)"
                ),
                "x": 0.5, "xanchor": "center",
                "font": {"size": 13},
            },
            "xaxis": {
                "title": {
                    "text": "contact id",
                    "font": {"size": 11},
                },
                "tickfont": {"size": 10},
            },
            "yaxis": {
                "title": {
                    "text": "|Z_access| (Ω)",
                    "font": {"size": 11},
                },
                "type": "log",
                "tickfont": {"size": 10},
            },
            "barmode": "group",
            "margin": {"l": 60, "r": 16, "t": 44, "b": 50},
            "bargap": 0.18,
            "showlegend": len(cids) > 1,
            "legend": {
                "orientation": "h", "x": 0.5,
                "xanchor": "center", "y": -0.18,
                "font": {"size": 10},
            },
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
        },
    }
    return _maybe_apply_preset(fig, preset)


def build_impedance_per_pair_figure(
    per_config_impedance: dict[str, dict],
    *,
    preset: Optional[Any] = None,
) -> dict:
    """Per-pair impedance bar chart. X-axis = "anode → cathode"
    label, y-axis = |Z_pair| (Ω, log scale). One bar group per
    config when more than one is supplied (Compare panel)."""
    cids = list(per_config_impedance.keys())
    all_pairs = sorted({
        (int(p.get("anode", 0)), int(p.get("cathode", 0)))
        for cfg in per_config_impedance.values()
        for p in cfg.get("per_pair", [])
    })
    if not all_pairs:
        return {"data": [], "layout": {
            "title": {
                "text": "No per-pair access impedance — set at "
                        "least one anode/cathode polarity",
            },
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
            "xaxis": {"visible": False},
            "yaxis": {"visible": False},
            "margin": {"l": 40, "r": 16, "t": 40, "b": 40},
        }}
    pair_labels = [f"{a} → {c}" for (a, c) in all_pairs]

    traces: list[dict] = []
    for cid in cids:
        entry = per_config_impedance[cid]
        label = str(entry.get("label", cid))
        by_pair = {
            (int(p.get("anode", 0)),
             int(p.get("cathode", 0))): p
            for p in entry.get("per_pair", [])
        }
        y_vals: list[float] = []
        hovers: list[str] = []
        for pair in all_pairs:
            row = by_pair.get(pair)
            if row is None:
                y_vals.append(float("nan"))
                hovers.append("n/a<extra></extra>")
                continue
            z = float(row.get("Z_pair_ohm", float("nan")))
            y_vals.append(z if np.isfinite(z) else 0.0)
            hovers.append(
                f"<b>{label}</b><br>"
                f"anode {pair[0]} → cathode {pair[1]}<br>"
                f"Z_access,pair = {fmt_ohms(z)}<extra></extra>"
            )
        traces.append({
            "type": "bar",
            "name": label,
            "x": pair_labels,
            "y": y_vals,
            "hovertemplate": hovers,
        })

    fig: dict = {
        "data": traces,
        "layout": {
            "title": {
                "text": (
                    "Per-pair access impedance "
                    "(tissue spreading; excludes interface)"
                ),
                "x": 0.5, "xanchor": "center",
                "font": {"size": 13},
            },
            "xaxis": {
                "title": {
                    "text": "anode → cathode",
                    "font": {"size": 11},
                },
                "tickfont": {"size": 10},
            },
            "yaxis": {
                "title": {
                    "text": "|Z_access,pair| (Ω)",
                    "font": {"size": 11},
                },
                "type": "log",
                "tickfont": {"size": 10},
            },
            "barmode": "group",
            "margin": {"l": 60, "r": 16, "t": 44, "b": 50},
            "bargap": 0.18,
            "showlegend": len(cids) > 1,
            "legend": {
                "orientation": "h", "x": 0.5,
                "xanchor": "center", "y": -0.18,
                "font": {"size": 10},
            },
            "paper_bgcolor": "white",
            "plot_bgcolor": "white",
        },
    }
    return _maybe_apply_preset(fig, preset)
