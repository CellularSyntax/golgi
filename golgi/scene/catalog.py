# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""SceneCatalog — single source of truth for what renders, in
what frame, in what units.

Each renderable (nerve, per-design region, fiber bundle, FEM
overlay) is registered as a SceneEntry. The catalog applies a
central frame/unit transform — `to_pca_mm()` — so no entry can
silently end up in cuff frame when others are in PCA, or in m
when others are in mm.

See `docs/scene_frames.md` for the invariants this enforces and
`docs/scene_regression_checklist.md` for the test scenarios each
refactor phase must preserve.

Phase 1 (this commit) ships the schema + a dual-run adapter:
the catalog produces a parallel scene_state alongside the
existing `_set_*_group` functions, and any divergences are
logged to stdout. The renderer continues to consume the inline
scene_state during Phase 1, so behaviour is unchanged. Phases
2-5 cut over each section to the catalog; Phase 6 retires the
inline code and the comparison harness.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Literal

import numpy as np


# Frames that an entry's source data can live in. The catalog
# translates everything to PCA, millimetres, at viewport mount
# time.
SourceFrame = Literal["raw", "pca", "cuff_local"]
SourceUnits = Literal["m", "mm"]


@dataclass
class FrameContext:
    """The set of geom fields needed for a PCA conversion. Built
    once per `Catalog.apply()` call and passed to every entry's
    fold so they don't fish around in `geom` for these.
    """
    centroid: np.ndarray | None  # (3,) in metres
    R_global: np.ndarray | None  # (3, 3) PCA rotation

    def is_pca_ready(self) -> bool:
        return self.centroid is not None and self.R_global is not None


def to_pca_mm(
    pts: np.ndarray,
    source_frame: SourceFrame,
    source_units: SourceUnits,
    ctx: FrameContext,
    cuff_origin_pca_m: np.ndarray | None = None,
    M_design: np.ndarray | None = None,
) -> np.ndarray:
    """Convert a point array from its source frame/units to the
    viewport's pure-PCA frame, millimetres.

    This is the SINGLE chokepoint for unit / frame conversions in
    the scene. Bugs like "metres-scale offset added to mm-scale
    points" or "fiber undo used current cuff instead of FEM-time
    cuff" become structurally impossible because every renderable
    routes through here with its source frame and units declared
    by the entry, not inferred at the call site.

    Arguments:
      pts: (N, 3) point array in `source_units`.
      source_frame:
        - "raw"        → STL coords as loaded; convert via
                         `(p - centroid) @ R_global`.
        - "pca"        → already centroid-at-origin + R_global-
                         applied; passthrough.
        - "cuff_local" → PCA rotated by M_design and translated so
                         `cuff_origin_pca_m` is at zero; undo via
                         `p @ M_design.T + cuff_origin_pca_m`.
      source_units: "m" or "mm".
      ctx: FrameContext from `Catalog.apply()`.
      cuff_origin_pca_m: required iff source_frame == "cuff_local".
        The PCA-frame origin (metres) the cuff_local frame was
        defined against. MUST be the origin used at the time
        `pts` was computed — NOT `geom.cuff_origin_pca` blindly
        (that field tracks the last-fit cuff and can be stale
        for FEM outputs / multi-design renders).
      M_design: required iff source_frame == "cuff_local". The
        per-design rotation that took PCA → cuff-local.
    """
    pts = np.asarray(pts, dtype=np.float64)
    if source_units == "mm":
        pts_m = pts / 1000.0
    else:
        pts_m = pts
    if source_frame == "raw":
        if not ctx.is_pca_ready():
            # Best effort: hand back raw mm so the entry still
            # mounts somewhere visible. Real callers should check
            # `ctx.is_pca_ready()` upstream and skip.
            return pts_m * 1000.0
        pts_pca_m = (pts_m - ctx.centroid) @ ctx.R_global
    elif source_frame == "pca":
        pts_pca_m = pts_m
    elif source_frame == "cuff_local":
        if cuff_origin_pca_m is None or M_design is None:
            raise ValueError(
                "cuff_local frame requires both cuff_origin_pca_m "
                "and M_design"
            )
        pts_pca_m = (
            pts_m @ np.asarray(M_design, dtype=np.float64).T
            + np.asarray(cuff_origin_pca_m, dtype=np.float64)
        )
    else:
        raise ValueError(
            f"unknown source_frame: {source_frame!r}"
        )
    return pts_pca_m * 1000.0


# A SceneGroup matches the existing renderer's payload shape:
# {"payload": pv.PolyData | None, "style": dict, "visible": bool,
#  "signature": int}. The catalog produces these so the renderer
# can consume them unchanged once the cut-over completes.
SceneGroup = dict


@dataclass
class SceneEntry:
    """One renderable in the catalog.

    The `fold` callable runs once per scene-state rebuild,
    receives the geom + state + FrameContext, and returns a
    SceneGroup dict (or None for "no payload right now"). Folds
    MUST use `to_pca_mm` for any coordinate transform — that's
    the whole point of the catalog.

    `section` and `key` together address a slot in the existing
    scene_state structure (so Phase 1's dual-run can compare
    apples to apples):
      - section="nerve",      key="nerve"            → state["nerve"]
      - section="regions",    key=f"{eid}_{tag}"     → state["regions"][key]
      - section="fibers",     key=("branches", i)    → state["fibers"]["branches"][i]
      - section="fibers",     key="ve"               → state["fibers"]["ve"]
      - section="field",      key="tubes" | "arrows" → state["field"][key]
      - section="electrodes", key=(eid, part)        → state["electrodes"][eid][part]
    """
    section: Literal["nerve", "regions", "fibers", "field", "electrodes"]
    key: Any
    fold: Callable[[Any, Any, FrameContext], SceneGroup | None]
    label: str = ""
    color_rgb: tuple[float, float, float] | None = None


@dataclass
class Catalog:
    """Mutable registry of SceneEntry objects.

    Build once at startup (entries can be registered eagerly or
    lazily as designs are added). Call `apply()` on every scene-
    state rebuild to produce the parallel scene_state used by the
    Phase 1 dual-run, or by the renderer directly post-Phase-6.
    """
    entries: list[SceneEntry] = field(default_factory=list)

    def register(self, entry: SceneEntry) -> None:
        self.entries.append(entry)

    def clear_section(self, section: str) -> None:
        """Drop every entry in a section. Useful when re-registering
        per-design entries after `state.designs` changes."""
        self.entries = [
            e for e in self.entries if e.section != section
        ]

    def apply_in_place(
        self,
        scene_state: dict,
        geom,
        state,
        mkgrp_empty: Callable[[], SceneGroup],
    ) -> None:
        """Run every entry's fold and write the results directly
        into the existing `scene_state` dict. Only sections /
        slots that have a registered entry are touched; sections
        whose registered entry returns None get cleared (whole-
        section entries) or replaced with an empty group (per-
        slot entries). Other sections are left untouched so any
        legacy inline `_set_*_group` co-existing with the catalog
        keeps owning them.

        Used by `_rebuild_scene_state_real` post-Phase-6a as the
        sole writer for sections that have been ported. The
        Phase 1 `apply()` (which builds a parallel dict for
        dual-run comparison) is no longer called.
        """
        ctx = FrameContext(
            centroid=geom.centroid,
            R_global=geom.R_global,
        )
        # Track which whole-section keys appeared so we can wipe
        # any per-slot stale entries first. For per-slot entries,
        # we wipe each slot's prior content as we write.
        whole_sections: set[str] = set()
        for entry in self.entries:
            if entry.key == "*":
                whole_sections.add(entry.section)
        # Reset whole-section slots before populating so stale
        # entries (e.g. a previous design's electrodes) drop out.
        for sec in whole_sections:
            if sec == "regions":
                scene_state["regions"] = {}
            elif sec == "electrodes":
                scene_state["electrodes"] = {}
            elif sec == "field":
                scene_state["field"] = {
                    "tubes": mkgrp_empty(),
                    "arrows": mkgrp_empty(),
                }
        for entry in self.entries:
            try:
                g = entry.fold(geom, state, ctx)
            except Exception as ex:
                print(
                    f"[catalog] fold failed for "
                    f"{entry.section}/{entry.key}: {ex}",
                    flush=True,
                )
                continue
            if g is None:
                g = mkgrp_empty()
            self._assign(scene_state, entry.section, entry.key, g)

    def apply(
        self,
        geom,
        state,
        mkgrp_empty: Callable[[], SceneGroup],
    ) -> dict:
        """Run every entry's fold and assemble a scene_state-
        shaped dict. Empty sections still get the right structural
        scaffolding so `compare_scene_states` can address them.
        """
        ctx = FrameContext(
            centroid=geom.centroid,
            R_global=geom.R_global,
        )
        out: dict = {
            "nerve": mkgrp_empty(),
            "regions": {},
            "fibers": {
                "mode": "off",
                "branches": {},
                "ve": mkgrp_empty(),
                "pop_types": {},
                "selected": {},
            },
            "field": {
                "tubes": mkgrp_empty(),
                "arrows": mkgrp_empty(),
            },
            "electrodes": {},
        }
        for entry in self.entries:
            try:
                g = entry.fold(geom, state, ctx)
            except Exception as ex:
                print(
                    f"[catalog] fold failed for "
                    f"{entry.section}/{entry.key}: {ex}",
                    flush=True,
                )
                continue
            if g is None:
                g = mkgrp_empty()
            self._assign(out, entry.section, entry.key, g)
        return out

    @staticmethod
    def _assign(
        state: dict, section: str, key: Any, group: SceneGroup | dict,
    ) -> None:
        # Special "whole-section" key — the fold returns a dict
        # shaped like the section's full content, and we wholesale
        # replace the section. Useful for sections (electrodes,
        # regions) where the per-(design, part) groups are built
        # together by an existing helper that returns them as a
        # nested dict, so splitting into per-key entries would
        # duplicate work.
        if key == "*":
            if section == "nerve":
                state["nerve"] = group  # type: ignore[assignment]
            elif section == "regions":
                state["regions"] = (
                    group if isinstance(group, dict) else {}
                )
            elif section == "fibers":
                # Fibers has nested structure (mode/branches/ve/
                # pop_types/selected). The "*" entry's group must
                # match that shape.
                if isinstance(group, dict):
                    state["fibers"].update(group)
            elif section == "field":
                if isinstance(group, dict):
                    state["field"].update(group)
            elif section == "electrodes":
                state["electrodes"] = (
                    group if isinstance(group, dict) else {}
                )
            return
        if section == "nerve":
            state["nerve"] = group  # type: ignore[assignment]
        elif section == "regions":
            state["regions"][key] = group
        elif section == "fibers":
            if isinstance(key, tuple) and len(key) == 2 and key[0] == "branches":
                state["fibers"]["branches"][key[1]] = group
            elif key == "ve":
                state["fibers"]["ve"] = group
            elif isinstance(key, tuple) and len(key) == 2 and key[0] == "pop_types":
                state["fibers"]["pop_types"][key[1]] = group
            elif isinstance(key, tuple) and len(key) == 2 and key[0] == "selected":
                state["fibers"]["selected"][key[1]] = group
        elif section == "field":
            state["field"][key] = group
        elif section == "electrodes":
            if isinstance(key, tuple) and len(key) == 2:
                eid, part = key
                state["electrodes"].setdefault(eid, {})[part] = group


def compare_scene_states(
    inline: dict,
    catalog: dict,
    *,
    atol_mm: float = 0.05,
    sections_to_check: tuple[str, ...] | None = None,
) -> list[str]:
    """Diff inline (legacy) and catalog (refactor) scene-state
    dicts. Returns a list of human-readable divergence strings.

    Phase 1's catalog has ZERO entries registered, so the catalog
    side is empty — `sections_to_check` lets us scope the
    comparison to ONLY sections the catalog has actually started
    populating, avoiding "presence mismatch" noise from the
    not-yet-ported sections. As Phase 2-5 add entries, expand
    `sections_to_check` to include those sections.

    Per-slot checks:
      - Both sides have a payload, or both don't.
      - Visibility flag matches.
      - Payload point-cloud centroid (mm) matches within
        `atol_mm` (default 0.05 mm = 50 µm).
    Skips the `signature` field (always differs between runs).
    """
    diffs: list[str] = []
    sections = sections_to_check or ()

    def _check(name: str, a: SceneGroup | None, b: SceneGroup | None) -> None:
        if a is None and b is None:
            return
        if (a is None) != (b is None):
            diffs.append(
                f"{name}: slot presence mismatch "
                f"(inline={a is not None}, catalog={b is not None})"
            )
            return
        a_payload = a.get("payload")
        b_payload = b.get("payload")
        if (a_payload is None) != (b_payload is None):
            diffs.append(
                f"{name}: payload presence mismatch "
                f"(inline={a_payload is not None}, "
                f"catalog={b_payload is not None})"
            )
            return
        if bool(a.get("visible", True)) != bool(b.get("visible", True)):
            diffs.append(
                f"{name}: visible mismatch "
                f"(inline={a.get('visible')}, "
                f"catalog={b.get('visible')})"
            )
        if a_payload is not None and b_payload is not None:
            try:
                a_pts = np.asarray(a_payload.points, dtype=np.float64)
                b_pts = np.asarray(b_payload.points, dtype=np.float64)
            except Exception:
                return
            if a_pts.shape != b_pts.shape:
                diffs.append(
                    f"{name}: point-count mismatch "
                    f"(inline={len(a_pts)}, catalog={len(b_pts)})"
                )
                return
            if len(a_pts) > 0:
                a_c = a_pts.mean(axis=0)
                b_c = b_pts.mean(axis=0)
                d = float(np.linalg.norm(a_c - b_c))
                if d > atol_mm:
                    diffs.append(
                        f"{name}: centroid drift {d:.3f} mm "
                        f"(inline={a_c}, catalog={b_c})"
                    )

    if "nerve" in sections:
        _check("nerve", inline.get("nerve"), catalog.get("nerve"))
    if "regions" in sections:
        a_regions = inline.get("regions", {}) or {}
        b_regions = catalog.get("regions", {}) or {}
        for k in sorted(set(a_regions) | set(b_regions),
                          key=str):
            _check(
                f"regions[{k}]",
                a_regions.get(k), b_regions.get(k),
            )
    if "fibers" in sections:
        a_fibers = inline.get("fibers", {}) or {}
        b_fibers = catalog.get("fibers", {}) or {}
        a_branches = a_fibers.get("branches", {}) or {}
        b_branches = b_fibers.get("branches", {}) or {}
        for k in sorted(set(a_branches) | set(b_branches),
                          key=str):
            _check(
                f"fibers.branches[{k}]",
                a_branches.get(k), b_branches.get(k),
            )
        _check(
            "fibers.ve",
            a_fibers.get("ve"), b_fibers.get("ve"),
        )
    if "field" in sections:
        a_field = inline.get("field", {}) or {}
        b_field = catalog.get("field", {}) or {}
        for k in ("tubes", "arrows"):
            _check(
                f"field.{k}",
                a_field.get(k), b_field.get(k),
            )
    if "electrodes" in sections:
        a_elecs = inline.get("electrodes", {}) or {}
        b_elecs = catalog.get("electrodes", {}) or {}
        for eid in sorted(set(a_elecs) | set(b_elecs), key=str):
            a_parts = a_elecs.get(eid, {}) or {}
            b_parts = b_elecs.get(eid, {}) or {}
            for part in sorted(set(a_parts) | set(b_parts),
                                  key=str):
                _check(
                    f"electrodes[{eid}].{part}",
                    a_parts.get(part), b_parts.get(part),
                )
    return diffs
