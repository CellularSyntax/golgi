# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""V1 Phase A.3b + A.4 — Segment-µCT dialog action handlers.

Handlers wire the dialog to the `golgi.segmentation` backend
and to per-project persistence under `<project>/uct/`.

  - do_open_segment_uct_dialog       open + try to restore an
                                     existing segmentation from
                                     <project>/uct/
                                     segmentation.json
  - do_close_segment_uct_dialog      close (keeps loaded stack so
                                     re-opening doesn't re-read)
  - do_load_uct_stack                read metadata from
                                     state.uct_file_path (any
                                     supported format), render
                                     middle slice
  - do_run_uct_segmentation          call segmenter.propose_all on
                                     the current slice, stash
                                     proposals + meta
  - do_label_uct_proposal(idx, lbl)  set a proposal's label,
                                     re-render overlay + counts
  - do_save_uct_segmentation         write segmentation.json +
                                     labels_slice<N>.png under
                                     <project>/uct/; emit audit
                                     event

A state.change watcher on `uct_slice_idx` re-renders the preview
lazily when the user moves the scrubber.

Stack + per-slice numpy arrays + the resolved Segmenter live in
a closure dict so they don't pollute Trame state (which msgpack-
encodes everything for the WebSocket). Only display strings +
JSON-safe metadata go through state.
"""
from __future__ import annotations

# OpenMP duplicate-init workaround for macOS conda envs where
# multiple libraries link against their own libomp.dylib:
# PyTorch (loaded by sam2 during Segment) and scipy/skimage
# (loaded by do_refine_masks below) both bring one. Without
# this env var, the second one to initialise aborts with
#   OMP: Error #15: Initializing libomp.dylib, but found
#   libomp.dylib already initialized.
# crashing the whole process. KMP_DUPLICATE_LIB_OK is the
# Intel-OMP-blessed escape hatch — it disables the duplicate
# check, allowing both runtimes to coexist. In practice this
# is safe for our use case (single-threaded mask refinement;
# no cross-runtime contention with torch's thread pool). Must
# be set BEFORE the offending libraries import their native
# extensions; module load time is the earliest reliable point.
# `setdefault` preserves any explicit user override.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import asyncio
import json
import traceback
from pathlib import Path
from typing import Callable, Optional

import numpy as np

from golgi.segmentation import (
    Stack,
    color_to_hex,
    compose_overlay,
    generate_proposal_colors,
    load_stack,
    read_slice,
    resolve_segmenter,
    to_data_url,
    to_display,
)
from golgi.segmentation.segmenter import MaskProposal
from golgi.segmentation import reconstruct3d as r3d


# Persistence schema versioning — bump when we change the
# segmentation.json layout in an incompatible way.
SEGMENTATION_SCHEMA = "v2"


# Label → integer index in the saved labels PNG. Must be a
# closed enumeration; new labels need a schema bump.
_LABEL_INDEX = {
    "background": 0,
    "discard":    0,    # treated identically downstream
    "epi":        1,
    "fascicle":   2,
    "unlabeled":  255,  # transient; never written to disk
}
_INDEX_LABEL = {
    0: "background",
    1: "epi",
    2: "fascicle",
}


def register(
    state,
    *,
    get_active_project_dir: Optional[
        Callable[[], Path]
    ] = None,
    on_recon_meshes_ready: Optional[
        Callable[[list], None]
    ] = None,
    cancel_token=None,
) -> dict[str, Callable]:
    """Wire the segment-µCT handlers.

    `on_recon_meshes_ready` is the bridge to the Step-3 PyVista
    plotter — app.py registers a callback that builds actors,
    updates the legend chip items, and refreshes the quality
    histogram. The action layer only knows that meshes are
    ready; it does not own the plotter itself, so the heavy
    VTK state stays out of the closure cache here.
    """

    # Mutable closure cache — keeps the heavy objects (Stack
    # handle, raw slice numpy array, Segmenter, MaskProposal
    # list) off the Trame state proxy. Wrapped in a dict so
    # individual handlers can reset its keys.
    _ctx: dict = {
        "stack": None,            # type: Stack | None
        "slice_arr_full": None,   # full slice (raw, native dtype)
        "slice_disp_full": None,  # full slice (uint8 stretched)
        "slice_arr": None,        # cropped slice (raw)
        "slice_disp": None,       # cropped slice (uint8)
        # CLAHE-enhanced cropped slice — lazy; computed on the
        # first render with uct_clahe=True and invalidated
        # whenever slice_disp changes (slice scrub / crop).
        "slice_disp_clahe": None,
        "crop": (0, 0, 0, 0),     # (x0, y0, x1, y1) inclusive
        # The CURRENT slice's working state — gets swapped in
        # and out of `per_slice` on scrolls so each slice
        # keeps its own proposals + labels.
        "current_slice_idx": -1,
        "proposals": [],
        "labels": [],
        "proposal_colors": [],
        # Per-slice cache. Each entry is {proposals, labels,
        # colors}. Filled by the stack-wide Segment pass; the
        # slice-change handler swaps the current slice's
        # entry into the working `proposals`/`labels`/
        # `proposal_colors` keys above. Crop change clears
        # everything since the per-slice masks live in cropped-
        # local coordinates.
        "per_slice": {},          # dict[int, dict]
        "segmenter": None,
    }

    def _ensure_segmenter() -> None:
        """Lazy-resolve the segmenter on first segmentation
        call so dialog open is fast even when MedSAM2 weights
        are present (which take a few seconds to load into a
        torch module). Honours the user's backend choice from
        state.uct_backend_choice; flipping the choice clears
        the cache (see _on_backend_choice_change) so a fresh
        backend can be loaded."""
        if _ctx["segmenter"] is not None:
            return
        _fallback: list[str] = []
        prefer = str(
            getattr(state, "uct_backend_choice", "auto") or "auto",
        )
        seg = resolve_segmenter(
            prefer=prefer,
            on_fallback=lambda r: _fallback.append(r),
        )
        _ctx["segmenter"] = seg
        with state:
            state.uct_segmenter_name = seg.name
            state.uct_segmenter_warning = (
                _fallback[0] if _fallback else ""
            )

    def _on_backend_choice_change(
        uct_backend_choice, **_kw,
    ) -> None:
        """Drop the cached segmenter so the next
        do_run_uct_segmentation rebuilds with the new backend.
        Clears warning + name so the engine chip reflects the
        impending switch (gets repopulated on next propose)."""
        _ctx["segmenter"] = None
        with state:
            state.uct_segmenter_name = ""
            state.uct_segmenter_warning = ""

    def _on_clahe_toggle(uct_clahe, **_kw) -> None:
        """Re-render the preview when the user toggles CLAHE
        so they immediately see the enhanced (or raw) view.
        The CLAHE result is cached in _ctx['slice_disp_clahe']
        so toggling on→off→on doesn't recompute."""
        _rerender_overlay()

    def _on_zoom_change(
        uct_zoom_x_range, uct_zoom_y_range, **_kw,
    ) -> None:
        """Display zoom toggled — re-render the slice through
        the new viewport. The crop + proposals are untouched."""
        _rerender_overlay()

    def _invalidate_finalize() -> None:
        """Mark the saved segmentation as stale. Called after
        any user action that changes labels / masks (paint,
        erase, chip click, active-stamp click, refine, re-
        segment). Gates the Step-2 Next button so the user
        re-runs Finalize before advancing."""
        if getattr(state, "uct_step2_finalized", False):
            with state:
                state.uct_step2_finalized = False

    def _read_2d_cleanup_params() -> tuple[int, int, int]:
        """Read the 2D-cleanup state vars and return them as a
        (min_component_px, min_hole_px, closing_radius_px)
        tuple. Defensive int-cast since the VTextField bindings
        can emit string values when the user types into them."""
        def _to_int(v, fallback=0):
            try:
                return max(0, int(v))
            except (TypeError, ValueError):
                return int(fallback)
        return (
            _to_int(getattr(
                state, "uct_recon_clean_min_component_px", 0,
            )),
            _to_int(getattr(
                state, "uct_recon_clean_min_hole_px", 0,
            )),
            _to_int(getattr(
                state, "uct_recon_clean_closing_radius_px", 0,
            )),
        )

    def _apply_2d_cleanup_to_proposals(
        proposals: list,
        labels: list,
        colors: Optional[list] = None,
    ) -> tuple[list, list, Optional[list]]:
        """Run `cleanup_2d_mask` on each proposal's binary mask
        and rebuild the MaskProposal. Drops proposals (and their
        paired labels / colors) whose mask becomes empty after
        cleanup (typical when `min_component_px` filters out a
        tiny noise blob).

        Applied per-proposal rather than per-class because the
        segmenter / video-propagator returns one MaskProposal per
        connected blob — running cleanup at this stage gives the
        user a clean overlay immediately, and the downstream
        reconstruct step (which still applies a class-level
        cleanup at extrude time) becomes a safety net rather
        than the primary cleanup pass.

        No-ops when every cleanup knob is 0.
        """
        mc, mh, cr = _read_2d_cleanup_params()
        if mc <= 0 and mh <= 0 and cr <= 0:
            return proposals, labels, colors
        out_props: list = []
        out_labels: list = []
        out_colors: Optional[list] = (
            [] if colors is not None else None
        )
        for i, p in enumerate(proposals):
            try:
                cleaned = r3d.cleanup_2d_mask(
                    p.mask.astype(bool),
                    min_component_px=mc,
                    min_hole_px=mh,
                    closing_radius_px=cr,
                )
            except Exception as ex:                       # noqa: BLE001
                print(
                    f"[segment] cleanup_2d_mask failed on "
                    f"proposal {i}: {ex} — keeping raw mask",
                    flush=True,
                )
                out_props.append(p)
                out_labels.append(
                    labels[i] if i < len(labels) else "unlabeled",
                )
                if out_colors is not None and colors is not None:
                    out_colors.append(
                        colors[i] if i < len(colors) else None,
                    )
                continue
            cleaned = cleaned.astype(bool)
            if not cleaned.any():
                # Whole proposal filtered out — drop label /
                # colour in lock-step.
                continue
            ys, xs = np.where(cleaned)
            bbox = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
            out_props.append(MaskProposal(
                mask=cleaned,
                score=float(getattr(p, "score", 0.0) or 0.0),
                bbox=bbox,
                area_px=int(cleaned.sum()),
                meta=dict(getattr(p, "meta", {}) or {}),
            ))
            out_labels.append(
                labels[i] if i < len(labels) else "unlabeled",
            )
            if out_colors is not None and colors is not None:
                out_colors.append(
                    colors[i] if i < len(colors) else None,
                )
        return out_props, out_labels, out_colors

    def _rebuild_meta_after_mutation() -> None:
        """Recompute per-proposal colours + chip meta after
        paint/erase / Generate-Epi mutates the proposal list.
        Centralised so the (n changes, recolour, rebuild meta,
        push state) dance only lives in one place.

        Side-effect: clears the Step-2 Finalize flag, so the
        Next button greys back out until the user re-runs
        Finalize.
        """
        n = len(_ctx["proposals"])
        colors = generate_proposal_colors(n)
        _ctx["proposal_colors"] = colors
        meta = [
            {
                "idx": i,
                "area_px": int(p.area_px),
                "bbox_str": (
                    f"({p.bbox[0]},{p.bbox[1]})-"
                    f"({p.bbox[2]},{p.bbox[3]})"
                ),
                "label": _ctx["labels"][i],
                "color_hex": color_to_hex(colors[i]),
            }
            for i, p in enumerate(_ctx["proposals"])
        ]
        with state:
            state.uct_proposals_meta = meta
        _refresh_label_counts()
        _rerender_overlay()
        _invalidate_finalize()

    def _on_paint_payload(uct_paint_payload, **_kw) -> None:
        """Paint or erase a brush stroke.

        Payload layout (flat array):
            [is_paint, ts, x0, y0, x1, y1, ..., xN, yN]
          • is_paint: 1 = paint, 0 = erase
          • ts: timestamp_ms (sentinel 0 → ignore)
          • (x_i, y_i): stroke point pairs in FULL-IMAGE pixel
            coords. Single-click gestures send 1 point; drags
            send the start, mousemove samples, and the end.

        We union a brush circle at every point into a single
        stroke mask, then apply it once. For paint, the stroke
        mask grows the LARGEST existing proposal of
        `state.uct_paint_label` (or creates a fresh one). For
        erase, the stroke mask is subtracted from every
        proposal — empty proposals get dropped.
        """
        if (
            not uct_paint_payload
            or len(uct_paint_payload) < 4
        ):
            return
        try:
            is_paint = bool(int(uct_paint_payload[0]))
            ts = int(uct_paint_payload[1])
        except (TypeError, ValueError, IndexError):
            return
        if ts == 0:
            return
        if _ctx.get("slice_disp") is None:
            return
        crop = _ctx.get("crop", (0, 0, 0, 0))
        try:
            cx0, cy0 = int(crop[0]), int(crop[1])
        except (TypeError, ValueError, IndexError):
            return
        h, w = _ctx["slice_disp"].shape[:2]
        try:
            radius = max(
                1, int(getattr(state, "uct_brush_radius", 12)),
            )
        except (TypeError, ValueError):
            radius = 12

        # Extract local-frame stroke points; clip to slice.
        points: list[tuple[int, int]] = []
        for i in range(2, len(uct_paint_payload) - 1, 2):
            try:
                px = int(uct_paint_payload[i]) - cx0
                py = int(uct_paint_payload[i + 1]) - cy0
            except (TypeError, ValueError, IndexError):
                continue
            if 0 <= px < w and 0 <= py < h:
                points.append((px, py))
        if not points:
            return

        # Union of brush circles at every stroke point. Each
        # disk is broadcast-evaluated against the full grid
        # then OR'd into the stroke mask. For long strokes
        # this is O(N_points × W × H) but with bool masks at
        # the cropped-slice size (typically < 2 MP) it's fast
        # — a 50-point stroke on a 1 MP slice is ~20 ms.
        yy, xx = np.ogrid[:h, :w]
        brush = np.zeros((h, w), dtype=bool)
        r2 = radius * radius
        for (px, py) in points:
            brush |= (yy - py) ** 2 + (xx - px) ** 2 <= r2
        if is_paint:
            target_label = str(
                getattr(state, "uct_paint_label", "fascicle"),
            )
            # Find largest existing proposal of target_label
            # so successive paint strokes grow it rather than
            # creating a new mask per click.
            target_idx = -1
            target_area = -1
            for i, lab in enumerate(_ctx["labels"]):
                if lab == target_label:
                    a = int(_ctx["proposals"][i].area_px)
                    if a > target_area:
                        target_area = a
                        target_idx = i
            if target_idx >= 0:
                p = _ctx["proposals"][target_idx]
                m = p.mask | brush
                ys_n, xs_n = np.where(m)
                bbox = (
                    int(xs_n.min()), int(ys_n.min()),
                    int(xs_n.max()), int(ys_n.max()),
                )
                _ctx["proposals"][target_idx] = MaskProposal(
                    mask=m, score=p.score,
                    bbox=bbox, area_px=int(m.sum()),
                )
            else:
                ys_n, xs_n = np.where(brush)
                bbox = (
                    int(xs_n.min()), int(ys_n.min()),
                    int(xs_n.max()), int(ys_n.max()),
                )
                _ctx["proposals"].append(MaskProposal(
                    mask=brush.copy(), score=1.0,
                    bbox=bbox, area_px=int(brush.sum()),
                ))
                _ctx["labels"].append(target_label)
        else:
            # Erase: subtract brush from every mask, drop
            # proposals that go empty.
            keep_props: list[MaskProposal] = []
            keep_labels: list[str] = []
            for p, lab in zip(
                _ctx["proposals"], _ctx["labels"],
            ):
                m = p.mask & ~brush
                if not m.any():
                    continue
                ys_n, xs_n = np.where(m)
                bbox = (
                    int(xs_n.min()), int(ys_n.min()),
                    int(xs_n.max()), int(ys_n.max()),
                )
                keep_props.append(MaskProposal(
                    mask=m, score=p.score,
                    bbox=bbox, area_px=int(m.sum()),
                ))
                keep_labels.append(lab)
            _ctx["proposals"] = keep_props
            _ctx["labels"] = keep_labels
        _refresh_epi_if_present()
        _rebuild_meta_after_mutation()

    # Pixel tolerance for click-to-label hit detection. 15 px
    # in image-pixel space — at the ~1024-wide display the
    # browser typically shows, that's ~7-8 screen pixels of
    # slop, which keeps very small (1-5 px) proposals
    # clickable even when the cursor lands a couple of image
    # pixels off the actual mask. Combined with the Tier-2
    # unlabeled-first preference (see `_on_click_payload`),
    # the generous tolerance can't accidentally pull a
    # neighbouring fascicle.
    _CLICK_HIT_RADIUS = 15

    def _on_click_payload(uct_click_payload, **_kw) -> None:
        """Click on the image preview → assign
        `state.uct_active_label` to the proposal under the
        cursor. Multi-tier hit detection prioritises what the
        user most likely meant:

          Tier 1 (direct hits) — proposals whose mask is True
          at the EXACT click pixel. Smallest area wins, so
          nested structures (fascicle inside epi outline)
          correctly pick the most-specific match. The
          auto-derived epi proposal is excluded from this
          tier so a click on an unlabelled region picks the
          unlabelled small proposal sitting on top of the
          epi mask, NOT the giant epi mask itself.

          Tier 2 (within tolerance) — if no direct hit, look
          within ±_CLICK_HIT_RADIUS pixels of the click.
          UNLABELED proposals are preferred over already-
          classified ones at this stage: in label mode the
          user is almost always trying to ASSIGN a class to
          an un-coloured blob, not re-classify a neighbouring
          fascicle whose mask happens to extend a couple of
          pixels closer. Without this preference, tiny blobs
          nestled against a fascicle become un-clickable —
          every tolerance click silently pulls the fascicle.
          Within each priority tier we then sort by distance,
          area as tiebreaker.

          Tier 3 (auto-epi only) — if no direct or tolerance
          hit on any user-affirmative proposal, fall back to
          the auto-derived epi (so it CAN still be relabelled
          when the user actually means to). Closest pixel of
          the auto-epi wins, but in practice that's the click
          point itself since auto-epi covers everything else.

        Hit search uses each proposal's bbox (not a fixed
        click-box) so very tiny proposals are still picked up
        when they're within tolerance but their mask doesn't
        intersect a small click-box around the cursor.

        Only fires when `state.uct_tool_mode == "label"`.
        """
        if not uct_click_payload:
            return
        try:
            x = int(uct_click_payload[0])
            y = int(uct_click_payload[1])
            ts = int(uct_click_payload[2])
        except (TypeError, ValueError, IndexError):
            return
        if ts == 0:
            return
        mode = str(
            getattr(state, "uct_tool_mode", "") or "",
        )
        if mode != "label":
            return
        if not _ctx.get("proposals"):
            return
        crop = _ctx.get("crop", (0, 0, 0, 0))
        try:
            cx0, cy0 = int(crop[0]), int(crop[1])
        except (TypeError, ValueError, IndexError):
            return
        local_x = x - cx0
        local_y = y - cy0
        disp = _ctx.get("slice_disp")
        if disp is None:
            return
        h, w = disp.shape[:2]
        if not (0 <= local_x < w and 0 <= local_y < h):
            return
        r = _CLICK_HIT_RADIUS
        r2 = r * r
        # Collect candidates, splitting auto-epi out so it
        # only fires as a last-resort fallback. For each
        # candidate compute (is_direct, distance, area).
        # Entries: (direct, dist, area, idx).
        normal: list[tuple[bool, float, int, int]] = []
        epi: list[tuple[bool, float, int, int]] = []
        for i, p in enumerate(_ctx["proposals"]):
            if (
                p.mask.shape[0] != h
                or p.mask.shape[1] != w
            ):
                continue
            # Fast bbox-distance reject: if the click is
            # > r pixels from the proposal's bbox in either
            # axis, the whole mask is out of range.
            bx0, by0, bx1, by1 = p.bbox
            dx_box = max(0, max(bx0 - local_x, local_x - bx1))
            dy_box = max(0, max(by0 - local_y, local_y - by1))
            if dx_box * dx_box + dy_box * dy_box > r2:
                continue
            direct = bool(p.mask[local_y, local_x])
            if direct:
                dist = 0.0
            else:
                # Search the proposal's full mask within its
                # bbox (clipped to the slice). Cheaper than
                # scanning a click-box and finds tiny far
                # masks the click-box would miss.
                y_lo_b = max(0, int(by0))
                y_hi_b = min(h, int(by1) + 1)
                x_lo_b = max(0, int(bx0))
                x_hi_b = min(w, int(bx1) + 1)
                sub = p.mask[y_lo_b:y_hi_b, x_lo_b:x_hi_b]
                if not sub.any():
                    continue
                ys_s, xs_s = np.where(sub)
                dy = (y_lo_b + ys_s) - local_y
                dx = (x_lo_b + xs_s) - local_x
                d2 = (dy * dy + dx * dx).min()
                if d2 > r2:
                    continue
                dist = float(np.sqrt(d2))
            entry = (direct, dist, int(p.area_px), i)
            if (
                _ctx["labels"][i] == "epi"
                and _is_auto_epi(p)
            ):
                epi.append(entry)
            else:
                normal.append(entry)
        # Tier 1: direct hit on a non-auto-epi proposal.
        direct_normal = [e for e in normal if e[0]]
        if direct_normal:
            # smallest area wins
            direct_normal.sort(key=lambda e: e[2])
            hit_idx = direct_normal[0][3]
        elif normal:
            # Tier 2: within tolerance. Unlabeled wins over
            # classified — see docstring for why. Distance
            # then area as tiebreakers within each group.
            def _tier2_key(
                e: tuple[bool, float, int, int],
            ) -> tuple[int, float, int]:
                is_unlabeled = (
                    _ctx["labels"][e[3]] == "unlabeled"
                )
                return (0 if is_unlabeled else 1, e[1], e[2])
            normal.sort(key=_tier2_key)
            hit_idx = normal[0][3]
        elif epi:
            # Tier 3: only auto-epi is under the cursor — let
            # the user relabel it if they want.
            epi.sort(key=lambda e: (e[1], e[2]))
            hit_idx = epi[0][3]
        else:
            return
        new_lab = str(
            getattr(state, "uct_active_label", "fascicle")
            or "fascicle"
        )
        labels = list(_ctx["labels"])
        labels[hit_idx] = new_lab
        _ctx["labels"] = labels
        # Same fresh-flag bookkeeping as the per-chip handler
        # below: clicking with the "None" active stamp must
        # hide the proposal on the overlay; any other stamp
        # restores the default rendering.
        try:
            prop = _ctx["proposals"][hit_idx]
            if not hasattr(prop, "meta") or prop.meta is None:
                prop.meta = {}
            if new_lab == "unlabeled":
                prop.meta["fresh"] = False
            else:
                prop.meta.pop("fresh", None)
        except (IndexError, AttributeError):
            pass
        # If an epi proposal exists, the label change invalidates
        # its mask — re-derive so the green tint always reflects
        # the CURRENT fascicle + background set. _recompute_epi
        # mutates _ctx, so the meta rebuild below picks up the
        # new epi proposal.
        if _epi_exists():
            _recompute_epi()
            _rebuild_meta_after_mutation()
            return
        meta = list(getattr(state, "uct_proposals_meta", []))
        if hit_idx < len(meta):
            meta[hit_idx] = {
                **meta[hit_idx],
                "label": new_lab,
            }
        with state:
            state.uct_proposals_meta = meta
        _refresh_label_counts()
        _rerender_overlay()

    def _rerender_overlay() -> None:
        """Compose the labelled overlay PNG for the current
        slice + proposals + labels, push as a base64 data URL
        the dialog's <img> reads.

        Two cropping concepts:

          • Data crop (uct_crop_x/y_range): persistent across
            slice scrolls; bound to `_ctx["slice_disp"]` and
            the proposal mask dimensions. This is what the
            segmenter sees.
          • Display zoom (uct_zoom_x/y_range): ephemeral; just
            slices the rendered image (and the masks) further
            so the user can inspect detail without re-running
            the segmenter. Reset zoom = back to data-crop view.

        When `state.uct_clahe` is True, renders on the CLAHE-
        enhanced base. Cached in `_ctx["slice_disp_clahe"]`."""
        if _ctx["slice_disp"] is None:
            with state:
                state.uct_overlay_url = ""
                state.uct_image_orig_w = 0
                state.uct_image_orig_h = 0
            return
        base_img = _ctx["slice_disp"]
        if bool(getattr(state, "uct_clahe", False)):
            if _ctx.get("slice_disp_clahe") is None:
                _ctx["slice_disp_clahe"] = _apply_clahe(
                    base_img,
                )
            base_img = _ctx["slice_disp_clahe"]

        # Resolve the display zoom (if any). uct_zoom_*_range
        # is in full-image coords; translate to data-crop-
        # local coords before slicing base_img + the proposal
        # masks.
        render_img = base_img
        render_props = _ctx["proposals"]
        z_view = _resolve_zoom_view()
        if z_view is not None:
            (zx0, zy0, zx1, zy1) = z_view
            render_img = base_img[zy0:zy1 + 1, zx0:zx1 + 1]
            # Slice every proposal mask to the same view so the
            # tints land on the right pixels. Bboxes are
            # recomputed (purely for the chip readout).
            sliced: list[MaskProposal] = []
            for p in _ctx["proposals"]:
                m = p.mask[zy0:zy1 + 1, zx0:zx1 + 1]
                if m.any():
                    ys, xs = np.where(m)
                    bbox = (
                        int(xs.min()), int(ys.min()),
                        int(xs.max()), int(ys.max()),
                    )
                else:
                    bbox = (0, 0, 0, 0)
                sliced.append(MaskProposal(
                    mask=m, score=p.score,
                    bbox=bbox, area_px=int(m.sum()),
                ))
            render_props = sliced

        # Record the slice's native pixel dimensions (post-
        # crop + zoom, pre-PNG-downsample) so the dialog can
        # compute a physical-scale bar. The bar-width-as-
        # percentage-of-displayed-image-width only depends on
        # this and `uct_voxel_size_um`, so it stays correct
        # regardless of how the browser scales the PNG.
        _orig_h, _orig_w = render_img.shape[:2]

        png = compose_overlay(
            render_img,
            render_props,
            _ctx["labels"],
            proposal_colors=_ctx.get("proposal_colors") or None,
            max_width=1024,
        )
        url = to_data_url(png)
        with state:
            state.uct_overlay_url = url
            state.uct_image_orig_w = int(_orig_w)
            state.uct_image_orig_h = int(_orig_h)

    def _resolve_zoom_view() -> Optional[tuple[int, int, int, int]]:
        """Translate state.uct_zoom_x/y_range (image-space) to
        cropped-slice-local coords + clip. Returns None when
        no zoom is set (range collapsed) or zoom equals crop.
        """
        if _ctx.get("slice_disp") is None:
            return None
        zx = getattr(state, "uct_zoom_x_range", None) or [0, 0]
        zy = getattr(state, "uct_zoom_y_range", None) or [0, 0]
        try:
            zx0, zx1 = int(zx[0]), int(zx[1])
            zy0, zy1 = int(zy[0]), int(zy[1])
        except (TypeError, ValueError, IndexError):
            return None
        if zx1 <= zx0 + 4 or zy1 <= zy0 + 4:
            return None
        crop = _ctx.get("crop", (0, 0, 0, 0))
        cx0, cy0 = int(crop[0]), int(crop[1])
        h, w = _ctx["slice_disp"].shape[:2]
        # full-image → cropped-local
        lx0 = max(0, min(w - 1, zx0 - cx0))
        ly0 = max(0, min(h - 1, zy0 - cy0))
        lx1 = max(0, min(w - 1, zx1 - cx0))
        ly1 = max(0, min(h - 1, zy1 - cy0))
        if lx1 <= lx0 + 4 or ly1 <= ly0 + 4:
            return None
        # Zoom that covers (essentially) the whole crop → no-op.
        if (
            lx0 <= 1 and ly0 <= 1
            and lx1 >= w - 2 and ly1 >= h - 2
        ):
            return None
        return (lx0, ly0, lx1, ly1)

    def _refresh_label_counts() -> None:
        counts: dict[str, int] = {}
        for lab in _ctx["labels"]:
            counts[lab] = counts.get(lab, 0) + 1
        with state:
            state.uct_label_counts = counts

    def _commit_current_slice_to_cache() -> None:
        """Snapshot the working proposals into per_slice for
        the current slice idx. Called BEFORE switching slices
        so edits made on slice N (paint / erase / label) stay
        with slice N."""
        cur = _ctx.get("current_slice_idx", -1)
        if cur < 0:
            return
        _ctx["per_slice"][cur] = {
            "proposals": list(_ctx["proposals"]),
            "labels": list(_ctx["labels"]),
            "colors": list(_ctx["proposal_colors"]),
        }

    def _restore_slice_from_cache(idx: int) -> None:
        """Pull proposals/labels for slice idx out of
        per_slice into the working keys. Empty entry =
        slice has no segmentation (the user can run
        propose_all on that slice individually, or rely on
        the stack-wide pass having populated it)."""
        entry = _ctx["per_slice"].get(idx)
        if entry is not None:
            _ctx["proposals"] = list(entry["proposals"])
            _ctx["labels"] = list(entry["labels"])
            _ctx["proposal_colors"] = list(entry["colors"])
        else:
            _ctx["proposals"] = []
            _ctx["labels"] = []
            _ctx["proposal_colors"] = []
        # Push chip meta for the new slice.
        meta = [
            {
                "idx": i,
                "area_px": int(p.area_px),
                "bbox_str": (
                    f"({p.bbox[0]},{p.bbox[1]})-"
                    f"({p.bbox[2]},{p.bbox[3]})"
                ),
                "label": _ctx["labels"][i],
                "color_hex": color_to_hex(
                    _ctx["proposal_colors"][i]
                ) if i < len(_ctx["proposal_colors"]) else (
                    "#888"
                ),
            }
            for i, p in enumerate(_ctx["proposals"])
        ]
        with state:
            state.uct_proposals_meta = meta

    def _read_and_render_slice(idx: int) -> None:
        """Read raw slice → contrast-stretch → re-apply the
        CURRENT crop → swap proposals from per_slice cache
        → re-render overlay.

        Crop persists across slice scrolls; per-slice
        proposals are independent so each frame keeps its
        own segmentation + edits.
        """
        s = _ctx["stack"]
        if s is None:
            return
        idx = max(0, min(s.n_frames - 1, int(idx)))
        # Commit edits to the OLD slice's cache before pulling
        # in the new one.
        _commit_current_slice_to_cache()

        arr = read_slice(s, idx)
        disp = to_display(
            arr,
            window=getattr(
                _ctx.get("stack"), "display_window", None,
            ),
        )
        h, w = disp.shape
        _ctx["slice_arr_full"] = arr
        _ctx["slice_disp_full"] = disp

        # First load → init crop to full extent.
        cur_crop = _ctx.get("crop") or (0, 0, w - 1, h - 1)
        cx0, cy0, cx1, cy1 = cur_crop
        first_load = (
            cx1 <= cx0 or cy1 <= cy0
            or cx1 >= w or cy1 >= h
        )
        if first_load:
            cx0, cy0 = 0, 0
            cx1, cy1 = w - 1, h - 1
        _ctx["crop"] = (cx0, cy0, cx1, cy1)
        _ctx["slice_arr"] = arr[cy0:cy1 + 1, cx0:cx1 + 1]
        _ctx["slice_disp"] = disp[cy0:cy1 + 1, cx0:cx1 + 1]
        _ctx["slice_disp_clahe"] = None
        _ctx["current_slice_idx"] = idx
        # Pull this slice's proposals from the per-slice cache.
        _restore_slice_from_cache(idx)
        _refresh_label_counts()

        if first_load:
            with state:
                state.uct_crop_max_x = int(w - 1)
                state.uct_crop_max_y = int(h - 1)
                state.uct_crop_x_range = [0, int(w - 1)]
                state.uct_crop_y_range = [0, int(h - 1)]
        else:
            with state:
                state.uct_crop_max_x = int(w - 1)
                state.uct_crop_max_y = int(h - 1)
        _rerender_overlay()

    def _remap_mask_to_new_crop(
        mask: np.ndarray,
        old_crop: tuple[int, int, int, int],
        new_crop: tuple[int, int, int, int],
        full_shape: tuple[int, int],
    ) -> np.ndarray:
        """Embed `mask` (in old_crop frame) into a full-image
        canvas, then re-slice it into new_crop frame. Pixels
        outside the new crop get dropped; pixels inside the
        new crop but outside the old crop are False.

        Used by `_on_crop_change` to preserve annotations when
        the user resets the crop or picks a different one —
        previously the whole per-slice cache was wiped, which
        meant clicking "Reset crop" or accidentally moving the
        crop slider threw away all the labelling work.
        """
        full_h, full_w = full_shape
        full = np.zeros((full_h, full_w), dtype=bool)
        ox0, oy0, ox1, oy1 = old_crop
        # Clamp into full-image bounds defensively (proposals
        # from a stale crop set might extend past the image if
        # the source was reloaded at a different resolution).
        ox0_c = max(0, min(full_w - 1, int(ox0)))
        oy0_c = max(0, min(full_h - 1, int(oy0)))
        ox1_c = max(ox0_c, min(full_w - 1, int(ox1)))
        oy1_c = max(oy0_c, min(full_h - 1, int(oy1)))
        target = full[oy0_c:oy1_c + 1, ox0_c:ox1_c + 1]
        # Mask may be sized for a different-sized old crop than
        # what got clamped above. Slice to the overlapping
        # rectangle so the assignment never raises.
        m_h, m_w = mask.shape[:2]
        copy_h = min(target.shape[0], m_h)
        copy_w = min(target.shape[1], m_w)
        target[:copy_h, :copy_w] = mask[:copy_h, :copy_w]
        nx0, ny0, nx1, ny1 = new_crop
        return full[ny0:ny1 + 1, nx0:nx1 + 1].copy()

    def _on_crop_change(
        uct_crop_x_range, uct_crop_y_range, **_kw,
    ) -> None:
        """Re-crop the active slice when either VRangeSlider
        moves. The proposal masks are stored in crop-local
        coordinates, so a crop change has to either DROP them
        (legacy) or REMAP them to the new frame (current).

        M33 — Switched from drop to remap so the user's
        labelling survives crop resets. Each proposal's mask
        gets embedded back to full-image coords (using the
        OLD crop offset) and then sliced to the new crop
        rectangle. Bbox + area_px are recomputed in the new
        frame. Empty proposals after the remap are dropped
        (their content was outside the new crop window).
        """
        if _ctx.get("slice_disp_full") is None:
            return
        try:
            x0, x1 = (
                int(uct_crop_x_range[0]),
                int(uct_crop_x_range[1]),
            )
            y0, y1 = (
                int(uct_crop_y_range[0]),
                int(uct_crop_y_range[1]),
            )
        except (TypeError, ValueError, IndexError):
            return
        # Vuetify's VRangeSlider can briefly emit lo == hi when
        # the user pins one thumb against the other. Guard
        # against the degenerate 1-pixel crop.
        if x1 <= x0 + 1 or y1 <= y0 + 1:
            return
        old_crop = tuple(_ctx.get("crop", (0, 0, 0, 0)))
        new_crop = (x0, y0, x1, y1)
        if old_crop == new_crop:
            return
        full_h, full_w = _ctx["slice_disp_full"].shape[:2]
        _ctx["crop"] = new_crop
        _ctx["slice_arr"] = _ctx["slice_arr_full"][
            y0:y1 + 1, x0:x1 + 1,
        ]
        _ctx["slice_disp"] = _ctx["slice_disp_full"][
            y0:y1 + 1, x0:x1 + 1,
        ]
        # CLAHE result belongs to a different slice/crop now.
        _ctx["slice_disp_clahe"] = None

        # Remap CURRENT-slice proposals.
        new_props: list[MaskProposal] = []
        new_labels: list[str] = []
        new_colors: list = []
        for p, lab, col in zip(
            _ctx.get("proposals", []) or [],
            _ctx.get("labels", []) or [],
            (_ctx.get("proposal_colors") or [None] * len(
                _ctx.get("proposals") or [],
            )),
        ):
            try:
                new_mask = _remap_mask_to_new_crop(
                    p.mask, old_crop, new_crop,
                    (full_h, full_w),
                )
            except Exception:                             # noqa: BLE001
                continue
            if not new_mask.any():
                continue
            ys, xs = np.where(new_mask)
            bbox = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
            new_props.append(MaskProposal(
                mask=new_mask,
                score=float(getattr(p, "score", 0.0) or 0.0),
                bbox=bbox,
                area_px=int(new_mask.sum()),
                meta=dict(getattr(p, "meta", {}) or {}),
            ))
            new_labels.append(lab)
            new_colors.append(col)
        _ctx["proposals"] = new_props
        _ctx["labels"] = new_labels
        _ctx["proposal_colors"] = (
            new_colors if all(c is not None for c in new_colors)
            else generate_proposal_colors(len(new_props))
        )

        # Remap EVERY slice in per_slice cache too.
        new_per_slice: dict[int, dict] = {}
        for idx, entry in (_ctx.get("per_slice") or {}).items():
            old_ps_props = entry.get("proposals") or []
            old_ps_labels = entry.get("labels") or []
            old_ps_colors = entry.get("colors") or [None] * len(
                old_ps_props,
            )
            ps_props: list[MaskProposal] = []
            ps_labels: list[str] = []
            ps_colors: list = []
            for p, lab, col in zip(
                old_ps_props, old_ps_labels, old_ps_colors,
            ):
                try:
                    new_mask = _remap_mask_to_new_crop(
                        p.mask, old_crop, new_crop,
                        (full_h, full_w),
                    )
                except Exception:                         # noqa: BLE001
                    continue
                if not new_mask.any():
                    continue
                ys, xs = np.where(new_mask)
                bbox = (
                    int(xs.min()), int(ys.min()),
                    int(xs.max()), int(ys.max()),
                )
                ps_props.append(MaskProposal(
                    mask=new_mask,
                    score=float(
                        getattr(p, "score", 0.0) or 0.0,
                    ),
                    bbox=bbox,
                    area_px=int(new_mask.sum()),
                    meta=dict(getattr(p, "meta", {}) or {}),
                ))
                ps_labels.append(lab)
                ps_colors.append(col)
            if ps_props:
                new_per_slice[int(idx)] = {
                    "proposals": ps_props,
                    "labels": ps_labels,
                    "colors": (
                        ps_colors
                        if all(c is not None for c in ps_colors)
                        else generate_proposal_colors(
                            len(ps_props),
                        )
                    ),
                }
        _ctx["per_slice"] = new_per_slice

        with state:
            # Crop change invalidates any active zoom.
            state.uct_zoom_x_range = [0, 0]
            state.uct_zoom_y_range = [0, 0]
        _rebuild_meta_after_mutation()

    # ---- public handlers ----

    def _uct_dir() -> Optional[Path]:
        """`<active_project>/uct/`, mkdir'd on demand. Returns
        None when no project is active (callers fall back to a
        no-op or surface an error to the user)."""
        if get_active_project_dir is None:
            return None
        try:
            base = Path(get_active_project_dir())
        except Exception:                             # noqa: BLE001
            return None
        if not base.exists():
            return None
        d = base / "uct"
        d.mkdir(exist_ok=True)
        return d

    def do_open_segment_uct_dialog() -> None:
        with state:
            state.show_segment_uct_dialog = True
            state.uct_status = ""
            # Always land on Step 1 (Upload) when opening. The
            # restore path below may bump us straight to Step 2
            # if a saved segmentation exists for the project.
            state.uct_step = "1"
            state.uct_step2_finalized = False
        # Phase A.4 — restore an existing segmentation if the
        # active project already has one. Walks
        # <project>/uct/segmentation.json + the labelled mask
        # PNG, rebuilds proposals (one per connected component
        # per label class) and re-renders the overlay. On
        # success it sets uct_stack_loaded=True, so we also
        # bump uct_step → "2" so the user sees their work
        # immediately instead of an empty Upload step.
        _try_restore_segmentation()
        if bool(getattr(state, "uct_stack_loaded", False)):
            with state:
                state.uct_step = "2"

    def do_close_segment_uct_dialog() -> None:
        # M33 — auto-save the segmentation when the dialog is
        # closed (cross button or escape). User previously had
        # to remember to click Finalize before closing, which
        # meant accidental closes lost all annotations.
        # Best-effort: a save failure is logged but doesn't
        # block closing the dialog.
        try:
            if (
                _ctx.get("slice_disp") is not None
                and (_ctx.get("per_slice") or {})
            ):
                do_save_uct_segmentation()
        except Exception as ex:                           # noqa: BLE001
            print(
                f"[seg-close] autosave failed (continuing): "
                f"{ex}",
                flush=True,
            )
        with state:
            state.show_segment_uct_dialog = False

    def do_load_uct_stack(*_args) -> None:
        path = str(getattr(state, "uct_file_path", "") or "")
        if not path:
            with state:
                state.uct_status = "Pick a file path first."
            return
        with state:
            state.uct_busy = True
            state.busy = True
            state.busy_msg = "Preparing stack…"
            state.busy_log = ""
            state.uct_status = f"Loading {path}…"
        state.flush()

        # If the upload landed in a directory of DICOM files
        # AND there's no pre-compressed volume.nii.gz yet,
        # compress in place first. This is the slow step the
        # user used to see no feedback for — drive it through
        # the dialog's busy lightbox so it's visible from any
        # panel. Compression cuts disk usage 4-7× AND makes
        # subsequent loads instant (single .nii.gz vs N .dcm
        # round-trips).
        p = Path(path)
        if p.is_dir() and not (p / "volume.nii.gz").is_file():
            try:
                from golgi.segmentation.image import (
                    _is_dicom_dir,
                    compress_dicom_series_to_nifti,
                )
                if _is_dicom_dir(p):
                    n_dcm = sum(
                        1 for f in p.iterdir()
                        if f.is_file()
                    )
                    with state:
                        state.busy_msg = (
                            f"Compressing DICOM series "
                            f"({n_dcm} files) → volume.nii.gz"
                        )
                        state.busy_log = (
                            "Reading DICOM headers…"
                        )
                    state.flush()

                    # Stream progress lines from the
                    # compressor into busy_log so the user
                    # sees "casting int32 → int16", "writing
                    # …", "deleted N originals" in real time.
                    def _on_compress_log(msg: str) -> None:
                        with state:
                            state.busy_log = msg
                        try:
                            state.flush()
                        except Exception:               # noqa: BLE001
                            pass

                    compress_dicom_series_to_nifti(
                        p,
                        delete_originals=True,
                        on_log=_on_compress_log,
                    )
            except Exception as ex:                   # noqa: BLE001
                # Compression failed → keep originals and
                # let load_stack fall back to the DICOM
                # series path. Print a warning but don't
                # block the load.
                print(
                    f"[uct-load] compression skipped: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )

        # Phase-specific status — for large stacks, the
        # `load_stack` call below can take seconds (DICOM
        # series re-scan after compression) or stay nearly
        # instant (TIFF/NIfTI metadata only). Either way the
        # user needs to see SOMETHING here rather than a blank
        # transition between "Compressing…" and the first
        # rendered slice. The drop-zone overlay in the dialog
        # reads busy_msg as its primary status line.
        try:
            _fmt_hint = Path(path).suffix.lower()
            if _fmt_hint in (".tif", ".tiff"):
                _fmt_label = "TIFF"
            elif _fmt_hint in (".nii", ".gz"):
                _fmt_label = "NIfTI"
            elif _fmt_hint in (".nrrd",):
                _fmt_label = "NRRD"
            elif Path(path).is_dir():
                _fmt_label = "DICOM series"
            else:
                _fmt_label = "image stack"
        except Exception:                                # noqa: BLE001
            _fmt_label = "image stack"
        with state:
            state.busy_msg = f"Reading {_fmt_label} headers…"
            state.busy_log = f"{path}"
        state.flush()

        try:
            stack = load_stack(path)
        except Exception as ex:                       # noqa: BLE001
            with state:
                state.uct_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_stack_loaded = False
                state.uct_status = f"Load failed: {ex}"
            return
        _ctx["stack"] = stack
        with state:
            state.busy_msg = (
                f"Decoded {stack.n_frames}-frame stack · "
                f"{stack.height} × {stack.width} · "
                f"rendering first slice…"
            )
            state.busy_log = ""
        state.flush()
        # Pre-populate voxel size from TIFF tags when available;
        # leave user value alone if they've already overridden.
        vsz = stack.voxel_size_um
        with state:
            state.uct_stack_loaded = True
            state.uct_slice_max = max(0, stack.n_frames - 1)
            state.uct_slice_idx = stack.n_frames // 2
            state.uct_stack_info_html = (
                f"<b>{stack.n_frames}</b> frames · "
                f"{stack.height} × {stack.width} · "
                f"dtype <code>{stack.dtype}</code>"
            )
            if (
                vsz is not None
                and float(getattr(state, "uct_voxel_size_um", 0))
                    <= 0.0
            ):
                state.uct_voxel_size_um = float(vsz[0])
            state.uct_status = (
                f"Loaded {stack.n_frames}-frame stack."
            )
            state.uct_busy = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            # Auto-advance Step 1 (Upload) → Step 2 (Segment) on
            # a successful load. The user just dropped a file or
            # picked one in the file picker; pushing them into
            # the segment view immediately avoids a redundant
            # "Next →" click. We only advance from "1" — if the
            # user re-uploads while already in "2" or "3", we
            # leave them where they are.
            if str(
                getattr(state, "uct_step", "1") or "1",
            ) == "1":
                state.uct_step = "2"
        _read_and_render_slice(stack.n_frames // 2)

    def do_clear_uct_stack(*_args) -> None:
        """Drop the currently-loaded stack so the user can
        upload a different one without restarting the app.

        Resets all per-stack closure state in `_ctx` and the
        trame state vars the dialog binds to. Releases the SAM2
        video inference state (frees per-frame feature
        tensors) but keeps the model loaded — so the next
        Propagate call doesn't re-pay the model load.

        Files on disk are preserved: the uploaded source
        files in `<project>/uct/uploads/` and the SAM2 JPEG
        cache in `<project>/uct/sam2_cache/` stay as archives
        so the user can re-upload / re-prepare without
        re-streaming the same bytes.
        """
        # Free SAM2 video state if it exists. forget() drops
        # the per-frame feature tensors for this stack but
        # keeps the predictor module alive.
        video_seg = _ctx.get("video_seg")
        prev_stack_id = _ctx.get("video_stack_id") or ""
        if video_seg is not None and prev_stack_id:
            try:
                video_seg.forget(prev_stack_id)
            except Exception:                            # noqa: BLE001
                pass

        # Reset the heavy closure state. The segmenter
        # instance stays cached — switching stacks doesn't
        # require reloading the model.
        _ctx["stack"] = None
        _ctx["slice_arr_full"] = None
        _ctx["slice_disp_full"] = None
        _ctx["slice_arr"] = None
        _ctx["slice_disp"] = None
        _ctx["slice_disp_clahe"] = None
        _ctx["crop"] = (0, 0, 0, 0)
        _ctx["current_slice_idx"] = -1
        _ctx["proposals"] = []
        _ctx["labels"] = []
        _ctx["proposal_colors"] = []
        _ctx["per_slice"] = {}
        _ctx["video_stack_id"] = ""
        _ctx["video_keyframe_obj_map"] = {}

        # Reset the trame-side state. Keep `uct_voxel_size_um`
        # at its current value (next stack may share the same
        # scanner pitch) and keep the segmenter / backend
        # picker untouched.
        with state:
            state.uct_file_path = ""
            state.uct_file_input = None
            state.uct_stack_loaded = False
            state.uct_stack_info_html = ""
            state.uct_slice_idx = 0
            state.uct_slice_max = 0
            state.uct_overlay_url = ""
            state.uct_image_orig_w = 0
            state.uct_image_orig_h = 0
            state.uct_proposals_meta = []
            state.uct_label_counts = {}
            state.uct_status = ""
            state.uct_step = "1"
            state.uct_step2_finalized = False
            state.uct_crop_x_range = [0, 0]
            state.uct_crop_y_range = [0, 0]
            state.uct_zoom_x_range = [0, 0]
            state.uct_zoom_y_range = [0, 0]
            state.uct_keyframe_slices = []
            state.uct_keyframe_summary = ""
            state.uct_propagation_busy = False
            state.uct_upload_progress = 0
            state.uct_upload_status = ""
            state.uct_upload_error = ""
            # Also bounce out of Step-3 reconstruction state.
            state.uct_recon_files = []
            state.uct_recon_status = ""

    def _on_slice_change(uct_slice_idx, **_kw) -> None:
        """Slice-scrubber watcher. Bound to state.uct_slice_idx
        by build_app's @state.change decorator (registered
        separately because @state.change needs the state
        instance)."""
        if _ctx["stack"] is None:
            return
        _read_and_render_slice(int(uct_slice_idx))

    async def do_run_uct_segmentation(*_args) -> None:
        """Run the selected segmenter on either the current
        slice or every slice in the stack — picked by
        `state.uct_segment_scope` ∈ {"current", "all"}.

        Results land in `_ctx["per_slice"][idx]`. The slice-
        change handler swaps that into the working keys on
        every scroll, so per-slice edits (paint / erase /
        label) stay with their slice.

        Async + run_in_executor so the busy lightbox + log
        update live during long stack runs.
        """
        stack = _ctx.get("stack")
        if stack is None or _ctx["slice_disp"] is None:
            with state:
                state.uct_status = "No stack loaded."
            return

        # Snapshot the current slice's working state so the
        # loop can write to per_slice without losing edits
        # already made.
        _commit_current_slice_to_cache()

        n_frames = int(stack.n_frames)
        crop = _ctx.get("crop", (0, 0, 0, 0))
        cx0, cy0, cx1, cy1 = (
            int(crop[0]), int(crop[1]),
            int(crop[2]), int(crop[3]),
        )
        use_clahe = bool(getattr(state, "uct_clahe", False))
        scope = str(
            getattr(state, "uct_segment_scope", "all") or "all",
        )
        cur_idx = int(_ctx.get("current_slice_idx", 0))
        # Step size for "all slices" scope. 1 = every slice
        # (legacy behaviour); N > 1 = sweep every Nth slice
        # so the user can segment a sparser set and let the
        # Step-3 ZOH-fill carry the intermediate slices.
        try:
            sweep_step = int(
                getattr(state, "uct_segment_step", 1) or 1,
            )
        except (TypeError, ValueError):
            sweep_step = 1
        sweep_step = max(1, sweep_step)
        if scope == "current":
            indices = [cur_idx]
            scope_label = f"slice {cur_idx + 1} / {n_frames}"
        else:
            indices = list(range(0, n_frames, sweep_step))
            if sweep_step == 1:
                scope_label = f"stack ({n_frames} slices)"
            else:
                scope_label = (
                    f"sweep every {sweep_step} slices · "
                    f"{len(indices)} / {n_frames}"
                )

        with state:
            state.uct_busy = True
            state.busy = True
            state.busy_msg = f"Segmenting {scope_label}"
            state.busy_log = "Loading model…"
            state.uct_status = (
                f"Segmenting {scope_label}…"
            )
        state.flush()
        await asyncio.sleep(0)

        loop = asyncio.get_event_loop()
        try:
            await loop.run_in_executor(
                None, _ensure_segmenter,
            )
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_status = (
                    f"Segmenter load failed: {ex}"
                )
            return

        engine_name = _ctx["segmenter"].name
        per_slice = _ctx["per_slice"]

        def _segment_one(arr_disp_in: "np.ndarray"):
            # Inner blocking job for run_in_executor — applies
            # CLAHE if requested and runs propose_all.
            seg_in = arr_disp_in
            if use_clahe:
                seg_in = _apply_clahe(arr_disp_in)
            return _ctx["segmenter"].propose_all(seg_in)

        for step, idx in enumerate(indices, start=1):
            # Read + apply crop fresh per slice. We don't keep
            # all 64 cropped uint8 slices in memory — TIFF
            # access is fast enough that re-reading is fine.
            try:
                arr = read_slice(stack, idx)
            except Exception as ex:                   # noqa: BLE001
                print(
                    f"[segment] read_slice({idx}) failed: "
                    f"{ex}",
                    flush=True,
                )
                continue
            disp = to_display(
            arr,
            window=getattr(
                _ctx.get("stack"), "display_window", None,
            ),
        )
            cropped = disp[cy0:cy1 + 1, cx0:cx1 + 1]

            with state:
                state.busy_log = (
                    f"Slice {idx + 1} / {n_frames}  "
                    f"({step} / {len(indices)} in this run · "
                    f"{engine_name}"
                    f"{' + CLAHE' if use_clahe else ''})"
                )
            state.flush()
            await asyncio.sleep(0)

            try:
                proposals = await loop.run_in_executor(
                    None, _segment_one, cropped,
                )
            except Exception as ex:                   # noqa: BLE001
                traceback.print_exc()
                print(
                    f"[segment] slice {idx} failed: {ex}",
                    flush=True,
                )
                continue

            # Apply 2D mask cleanup BEFORE assigning colours so
            # that dropped proposals (e.g. sub-min-component-px
            # noise blobs) don't waste colour slots and don't
            # appear in the legend. Cleanup runs per-proposal:
            # see `_apply_2d_cleanup_to_proposals` for the
            # design rationale.
            cleaned_props, cleaned_labels, _ = (
                _apply_2d_cleanup_to_proposals(
                    list(proposals),
                    ["unlabeled"] * len(proposals),
                    None,
                )
            )
            colors = generate_proposal_colors(len(cleaned_props))
            per_slice[idx] = {
                "proposals": cleaned_props,
                "labels": cleaned_labels,
                "colors": list(colors),
            }

        # Swap the current slice's freshly-computed entry back
        # into the working keys so the dialog reflects it.
        _restore_slice_from_cache(cur_idx)
        _refresh_label_counts()

        # Status line — varies by scope so the user knows what
        # was just run.
        if scope == "current":
            n_props = len(_ctx["proposals"])
            status = (
                f"Segmented slice {cur_idx + 1} · "
                f"{n_props} masks."
            )
        else:
            total = sum(
                len(per_slice[i]["proposals"])
                for i in per_slice
            )
            status = (
                f"Segmented {len(per_slice)} / {n_frames} "
                f"slices · {total} total masks. Scroll to "
                f"review."
            )
        with state:
            state.uct_busy = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.uct_status = status
        _rerender_overlay()

    def do_label_uct_proposal(idx, label) -> None:
        """Set proposal `idx`'s label and re-render overlay +
        counts. Bound to the per-chip 4-button rows in the
        legend. Triggers epi auto-regen if an auto-derived
        epi proposal is present, since changing a label can
        invalidate the derived mask.

        When the user clicks "None", we set the proposal's
        `meta.fresh = False` so the overlay renderer skips it
        (an explicit dismiss). Without that, the renderer
        falls back to the per-proposal hue and the user just
        sees a colour swap instead of the region clearing —
        which doesn't match the "None means hide" UX.
        """
        try:
            i = int(idx)
            lab = str(label)
        except (TypeError, ValueError):
            return
        labels = list(_ctx["labels"])
        if i < 0 or i >= len(labels):
            return
        labels[i] = lab
        _ctx["labels"] = labels
        # Update the per-proposal fresh flag. MaskProposal's
        # `meta` is a regular dict so we can mutate in place.
        try:
            prop = _ctx["proposals"][i]
            if not hasattr(prop, "meta") or prop.meta is None:
                prop.meta = {}
            if lab == "unlabeled":
                # Explicit user-dismiss → hide on overlay.
                prop.meta["fresh"] = False
            else:
                # User relabelled to a concrete class → drop
                # the dismiss flag so subsequent "None" toggles
                # restore the original meaning.
                prop.meta.pop("fresh", None)
        except (IndexError, AttributeError):
            pass
        # If an auto-derived epi exists, it must be re-derived
        # — re-running the full meta rebuild path keeps the
        # proposal index + colour mapping consistent.
        if _epi_exists():
            _recompute_epi()
            _rebuild_meta_after_mutation()
            return
        meta = list(getattr(state, "uct_proposals_meta", []))
        if i < len(meta):
            meta[i] = {**meta[i], "label": lab}
        with state:
            state.uct_proposals_meta = meta
        _refresh_label_counts()
        _rerender_overlay()

    def _is_auto_epi(p: MaskProposal) -> bool:
        """True for the auto-derived epi proposal (created by
        do_generate_epi / _recompute_epi). Distinguishes it
        from user-labelled epi proposals so the auto-regen
        path only replaces the derived one."""
        try:
            return bool(p.meta.get("auto_derived"))
        except AttributeError:
            return False

    def _epi_exists() -> bool:
        """True iff the AUTO-derived epi proposal is present
        — user-labelled epi proposals don't count, since the
        auto-regen path shouldn't keep recreating one when the
        user is explicitly maintaining their own."""
        return any(
            lab == "epi" and _is_auto_epi(p)
            for p, lab in zip(
                _ctx.get("proposals", []),
                _ctx.get("labels", []),
            )
        )

    def _recompute_epi() -> tuple[bool, int]:
        """Drop any auto-derived epi proposal and re-derive a
        fresh one from the CURRENT label state.

            epi_mask = full_slice
                       AND NOT (fascicle masks ∪)
                       AND NOT (background masks ∪)
                       AND NOT (user-labelled epi masks ∪)

        User-labelled epi proposals (label=="epi" and NOT
        auto-derived) are preserved AND contribute to the
        "already covered" union, so the derived auto-epi
        only fills the remaining gaps.

        Returns (added, pixel_count).
        """
        if _ctx.get("slice_disp") is None:
            return False, 0
        h, w = _ctx["slice_disp"].shape[:2]
        fasc_union = np.zeros((h, w), dtype=bool)
        bg_union = np.zeros((h, w), dtype=bool)
        manual_epi_union = np.zeros((h, w), dtype=bool)
        keep_props: list[MaskProposal] = []
        keep_labels: list[str] = []
        for p, lab in zip(
            _ctx["proposals"], _ctx["labels"],
        ):
            if lab == "epi" and _is_auto_epi(p):
                # Drop the OLD auto-derived epi only — re-built
                # below. User-labelled epi stays.
                continue
            if lab == "fascicle":
                fasc_union |= p.mask
            elif lab == "background":
                bg_union |= p.mask
            elif lab == "epi":
                # User-labelled epi mask — counts as already
                # affirmatively-labelled territory.
                manual_epi_union |= p.mask
            keep_props.append(p)
            keep_labels.append(lab)
        any_labelled = (
            fasc_union.any()
            or bg_union.any()
            or manual_epi_union.any()
        )
        if not any_labelled:
            _ctx["proposals"] = keep_props
            _ctx["labels"] = keep_labels
            return False, 0
        epi_mask = ~(
            fasc_union | bg_union | manual_epi_union
        )
        if not epi_mask.any():
            _ctx["proposals"] = keep_props
            _ctx["labels"] = keep_labels
            return False, 0
        ys, xs = np.where(epi_mask)
        bbox = (
            int(xs.min()), int(ys.min()),
            int(xs.max()), int(ys.max()),
        )
        keep_props.append(MaskProposal(
            mask=epi_mask, score=1.0,
            bbox=bbox, area_px=int(epi_mask.sum()),
            meta={"auto_derived": True},
        ))
        keep_labels.append("epi")
        _ctx["proposals"] = keep_props
        _ctx["labels"] = keep_labels
        return True, int(epi_mask.sum())

    def _refresh_epi_if_present() -> None:
        """Auto-regenerate the epi mask IF one was already
        present. No-op when epi hasn't been generated yet (the
        user can still call do_generate_epi explicitly). This
        is what makes "label fascicle → generate epi → paint
        more → epi auto-updates" feel seamless instead of
        leaving a stale green overlay covering the new
        fascicle.
        """
        if _epi_exists():
            _recompute_epi()

    def do_refine_masks(*_args) -> None:
        """Apply standard refinement to every labelled mask:

          1. Fill internal holes (`scipy.ndimage.binary_fill_holes`)
             — fixes speckle pixels SAM2 missed inside a
             fascicle that's otherwise solidly classified.
          2. Morphological closing with disk(2) — bridges
             few-pixel gaps in the boundary.
          3. Morphological opening with disk(1) — removes
             small spike artefacts on the boundary.
          4. Gaussian smoothing (σ=1.5) + threshold 0.5 —
             gives a smoother outline without significantly
             changing the enclosed area.
          5. Remove sub-30-px isolated components left from
             SAM2 noise.
          6. DROP every proposal labelled "unlabeled" — the
             user explicitly chose to leave them unclassified,
             so they're scaffolding we no longer need.
          7. Auto-regen the derived epi mask from the refined
             fascicle / background unions.

        Auto-derived epi is skipped in the loop (it gets
        rebuilt at the end) so smoothing doesn't tug its
        boundary in a way that conflicts with the fresh
        derivation. User-labelled epi proposals get the same
        refinement treatment as fascicles.
        """
        if not _ctx.get("proposals"):
            with state:
                state.uct_status = (
                    "No proposals to refine."
                )
            return
        try:
            # skimage 0.26 deprecated `binary_closing` /
            # `binary_opening` in favour of `closing` / `opening`
            # (both work identically on bool input — only the
            # name changed), and `remove_small_objects` swapped
            # its `min_size=<lo>` knob for `max_size=<lo - 1>`
            # with inverted threshold semantics (removes objects
            # ≤ max_size instead of < min_size). The aliases
            # below pin the imports + threshold to whichever
            # version is installed so we silence the future-
            # warning spam without breaking older skimages.
            from skimage import __version__ as _sk_ver
            from skimage.morphology import disk
            _sk_major, _sk_minor = (
                int(x) for x in _sk_ver.split(".")[:2]
            )
            if (_sk_major, _sk_minor) >= (0, 26):
                from skimage.morphology import (
                    closing as binary_closing,
                    opening as binary_opening,
                    remove_small_objects as _rso_new,
                )

                def remove_small_objects(arr, min_size, **kw):
                    # Old min_size=N removed objects with area
                    # < N. New max_size=M removes objects with
                    # area ≤ M. So max_size = min_size - 1.
                    return _rso_new(
                        arr,
                        max_size=int(min_size) - 1,
                        **kw,
                    )
            else:
                from skimage.morphology import (
                    binary_closing, binary_opening,
                    remove_small_objects,
                )
            from scipy import ndimage as ndi
        except ImportError as ex:
            with state:
                state.uct_status = (
                    f"Refinement deps missing: {ex}"
                )
            return

        had_auto_epi = _epi_exists()
        keep_props: list[MaskProposal] = []
        keep_labels: list[str] = []
        n_dropped_unlabeled = 0
        n_dropped_tiny = 0
        n_refined = 0

        for p, lab in zip(
            _ctx["proposals"], _ctx["labels"],
        ):
            # Drop unlabeled — they were just SAM2 candidates
            # the user decided not to classify.
            if lab == "unlabeled":
                n_dropped_unlabeled += 1
                continue
            # Skip the auto-derived epi — it's rebuilt below.
            if lab == "epi" and _is_auto_epi(p):
                continue

            m = p.mask.copy()
            # Background masks usually represent the SURROUND
            # — one big region with the nerve cross-section
            # as a hole inside. fill_holes would close that
            # hole and expand bg to cover the entire image,
            # which (a) wipes out the auto-derived epi mask
            # (auto_epi = ~(fasc ∪ bg)) and (b) re-labels
            # everything-but-fascicles as background.
            # binary_closing has the same direction of harm:
            # it grows bg outward, eating 1-2 px of the
            # nerve outline. Skip both for background; keep
            # the gentler ops (opening + smoothing +
            # speckle-removal) which only shrink or smooth
            # the boundary.
            if lab != "background":
                m = ndi.binary_fill_holes(m)
                m = binary_closing(m, disk(2))
            m = binary_opening(m, disk(1))
            # Gaussian + threshold for smooth outline. Cast
            # to float32 once; convert back at the threshold.
            m_smooth = ndi.gaussian_filter(
                m.astype(np.float32), sigma=1.5,
            )
            m = m_smooth > 0.5
            m = remove_small_objects(
                m, min_size=30, connectivity=2,
            )
            if not m.any():
                n_dropped_tiny += 1
                continue
            ys, xs = np.where(m)
            bbox = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
            # Preserve meta (e.g. auto_derived flag would have
            # already been filtered above, but if the user
            # manually labelled something as epi the meta is
            # empty — defensive copy preserves whatever's
            # there).
            keep_props.append(MaskProposal(
                mask=m, score=p.score,
                bbox=bbox, area_px=int(m.sum()),
                meta=dict(getattr(p, "meta", None) or {}),
            ))
            keep_labels.append(lab)
            n_refined += 1

        _ctx["proposals"] = keep_props
        _ctx["labels"] = keep_labels

        msg_parts = [f"Refined {n_refined} mask"
                     + ("s" if n_refined != 1 else "")]
        if n_dropped_unlabeled:
            msg_parts.append(
                f"dropped {n_dropped_unlabeled} unlabeled"
            )
        if n_dropped_tiny:
            msg_parts.append(
                f"removed {n_dropped_tiny} tiny"
            )

        # Rebuild the derived epi using the refined masks, but
        # only if one was already present (the user already
        # opted in to having a derived overlay; otherwise
        # respect that they haven't asked for one yet).
        if had_auto_epi:
            _recompute_epi()
            msg_parts.append("re-derived epi")

        _rebuild_meta_after_mutation()
        with state:
            state.uct_status = " · ".join(msg_parts) + "."

    def do_generate_epi(*_args) -> None:
        """User-driven epineurium generation. See
        `_recompute_epi` for the math. This wrapper handles
        the status messages + meta rebuild that the manual
        click triggers."""
        if (
            _ctx.get("slice_disp") is None
            or not _ctx.get("proposals")
        ):
            with state:
                state.uct_status = (
                    "Run segmentation first."
                )
            return
        # Sanity check before recomputing: do we have any
        # affirmative labels?
        has_fasc = any(
            lab == "fascicle"
            for lab in _ctx.get("labels", [])
        )
        has_bg = any(
            lab == "background"
            for lab in _ctx.get("labels", [])
        )
        if not has_fasc and not has_bg:
            with state:
                state.uct_status = (
                    "Label at least one fascicle or "
                    "background region before generating "
                    "the epineurium."
                )
            return
        added, n_px = _recompute_epi()
        if not added:
            with state:
                state.uct_status = (
                    "Nothing left after subtracting "
                    "fascicles + background."
                )
            return
        _rebuild_meta_after_mutation()
        with state:
            state.uct_status = (
                f"Generated epineurium ({n_px:,} px)."
            )

    def do_save_uct_segmentation(*_args) -> None:
        """M33 — Persist EVERY annotated slice, not just the
        currently-displayed one. Writes:

          - `segmentation.json` — metadata + array of saved
            slices (`slices: [{slice_idx, mask_path,
            label_counts}, …]`).
          - `labels_slice<N>.png` per annotated slice — uint8
            mask where 0 = background, 1 = epi, 2 = fascicle.

        Pre-M33 (v1 schema) only saved the current slice. v2
        adds the slices array. `_try_restore_segmentation`
        reads both shapes for backwards compatibility.
        """
        # Make sure the working slice's edits are committed into
        # per_slice before we iterate — otherwise the last few
        # paint strokes / labels won't be on disk.
        try:
            _commit_current_slice_to_cache()
        except Exception:                                 # noqa: BLE001
            pass
        per_slice = _ctx.get("per_slice") or {}
        if _ctx.get("slice_disp") is None or not per_slice:
            with state:
                state.uct_status = (
                    "Nothing to save — load a slice and run "
                    "segmentation first."
                )
            return

        uct_dir = _uct_dir()
        if uct_dir is None:
            with state:
                state.uct_status = (
                    "Save failed: no active project. Open a "
                    "project first."
                )
            return

        _v = getattr(state, "uct_voxel_size_um", 0.0)
        try:
            v_um = float(_v) if _v is not None else 0.0
        except (TypeError, ValueError):
            v_um = 0.0
        source_path = str(
            getattr(state, "uct_file_path", "") or "",
        )

        # Walk every annotated slice. Each entry stores
        # (proposals, labels) at the CURRENT crop frame, so
        # we composite a labelled mask in that frame and save
        # it as `labels_slice<N>.png`. The crop tuple is also
        # persisted so a future restore can reconstruct the
        # display frame even if the user resized the window.
        try:
            from PIL import Image
        except Exception as ex:                           # noqa: BLE001
            with state:
                state.uct_status = (
                    f"Save failed (PIL missing): {ex}"
                )
            return

        crop_tuple = tuple(_ctx.get("crop", (0, 0, 0, 0)))
        slice_disp_h, slice_disp_w = (
            int(_ctx["slice_disp"].shape[0]),
            int(_ctx["slice_disp"].shape[1]),
        )
        total_epi = 0
        total_fasc = 0
        slices_out: list[dict] = []
        for idx, entry in sorted(per_slice.items()):
            props = entry.get("proposals") or []
            labels_list = entry.get("labels") or []
            if not props:
                continue
            n_epi = sum(1 for l in labels_list if l == "epi")
            n_fasc = sum(
                1 for l in labels_list if l == "fascicle"
            )
            if n_epi == 0 and n_fasc == 0:
                continue
            # Compose the labelled mask at the slice_disp
            # frame for this slice. All proposals are at the
            # same frame so the masks line up.
            labels_img = np.zeros(
                (slice_disp_h, slice_disp_w), dtype=np.uint8,
            )
            for prop, lab in zip(props, labels_list):
                label_idx = _LABEL_INDEX.get(lab, 0)
                if label_idx == 0 or label_idx == 255:
                    continue
                try:
                    labels_img[prop.mask] = label_idx
                except Exception:                         # noqa: BLE001
                    continue
            mask_name = f"labels_slice{int(idx)}.png"
            mask_path = uct_dir / mask_name
            try:
                Image.fromarray(labels_img).save(mask_path)
            except Exception as ex:                       # noqa: BLE001
                print(
                    f"[seg-save] slice {idx} mask write "
                    f"failed: {ex}",
                    flush=True,
                )
                continue
            slices_out.append({
                "slice_idx": int(idx),
                "mask_path": mask_name,
                "label_counts": {
                    "epi": int(n_epi),
                    "fascicle": int(n_fasc),
                },
            })
            total_epi += n_epi
            total_fasc += n_fasc

        if not slices_out:
            with state:
                state.uct_status = (
                    "Save: nothing labelled yet."
                )
            return

        # Keep the v1 top-level fields populated for backwards-
        # compat readers (they'll see the current/working
        # slice).
        cur_idx = int(getattr(state, "uct_slice_idx", 0) or 0)
        cur_entry = next(
            (s for s in slices_out
             if int(s["slice_idx"]) == cur_idx),
            slices_out[0],
        )

        payload = {
            "schema": SEGMENTATION_SCHEMA,
            "source_path": source_path,
            "source_name": Path(source_path).name,
            "voxel_size_um": v_um,
            "image_shape": [slice_disp_h, slice_disp_w],
            "crop": [int(c) for c in crop_tuple],
            "label_index": {
                k: v for k, v in _LABEL_INDEX.items()
                if v in (0, 1, 2)
            },
            "slices": slices_out,
            # v1-shaped top-level fields, for back-compat with
            # any older reader code in the wild.
            "slice_idx": int(cur_entry["slice_idx"]),
            "mask_path": cur_entry["mask_path"],
            "label_counts": {
                "epi": int(total_epi),
                "fascicle": int(total_fasc),
            },
            "n_proposals": int(
                sum(len(s.get("proposals") or [])
                    for s in per_slice.values())
            ),
        }
        json_path = uct_dir / "segmentation.json"
        try:
            with open(json_path, "w") as f:
                json.dump(payload, f, indent=2)
        except Exception as ex:                       # noqa: BLE001
            with state:
                state.uct_status = (
                    f"Save failed (JSON): {ex}"
                )
            return

        with state:
            state.uct_status = (
                f"Saved {len(slices_out)} slice"
                f"{'s' if len(slices_out) != 1 else ''} · "
                f"{total_epi} epi + {total_fasc} fascicle masks"
            )

    def do_finalize_segmentation(*_args) -> None:
        """Bundles the three end-of-Segment-step actions into
        one click:

          1. `do_refine_masks` — cleanup pass (fill holes, close
             gaps, smooth outlines, drop tiny / unlabelled
             proposals).
          2. `do_generate_epi` — derive the epineurium mask as
             `slice - background - fascicles` so the saved
             segmentation includes it.
          3. `do_save_uct_segmentation` — write `segmentation.json`
             + the labelled mask PNG into
             `<project>/uct/`.

        Each handler does its own no-op / status-message check
        for the "nothing to do" case (no proposals etc.), so
        calling them in sequence on a half-finished slice
        gracefully just runs the parts that have something to
        do — no need for guards here. The status line ends up
        showing the LAST handler's message (the save line);
        the previous two are visible in the busy log if needed.
        """
        do_refine_masks()
        do_generate_epi()
        do_save_uct_segmentation()
        # Flag the Step-2 "Next" button as unlocked. Any
        # subsequent label edit (paint / erase / chip click /
        # active-stamp click / re-segment) flips this back to
        # False via `_invalidate_finalize` so the user re-runs
        # Finalize before advancing — that way the saved
        # segmentation always reflects what they ACTUALLY end
        # up using in Step 3.
        with state:
            state.uct_step2_finalized = True

    def _try_restore_segmentation() -> None:
        """If `<project>/uct/segmentation.json` exists, rebuild
        the dialog state from it: re-load the source stack, jump
        to the saved slice, decompose the labelled mask into
        per-label connected components as fresh proposals, then
        re-render the overlay.

        Silent no-op when no project or no saved segmentation —
        the user just sees a clean dialog."""
        uct_dir = _uct_dir()
        if uct_dir is None:
            return
        json_path = uct_dir / "segmentation.json"
        if not json_path.is_file():
            return
        try:
            with open(json_path) as f:
                payload = json.load(f)
        except Exception as ex:                       # noqa: BLE001
            with state:
                state.uct_status = (
                    f"Found segmentation.json but could not "
                    f"read it: {ex}"
                )
            return
        # Re-load the source stack.
        source_path = str(payload.get("source_path", "") or "")
        if not source_path or not Path(source_path).exists():
            with state:
                state.uct_status = (
                    "Restored segmentation references a source "
                    "file that's missing — re-upload to "
                    "continue."
                )
            return
        with state:
            state.uct_file_path = source_path
        try:
            stack = load_stack(source_path)
        except Exception as ex:                       # noqa: BLE001
            with state:
                state.uct_status = (
                    f"Restore failed to open source: {ex}"
                )
            return
        _ctx["stack"] = stack
        slice_idx = int(payload.get("slice_idx", 0))
        slice_idx = max(0, min(stack.n_frames - 1, slice_idx))
        with state:
            state.uct_stack_loaded = True
            state.uct_slice_max = max(0, stack.n_frames - 1)
            state.uct_slice_idx = slice_idx
            state.uct_stack_info_html = (
                f"<b>{stack.n_frames}</b> frames · "
                f"{stack.height} × {stack.width} · "
                f"dtype <code>{stack.dtype}</code>"
            )
            vsz = payload.get("voxel_size_um", 0.0)
            if vsz:
                state.uct_voxel_size_um = float(vsz)
        # Read the slice + labelled mask, decompose into
        # connected components per label.
        arr = read_slice(stack, slice_idx)
        disp = to_display(
            arr,
            window=getattr(
                _ctx.get("stack"), "display_window", None,
            ),
        )
        _ctx["slice_arr"] = arr
        _ctx["slice_disp"] = disp
        # M33 — Build the list of (slice_idx, mask_path) pairs
        # to restore. v2 schema has the `slices` array;
        # v1 only has top-level slice_idx + mask_path.
        slices_to_restore: list[dict] = list(
            payload.get("slices") or [],
        )
        if not slices_to_restore:
            _legacy_mask = str(
                payload.get("mask_path", "") or "",
            )
            if _legacy_mask:
                slices_to_restore = [{
                    "slice_idx": int(
                        payload.get("slice_idx", 0) or 0,
                    ),
                    "mask_path": _legacy_mask,
                }]
        if not slices_to_restore:
            with state:
                state.uct_status = (
                    "Restored metadata but no slice masks "
                    "listed."
                )
            return

        try:
            from PIL import Image
            from skimage.measure import label, regionprops
        except Exception as ex:                       # noqa: BLE001
            with state:
                state.uct_status = (
                    f"Restored metadata but PIL/skimage "
                    f"missing: {ex}"
                )
            return

        # If a crop was saved, restore it BEFORE we decode
        # the masks so the slice_disp shape matches what was
        # saved (mask PNGs were saved in slice_disp frame).
        saved_crop = payload.get("crop")
        if (
            isinstance(saved_crop, (list, tuple))
            and len(saved_crop) == 4
        ):
            try:
                cx0, cy0, cx1, cy1 = (
                    int(saved_crop[0]),
                    int(saved_crop[1]),
                    int(saved_crop[2]),
                    int(saved_crop[3]),
                )
            except (TypeError, ValueError):
                cx0 = cy0 = cx1 = cy1 = 0
            full_h, full_w = (
                int(_ctx["slice_disp_full"].shape[0])
                if _ctx.get("slice_disp_full") is not None
                else 0,
                int(_ctx["slice_disp_full"].shape[1])
                if _ctx.get("slice_disp_full") is not None
                else 0,
            )
            if (
                full_h > 0 and full_w > 0
                and 0 <= cx0 < cx1 <= full_w
                and 0 <= cy0 < cy1 <= full_h
            ):
                _ctx["crop"] = (cx0, cy0, cx1, cy1)
                _ctx["slice_arr"] = _ctx["slice_arr_full"][
                    cy0:cy1 + 1, cx0:cx1 + 1,
                ]
                _ctx["slice_disp"] = _ctx["slice_disp_full"][
                    cy0:cy1 + 1, cx0:cx1 + 1,
                ]
                with state:
                    state.uct_crop_x_range = [cx0, cx1]
                    state.uct_crop_y_range = [cy0, cy1]

        # Rebuild per_slice from disk masks.
        new_per_slice: dict[int, dict] = {}
        active_slice_proposals: list[MaskProposal] = []
        active_slice_labels: list[str] = []
        cur_slice_idx = int(
            getattr(state, "uct_slice_idx", 0) or 0,
        )
        for entry in slices_to_restore:
            try:
                s_idx = int(entry.get("slice_idx", 0))
                m_name = str(entry.get("mask_path", "") or "")
            except (TypeError, ValueError):
                continue
            if not m_name:
                continue
            m_path = uct_dir / m_name
            if not m_path.is_file():
                print(
                    f"[seg-restore] slice {s_idx} mask missing "
                    f"({m_path}) — skipping",
                    flush=True,
                )
                continue
            try:
                labels_img = np.asarray(
                    Image.open(m_path), dtype=np.uint8,
                )
            except Exception as ex:                   # noqa: BLE001
                print(
                    f"[seg-restore] slice {s_idx} mask load "
                    f"failed: {ex}",
                    flush=True,
                )
                continue
            s_props: list[MaskProposal] = []
            s_labels: list[str] = []
            for lab_idx, lab_name in _INDEX_LABEL.items():
                if lab_idx == 0:
                    continue
                comp = label(
                    labels_img == lab_idx, connectivity=2,
                )
                for r in regionprops(comp):
                    m = comp == r.label
                    y0, x0, y1, x1 = r.bbox
                    s_props.append(MaskProposal(
                        mask=m, score=1.0,
                        bbox=(int(x0), int(y0),
                              int(x1) - 1, int(y1) - 1),
                        area_px=int(r.area),
                    ))
                    s_labels.append(lab_name)
            if s_props:
                new_per_slice[s_idx] = {
                    "proposals": s_props,
                    "labels": s_labels,
                    "colors": list(
                        generate_proposal_colors(len(s_props)),
                    ),
                }
                if s_idx == cur_slice_idx:
                    active_slice_proposals = s_props
                    active_slice_labels = s_labels

        if not new_per_slice:
            with state:
                state.uct_status = (
                    "Restored metadata but no slice masks "
                    "could be decoded."
                )
            return

        _ctx["per_slice"] = new_per_slice
        # If the current slice isn't in the saved set, pick
        # the first one we have so the dialog has something
        # to show on first open.
        if not active_slice_proposals:
            first_idx = next(iter(new_per_slice))
            with state:
                state.uct_slice_idx = first_idx
            _read_and_render_slice(int(first_idx))
            active_slice_proposals = new_per_slice[first_idx][
                "proposals"
            ]
            active_slice_labels = new_per_slice[first_idx][
                "labels"
            ]
        _ctx["proposals"] = active_slice_proposals
        _ctx["labels"] = active_slice_labels
        colors = generate_proposal_colors(
            len(active_slice_proposals),
        )
        _ctx["proposal_colors"] = colors
        meta = [
            {
                "idx": i,
                "area_px": int(p.area_px),
                "bbox_str": (
                    f"({p.bbox[0]},{p.bbox[1]})-"
                    f"({p.bbox[2]},{p.bbox[3]})"
                ),
                "label": active_slice_labels[i],
                "color_hex": color_to_hex(colors[i]),
            }
            for i, p in enumerate(active_slice_proposals)
        ]
        with state:
            state.uct_proposals_meta = meta
            state.uct_status = (
                f"Restored {len(new_per_slice)} slice"
                f"{'s' if len(new_per_slice) != 1 else ''} "
                f"from {json_path.name}."
            )
        _refresh_label_counts()
        _rerender_overlay()

    # ----------------------------------------------------------
    # V1 Phase B — Step-2 3D reconstruction.
    #
    # Three handlers + one helper. The flow:
    #   do_recon_next   → flip uct_step to 2, prefill Z-spacing
    #                     from stack metadata, refresh coverage
    #   do_recon_back   → flip uct_step back to 1
    #   do_run_reconstruction → async; build per-slice mask map,
    #                           call reconstruct3d, write STLs
    #                           to <project>/uct/nerve_3d/<ts>/
    # ----------------------------------------------------------

    def _collect_per_slice_masks() -> tuple[
        dict[int, np.ndarray], dict[int, np.ndarray],
    ]:
        """Walk `_ctx["per_slice"]` plus the current working
        slice and return two sparse dicts:
            (epi_per_slice, fasc_per_slice)
        keyed by slice idx, value = unioned bool mask for that
        class on that slice. Only slices that have at least one
        epi OR fascicle proposal contribute an entry (background-
        only slices are gaps, by definition).

        **Epi is SOLIDIFIED here**: the auto-derived epi mask
        from `_recompute_epi` has the fascicles subtracted
        (`~(fasc ∪ bg ∪ manual_epi)`) so the Step-2 overlay
        renders correctly — but extruding *that* mask into 3D
        gives a genus-N shell with N cylindrical tunnels
        carved through it for the fascicles. Marching cubes
        then produces the inner-wall slivers that show up as
        the red bar in the quality histogram + the sharp
        edges the user flagged.

        For the 3D meshing path we want the epi to be a
        SOLID prism — `epi_solid = epi ∪ fasc = ~bg`. The
        fascicles are still passed separately as inner
        surfaces, and TetGen carves the per-fascicle regions
        out of the epi at meshing time (see the
        `inner_surfaces=` kwarg in
        `pipeline.plc.assemble_multi_domain_plc`). No need
        to pre-subtract them in the segmentation masks.
        """
        # Snapshot the current slice into per_slice first so
        # the in-progress edits are picked up even when the
        # user hasn't scrolled.
        _commit_current_slice_to_cache()
        epi_map: dict[int, np.ndarray] = {}
        fasc_map: dict[int, np.ndarray] = {}
        for idx, entry in _ctx["per_slice"].items():
            props = entry.get("proposals", [])
            labels = entry.get("labels", [])
            if not props:
                continue
            epi_u: Optional[np.ndarray] = None
            fasc_u: Optional[np.ndarray] = None
            bg_u: Optional[np.ndarray] = None
            for p, lab in zip(props, labels):
                if lab == "epi":
                    epi_u = (
                        p.mask if epi_u is None
                        else (epi_u | p.mask)
                    )
                elif lab == "fascicle":
                    fasc_u = (
                        p.mask if fasc_u is None
                        else (fasc_u | p.mask)
                    )
                elif lab == "background":
                    bg_u = (
                        p.mask if bg_u is None
                        else (bg_u | p.mask)
                    )
            # M32 — subtract BG from EPI. By definition any pixel
            # labelled `background` is NOT epineurium, so it has
            # no business being in the EPI mask that gets
            # extruded into the 3D volume. Without this, a
            # SAM2-emitted "whole-image" proposal labelled as EPI
            # (which happens when the segmenter picks a coarse
            # bounding-rectangle proposal that the user assigns
            # to EPI thinking it's the perineurium ring) would
            # extrude as a cuboid covering the entire image
            # rectangle — exactly the symptom the user reported.
            if epi_u is not None and bg_u is not None:
                epi_u = epi_u & ~bg_u
            # Solidify the epi by unioning the fascicles back in
            # — the overlay-display mask had them subtracted, but
            # the 3D mesh wants a hole-free outer hull (see the
            # docstring).
            if epi_u is not None and fasc_u is not None:
                epi_u = epi_u | fasc_u
            if epi_u is not None and epi_u.any():
                epi_map[int(idx)] = epi_u
            if fasc_u is not None and fasc_u.any():
                fasc_map[int(idx)] = fasc_u
        return epi_map, fasc_map

    def _annotated_indices() -> list[int]:
        """Sorted list of slice indices with at least one
        epi or fascicle proposal. Drives the Step-2 coverage
        readout + default slice-range picker."""
        epi_map, fasc_map = _collect_per_slice_masks()
        return sorted(set(epi_map.keys()) | set(fasc_map.keys()))

    def _refresh_recon_coverage() -> None:
        """Push the annotated-slice list + a coverage string
        into state for the Step-3 panel to render.

        Also rebuilds `uct_recon_annotated_items` — the VSelect
        items list for the single-slice picker. Each entry is
        `{value: <int slice idx>, title: <slice N · counts>}`
        so the dropdown reads "slice 12 · 3 fascicles + epi"
        and only annotated slices are pickable (no gaps).
        """
        epi_map, fasc_map = _collect_per_slice_masks()
        annotated = sorted(
            set(epi_map.keys()) | set(fasc_map.keys()),
        )
        items = []
        for idx in annotated:
            has_epi = idx in epi_map
            has_fasc = idx in fasc_map
            parts: list[str] = []
            if has_fasc:
                # Count fascicle CCs on this slice so the
                # picker is informative — "3 fascicles" reads
                # better than just "with fascicles".
                try:
                    from skimage.measure import label as _lbl
                    n_cc = int(_lbl(
                        fasc_map[idx], connectivity=2,
                    ).max())
                except Exception:                        # noqa: BLE001
                    n_cc = 1
                parts.append(
                    f"{n_cc} fascicle"
                    f"{'s' if n_cc != 1 else ''}"
                )
            if has_epi:
                parts.append("epi")
            tag = " + ".join(parts) if parts else "labelled"
            items.append({
                "value": int(idx),
                "title": f"slice {int(idx)} · {tag}",
            })
        try:
            s_lo = int(
                getattr(state, "uct_recon_slice_start", 0) or 0,
            )
            s_hi = int(
                getattr(state, "uct_recon_slice_end", 0) or 0,
            )
        except (TypeError, ValueError):
            s_lo, s_hi = 0, 0
        in_range = [i for i in annotated if s_lo <= i <= s_hi]
        total = max(0, s_hi - s_lo + 1)
        n_gaps = max(0, total - len(in_range))
        with state:
            state.uct_recon_annotated = annotated
            state.uct_recon_annotated_items = items
            state.uct_recon_coverage_msg = (
                f"{len(in_range)} / {total} slices annotated in "
                f"range · {n_gaps} ZOH-filled"
                if total > 0 else
                "Pick a slice range to see coverage."
            )

    def do_recon_next(*_args) -> None:
        """Step 1 → Step 2. Prefills Z-spacing from stack
        metadata if available (and the user hasn't already
        edited it) and defaults the slice range to all
        annotated slices."""
        annotated = _annotated_indices()
        if not annotated:
            with state:
                state.uct_status = (
                    "Annotate at least one slice "
                    "(fascicle or epineurium) before "
                    "reconstructing."
                )
            return
        stack = _ctx.get("stack")
        # Auto-fill Z spacing (mm) from voxel_size_um[0] when
        # available and the user hasn't already entered one.
        try:
            cur_z = float(
                getattr(state, "uct_recon_voxel_z_mm", 0.0)
                or 0.0,
            )
        except (TypeError, ValueError):
            cur_z = 0.0
        z_mm = cur_z
        if z_mm <= 0 and stack is not None:
            vsz = stack.voxel_size_um
            if vsz is not None and len(vsz) >= 1:
                try:
                    z_mm = float(vsz[0]) / 1000.0
                except (TypeError, ValueError):
                    z_mm = 0.0
        if z_mm <= 0:
            # Fall back to the XY voxel size (assume isotropic).
            try:
                xy_um = float(
                    getattr(state, "uct_voxel_size_um", 0.0)
                    or 0.0,
                )
            except (TypeError, ValueError):
                xy_um = 0.0
            if xy_um > 0:
                z_mm = xy_um / 1000.0
        # Default slice range = annotated extent.
        cur_lo = int(
            getattr(state, "uct_recon_slice_start", 0) or 0,
        )
        cur_hi = int(
            getattr(state, "uct_recon_slice_end", 0) or 0,
        )
        if cur_lo == 0 and cur_hi == 0:
            cur_lo = annotated[0]
            cur_hi = annotated[-1]
        # Default single-slice idx = current slice.
        cur_single = int(
            getattr(state, "uct_recon_single_slice_idx", -1)
            or -1,
        )
        if cur_single < 0:
            cur_single = int(
                _ctx.get("current_slice_idx", annotated[0])
            )
        with state:
            # VStepper v-model is string-typed — "3" advances
            # the visible window to the Reconstruct-3D pane.
            state.uct_step = "3"
            state.uct_recon_voxel_z_mm = float(z_mm)
            state.uct_recon_slice_start = cur_lo
            state.uct_recon_slice_end = cur_hi
            state.uct_recon_single_slice_idx = cur_single
            state.uct_recon_files = []
            state.uct_recon_status = ""
        _refresh_recon_coverage()

    def do_recon_back(*_args) -> None:
        """Step 3 → Step 2. State stays — the user can come
        back to Reconstruct without losing their settings."""
        with state:
            state.uct_step = "2"

    def _on_recon_range_change(
        uct_recon_slice_start, uct_recon_slice_end, **_kw,
    ) -> None:
        _refresh_recon_coverage()

    async def do_run_reconstruction(*_args) -> None:
        """Async — runs the chosen reconstruction mode, writes
        per-class STLs into `<project>/uct/nerve_3d/<ts>/`,
        and updates `state.uct_recon_files` with the relative
        paths. `state.busy` is held high during the run so the
        global lightbox covers it.
        """
        import asyncio
        import datetime as _dt
        mode = str(
            getattr(state, "uct_recon_mode", "multi") or "multi",
        )
        try:
            xy_um = float(
                getattr(state, "uct_voxel_size_um", 0.0) or 0.0,
            )
            z_mm = float(
                getattr(state, "uct_recon_voxel_z_mm", 0.0)
                or 0.0,
            )
            thickness_mm = float(
                getattr(state, "uct_recon_thickness_mm", 0.0)
                or 0.0,
            )
        except (TypeError, ValueError):
            with state:
                state.uct_recon_status = (
                    "Voxel / thickness fields must be "
                    "numbers."
                )
            return
        xy_mm = xy_um / 1000.0
        if xy_mm <= 0:
            with state:
                state.uct_recon_status = (
                    "Set the XY voxel size (µm) in Step 1 "
                    "before reconstructing."
                )
            return
        if mode == "multi" and z_mm <= 0:
            with state:
                state.uct_recon_status = (
                    "Z spacing (mm) is required for "
                    "marching-cubes mode."
                )
            return
        if mode == "single" and thickness_mm <= 0:
            with state:
                state.uct_recon_status = (
                    "Extrusion thickness (mm) is required "
                    "for single-slice mode."
                )
            return
        smooth_on = bool(
            getattr(state, "uct_recon_smooth", True),
        )
        try:
            sigma = float(
                getattr(state, "uct_recon_smooth_sigma", 1.0)
                or 1.0,
            )
        except (TypeError, ValueError):
            sigma = 1.0
        smooth_sigma: Optional[float] = (
            sigma if (smooth_on and sigma > 0) else None
        )
        # M30 — decoupled physical-mm sigmas. Read separately;
        # the geometry layer uses them when either is non-zero
        # (overriding the legacy isotropic-voxel `smooth_sigma`).
        try:
            _sxy_mm = float(
                getattr(
                    state,
                    "uct_recon_smooth_sigma_xy_mm",
                    0.005,
                ) or 0.0,
            )
        except (TypeError, ValueError):
            _sxy_mm = 0.005
        try:
            _sz_mm = float(
                getattr(
                    state,
                    "uct_recon_smooth_sigma_z_mm",
                    0.3,
                ) or 0.0,
            )
        except (TypeError, ValueError):
            _sz_mm = 0.3
        smooth_sigma_xy_mm: Optional[float] = (
            _sxy_mm if (smooth_on and _sxy_mm > 0) else None
        )
        smooth_sigma_z_mm: Optional[float] = (
            _sz_mm if (smooth_on and _sz_mm > 0) else None
        )

        epi_map, fasc_map = _collect_per_slice_masks()
        if not epi_map and not fasc_map:
            with state:
                state.uct_recon_status = (
                    "No annotated slices found. Label at "
                    "least one slice (fascicle or "
                    "epineurium) first."
                )
            return

        uct_dir = _uct_dir()
        if uct_dir is None:
            with state:
                state.uct_recon_status = (
                    "No active project — open a project "
                    "before saving the 3D nerve."
                )
            return

        ts = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        out_dir = uct_dir / "nerve_3d" / ts
        out_dir.mkdir(parents=True, exist_ok=True)

        with state:
            state.uct_busy = True
            state.busy = True
            state.busy_msg = (
                "Extruding nerve…" if mode == "single"
                else "Reconstructing nerve volume…"
            )
            # M25 — start busy_log empty so the on_progress
            # callback's accumulating tail isn't sitting on
            # top of a stale "Building masks…" line. The
            # first stage emission ("ZOH-fill slices") fires
            # within a few ms anyway.
            state.busy_log = ""
            state.uct_recon_status = ""
            state.uct_recon_files = []
        state.flush()
        await asyncio.sleep(0)

        loop = asyncio.get_event_loop()
        # Optional refinement (drop specks → Taubin → pymeshfix
        # → defensive trimesh pass → optional isotropic remesh
        # → optimesh CVT). NO decimation — see the comment on
        # `refine_mesh` for why.
        do_refine = bool(
            getattr(state, "uct_recon_refine", True),
        )
        do_remesh = bool(
            getattr(state, "uct_recon_remesh", False),
        )
        do_optimesh = bool(
            getattr(state, "uct_recon_use_optimesh", False),
        )
        try:
            remesh_edge_len_mm: Optional[float] = (
                float(
                    getattr(
                        state,
                        "uct_recon_edge_len_um",
                        50.0,
                    ) or 50.0,
                ) / 1000.0
            )
        except (TypeError, ValueError):
            remesh_edge_len_mm = 0.050
        # User-driven per-mesh decimation budget. 0 / unset
        # means "no decimation" (legacy behaviour).
        try:
            _dec_target = int(
                getattr(
                    state,
                    "uct_recon_decimate_target_tris",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _dec_target = 0
        decimate_target_tris: Optional[int] = (
            _dec_target if _dec_target > 0 else None
        )
        # M27 — surface size-control mode override. The combobox
        # `uct_recon_size_mode` dictates which knob is honoured:
        #   "off"          → both decimate paths disabled +
        #                    remesh forced off
        #   "fraction"     → decimate_target_fraction takes
        #                    precedence; decimate_target_tris off
        #   "target_tris"  → legacy behaviour, decimate_target_tris
        #                    honored as-is
        #   "isotropic"    → remesh on (forces do_remesh=True
        #                    using uct_recon_edge_len_um), both
        #                    decimate paths off
        _size_mode = str(
            getattr(state, "uct_recon_size_mode", "off")
            or "off",
        ).lower()
        try:
            _dec_fraction = float(
                getattr(
                    state,
                    "uct_recon_decimate_fraction",
                    0.5,
                ) or 0.5,
            )
        except (TypeError, ValueError):
            _dec_fraction = 0.5
        decimate_target_fraction: Optional[float] = None
        if _size_mode == "fraction":
            decimate_target_tris = None
            if 0.0 < _dec_fraction < 1.0:
                decimate_target_fraction = _dec_fraction
            do_remesh = False
        elif _size_mode == "target_tris":
            decimate_target_fraction = None
            do_remesh = False
        elif _size_mode == "isotropic":
            decimate_target_tris = None
            decimate_target_fraction = None
            do_remesh = True
        else:  # "off"
            decimate_target_tris = None
            decimate_target_fraction = None
            do_remesh = False
        # 2D mask cleanup before marching cubes. Both knobs in
        # pixels at source resolution; 0 = pass that
        # direction through unchanged.
        try:
            _clean_comp_px = int(
                getattr(
                    state,
                    "uct_recon_clean_min_component_px",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _clean_comp_px = 0
        try:
            _clean_hole_px = int(
                getattr(
                    state,
                    "uct_recon_clean_min_hole_px",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _clean_hole_px = 0
        # 2D morphological closing radius (pixels). 0 = off.
        try:
            _clean_close_r = int(
                getattr(
                    state,
                    "uct_recon_clean_closing_radius_px",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _clean_close_r = 0
        # 3D volume cleanup (voxels). 0 = off.
        try:
            _clean_3d_comp_vox = int(
                getattr(
                    state,
                    "uct_recon_clean_3d_min_component_vox",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _clean_3d_comp_vox = 0
        try:
            _clean_3d_hole_vox = int(
                getattr(
                    state,
                    "uct_recon_clean_3d_min_hole_vox",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _clean_3d_hole_vox = 0
        try:
            _fasc_inset_vox = int(
                getattr(
                    state,
                    "uct_recon_fasc_inset_vox",
                    2,
                ) or 0,
            )
        except (TypeError, ValueError):
            _fasc_inset_vox = 2

        def _build_meshes() -> list[r3d.Mesh]:
            if mode == "single":
                idx = int(
                    getattr(
                        state, "uct_recon_single_slice_idx", 0,
                    ) or 0,
                )
                epi_mask = epi_map.get(idx)
                fasc_mask = fasc_map.get(idx)
                shape = None
                if epi_mask is not None:
                    shape = epi_mask.shape
                elif fasc_mask is not None:
                    shape = fasc_mask.shape
                else:
                    raise RuntimeError(
                        f"Slice {idx} has no labelled "
                        f"epi / fascicle masks."
                    )
                if epi_mask is None:
                    epi_mask = np.zeros(shape, dtype=bool)
                if fasc_mask is None:
                    fasc_mask = np.zeros(shape, dtype=bool)
                return r3d.extrude_single_slice(
                    epi_mask, fasc_mask,
                    voxel_xy_mm=xy_mm,
                    thickness_mm=thickness_mm,
                    smooth_sigma=smooth_sigma,
                    smooth_sigma_xy_mm=smooth_sigma_xy_mm,
                    smooth_sigma_z_mm=smooth_sigma_z_mm,
                    refine=do_refine,
                    refine_use_optimesh=do_optimesh,
                    refine_remesh=do_remesh,
                    refine_remesh_edge_len_mm=(
                        remesh_edge_len_mm
                    ),
                    decimate_target_tris=(
                        decimate_target_tris
                    ),
                    decimate_target_fraction=(
                        decimate_target_fraction
                    ),
                    clean_min_component_px=_clean_comp_px,
                    clean_min_hole_px=_clean_hole_px,
                    clean_closing_radius_px=(
                        _clean_close_r
                    ),
                    fasc_inset_vox=_fasc_inset_vox,
                )
            # multi
            s_lo = int(
                getattr(state, "uct_recon_slice_start", 0) or 0,
            )
            s_hi = int(
                getattr(state, "uct_recon_slice_end", 0) or 0,
            )
            if s_hi < s_lo:
                s_lo, s_hi = s_hi, s_lo

            # Progress callback — fired by reconstruct_stack
            # at every major checkpoint AND by refine_mesh via
            # the same on_progress hook. APPENDS to busy_log
            # (rather than overwriting) so the user sees an
            # accumulating multi-line log of the actual stages
            # in flight (drop_specks → Taubin → pymeshfix →
            # trimesh → optimesh → fasc inset → marching cubes
            # → ...). Tail-truncates at MAX_LINES so the log
            # stays bounded and the most recent N events
            # always fit in the lightbox without scrolling.
            # Runs from the executor thread; Trame's state
            # writes are safe from worker threads (the next
            # event-loop tick pushes them).
            _BUSY_LOG_MAX_LINES = 15

            def _recon_progress(stage: str, elapsed: float) -> None:
                msg = f"[{elapsed:5.1f}s] {stage}"
                try:
                    prev = str(
                        getattr(state, "busy_log", "") or "",
                    )
                    lines = prev.split("\n") if prev else []
                    lines.append(msg)
                    if len(lines) > _BUSY_LOG_MAX_LINES:
                        lines = lines[-_BUSY_LOG_MAX_LINES:]
                    state.busy_log = "\n".join(lines)
                except Exception:                     # noqa: BLE001
                    pass

            return r3d.reconstruct_stack(
                epi_map, fasc_map,
                voxel_xy_mm=xy_mm,
                voxel_z_mm=z_mm,
                slice_range=(s_lo, s_hi),
                smooth_sigma=smooth_sigma,
                smooth_sigma_xy_mm=smooth_sigma_xy_mm,
                smooth_sigma_z_mm=smooth_sigma_z_mm,
                refine=do_refine,
                refine_use_optimesh=do_optimesh,
                refine_remesh=do_remesh,
                refine_remesh_edge_len_mm=remesh_edge_len_mm,
                decimate_target_tris=(
                    decimate_target_tris
                ),
                decimate_target_fraction=(
                    decimate_target_fraction
                ),
                clean_min_component_px=_clean_comp_px,
                clean_min_hole_px=_clean_hole_px,
                clean_closing_radius_px=_clean_close_r,
                clean_3d_min_component_vox=(
                    _clean_3d_comp_vox
                ),
                clean_3d_min_hole_vox=_clean_3d_hole_vox,
                fasc_inset_vox=_fasc_inset_vox,
                on_progress=_recon_progress,
            )

        try:
            meshes = await loop.run_in_executor(
                None, _build_meshes,
            )
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_recon_status = (
                    f"Reconstruction failed: {ex}"
                )
            return

        if not meshes:
            with state:
                state.uct_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_recon_status = (
                    "Reconstruction produced no surfaces — "
                    "check that the chosen slice/range "
                    "actually contains labelled masks."
                )
            return

        # Write STLs + manifest. Manifest is the same one V1
        # Phase D will read from the import wizard tile to
        # list candidate nerves.
        with state:
            state.busy_log = (
                f"Writing {len(meshes)} STL file"
                + ("s" if len(meshes) != 1 else "")
                + "…"
            )
        state.flush()
        await asyncio.sleep(0)
        files: list[str] = []
        try:
            for m in meshes:
                stl_path = out_dir / f"{m.name}.stl"
                await loop.run_in_executor(
                    None, r3d.write_binary_stl, m, stl_path,
                )
                files.append(stl_path.name)
            # Count fascicle / endo regions for the bundle
            # summary the import-wizard tile shows. Single-
            # slice extrude emits one `fascicle_<i>.stl` per 2D
            # connected component (true per-fascicle identity).
            # Multi-slice marching-cubes emits one combined
            # `endoneurium.stl` (the union — see reconstruct_-
            # stack for why we don't split there). Count an
            # endoneurium.stl as 1 region for display purposes.
            n_fascicles = sum(
                1 for fn in files
                if fn.startswith("fascicle_")
                or fn == "endoneurium.stl"
            )
            manifest = {
                # Identity fields — the import-wizard bundle
                # picker matches on `kind` to distinguish a
                # Golgi-generated bundle from an arbitrary dir
                # of STLs the user might drop in. `schema`
                # versions the bundle layout; bump it when the
                # set of files or the manifest shape changes
                # in a backwards-incompatible way.
                "kind": "golgi-uct-nerve",
                "schema": "v1",
                "mode": mode,
                "n_fascicles": int(n_fascicles),
                "voxel_xy_mm": xy_mm,
                "voxel_z_mm": (
                    z_mm if mode == "multi" else None
                ),
                "thickness_mm": (
                    thickness_mm if mode == "single" else None
                ),
                "smooth_sigma": smooth_sigma,
                "slice_idx": (
                    int(getattr(
                        state,
                        "uct_recon_single_slice_idx",
                        0,
                    ) or 0)
                    if mode == "single" else None
                ),
                "slice_range": (
                    [
                        int(getattr(
                            state,
                            "uct_recon_slice_start", 0,
                        ) or 0),
                        int(getattr(
                            state,
                            "uct_recon_slice_end", 0,
                        ) or 0),
                    ] if mode == "multi" else None
                ),
                "annotated_slices": _annotated_indices(),
                "files": files,
            }
            with open(out_dir / "manifest.json", "w") as f:
                json.dump(manifest, f, indent=2)
            # M13 Phase 1 — per-surface mesh quality diagnostics.
            # M14 Phase 2 — inter-surface diagnostics (pairwise
            # implicit-distance + KDTree) for PLC / TetGen triage.
            # Both written into mesh_diagnostics.json next to the
            # STLs + manifest so a later PLC failure can be
            # triaged offline. Best-effort: a report exception
            # mustn't fail the reconstruct step.
            try:
                reports = r3d.report_mesh_quality_batch(
                    meshes, prefix="[mesh-quality:json]",
                )
                inter = r3d.inter_surface_report(meshes)
                r3d.print_inter_surface_summary(inter)
                diagnostics = {
                    "per_surface": reports,
                    "inter_surface": inter,
                }
                with open(
                    out_dir / "mesh_diagnostics.json", "w",
                ) as f:
                    json.dump(diagnostics, f, indent=2)
            except Exception as ex:                     # noqa: BLE001
                traceback.print_exc()
                print(
                    "[recon] mesh-quality report write "
                    f"failed (continuing): {ex}",
                    flush=True,
                )
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_recon_status = (
                    f"STL write failed: {ex}"
                )
            return

        # Relative path (from project root) for display +
        # downstream import-wizard reference.
        rel_dir = out_dir.relative_to(uct_dir.parent)
        # Push freshly-built meshes into the in-dialog plotter
        # via the callback app.py wired in. The callback owns
        # the PyVista actor management + legend / histogram
        # state — we just hand it the meshes.
        if on_recon_meshes_ready is not None:
            try:
                on_recon_meshes_ready(meshes)
            except Exception:                            # noqa: BLE001
                traceback.print_exc()
        with state:
            state.uct_busy = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.uct_recon_files = [
                {
                    "name": fn,
                    "path": str(rel_dir / fn),
                }
                for fn in files
            ]
            # Stash the bundle id (the timestamp subdir name) so
            # the "Done → Import wizard" button knows what to
            # hand off. Also flips the import-wizard source
            # type so the bundle tile is pre-selected when the
            # wizard opens.
            state.uct_last_bundle_id = str(ts)
            state.uct_recon_status = (
                f"Wrote {len(files)} STL file"
                + ("s" if len(files) != 1 else "")
                + f" → {rel_dir}/"
            )

    async def do_run_reconstruction_preview(*_args) -> None:
        """Build the same meshes `do_run_reconstruction` would
        and render an offscreen preview PNG, but DON'T write
        STLs to disk. The user can iterate on parameters
        cheaply (single-slice extrude on a 1k-wide mask is
        ~1 s) and only commit to writing files when they're
        happy with the geometry.

        Shares the heavy mesh-building helper with
        `do_run_reconstruction` by leaning on the same closure
        cache for masks + voxel sizes. Marks `state.uct_busy`
        but uses a different message so the user can tell a
        preview pass apart from a write-to-disk pass.
        """
        import asyncio
        import base64 as _b64
        mode = str(
            getattr(state, "uct_recon_mode", "multi") or "multi",
        )
        try:
            xy_um = float(
                getattr(state, "uct_voxel_size_um", 0.0)
                or 0.0,
            )
            z_mm = float(
                getattr(state, "uct_recon_voxel_z_mm", 0.0)
                or 0.0,
            )
            thickness_mm = float(
                getattr(state, "uct_recon_thickness_mm", 0.0)
                or 0.0,
            )
        except (TypeError, ValueError):
            with state:
                state.uct_recon_status = (
                    "Voxel / thickness fields must be "
                    "numbers."
                )
            return
        xy_mm = xy_um / 1000.0
        if xy_mm <= 0 or (
            mode == "multi" and z_mm <= 0
        ) or (
            mode == "single" and thickness_mm <= 0
        ):
            with state:
                state.uct_recon_status = (
                    "Set XY voxel + (thickness | Z spacing) "
                    "before previewing."
                )
            return
        smooth_on = bool(
            getattr(state, "uct_recon_smooth", True),
        )
        try:
            sigma = float(
                getattr(state, "uct_recon_smooth_sigma", 1.0)
                or 1.0,
            )
        except (TypeError, ValueError):
            sigma = 1.0
        smooth_sigma = (
            sigma if (smooth_on and sigma > 0) else None
        )
        # M30 — decoupled physical-mm sigmas (preview).
        try:
            _sxy_mm_prev = float(
                getattr(
                    state,
                    "uct_recon_smooth_sigma_xy_mm",
                    0.005,
                ) or 0.0,
            )
        except (TypeError, ValueError):
            _sxy_mm_prev = 0.005
        try:
            _sz_mm_prev = float(
                getattr(
                    state,
                    "uct_recon_smooth_sigma_z_mm",
                    0.3,
                ) or 0.0,
            )
        except (TypeError, ValueError):
            _sz_mm_prev = 0.3
        smooth_sigma_xy_mm_prev = (
            _sxy_mm_prev if (smooth_on and _sxy_mm_prev > 0)
            else None
        )
        smooth_sigma_z_mm_prev = (
            _sz_mm_prev if (smooth_on and _sz_mm_prev > 0)
            else None
        )
        do_refine = bool(
            getattr(state, "uct_recon_refine", True),
        )
        do_remesh = bool(
            getattr(state, "uct_recon_remesh", False),
        )
        do_optimesh = bool(
            getattr(state, "uct_recon_use_optimesh", False),
        )
        # M27 — size-mode dispatch (preview). Match Generate.
        _size_mode_prev = str(
            getattr(state, "uct_recon_size_mode", "off")
            or "off",
        ).lower()
        try:
            _dec_target_prev = int(
                getattr(
                    state,
                    "uct_recon_decimate_target_tris",
                    0,
                ) or 0,
            )
        except (TypeError, ValueError):
            _dec_target_prev = 0
        try:
            _dec_fraction_prev = float(
                getattr(
                    state,
                    "uct_recon_decimate_fraction",
                    0.5,
                ) or 0.5,
            )
        except (TypeError, ValueError):
            _dec_fraction_prev = 0.5
        decimate_target_tris_prev: Optional[int] = None
        decimate_target_fraction_prev: Optional[float] = None
        if _size_mode_prev == "fraction":
            if 0.0 < _dec_fraction_prev < 1.0:
                decimate_target_fraction_prev = (
                    _dec_fraction_prev
                )
            do_remesh = False
        elif _size_mode_prev == "target_tris":
            if _dec_target_prev > 0:
                decimate_target_tris_prev = _dec_target_prev
            do_remesh = False
        elif _size_mode_prev == "isotropic":
            do_remesh = True
        else:
            do_remesh = False
        try:
            remesh_edge_len_mm: Optional[float] = (
                float(
                    getattr(
                        state,
                        "uct_recon_edge_len_um",
                        50.0,
                    ) or 50.0,
                ) / 1000.0
            )
        except (TypeError, ValueError):
            remesh_edge_len_mm = 0.050
        epi_map, fasc_map = _collect_per_slice_masks()
        if not epi_map and not fasc_map:
            with state:
                state.uct_recon_status = (
                    "No annotated slices found. Label at "
                    "least one slice first."
                )
            return

        with state:
            state.uct_busy = True
            state.busy = True
            state.busy_msg = "Previewing 3D nerve…"
            # M25 — start busy_log empty so the per-stage
            # progress callback's accumulating log isn't
            # sitting on top of a stale "Building meshes…".
            state.busy_log = ""
        state.flush()
        await asyncio.sleep(0)

        loop = asyncio.get_event_loop()

        def _build_meshes_for_preview() -> list:
            if mode == "single":
                idx = int(
                    getattr(
                        state,
                        "uct_recon_single_slice_idx",
                        0,
                    ) or 0,
                )
                epi_mask = epi_map.get(idx)
                fasc_mask = fasc_map.get(idx)
                shape = None
                if epi_mask is not None:
                    shape = epi_mask.shape
                elif fasc_mask is not None:
                    shape = fasc_mask.shape
                else:
                    raise RuntimeError(
                        f"Slice {idx} has no labelled masks."
                    )
                if epi_mask is None:
                    epi_mask = np.zeros(shape, dtype=bool)
                if fasc_mask is None:
                    fasc_mask = np.zeros(shape, dtype=bool)
                _fasc_inset_vox_preview = int(
                    getattr(
                        state,
                        "uct_recon_fasc_inset_vox",
                        2,
                    ) or 0,
                )
                meshes = r3d.extrude_single_slice(
                    epi_mask, fasc_mask,
                    voxel_xy_mm=xy_mm,
                    thickness_mm=thickness_mm,
                    smooth_sigma=smooth_sigma,
                    smooth_sigma_xy_mm=smooth_sigma_xy_mm_prev,
                    smooth_sigma_z_mm=smooth_sigma_z_mm_prev,
                    refine=do_refine,
                    refine_use_optimesh=do_optimesh,
                    refine_remesh=do_remesh,
                    refine_remesh_edge_len_mm=(
                        remesh_edge_len_mm
                    ),
                    decimate_target_tris=(
                        decimate_target_tris_prev
                    ),
                    decimate_target_fraction=(
                        decimate_target_fraction_prev
                    ),
                    fasc_inset_vox=_fasc_inset_vox_preview,
                )
            else:
                s_lo = int(
                    getattr(
                        state,
                        "uct_recon_slice_start", 0,
                    ) or 0,
                )
                s_hi = int(
                    getattr(
                        state,
                        "uct_recon_slice_end", 0,
                    ) or 0,
                )
                if s_hi < s_lo:
                    s_lo, s_hi = s_hi, s_lo
                _PREV_BUSY_LOG_MAX_LINES = 15

                def _preview_progress(
                    stage: str, elapsed: float,
                ) -> None:
                    msg = (
                        f"[preview {elapsed:5.1f}s] {stage}"
                    )
                    try:
                        prev = str(
                            getattr(
                                state, "busy_log", "",
                            ) or "",
                        )
                        lines = (
                            prev.split("\n") if prev else []
                        )
                        lines.append(msg)
                        if len(lines) > _PREV_BUSY_LOG_MAX_LINES:
                            lines = (
                                lines[-_PREV_BUSY_LOG_MAX_LINES:]
                            )
                        state.busy_log = "\n".join(lines)
                    except Exception:                 # noqa: BLE001
                        pass
                _fasc_inset_vox_preview = int(
                    getattr(
                        state,
                        "uct_recon_fasc_inset_vox",
                        2,
                    ) or 0,
                )
                meshes = r3d.reconstruct_stack(
                    epi_map, fasc_map,
                    voxel_xy_mm=xy_mm,
                    voxel_z_mm=z_mm,
                    slice_range=(s_lo, s_hi),
                    smooth_sigma=smooth_sigma,
                    smooth_sigma_xy_mm=smooth_sigma_xy_mm_prev,
                    smooth_sigma_z_mm=smooth_sigma_z_mm_prev,
                    refine=do_refine,
                    refine_use_optimesh=do_optimesh,
                    refine_remesh=do_remesh,
                    refine_remesh_edge_len_mm=(
                        remesh_edge_len_mm
                    ),
                    decimate_target_tris=(
                        decimate_target_tris_prev
                    ),
                    decimate_target_fraction=(
                        decimate_target_fraction_prev
                    ),
                    fasc_inset_vox=_fasc_inset_vox_preview,
                    on_progress=_preview_progress,
                )
            return meshes

        try:
            built = await loop.run_in_executor(
                None, _build_meshes_for_preview,
            )
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_recon_status = (
                    f"Preview failed: {ex}"
                )
            return

        # Hand the meshes off to the embedded plotter via the
        # app.py callback (set during `register`). The callback
        # builds actors, refreshes the legend, and computes the
        # quality histogram.
        if on_recon_meshes_ready is not None:
            try:
                on_recon_meshes_ready(built)
            except Exception:                            # noqa: BLE001
                traceback.print_exc()

        with state:
            state.uct_busy = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.uct_recon_status = (
                f"Preview: {len(built)} surface"
                + ("s" if len(built) != 1 else "")
                + " rendered (not saved to disk)."
            )

    # ------------------------------------------------------------------
    # SAM2 video / keyframe-driven propagation. The heavy backend is
    # cached here in `_ctx["video_seg"]`; the per-stack inference state
    # lives inside the backend keyed by stack id (see SAM2VideoSegmenter
    # for the lifecycle). `_ctx["video_stack_id"]` is the id we last
    # prepared on, which lets us notice when the user opens a different
    # stack (or re-uploads the same path) and re-prepare from scratch.
    # ------------------------------------------------------------------
    _ctx["video_seg"] = None
    _ctx["video_stack_id"] = ""
    _ctx["video_keyframe_obj_map"] = {}
    # video_keyframe_obj_map: dict[int slice_idx -> list[(obj_id, label)]]
    # Lets us reproduce the obj-id assignment on re-propagation without
    # re-running IoU matching. Populated by do_propagate_from_keyframes.

    def _ensure_video_seg() -> None:
        """Lazy-build the SAM2VideoSegmenter on first propagate.
        Mirrors `_ensure_segmenter` — the model is only loaded into
        memory when the user actually clicks Propagate, so dialog
        open stays fast even with SAM2 available."""
        if _ctx["video_seg"] is not None:
            return
        from golgi.segmentation.segmenter import SAM2VideoSegmenter
        _ctx["video_seg"] = SAM2VideoSegmenter()

    def _update_keyframe_summary() -> None:
        kfs = sorted(set(int(s) for s in (
            getattr(state, "uct_keyframe_slices", []) or []
        )))
        if not kfs:
            summary = ""
        else:
            # 0-indexed to match the slice scrubber thumb +
            # mask-proposal numbering. The "Slice X / N" busy
            # log is the odd one out (it adds +1); we keep
            # this readout consistent with what the user sees
            # in the scrubber.
            shown = ", ".join(str(s) for s in kfs[:8])
            if len(kfs) > 8:
                shown += f", … (+{len(kfs) - 8} more)"
            summary = f"{len(kfs)} keyframe{'s' if len(kfs) != 1 else ''}: slice {shown}"
        with state:
            state.uct_keyframe_slices = kfs
            state.uct_keyframe_summary = summary

    def do_toggle_keyframe(*_args) -> None:
        """Toggle the current slice in/out of `uct_keyframe_slices`.
        Does NOT trigger propagation — that's a separate button so
        the user can mark several slices and only pay the propagation
        cost once at the end.
        """
        if not bool(getattr(state, "uct_stack_loaded", False)):
            return
        cur = int(_ctx.get("current_slice_idx", 0))
        kfs = set(int(s) for s in (
            getattr(state, "uct_keyframe_slices", []) or []
        ))
        if cur in kfs:
            kfs.discard(cur)
        else:
            kfs.add(cur)
        with state:
            state.uct_keyframe_slices = sorted(kfs)
        _update_keyframe_summary()

    async def do_propagate_from_keyframes(*_args) -> None:
        """Drive SAM2 video propagation from the marked keyframes.

        Pipeline:
          1. Validate prerequisites (≥1 keyframe, ≥1 labelled mask
             on at least one keyframe, stack loaded).
          2. Lazy-load `SAM2VideoSegmenter`.
          3. Dump all cropped slices as JPEGs into the stack's cache
             dir (idempotent — skips files that already exist).
          4. For each keyframe in increasing slice order: gather the
             *labelled* masks (anything except "unlabeled") from the
             per-slice cache. Walk through them and assign each one
             an obj_id by IoU-matching against the most-recent
             propagation output at that slice — falls back to a
             fresh obj_id when no good match is found. Add to SAM2
             via `add_new_mask`. The matching only kicks in for
             keyframes 2..N; the first keyframe seeds the obj_id set
             from scratch.
          5. Drain forward + backward propagation, rebuild per_slice
             cache entries for every non-keyframe slice.
          6. Re-render the current slice + push label counts.
        """
        if bool(getattr(state, "uct_propagation_busy", False)):
            return
        if not bool(getattr(state, "uct_stack_loaded", False)):
            return
        # Flush the in-flight working slice (proposals/labels the
        # user has assigned but not yet committed by scrolling
        # away) into per_slice. Without this, marking the
        # CURRENT slice as a keyframe and clicking Propagate
        # immediately afterwards trips the "no labelled masks"
        # check, because the labels still live in the working
        # _ctx["labels"] list rather than the per_slice cache.
        _commit_current_slice_to_cache()
        kfs = sorted(set(int(s) for s in (
            getattr(state, "uct_keyframe_slices", []) or []
        )))
        if not kfs:
            with state:
                state.uct_status = (
                    "Mark at least one slice as keyframe first."
                )
            return
        # Make sure every keyframe actually has labelled masks.
        # Unlabeled-only keyframes don't anchor anything in SAM2.
        kfs_with_masks = []
        for k in kfs:
            entry = _ctx["per_slice"].get(k)
            if not entry:
                continue
            has_labelled = any(
                lab and lab != "unlabeled"
                for lab in entry.get("labels", [])
            )
            if has_labelled:
                kfs_with_masks.append(k)
        if not kfs_with_masks:
            with state:
                state.uct_status = (
                    "Keyframes need at least one labelled mask. "
                    "Assign endo / epi / etc. on a keyframe slice."
                )
            return

        stack = _ctx.get("stack")
        if stack is None:
            return

        with state:
            state.uct_propagation_busy = True
            state.busy = True
            state.busy_msg = "Propagating from keyframes"
            state.busy_log = "Loading SAM2 video predictor…"
            state.uct_status = "Propagating…"
            # Single-click-cancel mode: SAM2 propagation is
            # restartable + the operation is in-process (no
            # subprocess to kill), so the confirm dialog
            # adds friction without safety. The busy-lightbox
            # Cancel button reads this flag and bypasses the
            # confirm step when set. Reset on every exit path
            # below.
            state.busy_cancel_no_confirm = True
        # Reset the cancel token so a stale "requested" flag
        # from a prior operation (mesh build, FEM solve) can't
        # cause this propagate to bail before it starts.
        if cancel_token is not None:
            try:
                cancel_token.clear()
            except Exception:                                # noqa: BLE001
                pass
        state.flush()
        await asyncio.sleep(0)

        loop = asyncio.get_event_loop()

        # 1. Lazy-load backend.
        try:
            await loop.run_in_executor(None, _ensure_video_seg)
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_status = (
                    f"SAM2 video load failed: {ex}"
                )
            return

        video_seg = _ctx["video_seg"]

        # 2. Build the cropped slice stack at the cropped
        # resolution the user has been editing on, then dump
        # JPEGs into <project>/uct/sam2_cache/<stack_id>/.
        crop = _ctx.get("crop", (0, 0, 0, 0))
        cx0, cy0, cx1, cy1 = (
            int(crop[0]), int(crop[1]),
            int(crop[2]), int(crop[3]),
        )
        n_frames = int(getattr(stack, "n_frames", 0))
        if n_frames <= 0:
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_status = (
                    "Stack has 0 frames; nothing to propagate."
                )
            return
        # stack_id: prefer the project's uct stack metadata if
        # present (mirror the bundle import naming), else fall
        # back to a hash of the upload path so repeated re-uploads
        # of the same file reuse the cache.
        uct_dir = _uct_dir()
        if uct_dir is None:
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_status = (
                    "No active project — open one before "
                    "propagating."
                )
            return
        stack_id = (
            getattr(stack, "stack_id", None)
            or str(getattr(state, "uct_file_path", "") or "default")
        )
        # Sanitize: the stack id becomes a directory name, so
        # collapse anything that isn't alnum / underscore / dash
        # to underscore. Long ids get truncated to 40 chars.
        import re as _re
        stack_id = _re.sub(r"[^A-Za-z0-9_-]+", "_", stack_id)[:40]
        cache_dir = uct_dir / "sam2_cache" / stack_id

        def _build_frames_ndarray():
            # Read every slice + apply the current crop. Cropped
            # uint8 frames are typically ~0.5–2 MB each, so the
            # full stack fits comfortably in RAM (64 × 1 MB = 64 MB
            # for the user's earlier example).
            frames = []
            for fi in range(n_frames):
                arr = read_slice(stack, fi)
                disp = to_display(
            arr,
            window=getattr(
                _ctx.get("stack"), "display_window", None,
            ),
        )
                cropped = disp[cy0:cy1 + 1, cx0:cx1 + 1]
                frames.append(np.ascontiguousarray(cropped))
            return np.stack(frames, axis=0)

        with state:
            state.busy_log = (
                f"Reading {n_frames} slices + dumping JPEGs…"
            )
        state.flush()
        await asyncio.sleep(0)
        try:
            frames_uint8 = await loop.run_in_executor(
                None, _build_frames_ndarray,
            )
            summary = await loop.run_in_executor(
                None,
                lambda: video_seg.prepare_video(
                    stack_id, frames_uint8, cache_dir,
                ),
            )
            _ctx["video_stack_id"] = stack_id
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_status = f"Frame dump failed: {ex}"
            return

        with state:
            state.busy_log = (
                f"Prepared SAM2 state · "
                f"{summary['n_frames']} frames, "
                f"{summary['video_h']}×{summary['video_w']} "
                f"on {summary['device']} "
                f"(wrote {summary['n_jpegs_written']} new jpegs)"
            )
        state.flush()
        await asyncio.sleep(0)

        # 3. Feed keyframe masks into SAM2. First keyframe seeds
        # fresh obj_ids; subsequent keyframes IoU-match against
        # the current propagation output to reuse obj_ids on
        # already-seen fascicles.
        def _gather_kf_masks(slice_idx: int):
            """Returns list[(mask_bool, label_str)] for one
            keyframe, sourced from the per_slice cache. Skips
            unlabeled masks (they don't anchor anything)."""
            entry = _ctx["per_slice"].get(int(slice_idx)) or {}
            out = []
            for prop, lab in zip(
                entry.get("proposals", []),
                entry.get("labels", []),
            ):
                if not lab or lab == "unlabeled":
                    continue
                m = prop.mask
                if m is None or not m.any():
                    continue
                out.append((m.astype(bool), str(lab)))
            return out

        def _add_keyframe_masks_sync():
            """Synchronous keyframe ingest — runs in the executor.
            Builds the obj_id_map as it goes so we can re-render
            after propagation knows which obj is which fascicle."""
            obj_map: dict[int, list[tuple[int, str]]] = {}
            next_obj_id = 1
            # First keyframe: seed obj_ids 1..N from its labelled
            # masks. No matching needed.
            first_kf = kfs_with_masks[0]
            for m, lab in _gather_kf_masks(first_kf):
                video_seg.add_keyframe(
                    stack_id, first_kf, next_obj_id, m,
                )
                obj_map.setdefault(first_kf, []).append(
                    (next_obj_id, lab),
                )
                next_obj_id += 1
            # For keyframes 2..N: we need the running propagation
            # output at that slice to know what obj_id to reuse.
            # Run a forward pass after each keyframe injection so
            # we can match against the latest predictions.
            # Cheap because SAM2 caches per-frame features.
            for kf in kfs_with_masks[1:]:
                # Tap the forward propagator and grab whatever
                # mask it predicts at kf — that's the "what does
                # SAM2 think this slice looks like?" snapshot we
                # match against. We drain the generator fully
                # because SAM2 is stateful and skipping ahead
                # corrupts the rolling memory window.
                snapshot: dict[int, dict[int, np.ndarray]] = {}
                pred = video_seg._predictor
                for fi, oids, logits in pred.propagate_in_video(
                    inference_state=video_seg._states[stack_id],
                    reverse=False,
                ):
                    if int(fi) == int(kf):
                        bool_arr = (logits > 0.0).cpu().numpy()
                        snapshot[int(fi)] = {
                            int(oids[k]): (
                                bool_arr[k, 0].astype(bool)
                            )
                            for k in range(len(oids))
                        }
                        break
                kf_preds = snapshot.get(int(kf), {})
                # IoU match each labelled keyframe mask against
                # the predicted obj_id set. Threshold 0.30 — at
                # this point we KNOW the user marked these as
                # keyframes because something looked wrong, so we
                # err on "create a fresh obj_id" if the match is
                # weak (would rather track a new object than glue
                # an unrelated correction onto the wrong fascicle).
                for m, lab in _gather_kf_masks(kf):
                    best_oid = None
                    best_iou = 0.0
                    for oid, pmask in kf_preds.items():
                        if oid >= 1000:  # reserved
                            continue
                        inter = int(np.logical_and(m, pmask).sum())
                        if inter == 0:
                            continue
                        union = int(np.logical_or(m, pmask).sum())
                        if union == 0:
                            continue
                        iou = inter / union
                        if iou > best_iou:
                            best_iou = iou
                            best_oid = oid
                    if best_oid is not None and best_iou >= 0.30:
                        use_oid = int(best_oid)
                    else:
                        use_oid = next_obj_id
                        next_obj_id += 1
                    video_seg.add_keyframe(
                        stack_id, kf, use_oid, m,
                    )
                    obj_map.setdefault(kf, []).append(
                        (use_oid, lab),
                    )
            return obj_map

        with state:
            state.busy_log = (
                f"Anchoring {len(kfs_with_masks)} keyframe(s) "
                "with labelled masks…"
            )
        state.flush()
        await asyncio.sleep(0)
        try:
            obj_map = await loop.run_in_executor(
                None, _add_keyframe_masks_sync,
            )
            _ctx["video_keyframe_obj_map"] = obj_map
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.uct_status = (
                    f"Keyframe ingest failed: {ex}"
                )
            return

        # 4. Run the full propagation. Forward + backward.
        with state:
            state.busy_log = (
                "Propagating forward + backward through the "
                f"{n_frames}-slice stack…"
            )
        state.flush()
        await asyncio.sleep(0)

        # Live progress + cancellation. The propagator is a
        # generator under the hood; `on_progress` fires per
        # yielded frame so the busy lightbox shows real progress
        # instead of SAM2's tqdm scribble in the terminal; and
        # `should_cancel` polls the active cancel token between
        # frames so the busy lightbox's Cancel button actually
        # stops the work (the in-process SAM2 inference isn't
        # killable any other way).
        from golgi.segmentation.segmenter import (
            PropagationCancelled,
        )
        _last_pct = {"v": -1}

        def _on_prop_progress(frame_idx: int, n_total: int):
            # Throttle to 1%-of-total granularity so we don't
            # spam the WebSocket on a fast machine.
            denom = max(int(n_total), 1)
            pct = int(100.0 * (int(frame_idx) + 1) / denom)
            if pct == _last_pct["v"]:
                return
            _last_pct["v"] = pct
            try:
                with state:
                    state.busy_log = (
                        f"propagate frame "
                        f"{int(frame_idx) + 1}/{denom} "
                        f"({pct}%)"
                    )
                state.flush()
            except Exception:                            # noqa: BLE001
                pass

        # CancelToken is the same instance the build-app-level
        # cancel_busy action wires to the cancel button.
        # `was_requested()` flips True the moment the user
        # clicks Cancel in the busy lightbox.
        def _should_cancel():
            if cancel_token is None:
                return False
            try:
                return bool(cancel_token.was_requested())
            except Exception:                            # noqa: BLE001
                return False

        try:
            propagated = await loop.run_in_executor(
                None,
                lambda: video_seg.propagate(
                    stack_id,
                    on_progress=_on_prop_progress,
                    should_cancel=_should_cancel,
                ),
            )
        except PropagationCancelled:
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.busy_cancel_no_confirm = False
                state.uct_status = (
                    "Propagation cancelled. Per-slice cache "
                    "left in pre-propagation state."
                )
            return
        except Exception as ex:                       # noqa: BLE001
            traceback.print_exc()
            with state:
                state.uct_propagation_busy = False
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
                state.busy_cancel_no_confirm = False
                state.uct_status = f"Propagation failed: {ex}"
            return

        # 5. Rebuild per_slice cache entries for every NON-
        # keyframe slice. Keyframes themselves are left
        # untouched — the user explicitly marked them as the
        # source of truth.
        # Build a flat obj_id → label map by combining all
        # keyframes' obj_map entries. Conflicts (same obj_id,
        # different label across keyframes) should be rare; we
        # take the most recent (last keyframe) as authoritative.
        kf_set = set(int(k) for k in kfs_with_masks)
        obj_label: dict[int, str] = {}
        for kf in kfs_with_masks:
            for oid, lab in (obj_map.get(kf, []) or []):
                obj_label[int(oid)] = lab

        n_rewritten = 0
        for fi in range(n_frames):
            if fi in kf_set:
                continue
            per_obj = propagated.get(int(fi)) or {}
            if not per_obj:
                continue
            new_proposals: list[MaskProposal] = []
            new_labels: list[str] = []
            for oid in sorted(per_obj.keys()):
                m = per_obj[oid].astype(bool)
                if not m.any():
                    continue
                ys, xs = np.where(m)
                bbox = (
                    int(xs.min()), int(ys.min()),
                    int(xs.max()), int(ys.max()),
                )
                new_proposals.append(MaskProposal(
                    mask=m,
                    score=1.0,
                    bbox=bbox,
                    area_px=int(m.sum()),
                    meta={
                        "fresh": True,
                        "source": "sam2_video",
                        "obj_id": int(oid),
                    },
                ))
                new_labels.append(
                    obj_label.get(int(oid), "unlabeled"),
                )
            # Apply the same 2D cleanup as the per-slice
            # Segment path, so a sweep + propagate pair produce
            # consistent overlays (sub-component-px noise out,
            # holes filled, gaps closed). Labels stay paired
            # with surviving proposals; colour generation is
            # deferred until after cleanup so dropped proposals
            # don't waste slots in the chip legend.
            new_proposals, new_labels, _ = (
                _apply_2d_cleanup_to_proposals(
                    new_proposals, new_labels, None,
                )
            )
            colors = generate_proposal_colors(
                len(new_proposals),
            )
            _ctx["per_slice"][fi] = {
                "proposals": new_proposals,
                "labels": new_labels,
                "colors": list(colors),
            }
            n_rewritten += 1

        # 6. Refresh current slice overlay + label counts.
        _restore_slice_from_cache(
            int(_ctx.get("current_slice_idx", 0)),
        )
        _refresh_label_counts()
        _invalidate_finalize()

        with state:
            state.uct_propagation_busy = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.busy_cancel_no_confirm = False
            state.uct_status = (
                f"Propagated {n_rewritten} slice(s) from "
                f"{len(kfs_with_masks)} keyframe(s). "
                "Scroll to review; mark more keyframes + re-"
                "propagate to refine."
            )
        state.flush()
        _rerender_overlay()

    return {
        "do_open_segment_uct_dialog": do_open_segment_uct_dialog,
        "do_close_segment_uct_dialog": do_close_segment_uct_dialog,
        "do_load_uct_stack": do_load_uct_stack,
        "do_clear_uct_stack": do_clear_uct_stack,
        "do_run_uct_segmentation": do_run_uct_segmentation,
        "do_label_uct_proposal": do_label_uct_proposal,
        "do_generate_epi": do_generate_epi,
        "do_refine_masks": do_refine_masks,
        "do_save_uct_segmentation": do_save_uct_segmentation,
        "do_finalize_segmentation": do_finalize_segmentation,
        "do_toggle_keyframe": do_toggle_keyframe,
        "do_propagate_from_keyframes": (
            do_propagate_from_keyframes
        ),
        # V1 Phase B — Step-2 3D reconstruction.
        "do_recon_next": do_recon_next,
        "do_recon_back": do_recon_back,
        "do_run_reconstruction": do_run_reconstruction,
        "do_run_reconstruction_preview": (
            do_run_reconstruction_preview
        ),
        # Watcher callbacks — build_app registers them via
        # @state.change(...).
        "_on_uct_slice_change": _on_slice_change,
        "_on_uct_crop_change": _on_crop_change,
        "_on_uct_backend_change": _on_backend_choice_change,
        "_on_uct_clahe_change": _on_clahe_toggle,
        "_on_uct_click_payload": _on_click_payload,
        "_on_uct_zoom_change": _on_zoom_change,
        "_on_uct_paint_payload": _on_paint_payload,
        "_on_uct_recon_range_change": _on_recon_range_change,
    }


def _apply_clahe(image_uint8: "np.ndarray") -> "np.ndarray":
    """Contrast Limited Adaptive Histogram Equalization on a
    grayscale uint8 image. Returns a fresh array of the same
    shape — caller decides whether to keep the original raw
    slice or the enhanced view.

    Tile / clip-limit defaults are the OpenCV-standard 8×8 / 2.0
    which is the value cited in the medical-imaging literature
    (see e.g. Pisano 1998, Zuiderveld 1994). For µCT they
    typically lift fascicle / perineurium contrast 2-3× without
    introducing the haloing you see at higher clip limits."""
    try:
        import cv2
        clahe = cv2.createCLAHE(
            clipLimit=2.0, tileGridSize=(8, 8),
        )
        return clahe.apply(image_uint8)
    except ImportError:
        # OpenCV is in the env per the deps probe but the
        # silent-fallback path lets headless smoke tests pass
        # on a stripped install. skimage equivalent below.
        from skimage.exposure import equalize_adapthist
        out = equalize_adapthist(
            image_uint8, clip_limit=0.01,
        )
        return (out * 255.0).astype(image_uint8.dtype)
