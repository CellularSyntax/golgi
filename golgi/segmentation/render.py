# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Server-side mask overlay rendering.

The Segment-µCT dialog uses a "click chips to relabel" UX rather
than a true canvas-with-SVG-overlay (much simpler to build, no
custom JS, works on any browser). The flow is:

  1. dialog gets a list of MaskProposal from the segmenter
  2. user assigns each proposal a label in
     {background, epi, fascicle, discard} via VChips
  3. on every relabel, the server re-renders the slice with the
     current label assignments → PNG bytes → base64 data URL →
     pushed back to the browser as the image src

`compose_overlay()` does step 3 in a single NumPy pass. Fast
enough (~50 ms for a 1024-wide preview with ~20 proposals) that
the round-trip feels interactive.
"""
from __future__ import annotations

import base64
import colorsys
import io
from typing import Optional, Sequence

import numpy as np

from .segmenter import MaskProposal


# Label → RGBA tint. "discard" matches the background so a user
# can quickly demote a wrongly-proposed blob without it being
# distracting on the preview.
LABEL_COLORS: dict[str, tuple[int, int, int, int]] = {
    # Background = "non-nerve tissue" (sample mount / chamber
    # arc / voids). Red so the user can SEE what they marked
    # — a transparent value would make labelled background
    # blobs disappear, leaving the user unsure whether a click
    # landed.
    "background": (226, 75,  74,  140),   # golgi red @ ~55 % alpha
    # Epineurium is DERIVED (not click-cycled) — added by
    # do_generate_epi as a single mask covering everything in
    # the slice that wasn't labelled fascicle or background.
    # Green visually separates it from the user-set classes.
    "epi":        (76,  175, 80,  150),   # green @ ~59 % alpha
    "fascicle":   (96,  165, 250, 160),   # blue @ ~63 % alpha
    # "discard" is the legacy label kept for backwards
    # compatibility with old projects. New workflow uses
    # "background" instead — see _CLICK_CYCLE in
    # actions/segment_uct.py.
    "discard":    (130, 130, 130, 60),    # grey @ ~24 %
    "unlabeled":  (250, 220, 0,   100),   # amber prompt colour
}


def generate_proposal_colors(
    n: int, *, alpha: int = 150,
) -> list[tuple[int, int, int, int]]:
    """Build `n` visually-distinct RGBA tints — one per proposal
    so the dialog can show a colour swatch on each chip and the
    user can match it to the overlay tint.

    Uses the golden-ratio HSV trick (Knuth) — successive hues
    spaced by 360°/φ ≈ 137.5° look pseudo-random + maximally
    different to the eye. Saturation/value held high so the
    swatches read clearly against both the dark µCT background
    and the dialog's white chip rows.
    """
    out: list[tuple[int, int, int, int]] = []
    for i in range(n):
        # 0.618 ≈ 1/φ — golden-ratio offset in hue space.
        hue = (i * 0.61803398875) % 1.0
        r, g, b = colorsys.hsv_to_rgb(hue, 0.7, 0.95)
        out.append((
            int(r * 255), int(g * 255), int(b * 255),
            int(alpha),
        ))
    return out


def color_to_hex(col: tuple[int, int, int, int]) -> str:
    """RGBA tuple → #rrggbb hex string for CSS / chip swatches.
    Alpha is intentionally dropped since the swatch is rendered
    at full opacity for clarity."""
    return f"#{col[0]:02x}{col[1]:02x}{col[2]:02x}"


def compose_overlay(
    image_uint8: np.ndarray,
    proposals: Sequence[MaskProposal],
    labels: Sequence[str],
    *,
    proposal_colors: Optional[
        Sequence[tuple[int, int, int, int]]
    ] = None,
    max_width: Optional[int] = 1024,
) -> bytes:
    """Render the slice + per-proposal coloured overlays into a
    PNG byte string.

    `labels` must have the same length as `proposals` — each entry
    is one of `LABEL_COLORS`'s keys ('unlabeled' for proposals
    not yet classified). Unknown labels fall back to 'unlabeled'.

    `proposal_colors`: when supplied, **unlabeled** proposals
    use their per-index tint from this sequence instead of the
    single shared 'unlabeled' amber. Lets the dialog map each
    chip to its own coloured blob on the overlay. Already-
    labelled proposals (epi / fascicle / discard) keep their
    LABEL_COLORS tint so the canonical class colours stay
    stable across renders.

    `max_width`: optionally downsample for the browser preview;
    None → native resolution. Aspect ratio is preserved.
    """
    if len(proposals) != len(labels):
        raise ValueError(
            f"proposals/labels length mismatch: "
            f"{len(proposals)} vs {len(labels)}",
        )
    h, w = image_uint8.shape[:2]
    # Promote to RGBA so alpha-blending the tints is one expression
    # per proposal — saves a copy + a clip per pass.
    rgb = np.stack(
        [image_uint8, image_uint8, image_uint8], axis=-1,
    ).astype(np.float32)

    # Render-priority order — lowest first, highest last.
    # Fascicle (the user-affirmative target class) renders on
    # top of everything else, so even if the epi mask were
    # transiently stale (e.g. mid-update), the user can't end
    # up looking at green over a region they've marked
    # fascicle. Same logic for background → user intent wins
    # over auto-derived epi.
    _PRIORITY = {
        "unlabeled":  0,
        "epi":        1,
        "background": 2,
        "fascicle":   3,
    }
    order = sorted(
        range(len(proposals)),
        key=lambda i: _PRIORITY.get(labels[i], 0),
    )
    for i in order:
        prop = proposals[i]
        lab = labels[i]
        # An "unlabeled" proposal can be in one of two states:
        #   - `meta.fresh=True` (the segmenter-default): the user
        #     hasn't touched it yet. Render with its per-proposal
        #     hue so it's identifiable on the overlay.
        #   - `meta.fresh=False`: the user explicitly clicked
        #     "None" to dismiss it. Skip rendering so the user's
        #     intent ("get this off my overlay") is honoured
        #     immediately without needing the auto-epi to come
        #     in and cover it.
        if lab == "unlabeled":
            is_fresh = True
            try:
                is_fresh = bool(prop.meta.get("fresh", True))
            except AttributeError:
                pass
            if not is_fresh:
                continue
            if proposal_colors:
                col = proposal_colors[i % len(proposal_colors)]
            else:
                col = LABEL_COLORS["unlabeled"]
        else:
            col = LABEL_COLORS.get(lab, LABEL_COLORS["unlabeled"])
        alpha = col[3] / 255.0
        tint = np.array(col[:3], dtype=np.float32)
        # Boolean indexing is the fastest tint-in-place pass here;
        # cv2.addWeighted would force a full-image blend.
        m = prop.mask
        rgb[m] = rgb[m] * (1.0 - alpha) + tint * alpha

    rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    # Optional downsample for the preview path.
    if max_width is not None and w > max_width:
        from PIL import Image
        im = Image.fromarray(rgb)
        new_w = int(max_width)
        new_h = max(1, int(h * (max_width / w)))
        im = im.resize((new_w, new_h), Image.BILINEAR)
    else:
        from PIL import Image
        im = Image.fromarray(rgb)

    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=False)
    return buf.getvalue()


def to_data_url(png_bytes: bytes) -> str:
    """Wrap PNG bytes in a base64 `data:` URL the browser can use
    directly as an `<img src>`. Avoids a round-trip through a
    static-file endpoint."""
    b64 = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{b64}"
