# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""V1 — µCT slice segmentation.

Lives behind the Import-wizard "µCT segment" nerve-source tile.
Loads multi-page TIFF stacks, prompts MedSAM2 (optional dep) or
falls back to Otsu+morphology, produces labeled masks → polygons
that the existing `pipeline/plc.py` extrudes into a prismatic
nerve mesh.

Phase A — image loader + segmentation backends + dialog UI.
Phase B — polygon extraction + PLC layer build.
Phase C — classical Otsu fallback for headless runs.
Phase D — Import-wizard integration.

This package is intentionally minimal-surface — only the
loader/segmentation primitives live here; persistence + the
Trame dialog live one level up in `golgi/ui/dialogs/` and
`golgi/projects/` respectively.
"""
from __future__ import annotations

from .image import (
    Stack,
    StackError,
    load_stack,
    read_slice,
    to_display,
)
from .segmenter import (
    MaskProposal,
    MedSAM2Segmenter,
    Segmenter,
    StubSegmenter,
    resolve_segmenter,
)
from .render import (
    LABEL_COLORS,
    color_to_hex,
    compose_overlay,
    generate_proposal_colors,
    to_data_url,
)

__all__ = [
    "Stack",
    "StackError",
    "load_stack",
    "read_slice",
    "to_display",
    "MaskProposal",
    "Segmenter",
    "StubSegmenter",
    "MedSAM2Segmenter",
    "resolve_segmenter",
    "LABEL_COLORS",
    "color_to_hex",
    "compose_overlay",
    "generate_proposal_colors",
    "to_data_url",
]
