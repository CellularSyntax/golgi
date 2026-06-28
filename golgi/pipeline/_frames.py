# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff-frame migration helper, shared between fem.py + fibers.py.

solve_nerve.py (FEM solve) reads fibers in CUFF frame.
do_generate_fibers writes them in RAW frame (pts_cuff = pts_raw so
the Procrustes inside solve_fiber_paths_nerve.py is the identity).
This helper transforms the on-disk nerve_paths_fibers.npz +
in-memory geom.fiber_paths_raw raw → cuff in-place, stamping a
`frame_is_cuff` flag so re-runs are idempotent.

Was a closure inside build_app; extracted in step 4.4 so the FEM
pipeline driver (which still calls it as a pre-FEM step) can
import it cleanly from a stable module path.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import numpy as np


def ensure_fibers_in_cuff_frame(
    *,
    geom,
    out_dir: Path,
    transform_to_cuff_frame_fn: Callable,
    say: "Callable[[str], None] | None" = None,
) -> bool:
    """Ensure nerve_paths_fibers.npz (and the in-memory
    geom.fiber_paths_raw mirror) are stored in cuff frame
    before solve_nerve.py reads them. Transforms raw → cuff
    in-place, rewrites the npz with a `frame_is_cuff` flag,
    and updates geom.fibers_in_cuff_frame. Idempotent: if
    the flag is already set on disk or in memory, no-op.
    Returns True if a transform actually ran."""
    _say = say if say is not None else (lambda *_: None)
    if geom.fibers_in_cuff_frame:
        return False
    npz_path = out_dir / "nerve_paths_fibers.npz"
    if not npz_path.exists() or geom.fiber_paths_raw is None:
        return False
    if (geom.centroid is None or geom.R_global is None
            or geom.cuff_origin_pca is None
            or geom.R_local is None):
        _say(
            "  ⚠ cuff transform not available — paths stay "
            "in raw frame; FEM Ve along fibers will be wrong"
        )
        return False
    try:
        existing = np.load(npz_path, allow_pickle=True)
        if ("frame_is_cuff" in existing.files
                and int(existing["frame_is_cuff"]) == 1):
            # Already cuff frame on disk — just sync the
            # in-memory flag without touching the file.
            geom.fibers_in_cuff_frame = True
            return False
        _say(
            "  transforming fibers raw → cuff frame for "
            "FEM Vₑ sampling (and rewriting "
            "nerve_paths_fibers.npz)"
        )
        # Per-fiber transform → flat layout + lengths.
        transformed = [
            transform_to_cuff_frame_fn(
                np.asarray(p, dtype=np.float64),
                geom.centroid, geom.R_global,
                geom.cuff_origin_pca, geom.R_local,
            )
            for p in geom.fiber_paths_raw
        ]
        flat = np.vstack(transformed)
        lens = np.array(
            [len(p) for p in transformed], dtype=np.int64,
        )
        # Preserve every other field that was in the npz
        # (step_m, seed_end, sign, …) so downstream consumers
        # don't lose metadata. We can't just `dict(existing)`
        # because NpzFile values are lazy — read each one.
        preserved = {}
        for _k in existing.files:
            if _k in ("paths_flat", "path_lengths"):
                continue
            try:
                preserved[_k] = existing[_k]
            except Exception:
                pass
        np.savez(
            npz_path,
            paths_flat=flat,
            path_lengths=lens,
            frame_is_cuff=np.int8(1),
            **preserved,
        )
        geom.fiber_paths_raw = transformed
        geom.fibers_in_cuff_frame = True
        _say(
            f"  ✓ {len(transformed)} fibers now in cuff frame"
        )
        # Also migrate nerve_paths_caps.json — both cap
        # centroids ARE in raw frame, and the branch
        # classifier compares cuff-frame fiber endpoints
        # against them. Mixing frames silently routes ~all
        # fibers to whichever centroid is closer to the
        # origin in cuff frame, which is the "everything
        # lumped into one branch" bug the user kept seeing
        # in the §9/§10 ribbon plots.
        caps_path = out_dir / "nerve_paths_caps.json"
        if caps_path.exists():
            try:
                caps = json.loads(caps_path.read_text())
                if not bool(caps.get("frame_is_cuff", False)):
                    def _xform_pt(v):
                        arr = np.asarray(
                            v, dtype=np.float64,
                        ).reshape(1, 3)
                        out = transform_to_cuff_frame_fn(
                            arr, geom.centroid,
                            geom.R_global,
                            geom.cuff_origin_pca,
                            geom.R_local,
                        )[0]
                        return out.tolist()
                    if "trunk_cap_centroid_m" in caps:
                        caps["trunk_cap_centroid_m"] = (
                            _xform_pt(
                                caps["trunk_cap_centroid_m"],
                            )
                        )
                    if "branch_cap_centroids_m" in caps:
                        caps["branch_cap_centroids_m"] = [
                            _xform_pt(c)
                            for c in caps[
                                "branch_cap_centroids_m"
                            ]
                        ]
                    caps["frame_is_cuff"] = True
                    caps_path.write_text(
                        json.dumps(caps, indent=2),
                    )
                    _say(
                        f"  ✓ migrated caps_json centroids "
                        f"raw → cuff "
                        f"({len(caps.get('branch_cap_centroids_m', []))} "
                        f"branch centroids)"
                    )
            except Exception as ex:
                _say(
                    f"  ⚠ caps_json migration failed: {ex}"
                )
        return True
    except Exception as ex:
        _say(f"  ⚠ raw→cuff transform failed: {ex}")
        return False
