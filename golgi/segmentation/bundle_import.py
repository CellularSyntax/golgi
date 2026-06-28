# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Histology bundle import — TIFF-quartet → extruded nerve.

The "bundle" is a set of 4 TIFFs for one histology / µCT slice:
  * the slide image itself (RGB or grayscale)
  * an epi (whole-nerve) bool mask
  * a fascicle bool mask
  * a scale-bar bool mask

Filename role detection is by suffix (`*NerveMask`, `*FascMask`,
`*ScaleMask`); the residual file is the slide image. Pixel pitch
is calibrated from the scale mask's longest dimension given a
user-supplied physical scale-bar length.

The output is a list of `Mesh` objects (epi + per-fascicle prism)
identical in shape to what the existing single-slice µCT
reconstruct path produces via `extrude_single_slice`. Downstream
(STL save, PLC build, TetGen) is unchanged.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

from golgi.segmentation.reconstruct3d import (
    Mesh, extrude_single_slice,
)


# --------------------------------------------------------------
# Role detection
# --------------------------------------------------------------

# Suffix patterns identifying each role. Case-insensitive. The
# residual file (no suffix match) is treated as the slide image.
# Order matters: the human peri-inner/ext patterns must be tried BEFORE the
# bare "peri mask" so "PeriInnerMask" doesn't match the perineurium role.
_ROLE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("peri_inner", re.compile(r"peri\s*inner\s*mask", re.IGNORECASE)),
    ("peri_ext", re.compile(r"peri\s*ext\s*mask", re.IGNORECASE)),
    ("peri", re.compile(r"peri\s*mask", re.IGNORECASE)),
    ("nerve_mask", re.compile(r"nerve\s*mask", re.IGNORECASE)),
    ("fasc_mask", re.compile(r"fasc(icle)?\s*mask", re.IGNORECASE)),
    ("scale_mask", re.compile(r"scale\s*mask", re.IGNORECASE)),
)


@dataclass
class BundleRoles:
    """One bundle's files keyed by role. Absolute paths.

    Swine packages: slide + NerveMask + FascMask + ScaleMask. Human packages
    carry explicit perineurium masks — PeriInnerMask (endoneurium, used as the
    fascicle mask), PeriExtMask (fascicle+sheath), PeriMask (the annulus) —
    plus a ScaleLength.mat. `human` flags which convention was detected.
    """
    slide: Path | None
    nerve_mask: Path
    fasc_mask: Path                  # FascMask (swine) or PeriInnerMask (human)
    scale_mask: Path
    peri_inner: Path | None = None
    peri_ext: Path | None = None
    peri: Path | None = None
    scale_mat: Path | None = None
    human: bool = False

    @classmethod
    def from_files(cls, files: list[Path]) -> "BundleRoles":
        """Group a list of paths into a bundle by filename-suffix matching.

        Tolerant of extra files (human packages add Claudin stains, an
        AllTissue mask, a .nd2, a .mat): the residual image becomes the
        slide, AllTissue is ignored, the .mat is captured as scale_mat.
        Requires NerveMask + ScaleMask + (FascMask or PeriInnerMask).
        """
        roles: dict[str, Path] = {}
        residual: list[Path] = []
        scale_mat: Path | None = None
        for f in sorted(files):
            stem_l = f.stem.lower()
            if f.suffix.lower() == ".mat" and "scalelength" in stem_l:
                scale_mat = f
                continue
            if "alltissue" in stem_l:
                continue              # a coarser nerve mask — not used
            matched = None
            for role, pat in _ROLE_PATTERNS:
                if pat.search(f.stem):
                    matched = role
                    break
            if matched is None:
                if f.suffix.lower() in (".tif", ".tiff", ".png",
                                        ".jpg", ".jpeg"):
                    residual.append(f)
            elif matched not in roles:   # first match wins; ignore dups
                roles[matched] = f
        human = "peri_inner" in roles
        fasc = roles.get("peri_inner") if human else roles.get("fasc_mask")
        missing = [r for r, v in (("nerve_mask", roles.get("nerve_mask")),
                                  ("fasc/peri_inner", fasc),
                                  ("scale_mask", roles.get("scale_mask")))
                   if v is None]
        if missing:
            raise ValueError(
                f"bundle missing required role(s): {missing}; "
                f"got {[p.name for p in files]}")
        return cls(
            slide=(residual[0] if residual else None),
            nerve_mask=roles["nerve_mask"],
            fasc_mask=fasc,
            scale_mask=roles["scale_mask"],
            peri_inner=roles.get("peri_inner"),
            peri_ext=roles.get("peri_ext"),
            peri=roles.get("peri"),
            scale_mat=scale_mat,
            human=human,
        )


# --------------------------------------------------------------
# Image loading
# --------------------------------------------------------------

def _load_bool_mask(path: Path) -> np.ndarray:
    """Load a TIFF as a 2D bool mask. Accepts mode='1', uint8,
    or float — non-zero pixels become True."""
    from PIL import Image
    img = Image.open(str(path))
    arr = np.asarray(img)
    if arr.ndim == 3:
        # RGB mask: foreground iff any channel != 0.
        arr = arr.any(axis=2)
    return arr.astype(bool)


def _load_slide_rgb(path: Path) -> np.ndarray:
    """Load the slide image as (H, W, 3) uint8. Grayscale gets
    promoted by broadcasting."""
    from PIL import Image
    img = Image.open(str(path)).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


# --------------------------------------------------------------
# Scale calibration
# --------------------------------------------------------------

@dataclass
class ScaleCalibration:
    pixel_pitch_um: float
    scale_bar_length_um: float
    measured_bar_length_px: int
    aspect_ratio: float


def calibrate_pixel_pitch(
    scale_mask: np.ndarray,
    scale_bar_length_um: float,
    *,
    min_aspect_ratio: float = 3.0,
) -> ScaleCalibration:
    """Derive pixel pitch (µm/px) from a scale-bar mask.

    The scale mask must contain exactly one connected component
    that's substantially elongated (aspect ratio ≥ 3:1 by
    default). The bar's physical length is supplied as
    `scale_bar_length_um`; pixel pitch is computed as
    `scale_bar_length_um / bar_length_px` where `bar_length_px`
    is the longer of the bounding-box width and height.

    Bounding-box length is preferred over PCA 4σ because anti-
    aliased bar ends inflate the PCA spread; bbox is robust as
    long as the bar is axis-aligned (the common case).
    """
    if scale_mask.ndim != 2:
        raise ValueError(
            f"scale_mask must be 2D, got shape {scale_mask.shape}"
        )
    if scale_bar_length_um <= 0:
        raise ValueError(
            f"scale_bar_length_um must be > 0 (got "
            f"{scale_bar_length_um})"
        )
    from scipy.ndimage import label
    lab, n_cc = label(scale_mask.astype(bool))
    if n_cc == 0:
        raise ValueError("scale mask is empty")
    # Keep the largest connected component (in case there are
    # micro-specks around the bar).
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0
    biggest = int(np.argmax(sizes))
    ys, xs = np.where(lab == biggest)
    bbox_h = int(ys.max() - ys.min() + 1)
    bbox_w = int(xs.max() - xs.min() + 1)
    bar_len_px = max(bbox_h, bbox_w)
    bar_thick_px = min(bbox_h, bbox_w)
    aspect = bar_len_px / max(bar_thick_px, 1)
    if aspect < min_aspect_ratio:
        raise ValueError(
            f"scale-bar candidate has aspect ratio "
            f"{aspect:.1f} < {min_aspect_ratio} — doesn't look "
            f"like a bar (bbox {bbox_h}×{bbox_w} px). Check "
            f"that the ScaleMask file is correct."
        )
    pitch_um = float(scale_bar_length_um) / float(bar_len_px)
    return ScaleCalibration(
        pixel_pitch_um=pitch_um,
        scale_bar_length_um=float(scale_bar_length_um),
        measured_bar_length_px=bar_len_px,
        aspect_ratio=aspect,
    )


# --------------------------------------------------------------
# Bundle → meshes
# --------------------------------------------------------------

@dataclass
class BundleData:
    slide_rgb: np.ndarray            # (H, W, 3) uint8
    nerve_mask: np.ndarray           # (H, W) bool
    fasc_mask: np.ndarray            # (H, W) bool (FascMask | PeriInnerMask)
    scale_mask: np.ndarray           # (H, W) bool
    calibration: ScaleCalibration
    bundle_stem: str                 # e.g. "sub-10sam-1P10-1"
    human: bool = False
    peri_thk_um: float | None = None  # MEASURED perineurium (human) else None


def measured_perineurium_um(peri_inner, peri_ext, pitch_mm) -> float | None:
    """Area-weighted mean perineurium thickness (µm) from the human
    PeriInner (endo) and PeriExt (endo+sheath) masks: per endo component,
    thk = (d_ext − d_in)/2 of equivalent-circle diameters."""
    from scipy import ndimage
    px2 = (float(pitch_mm) * 1000.0) ** 2
    li, ni = ndimage.label(peri_inner)
    le, _ = ndimage.label(peri_ext)
    areas, thk = [], []
    for k in range(1, ni + 1):
        comp = li == k
        a_in = float(comp.sum()) * px2
        if a_in <= 0:
            continue
        d_in = 2.0 * np.sqrt(a_in / np.pi)
        lab = le[comp]
        lab = lab[lab > 0]
        if lab.size == 0:
            continue
        a_ext = float((le == np.bincount(lab).argmax()).sum()) * px2
        d_ext = 2.0 * np.sqrt(a_ext / np.pi)
        areas.append(a_in)
        thk.append(max(0.0, 0.5 * (d_ext - d_in)))
    if not areas:
        return None
    A = np.asarray(areas)
    return float((A * np.asarray(thk)).sum() / A.sum())


def _scale_bar_um_from_mat(mat_path: Path, default: float) -> float:
    if mat_path is None:
        return default
    try:
        import scipy.io as sio
        return float(np.asarray(sio.loadmat(mat_path)["slength"]).ravel()[0])
    except Exception:                                      # noqa: BLE001
        return default


def _bundle_stem(roles: BundleRoles) -> str:
    """Longest common prefix of the file stems — used for output naming."""
    stems = [p.stem for p in (roles.slide, roles.nerve_mask,
                              roles.fasc_mask, roles.scale_mask)
             if p is not None]
    if not stems:
        return "bundle"
    common = stems[0]
    for s in stems[1:]:
        i = 0
        n = min(len(common), len(s))
        while i < n and common[i] == s[i]:
            i += 1
        common = common[:i]
    return common.rstrip("_-.") or stems[0]


def load_bundle(
    roles: BundleRoles,
    *,
    scale_bar_length_um: float,
) -> BundleData:
    """Load + validate a 4-file bundle, returning the masks and
    derived pixel pitch.

    Verifies that all four images share the same (H, W) shape —
    a 1-pixel mismatch on any of the masks would propagate into
    the extrude and break downstream code, so the check is
    strict.
    """
    nerve_mask = _load_bool_mask(roles.nerve_mask)
    fasc_mask = _load_bool_mask(roles.fasc_mask)
    scale_mask = _load_bool_mask(roles.scale_mask)
    H, W = nerve_mask.shape[:2]
    for name, arr in (("fasc_mask", fasc_mask), ("scale_mask", scale_mask)):
        if arr.shape != (H, W):
            raise ValueError(
                f"{name} shape {arr.shape} doesn't match nerve_mask "
                f"({H}×{W})")
    # slide is optional (human packages may have only a Claudin stain or
    # none of matching size) — fall back to a blank canvas.
    slide_rgb = np.zeros((H, W, 3), np.uint8)
    if roles.slide is not None:
        try:
            s = _load_slide_rgb(roles.slide)
            if s.shape[:2] == (H, W):
                slide_rgb = s
        except Exception:                                  # noqa: BLE001
            pass
    # human ScaleLength.mat overrides the supplied bar length
    bar_um = _scale_bar_um_from_mat(roles.scale_mat, scale_bar_length_um)
    calib = calibrate_pixel_pitch(scale_mask, scale_bar_length_um=bar_um)
    peri_thk_um = None
    if roles.human and roles.peri_inner is not None \
            and roles.peri_ext is not None:
        peri_thk_um = measured_perineurium_um(
            _load_bool_mask(roles.peri_inner),
            _load_bool_mask(roles.peri_ext),
            calib.pixel_pitch_um / 1000.0)
    return BundleData(
        slide_rgb=slide_rgb,
        nerve_mask=nerve_mask,
        fasc_mask=fasc_mask,
        scale_mask=scale_mask,
        calibration=calib,
        bundle_stem=_bundle_stem(roles),
        human=roles.human,
        peri_thk_um=peri_thk_um,
    )


def extrude_bundle(
    bundle: BundleData,
    *,
    thickness_mm: float,
    fasc_inset_vox: int = 2,
    clean_min_component_px: int = 50,
    clean_min_hole_px: int = 50,
    clean_closing_radius_px: int = 2,
) -> list[Mesh]:
    """Build the watertight prism meshes for the bundle by
    handing the supplied masks straight to the M45 polygon-
    extrude path. No SAM2, no per-slice cache, no segmenter
    state — just masks in, meshes out.

    The `nerve_mask` is treated as the epi (solid prism); the
    `fasc_mask` is treated as fascicles (per-component prisms
    inset inside the epi by `fasc_inset_vox` voxels).

    M48 — Mask cleanup defaults match the µCT segmentation
    flow's state-default values: drop foreground components
    < 50 px (kills 1-7 px SAM2 / segmenter speckles), fill
    background holes < 50 px (closes small false-negative
    holes inside fascicles), morphological close with a
    2-px disk (seals 1-2 px gaps in the foreground boundary
    from segmenter jitter). Without these the polygon-extrude
    inherits every staircase artefact the segmentation pass
    left behind, which compounds with axial-slab aspect ratio
    issues to give very low q_radius_ratio.
    """
    voxel_xy_mm = bundle.calibration.pixel_pitch_um / 1000.0
    # human PeriInnerMask is already the endoneurium boundary (perineurium
    # removed) → no inset; swine FascMask is the fascicle outline → inset.
    inset = 0 if bundle.human else int(fasc_inset_vox)
    return extrude_single_slice(
        epi_mask=bundle.nerve_mask,
        fasc_mask=bundle.fasc_mask,
        voxel_xy_mm=voxel_xy_mm,
        thickness_mm=float(thickness_mm),
        fasc_inset_vox=inset,
        clean_min_component_px=int(clean_min_component_px),
        clean_min_hole_px=int(clean_min_hole_px),
        clean_closing_radius_px=int(clean_closing_radius_px),
    )


__all__ = [
    "BundleRoles",
    "BundleData",
    "ScaleCalibration",
    "calibrate_pixel_pitch",
    "measured_perineurium_um",
    "load_bundle",
    "extrude_bundle",
]
