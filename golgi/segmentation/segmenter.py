# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Segmentation backends — MedSAM2 (preferred) + stub fallback.

The dialog calls into a `Segmenter` instance via two operations:

    segmenter.propose_all(image_uint8)
        → list[MaskProposal] of candidate masks ("everything"
          mode). The dialog renders these as overlays and lets
          the user click-to-keep + assign labels.

    segmenter.segment_at(image_uint8, points, point_labels, box)
        → MaskProposal — refine a single mask from positive/
          negative point clicks and/or a bounding box.

Two concrete backends:

  • `MedSAM2Segmenter` — wraps Bo Wang's MedSAM2 (SAM2 fine-
    tuned on 1.5 M medical image/mask pairs). Optional dep;
    requires the `sam2` Python package + a downloaded
    checkpoint (~1 GB). Auto-detects checkpoint via the
    `GOLGI_MEDSAM2_CHECKPOINT` env var or `~/.cache/golgi/
    medsam2/medsam2.pt`.

  • `StubSegmenter` — produces geometric placeholders (a
    bounding-box outer mask + a few inner blobs) so the dialog
    can be built and tested without MedSAM2 installed. Pure
    NumPy; works headless.

`resolve_segmenter()` picks the best available backend and is
the entry point the dialog uses. Falling back to stub triggers
a one-line warning the dialog surfaces to the user.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Protocol

import numpy as np


@dataclass
class MaskProposal:
    """A candidate segmentation mask + metadata.

    `mask` is a boolean HxW array (True = foreground).
    `score` is the backend's confidence in [0, 1] (1.0 for
    deterministic backends like the stub).
    `bbox` is the tight axis-aligned bounding box in pixel
    coords as (x_min, y_min, x_max, y_max), inclusive bounds.
    `area_px` is the pixel count of the mask.
    `meta` carries backend-specific extras (e.g. SAM logit
    map) for refinement passes.
    """
    mask: np.ndarray
    score: float
    bbox: tuple[int, int, int, int]
    area_px: int
    meta: dict = field(default_factory=dict)


class Segmenter(Protocol):
    """Backend-agnostic segmentation API the dialog binds to."""

    name: str
    """Short human-readable backend label, e.g. 'MedSAM2',
    'SAM2', 'stub'."""

    def propose_all(
        self, image_uint8: np.ndarray,
    ) -> list[MaskProposal]:
        """Generate candidate masks for the whole slice
        ('everything' mode). The dialog presents these as
        overlays the user can keep/discard + label."""

    def segment_at(
        self,
        image_uint8: np.ndarray,
        *,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> MaskProposal:
        """Refine a single mask from a prompt. `points` is
        (N, 2) in (x, y) pixel coords; `point_labels` is (N,)
        with 1 = positive / 0 = negative. `box` is
        (x_min, y_min, x_max, y_max) inclusive. At least one
        of points or box must be provided."""


# --------------------------------------------------------------
# Stub backend — always available; geometric placeholders so the
# dialog + persistence + polygon pipeline can be built + tested
# without GPU / MedSAM2 weights.
# --------------------------------------------------------------

class StubSegmenter:
    """Deterministic placeholder segmenter — returns Otsu+
    morphology mask for `propose_all` and a disk centred on the
    last positive point for `segment_at`. Useful for headless
    smoke tests and for letting users explore the dialog UX
    before installing the real backend."""

    name = "stub"

    def __init__(self, *, max_proposals: int = 20) -> None:
        self.max_proposals = int(max_proposals)

    def propose_all(
        self, image_uint8: np.ndarray,
    ) -> list[MaskProposal]:
        """Use Otsu threshold + connected components as a
        crude approximation of 'segment everything'. Drops
        components smaller than 0.1 % of the image area to
        filter out speckle."""
        if image_uint8.ndim == 3:
            image_uint8 = image_uint8[..., 0]
        from skimage.filters import threshold_otsu
        from skimage.measure import label, regionprops
        from skimage.morphology import (
            binary_closing,
            disk,
            remove_small_objects,
        )
        try:
            t = threshold_otsu(image_uint8)
        except ValueError:
            t = int(image_uint8.mean())
        binary = image_uint8 > t
        binary = binary_closing(binary, disk(3))
        min_size = max(
            16, int(0.001 * image_uint8.size),
        )
        binary = remove_small_objects(binary, min_size=min_size)
        lbl = label(binary, connectivity=2)
        out: list[MaskProposal] = []
        for r in regionprops(lbl):
            if r.area < min_size:
                continue
            m = lbl == r.label
            y0, x0, y1, x1 = r.bbox
            out.append(MaskProposal(
                mask=m, score=1.0,
                bbox=(int(x0), int(y0),
                       int(x1) - 1, int(y1) - 1),
                area_px=int(r.area),
            ))
            if len(out) >= self.max_proposals:
                break
        # Sort largest-first so the UI renders the epi (outer
        # nerve outline) on top of the smaller fascicle blobs.
        out.sort(key=lambda p: -p.area_px)
        return out

    def segment_at(
        self,
        image_uint8: np.ndarray,
        *,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> MaskProposal:
        """Build a disk around the last positive point, or fill
        the bounding box if no points were supplied. This is
        intentionally dumb — the stub exists to exercise the
        downstream pipeline, not to do real segmentation."""
        h, w = image_uint8.shape[:2]
        mask = np.zeros((h, w), dtype=bool)
        if points is not None and len(points) > 0:
            pts = np.asarray(points, dtype=np.int32)
            labs = (
                np.asarray(point_labels, dtype=np.int32)
                if point_labels is not None
                else np.ones(len(pts), dtype=np.int32)
            )
            pos = pts[labs == 1]
            if len(pos) > 0:
                cy, cx = int(pos[-1, 1]), int(pos[-1, 0])
                r = max(8, min(h, w) // 30)
                yy, xx = np.ogrid[:h, :w]
                mask = (yy - cy) ** 2 + (xx - cx) ** 2 <= r ** 2
        elif box is not None:
            x0, y0, x1, y1 = box
            mask[
                max(0, y0):min(h, y1 + 1),
                max(0, x0):min(w, x1 + 1),
            ] = True
        else:
            raise ValueError(
                "segment_at needs either points or box",
            )
        ys, xs = np.where(mask)
        if len(ys) == 0:
            bbox = (0, 0, 0, 0)
        else:
            bbox = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
        return MaskProposal(
            mask=mask, score=1.0,
            bbox=bbox, area_px=int(mask.sum()),
        )


# --------------------------------------------------------------
# MedSAM2 backend — optional. Auto-detects checkpoint + skips
# silently when sam2 isn't importable.
# --------------------------------------------------------------

def _default_medsam_checkpoint() -> Optional[Path]:
    """Resolve the MedSAM2 weights path.

    Priority:
      1. `GOLGI_MEDSAM2_CHECKPOINT` env var (explicit override)
      2. `~/.cache/golgi/medsam2/medsam2.pt` (golgi default)
      3. `~/.cache/medsam2/medsam2.pt` (shared MedSAM2 default)
    Returns None when none of these exist on disk.
    """
    env = os.environ.get("GOLGI_MEDSAM2_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    for candidate in (
        Path.home() / ".cache/golgi/medsam2/medsam2.pt",
        Path.home() / ".cache/medsam2/medsam2.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _default_sam2_checkpoint() -> Optional[Path]:
    """Resolve the vanilla SAM2 weights path. Separate from
    MedSAM2 so users can A/B both backends.

    Priority:
      1. `GOLGI_SAM2_CHECKPOINT` env var
      2. `~/.cache/golgi/sam2/sam2.pt` (golgi default)
      3. `~/.cache/sam2/sam2_hiera_tiny.pt` (SAM2 default)
    """
    env = os.environ.get("GOLGI_SAM2_CHECKPOINT")
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    for candidate in (
        Path.home() / ".cache/golgi/sam2/sam2.pt",
        Path.home() / ".cache/sam2/sam2_hiera_tiny.pt",
        Path.home() / ".cache/sam2/sam2.1_hiera_tiny.pt",
    ):
        if candidate.is_file():
            return candidate
    return None


def _medsam2_available() -> tuple[bool, str]:
    """Probe for the MedSAM2 install + a usable checkpoint.
    Returns (ok, reason). `reason` is empty when ok=True;
    a one-line user-facing diagnostic otherwise."""
    try:
        import sam2  # noqa: F401  — module probe only
    except ImportError:
        return False, (
            "sam2 Python package not installed "
            "(pip install sam2 + download MedSAM2 weights)"
        )
    ckpt = _default_medsam_checkpoint()
    if ckpt is None:
        return False, (
            "no MedSAM2 checkpoint found — set "
            "GOLGI_MEDSAM2_CHECKPOINT or place "
            "medsam2.pt under ~/.cache/golgi/medsam2/"
        )
    return True, ""


class MedSAM2Segmenter:
    """MedSAM2 (Bo Wang et al., 2024 — arXiv 2408.03322).
    Fine-tuned SAM2 on 1.5 M medical image/mask pairs spanning
    CT / MRI / microscopy / µCT.

    The backend lazy-loads — the model is built on the first
    `propose_all` / `segment_at` call, not at construction
    time, so dialog open is fast even without GPU."""

    name = "MedSAM2"

    def __init__(
        self,
        *,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        config: Optional[str] = None,
        config_env_var: str = "GOLGI_MEDSAM2_CFG",
        default_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
    ) -> None:
        self.checkpoint = (
            checkpoint or _default_medsam_checkpoint()
        )
        if self.checkpoint is None:
            raise RuntimeError(
                "no MedSAM2 checkpoint provided — set "
                "GOLGI_MEDSAM2_CHECKPOINT or pass `checkpoint`",
            )
        self.device = device or (
            "cuda" if _torch_cuda_ok() else "cpu"
        )
        # config: explicit override > env var > class default.
        # `default_config` lets resolve_segmenter pick a different
        # default per backend (SAM2 needs sam2_hiera_t.yaml,
        # MedSAM2 needs sam2.1_hiera_t.yaml).
        self._config = (
            config
            or os.environ.get(config_env_var)
            or default_config
        )
        self._config_env_var = config_env_var
        self._predictor = None
        self._mask_gen = None
        self._loaded_for_image: int = -1

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import (
            SAM2ImagePredictor,
        )
        from sam2.automatic_mask_generator import (
            SAM2AutomaticMaskGenerator,
        )
        # Use the config resolved at construction time. SAM2 1.x
        # tiny / base+ / large differ in (blocks, embed_dim):
        # tiny=12/96, small=24/96, base+=24/112, large=48/144.
        # Picked config must match the checkpoint shape.
        cfg = self._config
        try:
            model = build_sam2(
                cfg, str(self.checkpoint),
                device=self.device,
            )
        except RuntimeError as ex:
            # Surface a focused diagnostic for the most common
            # cause (config ↔ checkpoint mismatch) without
            # burying it in a wall of state_dict diff.
            msg = str(ex)
            if "size mismatch" in msg or "Missing key" in msg:
                raise RuntimeError(
                    f"{self.name} config {cfg!r} doesn't "
                    f"match the checkpoint at "
                    f"{self.checkpoint}. Set "
                    f"{self._config_env_var} to a matching "
                    f"variant — common values: SAM2.1 tiny = "
                    f"configs/sam2.1/sam2.1_hiera_t.yaml "
                    f"(12 blocks, embed 96), SAM2.1 base+ = "
                    f"configs/sam2.1/sam2.1_hiera_b+.yaml "
                    f"(24, 112), SAM2.1 large = "
                    f"configs/sam2.1/sam2.1_hiera_l.yaml "
                    f"(48, 144). SAM2.0 tiny = "
                    f"configs/sam2/sam2_hiera_t.yaml (same "
                    f"shape as 2.1 tiny but different head). "
                    f"Original PyTorch error: "
                    f"{msg.splitlines()[0]}"
                ) from ex
            raise
        self._predictor = SAM2ImagePredictor(model)
        # Medical-imaging tuning for the everything-mode mask
        # generator. SAM2's defaults (pred_iou_thresh=0.8,
        # stability_score_thresh=0.95) are calibrated for
        # natural images and routinely return 0 masks on µCT /
        # MRI / OCT where edges are soft and contrast is low.
        # The values below match the medical-segmentation
        # literature (MedSAM2 paper + nnUNet defaults adapted
        # for SAM-style mask generators). Each is overridable
        # via GOLGI_SAM2_* env vars so power users can re-tune
        # without code changes.
        self._mask_gen = SAM2AutomaticMaskGenerator(
            model,
            points_per_side=int(os.environ.get(
                "GOLGI_SAM2_POINTS_PER_SIDE", "32",
            )),
            pred_iou_thresh=float(os.environ.get(
                "GOLGI_SAM2_IOU_THRESH", "0.6",
            )),
            stability_score_thresh=float(os.environ.get(
                "GOLGI_SAM2_STABILITY_THRESH", "0.85",
            )),
            min_mask_region_area=int(os.environ.get(
                "GOLGI_SAM2_MIN_AREA", "100",
            )),
            # Tightened from the SAM2 default 0.7 — the
            # everything-mode grid sampler on µCT often yields
            # near-duplicate masks (same fascicle, slightly
            # different boundary), and NMS at 0.45 drops the
            # second copy while keeping nested-but-distinct
            # masks (e.g. fascicle vs. its central lacuna).
            box_nms_thresh=float(os.environ.get(
                "GOLGI_SAM2_NMS_THRESH", "0.45",
            )),
        )

    def _set_image(self, image_uint8: np.ndarray) -> None:
        self._ensure_loaded()
        # SAM2 expects HxWx3 uint8 RGB. Promote grayscale by
        # tiling the channel; identity-preserving for the model.
        img = image_uint8
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        elif img.ndim == 3 and img.shape[2] == 1:
            img = np.repeat(img, 3, axis=2)
        # Cache the embedded image by id() so repeated clicks
        # on the same slice don't re-encode.
        key = id(image_uint8)
        if key != self._loaded_for_image:
            self._predictor.set_image(img)
            self._loaded_for_image = key

    def propose_all(
        self, image_uint8: np.ndarray,
    ) -> list[MaskProposal]:
        self._ensure_loaded()
        img = image_uint8
        if img.ndim == 2:
            img = np.stack([img, img, img], axis=-1)
        raw = self._mask_gen.generate(img)
        out: list[MaskProposal] = []
        for r in raw:
            m = np.asarray(r["segmentation"], dtype=bool)
            x, y, w, h = r["bbox"]
            out.append(MaskProposal(
                mask=m,
                score=float(r.get("predicted_iou", 1.0)),
                bbox=(int(x), int(y),
                       int(x + w - 1), int(y + h - 1)),
                area_px=int(r.get("area", int(m.sum()))),
            ))
        out.sort(key=lambda p: -p.area_px)
        return out

    def segment_at(
        self,
        image_uint8: np.ndarray,
        *,
        points: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[tuple[int, int, int, int]] = None,
    ) -> MaskProposal:
        if points is None and box is None:
            raise ValueError(
                "segment_at needs either points or box",
            )
        self._set_image(image_uint8)
        kw: dict = {"multimask_output": True}
        if points is not None and len(points) > 0:
            kw["point_coords"] = np.asarray(points, dtype=np.float32)
            kw["point_labels"] = (
                np.asarray(point_labels, dtype=np.int32)
                if point_labels is not None
                else np.ones(len(points), dtype=np.int32)
            )
        if box is not None:
            kw["box"] = np.asarray(box, dtype=np.float32)
        masks, scores, logits = self._predictor.predict(**kw)
        # Pick the best-scoring mask out of the 3 candidates.
        best = int(np.argmax(scores))
        m = np.asarray(masks[best], dtype=bool)
        ys, xs = np.where(m)
        if len(ys) == 0:
            bbox = (0, 0, 0, 0)
        else:
            bbox = (
                int(xs.min()), int(ys.min()),
                int(xs.max()), int(ys.max()),
            )
        return MaskProposal(
            mask=m, score=float(scores[best]),
            bbox=bbox, area_px=int(m.sum()),
            meta={"logits": logits[best]},
        )


# --------------------------------------------------------------
# SAM2 video predictor — keyframe-driven propagation across the
# z-stack. Treats the stack as a video, ingests user-corrected
# masks as conditioning prompts via `add_new_mask`, then
# propagates forward + backward to fill the rest of the volume.
#
# Lifecycle:
#   1. `prepare_video(stack_id, frames_uint8, cache_dir)` dumps
#      frames as JPEGs (idempotent — skips existing files) and
#      calls SAM2VideoPredictor.init_state on the cache dir.
#      The inference state is stashed in `self._states[stack_id]`.
#   2. `add_keyframe(stack_id, frame_idx, obj_id, mask)` injects
#      a user-corrected mask as a conditioning prompt.
#   3. `propagate(stack_id)` runs forward + backward
#      propagation from the earliest / latest conditioning
#      frame, drains the generator, and returns
#      `dict[frame_idx -> dict[obj_id -> mask_bool]]`. Object
#      IDs >= `RESERVED_OBJ_ID_BG` are stripped from the output
#      (they're the internal "not-fascicle" tracks used to give
#      vessels their own competing object so SAM2's non-overlap
#      consolidation pushes them out of the fascicle masks).
#   4. `remove_object(stack_id, obj_id)` / `reset(stack_id)` —
#      pass-through to the predictor's matching methods.
#
# Memory: the inference state holds image features for every
# frame at the model's resolution (~ 256×256 for tiny). With
# offload_state_to_cpu=True it sits on host RAM and only the
# active frame's slice moves to the compute device — keeps the
# 8 GB Macs (no CUDA) happy.

# Object IDs at or above this value are treated as internal-
# only (vessel / not-fascicle competition tracks) and are
# stripped from the propagate() output.
RESERVED_OBJ_ID_BG = 1000


class PropagationCancelled(Exception):
    """Raised by `SAM2VideoSegmenter.propagate(...)` when the
    optional `should_cancel()` callback returns True between
    frames. The caller is expected to catch this and unwind any
    "propagation in progress" state — the partial results are
    NOT returned because a half-propagated mask cache is harder
    for the user to recover from than no propagation at all."""


class SAM2VideoSegmenter:
    """SAM2 video predictor wrapped for keyframe propagation
    through the µCT stack. Sibling to MedSAM2Segmenter — same
    checkpoint, different predictor head. Not a `Segmenter`
    (different operation surface): see method signatures.

    Threading: not safe for concurrent calls per stack_id; the
    UI serialises via the busy lightbox.
    """

    name = "SAM2-video"

    def __init__(
        self,
        *,
        checkpoint: Optional[Path] = None,
        device: Optional[str] = None,
        config: Optional[str] = None,
        config_env_var: str = "GOLGI_MEDSAM2_CFG",
        default_config: str = "configs/sam2.1/sam2.1_hiera_t.yaml",
        offload_state_to_cpu: bool = True,
    ) -> None:
        self.checkpoint = (
            checkpoint or _default_medsam_checkpoint()
        )
        if self.checkpoint is None:
            raise RuntimeError(
                "no MedSAM2 checkpoint provided — set "
                "GOLGI_MEDSAM2_CHECKPOINT or pass `checkpoint`",
            )
        self.device = device or (
            "cuda" if _torch_cuda_ok() else "cpu"
        )
        self._config = (
            config
            or os.environ.get(config_env_var)
            or default_config
        )
        self._config_env_var = config_env_var
        self._offload_state_to_cpu = bool(offload_state_to_cpu)
        # Built on first prepare_video() call. None until the
        # first stack opens; persists for the dialog lifetime.
        self._predictor = None
        # stack_id -> dict (the inference state SAM2 returns).
        self._states: dict[str, dict] = {}

    def _ensure_loaded(self) -> None:
        if self._predictor is not None:
            return
        from sam2.build_sam import build_sam2_video_predictor
        cfg = self._config
        try:
            self._predictor = build_sam2_video_predictor(
                cfg, str(self.checkpoint),
                device=self.device,
            )
        except RuntimeError as ex:
            msg = str(ex)
            if "size mismatch" in msg or "Missing key" in msg:
                raise RuntimeError(
                    f"{self.name} config {cfg!r} doesn't "
                    f"match the checkpoint at "
                    f"{self.checkpoint}. Set "
                    f"{self._config_env_var} to the matching "
                    "variant — same convention as the image "
                    "predictor (sam2.1_hiera_t.yaml for the "
                    "tiny checkpoint, etc.). Original PyTorch "
                    f"error: {msg.splitlines()[0]}"
                ) from ex
            raise

    @staticmethod
    def _dump_frames_as_jpeg(
        frames_uint8: np.ndarray, cache_dir: Path,
    ) -> int:
        """Dump each (H, W) or (H, W, 3) uint8 slice as
        `<cache_dir>/<5-digit-idx>.jpg`. Idempotent: skips files
        that already exist on disk so re-opening a project's
        existing stack reuses the cache. Returns the frame
        count actually written.
        """
        from PIL import Image
        cache_dir.mkdir(parents=True, exist_ok=True)
        n = int(frames_uint8.shape[0])
        n_written = 0
        for fi in range(n):
            out = cache_dir / f"{fi:05d}.jpg"
            if out.exists():
                continue
            f = frames_uint8[fi]
            if f.ndim == 2:
                # SAM2's video loader expects RGB; replicate
                # grayscale µCT into all three channels.
                f = np.stack([f, f, f], axis=-1)
            Image.fromarray(f.astype(np.uint8)).save(
                out, format="JPEG", quality=92,
            )
            n_written += 1
        return n_written

    def prepare_video(
        self,
        stack_id: str,
        frames_uint8: np.ndarray,
        cache_dir: Path,
    ) -> dict:
        """Open / re-open the inference state for `stack_id`.
        `frames_uint8` is (N, H, W) or (N, H, W, 3) of the
        currently-displayed slices (post-crop, post-zoom — the
        same pixels the user has been editing on). Returns
        a summary dict for the caller's status banner.
        """
        self._ensure_loaded()
        # Idempotent dump first; then init_state from the
        # folder. The predictor caches image features inside
        # its returned state, so re-init on every prepare is
        # the simplest contract — we don't try to surgically
        # update features when frames change.
        n_written = self._dump_frames_as_jpeg(
            frames_uint8, cache_dir,
        )
        state = self._predictor.init_state(
            video_path=str(cache_dir),
            offload_state_to_cpu=self._offload_state_to_cpu,
        )
        self._states[stack_id] = state
        return {
            "n_frames": int(state["num_frames"]),
            "n_jpegs_written": int(n_written),
            "video_h": int(state["video_height"]),
            "video_w": int(state["video_width"]),
            "device": str(state["device"]),
        }

    def add_keyframe(
        self,
        stack_id: str,
        frame_idx: int,
        obj_id: int,
        mask: np.ndarray,
    ) -> None:
        """Inject a user-corrected mask at `frame_idx` for
        `obj_id`. Object IDs >= RESERVED_OBJ_ID_BG are passed
        through to SAM2 unchanged (so they compete for pixels
        against fascicle objects) but the propagate() result
        strips them before returning."""
        state = self._require_state(stack_id)
        # `add_new_mask` expects a 2D bool/float ndarray; the
        # predictor handles resize to model resolution.
        self._predictor.add_new_mask(
            inference_state=state,
            frame_idx=int(frame_idx),
            obj_id=int(obj_id),
            mask=mask.astype(bool),
        )

    def propagate(
        self,
        stack_id: str,
        *,
        on_progress=None,
        should_cancel=None,
    ) -> dict[int, dict[int, np.ndarray]]:
        """Run forward + backward propagation from the
        keyframes, drain both generators, and return per-frame
        per-object boolean masks.

        Hooks (both optional):
          * `on_progress(frame_idx, n_frames_total)` is called
            once per yielded frame so the caller can drive a
            progress UI.
          * `should_cancel()` is polled once per yielded frame
            BEFORE the next iteration; returning True breaks
            out of both sweeps and raises
            `PropagationCancelled` so the caller can clean up
            state. The forward sweep's partial output is
            DISCARDED on cancellation (we don't want to leave
            the per_slice cache in a half-propagated state
            that the user has to manually back out of).

        Frames the user marked as keyframes appear in the
        output with the user's exact mask (SAM2 reuses the
        conditioning output without re-prediction). Internal
        non-fascicle tracks (obj_id >= RESERVED_OBJ_ID_BG) are
        dropped from the returned dict.
        """
        state = self._require_state(stack_id)
        n_frames = int(state["num_frames"])
        out: dict[int, dict[int, np.ndarray]] = {}

        def _drain(gen):
            for frame_idx, obj_ids, mask_logits in gen:
                # mask_logits is (num_objs, 1, H, W) float;
                # > 0 = inside the predicted mask.
                _bool = (mask_logits > 0.0).cpu().numpy()
                per_obj: dict[int, np.ndarray] = {}
                for k, oid in enumerate(obj_ids):
                    oid_int = int(oid)
                    if oid_int >= RESERVED_OBJ_ID_BG:
                        continue
                    per_obj[oid_int] = (
                        _bool[k, 0].astype(bool)
                    )
                # If a frame is hit twice (forward + backward
                # sweeps meeting), the later write wins. For
                # the keyframe frames both passes yield the
                # same conditioning mask, so this is fine.
                # For non-conditioning frames the second pass
                # generally has better context (more memory
                # anchors on both sides) so favouring it is
                # usually correct.
                out[int(frame_idx)] = per_obj
                if on_progress is not None:
                    try:
                        on_progress(int(frame_idx), n_frames)
                    except Exception:               # noqa: BLE001
                        pass
                if should_cancel is not None:
                    try:
                        cancelled = bool(should_cancel())
                    except Exception:               # noqa: BLE001
                        cancelled = False
                    if cancelled:
                        raise PropagationCancelled()

        # Forward sweep first — defaults start from the
        # earliest conditioning frame and walk to the end.
        _drain(
            self._predictor.propagate_in_video(
                inference_state=state,
                reverse=False,
            ),
        )
        # Backward sweep — start from the earliest cond frame
        # walking back to frame 0. SAM2's `propagate_in_video`
        # skips reverse entirely when start_frame_idx == 0, so
        # we don't need to special-case the "keyframe is the
        # first slice" case here.
        _drain(
            self._predictor.propagate_in_video(
                inference_state=state,
                reverse=True,
            ),
        )
        return out

    def remove_object(
        self, stack_id: str, obj_id: int,
    ) -> None:
        """Drop an object's tracks from the inference state.
        Used when the user re-labels a propagated mask as
        background / not-a-fascicle."""
        state = self._require_state(stack_id)
        self._predictor.remove_object(
            state, int(obj_id), strict=False,
            need_output=False,
        )

    def reset(self, stack_id: str) -> None:
        """Wipe all prompts + tracking results for the stack
        but keep image features. Cheap re-arm; full re-init
        only needed when the underlying frames change."""
        state = self._states.get(stack_id)
        if state is None:
            return
        self._predictor.reset_state(state)

    def forget(self, stack_id: str) -> None:
        """Drop the inference state entirely — releases the
        per-frame feature tensors. Call when the user closes
        a stack."""
        self._states.pop(stack_id, None)

    def _require_state(self, stack_id: str) -> dict:
        state = self._states.get(stack_id)
        if state is None:
            raise RuntimeError(
                f"SAM2VideoSegmenter: no prepared state for "
                f"stack_id={stack_id!r}. Call "
                "prepare_video() first."
            )
        return state


def sam2_video_available() -> tuple[bool, str]:
    """Probe for the SAM2 video predictor module + a usable
    checkpoint. Returns (ok, reason) using the same one-line-
    diagnostic convention as `_medsam2_available`. Drives the
    UI gate that disables the Propagate button + tooltip text."""
    try:
        import sam2.sam2_video_predictor  # noqa: F401
    except ImportError:
        return False, (
            "sam2 Python package not installed "
            "(pip install sam2 + checkpoint)"
        )
    ckpt = _default_medsam_checkpoint() or _default_sam2_checkpoint()
    if ckpt is None:
        return False, (
            "no SAM2 checkpoint found — set "
            "GOLGI_MEDSAM2_CHECKPOINT or GOLGI_SAM2_CHECKPOINT"
        )
    return True, ""


def _torch_cuda_ok() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except ImportError:
        return False


# --------------------------------------------------------------
# Factory
# --------------------------------------------------------------

def resolve_segmenter(
    *,
    prefer: str = "auto",
    on_fallback: Optional[callable] = None,
) -> Segmenter:
    """Return the best available `Segmenter`.

    `prefer` is one of:
      • 'auto'     — MedSAM2 if available, else stub
      • 'medsam2'  — MedSAM2 or raise
      • 'sam2'     — vanilla SAM2 (different checkpoint path
                     than MedSAM2, see _default_sam2_checkpoint)
                     or raise
      • 'stub'     — always stub (useful for tests / quick
                     fallback when GPU / weights are missing)

    When MedSAM2 is unavailable and `prefer='auto'`, falls
    back to stub and calls `on_fallback(reason)` with a one-
    line diagnostic so the dialog can surface a warning chip.
    """
    p = prefer.lower()
    if p == "stub":
        return StubSegmenter()
    if p == "sam2":
        # Vanilla SAM2 — uses MedSAM2Segmenter's plumbing but
        # different checkpoint + config defaults. The HF
        # facebook/sam2-hiera-tiny checkpoint is the SAM2.0
        # (not 2.1) tiny variant, which uses
        # configs/sam2/sam2_hiera_t.yaml.
        # Override via GOLGI_SAM2_CFG for other variants.
        ckpt = _default_sam2_checkpoint()
        if ckpt is None:
            raise RuntimeError(
                "Vanilla SAM2 requested but no checkpoint "
                "found. Set GOLGI_SAM2_CHECKPOINT or place "
                "sam2.pt under ~/.cache/golgi/sam2/.",
            )
        try:
            import sam2  # noqa: F401
        except ImportError as ex:
            raise RuntimeError(
                "sam2 Python package not installed — "
                "needed for vanilla SAM2 backend "
                "(pip install sam2)."
            ) from ex
        seg = MedSAM2Segmenter(
            checkpoint=ckpt,
            config_env_var="GOLGI_SAM2_CFG",
            default_config="configs/sam2/sam2_hiera_t.yaml",
        )
        seg.name = "SAM2"
        return seg
    if p in ("medsam2", "auto"):
        ok, reason = _medsam2_available()
        if ok:
            return MedSAM2Segmenter()
        if p == "medsam2":
            raise RuntimeError(
                f"MedSAM2 requested but unavailable: {reason}",
            )
        if on_fallback is not None:
            try:
                on_fallback(reason)
            except Exception:                       # noqa: BLE001
                pass
        return StubSegmenter()
    raise ValueError(f"unknown prefer={prefer!r}")
