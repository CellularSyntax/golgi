# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""µCT / medical-image stack loader.

Format coverage (auto-dispatched by suffix + magic bytes):

  • TIFF      .tif / .tiff           tifffile  (lazy, multi-page)
  • DICOM     .dcm / .dicom          SimpleITK (single-file or
                                                multi-frame)
  • DICOM dir <directory of .dcm>    SimpleITK series reader
  • NIfTI     .nii / .nii.gz         SimpleITK
  • NRRD      .nrrd / .nhdr          SimpleITK
  • MetaImage .mha / .mhd            SimpleITK
  • Analyze   .img + .hdr            SimpleITK
  • JPEG 2000 .jp2 / .j2k / .jpx     SimpleITK (with OpenJPEG)
                                     → falls back to PIL
  • PNG/JPEG  .png .jpg .jpeg .bmp   PIL (single frame)

The TIFF path stays lazy (per-slice tifffile reads) because µCT
exports are routinely 100s of MB to several GB. Every other
format is loaded once into a NumPy volume cached on the Stack —
SimpleITK doesn't expose lazy slicing as cleanly, and most non-
TIFF medical formats are smaller than a TIFF stack anyway.

`to_display` does a percentile contrast stretch into uint8 for
UI rendering; the raw data stays in its native dtype for
downstream segmentation precision.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np


class StackError(RuntimeError):
    """Raised when a file can't be read as a µCT/medical image
    stack — bad format, missing pages, missing optional dep, or
    unsupported pixel layout."""


# Suffixes we recognise on the fast path. Magic-byte sniffing
# kicks in only when the suffix is ambiguous or missing.
_TIFF_SUFFIXES = {".tif", ".tiff"}
_DICOM_SUFFIXES = {".dcm", ".dicom"}
_NIFTI_SUFFIXES = {".nii"}        # .nii.gz handled separately
_NRRD_SUFFIXES = {".nrrd", ".nhdr"}
_METAIMG_SUFFIXES = {".mha", ".mhd"}
_ANALYZE_SUFFIXES = {".img", ".hdr"}
_JP2K_SUFFIXES = {".jp2", ".j2k", ".jpx", ".jpf"}
_PIL_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif"}


@dataclass
class Stack:
    """Handle to a (possibly multi-page) image stack.

    `volume` is the pre-loaded numpy volume of shape
    (n_frames, height, width) for formats SimpleITK / PIL read
    eagerly. For TIFF the field stays None and `read_slice`
    re-opens the file per request — keeping multi-GB scans off
    the heap.

    `voxel_size_um` is the (z, y, x) pixel pitch in micrometres
    when the format carries it (most medical formats do; basic
    PNG/JPEG/TIFF often don't). The segment dialog uses this for
    the pixel-to-mm scaling in Phase B; if absent the user
    supplies it manually.
    """
    path: Path
    format_kind: str
    n_frames: int
    height: int
    width: int
    dtype: str
    voxel_size_um: Optional[tuple[float, float, float]] = None
    is_multipage: bool = True
    volume: Optional[np.ndarray] = field(default=None, repr=False)
    extra: dict = field(default_factory=dict)
    # Display-range hint pulled from DICOM Window Center /
    # Window Width tags (or any other format-specific source).
    # `to_display` uses this as the contrast-stretch window
    # when present, falling back to the percentile heuristic
    # otherwise. CT scanners pack scene-appropriate presets
    # here (e.g. "40 / 400" for soft tissue), which makes a
    # massive visual difference vs. blind percentile clipping
    # when the structure of interest occupies a large fraction
    # of the slice.
    display_window: Optional[tuple[float, float]] = None


# ----- Format-specific loaders -----

def _is_dicom_file(path: Path) -> bool:
    """Probe a single file for DICOM by reading the 128-byte
    preamble + 'DICM' marker. Survives mis-named files (some
    scanners drop the .dcm extension)."""
    try:
        with open(path, "rb") as f:
            f.seek(128)
            return f.read(4) == b"DICM"
    except OSError:
        return False


def _is_dicom_dir(path: Path) -> bool:
    """A directory is a DICOM series if at least one file inside
    smells like DICOM. We don't scan recursively — series live
    in flat directories in the standard layout."""
    if not path.is_dir():
        return False
    for child in path.iterdir():
        if child.is_file() and _is_dicom_file(child):
            return True
    return False


def _vsz_from_sitk(image) -> tuple[float, float, float]:
    """SimpleITK exposes spacing in mm in (x, y, z) order; we
    return (z, y, x) in micrometres to match the Stack
    convention."""
    sp = image.GetSpacing()              # (x, y, z) in mm
    if len(sp) == 2:
        sp = (sp[0], sp[1], 1.0)
    z, y, x = float(sp[2]), float(sp[1]), float(sp[0])
    return (z * 1000.0, y * 1000.0, x * 1000.0)


def _vol_from_sitk(image) -> np.ndarray:
    """SimpleITK arrays come back as (z, y, x) which matches the
    Stack convention. 2D images get a leading 1-axis so the rest
    of the pipeline can iterate 'n_frames' uniformly."""
    import SimpleITK as sitk
    arr = sitk.GetArrayFromImage(image)
    if arr.ndim == 2:
        arr = arr[None, ...]
    elif arr.ndim == 4 and arr.shape[-1] in (3, 4):
        # RGB/RGBA volume — collapse to luma.
        arr = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        ).astype(arr.dtype, copy=False)
    return arr


def _read_dicom_window(file_path: Path) -> Optional[tuple[float, float]]:
    """Pull Window Center / Window Width from a single DICOM
    file. Returns `(lo, hi)` describing the display range, or
    None if either tag is missing / unparsable.

    DICOM `0028|1050` (Window Center) and `0028|1051` (Window
    Width) are IS / DS strings that can contain multiple
    preset values separated by backslash (CT often packs
    "40\\400" for soft tissue, "1500\\1500" for lung, etc.).
    We take the FIRST value of each — the scanner's primary
    preset, which is what viewers like ITK-SNAP default to."""
    try:
        import SimpleITK as sitk
    except ImportError:
        return None
    try:
        rdr = sitk.ImageFileReader()
        rdr.SetFileName(str(file_path))
        rdr.LoadPrivateTagsOn()
        rdr.ReadImageInformation()
    except Exception:                                    # noqa: BLE001
        return None

    def _first_float(s: str) -> Optional[float]:
        for tok in str(s).replace("\\", " ").split():
            try:
                return float(tok)
            except ValueError:
                continue
        return None

    try:
        if not (rdr.HasMetaDataKey("0028|1050")
                and rdr.HasMetaDataKey("0028|1051")):
            return None
        wc = _first_float(rdr.GetMetaData("0028|1050"))
        ww = _first_float(rdr.GetMetaData("0028|1051"))
    except Exception:                                    # noqa: BLE001
        return None
    if wc is None or ww is None or ww <= 0.0:
        return None
    return (wc - 0.5 * ww, wc + 0.5 * ww)


def compress_dicom_series_to_nifti(
    series_dir: Path,
    *,
    delete_originals: bool = True,
    on_log=None,
) -> Path:
    """Convert N DICOM files in `series_dir` into one
    `volume.nii.gz` (lossless gzip) + a `volume.json` sidecar
    carrying the display window pulled from the DICOM tags.

    Disk savings on a typical sheep-VN µCT series: ~4-7×
    (gzip alone is ~2-3× on int16 CT data; the int32→int16
    cast adds another ~2× whenever the dynamic range fits,
    which it does for all CT scanners). The DICOM series
    reader otherwise loads the full uncompressed volume into
    RAM on every dialog open, which is the immediate pain
    point for 300+ slice stacks.

    `delete_originals=True` removes the .dcm files after a
    successful write; the .nii.gz becomes the sole source of
    truth. `on_log(str)` is called with progress lines so the
    caller can stream them to the dialog's busy log.

    Returns the .nii.gz path. Raises `StackError` if the
    SimpleITK series read or write fails.
    """
    say = on_log if on_log is not None else (lambda *_: None)
    try:
        import SimpleITK as sitk
    except ImportError as ex:
        raise StackError(
            "SimpleITK not installed — required to compress "
            "DICOM series (pip install SimpleITK)"
        ) from ex
    reader = sitk.ImageSeriesReader()
    try:
        series_ids = reader.GetGDCMSeriesIDs(str(series_dir))
    except Exception as ex:                              # noqa: BLE001
        raise StackError(
            f"DICOM probe failed on {series_dir}: {ex}"
        ) from ex
    if not series_ids:
        raise StackError(
            f"no DICOM series found in {series_dir}",
        )
    files = reader.GetGDCMSeriesFileNames(
        str(series_dir), series_ids[0],
    )
    n_files = len(files)
    reader.SetFileNames(files)
    say(f"reading {n_files} DICOM file(s) …")
    try:
        image = reader.Execute()
    except Exception as ex:                              # noqa: BLE001
        raise StackError(
            f"DICOM series read failed: {ex}",
        ) from ex

    # int32 → int16 cast whenever the data range fits. CT
    # scanners emit int16 natively; the int32 promotion is a
    # SimpleITK-side artefact (rescale-slope flag in DICOM).
    # Casting back is lossless and halves the on-disk size
    # before gzip even runs.
    arr = sitk.GetArrayFromImage(image)
    if arr.dtype in (np.int32, np.uint32, np.int64, np.uint64):
        vmin = int(arr.min())
        vmax = int(arr.max())
        if vmin >= -32768 and vmax <= 32767:
            say(
                f"casting {arr.dtype} → int16 (range "
                f"[{vmin}, {vmax}] fits)"
            )
            image = sitk.Cast(image, sitk.sitkInt16)
        elif vmin >= 0 and vmax <= 65535:
            say(
                f"casting {arr.dtype} → uint16 (range "
                f"[{vmin}, {vmax}] fits)"
            )
            image = sitk.Cast(image, sitk.sitkUInt16)

    nii_path = series_dir / "volume.nii.gz"
    say(f"writing {nii_path.name} (gzip-compressed) …")
    try:
        sitk.WriteImage(
            image, str(nii_path), useCompression=True,
        )
    except Exception as ex:                              # noqa: BLE001
        # Wipe partial output so a subsequent load doesn't
        # pick up a half-written file.
        try:
            nii_path.unlink()
        except OSError:
            pass
        raise StackError(
            f"NIfTI write failed: {ex}",
        ) from ex

    # Sidecar — display window + provenance. NIfTI's own
    # header doesn't have a slot for DICOM Window Center /
    # Width, so we keep them next to the volume as JSON.
    import json as _json
    sidecar_data: dict = {
        "source": "dicom_series",
        "series_id": series_ids[0],
        "n_source_files": n_files,
    }
    win = _read_dicom_window(Path(files[0]))
    if win is not None:
        sidecar_data["display_window"] = list(win)
    sidecar_path = series_dir / "volume.json"
    try:
        sidecar_path.write_text(
            _json.dumps(sidecar_data, indent=2),
         encoding="utf-8")
    except OSError:
        pass

    nii_size = nii_path.stat().st_size
    say(
        f"wrote {nii_path.name} "
        f"({nii_size / (1024 * 1024):.1f} MB)"
    )

    if delete_originals:
        n_deleted = 0
        for f in files:
            try:
                Path(f).unlink()
                n_deleted += 1
            except OSError:
                pass
        say(f"deleted {n_deleted} / {n_files} originals")

    return nii_path


def _load_sitk(path: Path, kind: str) -> Stack:
    """Read any single-file format SimpleITK supports
    (DICOM single-frame / multi-frame, NIfTI, NRRD, MetaImage,
    Analyze, JPEG2000 when OpenJPEG is compiled in)."""
    try:
        import SimpleITK as sitk
    except ImportError as ex:
        raise StackError(
            f"SimpleITK not installed — required to read "
            f"{kind} files (pip install SimpleITK)"
        ) from ex
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    try:
        image = reader.Execute()
    except Exception as ex:                              # noqa: BLE001
        raise StackError(
            f"SimpleITK failed to read {kind} file {path}: "
            f"{ex}"
        ) from ex
    arr = _vol_from_sitk(image)
    # DICOM window only meaningful for the dicom kind — other
    # SimpleITK-routed formats (NIfTI / NRRD / MetaImage) don't
    # carry these tags.
    win = (
        _read_dicom_window(path) if kind == "dicom" else None
    )
    return Stack(
        path=path,
        format_kind=kind,
        n_frames=int(arr.shape[0]),
        height=int(arr.shape[1]),
        width=int(arr.shape[2]),
        dtype=str(arr.dtype),
        voxel_size_um=_vsz_from_sitk(image),
        is_multipage=arr.shape[0] > 1,
        volume=arr,
        extra={
            "sitk_pixel_id": image.GetPixelIDTypeAsString(),
        },
        display_window=win,
    )


def _load_dicom_series(path: Path) -> Stack:
    """Load a directory of single-frame DICOM files as a 3D
    volume. SimpleITK's ImageSeriesReader sorts by image-
    position-patient (ipp) automatically so axial / coronal /
    sagittal series load in the right order."""
    try:
        import SimpleITK as sitk
    except ImportError as ex:
        raise StackError(
            "SimpleITK not installed — required for DICOM "
            "series (pip install SimpleITK)"
        ) from ex
    reader = sitk.ImageSeriesReader()
    # Try the primary series in the dir. Most µCT exports
    # contain a single series; if multiple are present we pick
    # the first.
    try:
        series_ids = reader.GetGDCMSeriesIDs(str(path))
    except Exception as ex:                              # noqa: BLE001
        raise StackError(
            f"DICOM probe failed on {path}: {ex}"
        ) from ex
    if not series_ids:
        raise StackError(
            f"no DICOM series found in directory {path}",
        )
    files = reader.GetGDCMSeriesFileNames(
        str(path), series_ids[0],
    )
    reader.SetFileNames(files)
    try:
        image = reader.Execute()
    except Exception as ex:                              # noqa: BLE001
        raise StackError(
            f"DICOM series read failed: {ex}",
        ) from ex
    arr = _vol_from_sitk(image)
    # Pull display window from the FIRST file's DICOM tags.
    # The series reader doesn't expose per-file metadata in a
    # convenient way, but every file in the series shares the
    # same WC / WW preset (scanner-set, study-wide), so probing
    # file 0 is sufficient and cheap.
    win = (
        _read_dicom_window(Path(files[0]))
        if files else None
    )
    return Stack(
        path=path,
        format_kind="dicom_series",
        n_frames=int(arr.shape[0]),
        height=int(arr.shape[1]),
        width=int(arr.shape[2]),
        dtype=str(arr.dtype),
        voxel_size_um=_vsz_from_sitk(image),
        is_multipage=arr.shape[0] > 1,
        volume=arr,
        extra={
            "series_id": series_ids[0],
            "n_files": len(files),
        },
        display_window=win,
    )


def _load_tiff(path: Path) -> Stack:
    """Lazy TIFF — metadata only at load time. The slice scrubber
    pulls pages on demand via `read_slice`."""
    try:
        import tifffile
    except ImportError as ex:
        raise StackError(
            "tifffile not installed — required for TIFF stacks "
            "(pip install tifffile)"
        ) from ex
    try:
        with tifffile.TiffFile(str(path)) as tif:
            n_frames = len(tif.pages)
            if n_frames == 0:
                raise StackError(f"TIFF has zero pages: {path}")
            page0 = tif.pages[0]
            height = int(page0.imagelength)
            width = int(page0.imagewidth)
            dtype = str(page0.dtype)
            vsz = _read_tiff_resolution_um(tif)
            extra = {
                "compression": str(
                    getattr(page0, "compression", "") or "",
                ),
            }
    except StackError:
        raise
    except Exception as ex:                              # noqa: BLE001
        raise StackError(f"failed to open TIFF {path}: {ex}") from ex
    return Stack(
        path=path,
        format_kind="tiff",
        n_frames=n_frames,
        height=height,
        width=width,
        dtype=dtype,
        voxel_size_um=vsz,
        is_multipage=n_frames > 1,
        extra=extra,
    )


def _read_tiff_resolution_um(
    tif,
) -> Optional[tuple[float, float, float]]:
    """Pull voxel size (z, y, x) in micrometres from a TIFF.

    Tries the most informative sources first:

      1. ImageJ-style metadata — `spacing` (z) + xy resolution
         tags + `unit` (covers ImageJ / Fiji-processed stacks).
      2. OME-XML metadata — `PhysicalSizeX/Y/Z` + their `Unit`
         attributes (covers Bio-Formats exports).
      3. Standard TIFF tags XResolution + YResolution +
         ResolutionUnit (covers most scanner exports).

    Returns None only when none of these yield a sensible
    value — the segment dialog then asks the user manually.
    """
    page0 = tif.pages[0]

    # 1) ImageJ-style metadata.
    try:
        meta = tif.imagej_metadata or {}
        if meta:
            unit = str(meta.get("unit", "") or "").lower()
            z_um = _to_um(
                meta.get("spacing"), unit,
            ) if "spacing" in meta else None
            # ImageJ also stores XResolution / YResolution in
            # the TIFF tags themselves (one resolution shared
            # for both axes). Read those if present so we get
            # a per-axis xy pitch rather than assuming
            # isotropy.
            xy_um = _tiff_xy_resolution_um(page0)
            if z_um is not None or xy_um is not None:
                x_um = (xy_um[0] if xy_um else z_um)
                y_um = (xy_um[1] if xy_um else z_um)
                z_um = z_um if z_um is not None else (
                    xy_um[0] if xy_um else 0.0
                )
                if x_um and y_um and z_um:
                    return (
                        float(z_um),
                        float(y_um),
                        float(x_um),
                    )
    except (AttributeError, ValueError, KeyError):
        pass

    # 2) OME-XML metadata (Bio-Formats / many synchrotron
    # exports). Carries explicit physical pixel sizes + units.
    try:
        ome = getattr(tif, "ome_metadata", None)
        if ome:
            vsz = _ome_voxel_size_um(ome)
            if vsz is not None:
                return vsz
    except (AttributeError, ValueError):
        pass

    # 3) Plain TIFF resolution tags (no ImageJ wrapper).
    xy_um = _tiff_xy_resolution_um(page0)
    if xy_um is not None:
        # No z info — assume isotropic (common for single-
        # slice scans + good-enough default for prismatic
        # extrusion which only needs the in-plane pitch).
        return (xy_um[0], xy_um[1], xy_um[0])

    return None


def _to_um(value, unit: str) -> Optional[float]:
    """Convert (value, unit string) to micrometres. Handles
    µm / um / micron / mm / cm / m / inch / pixel."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if v <= 0.0:
        return None
    u = unit.lower().strip()
    if u in ("", "pixel", "px"):
        return None  # dimensionless — useless for FEM
    if u in ("µm", "um", "micron", "microns", "micrometer",
              "micrometers", "micrometre", "micrometres"):
        return v
    if u in ("mm", "millimeter", "millimeters",
              "millimetre", "millimetres"):
        return v * 1000.0
    if u in ("cm", "centimeter", "centimeters"):
        return v * 10000.0
    if u in ("m", "meter", "meters", "metre", "metres"):
        return v * 1_000_000.0
    if u in ("inch", "inches", "in"):
        return v * 25400.0
    return None


def _tiff_xy_resolution_um(
    page,
) -> Optional[tuple[float, float]]:
    """Read XResolution / YResolution from a TIFF page and
    convert to micrometres per pixel. Returns (y_um, x_um) or
    None.

    TIFF resolution tags store DOTS PER UNIT, so the per-pixel
    size is the reciprocal. ResolutionUnit values per the
    TIFF 6.0 spec: 1=no absolute unit, 2=inch, 3=centimeter.
    """
    try:
        tags = page.tags
        xres = tags.get("XResolution")
        yres = tags.get("YResolution")
        unit = tags.get("ResolutionUnit")
        if xres is None or yres is None:
            return None
        # `value` may be a (num, denom) fraction or a float.
        x_dpu = _frac_to_float(xres.value)
        y_dpu = _frac_to_float(yres.value)
        if (
            x_dpu is None or y_dpu is None
            or x_dpu <= 0.0 or y_dpu <= 0.0
        ):
            return None
        # 1/dpu → unit per pixel; convert to µm.
        u_val = int(unit.value) if unit is not None else 2
        if u_val == 2:        # inch
            x_um = (1.0 / x_dpu) * 25400.0
            y_um = (1.0 / y_dpu) * 25400.0
        elif u_val == 3:      # centimetre
            x_um = (1.0 / x_dpu) * 10000.0
            y_um = (1.0 / y_dpu) * 10000.0
        else:                 # u_val == 1: no absolute unit
            return None
        return (y_um, x_um)
    except (AttributeError, ValueError, KeyError, TypeError):
        return None


def _frac_to_float(v) -> Optional[float]:
    """TIFF rationals come back as (numerator, denominator)
    tuples or already-divided floats — normalise both."""
    if v is None:
        return None
    try:
        if isinstance(v, (tuple, list)) and len(v) == 2:
            n, d = float(v[0]), float(v[1])
            return n / d if d != 0 else None
        return float(v)
    except (TypeError, ValueError):
        return None


def _ome_voxel_size_um(
    ome_xml: str,
) -> Optional[tuple[float, float, float]]:
    """Parse OME-XML for PhysicalSizeX/Y/Z + their units.
    OME-XML is the de-facto medical / microscopy metadata
    standard exported by Bio-Formats."""
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(ome_xml)
    except (ET.ParseError, ValueError):
        return None
    # OME wraps everything in a namespace; iterate over
    # 'Pixels' elements regardless of namespace.
    for pix in root.iter():
        if not pix.tag.endswith("Pixels"):
            continue
        attrs = pix.attrib
        x = _to_um(
            attrs.get("PhysicalSizeX"),
            attrs.get("PhysicalSizeXUnit", "µm"),
        )
        y = _to_um(
            attrs.get("PhysicalSizeY"),
            attrs.get("PhysicalSizeYUnit", "µm"),
        )
        z = _to_um(
            attrs.get("PhysicalSizeZ"),
            attrs.get("PhysicalSizeZUnit", "µm"),
        ) or x  # fall back to isotropic
        if x and y:
            return (float(z or x), float(y), float(x))
    return None


def _load_pil(path: Path, kind: str) -> Stack:
    """PNG / JPEG / BMP / GIF / JPEG 2000 fallback. Most formats
    are single-frame; GIF can be multi-frame but we don't expose
    those frames as a stack — pick a representative one and
    treat as 1 page."""
    try:
        from PIL import Image
    except ImportError as ex:
        raise StackError(
            "Pillow not installed — required for "
            f"{kind} files (pip install pillow)"
        ) from ex
    try:
        with Image.open(path) as im:
            im.load()
            arr = np.asarray(im)
    except Exception as ex:                              # noqa: BLE001
        raise StackError(
            f"PIL failed to open {kind} file {path}: {ex}"
        ) from ex
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        arr = (
            0.299 * arr[..., 0]
            + 0.587 * arr[..., 1]
            + 0.114 * arr[..., 2]
        ).astype(arr.dtype, copy=False)
    if arr.ndim != 2:
        raise StackError(
            f"image has unexpected shape: {arr.shape}",
        )
    arr3 = arr[None, ...]
    return Stack(
        path=path,
        format_kind=kind,
        n_frames=1,
        height=int(arr.shape[0]),
        width=int(arr.shape[1]),
        dtype=str(arr.dtype),
        voxel_size_um=None,
        is_multipage=False,
        volume=arr3,
    )


# ----- Public dispatch -----

def load_stack(path: str | Path) -> Stack:
    """Open any supported µCT / medical-imaging file and return
    its `Stack` handle. See module docstring for the format
    table.

    A directory is interpreted as a DICOM series."""
    p = Path(path).expanduser()
    if not p.exists():
        raise StackError(f"path does not exist: {p}")

    if p.is_dir():
        # Prefer a pre-compressed volumetric file in the dir
        # if one is present. The upload route writes this
        # alongside the original DICOMs (and deletes them
        # after a successful compress) — so post-compression
        # reload is instant instead of re-reading N DICOMs.
        for cand_name in ("volume.nii.gz", "volume.nii"):
            cand = p / cand_name
            if cand.is_file():
                kind = "nifti"
                stack = _load_sitk(cand, kind=kind)
                # Apply the sidecar display window — the
                # DICOM WC / WW didn't survive the NIfTI
                # write, but we preserved it in volume.json
                # at compress time.
                sidecar = p / "volume.json"
                if sidecar.is_file():
                    import json as _json
                    try:
                        meta = _json.loads(
                            sidecar.read_text(encoding="utf-8"),
                        )
                        win = meta.get("display_window")
                        if (
                            isinstance(win, (list, tuple))
                            and len(win) == 2
                        ):
                            stack.display_window = (
                                float(win[0]),
                                float(win[1]),
                            )
                    except Exception:                    # noqa: BLE001
                        pass
                # Keep the directory path on the Stack so the
                # SAM2 cache + per-stack ids hash on the
                # series dir, not on the .nii.gz inside.
                stack.path = p
                return stack
        if _is_dicom_dir(p):
            return _load_dicom_series(p)
        raise StackError(
            f"directory {p} contains no DICOM files "
            f"(other directory formats not supported)",
        )

    suffix = p.suffix.lower()
    # NIfTI's compressed variant has a double suffix .nii.gz —
    # the Path-level `.suffix` only sees `.gz` which doesn't help.
    if suffix == ".gz" and p.name.lower().endswith(".nii.gz"):
        return _load_sitk(p, kind="nifti")

    if suffix in _TIFF_SUFFIXES:
        return _load_tiff(p)
    if suffix in _DICOM_SUFFIXES or _is_dicom_file(p):
        return _load_sitk(p, kind="dicom")
    if suffix in _NIFTI_SUFFIXES:
        return _load_sitk(p, kind="nifti")
    if suffix in _NRRD_SUFFIXES:
        return _load_sitk(p, kind="nrrd")
    if suffix in _METAIMG_SUFFIXES:
        return _load_sitk(p, kind="metaimage")
    if suffix in _ANALYZE_SUFFIXES:
        return _load_sitk(p, kind="analyze")
    if suffix in _JP2K_SUFFIXES:
        # SimpleITK reads JP2 when built with OpenJPEG; fall
        # back to PIL when not.
        try:
            return _load_sitk(p, kind="jpeg2000")
        except StackError:
            return _load_pil(p, kind="jpeg2000")
    if suffix in _PIL_SUFFIXES:
        return _load_pil(p, kind=suffix.lstrip("."))
    raise StackError(
        f"unsupported file format: {suffix or '(no suffix)'}",
    )


def read_slice(stack: Stack, idx: int) -> np.ndarray:
    """Return slice `idx` as a 2D numpy array in the source
    dtype. Lazy TIFF reads re-open the file per request; pre-
    loaded volumes (SimpleITK + PIL) return a view into the
    cached numpy volume."""
    if idx < 0 or idx >= stack.n_frames:
        raise StackError(
            f"slice index {idx} out of range "
            f"[0, {stack.n_frames})",
        )
    if stack.volume is not None:
        return np.ascontiguousarray(stack.volume[idx])
    if stack.format_kind == "tiff":
        import tifffile
        with tifffile.TiffFile(str(stack.path)) as tif:
            arr = tif.pages[idx].asarray()
        if arr.ndim == 3 and arr.shape[2] in (3, 4):
            arr = (
                0.299 * arr[..., 0]
                + 0.587 * arr[..., 1]
                + 0.114 * arr[..., 2]
            ).astype(arr.dtype, copy=False)
        if arr.ndim != 2:
            raise StackError(
                f"TIFF slice has unexpected shape: {arr.shape}",
            )
        return arr
    raise StackError(
        f"no slice reader for format_kind={stack.format_kind!r}",
    )


def to_display(
    arr: np.ndarray,
    *,
    contrast_pct: tuple[float, float] = (1.0, 99.0),
    invert: bool = False,
    window: Optional[tuple[float, float]] = None,
) -> np.ndarray:
    """Map a slice into uint8 for UI display.

    When `window=(lo, hi)` is provided (typically from DICOM
    Window Center / Window Width tags via Stack.display_window),
    that range is used directly — this matches what medical
    viewers like ITK-SNAP do and produces clean tissue contrast
    even when the structure of interest fills a large fraction
    of the slice. Otherwise falls back to a percentile clip
    (default 1-99) which is robust to beam-hardening / hot-pixel
    artefacts and works well on µCT TIFF where the sample fills
    a known fraction of the frame.

    `invert=True` flips the LUT (some medical modalities display
    tissue dark on light)."""
    if arr.size == 0:
        return np.zeros_like(arr, dtype=np.uint8)
    a = arr.astype(np.float32, copy=False)
    if window is not None:
        lo, hi = float(window[0]), float(window[1])
    else:
        lo_pct, hi_pct = contrast_pct
        lo = float(np.percentile(a, lo_pct))
        hi = float(np.percentile(a, hi_pct))
    if hi <= lo:
        hi = lo + 1.0
    out = np.clip((a - lo) / (hi - lo), 0.0, 1.0)
    if invert:
        out = 1.0 - out
    return (out * 255.0).astype(np.uint8)
