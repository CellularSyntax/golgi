# Geometry Import & Segmentation

The pipeline starts from nerve geometry. golgi accepts ready-made surfaces, pre-built bundles, and —
for raw imaging — provides **promptable segmentation** and 3-D reconstruction to turn microscopy /
µCT into a meshable nerve.

Source: `golgi/segmentation/` (`image.py`, `segmenter.py`, `reconstruct3d.py`, `deform.py`),
`golgi/projects/uct_route.py` (upload), and `Study.import_nerve` for surfaces.

---

## Ways to get geometry in

| Path | Input | Notes |
|---|---|---|
| **Surface import** | STL / NAS / OBJ | a watertight nerve surface; the fastest route. `Study.import_nerve(...)` or the Import wizard's *STL surface* tile. |
| **µCT / histology bundle** | a golgi bundle (epineurium + per-fascicle endoneurium) | produced by segmentation; carries multiple fascicles. |
| **Segment from images** | TIFF / DICOM / NIfTI / NRRD / PNG / … | segment slices in-app, then reconstruct to surfaces. |

On surface import, golgi computes the global **PCA frame**, per-triangle **surface quality**, and
topology stats (component count, watertightness, bounding box). `scale_factor` converts file units to
metres (default mm→m). golgi assumes a **watertight** surface — repair broken meshes upstream (MeshLab
/ Blender); see [Meshing](Meshing).

---

## Promptable segmentation

The **Segment µCT slice** dialog segments image slices interactively:

- **Image loading** (`image.py`) reads many formats — **TIFF** (lazy per-slice, multi-GB-safe),
  **DICOM** (single file or series), **NIfTI**, **NRRD**, **MetaImage**, **Analyze**, **JPEG2000**,
  and ordinary **PNG/JPEG** — capturing voxel size where available.
- **Segmenter** (`segmenter.py`) uses a **MedSAM2 / SAM-style** backend when a model checkpoint is
  available: *propose-all* ("everything") masks, or *segment-at* refinement from positive/negative
  point clicks and a box. Each proposal carries a confidence score, bounding box, and area.
- **Without a model checkpoint**, golgi falls back to a **stub segmenter** so the workflow still runs;
  the checkpoint is an optional install (see [Installation](Installation)).

Accept and label masks per slice (e.g. fascicle IDs), and golgi assembles them into a nerve.

## 3-D reconstruction

`reconstruct3d.py` turns segmented slices into surfaces (mm units):

- **`extrude_single_slice`** — one annotated cross-section + a thickness → a prismatic nerve (useful
  for extruded-cross-section studies and as a control against full 3-D reconstruction).
- **`reconstruct_stack`** — a slice range (gaps filled) → per-fascicle surfaces via 3-D connected
  components + marching cubes, with optional smoothing.

The result is the multifascicular geometry (epineurium + per-fascicle endoneurium) the rest of the
pipeline meshes and solves.

## Uploads

Large images upload through a streaming route (`uct_route.py`) — multipart POST written to the project
in 64 KB chunks (so multi-GB DICOM series don't blow up memory), with progress reported back to the
GUI.

---

### See also
[GUI Walkthrough](GUI-Walkthrough) · [Meshing](Meshing) ·
[Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories) · [Getting Started](Getting-Started)
