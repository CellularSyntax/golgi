# Meshing

golgi builds **multi-region tetrahedral meshes** of the whole modeling domain — endoneurium,
epineurium, perineurium interface, electrode contacts, cuff silicone, saline, and a surrounding
tissue (muscle) bath — on an open meshing stack: **TetGen** (primary volume mesher) and **Gmsh**
(geometry/extrusion). No commercial mesher is required.

Source: `golgi/pipeline/mesh.py` (driver), `golgi/pipeline/plc.py` (PLC assembly),
`golgi/pipeline/mesh_quality.py` (quality metrics), `golgi/compute/tetgen_runner.py` and
`golgi/compute/gmsh_mesher.py` (backends).

---

## How a mesh is built

1. **Assemble the PLC** (piecewise-linear complex) — the watertight surface description of every
   region and the electrode contacts, with region markers (`plc.py`,
   `assemble_multi_domain_plc` / `assemble_bare_nerve_in_bath`).
2. **Tetrahedralize** with TetGen, honoring per-region target sizes and quality switches.
3. **Tag cells** by region (the tags consumed by the [solver](Finite-Element-Solver) and the
   [conductivity model](Conductivity-and-Tissue-Properties)) and **tag facets** for the electrode
   contacts.
4. **Score quality** (per-region histograms) and surface it in the Mesh drawer.

The mesh is shared across electrode designs on the same nerve (the nerve and muscle bath are
canonical), while each design's cuff is placed in its own local frame — so switching designs only
moves the cuff, not the nerve geometry.

---

## Characteristic lengths

Each region has its own target element size (characteristic length, `lc_*`), set in the Mesh drawer
or via [`Study.set_mesh(...)`](Python-API). Finer near the contacts and inside the nerve, coarser in
the far-field bath:

| Parameter | Region |
|---|---|
| `lc_endo_um` | endoneurium (fascicle interior) |
| `lc_epi_um` | epineurium |
| `lc_muscle_um` | muscle / far-field bath |
| `lc_scar_um` | encapsulation / scar shell |
| `lc_saline_um` | cuff saline |
| `lc_silicone_um` | cuff silicone |
| `lc_contact_um` | electrode contacts |

Tissue-region sizes are shared across designs; cuff-region sizes are per design. Smaller values give
more accurate fields at higher mesh cost — the contacts and endoneurium dominate accuracy, so refine
those first.

There is also a **muscle bounding-box** placement (radial/axial padding and x/y/z offsets) controlling
the extent of the far-field bath and the ground boundary, plus a surface **decimation target**
(`decim_target_k`) applied to the imported nerve before meshing.

---

## Mesh quality

After a build, golgi computes per-element shape quality and shows **per-region histograms** plus a
combined view in the Mesh drawer (`mesh_quality.py`, `figures/mesh_stats.py`). Use these to catch
slivers or under-refinement before spending solve time. Quality figures are exportable like any other
([Figures & Reports](Figures-and-Reports)).

> golgi assumes the **imported nerve surface is watertight**. In-app mesh repair / Boolean editing is
> intentionally out of scope — repair non-manifold surfaces in MeshLab or Blender first. A light
> repair pass (pymeshfix) and decimation are applied, but a badly broken surface should be fixed
> upstream.

---

## TetGen vs Gmsh

- **TetGen** does the multi-region volume tetrahedralization from the PLC, with nerve_studio-style
  switches (small `epsilon`, near-180° angle tolerances) tuned for the thin cuff/contact features.
  TetGen's core is AGPL-licensed — which is part of why golgi is AGPL (see
  [License & Citation](License-and-Citation)).
- **Gmsh** provides OpenCASCADE geometry and extrusion (e.g. building a prismatic nerve from an
  extruded cross-section) and is available as an alternative mesher path.

The fiber-trajectory generator builds its **own** nerve-only tet mesh (decoupled from cuff position)
so cap detection is robust regardless of where the cuff sits — see
[Fiber Populations & Trajectories](Fiber-Populations-and-Trajectories).

---

## References

- Si H (2015). TetGen, a Delaunay-based quality tetrahedral mesh generator. *ACM Trans Math Softw* 41:11. — the volume mesher.
- Geuzaine C, Remacle J-F (2009). Gmsh: a 3-D finite element mesh generator with built-in pre- and post-processing facilities. *Int J Numer Methods Eng* 79:1309. — geometry/extrusion mesher.
- Pelot NA, Thio BJ, Grill WM (2019). On the parameters used in finite element modeling of compound peripheral nerves. *J Neural Eng* 16:016007. — multi-region nerve FE-model conventions.

---

### See also
[Geometry Import & Segmentation](Geometry-Import-and-Segmentation) ·
[Finite-Element Solver](Finite-Element-Solver) ·
[Conductivity & Tissue Properties](Conductivity-and-Tissue-Properties) ·
[Configuration Reference](Configuration-Reference)
