# Electrodes & Cuff Designer

golgi models a broad range of cuff and intrafascicular electrodes, fits them to the imported nerve,
and supports arbitrary multi-contact (N-polar) montages with per-contact current fractions — the
basis for **current steering** and selective stimulation.

Source: `cuff_designer.py` (parametric ASCENT-style primitives), `golgi/scene/cuff_fit.py` (PCA
fitting), `golgi/scene/electrode_patches.py` (renderable + meshable contacts),
`golgi/state_defaults/electrode.py` + `cuff.py` (defaults).

---

## Electrode types

Set via the **Designs** drawer (GUI) or `electrode_type` in [`set_electrodes`](Python-API). Each type
has its own geometry parameters (defaults shown):

| Type (`electrode_type`) | Contacts | Models | Key parameters |
|---|---|---|---|
| `bipolar ring-pair` | 2 | standard two-ring cuff | axial separation (4 mm), ring width (0.6 mm) |
| `tripolar (anode-cathode-anode)` | 3 | guarded tripole | ring separation (2 mm), ring width (0.6 mm) |
| `ring-array (NxM)` | rows × cols | segmented multi-contact ring array | rows (2), cols (4), row separation (3 mm), contact width (0.6 mm), contact arc (60°) |
| `helical (Livanova-style)` | bands (2) | LivaNova helical spiral | bands (2), pitch (12 mm/turn), band arc (180°), band separation (8 mm) |
| `LIFE (longitudinal intrafascicular)` | rows × cols | Boretius/Rossini intrafascicular wires | contacts/wire, # wires, contact spacing, wire spacing, contact length, wire Ø (80 µm), target fascicle |
| `TIME (transverse intrafascicular)` | rows × cols | Boretius 2010 transverse ribbon | rows, cols, row/col spacing (230 µm pitch), contact width, ribbon width/thickness, chord angle, target |
| `DUKE (ASCENT preset)` | preset-dependent | ASCENT cuff library (LivaNova, MultiContact) | via the **Cuff Designer** dialog (see below) |

Intrafascicular electrodes (LIFE/TIME) are auto-positioned to the nerve cross-section centroid — or to
a chosen **target fascicle** — when the design is refit.

---

## Contact polarities & current steering

Every contact carries a **polarity** and an optional **current fraction**:

```
POLARITY_CHOICES = ("off", "anode", "cathode", "ground")
```

- **Defaults are derived from the electrode type** — e.g. bipolar → `[anode, cathode]`, tripolar →
  `[anode, cathode, anode]`, arrays → a checkerboard of cathodes/anodes.
- **Current fractions** split the total stimulus current within a polarity group. `None` means "equal
  share" (the solver uses `1/N`); explicit fractions let you specify, say, a 60/40 split between two
  cathodes. The GUI shows a per-group **sum-check chip** (green at Σ = 1).
- The FEM driver injects `I_patch = I_stim · sign(polarity) · fraction`, conserving charge
  (Σ anode = −Σ cathode = I_stim).

Because the field is solved as a **per-contact lead-field basis**, you can re-weight contacts (steer
current) and sweep amplitude without re-solving the FEM — see
[Finite-Element Solver](Finite-Element-Solver) and
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity). Legacy projects that used the
old `"active"` role are migrated to `"anode"` automatically.

---

## Fitting the cuff to the nerve

golgi places a cuff on a real, possibly curved nerve via a PCA pipeline (`cuff_fit.py`):

1. **Global PCA** aligns the nerve's long axis with +z.
2. **Cuff origin** is chosen along that axis from an **anchor** (`trunk` low-z end, `branched` high-z
   end, or `centroid`) plus an axial **offset** and small transverse nudges.
3. **Local PCA refinement** re-aligns to the nerve's *local* trajectory at that location (handling
   curvature and oblique cross-sections).
4. **Auto-sizing** sets the cuff inner radius to clear the nerve by a user margin; the outer radius
   adds the wall thickness.

`refit_design_geometry()` re-runs this whenever a design's geometry or position changes (the per-row
**Refit** button, the design-sweep generator, and headless `run_mesh` all call it). With multiple
designs on one nerve, the first design anchors the shared mesh frame and each cuff is placed in its
own local frame — so switching designs moves only the cuff.

---

## The Cuff Designer (ASCENT presets)

The **DUKE** type opens the **Cuff Designer** dialog — a pure-Python re-implementation of ASCENT's
COMSOL "part primitive" library, with a **live 3-D preview**. It builds complex, literature-accurate
cuffs from parametric primitives:

| Primitive | Role | Builds |
|---|---|---|
| `CuffFill_Primitive` | fill | solid/annular bore fill (saline / insulation) |
| `TubeCuff_Primitive` | insulator | hollow silicone shell (optional angular gap) |
| `CircleContact_Primitive` | conductor (+ recess) | curved disc contact on the inner wall |
| `LivaNova_Primitive` | insulator + conductor | helical insulator with a central conductor band |

Bundled preset families include **LivaNova** (`LN`, helical bipolar) and **MultiContact** (`MCT`,
segmented ring array). The dialog exposes only the user-editable parameters per preset (cuff radius,
wall thickness, length, contact size/recess, helix pitch/turns, contact pitch, …); everything else is
derived from a COMSOL-flavored expression DAG (`"1.109 [mm]"`, unit brackets, cross-references, math
functions). Roles are colour-coded in the preview — silicone grey, contacts gold, fill translucent
blue.

> The standalone `cuff_designer.py` module can also be used programmatically:
> `load_cuff_presets(dir)` → preset dicts, `render_design(preset, param_overrides, ns_extras)` →
> a list of `(instance, primitive, mesh, role)` tuples (PyVista meshes, SI units).

---

## From contacts to mesh and field

`electrode_patches.py` turns each electrode into two things: **renderable PolyData** (tinted by
polarity in the viewport) and **patch dictionaries** (the `electrode_config.json` the solver reads as
current boundary conditions). Ring contacts become axial/helical patches; LIFE/TIME contacts become
cylindrical bands / ribbon rectangles. These tagged facets are what the
[FEM solver](Finite-Element-Solver) injects current through.

---

## References

- Navarro X, et al. (2005). A critical review of interfaces with the peripheral nervous system for the control of neuroprostheses and hybrid bionic systems. *J Peripher Nerv Syst.* — cuff vs intrafascicular (LIFE/TIME) interface overview.
- Veraart C, Grill WM, Mortimer JT (1993). Selective control of muscle activation with a multipolar nerve cuff electrode. *IEEE Trans Biomed Eng* 40:640. — multipolar cuffs and current steering.
- Musselman ED, Cariello JE, Grill WM, Pelot NA (2021). ASCENT: a pipeline for sample-specific computational modeling of electrical stimulation of peripheral nerves. *PLOS Comput Biol.* — the cuff part-primitive library the DUKE designer reimplements.
- Bucksot JE, et al. (2019). Flat electrode contacts for vagus nerve stimulation. *PLOS ONE.* — cuff contact geometry.

---

### See also
[Meshing](Meshing) · [Finite-Element Solver](Finite-Element-Solver) ·
[Recruitment Sweeps & Selectivity](Recruitment-Sweeps-and-Selectivity) ·
[GUI Walkthrough](GUI-Walkthrough) · [Configuration Reference](Configuration-Reference)
