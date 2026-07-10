# Conductivity & Tissue Properties

The accuracy of the extracellular field (and therefore of every threshold and selectivity number)
rests on the tissue conductivities. golgi ships a physically grounded, frequency-aware, **anisotropic**
material model with literature-backed defaults that are fully editable in the
[Materials drawer](GUI-Walkthrough).

Source: `golgi/conductivity/` — `materials.py` (canonical σ table), `cole_cole.py` (frequency
dispersion), `itis_db.py` (IT'IS database), `perineurium.py` (perineurium morphology).

---

## Tissue regions

The model uses tagged regions in the tetrahedral mesh. The default conductivities (S/m), drawn from
the peripheral-nerve modeling literature, are:

| Tag | Region | σ (transverse) | Anisotropy | Source |
|---|---|---|---|---|
| 1 | **Endoneurium** | ≈0.167 (1/6) | longitudinal ≈0.571 (1/1.75) | fascicle interior — strongly anisotropic (Ranck & BeMent 1965; Pelot et al. 2019) |
| 2 | **Saline** | 1.76 | isotropic | cuff fill (Pelot et al. 2019) |
| 3 | **Silicone** | ≈1e-12 | isotropic | cuff body — near-insulator (Pelot et al. 2019) |
| 4 | **Muscle** | 0.086 | longitudinal 0.35 | far-field tissue — anisotropic (Gielen et al. 1984; Pelot et al. 2017) |
| 5 | **Epineurium** | ≈0.159 (1/6.3) | isotropic | outer nerve sheath (Grill & Mortimer 1994; Pelot et al. 2019) |
| 6 / 8 | **Contact metal** | ≈9.43e6 | isotropic | platinum bulk, equipotential (De Podesta 2001) |
| 7 | **Encapsulation / scar** | ≈0.159 (1/6.3) | isotropic | fibrotic shell around the implant (Grill & Mortimer 1994) |
| — | **Perineurium** | ≈1/1149 (8.7e-4) | sheet | contact-impedance sheet, not a meshed volume — see below (Weerasuriya et al. 1984; Pelot et al. 2019) |

These follow the finite-element conductivity conventions compiled for peripheral nerve by Pelot et al.
(2019) and used in the ASCENT pipeline (Musselman et al. 2021); full citations are in the
[References](#references) below. Every value is overridable per project (Materials drawer or
`MeshConfig`/state). Anisotropy is represented as a per-cell diagonal tensor `σ = diag(σ_T, σ_T, σ_L)`
with the longitudinal axis along the nerve, so the endoneurium and muscle conduct preferentially along
the fiber direction.

> Available electrode-material presets in the GUI include Au, Pt, Pt-Ir, stainless steel, Ti, TiN,
> IrOx, and a perfect-conductor option.

---

## Frequency dependence: the Cole–Cole model

Tissue conductivity is frequency dependent. golgi evaluates a **Cole–Cole dispersion model**
(Gabriel et al. 1996):

$$
\sigma(\omega) = \sigma_{\text{ionic}} + \omega\,\varepsilon_0\,\varepsilon''(\omega),
\qquad
\varepsilon^{*}(\omega) = \varepsilon_\infty + \sum_i \frac{\Delta\varepsilon_i}{1+(j\omega\tau_i)^{1-\alpha_i}}
$$

The evaluator (`cole_cole_sigma`) accepts an arbitrary list of dispersion terms. The
[IT'IS database](#itis-material-database) supplies up to **four** dispersions per tissue (the full
Gabriel et al. 1996 parametric fit); the in-app Cole–Cole dialog uses a **three-term** fit as the
default working model. The **Cole–Cole dialog** (opened per tissue in the Materials drawer) lets you pick a tissue,
evaluate σ(f) across the spectrum, and commit a value at your stimulation frequency.

---

## IT'IS material database

golgi bundles the **IT'IS Foundation tissue database, version 4.2** (Hasgall et al.) — Gabriel-model
parameters (Δε₁–₄, τ₁–₄, α₁–₄, σ_ionic) for hundreds of tissues. A curated subset (nerve, muscle, fat,
blood, CNS tissues, bone, skin, …) is surfaced in the UI dropdowns, and `itis_sigma_at(tissue,
freq_hz)` returns the Cole–Cole σ at any frequency. Standard preset frequencies are 100 Hz, 1 kHz,
10 kHz, 100 kHz, and 1 MHz. The Nerve entry is included as a peripheral-nerve-specific addition over
the generic Gabriel et al. 1996 fits.

---

## Perineurium contact impedance

The perineurium is thin (microns) but highly resistive, and its trans-sheath voltage drop dominates
how much of the applied field reaches the axons inside a fascicle. Meshing it as a volume would force
µm-scale refinement, so golgi models it as a **contact-impedance (Robin) sheet** — ASCENT's
thin-layer approximation (Musselman et al. 2021):

- **Area-specific sheet resistance** `Rs = peri_thickness / σ_peri`, with σ_peri ≈ 1/1149 S/m
  (Weerasuriya et al. 1984; adopted by Pelot et al. 2019).
- **Thickness as a function of fascicle diameter**, `peri_thk(d) = slope·d + intercept`, with
  species-specific coefficients (rat / pig / human) after Pelot et al. 2019, with vagus morphology
  refined in Pelot et al. 2020.
- A **two-field formulation** (`solve_nerve_ci.py`) couples the endoneurium potential and the
  surrounding potential across the endo↔epi interface through the sheet conductance `g = 1/Rs`.

Enable it with `GOLGI_RECIP_CI=1` plus `perineurium_ci=True` and a positive thickness. It is **off by
default** (single-field solve). Turning CI on attenuates the field reaching the axons and therefore
raises thresholds and reshapes selectivity — it is the physically correct choice when fascicular
voltage gradients matter, at ~10–20% extra solve cost. See the
[Finite-Element Solver](Finite-Element-Solver) page for the block-form details.

---

## References

The conductivity and perineurium references below are those used in the golgi methods paper (and
recorded in `golgi/conductivity/materials.py` and `perineurium.py`):

- De Podesta M (2001). *Understanding the Properties of Matter.* Taylor & Francis. — platinum bulk conductivity.
- Gabriel S, Lau RW, Gabriel C (1996). The dielectric properties of biological tissues: III. Parametric models for the dielectric spectrum of tissues. *Phys Med Biol* 41:2271. — Cole–Cole / IT'IS dispersion model.
- Gielen FLH, Wallinga-de Jonge W, Boon KL (1984). Electrical conductivity of skeletal muscle tissue: experimental results from different muscles in vivo. *Med Biol Eng Comput* 22:569. — muscle σ.
- Grill WM, Mortimer JT (1994). Electrical properties of implant encapsulation tissue. *Ann Biomed Eng* 22:23. — encapsulation/scar (and epineurium).
- Hasgall PA, Di Gennaro F, Baumgartner C, Neufeld E, Lloyd B, Gosselin MC, Payne D, Klingenböck A, Kuster N. *IT'IS Database for thermal and electromagnetic parameters of biological tissues,* Version 4.2. IT'IS Foundation. https://itis.swiss/database
- Musselman ED, Cariello JE, Grill WM, Pelot NA (2021). ASCENT (Automated Simulations to Characterize Electrical Nerve Thresholds): a pipeline for sample-specific computational modeling of electrical stimulation of peripheral nerves. *PLOS Comput Biol.* — thin-layer perineurium approximation.
- Pelot NA, Behrend CE, Grill WM (2017). Modeling the response of small myelinated axons in a compound nerve to kilohertz frequency signals. *J Neural Eng* 14:046022. — muscle anisotropy.
- Pelot NA, Thio BJ, Grill WM (2019). On the parameters used in finite element modeling of compound peripheral nerves. *J Neural Eng* 16:016007. — adopted FE conductivities (endo/epi/saline/silicone/perineurium).
- Pelot NA, Goldhagen GB, Cariello JE, Musselman ED, Clissold KA, Ezzell JA, Grill WM (2020). Quantified morphology of the cervical and subdiaphragmatic vagus nerves of human, pig, and rat. *Front Neurosci* 14:1148. — perineurium thickness vs fascicle diameter.
- Ranck JB, BeMent SL (1965). The specific impedance of the dorsal columns of cat: an anisotropic medium. *Exp Neurol* 11:451. — endoneurium anisotropy.
- Weerasuriya A, Spangler RA, Rapoport SI, Taylor RE (1984). AC impedance of the perineurium of the frog sciatic nerve. *Biophys J* 46:167. — perineurium conductivity.

---

### See also
[Finite-Element Solver](Finite-Element-Solver) · [Meshing](Meshing) ·
[Configuration Reference](Configuration-Reference) · [Validation](Validation)
