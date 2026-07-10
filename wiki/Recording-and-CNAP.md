# Recording & CNAP

Beyond stimulation, golgi can predict what a cuff **records**: the compound nerve action potential
(CNAP) evoked on the fiber population, using a reciprocity-based forward model. This closes the loop
for studies of evoked CNAP / ECAP biomarkers.

Source: `golgi/pipeline/recording.py`, with FEM lead fields from `golgi/compute/solve_nerve.py`.
Unit-tested in `tests/test_recording_cable.py` (no solver stack required).

---

## The forward model

The recorded potential at a contact is computed by **reciprocity**: the contribution of each fiber is
its transmembrane current weighted by the contact's lead field along the fiber, integrated over the
fiber and summed over the population.

1. **Lead fields.** The [FEM solver](Finite-Element-Solver) emits a per-contact unit-current lead
   field `Vₑᴿ(s)` sampled on the fiber nodes (one `V_e_rec_<id>.npz` per recording contact). Enable
   with `GOLGI_EMIT_RECORDING=1`.
2. **Transmembrane current.** From each fiber's membrane potential `Vₘ(s,t)` (from the
   [activation](Fiber-Models-and-Activation) simulation), golgi computes the cable transmembrane
   current

   $$I_m(n,t) = \frac{\pi d^2}{4\,\rho_a\,\Delta x}\,\big(V_m(n{+}1) - 2V_m(n) + V_m(n{-}1)\big)$$

   with axoplasmic resistivity `ρ_a ≈ 0.7 Ω·m`.
3. **Single-fiber contribution.** Interpolate the lead field onto the fiber's nodes and sum
   `φ(t) = Σ_n I_m(n,t) · (Vₑᴿ⁺ − Vₑᴿ⁻)(n)` for a bipolar recording montage (plus/minus contacts).
4. **Population CNAP.** Sum over all fibers (optionally decomposed by fiber type), giving the
   compound response.

A **recording montage** (`RecordingMontage`) names the plus/minus contacts; any bipolar/multipolar
montage is a linear combination of the cached single-contact lead fields, so montages are cheap to
evaluate once the basis is solved.

---

## What's verified

The cable-current and summation math is covered by pure-numpy unit tests:

- zero current for a spatially uniform `Vₘ`,
- constant second difference for a quadratic `Vₘ`,
- current conservation along the fiber,
- correct lead-field subtraction for a bipolar montage,
- population sum / type decomposition, and shape validation.

Run them with `pytest tests/test_recording_cable.py -q` (no FEniCSx needed).

---

## References

- McIntyre CC, Richardson AG, Grill WM (2002). Modeling the excitability of mammalian nerve fibers: influence of afterpotentials on the recovery cycle. *J Neurophysiol* 87:995. — the cable model supplying the transmembrane currents.
- Lubba CH, et al. (2019). PyPNS: multiscale simulation of a peripheral nerve in Python. *Neuroinformatics.* — related compound-action-potential modeling.

The reciprocity (lead-field) approach is a standard volume-conductor result for recording; see the
[Finite-Element Solver](Finite-Element-Solver) page for how the per-contact lead fields are computed.

---

### See also
[Finite-Element Solver](Finite-Element-Solver) · [Fiber Models & Activation](Fiber-Models-and-Activation) ·
[Electrodes & Cuff Designer](Electrodes-and-Cuff-Designer) · [Validation](Validation)
