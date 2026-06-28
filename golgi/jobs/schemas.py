# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Typed JSON-on-disk payload schemas shared between pipeline
drivers and compute scripts.

Step 6.2 of migration.md: the goal is that BOTH sides of every
compute boundary read the same dataclass instead of accessing a
free-form dict by key. Renaming a field then becomes a single-PR
change that the type checker (and the human reviewer) can catch.

This module avoids any heavyweight imports — numpy / pyvista /
trame stay out — so compute scripts spawned in slim service
containers can import their schema without dragging the viz
stack in.

6.2a: TetGenPayload (mesh build).
6.2b: MeshConfig + ElectrodeConfig + ElectrodePatch (FEM solve).
6.2c: FiberSeedConfig (fiber trajectory generation)."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


@dataclass
class TetGenPayload:
    """The flat dict that golgi/compute/tetgen_runner.py reads
    off sys.argv[1]. Driven by `pipeline/mesh.py:run_mesh_build`;
    consumed by `compute/tetgen_runner.py:main`. If the field
    names diverge between those two sites the build breaks at
    deserialize, which is the contract we want."""
    plc_path: Path                # input PLC file (pv.read-able)
    out_npz: Path                 # output .npz (nodes, elems, attribs)
    switches: str                 # tetgen switches, e.g. "pzAa"
    seeds: list[list]             # [[tag, [x,y,z], maxv], ...]
    verbose: int = 2
    epsilon: float = 1.0e-6
    collinear_ang_tol: float = 178.0
    facet_separate_ang_tol: float = 178.0

    def serialize(self) -> dict:
        return {
            "plc_path": str(self.plc_path),
            "out_npz": str(self.out_npz),
            "switches": str(self.switches),
            # Coerce inner seeds defensively — UI state can pass
            # tuples / numpy scalars in here. We want a stable
            # JSON shape: [[int, [float,float,float], float], …].
            "seeds": [
                [
                    int(_tag),
                    [float(_c) for _c in _seed],
                    float(_maxv),
                ]
                for _tag, _seed, _maxv in self.seeds
            ],
            "verbose": int(self.verbose),
            "epsilon": float(self.epsilon),
            "collinear_ang_tol": float(self.collinear_ang_tol),
            "facet_separate_ang_tol": float(
                self.facet_separate_ang_tol
            ),
        }

    @classmethod
    def deserialize(cls, d: dict) -> "TetGenPayload":
        return cls(
            plc_path=Path(d["plc_path"]),
            out_npz=Path(d["out_npz"]),
            switches=str(d["switches"]),
            seeds=[
                [
                    int(_tag),
                    [float(_c) for _c in _seed],
                    float(_maxv),
                ]
                for _tag, _seed, _maxv in d["seeds"]
            ],
            verbose=int(d.get("verbose", 2)),
            epsilon=float(d.get("epsilon", 1.0e-6)),
            collinear_ang_tol=float(
                d.get("collinear_ang_tol", 178.0),
            ),
            facet_separate_ang_tol=float(
                d.get("facet_separate_ang_tol", 178.0),
            ),
        )


# ---------------------------------------------------------------------------
# 6.2b — FEM solve schemas.
# ---------------------------------------------------------------------------
# Two files cross the boundary at every FEM solve:
#   * mesh_config.json     — geometry + σ values + axis-sampling extents
#   * electrode_config.json — stim current + per-contact patch list
# Pipeline driver (pipeline/fem.py) constructs both as typed dataclasses
# and writes their .serialize() form. Compute side (compute/solve_nerve.py)
# deserialises at startup, then flattens patches back to dicts so the
# existing facet-mask code that does `p["type"] == "axial"` etc. stays
# unchanged.


@dataclass
class MeshConfig:
    """mesh_config.json — solve_nerve.py reads these at startup.
    All fields are optional with sensible defaults on the consumer
    side; this dataclass just documents the shape and lets a field
    rename break at deserialize."""
    mode: str = "cylinder"            # "cylinder" | "imported"
    R_cuff_inner: Optional[float] = None
    L_cuff: Optional[float] = None
    axis_z_lo_m: Optional[float] = None
    axis_z_hi_m: Optional[float] = None
    slice_xy_half_m: Optional[float] = None
    sigma_endo: Optional[float] = None
    sigma_saline: Optional[float] = None
    sigma_silicone: Optional[float] = None
    sigma_muscle: Optional[float] = None
    sigma_epi: Optional[float] = None
    sigma_contact: Optional[float] = None
    sigma_scar: Optional[float] = None        # tag 7 (scar / encapsulation)
    # Anisotropy — longitudinal (along +z) σ of the anisotropic tissues.
    # When set, solve_nerve builds a tensor σ = diag(σ_T, σ_T, σ_L) for
    # that tag (transverse σ_T = the matching sigma_* above). Omit → the
    # solver falls back to the materials-table anisotropy.
    sigma_endo_long: Optional[float] = None
    sigma_muscle_long: Optional[float] = None
    # Perineurium contact-impedance sheet at the endo↔epi interface.
    # Active iff perineurium_ci is true; the area-specific sheet
    # resistance is Rs = peri_thk_m / sigma_peri.
    perineurium_ci: bool = False
    sigma_peri: Optional[float] = None        # perineurium bulk σ [S/m]
    peri_thk_m: Optional[float] = None        # representative thickness [m]
    perineurium_species: Optional[str] = None  # provenance, e.g. "pig"
    # Step 7.1b: solver preset baked into the mesh_config so that
    # re-running the same mesh from a checkpoint picks the right
    # solver hierarchy. CLI `--preset` on solve_nerve.py overrides
    # this; missing → solve_nerve falls back to its DEFAULT_PRESET
    # ("Balanced"). Valid values: "Quick" | "Balanced" | "HPC".
    solver_preset: Optional[str] = None

    # F3.2 — cuff placement in the SHARED canonical mesh frame.
    # Used by solve_nerve.py to transform mesh facet midpoints +
    # axis-sampling pts between canonical (mesh) and cuff-local
    # coordinates. When omitted, solve_nerve falls back to the
    # legacy single-design assumption that the mesh IS in
    # cuff-local frame (cuff at origin, axis = +z).
    #   cuff_offset_m  — 3 floats, canonical-frame cuff origin.
    #   cuff_R_flat    — 9 floats, flattened row-major 3×3
    #                    rotation such that
    #                    p_canon = p_local @ R + offset.
    cuff_offset_m: Optional[list] = None
    cuff_R_flat: Optional[list] = None

    _OPT_FLOAT_FIELDS = (
        "R_cuff_inner", "L_cuff",
        "axis_z_lo_m", "axis_z_hi_m", "slice_xy_half_m",
        "sigma_endo", "sigma_saline", "sigma_silicone",
        "sigma_muscle", "sigma_epi", "sigma_contact", "sigma_scar",
        "sigma_endo_long", "sigma_muscle_long",
        "sigma_peri", "peri_thk_m",
    )

    def serialize(self) -> dict:
        # Omit unset (None) optional fields so the on-disk JSON
        # only carries what's actually populated — matches the
        # pre-6.2b dict-construction behaviour where the driver
        # only emitted keys it had values for.
        out: dict[str, Any] = {"mode": str(self.mode)}
        for f in self._OPT_FLOAT_FIELDS:
            v = getattr(self, f)
            if v is not None:
                out[f] = float(v)
        if self.perineurium_ci:
            out["perineurium_ci"] = True
        if self.perineurium_species is not None:
            out["perineurium_species"] = str(self.perineurium_species)
        if self.solver_preset is not None:
            out["solver_preset"] = str(self.solver_preset)
        if self.cuff_offset_m is not None:
            out["cuff_offset_m"] = [
                float(v) for v in self.cuff_offset_m
            ]
        if self.cuff_R_flat is not None:
            out["cuff_R_flat"] = [
                float(v) for v in self.cuff_R_flat
            ]
        return out

    @classmethod
    def deserialize(cls, d: dict) -> "MeshConfig":
        kwargs: dict[str, Any] = {
            "mode": str(d.get("mode", "cylinder")),
        }
        for f in cls._OPT_FLOAT_FIELDS:
            if f in d and d[f] is not None:
                kwargs[f] = float(d[f])
        if d.get("perineurium_ci"):
            kwargs["perineurium_ci"] = bool(d["perineurium_ci"])
        if d.get("perineurium_species") is not None:
            kwargs["perineurium_species"] = str(d["perineurium_species"])
        if "solver_preset" in d and d["solver_preset"] is not None:
            kwargs["solver_preset"] = str(d["solver_preset"])
        if "cuff_offset_m" in d and d["cuff_offset_m"] is not None:
            kwargs["cuff_offset_m"] = [
                float(v) for v in d["cuff_offset_m"]
            ]
        if "cuff_R_flat" in d and d["cuff_R_flat"] is not None:
            kwargs["cuff_R_flat"] = [
                float(v) for v in d["cuff_R_flat"]
            ]
        return cls(**kwargs)


@dataclass
class ElectrodePatch:
    """One contact patch in `electrode_config.json`. Flat tagged-
    union: `type` ∈ {"axial", "helical"} discriminates which
    subset of the geometric fields is populated. Axial patches
    use (z, dz, phi, dphi); helical patches use (z_start, z_end,
    phi0, pitch, dphi). `R` is optional on both (defaults to the
    mesh's R_cuff_inner on the consumer side).

    `role` ∈ {"anode", "cathode", "ground", "off"} (M1). Legacy
    projects with `role == "active"` deserialise as "anode" so
    nothing on disk needs migration. Tripolar / quadripolar /
    N-polar configurations let the user assign multiple contacts
    to the same polarity; `current_fraction` (0..1) controls how
    the polarity-group's total current is split. When None, the
    FEM driver assigns an equal share (1 / N_in_group)."""
    id: int
    type: str            # "axial" | "helical"
    role: str            # "anode" | "cathode" | "ground" | "off"
    # Common.
    R: Optional[float] = None
    dphi: Optional[float] = None
    # Axial-only.
    z: Optional[float] = None
    dz: Optional[float] = None
    phi: Optional[float] = None
    # Helical-only.
    z_start: Optional[float] = None
    z_end: Optional[float] = None
    phi0: Optional[float] = None
    pitch: Optional[float] = None
    # Intrafascicular-only (LIFE "life_band" / TIME "time_rect"): wire/ribbon
    # contact centroid (x, y) + wire radius, used by the FEM nearest-facet lookup.
    x: Optional[float] = None
    y: Optional[float] = None
    R_wire: Optional[float] = None
    # M1: per-contact share of its polarity group's current.
    # Normalised to [0, 1] within each polarity group; None =
    # "equal share" (the FEM driver computes 1 / N_in_group at
    # solve time).
    current_fraction: Optional[float] = None

    _OPT_FLOAT_FIELDS = (
        "R", "dphi", "z", "dz", "phi",
        "z_start", "z_end", "phi0", "pitch",
        "x", "y", "R_wire",
        "current_fraction",
    )

    # Back-compat: legacy "active" → "cathode" (the old
    # solver's "active" was the Neumann current source =
    # stimulating contact = cathode by physiological
    # convention). "ground" stays the same; unknown strings
    # pass through so a typo is loud at FEM-driver time rather
    # than silently corrupted here.
    _LEGACY_ROLE_MAP = {
        "active": "cathode",
    }

    def serialize(self) -> dict:
        out: dict[str, Any] = {
            "id": int(self.id),
            "type": str(self.type),
            "role": str(self.role),
        }
        for f in self._OPT_FLOAT_FIELDS:
            v = getattr(self, f)
            if v is not None:
                out[f] = float(v)
        return out

    @classmethod
    def deserialize(cls, d: dict) -> "ElectrodePatch":
        raw_role = str(d["role"])
        role = cls._LEGACY_ROLE_MAP.get(raw_role, raw_role)
        kwargs: dict[str, Any] = {
            "id": int(d["id"]),
            "type": str(d["type"]),
            "role": role,
        }
        for f in cls._OPT_FLOAT_FIELDS:
            if f in d and d[f] is not None:
                kwargs[f] = float(d[f])
        return cls(**kwargs)


@dataclass
class RecordingMontage:
    """R1 — one bipolar (or future N-polar) recording montage.

    `mid` is the stable montage id ("rec_A", "rec_B", …) used to
    name lead-field files on disk (`V_e_rec_<contact_id>.npz`)
    and to key per-montage results in `geom.cnap_*`. `label` is
    the user-facing display name. `plus_contact` / `minus_contact`
    are integer contact ids matching `ElectrodePatch.id`. `kind`
    is "bipolar" in R1; later iterations may add "tripolar" /
    "quasi-tripolar" with weighted contact lists, hence the field
    rather than implicit two-contact-only shape."""
    mid: str
    label: str
    plus_contact: int
    minus_contact: int
    kind: str = "bipolar"
    color: Optional[str] = None       # hex; UI assigns from palette

    def serialize(self) -> dict:
        out: dict[str, Any] = {
            "mid": str(self.mid),
            "label": str(self.label),
            "plus_contact": int(self.plus_contact),
            "minus_contact": int(self.minus_contact),
            "kind": str(self.kind),
        }
        if self.color is not None:
            out["color"] = str(self.color)
        return out

    @classmethod
    def deserialize(cls, d: dict) -> "RecordingMontage":
        return cls(
            mid=str(d["mid"]),
            label=str(d.get("label", d["mid"])),
            plus_contact=int(d["plus_contact"]),
            minus_contact=int(d["minus_contact"]),
            kind=str(d.get("kind", "bipolar")),
            color=(
                str(d["color"])
                if d.get("color") is not None
                else None
            ),
        )


@dataclass
class ElectrodeConfig:
    """electrode_config.json — stim current + per-contact patches.

    `I_stim` is in AMPERES (not mA — the driver multiplies the
    mA UI knob by 1e-3 before constructing this).

    `recording_montages` (R1) carries the user's bipolar pairs.
    Empty by default — projects with no recording montages
    serialise without the key, and the FEM driver only triggers
    reciprocity solves when this list is non-empty."""
    name: str
    I_stim: float
    patches: list[ElectrodePatch] = field(default_factory=list)
    recording_montages: list[RecordingMontage] = field(
        default_factory=list,
    )

    def serialize(self) -> dict:
        out: dict[str, Any] = {
            "name": str(self.name),
            "I_stim": float(self.I_stim),
            "patches": [p.serialize() for p in self.patches],
        }
        if self.recording_montages:
            out["recording_montages"] = [
                m.serialize() for m in self.recording_montages
            ]
        return out

    @classmethod
    def deserialize(cls, d: dict) -> "ElectrodeConfig":
        return cls(
            name=str(d.get("name", "custom")),
            I_stim=float(d.get("I_stim", 0.0)),
            patches=[
                ElectrodePatch.deserialize(p)
                for p in d.get("patches", [])
            ],
            recording_montages=[
                RecordingMontage.deserialize(m)
                for m in d.get("recording_montages", [])
            ],
        )


# ---------------------------------------------------------------------------
# 6.2c — Fiber-trajectory seed config.
# ---------------------------------------------------------------------------
# One file crosses the boundary: nerve_paths_seed_config.json.
# Driver (pipeline/fibers.py) writes it from the Fibers-drawer state;
# compute (compute/solve_fiber_paths_nerve.py) reads it twice — early
# (cap-detection knobs) and late (streamline integration knobs).


@dataclass
class FiberSeedConfig:
    """nerve_paths_seed_config.json — knobs the fiber-paths solver
    reads at startup. Defaults match the historical hard-coded
    values on the consumer side so a missing file or missing field
    reproduces prior behaviour."""
    # Streamline integration.
    n_seeds: int = 50
    seed_end: str = "low"           # "low" | "high"
    step_um: float = 200.0
    max_steps: int = 5000
    # Cap detection / DBSCAN clustering.
    cluster_eps_m: float = 0.002
    cap_band_frac: float = 0.15
    min_rel_size: float = 0.20
    axial_normal_thresh: float = 0.7

    def serialize(self) -> dict:
        return {
            "n_seeds": int(self.n_seeds),
            "seed_end": str(self.seed_end),
            "step_um": float(self.step_um),
            "max_steps": int(self.max_steps),
            "cluster_eps_m": float(self.cluster_eps_m),
            "cap_band_frac": float(self.cap_band_frac),
            "min_rel_size": float(self.min_rel_size),
            "axial_normal_thresh": float(self.axial_normal_thresh),
        }

    @classmethod
    def deserialize(cls, d: dict) -> "FiberSeedConfig":
        return cls(
            n_seeds=int(d.get("n_seeds", 50)),
            seed_end=str(d.get("seed_end", "low")),
            step_um=float(d.get("step_um", 200.0)),
            max_steps=int(d.get("max_steps", 5000)),
            cluster_eps_m=float(d.get("cluster_eps_m", 0.002)),
            cap_band_frac=float(d.get("cap_band_frac", 0.15)),
            min_rel_size=float(d.get("min_rel_size", 0.20)),
            axial_normal_thresh=float(
                d.get("axial_normal_thresh", 0.7),
            ),
        )


# ---------------------------------------------------------------------------
# F2.1 — Parameter sweep + threshold finder.
# ---------------------------------------------------------------------------
# Two modes share the same request/result envelope:
#   * "recruitment" — sweep over a list of stim amplitudes; record
#     activation (≥1 AP fires anywhere) per (fiber, amplitude) cell.
#   * "threshold"   — per-fiber bisection over amplitude to find the
#     minimum activating amplitude (µA precision).
# The driver in `golgi/pipeline/sweep.py` consumes a SweepRequest and
# emits a SweepResult; both modes reuse the existing per-fiber sim
# pipeline (`pipeline/fiber_sim.py::_do_one_fiber`) for the underlying
# axonml / pyfibers simulation step.


@dataclass
class SweepRequest:
    """Inputs to a sweep run. F2.1 ships single-axis (amplitude) only;
    multi-axis Cartesian sweeps land in a later iteration. Fiber
    filters narrow the per-fiber set; default `None` = all fibers."""
    mode: str                            # "recruitment" | "threshold"

    # Recruitment-mode axis values. Resolved by the UI (linspace /
    # logspace expansion happens before SweepRequest construction).
    amplitudes_mA: list[float] = field(default_factory=list)

    # Threshold-mode bisection bounds + tolerance.
    bisect_lo_mA: float = 0.01
    bisect_hi_mA: float = 5.0
    bisect_tol_uA: float = 10.0          # ±10 µA threshold precision

    # Fiber filters (None = no filter on that axis).
    fiber_indices: Optional[list[int]] = None    # explicit subset
    branch_filter: Optional[int] = None          # 0/1/... or None
    fiber_type_filter: Optional[str] = None      # row label or None

    # Sim params (carried from state — same shape pipeline/fiber_sim
    # already builds via H.fiber_pulse_params()). `backend` and
    # `model_name` are the Single-fiber tab defaults; per-fiber
    # values from the Population can override them when
    # `model_source == "population"`.
    pulse_params: dict = field(default_factory=dict)
    backend: str = "pyfibers"            # "pyfibers" | "axonml"
    model_name: str = "MRG_INTERPOLATION"

    # Where to pull each fiber's model + backend from. "population"
    # = per-fiber lookup against geom.fiber_pop_types (and the
    # row's backend); falls back to single-fiber values when a
    # fiber has no population assignment. "single_fiber" = use
    # the request's backend + model_name for every fiber.
    model_source: str = "population"     # "population" | "single_fiber"

    def serialize(self) -> dict:
        out: dict[str, Any] = {
            "mode": str(self.mode),
            "amplitudes_mA": [float(a) for a in self.amplitudes_mA],
            "bisect_lo_mA": float(self.bisect_lo_mA),
            "bisect_hi_mA": float(self.bisect_hi_mA),
            "bisect_tol_uA": float(self.bisect_tol_uA),
            "pulse_params": dict(self.pulse_params),
            "backend": str(self.backend),
            "model_name": str(self.model_name),
            "model_source": str(self.model_source),
        }
        if self.fiber_indices is not None:
            out["fiber_indices"] = [int(i) for i in self.fiber_indices]
        if self.branch_filter is not None:
            out["branch_filter"] = int(self.branch_filter)
        if self.fiber_type_filter is not None:
            out["fiber_type_filter"] = str(self.fiber_type_filter)
        return out

    @classmethod
    def deserialize(cls, d: dict) -> "SweepRequest":
        return cls(
            mode=str(d.get("mode", "recruitment")),
            amplitudes_mA=[
                float(a) for a in d.get("amplitudes_mA", [])
            ],
            bisect_lo_mA=float(d.get("bisect_lo_mA", 0.01)),
            bisect_hi_mA=float(d.get("bisect_hi_mA", 5.0)),
            bisect_tol_uA=float(d.get("bisect_tol_uA", 10.0)),
            fiber_indices=(
                [int(i) for i in d["fiber_indices"]]
                if "fiber_indices" in d and d["fiber_indices"] is not None
                else None
            ),
            branch_filter=(
                int(d["branch_filter"])
                if "branch_filter" in d and d["branch_filter"] is not None
                else None
            ),
            fiber_type_filter=(
                str(d["fiber_type_filter"])
                if "fiber_type_filter" in d
                and d["fiber_type_filter"] is not None
                else None
            ),
            pulse_params=dict(d.get("pulse_params", {})),
            backend=str(d.get("backend", "pyfibers")),
            model_name=str(d.get("model_name", "MRG_INTERPOLATION")),
            model_source=str(d.get("model_source", "population")),
        )


@dataclass
class SweepResult:
    """Output envelope for both sweep modes. Numpy-array fields
    serialise cleanly via `np.savez_compressed`; the small scalar
    fields go in a sibling JSON manifest.

    Recruitment-mode populates: amplitudes_mA, activated.
    Threshold-mode populates:   thresholds_uA, bisect_iters.
    Both populate the metadata block (fiber_indices, diameters, …)."""
    request: SweepRequest

    # Per-fiber metadata (in the swept order; length n_fibers).
    fiber_indices: np.ndarray = field(  # type: ignore[name-defined]
        default_factory=lambda: _np_empty_i64(),
    )
    fiber_diameters_um: np.ndarray = field(
        default_factory=lambda: _np_empty_f64(),
    )
    fiber_branch_idx: np.ndarray = field(
        default_factory=lambda: _np_empty_i32(),
    )
    fiber_type_labels: list[str] = field(default_factory=list)

    # Recruitment-mode payload. shape: (n_fibers, n_amplitudes).
    activated: Optional[np.ndarray] = None

    # Threshold-mode payload. shape: (n_fibers,) — NaN where the
    # fiber didn't activate inside [bisect_lo, bisect_hi].
    thresholds_uA: Optional[np.ndarray] = None
    bisect_iters: Optional[np.ndarray] = None

    # Run metadata.
    elapsed_s: float = 0.0
    sha: str = ""                        # cache key, see pipeline/sweep
    n_sims_total: int = 0                # for progress + cost reporting

    def to_npz_payload(self) -> dict:
        """Pack into a flat dict ready for np.savez_compressed.
        Tiny scalars / strings co-travel; the consumer reconstructs
        via from_npz_payload."""
        out: dict[str, Any] = {
            "fiber_indices": np.asarray(self.fiber_indices,
                                          dtype=np.int64),
            "fiber_diameters_um": np.asarray(
                self.fiber_diameters_um, dtype=np.float64,
            ),
            "fiber_branch_idx": np.asarray(
                self.fiber_branch_idx, dtype=np.int32,
            ),
            "fiber_type_labels": np.asarray(
                self.fiber_type_labels, dtype=object,
            ),
            "elapsed_s": np.float64(self.elapsed_s),
            "sha": np.asarray(self.sha),
            "n_sims_total": np.int64(self.n_sims_total),
        }
        if self.activated is not None:
            out["activated"] = np.asarray(
                self.activated, dtype=np.bool_,
            )
        if self.thresholds_uA is not None:
            out["thresholds_uA"] = np.asarray(
                self.thresholds_uA, dtype=np.float64,
            )
        if self.bisect_iters is not None:
            out["bisect_iters"] = np.asarray(
                self.bisect_iters, dtype=np.int32,
            )
        return out


# Forward-import for the SweepResult numpy defaults. np itself is
# the only heavy import this schemas module carries, but we keep it
# isolated to these factory functions so the rest of the module
# stays importable in the slim compute-side environment.
try:
    import numpy as np  # noqa: E402
except ImportError:                                          # pragma: no cover
    np = None  # type: ignore[assignment]


def _np_empty_i64():
    return np.empty(0, dtype=np.int64) if np is not None else []


def _np_empty_f64():
    return np.empty(0, dtype=np.float64) if np is not None else []


def _np_empty_i32():
    return np.empty(0, dtype=np.int32) if np is not None else []
