#!/usr/bin/env python
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.
"""Per-stage wall-time and peak-memory benchmark of the headless pipeline.

Runs the same reference study as ``examples/recruitment_sweep.py`` (a
synthetic monofascicular nerve with a bipolar ring cuff) and records, for
every pipeline stage, the wall-clock time and the peak resident set size of
the whole process tree (golgi runs the TetGen/Gmsh mesher, the FEniCSx field
solve and the NEURON fiber simulations in worker subprocesses, so the parent
process alone would under-report memory).

    python examples/benchmark.py                      # 12-fiber quick run
    python examples/benchmark.py --fibers 50 --out bench_50.json

The report (JSON, plus a Markdown table on stdout) contains the hardware and
software versions it was produced with, so numbers from different machines
or containers can be compared side by side.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import threading
import time
from pathlib import Path

import psutil

sys.path.insert(0, str(Path(__file__).resolve().parent))
from recruitment_sweep import NERVE_LENGTH_MM, make_synthetic_nerve  # noqa: E402

import golgi  # noqa: E402


# ---------------------------------------------------------------------------
# Process-tree memory sampler
# ---------------------------------------------------------------------------

class TreeMemorySampler:
    """Background thread sampling the summed RSS of this process and all
    of its live descendants every ``interval`` seconds."""

    def __init__(self, interval: float = 0.2) -> None:
        self._interval = interval
        self._proc = psutil.Process(os.getpid())
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_rss_bytes = 0
        self._stage_peak = 0
        self._lock = threading.Lock()

    # Sandboxed environments may forbid process enumeration; then fall
    # back to getrusage(): own RSS + the largest terminated child's peak.
    _use_rusage = False

    def _sample(self) -> int:
        if not self._use_rusage:
            try:
                procs = [self._proc] + self._proc.children(recursive=True)
                total = 0
                for p in procs:
                    try:
                        total += p.memory_info().rss
                    except psutil.Error:
                        pass
                return total
            except (psutil.Error, PermissionError, OSError):
                self._use_rusage = True
                self.method = "rusage (self RSS + max terminated child RSS)"
        import resource
        own = self._proc.memory_info().rss
        ch = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        if sys.platform != "darwin":
            ch *= 1024          # Linux reports kB, macOS bytes
        return own + ch

    method = "psutil process tree"

    def _run(self) -> None:
        while not self._stop.is_set():
            rss = self._sample()
            with self._lock:
                self.peak_rss_bytes = max(self.peak_rss_bytes, rss)
                self._stage_peak = max(self._stage_peak, rss)
            self._stop.wait(self._interval)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)

    def reset_stage(self) -> None:
        with self._lock:
            self._stage_peak = self._sample()

    def stage_peak(self) -> int:
        with self._lock:
            return self._stage_peak


# ---------------------------------------------------------------------------
# Environment description
# ---------------------------------------------------------------------------

def _neuron_version() -> str:
    try:
        import neuron
        return str(neuron.__version__)
    except Exception:  # noqa: BLE001
        return _pkg_version("neuron")


def _pkg_version(name: str) -> str:
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:  # noqa: BLE001
        return "n/a"


def describe_environment() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        # Linux: a readable CPU model name from /proc/cpuinfo.
        with open("/proc/cpuinfo") as fh:
            for line in fh:
                if line.lower().startswith("model name"):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    if sys.platform == "darwin":
        try:
            import subprocess
            cpu = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:  # noqa: BLE001
            pass
    in_container = Path("/.dockerenv").exists() or bool(
        os.environ.get("GOLGI_IN_CONTAINER"))
    return {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "cpu_model": cpu,
        "cpu_count_logical": psutil.cpu_count(logical=True),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "ram_total_gb": round(psutil.virtual_memory().total / 2**30, 1),
        "container": in_container,
        "golgi": _pkg_version("golgi"),
        "fenics-dolfinx": _pkg_version("fenics-dolfinx"),
        "petsc4py": _pkg_version("petsc4py"),
        "gmsh": _pkg_version("gmsh"),
        "tetgen": _pkg_version("tetgen"),
        "neuron": _neuron_version(),
        "pyfibers": _pkg_version("pyfibers"),
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "pyvista": _pkg_version("pyvista"),
        "trame": _pkg_version("trame"),
    }


# ---------------------------------------------------------------------------
# Benchmark
# ---------------------------------------------------------------------------

def _count_msh_elements(msh_path: Path) -> dict:
    """Number of tetrahedra (and nodes) in a Gmsh .msh, by physical region."""
    try:
        import meshio
        m = meshio.read(msh_path)
        n_tets = sum(int(len(c.data)) for c in m.cells if c.type == "tetra")
        per_region: dict[str, int] = {}
        tags = m.cell_data.get("gmsh:physical")
        if tags is not None:
            import numpy as np
            for c, t in zip(m.cells, tags):
                if c.type != "tetra":
                    continue
                for tag, cnt in zip(*np.unique(t, return_counts=True)):
                    name = str(int(tag))
                    for k, v in (m.field_data or {}).items():
                        if int(v[0]) == int(tag):
                            name = k
                    per_region[name] = per_region.get(name, 0) + int(cnt)
        return {"n_nodes": int(len(m.points)), "n_tets": n_tets,
                "tets_per_region": per_region}
    except Exception as ex:  # noqa: BLE001
        return {"error": f"{type(ex).__name__}: {ex}"}


def run_benchmark(project_dir: Path, n_fibers: int) -> dict:
    from golgi.jobs.schemas import SweepRequest

    if project_dir.exists():
        shutil.rmtree(project_dir)

    sampler = TreeMemorySampler()
    sampler.start()
    stages: list[dict] = []

    def timed(name: str, fn):
        sampler.reset_stage()
        t0 = time.perf_counter()
        out = fn()
        dt = time.perf_counter() - t0
        stages.append({
            "stage": name,
            "wall_s": round(dt, 2),
            "peak_rss_gb": round(sampler.stage_peak() / 2**30, 2),
        })
        print(f"  {name:<28s} {dt:8.1f} s   "
              f"peak RSS {sampler.stage_peak() / 2**30:5.2f} GB", flush=True)
        return out

    s = golgi.Study.create(project_dir)
    stl = project_dir / "synthetic_nerve.stl"
    make_synthetic_nerve(stl)
    info = timed("import_nerve", lambda: s.import_nerve(stl))

    s.set_mesh(use_epi=True, epi_thickness_um=50, lc_endo_um=200,
               lc_epi_um=150, lc_muscle_um=1000, lc_saline_um=150,
               lc_silicone_um=300, lc_contact_um=100, lc_scar_um=150,
               muscle_radial_pad_mm=5, muscle_axial_pad_mm=10)
    s.set_electrodes([{"eid": "elec_01", "name": "Bipolar cuff",
                       "cuff_offset_mm": NERVE_LENGTH_MM / 2.0,
                       "electrode_type": "bipolar ring-pair"}])
    meshes = timed("run_mesh (TetGen/Gmsh)", s.run_mesh)
    mesh_stats = {eid: _count_msh_elements(p) for eid, p in meshes.items()}

    s.set_fiber_seed(n_fibers=n_fibers, fiber_auto_detect_branches=True)
    fib = timed("run_fibers (trajectories)", s.run_fibers)
    timed("run_fem (FEniCSx lead fields)", s.run_fem)
    result = timed(
        f"run_sweep ({n_fibers} NEURON thresholds)",
        lambda: s.run_sweep(SweepRequest(
            mode="threshold", bisect_lo_mA=0.01, bisect_hi_mA=10.0,
            bisect_tol_uA=10.0, model_source="single_fiber")),
    )
    out_zip = project_dir.parent / f"{project_dir.name}_study.zip"
    timed("export_bundle", lambda: s.export_bundle(out_zip))
    s.close()
    sampler.stop()

    import numpy as np
    thr = np.asarray(result.thresholds_uA, dtype=float)
    thr_ok = thr[np.isfinite(thr) & (thr > 0)]
    total = sum(st["wall_s"] for st in stages)
    return {
        "environment": describe_environment(),
        "study": {
            "geometry": "synthetic cylindrical monofascicular nerve, "
                        f"r = 1 mm, L = {NERVE_LENGTH_MM:g} mm, "
                        "50 µm epineurium, bipolar ring-pair cuff",
            "nerve_surface": {k: info.get(k) for k in
                              ("n_pts", "n_tris", "watertight")},
            "mesh": mesh_stats,
            "fibers": {"n_requested": n_fibers,
                       "n_paths": fib.get("n_paths"),
                       "n_branches": fib.get("n_branches")},
            "thresholds_uA": {
                "n_fibers": int(thr.size),
                "n_activated": int(thr_ok.size),
                "median": float(np.median(thr_ok)) if thr_ok.size else None,
                "min": float(thr_ok.min()) if thr_ok.size else None,
                "max": float(thr_ok.max()) if thr_ok.size else None,
            },
            "bundle_bytes": out_zip.stat().st_size if out_zip.exists() else None,
        },
        "stages": stages,
        "total_wall_s": round(total, 1),
        "peak_rss_gb_overall": round(sampler.peak_rss_bytes / 2**30, 2),
        "memory_method": sampler.method,
    }


def markdown_table(rep: dict) -> str:
    env = rep["environment"]
    lines = [
        f"Machine: {env['cpu_model']} ({env['cpu_count_physical']} cores, "
        f"{env['ram_total_gb']} GB RAM), {env['platform']}"
        + (" [container]" if env["container"] else ""),
        "",
        "| Stage | Wall time (s) | Peak RSS (GB) |",
        "|---|---:|---:|",
    ]
    for st in rep["stages"]:
        lines.append(f"| {st['stage']} | {st['wall_s']:.1f} | "
                     f"{st['peak_rss_gb']:.2f} |")
    lines.append(f"| **Total** | **{rep['total_wall_s']:.1f}** | "
                 f"**{rep['peak_rss_gb_overall']:.2f}** |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fibers", type=int, default=12)
    ap.add_argument("--project", type=Path,
                    default=Path.cwd() / "golgi_benchmark_project")
    ap.add_argument("--out", type=Path, default=Path.cwd() / "benchmark.json")
    ap.add_argument("--label", default="", help="free-text tag for this run")
    args = ap.parse_args(argv)

    print(f"golgi benchmark — {args.fibers} fibers", flush=True)
    rep = run_benchmark(args.project, args.fibers)
    rep["label"] = args.label
    rep["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print()
    print(markdown_table(rep))
    print(f"\nreport written to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
