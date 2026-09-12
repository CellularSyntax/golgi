# tests/

```bash
pytest -q                    # tier 1 (seconds, no solver stack needed)
pytest -q -m integration     # tiers 2 + 3 (full FEniCSx/NEURON stack, 10–30 min)
```

| Tier | File | Needs | What it checks |
|---|---|---|---|
| 1 | `test_headless_api.py` (tiers 1–2) | pure Python | `golgi.Study` exposes every compute method un-stubbed; project create/open lifecycle |
| 1 | `test_bundle_integrity.py` | pure Python | bundle export writes a MANIFEST with SHA-256 per file + stage DAG; export is deterministic; `replay_study` passes on an intact bundle and pinpoints a single flipped byte; import round-trip |
| 1 | `test_cli.py` | pure Python | `golgi export / import / replay [--json]` exit codes and output |
| 1 | `test_recording_cable.py` | numpy | cable-equation current conservation for the recording model |
| 2 | `test_headless_api.py::test_end_to_end_pipeline` | FEniCSx, Gmsh/TetGen, NEURON/PyFibers | `import_nerve → run_mesh → run_fibers → run_fem → run_sweep → export_bundle` on a synthetic capsule nerve; every stage's artifacts exist |
| 3 | `test_e2e_reference.py` | as tier 2 | the 12-fiber reference study of `examples/recruitment_sweep.py` reproduces the recorded activation thresholds (`reference/e2e_synthetic_reference.json`) within tolerance and its bundle verifies |

Tier 2–3 tests auto-skip when the solver stack is absent. CI (`.github/workflows/ci.yml`) runs tier 1 on
Linux and macOS, tiers 2–3 on Linux inside the pinned conda environment, and the whole suite once more
inside the Docker image. GUI code (Trame/VTK) is exercised through the shared `golgi.Study` state and
the pipeline drivers it calls; the browser layer itself is not covered by automated tests.

---

Test scaffolding + shims. The actual pytest suite is sparse for
the GUI app (Trame + VTK + FEniCSx make CI heavy) — this dir
ships **dev / smoke helpers** that let you exercise the
trickier code paths without a full cluster or solver install.

## `fake_sbatch.py` (F4.2)

Python shim that pretends to be the SLURM `sbatch` /
`squeue` / `scancel` trio. Lets `SlurmJobRunner` be exercised
end-to-end on a workstation:

```bash
# Smoke a noop SLURM submission via the fake shim:
mkdir -p /tmp/golgi_slurm_smoke && cd /tmp/golgi_slurm_smoke
cat > payload.json <<'EOF'
{"kind": "noop"}
EOF

GOLGI_SBATCH=/path/to/golgi/tests/fake_sbatch.py \
python -c "
from pathlib import Path
from golgi.jobs import SlurmJobRunner, CancelToken, JobRequest
import dataclasses

@dataclasses.dataclass
class _NoopReq(JobRequest):
    out_dir: Path = Path.cwd()
    kind: str = 'noop'

runner = SlurmJobRunner(
    partition='dev', cpus=1, memory_gb=1, time_limit='00:05:00',
)
result = runner.run(
    _NoopReq(),
    on_line=lambda s: print('  >', s),
    cancel=CancelToken(),
)
print('exit:', result.return_code)
"
```

What `fake_sbatch.py` simulates:

- `sbatch --parsable <wrapper.sh>` → spawns the wrapper as a
  background process, returns a fake job id on stdout.
- `squeue -j <fakeid> -h -o %T` → reads the fake state file
  written by the wrapper exit hook (RUNNING while running,
  COMPLETED on exit 0, FAILED otherwise).
- `scancel <fakeid>` → SIGTERM the wrapper pid.

Scratch state lives under `$TMPDIR/golgi_fake_sbatch/` so
concurrent test runs don't collide. Wipe between runs with:

```bash
python tests/fake_sbatch.py cleanup
```

## Real-cluster integration

Once you're on a node with real `sbatch`, drop the
`GOLGI_SBATCH` override and the runner picks up the real
binary. The `resolve_fem_runner()` helper in
`golgi/jobs/__init__.py` reads `GOLGI_FEM_RUNNER` to dispatch
between the local subprocess runner and `SlurmJobRunner`:

```bash
export GOLGI_FEM_RUNNER=slurm
export GOLGI_SLURM_PARTITION=cpu-long
export GOLGI_SLURM_ACCOUNT=vagusgrant
export GOLGI_SLURM_CPUS=8
export GOLGI_SLURM_MEM_GB=32
export GOLGI_SLURM_TIME=04:00:00
export GOLGI_SLURM_SCRATCH=/scratch/$USER/golgi
export GOLGI_SLURM_MODULES=fenicsx/0.7:openmpi/4.1
```

Then run the GUI as normal — every FEM solve dispatches to the
cluster instead of running locally.

> ⚠️ **F4.2 Phase A** ships the SlurmJobRunner + env-var
> dispatch surface. The actual `pipeline/fem.py` wiring that
> calls `resolve_fem_runner()` instead of constructing the
> local FEMRunner directly lands in **Phase B**, alongside the
> per-band checkpoint-resume work (the SLURM path needs Phase
> B's output-collection logic).
