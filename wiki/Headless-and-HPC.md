# Headless / HPC

golgi runs without a browser for scripting, CI, and cluster work. The [`golgi.Study`](Python-API) API
is the headless surface; compute stages are dispatched through a pluggable **JobRunner** abstraction
that can run in-process, in a subprocess, or on a **SLURM** cluster.

Source: `golgi/api.py` (headless `Study`), `golgi/jobs/` (`protocol.py`, `schemas.py`,
`in_process.py`, `local_subprocess.py`, `slurm_runner.py`, `cancel.py`),
`golgi/cli.py` (`compute-worker`).

---

## Headless `Study`

`golgi.Study` drives the identical pipeline drivers the GUI uses, with no Trame/VTK. It synthesises a
local `headless` user so the auth-gated drivers run unchanged, and the
[audit](Authentication-and-Audit) writer still records every action. See [Python API](Python-API) for
the full method reference and [`examples/recruitment_sweep.py`](https://github.com/CellularSyntax/golgi/blob/main/examples/recruitment_sweep.py)
for an end-to-end script.

```python
import golgi
s = golgi.Study.create("/scratch/study_42")
s.import_nerve("nerve.stl"); s.run_mesh(); s.run_fem(); s.run_fibers()
...
```

Studies are independent on disk, so the simplest way to parallelize a parameter study is to fan out
**one process per `Study`** (each in its own project directory).

---

## Job runners

Compute requests are typed `dataclass` schemas (`golgi/jobs/schemas.py`) with `serialize()` /
`deserialize()` — e.g. `MeshConfig`, `ElectrodeConfig`/`ElectrodePatch`, `FiberSeedConfig`,
`SweepRequest`/`SweepResult`, `TetGenPayload`. A `JobRunner` takes a request, streams stdout lines to
a callback, honours a `CancelToken`, and returns `JobOutputs(return_code, outputs)`:

| Runner | Where it runs | Use |
|---|---|---|
| **InProcessRunner** | same process | tests, small jobs |
| **LocalSubprocessRunner** | child `python` process | isolates the solver; streams its log; cancellable |
| **SlurmJobRunner** | a SLURM cluster | large meshes / big sweeps |

Cancellation (`cancel.py`) sends `SIGTERM` with a hard-kill fallback (and `scancel` on SLURM), so the
GUI's **Cancel** button and headless cancellation both work cleanly.

---

## Running on SLURM

`SlurmJobRunner` serializes the request to a payload, renders an `sbatch` wrapper, submits with
`sbatch --parsable`, polls `squeue`, tails the job's `.out` file into the log callback, and (optionally)
rsyncs results back. The remote side is `golgi compute-worker <payload.json>` (see
[Command-Line Interface](Command-Line-Interface)), which dispatches on the payload's `kind`.

It is configurable (partition, account, cpus, memory, time limit, scratch root, poll interval, module
loads, sync mode). Select it via environment:

```bash
export GOLGI_FEM_RUNNER=slurm
export GOLGI_SLURM_PARTITION=cpu-long
export GOLGI_SLURM_CPUS=8 GOLGI_SLURM_MEM_GB=32 GOLGI_SLURM_TIME=04:00:00
```

### Test it without a cluster

A bundled shim fakes `sbatch`/`squeue`/`scancel` so the SLURM path can be exercised on a workstation:

```bash
export GOLGI_SBATCH="$PWD/tests/fake_sbatch.py"   # run the wrapped command locally
python tests/fake_sbatch.py cleanup               # wipe shim state
```

> **Status.** The SLURM runner, worker entry point, env dispatch, and fake-`sbatch` shim are in place;
> the per-pipeline remote integration (and FEM checkpoint-resume) is staged work — see the
> [feature roadmap](https://github.com/CellularSyntax/golgi/blob/main/FEATURES.md) (F4.2). For most
> users, fan-out over `Study` processes (or the local subprocess runner) is the path today.

---

### See also
[Python API](Python-API) · [Command-Line Interface](Command-Line-Interface) ·
[Configuration Reference](Configuration-Reference) · [Architecture](Architecture)
