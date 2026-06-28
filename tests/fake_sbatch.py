#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F4.2 — fake_sbatch shim for testing `SlurmJobRunner` without
a real SLURM install.

Pretends to be the `sbatch` binary: parses `--parsable`, extracts
the wrapper script path, runs it locally, and writes a
`slurm-<fakeid>.out` file the runner can tail. Returns the fake
job id on stdout so `SlurmJobRunner._submit` parses it the same
way it would a real SLURM response.

## Usage

Drop in as the `GOLGI_SBATCH` env var:

    GOLGI_SBATCH=$(pwd)/tests/fake_sbatch.py \\
        python -m golgi.cli compute-worker payload.json

Or wire it into a SlurmJobRunner explicitly:

    runner = SlurmJobRunner(
        ...,
        sbatch_path=str(Path("tests/fake_sbatch.py").resolve()),
    )

## What this fakes

  * `sbatch --parsable <wrapper.sh>` → spawns the wrapper as a
    background process, returns "<fakeid>" on stdout.
  * `squeue -j <fakeid> -h -o %T` → reads the fake state file
    written by the wrapper exit hook ("RUNNING" while running,
    "COMPLETED" on exit code 0, "FAILED" otherwise).
  * `scancel <fakeid>` → SIGTERM the wrapper pid.

The shim writes its scratch state under `${TMPDIR}/golgi_fake_sbatch/`
so concurrent tests don't collide.
"""
from __future__ import annotations

import os
import shutil
import signal
import subprocess
import sys
import tempfile
from pathlib import Path


_STATE_DIR = (
    Path(os.environ.get("TMPDIR", "/tmp"))
    / "golgi_fake_sbatch"
)
_STATE_DIR.mkdir(parents=True, exist_ok=True)


def _next_fake_id() -> int:
    """Monotonic fake job id, written to a counter file so
    successive `sbatch` calls in the same test run get
    different ids."""
    counter = _STATE_DIR / "next_id.txt"
    try:
        n = int(counter.read_text().strip())
    except (OSError, ValueError):
        n = 1_000_000
    counter.write_text(str(n + 1))
    return n


def _state_file(job_id: int) -> Path:
    return _STATE_DIR / f"state_{job_id}.txt"


def _pid_file(job_id: int) -> Path:
    return _STATE_DIR / f"pid_{job_id}.txt"


def _exit_file(job_id: int) -> Path:
    return _STATE_DIR / f"exit_{job_id}.txt"


def cmd_sbatch(argv: list[str]) -> int:
    """sbatch --parsable <wrapper.sh>"""
    # Real sbatch accepts many flags; we only care about
    # extracting the wrapper path.
    parsable = False
    wrapper: str | None = None
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "--parsable":
            parsable = True
            i += 1
            continue
        if a.startswith("-"):
            # Skip flag-with-value or just a flag.
            if "=" not in a and i + 1 < len(argv):
                i += 2
            else:
                i += 1
            continue
        wrapper = a
        i += 1
        break
    if wrapper is None:
        print(
            "fake_sbatch: missing wrapper script", file=sys.stderr,
        )
        return 1

    job_id = _next_fake_id()
    _state_file(job_id).write_text("RUNNING")

    # Parse the wrapper for --output= so the spawned process
    # writes its stdout where the real one would.
    out_file: Path | None = None
    try:
        wrapper_text = Path(wrapper).read_text()
        for line in wrapper_text.splitlines():
            if line.startswith("#SBATCH --output="):
                spec = line.split("=", 1)[1].strip()
                spec = spec.replace("%j", str(job_id))
                out_file = Path(spec)
                break
    except OSError:
        pass

    if out_file is not None:
        out_file.parent.mkdir(parents=True, exist_ok=True)
        log_handle = open(out_file, "w")
    else:
        log_handle = subprocess.DEVNULL

    # Spawn the wrapper. Detach with start_new_session so the
    # parent (this fake_sbatch) can exit immediately, matching
    # real sbatch behaviour.
    try:
        proc = subprocess.Popen(
            ["/bin/bash", str(wrapper)],
            stdout=log_handle, stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    except OSError as ex:
        if log_handle is not subprocess.DEVNULL:
            log_handle.close()
        _state_file(job_id).write_text("FAILED")
        print(
            f"fake_sbatch: failed to spawn wrapper: {ex}",
            file=sys.stderr,
        )
        return 1
    _pid_file(job_id).write_text(str(proc.pid))

    # Spawn a watcher that updates state on exit. We use a tiny
    # background bash one-liner so the watcher survives the
    # parent dying.
    watcher_script = (
        f"wait {proc.pid} 2>/dev/null; "
        f"rc=$?; "
        f"echo $rc > {_exit_file(job_id)}; "
        f"if [ $rc = 0 ]; then "
        f"  echo COMPLETED > {_state_file(job_id)}; "
        f"else "
        f"  echo FAILED > {_state_file(job_id)}; "
        f"fi"
    )
    subprocess.Popen(
        ["/bin/bash", "-c", watcher_script],
        start_new_session=True,
    )

    if parsable:
        print(str(job_id))
    else:
        print(f"Submitted batch job {job_id}")
    return 0


def cmd_squeue(argv: list[str]) -> int:
    """Minimal squeue: supports `-j <jobid> -h -o %T`."""
    job_id_str: str | None = None
    fmt = ""
    headerless = False
    i = 0
    while i < len(argv):
        a = argv[i]
        if a == "-j" and i + 1 < len(argv):
            job_id_str = argv[i + 1]
            i += 2
            continue
        if a == "-h":
            headerless = True
            i += 1
            continue
        if a == "-o" and i + 1 < len(argv):
            fmt = argv[i + 1]
            i += 2
            continue
        i += 1
    if job_id_str is None:
        return 0
    try:
        job_id = int(job_id_str)
    except ValueError:
        return 0
    sf = _state_file(job_id)
    if not sf.is_file():
        # Real squeue returns nothing for unknown jobs.
        return 0
    state = sf.read_text().strip()
    # Real squeue forgets COMPLETED / FAILED jobs after a few
    # minutes; mimic by returning empty stdout for terminal
    # states so the runner falls back to sacct (which we don't
    # implement — runner's sacct-failure fallback assumes
    # COMPLETED).
    if state in ("COMPLETED", "FAILED", "CANCELLED", "TIMEOUT"):
        # Stay visible for a few squeue polls so the runner can
        # observe the terminal state at least once before we
        # disappear.
        pass
    if fmt == "%T":
        print(state)
    else:
        # Generic line.
        prefix = (
            "" if headerless
            else "JOBID PARTITION    NAME     USER ST       TIME  NODES NODELIST(REASON)\n"
        )
        print(
            prefix
            + f"{job_id:>6}      fake    fake     fake "
            + ("R " if state == "RUNNING" else "PD") + "  0:00      1 fake1"
        )
    return 0


def cmd_scancel(argv: list[str]) -> int:
    """scancel <jobid> — SIGTERM the recorded pid + flip state
    to CANCELLED."""
    if not argv:
        return 1
    try:
        job_id = int(argv[0])
    except ValueError:
        return 1
    pf = _pid_file(job_id)
    if pf.is_file():
        try:
            pid = int(pf.read_text().strip())
            os.kill(pid, signal.SIGTERM)
        except (OSError, ValueError):
            pass
    _state_file(job_id).write_text("CANCELLED")
    return 0


def cleanup(_argv: list[str]) -> int:
    """fake_sbatch cleanup — wipe the scratch state dir. Tests
    call this between runs to avoid stale state."""
    if _STATE_DIR.is_dir():
        shutil.rmtree(_STATE_DIR, ignore_errors=True)
    _STATE_DIR.mkdir(parents=True, exist_ok=True)
    return 0


def main(argv: list[str]) -> int:
    # Dispatch on argv[0] basename so the same script can stand
    # in for sbatch / squeue / scancel by symlink. When invoked
    # directly without a symlink, the first positional arg
    # picks the mode.
    name = Path(sys.argv[0]).name.lower()
    if name in ("squeue", "fake_squeue.py", "fake_squeue"):
        return cmd_squeue(argv)
    if name in ("scancel", "fake_scancel.py", "fake_scancel"):
        return cmd_scancel(argv)
    if name in ("sbatch", "fake_sbatch.py", "fake_sbatch"):
        return cmd_sbatch(argv)
    # No symlink — first arg picks the mode.
    if argv and argv[0] in (
        "sbatch", "squeue", "scancel", "cleanup",
    ):
        mode = argv[0]
        rest = argv[1:]
        return {
            "sbatch": cmd_sbatch,
            "squeue": cmd_squeue,
            "scancel": cmd_scancel,
            "cleanup": cleanup,
        }[mode](rest)
    print(
        "usage: fake_sbatch.py {sbatch|squeue|scancel|cleanup} "
        "[args...]",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
