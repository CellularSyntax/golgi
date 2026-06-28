# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F4.2 — SLURM `JobRunner`.

Submits the same JobRequest a `LocalSubprocessRunner` would handle
locally as a SLURM batch job instead. Lets long-running FEM
solves run on a cluster while the user keeps iterating in the
GUI on their workstation, and unblocks the F5.1 UQ orchestration
which fans out N parameter samples.

## Lifecycle

  1. Serialise the JobRequest to JSON under
     `<project>/sbatch/<jobname>/payload.json`.
  2. Render an sbatch wrapper script that calls
     `python -m golgi.cli compute-worker <payload.json>` with
     the same env the local runner would set.
  3. Submit via `sbatch --parsable` → capture the SLURM job id.
  4. Arm the CancelToken — `cancel.was_requested()` triggers
     `scancel <job_id>`.
  5. Poll `squeue -j <job_id>` for state (PENDING / RUNNING /
     COMPLETED / FAILED); tail the cluster's
     `slurm-<job_id>.out` file → `on_line(...)` so the UI sees
     live progress.
  6. On completion: optionally rsync the result files back to
     `req.out_dir` (a no-op when the cluster filesystem is the
     same as the workstation's).
  7. Return JobOutputs(return_code, outputs).

## Configuration

```python
SlurmJobRunner(
    partition="cpu-long",
    account="vagusgrant",        # optional
    cpus=8,
    memory_gb=32,
    time_limit="04:00:00",
    scratch_root=Path("/scratch/golgi"),
    remote_root=None,            # set if remote != local FS
    sync="rsync",                # "rsync" | "none" | "scp"
    poll_interval_s=10.0,
)
```

## Testing without a real cluster

Set `GOLGI_SBATCH=/path/to/tests/fake_sbatch.py` to use the
bundled fake_sbatch shim. The shim runs the wrapped command
locally + writes a `slurm-<fakeid>.out` file → exercises the
runner's poll / tail / cancel paths without needing a SLURM
install.
"""
from __future__ import annotations

import dataclasses
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

from .cancel import CancelToken
from .protocol import JobOutputs, JobRequest


__all__ = ["SlurmJobRunner", "is_slurm_available"]


# Default poll interval. Cluster squeue calls are cheap but
# excessive polling thrashes the controller; 10 s is the
# typical sweet spot for jobs measured in minutes.
_DEFAULT_POLL_S = 10.0


def is_slurm_available() -> bool:
    """True if the `sbatch` executable resolves on PATH OR the
    `GOLGI_SBATCH` env var points at a working script. Used by
    the env-var dispatch in app.py to fall back to local mode
    when SLURM isn't installed."""
    override = os.environ.get("GOLGI_SBATCH", "").strip()
    if override:
        return Path(override).is_file()
    return shutil.which("sbatch") is not None


class SlurmJobRunner:
    """JobRunner that submits via sbatch + polls + rsyncs back.

    Conforms to the same `.run(req, on_line, cancel)` shape every
    other runner exposes; the pipeline driver doesn't know it's
    on a cluster.
    """

    def __init__(
        self,
        *,
        partition: str,
        account: Optional[str] = None,
        cpus: int = 4,
        memory_gb: int = 16,
        time_limit: str = "02:00:00",
        scratch_root: Optional[Path] = None,
        remote_root: Optional[Path] = None,
        sync: str = "rsync",
        poll_interval_s: float = _DEFAULT_POLL_S,
        sbatch_path: Optional[str] = None,
        extra_modules: Optional[list[str]] = None,
    ):
        self.partition = str(partition)
        self.account = (
            str(account) if account else None
        )
        self.cpus = int(cpus)
        self.memory_gb = int(memory_gb)
        self.time_limit = str(time_limit)
        # `scratch_root` is where per-job files live on the
        # cluster's shared FS. None → use the project dir
        # directly (works when running on the cluster head node).
        self.scratch_root = (
            Path(scratch_root) if scratch_root else None
        )
        self.remote_root = (
            Path(remote_root) if remote_root else None
        )
        self.sync = str(sync)
        self.poll_interval_s = float(poll_interval_s)
        # Allow the fake_sbatch shim to override the real binary.
        # Falls back to env var, then PATH lookup.
        self.sbatch_path = (
            str(sbatch_path) if sbatch_path
            else (
                os.environ.get("GOLGI_SBATCH", "").strip()
                or "sbatch"
            )
        )
        self.extra_modules = list(extra_modules or [])

    # ----- Lifecycle ----------------------------------------------------

    def run(
        self,
        req: JobRequest,
        on_line: Callable[[str], None],
        cancel: CancelToken,
    ) -> JobOutputs:
        """Submit + poll + collect.

        On cancel.was_requested(), issues `scancel <jobid>` and
        returns JobOutputs(return_code=130, outputs={}).
        """
        # 1. Stage payload + outputs dirs.
        job_dir = self._job_dir(req)
        job_dir.mkdir(parents=True, exist_ok=True)
        payload_path = job_dir / "payload.json"
        payload_path.write_text(
            json.dumps(self._serialize_payload(req), default=str),
        )
        wrapper_path = job_dir / "sbatch_wrapper.sh"
        wrapper_path.write_text(
            self._render_wrapper(req, payload_path, job_dir),
        )
        wrapper_path.chmod(0o755)

        # 2. Submit.
        on_line(
            f"[slurm] submitting {wrapper_path.name} "
            f"(partition={self.partition}, cpus={self.cpus}, "
            f"mem={self.memory_gb}GB, time={self.time_limit})"
        )
        job_id = self._submit(wrapper_path)
        if job_id is None:
            on_line("[slurm] ✗ submit failed — see stderr above")
            return JobOutputs(return_code=2, outputs={})
        on_line(f"[slurm] submitted as job {job_id}")

        # 3. Poll + tail. The cancel hook bridges
        #    cancel.was_requested() → scancel.
        try:
            return_code = self._poll_until_done(
                job_id, job_dir, on_line, cancel,
            )
        finally:
            cancel.clear()

        # 4. Optionally sync results back. The default `rsync`
        #    is a no-op when the scratch IS the project dir
        #    (rsync of identical dirs is a no-op too).
        if return_code == 0:
            self._sync_back(req, job_dir, on_line)

        # 5. Collect.
        outputs = self._collect_outputs(req, job_dir)
        return JobOutputs(
            return_code=int(return_code),
            outputs=outputs,
        )

    # ----- Hooks (overridable per pipeline) -----------------------------

    def _job_dir(self, req: JobRequest) -> Path:
        """The directory holding payload.json + sbatch_wrapper.sh
        + the cluster's slurm-<jobid>.out. Default: under the
        request's out_dir (when present) or the configured
        scratch root, with a per-job subdir keyed off a hash of
        the payload."""
        out_dir = (
            self._extract_out_dir(req)
            or self.scratch_root
            or Path.cwd()
        )
        # Per-request subdir so concurrent submissions don't
        # collide. The hash is short + deterministic so re-runs
        # of the same request reuse the same sbatch dir
        # (relevant if the F4.2 Phase B resume logic ever wants
        # to inspect a prior run's payload).
        import hashlib
        payload = self._serialize_payload(req)
        digest = hashlib.sha256(
            json.dumps(payload, default=str, sort_keys=True)
            .encode("utf-8"),
        ).hexdigest()[:12]
        return Path(out_dir) / "sbatch" / digest

    def _serialize_payload(self, req: JobRequest) -> dict:
        """JSON-encodable representation of the request. Default:
        dataclass-asdict, matching LocalSubprocessRunner's
        default serialisation."""
        if hasattr(req, "serialize") and callable(req.serialize):
            return req.serialize()
        if dataclasses.is_dataclass(req):
            return dataclasses.asdict(req)
        return dict(req.__dict__)

    def _extract_out_dir(self, req: JobRequest) -> Optional[Path]:
        """Find the out_dir on the request. Each pipeline's
        JobRequest subclass carries its own out_dir field
        (mesh: out_npz, fem: out_dir, fibers: fiber_out_dir). We
        sniff the common names; subclasses can override."""
        for name in ("out_dir", "fiber_out_dir", "out_npz"):
            v = getattr(req, name, None)
            if v is None:
                continue
            p = Path(v)
            return p if p.is_dir() else p.parent
        return None

    def _collect_outputs(
        self,
        req: JobRequest,
        job_dir: Path,
    ) -> dict:
        """Read the worker's `outputs.json` (written by the
        compute-worker CLI entry on success) and return its
        contents. Pipelines that need richer collection can
        subclass and override."""
        outputs_path = job_dir / "outputs.json"
        if not outputs_path.is_file():
            return {}
        try:
            data = json.loads(outputs_path.read_text())
        except Exception:                                # noqa: BLE001
            return {}
        return dict(data.get("outputs", {}))

    def _render_wrapper(
        self,
        req: JobRequest,                                # noqa: ARG002
        payload_path: Path,
        job_dir: Path,
    ) -> str:
        """Build the sbatch wrapper script. Standard SBATCH
        directives + module loads + a single call to the
        compute-worker CLI with the payload path. Subclasses
        can override to inject pipeline-specific env / MPI
        wrappers."""
        sbatch_directives = [
            f"#SBATCH --job-name=golgi_{job_dir.name}",
            f"#SBATCH --partition={self.partition}",
            f"#SBATCH --cpus-per-task={self.cpus}",
            f"#SBATCH --mem={self.memory_gb}G",
            f"#SBATCH --time={self.time_limit}",
            f"#SBATCH --output={job_dir}/slurm-%j.out",
            f"#SBATCH --error={job_dir}/slurm-%j.out",
        ]
        if self.account:
            sbatch_directives.append(
                f"#SBATCH --account={self.account}",
            )
        module_loads = "\n".join(
            f"module load {m}" for m in self.extra_modules
        )
        # Use a heredoc-style script so quoting is robust against
        # paths with spaces.
        return (
            "#!/bin/bash\n"
            "set -euo pipefail\n"
            + "\n".join(sbatch_directives) + "\n\n"
            + (module_loads + "\n\n" if module_loads else "")
            + f'cd "{job_dir}"\n'
            + (
                f'python -u -m golgi.cli compute-worker '
                f'"{payload_path}"\n'
            )
        )

    # ----- Submission + polling internals -------------------------------

    def _submit(self, wrapper_path: Path) -> Optional[str]:
        """`sbatch --parsable <wrapper>` returns "<jobid>[;<cluster>]"
        on stdout on success. Returns the parsed jobid string,
        or None on failure."""
        try:
            proc = subprocess.run(
                [self.sbatch_path, "--parsable", str(wrapper_path)],
                capture_output=True, text=True, check=False,
            )
        except FileNotFoundError:
            return None
        if proc.returncode != 0:
            return None
        out = proc.stdout.strip()
        if not out:
            return None
        # --parsable format: "12345" or "12345;clustername".
        return out.split(";", 1)[0].strip()

    def _poll_until_done(
        self,
        job_id: str,
        job_dir: Path,
        on_line: Callable[[str], None],
        cancel: CancelToken,
    ) -> int:
        """Wait for the job to leave squeue. Tail the slurm-out
        file while we wait; honour cancel.was_requested()."""
        out_file = job_dir / f"slurm-{job_id}.out"
        tailed_lines = 0
        # Arm the cancel token's external trigger so the UI's
        # cancel button can call scancel even if no subprocess
        # is on the local side.
        cancel.arm(_ScancelHandle(job_id, self.sbatch_path))

        while True:
            if cancel.was_requested():
                on_line(f"[slurm] ⚠ cancelling job {job_id}")
                self._scancel(job_id)
                return 130
            # Tail any new lines from the slurm-out file.
            if out_file.is_file():
                try:
                    with open(out_file, "r") as f:
                        lines = f.readlines()
                    for line in lines[tailed_lines:]:
                        on_line(line.rstrip("\n"))
                    tailed_lines = len(lines)
                except OSError:
                    pass
            # Check state.
            state = self._job_state(job_id)
            if state in ("COMPLETED",):
                on_line(f"[slurm] ✓ job {job_id} completed")
                # Flush any final lines.
                if out_file.is_file():
                    try:
                        with open(out_file, "r") as f:
                            lines = f.readlines()
                        for line in lines[tailed_lines:]:
                            on_line(line.rstrip("\n"))
                    except OSError:
                        pass
                return 0
            if state in (
                "FAILED", "CANCELLED", "TIMEOUT",
                "NODE_FAIL", "BOOT_FAIL",
            ):
                on_line(
                    f"[slurm] ✗ job {job_id} ended in state "
                    f"{state}"
                )
                return 1
            time.sleep(self.poll_interval_s)

    def _job_state(self, job_id: str) -> str:
        """Read SLURM's state for `job_id` via squeue. Returns
        empty string when squeue doesn't know about the job
        (which usually means COMPLETED — squeue forgets
        finished jobs after a few minutes — so callers also
        check `sacct` if available)."""
        try:
            proc = subprocess.run(
                [
                    "squeue", "-j", str(job_id),
                    "-h", "-o", "%T",
                ],
                capture_output=True, text=True, check=False,
                timeout=10.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return ""
        out = proc.stdout.strip()
        if out:
            return out.split()[0]
        # squeue empty: try sacct.
        try:
            proc = subprocess.run(
                [
                    "sacct", "-j", str(job_id), "-n",
                    "-o", "State", "-P",
                ],
                capture_output=True, text=True, check=False,
                timeout=10.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            # Best-effort: assume completed (the squeue
            # disappearing typically means the job ended).
            return "COMPLETED"
        out2 = proc.stdout.strip().split("\n")[0].strip()
        # sacct returns e.g. "COMPLETED" or "FAILED 1:0".
        return out2.split()[0] if out2 else "COMPLETED"

    def _scancel(self, job_id: str) -> None:
        try:
            subprocess.run(
                ["scancel", str(job_id)],
                capture_output=True, check=False, timeout=10.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def _sync_back(
        self,
        req: JobRequest,
        job_dir: Path,
        on_line: Callable[[str], None],
    ) -> None:
        """rsync the per-job dir back to the request's out_dir
        when they differ (i.e. when scratch_root or remote_root
        was configured). No-op when sync='none' or both paths
        identify the same FS location."""
        if self.sync == "none":
            return
        target = self._extract_out_dir(req)
        if target is None or target.resolve() == (
            job_dir.parent.parent.resolve()
        ):
            # Project dir already IS the parent of sbatch/<hash>/
            return
        if self.sync == "rsync":
            cmd = [
                "rsync", "-a", "--info=stats1",
                str(job_dir) + "/", str(target) + "/",
            ]
        elif self.sync == "scp":
            cmd = [
                "scp", "-r", str(job_dir) + "/.",
                str(target) + "/",
            ]
        else:
            on_line(
                f"[slurm] ⚠ unknown sync mode {self.sync!r}; "
                f"skipping result-sync"
            )
            return
        try:
            proc = subprocess.run(
                cmd, capture_output=True, text=True,
                check=False, timeout=300.0,
            )
            if proc.returncode == 0:
                on_line(f"[slurm] synced results → {target}")
            else:
                on_line(
                    f"[slurm] ⚠ sync exit={proc.returncode} "
                    f"({proc.stderr.strip()[:120]})"
                )
        except (FileNotFoundError, subprocess.TimeoutExpired) as ex:
            on_line(f"[slurm] ⚠ sync failed: {ex}")


# ----- CancelToken handle adapter --------------------------------------


class _ScancelHandle:
    """Quack like a subprocess.Popen for the CancelToken's
    `terminate()` contract. The CancelToken calls .terminate()
    on whatever is armed; for SLURM we forward to scancel."""

    def __init__(self, job_id: str, sbatch_path: str):
        self.job_id = job_id
        # We resolve the scancel binary alongside sbatch — same
        # cluster install, same PATH.
        self._scancel = (
            "scancel"
            if sbatch_path == "sbatch"
            else str(Path(sbatch_path).parent / "scancel")
        )

    def terminate(self) -> None:
        try:
            subprocess.run(
                [self._scancel, str(self.job_id)],
                capture_output=True, check=False, timeout=10.0,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

    def kill(self) -> None:
        # SLURM doesn't distinguish terminate / kill; both
        # forward to scancel.
        self.terminate()

    def poll(self) -> Optional[int]:
        # Always None — the real status check happens via squeue
        # in the poll loop above.
        return None


def parse_squeue_state_letter(letter: str) -> str:
    """Convert squeue's one-letter state codes (R, PD, CD, F, ...)
    to the longer names we check against. Exposed for the
    fake_sbatch shim's testing."""
    return {
        "R": "RUNNING",
        "PD": "PENDING",
        "CD": "COMPLETED",
        "F": "FAILED",
        "CA": "CANCELLED",
        "TO": "TIMEOUT",
        "CG": "COMPLETING",
        "S": "SUSPENDED",
    }.get(letter.strip().upper(), letter.strip().upper())


# Validate the time-limit format up front so an invalid string
# from the UI fails at config time instead of at sbatch-rejection
# time. SLURM accepts:
#   * minutes              "60"
#   * minutes:seconds      "60:30"
#   * hours:min:sec        "02:30:00"
#   * days-hours           "1-12"
#   * days-hours:min:sec   "1-12:30:00"
_TIME_LIMIT_RE = re.compile(
    r"^("
    r"\d+"
    r"|\d+:\d+"
    r"|\d+:\d+:\d+"
    r"|\d+-\d+"
    r"|\d+-\d+:\d+:\d+"
    r")$"
)


def validate_time_limit(s: str) -> bool:
    return bool(_TIME_LIMIT_RE.match(str(s)))
