# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""JobRequest / JobOutputs / JobRunner — the viz/compute boundary.

The migration plan's long-term goal is a service-oriented split:
  • viz container — Trame UI + plotter + project state + auth.
  • compute containers — TetGen, FEM, fiber-paths, fiber-sim, pop-sim.

This module defines the surface every pipeline driver consumes:

  req = MeshJobRequest(plc_path=..., switches=..., seeds=...)
  outputs = runner.run(req, on_line=log_callback, cancel=tok)
  if outputs.return_code == 0:
      nodes = np.load(outputs.outputs["mesh_npz"])
      ...

Same call works with LocalSubprocessRunner today and (eventually)
RemoteServiceRunner. JobRequest subclasses for each pipeline get
added in 4.3-4.7; this file is the abstract base.

Step 6.2 of migration.md added `serialize()` / `deserialize()`
on JobRequest + JobOutputs so subclasses can round-trip through
JSON-on-disk. Today only LocalSubprocessRunner's payload-file
flow uses serialize(); RemoteServiceRunner (Step 6.3) will use
the deserialize side over the wire.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Protocol


def _to_json_safe(value: Any) -> Any:
    """Coerce one value to a JSON-encodable form. Walks lists,
    tuples and dicts; Path → str; dataclass → serialize()/asdict.
    Anything else passes through (json.dumps will reject it if
    it's still non-encodable)."""
    if isinstance(value, Path):
        return str(value)
    if dataclasses.is_dataclass(value):
        if hasattr(value, "serialize") and callable(value.serialize):
            return value.serialize()
        return dataclasses.asdict(value)
    if isinstance(value, (list, tuple)):
        return [_to_json_safe(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _to_json_safe(v) for k, v in value.items()}
    return value


@dataclass
class JobRequest:
    """Base for per-job request payloads. Subclasses (one per
    pipeline) add the typed inputs that pipeline needs. Plain
    dataclass so we can use `dataclasses.asdict` for JSON
    serialization when the runner is remote."""

    def serialize(self) -> dict:
        """JSON-encodable dict representation. Default walks all
        dataclass fields and runs the shared `_to_json_safe`
        coercion (Path → str, nested dataclass → its own
        serialize()). Override when a subclass needs custom
        encoding (e.g. numpy arrays)."""
        out: dict[str, Any] = {}
        for f in dataclasses.fields(self):
            out[f.name] = _to_json_safe(getattr(self, f.name))
        return out

    @classmethod
    def deserialize(cls, d: dict) -> "JobRequest":
        """Reconstruct from the dict produced by `serialize()`.
        Default introspects field type hints and coerces Path
        strings back. Override when subclass fields need richer
        decoding (nested typed schemas, numpy arrays, …)."""
        from typing import get_type_hints
        hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}
        for f in dataclasses.fields(cls):
            if f.name not in d:
                continue
            v = d[f.name]
            t = hints.get(f.name, None)
            if t is Path and v is not None:
                kwargs[f.name] = Path(v)
            else:
                kwargs[f.name] = v
        return cls(**kwargs)


@dataclass
class JobOutputs:
    """Whatever a runner produces. Files-by-name + status code.

    `outputs` is a dict of logical name → absolute Path. Pipeline
    drivers know what keys to expect (e.g., mesh expects
    "mesh_npz"; FEM expects "axis_line" / "slice_volume" /
    "nerve_surface_Ve" / optionally "paths_Ve"). The shared
    contract keeps file naming the same whether the runner is
    local or remote.

    `return_code == 0` is success. Non-zero is failure; the
    pipeline driver decides how to surface that to the user."""
    return_code: int = 0
    outputs: dict = field(default_factory=dict)

    def serialize(self) -> dict:
        """JSON-encodable dict. Outputs values pass through the
        shared `_to_json_safe` coercion (Path → str), so the
        future remote runner can return its outputs dict over
        the wire."""
        return {
            "return_code": int(self.return_code),
            "outputs": _to_json_safe(self.outputs),
        }

    @classmethod
    def deserialize(cls, d: dict) -> "JobOutputs":
        """Reconstruct from `serialize()` output. Outputs values
        stay as-is — pipeline drivers know what type to coerce
        each named output to (Path, numpy array, …)."""
        return cls(
            return_code=int(d.get("return_code", 0)),
            outputs=dict(d.get("outputs", {})),
        )


class JobRunner(Protocol):
    """The interface every pipeline driver depends on. Concrete
    implementations:
        LocalSubprocessRunner — subprocess.Popen + stdout stream
        InProcessRunner       — direct callable, same-process
        (future) RemoteServiceRunner — HTTP POST + poll, payload
                                       serialized via dataclasses
                                       .asdict.

    Cancel contract: the runner is responsible for calling
    cancel.arm(proc) right after spawn (for subprocess runners)
    and clearing the cancel state via cancel.clear() on exit.
    """

    def run(
        self,
        req: JobRequest,
        on_line: Callable[[str], None],
        cancel,  # CancelToken
    ) -> JobOutputs:
        ...
