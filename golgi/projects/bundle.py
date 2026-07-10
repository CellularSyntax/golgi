# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Reproducible study bundle — export / import / DAG manifest
(F2.2 Phase 1 + 2).

`export_study(project_dir) -> bytes` packs a project's full state
into a single in-memory `.zip`. The recipient unpacks via
`import_study(zip_bytes, target_dir, owner_user_id)` and gets a
project that opens in golgi exactly as the exporter saw it.

The bundle's `MANIFEST.json` describes the per-stage DAG with
inputs / outputs / sha256s, so a later replay step can verify
that re-running each stage from inputs reproduces the exporter's
outputs byte-for-byte (see `golgi/projects/replay.py`).

Bundle layout (flat, mirrors the on-disk project layout — the
extracted directory IS the project dir):

    MANIFEST.json                  Bundle metadata + DAG + hashes
    project.json                   Original project manifest
    thumbnail.png                  Project tile thumbnail (if any)
    source/<orig>.<ext>            Original geometry import
    conductivities.json            Tissue σ committed values
    mesh_config.json               Mesh build inputs
    electrode_config.json          Electrode config + polarities
    nerve_paths_seed_config.json   Fiber seed config
    nerve.msh                      Built TetGen mesh
    nerve_paths_fibers.npz         Fiber trajectories
    nerve_paths_caps.json          Fiber-segment caps
    axis_line.npz                  FEM outputs (legacy flat) ↓
    slice_volume.npz
    paths_Ve.npz
    nerve_surface_Ve.npz
    Ve.xdmf / Ve.h5                FEM result fields
    E.xdmf / E.h5
    designs/<id>/manifest.json     F3.2a per-design manifest
    designs/<id>/nerve.msh         F3.2a per-design multi-domain
                                   mesh (cuff silicone is baked
                                   in, so each design has its own)
    designs/<id>/mesh_config.json
    designs/<id>/electrode_config.json
    designs/<id>/nerve_surface_pts.npz
    designs/<id>/nerve_paths_fibers.npz
    designs/<id>/axis_line.npz     F3.2a per-design FEM outputs
    designs/<id>/slice_volume.npz
    designs/<id>/paths_Ve.npz
    designs/<id>/nerve_surface_Ve.npz
    designs/<id>/Ve.xdmf / Ve.h5
    designs/<id>/E.xdmf / E.h5
    fiber_sim_results.pkl          Single-fiber sims (legacy flat)
    fiber_sim_cache.json
    sims/<id>/fiber_sim_results.pkl  F3.2 per-design sims
    sims/<id>/pop_state.pkl
    pop_state.json                 Population sims
    pop_state.pkl
    sweep_<sha>.npz × N            F2.1 sweep results
    sweep_<sha>.json × N           F2.1 sweep manifests
    env/golgi_version.txt          Reproducibility metadata
    env/requirements-frozen.txt
    audit/audit_excerpt.json       Project-scoped audit rows

Nothing in the bundle is sensitive — auth DB rows are NOT
included, only audit excerpts scoped to the project dir."""
from __future__ import annotations

import hashlib
import io
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Canonical file lists
# ---------------------------------------------------------------------------
#
# The exporter walks these globs/explicit names in the project
# root. Missing files are silently skipped — partial-run projects
# bundle whatever they've got + the MANIFEST DAG marks the
# absent stages as such.
#
# Sweep / cache files are picked up via glob since their names
# include a content-derived sha; explicit list-by-name would
# miss them.


# Explicit names — present at the project root.
_ROOT_FILES_EXPLICIT: tuple[str, ...] = (
    # Project metadata
    "project.json",
    "thumbnail.png",
    # Configs
    "conductivities.json",
    "mesh_config.json",
    "electrode_config.json",
    "nerve_paths_seed_config.json",
    # Mesh stage
    "nerve.msh",
    # Fiber stage
    "nerve_paths_fibers.npz",
    "nerve_paths_caps.json",
    "nerve_only_surface.npz",
    "current_plc.vtp",
    "current_tetgen.npz",
    "current_tetgen_payload.json",
    # FEM stage
    "axis_line.npz",
    "slice_volume.npz",
    "paths_Ve.npz",
    "nerve_surface_Ve.npz",
    "Ve.xdmf",
    "Ve.h5",
    "E.xdmf",
    "E.h5",
    "fem_results.npz",
    "fem_results.npy",
    # Sim caches
    "fiber_sim_results.pkl",
    "fiber_sim_cache.json",
    "pop_state.pkl",
    "pop_state.json",
)


# Glob patterns relative to the project root — picked up alongside
# the explicit names.
_ROOT_FILES_GLOBS: tuple[str, ...] = (
    "sweep_*.npz",
    "sweep_*.json",
    "sweep_*.csv",
    "sweep_*latest.txt",
)


# Subdirectories copied recursively into the bundle.
_SUBDIRS: tuple[str, ...] = (
    "source",        # original STL / NAS / OBJ
    "electrodes",    # per-electrode JSON definitions
    "designs",       # F3.2a per-design folder — own mesh + FEM
                     # outputs + manifest.json per design
    "configs",       # F3.2 per-config FEM lead-fields (Ve/E
                     # fields, paths_Ve, slice/axis, facet tags,
                     # electrode_config) — the volumetric solve
                     # for each montage. The current GUI + headless
                     # pipelines write the per-config FEM here, so
                     # without it a bundle ships the recordings but
                     # not the field solution it came from.
    "sims",          # F3.2 per-design sim caches (fiber + pop)
    "sweeps",        # F2.1 threshold / recruitment sweep cache
                     # (sweep_<sha>.npz + .json + CSVs + latest.txt).
                     # Without it a bundle can't surface the
                     # activation-threshold results in the Sweep tab
                     # on import — the reviewer would have to re-run.
    "fem",           # legacy F3.1 layout (kept for back-compat
                     # with bundles exported between F3.1 and
                     # F3.2a — copied if present, ignored
                     # otherwise)
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _sha256_of_bytes(blob: bytes) -> str:
    """Hex sha256 of a bytes payload — used to fingerprint every
    file that lands in the bundle."""
    return hashlib.sha256(blob).hexdigest()


def _sha256_of_file(path: Path) -> str:
    """Stream a file's bytes through sha256 in 64 KB chunks so
    large npz / xdmf payloads don't load whole into memory."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _enumerate_project_files(
    project_dir: Path,
) -> list[tuple[Path, str]]:
    """Walk the project dir and return [(absolute_path, arcname)]
    pairs for every file that should land in the bundle.

    Arcname is the path INSIDE the zip — for root-level files
    it's just the basename so the bundle dir IS a valid project
    dir on extraction. Subdirectories preserve their relative
    layout.

    Files are emitted in deterministic order (sorted) so the
    same project produces the same bundle bytes on every export,
    which keeps the MANIFEST hashes stable."""
    out: list[tuple[Path, str]] = []
    # Root-level explicit names.
    for name in _ROOT_FILES_EXPLICIT:
        p = project_dir / name
        if p.is_file():
            out.append((p, name))
    # Root-level globs.
    for pattern in _ROOT_FILES_GLOBS:
        for p in sorted(project_dir.glob(pattern)):
            if p.is_file():
                out.append((p, p.name))
    # Subdirectory recursion.
    for subdir in _SUBDIRS:
        sub = project_dir / subdir
        if not sub.is_dir():
            continue
        for p in sorted(sub.rglob("*")):
            if not p.is_file():
                continue
            arc = p.relative_to(project_dir).as_posix()
            out.append((p, arc))
    return out


def _read_env_metadata() -> dict[str, str]:
    """Build the `env/` payload — golgi version + frozen
    requirements snapshot. golgi version comes from importlib
    metadata; requirements via `pip freeze` (best-effort, falls
    back to empty when pip isn't available)."""
    info: dict[str, str] = {}
    try:
        from importlib.metadata import version as _ver
        info["golgi_version"] = _ver("golgi")
    except Exception:                                    # noqa: BLE001
        try:
            import subprocess
            out = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL, timeout=2,
            ).decode().strip()
            info["golgi_version"] = f"dev-{out}"
        except Exception:                                # noqa: BLE001
            info["golgi_version"] = "dev"
    try:
        import subprocess
        out = subprocess.check_output(
            ["pip", "freeze"],
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode("utf-8", errors="replace")
        info["requirements_frozen"] = out
    except Exception:                                    # noqa: BLE001
        info["requirements_frozen"] = (
            "# pip freeze unavailable at export time\n"
        )
    return info


def _audit_excerpt(
    project_dir: Path,
) -> list[dict[str, Any]]:
    """Pull project-scoped audit rows from the auth DB. Returns
    an empty list when no DB / no rows. Best-effort — the bundle
    survives an audit query failure with a stub list."""
    try:
        from golgi.auth.models import (
            _AuditEvent, _User, get_session,
        )
    except Exception:                                    # noqa: BLE001
        return []
    out: list[dict[str, Any]] = []
    try:
        with get_session() as session:
            user_lookup = {
                int(u.id): str(
                    u.username or u.email or f"user_{u.id}"
                )
                for u in session.query(_User).all()
            }
            events = (
                session.query(_AuditEvent)
                .filter(
                    _AuditEvent.project_dir == str(project_dir),
                )
                .order_by(_AuditEvent.ts.asc())
                .all()
            )
            for ev in events:
                out.append({
                    "ts": (
                        ev.ts.isoformat() if ev.ts else None
                    ),
                    "user_id": (
                        int(ev.user_id)
                        if ev.user_id is not None else None
                    ),
                    "user_label": (
                        user_lookup.get(int(ev.user_id), "—")
                        if ev.user_id is not None else "—"
                    ),
                    "action": str(ev.action or ""),
                    "status": str(ev.status or ""),
                    "payload": (
                        json.loads(ev.payload)
                        if ev.payload else None
                    ),
                })
    except Exception as ex:                              # noqa: BLE001
        print(
            f"[bundle] audit query failed: "
            f"{type(ex).__name__}: {ex}",
            flush=True,
        )
    return out


# ---------------------------------------------------------------------------
# DAG builder
# ---------------------------------------------------------------------------
#
# Six pipeline stages, each with explicit inputs + outputs that
# pin down the dependency edges in the manifest. A stage's
# `sha256` is the SHA-256 of the concatenated SHAs of its
# outputs (in canonical order), which gives a stable per-stage
# identifier — `golgi replay --check-only` can verify the stage
# hash without recomputing every file's hash individually.


def _stage_def(
    name: str,
    inputs: list[str],
    outputs: list[str],
    file_shas: dict[str, str],
) -> dict[str, Any]:
    """Build one stage entry for the manifest. `file_shas` is the
    map of arcname → sha256 already computed by `export_study`.
    Stages whose outputs aren't all present in `file_shas`
    report `present=False` so replay can skip them gracefully."""
    out_shas = [
        file_shas[name_] for name_ in outputs
        if name_ in file_shas
    ]
    present = (
        bool(out_shas) and len(out_shas) == len(outputs)
    )
    # Stage hash = sha256 of the concat'd sha hex strings of the
    # outputs in the listed order. Stable + cheap to recompute.
    if out_shas:
        stage_hash = hashlib.sha256(
            "|".join(out_shas).encode("ascii"),
        ).hexdigest()
    else:
        stage_hash = ""
    return {
        "stage": name,
        "present": present,
        "inputs": list(inputs),
        "outputs": list(outputs),
        "sha256": stage_hash,
    }


def _build_dag(
    file_shas: dict[str, str],
) -> list[dict[str, Any]]:
    """Six pipeline stages — each `inputs` / `outputs` list is
    the same canonical naming the existing pipeline drivers use
    (look in golgi/pipeline/*.py for the producer side of each
    file). Stages are listed in DAG order so a replay can walk
    them top-to-bottom."""
    return [
        _stage_def(
            "mesh",
            inputs=[
                "mesh_config.json",
                # source/* — the original geometry import
            ],
            outputs=["nerve.msh"],
            file_shas=file_shas,
        ),
        _stage_def(
            "fem",
            inputs=[
                "nerve.msh",
                "electrode_config.json",
                "conductivities.json",
            ],
            outputs=[
                "axis_line.npz",
                "slice_volume.npz",
                "paths_Ve.npz",
                "nerve_surface_Ve.npz",
            ],
            file_shas=file_shas,
        ),
        _stage_def(
            "fibers",
            inputs=[
                "nerve.msh",
                "nerve_paths_seed_config.json",
            ],
            outputs=[
                "nerve_paths_fibers.npz",
                "nerve_paths_caps.json",
            ],
            file_shas=file_shas,
        ),
        _stage_def(
            "fiber_sim",
            inputs=[
                "paths_Ve.npz",
                "nerve_paths_fibers.npz",
                "fiber_sim_cache.json",
            ],
            outputs=["fiber_sim_results.pkl"],
            file_shas=file_shas,
        ),
        _stage_def(
            "pop_sim",
            inputs=[
                "paths_Ve.npz",
                "nerve_paths_fibers.npz",
                "pop_state.json",
            ],
            outputs=["pop_state.pkl"],
            file_shas=file_shas,
        ),
        # Sweep stage — outputs are all `sweep_*.npz` files in
        # the bundle. Compute the stage hash off whichever
        # sweep_*.npz arcnames exist.
        _stage_def(
            "sweep",
            inputs=[
                "paths_Ve.npz",
                "nerve_paths_fibers.npz",
            ],
            outputs=sorted([
                name for name in file_shas
                if name.startswith("sweep_")
                and name.endswith(".npz")
            ]),
            file_shas=file_shas,
        ),
    ]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def export_study(
    project_dir: Path,
    *,
    exported_by_user: str = "",
    exported_by_user_id: int | None = None,
    on_progress=None,
) -> bytes:
    """Pack `project_dir` into a self-contained study bundle.
    Returns the zip as `bytes` — the caller wires it through the
    same data-URI download flow as the bulk Exports drawer.

    `on_progress(stage_name, fraction)` is invoked as files are
    added so the action handler can drive a busy-lightbox
    progress bar. `stage_name` is one of "metadata", "scan",
    "files", "manifest"; `fraction` is 0.0–1.0 within that stage."""
    project_dir = Path(project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        raise FileNotFoundError(
            f"project dir does not exist: {project_dir}",
        )

    def _emit(stage: str, frac: float) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, max(0.0, min(1.0, float(frac))))
            except Exception:                            # noqa: BLE001
                pass

    _emit("scan", 0.0)
    entries = _enumerate_project_files(project_dir)
    _emit("scan", 1.0)
    if not entries:
        # Empty projects still get a bundle — only MANIFEST +
        # env/. Caller (UI) should warn the user.
        print(
            "[bundle] WARNING: no project files matched "
            "the bundle enumeration — exporting empty study",
            flush=True,
        )

    # Build the bundle in memory. Streamed write is ~3× the
    # memory of a typical bundle (≤ 50 MB usually) — fine for
    # interactive use; the headless CLI path could swap to a
    # file-backed zipfile if this becomes a bottleneck.
    buf = io.BytesIO()
    file_shas: dict[str, str] = {}
    with zipfile.ZipFile(
        buf, "w", zipfile.ZIP_DEFLATED, allowZip64=True,
    ) as zf:
        n = len(entries)
        for idx, (src_path, arcname) in enumerate(entries):
            try:
                blob = src_path.read_bytes()
            except Exception as ex:                      # noqa: BLE001
                print(
                    f"[bundle] skip {arcname}: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )
                continue
            file_shas[arcname] = _sha256_of_bytes(blob)
            zf.writestr(arcname, blob)
            _emit(
                "files",
                (idx + 1) / max(1, n),
            )
        # env/ payload. The shas need to match the EXACT bytes
        # we write — both the version file (with trailing \n)
        # and the requirements file (verbatim) get encoded once,
        # then both written + hashed off the same bytes.
        env = _read_env_metadata()
        version_blob = (env["golgi_version"] + "\n").encode(
            "utf-8",
        )
        requirements_blob = env["requirements_frozen"].encode(
            "utf-8",
        )
        zf.writestr("env/golgi_version.txt", version_blob)
        zf.writestr(
            "env/requirements-frozen.txt", requirements_blob,
        )
        file_shas["env/golgi_version.txt"] = (
            _sha256_of_bytes(version_blob)
        )
        file_shas["env/requirements-frozen.txt"] = (
            _sha256_of_bytes(requirements_blob)
        )
        # audit/ payload.
        audit_rows = _audit_excerpt(project_dir)
        audit_blob = json.dumps(audit_rows, indent=2).encode(
            "utf-8",
        )
        zf.writestr("audit/audit_excerpt.json", audit_blob)
        file_shas["audit/audit_excerpt.json"] = (
            _sha256_of_bytes(audit_blob)
        )
        # MANIFEST.json — the heart of the bundle. Comes last so
        # the DAG can reference every other arcname's sha256.
        _emit("manifest", 0.0)
        try:
            manifest_disk = json.loads(
                (project_dir / "project.json").read_text(
                    encoding="utf-8",
                ),
            )
        except Exception:                                # noqa: BLE001
            manifest_disk = {}
        manifest = {
            "schema": "golgi.study.bundle.v1",
            "golgi_version": env["golgi_version"],
            "exported_at": datetime.now(timezone.utc).isoformat(
                timespec="seconds",
            ),
            "exported_by": str(exported_by_user or ""),
            "exported_by_user_id": (
                int(exported_by_user_id)
                if exported_by_user_id is not None else None
            ),
            "project": {
                "name": str(
                    manifest_disk.get("name", project_dir.name),
                ),
                "created": str(
                    manifest_disk.get("created", ""),
                ),
                "last_modified": str(
                    manifest_disk.get("last_modified", ""),
                ),
                "source_file": str(
                    manifest_disk.get("source_file", ""),
                ),
            },
            "files": [
                {"name": name, "sha256": sha}
                for name, sha in sorted(file_shas.items())
            ],
            "dag": _build_dag(file_shas),
        }
        manifest_blob = json.dumps(manifest, indent=2).encode(
            "utf-8",
        )
        zf.writestr("MANIFEST.json", manifest_blob)
        _emit("manifest", 1.0)
    return buf.getvalue()


def import_study(
    source: "bytes | Path | str",
    target_dir: Path,
    *,
    owner_user_id: int | None = None,
    on_progress=None,
) -> dict[str, Any]:
    """Unpack a bundle into `target_dir` and return the parsed
    MANIFEST.

    `source` is either:
      * `bytes` — bundle loaded into memory (small WS uploads).
      * `Path | str` — path to a `.zip` on the SERVER's local
        filesystem. zipfile opens it directly + streams per-
        entry, so multi-GB bundles don't have to live in RAM.

    `target_dir` must NOT already exist — the caller (UI) is
    expected to suggest a fresh dir under PROJECTS_ROOT.

    `owner_user_id` is written into the imported project's
    `project.json` as the new owner; the original exporter's
    identity is preserved in the `imported_from` field so the
    welcome view can mark the project as imported.

    Audit-event replay: every audit_excerpt entry is logged
    under the importing user with a "(imported from X@Y)"
    suffix on the action string."""
    target_dir = Path(target_dir).expanduser().resolve()
    if target_dir.exists():
        raise FileExistsError(
            f"target dir already exists: {target_dir} "
            f"(pick a fresh path)",
        )
    target_dir.mkdir(parents=True)

    def _emit(stage: str, frac: float) -> None:
        if on_progress is not None:
            try:
                on_progress(stage, max(0.0, min(1.0, float(frac))))
            except Exception:                            # noqa: BLE001
                pass

    # zipfile.ZipFile accepts either a file-like object or a
    # path string. For bytes we wrap in BytesIO (memory-bound);
    # for paths we pass through so zipfile mmaps + streams.
    # `_zip_bytes_for_audit` is the bytes payload we hash for
    # the imported project's `source_zip_sha256` — fast for
    # bytes, requires a separate sha256 pass for path sources.
    if isinstance(source, (bytes, bytearray)):
        zip_handle = io.BytesIO(bytes(source))
        zip_bytes = bytes(source)
    else:
        zip_handle = str(Path(source).expanduser().resolve())
        zip_bytes = None  # lazily computed below if needed

    manifest: dict[str, Any] = {}
    audit_rows: list[dict[str, Any]] = []
    with zipfile.ZipFile(zip_handle, "r") as zf:
        names = zf.namelist()
        n = len(names)
        _emit("scan", 1.0)
        for idx, name in enumerate(names):
            # Defensive — refuse path traversal escapes. zipfile
            # extracts safely with `extract` API but we want a
            # belt-and-braces check since the bundle came from
            # an external party.
            if (".." in Path(name).parts
                    or name.startswith("/")
                    or "\\" in name):
                print(
                    f"[bundle] refusing suspicious arcname: "
                    f"{name!r}",
                    flush=True,
                )
                continue
            # MANIFEST + audit are read into memory (small).
            # Every other file streams through a 64 KB shutil
            # copy so we never hold more than one big payload
            # in RAM at a time — important for the
            # paths_Ve.npz / Ve.h5 / slice_volume.npz that can
            # each be hundreds of MB.
            if name == "MANIFEST.json":
                manifest = json.loads(zf.read(name).decode("utf-8"))
                continue
            if name == "audit/audit_excerpt.json":
                try:
                    audit_rows = json.loads(
                        zf.read(name).decode("utf-8"),
                    )
                except Exception:                        # noqa: BLE001
                    audit_rows = []
                continue
            out_path = target_dir / name
            out_path.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(name) as src, out_path.open("wb") as dst:
                import shutil as _sh
                _sh.copyfileobj(src, dst, length=64 * 1024)
            _emit("files", (idx + 1) / max(1, n))

    # Re-write project.json with the importing user as owner,
    # carrying the exporter's identity in `imported_from`.
    project_json_path = target_dir / "project.json"
    if project_json_path.is_file():
        try:
            pj = json.loads(
                project_json_path.read_text(encoding="utf-8"),
            )
        except Exception:                                # noqa: BLE001
            pj = {}
    else:
        pj = {}
    pj["owner_user_id"] = owner_user_id
    # source_zip_sha256 — for the bytes path it's free; for the
    # path source we stream-hash so we don't have to slurp the
    # whole archive again.
    if zip_bytes is not None:
        zip_sha = _sha256_of_bytes(zip_bytes)
    else:
        try:
            zip_sha = _sha256_of_file(Path(zip_handle))
        except Exception:                                # noqa: BLE001
            zip_sha = ""
    pj["imported_from"] = {
        "exported_by": manifest.get("exported_by", ""),
        "exported_at": manifest.get("exported_at", ""),
        "schema": manifest.get("schema", ""),
        "source_zip_sha256": zip_sha,
    }
    # Reset last_modified to import time so the welcome tile
    # sorts the imported project to the top of the list.
    pj["last_modified"] = datetime.now().isoformat(
        timespec="seconds",
    )
    project_json_path.write_text(
        json.dumps(pj, indent=2), encoding="utf-8",
    )

    # Replay audit events under the importing user.
    if audit_rows and owner_user_id is not None:
        try:
            from golgi.auth.audit import _audit_log
            from_label = manifest.get("exported_by", "?")
            from_at = manifest.get("exported_at", "?")
            for row in audit_rows:
                action = row.get("action", "")
                payload = row.get("payload")
                _audit_log(
                    user_id=int(owner_user_id),
                    action=f"{action}/imported",
                    payload={
                        "original_user": row.get("user_label"),
                        "original_ts": row.get("ts"),
                        "imported_from": from_label,
                        "imported_from_ts": from_at,
                        "original_payload": payload,
                    },
                    project_dir=str(target_dir),
                    status="info",
                )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[bundle] audit replay failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )

    _emit("manifest", 1.0)
    return manifest


def read_manifest(
    source: "bytes | Path | str",
) -> dict[str, Any]:
    """Peek at a bundle's MANIFEST.json without extracting any
    files. `source` is either bundle bytes (small WS upload) or
    a path on the server's local filesystem (multi-GB bundle).

    Path-source reads ONLY the manifest entry from the zip
    central directory + the one entry's bytes — never touches
    the rest of the archive. Cheap even for 10+ GB bundles."""
    if isinstance(source, (bytes, bytearray)):
        handle = io.BytesIO(bytes(source))
    else:
        handle = str(Path(source).expanduser().resolve())
    with zipfile.ZipFile(handle, "r") as zf:
        try:
            blob = zf.read("MANIFEST.json")
        except KeyError as ex:
            raise ValueError(
                "not a golgi study bundle — MANIFEST.json missing",
            ) from ex
    return json.loads(blob.decode("utf-8"))
