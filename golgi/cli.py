# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F2.2 Phase 4 — CLI subcommands for study bundles.

`golgi/app.py:main()` dispatches positional arguments to these
subcommands BEFORE it falls through to `build_app()` (the
default server-start path). That keeps the single-entry-point
convention while exposing headless study tooling:

    python -m golgi.app export <project_dir> [<out.zip>]
    python -m golgi.app import <bundle.zip> [<target_dir>]
    python -m golgi.app replay <bundle.zip | bundle_dir>
                          [--check-only] [--full] [--keep-tmp]
                          [--json]

All three exit 0 on success, 1 on user-facing errors (file
missing, sha mismatch, replay failure) and 2 on internal
exceptions (with the traceback on stderr).
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path


def _cmd_export(args) -> int:
    from golgi.projects import bundle as _bundle
    project_dir = Path(args.project_dir).expanduser().resolve()
    if not project_dir.is_dir():
        print(
            f"error: project dir not found: {project_dir}",
            file=sys.stderr,
        )
        return 1
    out_zip: Path
    if args.out_zip:
        out_zip = Path(args.out_zip).expanduser().resolve()
    else:
        # Default → next to the project, suffixed with timestamp.
        import datetime as _dt
        stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_zip = project_dir.parent / (
            f"{project_dir.name}_study_{stamp}.zip"
        )

    def _emit(stage: str, frac: float) -> None:
        # Carriage-return progress on a TTY; line-each on a pipe.
        if sys.stderr.isatty():
            print(
                f"\r[export] {stage:10s} {int(frac * 100):3d}%",
                end="",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[export] {stage:10s} {int(frac * 100):3d}%",
                file=sys.stderr,
                flush=True,
            )

    try:
        blob = _bundle.export_study(
            project_dir,
            exported_by_user=args.user or "",
            on_progress=_emit,
        )
    except Exception as ex:                              # noqa: BLE001
        print(file=sys.stderr)
        print(f"error: {type(ex).__name__}: {ex}", file=sys.stderr)
        traceback.print_exc()
        return 2
    if sys.stderr.isatty():
        print(file=sys.stderr)
    out_zip.write_bytes(blob)
    print(
        f"✓ wrote {out_zip}  "
        f"({len(blob) / (1024 * 1024):.2f} MB)",
        flush=True,
    )
    return 0


def _cmd_import(args) -> int:
    from golgi.projects import bundle as _bundle
    zip_path = Path(args.zip_path).expanduser().resolve()
    if not zip_path.is_file():
        print(
            f"error: zip not found: {zip_path}",
            file=sys.stderr,
        )
        return 1
    if args.target_dir:
        target = Path(args.target_dir).expanduser().resolve()
    else:
        # Peek the manifest for the project name; suggest a dir
        # next to the zip.
        try:
            zip_bytes = zip_path.read_bytes()
            manifest = _bundle.read_manifest(zip_bytes)
            slug = "".join(
                c if c.isalnum() or c in "._-" else "_"
                for c in manifest.get("project", {})
                .get("name", "imported")
            ).strip("_") or "imported"
        except Exception:                                # noqa: BLE001
            slug = "imported"
        target = zip_path.parent / f"{slug}_imported"
        n = 1
        while target.exists():
            n += 1
            target = zip_path.parent / f"{slug}_imported_{n}"

    try:
        zip_bytes = zip_path.read_bytes()
        manifest = _bundle.import_study(
            zip_bytes, target,
            owner_user_id=None,
        )
    except Exception as ex:                              # noqa: BLE001
        print(f"error: {type(ex).__name__}: {ex}", file=sys.stderr)
        traceback.print_exc()
        return 2
    print(
        f"✓ unpacked into {target} · "
        f"{len(manifest.get('files', []))} files · "
        f"exporter={manifest.get('exported_by', '?')}",
        flush=True,
    )
    return 0


def _cmd_replay(args) -> int:
    from golgi.projects import replay as _replay
    src = Path(args.bundle).expanduser().resolve()
    if not (src.is_file() or src.is_dir()):
        print(
            f"error: bundle not found: {src}",
            file=sys.stderr,
        )
        return 1

    def _emit(stage: str, frac: float) -> None:
        if args.json:
            return
        if sys.stderr.isatty():
            print(
                f"\r[replay] {stage:14s} {int(frac * 100):3d}%",
                end="",
                file=sys.stderr,
                flush=True,
            )
        else:
            print(
                f"[replay] {stage:14s} {int(frac * 100):3d}%",
                file=sys.stderr,
                flush=True,
            )

    try:
        report = _replay.replay_study(
            src,
            check_only=not args.full,
            keep_tmp=args.keep_tmp,
            on_progress=_emit,
        )
    except Exception as ex:                              # noqa: BLE001
        print(f"error: {type(ex).__name__}: {ex}", file=sys.stderr)
        traceback.print_exc()
        return 2
    if sys.stderr.isatty():
        print(file=sys.stderr)
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(report.short_summary())
        if not report.ok:
            for s in report.stages:
                if not s.matched:
                    print(f"  stage `{s.stage}` diverged:")
                    for f in s.outputs:
                        if not f.matched:
                            print(
                                f"    {f.name}: {f.note}"
                            )
    return 0 if report.ok else 1


def _cmd_figure(args) -> int:
    """Render a bundle's result figures to PNG, headless.

    Hydrates the bundle's CACHED fibers + per-fibre lead field + the
    cached threshold sweep (no FEM re-solve, no re-run) and calls the
    same figure builders the GUI uses, so the output matches what the
    bundle shows on import — the exact published numbers. Produces an
    activation-threshold scatter (from sweeps/) and the Vₑ FEM slice
    (from the config's slice_volume)."""
    import shutil
    import tempfile
    src = Path(args.bundle).expanduser().resolve()
    if not src.exists():
        print(f"error: not found: {src}", file=sys.stderr)
        return 1
    tmp = None
    if src.is_dir():
        proj = src
    else:
        from golgi.projects import bundle as _bundle
        tmp = Path(tempfile.mkdtemp(prefix="golgi_fig_"))
        proj = tmp / "project"
        try:
            _bundle.import_study(src.read_bytes(), proj, owner_user_id=None)
        except Exception as ex:                              # noqa: BLE001
            print(f"error: import failed: {ex}", file=sys.stderr)
            shutil.rmtree(tmp, ignore_errors=True)
            return 2
    out_dir = (Path(args.out).expanduser().resolve()
               if args.out else Path.cwd())
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.name or proj.name
    written: list[Path] = []
    try:
        import numpy as _np
        import plotly.graph_objects as _go
        import plotly.io as _pio
        from golgi.projects import sweep_cache as _swc
        from golgi.pipeline import fem_layout as _fl

        # The result figures read the cached sweep + slice_volume
        # directly off disk — no geometry hydration / FEM re-solve.

        def _save(fig_dict, suffix, w, h):
            p = out_dir / f"{stem}_{suffix}.png"
            _pio.write_image(_go.Figure(fig_dict), str(p),
                             format="png", width=w, height=h, scale=2)
            written.append(p)

        # 1) Activation-threshold scatter from the cached sweep.
        result = _swc.load_latest(proj)
        if result is not None:
            from golgi.figures.recruitment import (
                build_threshold_scatter_figure,
            )
            _save(build_threshold_scatter_figure(result),
                  "thresholds", 1100, 750)
        else:
            print("  (no sweep cache — threshold figure skipped)",
                  flush=True)

        # 2) Vₑ FEM slice heatmap from the config's slice_volume.
        sv = None
        for c in (_fl.enumerate_configs(proj) or []):
            cand = _fl.config_dir(proj, c["id"]) / "slice_volume.npz"
            if cand.is_file():
                sv = cand
                break
        if sv is not None:
            from golgi.figures.fem import _build_fem_slice_figure
            d = _np.load(sv, allow_pickle=True)
            sd = {k: d[k] for k in
                  ("x", "y", "z", "Ve", "Ex", "Ey", "Ez")
                  if k in d.files}
            _save(_build_fem_slice_figure(sd), "fem_slice", 900, 820)
        else:
            print("  (no slice_volume — FEM figure skipped)", flush=True)
    except Exception as ex:                                  # noqa: BLE001
        print(f"error: {type(ex).__name__}: {ex}", file=sys.stderr)
        traceback.print_exc()
        return 2
    finally:
        if tmp is not None:
            shutil.rmtree(tmp, ignore_errors=True)
    for p in written:
        print(f"✓ wrote {p}", flush=True)
    if not written:
        print("no figures produced (bundle had no cached results)",
              file=sys.stderr)
        return 3
    return 0


def _cmd_compute_worker(args) -> int:
    """F4.2 — remote-side entry point. Reads a JobRequest
    payload JSON, dispatches to the pipeline-specific runner,
    writes outputs.json next to the payload.

    Called by the sbatch wrapper script that `SlurmJobRunner`
    generates. Also reachable directly for debugging:

        python -m golgi.cli compute-worker /path/to/payload.json

    The payload schema is the JSON form of one of the
    pipeline JobRequest subclasses (MeshJobRequest,
    FEMJobRequest, FiberSimJobRequest, ...). The dispatch
    looks at `payload["kind"]` to pick the runner.
    """
    payload_path = Path(args.payload).expanduser().resolve()
    if not payload_path.is_file():
        print(
            f"error: payload not found: {payload_path}",
            file=sys.stderr,
        )
        return 1
    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
    except Exception as ex:                              # noqa: BLE001
        print(f"error: payload not JSON: {ex}", file=sys.stderr)
        return 1
    kind = str(payload.get("kind", "")).strip()
    if not kind:
        print(
            "error: payload missing 'kind' discriminator. The "
            "remote-execution contract requires JobRequest "
            "subclasses to include a 'kind' field naming the "
            "pipeline (e.g. 'mesh', 'fem', 'fiber_sim') so the "
            "worker can dispatch to the right runner.",
            file=sys.stderr,
        )
        return 1
    # F4.2 Phase A — dispatch table is intentionally a stub.
    # Each pipeline's runner integration with the remote
    # worker lands as part of its own SLURM enablement (mesh
    # first, then FEM, then fibers / fiber-sim). For now the
    # worker only knows how to NO-OP a payload — useful for the
    # fake_sbatch shim to exercise the runner's submit / poll /
    # collect path without touching real compute.
    if kind == "noop":
        outputs_path = payload_path.parent / "outputs.json"
        outputs_path.write_text(json.dumps(
            {"return_code": 0, "outputs": {}}, indent=2,
        ), encoding="utf-8")
        print(f"[worker] noop payload — wrote {outputs_path}")
        return 0
    print(
        f"error: no remote runner registered for kind={kind!r}. "
        f"F4.2 Phase A ships the SLURM runner + worker entry; "
        f"per-pipeline remote integrations land in F4.2 Phase B "
        f"alongside the FEM checkpoint-resume work.",
        file=sys.stderr,
    )
    return 2


def _cmd_fetch_tissue_db(args) -> int:
    """Download + install the IT'IS tissue-properties database (from itis.swiss)."""
    from golgi.conductivity.fetch_itis import fetch_itis_db
    try:
        fetch_itis_db(force=getattr(args, "force", False))
        return 0
    except Exception as ex:                              # noqa: BLE001
        import sys
        print(f"[golgi] IT'IS download failed: {ex}\n"
              "        Download it manually — see resources/tissue_db/README.md.",
              file=sys.stderr)
        return 2


def dispatch(argv: list[str]) -> "int | None":
    """Parse the leading subcommand off `argv` and run it.
    Returns the exit code (int) when a CLI command ran, or
    `None` to signal "no CLI subcommand recognised — fall
    through to the default server-start path"."""
    if not argv or argv[0] in ("--port", "-p", "--help", "-h"):
        return None
    if argv[0] not in (
        "export", "import", "replay", "figure", "compute-worker",
        "fetch-tissue-db",
    ):
        return None
    parser = argparse.ArgumentParser(
        prog="golgi",
        description=(
            "GOLGI study bundle CLI (F2.2). "
            "Run with --help on any subcommand for usage."
        ),
    )
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_export = subs.add_parser(
        "export",
        help="Pack a project dir into a study .zip.",
    )
    p_export.add_argument(
        "project_dir",
        help="Path to the project directory.",
    )
    p_export.add_argument(
        "out_zip", nargs="?",
        help=(
            "Output .zip path. Defaults to "
            "<project>_study_<timestamp>.zip alongside the "
            "project."
        ),
    )
    p_export.add_argument(
        "--user",
        help=(
            "Override the 'exported_by' field in MANIFEST.json "
            "(headless mode — no auth user available)."
        ),
        default="",
    )
    p_export.set_defaults(func=_cmd_export)

    p_import = subs.add_parser(
        "import",
        help="Unpack a study .zip into a new project dir.",
    )
    p_import.add_argument(
        "zip_path",
        help="Path to the bundle .zip.",
    )
    p_import.add_argument(
        "target_dir", nargs="?",
        help=(
            "Target dir for the unpacked project. Defaults to "
            "a fresh dir alongside the zip."
        ),
    )
    p_import.set_defaults(func=_cmd_import)

    p_replay = subs.add_parser(
        "replay",
        help="Verify a study bundle's hashes (or re-run with --full).",
    )
    p_replay.add_argument(
        "bundle",
        help="Bundle .zip path OR already-extracted dir.",
    )
    grp = p_replay.add_mutually_exclusive_group()
    grp.add_argument(
        "--check-only", action="store_true",
        help=(
            "Default — re-hash every file + compare to "
            "MANIFEST.files[].sha256. Detects byte tampering."
        ),
    )
    grp.add_argument(
        "--full", action="store_true",
        help=(
            "Re-run each pipeline stage from inputs + hash "
            "the outputs. (Phase 3b — currently falls back "
            "to check-only.)"
        ),
    )
    p_replay.add_argument(
        "--keep-tmp", action="store_true",
        help=(
            "Keep the extracted temp dir for inspection."
        ),
    )
    p_replay.add_argument(
        "--json", action="store_true",
        help="Emit the full ReplayReport as JSON on stdout.",
    )
    p_replay.set_defaults(func=_cmd_replay)

    p_figure = subs.add_parser(
        "figure",
        help=(
            "Render a bundle's result figures (activation-threshold "
            "scatter + Vₑ FEM slice) to PNG, headless."
        ),
    )
    p_figure.add_argument(
        "bundle",
        help="Bundle .golgi/.zip path OR an already-extracted dir.",
    )
    p_figure.add_argument(
        "--out", default="",
        help="Output directory for the PNGs (default: cwd).",
    )
    p_figure.add_argument(
        "--name", default="",
        help="Filename stem for the PNGs (default: project name).",
    )
    p_figure.set_defaults(func=_cmd_figure)

    # F4.2 — remote-side compute worker (called by the SLURM
    # sbatch wrapper or directly for debugging).
    p_worker = subs.add_parser(
        "compute-worker",
        help=(
            "Remote-side entry point: read a JobRequest "
            "payload JSON and dispatch to the appropriate "
            "pipeline runner. Used by SlurmJobRunner."
        ),
    )
    p_worker.add_argument(
        "payload",
        help=(
            "Path to a JobRequest JSON file with a 'kind' "
            "discriminator field naming the pipeline."
        ),
    )
    p_worker.set_defaults(func=_cmd_compute_worker)

    p_fetch = subs.add_parser(
        "fetch-tissue-db",
        help="Download the IT'IS tissue-properties database from itis.swiss.",
    )
    p_fetch.add_argument(
        "--force", action="store_true",
        help="Re-download even if a database is already installed.",
    )
    p_fetch.set_defaults(func=_cmd_fetch_tissue_db)

    args = parser.parse_args(argv)
    return args.func(args)
