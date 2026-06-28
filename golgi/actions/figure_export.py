# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Figure-export handlers (F2.3.a + F2.3.b).

F2.3.a — single-figure export:
  1. Button click on a panel → `do_export_single_figure(fig_id)`
  2. Resolve the FigureSpec from `golgi.figures.registry`.
  3. Materialise the figure (deepcopy of the live state dict for
     Plotly figures).
  4. Apply the selected preset (`state.export_default_preset`).
  5. Render via Plotly + kaleido → bytes.
  6. Base64-encode → push to `state.export_pending_*` so the
     popover's Download anchor activates.

F2.3.b — bulk export:
  1. Exports drawer's CTA → `do_bulk_export()`
  2. Iterate `state.exports_selected_fig_ids`, render each via the
     same per-figure path, accumulate into an in-memory ZIP.
  3. Push the ZIP as a base64 data URI on
     `state.bulk_export_pending_*` so the drawer's Download anchor
     activates. Separate state slot from the per-figure popover so
     the two flows don't collide.

F2.3.a viewport — 3D plotter screenshot:
  do_export_viewport_screenshot(viewport_id) — PyVista
  Plotter.screenshot() into a PNG, same data-URI download pattern.

The browser download is browser-side ONLY — no path is shown to the
user, matching the design rule established by the sweep CSV
download (server-deployable; no filesystem access from the client).
"""
from __future__ import annotations

import asyncio
import base64
import datetime as _dt
import io
import traceback
import zipfile
from typing import Callable

from golgi.figures import registry as _registry
from golgi.figures.export import (
    PRESETS,
    _plotly_render_kwargs,
    apply_preset_to_plotly_fig,
)


_MIME_BY_FMT = {
    "pdf": "application/pdf",
    "svg": "image/svg+xml",
    "png": "image/png",
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "eps": "application/postscript",
}


def _safe_id_for_filename(fig_id: str) -> str:
    """`pop.kde` → `pop_kde`. Keeps filenames POSIX-friendly
    + reversible to the registry id."""
    return fig_id.replace(".", "_").replace("/", "_")


def _timestamp() -> str:
    return _dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def _plotly_bytes(fig_dict: dict, fmt: str, preset) -> bytes:
    """Render a Plotly fig dict to bytes via kaleido. Width / height
    / scale come from `_plotly_render_kwargs` so match-ui and
    override presets follow the same code path as the on-disk
    `render_publication_plotly`. The fmt argument is honoured even
    if the preset's default fmt differs (the popover Format
    dropdown can override per export)."""
    import plotly.io as pio
    try:
        import kaleido  # noqa: F401
    except ImportError as ex:
        raise RuntimeError(
            "Plotly vector export requires `kaleido`. "
            "Install with: pip install kaleido"
        ) from ex
    # Pass fmt explicitly so the scale-vs-no-scale rule honours
    # the popover's Format dropdown (which can override preset.fmt).
    kwargs = _plotly_render_kwargs(preset, fmt=fmt)
    return pio.to_image(fig_dict, **kwargs)


def _render_spec_bytes(
    spec, fig_obj, fmt: str, preset,
) -> tuple[bytes, str, str]:
    """Unified render path. Returns (bytes, mime, file_ext).

    Plotly state-var specs: apply the preset to a deepcopy of the
    state dict and render via kaleido at the requested fmt.

    render3d_variant specs: `fig_obj` is already PNG bytes from
    render3d.render_variant — return as-is and force the file
    extension to .png (PyVista off-screen captures are raster
    only; vector export would need a totally different pipeline)."""
    if spec.source == "plotly_state":
        apply_preset_to_plotly_fig(fig_obj, preset)
        blob = _plotly_bytes(fig_obj, fmt, preset)
        mime = _MIME_BY_FMT.get(fmt, "application/octet-stream")
        return blob, mime, fmt
    if spec.source == "render3d_variant":
        # fig_obj from materialize() is already PNG bytes.
        if not isinstance(fig_obj, (bytes, bytearray)):
            raise RuntimeError(
                f"render3d variant {spec.id} returned "
                f"{type(fig_obj).__name__} — expected PNG bytes."
            )
        return bytes(fig_obj), "image/png", "png"
    raise ValueError(
        f"unknown FigureSpec.source {spec.source!r}"
    )


def _viewport_png_bytes(plotter) -> bytes:
    """PyVista plotter → PNG bytes via the offscreen framebuffer.
    `plotter.screenshot(return_img=True)` returns an (H, W, 4) RGBA
    uint8 array; matplotlib's PNG writer is already a project dep
    so we use it here instead of pulling PIL just for the encode."""
    import io
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.image
    img = plotter.screenshot(
        return_img=True,
        transparent_background=False,
    )
    buf = io.BytesIO()
    matplotlib.image.imsave(buf, img, format="png")
    return buf.getvalue()


def register(
    state,
    *,
    pipeline_ctx,
    viewports: dict | None = None,
    render3d_kwargs: dict | None = None,
) -> dict[str, Callable]:
    """Wire the figure-export handler. Returns the do_* callable
    bag for build_app to splat into ctrl.* + the UI bindings.

    `viewports` maps a string id to a `pyvista.Plotter`:
        {"main": pl, "cuff_designer": pl_cuff}
    The corresponding 3D-viewport export buttons pass these ids as
    do_export_viewport_screenshot("main"), and the handler looks up
    the plotter, screenshots it, and pushes a PNG data URI on the
    shared `state.export_pending_*` slot.

    `render3d_kwargs` carries the styling constants (DEFAULTS dict,
    GOLD_STYLE, BRANCH_PALETTE) that golgi.figures.render3d needs
    to render off-screen variants. Threaded into the
    FigureExportContext so the registry's render3d_variant source
    can reach them without importing app.py."""
    viewports = dict(viewports or {})
    render3d_kwargs = dict(render3d_kwargs or {})

    async def do_export_single_figure(*args) -> None:
        """Render `fig_id` with the current default preset + format
        and push the bytes as a base64 data URI on
        `state.export_pending_*` so the popover Download anchor
        activates. `fig_id` arrives as the FIRST positional arg
        (trame passes button args as a tuple).

        Errors land on `state.export_pending_error`; the popover
        surfaces them inline so a missing input or kaleido import
        problem is visible without checking the terminal."""
        if not args:
            print(
                "[export] missing fig_id — button wired without arg?",
                flush=True,
            )
            return
        fig_id = str(args[0])
        fmt = str(state.export_default_format or "pdf")
        preset_name = str(state.export_default_preset or "paper-300")
        # Reset the slot to this figure's id BEFORE the render so
        # the popover hides any stale download from a prior export
        # while the busy spinner is up.
        with state:
            state.export_pending_fig_id = fig_id
            state.export_pending_data_uri = ""
            state.export_pending_filename = ""
            state.export_pending_busy = True
            state.export_pending_error = ""
        state.flush()
        try:
            spec = _registry.get(fig_id)
        except KeyError:
            with state:
                state.export_pending_busy = False
                state.export_pending_error = (
                    f"unknown figure id '{fig_id}' "
                    f"— not in registry"
                )
            state.flush()
            return
        preset = PRESETS.get(preset_name)
        if preset is None:
            with state:
                state.export_pending_busy = False
                state.export_pending_error = (
                    f"unknown preset '{preset_name}'"
                )
            state.flush()
            return
        # Build the export context. Project dir comes from the
        # active project entry; None when no project is open (the
        # button is disabled in that case but stay defensive).
        # render3d_kwargs is threaded through so render3d_variant
        # specs can reach DEFAULTS / GOLD_STYLE / BRANCH_PALETTE.
        ctx = _registry.FigureExportContext(
            state=state,
            geom=getattr(pipeline_ctx, "geom", None),
            project_dir=None,
            render3d_kwargs=render3d_kwargs,
        )
        try:
            fig = _registry.materialize(ctx, fig_id)
        except Exception as ex:                              # noqa: BLE001
            with state:
                state.export_pending_busy = False
                state.export_pending_error = (
                    f"figure not ready: "
                    f"{type(ex).__name__}: {ex}"
                )
            state.flush()
            return
        # Render — for Plotly figures this hits kaleido (slow first
        # call). For render3d variants `fig` is already bytes;
        # _render_spec_bytes returns immediately. Both paths run
        # on the executor so the WS loop keeps heart-beating.
        loop = asyncio.get_event_loop()

        def _render() -> tuple[bytes, str, str]:
            return _render_spec_bytes(spec, fig, fmt, preset)

        try:
            blob, mime, out_ext = await loop.run_in_executor(
                None, _render,
            )
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[export] {fig_id} render failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )
            traceback.print_exc()
            with state:
                state.export_pending_busy = False
                state.export_pending_error = (
                    f"render failed: "
                    f"{type(ex).__name__}: {ex}"
                )
            state.flush()
            return
        b64 = base64.b64encode(blob).decode("ascii")
        data_uri = f"data:{mime};base64,{b64}"
        filename = (
            f"{_safe_id_for_filename(fig_id)}_"
            f"{preset.name}_{_timestamp()}.{out_ext}"
        )
        with state:
            state.export_pending_data_uri = data_uri
            state.export_pending_filename = filename
            state.export_pending_busy = False
            state.export_pending_error = ""
        state.flush()
        print(
            f"[export] {fig_id} ready · "
            f"{preset.name}/{fmt} · "
            f"{len(blob) / 1024:.1f} KB · {filename}",
            flush=True,
        )

    async def do_export_viewport_screenshot(*args) -> None:
        """Screenshot a registered 3D viewport (PyVista plotter) and
        push the PNG as a data URI on the shared export_pending_*
        slot. Same UX as do_export_single_figure: the floating
        camera button on the viewport activates a popover that
        kicks this; once the bytes are ready the Download anchor
        becomes clickable.

        `viewport_id` is the first positional arg. The popover
        always emits PNG (kaleido + preset don't apply to a raw
        PyVista framebuffer); the filename carries a timestamp so
        sequential captures don't collide."""
        if not args:
            print(
                "[export] missing viewport_id "
                "— button wired without arg?",
                flush=True,
            )
            return
        viewport_id = str(args[0])
        fig_id = f"viewport.{viewport_id}"
        with state:
            state.export_pending_fig_id = fig_id
            state.export_pending_data_uri = ""
            state.export_pending_filename = ""
            state.export_pending_busy = True
            state.export_pending_error = ""
        state.flush()
        plotter = viewports.get(viewport_id)
        if plotter is None:
            with state:
                state.export_pending_busy = False
                state.export_pending_error = (
                    f"unknown viewport '{viewport_id}'"
                )
            state.flush()
            return
        loop = asyncio.get_event_loop()
        try:
            blob = await loop.run_in_executor(
                None, _viewport_png_bytes, plotter,
            )
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[export] viewport {viewport_id} screenshot "
                f"failed: {type(ex).__name__}: {ex}",
                flush=True,
            )
            traceback.print_exc()
            with state:
                state.export_pending_busy = False
                state.export_pending_error = (
                    f"screenshot failed: "
                    f"{type(ex).__name__}: {ex}"
                )
            state.flush()
            return
        b64 = base64.b64encode(blob).decode("ascii")
        data_uri = f"data:image/png;base64,{b64}"
        filename = (
            f"viewport_{viewport_id}_{_timestamp()}.png"
        )
        with state:
            state.export_pending_data_uri = data_uri
            state.export_pending_filename = filename
            state.export_pending_busy = False
            state.export_pending_error = ""
        state.flush()
        print(
            f"[export] viewport {viewport_id} ready · png · "
            f"{len(blob) / 1024:.1f} KB · {filename}",
            flush=True,
        )

    async def do_bulk_export(*_args) -> None:
        """Render every fig_id in `state.exports_selected_fig_ids`
        and pack them into an in-memory ZIP. Result lands on
        `state.bulk_export_pending_*` so the drawer's Download
        anchor activates.

        Per-figure failures are non-fatal — the corresponding
        entry in the ZIP is skipped and the failure is logged
        into `state.bulk_export_progress` so the user knows what
        to re-build before retrying."""
        selected = list(state.exports_selected_fig_ids or [])
        if not selected:
            with state:
                state.bulk_export_pending_error = (
                    "Pick at least one figure to export."
                )
                state.bulk_export_pending_status = ""
            state.flush()
            return
        fmt = str(state.export_default_format or "pdf")
        preset_name = str(state.export_default_preset or "match-ui")
        preset = PRESETS.get(preset_name)
        if preset is None:
            with state:
                state.bulk_export_pending_error = (
                    f"unknown preset '{preset_name}'"
                )
            state.flush()
            return
        ctx_obj = _registry.FigureExportContext(
            state=state,
            geom=getattr(pipeline_ctx, "geom", None),
            project_dir=None,
            render3d_kwargs=render3d_kwargs,
        )
        # Raise the global busy lightbox so the user gets a clear
        # overlay during the (potentially multi-minute) bulk render.
        with state:
            state.bulk_export_pending_busy = True
            state.bulk_export_pending_data_uri = ""
            state.bulk_export_pending_filename = ""
            state.bulk_export_pending_error = ""
            state.bulk_export_pending_status = (
                f"Exporting 0 / {len(selected)}…"
            )
            state.bulk_export_progress = ""
            state.busy = True
            state.busy_msg = (
                f"Bulk export · 0 / {len(selected)}…"
            )
            state.busy_log = ""
        state.flush()
        # Build the ZIP in-memory. Each figure goes into a flat
        # archive — no per-category folders for v1 (the fig_id is
        # already "<category>.<name>" so a sort still groups them).
        loop = asyncio.get_event_loop()
        buf = io.BytesIO()
        ok = 0
        fail = 0
        manifest: list[dict] = []
        progress_lines: list[str] = []
        with zipfile.ZipFile(
            buf, "w", zipfile.ZIP_DEFLATED,
        ) as zf:
            for idx, fig_id in enumerate(selected, start=1):
                with state:
                    state.bulk_export_pending_status = (
                        f"Exporting {idx} / {len(selected)} "
                        f"— {fig_id}"
                    )
                    state.busy_msg = (
                        f"Bulk export · {idx} / "
                        f"{len(selected)} — {fig_id}"
                    )
                state.flush()
                try:
                    spec = _registry.get(fig_id)
                    fig = _registry.materialize(ctx_obj, fig_id)
                except Exception as ex:                       # noqa: BLE001
                    fail += 1
                    line = (
                        f"  ⚠ {fig_id}: "
                        f"{type(ex).__name__}: {ex}"
                    )
                    progress_lines.append(line)
                    with state:
                        state.bulk_export_progress = "\n".join(
                            progress_lines[-12:]
                        )
                        state.busy_log = "\n".join(
                            progress_lines[-10:]
                        )
                    state.flush()
                    print(f"[bulk-export] {line}", flush=True)
                    continue

                def _render(
                    s=spec, f=fig, p=preset, x=fmt,
                ) -> tuple[bytes, str, str]:
                    return _render_spec_bytes(s, f, x, p)

                try:
                    blob, _mime, out_ext = (
                        await loop.run_in_executor(
                            None, _render,
                        )
                    )
                except Exception as ex:                       # noqa: BLE001
                    fail += 1
                    line = (
                        f"  ⚠ {fig_id} render failed: "
                        f"{type(ex).__name__}: {ex}"
                    )
                    progress_lines.append(line)
                    traceback.print_exc()
                    with state:
                        state.bulk_export_progress = "\n".join(
                            progress_lines[-12:]
                        )
                        state.busy_log = "\n".join(
                            progress_lines[-10:]
                        )
                    state.flush()
                    print(f"[bulk-export] {line}", flush=True)
                    continue
                arcname = (
                    f"{_safe_id_for_filename(fig_id)}.{out_ext}"
                )
                zf.writestr(arcname, blob)
                manifest.append({
                    "id": fig_id,
                    "title": spec.title,
                    "category": spec.category,
                    "filename": arcname,
                    "fmt": out_ext,
                    "preset": preset_name,
                    "bytes": len(blob),
                })
                ok += 1
                line = (
                    f"  ✓ {fig_id} "
                    f"({len(blob) / 1024:.1f} KB)"
                )
                progress_lines.append(line)
                with state:
                    state.bulk_export_progress = "\n".join(
                        progress_lines[-12:]
                    )
                    state.busy_log = "\n".join(
                        progress_lines[-10:]
                    )
                state.flush()
                print(f"[bulk-export] {line}", flush=True)
            # Append a MANIFEST.json so the recipient knows which
            # figure id maps to which file in the archive.
            import json
            zf.writestr(
                "MANIFEST.json",
                json.dumps(
                    {
                        "preset": preset_name,
                        "format": fmt,
                        "n_figures": ok,
                        "n_failed": fail,
                        "figures": manifest,
                        "exported_at": _dt.datetime.now().isoformat(
                            timespec="seconds",
                        ),
                    },
                    indent=2,
                ),
            )
        if ok == 0:
            with state:
                state.bulk_export_pending_busy = False
                state.bulk_export_pending_error = (
                    f"all {len(selected)} export"
                    f"{'' if len(selected) == 1 else 's'} "
                    f"failed — see progress log"
                )
                state.bulk_export_pending_status = ""
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
            state.flush()
            return
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        data_uri = (
            f"data:application/zip;base64,{b64}"
        )
        filename = f"golgi_figures_{_timestamp()}.zip"
        with state:
            state.bulk_export_pending_busy = False
            state.bulk_export_pending_data_uri = data_uri
            state.bulk_export_pending_filename = filename
            state.bulk_export_pending_status = (
                f"✓ {ok} / {len(selected)} exported"
                + (
                    f" ({fail} failed)"
                    if fail else ""
                )
            )
            state.bulk_export_pending_error = ""
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
        state.flush()
        print(
            f"[bulk-export] ZIP ready · {ok}/{len(selected)} figs · "
            f"{len(buf.getvalue()) / 1024:.1f} KB · {filename}",
            flush=True,
        )

    def do_bulk_export_select_all_available(*_args) -> None:
        """Populate exports_selected_fig_ids with every spec whose
        availability predicate currently returns True. Greyed-out
        unavailable figures (no source data) are excluded."""
        ctx_obj = _registry.FigureExportContext(
            state=state,
            geom=getattr(pipeline_ctx, "geom", None),
            render3d_kwargs=render3d_kwargs,
        )
        ids: list[str] = []
        for spec in _registry.REGISTRY:
            avail = spec.availability
            try:
                ok = (avail is None) or bool(avail(ctx_obj))
            except Exception:                                    # noqa: BLE001
                ok = False
            if ok:
                ids.append(spec.id)
        state.exports_selected_fig_ids = ids
        state.flush()

    def do_bulk_export_clear(*_args) -> None:
        state.exports_selected_fig_ids = []
        state.flush()

    async def do_generate_report(*_args) -> None:
        """Multi-page PDF (F2.3.c). Reads section toggles from
        state.report_section_*, takes a live snapshot of the main
        viewport, calls golgi.figures.report.generate_report, and
        pushes the PDF bytes as a base64 data URI on
        state.report_pending_*."""
        from pathlib import Path
        from golgi.figures import report as _report
        sections = {
            "electrode_design": bool(
                state.report_section_electrode,
            ),
            "mesh_results": bool(state.report_section_mesh),
            "fiber_trajectories": bool(
                state.report_section_fibers,
            ),
            "fem_results": bool(state.report_section_fem),
            "single_fiber_sim": bool(
                state.report_section_single_fiber,
            ),
            "population_sim": bool(
                state.report_section_population,
            ),
            "sweep": bool(state.report_section_sweep),
        }
        with state:
            state.report_pending_busy = True
            state.report_pending_data_uri = ""
            state.report_pending_filename = ""
            state.report_pending_error = ""
            state.report_pending_status = "Capturing 3D viewport…"
            state.busy = True
            state.busy_msg = "Generating report — capturing 3D viewport…"
            state.busy_log = ""
        state.flush()
        # Snapshot the workspace viewport (whatever the user has
        # currently visible). v1 reuses this single snapshot for
        # every "3D" section of the report; v2 will swap in the
        # multi-variant renders from figures/render3d.py.
        viewport_png: bytes | None = None
        plotter = viewports.get("main")
        if plotter is not None:
            loop = asyncio.get_event_loop()
            try:
                viewport_png = await loop.run_in_executor(
                    None, _viewport_png_bytes, plotter,
                )
            except Exception as ex:                       # noqa: BLE001
                print(
                    f"[report] viewport snapshot failed: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )
                viewport_png = None
        # Project context. The state holds username + email + name
        # parts separately — pick the friendliest non-empty option
        # (first+last name → username → email → "anonymous").
        project_name = str(
            getattr(state, "current_project_name", "") or "",
        )
        first = str(
            getattr(state, "current_user_first_name", "") or "",
        )
        last = str(
            getattr(state, "current_user_last_name", "") or "",
        )
        username = str(
            getattr(state, "current_user_username", "") or "",
        )
        email = str(
            getattr(state, "current_user_email", "") or "",
        )
        if first or last:
            user_name = f"{first} {last}".strip()
            if username:
                user_name += f" (@{username})"
        elif username:
            user_name = f"@{username}"
        elif email:
            user_name = email
        else:
            user_name = ""
        # Extra profile fields shown under the user name on the
        # cover. All optional — empty strings get filtered out by
        # the cover layout, no need to defend here.
        user_meta = {
            "email": email,
            "position": str(
                getattr(state, "current_user_position", "") or "",
            ),
            "institution": str(
                getattr(state, "current_user_institution", "") or "",
            ),
            "country": str(
                getattr(state, "current_user_country", "") or "",
            ),
        }
        # Resolve the active project directory FIRST — the manifest
        # read below needs `project_dir`, and so does the
        # FigureExportContext we build a few lines further down.
        # (v1 of this block referenced `project_dir` before it was
        # bound and threw UnboundLocalError.)
        project_dir: Path | None = None
        try:
            from golgi import get_active
            ap = get_active()
            if ap is not None and getattr(ap, "out_dir", None):
                project_dir = Path(ap.out_dir)
        except Exception:                                  # noqa: BLE001
            project_dir = None
        # Project timestamps from project.json. The manifest carries
        # `created` + `last_modified` ISO strings; we read them
        # here so the cover doesn't have to know the manifest
        # layout. Returns empty strings when the manifest is
        # unreadable so the cover gracefully skips those rows.
        project_created = ""
        project_modified = ""
        if project_dir is not None:
            try:
                import json
                manifest = json.loads(
                    (project_dir / "project.json").read_text(
                        encoding="utf-8",
                    ),
                )
                project_created = str(
                    manifest.get("created", "") or "",
                )
                project_modified = str(
                    manifest.get("last_modified", "") or "",
                )
            except Exception as ex:                          # noqa: BLE001
                print(
                    f"[report] manifest read failed: "
                    f"{type(ex).__name__}: {ex}",
                    flush=True,
                )

        ctx_obj = _registry.FigureExportContext(
            state=state,
            geom=getattr(pipeline_ctx, "geom", None),
            project_dir=project_dir,
            render3d_kwargs=render3d_kwargs,
        )
        with state:
            state.report_pending_status = (
                "Rendering report — embedding figures + tables…"
            )
            state.busy_msg = (
                "Generating report — rendering 3D variants + "
                "embedding figures…"
            )
        state.flush()
        loop = asyncio.get_event_loop()
        try:
            pdf_bytes = await loop.run_in_executor(
                None,
                lambda: _report.generate_report(
                    ctx_obj,
                    sections=sections,
                    viewport_png=viewport_png,
                    project_name=project_name,
                    user_name=user_name,
                    user_meta=user_meta,
                    project_created=project_created,
                    project_modified=project_modified,
                    project_dir=project_dir,
                ),
            )
        except Exception as ex:                              # noqa: BLE001
            print(
                f"[report] generation failed: "
                f"{type(ex).__name__}: {ex}",
                flush=True,
            )
            traceback.print_exc()
            with state:
                state.report_pending_busy = False
                state.report_pending_error = (
                    f"report generation failed: "
                    f"{type(ex).__name__}: {ex}"
                )
                state.report_pending_status = ""
                state.busy = False
                state.busy_msg = ""
                state.busy_log = ""
            state.flush()
            return
        b64 = base64.b64encode(pdf_bytes).decode("ascii")
        data_uri = f"data:application/pdf;base64,{b64}"
        safe_name = (
            (project_name or "project")
            .replace(" ", "_").replace("/", "_")
        )
        filename = (
            f"golgi_report_{safe_name}_{_timestamp()}.pdf"
        )
        with state:
            state.report_pending_busy = False
            state.report_pending_data_uri = data_uri
            state.report_pending_filename = filename
            state.report_pending_status = (
                f"✓ Report ready · "
                f"{len(pdf_bytes) / 1024:.0f} KB"
            )
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
        state.flush()
        print(
            f"[report] ready · {len(pdf_bytes) / 1024:.1f} KB · "
            f"{filename}",
            flush=True,
        )

    def do_open_generate_report_dialog(*_args) -> None:
        state.show_generate_report_dialog = True

    def do_close_generate_report_dialog(*_args) -> None:
        state.show_generate_report_dialog = False

    return {
        "do_export_single_figure": do_export_single_figure,
        "do_export_viewport_screenshot": (
            do_export_viewport_screenshot
        ),
        "do_bulk_export": do_bulk_export,
        "do_bulk_export_select_all_available": (
            do_bulk_export_select_all_available
        ),
        "do_bulk_export_clear": do_bulk_export_clear,
        "do_generate_report": do_generate_report,
        "do_open_generate_report_dialog": (
            do_open_generate_report_dialog
        ),
        "do_close_generate_report_dialog": (
            do_close_generate_report_dialog
        ),
    }
