# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Conductivity-tab + Cole-Cole-dialog action handlers.

W1.8a (step 1/5 of the do_* handler extraction). Four handlers:
- do_reset_sigma         — restore all σ to solve_nerve.py defaults
- do_update_sigma        — persist σ JSON + emit audit-log event
- do_cole_cole_apply     — write computed σ to target tissue field
- do_cole_cole_cancel    — close the Cole-Cole dialog
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from golgi.auth.audit import _audit_log
from golgi.auth.session import _auth_session


def register(
    state,
    *,
    default_sigma: dict,
    autosave: Callable[..., None],
    get_out_dir: Callable[[], Path],
) -> dict[str, Callable]:
    """Wire the four conductivity / Cole-Cole handlers."""

    def do_reset_sigma():
        """Restore all conductivities to their solve_nerve.py
        defaults and overwrite the persisted JSON."""
        with state:
            for _k, _v in default_sigma.items():
                state[_k] = float(_v)

    def do_update_sigma(*_args) -> None:
        """Persist the current σ values to results_golgi/
        conductivities.json AND emit an audit-log event.

        The per-key edit watcher (`_on_sigma_change`) already
        auto-saves on every slider tick, so the file write here is
        idempotent. The user-facing value is that this button
        leaves a single deliberate "conductivities updated" row in
        the activity log — a clear marker that the operator
        reviewed and committed the values, vs the flurry of
        intermediate writes during typing/dragging.

        Payload captures the σ values + diff vs default_sigma so
        the activity row's expandable JSON tells you what actually
        changed without having to cross-reference the defaults.
        """
        cfg: dict[str, float] = {}
        for _k in default_sigma:
            try:
                cfg[_k] = float(state[_k])
            except (TypeError, ValueError):
                cfg[_k] = float(default_sigma[_k])
        diffs: dict[str, dict] = {}
        for _k, _default in default_sigma.items():
            _cur = float(cfg.get(_k, _default))
            if abs(_cur - float(_default)) > 1e-15:
                diffs[_k] = {
                    "default": float(_default),
                    "current": _cur,
                }
        write_ok = True
        if state.has_active_project:
            try:
                (Path(get_out_dir())
                 / "conductivities.json").write_text(
                    json.dumps(cfg, indent=2),
                 encoding="utf-8")
            except Exception as ex:                      # noqa: BLE001
                write_ok = False
                print(
                    f"[sigma] update failed: {ex}",
                    flush=True,
                )
        _audit_log(
            user_id=_auth_session.get("user_id"),
            action="conductivities_update",
            payload={
                "sigma": cfg,
                "modified_from_defaults": diffs,
            },
            project_dir=(
                str(state.current_project_dir)
                if state.current_project_dir else None
            ),
            status="success" if write_ok else "failure",
        )
        # Persisted marker — flips the "Conductivities configured"
        # status row to ✓ for this project, and survives close /
        # reopen via the manifest's ui_state.
        if write_ok:
            state.sigma_committed = True
            # Force an immediate autosave so the manifest captures
            # the new flag — without this the marker only lands
            # when the next stage triggers a save, which can leave
            # a confusing window where the activity log shows
            # "Conductivities updated" but the status row still
            # says pending.
            autosave(capture_thumb=False)
        # User-visible confirmation chip in the drawer.
        state.sigma_update_status = (
            "✓ Conductivities updated"
            if write_ok else
            "⚠ Update failed — see server log"
        )

    def do_cole_cole_apply():
        """Write the computed σ to the target tissue's σ field and
        close the dialog. The σ → preset watcher in the
        Conductivities drawer will auto-flip the preset dropdown
        to "Custom value" since the typed value won't match a
        preset (Cole-Cole derived values almost never do)."""
        target = str(state.cole_cole_target or "")
        if target in default_sigma:
            try:
                state[target] = float(state.cc_sigma_result)
            except Exception:                            # noqa: BLE001
                pass
        state.show_cole_cole_dialog = False

    def do_cole_cole_cancel():
        state.show_cole_cole_dialog = False

    return {
        "do_reset_sigma": do_reset_sigma,
        "do_update_sigma": do_update_sigma,
        "do_cole_cole_apply": do_cole_cole_apply,
        "do_cole_cole_cancel": do_cole_cole_cancel,
    }
