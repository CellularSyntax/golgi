# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Flight-recorder decorator pair — `@log_action` / `@gated`.

Originally lived as nested functions inside build_app(),
closing over `state` + `_auth_session` + `_audit_log`. To make
them usable from code that lives outside build_app (everything
the migration plan moves into golgi/), they're now exposed as
factories that take an AuthContext holding the captured `state`
proxy.

Usage from build_app() (or any module that has the trame state):

    from golgi.auth.decorators import (
        AuthContext, make_log_action, make_gated,
    )
    _auth_ctx = AuthContext(state=server.state)
    log_action = make_log_action(_auth_ctx)
    gated = make_gated(_auth_ctx)

    @log_action("load_geometry")
    async def do_load_geometry(): ...

    @gated("mesh_build")
    async def do_build_mesh(): ...

Behaviour matches the previous in-build_app definitions:
  * `log_action` wraps any sync/async handler; emits one
    audit row per call (status="success" on return,
    "failure" on raise — the exception is re-raised).
  * `gated` adds an auth check on top; refuses to run when
    `_auth_session["user_id"]` is None, surfaces the login
    card, and audits the attempt with status="blocked".
"""
from __future__ import annotations

import functools
import inspect
from dataclasses import dataclass

from .audit import _audit_log
from .session import _auth_session


@dataclass
class AuthContext:
    """Wires the decorators to the live trame state proxy + any
    other per-server context they need to read at call time."""
    state: object  # trame.state proxy, used for current_project_dir + show_auth_dialog


def _flight_recorder_capture(
    fn, args: tuple, kwargs: dict,
    capture_args: bool,
) -> dict | None:
    """Build the `parameters` payload for one wrapped call.
    Skips the audit's arg capture when disabled OR when the
    function signature has no useful params; truncates long
    positional args so a stray big array doesn't bloat the
    log."""
    if not capture_args:
        return None
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters)
    except (TypeError, ValueError):
        params = []
    out: dict = {}
    if args:
        named = []
        for i, a in enumerate(args):
            name = params[i] if i < len(params) else f"arg{i}"
            named.append({name: str(a)[:160]})
        out["args"] = named
    if kwargs:
        out["kwargs"] = {
            str(k): str(v)[:160] for k, v in kwargs.items()
        }
    return out or None


def _flight_recorder_emit(
    ctx: AuthContext,
    action_name: str,
    params: dict | None,
    status: str,
    error: BaseException | None = None,
) -> None:
    """Build the audit payload and enqueue it. Always called
    from the action's own thread (sync or async) — the
    enqueue itself is non-blocking."""
    uid = _auth_session.get("user_id")
    pdir = str(ctx.state.current_project_dir or "") or None
    payload = dict(params or {})
    if error is not None:
        payload["error"] = (
            f"{type(error).__name__}: {error}"
        )[:240]
    _audit_log(
        user_id=uid,
        action=action_name,
        payload=payload or None,
        project_dir=pdir,
        status=status,
    )


def make_log_action(ctx: AuthContext):
    """Factory returning the @log_action(...) decorator bound to
    this auth context."""

    def log_action(
        action_name: str,
        capture_args: bool = True,
    ):
        def _decorator(fn):
            is_coro = inspect.iscoroutinefunction(fn)

            if is_coro:
                @functools.wraps(fn)
                async def _async_wrap(*args, **kwargs):
                    params = _flight_recorder_capture(
                        fn, args, kwargs, capture_args,
                    )
                    try:
                        result = await fn(*args, **kwargs)
                    except BaseException as ex:
                        _flight_recorder_emit(
                            ctx, action_name, params,
                            status="failure", error=ex,
                        )
                        raise
                    _flight_recorder_emit(
                        ctx, action_name, params,
                        status="success",
                    )
                    return result
                return _async_wrap

            @functools.wraps(fn)
            def _sync_wrap(*args, **kwargs):
                params = _flight_recorder_capture(
                    fn, args, kwargs, capture_args,
                )
                try:
                    result = fn(*args, **kwargs)
                except BaseException as ex:
                    _flight_recorder_emit(
                        ctx, action_name, params,
                        status="failure", error=ex,
                    )
                    raise
                _flight_recorder_emit(
                    ctx, action_name, params,
                    status="success",
                )
                return result
            return _sync_wrap

        return _decorator

    return log_action


def make_gated(ctx: AuthContext):
    """Factory returning the @gated(...) decorator bound to this
    auth context."""

    def gated(action_name: str | None = None,
              audit: bool = True):
        def _decorator(fn):
            is_coro = inspect.iscoroutinefunction(fn)

            def _check_or_block(args, kwargs):
                uid = _auth_session.get("user_id")
                if uid is None:
                    ctx.state.show_auth_dialog = True
                    ctx.state.auth_error = (
                        "Please sign in to continue."
                    )
                    if audit and action_name:
                        # Record the attempted-while-logged-out
                        # access so it shows up in the audit log
                        # (with user_id=NULL since nobody is in).
                        _flight_recorder_emit(
                            ctx, action_name,
                            _flight_recorder_capture(
                                fn, args, kwargs, True,
                            ),
                            status="blocked",
                        )
                    return None
                return uid

            if is_coro:
                @functools.wraps(fn)
                async def _async_wrap(*args, **kwargs):
                    if _check_or_block(args, kwargs) is None:
                        return None
                    params = (
                        _flight_recorder_capture(
                            fn, args, kwargs, True,
                        )
                        if audit and action_name else None
                    )
                    try:
                        result = await fn(*args, **kwargs)
                    except BaseException as ex:
                        if audit and action_name:
                            _flight_recorder_emit(
                                ctx, action_name, params,
                                status="failure", error=ex,
                            )
                        raise
                    if audit and action_name:
                        _flight_recorder_emit(
                            ctx, action_name, params,
                            status="success",
                        )
                    return result
                return _async_wrap

            @functools.wraps(fn)
            def _sync_wrap(*args, **kwargs):
                if _check_or_block(args, kwargs) is None:
                    return None
                params = (
                    _flight_recorder_capture(
                        fn, args, kwargs, True,
                    )
                    if audit and action_name else None
                )
                try:
                    result = fn(*args, **kwargs)
                except BaseException as ex:
                    if audit and action_name:
                        _flight_recorder_emit(
                            ctx, action_name, params,
                            status="failure", error=ex,
                        )
                    raise
                if audit and action_name:
                    _flight_recorder_emit(
                        ctx, action_name, params,
                        status="success",
                    )
                return result
            return _sync_wrap

        return _decorator

    return gated
