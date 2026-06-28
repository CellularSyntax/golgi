# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Auth-dialog + profile-dialog + logout action handlers.

W1.8b (step 2/5 of the do_* handler extraction). Eleven handlers:
- do_open_auth_dialog        open the login/register card
- do_close_auth_dialog       close it
- do_switch_auth_mode        toggle login ↔ register
- do_submit_login            verify (id, password) → claim session
- do_submit_register         create user → audit → auto-login
- do_open_profile_dialog     populate from users row
- do_close_profile_dialog    close + clear status
- do_save_profile            persist edits, mirror back to state
- do_dismiss_logout_dialog   dismiss confirm
- do_confirm_logout          chain into do_logout
- do_logout                  close active project then clear session

Six build_app helpers are threaded as kwargs (they stay co-located
with `_push_auth_session` / `_clear_auth_session` in app.py because
those two share state writes with the handlers and the whole cluster
is tightly cohesive):
  username_re, decode_avatar_data_uri, validate_avatar_bytes,
  push_auth_session, clear_auth_session, do_confirm_close.
"""
from __future__ import annotations

import asyncio
from typing import Callable

import sqlalchemy as _sa

from golgi.auth.audit import _audit_log
from golgi.auth.models import _User, get_session
from golgi.auth.session import (
    _auth_session,
    _bcrypt_hash,
    _bcrypt_verify,
    _session_lock,
    _user_avatar_data_uri,
)


def register(
    state,
    *,
    username_re,
    decode_avatar_data_uri: Callable,
    validate_avatar_bytes: Callable,
    push_auth_session: Callable,
    clear_auth_session: Callable,
    do_confirm_close: Callable,
) -> dict[str, Callable]:
    """Wire the 11 auth/profile/logout handlers.

    do_confirm_close is a cross-domain dep (project-lifecycle
    handler) — passed as a callable so this module doesn't need to
    know about W1.8c's actions/project module ordering."""

    # ------------------------------------------------------------
    # Auth dialog open/close + mode switch
    # ------------------------------------------------------------
    def do_open_auth_dialog():
        """Open the login/register card. Reset transient form fields
        so a previous failed attempt doesn't leak in."""
        state.auth_login_id = ""
        state.auth_email = ""
        state.auth_username = ""
        state.auth_first_name = ""
        state.auth_last_name = ""
        state.auth_country = ""
        state.auth_institution = ""
        state.auth_position = ""
        state.auth_password = ""
        state.auth_password_confirm = ""
        state.auth_image_data_uri = ""
        state.auth_image_file = None
        state.auth_error = ""
        state.auth_busy = False
        state.show_auth_dialog = True

    def do_close_auth_dialog():
        state.show_auth_dialog = False
        state.auth_error = ""

    def do_switch_auth_mode(mode: str):
        if mode not in ("login", "register"):
            return
        state.auth_mode = mode
        state.auth_error = ""

    # ------------------------------------------------------------
    # Login (async)
    # ------------------------------------------------------------
    async def do_submit_login():
        """Verify (email-or-username, password) against the users
        table. `auth_login_id` matches both columns; bcrypt runs in
        an executor so the asyncio loop isn't blocked. Successful
        login claims the session lock atomically."""
        login_id = str(state.auth_login_id or "").strip()
        password = str(state.auth_password or "")
        if not login_id or not password:
            state.auth_error = (
                "Email or username and password are required."
            )
            return
        lookup_lower = login_id.lower()
        state.auth_busy = True
        state.auth_error = ""
        state.flush()
        loop = asyncio.get_event_loop()
        try:
            def _verify():
                with get_session() as session:
                    row = session.execute(
                        _sa.select(_User).where(
                            (_User.email == lookup_lower)
                            | (_User.username == login_id)
                            | (_User.username == lookup_lower),
                        ),
                    ).scalar_one_or_none()
                    if row is None:
                        return None, "no-such-user"
                    if not _bcrypt_verify(
                        password, row.hashed_password,
                    ):
                        return None, "bad-password"
                    return row, "ok"

            user_row, status = await loop.run_in_executor(
                None, _verify,
            )
        except Exception as ex:                              # noqa: BLE001
            state.auth_busy = False
            state.auth_error = f"Login error: {ex}"
            _audit_log(None, "login_error",
                       payload={"login_id": login_id,
                                "error": str(ex)},
                       status="failure")
            return
        if status != "ok":
            state.auth_busy = False
            state.auth_error = (
                "Invalid email/username or password."
            )
            _audit_log(None, "login_failed",
                       payload={"login_id": login_id,
                                "reason": status},
                       status="failure")
            return
        # Atomic claim of the single-active-session lock.
        with _session_lock:
            existing_uid = _auth_session.get("user_id")
            existing_email = _auth_session.get("email")
            if (existing_uid is not None
                    and existing_uid != int(user_row.id)):
                state.auth_busy = False
                state.auth_error = (
                    f"Workspace currently in use by "
                    f"{existing_email}. Ask them to sign out, "
                    "or try again later."
                )
                _audit_log(
                    int(user_row.id), "login_rejected_busy",
                    payload={
                        "login_id": login_id,
                        "active_email": existing_email,
                    },
                    status="blocked",
                )
                return
            push_auth_session(user_row)

    # ------------------------------------------------------------
    # Register (async)
    # ------------------------------------------------------------
    async def do_submit_register():
        """Open registration: create a new (email, username,
        hashed_password [, image_blob]) row, then log the user in.
        Email + username uniqueness enforced by column constraints;
        IntegrityError mapped to a friendly message identifying which
        collided."""
        email = str(state.auth_email or "").strip().lower()
        username = str(state.auth_username or "").strip()
        password = str(state.auth_password or "")
        confirm = str(state.auth_password_confirm or "")
        first_name = str(state.auth_first_name or "").strip()
        last_name = str(state.auth_last_name or "").strip()
        country_raw = str(state.auth_country or "").strip()
        # Treat the "—" sentinel (rendered at the top of the country
        # dropdown as an "unset" option) as empty — never store the
        # placeholder.
        country = "" if country_raw in ("", "—") else country_raw
        institution = str(state.auth_institution or "").strip()
        position = str(state.auth_position or "").strip()
        avatar_b = decode_avatar_data_uri(
            str(state.auth_image_data_uri or ""),
        )
        if not email or "@" not in email:
            state.auth_error = "Enter a valid email address."
            return
        if not username_re.match(username):
            state.auth_error = (
                "Username: 3–64 chars; letters, digits, and any "
                "of . _ - @ allowed."
            )
            return
        if len(password) < 8:
            state.auth_error = (
                "Password must be at least 8 characters."
            )
            return
        if password != confirm:
            state.auth_error = "Passwords do not match."
            return
        if len(password.encode("utf-8")) > 72:
            state.auth_error = (
                "Password too long (bcrypt limit: 72 bytes). "
                "Use a shorter passphrase."
            )
            return
        # All non-image profile fields are required at sign-up.
        if not first_name:
            state.auth_error = "First name is required."
            return
        if not last_name:
            state.auth_error = "Last name is required."
            return
        if not country:
            state.auth_error = "Country is required."
            return
        if not institution:
            state.auth_error = "Institution is required."
            return
        if not position:
            state.auth_error = "Position is required."
            return
        avatar_err = validate_avatar_bytes(avatar_b)
        if avatar_err:
            state.auth_error = avatar_err
            return
        state.auth_busy = True
        state.auth_error = ""
        state.flush()
        loop = asyncio.get_event_loop()
        try:
            def _create():
                hashed = _bcrypt_hash(password)
                with get_session() as session:
                    user = _User(
                        email=email,
                        username=username,
                        hashed_password=hashed,
                        image_blob=avatar_b,
                        first_name=first_name or None,
                        last_name=last_name or None,
                        country=country or None,
                        institution=institution or None,
                        position=position or None,
                    )
                    session.add(user)
                    try:
                        session.commit()
                    except _sa.exc.IntegrityError as ex:
                        session.rollback()
                        existing_email = session.execute(
                            _sa.select(_User.id).where(
                                _User.email == email,
                            ),
                        ).scalar_one_or_none()
                        existing_user = session.execute(
                            _sa.select(_User.id).where(
                                _User.username == username,
                            ),
                        ).scalar_one_or_none()
                        if existing_email is not None:
                            return None, "email-taken"
                        if existing_user is not None:
                            return None, "username-taken"
                        return None, f"integrity:{ex}"
                    session.refresh(user)
                    return user, "ok"

            user_row, status = await loop.run_in_executor(
                None, _create,
            )
        except Exception as ex:                              # noqa: BLE001
            state.auth_busy = False
            state.auth_error = f"Registration error: {ex}"
            _audit_log(None, "register_error",
                       payload={"email": email,
                                "username": username,
                                "error": str(ex)},
                       status="failure")
            return
        if status == "email-taken":
            state.auth_busy = False
            state.auth_error = (
                "An account with this email already exists. "
                "Sign in instead?"
            )
            return
        if status == "username-taken":
            state.auth_busy = False
            state.auth_error = (
                "That username is already taken. "
                "Pick a different one."
            )
            return
        if status != "ok":
            state.auth_busy = False
            state.auth_error = f"Registration failed: {status}"
            return
        _audit_log(int(user_row.id), "register",
                   payload={"email": email,
                            "username": username,
                            "has_avatar": bool(avatar_b)},
                   status="success")
        with _session_lock:
            existing_uid = _auth_session.get("user_id")
            existing_email = _auth_session.get("email")
            if (existing_uid is not None
                    and existing_uid != int(user_row.id)):
                state.auth_busy = False
                state.auth_error = (
                    f"Account created. Workspace is currently "
                    f"in use by {existing_email} — please try "
                    "signing in once they've signed out."
                )
                return
            push_auth_session(user_row)

    # ------------------------------------------------------------
    # Profile dialog
    # ------------------------------------------------------------
    def do_open_profile_dialog():
        """Open the profile-edit card pre-populated with the
        currently-logged-in user's values. Reads from the users row
        by id so any out-of-band edits made via the DB (or another
        tab) are picked up."""
        uid = _auth_session.get("user_id")
        if uid is None:
            return
        try:
            with get_session() as session:
                row = session.get(_User, int(uid))
                if row is None:
                    return
                state.profile_email = str(row.email or "")
                state.profile_username = str(
                    row.username or row.email or "",
                )
                state.profile_first_name = str(
                    row.first_name or "",
                )
                state.profile_last_name = str(
                    row.last_name or "",
                )
                state.profile_country = str(row.country or "")
                state.profile_institution = str(
                    row.institution or "",
                )
                state.profile_position = str(row.position or "")
                state.profile_image_data_uri = (
                    _user_avatar_data_uri(row) if row.image_blob
                    else ""
                )
        except Exception as ex:                              # noqa: BLE001
            state.profile_error = (
                f"Could not load profile: {ex}"
            )
            return
        state.profile_image_file = None
        state.profile_remove_image = False
        state.profile_error = ""
        state.profile_status = ""
        state.profile_busy = False
        state.show_user_menu = False
        state.show_profile_dialog = True

    def do_close_profile_dialog():
        state.show_profile_dialog = False
        state.profile_error = ""
        state.profile_status = ""

    async def do_save_profile():
        """Persist edits to the user's row (email, username, avatar).
        Password change is NOT exposed here — out of scope for the
        current ask. Server-side image-size + mime validation
        matches the register form."""
        uid = _auth_session.get("user_id")
        if uid is None:
            state.profile_error = "Not signed in."
            return
        new_email = str(
            state.profile_email or "",
        ).strip().lower()
        new_username = str(
            state.profile_username or "",
        ).strip()
        new_first = str(state.profile_first_name or "").strip()
        new_last = str(state.profile_last_name or "").strip()
        new_country_raw = str(
            state.profile_country or "",
        ).strip()
        new_country = (
            ""
            if new_country_raw in ("", "—")
            else new_country_raw
        )
        new_institution = str(
            state.profile_institution or "",
        ).strip()
        new_position = str(state.profile_position or "").strip()
        if not new_email or "@" not in new_email:
            state.profile_error = "Enter a valid email address."
            return
        if not username_re.match(new_username):
            state.profile_error = (
                "Username: 3–64 chars; letters, digits, and any "
                "of . _ - @ allowed."
            )
            return
        avatar_b: bytes | None = None
        if bool(state.profile_remove_image):
            avatar_b = b""  # sentinel: "explicitly remove"
        elif str(state.profile_image_data_uri or "").startswith(
                "data:"):
            avatar_b = decode_avatar_data_uri(
                str(state.profile_image_data_uri or ""),
            )
            err = validate_avatar_bytes(avatar_b)
            if err:
                state.profile_error = err
                return
        # `None` here means "no new image uploaded; leave the
        # existing blob alone". We use `b""` as the explicit remove
        # sentinel above so the worker can distinguish.

        state.profile_busy = True
        state.profile_error = ""
        state.profile_status = "saving …"
        state.flush()
        loop = asyncio.get_event_loop()
        try:
            def _save():
                with get_session() as session:
                    row = session.get(_User, int(uid))
                    if row is None:
                        return None, "no-such-user"
                    row.email = new_email
                    row.username = new_username
                    row.first_name = new_first or None
                    row.last_name = new_last or None
                    row.country = new_country or None
                    row.institution = new_institution or None
                    row.position = new_position or None
                    if avatar_b is None:
                        pass  # leave existing image
                    elif avatar_b == b"":
                        row.image_blob = None
                    else:
                        row.image_blob = avatar_b
                    try:
                        session.commit()
                    except _sa.exc.IntegrityError:
                        session.rollback()
                        existing_email = session.execute(
                            _sa.select(_User.id).where(
                                _User.email == new_email,
                                _User.id != int(uid),
                            ),
                        ).scalar_one_or_none()
                        existing_user = session.execute(
                            _sa.select(_User.id).where(
                                _User.username == new_username,
                                _User.id != int(uid),
                            ),
                        ).scalar_one_or_none()
                        if existing_email is not None:
                            return None, "email-taken"
                        if existing_user is not None:
                            return None, "username-taken"
                        return None, "integrity"
                    session.refresh(row)
                    return row, "ok"

            user_row, status = await loop.run_in_executor(
                None, _save,
            )
        except Exception as ex:                              # noqa: BLE001
            state.profile_busy = False
            state.profile_error = f"Save failed: {ex}"
            return
        if status == "email-taken":
            state.profile_busy = False
            state.profile_error = (
                "Another account already uses this email."
            )
            return
        if status == "username-taken":
            state.profile_busy = False
            state.profile_error = (
                "Another account already uses this username."
            )
            return
        if status != "ok" or user_row is None:
            state.profile_busy = False
            state.profile_error = f"Save failed: {status}"
            return
        # Mirror the new identity into session + state immediately
        # (no re-login needed).
        _auth_session["email"] = str(user_row.email)
        _auth_session["username"] = str(
            user_row.username or user_row.email,
        )
        avatar_uri = _user_avatar_data_uri(user_row)
        with state:
            state.current_user_email = str(user_row.email)
            state.current_user_username = str(
                user_row.username or user_row.email,
            )
            state.current_user_first_name = str(
                user_row.first_name or "",
            )
            state.current_user_last_name = str(
                user_row.last_name or "",
            )
            state.current_user_country = str(
                user_row.country or "",
            )
            state.current_user_institution = str(
                user_row.institution or "",
            )
            state.current_user_position = str(
                user_row.position or "",
            )
            state.current_user_avatar = avatar_uri
            state.profile_image_file = None
            state.profile_remove_image = False
            state.profile_busy = False
            state.profile_status = "✓ saved"
        _audit_log(int(user_row.id), "profile_update",
                   payload={"email": str(user_row.email),
                            "username": str(user_row.username),
                            "image_changed": avatar_b is not None},
                   status="success")

    # ------------------------------------------------------------
    # Logout
    # ------------------------------------------------------------
    def do_dismiss_logout_dialog(*_args) -> None:
        state.show_logout_dialog = False

    def do_confirm_logout(*_args) -> None:
        """Logout-confirm handler. Closes the dialog (so it doesn't
        linger over the welcome view), then chains into do_logout —
        which itself closes any open project before clearing the
        auth session."""
        state.show_logout_dialog = False
        do_logout()

    def do_logout():
        """Sign out the current user. If a project is open we
        defensively close it first (workspace → welcome view) so the
        next user sees a clean slate."""
        # Tear the workspace down BEFORE clearing the auth state so
        # the audit log attributes the close to the current user,
        # not the soon-to-be-empty session.
        if bool(state.has_active_project):
            try:
                async def _close_then_clear():
                    try:
                        await do_confirm_close()
                    finally:
                        clear_auth_session()
                asyncio.ensure_future(_close_then_clear())
                return
            except Exception:                                # noqa: BLE001
                pass
        clear_auth_session()

    return {
        "do_open_auth_dialog": do_open_auth_dialog,
        "do_close_auth_dialog": do_close_auth_dialog,
        "do_switch_auth_mode": do_switch_auth_mode,
        "do_submit_login": do_submit_login,
        "do_submit_register": do_submit_register,
        "do_open_profile_dialog": do_open_profile_dialog,
        "do_close_profile_dialog": do_close_profile_dialog,
        "do_save_profile": do_save_profile,
        "do_dismiss_logout_dialog": do_dismiss_logout_dialog,
        "do_confirm_logout": do_confirm_logout,
        "do_logout": do_logout,
    }
