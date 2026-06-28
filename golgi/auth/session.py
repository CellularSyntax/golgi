# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Session-state singleton + user/bcrypt/avatar helpers.

Holds the process-global single-active-session lock
(_session_lock) and the live session dict (_auth_session). All
user/profile read helpers (_list_users, _user_brief_by_id,
avatar rendering) live here too — they share the same
get_session() backing.

Pure helpers (_bcrypt_hash, _bcrypt_verify, _sniff_image_mime)
keep their inline imports; matplotlib / heavy stuff is dodged.
"""
from __future__ import annotations

import base64
import threading

import bcrypt

from .models import _User, get_session


# Maximum avatar size (raw bytes — checked AFTER any client-side
# downsampling). 500 KB is plenty for a 256×256 profile pic in
# JPEG/PNG/WEBP and keeps SQLite row sizes manageable.
AVATAR_MAX_BYTES = 500 * 1024


# Process-global single-active-session lock. Source of truth for
# the gatekeeper — `state.authenticated` is just a UI mirror that
# the client cannot use to forge access (the gatekeeper does NOT
# read it). On every login the lock is checked atomically under
# this mutex; the loser gets the "in use" rejection.
_session_lock = threading.Lock()
_auth_session: dict = {
    "user_id": None,        # int when logged in
    "email": None,          # str when logged in
    "since": None,          # datetime when logged in
    "session_token": None,  # opaque secret for this session
}


def _sniff_image_mime(b: bytes) -> str | None:
    """Identify supported image formats from magic bytes.
    Returns None for unknown / unsupported. Conservative: we
    only accept PNG, JPEG, and WEBP — the three the browser
    `<input type=file accept="image/*">` produces from any
    common upload."""
    if not b or len(b) < 12:
        return None
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        return "png"
    if b[:3] == b"\xff\xd8\xff":
        return "jpeg"
    if b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        return "webp"
    return None


def _user_avatar_data_uri(user_row: "_User") -> str:
    """Return the user's avatar as a base64 data URI suitable for
    `<img src=…>`. Prefers `image_blob` (sniffed for MIME) and
    falls back to a server-rendered SVG showing the first letter
    of the username on a deterministic-hue circle (so the same
    account always gets the same colour)."""
    if user_row.image_blob:
        b = bytes(user_row.image_blob)
        mime = _sniff_image_mime(b) or "png"
        return (
            f"data:image/{mime};base64,"
            + base64.b64encode(b).decode("ascii")
        )
    name = (user_row.username or user_row.email or "?").strip()
    letter = (name[0] if name else "?").upper()
    hue = (int(user_row.id or 0) * 47) % 360
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'width="64" height="64" viewBox="0 0 64 64">'
        f'<circle cx="32" cy="32" r="32" '
        f'fill="hsl({hue},65%,42%)"/>'
        f'<text x="32" y="42" text-anchor="middle" '
        f'font-family="-apple-system,system-ui,sans-serif" '
        f'font-size="30" font-weight="600" '
        f'fill="white">{letter}</text>'
        f'</svg>'
    )
    return (
        "data:image/svg+xml;base64,"
        + base64.b64encode(svg.encode("utf-8")).decode("ascii")
    )


def _list_users() -> list[dict]:
    """Return every registered user as a {id, username, email,
    avatar_data_uri} dict, sorted by username. Drives the
    share-users VAutocomplete in the project-details dialog."""
    try:
        with get_session() as session:
            rows = session.query(_User).order_by(
                _User.username.asc(), _User.email.asc(),
            ).all()
            return [
                {
                    "id": int(u.id),
                    "username": str(u.username or u.email),
                    "email": str(u.email),
                    "first_name": str(u.first_name or ""),
                    "last_name": str(u.last_name or ""),
                    "avatar_data_uri": _user_avatar_data_uri(u),
                }
                for u in rows
            ]
    except Exception as ex:
        print(f"[auth] _list_users failed: {ex}", flush=True)
        return []


def _user_brief_by_id(user_id: int | None) -> dict:
    """Return a compact {id, username, email, first_name,
    last_name, avatar_data_uri} record for a single user, or
    an empty dict when the id is unknown / null. Used by the
    detail dialog to render owner + last-modifier chips."""
    if user_id is None:
        return {}
    try:
        with get_session() as session:
            u = session.get(_User, int(user_id))
            if u is None:
                return {}
            return {
                "id": int(u.id),
                "username": str(u.username or u.email),
                "email": str(u.email),
                "first_name": str(u.first_name or ""),
                "last_name": str(u.last_name or ""),
                "avatar_data_uri": _user_avatar_data_uri(u),
            }
    except Exception:
        return {}


def _bcrypt_hash(password: str) -> bytes:
    """Synchronous bcrypt hash. CPU-bound (~100–300 ms). Callers
    in async contexts must invoke this via `loop.run_in_executor`
    so the asyncio loop isn't blocked while the KDF runs.
    Password is utf-8 encoded; bcrypt's 72-byte input cap is
    documented in the help text on the register form."""
    pw_bytes = password.encode("utf-8")[:72]
    return bcrypt.hashpw(pw_bytes, bcrypt.gensalt(rounds=12))


def _bcrypt_verify(password: str, stored_hash: bytes) -> bool:
    """Synchronous bcrypt verify. Same async-context note as
    `_bcrypt_hash`."""
    try:
        return bcrypt.checkpw(
            password.encode("utf-8")[:72], stored_hash,
        )
    except Exception:
        return False
