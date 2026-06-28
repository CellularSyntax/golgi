# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""SQLAlchemy ORM classes + engine setup + migrations for the
auth + audit-log tables.

State:
  _auth_engine   — engine handle, None until _init_auth_db() runs
  _AuthSession   — sessionmaker(), None until _init_auth_db() runs
  Both stay private; external callers should use get_session()
  so they fail loudly if they forget the init.

Callers (golgi.py + golgi/auth/audit.py / session.py once those
land) get a working Session by calling get_session(), not by
referencing _AuthSession directly — that would snapshot None at
import time.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import sqlalchemy as _sa
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, sessionmaker,
)


class _AuthBase(DeclarativeBase):
    pass


class _User(_AuthBase):
    __tablename__ = "users"
    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(
        _sa.String(255), unique=True, nullable=False, index=True,
    )
    # Optional separate display handle. Login accepts EITHER email
    # or username in the same input field. New accounts are
    # required to set one in the register form; legacy rows are
    # backfilled to `username = email` by _migrate_users_table
    # below so the unique-index constraint always has a value
    # (nullable column kept so the migration can run incrementally).
    username: Mapped[str | None] = mapped_column(
        _sa.String(255), unique=True, nullable=True, index=True,
    )
    hashed_password: Mapped[bytes] = mapped_column(
        _sa.LargeBinary, nullable=False,
    )
    # Optional avatar BLOB (PNG / JPEG / WEBP). Capped at 500 KB
    # by the upload handler — larger uploads are rejected client-
    # side and again here for defense in depth. Served to the
    # client as a base64 data URI built by `_user_avatar_data_uri`.
    image_blob: Mapped[bytes | None] = mapped_column(
        _sa.LargeBinary, nullable=True,
    )
    # Optional profile fields — all nullable so legacy accounts
    # (registered before these were introduced) keep working,
    # and so new users can skip them at registration and fill
    # them in later from the Profile dialog. The welcome-page
    # greeting uses `first_name` (falls back to username).
    first_name: Mapped[str | None] = mapped_column(
        _sa.String(120), nullable=True,
    )
    last_name: Mapped[str | None] = mapped_column(
        _sa.String(120), nullable=True,
    )
    country: Mapped[str | None] = mapped_column(
        _sa.String(120), nullable=True,
    )
    institution: Mapped[str | None] = mapped_column(
        _sa.String(255), nullable=True,
    )
    position: Mapped[str | None] = mapped_column(
        _sa.String(255), nullable=True,
    )
    created_at: Mapped[datetime] = mapped_column(
        _sa.DateTime, nullable=False,
        default=lambda: datetime.now(timezone.utc),
    )


class _AuditEvent(_AuthBase):
    __tablename__ = "audit_events"
    id: Mapped[int] = mapped_column(primary_key=True)
    ts: Mapped[datetime] = mapped_column(
        _sa.DateTime, nullable=False, index=True,
        default=lambda: datetime.now(timezone.utc),
    )
    # Foreign-key NULL when the event was triggered before login
    # (e.g. failed login attempts) so we can still attribute the
    # IP / email-attempted in payload without dropping the row.
    user_id: Mapped[int | None] = mapped_column(
        _sa.ForeignKey("users.id"), nullable=True, index=True,
    )
    action: Mapped[str] = mapped_column(
        _sa.String(64), nullable=False, index=True,
    )
    payload: Mapped[str] = mapped_column(_sa.Text, nullable=True)
    project_dir: Mapped[str] = mapped_column(
        _sa.String(512), nullable=True,
    )
    # Outcome marker — "success" | "failure" | "info". The flight-
    # recorder decorator (`@log_action` / `@gated`) sets this
    # automatically based on whether the wrapped function raised.
    # Older direct-write call sites pass "info" / "failure"
    # explicitly. Nullable so the migration can backfill for
    # rows that pre-date this column.
    status: Mapped[str | None] = mapped_column(
        _sa.String(16), nullable=True, index=True,
    )


# ---------------------------------------------------------------------------
# Engine + session factory state (mutated by _init_auth_db)
# ---------------------------------------------------------------------------

_auth_engine: "_sa.Engine | None" = None
_AuthSession = None


def get_session():
    """Return a new SQLAlchemy Session bound to the auth engine.

    Raises RuntimeError if called before _init_auth_db() has
    run. Always invoke this — never reference _AuthSession
    directly across module boundaries (the import would
    snapshot None at module-load time)."""
    if _AuthSession is None:
        raise RuntimeError(
            "Auth DB not initialised — call _init_auth_db() "
            "first (normally via _ensure_initialized() at the "
            "top of main() / build_app())"
        )
    return _AuthSession()


def _init_auth_db(db_path: Path) -> None:
    """Open the auth SQLite engine at `db_path`, create tables,
    run migrations. Idempotent — subsequent calls return early.

    Deferred from module-import time to startup so `import
    golgi` (e.g. from tests or a compute-side worker) doesn't
    open a connection."""
    global _auth_engine, _AuthSession
    if _auth_engine is not None:
        return
    _auth_engine = _sa.create_engine(
        f"sqlite:///{db_path}",
        # check_same_thread=False so we can use the same engine
        # from the asyncio main thread AND from executor worker
        # threads (login bcrypt verify runs in
        # loop.run_in_executor). Safe here because every commit
        # goes through its own Session() which we close
        # immediately.
        connect_args={"check_same_thread": False},
    )
    _AuthBase.metadata.create_all(_auth_engine)
    _AuthSession = sessionmaker(
        bind=_auth_engine, expire_on_commit=False,
    )
    _migrate_users_table()
    _migrate_audit_events_table()


def _migrate_users_table() -> None:
    """Add optional columns to pre-existing `users` tables that
    were created before those fields existed. SQLite
    ALTER TABLE ADD COLUMN is idempotent-by-precheck: we
    inspect PRAGMA table_info first to avoid 'duplicate column'
    errors on second runs.

    Also backfills `username = email` for every row whose
    username is empty / null, so legacy accounts can immediately
    sign in with their email as the handle and the
    unique-index constraint has a value to enforce.
    """
    with _auth_engine.begin() as conn:
        cols = {
            r[1] for r in conn.exec_driver_sql(
                "PRAGMA table_info(users)",
            ).fetchall()
        }
        if "username" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN username "
                "VARCHAR(255)"
            )
        if "image_blob" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE users ADD COLUMN image_blob BLOB"
            )
        # Profile fields (added later). All optional / nullable
        # so legacy rows stay valid; the Profile dialog +
        # registration form expose them but accept blanks.
        for _col, _ty in (
            ("first_name", "VARCHAR(120)"),
            ("last_name", "VARCHAR(120)"),
            ("country", "VARCHAR(120)"),
            ("institution", "VARCHAR(255)"),
            ("position", "VARCHAR(255)"),
        ):
            if _col not in cols:
                conn.exec_driver_sql(
                    f"ALTER TABLE users ADD COLUMN {_col} {_ty}"
                )
        # Backfill: existing rows get username = email so legacy
        # accounts can log in by email-as-username on day one.
        conn.exec_driver_sql(
            "UPDATE users SET username = email "
            "WHERE username IS NULL OR username = ''"
        )
        # Unique index on username (SQLite treats NULLs as
        # distinct, so this is safe even if a future schema
        # change allows nulls again).
        conn.exec_driver_sql(
            "CREATE UNIQUE INDEX IF NOT EXISTS "
            "ix_users_username ON users (username)"
        )


def _migrate_audit_events_table() -> None:
    """Add the `status` column (success | failure | info) to
    pre-existing `audit_events` tables. Same `PRAGMA table_info`
    precheck pattern as `_migrate_users_table` — idempotent on
    re-launch. Old rows get `status = NULL` and stay queryable;
    new rows always carry a status set by the flight-recorder
    decorator or by direct callers."""
    with _auth_engine.begin() as conn:
        cols = {
            r[1] for r in conn.exec_driver_sql(
                "PRAGMA table_info(audit_events)",
            ).fetchall()
        }
        if "status" not in cols:
            conn.exec_driver_sql(
                "ALTER TABLE audit_events ADD COLUMN status "
                "VARCHAR(16)"
            )
        # Index for status-based queries (e.g. show me failed
        # solves). Idempotent thanks to IF NOT EXISTS.
        conn.exec_driver_sql(
            "CREATE INDEX IF NOT EXISTS "
            "ix_audit_events_status ON audit_events (status)"
        )
