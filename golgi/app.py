# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""golgi · CAD-style web app for designing nerve cuff meshes.

Implements §1–§5 of nerve_studio.py (load STL/NAS, fit cuff, place
electrode contacts, set mesh sizes, build multi-domain mesh with
TetGen) inside a single trame-served web app.

Layout
------
Top: VAppBar with logo + menu items (Import · Cuff & Electrodes ·
     Mesh · Render).
Center: persistent PyVista 3D viewport (white background, Phong
     materials, same tuned look as nerve_viz.py).
Right side: VNavigationDrawer that opens with the relevant property
     panel when a menu item is clicked. Closes on outside-click.

Run:
    python golgi.py [--port 8080]

Outputs land in `results_golgi/` so this stays decoupled from
nerve_studio's caches.
"""
from __future__ import annotations

# F2.2 — the WebSocket message-size cap. Importing a study bundle
# pushes the whole reconstituted VTK scene to the browser in one
# frame; for a built mesh + FEM outputs that easily exceeds 10 MB
# and the connection is reset mid-render ("Message size … exceeds
# limit"), leaving the GUI blank with every menu disabled.
#
# There are TWO caps and they must agree:
#   * wslink reads WSLINK_MAX_MSG_SIZE at import time, BUT
#   * trame OVERWRITES WSLINK_MAX_MSG_SIZE at server.start() from
#     its own `ws_max_msg_size` option, whose default is only
#     10_000_000 (10 MB) unless TRAME_WS_MAX_MSG_SIZE is set.
# So setting WSLINK_MAX_MSG_SIZE alone is silently clobbered by
# trame — TRAME_WS_MAX_MSG_SIZE is the knob that actually wins.
# Set both, BEFORE any import below pulls wslink/trame in.
#
# 4 GB default: some validation bundles (dense meshes + full FEM
# fields) push a scene well over 1 GB in a single frame. Both are
# `setdefault`, so a user can tune it up or down via the shell env
# (e.g. TRAME_WS_MAX_MSG_SIZE) — a very large scene is memory-heavy
# on both server and browser, but the cap must not be the blocker.
import os as _os
_MAX_WS_BYTES = str(4 * 1024 * 1024 * 1024)
_os.environ.setdefault("WSLINK_MAX_MSG_SIZE", _MAX_WS_BYTES)
_os.environ.setdefault("TRAME_WS_MAX_MSG_SIZE", _MAX_WS_BYTES)

# R1.4 fix-up #6 — macOS OpenMP thread-pool-init crash.
# Symptoms when these vars are missing: clicking "Generate fiber
# population" crashes with
#     OMP: Error #179: Function pthread_mutex_init failed:
#     OMP: System error #22: Invalid argument
#     zsh: abort  python golgi.py ...
# inside `scipy.stats.gaussian_kde`. KMP_DUPLICATE_LIB_OK=TRUE
# alone (the canonical duplicate-libomp workaround) did NOT fix
# it — pthread_mutex_init returning EINVAL is libomp failing to
# CREATE a mutex, not failing to re-init. Cause appears to be
# Apple Accelerate / OpenBLAS / libomp tripping over each other
# when the BLAS / OpenMP thread pool gets initialised by scipy
# on top of an already-initialised numpy/VTK state.
#
# Hard fix: cap every BLAS / OpenMP variant to a single thread
# in the main app. None of the workloads in the GUI process
# benefit from multi-threading (KDE on ~100 fibres, plotly
# figure dicts, VTK rendering — all trivial). Heavy parallelism
# happens in the FEM subprocess via MPI ranks, not OpenMP
# threads, so capping here doesn't slow that path.
#
# These env vars only take effect if set BEFORE numpy / scipy /
# matplotlib / vtk import — hence this block sits at the very
# top of golgi/app.py alongside WSLINK_MAX_MSG_SIZE.
_os.environ.setdefault("OMP_NUM_THREADS", "1")
_os.environ.setdefault("MKL_NUM_THREADS", "1")
_os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
_os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
_os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
# Belt + suspenders: keep the duplicate-libomp escape hatch in
# case the underlying issue mutates to that variant later.
_os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import asyncio
import atexit
import base64
import functools
import pickle
import inspect
import json
import math
import os
import queue as _stdqueue
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

# Auth + audit log dependencies. bcrypt is the password-hash KDF;
# SQLAlchemy backs the users + audit_events tables in a local
# SQLite file. Both are HARD requirements once auth is enabled —
# fail loudly at import time rather than at first login so the
# user gets a clear "pip install bcrypt sqlalchemy" instead of a
# baffling 500 error from a click handler.
import bcrypt
import sqlalchemy as _sa
from sqlalchemy.orm import (
    DeclarativeBase, Mapped, mapped_column, sessionmaker,
)

# wslink/aiohttp defaults to a 30 s WebSocket protocol heartbeat,
# which sounds harmless but is actually counter-productive: aiohttp
# also auto-closes the connection if it doesn't receive a PONG
# within ~heartbeat/2 seconds. During heavy fiber-solve compute
# (multi-minute RK4 stretches with no log output) the browser can
# be slow to PONG — Chrome's tab throttling, OS scheduling, a
# pending Vue render — and aiohttp tears the WS down. Trame then
# shows its reconnect spinner instead of our app until the user
# manually refreshes.
#
# Fix: bump the heartbeat to 1 hour so the auto-close-on-no-PONG
# path effectively never fires. On localhost the TCP connection
# stays alive on its own; the application-level heartbeat
# (loop.create_task(_heartbeat()) inside do_generate_fibers) gives
# the user the visible "still running …" feedback. Also bump the
# per-message size cap so a big scene push doesn't fragment.
#
# Both `setdefault` so the user can override via shell env.
# MUST be set before importing trame — wslink reads them at module
# import time.
os.environ.setdefault("WSLINK_HEART_BEAT", "3600")
# Message-size cap is set to 4 GB at the top of this file (both
# WSLINK_MAX_MSG_SIZE and the TRAME_WS_MAX_MSG_SIZE knob that trame
# actually honours). Do NOT re-set a smaller value here — a 32 MB
# setdefault used to shadow the intent and blanked the GUI on large
# bundle imports.

import meshio
import numpy as np
import pyvista as pv

import cuff_designer

from pyvista.trame.ui import plotter_ui
from trame.app import get_server
from trame.ui.vuetify3 import VAppLayout
from trame.widgets import html, vuetify3 as v3
try:
    # trame-plotly bundles the interactive heatmap/ribbon/AF tiles
    # on the Solve panel. If the package is missing, the widget
    # will be None and we render a small "install trame-plotly"
    # notice instead of crashing the whole app.
    from trame.widgets import plotly as twp
except Exception:
    twp = None

# Step 1.2a of migration.md — figure-builder helpers extracted to
# golgi.figures.util. Names are re-imported here so existing call
# sites (inside this file) keep working unchanged.
from golgi.figures.util import (  # noqa: E402
    _fig_to_data_uri,
    _hex_to_rgba,
    _plotly_placeholder,
)
from golgi.figures.cole_cole import (  # noqa: E402
    _build_cole_cole_figure,
)
from golgi.figures.cuff import (  # noqa: E402
    _render_cuff_design_preview,
)
from golgi.figures.mesh_stats import (  # noqa: E402
    _build_combined_quality_histogram_figure,
    _build_quality_histogram_figure,
    _compute_mesh_stats_html,
)
from golgi.figures.fem import (  # noqa: E402
    _build_fem_af_figure,
    _build_fem_axis_figure,
    _build_fem_slice_figure,
    _render_fem_af_plot,
    _render_fem_axis_plot,
    _render_fem_slice_plot,
    _render_ve_colorbar_png,
    _slice_tris_at_z,
)
from golgi.figures.fiber import (  # noqa: E402
    _build_fiber_propagation_figure,
    _build_fiber_pulse_figure,
    _build_fiber_waterfall_figure,
)
from golgi.figures.population import (  # noqa: E402
    _build_pop_kde_figure,
    _build_pop_xsec_at_cuff_figure,
    _build_pop_xsec_figure,
)
from golgi.figures.compare import (  # noqa: E402
    build_compare_axis_figure,
    build_compare_slice_grid,
)
from golgi.figures.selectivity import (  # noqa: E402
    build_selectivity_bar_figure,
    build_threshold_ratio_table,
)
from golgi.figures.impedance import (  # noqa: E402
    build_impedance_bar_figure,
    build_impedance_per_pair_figure,
    fmt_ohms as _fmt_ohms_imp,
)
from golgi.figures.recording import (  # noqa: E402
    build_fiber_cnap_figure,
    build_pop_cnap_figure,
)
from golgi.segmentation import reconstruct3d as _r3d  # noqa: E402
from golgi.pipeline.selectivity import (  # noqa: E402
    branch_ids_present,
    compute_branch_recruitment,
    compute_threshold_ratio,
    compute_threshold_stats_per_branch,
    compute_veraart_si,
)

# ---------------------------------------------------------------------------
# Paths + defaults
# ---------------------------------------------------------------------------

HERE = Path(__file__).resolve().parent.parent  # repo root (parent of golgi/)
# Bundled-resource layout:
#   resources/images/     — logos + icons + favicon source
#   resources/styles/     — golgi.css (single stylesheet, sidecar'd into
#                            _golgi_assets/loader.css at startup)
#   resources/cuffs/      — ASCENT DUKE cuff preset library (JSON)
#   resources/tissue_db/  — IT'IS Tissue Properties Database (sqlite + docs)
# The runtime-generated `_golgi_assets/` dir (where trame serves from)
# is unrelated and stays at the repo root, gitignored.
RESOURCES_DIR = HERE / "resources"
DATA_DIR = HERE / "data"
CUFF_DUKE_DIR = RESOURCES_DIR / "cuffs"

# Cuff-electrode preset library — ASCENT-style JSON, loaded once at
# module import. Available throughout the app via the Electrode
# Designer dialog. Each preset's params get evaluated lazily when
# the user picks it, against a namespace seeded from the current
# cuff fit (R_ci, etc.).
_CUFF_PRESETS = cuff_designer.load_cuff_presets(CUFF_DUKE_DIR)

# Projects live under ~/Documents/Golgi/Projects/, one folder per
# project. Each project folder doubles as that project's GOLGI_OUT
# (subprocess outputs land directly inside it, alongside the
# project.json manifest + thumbnail.png + source/<orig>.<ext>).
# Before any project is opened, GOLGI_OUT points at a temporary
# orphan dir under HERE/ so module-load code that wants to write
# (the old static assets path, sigma-default load) doesn't crash.
PROJECTS_ROOT = Path.home() / "Documents" / "Golgi" / "Projects"
PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)

_NO_PROJECT_FALLBACK = HERE / "_golgi_no_project"
_NO_PROJECT_FALLBACK.mkdir(exist_ok=True)

# Active-project context — what GOLGI_OUT and UPLOAD_DIR resolve
# to at any moment. Pre-project (welcome screen) the paths point
# at _NO_PROJECT_FALLBACK; _activate_project(pdir) rewrites
# _active so subsequent reads of the proxies below see the new
# project folder. Step 2.2 of migration.md: this replaces the
# old `global GOLGI_OUT` rebinding pattern, so future module
# imports of golgi.* can read the live paths via get_active().


@dataclass
class ActiveProject:
    """Holder for the currently-active project's filesystem
    paths. Mutated by set_active(); never write to its fields
    directly — they'd drift out of sync with the on-disk
    upload_dir."""
    out_dir: Path
    upload_dir: Path


_active: ActiveProject = ActiveProject(
    out_dir=_NO_PROJECT_FALLBACK,
    upload_dir=_NO_PROJECT_FALLBACK / "uploads",
)


def get_active() -> ActiveProject:
    """Return the live ActiveProject. Lookup is at call time —
    don't cache the return value across project switches."""
    return _active


def set_active(pdir: Path) -> ActiveProject:
    """Switch the active project. Creates the project dir + its
    uploads/ subdir as a side effect."""
    global _active
    pdir = Path(pdir)
    pdir.mkdir(parents=True, exist_ok=True)
    upload_dir = pdir / "uploads"
    upload_dir.mkdir(exist_ok=True)
    _active = ActiveProject(out_dir=pdir, upload_dir=upload_dir)
    return _active


class _ActiveDirProxy:
    """Path-shaped proxy that always delegates to the live
    ActiveProject. Lets existing call sites keep using
    `GOLGI_OUT / "x"` without knowing they go through
    set_active(). A later sub-step may rewrite those sites to
    import get_active() directly and remove this shim.

    Forwards every undefined attribute (.is_dir, .glob, .parent,
    .exists, .mkdir, ...) to the underlying Path via
    __getattr__, so it's behaviourally indistinguishable from a
    Path in every call site found in the codebase today."""

    __slots__ = ("_attr",)

    def __init__(self, attr: str) -> None:
        self._attr = attr

    def _path(self) -> Path:
        return getattr(_active, self._attr)

    def __truediv__(self, other):
        return self._path() / other

    def __rtruediv__(self, other):
        return other / self._path()

    def __str__(self) -> str:
        return str(self._path())

    def __repr__(self) -> str:
        return repr(self._path())

    def __fspath__(self) -> str:
        return str(self._path())

    def __eq__(self, other) -> bool:
        if isinstance(other, _ActiveDirProxy):
            return self._path() == other._path()
        return self._path() == other

    def __ne__(self, other) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(self._path())

    def __getattr__(self, name):
        return getattr(self._path(), name)


GOLGI_OUT = _ActiveDirProxy("out_dir")
UPLOAD_DIR = _ActiveDirProxy("upload_dir")

# Make sure the orphan upload dir exists so uploads work even
# before any project is opened. Tiny one-shot mkdir.
(_NO_PROJECT_FALLBACK / "uploads").mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Auth + audit log: SQLite + SQLAlchemy + bcrypt.
# ---------------------------------------------------------------------------
# Lightweight in-app authentication for LAN-style multi-user use.
# Hard limits of this design (deliberate, called out in the rework
# planning conversation):
#
#   * `server.state` in trame is per-browser-tab BUT `pl` / `geom`
#     / `_scene_state` are process-global. We allow exactly one
#     active session (one logged-in user) at a time; concurrent
#     logins are rejected with an "in use" overlay. See
#     `_session_lock` below.
#
#   * NOT a production identity system: no email verification,
#     no password reset flow, no lockout / rate-limit, no TLS,
#     no CSRF tokens, no client→server write-protection on the
#     `authenticated` mirror state var. The gatekeeper relies on
#     the server-local `_auth_session` dict (not on the mirror)
#     for actual gating, so a hostile client cannot bypass by
#     forging `state.authenticated = True`.
#
#   * Audit log captures (timestamp, user_id, action, payload,
#     project_dir). Combined with inline `[email]` log-line
#     stamping it gives both queryable history + live UI
#     visibility of who triggered each action.
# ---------------------------------------------------------------------------

AUTH_DB_DIR = PROJECTS_ROOT / "_auth"
AUTH_DB_DIR.mkdir(parents=True, exist_ok=True)
AUTH_DB_PATH = AUTH_DB_DIR / "golgi_auth.db"


# Auth-DB internals (ORM classes, engine state, init, migrations,
# get_session helper) extracted to golgi.auth.models in steps
# 2.3a / 2.3b. The engine + session factory live in models as
# private state; this file uses get_session() to make sessions
# and calls _init_auth_db(AUTH_DB_PATH) from _ensure_initialized.
from golgi.auth.models import (  # noqa: E402
    _AuthBase, _User, _AuditEvent,
    _init_auth_db, get_session,
)
from golgi.auth.audit import (  # noqa: E402
    _audit_log, _init_audit_writer,
)
from golgi.auth.session import (  # noqa: E402
    AVATAR_MAX_BYTES,
    _auth_session,
    _bcrypt_hash,
    _bcrypt_verify,
    _list_users,
    _session_lock,
    _sniff_image_mime,
    _user_avatar_data_uri,
    _user_brief_by_id,
)
from golgi.auth.decorators import (  # noqa: E402
    AuthContext, make_gated, make_log_action,
)


def _ensure_initialized() -> None:
    """Run every deferred module-load side effect exactly once.
    Called from the top of main() AND from the top of
    build_app() so reaching either entry point lazily initialises
    the auth DB, the audit writer, and the static-asset dir.

    Each init_* function is itself idempotent, so double-invocation
    (test paths that call build_app directly without going through
    main, etc.) is safe."""
    _init_auth_db(AUTH_DB_PATH)
    _init_audit_writer(AUTH_DB_DIR / "audit_fallback.jsonl")
    _init_static_assets()


# Academic / research positions surfaced as a dropdown on the
# registration + profile dialogs. Curated for the typical user
# base of this app (neural / bioelectronics labs); ends with
# "Other" so anyone outside the list still has a valid choice.
POSITION_OPTIONS = [
    "Undergraduate Student",
    "Master's Student",
    "PhD Candidate",
    "Postdoctoral Researcher",
    "Research Assistant",
    "Research Scientist",
    "Lecturer",
    "Assistant Professor",
    "Associate Professor",
    "Professor",
    "Principal Investigator",
    "Lab Manager",
    "Research Engineer",
    "Clinical Researcher",
    "Industry Research Scientist",
    "Other",
]

# Bundled country list — ISO 3166-1 short English names, plus
# a "—" sentinel at the top so the dropdown can render an
# "unset" choice without a custom empty-state slot. Sorted
# alphabetically (the sentinel sticks to the front). Kept here
# rather than pulled from a 3rd-party lib so install footprint
# stays minimal.
COUNTRY_NAMES = [
    "—",
    "Afghanistan", "Albania", "Algeria", "Andorra", "Angola",
    "Antigua and Barbuda", "Argentina", "Armenia", "Australia",
    "Austria", "Azerbaijan", "Bahamas", "Bahrain", "Bangladesh",
    "Barbados", "Belarus", "Belgium", "Belize", "Benin",
    "Bhutan", "Bolivia", "Bosnia and Herzegovina", "Botswana",
    "Brazil", "Brunei", "Bulgaria", "Burkina Faso", "Burundi",
    "Cabo Verde", "Cambodia", "Cameroon", "Canada",
    "Central African Republic", "Chad", "Chile", "China",
    "Colombia", "Comoros", "Congo", "Costa Rica",
    "Côte d'Ivoire", "Croatia", "Cuba", "Cyprus",
    "Czech Republic", "Democratic Republic of the Congo",
    "Denmark", "Djibouti", "Dominica", "Dominican Republic",
    "Ecuador", "Egypt", "El Salvador", "Equatorial Guinea",
    "Eritrea", "Estonia", "Eswatini", "Ethiopia", "Fiji",
    "Finland", "France", "Gabon", "Gambia", "Georgia",
    "Germany", "Ghana", "Greece", "Grenada", "Guatemala",
    "Guinea", "Guinea-Bissau", "Guyana", "Haiti", "Honduras",
    "Hungary", "Iceland", "India", "Indonesia", "Iran", "Iraq",
    "Ireland", "Israel", "Italy", "Jamaica", "Japan", "Jordan",
    "Kazakhstan", "Kenya", "Kiribati", "Kuwait", "Kyrgyzstan",
    "Laos", "Latvia", "Lebanon", "Lesotho", "Liberia", "Libya",
    "Liechtenstein", "Lithuania", "Luxembourg", "Madagascar",
    "Malawi", "Malaysia", "Maldives", "Mali", "Malta",
    "Marshall Islands", "Mauritania", "Mauritius", "Mexico",
    "Micronesia", "Moldova", "Monaco", "Mongolia",
    "Montenegro", "Morocco", "Mozambique", "Myanmar",
    "Namibia", "Nauru", "Nepal", "Netherlands", "New Zealand",
    "Nicaragua", "Niger", "Nigeria", "North Korea",
    "North Macedonia", "Norway", "Oman", "Pakistan", "Palau",
    "Palestine", "Panama", "Papua New Guinea", "Paraguay",
    "Peru", "Philippines", "Poland", "Portugal", "Qatar",
    "Romania", "Russia", "Rwanda", "Saint Kitts and Nevis",
    "Saint Lucia", "Saint Vincent and the Grenadines",
    "Samoa", "San Marino", "São Tomé and Príncipe",
    "Saudi Arabia", "Senegal", "Serbia", "Seychelles",
    "Sierra Leone", "Singapore", "Slovakia", "Slovenia",
    "Solomon Islands", "Somalia", "South Africa",
    "South Korea", "South Sudan", "Spain", "Sri Lanka",
    "Sudan", "Suriname", "Sweden", "Switzerland", "Syria",
    "Taiwan", "Tajikistan", "Tanzania", "Thailand",
    "Timor-Leste", "Togo", "Tonga", "Trinidad and Tobago",
    "Tunisia", "Turkey", "Turkmenistan", "Tuvalu", "Uganda",
    "Ukraine", "United Arab Emirates", "United Kingdom",
    "United States", "Uruguay", "Uzbekistan", "Vanuatu",
    "Vatican City", "Venezuela", "Vietnam", "Yemen", "Zambia",
    "Zimbabwe",
]


BG_COLOR = "#ffffff"

# ---------------------------------------------------------------------------
# Stylesheet — lives in a sidecar `golgi.css` next to this file
# so editors get real CSS syntax-highlighting + linting, and so
# golgi.py itself stays focused on Python.
#
# Loaded into trame via the static-asset module (see the
# `_GolgiAssetsModule` / `serve = {"golgi_static": ...}` wiring
# further down — we copy `golgi.css` to the served static dir as
# `loader.css` so the URL the layout references stays stable
# across the refactor). Includes the fullscreen "wave" loading
# overlay, the Vuetify-primary recolour, every workspace shell
# style, the welcome/auth page styling, the legend, etc.
# ---------------------------------------------------------------------------
# Trame strips inline <style> tags from Vue templates, so we serve
# the CSS as a static asset and let the module system inject a
# <link rel="stylesheet"> tag into the document <head>. The logo
# is copied into the same dir so we can <img src="..."> it too.
# Static assets live next to the app (HERE/_golgi_assets) rather
# than inside GOLGI_OUT — they're shared across all projects and
# must stay reachable when GOLGI_OUT gets rebound on project open.
_GOLGI_STATIC_DIR = HERE / "_golgi_assets"
_GOLGI_CSS_FILE = _GOLGI_STATIC_DIR / "loader.css"
_GOLGI_CSS_SOURCE = RESOURCES_DIR / "styles" / "golgi.css"

# All image sources live in resources/images/ — copied at startup
# into _golgi_assets/ where trame serves them at golgi_static/<name>.
_IMG_DIR = RESOURCES_DIR / "images"

_LOGO_SRC = _IMG_DIR / "logo.png"
_LOGO_STATIC = _GOLGI_STATIC_DIR / "logo.png"
_LOGO_URL = "golgi_static/logo.png"

# Big logo (waveform + "GOLGI" wordmark) used on the welcome
# screen — distinct from the small navbar logo above. Prefer the
# animated GIF when available so the wordmark gently animates on
# the welcome page; falls back to the static PNG otherwise.
_LOGO_TEXT_ANIM_SRC = _IMG_DIR / "logo_animated.gif"
_LOGO_TEXT_STATIC_SRC = _IMG_DIR / "logo_with_text.png"
# URL chosen by which source file exists; copy happens in
# _init_static_assets().
if _LOGO_TEXT_ANIM_SRC.exists():
    _LOGO_TEXT_URL = "golgi_static/logo_animated.gif"
elif _LOGO_TEXT_STATIC_SRC.exists():
    _LOGO_TEXT_URL = "golgi_static/logo_with_text.png"
else:
    _LOGO_TEXT_URL = ""

# Browser-tab favicon — the wordmark on a coloured background so
# the icon stays legible at 16×16. loader_variants.js wires this
# into <link rel="icon"> at page-ready time (trame's module
# system doesn't expose a direct head-link API, so we inject it
# from JS).
_FAVICON_SRC = _IMG_DIR / "logo_with_bg.png"
_FAVICON_STATIC = _GOLGI_STATIC_DIR / "favicon.png"
_FAVICON_URL = "golgi_static/favicon.png"

# Inline icon used on the "Documentation" button. Recoloured to
# white at render time via `filter: brightness(0) invert(1)` so we
# don't have to ship a recoloured variant.
_EXT_LINK_SRC = _IMG_DIR / "docs_icon.svg"
_EXT_LINK_STATIC = _GOLGI_STATIC_DIR / "docs_icon.svg"
_EXT_LINK_URL = "golgi_static/docs_icon.svg"

# Companion icon for the welcome page's primary CTA. Same
# recolour-via-filter pattern as _EXT_LINK_URL.
_EXT_SITE_SRC = _IMG_DIR / "start_icon.svg"
_EXT_SITE_STATIC = _GOLGI_STATIC_DIR / "start_icon.svg"
_EXT_SITE_URL = "golgi_static/start_icon.svg"

# Login icon — used on the welcome-view CTA when the user is
# logged out ("Sign in" button). Mirrors the same copy-into-
# static-dir pattern as `_EXT_SITE_URL` above.
_LOGIN_ICON_SRC = _IMG_DIR / "login_icon.svg"
_LOGIN_ICON_STATIC = _GOLGI_STATIC_DIR / "login_icon.svg"
_LOGIN_ICON_URL = "golgi_static/login_icon.svg"

# Pencil-edit icon used next to the project name in the details
# lightbox. Same recolour trick — the icon ships with a dark stroke
# (#464455) which we override via filter at render time.
_EDIT_ICON_SRC = _IMG_DIR / "edit_icon.svg"
_EDIT_ICON_STATIC = _GOLGI_STATIC_DIR / "edit_icon.svg"
_EDIT_ICON_URL = "golgi_static/edit_icon.svg"

# Loader-variant cycler + SVG-filter recolour injection. The
# loader uses `filter: ... url(#loader-recolor)` to recolour its
# black wave shapes into magma red via a precomputed
# feColorMatrix. The matrix is row N = (1 − channel_N) on the
# diagonal, channel_N in column 5 (offset) — so black maps to
# (R, G, B) and white stays white. Five variant classes (v1..v5)
# swap the animation; cycle them every 10 s with random no-repeat.
_GOLGI_LOADER_JS = r"""
(function () {
  // Set the browser-tab title + favicon. trame's module system
  // injects scripts/styles but doesn't expose a hook for raw
  // <link rel="icon"> tags, so we install ours at page load
  // and remove any default favicons the trame template ships.
  function installBranding() {
    if (document.title !== 'GOLGI.IO') document.title = 'GOLGI.IO';
    document.querySelectorAll(
      'link[rel*="icon"]'
    ).forEach(function (el) {
      if (!el.hasAttribute('data-golgi')) el.remove();
    });
    if (!document.querySelector('link[rel="icon"][data-golgi]')) {
      const link = document.createElement('link');
      link.rel = 'icon';
      link.type = 'image/png';
      link.href = 'golgi_static/favicon.png';
      link.setAttribute('data-golgi', '1');
      document.head.appendChild(link);
    }
  }

  // Tailwind Play CDN loader. Set the `tailwind` config global
  // BEFORE the script loads so the CDN picks it up on init —
  // notably `corePlugins.preflight = false` to keep Tailwind's
  // aggressive CSS reset from clobbering the existing Vuetify +
  // custom styles. Also extends `theme.animation` with the
  // `border` keyframe used by the SaaS-style animated-gradient
  // border pattern. Safe to call once: bail if already loaded.
  function installTailwind() {
    if (document.querySelector('script[data-golgi-tailwind]')) {
      return;
    }
    if (!window.tailwind) {
      window.tailwind = {};
    }
    window.tailwind.config = {
      corePlugins: { preflight: false },
      theme: {
        extend: {
          animation: {
            'border': 'border 4s linear infinite',
          },
          keyframes: {
            'border': {
              to: { '--border-angle': '360deg' },
            },
          },
        },
      },
    };
    const tw = document.createElement('script');
    tw.src = 'https://cdn.tailwindcss.com';
    tw.setAttribute('data-golgi-tailwind', '1');
    document.head.appendChild(tw);
  }

  const FILTER_ID = 'loader-recolor';
  const SVG_NS = 'http://www.w3.org/2000/svg';
  // (R, G, B) for the target loader colour, precomputed for #e24b4a.
  const MATRIX_VALUES = (
    '0.114 0 0 0 0.886 ' +
    '0 0.706 0 0 0.294 ' +
    '0 0 0.710 0 0.290 ' +
    '0 0 0 1 0'
  );

  function ensureSvgFilter() {
    if (document.getElementById(FILTER_ID)) return;
    const svg = document.createElementNS(SVG_NS, 'svg');
    svg.setAttribute('style',
      'position: absolute; width: 0; height: 0; overflow: hidden;'
    );
    svg.setAttribute('aria-hidden', 'true');
    const filter = document.createElementNS(SVG_NS, 'filter');
    filter.setAttribute('id', FILTER_ID);
    filter.setAttribute('color-interpolation-filters', 'sRGB');
    const m = document.createElementNS(SVG_NS, 'feColorMatrix');
    m.setAttribute('type', 'matrix');
    m.setAttribute('values', MATRIX_VALUES);
    filter.appendChild(m);
    svg.appendChild(filter);
    document.body.appendChild(svg);
  }

  const variants = ['v1', 'v2', 'v3', 'v4', 'v5'];
  let currentIdx = 4;

  function apply(idx) {
    document.querySelectorAll('.loader').forEach(function (el) {
      variants.forEach(function (v) { el.classList.remove(v); });
      el.classList.add(variants[idx]);
    });
    currentIdx = idx;
  }

  function start() {
    installBranding();
    installTailwind();
    ensureSvgFilter();
    if (!document.querySelector('.loader')) {
      // Vue hasn't mounted the lightbox yet — retry shortly.
      // Re-run installBranding in the retry loop so trame's
      // late-loaded HTML can't blow away our favicon/title.
      setTimeout(start, 250);
      return;
    }
    apply(currentIdx);  // sync DOM class with our index
    setInterval(function () {
      let next;
      do {
        next = Math.floor(Math.random() * variants.length);
      } while (next === currentIdx);
      apply(next);
    }, 10000);
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
"""
_GOLGI_JS_FILE = _GOLGI_STATIC_DIR / "loader_variants.js"


# Per-tile Plotly export helper exposed as `window.golgi_export_plot`.
# The Solve-tab tiles call this with `(tileId, format)` so the user
# can save PNG / SVG snapshots of any Plotly figure without leaving
# the page. Looks up the .js-plotly-plot div inside the addressed
# tile and delegates to Plotly.downloadImage; no-ops gracefully when
# the global isn't ready yet (Vue mount races).
_GOLGI_EXPORT_JS = """
(function () {
  window.golgi_export_plot = function (tileId, format) {
    var root = document.getElementById(tileId);
    if (!root) {
      console.warn('[golgi-export] tile not found:', tileId);
      return;
    }
    var gd = root.querySelector('.js-plotly-plot');
    if (!gd) {
      console.warn(
        '[golgi-export] no plotly div in tile:', tileId
      );
      return;
    }
    if (!window.Plotly) {
      console.warn('[golgi-export] Plotly global missing');
      return;
    }
    var opts = {
      format: (format || 'png'),
      filename: tileId,
      scale: 2,
    };
    try {
      window.Plotly.downloadImage(gd, opts);
    } catch (err) {
      console.warn('[golgi-export] downloadImage failed', err);
    }
  };
})();
"""
_GOLGI_EXPORT_FILE = _GOLGI_STATIC_DIR / "golgi_export.js"


# F2.2 — study-bundle uploader. POSTs the picked file directly to
# /api/study/upload via XHR (bypassing the WS msgpack ArrayBuffer
# cap that wrecks big-bundle uploads through VFileInput). Exposes
# `window.golgi_study_upload(files)` which the Vuetify
# `v-file-input` change handler invokes with the File[] array.
_GOLGI_STUDY_UPLOAD_JS = r"""
(function () {
  // trame-client 3.12 exposes the live state proxy at
  // `window.trame.state` (post-connect). Earlier versions of
  // this helper checked `window.trame.app || window.trame` and
  // then `tr.state.update` — but `window.trame.app` is the
  // Vue app instance and DOESN'T carry a `.state.update()`
  // method, so the check silently failed on every push and
  // the upload progress never made it into the busy lightbox.
  function setTrameStatePatch(patch) {
    var s = (window.trame && window.trame.state) || null;
    if (s && typeof s.update === "function") {
      s.update(patch);
      return true;
    }
    // Fallback: direct setter on each key. trame's $i class
    // also exposes `set(key, value)`. If even that's missing,
    // give up and warn once per call.
    if (s && typeof s.set === "function") {
      Object.keys(patch).forEach(function (k) {
        s.set(k, patch[k]);
      });
      return true;
    }
    console.warn(
      "[golgi-upload] could not push state — " +
        "window.trame.state.update missing"
    );
    return false;
  }
  function setTrameState(key, value) {
    var p = {};
    p[key] = value;
    return setTrameStatePatch(p);
  }

  window.golgi_study_upload = function (files) {
    var file = null;
    if (Array.isArray(files)) {
      file = files[0];
    } else if (files instanceof FileList) {
      file = files[0];
    } else if (files instanceof File) {
      file = files;
    } else if (files && files.target && files.target.files) {
      file = files.target.files[0];
    }
    if (!file) {
      console.warn("[golgi-upload] no file in change payload");
      return;
    }
    console.log(
      "[golgi-upload] uploading", file.name,
      "(" + (file.size / 1024 / 1024).toFixed(2) + " MB)"
    );
    setTrameState("study_import_uploading", true);
    setTrameState("study_import_pending_progress", 0);
    setTrameState("study_import_pending_status", "Uploading…");
    setTrameState("study_import_pending_error", "");
    setTrameState("study_import_ready", false);
    setTrameState("study_import_manifest_summary", "");

    var fd = new FormData();
    fd.append("file", file, file.name);

    var xhr = new XMLHttpRequest();
    xhr.upload.onprogress = function (e) {
      if (!e.lengthComputable) { return; }
      var pct = Math.round((e.loaded / e.total) * 100);
      setTrameState("study_import_pending_progress", pct);
      setTrameState(
        "study_import_pending_status",
        "Uploading… " + pct + "% (" +
          (e.loaded / 1024 / 1024).toFixed(1) + " / " +
          (e.total / 1024 / 1024).toFixed(1) + " MB)"
      );
    };
    xhr.onerror = function () {
      console.error("[golgi-upload] network error");
      setTrameState(
        "study_import_pending_error",
        "Upload failed — network error."
      );
      setTrameState("study_import_uploading", false);
    };
    xhr.onload = function () {
      if (xhr.status < 200 || xhr.status >= 300) {
        var msg = "HTTP " + xhr.status;
        try {
          var j = JSON.parse(xhr.responseText);
          if (j && j.error) { msg += " — " + j.error; }
        } catch (_e) {}
        setTrameState("study_import_pending_error", msg);
        setTrameState("study_import_uploading", false);
        return;
      }
      var data = {};
      try {
        data = JSON.parse(xhr.responseText);
      } catch (e) {
        setTrameState(
          "study_import_pending_error",
          "Bad JSON from server."
        );
        setTrameState("study_import_uploading", false);
        return;
      }
      setTrameState(
        "study_import_pending_status", "Reading manifest…"
      );
      setTrameState("study_import_path_on_disk", data.path);
    };
    xhr.open("POST", "/api/study/upload");
    xhr.send(fd);
  };
})();
"""
_GOLGI_STUDY_UPLOAD_FILE = (
    _GOLGI_STATIC_DIR / "golgi_study_upload.js"
)

# V1 Phase A.7 — µCT/medical-image uploader. Same XHR-bypass-WS
# trick as study uploads, posting to /api/uct/upload. The server
# callback (registered in build_app) sets state.uct_file_path +
# kicks off do_load_uct_stack as soon as the bytes are on disk,
# so the dialog auto-advances to "slice scrubber" without a
# second click.
_GOLGI_UCT_UPLOAD_JS = r"""
(function () {
  // See _GOLGI_STUDY_UPLOAD_JS for the rationale — trame-
  // client 3.12 exposes the live state at `window.trame.state`
  // (post-connect), NOT on `window.trame.app`. Going through
  // app silently no-op'd every push.
  function setTrameStatePatch(patch) {
    var s = (window.trame && window.trame.state) || null;
    if (s && typeof s.update === "function") {
      s.update(patch);
      return true;
    }
    if (s && typeof s.set === "function") {
      Object.keys(patch).forEach(function (k) {
        s.set(k, patch[k]);
      });
      return true;
    }
    console.warn(
      "[golgi-uct-upload] could not push state — " +
        "window.trame.state.update missing"
    );
    return false;
  }
  function setTrameState(key, value) {
    var p = {};
    p[key] = value;
    return setTrameStatePatch(p);
  }

  window.golgi_uct_upload = function (files) {
    // Normalise to a real Array so the same code handles every
    // plumbed source: FileList from <input multiple>, a single
    // File, the drag-and-drop dataTransfer.files, or an Array
    // (some Vue plumbing wraps it).
    var list = [];
    if (Array.isArray(files)) {
      list = files.slice();
    } else if (files instanceof FileList) {
      for (var i = 0; i < files.length; i++) {
        list.push(files[i]);
      }
    } else if (files instanceof File) {
      list = [files];
    } else if (files && files.target && files.target.files) {
      var fl = files.target.files;
      for (var j = 0; j < fl.length; j++) {
        list.push(fl[j]);
      }
    }
    list = list.filter(function (f) { return f; });
    if (list.length === 0) {
      console.warn("[golgi-uct-upload] no file in change payload");
      return;
    }
    var totalBytes = 0;
    list.forEach(function (f) { totalBytes += f.size; });
    var label = list.length === 1
      ? list[0].name
      : list.length + " files (" +
        (totalBytes / 1024 / 1024).toFixed(2) + " MB)";
    console.log("[golgi-uct-upload] uploading", label);
    setTrameState("uct_uploading", true);
    setTrameState("uct_upload_progress", 0);
    setTrameState(
      "uct_upload_status",
      "Uploading " + label + "…"
    );
    setTrameState("uct_upload_error", "");
    // Drive the dialog's busy lightbox too — it's the
    // overlay the user already associates with "something
    // is happening, don't touch the UI", and it's the only
    // thing visible regardless of which panel they're
    // looking at. The action layer takes over busy_msg /
    // busy_log when the upload finishes (compression +
    // load), so this initial state is just the first phase.
    setTrameState("busy", true);
    setTrameState("busy_msg", "Uploading " + label);
    setTrameState("busy_log", "");

    var fd = new FormData();
    // Append every file under the same `file` field so the
    // server's multipart reader iterates them all into a shared
    // series subdir. Used for DICOM stacks (N .dcm files = one
    // series); single-file uploads still work unchanged.
    list.forEach(function (f) {
      fd.append("file", f, f.name);
    });

    var xhr = new XMLHttpRequest();
    xhr.upload.onprogress = function (e) {
      if (!e.lengthComputable) { return; }
      var pct = Math.round((e.loaded / e.total) * 100);
      var mbLine = (e.loaded / 1024 / 1024).toFixed(1) +
        " / " + (e.total / 1024 / 1024).toFixed(1) + " MB";
      var multiSuffix = list.length > 1
        ? " · " + list.length + " files"
        : "";
      // One batched patch per progress tick — keeps the
      // WS write count to a single message rather than 4.
      setTrameStatePatch({
        "uct_upload_progress": pct,
        "uct_upload_status":
          "Uploading " + pct + "% (" + mbLine + ")" + multiSuffix,
        "busy_msg":
          "Uploading " + label + " · " + pct + "%",
        "busy_log": mbLine + multiSuffix,
      });
    };
    xhr.onerror = function () {
      console.error("[golgi-uct-upload] network error");
      setTrameState(
        "uct_upload_error",
        "Upload failed — network error."
      );
      setTrameState("uct_uploading", false);
      setTrameState("busy", false);
      setTrameState("busy_msg", "");
      setTrameState("busy_log", "");
    };
    xhr.onload = function () {
      if (xhr.status < 200 || xhr.status >= 300) {
        var msg = "HTTP " + xhr.status;
        try {
          var j = JSON.parse(xhr.responseText);
          if (j && j.error) { msg += " — " + j.error; }
        } catch (_e) {}
        setTrameState("uct_upload_error", msg);
        setTrameState("uct_uploading", false);
        setTrameState("busy", false);
        setTrameState("busy_msg", "");
        setTrameState("busy_log", "");
        return;
      }
      // Server-side callback (registered as
      // on_upload_complete) takes over from here: fires
      // do_load_uct_stack, which owns busy_msg / busy_log
      // through the compression + load phases. We just clear
      // the upload-specific fields and hand off the busy
      // overlay (don't reset busy=false here; the action
      // layer flips it when load finishes).
      setTrameState("uct_upload_progress", 100);
      setTrameState("uct_upload_status", "Upload complete.");
      setTrameState("uct_uploading", false);
      setTrameState("busy_msg", "Preparing stack…");
      setTrameState("busy_log", "");
    };
    xhr.open("POST", "/api/uct/upload");
    xhr.send(fd);
  };
})();
"""
_GOLGI_UCT_UPLOAD_FILE = (
    _GOLGI_STATIC_DIR / "golgi_uct_upload.js"
)

# V1 Phase A — drag-to-crop widget for the segment-µCT dialog.
# Self-contained: no external lib. Three `window.golgi_uct_*`
# functions handle the DOM side of a drag-rectangle crop, plus
# a wheel helper for slice scrolling:
#
#   golgi_uct_crop_start(evt) — mousedown: capture image-space
#                              start coords, install live
#                              preview rectangle
#   golgi_uct_crop_move(evt)  — mousemove: update preview rect
#   golgi_uct_crop_end(evt)   — mouseup: return {x0, x1, y0, y1}
#                              in IMAGE pixel coords (or null
#                              for clicks/aborts). Hides the
#                              preview but does NOT push state.
#
# State mutation (uct_crop_x_range / uct_crop_y_range /
# uct_slice_idx) happens via Vue inline expressions wired in
# the dialog's @mouseup.left and @wheel.prevent attrs — that
# goes through the same reactive sync path as VSlider /
# VRangeSlider v-models, so the server-side @state.change
# watchers fire reliably. Earlier revision tried to push state
# from inside the JS via trame.state.update; the client-side
# UI updated but the server-side watchers didn't always fire,
# so the cropped slice didn't re-render.
_GOLGI_UCT_CROPPER_JS = r"""
(function () {
  // Closure state for the in-flight drag.
  var startClient = null;
  var startImage = null;
  var imgEl = null;
  var panelEl = null;
  var previewEl = null;
  // Cache the CURRENT crop window (in full-image pixel space)
  // captured at mousedown — drives the screen-to-image coord
  // conversion in move + end. The PNG preview shows the
  // CROPPED slice, so screen pixels map to image pixels via
  // the current crop range, not via img.naturalWidth (which
  // is the cropped-region width, not the full image width).
  var dragCropX = null;
  var dragCropY = null;
  // Tool mode captured at mousedown — disambiguates what the
  // gesture should do at mouseup. Paint/erase collect a
  // stroke (list of image-pixel points) instead of a
  // rectangle, and crop_move skips the preview rect in those
  // modes.
  var dragMode = "";
  var strokePoints = [];   // array of {x, y} in image coords

  function findPanel(target) {
    var p = target;
    while (p && p !== document.body) {
      if (
        p.id === "golgi-uct-crop-panel" ||
        (p.classList &&
         p.classList.contains("golgi-uct-crop-panel"))
      ) {
        return { panel: p, img: p.querySelector("img") };
      }
      p = p.parentNode;
    }
    return null;
  }

  // Pick the active view range — zoom if set, otherwise crop.
  // Zoom is a display-only further crop on top of the data
  // crop, so when it's active the displayed <img> shows the
  // ZOOM region (not the full crop). Click coords need to map
  // to the zoom range in that case; otherwise to the crop.
  function activeRange(cropV, zoomV) {
    if (
      zoomV && zoomV.length >= 2
      && zoomV[1] > zoomV[0] + 4
    ) {
      return zoomV;
    }
    return cropV;
  }

  // Compute the ACTUAL VISIBLE image area within the <img>
  // element, accounting for object-fit: contain. The element
  // gets `width:100%;height:100%` so it fills the panel, but
  // when the image's aspect ratio doesn't match, CSS adds
  // letterbox (top/bottom) or pillarbox (left/right) inside
  // the element. getBoundingClientRect() returns the ELEMENT
  // bounds — the visible image is centred within that, so
  // raw `clientX - r.left` overcounts by the letterbox edge.
  //
  // Returns {left, top, width, height} in viewport coords,
  // describing the actual visible image. Falls back to the
  // raw element rect when naturalWidth/Height aren't set
  // yet (image still loading).
  function imageDisplayedArea(img) {
    var r = img.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) { return null; }
    var nw = img.naturalWidth || 0;
    var nh = img.naturalHeight || 0;
    if (!nw || !nh) {
      return {
        left: r.left, top: r.top,
        width: r.width, height: r.height,
      };
    }
    var imgAspect = nw / nh;
    var contAspect = r.width / r.height;
    var dw, dh, ox, oy;
    if (imgAspect > contAspect) {
      // Image wider than container → fit width, top/bottom
      // letterboxing.
      dw = r.width;
      dh = r.width / imgAspect;
      ox = 0;
      oy = (r.height - dh) / 2;
    } else {
      // Image taller → fit height, left/right pillarboxing.
      dw = r.height * imgAspect;
      dh = r.height;
      ox = (r.width - dw) / 2;
      oy = 0;
    }
    return {
      left: r.left + ox,
      top: r.top + oy,
      width: dw,
      height: dh,
    };
  }

  // Screen pixel → full-image pixel. Uses the actual visible
  // image area (not the element bounds) so clicks line up
  // pixel-perfect even when CSS object-fit lets-box the image
  // inside its element. Linear interpolation maps a screen
  // position across the active view range (zoom-or-crop).
  function screenToImage(
    evt, img, cropX, cropY, zoomX, zoomY,
  ) {
    var area = imageDisplayedArea(img);
    if (!area) { return null; }
    var vx = activeRange(cropX, zoomX);
    var vy = activeRange(cropY, zoomY);
    if (!vx || !vy) { return null; }
    var sx = (evt.clientX - area.left) / area.width;
    var sy = (evt.clientY - area.top) / area.height;
    sx = Math.max(0, Math.min(1, sx));
    sy = Math.max(0, Math.min(1, sy));
    var x = Math.round(vx[0] + sx * (vx[1] - vx[0]));
    var y = Math.round(vy[0] + sy * (vy[1] - vy[0]));
    return { x: x, y: y };
  }

  function ensurePreview(panel) {
    var el = panel.querySelector(".golgi-uct-crop-preview");
    if (el) { return el; }
    el = document.createElement("div");
    el.className = "golgi-uct-crop-preview";
    el.style.cssText =
      "position:absolute;border:2px dashed #e24b4a;" +
      "background:rgba(226,75,74,0.18);" +
      "pointer-events:none;z-index:8;display:none;" +
      "box-sizing:border-box;";
    panel.appendChild(el);
    return el;
  }

  function updatePreview(evt) {
    if (!startClient || !panelEl || !previewEl) { return; }
    var rp = panelEl.getBoundingClientRect();
    var x0 = Math.min(startClient.x, evt.clientX) - rp.left;
    var y0 = Math.min(startClient.y, evt.clientY) - rp.top;
    var w = Math.abs(evt.clientX - startClient.x);
    var h = Math.abs(evt.clientY - startClient.y);
    previewEl.style.left = x0 + "px";
    previewEl.style.top = y0 + "px";
    previewEl.style.width = w + "px";
    previewEl.style.height = h + "px";
    previewEl.style.display = "block";
  }

  function resetDrag() {
    startClient = null;
    startImage = null;
    dragCropX = null;
    dragCropY = null;
    // Clear any in-progress pan transform on the image
    // element so the next render snaps back to the natural
    // position. resetDrag() is the single funnel every
    // gesture exits through, so this catches every code path.
    if (imgEl) {
      imgEl.style.transform = "";
    }
    dragMode = "";
    strokePoints = [];
    if (previewEl) { previewEl.style.display = "none"; }
  }

  // start(evt, cropX, cropY, zoomX, zoomY, mode)
  // Vue template now also passes the current tool mode so
  // crop_move / crop_end can dispatch correctly without
  // re-reading state. For paint/erase modes, we initialise
  // strokePoints with the start point and skip the drag-
  // preview rectangle.
  window.golgi_uct_crop_start = function (
    evt, cropX, cropY, zoomX, zoomY, mode,
  ) {
    var info = findPanel(evt.currentTarget);
    if (!info || !info.img) { return; }
    var vx = activeRange(cropX, zoomX);
    var vy = activeRange(cropY, zoomY);
    if (!vx || !vy) { return; }
    dragCropX = [vx[0], vx[1]];
    dragCropY = [vy[0], vy[1]];
    var coords = screenToImage(
      evt, info.img,
      dragCropX, dragCropY,
      null, null,
    );
    if (!coords) { return; }
    imgEl = info.img;
    panelEl = info.panel;
    startClient = { x: evt.clientX, y: evt.clientY };
    startImage = coords;
    // Pan trigger: Alt held at mousedown OR middle-button
    // drag. Alt-drag is the user-preferred gesture (right-drag
    // proved unreliable across browsers — Vue's `.right`
    // modifier and the contextmenu event interfered with the
    // mousedown filter). Left button without Alt uses the
    // active tool mode (crop / zoom / paint / erase / label).
    if (evt.altKey || evt.button === 1) {
      dragMode = "pan";
    } else {
      dragMode = String(mode || "crop");
    }
    strokePoints = [];
    if (dragMode === "paint" || dragMode === "erase") {
      strokePoints.push({ x: coords.x, y: coords.y });
    } else if (dragMode === "pan") {
      // No preview rect for pan; CSS transform on imgEl
      // drives the live feedback in golgi_uct_crop_move.
    } else {
      previewEl = ensurePreview(panelEl);
      previewEl.style.display = "block";
      previewEl.style.width = "0px";
      previewEl.style.height = "0px";
    }
    evt.preventDefault();
  };

  window.golgi_uct_crop_move = function (evt) {
    if (!startClient) { return; }
    if (dragMode === "paint" || dragMode === "erase") {
      // Continuous brush stroke — sample the cursor and add
      // distinct points only (dedup repeats so we don't push
      // 60+ identical pixels for a stationary mouse).
      var pt = screenToImage(
        evt, imgEl, dragCropX, dragCropY, null, null,
      );
      if (!pt) { return; }
      var last = strokePoints[strokePoints.length - 1];
      if (!last || last.x !== pt.x || last.y !== pt.y) {
        strokePoints.push(pt);
      }
      return;
    }
    if (dragMode === "pan") {
      // Live preview via CSS transform on the image element.
      // Cheap: no re-render, no state push, no WebSocket
      // round-trip. The actual range update happens at
      // mouseup via golgi_uct_apply_pan().
      if (imgEl) {
        var dxs = evt.clientX - startClient.x;
        var dys = evt.clientY - startClient.y;
        imgEl.style.transform =
          "translate(" + dxs + "px, " + dys + "px)";
      }
      return;
    }
    updatePreview(evt);
  };

  // ----- Brush cursor indicator -----
  // Replaces the native cursor in paint/erase modes with a
  // circle sized to the actual brush radius. Bound to
  // @mousemove on the panel; receives the current tool mode,
  // brush radius, and crop window so it can map image-pixel
  // radius to on-screen pixels. The crop window matters
  // because the rendered <img> scales the cropped region to
  // fill the panel — display-pixel size depends on the
  // current crop width.
  function ensureBrushCursor(panel) {
    var el = panel.querySelector(".golgi-uct-brush-cursor");
    if (el) { return el; }
    el = document.createElement("div");
    el.className = "golgi-uct-brush-cursor";
    el.style.cssText =
      "position:absolute;pointer-events:none;" +
      "border-radius:50%;box-sizing:border-box;" +
      "z-index:9;display:none;" +
      "border:2px solid #fff;" +
      "box-shadow:0 0 0 1px rgba(0,0,0,0.5);";
    panel.appendChild(el);
    return el;
  }

  // Sets the native CSS cursor on the panel based on the
  // current tool mode AND any active gesture / modifier.
  // Priority order:
  //   1. Active pan drag (dragMode === "pan") → grabbing
  //   2. Alt held (potential pan)             → grab
  //   3. Tool-mode-specific cursor:
  //        zoom-in   → browser magnifying-glass (with + inside)
  //        crosshair → standard "+" select-region cursor
  //        cell      → small dotted-square cursor (suggests
  //                    "select / pick" — used for label mode)
  //        none      → hide (paint/erase use a custom brush
  //                    indicator div instead)
  function setPanelCursor(panel, mode) {
    var c;
    if (dragMode === "pan") {
      c = "grabbing";
    } else if (window._golgi_uct_alt_pressed) {
      c = "grab";
    } else if (mode === "paint" || mode === "erase") {
      c = "none";
    } else if (mode === "zoom") {
      c = "zoom-in";
    } else if (mode === "crop") {
      c = "crosshair";
    } else if (mode === "label") {
      c = "cell";
    } else {
      c = "";
    }
    panel.style.cursor = c;
  }

  // Document-level Alt-key tracker so the panel cursor flips
  // to a "grab" hand the moment the user holds Option/Alt,
  // signalling that an Alt-drag will pan. Bound at module
  // load (once per page); the inside of the listeners read
  // the current tool mode from window.trame.state because the
  // mousemove handler isn't guaranteed to fire between the
  // keypress and the user starting a drag (consider Alt held
  // while the cursor is stationary). The hide-brush-cursor
  // helper below clears the custom paint/erase circle when
  // Alt activates, so the user sees the hand cursor
  // unambiguously instead of two cursors stacked.
  function _currentToolMode() {
    var s = (window.trame && window.trame.state) || null;
    if (!s || typeof s.get !== "function") { return ""; }
    return String(s.get("uct_tool_mode") || "");
  }
  function _hideBrushCursor(panel) {
    var el = panel.querySelector(".golgi-uct-brush-cursor");
    if (el) { el.style.display = "none"; }
  }
  function _refreshCropPanelCursors() {
    var panels = document.querySelectorAll(
      "#golgi-uct-crop-panel, .golgi-uct-crop-panel"
    );
    var mode = _currentToolMode();
    for (var i = 0; i < panels.length; i++) {
      setPanelCursor(panels[i], mode);
      if (window._golgi_uct_alt_pressed) {
        _hideBrushCursor(panels[i]);
      }
    }
  }
  if (!window._golgi_uct_alt_listeners_installed) {
    window._golgi_uct_alt_listeners_installed = true;
    window._golgi_uct_alt_pressed = false;
    document.addEventListener("keydown", function (e) {
      // e.key is "Alt" on all modern browsers per UI Events
      // spec. We also OR in e.altKey to catch synthesised
      // events where key is unset (Mac sometimes reports
      // "Dead" for Option+character pre-emit).
      if (e.key === "Alt" || e.altKey) {
        if (!window._golgi_uct_alt_pressed) {
          window._golgi_uct_alt_pressed = true;
          _refreshCropPanelCursors();
        }
      }
    });
    document.addEventListener("keyup", function (e) {
      if (e.key === "Alt" || !e.altKey) {
        if (window._golgi_uct_alt_pressed) {
          window._golgi_uct_alt_pressed = false;
          _refreshCropPanelCursors();
        }
      }
    });
    // Blur safety net — if the window loses focus while Alt
    // is held (Cmd-Tab away on macOS) the keyup never fires
    // and the cursor would be stuck on "grab". Treat blur
    // as an Alt release.
    window.addEventListener("blur", function () {
      if (window._golgi_uct_alt_pressed) {
        window._golgi_uct_alt_pressed = false;
        _refreshCropPanelCursors();
      }
    });
  }

  window.golgi_uct_brush_cursor = function (
    evt, mode, brushRadius, cropX, cropY, zoomX, zoomY,
  ) {
    var info = findPanel(evt.currentTarget);
    if (!info) { return; }
    var panel = info.panel;
    var img = info.img;
    var el = ensureBrushCursor(panel);
    // Always update the cursor — handles every mode, not just
    // paint/erase. Pulled out into setPanelCursor() so the
    // brush-indicator logic below can focus on the radius
    // math without re-deciding cursor strings.
    setPanelCursor(panel, mode);
    if (mode !== "paint" && mode !== "erase") {
      el.style.display = "none";
      return;
    }
    // Alt held → showing the grab/grabbing cursor; hide the
    // paint/erase brush circle so the user sees one cursor
    // affordance, not two stacked.
    if (window._golgi_uct_alt_pressed || dragMode === "pan") {
      el.style.display = "none";
      return;
    }
    if (!img) {
      el.style.display = "none";
      return;
    }
    // Use the ACTUAL displayed image area (object-fit: contain
    // means the visible image can be narrower / shorter than
    // the element bounds — letterbox / pillarbox).
    var area = imageDisplayedArea(img);
    if (!area || area.width <= 0) {
      el.style.display = "none";
      return;
    }
    // Active range (zoom-if-set else crop) so the on-screen
    // brush size matches what the user will actually affect —
    // when zoomed in the same image-pixel brush covers more
    // display pixels.
    var vx = activeRange(cropX, zoomX);
    if (!vx) { el.style.display = "none"; return; }
    var spanX = (vx[1] - vx[0]) || 1;
    var pxPerImg = area.width / spanX;
    var rPx = Math.max(2, brushRadius * pxPerImg);
    var rect = panel.getBoundingClientRect();
    var sx = evt.clientX - rect.left;
    var sy = evt.clientY - rect.top;
    el.style.left = (sx - rPx) + "px";
    el.style.top = (sy - rPx) + "px";
    el.style.width = (2 * rPx) + "px";
    el.style.height = (2 * rPx) + "px";
    el.style.borderColor =
      mode === "paint" ? "#4caf50" : "#f44336";
    el.style.display = "block";
  };

  // Returns one of:
  //   { type: "crop",   x0, x1, y0, y1 }    — real drag rect
  //   { type: "click",  x, y }              — click-without-drag
  //   { type: "stroke", flat: [is_paint, ts, x0, y0, x1, y1,
  //                            ...] }       — paint/erase
  //   null                                   — aborted gesture
  // All coords are full-image pixel space.
  window.golgi_uct_crop_end = function (evt) {
    if (!startClient || !imgEl || !dragCropX || !dragCropY) {
      resetDrag();
      return null;
    }
    var endImage = screenToImage(
      evt, imgEl, dragCropX, dragCropY,
      null, null,
    );
    var clickStart = startImage;
    var modeAtStart = dragMode;

    // Paint / erase always return a stroke. Single click =
    // 1-point stroke; drag = N-point stroke. Flat array so
    // Vue inline expression can assign it directly to the
    // payload state var without a JS loop.
    if (
      modeAtStart === "paint"
      || modeAtStart === "erase"
    ) {
      if (endImage) {
        var last = strokePoints[strokePoints.length - 1];
        if (
          !last
          || last.x !== endImage.x
          || last.y !== endImage.y
        ) {
          strokePoints.push(endImage);
        }
      }
      var pts = strokePoints.slice();
      resetDrag();
      if (pts.length === 0) { return null; }
      var flat = [
        modeAtStart === "paint" ? 1 : 0,
        Date.now(),
      ];
      for (var i = 0; i < pts.length; i++) {
        flat.push(pts[i].x);
        flat.push(pts[i].y);
      }
      return { type: "stroke", flat: flat };
    }

    // Pan delta is computed from the RAW screen-pixel drag
    // (clientX/Y delta), NOT from screenToImage(end) -
    // screenToImage(start). screenToImage clamps the endpoint
    // to [0, 1] of the visible image, so a drag that ends
    // outside the panel undercounts the delta and the final
    // applied pan is smaller than the live CSS-transform
    // preview — the image visibly snaps backwards on release,
    // which reads as "panning is broken". Compute the screen
    // delta first, then scale by (active-range / displayed-
    // area) so the result is unclamped image pixels.
    if (modeAtStart === "pan") {
      var dxsScreen = evt.clientX - startClient.x;
      var dysScreen = evt.clientY - startClient.y;
      var areaP = imgEl ? imageDisplayedArea(imgEl) : null;
      var spanXp =
        dragCropX ? (dragCropX[1] - dragCropX[0]) : 0;
      var spanYp =
        dragCropY ? (dragCropY[1] - dragCropY[0]) : 0;
      var dxImg = 0, dyImg = 0;
      if (areaP && areaP.width > 0 && spanXp > 0) {
        dxImg = (dxsScreen / areaP.width) * spanXp;
      }
      if (areaP && areaP.height > 0 && spanYp > 0) {
        dyImg = (dysScreen / areaP.height) * spanYp;
      }
      // Sub-pixel drag → snap back, no pan. resetDrag clears
      // imgEl.style.transform so the (zero-effect) live preview
      // doesn't linger.
      if (Math.abs(dxImg) < 1 && Math.abs(dyImg) < 1) {
        resetDrag();
        return null;
      }
      // Real pan — leave imgEl.style.transform in place so the
      // user keeps seeing the dragged-to position while the
      // state update round-trips and the server-side
      // _on_crop_change re-renders the slice. The img's @load
      // handler (golgi_uct_scalebar_update) clears the
      // transform once the new image arrives, giving a clean
      // hand-off with no visible snap-back. Manually clear the
      // rest of the drag state (startClient, dragMode, …)
      // since we're bypassing resetDrag() to preserve the
      // transform.
      startClient = null;
      startImage = null;
      dragCropX = null;
      dragCropY = null;
      dragMode = "";
      strokePoints = [];
      if (previewEl) { previewEl.style.display = "none"; }
      return { type: "pan", dx: dxImg, dy: dyImg };
    }

    if (!endImage) {
      resetDrag();
      return null;
    }
    var x0 = Math.min(clickStart.x, endImage.x);
    var x1 = Math.max(clickStart.x, endImage.x);
    var y0 = Math.min(clickStart.y, endImage.y);
    var y1 = Math.max(clickStart.y, endImage.y);
    resetDrag();
    if (x1 - x0 < 8 || y1 - y0 < 8) {
      return {
        type: "click",
        x: clickStart.x,
        y: clickStart.y,
      };
    }
    return {
      type: "crop", x0: x0, x1: x1, y0: y0, y1: y1,
    };
  };

  // Apply a pan emitted by golgi_uct_crop_end. Shifts the
  // active range (zoom-if-set else crop) by (-dx, -dy) image
  // pixels — dragging right pans the image right, which
  // visually means looking at the LEFT part of it, which
  // means the view range moves LEFT (smaller x). Clamped to
  // image bounds (or crop bounds when shifting a zoom inside
  // a crop). Uses window.trame.state.update so the change
  // round-trips through Trame and the slice re-renders.
  window.golgi_uct_apply_pan = function (dx, dy) {
    var s = (window.trame && window.trame.state) || null;
    if (!s || typeof s.get !== "function") { return; }
    var imgW = Number(s.get("uct_image_orig_w") || 0);
    var imgH = Number(s.get("uct_image_orig_h") || 0);
    if (imgW <= 0 || imgH <= 0) { return; }
    var zx = s.get("uct_zoom_x_range") || [0, 0];
    var zy = s.get("uct_zoom_y_range") || [0, 0];
    var cx = s.get("uct_crop_x_range") || [0, 0];
    var cy = s.get("uct_crop_y_range") || [0, 0];
    var zoomActive = zx[0] !== zx[1];
    var cropActive = cx[0] !== cx[1];

    function clampShift(r0, r1, lo, hi, shift) {
      // Try to shift the [r0, r1] interval by `shift` while
      // keeping it inside [lo, hi]. If the shift would push
      // either endpoint out, clamp the shift so the range
      // stays in-bounds and preserves its width.
      var n0 = r0 + shift;
      var n1 = r1 + shift;
      if (n0 < lo) { var off = lo - n0; n0 += off; n1 += off; }
      if (n1 > hi) { var off2 = n1 - hi; n0 -= off2; n1 -= off2; }
      n0 = Math.max(lo, Math.min(hi, n0));
      n1 = Math.max(lo, Math.min(hi, n1));
      return [n0, n1];
    }

    // Helper: if the pan landed against a boundary so the
    // active range didn't actually change, the server-side
    // state watcher won't re-fire and the img's @load will
    // never reach scalebar_update — leaving the CSS pan-
    // preview transform stuck on the image. Clear it here as
    // a fallback. Same applies when no range is active to
    // pan in the first place.
    function _clearPanTransform() {
      var imgs = document.querySelectorAll(
        "#golgi-uct-crop-panel img, " +
          ".golgi-uct-crop-panel img"
      );
      for (var i = 0; i < imgs.length; i++) {
        imgs[i].style.transform = "";
      }
    }
    function _sameRange(a, b) {
      return a && b &&
        a[0] === b[0] && a[1] === b[1];
    }

    if (zoomActive) {
      // Pan inside zoom — clamp to the surrounding crop (or
      // image bounds when there's no crop yet).
      var bxLo = cropActive ? cx[0] : 0;
      var bxHi = cropActive ? cx[1] : imgW;
      var byLo = cropActive ? cy[0] : 0;
      var byHi = cropActive ? cy[1] : imgH;
      var nzx = clampShift(zx[0], zx[1], bxLo, bxHi, -dx);
      var nzy = clampShift(zy[0], zy[1], byLo, byHi, -dy);
      if (_sameRange(nzx, zx) && _sameRange(nzy, zy)) {
        _clearPanTransform();
        return;
      }
      s.update({
        uct_zoom_x_range: nzx,
        uct_zoom_y_range: nzy,
      });
    } else if (cropActive) {
      // Pan the crop within the image.
      var ncx = clampShift(cx[0], cx[1], 0, imgW, -dx);
      var ncy = clampShift(cy[0], cy[1], 0, imgH, -dy);
      if (_sameRange(ncx, cx) && _sameRange(ncy, cy)) {
        _clearPanTransform();
        return;
      }
      s.update({
        uct_crop_x_range: ncx,
        uct_crop_y_range: ncy,
      });
    } else {
      // Neither zoom nor crop is set — full-image view, nothing
      // to pan. Clear the live preview transform so the image
      // snaps back instead of staying at the dragged offset.
      _clearPanTransform();
    }
  };

  // Reset the crop + zoom ranges to "full image, no zoom".
  // Wired to the crop-panel @dblclick handler so a double-
  // click anywhere on the image clears the current crop /
  // zoom selection. Uses window.trame.state.update so the
  // change round-trips through Trame and the slice re-renders
  // immediately. We also log to console so the user can
  // verify the gesture fired even when the browser dropped
  // the dblclick event (which happens occasionally when the
  // preceding mousedowns / mouseups land on different
  // children of the panel).
  window.golgi_uct_reset_crop = function () {
    var s = (window.trame && window.trame.state) || null;
    if (!s || typeof s.update !== "function") {
      console.warn(
        "[golgi-uct] reset_crop: window.trame.state.update " +
          "not available — page may still be connecting"
      );
      return;
    }
    // Reset to the FULL image extent, not [0, 0]. The server-
    // side _on_crop_change guard bails on degenerate ranges
    // (lo == hi or |hi-lo| < 2) to dodge VRangeSlider transient
    // emissions during a thumb-pin. So pushing [0, 0] leaves
    // the displayed slice cropped while state says "no crop",
    // and every subsequent screenToImage maps everything to
    // (0, 0) — brush / erase / label all silently target the
    // top-left pixel. The right-panel "Reset crop" button
    // works because it pushes [0, uct_crop_max_x]; we do the
    // same here.
    var maxX = Number(s.get("uct_crop_max_x") || 0);
    var maxY = Number(s.get("uct_crop_max_y") || 0);
    if (maxX <= 0 || maxY <= 0) {
      console.warn(
        "[golgi-uct] reset_crop: uct_crop_max_x / _max_y not " +
          "set (stack not loaded?) — skipping reset"
      );
      return;
    }
    s.update({
      uct_crop_x_range: [0, maxX],
      uct_crop_y_range: [0, maxY],
      uct_zoom_x_range: [0, 0],
      uct_zoom_y_range: [0, 0],
    });
    console.log(
      "[golgi-uct] reset_crop: cleared to full extent " +
        maxX + " x " + maxY
    );
  };

  // Mousewheel slice scrubber with accumulation. Native wheel
  // events fire ~10× per touchpad gesture (with inertia tail),
  // so advancing 1 slice per event makes scrolling feel like
  // jumping. Accumulate deltaY until it crosses a threshold
  // (~100 = one notch detent on a discrete mouse wheel; one
  // moderate-velocity touchpad swipe).
  //
  // wheel_step(evt, curIdx, maxIdx) returns the new slice idx
  // when the threshold has been crossed, or null to skip.
  window.golgi_uct_wheel_step = function (evt, curIdx, maxIdx) {
    var d = evt.deltaY;
    if (typeof d !== "number" || !isFinite(d)) { return null; }
    window._uct_wheel_acc = (window._uct_wheel_acc || 0) + d;
    var threshold = 100;
    var acc = window._uct_wheel_acc;
    if (Math.abs(acc) < threshold) { return null; }
    var step = acc > 0 ? 1 : -1;
    window._uct_wheel_acc = 0;
    var next = curIdx + step;
    if (next < 0) { next = 0; }
    if (next > maxIdx) { next = maxIdx; }
    if (next === curIdx) { return null; }
    return next;
  };
})();
"""
_GOLGI_UCT_CROPPER_FILE = (
    _GOLGI_STATIC_DIR / "golgi_uct_cropper.js"
)

# 1 mm physical-scale bar updater for the µCT segmentation
# dialog. Kept in its own bundle (separate from the cropper
# IIFE) so an edit here doesn't trigger the linter pass that
# rewrites the cropper module — every time the scalebar
# function was inlined into _GOLGI_UCT_CROPPER_JS it got
# stripped out within minutes of the next save.
_GOLGI_UCT_SCALEBAR_JS = r"""
(function () {
  // Local copy of the cropper's imageDisplayedArea helper —
  // the cropper IIFE doesn't export it, and duplicating ~30
  // lines is cheaper than coupling the two files.
  function imageDisplayedArea(img) {
    var r = img.getBoundingClientRect();
    if (r.width <= 0 || r.height <= 0) { return null; }
    var nw = img.naturalWidth || 0;
    var nh = img.naturalHeight || 0;
    if (!nw || !nh) {
      return {
        left: r.left, top: r.top,
        width: r.width, height: r.height,
      };
    }
    var imgAspect = nw / nh;
    var contAspect = r.width / r.height;
    var dw, dh, ox, oy;
    if (imgAspect > contAspect) {
      dw = r.width;
      dh = r.width / imgAspect;
      ox = 0;
      oy = (r.height - dh) / 2;
    } else {
      dw = r.height * imgAspect;
      dh = r.height;
      ox = (r.width - dw) / 2;
      oy = 0;
    }
    return {
      left: r.left + ox,
      top: r.top + oy,
      width: dw,
      height: dh,
    };
  }

  // 1 mm physical-scale bar updater.
  //
  // Called from:
  //   1. Img @load — fires after every slice / crop / zoom
  //      because re-rendering the overlay PNG changes the
  //      <img>'s src.
  //   2. MutationObserver on data-voxel-um / data-orig-w —
  //      Vue binds those attrs to the matching state vars,
  //      so editing the voxel-size text field or loading a
  //      new stack re-triggers the observer.
  //   3. ResizeObserver on the panel — dialog drag, window
  //      resize, Vuetify drawer-toggle.
  //
  // Geometry: bar_px = displayed.width * (1000/voxelUm) / origW
  // Sits at the bottom-RIGHT of the displayed image (not the
  // panel — matters when CSS letterboxes the image).
  //
  // Hides itself when voxelUm <= 0, origW <= 0, or the bar
  // would overflow the displayed image width.
  window.golgi_uct_scalebar_update = function (
    img, voxelUm, origW,
  ) {
    if (!img) { return; }
    // The new image bytes have arrived from the server (this
    // handler is wired to @load on the <img>), so any pan
    // CSS-transform we left in place from the most-recent
    // pan-drag release can now be cleared — the new src is
    // already showing the panned crop, so the transform is
    // redundant and would compound visually. See the pan
    // branch in golgi_uct_crop_end for the design rationale.
    img.style.transform = "";
    var panel = img.closest("#golgi-uct-crop-panel");
    if (!panel) { return; }
    var bar = panel.querySelector(".golgi-uct-scalebar");
    if (!bar) { return; }
    if (!panel._golgi_scalebar_installed) {
      panel._golgi_scalebar_installed = true;
      var refire = function () {
        var vum = parseFloat(img.dataset.voxelUm || "0");
        var ow = parseInt(img.dataset.origW || "0", 10);
        window.golgi_uct_scalebar_update(img, vum, ow);
      };
      try {
        var ro = new ResizeObserver(refire);
        ro.observe(panel);
      } catch (e) { /* old browser */ }
      try {
        var mo = new MutationObserver(refire);
        mo.observe(img, {
          attributes: true,
          attributeFilter: [
            "data-voxel-um", "data-orig-w",
          ],
        });
      } catch (e) { /* old browser */ }
    }
    var vu = parseFloat(voxelUm);
    var ow = parseInt(origW, 10);
    if (!(vu > 0) || !(ow > 0)) {
      bar.style.display = "none";
      return;
    }
    var area = imageDisplayedArea(img);
    if (!area || area.width <= 0) {
      bar.style.display = "none";
      return;
    }
    var barPx = area.width * (1000 / vu) / ow;
    if (!(barPx > 0) || barPx > area.width) {
      bar.style.display = "none";
      return;
    }
    var panelRect = panel.getBoundingClientRect();
    bar.style.display = "flex";
    bar.style.left = "";
    bar.style.transform = "";
    bar.style.right =
      (panelRect.right - (area.left + area.width) + 8) + "px";
    bar.style.bottom =
      (panelRect.bottom - (area.top + area.height) + 8) + "px";
    var tick = bar.querySelector(".golgi-uct-scalebar-tick");
    if (tick) { tick.style.width = barPx + "px"; }
  };
})();
"""
_GOLGI_UCT_SCALEBAR_FILE = (
    _GOLGI_STATIC_DIR / "golgi_uct_scalebar.js"
)


# Auto-reload-on-disconnect backstop. The trame_client WS
# layer fires `connection.onclose` whenever the browser-side
# socket gives up (macOS network suspend, browser background
# throttling, etc.) and replaces the entire UI with a
# `<trame-loading message="Connection closed"/>` element.
# Trame has built-in reconnect machinery but a TypeError in
# wslink-js cleanup (`notifyListeners → dirty → set` on
# `e.connection.onclose`) crashes mid-chain and the reconnect
# never fires. We can't fix that bug from the app side, so
# instead we poll for the loading overlay and trigger a soft
# page reload when it appears — since trame keeps every UI
# state var server-side, the reload picks up exactly where
# things were. The user sees a brief flash instead of a
# permanent "Connection closed" screen.
#
# Belt-and-braces fallback also patches WebSocket so that any
# `close` event triggers the same reload — catches the case
# where the trame overlay never paints because the cleanup
# crash happened too early.
_GOLGI_AUTORELOAD_JS = r"""
(function () {
  if (window.__golgi_autoreload_installed) {
    console.log('[golgi-autoreload] already installed, skipping.');
    return;
  }
  window.__golgi_autoreload_installed = true;
  console.log('[golgi-autoreload] script loaded at',
              new Date().toISOString());

  // ---- Constants -------------------------------------------
  // Selector that matches "the app is fully mounted" — once
  // any of these appear, we know wslink connected, state
  // synced, and Vue painted. Used both for "reveal after
  // silent reload" and for "we have seen the app, so any
  // future loading overlay is a reconnect spinner".
  var APP_ROOT_SELECTOR =
    '.golgi-navbar-userchip, .golgi-welcome, .golgi-central';

  // sessionStorage flag set just before a silent reload so the
  // reloaded page can hide trame's "Loading…" overlay until
  // the app is back up.
  var SILENT_RELOAD_KEY = '__golgi_silent_reload';

  // Don't reload if the page was just reloaded — prevents
  // loops if the server itself is down at startup.
  var GRACE_AFTER_LOAD_MS = 15 * 1000;
  var pageLoadedAt = Date.now();

  // ---- Reveal helper ---------------------------------------
  // Used for both Path 0 (silent reload startup) and Path 3
  // (mid-session reconnect spinner). Removes the hide style
  // once the app root is visible (or after `safetyMs` ms).
  function waitForAppThenReveal(hideStyle, label, safetyMs) {
    var startedAt = Date.now();
    var iv = setInterval(function () {
      if (document.querySelector(APP_ROOT_SELECTOR)) {
        clearInterval(iv);
        if (hideStyle.parentNode) hideStyle.remove();
        console.log(
          '[golgi-autoreload] ' + label + ' — app ready after ' +
          (Date.now() - startedAt) + ' ms; revealing.'
        );
      }
    }, 100);
    setTimeout(function () {
      clearInterval(iv);
      if (hideStyle.parentNode) {
        hideStyle.remove();
        console.warn(
          '[golgi-autoreload] ' + label + ' — app did NOT mount ' +
          'in ' + safetyMs + ' ms; revealing anyway to avoid ' +
          'leaving the page blank.'
        );
      }
    }, safetyMs);
  }

  // ---- Path 0: SILENT RELOAD STARTUP -----------------------
  // If this load is the result of our auto-reload, paint a
  // white screen and keep the page hidden until the app is
  // back. Otherwise the user sees trame's "Loading…" spinner
  // mid-cycle. Set BEFORE Path 1/2 so it takes effect even
  // when scheduleReload fires synchronously below.
  if (sessionStorage.getItem(SILENT_RELOAD_KEY) === '1') {
    sessionStorage.removeItem(SILENT_RELOAD_KEY);
    console.log(
      '[golgi-autoreload] silent reload detected — hiding ' +
      'page until app is back.'
    );
    var silentStyle = document.createElement('style');
    silentStyle.id = '__golgi_silent_reload_style';
    silentStyle.textContent =
      'html, body { background: #ffffff !important; } ' +
      'body { visibility: hidden !important; }';
    document.head.appendChild(silentStyle);
    waitForAppThenReveal(silentStyle, 'silent reload', 15000);
  }

  // ---- Path 3: MID-SESSION RECONNECT-SPINNER HIDE ----------
  // Once the app has mounted, trame's own reconnect mechanism
  // may paint a "Loading…" overlay between WS drops and the
  // re-sync. We can't fix wslink-js, but we can hide the
  // overlay so the user doesn't see anything until either
  // (a) trame finishes its own reconnect and the app reappears
  // or (b) we trigger a full reload as a backstop.
  //
  // Strategy: once we've seen the app root once, watch the
  // DOM for new elements that look like trame's session-level
  // loading overlay (NOT the inline button/input spinners,
  // which are nested deep) and hide them.
  var appHasMountedOnce = false;
  function isReconnectOverlay(el) {
    if (!el || el.nodeType !== 1) return false;
    // Match trame's <trame-loading> render output by text.
    // The reconnect overlay has small textContent (a message
    // or just a spinner); the rest of the app has lots.
    var text = (el.textContent || '').trim();
    if (text.length > 200) return false;
    var cls = (el.className || '').toString();
    var lc = cls.toLowerCase();
    // Match obvious trame-loading / vuetify-overlay classes.
    if (lc.indexOf('trame-loading') !== -1) return true;
    if (lc.indexOf('loading-overlay') !== -1) return true;
    if (lc.indexOf('v-overlay--scrim') !== -1) return true;
    // Also match by inner spinner class (Vuetify's
    // v-progress-circular is what trame shows during reconnect).
    if (el.querySelector &&
        el.querySelector('.v-progress-circular, ' +
                         '[class*="loading-spinner"]')) {
      // Only flag it if the element is a top-level overlay
      // (direct body child or in the app root) — otherwise we'd
      // hide every legitimate in-app spinner.
      var p = el.parentElement;
      if (p && (p === document.body ||
                p.className.indexOf('v-application') !== -1 ||
                p.tagName === 'HTML')) {
        return true;
      }
    }
    return false;
  }
  function hideReconnectOverlays() {
    var candidates = document.querySelectorAll(
      'body > *, .v-application > *'
    );
    for (var i = 0; i < candidates.length; i++) {
      if (isReconnectOverlay(candidates[i])) {
        if (candidates[i].style.visibility !== 'hidden') {
          candidates[i].style.visibility = 'hidden';
          console.log(
            '[golgi-autoreload] hiding mid-session reconnect ' +
            'overlay:', candidates[i].className || '<no class>'
          );
        }
      }
    }
  }
  // Cheap interval poll instead of MutationObserver (the
  // observer's callback would fire on EVERY DOM mutation
  // including every Vue re-render, which is far too noisy).
  setInterval(function () {
    if (!appHasMountedOnce) {
      if (document.querySelector(APP_ROOT_SELECTOR)) {
        appHasMountedOnce = true;
        console.log(
          '[golgi-autoreload] app mounted; reconnect-spinner ' +
          'suppression now active.'
        );
      }
      return;
    }
    hideReconnectOverlays();
  }, 250);

  // ---- Reload trigger --------------------------------------
  // Based on user observation: trame's own reconnect does NOT
  // recover in this deployment — the spinner sits forever
  // unless we force a reload. So fire as soon as any disconnect
  // signal appears (no 12s grace).
  var reloadScheduled = false;
  function scheduleReload(reason) {
    if (reloadScheduled) return;
    if (Date.now() - pageLoadedAt < GRACE_AFTER_LOAD_MS) {
      console.log(
        '[golgi-autoreload] disconnect (' + reason +
        ') but within ' + GRACE_AFTER_LOAD_MS +
        ' ms of page load — NOT reloading (prevents boot loop).'
      );
      return;
    }
    reloadScheduled = true;
    console.log(
      '[golgi-autoreload] reloading (' + reason +
      ') — silent reload flagged.'
    );
    sessionStorage.setItem(SILENT_RELOAD_KEY, '1');
    var hideStyle = document.createElement('style');
    hideStyle.id = '__golgi_autoreload_hide';
    hideStyle.textContent =
      'html, body { background: #ffffff !important; } ' +
      'body { visibility: hidden !important; }';
    document.head.appendChild(hideStyle);
    setTimeout(function () { window.location.reload(); }, 150);
  }

  // Path 1: poll the DOM for "Connection closed" text. Fires
  // ASAP — trame paints this state when its wslink-js detects
  // the close, regardless of whether the user can see the text
  // (it's usually behind the spinner). This is what actually
  // triggered reloads in earlier versions, even though the
  // user "never saw Connection closed" visually.
  function detectClosedOverlay() {
    var candidates = document.querySelectorAll(
      'body > div, body > div > div, .v-application, ' +
      '[class*="loading"], [class*="Loading"]'
    );
    for (var i = 0; i < candidates.length; i++) {
      var t = (candidates[i].textContent || '').trim();
      if (t.length > 100) continue;
      if (t.indexOf('Connection closed') !== -1) return true;
    }
    return false;
  }
  setInterval(function () {
    if (reloadScheduled) return;
    if (detectClosedOverlay()) {
      scheduleReload('"Connection closed" text in overlay');
    }
  }, 500);

  // Path 4: if Path 3 has been hiding the reconnect overlay
  // for >3 s without it going away, trame isn't going to
  // reconnect on its own — fall through to a silent reload.
  // This catches the case where the overlay carries a spinner
  // but no "Connection closed" text (so Path 1 misses it).
  var spinnerFirstHiddenAt = null;
  setInterval(function () {
    if (reloadScheduled) return;
    var visible = false;
    var candidates = document.querySelectorAll(
      'body > *, .v-application > *'
    );
    for (var i = 0; i < candidates.length; i++) {
      if (isReconnectOverlay(candidates[i])) {
        visible = true;
        break;
      }
    }
    if (visible) {
      if (spinnerFirstHiddenAt === null) {
        spinnerFirstHiddenAt = Date.now();
      } else if (Date.now() - spinnerFirstHiddenAt > 3000) {
        scheduleReload('reconnect spinner persisted >3s');
      }
    } else {
      spinnerFirstHiddenAt = null;
    }
  }, 500);

  // Path 2: wrap WebSocket to catch the raw close. Only useful
  // if this script loads BEFORE trame's main bundle. In the
  // current trame_client + vite build it loads after — you
  // won't see a "WS open" log line — so Path 2 is dormant.
  // Left in for environments where script load order is more
  // favourable (or future trame versions that expose the WS
  // lazily).
  var OrigWS = window.WebSocket;
  if (OrigWS) {
    console.log(
      '[golgi-autoreload] wrapping window.WebSocket (Path 2).'
    );
    var WrappedWS = function (url, protocols) {
      console.log('[golgi-autoreload] WS open:', url);
      var ws = protocols !== undefined
        ? new OrigWS(url, protocols) : new OrigWS(url);
      ws.addEventListener('close', function (ev) {
        console.log(
          '[golgi-autoreload] WS close fired url=' + ws.url +
          ' code=' + ev.code + ' clean=' + ev.wasClean
        );
        try {
          if (ws.url.indexOf(window.location.host) < 0) return;
        } catch (e) { return; }
        scheduleReload(
          'WS close code=' + ev.code + ' clean=' + ev.wasClean
        );
      });
      return ws;
    };
    WrappedWS.prototype = OrigWS.prototype;
    WrappedWS.CONNECTING = OrigWS.CONNECTING;
    WrappedWS.OPEN = OrigWS.OPEN;
    WrappedWS.CLOSING = OrigWS.CLOSING;
    WrappedWS.CLOSED = OrigWS.CLOSED;
    window.WebSocket = WrappedWS;
  }
})();
"""


# Cache-bust via content hash so an asset edit forces a refetch
# instead of the browser serving a stale copy (same trick as
# _GOLGI_VANTA_FNAME below). The previous static-name version
# hit exactly this: the user's browser kept loading the old
# script and the disconnect path was never triggered.
# (Local import — `_hashlib` is imported once further down for
# vanta; pull it in here so this section is order-independent.)
import hashlib as _autoreload_hashlib
_GOLGI_AUTORELOAD_HASH = _autoreload_hashlib.sha1(
    _GOLGI_AUTORELOAD_JS.encode("utf-8"),
).hexdigest()[:10]
_GOLGI_AUTORELOAD_FNAME = (
    f"golgi_autoreload_{_GOLGI_AUTORELOAD_HASH}.js"
)
_GOLGI_AUTORELOAD_FILE = (
    _GOLGI_STATIC_DIR / _GOLGI_AUTORELOAD_FNAME
)


# Vanta FOG initialiser — mounts the WebGL fog effect on the
# welcome page once `.golgi-welcome` exists in the DOM and the
# VANTA global has been loaded from CDN. Polls for both because
# trame's Vue mount + the async <script> loads from cdnjs/jsdelivr
# both race against page-ready, and we can't assume an order. Idle
# CPU cost is negligible — Vanta only re-renders while the page
# is visible. Colours intentionally match the navy gradient still
# used on the workspace shell (.golgi-central) so swapping between
# welcome and workspace stays visually cohesive.
# Feature flag — flip to `True` to re-enable the WebGL fog
# backdrop on the welcome view. Disabled for now: the static
# animated CSS-gradient backdrop alone is enough, and Vanta
# adds ~700 KB of CDN-loaded JS plus a continuously-running
# WebGL render loop on the welcome view. When False, the
# Vanta mount script just no-ops and the CSS gradient
# fallback shows on its own.
_GOLGI_VANTA_ENABLED = False

_GOLGI_VANTA_JS = (
    "/* Vanta disabled — _GOLGI_VANTA_ENABLED = False at "
    "module-level. Re-enable to bring back the WebGL fog. */\n"
    "/* no-op */\n"
) if not _GOLGI_VANTA_ENABLED else """
/* Vanta FOG mount for golgi's welcome page.
 *
 * IMPORTANT — we LOAD three.js + vanta.fog OURSELVES via
 * dynamically-injected <script> tags instead of relying on
 * trame's Module.scripts list. Reason: the trame vue3 bundle
 * appears to put a stripped-down `THREE` namespace on window
 * before our CDN three.js gets a chance to run, so
 * `window.THREE` exists but `THREE.Color` is undefined,
 * which crashes Vanta's FOG constructor at
 *   `Cannot read properties of undefined (reading 'Color')`.
 * Dynamic injection guarantees the real UMD three.js attaches
 * its full API to window AFTER any prior stripped namespace
 * was set, and Vanta then picks it up cleanly. */
(function () {
  var THREE_URL =
    "https://cdn.jsdelivr.net/npm/three@0.134.0/build/three.min.js";
  var VANTA_URL =
    "https://cdn.jsdelivr.net/npm/vanta@0.5.24/dist/vanta.fog.min.js";
  var instance = null;
  var resizeBound = false;

  function loadScript(url, cb) {
    // Reuse an existing tag if a previous load already started
    // for this URL (e.g. on Vue hot-reload during dev).
    var existing = document.querySelector(
      'script[data-golgi-src="' + url + '"]'
    );
    if (existing) {
      if (existing.dataset.golgiReady === '1') {
        cb();
      } else {
        existing.addEventListener('load', cb);
      }
      return;
    }
    var s = document.createElement('script');
    s.src = url;
    s.async = false;          // preserve execution order
    s.dataset.golgiSrc = url;
    s.onload = function () {
      s.dataset.golgiReady = '1';
      cb();
    };
    s.onerror = function (e) {
      console.error('[golgi-vanta] failed to load', url, e);
    };
    document.head.appendChild(s);
  }

  function haveThree() {
    return typeof window.THREE !== 'undefined'
      && window.THREE.Color
      && window.THREE.WebGLRenderer;
  }
  function haveVanta() {
    return typeof window.VANTA !== 'undefined'
      && window.VANTA.FOG;
  }

  function mount() {
    if (instance) return;
    if (!haveThree()) {
      console.warn(
        '[golgi-vanta] THREE.Color still missing after CDN load '
        + '— the bundle on window is not a UMD three.js. Vanta '
        + 'will stay on the CSS-gradient fallback.'
      );
      return;
    }
    if (!haveVanta()) {
      console.warn('[golgi-vanta] VANTA.FOG missing after CDN load');
      return;
    }
    var el = document.querySelector('.golgi-welcome');
    if (!el) {
      setTimeout(mount, 300);
      return;
    }
    try {
      instance = window.VANTA.FOG({
        el: el,
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.0,
        minWidth: 200.0,
        // Palette matched to the navy gradient on .golgi-central:
        //   baseColor       ≈ hsla(205,63%,11%) → 0x0A2334
        //   lowlightColor   ≈ hsla(205,68%,8%)  → 0x05131F
        //   midtoneColor    ≈ hsla(205,83%,17%) → 0x074A73
        //   highlightColor — soft sky blue accent in the same
        //   family (instead of the white of the old gradient,
        //   which read as a hot spot under the fog blur).
        highlightColor: 0x88c0e8,
        midtoneColor: 0x0f3a5b,
        lowlightColor: 0x041020,
        baseColor: 0x0a2030,
        blurFactor: 0.6,
        zoom: 0.4,
        speed: 1.0
      });
      console.log('[golgi-vanta] ready');
    } catch (err) {
      console.warn('[golgi-vanta] init failed', err);
      return;
    }
    if (!resizeBound) {
      window.addEventListener('resize', function () {
        if (instance && instance.resize) instance.resize();
      });
      resizeBound = true;
    }
  }

  function start() {
    // Force-load our own three.js even if `window.THREE`
    // already exists — the trame bundle ships a partial
    // namespace that lacks the `.Color` ctor, and we'd rather
    // overwrite it with the real UMD payload than try to use
    // the broken one. Vanta then picks the real THREE up.
    loadScript(THREE_URL, function () {
      loadScript(VANTA_URL, mount);
    });
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', start);
  } else {
    start();
  }
})();
"""
# Cache-bust the static init script via its content hash so the
# browser refetches it whenever the JS body changes. Without this,
# Chrome / Safari aggressively cache `golgi_vanta.js` and the user
# can stay on a stale init for hours after a server restart.
import hashlib as _hashlib
_GOLGI_VANTA_HASH = _hashlib.sha1(
    _GOLGI_VANTA_JS.encode("utf-8"),
).hexdigest()[:10]
_GOLGI_VANTA_FNAME = f"golgi_vanta_{_GOLGI_VANTA_HASH}.js"
_GOLGI_VANTA_FILE = _GOLGI_STATIC_DIR / _GOLGI_VANTA_FNAME


def _init_static_assets() -> None:
    """Create _golgi_assets/ + copy bundled CSS/logos/SVG files
    + write the three inline-JS helper files into it. Idempotent
    (mkdir(exist_ok=True), shutil.copyfile, write_text are all
    safe to repeat). Deferred from module-load so `import golgi`
    no longer performs 10+ filesystem writes."""
    _GOLGI_STATIC_DIR.mkdir(exist_ok=True)

    # CSS sidecar — fall back to an empty stylesheet when the
    # source is missing so the static-mount initialises cleanly
    # and the app boots; styling will be missing but the user
    # gets a usable UI to diagnose with.
    if _GOLGI_CSS_SOURCE.is_file():
        shutil.copyfile(_GOLGI_CSS_SOURCE, _GOLGI_CSS_FILE)
    else:
        print(
            f"[css] sidecar not found at {_GOLGI_CSS_SOURCE} — "
            f"running without app stylesheet",
            flush=True,
        )
        _GOLGI_CSS_FILE.write_text("", encoding="utf-8")

    # Single-source asset copies — each guarded by .exists() so a
    # partial install doesn't crash startup.
    for src, dst in (
        (_LOGO_SRC,        _LOGO_STATIC),
        (_FAVICON_SRC,     _FAVICON_STATIC),
        (_EXT_LINK_SRC,    _EXT_LINK_STATIC),
        (_EXT_SITE_SRC,    _EXT_SITE_STATIC),
        (_LOGIN_ICON_SRC,  _LOGIN_ICON_STATIC),
        (_EDIT_ICON_SRC,   _EDIT_ICON_STATIC),
    ):
        if src.exists():
            shutil.copyfile(src, dst)

    # Text logo — prefer animated GIF, fall back to static PNG.
    # The URL is determined at module-load time (in the
    # _LOGO_TEXT_URL constant); this just materialises whichever
    # source the URL points at.
    if _LOGO_TEXT_ANIM_SRC.exists():
        shutil.copyfile(
            _LOGO_TEXT_ANIM_SRC,
            _GOLGI_STATIC_DIR / "logo_animated.gif",
        )
    elif _LOGO_TEXT_STATIC_SRC.exists():
        shutil.copyfile(
            _LOGO_TEXT_STATIC_SRC,
            _GOLGI_STATIC_DIR / "logo_with_text.png",
        )

    # Inline JS bundles — content lives in the *_JS string
    # constants above; materialised here so trame can serve them.
    _GOLGI_JS_FILE.write_text(_GOLGI_LOADER_JS, encoding="utf-8")
    _GOLGI_EXPORT_FILE.write_text(_GOLGI_EXPORT_JS, encoding="utf-8")
    _GOLGI_STUDY_UPLOAD_FILE.write_text(_GOLGI_STUDY_UPLOAD_JS, encoding="utf-8")
    _GOLGI_UCT_UPLOAD_FILE.write_text(_GOLGI_UCT_UPLOAD_JS, encoding="utf-8")
    _GOLGI_UCT_CROPPER_FILE.write_text(_GOLGI_UCT_CROPPER_JS, encoding="utf-8")
    _GOLGI_UCT_SCALEBAR_FILE.write_text(_GOLGI_UCT_SCALEBAR_JS, encoding="utf-8")
    _GOLGI_VANTA_FILE.write_text(_GOLGI_VANTA_JS, encoding="utf-8")
    _GOLGI_AUTORELOAD_FILE.write_text(_GOLGI_AUTORELOAD_JS, encoding="utf-8")

    # Prune any older content-hashed vanta + autoreload copies
    # so the static directory doesn't accumulate dead files
    # across server restarts.
    for _stale in _GOLGI_STATIC_DIR.glob("golgi_vanta_*.js"):
        if _stale.name != _GOLGI_VANTA_FNAME:
            try:
                _stale.unlink()
            except Exception:
                pass
    for _stale in _GOLGI_STATIC_DIR.glob("golgi_autoreload_*.js"):
        if _stale.name != _GOLGI_AUTORELOAD_FNAME:
            try:
                _stale.unlink()
            except Exception:
                pass
    # Also clean up the legacy unhashed name from the first
    # version of this asset.
    _legacy = _GOLGI_STATIC_DIR / "golgi_autoreload.js"
    if _legacy.exists():
        try:
            _legacy.unlink()
        except Exception:
            pass


class _GolgiAssetsModule:
    """Trame module — exposes the loader CSS as a stylesheet link
    and the variant-cycler JS as a <script src=...> in the head.
    The Vanta FOG background on the welcome page is handled
    entirely by `golgi_vanta_<hash>.js`, which dynamically loads
    three.js + vanta.fog from CDN on its own. We DELIBERATELY do
    NOT include those CDN URLs in this `scripts` list — trame's
    vue3 bundle appears to set a stripped-down `THREE` namespace
    on window before module scripts run, which crashes Vanta's
    FOG constructor (`Cannot read properties of undefined
    (reading 'Color')`). Loading them ourselves from a
    user-script context overrides that partial namespace with the
    real UMD payload."""
    serve = {"golgi_static": str(_GOLGI_STATIC_DIR)}
    styles = ["golgi_static/loader.css"]
    # Autoreload script FIRST so it monkey-patches WebSocket
    # before trame's wslink JS opens its connection. Otherwise
    # the original WebSocket constructor is captured by wslink
    # at module load and our patch never sees the trame WS.
    # Filename carries a content hash for cache-busting.
    scripts = [
        f"golgi_static/{_GOLGI_AUTORELOAD_FNAME}",
        "golgi_static/loader_variants.js",
        "golgi_static/golgi_export.js",
        "golgi_static/golgi_study_upload.js",
        "golgi_static/golgi_uct_upload.js",
        "golgi_static/golgi_uct_cropper.js",
        "golgi_static/golgi_uct_scalebar.js",
        f"golgi_static/{_GOLGI_VANTA_FNAME}",
    ]
    state: dict = {}
    vue_use: list = []


def _setup_golgi_assets(server):
    server.enable_module(_GolgiAssetsModule)


# ---------------------------------------------------------------------------
# Project bundle helpers — one folder per project under PROJECTS_ROOT.
#
# Folder layout:
#   <project>/
#     project.json               manifest (name, created, last_modified,
#                                source_file, stage_completed, ui_state)
#     thumbnail.png              auto-screenshot of the 3D viewport
#     source/<orig>.<ext>        copied-in nerve geometry (portable)
#     uploads/                   per-project upload sink (replaces UPLOAD_DIR)
#     nerve.msh, *.npz, *.json   subprocess outputs land here directly
#                                (project root IS the GOLGI_OUT for the
#                                 currently-active project)
#
# `_activate_project(pdir)` rebinds the module-level GOLGI_OUT + UPLOAD_DIR
# at runtime. Existing references inside build_app()'s closures look the
# globals up at call time, so rebinding works without rewriting them.
# ---------------------------------------------------------------------------
_PROJECT_MANIFEST_VERSION = 1


def _sanitize_proj_dirname(name: str) -> str:
    """Filesystem-safe slug for a project's folder name. Keeps
    alphanumerics + dash/underscore; collapses runs of other chars
    to '_'. Empty or all-special → 'untitled'."""
    s = re.sub(r"[^A-Za-z0-9._-]+", "_", name.strip()).strip("._-")
    return s or "untitled"


def _project_short_uid() -> str:
    return uuid.uuid4().hex[:8]


def _list_projects(owner_user_id: int | None = None) -> list[dict]:
    """Return one entry per valid project under PROJECTS_ROOT,
    sorted by last_modified DESC. Each entry: {name, dir, created,
    last_modified, thumbnail_data_uri, source_file}. Folders
    without a project.json are skipped silently.

    Owner-filtering rules:
      * `owner_user_id is None` (nobody logged in) → return [].
        The welcome view shows the "Please sign in" panel instead
        of leaking the global tile list.
      * `owner_user_id is int` → return projects whose manifest's
        `owner_user_id` matches OR is null (legacy / orphan
        projects — visible to every logged-in user for backward
        compatibility; users can claim them by editing the
        project from the workspace once opened).
    Note: the `_auth` directory (auth DB) lives under
    PROJECTS_ROOT and would otherwise be scanned here. We skip
    underscore-prefixed dirs so it never appears as a tile."""
    out: list[dict] = []
    if not PROJECTS_ROOT.exists():
        return out
    if owner_user_id is None:
        return out
    for pdir in PROJECTS_ROOT.iterdir():
        if not pdir.is_dir():
            continue
        # Skip internal directories like `_auth` so they never
        # appear as project tiles.
        if pdir.name.startswith("_") or pdir.name.startswith("."):
            continue
        manifest_path = pdir / "project.json"
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        # Access filter — show the project if the current user
        # OWNS it OR is in its `shared_user_ids` list. Legacy
        # projects with no owner field stay visible to every
        # logged-in user (backward compatibility for projects
        # created before the auth system existed).
        _owner = data.get("owner_user_id")
        _shared = data.get("shared_user_ids") or []
        try:
            shared_ids = {int(x) for x in _shared}
        except (TypeError, ValueError):
            shared_ids = set()
        _is_owner = (
            _owner is None
            or int(_owner) == int(owner_user_id)
        )
        _is_shared = int(owner_user_id) in shared_ids
        if not (_is_owner or _is_shared):
            continue
        thumb_path = pdir / "thumbnail.png"
        thumb_uri = ""
        if thumb_path.exists():
            try:
                raw = thumb_path.read_bytes()
                thumb_uri = (
                    "data:image/png;base64,"
                    + base64.b64encode(raw).decode("ascii")
                )
            except Exception:
                thumb_uri = ""
        lm = str(data.get("last_modified", ""))
        created_raw = str(data.get("created", ""))
        size_b = _dir_size_bytes(pdir)
        out.append({
            "name": str(data.get("name", pdir.name)),
            "dir": str(pdir),
            "created": created_raw,
            "created_short": _format_modified(created_raw),
            "last_modified": lm,
            "last_modified_short": _format_modified(lm),
            "thumbnail_data_uri": thumb_uri,
            "source_file": str(data.get("source_file", "")),
            # User-editable labels (replaces the old auto-tracked
            # stage_completed list). Read manifest["labels"] only;
            # legacy "stage_completed" entries are intentionally
            # ignored so projects start with an empty label slate.
            "labels": list(data.get("labels", [])),
            "size_bytes": size_b,
            "size_short": _format_bytes(size_b),
            # User-related fields surfaced into the welcome-view
            # tile + the navbar detail dialog. owner_user_id is
            # the creator; last_modified_user_id is whoever
            # touched the manifest most recently (stamped by
            # `_write_manifest` automatically); shared_user_ids
            # is the access-list the current user can edit from
            # the detail dialog.
            "owner_user_id": (
                int(_owner) if _owner is not None else None
            ),
            "last_modified_user_id": (
                int(data["last_modified_user_id"])
                if data.get("last_modified_user_id") is not None
                else None
            ),
            "shared_user_ids": sorted(shared_ids),
        })
    out.sort(key=lambda d: d["last_modified"], reverse=True)
    return out


# Pretty-print map for audit-log `action` strings + stage labels.
# Used by both the project status table and the activity scroller
# so the wording stays consistent between the two tabs.
_ACTION_PRETTY = {
    "load_geometry": "Geometry imported",
    "mesh_build": "Mesh built",
    "fem_solve": "FEM solved",
    "fiber_generate": "Fibers generated",
    "fiber_sim_run": "Single-fiber sim run",
    "pop_sim_run": "Population sim run",
    "conductivities_update": "Conductivities updated",
    "project_create": "Project created",
    "project_open": "Project opened",
    "project_close": "Project closed",
    "project_delete": "Project deleted",
    "login": "Signed in",
    "logout": "Signed out",
    "login_failed": "Sign-in failed",
    "login_error": "Sign-in error",
    "register": "Account registered",
    "register_error": "Registration error",
    "profile_update": "Profile updated",
}


def _pretty_action(action: str) -> str:
    """Map an audit-log action string to a human-readable label.
    Falls back to a title-cased version of the raw action so
    actions added after this map still render readably."""
    if not action:
        return ""
    if action in _ACTION_PRETTY:
        return _ACTION_PRETTY[action]
    return action.replace("_", " ").capitalize()


def _format_relative_time(iso: str) -> str:
    """Compact relative-time label ('5 m ago', '2 h ago',
    '3 d ago'). Falls back to the ISO timestamp if parsing
    fails. Used by the activity scroller so each row reads at
    a glance — the absolute timestamp lives on the hover."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
    except Exception:
        return iso
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    now = datetime.now(timezone.utc)
    diff = (now - dt).total_seconds()
    if diff < 0:
        return "just now"
    if diff < 60:
        return f"{int(diff)} s ago"
    if diff < 3600:
        return f"{int(diff/60)} m ago"
    if diff < 86400:
        return f"{int(diff/3600)} h ago"
    if diff < 86400 * 30:
        return f"{int(diff/86400)} d ago"
    if diff < 86400 * 365:
        return f"{int(diff/(86400*30))} mo ago"
    return f"{int(diff/(86400*365))} y ago"


def _compute_project_status(proj: dict) -> list[dict]:
    """Build the 8-row stage-status table for a project.

    Reads the manifest + checks disk-file presence so closed
    projects (welcome-page detail dialog) render exactly the
    same as live ones. Each row:
        {
          "id":      str,       # stable key for v-for
          "label":   str,       # human-readable stage name
          "done":    bool,
          "icon":    str,       # mdi-* icon (done / not-done)
          "color":   str,       # icon colour
          "status":  str,       # "done" | "pending"
          "details": str,       # secondary line (path / count / —)
        }

    Stage-completion criteria:
        Geometry imported  — manifest.source_file set OR
                              source/ folder has a file
        Electrodes placed  — ui_state.electrodes is non-empty
        Conductivities     — any σ differs from DEFAULT_SIGMA
                              (i.e. user moved a slider)
        Mesh built          — nerve.msh exists
        Fibers generated    — nerve_paths_fibers.npz exists
        FEM solved          — axis_line.npz AND
                              slice_volume.npz exist
        Single-fiber sims   — fiber_sim_results.pkl exists
        Population sims     — pop_state.pkl exists
    """
    pdir = Path(str(proj.get("dir", ""))) if proj else None
    ui_state = {}
    manifest = {}
    if pdir is not None and pdir.is_dir():
        try:
            manifest = json.loads(
                (pdir / "project.json").read_text(encoding="utf-8")
            )
            ui_state = dict(manifest.get("ui_state") or {})
        except Exception as _ex:
            print(
                f"[status] manifest read failed for "
                f"{pdir}: {_ex}",
                flush=True,
            )
            manifest = {}
            ui_state = {}
    else:
        print(
            f"[status] pdir invalid: "
            f"proj.dir={proj.get('dir') if proj else None!r}, "
            f"pdir={pdir!r}",
            flush=True,
        )

    def _exists(name: str) -> bool:
        return bool(pdir and (pdir / name).is_file())

    def _exists_in_sims(name: str) -> bool:
        """F3.1: a project "has" a single-fiber or population sim
        if the corresponding pickle lives EITHER at the legacy
        flat root OR under any of its per-design sim subdirs
        (`<out>/sims/<id>/<name>`). Used by the project-status
        chips so the welcome tile still lights up when only one
        design out of many has been simulated."""
        if not pdir:
            return False
        if (pdir / name).is_file():
            return True
        sims_root = pdir / "sims"
        if not sims_root.is_dir():
            return False
        for sub in sims_root.iterdir():
            if sub.is_dir() and (sub / name).is_file():
                return True
        return False

    def _row(rid, label, done, details):
        return {
            "id": rid,
            "label": label,
            "done": bool(done),
            "icon": (
                "mdi-check-circle"
                if done else "mdi-circle-outline"
            ),
            "color": (
                "success" if done else "grey-lighten-1"
            ),
            "status": "done" if done else "pending",
            "details": str(details or ""),
        }

    # 1) Geometry
    source_file = str(manifest.get("source_file", ""))
    has_source = bool(source_file)
    if not has_source and pdir is not None:
        src_dir = pdir / "source"
        if src_dir.is_dir():
            has_source = any(
                p.is_file() for p in src_dir.iterdir()
            )
    geom_details = ""
    if has_source:
        geom_details = source_file or "source/ folder populated"
        _sf = ui_state.get("scale_factor")
        if _sf is not None:
            try:
                geom_details += f"  (scale × {float(_sf):.3g})"
            except (TypeError, ValueError):
                pass
    rows = [_row("geometry", "Nerve geometry imported",
                  has_source, geom_details)]

    # 2) Designs (was "Electrodes" pre-F3.2a rename — manifests
    # written by older versions still carry the `electrodes` key,
    # so read both and prefer the new one).
    designs = list(
        ui_state.get("designs")
        or ui_state.get("electrodes")
        or []
    )
    has_designs = len(designs) > 0
    design_details = ""
    if has_designs:
        _types = [
            str(e.get("electrode_type", "")) for e in designs
            if e.get("electrode_type")
        ]
        _seen: list[str] = []
        for _t in _types:
            if _t and _t not in _seen:
                _seen.append(_t)
        design_details = (
            f"{len(designs)} "
            f"design{'s' if len(designs) != 1 else ''}"
        )
        if _seen:
            design_details += " · " + ", ".join(_seen[:3])
            if len(_seen) > 3:
                design_details += f", +{len(_seen)-3} more"
    rows.append(_row("designs", "Cuff designs placed",
                      has_designs, design_details))

    # 3) Conductivities — user explicitly committed via the
    # Update button? The `sigma_committed` flag lives in
    # ui_state and persists with the manifest. We also report
    # whether any σ differs from defaults as a secondary hint
    # (for committed projects only — uncommitted projects show
    # "Pending — open Conductivities and click Update").
    sigma_modified = False
    for _k, _default in DEFAULT_SIGMA.items():
        try:
            _v = float(ui_state.get(_k, _default))
        except (TypeError, ValueError):
            continue
        if abs(_v - float(_default)) > 1e-12:
            sigma_modified = True
            break
    sigma_done = bool(ui_state.get("sigma_committed", False))
    if sigma_done:
        sigma_details = (
            "Modified from defaults" if sigma_modified
            else "Committed at factory defaults"
        )
    else:
        sigma_details = (
            "Modified — click Update to commit"
            if sigma_modified
            else "Pending — click Update to commit"
        )
    rows.append(_row(
        "sigma", "Conductivities configured",
        sigma_done, sigma_details,
    ))

    # 4) Mesh
    has_mesh = _exists("nerve.msh")
    mesh_details = ""
    if has_mesh and pdir is not None:
        try:
            sz = (pdir / "nerve.msh").stat().st_size
            mesh_details = _format_bytes(sz)
        except Exception:
            mesh_details = "built"
    rows.append(_row("mesh", "Mesh built",
                      has_mesh, mesh_details))

    # 5) Fibers
    has_fibers = _exists("nerve_paths_fibers.npz")
    fib_details = ""
    if has_fibers and pdir is not None:
        try:
            sz = (pdir / "nerve_paths_fibers.npz").stat().st_size
            fib_details = _format_bytes(sz)
        except Exception:
            fib_details = "generated"
    rows.append(_row("fibers", "Fiber trajectories generated",
                      has_fibers, fib_details))

    # 6) FEM solve — solve_nerve.py writes axis_line.npz +
    # slice_volume.npz under the project dir. Both must exist
    # for `_restore_fem_from_disk` to succeed (we mirror that
    # gate here so the row matches what the runtime actually
    # considers "solved").
    def _any_design_has_fem() -> bool:
        """F3.1: a project counts as 'FEM solved' when EITHER the
        legacy flat layout has axis_line.npz+slice_volume.npz at
        the root OR any per-design subdir under fem/ has them."""
        if not pdir:
            return False
        if (
            (pdir / "axis_line.npz").is_file()
            and (pdir / "slice_volume.npz").is_file()
        ):
            return True
        fem_root = pdir / "fem"
        if fem_root.is_dir():
            for _sub in fem_root.iterdir():
                if not _sub.is_dir():
                    continue
                if (
                    (_sub / "axis_line.npz").is_file()
                    and (_sub / "slice_volume.npz").is_file()
                ):
                    return True
        return False

    has_fem = _any_design_has_fem()
    fem_details = ""
    if has_fem:
        try:
            _i = float(ui_state.get("I_stim_mA", 0.0))
            fem_details = f"I_stim = {_i:.3g} mA"
        except (TypeError, ValueError):
            fem_details = "solved"
    rows.append(_row("fem", "FEM extracellular field solved",
                      has_fem, fem_details))

    # 7) Single-fiber sims
    has_fiber_sim = _exists_in_sims("fiber_sim_results.pkl")
    rows.append(_row(
        "fiber_sim", "Single-fiber simulations run",
        has_fiber_sim,
        "Cached" if has_fiber_sim else "",
    ))

    # 8) Population sims
    has_pop = _exists_in_sims("pop_state.pkl")
    rows.append(_row(
        "pop_sim", "Population simulated",
        has_pop,
        "Cached" if has_pop else "",
    ))

    return rows


def _load_audit_events_for_project(
    project_dir: str | Path,
    limit: int = 500,
) -> list[dict]:
    """Pull recent audit events for `project_dir`, newest first.
    Each row is enriched with the username + avatar of the
    user_id so the activity scroller can render the chip without
    a second round-trip. Returns at most `limit` rows.

    Empty list when project_dir is falsy — the welcome-screen
    'global' detail view never opens with a project, so the
    callsite is safe to invoke unconditionally.
    """
    pdir = str(project_dir) if project_dir else ""
    if not pdir:
        print("[audit] _load: empty project_dir", flush=True)
        return []
    out: list[dict] = []
    try:
        with get_session() as session:
            # Match against the absolute path, its basename
            # (legacy rows), AND any stored project_dir that
            # endswith the basename (covers projects whose
            # absolute path string differs slightly — e.g.
            # symlinks, trailing slashes, abs vs canonical).
            pname = Path(pdir).name
            q = (
                session.query(_AuditEvent)
                .filter(
                    _sa.or_(
                        _AuditEvent.project_dir == pdir,
                        _AuditEvent.project_dir == pname,
                        _AuditEvent.project_dir.like(
                            f"%{pname}"
                        ),
                    )
                )
                .order_by(_AuditEvent.ts.desc())
                .limit(int(limit))
            )
            events = list(q.all())
            # Diagnostic — surface how many events we found
            # plus a sample of the project_dir values stored
            # in the DB so we can see why the filter failed
            # when it returns empty.
            print(
                f"[audit] _load: pdir={pdir!r}, "
                f"pname={pname!r}, matched={len(events)}",
                flush=True,
            )
            if not events:
                sample = (
                    session.query(_AuditEvent.project_dir)
                    .distinct()
                    .limit(10)
                    .all()
                )
                print(
                    f"[audit] DB has these project_dir values: "
                    f"{[r[0] for r in sample]}",
                    flush=True,
                )
            # Resolve user_id → username/avatar in one query.
            uids = sorted({
                int(e.user_id) for e in events
                if e.user_id is not None
            })
            users_by_id: dict[int, dict] = {}
            if uids:
                rows = (
                    session.query(_User)
                    .filter(_User.id.in_(uids))
                    .all()
                )
                for u in rows:
                    # Use the shared avatar helper — it reads
                    # `image_blob` (the actual column name) and
                    # falls back to a deterministic-colour SVG
                    # with the first letter of the username, so
                    # every chip always has an image to show.
                    users_by_id[int(u.id)] = {
                        "username": str(u.username or ""),
                        "avatar_data_uri": (
                            _user_avatar_data_uri(u)
                        ),
                    }
            for e in events:
                uid = (
                    int(e.user_id)
                    if e.user_id is not None else None
                )
                user = (
                    users_by_id.get(uid, {})
                    if uid is not None else {}
                )
                ts_iso = (
                    e.ts.isoformat()
                    if e.ts is not None else ""
                )
                # Pretty-print payload as 2-space JSON for the
                # expand panel. Empty / null payloads stay as
                # an empty string so the Vue template can hide
                # the panel cleanly.
                payload_pretty = ""
                if e.payload:
                    try:
                        _p = json.loads(e.payload)
                        payload_pretty = json.dumps(_p, indent=2)
                    except Exception:
                        payload_pretty = str(e.payload)
                out.append({
                    "id": int(e.id),
                    "ts_iso": ts_iso,
                    "ts_short": _format_modified(ts_iso),
                    "ts_relative": _format_relative_time(ts_iso),
                    "user_id": uid,
                    "username": user.get("username", ""),
                    "avatar_data_uri": user.get(
                        "avatar_data_uri", ""
                    ),
                    "action": str(e.action or ""),
                    "action_pretty": _pretty_action(
                        str(e.action or "")
                    ),
                    "status": str(e.status or "info"),
                    "payload_pretty": payload_pretty,
                    "has_payload": bool(payload_pretty),
                })
    except Exception as ex:
        print(f"[audit] load events failed: {ex}", flush=True)
        return []
    return out


def _create_project(name: str,
                     source_path: Path | None = None,
                     owner_user_id: int | None = None) -> Path:
    """Create a fresh project under PROJECTS_ROOT. If source_path
    is given, copy it into <project>/source/ and record its
    relative path in the manifest. `owner_user_id` is stamped
    into the manifest so per-user project filtering works in
    `_list_projects`. Returns the project directory."""
    name = (name or "").strip() or "Untitled project"
    slug = _sanitize_proj_dirname(name)
    pdir = PROJECTS_ROOT / f"{slug}__{_project_short_uid()}"
    pdir.mkdir(parents=True, exist_ok=False)
    (pdir / "uploads").mkdir(exist_ok=True)
    (pdir / "source").mkdir(exist_ok=True)
    source_rel = ""
    if source_path is not None and Path(source_path).is_file():
        sp = Path(source_path)
        dst = pdir / "source" / sp.name
        try:
            shutil.copy2(sp, dst)
            source_rel = f"source/{sp.name}"
        except Exception:
            source_rel = ""
    now = datetime.now().isoformat(timespec="seconds")
    manifest = {
        "version": _PROJECT_MANIFEST_VERSION,
        "name": name,
        "created": now,
        "last_modified": now,
        "source_file": source_rel,
        "labels": [],
        "ui_state": {},
        "notes": "",
        # int when set; null for legacy / scripted-create paths.
        "owner_user_id": (
            int(owner_user_id) if owner_user_id is not None else None
        ),
    }
    (pdir / "project.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return pdir


def _activate_project(pdir: Path) -> dict:
    """Switch the active project to `pdir`. Subsequent subprocess
    invocations + .json/.npz writes target the project folder
    via the GOLGI_OUT / UPLOAD_DIR proxies (which always read the
    live ActiveProject), so the bundle stays self-contained.
    Returns the manifest dict (empty if no project.json)."""
    pdir = Path(pdir)
    set_active(pdir)
    (pdir / "source").mkdir(exist_ok=True)
    mf = pdir / "project.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _deactivate_project() -> None:
    """Release the active project. The GOLGI_OUT/UPLOAD_DIR
    proxies fall back to the orphan path so any stray writes
    don't pollute a real project after a close."""
    set_active(_NO_PROJECT_FALLBACK)


def _read_manifest(pdir: Path) -> dict:
    mf = Path(pdir) / "project.json"
    if not mf.exists():
        return {}
    try:
        return json.loads(mf.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_manifest(pdir: Path, **updates) -> dict:
    """Merge `updates` into <pdir>/project.json and bump
    last_modified. Always stamps `last_modified_user_id` from
    the current auth session so the detail dialog can show
    who made the change. Returns the merged manifest."""
    mf = Path(pdir) / "project.json"
    data = _read_manifest(pdir)
    data.update(updates)
    data["last_modified"] = datetime.now().isoformat(timespec="seconds")
    # Stamp the actor — None if anonymous (script-run / migration
    # path). The detail dialog falls back to the owner's name
    # when this field is missing on legacy projects.
    _actor = _auth_session.get("user_id")
    if _actor is not None:
        data["last_modified_user_id"] = int(_actor)
    if "version" not in data:
        data["version"] = _PROJECT_MANIFEST_VERSION
    # Sharing list — make sure the field exists on every save so
    # downstream readers can rely on it.
    if "shared_user_ids" not in data:
        data["shared_user_ids"] = []
    mf.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _format_modified(iso: str) -> str:
    """Compact 'YYYY-MM-DD HH:MM' for display in project tiles."""
    if not iso:
        return ""
    try:
        dt = datetime.fromisoformat(iso)
        return dt.strftime("%Y-%m-%d %H:%M")
    except Exception:
        return iso


def _dir_size_bytes(path: Path) -> int:
    """Recursive on-disk size of a project folder, in bytes.
    Skips files that fail stat() (broken symlinks etc.) so a stray
    error in one entry can't take the whole sum down."""
    total = 0
    try:
        for entry in path.rglob("*"):
            try:
                if entry.is_file():
                    total += entry.stat().st_size
            except OSError:
                continue
    except OSError:
        pass
    return total


def _format_bytes(n: int) -> str:
    """Human-readable file size. Picks the unit so the value
    stays in [1, 1024), e.g. 0 B / 850 KB / 1.4 MB / 2.7 GB.
    Two decimals for GB so per-project sizes read clearly even
    when they hover around the 1 GB boundary."""
    if n < 0:
        return "—"
    units = ("B", "KB", "MB", "GB", "TB")
    size = float(n)
    for u in units:
        if size < 1024.0 or u == units[-1]:
            if u == "B":
                return f"{int(size)} {u}"
            if u in ("KB", "MB"):
                return f"{size:.1f} {u}"
            return f"{size:.2f} {u}"
        size /= 1024.0
    return f"{n} B"


# Locked material parameters — same values we hand-tuned for nerve_viz.
# Tag convention follows nerve_studio:
#   1 = endoneurium  2 = saline  3 = silicone cuff
#   4 = muscle       5 = epineurium  (epi shell skipped for v1)
# Gold electrode contacts get their own synthetic tag = 6.
DEFAULTS = {
    1: dict(label="endoneurium",
            color=(0.043, 0.000, 0.200),  # #0B0033 — deep indigo
            opacity=1.000, visible=True,
            ambient=0.28, diffuse=0.90,
            specular=0.20, specular_power=12.0),
    2: dict(label="saline",
            color=(0.369, 0.769, 0.941),
            opacity=0.840, visible=True,
            ambient=0.10, diffuse=0.65,
            specular=0.550, specular_power=5.0),
    3: dict(label="silicone",
            color=(0.940, 0.940, 0.940),
            opacity=0.550, visible=True,
            ambient=0.20, diffuse=0.55,
            specular=0.840, specular_power=2.0),
    4: dict(label="muscle",
            color=(0.788, 0.486, 0.486),
            opacity=0.200, visible=True,
            ambient=0.18, diffuse=0.85,
            specular=0.000, specular_power=1.0),
    5: dict(label="epineurium",
            color=(1.000, 0.949, 0.761),
            opacity=0.500, visible=True,
            ambient=0.15, diffuse=0.75,
            specular=0.350, specular_power=25.0),
    # Tag 7 — scar / connective tissue (per-design outward shell
    # around the nerve, inside the cuff saline pocket). Salmon /
    # dusty rose so it reads distinct from epi cream + saline
    # cyan in the legend swatches.
    7: dict(label="scar / connective tissue",
            color=(0.85, 0.55, 0.50),
            opacity=0.650, visible=True,
            ambient=0.18, diffuse=0.80,
            specular=0.050, specular_power=4.0),
}
# Gold contacts — metallic look. specular is the tuneable visual
# knob; high specular_power = tight bright highlight = polished metal.
GOLD_STYLE = dict(
    label="contacts",
    color=(0.95, 0.78, 0.18),
    opacity=1.000, visible=True,
    ambient=0.30, diffuse=0.50,
    specular=1.000, specular_power=40.0,
)
# Anode / cathode tints used to colour contact actors when the
# user has marked them so via the Electrodes drawer. Same lighting
# profile as GOLD_STYLE — only the diffuse colour differs — so
# polarised contacts read as "the same metal, just illuminated
# differently" rather than competing materials.
ANODE_STYLE = dict(
    label="anode",
    color=(0.95, 0.22, 0.25),   # saturated red
    opacity=1.000, visible=True,
    ambient=0.30, diffuse=0.55,
    specular=0.900, specular_power=35.0,
)
CATHODE_STYLE = dict(
    label="cathode",
    color=(0.18, 0.55, 0.97),   # saturated blue
    opacity=1.000, visible=True,
    ambient=0.30, diffuse=0.55,
    specular=0.900, specular_power=35.0,
)
# UI choices for the polarity dropdown. "off" leaves the contact
# at the gold neutral — no current source attached, treated as
# floating by the solver pass downstream.
POLARITY_CHOICES = ("off", "anode", "cathode", "ground")
TAG_GOLD = 6
TAG_SCAR = 7
# Outer-to-inner radial ordering for blend / depth-sort. Inside the
# cuff: muscle (4) → silicone (3) → saline (2) → scar (7) → nerve
# outer surface → epineurium shell (5) → endoneurium core (1).
# Gold contacts (6) are embedded in the silicone, rendered last.
TAG_ORDER = [4, 3, 2, TAG_SCAR, 5, 1, TAG_GOLD]

# Fiber-branch colour palette — matches nerve_studio.py's
# `BRANCH_COLORS` so a side-by-side comparison is consistent.
BRANCH_PALETTE = [
    "#1a3a8f",   # navy        (branch 0)
    "#d35400",   # persimmon   (branch 1)
    "#0c8a55",   # emerald     (branch 2)
    "#b88a00",   # amber       (branch 3)
    "#7a3aa7",   # violet      (branch 4)
    "#a83e6c",   # rose        (branch 5)
]
MAX_FIBER_BRANCHES = len(BRANCH_PALETTE)
FIBERS_MASTER_COLOUR = "#cc4778"   # used when no branch clustering

# Matplotlib/Seaborn tab10 palette. Used by the Single-fiber
# combobox to give each fiber chip a stable distinct colour, and
# by the 3-D viewport highlight to draw each selected fiber in
# the matching colour (so the chip ↔ trajectory mapping is
# unambiguous at a glance). Mirrors
# `sns.color_palette("tab10")`.
TAB10_PALETTE = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
    "#9467bd", "#8c564b", "#e377c2", "#7f7f7f",
    "#bcbd22", "#17becf",
]

# Fiber-type colour palette — used by the Population tab to colour
# fibers by their assigned membrane model after generate (so all
# MRG fibers share one colour, all Sundt fibers another, etc.).
# Distinct hues, all saturated enough to read against the white
# bg + dim grey unassigned fibers.
TYPE_PALETTE = [
    "#dc2626",   # red
    "#2563eb",   # blue
    "#16a34a",   # green
    "#ea580c",   # orange
    "#9333ea",   # purple
    "#0891b2",   # cyan
    "#ca8a04",   # gold
    "#db2777",   # pink
    "#65a30d",   # lime
    "#475569",   # slate
    "#7c3aed",   # violet
    "#0d9488",   # teal
]

# Saline-infill preview style — translucent cylinder at R_ci × L_cuff
# so the nerve stays visible through it. Separate from DEFAULTS[2]
# (which is for the meshed saline domain) because the preview wants
# a much lower opacity than the meshed render.
SALINE_OVERLAY_STYLE = dict(
    label="saline infill",
    color=(0.30, 0.72, 0.98),
    opacity=0.55, visible=True,
    ambient=0.35, diffuse=0.70,
    specular=0.20, specular_power=6.0,
)

# Parameter defaults — pulled from nerve_studio's UI initial values.
DEFAULT_CUFF = dict(
    L_cuff_mm=10.0,
    cuff_offset_mm=10.0,         # signed axial offset from anchor (mm)
    cuff_dx_mm=0.0,              # transverse fine-tune in PCA frame
    cuff_dy_mm=0.0,              # transverse fine-tune in PCA frame
    # F3.2a: cuff orientation as intrinsic Euler angles in the
    # cuff's local frame. Applied as Rx · Ry · Rz so the user
    # can pitch, then yaw, then twist around the cuff's own axis.
    cuff_rot_x_deg=0.0,          # pitch — tilt off-axis around local x
    cuff_rot_y_deg=0.0,          # yaw   — tilt off-axis around local y
    cuff_rot_z_deg=0.0,          # twist — rotation around local z (cuff axis)
    cuff_anchor="trunk",         # 'trunk', 'branched', or 'centroid'
    local_pca_radius_mm=15.0,
    cuff_clearance_mm=0.20,
    cuff_wall_mm=1.0,
    show_saline=True,            # render translucent saline cylinder
    # F3.2-M3: per-design scar / connective tissue shell. When
    # `use_scar=True`, the PLC builder inserts an outward offset
    # of the nerve surface by `scar_thickness_um` and TetGen
    # carves out a thin (tag 7) region between the nerve and
    # that offset. Saline auto-fills the remaining cuff interior.
    # Default off so existing projects re-mesh identically.
    use_scar=False,
    scar_thickness_um=100,
)
DEFAULT_ELECTRODE = dict(
    electrode_type="bipolar ring-pair",
    # bipolar (full-ring × 2)
    bipolar_axial_sep_mm=4.0,
    bipolar_ring_width_mm=0.6,
    # tripolar (anode-cathode-anode full rings)
    tripolar_axial_sep_mm=2.0,
    tripolar_ring_width_mm=0.6,
    # ring-array (N_rows × N_cols angular-arc contacts)
    array_n_rows=2,
    array_n_cols=4,
    array_row_sep_mm=3.0,        # axial spacing between rows
    array_contact_w_mm=0.6,
    array_contact_phi_deg=60.0,
    # helical (Livanova-style spiral bands)
    helix_n_bands=2,
    helix_pitch_mm=12.0,
    helix_dphi_deg=180.0,
    helix_band_sep_mm=8.0,       # contact separation, decoupled from cuff length (LivaNova ~8 mm)
    # LIFE (Longitudinal Intrafascicular Electrode) NxM array
    # — M parallel thin filaments running along local +z,
    # each carrying N contact bands. Single-wire LIFE is
    # `life_n_cols=1`. Defaults follow Boretius / Rossini
    # tfLIFE geometry (∼ 75 µm wire diameter, ∼ 500 µm
    # exposed sites, ∼ 2 mm contact spacing). M wires are
    # laid out along a chord at angle `life_chord_phi_deg`
    # (cuff-local frame), spaced by `life_col_sep_mm`.
    # `life_x_mm` / `life_y_mm` mark the array centre — set
    # by auto-fit to the nerve cross-section centroid
    # (overrideable via `life_target_fascicle_idx`: -1 =
    # nerve centroid, 0..N-1 = snap to that fascicle's
    # centroid on µCT bundles).
    life_n_rows=2,
    life_n_cols=1,
    life_row_sep_mm=2.0,
    life_col_sep_mm=0.5,
    life_contact_length_mm=0.5,
    life_diameter_um=80.0,
    life_chord_phi_deg=0.0,
    life_x_mm=0.0,
    life_y_mm=0.0,
    life_target_fascicle_idx=-1,
    # TIME (Transverse Intrafascicular Multichannel Electrode)
    # NxM array — flat ribbon punched perpendicular to the
    # nerve axis, threads through multiple fascicles, contacts
    # arranged as N axial rows × M transverse columns on the
    # ribbon's front face. N=1 reduces to single-row TIME
    # (Boretius 2010); typical research TIMEs have 1-2 rows
    # of 8-14 contacts at ∼ 230 µm pitch. `time_chord_phi_deg`
    # is the in-plane chord angle (0° = ribbon along +x).
    # `time_x_mm` / `time_y_mm` mark the ribbon midpoint;
    # auto-fit sets them to the nerve cross-section centroid
    # + chooses phi from fascicle-centroid PCA.
    time_n_rows=1,
    time_n_cols=8,
    time_row_sep_mm=0.5,
    time_col_sep_mm=0.230,
    time_contact_w_mm=0.080,
    time_ribbon_width_mm=1.5,
    time_ribbon_thickness_um=100.0,
    time_chord_phi_deg=0.0,
    time_x_mm=0.0,
    time_y_mm=0.0,
)
ELECTRODE_TYPES = [
    "bipolar ring-pair",
    "tripolar (anode-cathode-anode)",
    "ring-array (NxM)",
    "helical (Livanova-style)",
    # Intrafascicular electrodes (LIFE / TIME) — see DEFAULT_-
    # ELECTRODE comments above for the parameter set. The
    # geometry / FEM dispatchers no-op on these strings in
    # phase 1 (cuff shell renders alone); phase 2 wires up
    # the wire / ribbon patch builders, phase 5 the FEM BCs.
    "LIFE (longitudinal intrafascicular)",
    "TIME (transverse intrafascicular)",
    # When this is the selected type, the per-electrode params
    # area swaps the standard slider stack for an "Open designer"
    # button that opens the ASCENT cuff designer dialog scoped to
    # this electrode.
    "DUKE Cuff designer",
]
LIFE_ELECTRODE_TYPE = "LIFE (longitudinal intrafascicular)"
TIME_ELECTRODE_TYPE = "TIME (transverse intrafascicular)"
DUKE_ELECTRODE_TYPE = "DUKE Cuff designer"
# Adopted project defaults (single source of truth:
# golgi.conductivity.materials.MATERIAL_SIGMA). These are the
# TRANSVERSE / isotropic values shown per-tag in the Conductivities
# drawer; the longitudinal components of the anisotropic tissues
# (endoneurium, muscle) and the perineurium contact-impedance σ are
# carried by MeshConfig + the materials table, not as extra state
# keys here (the drawer / state plumbing is keyed one-per-tag).
DEFAULT_SIGMA = dict(
    sigma_endo=1.0 / 6.0,    # endoneurium transverse 1/6 [S/m] (Ranck 1965)
    sigma_saline=1.76,       # saline (Geddes & Baker 1967)
    sigma_silicone=1.0e-12,  # silicone cuff (near-insulator)
    sigma_muscle=0.086,      # muscle transverse [S/m] (Gielen 1984 / Pelot 2017)
    sigma_epi=1.0 / 6.3,     # epineurium 1/6.3 [S/m] (Stolinski/Grill/Pelot)
    # Scar / encapsulation tissue (tag 7) — fibrotic peri-implant
    # capsule, 1/6.3 S/m (Grill & Mortimer 1994). Override in the
    # Conductivities drawer.
    sigma_scar=1.0 / 6.3,
    # Electrode contact volume (tag 6). The contacts drive current via
    # a Neumann BC on their facets, so this σ only shapes the field
    # within the contact volume + aids conditioning. Default = bulk
    # platinum (the project's electrode metal); pick another from the
    # preset list for Pt-Ir / SS316LVM / TiN cuffs.
    sigma_contact=9.43e6,
)
SIGMA_TAG_MAP = {            # state-var key → mesh tag in TetGen output
    "sigma_endo": 1,
    "sigma_saline": 2,
    "sigma_silicone": 3,
    "sigma_muscle": 4,
    "sigma_epi": 5,
    "sigma_scar": 7,          # matches TAG_SCAR
    "sigma_contact": 6,       # matches TAG_GOLD
}
SIGMA_LABEL_MAP = {
    "sigma_endo": "endoneurium",
    "sigma_saline": "saline",
    "sigma_silicone": "silicone (insulator)",
    "sigma_muscle": "muscle",
    "sigma_epi": "epineurium",
    "sigma_scar": "scar / connective tissue",
    "sigma_contact": "contact metal",
}

# ---------------------------------------------------------------------------
# Tissue / material preset library — populates the per-tissue dropdown in
# the Conductivities drawer. Each tuple is (label, σ in S/m, source). Use
# label-only lookup; the source string is appended for display so the user
# can compare references at a glance.
#
# Curated from:
#   • IT'IS Foundation Tissue Properties Database v4.1 (Nerve entry —
#     critical because Gabriel 1996 lacks nerve; values at 37 °C across
#     frequency).
#   • Ranck (1965) — endoneurium anisotropic σ along/across fascicle.
#   • Stolinski (1995) — peripheral nerve tissue review.
#   • Geddes & Baker (1967) — muscle anisotropy.
#   • Schwan (1957) — muscle isotropic baseline.
#   • Grill & Mortimer (1994), Pelot et al. (2018) — epineurium ranges.
#   • Bedard et al. (2004) — saline / extracellular reference.
# The "Custom value" entry is a sentinel (σ = None) → picking it does
# nothing; it's the indicator state when the user typed a value that
# doesn't match any tabulated preset.
# ---------------------------------------------------------------------------
# Static literature entries + a Custom sentinel. The IT'IS values
# are appended below after the DB has been loaded (so they always
# reflect the bundled v4.2 data, not hand-transcribed numbers).
SIGMA_PRESETS: dict[str, list[tuple[str, float | None, str]]] = {
    "sigma_endo": [
        ("Custom value", None, ""),
        ("Ranck 1965 — longitudinal", 0.571, "Ranck (1965)"),
        ("Ranck 1965 — transverse", 0.083, "Ranck (1965)"),
        ("Stolinski 1995 — isotropic", 0.500, "Stolinski (1995)"),
    ],
    "sigma_saline": [
        ("Custom value", None, ""),
        ("0.9% saline / PBS (37 °C)", 1.50, "Bedard 2004"),
        ("Isotonic saline (37 °C)", 1.40, "common"),
        ("Ringer's solution", 1.50, "common"),
    ],
    "sigma_silicone": [
        ("Custom value", None, ""),
        ("Silicone — perfect insulator", 1.0e-12,
            "datasheet idealised"),
        ("Silicone — encapsulant grade", 1.0e-13,
            "datasheet typical"),
        ("Silicone — with surface leakage", 1.0e-9,
            "estimate"),
    ],
    "sigma_muscle": [
        ("Custom value", None, ""),
        ("Muscle longitudinal (Geddes 1967)", 0.520,
            "Geddes & Baker (1967)"),
        ("Muscle transverse (Geddes 1967)", 0.076,
            "Geddes & Baker (1967)"),
        ("Muscle isotropic (Schwan 1957)", 0.270,
            "Schwan (1957)"),
    ],
    "sigma_epi": [
        ("Custom value", None, ""),
        ("Stolinski 1995", 0.083, "Stolinski (1995)"),
        ("Grill & Mortimer 1994", 0.0083, "Grill & Mortimer (1994)"),
        ("Pelot et al. 2018", 0.066, "Pelot et al. (2018)"),
        ("Weerasuriya 1984 (rabbit sciatic)", 0.0085,
            "Weerasuriya (1984)"),
    ],
    # F3.2-M3 — scar / connective tissue (tag 7). Peri-implant
    # fibrotic capsule σ values reported in the cuff-stimulation
    # literature; the IT'IS / Gabriel "Connective Tissue" entry
    # gives a reasonable order-of-magnitude default at 0.1 S/m.
    "sigma_scar": [
        ("Custom value", None, ""),
        ("Connective tissue (IT'IS / Gabriel)", 0.10,
            "IT'IS database"),
        ("Encapsulation tissue (Grill 1999)", 0.05,
            "Grill (1999)"),
        ("Dense fibrotic capsule (literature)", 0.05,
            "literature"),
        ("Loose / vascularised scar (literature)", 0.20,
            "literature"),
    ],
    # Electrode contact metals — bulk DC conductivities. The
    # contacts are driven by a Neumann BC at their facets, so this
    # σ only shapes the field within the contact volume + helps
    # conditioning. For surface-rough materials (Pt black, TiN) the
    # impedance reduction is a surface-area effect — the bulk σ
    # entered here is the same as the smooth-metal value.
    "sigma_contact": [
        ("Custom value", None, ""),
        ("Gold (Au)", 4.10e7, "CRC handbook"),
        ("Platinum (Pt)", 9.43e6, "CRC handbook"),
        ("Pt-Ir 90/10", 4.7e6, "literature"),
        ("Pt-Ir 80/20", 3.5e6, "literature"),
        ("Stainless steel 316LVM", 1.4e6, "AISI / ASM data"),
        ("Titanium (Ti)", 2.38e6, "CRC handbook"),
        ("Titanium nitride (TiN)", 5.0e6, "literature"),
        ("Iridium oxide (IrOx)", 7.0e4, "literature"),
        ("Perfect conductor (idealised)", 1.0e8, "model assumption"),
    ],
}

# Per-domain IT'IS tissue routing — which entries from the curated
# 30-tissue subset get appended to each σ preset list. Each domain
# gets the tissues most likely to inform its σ choice. The actual
# extend() of SIGMA_PRESETS happens further down, AFTER
# golgi.conductivity is imported (which provides itis_preset_rows).
_SIGMA_DOMAIN_TISSUES: dict[str, list[str]] = {
    "sigma_endo": [
        "Nerve", "Spinal Cord",
        "Brain (Grey Matter)", "Brain (White Matter)",
    ],
    "sigma_saline": [
        "Cerebrospinal Fluid", "Extracellular Fluids",
        "Blood", "Bile", "Urine",
    ],
    "sigma_silicone": [],   # no IT'IS equivalent — insulator
    "sigma_contact": [],    # metals, not biological — see preset list
    "sigma_muscle": [
        "Muscle", "Heart Muscle", "Tongue", "Esophagus",
        "Stomach (Wall)",
    ],
    "sigma_epi": [
        "Connective Tissue", "Tendon (Ligaments)", "Cartilage",
        "Skin (Dry)", "Skin (Wet)", "Fat",
        "SAT (Subcutaneous Fat)",
    ],
    # F3.2-M3 — scar / connective tissue picks the same fibrotic
    # IT'IS family as epi (connective tissue / tendon / skin),
    # since peri-implant scar is histologically dense connective
    # tissue with similar dielectric behaviour.
    "sigma_scar": [
        "Connective Tissue", "Tendon (Ligaments)",
        "Skin (Dry)", "Skin (Wet)", "Fat",
        "SAT (Subcutaneous Fat)",
    ],
}

# Vue-ready preset items per tissue: [{"title": "<label> · <σ> S/m · <src>",
# "value": "<label>"}]. The display label includes σ + source inline so the
# user sees the citation without expanding the row.
def _build_preset_items() -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for k, presets in SIGMA_PRESETS.items():
        rows: list[dict] = []
        for lbl, sigma, src in presets:
            if sigma is None:
                title = lbl
            else:
                # Pick a sigma format that reads cleanly across the
                # 1e-13 … 10 S/m range we cover.
                if abs(sigma) < 1e-3 or abs(sigma) > 100:
                    sig_str = f"{sigma:.3g}"
                else:
                    sig_str = f"{sigma:g}"
                src_str = f" · {src}" if src else ""
                title = f"{lbl} · σ = {sig_str} S/m{src_str}"
            rows.append({"title": title, "value": lbl})
        out[k] = rows
    return out


SIGMA_PRESET_ITEMS: dict[str, list[dict]] = _build_preset_items()


def _sigma_preset_lookup(sigma_key: str, label: str) -> float | None:
    """Look up the σ associated with a preset label for a tissue."""
    for lbl, sigma, _src in SIGMA_PRESETS.get(sigma_key, []):
        if lbl == label:
            return sigma
    return None


def _sigma_match_label(sigma_key: str, value: float) -> str:
    """Find the preset whose σ exactly matches `value` (with a
    small relative tolerance). Falls back to 'Custom value' so
    the dropdown reflects that the field was typed, not picked."""
    for lbl, sigma, _src in SIGMA_PRESETS.get(sigma_key, []):
        if sigma is None:
            continue
        # Relative + absolute tolerance — abs handles σ ≈ 0
        # (silicone insulator ≈ 1e-12), rel handles everything else.
        if (abs(sigma - value)
                <= max(abs(sigma) * 1e-6, 1e-18)):
            return lbl
    return "Custom value"


# ---------------------------------------------------------------------------
# Cole-Cole conductivity model + IT'IS Material Database loader.
# Extracted into `golgi/conductivity/` in step W1.1 of FEATURES.md.
# The public surface (cole_cole_sigma, COLE_COLE_PRESETS,
# COLE_COLE_PRESET_ITEMS, ITIS_CURATED_30, itis_preset_rows, …)
# is unchanged; everything below this comment used to live inline.
# ---------------------------------------------------------------------------
from golgi.conductivity import (  # noqa: E402
    COLE_COLE_PRESET_ITEMS,
    COLE_COLE_PRESETS,
    ITIS_CURATED_30,
    cole_cole_sigma,
    itis_preset_rows,
)


# Now that the helpers above are imported, augment each σ-preset
# list with IT'IS-derived rows for the curated tissue subset.
for _domain, _tissues in _SIGMA_DOMAIN_TISSUES.items():
    for _t in _tissues:
        SIGMA_PRESETS[_domain].extend(itis_preset_rows(_t))


DEFAULT_MESH = dict(
    muscle_radial_pad_mm=20.0,
    muscle_axial_pad_mm=80.0,
    muscle_dx_mm=0.0,
    muscle_dy_mm=0.0,
    muscle_dz_mm=0.0,
    lc_endo_um=200,
    # Bumped from 150 → 300 µm. Saline carries the volume
    # conductor between nerve and cuff inner wall; the field
    # gradients there are gentle, so 300 µm captures the
    # potential drop without doubling the tet count. Halves
    # TetGen's refinement time vs. the previous default.
    lc_saline_um=300,
    lc_silicone_um=300,
    # Bumped from 1000 → 3000 µm. Muscle is a large bbox
    # padded ~20 mm radially and ~80 mm axially around the
    # nerve; its only role in the FEM is to provide a return
    # path for current. 3 mm tets are coarse enough to drop
    # the muscle from being a meshing bottleneck while still
    # resolving the bulk conductor.
    lc_muscle_um=3000,
    lc_contact_um=100,
    # Epineurium (optional shell domain — same defaults as
    # nerve_studio.py § 5)
    use_epi=False,
    epi_thickness_um=50,
    # Bumped from 150 → 250 µm. The epi annulus is the thinnest
    # region in the bundle build (between fascicles and outer
    # nerve hull), so it dominated the refine pass. 250 µm
    # still gives ≥ 1 tet across a typical 100 µm epi thickness.
    lc_epi_um=250,
    # F3.2-M3 — scar / connective tissue (tag 7). Mesh edge length
    # inside the scar shell; should be ≤ shell thickness so at
    # least one tet fits radially. Default matches `lc_epi_um`.
    lc_scar_um=150,
)


# ---------------------------------------------------------------------------
# Helper functions (geometry / math) — adapted from nerve_studio
# ---------------------------------------------------------------------------

def list_data_files() -> list[str]:
    """STL/NAS/OBJ files visible to the importer. Scans DATA_DIR +
    the active project's uploads/ + the active project's source/
    (when a project is open). Paths are returned relative to HERE
    when possible, otherwise as absolute paths (project files live
    outside HERE under ~/Documents/Golgi/Projects/)."""
    out: list[str] = []
    dirs: list[Path] = [DATA_DIR, UPLOAD_DIR]
    # When a project is active, also surface its bundled source
    # geometry so the user can re-pick / re-load it from the
    # Import drawer.
    if GOLGI_OUT != _NO_PROJECT_FALLBACK:
        src_dir = GOLGI_OUT / "source"
        if src_dir.is_dir():
            dirs.append(src_dir)
    for _d in dirs:
        if not _d.is_dir():
            continue
        for _ext in ("*.stl", "*.STL", "*.nas", "*.NAS",
                      "*.obj", "*.OBJ"):
            for _p in sorted(_d.glob(_ext)):
                try:
                    out.append(str(_p.relative_to(HERE)))
                except ValueError:
                    # Project source lives outside HERE → keep
                    # absolute path so load_nerve_file can resolve it.
                    out.append(str(_p))
    return out


def load_nerve_file(rel_path: str,
                     units_factor: float = 1.0e-3) -> dict:
    """Load STL/NAS/OBJ → {'pts_raw', 'tets_raw', 'boundary_raw'}.
    Returns positions in metres. `units_factor` is the multiplier to
    convert source-file units → metres (default 1e-3 = mm → m)."""
    full = HERE / rel_path
    m = meshio.read(str(full))
    pts = np.asarray(m.points, dtype=np.float64) * units_factor
    tets = None
    bnd = None
    if "tetra" in m.cells_dict:
        tets = np.asarray(m.cells_dict["tetra"], dtype=np.int64)
        bnd = boundary_tris_of_tets(tets)
    elif "triangle" in m.cells_dict:
        bnd = np.asarray(m.cells_dict["triangle"], dtype=np.int64)
    else:
        raise RuntimeError(
            f"Mesh has no tetra or triangle cells. "
            f"Cell types found: {list(m.cells_dict.keys())}"
        )
    return dict(pts_raw=pts, tets_raw=tets, boundary_raw=bnd,
                source_file=rel_path)


def boundary_tris_of_tets(tets: np.ndarray) -> np.ndarray:
    """Boundary triangles of a tet mesh — faces that appear once."""
    faces = np.vstack([
        tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]],
    ])
    s = np.sort(faces, axis=1)
    _, inv, counts = np.unique(
        s, axis=0, return_inverse=True, return_counts=True,
    )
    return faces[counts[inv] == 1]


# W1.2: per-element mesh-quality math lives in
# `golgi.pipeline.mesh_quality`. The leading-underscore aliases here
# preserve the historical call sites elsewhere in app.py without
# requiring a rename pass — the consumers in step 4 / step 5 will
# move to the new names as part of their extraction.
from golgi.pipeline.mesh_quality import (  # noqa: E402
    surface_quality as _surface_quality,
    tet_shape_quality as _tet_shape_quality,
)


def _build_viz_surfaces(
    region_surfaces: dict,
    target_max_tris: int = 30000,
) -> dict:
    """Build decimated viewport copies of each region surface.
    Anything above `target_max_tris` triangles is decimated down
    to that target; smaller regions are passed through untouched.
    The simulation pipeline (solve_nerve.py + Vₑ-on-surface
    overlay + colour-by-quality) keeps using the FULL surface
    via `geom.region_surfaces`; the viewport actors render from
    this decimated dict, which slashes the WebSocket payload for
    a 22 M-tet imported nerve from hundreds of MB down to a few
    MB and is the reason the lightbox feels like it "still has
    work to do" after closing — the client was downloading +
    GPU-uploading the full FEM boundary triangulation."""
    out: dict = {}
    for tag, surf in region_surfaces.items():
        n = int(surf.n_cells)
        if n <= target_max_tris:
            out[tag] = surf
            continue
        target_red = 1.0 - (target_max_tris / float(n))
        target_red = min(target_red, 0.95)
        try:
            dec = surf.decimate(
                target_red, progress_bar=False,
            )
            # decimate() can collapse pathologically when the
            # input has many degenerate triangles — fall back to
            # the original in that case so the region doesn't
            # vanish from the viewport.
            if dec.n_cells > 0:
                out[tag] = dec
            else:
                out[tag] = surf
        except Exception:
            out[tag] = surf
    return out


def _extract_region_surfaces_mm(pts_m: np.ndarray,
                                  tets: np.ndarray,
                                  tags: np.ndarray,
                                  q_tet: np.ndarray | None = None,
                                  on_line=None,
                                  ) -> dict:
    """Per-region boundary-surface extraction via VTK's
    vtkGeometryFilter (wrapped by pv.UnstructuredGrid.extract_
    surface). Orders-of-magnitude faster than numpy's np.unique on
    the stacked-face array for million+-tet meshes — the user's
    22 M-tet build would block the asyncio loop for minutes the
    other way, breaking the lightbox-close handshake.

    Coordinates are converted to **mm** here so render_built_mesh
    can attach the polydata directly without per-frame copies.
    Each region's surface inherits q_tet from its parent tet as
    cell data, so the colour-by-quality overlay works for free.
    """
    say = on_line if on_line is not None else (lambda *_: None)
    pts_mm = np.asarray(pts_m, dtype=np.float64) * 1000.0
    tags = np.asarray(tags, dtype=np.int32)
    tets = np.asarray(tets, dtype=np.int64)
    out: dict = {}
    utags = sorted({int(t) for t in np.unique(tags)})
    for tag in utags:
        mask = tags == tag
        n = int(mask.sum())
        if n == 0:
            continue
        sub_tets = tets[mask]
        cells = np.empty(n * 5, dtype=np.int64)
        cells[0::5] = 4
        cells[1::5] = sub_tets[:, 0]
        cells[2::5] = sub_tets[:, 1]
        cells[3::5] = sub_tets[:, 2]
        cells[4::5] = sub_tets[:, 3]
        celltypes = np.full(n, pv.CellType.TETRA, dtype=np.uint8)
        ug = pv.UnstructuredGrid(cells, celltypes, pts_mm)
        if q_tet is not None:
            ug.cell_data["q_tet"] = q_tet[mask].astype(np.float32)
        surf = ug.extract_surface(
            pass_pointid=False, pass_cellid=False,
            progress_bar=False,
            algorithm="dataset_surface",
        )
        out[int(tag)] = surf
        say(f"    tag {tag}: {n:,} tets → "
             f"{surf.n_cells:,} boundary tris")
        del ug, cells, celltypes, sub_tets
    return out


def boundary_tris_with_parents(
    tets: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Like `boundary_tris_of_tets` but also returns the parent
    tet index (into the input `tets` array) for each boundary
    face. Used for mapping per-tet quality scalars onto the
    boundary-surface render."""
    n_tets = len(tets)
    faces = np.vstack([
        tets[:, [0, 1, 2]], tets[:, [0, 1, 3]],
        tets[:, [0, 2, 3]], tets[:, [1, 2, 3]],
    ])
    s = np.sort(faces, axis=1)
    _, inv, counts = np.unique(
        s, axis=0,
        return_inverse=True, return_counts=True,
    )
    boundary_mask = counts[inv] == 1
    boundary_faces = faces[boundary_mask]
    # Row k of the stacked faces came from tet (k % n_tets),
    # local face (k // n_tets).
    parent_idx = np.where(boundary_mask)[0] % n_tets
    return boundary_faces, parent_idx.astype(np.int64)


def _topology_stats(pts: np.ndarray, tris: np.ndarray) -> dict:
    """Basic topology + bbox sanity check. Catches the common
    import-time disasters: open mesh, non-manifold edges, multiple
    disconnected pieces, units-off-by-1000, etc."""
    n_pts = int(len(pts))
    n_tris = int(len(tris))
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    bbox_mm = (bbox_max - bbox_min) * 1000.0  # source units → mm
    # Edge use-count: each edge of each triangle goes into one row.
    # Sorted so (a,b) and (b,a) are identified.
    edges = np.concatenate([
        tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]],
    ], axis=0)
    edges_sorted = np.sort(edges, axis=1)
    _, inv, counts = np.unique(
        edges_sorted, axis=0,
        return_inverse=True, return_counts=True,
    )
    n_boundary_edges = int((counts == 1).sum())
    n_nonmanifold_edges = int((counts > 2).sum())
    watertight = (n_boundary_edges == 0) and (n_nonmanifold_edges == 0)
    # Connected-component count via union-find on edges.
    parent = np.arange(n_pts, dtype=np.int64)
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    for u, v in edges_sorted[::3]:        # one edge per triangle is enough
        ru, rv = _find(int(u)), _find(int(v))
        if ru != rv:
            parent[ru] = rv
    referenced = np.unique(tris.ravel())
    roots = {_find(int(x)) for x in referenced}
    n_components = len(roots)
    return dict(
        n_pts=n_pts, n_tris=n_tris,
        bbox_mm=tuple(float(b) for b in bbox_mm),
        n_components=n_components,
        n_boundary_edges=n_boundary_edges,
        n_nonmanifold_edges=n_nonmanifold_edges,
        watertight=bool(watertight),
    )




# ---------------------------------------------------------------------------
# Cuff-fitting PCA helpers — extracted into golgi/scene/cuff_fit.py in
# step W1.5 of FEATURES.md. Five pure-numpy helpers imported under
# their original names so existing call sites (UI, do_fit_cuff, the H
# bundle, watchers/cuff.py, pipeline/_frames.py) don't change.
# ---------------------------------------------------------------------------
from golgi.scene.cuff_fit import (  # noqa: E402
    global_pca,
    find_cuff_origin_pca,
    local_pca_refine,
    transform_to_cuff_frame,
    autosize_R_ci,
)


# ---------------------------------------------------------------------------
# Electrode-patch generation — extracted into golgi/scene/electrode_patches.py
# in step W1.4 of FEATURES.md. The four functions are imported here under
# the same names so existing call sites (UI + FEM driver + H namespace)
# don't change.
# ---------------------------------------------------------------------------
from golgi.scene.electrode_patches import (  # noqa: E402
    axial_patch_polydata,
    helical_patch_polydata,
    build_electrode_patches_dicts,
    build_electrode_patches,
)


# ---------------------------------------------------------------------------
# PyVista plotter setup + render helpers
# ---------------------------------------------------------------------------

# build_plotter moved to golgi.scene.renderer in step 3.2;
# re-imported above.


def _add_phong_mesh(pl: pv.Plotter, polydata: pv.PolyData, *,
                     name: str, style: dict) -> "pv.Actor":
    surf = polydata.compute_normals(
        cell_normals=False, point_normals=True,
        consistent_normals=True, auto_orient_normals=True,
        non_manifold_traversal=False,
    )
    return pl.add_mesh(
        surf, name=name,
        color=style["color"],
        opacity=style["opacity"],
        pbr=False,
        ambient=style["ambient"],
        diffuse=style["diffuse"],
        specular=style["specular"],
        specular_power=style["specular_power"],
        smooth_shading=True,
        show_edges=False,
    )


def render_raw_nerve(pl: pv.Plotter,
                      nerve: dict,
                      cuff_origin: np.ndarray | None = None,
                      q: np.ndarray | None = None,
                      ) -> pv.PolyData:
    """Render JUST the nerve surface (no cuff overlay). Used for
    the initial 'Import' preview after loading the geometry.
    `q`: optional per-triangle quality scalar in [0, 1] — when
    provided, the nerve is coloured with the RdYlGn colormap
    instead of the solid DEFAULTS[1] indigo. Use this for the
    'Color by triangle quality' toggle in the Import drawer.

    Returns the vtkPolyData that is ACTUALLY mounted in the
    plotter (after compute_normals if the phong path is used),
    so the caller can stash it for in-place point updates later.
    Returning the pre-normals polydata would point at a stale
    object — the mapper holds the post-normals surf, and writing
    to the pre-normals copy would update nothing visible."""
    pl.remove_actor("nerve", reset_camera=False)
    pts_mm = nerve["pts_raw"] * 1000.0
    tris = nerve["boundary_raw"]
    n = len(tris)
    faces = np.empty(n * 4, dtype=np.int64)
    faces[0::4] = 3
    faces[1::4] = tris[:, 0]
    faces[2::4] = tris[:, 1]
    faces[3::4] = tris[:, 2]
    poly = pv.PolyData(pts_mm, faces)
    if q is not None and len(q) == n:
        poly.cell_data["q_radius_ratio"] = np.asarray(q,
                                                       dtype=np.float32)
        # No normals computation here — colour-by-scalar uses flat
        # shading per cell, which is correct for "show me the bad
        # triangles". Skip _add_phong_mesh's auto-orient pass to
        # keep both the scalar mapping and the cell layout intact.
        # Phong material params on the scalar-coloured path
        # too — otherwise the surface drops the 3-light
        # cinematic shading and reads as flat. Same values as
        # DEFAULTS[1] (endo spec) so the q-coloured nerve picks
        # up the same shape-from-shading as the plain nerve.
        _q_phong = DEFAULTS[1]
        pl.add_mesh(
            poly, name="nerve",
            scalars="q_radius_ratio",
            cmap="RdYlGn", clim=(0.0, 1.0),
            opacity=1.0, show_edges=False,
            pbr=False,
            ambient=_q_phong["ambient"],
            diffuse=_q_phong["diffuse"],
            specular=_q_phong["specular"],
            specular_power=_q_phong["specular_power"],
            smooth_shading=True,
            show_scalar_bar=False,
        )
        return poly
    # Phong path: compute normals here so we have a direct
    # reference to the polydata that ends up in the mapper.
    surf = poly.compute_normals(
        cell_normals=False, point_normals=True,
        consistent_normals=True, auto_orient_normals=True,
        non_manifold_traversal=False,
    )
    _style = DEFAULTS[1]
    pl.add_mesh(
        surf, name="nerve",
        color=_style["color"],
        opacity=_style["opacity"],
        pbr=False,
        ambient=_style["ambient"],
        diffuse=_style["diffuse"],
        specular=_style["specular"],
        specular_power=_style["specular_power"],
        smooth_shading=True,
        show_edges=False,
    )
    return surf


def _compute_fiber_branch_summary(
    paths: list,
    caps_json_path: Path,
    seed_end: str,
    branch_labels: dict[int, str] | None = None,
) -> list[dict]:
    """Build the structured branch-summary rows that drive the
    "Branch summary" table in the Fiber Trajectories drawer.
    Returns an ordered list with:
        [
          {idx: -1, label: "Overall", color: "",
           n_fibers, mean_mm, min_mm, max_mm, std_mm,
           editable: False},
          {idx: 0, label: <user-renamed or "Branch 0">,
           color: "#1a3a8f", ..., editable: True},
          ...
        ]
    where `idx == -1` marks the aggregate Overall row (no rename
    affordance, no colour swatch).

    `branch_labels` maps branch idx → user-renamed label; missing
    entries fall back to "Branch {idx}". Pass an empty dict / None
    to get default labels. Stats use trajectory arc length in mm.
    """
    if not paths:
        return []

    def _arc_len(p: np.ndarray) -> float:
        if len(p) < 2:
            return 0.0
        return float(
            np.linalg.norm(np.diff(p, axis=0), axis=1).sum()
        )

    labels = dict(branch_labels or {})
    lens_m = np.array([_arc_len(p) for p in paths])
    rows: list[dict] = [
        {
            "idx": -1,
            "label": "Overall",
            "color": "",
            "n_fibers": int(len(paths)),
            "mean_mm": float(lens_m.mean() * 1000.0),
            "min_mm": float(lens_m.min() * 1000.0),
            "max_mm": float(lens_m.max() * 1000.0),
            "std_mm": (
                float(lens_m.std() * 1000.0)
                if len(lens_m) > 1 else 0.0
            ),
            "editable": False,
        },
    ]

    branch_palette = [
        "#1a3a8f", "#d35400", "#0c8a55", "#b88a00",
        "#7a3aa7", "#a83e6c",
    ]
    if caps_json_path.exists():
        try:
            caps = json.loads(caps_json_path.read_text(encoding="utf-8"))
            centroids = np.asarray(
                caps.get("branch_cap_centroids_m", []),
                dtype=np.float64,
            )
        except Exception:
            centroids = np.zeros((0, 3))
        if centroids.size >= 3 and centroids.shape[0] > 0:
            if seed_end == "low":
                endpoints = np.array([p[-1] for p in paths])
            else:
                endpoints = np.array([p[0] for p in paths])
            d2 = np.linalg.norm(
                endpoints[:, None, :] - centroids[None, :, :],
                axis=2,
            )
            assigned = np.argmin(d2, axis=1)
            for bi in range(centroids.shape[0]):
                mask = assigned == bi
                if not mask.any():
                    continue
                bl = lens_m[mask]
                _user_label = str(labels.get(int(bi), "")).strip()
                rows.append({
                    "idx": int(bi),
                    "label": (
                        _user_label or f"Branch {int(bi)}"
                    ),
                    "color": branch_palette[
                        int(bi) % len(branch_palette)
                    ],
                    "n_fibers": int(mask.sum()),
                    "mean_mm": float(bl.mean() * 1000.0),
                    "min_mm": float(bl.min() * 1000.0),
                    "max_mm": float(bl.max() * 1000.0),
                    "std_mm": (
                        float(bl.std() * 1000.0)
                        if mask.sum() > 1 else 0.0
                    ),
                    "editable": True,
                })
    return rows


def _classify_fibers_by_branch(
    paths_raw: list,
    caps_json_path: Path,
    seed_end: str,
) -> tuple[np.ndarray, int]:
    """Assign each fiber to a branch based on its endpoint's
    nearest-cap kNN against the branch_cap_centroids the solver
    wrote to nerve_paths_caps.json. Returns (branch_idx, N).

    If no caps file or only one branch cap, every fiber gets
    branch 0 (and N = 1, which the legend treats as 'no
    sub-rows', just the master Trajectories toggle).
    """
    n = len(paths_raw)
    if n == 0:
        return np.zeros(0, dtype=int), 0
    if not caps_json_path.exists():
        return np.zeros(n, dtype=int), 1
    try:
        caps = json.loads(caps_json_path.read_text(encoding="utf-8"))
        centroids = np.asarray(
            caps.get("branch_cap_centroids_m", []),
            dtype=np.float64,
        )
    except Exception:
        return np.zeros(n, dtype=int), 1
    if centroids.ndim != 2 or centroids.shape[0] < 1:
        return np.zeros(n, dtype=int), 1
    if centroids.shape[0] == 1:
        return np.zeros(n, dtype=int), 1
    if seed_end == "low":
        endpoints = np.array([p[-1] for p in paths_raw])
    else:
        endpoints = np.array([p[0] for p in paths_raw])
    d2 = np.linalg.norm(
        endpoints[:, None, :] - centroids[None, :, :],
        axis=2,
    )
    return np.argmin(d2, axis=1).astype(int), int(centroids.shape[0])


def _polyline_polydata(paths_mm: list[np.ndarray]) -> pv.PolyData:
    """Pack a list of (N_i, 3) point arrays into a single PolyData
    with VTK line cells. Empty / degenerate paths are skipped."""
    pts_chunks: list[np.ndarray] = []
    cells_chunks: list[np.ndarray] = []
    offset = 0
    for p in paths_mm:
        n = int(p.shape[0])
        if n < 2:
            continue
        pts_chunks.append(p)
        cells_chunks.append(
            np.concatenate([[n], np.arange(n) + offset])
        )
        offset += n
    poly = pv.PolyData()
    if not pts_chunks:
        return poly
    poly.points = np.vstack(pts_chunks)
    poly.lines = np.concatenate(cells_chunks).astype(np.int64)
    return poly


def render_muscle_preview(pl: pv.Plotter,
                            R_mus_mm: float,
                            L_mus_mm: float,
                            cx_mm: float = 0.0,
                            cy_mm: float = 0.0,
                            cz_mm: float = 0.0,
                            ) -> None:
    """Translucent preview cylinder showing where the muscle
    bounding box will sit. As of F3.2-M2.1a this is mounted
    persistently whenever the nerve is loaded (was previously
    only shown while the Mesh drawer was open), so the user
    can see the bbox during stepper Step 4 and afterwards
    until any design's mesh is built."""
    pl.remove_actor("muscle_overlay", reset_camera=False)
    cyl = pv.Cylinder(
        center=(cx_mm, cy_mm, cz_mm), direction=(0, 0, 1),
        radius=R_mus_mm, height=L_mus_mm,
        resolution=96, capping=False,
    )
    spec = DEFAULTS[4]  # muscle styling, but override opacity to be
    pl.add_mesh(
        cyl, name="muscle_overlay",
        color=spec["color"], opacity=0.08,
        pbr=False, ambient=0.3, diffuse=0.55,
        specular=0.0, specular_power=1.0,
        smooth_shading=True, show_edges=False,
        culling=False, show_scalar_bar=False,
    )


def render_epi_preview(pl: pv.Plotter,
                         nerve_pts_mm: np.ndarray,
                         nerve_faces_flat: np.ndarray,
                         thickness_mm: float,
                         ) -> None:
    """Translucent preview of the epineurium shell — an inward-
    offset of the nerve boundary triangulation by `thickness_mm`.
    Pure geometric operation (no pymeshfix), so self-intersections
    in concave regions are tolerated; the user just sees an
    indicative shell, not a watertight FEM input. The "real" epi
    shell is built later by the mesh pipeline with pymeshfix
    repair so TetGen can carve out a clean (tag 5) volume.

    `nerve_pts_mm`: (N, 3) float array of nerve boundary vertices
        in viewport mm (PCA-aligned cuff frame).
    `nerve_faces_flat`: (M*4,) flat-array PyVista face format —
        each face is `[3, i0, i1, i2]`.
    """
    pl.remove_actor("epi_overlay", reset_camera=False)
    if (nerve_pts_mm is None
            or nerve_faces_flat is None
            or thickness_mm <= 0.0):
        return
    poly = pv.PolyData(nerve_pts_mm, nerve_faces_flat)
    poly = poly.compute_normals(
        point_normals=True, cell_normals=False,
        auto_orient_normals=True, consistent_normals=True,
        non_manifold_traversal=False,
    )
    normals = np.asarray(poly.point_data["Normals"], dtype=np.float64)
    poly.points = np.asarray(poly.points, dtype=np.float64) - (
        thickness_mm * normals
    )
    # F3.2-M3: the inward-offset surface IS the OUTER endoneurium
    # boundary in the eventual mesh (the epi shell sits BETWEEN
    # this surface and the raw nerve outer surface). Render as
    # indigo opaque so the user sees the endo through the cream
    # semi-transparent raw-nerve actor — together the two read as
    # "endo wrapped in epi shell" with clear colour separation.
    spec = DEFAULTS[1]  # endoneurium styling
    pl.add_mesh(
        poly, name="epi_overlay",
        color=spec["color"], opacity=1.0,
        pbr=False, ambient=0.3, diffuse=0.55,
        specular=0.0, specular_power=1.0,
        smooth_shading=True, show_edges=False,
        culling=False, show_scalar_bar=False,
    )


def render_fibers_by_branch(pl: pv.Plotter,
                              paths_display: list,
                              branch_idx: np.ndarray,
                              n_branches: int,
                              ve_per_path: list | None = None,
                              ve_clim_mV: tuple | None = None,
                              ) -> None:
    """Render one actor per branch with the matching palette
    colour. Old `fiber_branch_<i>` actors are wiped first so
    a re-render (e.g. after switching display frames) doesn't
    leak. `paths_display` is already in the current frame and
    in metres.

    When `ve_per_path` is provided (one Ve array per path,
    matching length), instead of per-branch solid colours the
    whole bundle is rendered as ONE tube-mesh actor coloured by
    per-point Ve via the plasma cmap. We extrude the polylines
    into a thin tube so the scalars actually colour a surface
    (line rendering through trame's vtk.js backend has ignored
    per-vertex cmaps in our setup — tubes have real triangles
    that the mapper colours reliably). NaNs (points outside the
    FEM mesh) are replaced with the median so they don't break
    the colour normalisation.
    """
    for _i in range(MAX_FIBER_BRANCHES):
        pl.remove_actor(f"fiber_branch_{_i}", reset_camera=False)
    if not paths_display:
        return

    # Ve overlay path
    if ve_per_path is not None and len(ve_per_path) == len(paths_display):
        pts_chunks: list[np.ndarray] = []
        cells_chunks: list[np.ndarray] = []
        ve_chunks: list[np.ndarray] = []
        offset = 0
        for p, ve in zip(paths_display, ve_per_path):
            p_mm = np.asarray(p, dtype=np.float64) * 1000.0
            n = int(p_mm.shape[0])
            if n < 2 or len(ve) != n:
                continue
            pts_chunks.append(p_mm)
            cells_chunks.append(
                np.concatenate([[n], np.arange(n) + offset]),
            )
            ve_chunks.append(np.asarray(ve, dtype=np.float32))
            offset += n
        if not pts_chunks:
            return
        ve_pts = np.concatenate(ve_chunks).astype(np.float32)
        good = np.isfinite(ve_pts)
        if good.any():
            ve_pts[~good] = np.float32(
                float(np.median(ve_pts[good])),
            )
        else:
            ve_pts[:] = 0.0
        # Build via the (points, lines=) constructor — assigning
        # to `.points` / `.lines` on an empty PolyData has been
        # observed to leave the underlying VTK cell array empty
        # in some pyvista builds, which causes the actor to fall
        # back to its default colour (often black). Passing both
        # at construction sidesteps that.
        poly = pv.PolyData(
            np.vstack(pts_chunks).astype(np.float64),
            lines=np.concatenate(cells_chunks).astype(np.int64),
        )
        poly.point_data["Ve"] = ve_pts
        poly.GetPointData().SetActiveScalars("Ve")
        # Extrude the polylines into a thin tube mesh so the
        # mapper has real surface triangles to colour. Radius is
        # in the same units as the points (mm) — 30 µm reads as
        # a fine but visible filament on top of a ~1 mm-diameter
        # nerve. tube() carries point scalars through to the new
        # surface points.
        tube = poly.tube(radius=0.03, n_sides=10, capping=False)
        # Convert V → mV so the tubes share the same numeric
        # scale as the endo/epi surface overlays + the colour-
        # bar legend. The mV unit is also what the §9 ribbon
        # uses, so the visual scale tracks the line plot.
        tube.point_data["Ve"] = (
            tube.point_data["Ve"].astype(np.float32) * 1.0e3
        )
        tube.GetPointData().SetActiveScalars("Ve")
        # Prefer the SHARED clim computed once per FEM solve
        # (so endo, epi, and the fiber tubes all map the same
        # mV value to the same colour, and the horizontal
        # colour bar applies to all three). Fall back to a
        # local 1/99 percentile clip when no shared clim is
        # available — keeps the legacy single-actor path
        # working before a FEM solve completes.
        if ve_clim_mV is not None:
            clim = ve_clim_mV
        else:
            ve_mv = ve_pts.astype(np.float32) * 1.0e3
            _good = np.isfinite(ve_mv)
            if _good.any():
                v_lo = float(np.percentile(ve_mv[_good], 1.0))
                v_hi = float(np.percentile(ve_mv[_good], 99.0))
            else:
                v_lo, v_hi = -1.0, 1.0
            if v_hi - v_lo < 1e-12:
                v_hi = v_lo + 1.0
            clim = (v_lo, v_hi)
        # Apply the same phong material params used by the
        # solid-colour fiber tubes so the Vₑ-coloured ones
        # don't drop the cinematic shading and look flat next
        # to the rest of the scene.
        _tube_phong = DEFAULTS[1]
        actor = pl.add_mesh(
            tube, name="fiber_branch_0",
            scalars="Ve",
            cmap="plasma", clim=clim,
            opacity=1.0,
            pbr=False,
            ambient=_tube_phong["ambient"],
            diffuse=_tube_phong["diffuse"],
            specular=_tube_phong["specular"],
            specular_power=_tube_phong["specular_power"],
            show_scalar_bar=False,
            smooth_shading=True,
            lighting=True,
        )
        try:
            _mapper = actor.GetMapper()
            _mapper.SetScalarModeToUsePointData()
            _mapper.SelectColorArray("Ve")
            _mapper.ScalarVisibilityOn()
        except Exception:
            pass
        return

    paths_mm = [
        np.asarray(p, dtype=np.float64) * 1000.0
        for p in paths_display
    ]
    if n_branches <= 1 or branch_idx is None:
        # Single bundle — use the master fibers colour.
        poly = _polyline_polydata(paths_mm)
        if poly.n_points == 0:
            return
        pl.add_mesh(
            poly, name="fiber_branch_0",
            color=FIBERS_MASTER_COLOUR,
            line_width=2, opacity=0.9,
            show_scalar_bar=False,
        )
        return
    # Multi-branch: one actor per branch, palette-coloured.
    for bi in range(n_branches):
        mask = branch_idx == bi
        if not mask.any():
            continue
        bpaths_mm = [paths_mm[k]
                       for k in np.where(mask)[0]]
        poly = _polyline_polydata(bpaths_mm)
        if poly.n_points == 0:
            continue
        pl.add_mesh(
            poly, name=f"fiber_branch_{bi}",
            color=BRANCH_PALETTE[bi % len(BRANCH_PALETTE)],
            line_width=2, opacity=0.9,
            show_scalar_bar=False,
        )


def _update_nerve_points_inplace(poly: pv.PolyData | None,
                                   pts_new: np.ndarray) -> bool:
    """Swap the nerve polydata's point coordinates in place — no
    remove+add, no normals recompute, no actor unmount. Used on
    every translate-only cuff fit so the nerve never disappears
    while the user is dragging position sliders.

    Operates directly on the polydata reference stored on
    `geom.nerve_poly` at render time. Bypassing the
    `pl.actors['nerve'].GetMapper().GetInput()` round-trip
    sidesteps two race conditions:
      1. `mapper.GetInput()` returning None on the same tick
         the actor was added (caused the "first-fit
         misalignment" bug),
      2. `pl.remove_actor("nerve") + pl.add_mesh(..., name="nerve")`
         leaving the prior actor in the renderer (caused the
         "two nerves" bug).

    Returns True on success, False if `poly` is None."""
    if poly is None:
        return False
    import vtk as _vtk
    from vtkmodules.util.numpy_support import numpy_to_vtk
    pts_mm = np.ascontiguousarray(pts_new * 1000.0, dtype=np.float64)
    vtk_arr = numpy_to_vtk(pts_mm, deep=True)
    vtk_pts = _vtk.vtkPoints()
    vtk_pts.SetData(vtk_arr)
    poly.SetPoints(vtk_pts)
    poly.Modified()
    return True


def render_cuff_preview(pl: pv.Plotter,
                          L_cuff_m: float,
                          R_ci_m: float,
                          R_co_m: float,
                          patches: list[pv.PolyData],
                          show_saline: bool = True,
                          ) -> None:
    """Solid silicone wall (annular tube) + optional translucent
    saline-infill cylinder + gold contacts overlay in cuff frame
    (cuff origin at 0, axis +z). The wall is the annulus between
    R_ci (inner) and R_co (outer): outer + inner cylinder surfaces
    + two flat annular caps at the cuff ends. Saline fills the
    interior of the cuff (radius < R_ci, |z| < L_cuff/2).
    """
    pl.remove_actor("silicone_overlay", reset_camera=False)
    pl.remove_actor("saline_overlay", reset_camera=False)
    # Wipe any contacts left over from a previous render — the
    # electrode-mode switch can change the count from 2 (bipolar)
    # to ~12 (ring-array 2×4 + headroom), so we strip a generous
    # range rather than only a single named actor.
    for _i in range(64):
        pl.remove_actor(f"gold_overlay_{_i}", reset_camera=False)

    L_mm = L_cuff_m * 1000.0
    R_ci_mm = R_ci_m * 1000.0
    R_co_mm = R_co_m * 1000.0
    outer = pv.Cylinder(
        center=(0, 0, 0), direction=(0, 0, 1),
        radius=R_co_mm, height=L_mm,
        resolution=96, capping=False,
    )
    inner = pv.Cylinder(
        center=(0, 0, 0), direction=(0, 0, 1),
        radius=R_ci_mm, height=L_mm,
        resolution=96, capping=False,
    )
    cap_top = pv.Disc(
        center=(0.0, 0.0, +L_mm / 2.0),
        inner=R_ci_mm, outer=R_co_mm,
        normal=(0.0, 0.0, 1.0), r_res=2, c_res=96,
    )
    cap_bot = pv.Disc(
        center=(0.0, 0.0, -L_mm / 2.0),
        inner=R_ci_mm, outer=R_co_mm,
        normal=(0.0, 0.0, -1.0), r_res=2, c_res=96,
    )
    wall = outer.merge([inner, cap_top, cap_bot])
    _add_phong_mesh(pl, wall, name="silicone_overlay",
                     style=DEFAULTS[3])

    # Saline infill — translucent cylinder filling the cuff bore.
    # Shrunk by a hair on both ends + radius so it doesn't z-fight
    # with the silicone annulus / its end caps. We bypass the
    # generic _add_phong_mesh path because its auto_orient_normals
    # heuristic can flip a closed cylinder's normals inward and
    # leave the back faces translucent-to-the-point-of-invisible.
    if show_saline:
        sal_R = R_ci_mm * 0.995
        sal_L = L_mm * 0.999
        saline = pv.Cylinder(
            center=(0, 0, 0), direction=(0, 0, 1),
            radius=sal_R, height=sal_L,
            resolution=96, capping=True,
        )
        _s = SALINE_OVERLAY_STYLE
        pl.add_mesh(
            saline, name="saline_overlay",
            color=_s["color"], opacity=_s["opacity"],
            pbr=False,
            ambient=_s["ambient"], diffuse=_s["diffuse"],
            specular=_s["specular"],
            specular_power=_s["specular_power"],
            smooth_shading=True, show_edges=False,
            culling=False,  # double-sided so we see the inside
        )
    # Gold contacts on the INNER surface (R_ci)
    for _i, _patch in enumerate(patches):
        _patch_mm = _patch.copy()
        _patch_mm.points *= 1000.0
        _add_phong_mesh(
            pl, _patch_mm,
            name=f"gold_overlay_{_i}",
            style=GOLD_STYLE,
        )


def render_one_cuff(pl: pv.Plotter,
                       name_prefix: str,
                       L_cuff_m: float,
                       R_ci_m: float,
                       R_co_m: float,
                       patches: list,
                       show_saline: bool = True,
                       offset_xyz_m: tuple = (0.0, 0.0, 0.0),
                       R_local_in_frame: np.ndarray | None = None,
                       polarities: list | None = None,
                       ) -> None:
    """Mount one electrode's cuff geometry under a namespaced
    actor-key prefix (e.g. `name_prefix='elec_01' →
    elec_01_silicone, elec_01_saline, elec_01_contact_<n>`).
    `offset_xyz_m` translates the cuff inside the global render
    frame; `R_local_in_frame` (3×3) rotates the cuff's local
    z-axis to point along the electrode's local nerve direction
    in that same frame. None = identity (cuff stays axis-aligned
    with global z, original behaviour)."""
    silicone_name = f"{name_prefix}_silicone"
    saline_name = f"{name_prefix}_saline"
    pl.remove_actor(silicone_name, reset_camera=False)
    pl.remove_actor(saline_name, reset_camera=False)
    for _i in range(64):
        pl.remove_actor(
            f"{name_prefix}_contact_{_i}", reset_camera=False,
        )
    L_mm = L_cuff_m * 1000.0
    R_ci_mm = R_ci_m * 1000.0
    R_co_mm = R_co_m * 1000.0
    dx_mm = offset_xyz_m[0] * 1000.0
    dy_mm = offset_xyz_m[1] * 1000.0
    dz_mm = offset_xyz_m[2] * 1000.0
    # Build every primitive in LOCAL frame (axis = +z, origin =
    # 0), then apply one 4×4 affine to land it in the global
    # render frame. Concentrating the transform here keeps the
    # rotation, translation, and per-vertex translation paths
    # consistent.
    R = (np.asarray(R_local_in_frame, dtype=np.float64)
         if R_local_in_frame is not None
         else np.eye(3, dtype=np.float64))
    M = np.eye(4, dtype=np.float64)
    M[:3, :3] = R
    M[:3, 3] = (dx_mm, dy_mm, dz_mm)

    def _to_frame(mesh: pv.DataSet) -> pv.DataSet:
        return mesh.transform(M, inplace=False)

    outer = pv.Cylinder(
        center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        radius=R_co_mm, height=L_mm,
        resolution=96, capping=False,
    )
    inner = pv.Cylinder(
        center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
        radius=R_ci_mm, height=L_mm,
        resolution=96, capping=False,
    )
    cap_top = pv.Disc(
        center=(0.0, 0.0, L_mm / 2.0),
        inner=R_ci_mm, outer=R_co_mm,
        normal=(0.0, 0.0, 1.0), r_res=2, c_res=96,
    )
    cap_bot = pv.Disc(
        center=(0.0, 0.0, -L_mm / 2.0),
        inner=R_ci_mm, outer=R_co_mm,
        normal=(0.0, 0.0, -1.0), r_res=2, c_res=96,
    )
    wall = outer.merge([inner, cap_top, cap_bot])
    wall = _to_frame(wall)
    _add_phong_mesh(
        pl, wall, name=silicone_name, style=DEFAULTS[3],
    )
    if show_saline:
        sal_R = R_ci_mm * 0.995
        sal_L = L_mm * 0.999
        saline = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=sal_R, height=sal_L,
            resolution=96, capping=True,
        )
        saline = _to_frame(saline)
        _s = SALINE_OVERLAY_STYLE
        pl.add_mesh(
            saline, name=saline_name,
            color=_s["color"], opacity=_s["opacity"],
            pbr=False,
            ambient=_s["ambient"], diffuse=_s["diffuse"],
            specular=_s["specular"],
            specular_power=_s["specular_power"],
            smooth_shading=True, show_edges=False,
            culling=False,
        )
    for _i, _patch in enumerate(patches):
        # `build_electrode_patches` returns contact polydatas in
        # metres, axis-aligned with +z. Scale to mm, then apply
        # the same affine.
        _patch_mm = _patch.copy()
        _patch_mm.points = _patch_mm.points * 1000.0
        _patch_mm = _to_frame(_patch_mm)
        # Polarity tint — only applied when the caller passed
        # an explicit polarities list (i.e. the Electrodes
        # drawer is open AND this is the selected electrode).
        # Off-state contacts stay gold; the actor name is the
        # same regardless of polarity so re-renders cleanly
        # overwrite the previous tint.
        if polarities is not None and _i < len(polarities):
            _pol = polarities[_i]
        else:
            _pol = "off"
        if _pol == "anode":
            _style = ANODE_STYLE
        elif _pol == "cathode":
            _style = CATHODE_STYLE
        elif _pol == "ground":
            # M1 — ground contacts get a desaturated grey
            # so they read as "passive reference" vs the
            # saturated red/blue stim contacts.
            _style = dict(GOLD_STYLE)
            _style["color"] = (0.55, 0.55, 0.55)
        else:
            _style = GOLD_STYLE
        _add_phong_mesh(
            pl, _patch_mm,
            name=f"{name_prefix}_contact_{_i}",
            style=_style,
        )


# ---------------------------------------------------------------------------
# PLC assembly — direct port of nerve_studio.py § 5. Extracted to
# golgi.pipeline.plc in step W1.3 of FEATURES.md. The single public
# entry point is `assemble_multi_domain_plc`; 11 internal helpers
# (_triangulate_*, _build_cylinder_lateral, _open_boundary_polylines,
# _signed_area, _orient, _count_self_intersections,
# _surgical_remove_intersections, _preprocess_nerve_surface,
# _assemble_plc) live there too and are not used elsewhere.
# ---------------------------------------------------------------------------
from golgi.pipeline.plc import assemble_multi_domain_plc  # noqa: E402


# Pipeline drivers (steps 4.3b / 4.4): the heavy do_* coroutines
# moved out of build_app into golgi.pipeline.<topic>. They take a
# PipelineContext bundling state/geom/scene + the closure hooks
# build_app defines later.
from types import SimpleNamespace  # noqa: E402
from golgi.pipeline import (  # noqa: E402
    mesh as _pipeline_mesh,
    fem as _pipeline_fem,
    fibers as _pipeline_fibers,
    fiber_sim as _pipeline_fiber_sim,
    pop_sim as _pipeline_pop_sim,
    sweep as _pipeline_sweep,
)
from golgi.pipeline._frames import (  # noqa: E402
    ensure_fibers_in_cuff_frame as _ensure_fibers_in_cuff_frame_impl,
)
from golgi.pipeline import fem_layout as _fem_layout  # noqa: E402
from golgi.pipeline.context import PipelineContext  # noqa: E402
from golgi.jobs.cancel import CancelToken as _CancelToken  # noqa: E402
# State-defaults registry — step 5.1.
from golgi import state_defaults as _state_defaults  # noqa: E402
# Watchers registry — step 5.2 (group-at-a-time).
from golgi import watchers as _watchers  # noqa: E402
# UI tier — step 5.3 (dialogs) / 5.4 (drawers).
from golgi import ui as _ui  # noqa: E402
# Action handlers — step W1.8 (extraction group-at-a-time).
from golgi import actions as _actions  # noqa: E402


def write_msh22(out_path: Path, nodes: np.ndarray,
                  elems: np.ndarray, tags: np.ndarray) -> None:
    """Write a gmsh v2.2 .msh file with per-tet physical tags."""
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("$MeshFormat\n2.2 0 8\n$EndMeshFormat\n")
        f.write("$PhysicalNames\n5\n")
        for _id, _name in [
            (1, "endo"), (2, "saline"), (3, "silicone"),
            (4, "muscle"), (5, "epi"),
        ]:
            f.write(f'3 {_id} "{_name}"\n')
        f.write("$EndPhysicalNames\n")
        f.write(f"$Nodes\n{len(nodes)}\n")
        for _i, _p in enumerate(nodes):
            f.write(f"{_i+1} {_p[0]:.9g} {_p[1]:.9g} {_p[2]:.9g}\n")
        f.write("$EndNodes\n")
        f.write(f"$Elements\n{len(elems)}\n")
        for _i, (_e, _tag) in enumerate(zip(elems, tags)):
            f.write(
                f"{_i+1} 4 2 {int(_tag)} {int(_tag)} "
                f"{_e[0]+1} {_e[1]+1} {_e[2]+1} {_e[3]+1}\n"
            )
        f.write("$EndElements\n")


# ---------------------------------------------------------------------------
# Trame app
# ---------------------------------------------------------------------------

# GeometryState (the per-project mutable data container) extracted
# to golgi.scene.geometry in step 3.1.
from golgi.scene.geometry import GeometryState  # noqa: E402
# Scene class + plotter factory (step 3.2). Owns pl + the
# declarative state_dict + the render-coalesce dispatch. The
# cuff-designer dialog also calls build_plotter() to get its
# second offscreen plotter.
from golgi.scene.renderer import Scene, build_plotter  # noqa: E402


# ---------------------------------------------------------------------------
# Single-fiber simulation backends (pyfibers + axonml).
# ---------------------------------------------------------------------------
# Lifted verbatim from nerve_studio.py §12 (pulse design helpers
# + backend wrappers). Both backends are LAZY-IMPORTED inside each
# function body: importing pyfibers triggers NEURON's HOC init
# which prints to stdout; axonml imports torch which is large; we
# don't want either to fire on `import golgi` when the user only
# wants to look at geometry. The first Run-button click is where
# the dep check happens, and the gatekeeper surfaces install
# instructions if the import fails.
#
# Sign convention (waveform): positive amp = "cathodic" (drives
# V_e negative on the cuff cathode → membrane depolarisation);
# negative = anodic.
# ---------------------------------------------------------------------------

AXON_BACKENDS = ("pyfibers", "axonml")

# pyfibers FiberModel choices, split by myelination. The list
# mirrors nerve_studio's §13 sweep; axonml's bundled surrogate is
# MRG-only, so the Fiber-tab dropdown filters to the first entry
# when backend == "axonml".
MYELINATED_MODELS = [
    "MRG_INTERPOLATION", "MRG_DISCRETE",
    "SMALL_MRG_INTERPOLATION", "SWEENEY",
]
# Thio autonomic/cutaneous fibers are UNMYELINATED (Thio et al. 2024,
# PLoS Comput Biol 20:e1012475 — "...autonomic and cutaneous unmyelinated
# fibers"); they were previously misfiled as myelinated.
UNMYELINATED_MODELS = [
    "SUNDT", "TIGERHOLM", "RATTAY", "SCHILD94",
    "SCHILD97", "THIO_AUTONOMIC", "THIO_CUTANEOUS",
]

# Per-model physiologically reasonable diameter ranges (µm).
# Used by the Single-fiber sim-row's slider+input pair to clamp
# the value to the right range for the picked model. MRG_DISCRETE
# is a SPECIAL case: only the listed discrete diameters are
# permitted (the model itself only ships tabulated parameters
# for these). For all other models we use a continuous slider
# with a small step.
#   key: model name
#   value: {
#       "min": float, "max": float, "step": float,
#       "default": float,  # used when the user switches models
#       "ticks": list[float] | None,  # discrete-only permitted
#                                     # values, else None
#   }
FIBER_MODEL_DIAMETER_CONFIG: dict[str, dict] = {
    # MRG family — large myelinated A-alpha fibres. Continuous
    # interpolation covers anything in this range; discrete uses
    # only the tabulated points the model ships with.
    "MRG_INTERPOLATION": {
        "min": 2.0, "max": 16.0, "step": 0.1, "default": 10.0,
        "ticks": None,
    },
    "MRG_DISCRETE": {
        "min": 5.7, "max": 16.0, "step": 0.1, "default": 10.0,
        "ticks": [
            5.7, 7.3, 8.7, 10.0, 11.5, 12.8, 14.0, 15.0, 16.0,
        ],
    },
    # Small myelinated A-delta / autonomic — narrower range.
    "SMALL_MRG_INTERPOLATION": {
        "min": 1.0, "max": 5.0, "step": 0.1, "default": 3.0,
        "ticks": None,
    },
    # Sweeney mammalian myelinated nerve (Sweeney et al. 1987).
    "SWEENEY": {
        "min": 2.0, "max": 16.0, "step": 0.1, "default": 10.0,
        "ticks": None,
    },
    "THIO_AUTONOMIC": {
        "min": 1.0, "max": 5.0, "step": 0.1, "default": 3.0,
        "ticks": None,
    },
    "THIO_CUTANEOUS": {
        "min": 1.0, "max": 5.0, "step": 0.1, "default": 3.0,
        "ticks": None,
    },
    # Unmyelinated C-fibres — sub-µm to ~2 µm. Finer step.
    "SUNDT": {
        "min": 0.2, "max": 1.5, "step": 0.05, "default": 1.0,
        "ticks": None,
    },
    "TIGERHOLM": {
        "min": 0.2, "max": 1.5, "step": 0.05, "default": 1.0,
        "ticks": None,
    },
    "RATTAY": {
        "min": 0.2, "max": 2.0, "step": 0.05, "default": 1.0,
        "ticks": None,
    },
    "SCHILD94": {
        "min": 0.2, "max": 1.5, "step": 0.05, "default": 1.0,
        "ticks": None,
    },
    "SCHILD97": {
        "min": 0.2, "max": 1.5, "step": 0.05, "default": 1.0,
        "ticks": None,
    },
}
# Sentinel returned when an unknown model is looked up — keeps
# the UI alive with a sensible default range.
_FIBER_MODEL_DIAMETER_DEFAULT = {
    "min": 0.1, "max": 20.0, "step": 0.1, "default": 5.7,
    "ticks": None,
}

# Simulation-duration range for the Single-fiber tab. Used by
# the duration slider+input pair on the sim-row.
FIBER_TSTOP_MIN_MS = 0.0
FIBER_TSTOP_MAX_MS = 1000.0
FIBER_TSTOP_STEP_MS = 0.5


def _fiber_effective_anod_pw_ms(
    cath_amp_mA: float, cath_pw_ms: float,
    anod_amp_mA: float, anod_pw_ms_user: float,
    charge_balance: bool,
) -> float:
    """Charge-balance the anodic phase. If `charge_balance` is on
    AND the anodic amplitude is non-zero, set anodic PW so net
    charge = 0 (anod_pw = cath_amp · cath_pw / anod_amp); else
    return the user-set PW unchanged."""
    if charge_balance and abs(float(anod_amp_mA)) > 1.0e-12:
        return (float(cath_amp_mA) * float(cath_pw_ms)
                / float(anod_amp_mA))
    return float(anod_pw_ms_user)


def build_pulse_waveform(
    t_grid_ms, t0_ms: float,
    cath_amp_mA: float, cath_pw_ms: float, gap_ms: float,
    anod_amp_mA: float, anod_pw_ms: float,
    anode_first: bool = False,
):
    """Sample the biphasic stim waveform on the simulator's `t_grid_ms`
    (ms), returning w(t) in mA. `anode_first` swaps phase 1 ↔ 2."""
    if anode_first:
        p1_amp, p1_pw = -float(anod_amp_mA), float(anod_pw_ms)
        p2_amp, p2_pw = +float(cath_amp_mA), float(cath_pw_ms)
    else:
        p1_amp, p1_pw = +float(cath_amp_mA), float(cath_pw_ms)
        p2_amp, p2_pw = -float(anod_amp_mA), float(anod_pw_ms)
    t_grid = np.asarray(t_grid_ms, dtype=np.float64)
    w = np.zeros_like(t_grid)
    t1_lo = float(t0_ms)
    t1_hi = t1_lo + p1_pw
    t2_lo = t1_hi + float(gap_ms)
    t2_hi = t2_lo + p2_pw
    if abs(p1_amp) > 1.0e-12 and p1_pw > 0:
        w[(t_grid >= t1_lo) & (t_grid < t1_hi)] = p1_amp
    if abs(p2_amp) > 1.0e-12 and p2_pw > 0:
        w[(t_grid >= t2_lo) & (t_grid < t2_hi)] = p2_amp
    return w


def build_pulse_breakpoints(
    t0_ms: float,
    cath_amp_mA: float, cath_pw_ms: float, gap_ms: float,
    anod_amp_mA: float, anod_pw_ms: float,
    anode_first: bool, tstop_ms: float,
):
    """Build (t_breakpoints, amp_breakpoints) suitable for
    `scipy.interp1d(kind='previous')`. pyfibers' ScaledStim gets
    the resulting waveform in mA directly, so pass amp=1.0 to
    run_sim — no extra scaling."""
    if anode_first:
        p1_amp, p1_pw = -float(anod_amp_mA), float(anod_pw_ms)
        p2_amp, p2_pw = +float(cath_amp_mA), float(cath_pw_ms)
    else:
        p1_amp, p1_pw = +float(cath_amp_mA), float(cath_pw_ms)
        p2_amp, p2_pw = -float(anod_amp_mA), float(anod_pw_ms)
    t1_hi = t0_ms + p1_pw
    t2_lo = t1_hi + gap_ms
    t2_hi = t2_lo + p2_pw
    tp = [0.0, float(t0_ms)]
    vp = [0.0, p1_amp]
    tp.append(t1_hi); vp.append(0.0)
    if abs(p2_amp) > 1.0e-12 and p2_pw > 0:
        tp.append(t2_lo); vp.append(p2_amp)
        tp.append(t2_hi); vp.append(0.0)
    tp.append(float(tstop_ms))
    vp.append(0.0)
    return np.asarray(tp), np.asarray(vp)


# ---------------------------------------------------------------------------
# axonml MRG-surrogate backend — extracted into
# golgi/pipeline/fiber_backends.py in step W1.6 of FEATURES.md. The
# only public name (axonml_run_single) is bundled into the H
# SimpleNamespace at build_app and consumed by
# pipeline/fiber_sim.py::_run_axonml_branch. Internal device /
# resampling / node-count helpers stay co-located in the new module.
# ---------------------------------------------------------------------------
from golgi.pipeline.fiber_backends import (  # noqa: E402, F401
    axonml_run_single,
)


# _FLARE_COLORSCALE / _FIBER_AXIS_TITLE_FONT / _FIBER_AXIS_TICK_FONT
# moved to golgi/figures/util.py — they were missed during step
# 1.2f's fiber.py extraction; the bug surfaced when single-fiber
# sim's _build_fiber_propagation_figure was first called after
# step 4.6 moved the call site out of build_app.



def build_app(port: int) -> None:
    _ensure_initialized()
    server = get_server(client_type="vue3")
    state, ctrl = server.state, server.controller

    # F2.2 — register a static-file endpoint for study-bundle
    # downloads so the action handler can stream large zips via
    # plain HTTP instead of a base64 data URI. The data URI path
    # falls over for 50+ MB bundles (browser stalls minutes
    # parsing the inline payload + the wslink WS transfer is
    # also slow). The dir is also reused by the report PDF +
    # bulk-figure-export ZIPs in a follow-up commit.
    _DOWNLOADS_DIR = PROJECTS_ROOT / "_downloads"
    _DOWNLOADS_DIR.mkdir(parents=True, exist_ok=True)
    server.serve["_downloads"] = str(_DOWNLOADS_DIR)
    # Best-effort cleanup of files older than 24h so the dir
    # doesn't grow unbounded across sessions.
    try:
        import time
        _now = time.time()
        for _p in _DOWNLOADS_DIR.iterdir():
            try:
                if (_now - _p.stat().st_mtime) > 24 * 3600:
                    _p.unlink(missing_ok=True)
            except Exception:                            # noqa: BLE001
                pass
    except Exception:                                    # noqa: BLE001
        pass
    geom = GeometryState()
    # Scene owns the main 3D viewport plotter (pl) + its actor
    # lifecycle. The local names `_scene_state`, `_rendered_sigs`,
    # `_main_loop_ref`, `_mkgrp`, `_next_sig`, `_apply_group`,
    # `_retire_unknown_actors`, `_render_scene`, `_request_render`,
    # `_set_actor_visible` are aliased below over the scene's
    # state + methods so existing call sites (~70) continue to
    # work unchanged.
    scene = Scene(
        geom=geom, state=state, ctrl=ctrl,
        loop_factory=asyncio.get_running_loop,
        region_tags=(1, 2, 3, 4, 5, TAG_SCAR, TAG_GOLD),
        max_fiber_branches=MAX_FIBER_BRANCHES,
    )
    pl = scene.pl
    # Second plotter dedicated to the electrode-designer dialog.
    # Built lazily by build_plotter() with the same tuned material
    # look so the cuff preview matches the workspace styling.
    pl_cuff = build_plotter()
    # Third plotter dedicated to the µCT-reconstruction preview
    # in the Segment dialog's Step 3. Same build_plotter() →
    # same lighting / camera style as the workspace plotter, so
    # the user sees the same "look" they'll get after import.
    pl_uct_recon = build_plotter()
    # Actor name → mesh cache, so view-option watchers (edges,
    # quality colormap, per-mesh visibility) can rebuild actors
    # without re-running marching cubes.
    _uct_recon_meshes_cache: list = []

    # Pipeline-driver context — forward-declared so the do_*
    # handlers can close over the name. Actual construction
    # happens further down (right before VAppLayout), after every
    # closure / helper the context wraps is defined. Late-binding
    # closure lookup means each do_* call sees the real ctx by
    # the time the UI is up.
    _pipeline_ctx: "PipelineContext | None" = None

    # -----------------------------------------------------------------
    # safe view update (registered when plotter_ui() runs)
    # -----------------------------------------------------------------
    def safe_update():
        try:
            ctrl.view_update()
        except Exception:
            pass

    # ================================================================
    # Auth handlers + gatekeeper decorator.
    # ================================================================
    # `_auth_session` (module-global, thread-safe via `_session_lock`)
    # is the source of truth — the `gated()` wrapper consults it, NOT
    # the client-mirrored `state.authenticated`. State writes from a
    # hostile client cannot bypass the gate.
    # ----------------------------------------------------------------

    def _stamp_user_line(line: str) -> str:
        """Prepend the current user's email to a log line. No-op when
        nobody is logged in (preserves pre-auth log formatting)."""
        email = _auth_session.get("email")
        if not email:
            return line
        return f"[{email}] {line}"

    def _refresh_projects_for_user() -> None:
        """Re-run the owner-filtered project scan and push the result
        to the welcome view's tile list. Also refreshes the
        registered-user list so the share-picker in the detail
        dialog has the latest set without needing a server
        restart."""
        uid = _auth_session.get("user_id")
        try:
            state.projects_list = _list_projects(owner_user_id=uid)
        except Exception:
            pass
        try:
            state.users_list = _list_users()
        except Exception:
            pass

    _USERNAME_RE = re.compile(r"^[A-Za-z0-9._@\-]{3,64}$")

    def _decode_avatar_data_uri(data_uri: str) -> bytes | None:
        """Decode a base64 `data:image/...;base64,…` URI into raw
        bytes for storage. Returns None for empty / malformed
        input. Caller validates size + mime before commit."""
        if not data_uri:
            return None
        s = str(data_uri).strip()
        if not s.startswith("data:"):
            return None
        try:
            _, b64 = s.split(",", 1)
        except ValueError:
            return None
        try:
            return base64.b64decode(b64)
        except Exception:
            return None

    def _validate_avatar_bytes(b: bytes | None) -> str:
        """Return "" if `b` is acceptable, else a user-facing
        error message. Empty / None bytes are acceptable (means
        "no avatar")."""
        if not b:
            return ""
        if len(b) > AVATAR_MAX_BYTES:
            return (
                f"Image too large "
                f"({len(b) / 1024:.0f} KB > "
                f"{AVATAR_MAX_BYTES // 1024} KB max). "
                "Pick a smaller picture or use an external "
                "tool to compress it."
            )
        if _sniff_image_mime(b) is None:
            return (
                "Unsupported image format. Use PNG, JPEG, or "
                "WEBP."
            )
        return ""

    def _push_auth_session(user_row: "_User") -> None:
        """Commit a successful login: capture the session lock,
        mirror identity to state vars (incl. username + avatar
        data URI), refresh per-user project list, audit-log the
        login. Caller already holds `_session_lock`."""
        _auth_session["user_id"] = int(user_row.id)
        _auth_session["email"] = str(user_row.email)
        _auth_session["username"] = str(
            user_row.username or user_row.email,
        )
        _auth_session["since"] = datetime.now(timezone.utc)
        _auth_session["session_token"] = secrets.token_urlsafe(24)
        avatar_uri = _user_avatar_data_uri(user_row)
        with state:
            state.authenticated = True
            state.current_user_id = int(user_row.id)
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
            state.session_locked_by = ""
            state.show_auth_dialog = False
            state.auth_mode = "login"
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
        _refresh_projects_for_user()
        _audit_log(int(user_row.id), "login",
                   payload={"email": str(user_row.email),
                            "username": str(user_row.username)},
                   status="success")

    def _clear_auth_session() -> None:
        """Release the session lock + clear identity. Audit the
        logout event using whatever user_id was active."""
        prev_uid = _auth_session.get("user_id")
        prev_email = _auth_session.get("email")
        with _session_lock:
            _auth_session["user_id"] = None
            _auth_session["email"] = None
            _auth_session["username"] = None
            _auth_session["since"] = None
            _auth_session["session_token"] = None
        with state:
            state.authenticated = False
            state.current_user_id = 0
            state.current_user_email = ""
            state.current_user_username = ""
            state.current_user_first_name = ""
            state.current_user_last_name = ""
            state.current_user_country = ""
            state.current_user_institution = ""
            state.current_user_position = ""
            state.current_user_avatar = ""
            state.session_locked_by = ""
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
            state.show_user_menu = False
        # Refresh tiles back to the logged-out (empty) list.
        try:
            state.projects_list = _list_projects(owner_user_id=None)
        except Exception:
            pass
        if prev_uid is not None:
            _audit_log(int(prev_uid), "logout",
                       payload={"email": prev_email},
                       status="success")


    # ----------------------------------------------------------------
    # Flight recorder — `@log_action` / `@gated` decorator pair.
    # ----------------------------------------------------------------
    # Both decorators emit one audit row per call via `_audit_log`
    # (which enqueues, never blocks). The row carries:
    #   * user_id        — from `_auth_session` (server-local).
    #   * timestamp      — UTC, stamped at enqueue.
    #   * action_name    — the string passed to the decorator.
    #   * parameters     — JSON: {"args": [...], "kwargs": {...},
    #                              "error": "..."}  (error only on
    #                                                 failure).
    #   * project_dir    — current project (if any).
    #   * status         — "success" | "failure" | "blocked".
    #
    # `@log_action` is for ANY state-mutating handler. Wraps both
    # sync + async callables. `@gated` adds the "must be signed
    # in" check on top — block-when-anonymous logs a "blocked"
    # status so we can see attempted-while-logged-out events.
    #
    # The decorator BODIES live in golgi/auth/decorators.py (lifted
    # out of build_app in step 2.4) — here we just bind them to the
    # live trame state via AuthContext.
    # ----------------------------------------------------------------
    _auth_ctx = AuthContext(state=state)
    log_action = make_log_action(_auth_ctx)
    gated = make_gated(_auth_ctx)

    # ----------------------------------------------------------------
    # Scene state — single declarative target for the 3D viewport.
    # ----------------------------------------------------------------
    # Every actor in `pl` is owned by exactly one entry in
    # `_scene_state`. Watchers update this dict (and `geom`/`state`);
    # only `_render_scene()` mutates the plotter, on the main thread.
    # `_request_render()` coalesces many updates per tick.
    #
    # Step 3.2 of migration.md: the actor-lifecycle machinery
    # (`_mkgrp`, `_next_sig`, `_apply_group`, `_retire_unknown_actors`,
    # `_render_scene`, `_request_render`, `_set_actor_visible`, the
    # `_main_loop_ref` capture, and the `_scene_state` /
    # `_rendered_sigs` dicts) all live on the Scene instance now.
    # The names below are aliases so existing call sites keep
    # resolving to the same dicts and functions.
    # ----------------------------------------------------------------
    _scene_state = scene.state_dict
    _rendered_sigs = scene.rendered_sigs
    _main_loop_ref = scene.main_loop_ref
    _mkgrp = scene.mkgrp
    _next_sig = scene.next_sig
    _apply_group = scene.apply_group
    _retire_unknown_actors = scene.retire_unknown

    _render_scene = scene.render_scene
    _request_render = scene.request_render

    # SceneCatalog Phase 1 — registry + comparison harness.
    # Phase 1 ships the schema only (no entries registered);
    # Phases 2-5 register entries section by section and grow
    # `_catalog_sections_active` to enable diff-checking of
    # newly-ported sections. Phase 6 retires the inline
    # `_set_*_group` functions entirely and the catalog becomes
    # the sole source for scene_state.
    from golgi.scene.catalog import (
        Catalog as _Catalog,
        SceneEntry as _SceneEntry,
        compare_scene_states as _compare_scene_states,
        to_pca_mm as _to_pca_mm,
    )
    _scene_catalog = _Catalog()
    _catalog_sections_active: tuple[str, ...] = ()

    # Forward declaration target. _rebuild_scene_state's body is
    # assigned later (closer to the per-group builders) so it can
    # close over geom/state/helpers defined further down. Until
    # then, calling _request_render is a no-op render.
    def _rebuild_scene_state() -> None:  # noqa: E306
        return None

    # The Scene's request_render dispatch needs to call the live
    # _rebuild_scene_state name at run time (not the stub
    # captured here). Lambda gives us late-binding lookup.
    scene.rebuild_callback = lambda: _rebuild_scene_state()

    def do_toggle_elec_vis(eid: str, field: str) -> None:
        """Flip a per-electrode visibility flag in-place + apply.
        Bound to the legend rows so clicking a "Cuff N" entry (or
        one of its silicone/saline/contacts sub-rows) directly
        mutates the electrode dict and updates the actors —
        no full re-render, no slider edit detour."""
        if not eid or field not in (
            "vis_master",
            "vis_endo", "vis_epi", "vis_muscle",
            "vis_silicone", "vis_saline", "vis_contacts",
            # F3.2-M3 — per-design scar legend toggle.
            "vis_scar",
            # F3.2-M2.1e — Meshes section's per-design toggles.
            "vis_mesh", "vis_mesh_quality",
        ):
            return
        electrodes = list(state.designs or [])
        for idx, e in enumerate(electrodes):
            if e.get("eid") != eid:
                continue
            # New dict so Vue + trame see a row-level reference
            # change and the legend's is-off class updates.
            new_e = dict(e)
            new_e[field] = not bool(new_e.get(field, True))
            electrodes[idx] = new_e
            break
        state.designs = electrodes
        # If the toggled electrode is the current selection,
        # mirror the new value to the state var so any drawer
        # control stays in sync (guarded so we don't bounce
        # through the watcher).
        if (str(state.selected_design_id or "") == eid):
            for e in electrodes:
                if e.get("eid") != eid:
                    continue
                _elec_sync_guard["loading"] = True
                try:
                    state[field] = bool(e.get(field, True))
                finally:
                    _elec_sync_guard["loading"] = False
                break
        _apply_electrode_visibility()
        safe_update()

    _set_actor_visible = scene.set_actor_visible

    def _apply_electrode_visibility() -> None:
        """Per-electrode visibility flip (vis_master + sub-flags).
        The scene-state pipeline folds per-electrode flags directly
        into the actor visibility in `_set_electrode_groups`; this
        thin shim now just requests a render pass.

        Important behaviour change (see plan G6/Q1, Decouple option):
        the FEM-mesh region actors (`region_2/3/TAG_GOLD`) are no
        longer driven by an OR-aggregate over per-electrode flags.
        Each UI control owns exactly one scene flag. The global
        `vis_2/3/TAG_GOLD` toggles in the legend remain the way to
        hide the FEM-mesh regions."""
        _request_render()

    def safe_reset_camera():
        """Push the *server's* reset camera to the client.

        Important: `view.reset_camera()` asks the JS client to
        compute its own reset from whatever scene it currently
        has — that races with the geometry push (the client may
        not have the new actors yet, in which case it resets to
        empty bounds and the nerve ends up outside the frustum).

        `view.push_camera()` instead serialises the server-side
        camera (which we already fitted via `pl.reset_camera()`
        right after adding the actors) and sends those params
        verbatim. No race — the client just adopts the camera
        the server has already computed correctly.
        """
        try:
            ctrl.view_push_camera()
        except Exception:
            # Fall back to client-side reset_camera if push isn't
            # wired yet (e.g. earliest startup window).
            try:
                ctrl.view_reset_camera()
            except Exception:
                pass

    # -----------------------------------------------------------------
    # State defaults — extracted into golgi.state_defaults per topic
    # (step 5.1 of migration.md). Each register() seeds the relevant
    # state.* keys to their factory defaults.
    # -----------------------------------------------------------------
    _state_defaults.ui_toggles.register(state)
    _state_defaults.fem.register(state)
    _state_defaults.fiber.register(
        state,
        fiber_diameter_config=FIBER_MODEL_DIAMETER_CONFIG,
        fiber_diameter_default=_FIBER_MODEL_DIAMETER_DEFAULT,
        tab10_palette=TAB10_PALETTE,
    )
    _state_defaults.pop.register(state)
    _state_defaults.sweep.register(state)  # F2.1.c
    _state_defaults.exports.register(state)  # F2.3.a
    _state_defaults.study_bundle.register(state)  # F2.2
    _state_defaults.import_state.register(
        state, list_data_files=list_data_files,
    )
    _state_defaults.mesh.register(state)
    _state_defaults.cuff.register(state, default_cuff=DEFAULT_CUFF)
    _state_defaults.electrode.register(
        state, default_electrode=DEFAULT_ELECTRODE,
    )

    # ----- Multi-electrode model (A1) -----
    # state.designs is the source of truth — a LIST of dicts,
    # each capturing the full per-electrode parameter set. The
    # existing legacy state vars (cuff_offset_mm, electrode_type,
    # …) mirror the *selected* electrode so the existing sliders
    # / watchers keep working without refactoring every callsite.
    #
    # Workflow:
    #   - User selects electrode N → save current state vars into
    #     N's dict, then load N's dict into the state vars.
    #   - User edits a slider → existing watchers fire (re-fit,
    #     re-render) AND `_save_selected_to_designs` keeps the
    #     dict in sync.
    #   - Renderer iterates state.designs and mounts each one
    #     under namespaced actor keys `elec_<eid>_…`.

    # Fields that belong to a single electrode and mirror the
    # legacy state vars. Tuple in order so save/load loops are
    # deterministic.
    _ELEC_MIRROR_KEYS: tuple = (
        "L_cuff_mm", "cuff_offset_mm",
        "cuff_dx_mm", "cuff_dy_mm",
        "cuff_rot_x_deg", "cuff_rot_y_deg", "cuff_rot_z_deg",
        "cuff_clearance_mm", "cuff_wall_mm",
        "show_saline",
        # F3.2-M3 — per-design scar / connective tissue shell.
        "use_scar", "scar_thickness_um",
        # Per-design visibility flags (master + tissue + cuff-
        # part sub-components). Setting any to False just toggles
        # actor.visibility — the geometry stays mounted so re-
        # enabling is instant. F3.2-M1 expanded the per-design
        # vocabulary with vis_endo / vis_epi / vis_muscle so each
        # design's tissue regions can toggle independently.
        "vis_master",
        "vis_endo",
        "vis_epi",
        "vis_muscle",
        "vis_silicone",
        "vis_saline",
        "vis_contacts",
        # F3.2-M3 — per-design scar / connective tissue visibility.
        "vis_scar",
        # M1.1: per-design has_mesh flag (whether nerve.msh
        # exists for this design). Persisted so on project open
        # the legend correctly hides tissue rows for designs
        # whose mesh hasn't been built yet.
        "has_mesh",
        # M2.0: per-design has_fem flag (whether any solved
        # config exists for this design). Gates Overlays
        # sub-section in legend.
        "has_fem",
    ) + tuple(DEFAULT_ELECTRODE.keys())

    def _contact_count(elec: dict) -> int:
        """How many distinct contacts does this electrode expose?
        Drives the contact-polarity table in the drawer + the
        per-contact tinting in the 3-D view."""
        kind = str(
            elec.get("electrode_type", "bipolar ring-pair"),
        )
        if kind == "bipolar ring-pair":
            return 2
        if kind == "tripolar (anode-cathode-anode)":
            return 3
        if kind == "ring-array (NxM)":
            try:
                rows = int(elec.get("array_n_rows", 2))
                cols = int(elec.get("array_n_cols", 4))
            except (TypeError, ValueError):
                rows, cols = 2, 4
            return max(0, rows * cols)
        if kind == "helical (Livanova-style)":
            return 2
        if kind == DUKE_ELECTRODE_TYPE:
            preset_name = str(elec.get("duke_preset", "") or "")
            preset = _CUFF_PRESETS.get(preset_name)
            if preset is None:
                return 0
            # One contact per LivaNova_Primitive (helical
            # conductor) or CircleContact_Primitive instance.
            n = 0
            for inst in preset.get("instances", []):
                if inst.get("type") in (
                    "LivaNova_Primitive",
                    "CircleContact_Primitive",
                ):
                    n += 1
            return n
        return 0

    def _default_polarities(elec: dict) -> list:
        """Polarity defaults to match each electrode's canonical
        firing pattern: bipolar = anode/cathode, tripolar =
        anode/cathode/anode, larger arrays alternate."""
        n = _contact_count(elec)
        if n <= 0:
            return []
        if n == 1:
            return ["anode"]
        if n == 2:
            return ["anode", "cathode"]
        if n == 3:
            return ["anode", "cathode", "anode"]
        return [
            "anode" if (i % 2 == 0) else "cathode"
            for i in range(n)
        ]

    def _ensure_polarities(elec: dict) -> list:
        """Returns a polarity list of the right length for the
        electrode, persisting defaults onto the dict the first
        time. Called from both the loader and the renderer so an
        electrode that pre-dates this feature picks up sensible
        defaults the moment it's touched.

        M1 migration: legacy on-disk projects might still carry
        the pre-M1 vocab ("active") — map it to "anode" here so
        the rest of the codebase sees a consistent set."""
        n = _contact_count(elec)
        existing = elec.get("contact_polarities", None)
        if isinstance(existing, list) and len(existing) == n:
            migrated = [
                ("anode" if p == "active" else p)
                for p in existing
            ]
            if all(p in POLARITY_CHOICES for p in migrated):
                elec["contact_polarities"] = migrated
                return list(migrated)
        pols = _default_polarities(elec)
        elec["contact_polarities"] = pols
        return list(pols)

    def _compute_polarity_sums(
        pols: list, fracs: list,
    ) -> list:
        """Build the sum-check chip data for the polarity table.
        One entry per polarity group present in `pols`; carries
        the sum of EXPLICIT fractions, the count of explicit-
        and implicit-fraction contacts, and a colour hint
        (success / warning / disabled).

        Sum-check semantics:
          * All implicit (no fraction set anywhere) → "equal
            share" → chip shows N contacts split evenly →
            success colour.
          * All explicit, sum ≈ 1.0 → success.
          * Mixed or explicit-but-not-1.0 → warning, FEM driver
            normalises but the user probably wanted explicit.
        """
        out: list[dict] = []
        # Stable order: cathode (the stimulating one) first,
        # then anode, then ground. "off" contacts don't
        # contribute to any group.
        for label in ("cathode", "anode", "ground"):
            idxs = [i for i, p in enumerate(pols) if p == label]
            n_total = len(idxs)
            if n_total == 0:
                continue
            explicit_vals = [
                float(fracs[i]) for i in idxs
                if (i < len(fracs)
                    and fracs[i] is not None)
            ]
            n_explicit = len(explicit_vals)
            s = sum(explicit_vals)
            # Colour: success when all-implicit (equal split)
            # OR all-explicit + sums to ~1.0 within 0.5%.
            if n_explicit == 0:
                color = "success"
                hint = f"Equal split ({n_total} × 1/{n_total})"
            elif n_explicit == n_total and abs(s - 1.0) <= 0.005:
                color = "success"
                hint = f"Σ = {s:.3f}"
            else:
                color = "warning"
                hint = (
                    f"Σ = {s:.3f} of explicit "
                    f"{n_explicit}/{n_total}"
                )
            out.append({
                "label": label,
                "n_total": n_total,
                "n_explicit": n_explicit,
                "sum": s,
                "color": color,
                "hint": hint,
            })
        return out

    def _ensure_current_fractions(elec: dict) -> list:
        """Returns a per-contact current-fraction list of the
        right length, persisting defaults the first time.
        Entry is `None` for "use equal share within polarity
        group" (the FEM driver computes the split at solve time)
        or a float in [0, 1] for an explicit fraction."""
        n = _contact_count(elec)
        existing = elec.get("contact_current_fractions", None)
        if (isinstance(existing, list) and len(existing) == n):
            # Coerce items to float | None.
            out = []
            for v in existing:
                if v is None or v == "":
                    out.append(None)
                else:
                    try:
                        out.append(float(v))
                    except (TypeError, ValueError):
                        out.append(None)
            elec["contact_current_fractions"] = out
            return list(out)
        out = [None] * n
        elec["contact_current_fractions"] = out
        return list(out)

    def _new_electrode_default(eid: str,
                                  name: str = "",
                                  z_offset_mm: float = 10.0,
                                  ) -> dict:
        """Build an Electrode dict with the same defaults the legacy
        single-cuff path used to start from. `z_offset_mm` lets
        the caller stagger newly-added electrodes along the nerve
        so they don't all stack on top of electrode #1."""
        d = {
            "eid": eid,
            "name": name or eid.replace("elec_", "Cuff "),
            # Mirror fields (initially copied from DEFAULT_*)
            **{k: DEFAULT_CUFF[k] for k in DEFAULT_CUFF
                 if k in _ELEC_MIRROR_KEYS},
            **{k: DEFAULT_ELECTRODE[k] for k in DEFAULT_ELECTRODE},
            # Resolved cuff radii (filled in by do_fit_cuff)
            "R_ci_m": None,
            "R_co_m": None,
            # DUKE designer config (used in A3)
            "duke_preset": "",
            "duke_overrides": {},
            # Contact polarity — list of "off" / "anode" /
            # "cathode" with one entry per contact. Populated
            # below once we know the electrode type.
            "contact_polarities": [],
            # M1: per-contact current_fraction (0..1) within
            # polarity group. None = "equal share" (FEM driver
            # computes 1/N_in_group at solve time).
            "contact_current_fractions": [],
            # Per-design visibility — master + sub-components.
            # Sub-toggles are gated by master in the renderer:
            # turning master off hides everything (tissues +
            # electrode parts) for THIS design regardless of the
            # sub-flag values. F3.2-M1 adds vis_endo / vis_epi /
            # vis_muscle so each design owns its own tissue
            # visibility too (was global vis_1 / vis_5 / vis_4
            # pre-M1).
            "vis_master": True,
            "vis_endo": True,
            "vis_epi": True,
            "vis_muscle": True,
            "vis_silicone": True,
            "vis_saline": True,
            "vis_contacts": True,
            # F3.2-M3 — per-design scar / connective tissue
            # visibility. Default visible; toggle from the
            # legend's scar row (gated on use_scar).
            "vis_scar": True,
            # F3.2-M2.1e — per-design whole-mesh visibility.
            # The legend's "Meshes" section drives this single
            # flag instead of the six per-tag sub-toggles above.
            # Renderer treats it as "design's master AND this
            # flag" — turning it off hides every meshed actor
            # (endo, epi, muscle, silicone, saline, contacts)
            # for this design. The per-tag flags stay in the
            # dict for backwards compatibility but the UI no
            # longer exposes them.
            "vis_mesh": True,
            # F3.2-M2.1e — per-design mesh-quality colour
            # overlay. Re-styles every region actor for this
            # design with the quality-ratio scalar mapped to
            # RdYlGn. Replaces the old global
            # show_mesh_quality_color flag.
            "vis_mesh_quality": False,
            # F3.2-M1.1: per-design has_mesh flag. Set True
            # when this design's nerve.msh is built; gates the
            # tissue sub-rows in the legend so the user doesn't
            # see toggles for actors that don't exist yet.
            "has_mesh": False,
            # F3.2-M2.0: per-design has_fem flag. Set True
            # when at least one config bound to this design
            # has completed an FEM solve; gates the Overlays
            # sub-section in the legend. Computed by
            # `_on_config_items_rebuild` from state.fem_configs.
            "has_fem": False,
        }
        d["cuff_offset_mm"] = float(z_offset_mm)
        d["contact_polarities"] = _default_polarities(d)
        d["contact_current_fractions"] = (
            [None] * len(d["contact_polarities"])
        )
        return d

    # Start with NO electrodes — the user explicitly adds them
    # via the drawer's "+ Add" button. Clicking the Electrodes
    # tab no longer auto-creates a cuff.
    state.designs = []
    state.selected_design_id = ""
    state.next_design_seq = 1
    # Mirror of the currently-selected electrode's contact
    # polarities — a list of "off" / "anode" / "cathode" entries,
    # one per contact. The drawer's polarity table is v-for over
    # this list; on change a watcher writes it back into the
    # selected electrode's dict and triggers a translate-only
    # re-render so the anode/cathode tints update in real time.
    state.contact_polarities = []
    state.contact_count = 0
    # M1 — per-contact current fractions (one entry per contact).
    # Mirrors the selected electrode's contact_current_fractions.
    # Each entry is float (0..1) or None ("equal share within
    # polarity group" — FEM driver computes 1/N_in_group at
    # solve time).
    state.contact_current_fractions = []
    # M1 — per-polarity sum-check chips. The drawer renders a
    # small chip per polarity group (anode / cathode / ground)
    # showing the current sum of explicit fractions; green when
    # ≈ 1.0, amber otherwise. Lists hold {label, sum,
    # n_explicit, n_total, color} dicts so the v-for has all the
    # info inline.
    state.contact_polarity_sums = []
    # M1 — quick-preset name currently selected (display only).
    state.contact_preset = ""
    # F3.2b — first-class contact configurations. A "config" is a
    # named polarity + current-fraction pattern attached to ONE
    # design. Multiple configs can share a design (same physical
    # cuff, different wirings → no remesh, just re-solve). State
    # vars:
    #   configs           : list of dicts, each
    #                       {cid, design_id, name,
    #                        contact_polarities[],
    #                        contact_current_fractions[],
    #                        I_stim_mA}
    #   selected_config_id: cid of the currently-active config
    #                       (drives state.contact_polarities etc.
    #                       via the load/save watchers below)
    #   next_config_seq   : monotonic id counter for cid
    #                       generation; survives close/reopen.
    state.configs = []
    state.selected_config_id = ""
    state.next_config_seq = 1
    # R1.1 — Recording montages. Each config carries its own
    # bipolar (or future N-polar) recording montages alongside
    # polarities + fractions. Per-config schema:
    #   recording_montages : list of dicts, each
    #                        {mid, label, plus_contact,
    #                         minus_contact, kind, color}
    # The mirror state.recording_montages tracks the active
    # config's list (drawer + viewport bind to this). The
    # derived state.contact_montage_map is keyed by 0-based
    # contact index → {mid, label, color, pole} for the
    # per-contact-row badge in the drawer.
    state.recording_montages = []
    state.contact_montage_map = {}
    state.next_montage_seq = 1
    # R1.4 — cNAP results. `geom.cnap_single` and `geom.cnap_pop`
    # are dicts keyed by montage id; the state mirrors below
    # hold the per-montage figure + active-mid selector so the
    # analysis panels can render the trace + react to the
    # montage dropdown without re-running the sim.
    state.fiber_cnap_figure = {"data": [], "layout": {}}
    state.fiber_cnap_status = ""
    state.pop_cnap_figure = {"data": [], "layout": {}}
    state.pop_cnap_status = ""
    state.active_montage_single = ""
    state.active_montage_pop = ""
    # Population panel toggle: stack per-fiber-type traces beneath
    # the total cNAP. On by default — it's the most informative
    # view for "which fibers produce this peak".
    state.cnap_decompose_by_type = True
    # Inline montage editor — modal-less form that opens below
    # the montage list. `editing_montage_id` is empty string in
    # "add new" mode, set to a mid in "edit existing" mode.
    state.show_montage_editor = False
    state.editing_montage_id = ""
    state.montage_form_label = ""
    state.montage_form_plus = -1     # 0-based contact id; -1 = unset
    state.montage_form_minus = -1
    state.montage_form_error = ""
    # Palette the UI cycles through for new montages. Kept in
    # state so the cuff drawer can read it without importing a
    # constant from app.py.
    state.montage_palette = [
        "#22c55e", "#a855f7", "#06b6d4", "#f59e0b",
        "#ef4444", "#3b82f6", "#ec4899", "#84cc16",
    ]
    # F3.2a — Mesh tab multi-select. List of design eids the
    # user wants to build a mesh for. Default empty → the
    # driver falls back to the currently-selected design (legacy
    # single-cuff workflow).
    state.mesh_design_selection = []
    # F3.2c — Solve tab multi-select. List of config cids the
    # user wants to run FEM solves for. Default empty → driver
    # falls back to the currently-active config
    # (state.selected_config_id).
    state.solve_config_selection = []
    # I1 Phase A — DC impedance state. Default-on toggle gates
    # the per-contact dirichlet dual-solve at the end of every
    # FEM solve. `fem_impedance` is a dict keyed by cid; each
    # value is the impedance.json payload (per_contact +
    # per_pair arrays).
    state.emit_impedance = True
    state.fem_impedance = {}
    state.impedance_bar_figure = {"data": [], "layout": {}}
    state.impedance_per_pair_figure = {"data": [], "layout": {}}

    # V1 Phase A — µCT segmentation dialog state.
    # `show_segment_uct_dialog` is the modal v-model. Stack
    # metadata, slice index, the rendered overlay (PNG data
    # URL) and per-proposal chip metadata live on the state
    # proxy; the raw NumPy arrays + Segmenter instance stay in
    # the segment_uct actions module's closure to avoid msgpack-
    # encoding heavy objects across the WebSocket.
    state.show_segment_uct_dialog = False
    # M47 — histology bundle import dialog state.
    state.show_bundle_import_dialog = False
    state.bundle_dir_path = ""
    state.bundle_slide_path = ""
    state.bundle_nerve_path = ""
    state.bundle_fasc_path = ""
    state.bundle_scale_path = ""
    state.bundle_scale_bar_um = 1000.0
    state.bundle_thickness_mm = 10.0
    state.bundle_pixel_pitch_um = 0.0
    state.bundle_detect_error = ""
    state.bundle_status = ""
    state.bundle_scale_preset_items = [
        {"title": "500 µm", "value": 500.0},
        {"title": "1 mm",   "value": 1000.0},
        {"title": "2 mm",   "value": 2000.0},
        {"title": "5 mm",   "value": 5000.0},
    ]
    state.uct_file_path = ""
    # Default voxel pitch — matches the µCT scanner the lab
    # uses by default. The user can override per-stack via the
    # field in the Segment dialog; the scale bar reads the
    # current value live.
    state.uct_voxel_size_um = 10.4
    state.uct_stack_loaded = False
    state.uct_stack_info_html = ""
    state.uct_slice_idx = 0
    state.uct_slice_max = 0
    state.uct_overlay_url = ""
    # Native pixel dimensions of the currently-rendered slice
    # (post-crop, post-zoom, pre-PNG-downsample). Together with
    # `uct_voxel_size_um` these let the dialog overlay a
    # physical-scale bar that stays correct under crop / zoom /
    # window resize.
    state.uct_image_orig_w = 0
    state.uct_image_orig_h = 0
    state.uct_segmenter_name = ""
    state.uct_segmenter_warning = ""
    state.uct_proposals_meta = []
    state.uct_label_counts = {}
    state.uct_busy = False
    state.uct_status = ""
    # SAM2 video / keyframe-propagation state. The Segment dialog
    # offers an opt-in workflow where the user marks 1-5 slices
    # as "keyframes" (their per-slice labelled masks become
    # SAM2 conditioning prompts) and clicks Propagate to fill the
    # rest of the stack via forward + backward video propagation.
    # The actions module owns the heavy SAM2VideoSegmenter
    # instance + per-stack inference state; these state vars are
    # the trame-side mirror that drives the UI section.
    state.uct_keyframe_slices = []
    state.uct_keyframe_summary = ""   # "3 (slices 0, 17, 32)"
    state.uct_propagation_busy = False
    state.uct_sam2_video_available = False
    state.uct_sam2_video_reason = ""
    # Probe SAM2 video at startup. Cheap (module import +
    # checkpoint path stat — no model load), so the dialog can
    # gate the Propagate button + show a tooltip the moment it
    # opens. The heavy `build_sam2_video_predictor` only fires
    # on the first Propagate click.
    try:
        from golgi.segmentation.segmenter import (
            sam2_video_available as _sam2v_probe,
        )
        _ok, _reason = _sam2v_probe()
        state.uct_sam2_video_available = bool(_ok)
        state.uct_sam2_video_reason = str(_reason)
    except Exception as _ex:                          # noqa: BLE001
        state.uct_sam2_video_available = False
        state.uct_sam2_video_reason = f"probe failed: {_ex}"
    # V1 Phase A.7 — browser-side upload progress. The JS XHR
    # uploader in golgi_uct_upload.js mutates these via
    # trame.state.update so the dialog's VProgressLinear +
    # status banner reflect bytes-on-the-wire in real time.
    state.uct_uploading = False
    state.uct_upload_progress = 0
    state.uct_upload_status = ""
    state.uct_upload_error = ""
    # VFileInput's v-model — Vuetify needs a writeable binding
    # so the picker UI updates after a selection. Our JS
    # uploader reads the file from the @update:modelValue event
    # directly, but the v-model has to exist as state to keep
    # Vuetify's internal state consistent.
    state.uct_file_input = None
    # Drag-and-drop active flag — flipped by @dragover/@dragleave
    # on the image panel so a "drop file to upload" overlay
    # appears while the user is dragging.
    state.uct_drag_active = False
    # Click payload — pushed by the dialog's @mouseup when the
    # gesture was a click (not a drag). Triple of
    # [image_x, image_y, timestamp_ms] so the @state.change
    # watcher re-fires even on repeated clicks at the same
    # pixel (otherwise an identical array value wouldn't
    # trigger the watcher). Server-side handler finds the
    # smallest proposal mask containing the point and cycles
    # its label: unlabeled → fascicle → background → unlabeled.
    state.uct_click_payload = [0, 0, 0]
    # Tool mode for the image-panel mouse interactions. The
    # toolbar in the dialog flips this; the @mousedown/
    # @mousemove/@mouseup expressions dispatch to different
    # behaviours per mode.
    state.uct_tool_mode = "crop"
    state.uct_tool_items = [
        {
            "value": "crop", "title": "Crop",
            "icon": "mdi-crop",
        },
        {
            "value": "zoom", "title": "Zoom",
            "icon": "mdi-magnify-plus-outline",
        },
        {
            "value": "paint", "title": "Paint",
            "icon": "mdi-brush",
        },
        {
            "value": "erase", "title": "Erase",
            "icon": "mdi-eraser",
        },
    ]
    # Active stamp label — clickable pills in the "Assign
    # labels" section flip this. In crop / zoom modes, a
    # single click on a proposal mask applies this label. In
    # paint / erase modes the brush gestures take over (paint
    # uses uct_paint_label which is a SEPARATE picker for the
    # paint target).
    state.uct_active_label = "fascicle"
    state.uct_active_label_items = [
        {
            "value": "fascicle",   "title": "Fascicle",
            "color": "primary",
        },
        {
            "value": "background", "title": "Background",
            "color": "error",
        },
        {
            "value": "epi",        "title": "Epineurium",
            "color": "success",
        },
        {
            "value": "unlabeled",  "title": "None",
            "color": "grey",
        },
    ]
    # Zoom range — display-only further crop on top of the data
    # crop. When [0, 0] / equal to crop, no zoom is applied.
    # Reset button restores to no-zoom.
    state.uct_zoom_x_range = [0, 0]
    state.uct_zoom_y_range = [0, 0]
    # Paint / erase parameters. uct_paint_label = target class
    # for new strokes; uct_brush_radius in image-pixel space.
    # uct_paint_payload is the per-stroke push from JS:
    # [image_x, image_y, paint_or_erase (1/0), timestamp].
    state.uct_paint_label = "fascicle"
    state.uct_paint_label_items = [
        {"value": "fascicle",   "title": "Fascicle"},
        {"value": "background", "title": "Background"},
        {"value": "epi",        "title": "Epineurium"},
    ]
    state.uct_brush_radius = 12
    state.uct_paint_payload = [0, 0, 0, 0]
    # Segmentation backend picker. Default to vanilla SAM2 —
    # the user found it works better on µCT than MedSAM2 in
    # practice (vanilla's 11M-mask training set generalises
    # better to out-of-distribution modalities; MedSAM2's
    # fine-tune target was dominated by clinical CT/MRI).
    state.uct_backend_choice = "sam2"
    # Segmentation scope — current slice only vs entire stack.
    # Current is fast (1 slice × backend time), All loops over
    # every frame in the stack and caches per-slice in
    # `_ctx["per_slice"]`. Default "all" matches the user's
    # stated workflow (segment the whole stack then scroll +
    # correct), but "current" is useful for fast iteration on
    # one slice.
    state.uct_segment_scope = "all"
    state.uct_segment_scope_items = [
        {
            "value": "current",
            "title": "Current slice",
            "icon": "mdi-image-outline",
        },
        {
            "value": "all",
            "title": "All slices",
            "icon": "mdi-image-multiple-outline",
        },
    ]
    state.uct_backend_items = [
        {"title": "Auto (MedSAM2 preferred)", "value": "auto"},
        {"title": "MedSAM2 (medical fine-tune)", "value": "medsam2"},
        {"title": "SAM2 (vanilla Meta)", "value": "sam2"},
        {"title": "Stub (Otsu + morphology)", "value": "stub"},
    ]
    # CLAHE pre-processing toggle. Adaptive histogram
    # equalisation tames the very low local contrast typical of
    # µCT scans, which makes SAM2's mask grid more discriminative
    # and dramatically reduces the "0 masks" failure mode on
    # full-frame views. Off by default — flicker the toggle if
    # the everything-mode results look thin.
    state.uct_clahe = False
    # Optional crop applied to the slice before segmentation.
    # Two VRangeSlider v-models in [lo, hi] pixel coords; the
    # _on_uct_crop_change watcher re-crops slice_disp on every
    # release and clears stale proposals. Defaults to [0, 0]
    # until a stack loads — load_stack sets the full extent.
    state.uct_crop_x_range = [0, 0]
    state.uct_crop_y_range = [0, 0]
    state.uct_crop_max_x = 0
    state.uct_crop_max_y = 0
    # V1 Phase B — 3-step segment-µCT dialog.
    # VStepper v-model is a STRING (Vuetify's VStepperItem `value`
    # attrs are strings). Steps:
    #   "1" = Upload (file picker + drag-drop landing zone)
    #   "2" = Segment (image + tools + label assignment)
    #   "3" = Reconstruct 3D (single-slice or marching-cubes)
    # `do_recon_next/back` flip between "2" and "3";
    # `do_load_uct_stack` auto-advances from "1" → "2" on a
    # successful load.
    state.uct_step = "1"
    # Flips True when `do_finalize_segmentation` runs at least
    # once for the current segmentation session. Gates Step-2's
    # Next button so the user can't skip past the refine →
    # generate-epi → save pipeline by accident. Reset on dialog
    # open + on any further label edit (so the user has to
    # re-finalize after fixing things).
    state.uct_step2_finalized = False
    # Sweep step size for "All slices" segmentation: 1 = every
    # slice, 5 = every 5th, etc. Default 1 = the legacy behaviour.
    state.uct_segment_step = 1
    state.uct_recon_mode = "multi"
    state.uct_recon_thickness_mm = 5.0
    state.uct_recon_slice_start = 0
    state.uct_recon_slice_end = 0
    state.uct_recon_voxel_z_mm = 0.0
    state.uct_recon_single_slice_idx = 0
    # Gaussian-smooth the volume before marching cubes —
    # σ = 1 voxel by default. Off renders the raw blocky
    # MC surface (true to labels, ugly stair-steps between
    # ZOH-filled slices); on smooths it for FEM-quality
    # surfaces.
    state.uct_recon_smooth = True
    state.uct_recon_smooth_sigma = 1.0
    # M30 — decoupled physical-mm smoothing sigmas.
    # `smooth_sigma_xy_mm` controls in-plane smoothing
    # (sub-voxel default, keeps fascicle detail intact).
    # `smooth_sigma_z_mm` controls cross-slice smoothing
    # — its default 0.3 mm covers a few Z slabs in a typical
    # single-slice extrude (50 mm prism / 500 slabs = 0.1 mm
    # pitch), erasing the periodic step-shaped ripples the
    # user reported on the lateral fascicle surfaces. Either
    # to 0 disables that axis; legacy `smooth_sigma` is used
    # when both are 0.
    state.uct_recon_smooth_sigma_xy_mm = 0.005
    state.uct_recon_smooth_sigma_z_mm = 0.3
    # Mesh refinement: runs the same Taubin-smooth + pymeshfix
    # repair pipeline that the legacy nerve importer applies to
    # raw STL surfaces. ON by default — most users want a clean
    # mesh out of the box; the toggle lets power users compare
    # raw MC output to the refined version. `decim_k` is the
    # decimation target in thousands of triangles (matches the
    # nerve-import knob naming).
    state.uct_recon_refine = True
    # Isotropic remesh (pyacvd) — re-tiles the surface with a
    # near-uniform edge length so small fascicles get few
    # vertices and the epi shell gets many, both at the same
    # local resolution. OFF by default; the raw MC + Taubin
    # + CVT pipeline is usually enough.
    state.uct_recon_remesh = False
    state.uct_recon_edge_len_um = 50.0
    # M37 — optimesh CVT default flipped to OFF. When M31
    # closed the epi caps the mesh became a genus-0 closed
    # bag, so the `_maybe_optimesh` genus-guard (which used
    # to fire by accident on the previous open-tube epi
    # because the Euler formula gave garbage on a non-closed
    # manifold) no longer bails. On µCT epi shells at 500 k
    # tris each of the 20 CVT iterations cost ~30 s, so the
    # refine step silently turned into 10+ minutes per
    # mesh. Default OFF; re-enable from the UI when you
    # want the polish (small fascicles still benefit).
    state.uct_recon_use_optimesh = False
    # Legacy state var — kept so any persisted UI snapshot
    # restores cleanly. Decimation itself is gone from the
    # refine pipeline; this just survives as a no-op slot.
    state.uct_recon_decim_k = 0
    # Per-mesh decimation target (triangles). 0 = no
    # decimation (legacy behaviour); > 0 enables a volume-
    # preserving pyvista.decimate pass on each output mesh
    # (epi + per-fascicle) after the refinement chain. Useful
    # when µCT geometry produces denser surfaces than the
    # downstream PLC / TetGen step can comfortably digest.
    state.uct_recon_decimate_target_tris = 0
    # M27 — surface size-control mode. The legacy decimate-to-
    # target-tris path uniformly caps every surface at the same
    # number, which crushes the epi (488 k → 20 k = 96 % loss
    # and ugly oversized tris) while barely touching the
    # fascicles. Combobox lets the user pick:
    #   "off"          → marching-cubes output passes through
    #   "fraction"     → keep a FRACTION of each surface's tris
    #                    (consistent edge length across surfaces)
    #   "target_tris"  → legacy absolute cap (backwards compat)
    #   "isotropic"    → pyacvd-based isotropic remesh, target
    #                    edge length controls tri size uniformly
    state.uct_recon_size_mode = "off"
    state.uct_recon_decimate_fraction = 0.5
    # M29 — items list for the surface-size-control VSelect.
    # Inlining as a JS array literal string proved brittle
    # (the v_show conditions on the sub-controls didn't fire,
    # so all three knobs showed regardless of mode). Pushing
    # the items as a proper Python list of dicts via state is
    # the pattern every other VSelect in the app uses
    # (uct_backend_items, fiber_branch_items, etc.).
    state.uct_recon_size_mode_items = [
        {
            "title": "Off (marching-cubes raw)",
            "value": "off",
        },
        {
            "title": "Decimate to fraction (per-surface)",
            "value": "fraction",
        },
        {
            "title": "Decimate to target tri count",
            "value": "target_tris",
        },
        {
            "title": "Isotropic remesh (target edge length)",
            "value": "isotropic",
        },
    ]
    # 2D mask cleanup, applied at SEGMENT time (each proposal
    # goes through cleanup_2d_mask before being added to the
    # per-slice cache). Drops small connected components
    # (false-positive speckle blobs the segmenter emitted) and
    # fills small holes inside the foreground (false-negative
    # pixels inside fascicles). Both in pixels at the source
    # image resolution. 0 = disable that direction.
    # Defaults tuned for typical ~1k² µCT slices: 50 px wipes
    # 1-7-px noise blobs but leaves thin fascicle arms intact;
    # 50 px hole-fill closes the few-pixel false-negative
    # speckles that MedSAM2 often leaves inside large
    # fascicles. The user can crank either knob up to ~100 if
    # the imagery is noisier or pull them down to 0 to inspect
    # the raw segmenter output.
    state.uct_recon_clean_min_component_px = 50
    state.uct_recon_clean_min_hole_px = 50
    # 2D morphological closing — seals thin gaps along the
    # foreground boundary (e.g. 1-2 px disconnects from
    # classifier jitter). Radius in pixels; the kernel is a
    # (2r+1)² square so a value of N closes gaps up to 2N
    # pixels wide in every direction. r=2 closes ≤4-px gaps
    # which is a good default for SAM2 output; bump to 3 if
    # the boundaries look jittery.
    state.uct_recon_clean_closing_radius_px = 2
    # 3D volume cleanup AFTER per-slice 2D cleanup. Catches
    # Z-direction speckle / voids 2D can't see (a streak that
    # lives in 3-4 adjacent slices but nowhere else, or a
    # background column punched through the middle of a
    # fascicle). Knobs in VOXELS — typically larger than the
    # 2D equivalents because the same artefact spans several
    # slices.
    # Default 500 vox ≈ a 50-px-radius speckle that survived
    # in ~6 slices; gets dropped. Same magnitude for hole-fill
    # so internal voids the size of a small fascicle are
    # treated as recovered foreground.
    state.uct_recon_clean_3d_min_component_vox = 500
    state.uct_recon_clean_3d_min_hole_vox = 500
    # M24 — Fascicle inward-offset (voxels). Before marching
    # cubes, erode `epi_vol` by this many voxels and intersect
    # `fasc_vol` against the eroded shape so each fascicle's
    # isosurface lands cleanly INSIDE the epi after marching
    # cubes + Gaussian smoothing. Inter-surface diagnostics
    # showed fascicles straddling the epi by sub-µm to ~3 µm —
    # TetGen then refuses to classify them as nested. Default
    # 2 voxels gives Gaussian-smooth-sigma-1 worth of margin.
    # 0 disables (legacy behaviour, straddles allowed).
    state.uct_recon_fasc_inset_vox = 2
    # Annotated-slice list + a one-line coverage summary
    # pushed by `_refresh_recon_coverage` in the action layer
    # whenever the slice range changes. The dialog's Step-2
    # panel renders both directly.
    state.uct_recon_annotated = []
    state.uct_recon_annotated_items = []
    state.uct_recon_coverage_msg = ""
    # Generated-file listing — populated after a successful
    # do_run_reconstruction. Each entry is {name, path} so the
    # Step-2 panel can list filename + relative project path.
    state.uct_recon_files = []
    state.uct_recon_status = ""
    # 3D preview PNG (data URL) — kept for backwards-compat
    # callers but no longer drives the dialog viewport
    # (replaced by pl_uct_recon below).
    state.uct_recon_preview_url = ""
    # ID (timestamp subdir name) of the most-recently-generated
    # bundle. Empty until the user clicks "Generate 3D nerve"
    # for the first time. The "Done → Import wizard" button is
    # disabled until this is non-empty; the finish handler uses
    # it to pre-select the bundle in the import-wizard picker.
    state.uct_last_bundle_id = ""
    # Per-mesh legend items + view-style toggles for the
    # in-dialog plotter. `uct_recon_mesh_items` is rebuilt by
    # `_update_recon_viewport` after every Preview / Generate
    # so the legend chips reflect the current mesh set; each
    # entry is `{name, color, visible}`. The visibility flag
    # is written through to the actor in the watcher below.
    state.uct_recon_mesh_items = []
    state.uct_recon_show_edges = False
    # M13 Phase 1 — default ON. Per-surface mesh quality is the
    # whole point of the reconstruct preview now; surfacing it
    # immediately (rather than gating behind a toggle) makes
    # the diagnostic visible at first glance after a build.
    state.uct_recon_color_by_quality = True
    # Quality-histogram figure (Plotly) — same shape the
    # nerve-import wizard uses. Empty until first build.
    state.uct_recon_quality_hist_figure = {
        "data": [], "layout": {},
    }
    # Solve-drawer summary chips for the ACTIVE config — kept as
    # pre-computed lists so the v-for in the drawer doesn't have
    # to dig into the nested fem_impedance dict via Vue template
    # gymnastics. Rebuilt by `_on_fem_impedance_chips_change`
    # whenever fem_impedance or active_config_id changes.
    state.fem_impedance_chips_contact = []
    state.fem_impedance_chips_pair = []
    state.fem_impedance_chips_meta = ""

    # F3.2c — analysis-chip switcher state. `fem_configs` is the
    # list of configs the user can pick from (populated by the
    # FEM driver as it completes solves); `active_config_id`
    # drives which config's outputs render in the analysis grid
    # + 3D overlays.
    state.fem_configs = []
    state.active_config_id = ""
    # F3.2c reliability — items lists for the three VSelects that
    # show configs (Solve-drawer multi-select, analysis-grid
    # chip, Compare panel multi-select). Pre-computed in Python
    # so VSelect sees a stable array reference rather than an
    # inline `.map()` expression that produces a new array on
    # every render (which made Vuetify drop multi-select state).
    state.solve_config_items = []
    state.fem_config_items = []
    state.compare_config_items = []
    # Unified design+config picker (replaces the separate analysis-
    # grid "Config" chip that used to overlap the viewport
    # toolbar). Items are adaptive: a design with 0 or 1 solved
    # configs contributes one entry showing just the design name;
    # a design with 2+ configs contributes one entry per config
    # ("Cuff 1 · Default", "Cuff 1 · Mono 1mA"). Value is "<eid>"
    # or "<eid>|<cid>"; a watcher splits it and writes
    # `selected_design_id` (+ `active_config_id` when present).
    state.design_config_items = []
    state.design_config_key = ""
    # F3.2e — Compare-view state. Multi-select of config cids
    # to overlay; z-slice index used by the slice-grid figure;
    # two Plotly figures (axis overlay + slice grid) rebuilt by
    # the compare-watcher whenever the selection / z changes.
    state.compare_config_selection = []
    state.compare_slice_z_idx = 20
    state.compare_axis_figure = {"data": [], "layout": {}}
    state.compare_slice_grid_figure = {"data": [], "layout": {}}
    # F3.2 — selectivity inputs + outputs. Driven by the per-cid
    # sweep results loaded in the Compare watcher. The target /
    # off-target branches are user-picked from the dropdown +
    # multi-select; the amplitude knob slides the SI bar chart
    # along the per-amplitude curve (recruitment-mode sweeps
    # only — threshold-mode sweeps drive the ratio table).
    state.selectivity_target_branch = ""        # branch id as string
    state.selectivity_offtarget_branches = []   # list of branch ids
    state.selectivity_branch_items = []         # combobox items
    state.selectivity_amplitude_mA = 1.0
    state.selectivity_bar_figure = {"data": [], "layout": {}}
    state.selectivity_table_html = (
        "<em style='color:#888; font-size:12px;'>"
        "Run a sweep (threshold-mode or recruitment-mode) on "
        "each config in the Sweep tab to populate this table."
        "</em>"
    )
    state.selectivity_status = ""
    # F3.2b — drawer-side helpers for the configs panel.
    # `rename_cfg_cid_active` is the cid currently in inline-edit
    # mode (empty when nobody is being renamed); `rename_cfg_value`
    # holds the in-progress edit string. `new_config_name` holds
    # the text in the "+ Save current as new config" field.
    state.rename_cfg_cid_active = ""
    state.rename_cfg_value = ""
    state.new_config_name = ""
    # F3.2b — sweep-generator dialog state. The random dialog
    # asks for (n_draws, k_cathodes, l_anodes, rest, seed). The
    # manual-pair dialog stages a list of {cathode_idx, anode_idx,
    # name} dicts; the "Generate" button materialises them as
    # configs and clears.
    state.show_sweep_random_dialog = False
    state.sweep_random_n_draws = 10
    state.sweep_random_k_cathodes = 1
    state.sweep_random_l_anodes = 1
    state.sweep_random_rest = "off"  # "off" | "ground"
    state.sweep_random_seed = ""     # blank → unseeded
    state.show_sweep_manual_dialog = False
    state.sweep_manual_pairs = []
    state.sweep_manual_new_cathode = 0
    state.sweep_manual_new_anode = 1
    state.sweep_manual_new_name = ""
    # F3.2b — design sweep wizard state. Generates a batch of
    # cloned designs from the currently-selected parent, varying
    # Z translation and/or rot_z. `axis` is one of:
    #   "z"     → linear sweep over cuff_offset_mm only
    #   "rot_z" → linear sweep over cuff_rot_z_deg only
    #   "grid"  → outer product over Z × rot_z → N_z × N_rot designs
    #   "scar"  → linear sweep over scar_thickness_um; force-
    #             enables use_scar on every clone (F3.2-M3).
    state.show_sweep_designs_dialog = False
    state.sweep_design_axis = "z"
    state.sweep_design_z_start_mm = 0.0
    state.sweep_design_z_end_mm = 20.0
    state.sweep_design_z_steps = 5
    state.sweep_design_rot_start_deg = -90.0
    state.sweep_design_rot_end_deg = 90.0
    state.sweep_design_rot_steps = 5
    # F3.2-M3 — scar thickness sweep params. Defaults span the
    # range that fits inside the default cuff clearance (200 µm):
    # 50 → 180 µm in 5 steps. User can adjust before running.
    state.sweep_design_scar_start_um = 50.0
    state.sweep_design_scar_end_um = 180.0
    state.sweep_design_scar_steps = 5
    state.sweep_design_name_prefix = "Sweep"
    # Per-electrode visibility mirrors (selected electrode).
    # Defaults to True; bound to checkboxes in the drawer. M1
    # added vis_endo / vis_epi / vis_muscle alongside the cuff-
    # part flags so each design owns its tissue visibility.
    state.vis_master = True
    state.vis_endo = True
    state.vis_epi = True
    state.vis_muscle = True
    state.vis_silicone = True
    state.vis_saline = True
    state.vis_contacts = True
    # Bridge state vars for inline list edits — Vue click writes
    # the target eid (or {eid, name}) here, the @state.change
    # watcher applies + clears.
    state.rename_design_request = None
    state.remove_design_request = ""
    state.refit_design_request = ""
    # Confirm-delete dialog: `confirm_delete_eid` holds the target
    # eid (or "" when the dialog is closed). `confirm_delete_name`
    # is the human-readable name displayed in the confirm prompt.
    state.confirm_delete_eid = ""
    state.confirm_delete_name = ""
    state.show_confirm_delete_dialog = False
    # Confirm-delete-mesh dialog: same shape as the
    # confirm-delete-electrode one above but targets a built
    # mesh (nerve.msh + cached PLC / TetGen artefacts under
    # <project>/designs/<eid>/). Per-row delete buttons in the
    # Mesh drawer set these three vars + open the dialog;
    # confirm posts the eid to `remove_mesh_request`, which the
    # `_on_remove_mesh_request` watcher routes to do_delete_mesh.
    state.confirm_delete_mesh_eid = ""
    state.confirm_delete_mesh_name = ""
    state.show_confirm_delete_mesh_dialog = False
    state.remove_mesh_request = ""
    # Confirm-remove-geometry dialog. Boolean toggle, no payload —
    # there's only ever one loaded geometry per project so no eid
    # bookkeeping is needed. Opens via the Remove button in the
    # Import drawer; the dialog's Remove action runs
    # `do_remove_geometry`.
    state.show_confirm_remove_geometry_dialog = False
    # Inline-rename UI state: `rename_eid_active` holds the eid
    # of the row currently in edit mode (or "" if none).
    # `rename_eid_value` is the text being typed.
    state.rename_eid_active = ""
    state.rename_eid_value = ""

    # Mesh params
    for _k, _v in DEFAULT_MESH.items():
        state[_k] = _v
    state.mesh_log = "Mesh not built yet."
    # Nerve surface preprocessing target (k tris). Mirrors
    # nerve_studio.py's `decim_target_k` slider — lower values are
    # simpler / more TetGen-robust, higher values preserve more
    # surface detail.
    state.decim_target_k = 50
    # Conductivity params — load persisted overrides if available
    _sigma_path = GOLGI_OUT / "conductivities.json"
    _sigma_saved: dict = {}
    if _sigma_path.exists():
        try:
            _sigma_saved = json.loads(_sigma_path.read_text(encoding="utf-8"))
        except Exception:
            _sigma_saved = {}
    for _k, _v in DEFAULT_SIGMA.items():
        state[_k] = float(_sigma_saved.get(_k, _v))
    # Transient confirmation message shown in the Conductivities
    # drawer after the user clicks the Update button. Empty
    # string = no message visible.
    state.sigma_update_status = ""
    # Persisted marker: True once the user has clicked the
    # Update button in this project. Drives the "Conductivities
    # configured" stage row in the project-detail Status tab.
    # We use an explicit commit flag instead of comparing values
    # to defaults — defaults are valid σ values and treating
    # them as "not configured" was misleading. Persisted via
    # the manifest's ui_state, so it survives close / reopen.
    state.sigma_committed = False
    # Per-tissue preset selection — initialised by matching the
    # current σ against the preset table. If no preset matches,
    # falls back to "Custom value".
    for _k in DEFAULT_SIGMA.keys():
        state[f"{_k}_preset"] = _sigma_match_label(
            _k, float(state[_k]),
        )
    # Vue items lists. Pushed into state so VSelect can read them
    # reactively without baking the full list into the v_for body.
    for _k, _items in SIGMA_PRESET_ITEMS.items():
        state[f"{_k}_preset_items"] = _items

    # ----- Cole-Cole evaluator -----
    # Shared dialog opened from any tissue row in the
    # Conductivities drawer. Compute σ(f) from a 4-term Cole-Cole
    # fit (matches IT'IS DB layout — leave d4.Δε=0 for an effective
    # 3-term fit), then "Apply" writes the result back to the
    # target σ field.
    state.show_cole_cole_dialog = False
    state.cole_cole_target = ""          # e.g. "sigma_endo"
    state.cole_cole_target_label = ""    # display: e.g. "endoneurium"
    state.cc_preset = "Custom"
    state.cc_preset_items = COLE_COLE_PRESET_ITEMS
    state.cc_n_presets = len(COLE_COLE_PRESETS)
    # Seed from the "Custom" preset (4 dispersions now).
    _cc0 = COLE_COLE_PRESETS["Custom"]
    state.cc_freq_hz = 1000.0
    state.cc_eps_inf = float(_cc0["eps_inf"])
    state.cc_sigma_ionic = float(_cc0["sigma_ionic"])
    for _i, _d in enumerate(_cc0["dispersions"], start=1):
        state[f"cc_d{_i}_de"] = float(_d[0])
        state[f"cc_d{_i}_tau"] = float(_d[1])
        state[f"cc_d{_i}_alpha"] = float(_d[2])
    state.cc_sigma_result = 0.0
    state.cc_sigma_result_str = "—"
    state.cc_plot_figure = {              # σ(f) Plotly figure dict
        "data": [], "layout": {},
    }

    # ----- Electrode designer (ASCENT presets) -----
    state.show_cuff_designer_dialog = False
    state.cuff_preset_name = ""
    state.cuff_preset_items = [
        {"title": _k.replace("_", " "), "value": _k}
        for _k in sorted(_CUFF_PRESETS.keys())
    ]
    state.cuff_preset_code = ""    # "LN" / "MCT" — drives which
                                       # row set the dialog shows
    state.cuff_designer_status = "Pick a preset to load."
    # Whether a designed cuff is currently mounted in the main
    # viewport. Drives the Clear button's enabled state.
    state.has_designer_cuff = False

    # One state var per visible parameter (across ALL presets) —
    # cuff_p_<name>, in DISPLAY units. The unified watcher below
    # listens to all of them at once.
    _CUFF_ALL_VISIBLE_NAMES = []
    for _code, _params in cuff_designer.DESIGNER_VISIBLE_PARAMS.items():
        for _p in _params:
            if _p["name"] not in _CUFF_ALL_VISIBLE_NAMES:
                _CUFF_ALL_VISIBLE_NAMES.append(_p["name"])
    for _n in _CUFF_ALL_VISIBLE_NAMES:
        state[f"cuff_p_{_n}"] = 0.0
    # Fiber params
    state.n_fibers = 100
    state.fiber_max_steps = 10000
    state.fiber_seed_end = "trunk (low z)"
    # Cap-detection / clustering knobs exposed in the Fiber
    # trajectories drawer. Solver script reads these from
    # nerve_paths_seed_config.json. Defaults match the historical
    # hard-coded values in solve_fiber_paths_nerve.py so existing
    # projects reproduce bit-for-bit.
    state.fiber_cluster_eps_mm = 2.0          # DBSCAN xy-radius
    state.fiber_cap_band_pct = 15.0           # z-extreme band (% of nerve length)
    state.fiber_min_rel_size_pct = 20.0       # min cluster size vs largest (%)
    state.fiber_axial_normal_thresh = 0.70    # |n·z| threshold for "cap-like"
    # User-renamed branch labels. Stored as flat per-branch state
    # vars (`fiber_branch_name_0`, `_1`, …) instead of a dict so
    # the Vue templates can bind them directly with `v_model` and
    # never touch object literals — earlier `(dict || {})[key]`
    # template expressions tripped Vue's render-function compiler
    # in this build. Empty string = "use the default Branch N".
    # `_branch_name(b)` resolves the active label for branch `b`.
    for _i in range(MAX_FIBER_BRANCHES):
        state[f"fiber_branch_name_{_i}"] = ""
    state.fiber_log = "No trajectories generated yet."
    state.fiber_status = "Load a nerve first to enable fibers."
    state.has_mesh = False
    state.mesh_stats_html = ""
    state.mesh_quality_hist_figure = {"data": [], "layout": {}}
    # F3.2-M2.1e — per-design Mesh-drawer panels. List of
    # {eid, name, stats_html, hist_fig} dicts, one per design
    # whose mesh has been built. The drawer iterates this with
    # v_for to stack the stats + quality histogram per design.
    state.designs_mesh_panels = []
    state.show_mesh_quality_color = False
    # Wireframe overlay on the built TetGen mesh region actors.
    # Auto-flips to True when the user opens the Mesh drawer so
    # they can immediately see the tet edges; stays where the user
    # left it after that. Wired into the style dict inside
    # `_set_region_groups`.
    state.show_mesh_edges = False
    state.has_fibers = False
    state.fiber_stats_html = ""    # legacy field — no longer rendered
    # Structured branch-summary rows for the table in the Fiber
    # Trajectories drawer. Each row carries the resolved label
    # (default "Branch N" or user-renamed) + per-branch trajectory
    # length stats. Rebuilt after fiber generation, project
    # restore, and any branch-name edit.
    state.fiber_branch_summary = []
    # Inline-rename bridge for the Branch summary table.
    # `branch_rename_active` = branch idx being edited (-1 = no
    # active edit). `branch_rename_value` holds the transient
    # value while the user types; `do_apply_branch_rename`
    # commits it into `fiber_branch_name_{idx}`.
    state.branch_rename_active = -1
    state.branch_rename_value = ""
    state.fiber_failed = False     # surface error log in sidebar
    state.fiber_n_branches = 0     # legend uses to v_show branch rows
    state.vis_fibers = True        # master toggle for ALL fibers
    for _i in range(MAX_FIBER_BRANCHES):
        state[f"vis_fiber_branch_{_i}"] = True
    # Visibility (for rendered mesh)
    for _tag, _spec in DEFAULTS.items():
        state[f"vis_{_tag}"] = bool(_spec["visible"])
    state[f"vis_{TAG_GOLD}"] = True

    # -----------------------------------------------------------------
    # Project lifecycle — startup shows the welcome screen until the
    # user creates or opens a project. Activating a project rebinds
    # GOLGI_OUT to that project's folder; autosaves write project.json
    # + thumbnail.png after each major step + on close.
    # -----------------------------------------------------------------
    state.view_mode = "welcome"          # "welcome" | "workspace"
    state.current_project_name = ""
    state.current_project_dir = ""
    state.current_project_modified = ""
    # `_list_projects` returns an owner-filtered list — at startup
    # nobody is logged in so the list is empty (legacy / orphan
    # projects only appear once a user logs in).
    state.projects_list = _list_projects(owner_user_id=None)
    state.has_active_project = False
    # ---- Auth state mirrors (client-readable; NOT trusted for
    #      gating — `_auth_session` is the source of truth). ----
    state.authenticated = False
    state.current_user_id = 0       # 0 = nobody (avoid None in JSON)
    state.current_user_email = ""
    state.current_user_username = ""
    # Optional profile fields — all start empty until login or
    # the user fills them in via Profile / register dialog.
    state.current_user_first_name = ""
    state.current_user_last_name = ""
    state.current_user_country = ""
    state.current_user_institution = ""
    state.current_user_position = ""
    state.current_user_avatar = ""  # base64 data URI; "" = no avatar
    # Country dropdown options for the registration + profile
    # dialogs. Bundled list — see module-level COUNTRY_NAMES.
    state.country_options = list(COUNTRY_NAMES)
    # Academic position options — same role as country_options.
    state.position_options = list(POSITION_OPTIONS)
    state.session_locked_by = ""    # email of session owner if not us
    state.show_auth_dialog = False
    state.auth_mode = "login"       # "login" | "register"
    # Login form — `auth_login_id` accepts EITHER email or username
    # in one field. Kept separate from the register-only `auth_email`
    # so the two modes don't clobber each other.
    state.auth_login_id = ""
    state.auth_email = ""
    state.auth_username = ""
    # Optional profile fields collected during registration. All
    # default to empty + are all optional — the user can fill
    # them in later via the Profile dialog.
    state.auth_first_name = ""
    state.auth_last_name = ""
    state.auth_country = ""
    state.auth_institution = ""
    state.auth_position = ""
    state.auth_password = ""
    state.auth_password_confirm = ""
    state.auth_image_data_uri = ""  # base64 data URI (built server-side)
    state.auth_image_file = None    # VFileInput v_model target
    state.auth_error = ""
    state.auth_busy = False
    # Profile dialog (open via the navbar avatar dropdown).
    state.show_profile_dialog = False
    state.profile_email = ""
    state.profile_username = ""
    state.profile_first_name = ""
    state.profile_last_name = ""
    state.profile_country = ""
    state.profile_institution = ""
    state.profile_position = ""
    state.profile_image_data_uri = ""
    state.profile_image_file = None  # VFileInput v_model target
    state.profile_remove_image = False
    state.profile_error = ""
    state.profile_status = ""
    state.profile_busy = False
    # Navbar avatar dropdown (Vuetify VMenu binding).
    state.show_user_menu = False
    # Navbar "Simulate" umbrella dropdown (Vuetify VMenu binding).
    state.show_sim_menu = False
    # Navbar "File" umbrella dropdown (Vuetify VMenu binding).
    # Contains Import / Save / Close project sub-items.
    state.show_file_menu = False
    state.show_new_project_dialog = False
    state.new_project_name = ""
    state.new_project_error = ""
    state.show_close_dialog = False
    # Confirmation dialog raised when the user clicks Sign out
    # from the avatar menu. Doubles as a heads-up that signing
    # out also closes the open project (since the workspace tears
    # down + state resets on logout).
    state.show_logout_dialog = False
    state.busy_open = False              # "opening project…" spinner flag
    # Bridge for the details-lightbox "Open" button → async handler.
    # Vue click can only emit assignments to state vars cleanly; an
    # @state.change watcher below converts a non-empty value into
    # the actual `await do_open_project(...)` call.
    state.open_project_request = ""
    # Details lightbox — opened when the user clicks a project
    # tile on the welcome screen. Mirrors the SPARC-style "open
    # before launch" panel: thumbnail + metadata + description,
    # with Open + Delete buttons.
    state.show_detail_dialog = False
    state.detail_project = None          # full dict from projects_list
    # Owner + last-modifier briefs resolved when the detail
    # dialog opens. Each is {id, username, email,
    # avatar_data_uri} or {} when unknown / legacy. Surfaced
    # next to the metadata grid so the user can see WHO
    # created / last touched a project.
    state.detail_project_owner = {}
    state.detail_project_modifier = {}
    # Detail-dialog tabbed layout:
    #   "overview"  — existing metadata + share picker
    #   "status"    — 8-row stage-completion table
    #   "activity"  — paginated audit-log scroller
    # All three are populated lazily by `_refresh_detail_brief`
    # when the dialog opens or the target project changes.
    state.detail_tab = "overview"
    state.detail_status_rows = []
    state.detail_activity_events = []
    state.detail_activity_expanded = []
    # Full registered-user list for the share VAutocomplete.
    # Each entry: {id, username, email, avatar_data_uri}.
    # Populated on login + project-detail open so the user
    # picker always has fresh data without round-tripping per
    # keystroke.
    state.users_list = _list_users()
    # Where the details dialog was opened from — "tile" (welcome-
    # view project tile) or "navbar" (clicking the active project
    # name in the navbar). The navbar entrypoint hides the
    # Open + Delete buttons (you already have it open; you can't
    # delete the open project from inside its own workspace).
    state.detail_dialog_source = ""
    # Delete-confirmation sub-dialog (chained off the details
    # lightbox). delete_project_dir is the absolute path of the
    # project the user just asked to remove.
    state.show_delete_dialog = False
    state.delete_project_dir = ""
    state.delete_project_name = ""
    state.delete_error = ""
    # Inline rename — pencil icon in the details lightbox header
    # toggles edit mode; the input is bound to edit_name_value.
    state.edit_name_mode = False
    state.edit_name_value = ""
    # Per-project labels — managed from the detail-dialog UI.
    # `add_label_mode` toggles between the "+ add label" pill and
    # the inline input. `remove_label_request` is the click-bridge
    # state var; a watcher dispatches the actual removal.
    state.add_label_mode = False
    state.add_label_value = ""
    state.remove_label_request = ""

    # Cancel-busy lightbox: when the user clicks the Cancel button
    # inside any busy spinner (mesh / FEM / fibers / project open),
    # a confirmation sub-dialog asks if they really mean it.
    state.show_cancel_dialog = False
    # When True (set by individual handlers for in-process,
    # restartable work — e.g. SAM2 video propagation), the
    # Cancel button on the busy lightbox skips the confirm
    # sub-dialog and fires `do_confirm_cancel` directly. Needed
    # because the confirm sub-dialog can render under the
    # currently-open modal (e.g. segment-µCT dialog) and become
    # un-clickable, AND the operation has no subprocess to kill
    # so the user clicking through a confirm step adds zero
    # safety. Handlers reset this to False on every exit path.
    state.busy_cancel_no_confirm = False

    # Cancellation registry shared by all subprocess-driven async
    # handlers. `proc` is the currently-running Popen (or None);
    # `requested` is flipped True when the user confirms cancel
    # so the post-subprocess code can short-circuit and skip
    # output loading.
    # W1.8c — Cancellation state lifted into the existing
    # golgi.jobs.cancel.CancelToken class (which migration.md
    # step 4.1 added but step 4.2 never wired into golgi.py).
    # The CancelToken now owns the dict shape + the hard-kill
    # fallback that used to live inline in do_confirm_cancel.
    _cancel = _CancelToken()

    # Which state keys are part of the persisted UI snapshot. Kept
    # explicit so we don't accidentally persist transient flags
    # (busy, view_mode, dialog open/close, etc.).
    _PERSISTED_UI_KEYS: list[str] = [
        "scale_preset", "scale_factor", "selected_file",
        *DEFAULT_CUFF.keys(),
        *DEFAULT_ELECTRODE.keys(),
        *DEFAULT_MESH.keys(),
        *DEFAULT_SIGMA.keys(),
        "sigma_committed",
        "decim_target_k",
        "I_stim_mA", "fem_field", "fem_slice_z_idx",
        "fem_fiber_sel", "fem_sg_window",
        "n_fibers", "fiber_max_steps", "fiber_seed_end",
        # Cap-detection clustering knobs (Fiber trajectories
        # drawer) + user-renamed branch labels (one flat var
        # per branch — see state init for the rationale).
        "fiber_cluster_eps_mm", "fiber_cap_band_pct",
        "fiber_min_rel_size_pct", "fiber_axial_normal_thresh",
        *[
            f"fiber_branch_name_{_i}"
            for _i in range(MAX_FIBER_BRANCHES)
        ],
        "show_quality_color", "show_mesh_quality_color",
        "show_mesh_edges",
        "show_ve_fibers", "show_ve_surface", "show_field_lines",
        "show_saline", "vis_fibers",
        # F3.2-M3 — one-way unlock flag for the muscle bbox
        # preview (True once the user reached Step 4 of the
        # import stepper). Persisted so a re-opened project
        # remembers that the user is past the "first time
        # seeing the muscle" moment.
        "muscle_preview_unlocked",
        # I1 Phase A — persisted impedance toggle. Default on
        # (FEM solves auto-compute Z); user can toggle off in
        # the Conductivities drawer to skip the dual-solves.
        "emit_impedance",
        *[f"vis_{_t}" for _t in DEFAULTS.keys()],
        f"vis_{TAG_GOLD}",
        *[f"vis_fiber_branch_{_i}" for _i in range(MAX_FIBER_BRANCHES)],
        # Multi-design model (F3.2a — was "multi-electrode" pre-
        # rename) — `designs` is the full list of per-cuff dicts,
        # `selected_design_id` restores the focused one in the
        # drawer, `next_design_seq` keeps the auto-generated IDs
        # unique across sessions.
        "designs", "selected_design_id", "next_design_seq",
        # F3.2b — first-class contact configurations. See state
        # init for the schema. Persisted so project reopen
        # restores both the configs list AND which one was
        # active in the drawer.
        "configs", "selected_config_id", "next_config_seq",
        # R1.1 — Recording montages. The montages themselves
        # live inside each config dict (persisted via "configs"
        # above); only the monotonic seq counter needs its own
        # slot so reopening doesn't reset to "rec_A" and clash.
        "next_montage_seq",
        # F3.2a — Mesh tab multi-select. Which designs to build
        # meshes for when the user clicks Build.
        "mesh_design_selection",
        # F3.2c — Solve tab multi-select + which config is
        # currently "active" in the analysis chip switcher.
        "solve_config_selection", "active_config_id",
        # F3.2e — Compare view persistent state.
        "compare_config_selection", "compare_slice_z_idx",
        # Single-fiber tab — selected trajectories + pulse design.
        # The actual sim results live in a pickle cache
        # (`fiber_sim_results.pkl`) handled separately.
        "fiber_sel_indices", "fiber_sel_tab", "fiber_sel_idx",
        "fiber_backend", "fiber_model", "fiber_diameter_um",
        "fiber_pulse_type", "fiber_onset_ms", "fiber_tstop_ms",
        "fiber_mono_polarity", "fiber_mono_amp_mA",
        "fiber_mono_pw_us",
        "fiber_bi_order", "fiber_bi_charge_balanced",
        "fiber_bi_phase1_amp_mA", "fiber_bi_phase1_pw_us",
        "fiber_bi_gap_us",
        "fiber_bi_phase2_amp_mA", "fiber_bi_phase2_pw_us",
        # Population tab — the per-branch type design. The
        # generated assignments + sim results live in
        # `pop_state.pkl` handled separately.
        "pop_branch_types", "pop_seed",
    ]

    # Factory defaults for every persisted key. Used as the single
    # source of truth when resetting state on project close — saves
    # us from having to remember each individual default in the
    # reset function (the bug the user was hitting: scale_factor
    # and other persisted state vars carried over from the prior
    # project because the reset only handled a handful explicitly).
    _FACTORY_DEFAULTS: dict = {
        "scale_preset": "mm → m (×1e-3)",
        "scale_factor": 1.0e-3,
        "selected_file": "",
        # V1 — µCT-bundle import. `import_source_type` flips
        # the Step-1 wizard between the legacy single-STL flow
        # ("stl") and the multi-surface Golgi bundle flow
        # ("uct_bundle"). The bundle picker lists timestamped
        # directories under <project>/uct/nerve_3d/ that carry
        # the manifest.json{kind: "golgi-uct-nerve"} marker.
        "import_source_type": "stl",
        "uct_bundle_items": [],
        "selected_uct_bundle": "",
        "uct_bundle_n_fasc": 0,
        # M47 — histology bundle source. Parallel to uct_bundle
        # but lists `<project>/histology/nerve_3d/*/`.
        "histo_bundle_items": [],
        "selected_histo_bundle": "",
        "histo_bundle_n_fasc": 0,
        # M47 — picker-visibility booleans, recomputed server-side
        # whenever `import_source_type` changes (see watcher in
        # `golgi/watchers/source_type.py`). Complex JS-expression
        # `v_show` bindings (`"import_source_type !== 'foo' && …"`)
        # are not reactively re-evaluated in this trame-client /
        # Vuetify build; simple boolean state vars are.
        "show_picker_stl": True,
        "show_picker_uct_bundle": False,
        "show_picker_histo_bundle": False,
        # M47 — also a server-computed flag for the "Load nerve"
        # button's disabled state. True iff the user has no
        # source selected for the current source type (e.g.
        # empty selected_histo_bundle when on the histo tile).
        # Recomputed by the same watcher.
        "load_nerve_blocked": True,
        **DEFAULT_CUFF,
        **DEFAULT_ELECTRODE,
        **DEFAULT_MESH,
        **DEFAULT_SIGMA,
        "sigma_committed": False,
        "decim_target_k": 50,
        "I_stim_mA": 1.0,
        "fem_field": "Ve",
        "fem_slice_z_idx": 20,
        "fem_fiber_sel": 0,
        "fem_sg_window": 9,
        "n_fibers": 100,
        "fiber_max_steps": 10000,
        "fiber_seed_end": "trunk (low z)",
        "fiber_cluster_eps_mm": 2.0,
        "fiber_cap_band_pct": 15.0,
        "fiber_min_rel_size_pct": 20.0,
        "fiber_axial_normal_thresh": 0.70,
        **{
            f"fiber_branch_name_{_i}": ""
            for _i in range(MAX_FIBER_BRANCHES)
        },
        "show_quality_color": False,
        "show_mesh_quality_color": False,
        "show_mesh_edges": False,
        "show_ve_fibers": False,
        "show_ve_surface": False,
        "show_field_lines": False,
        "show_saline": True,
        "vis_fibers": True,
        "muscle_preview_unlocked": False,
        "emit_impedance": True,
        # Multi-design model (F3.2a — was multi-electrode pre-rename)
        "designs": [],
        "selected_design_id": "",
        "next_design_seq": 1,
        # F3.2b — first-class contact configurations.
        "configs": [],
        "selected_config_id": "",
        "next_config_seq": 1,
        # R1.1 — Recording montage seq counter.
        "next_montage_seq": 1,
        # R1.4 — cNAP figure defaults. Reset on project close so
        # a fresh project doesn't show a stale trace.
        "fiber_cnap_figure": {"data": [], "layout": {}},
        "fiber_cnap_status": "",
        "pop_cnap_figure": {"data": [], "layout": {}},
        "pop_cnap_status": "",
        "active_montage_single": "",
        "active_montage_pop": "",
        "cnap_decompose_by_type": True,
        # F3.2a — Mesh tab multi-select.
        "mesh_design_selection": [],
        # F3.2c — Solve tab multi-select + active config in
        # analysis chip.
        "solve_config_selection": [],
        "active_config_id": "",
        # Unified design+config picker — see state init for schema.
        "design_config_items": [],
        "design_config_key": "",
        # F3.2e — Compare view selection + slice index.
        "compare_config_selection": [],
        "compare_slice_z_idx": 20,
        # Single-fiber tab defaults.
        "fiber_sel_indices": [],
        "fiber_sel_tab": "0",
        "fiber_sel_idx": 0,
        "fiber_backend": "pyfibers",
        "fiber_model": "MRG_INTERPOLATION",
        "fiber_diameter_um": 5.7,
        "fiber_pulse_type": "monophasic",
        "fiber_onset_ms": 1.0,
        "fiber_tstop_ms": 8.0,
        "fiber_mono_polarity": "cathodic",
        "fiber_mono_amp_mA": 1.0,
        "fiber_mono_pw_us": 1000.0,
        "fiber_bi_order": "cathodic-first",
        "fiber_bi_charge_balanced": False,
        "fiber_bi_phase1_amp_mA": 1.0,
        "fiber_bi_phase1_pw_us": 1000.0,
        "fiber_bi_gap_us": 0.0,
        "fiber_bi_phase2_amp_mA": 1.0,
        "fiber_bi_phase2_pw_us": 1000.0,
        # Population tab defaults.
        "pop_branch_types": {},
        "pop_seed": 42,
    }
    for _tag, _spec in DEFAULTS.items():
        _FACTORY_DEFAULTS[f"vis_{_tag}"] = bool(_spec["visible"])
    _FACTORY_DEFAULTS[f"vis_{TAG_GOLD}"] = True
    for _i in range(MAX_FIBER_BRANCHES):
        _FACTORY_DEFAULTS[f"vis_fiber_branch_{_i}"] = True

    # -----------------------------------------------------------------
    # Reactive computations
    # -----------------------------------------------------------------
    @log_action("load_geometry")
    async def do_load_geometry():
        """Heavy I/O runs in a thread-pool executor so the event
        loop stays responsive (spinner animates, state pushes
        happen). All `state.*` mutations stay on the main loop,
        so busy=True / busy=False both reach the client correctly.

        V1 — branches on `state.import_source_type`:
          * "stl" (legacy): load via `load_nerve_file` from
            `state.selected_file`, single boundary mesh.
          * "uct_bundle": load via `_r3d.load_bundle` from the
            picked `<project>/uct/nerve_3d/<id>/` dir. Builds a
            nerve dict where `pts_raw`/`boundary_raw` come from
            the epi outer hull (so legacy code paths that read
            those keep working) plus a `bundle` sub-dict
            carrying the per-fascicle endoneurium surfaces for
            the multi-region PLC pass + fiber-seed filter.
        """
        _src_type = str(
            getattr(state, "import_source_type", "stl") or "stl",
        )
        if _src_type == "uct_bundle":
            if not getattr(state, "selected_uct_bundle", ""):
                return
        elif _src_type == "histo_bundle":
            if not getattr(state, "selected_histo_bundle", ""):
                return
        else:
            if not state.selected_file:
                return
        try:
            _scale = float(state.scale_factor)
        except (TypeError, ValueError):
            _scale = 1.0e-3

        # When invoked as part of do_open_project the outer
        # handler owns the busy lightbox lifecycle — touching
        # state.busy here would either close the lightbox in our
        # finally block (the bug the user saw: lightbox closing
        # after nerve+electrodes loaded) or overwrite the outer
        # busy_msg. Detect the wrapper via state.busy_open and
        # skip the local busy management entirely.
        _owns_busy = not bool(state.busy_open)
        if _owns_busy:
            state.busy = True
            state.busy_msg = "Loading geometry"
            state.flush()

        loop = asyncio.get_event_loop()
        try:
            # Heavy steps batched in the executor so the spinner
            # can still animate: load file, compute global PCA,
            # compute per-triangle quality, gather topology stats,
            # render the histogram PNG.
            def _heavy_load():
                if _src_type in ("uct_bundle", "histo_bundle"):
                    # ---- bundle path (µCT or histology) ----
                    # Resolve the bundle dir from the picker id
                    # (a timestamp string written as the subdir
                    # name by `do_run_reconstruction` /
                    # `do_run_bundle_import`). The on-disk
                    # layout + manifest format is identical
                    # for both kinds, so the same `load_bundle`
                    # works — only the root sub-directory
                    # differs (`uct/` vs `histology/`).
                    pdir = Path(state.current_project_dir)
                    if _src_type == "uct_bundle":
                        bundle_dir = (
                            pdir / "uct" / "nerve_3d"
                            / str(state.selected_uct_bundle)
                        )
                    else:
                        bundle_dir = (
                            pdir / "histology" / "nerve_3d"
                            / str(state.selected_histo_bundle)
                        )
                    bundle = _r3d.load_bundle(bundle_dir)
                    epi_v_mm = bundle["epi"]["verts"]
                    epi_f = bundle["epi"]["faces"]
                    # Always mm → m for bundles; the bundle
                    # manifest dictates units.
                    epi_v_m = epi_v_mm * 1.0e-3
                    # Hydrate per-fascicle dicts with the m-scaled
                    # verts as a parallel field so downstream PLC
                    # / fiber code reads `verts_m` directly
                    # without having to know the mm/m convention.
                    fascicles_m = []
                    for f in bundle["fascicles"]:
                        fascicles_m.append({
                            "verts_m": f["verts"] * 1.0e-3,
                            "faces": f["faces"],
                            "stl_path": f["stl_path"],
                        })
                    # `pts_raw` / `boundary_raw` define the
                    # nerve's OUTER hull — they feed the cuff-
                    # fit autosize (`refit_design_geometry`),
                    # the global + local PCA frame, the FEM
                    # cross-section centroid, and the multi-
                    # domain PLC's outer wall. They have to be
                    # the EPI surface only. The viewport viz
                    # of the fascicles inside the translucent
                    # epi is now handled by separate scene
                    # catalog entries (`_catalog_fold_regions`
                    # mounts one `region_fascicle_<i>` actor
                    # per fascicle directly from
                    # `bundle["fascicles"]`), so the legacy
                    # combined epi+fascicles buffer is no
                    # longer needed. The earlier combined
                    # buffer also caused TetGen to choke on the
                    # multi-domain PLC, since `pipeline/mesh.py`
                    # added the fascicles a second time via
                    # `inner_surfaces` and the inward-offset
                    # shell sat between two close-packed copies
                    # of every fascicle boundary.
                    nerve = dict(
                        pts_raw=epi_v_m,
                        tets_raw=None,
                        boundary_raw=epi_f,
                        source_file=str(
                            bundle["epi"]["stl_path"].relative_to(
                                pdir,
                            ) if (
                                bundle["epi"]["stl_path"]
                                .is_relative_to(pdir)
                            ) else bundle["epi"]["stl_path"]
                        ),
                        kind="uct_bundle",
                        bundle=dict(
                            epi=dict(
                                verts_m=epi_v_m,
                                faces=epi_f,
                                stl_path=bundle["epi"]["stl_path"],
                            ),
                            fascicles=fascicles_m,
                            voxel_xy_mm=bundle["voxel_xy_mm"],
                            voxel_z_mm=bundle["voxel_z_mm"],
                            manifest=bundle["manifest"],
                            bundle_id=str(
                                state.selected_uct_bundle,
                            ),
                        ),
                    )
                else:
                    _epi_sel = str(
                        getattr(state, "selected_epi_file", "")
                        or "",
                    )
                    if _epi_sel:
                        # ---- explicit epi + endo STL pair ----
                        # The picked source file is the endoneurium;
                        # `selected_epi_file` is the outer epineurium
                        # hull. Assemble the same multi-region
                        # `uct_bundle` nerve the µCT/histology
                        # bundles use (and that fig 8's
                        # new_human_mesh.py builds): pts_raw/
                        # boundary_raw = epi outer hull, fascicles =
                        # [endo]. No inward-offset shell — the epi is
                        # real. Both surfaces share the scale factor.
                        endo_n = load_nerve_file(
                            state.selected_file,
                            units_factor=_scale,
                        )
                        epi_n = load_nerve_file(
                            _epi_sel, units_factor=_scale,
                        )
                        nerve = dict(
                            pts_raw=epi_n["pts_raw"],
                            tets_raw=None,
                            boundary_raw=epi_n["boundary_raw"],
                            source_file=str(_epi_sel),
                            kind="uct_bundle",
                            bundle=dict(
                                epi=dict(
                                    verts_m=epi_n["pts_raw"],
                                    faces=epi_n["boundary_raw"],
                                    stl_path=str(_epi_sel),
                                ),
                                fascicles=[dict(
                                    verts_m=endo_n["pts_raw"],
                                    faces=endo_n["boundary_raw"],
                                    stl_path=str(state.selected_file),
                                )],
                                voxel_xy_mm=0.0,
                                voxel_z_mm=0.0,
                                manifest={"source": "stl_pair"},
                                bundle_id="stl_pair",
                            ),
                        )
                    else:
                        nerve = load_nerve_file(
                            state.selected_file,
                            units_factor=_scale,
                        )
                centroid, R_global = global_pca(nerve["pts_raw"])
                q, _ = _surface_quality(
                    nerve["pts_raw"], nerve["boundary_raw"],
                )
                topo = _topology_stats(
                    nerve["pts_raw"], nerve["boundary_raw"],
                )
                hist_fig = _build_quality_histogram_figure(
                    q,
                    x_label="q_radius_ratio (triangle quality)",
                    y_label="# triangles",
                )
                return nerve, centroid, R_global, q, topo, hist_fig

            (nerve, centroid, R_global,
             nerve_q, topo, hist_fig) = await loop.run_in_executor(
                None, _heavy_load,
            )
            geom.nerve = nerve
            geom.centroid = centroid
            geom.R_global = R_global
            geom.nerve_q = nerve_q
            # Scene state owns the nerve actor now. Build the
            # initial polydata + mount via _request_render — never
            # touches `pl` directly. `geom.nerve_poly` is populated
            # by `_set_nerve_group` on the first fold pass.
            geom.nerve_poly = None
            geom._needs_camera_reset = True
            geom._fit_locked = False
            geom._R_local_cached = None
            geom._R_ci_cached = None
            # Render pass handles the reset_camera + push via the
            # `_needs_camera_reset` flag, after actors are mounted
            # (so the bounding box reflects reality).
            _request_render()
            # Topology + quality summary lines for the Import drawer.
            bbox = topo["bbox_mm"]
            q_min  = float(nerve_q.min())
            q_p10  = float(np.percentile(nerve_q, 10))
            q_med  = float(np.median(nerve_q))
            q_mean = float(nerve_q.mean())
            # Human-readable nerve name for the summary lines.
            # Bundle uses its timestamp id (e.g. 20260530-150000)
            # plus a "(N fascicles)" tag so the user can see at
            # a glance which import produced the loaded geometry.
            if _src_type in ("uct_bundle", "histo_bundle"):
                _n_fasc = len(
                    nerve["bundle"]["fascicles"],
                )
                _kind_label = (
                    "µCT bundle"
                    if _src_type == "uct_bundle"
                    else "histology bundle"
                )
                # Explicitly mention "epi + N fascicles" so the
                # loaded-geometry summary in the Import drawer
                # reflects every surface in the bundle, not
                # just the fascicle count.
                _src_name = (
                    f"{_kind_label} "
                    f"{nerve['bundle']['bundle_id']} "
                    f"(epi + {_n_fasc} fascicle"
                    f"{'s' if _n_fasc != 1 else ''})"
                )
            elif (
                isinstance(nerve.get("bundle"), dict)
                and nerve["bundle"].get("fascicles")
            ):
                # Explicit epi + endo STL pair (stl_pair) — name both
                # surfaces so the summary reflects the multi-region
                # build, not just the endoneurium file.
                _nf = len(nerve["bundle"]["fascicles"])
                _epi_nm = Path(
                    str(getattr(state, "selected_epi_file", "")
                        or "epi"),
                ).name
                _src_name = (
                    f"{Path(state.selected_file).name} + {_epi_nm} "
                    f"(epi + {_nf} endo)"
                )
            else:
                _src_name = Path(state.selected_file).name
            state.geom_summary = (
                f"loaded {_src_name}\n"
                f"  {topo['n_pts']:,} pts | "
                f"{topo['n_tris']:,} tris | "
                f"{topo['n_components']} component"
                f"{'s' if topo['n_components'] != 1 else ''}\n"
                f"  bbox: "
                f"{bbox[0]:.1f} × {bbox[1]:.1f} × {bbox[2]:.1f} mm\n"
                f"  watertight: "
                f"{'yes' if topo['watertight'] else 'no'} | "
                f"open edges: {topo['n_boundary_edges']:,} | "
                f"non-manifold: {topo['n_nonmanifold_edges']:,}\n"
                f"  q_radius_ratio: "
                f"min={q_min:.3f} p10={q_p10:.3f} "
                f"median={q_med:.3f} mean={q_mean:.3f}"
            )
            state.quality_hist_figure = hist_fig
            state.has_geometry = True
            state.fiber_status = (
                f"Nerve ready: {_src_name} — "
                f"generate trajectories"
            )
            # V1 — push fascicle count to Step-2's read-only
            # summary card.
            if _src_type in ("uct_bundle", "histo_bundle"):
                state.uct_bundle_n_fasc = len(
                    nerve["bundle"]["fascicles"],
                )
            else:
                state.uct_bundle_n_fasc = 0
            # Auto-select the fiber-generation method that fits
            # the geometry. µCT bundles + histology bundles are
            # both extruded prisms, so the in-process axial
            # straight-line method is both faster (no Laplace +
            # RK4) and physiologically correct (fibers run
            # parallel to the fascicle's principal axis).
            # Legacy STL imports stay on the streamlines path.
            # The user can still override either choice via the
            # combobox in Step 3.
            if _src_type in ("uct_bundle", "histo_bundle"):
                state.fiber_method = "axial"
                # M48 — the bundle's epi actor renders only when
                # `vis_epi_preview` is True (see catalog fold —
                # bundles route the epi shell visibility through
                # this flag rather than `vis_nerve_raw`). Force
                # it on at load time so the user sees the
                # translucent shell + fascicles immediately
                # (otherwise the default — which is whatever the
                # previous user-session left it at — can hide
                # the epi until they manually toggle it from the
                # legend).
                state.vis_epi_preview = True
            else:
                state.fiber_method = "streamlines"
            # Bundle the source into the active project so the
            # project stays self-contained — on reopen the manifest
            # points at this file. Skips the copy when the source
            # already lives inside the project folder.
            if state.has_active_project:
                try:
                    pdir = Path(state.current_project_dir)
                    if _src_type in (
                        "uct_bundle", "histo_bundle",
                    ):
                        # The bundle dir already lives under
                        # <pdir>/<kind>/nerve_3d/<id>/ — no
                        # copy. Persist a manifest pointer so
                        # re-open picks the same bundle. We
                        # record both the bundle id and the
                        # relative path to epi.stl as
                        # source_file (the legacy key the rest
                        # of the project uses); the id is the
                        # durable identifier the wizard
                        # restores from on next open.
                        if _src_type == "uct_bundle":
                            _bid = str(
                                getattr(
                                    state,
                                    "selected_uct_bundle", "",
                                ),
                            )
                            _root = "uct"
                        else:
                            _bid = str(
                                getattr(
                                    state,
                                    "selected_histo_bundle", "",
                                ),
                            )
                            _root = "histology"
                        source_file_rel = (
                            f"{_root}/nerve_3d/{_bid}/epi.stl"
                        )
                        state.selected_file = str(
                            pdir / source_file_rel,
                        )
                        _write_manifest(
                            pdir,
                            source_file=source_file_rel,
                            import_source_type=_src_type,
                            uct_bundle_id=_bid,
                        )
                    else:
                        src_full = Path(state.selected_file)
                        if not src_full.is_absolute():
                            src_full = HERE / state.selected_file
                        if src_full.is_file():
                            try:
                                rel = src_full.relative_to(pdir)
                                source_file_rel = str(rel)
                            except ValueError:
                                dst = (
                                    pdir / "source"
                                    / src_full.name
                                )
                                dst.parent.mkdir(exist_ok=True)
                                # Skip copy if an identical file
                                # is already there (re-loads
                                # should be a no-op rather than
                                # thrashing the disk).
                                same = (
                                    dst.exists()
                                    and dst.stat().st_size
                                        == src_full.stat().st_size
                                )
                                if not same:
                                    shutil.copy2(src_full, dst)
                                source_file_rel = (
                                    f"source/{src_full.name}"
                                )
                                state.selected_file = str(dst)
                            _write_manifest(
                                pdir,
                                source_file=source_file_rel,
                                import_source_type="stl",
                            )
                except Exception as _bex:
                    print(
                        f"[project] source bundle failed: {_bex}",
                        flush=True,
                    )
            # Autosave after a successful source load. Cheap (no
            # thumbnail at this stage since the camera hasn't been
            # framed yet on first load — the safe_reset_camera()
            # below handles that, then the next autosave will get
            # a properly-framed thumbnail).
            _autosave(stage="source", capture_thumb=False)
        except Exception as ex:
            state.geom_summary = f"⚠ {type(ex).__name__}: {ex}"
            state.quality_hist_figure = {"data": [], "layout": {}}
            state.has_geometry = False
        finally:
            if _owns_busy:
                state.busy = False
            state.flush()
            safe_update()
            # Camera-reset is owned by `_render_scene` — it fires
            # `pl.reset_camera()` + `view_push_camera()` after the
            # nerve actor lands, so the client camera frames the
            # actual scene (not the empty pre-mount bounds).

    def do_refresh_uct_bundles(*_args) -> None:
        """Rescan `<project>/uct/nerve_3d/` and push the bundle
        list into `state.uct_bundle_items` so the import-wizard
        Step-1 picker has up-to-date entries. Bound to the
        project-open lifecycle + the Segment-µCT dialog close
        watcher, so newly-saved bundles appear immediately
        without a page reload."""
        if not state.has_active_project:
            with state:
                state.uct_bundle_items = []
                state.selected_uct_bundle = ""
            return
        try:
            pdir = Path(state.current_project_dir)
        except Exception:                                # noqa: BLE001
            return
        listings = _r3d.list_bundles(pdir / "uct")
        items = [
            {
                "value": L["id"],
                "title": f"{L['id']}  ·  {L['summary']}",
                "summary": L["summary"],
            }
            for L in listings
        ]
        # Preserve current selection if still present; else fall
        # back to the newest bundle (first entry, since
        # list_bundles sorts newest-first).
        cur_sel = str(
            getattr(state, "selected_uct_bundle", "") or "",
        )
        valid = {it["value"] for it in items}
        if cur_sel not in valid:
            cur_sel = items[0]["value"] if items else ""
        with state:
            state.uct_bundle_items = items
            state.selected_uct_bundle = cur_sel

    def do_refresh_histo_bundles(*_args) -> None:
        """M47 — histology-bundle equivalent of
        `do_refresh_uct_bundles`. Scans
        `<project>/histology/nerve_3d/` for the
        `manifest.json{kind: "golgi-uct-nerve"}` bundles written by
        the histology bundle-import action and pushes the listing
        into `state.histo_bundle_items` for the third wizard tile."""
        if not state.has_active_project:
            with state:
                state.histo_bundle_items = []
                state.selected_histo_bundle = ""
            return
        try:
            pdir = Path(state.current_project_dir)
        except Exception:                                # noqa: BLE001
            return
        listings = _r3d.list_bundles(pdir / "histology")
        items = [
            {
                "value": L["id"],
                "title": f"{L['id']}  ·  {L['summary']}",
                "summary": L["summary"],
            }
            for L in listings
        ]
        cur_sel = str(
            getattr(state, "selected_histo_bundle", "") or "",
        )
        valid = {it["value"] for it in items}
        if cur_sel not in valid:
            cur_sel = items[0]["value"] if items else ""
        with state:
            state.histo_bundle_items = items
            state.selected_histo_bundle = cur_sel

    # ---- Remove imported source items (F: per-item delete) ----
    # Trash affordances in the import wizard let the user clean up
    # imported surfaces / bundles that clutter the pickers. Distinct
    # from `do_remove_geometry`, which unloads the *currently-loaded*
    # nerve + its derived artefacts; these delete unloaded source
    # inputs on disk so the pickers stay tidy.
    def _resolve_data_file(rel: str):
        """A data_files entry → existing Path (or None). Entries are
        relative-to-HERE when possible, else absolute (project files
        live outside HERE)."""
        if not rel:
            return None
        p = Path(rel)
        if not p.is_absolute():
            p = HERE / rel
        return p if p.is_file() else None

    def _delete_source_file(rel: str, *, which: str) -> None:
        p = _resolve_data_file(rel)
        if p is None:
            state.upload_info = f"{which}: file not found"
            return
        # Never delete the bundled repo examples under data/ — only
        # files the user uploaded into the project (uploads/, source/).
        try:
            in_examples = p.resolve().is_relative_to(
                DATA_DIR.resolve(),
            )
        except Exception:                                # noqa: BLE001
            in_examples = False
        if in_examples:
            state.upload_info = (
                f"{which}: bundled example — not deletable"
            )
            return
        try:
            p.unlink()
        except OSError as ex:
            state.upload_info = f"{which}: delete failed ({ex})"
            return
        state.data_files = list_data_files()
        if str(getattr(state, "selected_file", "") or "") == rel:
            state.selected_file = (
                state.data_files[0] if state.data_files else None
            )
        if str(getattr(state, "selected_epi_file", "") or "") == rel:
            state.selected_epi_file = ""
        state.upload_info = f"deleted {Path(rel).name}"

    def do_delete_source_file(*_args) -> None:
        _delete_source_file(
            str(getattr(state, "selected_file", "") or ""),
            which="source",
        )

    def do_delete_epi_file(*_args) -> None:
        _delete_source_file(
            str(getattr(state, "selected_epi_file", "") or ""),
            which="epineurium",
        )

    def _delete_bundle(bundle_id: str, sub: str, refresh) -> None:
        if not bundle_id:
            return
        try:
            pdir = Path(state.current_project_dir)
        except Exception:                                # noqa: BLE001
            return
        bdir = pdir / sub / "nerve_3d" / str(bundle_id)
        try:
            if bdir.is_dir():
                shutil.rmtree(bdir)
        except OSError as ex:
            print(f"[delete-bundle] {bdir}: {ex}", flush=True)
            return
        refresh()

    def do_delete_uct_bundle(*_args) -> None:
        _delete_bundle(
            str(getattr(state, "selected_uct_bundle", "") or ""),
            "uct", do_refresh_uct_bundles,
        )

    def do_delete_histo_bundle(*_args) -> None:
        _delete_bundle(
            str(getattr(state, "selected_histo_bundle", "") or ""),
            "histology", do_refresh_histo_bundles,
        )

    def _update_recon_viewport(
        meshes,
        *,
        keep_camera: bool = False,
    ) -> None:
        """Populate the Step-3 PyVista plotter from the latest
        reconstruction meshes, and refresh the legend +
        histogram state vars.

        Owned by app.py rather than the segmentation action
        layer because (a) the plotter (`pl_uct_recon`) is built
        here, and (b) the watchers below rebuild actors from
        the meshes cache — keeping all plotter access in one
        place avoids cross-module imports of pl_uct_recon.

        `keep_camera=True` is used by the toggle watchers so
        flipping "edges" / "quality" doesn't snap the camera
        back to a default view; for fresh meshes from Preview
        / Generate we reset the camera so the user sees the
        new geometry framed.
        """
        nonlocal _uct_recon_meshes_cache
        try:
            import pyvista as _pv
            import numpy as _np
            from golgi.pipeline.mesh_quality import (
                surface_quality as _sq,
            )
        except ImportError:
            return
        _uct_recon_meshes_cache = list(meshes)
        # Camera snapshot — only restored when keep_camera=True
        # and the plotter already had a meaningful view set.
        cam_position = None
        if keep_camera:
            try:
                cam_position = pl_uct_recon.camera_position
            except Exception:                            # noqa: BLE001
                cam_position = None
        pl_uct_recon.clear_actors()
        show_edges = bool(
            getattr(state, "uct_recon_show_edges", False),
        )
        color_by_q = bool(
            getattr(state, "uct_recon_color_by_quality", False),
        )
        # Preserve per-mesh visibility decisions across rebuilds.
        prev_vis: dict[str, bool] = {}
        for it in list(
            getattr(state, "uct_recon_mesh_items", []) or [],
        ):
            try:
                prev_vis[str(it["name"])] = bool(it["visible"])
            except (KeyError, TypeError):
                pass
        items: list[dict] = []
        # M13 Phase 1 follow-up — keep per-mesh quality arrays
        # paired with their mesh names so the histogram panel
        # below can render one subplot per surface instead of
        # one combined histogram. Concatenating across meshes
        # hides which structure has the low-quality tris, which
        # is exactly the diagnostic the user wants here.
        per_mesh_q: list[dict] = []
        for m in meshes:
            n_t = int(m.faces.shape[0])
            if n_t == 0:
                continue
            flat = _np.empty(n_t * 4, dtype=_np.int64)
            flat[0::4] = 3
            flat[1::4] = m.faces[:, 0]
            flat[2::4] = m.faces[:, 1]
            flat[3::4] = m.faces[:, 2]
            pd = _pv.PolyData(
                _np.asarray(m.verts, dtype=_np.float64),
                flat,
            )
            # Heron-radius-ratio triangle quality. Stored as a
            # named scalar field so `scalars="q_tri"` picks it
            # up when colour-by-quality is on.
            try:
                q_arr, _ = _sq(
                    _np.asarray(
                        m.verts, dtype=_np.float64,
                    ),
                    _np.asarray(
                        m.faces, dtype=_np.int64,
                    ),
                )
                pd["q_tri"] = _np.asarray(
                    q_arr, dtype=_np.float32,
                )
                per_mesh_q.append({
                    "name": str(m.name),
                    "q": _np.asarray(q_arr, dtype=_np.float64),
                })
            except Exception:                            # noqa: BLE001
                pass
            is_epi = (m.name == "epi")
            class_color = (
                "#4caf50" if is_epi else "#60a5fa"
            )
            class_opacity = 0.35 if is_epi else 0.9
            visible = prev_vis.get(m.name, True)
            if color_by_q and "q_tri" in pd.array_names:
                actor = pl_uct_recon.add_mesh(
                    pd,
                    name=m.name,
                    scalars="q_tri",
                    cmap="RdYlGn",
                    clim=(0.0, 1.0),
                    opacity=class_opacity,
                    show_edges=show_edges,
                    smooth_shading=True,
                    show_scalar_bar=False,
                )
            else:
                actor = pl_uct_recon.add_mesh(
                    pd,
                    name=m.name,
                    color=class_color,
                    opacity=class_opacity,
                    show_edges=show_edges,
                    smooth_shading=True,
                )
            try:
                actor.SetVisibility(int(bool(visible)))
            except Exception:                            # noqa: BLE001
                pass
            items.append({
                "name": m.name,
                "color": class_color,
                "visible": bool(visible),
                "n_tris": int(n_t),
            })

        # Per-mesh stacked histogram — one subplot per surface
        # so the user can see at a glance which structure has
        # the sliver triangles. Falls back to empty figure when
        # there's no mesh with a quality array.
        if per_mesh_q:
            try:
                hist = _build_combined_quality_histogram_figure(
                    per_mesh_q,
                    x_label=(
                        "q_radius_ratio (triangle quality)"
                    ),
                    y_label="# triangles",
                )
            except Exception:                            # noqa: BLE001
                hist = {"data": [], "layout": {}}
        else:
            hist = {"data": [], "layout": {}}

        # Camera: restore prior view OR reset to fit all actors.
        if cam_position is not None:
            try:
                pl_uct_recon.camera_position = cam_position
            except Exception:                            # noqa: BLE001
                pl_uct_recon.reset_camera()
        else:
            pl_uct_recon.reset_camera()

        with state:
            state.uct_recon_mesh_items = items
            state.uct_recon_quality_hist_figure = hist
        # Force a re-render so the embedded view updates
        # without waiting for a user mouse-move on the canvas.
        try:
            ctrl.view_uct_recon_update()
        except Exception:                                # noqa: BLE001
            pass

    def do_remove_geometry(*_args) -> None:
        """Unload the current geometry and every downstream
        artefact (mesh, FEM, fibers, population) so the project
        is ready for a fresh import. Preserves user parameter
        choices (cuff/electrode/σ/fiber-sim settings) — those
        are project preferences, not derived data.

        Hard-deletes the cached files on disk too (nerve.msh,
        nerve_paths_*, fem_results_*, source/* …) so the next
        time the project is opened nothing auto-restores. The
        confirm dialog in the Import drawer is the only callsite.
        """
        # 0) Close the confirm dialog right away so the user
        # doesn't see it linger while the teardown runs.
        state.show_confirm_remove_geometry_dialog = False
        state.flush()

        # 1) Drop in-memory geometry + derived caches. Mirrors
        # the slot list in `_reset_geom_and_state` but stops
        # short of resetting the persisted UI keys.
        for slot in (
            "nerve", "centroid", "R_global", "cuff_origin_pca",
            "R_local", "pts_cuff", "R_ci", "R_co", "msh_path",
            "mesh_nodes", "mesh_elems", "mesh_tags", "mesh_q",
            "region_surfaces", "region_surfaces_viz",
            "designs_meshes",
            "fem_axis", "fem_slice",
            "fiber_paths_Ve", "fiber_paths_Ez",
            "fiber_paths_for_Ve",
            "nerve_surface_Ve", "nerve_q",
            "fiber_paths_raw", "fiber_branch_idx",
            "fiber_pop_types", "fiber_pop_rows",
            "fiber_pop_diameters_um", "fiber_pop_sim_results",
            "fiber_sim_data", "fiber_sim_results",
            "_cuff_designer_parts", "nerve_poly",
        ):
            try:
                setattr(geom, slot, None)
            except Exception:
                pass
        geom.fiber_n_branches = 0
        geom.fibers_in_cuff_frame = False
        geom.ve_clim_mV = None
        geom.field_lines_poly = None
        geom._fit_locked = False
        geom._R_local_cached = None
        geom._R_ci_cached = None
        geom._needs_camera_reset = True

        # 2) Reset has_* flags + cached display strings so the
        # UI reflects a freshly empty project.
        with state:
            state.has_geometry = False
            state.has_mesh = False
            state.has_fem = False
            state.has_fibers = False
            state.has_designer_cuff = False
            state.geom_summary = "no geometry loaded"
            state.quality_hist_figure = {"data": [], "layout": {}}
            state.mesh_stats_html = ""
            state.mesh_quality_hist_figure = {"data": [], "layout": {}}
            state.designs_mesh_panels = []
            state.fiber_stats_html = ""
            state.fiber_branch_summary = []
            state.branch_rename_active = -1
            state.branch_rename_value = ""
            state.fiber_n_branches = 0
            state.fem_axis_b64 = ""
            state.fem_slice_b64 = ""
            state.fem_af_b64 = ""
            state.fem_ve_cbar_b64 = ""
            state.fem_slice_figure = {"data": [], "layout": {}}
            state.fem_axis_figure = {"data": [], "layout": {}}
            state.fem_af_figure = {"data": [], "layout": {}}
            state.fem_status = "No FEM run yet."
            state.fem_failed = False
            state.fem_log = ""
            state.mesh_log = "Mesh not built yet."
            state.fiber_log = "No trajectories generated yet."
            state.fiber_status = (
                "Load a nerve first to enable fibers."
            )
            state.fiber_failed = False
            # Population sim outputs — design choices in
            # `pop_branch_types` stay, but anything derived
            # from the now-gone fibers is gone too.
            state.pop_branches_meta = []
            state.pop_generated = False
            state.pop_status = "No population generated yet."
            state.pop_kde_figure = {"data": [], "layout": {}}
            state.pop_sim_done = False
            state.pop_sim_results_meta = []
            state.pop_activated_set = []
            state.pop_view_idx = 0
            state.pop_xsec_figure = {"data": [], "layout": {}}
            state.pop_xsec_cuff_figure = (
                {"data": [], "layout": {}}
            )
            state.pop_propagation_figure = (
                {"data": [], "layout": {}}
            )
            state.pop_waterfall_figure = (
                {"data": [], "layout": {}}
            )
            state.pop_row_meta = {}
            state.pop_row_visible = {}
            state.pop_row_colors = {}
            state.pop_type_colors = {}
            # Single-fiber tab — clear stale selections
            # and any cached sim outputs.
            state.fiber_sel_items = []
            state.fiber_sel_indices = []
            state.fiber_sim_results_meta = []
            state.fiber_sim_status = "No simulation run yet."
            state.fiber_sim_summary = ""
            state.fiber_sim_log = ""
            state.fiber_propagation_figure = (
                {"data": [], "layout": {}}
            )
            state.fiber_waterfall_figure = (
                {"data": [], "layout": {}}
            )

        # 3) Drop every actor and reset scene-state caches so
        # the next render pass starts from an empty world.
        try:
            pl.renderer.RemoveAllViewProps()
        except Exception:
            pass
        _rendered_sigs.clear()
        _scene_state["nerve"] = _mkgrp()
        for _tag in _scene_state["regions"]:
            _scene_state["regions"][_tag] = _mkgrp()
        _scene_state["fibers"]["mode"] = "off"
        for _i in _scene_state["fibers"]["branches"]:
            _scene_state["fibers"]["branches"][_i] = _mkgrp()
        _scene_state["fibers"]["ve"] = _mkgrp()
        _scene_state["fibers"]["selected"] = _mkgrp()
        _scene_state["field"]["tubes"] = _mkgrp()
        _scene_state["field"]["arrows"] = _mkgrp()

        # 4) Hard-delete the on-disk caches associated with
        # this geometry so opening the project later won't
        # auto-restore anything. The project manifest itself
        # is left alone (preserves user prefs).
        if state.has_active_project:
            for _name in (
                "nerve.msh",
                "nerve_paths.vtu",
                "nerve_paths_caps.json",
                "nerve_paths_fibers.npz",
                "nerve_paths_field.xdmf",
                "nerve_paths_field.h5",
                "nerve_paths_seed_config.json",
                "nerve_only_surface.npz",
                "current_plc.vtp",
                "current_tetgen.npz",
                "current_tetgen_payload.json",
                "mesh_config.json",
                "electrode_config.json",
                "fiber_sim_results.pkl",
                "pop_state.pkl",
                "fem_results.npz",
                "fem_results.npy",
                "axis_line.npz", "slice_volume.npz",
                "paths_Ve.npz",
                "Ve.xdmf", "Ve.h5",
            ):
                try:
                    (GOLGI_OUT / _name).unlink(missing_ok=True)
                except Exception:
                    pass
            # F3.1: nuke the per-design subdirs too so the next
            # solve starts from a clean slate. Each design's
            # outputs are non-recoverable derivatives of the
            # mesh + electrode config, so there's no data loss.
            for _subname in ("designs", "fem", "sims"):
                _subdir = GOLGI_OUT / _subname
                if _subdir.is_dir():
                    try:
                        shutil.rmtree(_subdir, ignore_errors=True)
                    except Exception:
                        pass
            # Wipe the per-project source/ folder so the
            # imported file is removed from the data file list
            # and the project starts blank.
            try:
                _src_dir = GOLGI_OUT / "source"
                if _src_dir.is_dir():
                    for _f in _src_dir.iterdir():
                        if _f.is_file():
                            try:
                                _f.unlink()
                            except Exception:
                                pass
            except Exception:
                pass

        # 5) Refresh the importer's file list + clear the
        # current selection so the dropdown doesn't dangle on
        # the now-deleted file.
        state.data_files = list_data_files()
        state.selected_file = (
            state.data_files[0]
            if state.data_files else ""
        )

        # 6) Push the cleared scene to the client and
        # persist the wiped state in the manifest.
        try:
            ctrl.view_update()
        except Exception:
            pass
        _request_render()
        _autosave(capture_thumb=True)

    # ----- Multi-electrode selection / sync helpers -----
    # Guard used by selection-load to silence the per-mirror-key
    # watchers while we're seeding the state vars from a freshly-
    # selected electrode's dict. Without it, loading would fire a
    # cascade of re-fit + re-render events for keys that didn't
    # actually change.
    _elec_sync_guard = {"loading": False}

    def _find_design(eid: str) -> dict | None:
        for e in (state.designs or []):
            if e.get("eid") == eid:
                return e
        return None

    def _refit_design_geometry(eid: str) -> bool:
        """Per-design refit. F4.1 Phase B: body lifted into
        `golgi.scene.cuff_fit.refit_design_geometry` so the
        headless `Study.run_mesh()` path can call the same
        logic without going through `build_app()`. This closure
        just delegates with the build_app-local `geom` + `state`
        captured."""
        from golgi.scene.cuff_fit import (
            refit_design_geometry as _refit,
        )
        return _refit(eid, geom=geom, state=state)

    # ----- F3.2b: contact configurations -----
    # A "config" is a named polarity + current-fraction snapshot
    # attached to ONE design. The design dict still carries the
    # CURRENTLY-ACTIVE polarities (so the FEM driver and the
    # cuff-render path don't have to learn about configs); configs
    # are named saved copies the user can load back into the
    # design or sweep over for batch FEM runs.
    def _find_config(cid: str) -> dict | None:
        for c in (state.configs or []):
            if c.get("cid") == cid:
                return c
        return None

    def _configs_for_design(eid: str) -> list[dict]:
        return [
            c for c in (state.configs or [])
            if c.get("design_id") == eid
        ]

    def _create_config(
        design_eid: str,
        name: str,
        polarities: list | None = None,
        fractions: list | None = None,
        i_stim_ma: float | None = None,
    ) -> str:
        """Append a new config bound to `design_eid`. If
        polarities/fractions are None, copies them from the
        design's current dict (i.e., "snapshot current"). Returns
        the new cid."""
        seq = int(state.next_config_seq or 1)
        cid = f"cfg_{seq:02d}"
        d = _find_design(design_eid)
        if polarities is None:
            polarities = list(
                (d or {}).get("contact_polarities") or []
            )
        if fractions is None:
            fractions = list(
                (d or {}).get("contact_current_fractions") or []
            )
        if i_stim_ma is None:
            i_stim_ma = float(state.I_stim_mA or 1.0)
        cfg = {
            "cid": cid,
            "design_id": str(design_eid),
            "name": str(name or cid),
            "contact_polarities": list(polarities),
            "contact_current_fractions": list(fractions),
            "I_stim_mA": float(i_stim_ma),
            # R1.1 — recording montages start empty; the user
            # adds them via the cuff drawer's Recording panel.
            "recording_montages": [],
        }
        state.configs = list(state.configs or []) + [cfg]
        state.next_config_seq = seq + 1
        return cid

    def _ensure_default_config_for_design(eid: str) -> str:
        """Make sure `eid` has at least one config. If a Default
        already exists return its cid; otherwise create one from
        the design's current polarities. Returns the cid of the
        (existing or new) Default config."""
        existing = _configs_for_design(eid)
        if existing:
            return str(existing[0].get("cid", ""))
        return _create_config(eid, "Default")

    def _rebuild_contact_montage_map(
        montages: list,
    ) -> dict:
        """R1.1 — derive {contact_idx: {mid, label, color, pole}}
        from a list of montage dicts. The cuff drawer reads this
        to render the per-contact-row badge ("🟢 Rec A +"). A
        contact in multiple montages keeps the FIRST it appears
        in (the editor's uniqueness rule prevents this in normal
        use, but the renderer stays safe if the rule is bypassed
        via project-file edits)."""
        out: dict[int, dict] = {}
        for m in (montages or []):
            mid = str(m.get("mid", ""))
            label = str(m.get("label", mid))
            color = str(m.get("color") or "#888")
            plus = int(m.get("plus_contact", -1))
            minus = int(m.get("minus_contact", -1))
            if plus >= 0 and plus not in out:
                out[plus] = {
                    "mid": mid, "label": label,
                    "color": color, "pole": "+",
                }
            if minus >= 0 and minus not in out:
                out[minus] = {
                    "mid": mid, "label": label,
                    "color": color, "pole": "-",
                }
        return out

    def _apply_config_to_design(cid: str) -> None:
        """Load a config's polarities + fractions back into its
        design dict + the state.contact_* mirrors, then mark this
        config as the active one. The drawer renders config
        polarities by walking the design dict, so writing here
        cascades through the existing render path."""
        cfg = _find_config(cid)
        if cfg is None:
            return
        eid = str(cfg.get("design_id", ""))
        d = _find_design(eid)
        if d is None:
            return
        pols = list(cfg.get("contact_polarities") or [])
        fracs = list(cfg.get("contact_current_fractions") or [])
        i_ma = float(cfg.get("I_stim_mA", state.I_stim_mA))
        montages = list(cfg.get("recording_montages") or [])
        # Write into the design dict so the FEM driver +
        # renderer see the new polarities.
        designs = list(state.designs or [])
        for i, dd in enumerate(designs):
            if dd.get("eid") == eid:
                dd = dict(dd)
                dd["contact_polarities"] = list(pols)
                dd["contact_current_fractions"] = list(fracs)
                # R1.1 — mirror montages to the design dict so
                # the renderer can iterate designs and draw
                # arcs without reaching back to state.configs.
                dd["recording_montages"] = list(montages)
                designs[i] = dd
                break
        state.designs = designs
        # Mirror to the state vars the drawer binds to.
        _elec_sync_guard["loading"] = True
        try:
            with state:
                state.contact_polarities = list(pols)
                state.contact_current_fractions = list(fracs)
                state.contact_count = len(pols)
                state.I_stim_mA = i_ma
                state.recording_montages = list(montages)
                state.contact_montage_map = (
                    _rebuild_contact_montage_map(montages)
                )
                state.selected_config_id = cid
                if _compute_polarity_sums is not None:
                    state.contact_polarity_sums = (
                        _compute_polarity_sums(pols, fracs)
                    )
        finally:
            _elec_sync_guard["loading"] = False

    def _save_design_polarities_to_config(cid: str) -> None:
        """Snapshot the design's current polarities + fractions
        into the named config. Used by the drawer's 'Save current'
        button to update an existing config in place."""
        cfg = _find_config(cid)
        if cfg is None:
            return
        eid = str(cfg.get("design_id", ""))
        d = _find_design(eid)
        if d is None:
            return
        configs = list(state.configs or [])
        for i, c in enumerate(configs):
            if c.get("cid") == cid:
                c = dict(c)
                c["contact_polarities"] = list(
                    d.get("contact_polarities") or []
                )
                c["contact_current_fractions"] = list(
                    d.get("contact_current_fractions") or []
                )
                c["I_stim_mA"] = float(state.I_stim_mA or 1.0)
                configs[i] = c
                break
        state.configs = configs

    def _save_selected_to_designs() -> None:
        """Copy the legacy state vars (which represent the
        currently-active electrode) into the selected electrode's
        dict in state.designs. Called after every slider edit
        via _on_legacy_electrode_state_change."""
        sel_id = str(state.selected_design_id or "")
        if not sel_id:
            return
        electrodes = list(state.designs or [])
        for idx, e in enumerate(electrodes):
            if e.get("eid") != sel_id:
                continue
            # Build a NEW dict instead of mutating in place so
            # Vue's reactivity sees a reference change for this
            # row. With in-place mutation, the dict object in the
            # client-side `state.designs` array stays the same
            # reference even after the server pushes the update,
            # and v-for + :key="elec.eid" reuses DOM nodes
            # without re-evaluating the `{{ elec.electrode_type }}`
            # binding — that was the "list label doesn't update
            # until reopen" bug.
            new_e = dict(e)
            for k in _ELEC_MIRROR_KEYS:
                try:
                    new_e[k] = state[k]
                except Exception:
                    pass
            # R_ci / R_co are auto-populated from the frame fit
            # ONLY for the frame anchor (electrodes[0]). Every
            # other electrode owns its R_ci_m / R_co_m through
            # explicit per-row Refit — otherwise the cached frame
            # value would silently undo per-row refits on the very
            # next render pass.
            if idx == 0:
                if geom.R_ci is not None:
                    new_e["R_ci_m"] = float(geom.R_ci)
                if geom.R_co is not None:
                    new_e["R_co_m"] = float(geom.R_co)
            else:
                # Seed the first time only (no value yet) so a
                # freshly-added non-anchor electrode still renders
                # with the frame R_ci until the user refits it.
                if (new_e.get("R_ci_m") is None
                        and geom.R_ci is not None):
                    new_e["R_ci_m"] = float(geom.R_ci)
                if (new_e.get("R_co_m") is None
                        and geom.R_co is not None):
                    new_e["R_co_m"] = float(geom.R_co)
            # Persist the polarity table. If the user touched the
            # electrode-type dropdown the contact count may have
            # changed since the polarities were last loaded — fall
            # back to defaults in that case so the list stays the
            # right length. Push the new defaults back to the
            # mirror under the loading guard so the drawer table
            # re-renders without re-triggering this save handler.
            expected_n = _contact_count(new_e)
            mirror_pols = list(state.contact_polarities or [])
            if (len(mirror_pols) == expected_n
                    and all(p in POLARITY_CHOICES
                              for p in mirror_pols)):
                new_e["contact_polarities"] = mirror_pols
            else:
                new_e["contact_polarities"] = _default_polarities(
                    new_e,
                )
                _elec_sync_guard["loading"] = True
                try:
                    state.contact_polarities = list(
                        new_e["contact_polarities"],
                    )
                    state.contact_count = len(
                        new_e["contact_polarities"],
                    )
                finally:
                    _elec_sync_guard["loading"] = False
            # M1: persist contact_current_fractions alongside
            # polarities. Coerce entries to float | None.
            mirror_fracs = list(
                state.contact_current_fractions or [],
            )
            if len(mirror_fracs) == expected_n:
                new_e["contact_current_fractions"] = [
                    (None if v is None or v == ""
                     else float(v))
                    for v in mirror_fracs
                ]
            else:
                new_e["contact_current_fractions"] = (
                    [None] * expected_n
                )
                _elec_sync_guard["loading"] = True
                try:
                    state.contact_current_fractions = list(
                        new_e["contact_current_fractions"],
                    )
                finally:
                    _elec_sync_guard["loading"] = False
            electrodes[idx] = new_e
            break
        # Re-publish so trame pushes the mutated list to the client.
        state.designs = electrodes

    def _load_design_to_selected(eid: str) -> None:
        """Restore the legacy state vars from the dict for `eid`.
        Wrapped in the loading guard so the mirror-key watcher
        doesn't fire a cascade of redundant re-renders.

        F3.2b: also ensures this design has at least one config
        (auto-creates a "Default" the first time a design is
        loaded), and points state.selected_config_id at whichever
        of the design's configs was most recently active — or the
        first one when no prior active is known."""
        e = _find_design(eid)
        if e is None:
            return
        _elec_sync_guard["loading"] = True
        try:
            with state:
                for k in _ELEC_MIRROR_KEYS:
                    if k in e:
                        state[k] = e[k]
                # Re-publish R_ci / R_co onto geom so downstream
                # rendering / fiber-fit logic sees the per-electrode
                # radii.
                if e.get("R_ci_m") is not None:
                    geom.R_ci = float(e["R_ci_m"])
                if e.get("R_co_m") is not None:
                    geom.R_co = float(e["R_co_m"])
                # Per-contact polarities mirror — auto-populates
                # defaults the first time an electrode is touched.
                pols = _ensure_polarities(e)
                fracs = _ensure_current_fractions(e)
                state.contact_polarities = list(pols)
                state.contact_count = len(pols)
                state.contact_current_fractions = list(fracs)
                state.contact_polarity_sums = (
                    _compute_polarity_sums(pols, fracs)
                )
                # R1.1 — recording montages mirror. The design
                # dict carries them (mirrored from the active
                # config); legacy projects without the field
                # default to [].
                _montages = list(
                    e.get("recording_montages") or [],
                )
                state.recording_montages = _montages
                state.contact_montage_map = (
                    _rebuild_contact_montage_map(_montages)
                )
                # Close any stale editor session that targeted a
                # different design's montage.
                state.show_montage_editor = False
                state.editing_montage_id = ""
                state.montage_form_error = ""
        finally:
            _elec_sync_guard["loading"] = False
        # F3.2b: ensure this design has at least one config, and
        # point selected_config_id at it. If a config for this
        # design is already active, keep it; otherwise pick the
        # first config (or create a Default).
        _existing_for_design = _configs_for_design(eid)
        _current_cid = str(state.selected_config_id or "")
        _current_cfg = _find_config(_current_cid)
        if (_current_cfg is None
                or _current_cfg.get("design_id") != eid):
            if _existing_for_design:
                state.selected_config_id = str(
                    _existing_for_design[0].get("cid", ""),
                )
            else:
                state.selected_config_id = (
                    _ensure_default_config_for_design(eid)
                )

    # ---- M1 quick-preset polarity assignment ----
    def _apply_polarity_preset_impl(preset: str) -> None:
        """Write a canonical N-polar assignment into the
        currently-selected electrode's contact_polarities +
        contact_current_fractions. The preset names match the
        items in the drawer's Quick-preset dropdown.

        Layout conventions:
          * Contacts are indexed in increasing-z order for axial
            contacts. For transverse-tripolar (φ-distributed
            ring of 3) the layout treats the FIRST contact as
            the cathode and the next two as anodes regardless
            of φ — selectivity depends on which contact is the
            stimulating one, not on absolute φ.
          * Quadripolar guarded bipole: A-C-C-A along z, with
            cathodes split 50/50 of I_stim.
          * Monopolar: every contact is a cathode; the user is
            expected to mark a remote ground via the saline
            boundary (future F3.1 work — for now this just
            populates polarities and warns via the sum-check
            chip)."""
        n = int(state.contact_count or 0)
        if n <= 0 or not preset:
            return
        pols: list[str]
        fracs: list[float | None]
        if preset == "bipolar":
            pols = ["anode", "cathode"] + ["off"] * max(0, n - 2)
            fracs = [None] * n
        elif preset == "tripolar_long":
            # Anode-Cathode-Anode along z. Cathode in middle.
            if n >= 3:
                pols = (
                    ["anode", "cathode", "anode"]
                    + ["off"] * (n - 3)
                )
            else:
                # Fall back to bipolar for n=2.
                pols = ["anode", "cathode"] + ["off"] * (n - 2)
            fracs = [None] * n
        elif preset == "tripolar_trans":
            # Same as tripolar_long for now — the geometric
            # difference (longitudinal vs transverse guard) is
            # in the cuff layout, not the polarity assignment.
            # Documented as such in the dropdown tooltip.
            if n >= 3:
                pols = (
                    ["cathode", "anode", "anode"]
                    + ["off"] * (n - 3)
                )
            else:
                pols = ["anode", "cathode"] + ["off"] * (n - 2)
            fracs = [None] * n
        elif preset == "quadripolar":
            # A-C-C-A guarded bipole. Two cathodes split 50/50.
            if n >= 4:
                pols = (
                    ["anode", "cathode", "cathode", "anode"]
                    + ["off"] * (n - 4)
                )
                fracs = [None, 0.5, 0.5, None] + [None] * (n - 4)
            else:
                pols = _default_polarities({"contact_count": n})
                if len(pols) != n:
                    pols = (pols + ["off"] * n)[:n]
                fracs = [None] * n
        elif preset == "monopolar":
            pols = ["cathode"] * n
            # Equal split across all cathodes — the FEM driver
            # will fill 1/N at solve time.
            fracs = [None] * n
        else:
            return
        with state:
            state.contact_polarities = list(pols)
            state.contact_current_fractions = list(fracs)
            state.contact_polarity_sums = (
                _compute_polarity_sums(pols, fracs)
            )

    server.trigger("do_apply_polarity_preset")(
        _apply_polarity_preset_impl,
    )

    # ---- F3.2b: contact-config triggers ----
    # All three are wired from the drawer's configs panel via
    # window.trame.trigger("…", …). The handlers keep state.configs
    # + the design dict + the contact_polarities mirrors in sync;
    # the renderer + FEM driver read straight from the design dict
    # so config switching just looks like "the design's polarities
    # changed."

    def _do_config_select(cid: str) -> None:
        if cid:
            _apply_config_to_design(str(cid))

    def _do_config_save_current(cid: str) -> None:
        """Overwrite an existing config with the design's CURRENT
        polarities + fractions + I_stim_mA — i.e. 'save current'."""
        if cid:
            _save_design_polarities_to_config(str(cid))

    def _do_config_save_as_new(name: str) -> None:
        """Create a new config from the currently-selected design's
        current polarities + fractions. `name` is the user-typed
        label; falls back to 'Config N' when blank."""
        eid = str(state.selected_design_id or "")
        if not eid:
            return
        cfg_name = str(name or "").strip() or (
            f"Config {int(state.next_config_seq or 1)}"
        )
        cid = _create_config(eid, cfg_name)
        state.selected_config_id = cid

    def _do_config_delete(cid: str) -> None:
        if not cid:
            return
        configs = [
            c for c in (state.configs or [])
            if c.get("cid") != cid
        ]
        state.configs = configs
        if str(state.selected_config_id) == cid:
            # Fall back to the first remaining config for the
            # currently-selected design, or clear when none left.
            eid = str(state.selected_design_id or "")
            siblings = _configs_for_design(eid)
            state.selected_config_id = (
                str(siblings[0].get("cid", ""))
                if siblings else ""
            )

    def _do_config_rename(payload) -> None:
        """Payload is {cid, name}. Renames the config in place."""
        if not isinstance(payload, dict):
            return
        cid = str(payload.get("cid", ""))
        name = str(payload.get("name", "")).strip()[:48]
        if not cid or not name:
            return
        configs = list(state.configs or [])
        for i, c in enumerate(configs):
            if c.get("cid") == cid:
                c = dict(c)
                c["name"] = name
                configs[i] = c
                break
        state.configs = configs

    server.trigger("do_config_select")(_do_config_select)
    server.trigger("do_config_save_current")(_do_config_save_current)
    server.trigger("do_config_save_as_new")(_do_config_save_as_new)
    server.trigger("do_config_delete")(_do_config_delete)
    server.trigger("do_config_rename")(_do_config_rename)

    # ---- R1.1: recording-montage CRUD triggers ----
    # All edits target the currently-active config
    # (state.selected_config_id). Adding / editing / deleting a
    # montage writes to state.configs[i].recording_montages,
    # re-mirrors to the design dict + state.recording_montages
    # + state.contact_montage_map.

    def _montage_letter(seq: int) -> str:
        """1→A, 2→B, ..., 26→Z, 27→AA, 28→AB, ..."""
        s = ""
        n = max(1, int(seq))
        while n > 0:
            n, r = divmod(n - 1, 26)
            s = chr(ord("A") + r) + s
        return s

    def _active_config_dict() -> tuple[list, int]:
        """Return (configs_list_copy, index_of_active_config) or
        ([], -1) if no active config."""
        configs = list(state.configs or [])
        cid = str(state.selected_config_id or "")
        if not cid:
            return [], -1
        for i, c in enumerate(configs):
            if c.get("cid") == cid:
                return configs, i
        return [], -1

    def _commit_active_montages(montages: list) -> None:
        """Write `montages` into the active config, mirror to
        the selected design dict + state mirrors + map. No-op if
        there's no active config."""
        configs, i = _active_config_dict()
        if i < 0:
            return
        c = dict(configs[i])
        c["recording_montages"] = list(montages)
        configs[i] = c
        state.configs = configs

        eid = str(c.get("design_id", ""))
        designs = list(state.designs or [])
        for j, dd in enumerate(designs):
            if dd.get("eid") == eid:
                dd = dict(dd)
                dd["recording_montages"] = list(montages)
                designs[j] = dd
                break
        state.designs = designs
        state.recording_montages = list(montages)
        state.contact_montage_map = (
            _rebuild_contact_montage_map(montages)
        )
        # Trigger scene rebuild so the viewport arcs update.
        _request_render()

    def _do_montage_open_add(*_args) -> None:
        """Open the editor in 'add new' mode — clear form, leave
        editing_montage_id empty so save creates a new entry."""
        state.editing_montage_id = ""
        state.montage_form_label = ""
        state.montage_form_plus = -1
        state.montage_form_minus = -1
        state.montage_form_error = ""
        state.show_montage_editor = True

    def _do_montage_open_edit(mid: str) -> None:
        """Open the editor in 'edit existing' mode — preload the
        form from the named montage. No-op if mid doesn't exist
        in the active config."""
        mid = str(mid or "")
        if not mid:
            return
        for m in (state.recording_montages or []):
            if str(m.get("mid", "")) == mid:
                state.editing_montage_id = mid
                state.montage_form_label = str(
                    m.get("label", mid),
                )
                state.montage_form_plus = int(
                    m.get("plus_contact", -1),
                )
                state.montage_form_minus = int(
                    m.get("minus_contact", -1),
                )
                state.montage_form_error = ""
                state.show_montage_editor = True
                return

    def _do_montage_cancel_edit(*_args) -> None:
        state.show_montage_editor = False
        state.editing_montage_id = ""
        state.montage_form_error = ""

    def _do_montage_save(payload) -> None:
        """Commit the editor form to the active config. Payload
        is {label, plus_contact, minus_contact}; the mid comes
        from state.editing_montage_id (empty = new). Validates
        plus ≠ minus and both chosen; sets
        state.montage_form_error and leaves the editor open on
        failure."""
        if not isinstance(payload, dict):
            return
        label = str(payload.get("label", "")).strip()[:48]
        try:
            plus = int(payload.get("plus_contact", -1))
            minus = int(payload.get("minus_contact", -1))
        except (TypeError, ValueError):
            state.montage_form_error = (
                "Pick + and − contacts."
            )
            return
        n_contacts = int(state.contact_count or 0)
        if plus < 0 or minus < 0:
            state.montage_form_error = (
                "Both + and − must be chosen."
            )
            return
        if plus == minus:
            state.montage_form_error = (
                "+ and − must be different contacts."
            )
            return
        if plus >= n_contacts or minus >= n_contacts:
            state.montage_form_error = (
                "Contact id out of range for this design."
            )
            return

        configs, i = _active_config_dict()
        if i < 0:
            state.montage_form_error = (
                "No active config — select one first."
            )
            return

        existing = list(
            configs[i].get("recording_montages") or []
        )
        edit_mid = str(state.editing_montage_id or "")

        if edit_mid:
            # Update existing in place.
            updated = []
            for m in existing:
                if str(m.get("mid", "")) == edit_mid:
                    m = dict(m)
                    m["label"] = label or edit_mid
                    m["plus_contact"] = plus
                    m["minus_contact"] = minus
                    # Preserve kind + color as-is.
                updated.append(m)
            _commit_active_montages(updated)
        else:
            # Create new — allocate a fresh mid + palette color.
            seq = int(state.next_montage_seq or 1)
            letter = _montage_letter(seq)
            mid = f"rec_{letter}"
            palette = list(state.montage_palette or []) or [
                "#22c55e",
            ]
            color = palette[(seq - 1) % len(palette)]
            new_m = {
                "mid": mid,
                "label": label or f"Rec {letter}",
                "plus_contact": plus,
                "minus_contact": minus,
                "kind": "bipolar",
                "color": color,
            }
            existing.append(new_m)
            state.next_montage_seq = seq + 1
            _commit_active_montages(existing)

        state.show_montage_editor = False
        state.editing_montage_id = ""
        state.montage_form_error = ""

    def _do_montage_delete(mid: str) -> None:
        mid = str(mid or "")
        if not mid:
            return
        existing = list(state.recording_montages or [])
        kept = [
            m for m in existing
            if str(m.get("mid", "")) != mid
        ]
        if len(kept) == len(existing):
            return
        _commit_active_montages(kept)
        # If the editor was on this one, close it.
        if str(state.editing_montage_id or "") == mid:
            state.show_montage_editor = False
            state.editing_montage_id = ""
            state.montage_form_error = ""

    server.trigger("do_montage_open_add")(_do_montage_open_add)
    server.trigger("do_montage_open_edit")(_do_montage_open_edit)
    server.trigger("do_montage_cancel_edit")(
        _do_montage_cancel_edit,
    )
    server.trigger("do_montage_save")(_do_montage_save)
    server.trigger("do_montage_delete")(_do_montage_delete)

    # ---- F3.2b: contact-config sweep generators ----
    # Each generator returns a list of (name, polarities, fractions)
    # triples that the trigger handler folds into _create_config
    # calls. Generators are pure functions of (n_contacts, ...args);
    # the design-id binding lives in the trigger wrapper.

    def _sweep_bipolar_adjacent(
        n: int,
    ) -> list[tuple]:
        """N-1 configs, each with one adjacent anode/cathode pair
        (cathode = lower index, anode = higher), all others off."""
        out = []
        for i in range(n - 1):
            pols = ["off"] * n
            pols[i] = "cathode"
            pols[i + 1] = "anode"
            out.append((
                f"BiAdj C{i + 1}↓ C{i + 2}↑",
                pols, [None] * n,
            ))
        return out

    def _sweep_bipolar_all_pairs(
        n: int,
    ) -> list[tuple]:
        """Every unordered pair (i < j) → cathode at i, anode at
        j, others off. N*(N-1)/2 configs total."""
        out = []
        for i in range(n):
            for j in range(i + 1, n):
                pols = ["off"] * n
                pols[i] = "cathode"
                pols[j] = "anode"
                out.append((
                    f"Pair C{i + 1}↓ C{j + 1}↑",
                    pols, [None] * n,
                ))
        return out

    def _sweep_tripolar_axial(
        n: int,
    ) -> list[tuple]:
        """Sliding tripolar along the contact array — centre
        cathode + flanking anodes (0.5/0.5 current). N-2 configs
        (excludes the two end contacts which can't be tripolar)."""
        if n < 3:
            return []
        out = []
        for i in range(1, n - 1):
            pols = ["off"] * n
            pols[i] = "cathode"
            pols[i - 1] = "anode"
            pols[i + 1] = "anode"
            fracs = [None] * n
            fracs[i - 1] = 0.5
            fracs[i + 1] = 0.5
            out.append((
                f"Tri C{i + 1}↓ "
                f"(C{i}/C{i + 2}↑)",
                pols, fracs,
            ))
        return out

    def _sweep_monopolar_each(
        n: int,
    ) -> list[tuple]:
        """N configs — each contact in turn as the sole cathode,
        all others wired to ground (Dirichlet 0)."""
        out = []
        for i in range(n):
            pols = ["ground"] * n
            pols[i] = "cathode"
            out.append((
                f"Mono C{i + 1}↓",
                pols, [None] * n,
            ))
        return out

    def _sweep_random(
        n: int, n_draws: int,
        k_cathodes: int, l_anodes: int,
        rest_ground: bool,
        seed: int | None = None,
    ) -> list[tuple]:
        """N_draws configs of random polarity assignments with
        exactly k cathodes + l anodes + (n-k-l) inactive contacts
        (ground when rest_ground=True, off otherwise). Uses a
        seeded RNG so the same params reproduce."""
        import random as _random_mod
        if k_cathodes + l_anodes > n:
            return []
        if n_draws <= 0:
            return []
        rng = _random_mod.Random(seed)
        rest = "ground" if rest_ground else "off"
        out = []
        for d in range(n_draws):
            idxs = list(range(n))
            rng.shuffle(idxs)
            cathodes = set(idxs[:k_cathodes])
            anodes = set(idxs[
                k_cathodes:k_cathodes + l_anodes
            ])
            pols = [rest] * n
            for c in cathodes:
                pols[c] = "cathode"
            for a in anodes:
                pols[a] = "anode"
            out.append((
                f"Rand #{d + 1:02d}",
                pols, [None] * n,
            ))
        return out

    def _sweep_manual_pairs(
        n: int, pairs: list,
    ) -> list[tuple]:
        """Manual list of {cathode_idx, anode_idx, name} → one
        bipolar config per row. Skips rows whose indices are
        out of range or where cathode == anode."""
        out = []
        for p in (pairs or []):
            if not isinstance(p, dict):
                continue
            try:
                cidx = int(p.get("cathode_idx", -1))
                aidx = int(p.get("anode_idx", -1))
            except (TypeError, ValueError):
                continue
            if not (0 <= cidx < n
                    and 0 <= aidx < n
                    and cidx != aidx):
                continue
            pols = ["off"] * n
            pols[cidx] = "cathode"
            pols[aidx] = "anode"
            name = str(
                p.get("name") or
                f"Manual C{cidx + 1}↓ C{aidx + 1}↑"
            ).strip()[:48]
            out.append((name, pols, [None] * n))
        return out

    def _run_sweep_and_persist(items: list[tuple]) -> int:
        """Take a list of (name, polarities, fractions) tuples and
        materialise them as configs bound to the currently-selected
        design. Returns the count of configs created."""
        eid = str(state.selected_design_id or "")
        if not eid or not items:
            return 0
        for name, pols, fracs in items:
            _create_config(
                eid, name,
                polarities=pols, fractions=fracs,
            )
        return len(items)

    def _do_sweep_bipolar_adjacent():
        n = int(state.contact_count or 0)
        _run_sweep_and_persist(_sweep_bipolar_adjacent(n))

    def _do_sweep_bipolar_all_pairs():
        n = int(state.contact_count or 0)
        _run_sweep_and_persist(_sweep_bipolar_all_pairs(n))

    def _do_sweep_tripolar_axial():
        n = int(state.contact_count or 0)
        _run_sweep_and_persist(_sweep_tripolar_axial(n))

    def _do_sweep_monopolar_each():
        n = int(state.contact_count or 0)
        _run_sweep_and_persist(_sweep_monopolar_each(n))

    def _do_sweep_random_run():
        """Runs the random-draw generator using state.sweep_random_*
        params (populated by the random-sweep dialog)."""
        n = int(state.contact_count or 0)
        try:
            n_draws = int(state.sweep_random_n_draws or 0)
            k = int(state.sweep_random_k_cathodes or 0)
            l_anodes = int(state.sweep_random_l_anodes or 0)
        except (TypeError, ValueError):
            return
        rest_ground = (
            str(state.sweep_random_rest or "off") == "ground"
        )
        seed_str = str(state.sweep_random_seed or "").strip()
        seed = int(seed_str) if seed_str else None
        items = _sweep_random(
            n, n_draws, k, l_anodes, rest_ground, seed,
        )
        _run_sweep_and_persist(items)
        state.show_sweep_random_dialog = False

    def _do_sweep_manual_run():
        """Materialises the manual-pair sweep using
        state.sweep_manual_pairs (built up by the dialog)."""
        n = int(state.contact_count or 0)
        pairs = list(state.sweep_manual_pairs or [])
        items = _sweep_manual_pairs(n, pairs)
        _run_sweep_and_persist(items)
        # Reset the dialog state on success so reopening starts
        # fresh.
        state.sweep_manual_pairs = []
        state.show_sweep_manual_dialog = False

    def _do_sweep_manual_add_row():
        """Append a pair to the manual-sweep working list. The
        dialog's row-add button writes {cathode_idx, anode_idx,
        name} via inline JS, but for consistency the server-side
        version pulls from the staging vars."""
        try:
            cidx = int(state.sweep_manual_new_cathode)
            aidx = int(state.sweep_manual_new_anode)
        except (TypeError, ValueError):
            return
        n = int(state.contact_count or 0)
        if not (0 <= cidx < n
                and 0 <= aidx < n
                and cidx != aidx):
            return
        name = str(
            state.sweep_manual_new_name or ""
        ).strip()[:48]
        pairs = list(state.sweep_manual_pairs or [])
        pairs.append({
            "cathode_idx": cidx,
            "anode_idx": aidx,
            "name": name,
        })
        state.sweep_manual_pairs = pairs
        # Clear the staging name so the next add is obvious.
        state.sweep_manual_new_name = ""

    def _do_sweep_manual_remove_row(idx):
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return
        pairs = list(state.sweep_manual_pairs or [])
        if 0 <= i < len(pairs):
            del pairs[i]
            state.sweep_manual_pairs = pairs

    server.trigger("do_sweep_bipolar_adjacent")(
        _do_sweep_bipolar_adjacent,
    )
    server.trigger("do_sweep_bipolar_all_pairs")(
        _do_sweep_bipolar_all_pairs,
    )
    server.trigger("do_sweep_tripolar_axial")(
        _do_sweep_tripolar_axial,
    )
    server.trigger("do_sweep_monopolar_each")(
        _do_sweep_monopolar_each,
    )
    server.trigger("do_sweep_random_run")(_do_sweep_random_run)
    server.trigger("do_sweep_manual_run")(_do_sweep_manual_run)
    server.trigger("do_sweep_manual_add_row")(
        _do_sweep_manual_add_row,
    )
    server.trigger("do_sweep_manual_remove_row")(
        _do_sweep_manual_remove_row,
    )

    # ---- F3.2b: design sweep generator ----
    # Generates a batch of NEW designs from the currently-selected
    # one as a template. Each new design clones the parent's
    # geometry + electrode hardware but varies cuff_offset_mm (Z
    # translation) and/or cuff_rot_z_deg (twist around the cuff
    # axis). Grid mode is N_z × N_rot.

    def _linspace(start: float, end: float, n: int) -> list[float]:
        """Pure-Python linspace (avoid importing numpy here; the
        sweep params are small lists)."""
        if n <= 0:
            return []
        if n == 1:
            return [float(start)]
        step = (float(end) - float(start)) / float(n - 1)
        return [float(start) + i * step for i in range(n)]

    def _clone_design(
        parent: dict, new_eid: str, new_name: str,
        overrides: dict | None = None,
    ) -> dict:
        """Deep-ish copy of a design dict for sweep purposes.
        Lists are copied to avoid aliasing the parent's polarity
        vector; the rest are scalar-like so a shallow dict copy
        is enough. `overrides` is applied after the copy so the
        caller can tweak cuff_offset_mm / rot_z / etc."""
        clone = dict(parent or {})
        # Lists & nested dicts that should NOT alias the parent.
        for k in (
            "contact_polarities",
            "contact_current_fractions",
            "R_local_elec",
        ):
            if k in clone and clone[k] is not None:
                clone[k] = list(clone[k])
        if "duke_overrides" in clone:
            clone["duke_overrides"] = dict(
                clone["duke_overrides"] or {},
            )
        clone["eid"] = new_eid
        clone["name"] = new_name
        # Each new design gets its own fitted radii — let the next
        # do_fit_cuff fill these in rather than aliasing the
        # parent's cached values.
        clone["R_ci_m"] = None
        clone["R_co_m"] = None
        if overrides:
            for k, v in overrides.items():
                clone[k] = v
        return clone

    def _next_design_eids(n: int) -> list[tuple[str, int]]:
        """Reserve N new design ids in order. Returns
        [(eid, seq), …]. Caller is responsible for bumping
        state.next_design_seq afterwards."""
        seq = int(state.next_design_seq or 1)
        out = []
        for _ in range(n):
            out.append((f"elec_{seq:02d}", seq))
            seq += 1
        return out

    def _do_sweep_designs_run():
        """Generate a batch of new designs cloned from the
        currently-selected one. Reads its params from
        state.sweep_design_*."""
        parent_eid = str(state.selected_design_id or "")
        parent = _find_design(parent_eid)
        if parent is None:
            return
        axis = str(state.sweep_design_axis or "z")
        prefix = (
            str(state.sweep_design_name_prefix or "").strip()
            or "Sweep"
        )
        # Pull the requested grid.
        try:
            z_n = int(state.sweep_design_z_steps or 1)
            r_n = int(state.sweep_design_rot_steps or 1)
            s_n = int(state.sweep_design_scar_steps or 1)
        except (TypeError, ValueError):
            return
        # F3.2-M3 — scar axis is single-dimension; it doesn't
        # touch z_vals / r_vals so we keep those at the parent's
        # current values and iterate over scar thickness only.
        scar_vals: list[float] = []
        if axis == "z":
            z_vals = _linspace(
                float(state.sweep_design_z_start_mm or 0.0),
                float(state.sweep_design_z_end_mm or 0.0),
                z_n,
            )
            r_vals = [
                float(parent.get("cuff_rot_z_deg", 0.0)),
            ]
        elif axis == "rot_z":
            z_vals = [
                float(parent.get("cuff_offset_mm", 0.0)),
            ]
            r_vals = _linspace(
                float(state.sweep_design_rot_start_deg or 0.0),
                float(state.sweep_design_rot_end_deg or 0.0),
                r_n,
            )
        elif axis == "grid":
            z_vals = _linspace(
                float(state.sweep_design_z_start_mm or 0.0),
                float(state.sweep_design_z_end_mm or 0.0),
                z_n,
            )
            r_vals = _linspace(
                float(state.sweep_design_rot_start_deg or 0.0),
                float(state.sweep_design_rot_end_deg or 0.0),
                r_n,
            )
        elif axis == "scar":
            z_vals = [
                float(parent.get("cuff_offset_mm", 0.0)),
            ]
            r_vals = [
                float(parent.get("cuff_rot_z_deg", 0.0)),
            ]
            scar_vals = _linspace(
                float(
                    state.sweep_design_scar_start_um or 0.0
                ),
                float(
                    state.sweep_design_scar_end_um or 0.0
                ),
                s_n,
            )
        else:
            return
        if axis == "scar":
            total = len(scar_vals)
        else:
            total = len(z_vals) * len(r_vals)
        if total <= 0:
            return
        new_eids = _next_design_eids(total)
        new_designs: list[dict] = []
        if axis == "scar":
            # Single-dimensional iteration over scar thickness.
            # use_scar=True is force-set on every clone so the
            # mesh actually includes the scar shell (otherwise
            # cloning a parent with scar disabled would produce
            # silently scar-less meshes).
            for idx, t_um in enumerate(scar_vals):
                new_eid, seq = new_eids[idx]
                label = (
                    f"{prefix} scar={t_um:.0f} µm"
                )
                new_designs.append(_clone_design(
                    parent,
                    new_eid=new_eid,
                    new_name=label,
                    overrides={
                        "use_scar": True,
                        "scar_thickness_um": float(t_um),
                    },
                ))
        else:
            idx = 0
            for zi, z_mm in enumerate(z_vals):
                for ri, rot_deg in enumerate(r_vals):
                    new_eid, seq = new_eids[idx]
                    # Naming: "Sweep_Z{z}_R{rot}_NNN" — keeps the
                    # parameters discoverable in the design list.
                    if axis == "z":
                        label = (
                            f"{prefix} Z={z_mm:+.2f} mm"
                        )
                    elif axis == "rot_z":
                        label = (
                            f"{prefix} φ={rot_deg:+.0f}°"
                        )
                    else:
                        label = (
                            f"{prefix} "
                            f"Z={z_mm:+.2f} "
                            f"φ={rot_deg:+.0f}°"
                        )
                    new_designs.append(_clone_design(
                        parent,
                        new_eid=new_eid,
                        new_name=label,
                        overrides={
                            "cuff_offset_mm": float(z_mm),
                            "cuff_rot_z_deg": float(rot_deg),
                        },
                    ))
                    idx += 1
        # Commit the new designs + bump the seq counter; auto-
        # create a Default config for each one to mirror
        # do_add_design's lifecycle.
        existing = list(state.designs or [])
        state.designs = existing + new_designs
        state.next_design_seq = (
            int(state.next_design_seq or 1) + total
        )
        for d in new_designs:
            _create_config(d["eid"], "Default")
        # F3.2b: refit every cloned design at its OWN Z position
        # / rotation. Skip for the scar axis — scar thickness
        # doesn't change R_local_elec / R_ci / the cuff pose, so
        # the clones already inherit the parent's correct fit.
        if geom.nerve is not None and axis != "scar":
            for d in new_designs:
                _refit_design_geometry(d["eid"])
            asyncio.create_task(do_fit_cuff(refit=False))
        elif geom.nerve is not None:
            # Scar axis: just trigger a render so the new
            # designs appear immediately in the legend +
            # combobox.
            asyncio.create_task(do_fit_cuff(refit=False))
        state.show_sweep_designs_dialog = False

    server.trigger("do_sweep_designs_run")(_do_sweep_designs_run)

    # ---- Electrode list CRUD ----
    def do_add_design():
        """Create a new electrode at the end of the list, stagger
        it along the nerve so it doesn't stack on top of an
        existing one, and select it for editing."""
        # Commit any pending edits to the currently-selected one
        # before switching focus.
        _save_selected_to_designs()
        seq = int(state.next_design_seq or 1)
        eid = f"elec_{seq:02d}"
        # Stagger Z: pick a position 6 mm further along than the
        # previous electrode (well outside the default 10 mm cuff
        # length so the new one doesn't clash with the previous).
        existing = list(state.designs or [])
        if existing:
            last_z = float(existing[-1].get("cuff_offset_mm", 0.0))
            z_new = last_z + 6.0
        else:
            z_new = float(DEFAULT_CUFF["cuff_offset_mm"])
        new = _new_electrode_default(
            eid, f"Cuff {seq}", z_offset_mm=z_new,
        )
        # F3.2-M1.1: adding a new design does NOT mutate
        # existing designs' visibility. Previous designs stay
        # exactly as the user left them (visible or hidden);
        # the new design defaults to vis_master=True from
        # `_new_electrode_default`. This preserves the
        # "see the nerve while placing a new cuff" workflow —
        # the previously-meshed design's nerve is still on
        # screen so the user has a reference to fit the new
        # cuff against.
        state.designs = existing + [new]
        state.next_design_seq = seq + 1
        state.selected_design_id = eid
        # `_load_design_to_selected` already auto-creates a Default
        # config when the design has none + sets selected_config_id
        # via `_ensure_default_config_for_design`. Calling
        # `_create_config(eid, "Default")` again here would create
        # a SECOND Default per design (so the "Configs to solve"
        # picker would show "Cuff 1 · Default" twice). Trust the
        # ensure-helper.
        _load_design_to_selected(eid)
        # Trigger a fit for the new one so it appears in the view.
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=True))

    def do_remove_design(eid: str):
        """Drop the electrode + tear down its plotter actors. The
        list IS allowed to go empty — clicking the Electrodes tab
        with no entries is a valid state (just no cuffs rendered)."""
        if not eid:
            return
        electrodes = [
            e for e in (state.designs or [])
            if e.get("eid") != eid
        ]
        state.designs = electrodes
        # F3.2b: cascade-delete this design's configs. A config
        # without a design has no meaning and would just clutter
        # the list.
        state.configs = [
            c for c in (state.configs or [])
            if c.get("design_id") != eid
        ]
        if _find_config(str(state.selected_config_id)) is None:
            state.selected_config_id = ""
        # Strip the dead electrode's actors from the plotter.
        # The eid already starts with "elec_" — actor names are
        # `<eid>_silicone`, NOT `elec_<eid>_silicone` (the old
        # double-prefix bug left every removed cuff visible).
        for sub in ("silicone", "saline"):
            pl.remove_actor(
                f"{eid}_{sub}", reset_camera=False,
            )
        for i in range(64):
            pl.remove_actor(
                f"{eid}_contact_{i}", reset_camera=False,
            )
        # DUKE-typed electrodes mount their design parts under
        # `<eid>_designer_<role>_<idx>` — strip any whose name
        # starts with the electrode's designer prefix.
        for _actor_name in list(pl.actors.keys()):
            if _actor_name.startswith(f"{eid}_designer_"):
                pl.remove_actor(
                    _actor_name, reset_camera=False,
                )
        # Pick the next survivor (or "" when the list is empty)
        # for the selected slot, and clear any in-flight rename
        # that was targeting the removed row.
        if str(state.selected_design_id) == eid:
            if electrodes:
                state.selected_design_id = electrodes[0]["eid"]
                _load_design_to_selected(
                    electrodes[0]["eid"],
                )
            else:
                state.selected_design_id = ""
        if str(state.rename_eid_active) == eid:
            state.rename_eid_active = ""
            state.rename_eid_value = ""
        # F3.2-M1: also evict the dead design's mesh from
        # geom.designs_meshes so its region actors get retired
        # by the scene-state pipeline (no per-design entry =>
        # no `region_<eid>_<tag>` groups => `retire_unknown`
        # sweeps the actors on next render).
        if geom.designs_meshes is not None:
            geom.designs_meshes.pop(eid, None)
        # Scene state: the dead eid is no longer in
        # `state.designs` → `_set_electrode_groups` won't
        # produce SceneGroups for it → `_retire_unknown_actors`
        # sweeps any lingering actors with that eid prefix.
        _request_render()
        safe_update()

    def do_delete_mesh(eid: str) -> None:
        """Delete the built mesh + cached PLC / TetGen / FEM
        artefacts for a single design while keeping the design
        itself in `state.designs`. Mirrors the user-facing
        confirm-delete flow on the cuff designs list — the
        design row stays, just the costly mesh outputs go.

        Removes (best-effort, ignore-missing):
          * <project>/designs/<eid>/ — nerve.msh + current_plc.vtp
            + current_tetgen* + nerve_surface_pts.npz
          * <project>/fem/<eid>/ — paths_Ve.npz + axis_line.npz +
            any other FEM caches keyed off this design
          * <project>/sims/<eid>/ — fiber / population sim caches
            that depend on the (now-removed) mesh

        Also strips the design's region actors from the plotter
        and flips its `has_mesh` flag to False so the legend
        rows for tissue tags collapse back to the no-mesh state.
        """
        if not eid:
            return
        if not state.has_active_project:
            return
        try:
            pdir = Path(state.current_project_dir)
        except Exception:                                # noqa: BLE001
            return

        # Remove the on-disk dirs for this design. shutil.rmtree
        # with ignore_errors so a partially-built mesh (missing
        # one of the artefact files) still cleans up.
        for _sub in ("designs", "fem", "sims"):
            _d = pdir / _sub / eid
            if _d.exists():
                try:
                    shutil.rmtree(_d, ignore_errors=True)
                    print(
                        f"[mesh-delete] removed {_d}",
                        flush=True,
                    )
                except Exception as _ex:                 # noqa: BLE001
                    print(
                        f"[mesh-delete] failed to remove "
                        f"{_d}: {_ex}",
                        flush=True,
                    )

        # Strip the per-region actors for this design from the
        # plotter. The scene catalog rebuilds on the next render
        # pass and `retire_unknown_actors` sweeps anything that's
        # no longer claimed.
        for _actor_name in list(pl.actors.keys()):
            if _actor_name.startswith(f"region_{eid}_"):
                try:
                    pl.remove_actor(
                        _actor_name, reset_camera=False,
                    )
                except Exception:                        # noqa: BLE001
                    pass

        # Evict this design's mesh from geom.designs_meshes so
        # downstream consumers (region surfaces, fiber-Ve sampler,
        # FEM viewport) don't try to use stale data.
        if geom.designs_meshes is not None:
            geom.designs_meshes.pop(eid, None)

        # Flip the per-design has_mesh flag (and clear the
        # singletons if this was the active design).
        _new_designs = []
        for _d in (state.designs or []):
            if _d.get("eid") == eid:
                _new_designs.append({**_d, "has_mesh": False})
            else:
                _new_designs.append(_d)
        state.designs = _new_designs

        # If the active design was this one, clear the global
        # has_mesh flag too (it gates the post-build stats panel).
        # Otherwise leave it — another design might still have a
        # mesh.
        _any_left = any(
            bool(_d.get("has_mesh"))
            for _d in (state.designs or [])
        )
        state.has_mesh = bool(_any_left)
        # Per-design stats panels are derived from the loaded
        # meshes; drop the deleted one.
        try:
            state.designs_mesh_panels = [
                _p for _p in (state.designs_mesh_panels or [])
                if _p.get("eid") != eid
            ]
        except Exception:                                # noqa: BLE001
            pass

        _request_render()
        safe_update()

    def do_select_design(eid: str):
        """Commit the current edits then switch focus to `eid`."""
        if not eid:
            return
        if str(state.selected_design_id) == eid:
            return
        _save_selected_to_designs()
        state.selected_design_id = eid
        _load_design_to_selected(eid)

    def do_pick_electrode_at(x_norm: float, y_norm: float) -> None:
        """Double-click in the workspace viewport picks the cuff
        under the cursor and selects its electrode. Coordinates
        arrive in [0, 1] (client-canvas normalised, origin
        bottom-left so the y-flip is already done by the JS event
        handler). We rescale to the server's render-window
        pixel grid so vtkPropPicker can resolve the actor."""
        try:
            xn = float(x_norm)
            yn = float(y_norm)
        except (TypeError, ValueError):
            return
        if geom.nerve is None:
            return
        try:
            rw = pl.render_window
            w, h = rw.GetSize()
        except Exception:
            return
        if w <= 0 or h <= 0:
            return
        px = max(0.0, min(1.0, xn)) * float(w)
        py = max(0.0, min(1.0, yn)) * float(h)
        import vtk as _vtk
        picker = _vtk.vtkPropPicker()
        try:
            picker.Pick(px, py, 0.0, pl.renderer)
        except Exception:
            return
        picked = picker.GetActor()
        if picked is None:
            return
        # Walk pl.actors and match by underlying VTK actor
        # reference (pl.actors may wrap with pv.Actor, which
        # forwards GetMapper but isn't `is` the raw vtkActor).
        target_name = None
        for nm, a in pl.actors.items():
            if a is picked:
                target_name = nm
                break
            underlying = getattr(a, "_actor", None)
            if underlying is picked:
                target_name = nm
                break
            try:
                if a.GetMapper() is picked.GetMapper():
                    target_name = nm
                    break
            except Exception:
                continue
        if not target_name:
            return
        # All cuff actors are namespaced as
        # `elec_<seq>_<sub>` (silicone / saline / contact_N /
        # designer_N / halo). Strip the `_<sub>` tail to get
        # the eid.
        if not target_name.startswith("elec_"):
            return
        parts = target_name.split("_")
        if len(parts) < 3:
            return
        eid = f"{parts[0]}_{parts[1]}"
        if _find_design(eid) is None:
            return
        # Commit any pending edits + switch selection. Also pop
        # the Electrodes drawer open so the user sees the
        # highlighted row immediately.
        _save_selected_to_designs()
        state.selected_design_id = eid
        _load_design_to_selected(eid)
        if not state.show_cuff:
            state.show_cuff = True

    # ---- Inline-rename helpers ----
    def do_cancel_rename_eid():
        state.rename_eid_active = ""
        state.rename_eid_value = ""

    def do_save_rename_eid():
        """Apply the typed value to the in-flight row and clear
        the edit-mode flag. Empty / whitespace-only values just
        cancel without touching the dict."""
        eid = str(state.rename_eid_active or "")
        new_name = str(state.rename_eid_value or "").strip()[:48]
        state.rename_eid_active = ""
        state.rename_eid_value = ""
        if not eid or not new_name:
            return
        electrodes = list(state.designs or [])
        for e in electrodes:
            if e.get("eid") == eid:
                if e.get("name") == new_name:
                    return
                e["name"] = new_name
                break
        state.designs = electrodes

    async def do_fit_cuff(refit: bool = False):
        """`refit=True` → recompute R_local + R_ci (cuff size +
        orientation) from the current cuff position. `refit=False`
        → reuse cached R_local + R_ci so position sliders (offset
        / dx / dy) become pure translation rather than triggering
        auto-resizing. Caller passes refit=True only when the
        anchor, local-PCA radius, L_cuff, clearance, or wall slider
        changes — those are the controls that should affect cuff
        shape/orientation."""
        if geom.nerve is None:
            return
        # No electrodes → nothing to fit. Skip the whole pass
        # rather than transforming the nerve into a cuff frame
        # that has nothing in it. The nerve stays in raw frame
        # until the user adds at least one electrode.
        if not state.designs:
            return
        # First fit after a load → must do a full fit so the cache
        # gets populated. Only the very first fit shows the busy
        # overlay; subsequent fits are incremental and fast enough
        # that flashing the lightbox would just make the viewport
        # stutter on every slider tick.
        is_first_fit = not geom._fit_locked
        if is_first_fit:
            refit = True
        # Same wrapper-detection as do_load_geometry — let the
        # outer project-open keep the lightbox open across the
        # whole restore chain instead of closing it after the
        # first-fit finally block.
        _owns_busy = is_first_fit and not bool(state.busy_open)
        if _owns_busy:
            state.busy = True
            state.busy_msg = "Fitting cuff"
            state.flush()

        loop = asyncio.get_event_loop()
        try:
            # The render frame is PURE PCA, translated so the FIRST
            # electrode's PCA origin sits at (0, 0, 0). No global
            # rotation is applied to the nerve — the local-PCA
            # alignment lives on each electrode dict as
            # `R_local_elec` and is applied to that electrode's
            # CUFF mesh only. This decouples electrodes completely:
            # refitting any one of them rotates ONLY its own cuff.
            anchor_elec = (state.designs[0]
                           if state.designs else {})

            def _heavy():
                pts_pca = ((geom.nerve["pts_raw"] - geom.centroid)
                            @ geom.R_global)
                frame_off = float(anchor_elec.get(
                    "cuff_offset_mm", state.cuff_offset_mm,
                ))
                frame_dx = float(anchor_elec.get(
                    "cuff_dx_mm", state.cuff_dx_mm,
                ))
                frame_dy = float(anchor_elec.get(
                    "cuff_dy_mm", state.cuff_dy_mm,
                ))
                frame_L = float(anchor_elec.get(
                    "L_cuff_mm", state.L_cuff_mm,
                )) * 1e-3
                frame_clearance = float(anchor_elec.get(
                    "cuff_clearance_mm", state.cuff_clearance_mm,
                )) * 1e-3
                frame_wall = float(anchor_elec.get(
                    "cuff_wall_mm", state.cuff_wall_mm,
                )) * 1e-3
                cuff_origin = find_cuff_origin_pca(
                    pts_pca, state.cuff_anchor,
                    frame_off, frame_dx, frame_dy,
                )
                L_cuff = frame_L
                if refit:
                    # Compute the anchor electrode's local nerve
                    # axis here (so we can seed
                    # electrodes[0].R_local_elec on first fit)
                    # and size its R_ci off the local slab.
                    R_local_anchor = local_pca_refine(
                        pts_pca, cuff_origin,
                        float(state.local_pca_radius_mm) * 1e-3,
                    )
                    pts_anchor_local = (
                        (pts_pca - cuff_origin)
                        @ R_local_anchor.T
                    )
                    R_ci = autosize_R_ci(
                        pts_anchor_local, L_cuff, frame_clearance,
                    )
                else:
                    R_local_anchor = geom._R_local_cached
                    R_ci = geom._R_ci_cached
                # Render coords for the NERVE: translate PCA so
                # the anchor's origin is at 0. NO rotation.
                pts_cuff = pts_pca - cuff_origin
                R_co = R_ci + frame_wall
                return (cuff_origin, R_local_anchor, pts_cuff,
                         L_cuff, R_ci, R_co, pts_pca)

            (cuff_origin, R_local_anchor, pts_cuff,
             L_cuff, R_ci, R_co,
             pts_pca) = await loop.run_in_executor(
                None, _heavy,
            )
            geom.cuff_origin_pca = cuff_origin
            # `geom.R_local` is kept on the geom for compatibility
            # with downstream code (fibers, etc.) but is now ALWAYS
            # identity — the nerve render frame is just translated
            # PCA. Per-electrode rotations live on each electrode.
            geom.R_local = np.eye(3, dtype=np.float64)
            geom.pts_cuff = pts_cuff
            geom.R_ci = R_ci
            geom.R_co = R_co
            if refit:
                # Cache the anchor's local-PCA rotation so a
                # follow-up translate-only fit can still
                # re-populate R_local_anchor for the auto-seed
                # below without re-running local_pca_refine.
                geom._R_local_cached = R_local_anchor
                geom._R_ci_cached = R_ci
                geom._fit_locked = True
                # Auto-seed electrodes[0].R_local_elec on the
                # FIRST fit so the anchor cuff aligns to local
                # nerve trajectory out-of-the-box. Subsequent
                # explicit Refits on electrodes[0] overwrite it
                # via the per-row Refit handler.
                _seed_list = list(state.designs or [])
                if (_seed_list
                        and _seed_list[0].get("R_local_elec")
                            is None):
                    _seed_list[0]["R_local_elec"] = [
                        float(R_local_anchor[i, j])
                        for i in range(3) for j in range(3)
                    ]
                    if _seed_list[0].get("R_ci_m") is None:
                        _seed_list[0]["R_ci_m"] = float(R_ci)
                        _seed_list[0]["R_co_m"] = float(R_co)
                    state.designs = _seed_list
            # Scene-state pipeline owns every actor from here on.
            # do_fit_cuff is now just "compute geometry, persist
            # electrode dicts, request a render pass." The nerve /
            # regions / electrodes / fibers / field-lines actors
            # are all (re)materialised by `_set_*_groups` inside
            # `_rebuild_scene_state`, which runs on the main loop
            # via `_request_render()`. The phantom-nerve race is
            # gone because the nerve group is only `present` when
            # `geom.region_surfaces is None` (pre-mesh); post-mesh
            # the renderer retires the nerve actor automatically.
            state.has_designer_cuff = False
            _save_selected_to_designs()
            # F3.2-M2.1b — refresh the pre-mesh previews so they
            # snap into cuff frame alongside the nerve actor.
            # `geom.pts_cuff` just flipped from None (raw frame)
            # to a real array (cuff frame), but neither preview
            # is watching that field directly — without these
            # re-renders the muscle cylinder + epi shell would
            # remain ghosted at the raw-frame position while the
            # nerve jumps to its fit-aligned location.
            if bool(state.vis_muscle_preview):
                _update_muscle_preview()
            if bool(state.vis_epi_preview):
                _update_epi_preview()
            # Electrode actor positioning + per-cuff geometry is
            # built in `_set_electrode_groups`. Request a render
            # pass; the scene-state folder reads geom.* and produces
            # one SceneGroup map per electrode.
            _request_render()
        except Exception:
            pass
        finally:
            if _owns_busy:
                state.busy = False
                state.flush()
            safe_update()
            # Camera-reset is handled by `_render_scene` AFTER the
            # actors are mounted, so the bounding box reflects the
            # actual scene. We do NOT touch `_needs_camera_reset`
            # here.

    def do_refit_cuff():
        """User-triggered full re-fit at the *current* cuff
        position. After translating the cuff with the offset / Δx /
        Δy sliders, the nerve cross-section under the cuff may
        differ from the one R_ci was originally fitted to — so
        the nerve can poke through the cuff wall. Hitting this
        button recomputes R_local + R_ci at the present position
        and updates the cache, snapping the cuff back around the
        nerve at wherever the user has placed it."""
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=True))


    def do_reset_cuff():
        """Restore every cuff + electrode parameter to its default
        and force a fresh full refit. Useful when the user has
        wandered into a broken slider combination and wants a
        clean slate."""
        with state:
            for _k, _v in DEFAULT_CUFF.items():
                state[_k] = _v
            for _k, _v in DEFAULT_ELECTRODE.items():
                state[_k] = _v
        # Clear the rigid-cuff cache so the next fit is a real
        # full refit rather than a translate-only against stale
        # cached values.
        geom._fit_locked = False
        geom._R_local_cached = None
        geom._R_ci_cached = None
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=True))


    # ----------------------------------------------------------------
    # Fiber-tab single-fiber simulation (Phase C / nerve_studio §12)
    # ----------------------------------------------------------------
    def _branch_name(b: int) -> str:
        """User-renamed label for branch `b`, falling back to the
        default `"Branch {b}"` when no rename has been set.

        Reads the flat state var `fiber_branch_name_{b}`. Empty
        string = use default. Flat-var storage (instead of a dict)
        avoids object-literal expressions in Vue templates.
        """
        try:
            key = f"fiber_branch_name_{int(b)}"
            name = str(state[key] or "").strip()
        except Exception:
            name = ""
        return name or f"Branch {int(b)}"

    def _refresh_fiber_sel_items() -> None:
        """Rebuild the `fiber_sel_items` dropdown list from the
        current `geom.fiber_paths_raw` + `fiber_branch_idx`.
        Each entry is `{"title": "Branch X · Fiber N",
                        "value": N}`, sorted by branch then by
        fiber-index-within-branch, so the dropdown reads as
        grouped sections. Empty when no fibers are loaded.

        Called after:
          - `do_generate_fibers` completes successfully,
          - `_restore_fibers_from_disk` repopulates the cache.
        """
        if (geom.fiber_paths_raw is None
                or len(geom.fiber_paths_raw) == 0):
            state.fiber_sel_items = []
            return
        n_fibers = int(len(geom.fiber_paths_raw))
        bidx = geom.fiber_branch_idx
        n_branches = int(geom.fiber_n_branches or 0)
        items: list = []
        # tab10 colour per fiber index — stable assignment so the
        # chip in the combobox and the actor in the 3-D viewport
        # use the SAME colour. Cycles every 10 fibers.
        def _color_for(i: int) -> str:
            return TAB10_PALETTE[int(i) % len(TAB10_PALETTE)]
        if bidx is not None and n_branches > 0:
            bidx_arr = np.asarray(bidx, dtype=np.int32)
            for branch in range(n_branches):
                fib_in_branch = np.where(bidx_arr == branch)[0]
                if fib_in_branch.size == 0:
                    continue
                for i in fib_in_branch:
                    items.append({
                        "title": (
                            f"{_branch_name(branch)} · "
                            f"Fiber {int(i)}"
                        ),
                        "value": int(i),
                        # `branch` is used by the VCombobox
                        # prepend-item tabs to filter the
                        # visible items per branch.
                        "branch": int(branch),
                        # `color` drives the per-chip swatch + the
                        # matching trajectory tint in the 3-D
                        # viewport.
                        "color": _color_for(i),
                    })
            # Catch fibers without a valid branch assignment
            # (shouldn't happen post-classify, but defensive).
            assigned = {item["value"] for item in items}
            for i in range(n_fibers):
                if i not in assigned:
                    items.append({
                        "title": f"Unassigned · Fiber {int(i)}",
                        "value": int(i),
                        "branch": -1,
                        "color": _color_for(i),
                    })
        else:
            # No branch indexing yet — fall back to a flat list.
            for i in range(n_fibers):
                items.append({
                    "title": f"Fiber {int(i)}",
                    "value": int(i),
                    "branch": -1,
                    "color": _color_for(i),
                })
        state.fiber_sel_items = items
        # Snap the active dropdown tab to the FIRST branch that
        # actually has fibers. Without this, the dropdown shows
        # an empty list when a project's branches don't include
        # the previously-active tab (e.g., re-loading a project
        # with different clustering).
        present_branches = sorted({
            it["branch"] for it in items
            if it.get("branch", -1) >= 0
        })
        if present_branches:
            try:
                cur_tab = int(state.fiber_sel_tab)
            except (TypeError, ValueError):
                cur_tab = -1
            if cur_tab not in present_branches:
                state.fiber_sel_tab = str(present_branches[0])
        # If the currently-viewed fiber index is out of range
        # (after a project switch / re-generate), snap to the
        # first available so the plot tiles + 3-D highlight
        # have a valid pick.
        if (int(state.fiber_sel_idx) < 0
                or int(state.fiber_sel_idx) >= n_fibers):
            state.fiber_sel_idx = 0
        # Seed `fiber_sel_indices` with the first fiber on a
        # fresh load — without this, the combobox starts empty
        # and the user can't run a sim until they pick one.
        # Filter the current selection to valid indices in case
        # we re-generated with fewer fibers than before.
        valid_set = set(range(n_fibers))
        current = [int(i) for i in (state.fiber_sel_indices or [])
                   if int(i) in valid_set]
        if not current:
            state.fiber_sel_indices = [int(state.fiber_sel_idx)]
        else:
            state.fiber_sel_indices = current

    # ----------------------------------------------------------------
    # Population tab — per-branch fiber-type mixture
    # ----------------------------------------------------------------
    def _refresh_pop_branches_meta() -> None:
        """Rebuild `pop_branches_meta` from the current geom so
        the Population panel knows which branches exist and how
        many fibers each contains. Called after fiber generate /
        project restore, and whenever the branch clustering
        re-runs."""
        n_branches = int(geom.fiber_n_branches or 0)
        bidx = geom.fiber_branch_idx
        if (bidx is None or n_branches <= 0
                or geom.fiber_paths_raw is None):
            state.pop_branches_meta = []
            return
        bidx_arr = np.asarray(bidx, dtype=np.int32)
        meta = []
        for b in range(n_branches):
            n = int((bidx_arr == b).sum())
            if n == 0:
                continue
            meta.append({
                "idx": int(b),
                "label": _branch_name(b),
                "n_fibers": n,
                "color": BRANCH_PALETTE[b % len(BRANCH_PALETTE)],
            })
        state.pop_branches_meta = meta
        # Re-generation invalidates any prior population
        # assignment AND any sim results (fiber indices may have
        # shifted; row colours / metadata definitely have).
        geom.fiber_pop_types = None
        geom.fiber_pop_rows = None
        geom.fiber_pop_diameters_um = None
        geom.fiber_pop_sim_results = None
        state.pop_generated = False
        state.pop_sim_done = False
        state.pop_sim_results_meta = []
        state.pop_activated_set = []
        state.pop_xsec_figure = {"data": [], "layout": {}}
        state.pop_xsec_cuff_figure = {"data": [], "layout": {}}
        state.pop_propagation_figure = {"data": [], "layout": {}}
        state.pop_waterfall_figure = {"data": [], "layout": {}}
        state.pop_status = (
            "Population is stale — re-generate to update."
            if state.pop_branch_types
            else "No population generated yet."
        )

    def _default_pop_type_row() -> dict:
        """Server-side fallback factory for a new fiber-type row.
        The client-side "+ Add fiber type" inline-JS handler is
        the canonical add path (it derives `name` + `color` from
        the live state count); this function exists for any
        future server-side caller. Picks the next tab10 colour
        by total row count across all branches."""
        total = sum(
            len(rs) for rs in
            (state.pop_branch_types or {}).values()
        )
        return {
            "id": secrets.token_hex(4),
            "name": f"Type {total + 1}",
            "backend": "pyfibers",
            "model": MYELINATED_MODELS[0],
            "mean_um": 10.0,
            "std_um": 1.5,
            "frac": 100.0,
            "color": TAB10_PALETTE[total % len(TAB10_PALETTE)],
        }

    def do_pop_add_type(branch_idx: int, *_args) -> None:
        bt = dict(state.pop_branch_types or {})
        key = str(int(branch_idx))
        rows = list(bt.get(key, []))
        rows.append(_default_pop_type_row())
        bt[key] = rows
        state.pop_branch_types = bt
        # Editing the design invalidates the previous generate.
        if state.pop_generated:
            state.pop_generated = False
            state.pop_status = "Design changed — re-generate."

    def do_pop_remove_type(branch_idx: int, row_id: str,
                            *_args) -> None:
        bt = dict(state.pop_branch_types or {})
        key = str(int(branch_idx))
        rows = [r for r in bt.get(key, []) if r.get("id") != row_id]
        bt[key] = rows
        state.pop_branch_types = bt
        if state.pop_generated:
            state.pop_generated = False
            state.pop_status = "Design changed — re-generate."

    def do_pop_update_type(branch_idx: int, row_id: str,
                            field: str, value, *_args) -> None:
        """Field-level update (model / mean_um / std_um / frac)
        from the per-row inputs. Coerces numerics defensively
        since trame returns strings from VTextField."""
        bt = dict(state.pop_branch_types or {})
        key = str(int(branch_idx))
        rows = list(bt.get(key, []))
        for r in rows:
            if r.get("id") != row_id:
                continue
            if field in ("mean_um", "std_um", "frac"):
                try:
                    r[field] = float(value)
                except (TypeError, ValueError):
                    r[field] = 0.0
            else:
                r[field] = str(value)
            break
        bt[key] = rows
        state.pop_branch_types = bt
        if state.pop_generated:
            state.pop_generated = False
            state.pop_status = "Design changed — re-generate."

    # ---- F1.1: curated fiber-population presets ----
    # The pop_preset_choice live-preview watcher moved into
    # golgi.watchers.fiber_panel in step W1.7a.

    def do_pop_apply_preset(*_args) -> None:
        """Materialise the currently-selected preset into
        pop_branch_types — one row group per detected branch. Overwrites
        any existing rows (the dropdown sits next to a confirm-by-click
        button so the destructive nature is visible). Invalidates
        pop_generated so the user is prompted to regenerate."""
        name = str(state.pop_preset_choice or "")
        if not name:
            state.pop_status = (
                "Pick a preset above before clicking Apply."
            )
            return
        meta_list = list(state.pop_branches_meta or [])
        if not meta_list:
            state.pop_status = (
                "No fiber branches detected yet — generate fiber "
                "trajectories first, then re-apply the preset."
            )
            return
        new_bt = _state_defaults.pop_presets.apply_preset(
            name, meta_list, tab10_palette=TAB10_PALETTE,
        )
        state.pop_branch_types = new_bt
        if state.pop_generated:
            state.pop_generated = False
        meta = _state_defaults.pop_presets.preset_meta(name)
        n_rows = sum(len(v) for v in new_bt.values())
        state.pop_status = (
            f"Applied preset '{meta.get('label', name)}': "
            f"{n_rows} fiber-type rows across "
            f"{len(meta_list)} branch(es). Click Generate."
        )

    async def do_pop_generate(*_args) -> None:
        # Body moved to golgi.pipeline.pop_sim.run_pop_generate
        # (step 4.7).
        await _pipeline_pop_sim.run_pop_generate(_pipeline_ctx)

    @gated("pop_sim_run")
    async def do_pop_run_sim():
        # Body moved to golgi.pipeline.pop_sim.run_pop_sim
        # (step 4.7). The per-fiber InProcessRunner machinery
        # (FiberSimJobRequest + _do_one_fiber + _fiber_preflight)
        # is shared with run_fiber_sim — collapses the temporary
        # duplication introduced in 4.6.
        await _pipeline_pop_sim.run_pop_sim(_pipeline_ctx)

    # Fiber-tab + Pop-tab + branch-name watchers — extracted to
    # golgi.watchers.fiber_panel in step 5.2. Returns the
    # _fiber_pulse_params closure (state-bound) which gets wired
    # into _pipeline_helpers for the sim drivers.
    _fiber_pulse_params = _watchers.fiber_panel.register(
        state, geom=geom,
        fiber_diameter_config=FIBER_MODEL_DIAMETER_CONFIG,
        fiber_diameter_default=_FIBER_MODEL_DIAMETER_DEFAULT,
        max_fiber_branches=MAX_FIBER_BRANCHES,
        request_render=_request_render,
        rebuild_scene_state=_rebuild_scene_state,
        refresh_fiber_sel_items=_refresh_fiber_sel_items,
        branch_name=_branch_name,
        build_pulse_waveform=build_pulse_waveform,
        effective_anod_pw_ms=_fiber_effective_anod_pw_ms,
    )

    def _fiber_label_and_color(idx: int) -> tuple[str, str]:
        """Build a "Branch X · Fiber N" label + tab10 colour for
        a given fiber index. Same convention as the combobox
        items so the result-picker chips and the colour dots in
        both UIs agree visually."""
        color = TAB10_PALETTE[int(idx) % len(TAB10_PALETTE)]
        bidx_arr = geom.fiber_branch_idx
        if (bidx_arr is not None
                and 0 <= int(idx) < len(bidx_arr)):
            b = int(bidx_arr[int(idx)])
            if b >= 0:
                return f"Branch {b} · Fiber {int(idx)}", color
        return f"Fiber {int(idx)}", color




    def _ensure_fibers_in_cuff_frame(say=None) -> bool:
        # Body extracted to golgi.pipeline._frames in step 4.4
        # so the FEM pipeline driver can import it directly.
        # Thin wrapper here keeps the in-build_app call sites
        # (do_generate_fibers, _restore_fibers_from_disk, ...)
        # unchanged.
        return _ensure_fibers_in_cuff_frame_impl(
            geom=geom,
            out_dir=Path(get_active().out_dir),
            transform_to_cuff_frame_fn=transform_to_cuff_frame,
            say=say,
        )

    def _render_fibers_current_frame() -> None:
        """Draw the cached fibers in PURE PCA frame so they stay
        glued to the nerve actor regardless of which cuff was
        fit last.

        F3.2-M2.1f — all viewport content lives in pure PCA frame
        (centroid at origin, R_global rotation applied). Fibers
        stored in raw frame get transformed via (p - centroid) @
        R_global. Fibers already in cuff frame (rewritten by the
        FEM-solve preflight `_ensure_fibers_in_cuff_frame`) get
        undone first: pts_pca = pts_cuff + cuff_origin_pca (R_local
        is identity in current code, so no rotation to undo)."""
        if geom.fiber_paths_raw is None:
            return
        if (geom.fibers_in_cuff_frame
                and geom.cuff_origin_pca is not None):
            _off = np.asarray(
                geom.cuff_origin_pca, dtype=np.float64,
            )
            paths_display = [
                np.asarray(p, dtype=np.float64) + _off
                for p in geom.fiber_paths_raw
            ]
        elif (geom.centroid is not None
                and geom.R_global is not None):
            _c = np.asarray(geom.centroid, dtype=np.float64)
            _R = np.asarray(geom.R_global, dtype=np.float64)
            paths_display = [
                (np.asarray(p, dtype=np.float64) - _c) @ _R
                for p in geom.fiber_paths_raw
            ]
        else:
            paths_display = geom.fiber_paths_raw
        # Optional Ve overlay: only when the user has toggled
        # "Ve on fibers" on AND a FEM solve has produced per-path
        # Ve data. The per-path arrays are indexed identically to
        # geom.fiber_paths_raw (same generation order).
        ve_per_path = None
        if (bool(state.show_ve_fibers)
                and geom.fiber_paths_Ve is not None
                and len(geom.fiber_paths_Ve)
                    == len(paths_display)):
            ve_per_path = geom.fiber_paths_Ve
        render_fibers_by_branch(
            pl, paths_display,
            geom.fiber_branch_idx,
            geom.fiber_n_branches,
            ve_per_path=ve_per_path,
            ve_clim_mV=geom.ve_clim_mV,
        )
        # Reapply current visibility toggles to the newly-mounted
        # actors (a fresh add_mesh resets visibility to True).
        _apply_fiber_visibility()

    def _compute_field_lines_polydata() -> object | None:
        """Build 3D E-field streamlines (in mm) from the cached
        slice_volume.npz. Returns a pyvista PolyData of polylines
        with a per-point `E_mag` scalar for colouring, or None
        if there's no slice volume yet or the streamline
        integration produced an empty result. Heavy-ish (a few
        seconds for a typical mesh) so we cache on geom and
        only recompute after a fresh FEM solve."""
        if geom.fem_slice is None:
            return None
        try:
            sd = geom.fem_slice
            x = np.asarray(sd["x"], dtype=np.float64)
            y = np.asarray(sd["y"], dtype=np.float64)
            z = np.asarray(sd["z"], dtype=np.float64)
            Ex = np.asarray(sd["Ex"], dtype=np.float64)
            Ey = np.asarray(sd["Ey"], dtype=np.float64)
            Ez = np.asarray(sd["Ez"], dtype=np.float64)
            nx, ny, nz = len(x), len(y), len(z)
            if nx < 2 or ny < 2 or nz < 2:
                return None
            # Spacing + origin in mm so the streamlines land in
            # the same frame as every other actor.
            dx_mm = (x[-1] - x[0]) / (nx - 1) * 1.0e3
            dy_mm = (y[-1] - y[0]) / (ny - 1) * 1.0e3
            dz_mm = (z[-1] - z[0]) / (nz - 1) * 1.0e3
            origin_mm = (
                x[0] * 1.0e3, y[0] * 1.0e3, z[0] * 1.0e3,
            )
            grid = pv.ImageData(
                dimensions=(nx, ny, nz),
                spacing=(dx_mm, dy_mm, dz_mm),
                origin=origin_mm,
            )
            # slice_volume is shape (nz, ny, nx) with z slowest,
            # x fastest — exactly what ImageData expects when
            # dimensions=(nx, ny, nz). C-order ravel preserves
            # this layout.
            Ex_flat = Ex.ravel(order="C")
            Ey_flat = Ey.ravel(order="C")
            Ez_flat = Ez.ravel(order="C")
            # Replace NaN (off-mesh) with zero — pyvista's
            # streamline integrator stops at zero-magnitude
            # cells, which is the behaviour we want at the
            # domain boundary.
            for _arr in (Ex_flat, Ey_flat, Ez_flat):
                _bad = ~np.isfinite(_arr)
                if _bad.any():
                    _arr[_bad] = 0.0
            E_vec = np.column_stack([Ex_flat, Ey_flat, Ez_flat])
            grid["E"] = E_vec
            grid["E_mag"] = np.linalg.norm(E_vec, axis=1)
            grid.set_active_vectors("E")
            # Seed: a sphere of points centred at the cuff
            # origin, with radius ~half the smaller XY extent so
            # streamlines start inside the cuff and integrate
            # outward into the saline / muscle bath. ~300 seed
            # points = ~600 streamline halves (forward+backward)
            # which reads as a dense but legible flow field.
            _R_seed_mm = min(nx * dx_mm, ny * dy_mm) * 0.25
            if _R_seed_mm <= 0:
                _R_seed_mm = 4.0
            res = 14   # theta_resolution × phi_resolution ≈ 200
            seed = pv.Sphere(
                radius=_R_seed_mm,
                center=(0.0, 0.0, 0.0),
                theta_resolution=res,
                phi_resolution=res,
            )
            lines = grid.streamlines_from_source(
                seed,
                vectors="E",
                max_steps=400,
                terminal_speed=1e-9,
                integration_direction="both",
            )
            if lines is None or lines.n_points == 0:
                return None
            # Per-point |E| for colouring (vectors are carried
            # through streamline integration as point data).
            if "E" in lines.point_data:
                lines["E_mag"] = np.linalg.norm(
                    np.asarray(
                        lines.point_data["E"],
                        dtype=np.float64,
                    ),
                    axis=1,
                )
                # Re-mark "E" as the ACTIVE vector field on the
                # streamlines polydata so the arrow-glyph filter
                # downstream can find it via `orient="E"`. The
                # streamline integrator copies point data
                # through but doesn't re-elect an active set.
                try:
                    lines.set_active_vectors("E")
                except Exception:
                    pass
            return lines
        except Exception as ex:
            print(
                f"[field-lines] streamline build failed: {ex}",
                flush=True,
            )
            return None

    _field_lines_compute_inflight = {"value": False}

    def _ensure_field_lines_async() -> None:
        """Ensure `geom.field_lines_poly` is populated, then request
        a scene-state render pass. Streamline integration runs on
        a worker thread; the render pass that mounts the tubes /
        arrow glyphs ALWAYS runs on the main loop (via
        `_request_render`), so the executor-thread plotter race
        that caused symptom 4 is gone.

        Called from:
          - `_on_show_field_lines_change` (user toggle, main loop)
          - `_refresh_fem_plots` (after a fresh FEM solve drops the
            cached streamlines; this fn itself runs in an executor
            during project restore — we MUST stay off `pl` here)
          - `do_open_project` finaliser
        Re-entrant: while a compute is in flight, subsequent calls
        are no-ops."""
        if not bool(state.show_field_lines):
            # Toggle off → just retire the actors via a render
            # pass. The folder sees `show_field_lines=False` and
            # produces empty field groups.
            _request_render()
            return
        if geom.field_lines_poly is not None:
            _request_render()
            return
        if _field_lines_compute_inflight["value"]:
            return
        if geom.fem_slice is None:
            _request_render()
            return
        # Always route the streamline job through the captured
        # main loop so the compute lands on a pyvista-safe
        # context. Calling threads (executor during project
        # restore) post-and-forget here.
        loop = _main_loop_ref["loop"]
        if loop is None:
            try:
                loop = asyncio.get_running_loop()
                _main_loop_ref["loop"] = loop
            except RuntimeError:
                loop = None
        if loop is None or not loop.is_running():
            # First-time-mount fallback. No async context →
            # compute inline on whichever thread we're on (safe;
            # `_compute_field_lines_polydata` only touches pv
            # data, never `pl`), then ask for a render.
            _field_lines_compute_inflight["value"] = True
            try:
                geom.field_lines_poly = (
                    _compute_field_lines_polydata()
                )
            finally:
                _field_lines_compute_inflight["value"] = False
            _request_render()
            return
        _field_lines_compute_inflight["value"] = True

        async def _job():
            try:
                lines = await loop.run_in_executor(
                    None, _compute_field_lines_polydata,
                )
                geom.field_lines_poly = lines
            except Exception as ex:
                print(
                    f"[field-lines] integration failed: {ex}",
                    flush=True,
                )
                geom.field_lines_poly = None
            finally:
                _field_lines_compute_inflight["value"] = False
            _request_render()

        # Schedule from any thread — `run_coroutine_threadsafe`
        # is the cross-thread analogue of `ensure_future`.
        try:
            asyncio.run_coroutine_threadsafe(_job(), loop)
        except Exception as ex:
            print(f"[field-lines] schedule failed: {ex}",
                  flush=True)
            _field_lines_compute_inflight["value"] = False

    # Legacy alias — kept callable so any straggling caller still
    # gets the safe behaviour. The body does NOT touch `pl`.
    def _render_field_lines_actor() -> None:
        _ensure_field_lines_async()

    def render_built_mesh(pl: pv.Plotter, msh_path: Path) -> None:
        """Mount the pre-extracted per-region boundary surfaces
        (geom.region_surfaces, in mm) as actors. Heavy work is
        done in do_build_mesh's executor; this function is cheap
        — just `add_mesh()` per region — so it doesn't block the
        asyncio loop even for 22 M-tet meshes."""
        pl.remove_actor("silicone_overlay", reset_camera=False)
        pl.remove_actor("saline_overlay", reset_camera=False)
        pl.remove_actor("muscle_overlay", reset_camera=False)
        pl.remove_actor("nerve", reset_camera=False)
        for _i in range(64):
            pl.remove_actor(f"region_{_i}", reset_camera=False)
            pl.remove_actor(f"gold_overlay_{_i}", reset_camera=False)

        if geom.region_surfaces is None:
            return
        colour_by_q = bool(state.show_mesh_quality_color)
        ve_on_surf = (
            bool(state.show_ve_surface)
            and geom.nerve_surface_Ve is not None
        )
        # Pick the viz (decimated) dict for default rendering and
        # fall back to the full dict only where we need it:
        # quality colour (per-cell q_tet doesn't survive decimate)
        # and Vₑ-on-surface (point parity with nerve_surface_Ve).
        viz_dict = (geom.region_surfaces_viz
                    if geom.region_surfaces_viz is not None
                    else geom.region_surfaces)
        for _tag in [t for t in TAG_ORDER
                      if t in geom.region_surfaces]:
            surf_full = geom.region_surfaces[_tag]
            surf_viz = viz_dict.get(_tag, surf_full)
            spec = DEFAULTS.get(_tag, DEFAULTS[1])
            # Default render target — overridden below for the
            # quality / Vₑ branches that need the full polydata.
            surf = surf_viz
            # Vₑ-on-nerve overlay wins over the quality cmap when
            # both toggles are on. Applies to BOTH the endo (tag
            # 1) and epi (tag 5) surfaces: tag 1 uses the direct
            # FEM samples that solve_nerve.py wrote (point parity
            # with nerve_surface_Ve), and tag 5 borrows tag 1's
            # samples via a nearest-vertex KDTree lookup — the
            # endo↔epi shell is thin (50-200 µm) so the field is
            # essentially constant across it for visualisation.
            _ve_on_this_tag = (ve_on_surf and _tag in (1, 5))
            _ve_endo_len_ok = (
                ve_on_surf
                and 1 in geom.region_surfaces
                and (len(geom.nerve_surface_Ve)
                     == geom.region_surfaces[1].n_points)
            )
            if (ve_on_surf and _tag == 1
                    and not _ve_endo_len_ok):
                # Loud diagnostic: solve_nerve sampled Ve at the
                # endo surface vertices that golgi wrote out, so
                # if the lengths disagree the user's mesh has
                # been rebuilt since the solve — they need to
                # re-run FEM or the overlay can't be mapped.
                print(
                    f"[Ve overlay] length mismatch on tag 1: "
                    f"nerve_surface_Ve has "
                    f"{len(geom.nerve_surface_Ve):,} pts but "
                    f"region_surfaces[1] has "
                    f"{geom.region_surfaces[1].n_points:,} pts "
                    f"— re-run FEM to refresh",
                    flush=True,
                )
            if _ve_on_this_tag and _ve_endo_len_ok:
                # Vₑ array is indexed by full-surface vertex IDs
                # → render off the FULL polydata so the scalar
                # lines up cleanly.
                surf = surf_full
                if _tag == 1:
                    ve = np.asarray(
                        geom.nerve_surface_Ve,
                        dtype=np.float32,
                    ).copy()
                else:
                    # tag 5 (epi): nearest-vertex sampling from
                    # the endo Vₑ array onto this surface's
                    # points. KDTree query is fast (~few ms for
                    # tens of thousands of points).
                    from scipy.spatial import cKDTree
                    _endo_surf = geom.region_surfaces[1]
                    _tree = cKDTree(
                        np.asarray(
                            _endo_surf.points,
                            dtype=np.float64,
                        ),
                    )
                    _, _nn = _tree.query(
                        np.asarray(
                            surf_full.points,
                            dtype=np.float64,
                        ),
                        k=1,
                    )
                    ve = np.asarray(
                        geom.nerve_surface_Ve,
                        dtype=np.float32,
                    )[_nn].copy()
                good = np.isfinite(ve)
                if good.any():
                    ve[~good] = float(np.median(ve[good]))
                else:
                    ve[:] = 0.0
                surf_ve = surf.copy()
                surf_ve.point_data["Ve"] = ve * 1.0e3  # V → mV
                surf_ve.GetPointData().SetActiveScalars("Ve")
                # Use the SHARED Vₑ clim computed once per solve
                # in _refresh_fem_plots so endo, epi, and the
                # fiber tubes all map the same value to the same
                # colour (and so the horizontal colour bar on
                # the viewport applies to all three).
                if geom.ve_clim_mV is not None:
                    clim = geom.ve_clim_mV
                else:
                    _good = np.isfinite(ve)
                    if _good.any():
                        ve_mv = ve * 1.0e3
                        v_lo = float(
                            np.percentile(ve_mv[_good], 1.0),
                        )
                        v_hi = float(
                            np.percentile(ve_mv[_good], 99.0),
                        )
                    else:
                        v_lo, v_hi = -1.0, 1.0
                    if v_hi - v_lo < 1e-12:
                        v_hi = v_lo + 1.0
                    clim = (v_lo, v_hi)
                # Per-tag opacity so BOTH the endo (inner) and
                # epi (outer shell) Vₑ paints are simultaneously
                # visible. Without this, the epi rendered at
                # opacity=1.0 completely hid the endo painted
                # underneath, leaving the user with only the
                # epi swatch on the screen.
                _ve_opacity = (
                    1.0 if _tag == 1
                    else 0.45  # tag 5 (epi): semi-transparent
                )
                # Phong material params so the Vₑ-coloured
                # surface still gets the cinematic shading.
                _ve_phong = DEFAULTS.get(_tag, DEFAULTS[1])
                actor = pl.add_mesh(
                    surf_ve, name=f"region_{_tag}",
                    scalars="Ve",
                    cmap="plasma", clim=clim,
                    opacity=_ve_opacity,
                    pbr=False,
                    ambient=_ve_phong["ambient"],
                    diffuse=_ve_phong["diffuse"],
                    specular=_ve_phong["specular"],
                    specular_power=_ve_phong["specular_power"],
                    show_edges=False, smooth_shading=True,
                    show_scalar_bar=False,
                )
                try:
                    _mapper = actor.GetMapper()
                    _mapper.SetScalarModeToUsePointData()
                    _mapper.SelectColorArray("Ve")
                    _mapper.ScalarVisibilityOn()
                except Exception:
                    pass
            elif colour_by_q and "q_tet" in surf_full.cell_data:
                # Quality-by-colour needs per-cell q_tet which
                # doesn't survive the decimation, so render from
                # the full polydata in this branch only. Phong
                # material params preserve the cinematic shading
                # under the q-colour mapping.
                surf = surf_full
                actor = pl.add_mesh(
                    surf, name=f"region_{_tag}",
                    scalars="q_tet",
                    cmap="RdYlGn", clim=(0.0, 1.0),
                    opacity=spec["opacity"],
                    pbr=False,
                    ambient=spec["ambient"],
                    diffuse=spec["diffuse"],
                    specular=spec["specular"],
                    specular_power=spec["specular_power"],
                    show_edges=False, smooth_shading=True,
                    show_scalar_bar=False,
                )
            else:
                actor = _add_phong_mesh(
                    pl, surf, name=f"region_{_tag}", style=spec,
                )
            actor.visibility = bool(state[f"vis_{_tag}"])

    def _refresh_fem_plots(slice_only: bool = False) -> None:
        """Re-render the §9 axis ribbon + §9 slice heatmap +
        §10 AF plot from the currently-cached FEM outputs. Called
        after every solve, after every slice slider tick, and on
        project open. `slice_only=True` skips the axis + AF
        plots (the cheap path for slice-slider scrubbing)."""
        if geom.fem_slice is None or geom.fem_axis is None:
            return
        # Slice plot geometry context — fed from the active mesh
        # config so the cuff outlines + electrode arcs match what
        # the FEM solver actually saw. Patches are reconstructed
        # from the current state so the side-by-side plot reflects
        # the user's latest electrode-type / sliders without
        # needing to write a sidecar file.
        try:
            L_cuff_m = float(state.L_cuff_mm) * 1.0e-3
        except Exception:
            L_cuff_m = 0.0
        R_ci_m = float(geom.R_ci or 0.0)
        R_co_m = float(geom.R_co or 0.0)
        # Muscle radius derived the same way as in do_solve_fem
        # (max nerve radius + radial pad), so the dashed outer
        # ring on the slice plot lands where the FEM mesh actually
        # ends.
        if geom.pts_cuff is not None:
            r_max = float(np.linalg.norm(
                geom.pts_cuff[:, :2], axis=1).max(),
            )
        else:
            r_max = R_co_m * 1.5
        try:
            muscle_pad_m = float(state.muscle_radial_pad_mm) * 1.0e-3
        except Exception:
            muscle_pad_m = 0.0
        muscle_R_m = max(R_co_m + muscle_pad_m, r_max * 1.05)
        try:
            elec_patches = build_electrode_patches_dicts(
                L_cuff_m, R_ci_m,
                kind=str(state.electrode_type),
                cfg={k: state[k]
                     for k in DEFAULT_ELECTRODE
                     if k != "electrode_type"},
            )
        except Exception:
            elec_patches = []
        # Triangulation indices come from the loaded nerve dict.
        b_raw = (np.asarray(
                    geom.nerve.get("boundary_raw"), dtype=np.int64,
                 )
                 if (geom.nerve is not None
                     and "boundary_raw" in geom.nerve)
                 else None)
        state.fem_slice_figure = _build_fem_slice_figure(
            geom.fem_slice,
            L_cuff_m=L_cuff_m,
            R_ci_m=R_ci_m, R_co_m=R_co_m,
            pts_cuff=geom.pts_cuff,
            boundary_raw=b_raw,
            muscle_R_m=muscle_R_m,
            electrode_patches=elec_patches,
            init_z_idx=int(state.fem_slice_z_idx),
        )
        if slice_only:
            return
        # Prefer the paths_flat that solve_nerve.py snapshotted
        # inside paths_Ve.npz — those are guaranteed to be the
        # exact coordinates where Ve was sampled, so the (s, Ve,
        # Ez) triple is consistent even if fibers got regenerated
        # after the FEM solve. Fall back to fiber_paths_raw only
        # for legacy projects where paths_Ve.npz pre-dates the
        # paths_flat snapshot.
        _af_paths = (geom.fiber_paths_for_Ve
                     if geom.fiber_paths_for_Ve is not None
                     else geom.fiber_paths_raw)
        state.fem_axis_figure = _build_fem_axis_figure(
            paths_Ve=geom.fiber_paths_Ve,
            paths_Ez=geom.fiber_paths_Ez,
            paths_raw=_af_paths,
            branch_idx=geom.fiber_branch_idx,
            I_stim_mA=float(state.I_stim_mA),
        )
        state.fem_af_figure = _build_fem_af_figure(
            paths_Ve=geom.fiber_paths_Ve,
            paths_raw=_af_paths,
            branch_idx=geom.fiber_branch_idx,
            sel_fiber=int(state.fem_fiber_sel),
            sg_window=int(state.fem_sg_window),
        )
        # Fresh slice_volume → previous streamlines are stale.
        # Drop the cache so the next render pass / toggle
        # recomputes against the new E field. If the toggle is
        # currently on, schedule a recompute + render on the
        # main loop (NEVER from the executor thread that called
        # _refresh_fem_plots during project restore — that was
        # symptom 4's race).
        geom.field_lines_poly = None
        if bool(state.show_field_lines):
            _ensure_field_lines_async()
        # Shared Vₑ clim (mV) + colourbar PNG. Computed from
        # nerve_surface_Ve (the full-volume FEM sample) with a
        # 1/99 percentile clip so outliers near the contact
        # facets don't crush the dynamic range. Stored on geom
        # so render_built_mesh + render_fibers_by_branch can
        # both use the same scale.
        if (geom.nerve_surface_Ve is not None
                and geom.nerve_surface_Ve.size > 0):
            _ve_mV = np.asarray(
                geom.nerve_surface_Ve, dtype=np.float64,
            ) * 1.0e3
            _good = np.isfinite(_ve_mV)
            if _good.any():
                v_lo = float(
                    np.percentile(_ve_mV[_good], 1.0),
                )
                v_hi = float(
                    np.percentile(_ve_mV[_good], 99.0),
                )
                if v_hi - v_lo < 1e-9:
                    v_hi = v_lo + 1.0
                geom.ve_clim_mV = (v_lo, v_hi)
                state.fem_ve_cbar_b64 = _render_ve_colorbar_png(
                    v_lo, v_hi,
                )
            else:
                geom.ve_clim_mV = None
                state.fem_ve_cbar_b64 = ""
        else:
            geom.ve_clim_mV = None
            state.fem_ve_cbar_b64 = ""

    # -----------------------------------------------------------------
    # Project lifecycle handlers (autosave / open / close / create).
    # All file I/O is bounded by the active project's directory
    # (GOLGI_OUT) so each project stays a self-contained bundle.
    # -----------------------------------------------------------------
    def _refresh_projects_list() -> None:
        """Rescan PROJECTS_ROOT and push the new tile list to the
        welcome view. Filters by the active user — logged-out
        sessions get an empty list (the welcome view shows
        a "Please sign in" panel instead)."""
        state.projects_list = _list_projects(
            owner_user_id=_auth_session.get("user_id"),
        )

    def _snapshot_ui_state() -> dict:
        """Collect the persisted state keys into a JSON-safe dict.
        Pulled from server state at call time so it always reflects
        the current UI."""
        snap: dict = {}
        for k in _PERSISTED_UI_KEYS:
            try:
                v = state[k]
            except KeyError:
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                snap[k] = v
            elif isinstance(v, (list, tuple, dict)):
                # Nested JSON-serialisable values (`electrodes`).
                # Round-trip through json to assert serialisability
                # — if it fails, fall back to a string so we don't
                # crash autosave on a stray non-JSON object.
                try:
                    snap[k] = json.loads(json.dumps(v))
                except Exception:
                    snap[k] = str(v)
            else:
                # Best-effort: stringify; restore path falls back
                # to defaults for that key if needed.
                snap[k] = str(v)
        return snap

    def _apply_ui_state(snap: dict) -> None:
        """Restore persisted state values onto the server state.
        Writes are batched in a single `with state:` so client gets
        one consolidated push + change watchers fire once.

        Critical: we set `_elec_sync_guard["loading"] = True`
        around the batch so the cuff-position / geometry /
        electrode / Vₑ-overlay / FEM-plot watchers all bail out.
        Without the guard, restoring ~20 state vars fires ~10
        fire-and-forget `asyncio.create_task(do_fit_cuff(...))`
        and a handful of render_built_mesh / refresh-FEM-plots
        calls — they pile up on the event loop and finish AFTER
        do_open_project's lightbox-close handshake, which is the
        2-3 minute "ghost work" the user kept seeing. The proper
        re-fit is called explicitly later in do_open_project."""
        if not snap:
            return
        # F3.2a back-compat: projects saved before the rename
        # carry the old "electrodes" / "selected_electrode_id" /
        # "next_electrode_seq" keys. Map them onto the new ones
        # at load time so legacy projects open without losing
        # their cuff layout.
        snap = dict(snap)
        _legacy_to_new = {
            "electrodes": "designs",
            "selected_electrode_id": "selected_design_id",
            "next_electrode_seq": "next_design_seq",
        }
        for _old, _new in _legacy_to_new.items():
            if _old in snap and _new not in snap:
                snap[_new] = snap.pop(_old)
        _elec_sync_guard["loading"] = True
        try:
            with state:
                for k, v in snap.items():
                    if k not in _PERSISTED_UI_KEYS:
                        continue
                    try:
                        state[k] = v
                    except Exception:
                        pass
        finally:
            _elec_sync_guard["loading"] = False

    def _apply_persisted_sigma() -> None:
        """Reload conductivities.json from the active project, if
        present, and overwrite the σ state vars. Called on project
        open so opening a saved project picks up its σ overrides
        instead of inheriting from the previously-open project."""
        path = GOLGI_OUT / "conductivities.json"
        if not path.exists():
            return
        try:
            saved = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return
        _elec_sync_guard["loading"] = True
        try:
            with state:
                for k, v in DEFAULT_SIGMA.items():
                    try:
                        state[k] = float(saved.get(k, v))
                    except Exception:
                        pass
        finally:
            _elec_sync_guard["loading"] = False

    def _capture_thumbnail() -> bool:
        """Write thumbnail.png into the active project from a
        screenshot of the current 3D viewport. No-op when no
        project is active or the plotter has no actors yet."""
        if not state.has_active_project:
            return False
        pdir = Path(state.current_project_dir)
        if not pdir.is_dir():
            return False
        try:
            # pl.screenshot writes a PNG to disk directly via VTK's
            # offscreen render window — no PIL/imageio dependency.
            # The image size matches the plotter's render window
            # (Trame default ≈ 1200x800), which is fine for tiles.
            pl.screenshot(
                filename=str(pdir / "thumbnail.png"),
                transparent_background=False,
                return_img=False,
            )
            return True
        except Exception as ex:
            print(f"[project] thumbnail capture failed: {ex}",
                   flush=True)
            return False

    def _override_preset_expr(expr_raw, val_si: float) -> str:
        """Replace the literal numeric in `expr_raw` with `val_si`
        while preserving its `[unit]` annotation, so a user-tuned
        SI value round-trips back through the ASCENT-style
        preset format with its original display unit intact.
        Falls back to a unit-less SI literal when the original
        expression has no `[unit]` suffix."""
        s = str(expr_raw or "")
        m = re.search(r"\s*\[\s*([a-zA-Z]+)\s*\]\s*$", s)
        if m is None:
            return repr(float(val_si))
        unit = m.group(1).strip()
        factor = cuff_designer._UNIT_FACTORS.get(unit, 1.0)
        if factor == 0:
            return repr(float(val_si))
        val_disp = float(val_si) / factor
        return f"{val_disp:g} [{unit}]"

    def _save_electrode_configs(pdir: Path) -> None:
        """Export one JSON per electrode under
        `<project>/electrodes/<eid>.json`. DUKE-typed electrodes
        write a self-contained preset in the ASCENT format
        (params + instances + local_params + offset block),
        with the user's slider overrides baked into the params
        block — so the file is directly editable / shareable
        with ASCENT or other tooling. Standard parametric
        electrodes write a simpler `_golgi_format: parametric`
        record capturing their slider state.
        A `_golgi_meta` block on every file carries the
        electrode's identity + placement (eid, name, position,
        per-electrode R_local, R_ci/R_co), so the file is
        re-importable without losing context. Stale files are
        cleaned up so renamed/removed electrodes don't leave
        orphans on disk."""
        if not pdir or not Path(pdir).is_dir():
            return
        out_dir = Path(pdir) / "electrodes"
        out_dir.mkdir(exist_ok=True)
        electrodes = list(state.designs or [])
        live_files = set()
        for elec in electrodes:
            eid = str(elec.get("eid", "")).strip()
            if not eid:
                continue
            common_meta = {
                "eid": eid,
                "name": str(elec.get("name", "")),
                "electrode_type": str(
                    elec.get(
                        "electrode_type", "bipolar ring-pair",
                    ),
                ),
                "cuff_offset_mm": float(
                    elec.get("cuff_offset_mm", 0.0),
                ),
                "cuff_dx_mm": float(elec.get("cuff_dx_mm", 0.0)),
                "cuff_dy_mm": float(elec.get("cuff_dy_mm", 0.0)),
                "L_cuff_mm": float(
                    elec.get("L_cuff_mm",
                                DEFAULT_CUFF["L_cuff_mm"]),
                ),
                "R_ci_m": elec.get("R_ci_m"),
                "R_co_m": elec.get("R_co_m"),
                "R_local_elec": elec.get("R_local_elec"),
                "show_saline": bool(
                    elec.get("show_saline", True),
                ),
            }
            if (str(elec.get("electrode_type", ""))
                    == DUKE_ELECTRODE_TYPE):
                preset_name = str(elec.get("duke_preset", ""))
                base = _CUFF_PRESETS.get(preset_name)
                if base is None:
                    # Preset unresolved (deleted / renamed file
                    # on disk). Skip — the manifest copy of the
                    # electrode is still the source of truth.
                    continue
                cfg = json.loads(json.dumps(base))   # deep copy
                overrides_si = {
                    str(k): float(v)
                    for k, v in (
                        elec.get("duke_overrides", {}) or {}
                    ).items()
                }
                for p in cfg.get("params", []):
                    pname = p.get("name")
                    if pname in overrides_si:
                        p["expression"] = _override_preset_expr(
                            p.get("expression"),
                            overrides_si[pname],
                        )
                cfg["_golgi_meta"] = {
                    **common_meta,
                    "duke_preset": preset_name,
                    "duke_overrides_si": overrides_si,
                }
            else:
                cfg = {
                    "_golgi_format": "parametric",
                    "_golgi_meta": common_meta,
                    "params": {
                        k: elec.get(k)
                        for k in DEFAULT_ELECTRODE
                        if k != "electrode_type"
                    },
                }
            path = out_dir / f"{eid}.json"
            path.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
            live_files.add(path.name)
        # Sweep orphan files: any *.json in the dir not produced
        # by this pass (e.g. removed electrode) gets deleted, so
        # the export view tracks `state.designs` exactly.
        for f in out_dir.glob("*.json"):
            if f.name not in live_files:
                try:
                    f.unlink()
                except OSError:
                    pass

    def _autosave(stage: str | None = None,
                    capture_thumb: bool = True) -> None:
        """Persist current UI state + manifest + thumbnail. Called
        after every major step (mesh / FEM / fibers / source load)
        and on project close. `stage` is now ignored — labels are
        user-managed via the detail-dialog UI rather than auto-
        accumulated. Kept in the signature for callsite stability."""
        del stage  # unused; auto-stage tracking has been removed
        if not state.has_active_project:
            return
        pdir = Path(state.current_project_dir)
        if not pdir.is_dir():
            return
        try:
            snap = _snapshot_ui_state()
            data = _write_manifest(
                pdir,
                name=state.current_project_name or pdir.name,
                ui_state=snap,
            )
            state.current_project_modified = _format_modified(
                data.get("last_modified", ""),
            )
            # Per-electrode JSON sidecars for external exchange.
            # Cheap to write, runs alongside the manifest.
            _save_electrode_configs(pdir)
            if capture_thumb:
                _capture_thumbnail()
        except Exception as ex:
            print(f"[project] autosave failed: {ex}", flush=True)

    def do_save_project(*_args) -> None:
        """Explicit user-triggered save (File → Save). Calls
        `_autosave` and surfaces a tiny confirmation in the
        navbar saved-chip via the manifest's last_modified
        timestamp. Quiet if no project is open."""
        if not state.has_active_project:
            state.fiber_sim_status = (
                "⚠ no active project to save."
            )
            return
        _autosave(capture_thumb=True)

    def _clear_plotter_actors() -> None:
        """Wipe every actor from BOTH plotters (main workspace +
        designer dialog) on project close / project switch so the
        next project starts on a blank canvas. Belt-and-suspenders:
        we explicitly remove every named actor we know about, then
        call `.clear()` AND `RemoveAllViewProps()` on the renderer
        as a final safety net. `.clear()` alone has been observed
        to leave named-mount actors in the underlying vtkRenderer
        in some pyvista versions, which then bleeds into the next
        project as ghost fibers / cuff overlays."""
        named_actors = (
            "nerve", "silicone_overlay", "saline_overlay",
            "muscle_overlay", "designer_nerve",
            # New scene-state actor names introduced by the rework
            # (Vₑ-merged tubes + streamlines + arrow glyphs).
            "fiber_ve",
            "field_lines", "field_lines_arrows",
        )
        for nm in named_actors:
            pl.remove_actor(nm, reset_camera=False)
        # Multi-fiber highlight: per-fiber actors mounted under
        # `fiber_selected_<idx>`. Selection size is bounded by
        # the total fiber count, but clearing up to 1024 here is
        # a safe over-shoot (project switching is rare and cheap).
        for _i in range(1024):
            pl.remove_actor(
                f"fiber_selected_{_i}", reset_camera=False,
            )
        for _i in range(64):
            pl.remove_actor(
                f"gold_overlay_{_i}", reset_camera=False,
            )
            pl.remove_actor(
                f"region_{_i}", reset_camera=False,
            )
        for _i in range(MAX_FIBER_BRANCHES):
            pl.remove_actor(
                f"fiber_branch_{_i}", reset_camera=False,
            )
        for _i in range(256):
            pl.remove_actor(
                f"designer_part_{_i}", reset_camera=False,
            )
        # Walk the live actor dict twice over — once for `elec_*`
        # (legacy + current per-electrode actors) and once for any
        # other name we didn't anticipate. Without the second
        # sweep, a renamed-but-still-present actor (e.g. the
        # post-rework `<eid>_contacts_<i>` plural form) could slip
        # through.
        for actor_name in list(pl.actors.keys()):
            try:
                pl.remove_actor(actor_name, reset_camera=False)
            except Exception:
                pass
        try:
            pl.clear()
        except Exception:
            pass
        # Final hammer: VTK-level eviction. `vtkRenderer
        # .RemoveAllViewProps()` drops every prop (actor, volume,
        # 2D actor) from the renderer regardless of whether
        # pyvista's wrapper dict knows about it. Lights are a
        # separate collection so they survive. This is what
        # actually closes the "ghost actor still in the renderer
        # even though pl.actors is empty" race we kept hitting.
        try:
            pl.renderer.RemoveAllViewProps()
        except Exception:
            pass
        # The scene-state pipeline tracks which actors are
        # currently mounted in `_rendered_sigs`. After a hard
        # `pl.clear()` those caches are lying — the next render
        # pass would compare signatures, see no change, and skip
        # the add_mesh, leaving the scene empty. Reset all scene
        # caches in lockstep with the plotter wipe.
        _rendered_sigs.clear()
        _scene_state["nerve"] = _mkgrp()
        for tag in _scene_state["regions"]:
            _scene_state["regions"][tag] = _mkgrp()
        _scene_state["fibers"]["mode"] = "off"
        for i in _scene_state["fibers"]["branches"]:
            _scene_state["fibers"]["branches"][i] = _mkgrp()
        _scene_state["fibers"]["ve"] = _mkgrp()
        _scene_state["fibers"]["selected"] = _mkgrp()
        _scene_state["field"]["tubes"] = _mkgrp()
        _scene_state["field"]["arrows"] = _mkgrp()
        _scene_state["electrodes"] = {}
        # Also wipe the designer plotter so an old cuff design
        # doesn't reappear when the user opens the Electrode
        # designer in the next project.
        try:
            pl_cuff.clear()
        except Exception:
            pass
        # Push the cleared scene to the client. Without an
        # explicit update, the WebGL view can hold the previous
        # actor list until the next interaction.
        try:
            ctrl.view_update()
        except Exception:
            pass
        try:
            ctrl.view_cuff_update()
        except Exception:
            pass

    def _reset_geom_and_state() -> None:
        """Drop in-memory geometry caches + reset all per-project
        state flags. Called on project close so the welcome view
        + next-opened project both start from a clean slate. This
        is INTENTIONALLY thorough: any state var that holds a
        per-project value gets reset, otherwise it bleeds into the
        next project as a "ghost" of the previous session."""
        for slot in (
            "nerve", "centroid", "R_global", "cuff_origin_pca",
            "R_local", "pts_cuff", "R_ci", "R_co", "msh_path",
            "mesh_nodes", "mesh_elems", "mesh_tags", "mesh_q",
            "region_surfaces", "region_surfaces_viz",
            "designs_meshes",
            "fem_axis", "fem_slice",
            "fiber_paths_Ve", "fiber_paths_Ez",
            "fiber_paths_for_Ve",
            "nerve_surface_Ve", "nerve_q",
            "fiber_paths_raw", "fiber_branch_idx",
            "fiber_pop_types", "fiber_pop_rows",
            "fiber_pop_diameters_um", "fiber_pop_sim_results",
            "fiber_sim_data", "fiber_sim_results",
            "_cuff_designer_parts", "nerve_poly",
        ):
            setattr(geom, slot, None)
        geom.fiber_n_branches = 0
        geom.fibers_in_cuff_frame = False
        geom.ve_clim_mV = None
        geom.field_lines_poly = None
        geom._fit_locked = False
        geom._R_local_cached = None
        geom._R_ci_cached = None
        geom._needs_camera_reset = False
        with state:
            # 1) Reset every persisted UI key back to its factory
            #    default. This is the catch-all that ensures slider
            #    values (scale_factor, cuff geometry, σ values,
            #    visibility toggles, etc.) don't bleed across
            #    projects. List the exceptions explicitly below if
            #    a particular key should NOT reset on close.
            for _k, _v in _FACTORY_DEFAULTS.items():
                try:
                    state[_k] = _v
                except Exception:
                    pass
            # 2) Stage / status flags + cached display strings.
            state.has_geometry = False
            state.has_mesh = False
            state.has_fem = False
            state.has_fibers = False
            state.has_designer_cuff = False
            state.geom_summary = "no geometry loaded"
            state.quality_hist_figure = {"data": [], "layout": {}}
            state.mesh_stats_html = ""
            state.mesh_quality_hist_figure = {"data": [], "layout": {}}
            state.designs_mesh_panels = []
            state.fiber_stats_html = ""
            state.fiber_branch_summary = []
            state.branch_rename_active = -1
            state.branch_rename_value = ""
            state.fiber_n_branches = 0
            # Population tab — wipe the per-branch type design and
            # the generated KDE so a new project doesn't inherit
            # the prior project's mixture.
            state.pop_branches_meta = []
            state.pop_branch_types = {}
            state.pop_generated = False
            state.pop_status = "No population generated yet."
            state.pop_type_colors = {}
            state.pop_row_colors = {}
            state.pop_row_meta = {}
            state.pop_row_visible = {}
            state.pop_kde_figure = {"data": [], "layout": {}}
            state.pop_sim_done = False
            state.pop_sim_results_meta = []
            state.pop_activated_set = []
            state.pop_view_idx = 0
            state.pop_xsec_figure = (
                {"data": [], "layout": {}}
            )
            state.pop_propagation_figure = (
                {"data": [], "layout": {}}
            )
            state.pop_waterfall_figure = (
                {"data": [], "layout": {}}
            )
            # Single-fiber tab — clear any stale multi-pick set
            # + batch results so a freshly loaded project starts
            # with an empty playground.
            state.fiber_sel_indices = []
            state.fiber_sel_tab = "all"
            state.fiber_sim_results_meta = []
            state.fiber_sim_status = "No simulation run yet."
            state.fiber_sim_summary = ""
            state.fiber_sim_log = ""
            state.fiber_propagation_figure = (
                {"data": [], "layout": {}}
            )
            state.fiber_waterfall_figure = (
                {"data": [], "layout": {}}
            )
            state.fem_axis_b64 = ""
            state.fem_slice_b64 = ""
            state.fem_af_b64 = ""
            state.fem_ve_cbar_b64 = ""
            state.fem_slice_figure = {"data": [], "layout": {}}
            state.fem_axis_figure = {"data": [], "layout": {}}
            state.fem_af_figure = {"data": [], "layout": {}}
            state.fem_status = "No FEM run yet."
            state.fem_log = ""
            state.fem_failed = False
            state.mesh_log = "Mesh not built yet."
            state.fiber_log = "No trajectories generated yet."
            state.fiber_status = (
                "Load a nerve first to enable fibers."
            )
            state.fiber_failed = False
            # 3) Transient electrode-list bridge state vars (not in
            #    _FACTORY_DEFAULTS because they're never persisted).
            state.rename_eid_active = ""
            state.rename_eid_value = ""
            state.rename_design_request = None
            state.remove_design_request = ""
            state.refit_design_request = ""
            state.confirm_delete_eid = ""
            state.confirm_delete_name = ""
            state.show_confirm_delete_dialog = False
            state.show_confirm_remove_geometry_dialog = False
            # 4) Electrode designer dialog state (also non-persisted).
            state.show_cuff_designer_dialog = False
            state.cuff_preset_name = ""
            state.cuff_preset_code = ""
            state.cuff_param_rows = []
            state.cuff_preview_b64 = ""
            state.cuff_designer_status = "Pick a preset to load."
            for _n in _CUFF_ALL_VISIBLE_NAMES:
                state[f"cuff_p_{_n}"] = 0.0
            # 5) Importer file list — fall back to whatever's
            #    globally visible (data/) since no project is active.
            state.data_files = list_data_files()
            # 6) Close any open drawers / dialogs so the next view
            #    starts tidy.
            state.show_import = False
            state.show_import_stepper = False
            state.import_stepper_step = "1"
            state.show_cuff = False
            state.show_mesh = False
            state.show_sigma = False
            state.show_fibers = False
            state.show_solve = False
            state.show_fiber = False
            state.show_pop = False
            state.show_close_dialog = False
            state.show_cancel_dialog = False
            state.show_new_project_dialog = False
            state.show_delete_dialog = False
            state.show_detail_dialog = False
            state.show_cole_cole_dialog = False

    async def _restore_mesh_from_disk() -> bool:
        """Reload EVERY design's nerve.msh + per-region surfaces
        from disk into geom.designs_meshes. Skips silently when no
        design has a mesh file on disk. Mirrors the post-TetGen
        tail of do_build_mesh without rerunning the (slow) TetGen
        step.

        F3.2-M1: loads ALL designs simultaneously (was: only the
        active one). Each design's on-disk nerve.msh is in its
        own cuff-local frame; we rotate per-design back to the
        shared PCA-translated viewport frame so all designs
        co-render.

        Single-slot fields (geom.mesh_nodes / region_surfaces /
        …) mirror the currently-selected design's data for back-
        compat with legacy callers (FEM driver bbox calc, project
        bundle export, etc.)."""
        designs = list(state.designs or [])
        if not designs:
            return False
        loop = asyncio.get_event_loop()

        # Precompute PCA + anchor origin ONCE; cheap and identical
        # for every design.
        anchor_origin_pca = None
        pts_pca_all = None
        if (geom.nerve is not None
                and geom.centroid is not None
                and geom.R_global is not None):
            pts_pca_all = (
                (geom.nerve["pts_raw"] - geom.centroid)
                @ geom.R_global
            )
            # SceneCatalog Phase 4 — must match the mesh-build
            # pipeline's canonical frame, which forces
            # `anchor_origin_pca = np.zeros(3)` (pure PCA, no
            # anchor offset). Pre-Phase-4 this branch called
            # `anchor_origin_pca_for_designs(...)` which returns
            # a non-zero anchor, so disk-restored meshes ended
            # up offset by `-anchor` from their build-time PCA
            # positions — the bug that broke "open project →
            # mesh and fibers misaligned". The previous
            # standalone M2.1h patch fixed this in isolation;
            # this commit re-applies it as part of the catalog
            # cut-over where the canonical-frame contract is
            # now structurally owned.
            anchor_origin_pca = np.zeros(3, dtype=np.float64)

        def _load_one(eid: str) -> dict | None:
            """Heavy per-design load — runs in the executor pool."""
            msh_dir = _fem_layout.design_dir(GOLGI_OUT, eid)
            msh_path = msh_dir / "nerve.msh"
            if not msh_path.exists():
                return None
            try:
                m = meshio.read(str(msh_path))
            except Exception as ex:
                print(
                    f"[mesh-restore] {eid}: meshio.read failed: "
                    f"{ex}", flush=True,
                )
                return None
            nodes = np.asarray(m.points, dtype=np.float64)
            if "tetra" not in m.cells_dict:
                return None
            elems = np.asarray(
                m.cells_dict["tetra"], dtype=np.int64,
            )
            tags = np.asarray(
                m.cell_data_dict.get(
                    "gmsh:physical", {},
                ).get("tetra", np.zeros(len(elems))),
                dtype=np.int32,
            )
            q_tet = _tet_shape_quality(nodes, elems)
            stats_html = _compute_mesh_stats_html(
                nodes, elems, tags, q_tet,
                defaults_by_tag=DEFAULTS,
            )
            hist_fig = _build_quality_histogram_figure(
                q_tet,
                x_label="tet quality (6√2·V / max_edge³)",
                y_label="# tetrahedra",
            )
            region_surfaces = _extract_region_surfaces_mm(
                nodes, elems, tags, q_tet,
            )
            return {
                "msh_path": msh_path,
                "nodes": nodes,
                "elems": elems,
                "tags": tags,
                "q_tet": q_tet,
                "region_surfaces": region_surfaces,
                "stats_html": stats_html,
                "hist_fig": hist_fig,
            }

        def _to_viewport_frame(loaded: dict, design: dict) -> dict:
            """Rotate this design's loaded nodes + region surfaces
            from D's cuff-local (on-disk frame) into the shared
            PCA-translated viewport frame. Returns a NEW dict
            with the transformed arrays."""
            if (pts_pca_all is None
                    or anchor_origin_pca is None):
                return loaded
            from golgi.scene.cuff_fit import (
                _design_M, find_cuff_origin_pca,
            )
            _M_D = _design_M(design)
            _cuff_origin_D_pca = find_cuff_origin_pca(
                pts_pca_all, state.cuff_anchor,
                float(design.get("cuff_offset_mm", 0.0)),
                float(design.get("cuff_dx_mm", 0.0)),
                float(design.get("cuff_dy_mm", 0.0)),
            )
            _design_offset_canon_m = (
                _cuff_origin_D_pca - anchor_origin_pca
            )
            # F3.2-M2.1f fix — mirror the unit-aware translation
            # split from pipeline/mesh.py. Nodes loaded from
            # nerve.msh come in METRES; region surfaces from
            # `extract_region_surfaces_mm` come in MILLIMETRES.
            # Using a single metres-scale offset for both was a
            # 1000× silent error that left region surfaces at
            # the design-local origin instead of their PCA cuff
            # origin (visible only with non-zero cuff offsets).
            _design_offset_canon_mm = (
                _design_offset_canon_m * 1000.0
            )
            _M_D_T = np.asarray(_M_D, dtype=np.float64).T
            xformed = dict(loaded)
            xformed["nodes"] = (
                loaded["nodes"] @ _M_D_T
                + _design_offset_canon_m
            )
            _rs_view = {}
            for _tag, _poly in (
                loaded.get("region_surfaces") or {}
            ).items():
                _pts = np.asarray(
                    _poly.points, dtype=np.float64,
                )
                _new = _poly.copy(deep=True)
                _new.points = (
                    _pts @ _M_D_T
                    + _design_offset_canon_mm
                )
                _rs_view[_tag] = _new
            xformed["region_surfaces"] = _rs_view
            return xformed

        # Heavy load — one design at a time in the executor.
        # Could parallelise via gather, but sequential is fine for
        # the typical 1-5 designs and avoids hammering the OS with
        # multiple meshio reads.
        designs_meshes: dict = {}
        active_id = str(state.selected_design_id or "")
        active_loaded: dict | None = None
        for design in designs:
            eid = str(design.get("eid", ""))
            if not eid:
                continue
            loaded = await loop.run_in_executor(
                None, _load_one, eid,
            )
            if loaded is None:
                continue
            loaded = _to_viewport_frame(loaded, design)
            designs_meshes[eid] = {
                "mesh_nodes": loaded["nodes"],
                "mesh_elems": loaded["elems"],
                "mesh_tags": loaded["tags"],
                "mesh_q": loaded["q_tet"],
                "region_surfaces": loaded["region_surfaces"],
                "region_surfaces_viz": _build_viz_surfaces(
                    loaded["region_surfaces"],
                ),
                "msh_path": loaded["msh_path"],
                "R_ci": (
                    float(design.get("R_ci_m") or 0.0)
                ),
                "R_co": (
                    float(design.get("R_co_m") or 0.0)
                ),
            }
            if eid == active_id:
                active_loaded = loaded

        if not designs_meshes:
            return False

        geom.designs_meshes = designs_meshes
        # M1.1: flag the freshly-loaded designs as has_mesh=True
        # so the legend exposes their tissue sub-rows. Designs
        # in state.designs that weren't loaded (no nerve.msh on
        # disk) keep their existing has_mesh value (typically
        # False).
        _loaded_eids = set(designs_meshes.keys())
        _ds_after = []
        for _d in (state.designs or []):
            if _d.get("eid") in _loaded_eids:
                _ds_after.append({**_d, "has_mesh": True})
            else:
                _ds_after.append(_d)
        state.designs = _ds_after

        # Single-slot back-compat fields: populate from the active
        # design's data (or the first one if active isn't present).
        if active_loaded is None:
            _first_eid = next(iter(designs_meshes))
            _ad = designs_meshes[_first_eid]
            geom.msh_path = _ad["msh_path"]
            geom.mesh_nodes = _ad["mesh_nodes"]
            geom.mesh_elems = _ad["mesh_elems"]
            geom.mesh_tags = _ad["mesh_tags"]
            geom.mesh_q = _ad["mesh_q"]
            geom.region_surfaces = _ad["region_surfaces"]
            geom.region_surfaces_viz = _ad["region_surfaces_viz"]
            geom.R_ci = _ad["R_ci"]
            geom.R_co = _ad["R_co"]
            # Use whatever was returned for the first design's
            # stats — we don't keep them per-design.
            state.mesh_stats_html = ""
            # Empty plotly stub — `None` makes the trame Figure
            # widget read `null.data` on the client and the error
            # bubbles up through Vue, blanking the rest of the
            # Mesh drawer body.
            state.mesh_quality_hist_figure = {
                "data": [], "layout": {},
            }
        else:
            geom.msh_path = active_loaded["msh_path"]
            geom.mesh_nodes = active_loaded["nodes"]
            geom.mesh_elems = active_loaded["elems"]
            geom.mesh_tags = active_loaded["tags"]
            geom.mesh_q = active_loaded["q_tet"]
            geom.region_surfaces = active_loaded["region_surfaces"]
            geom.region_surfaces_viz = _build_viz_surfaces(
                active_loaded["region_surfaces"],
            )
            geom.R_ci = float(
                _find_design(active_id) and
                _find_design(active_id).get("R_ci_m") or 0.0,
            )
            geom.R_co = float(
                _find_design(active_id) and
                _find_design(active_id).get("R_co_m") or 0.0,
            )
            state.mesh_stats_html = active_loaded.get(
                "stats_html", "",
            )
            # Fall back to the empty plotly stub if `hist_fig`
            # is missing or `None` — plotly throws on null.data.
            state.mesh_quality_hist_figure = (
                active_loaded.get("hist_fig")
                or {"data": [], "layout": {}}
            )
        state.has_mesh = True
        _request_render()
        state.mesh_log = (
            f"Restored {len(designs_meshes)} design "
            f"mesh{'es' if len(designs_meshes) != 1 else ''} "
            f"from disk."
        )
        return True

    def _restore_fem_from_disk() -> bool:
        """Load axis_line.npz + slice_volume.npz + the Vₑ overlays
        from the active project. No subprocess — purely re-reading
        cached results so the analysis plots come back without
        solving.

        F3.2c: outputs now live under `<out>/configs/<cid>/`
        (per-config layout — one solve per polarity wiring on a
        given design's mesh). Falls back to the F3.2a per-design
        layout (`<out>/designs/<eid>/`) and the legacy flat root
        for older projects."""
        # Prefer the per-config layout (F3.2c).
        configs_meta = _fem_layout.enumerate_configs(GOLGI_OUT)
        if configs_meta:
            _known = {c["id"] for c in configs_meta}
            _prev_cid = str(
                getattr(state, "active_config_id", "") or "",
            )
            active_cid = (
                _prev_cid if _prev_cid in _known
                else configs_meta[0]["id"]
            )
            design_dir = _fem_layout.config_dir(
                GOLGI_OUT, active_cid,
            )
            axis_path = design_dir / "axis_line.npz"
            slice_path = design_dir / "slice_volume.npz"
            if not axis_path.exists() or not slice_path.exists():
                return False
            try:
                state.fem_configs = list(configs_meta)
                state.active_config_id = active_cid
                # Keep active_design_id in sync with the config's
                # parent so the design-side selectors track.
                parent_eid = next(
                    (c["design_id"] for c in configs_meta
                     if c["id"] == active_cid),
                    "",
                )
                if parent_eid:
                    state.active_design_id = parent_eid
            except Exception:
                pass
            # Fall through to the rest of the restore using the
            # config dir as the source (axis_path/slice_path were
            # computed above; the inner try/except below loads
            # from it).
        else:
            # F3.2a fallback: per-design FEM outputs.
            designs = _fem_layout.enumerate_designs(GOLGI_OUT)
            if not designs:
                return False
            _known = {d["id"] for d in designs}
            _prev = str(
                getattr(state, "active_design_id", "") or "",
            )
            active_id = (
                _prev if _prev in _known else designs[0]["id"]
            )
            design_dir = _fem_layout.design_dir(
                GOLGI_OUT, active_id,
            )
            axis_path = design_dir / "axis_line.npz"
            slice_path = design_dir / "slice_volume.npz"
            if not axis_path.exists() or not slice_path.exists():
                return False
        try:
            if not configs_meta:
                state.fem_designs = list(designs)
                state.active_design_id = active_id
            geom.fem_axis = np.load(axis_path, allow_pickle=True)
            geom.fem_slice = np.load(slice_path, allow_pickle=True)
            paths_ve_path = design_dir / "paths_Ve.npz"
            if paths_ve_path.exists():
                try:
                    pvz = np.load(paths_ve_path, allow_pickle=True)
                    pv_lens = np.asarray(pvz["path_lengths"])
                    pv_Ve = np.asarray(pvz["Ve_flat"])
                    pv_Ez = (np.asarray(pvz["Ez_flat"])
                             if "Ez_flat" in pvz.files else None)
                    pv_flat = (np.asarray(pvz["paths_flat"])
                               if "paths_flat" in pvz.files
                               else None)
                    paths_Ve: list = []
                    paths_Ez: list = []
                    paths_xyz: list = []
                    _off = 0
                    for L in pv_lens:
                        _n = int(L)
                        paths_Ve.append(pv_Ve[_off:_off + _n].copy())
                        if pv_Ez is not None:
                            paths_Ez.append(
                                pv_Ez[_off:_off + _n].copy(),
                            )
                        if pv_flat is not None:
                            paths_xyz.append(
                                pv_flat[_off:_off + _n].copy(),
                            )
                        _off += _n
                    geom.fiber_paths_Ve = paths_Ve
                    geom.fiber_paths_Ez = (
                        paths_Ez if pv_Ez is not None else None
                    )
                    geom.fiber_paths_for_Ve = (
                        paths_xyz if pv_flat is not None else None
                    )
                except Exception:
                    geom.fiber_paths_Ve = None
                    geom.fiber_paths_Ez = None
                    geom.fiber_paths_for_Ve = None
            else:
                geom.fiber_paths_Ve = None
                geom.fiber_paths_Ez = None
                geom.fiber_paths_for_Ve = None
            nsv_path = design_dir / "nerve_surface_Ve.npz"
            if nsv_path.exists():
                try:
                    nsvd = np.load(nsv_path, allow_pickle=True)
                    geom.nerve_surface_Ve = np.asarray(
                        nsvd["Ve"], dtype=np.float64,
                    )
                except Exception:
                    geom.nerve_surface_Ve = None
            else:
                geom.nerve_surface_Ve = None
            _refresh_fem_plots()
            state.has_fem = True
            state.fem_status = (
                f"✓ Restored cached FEM — "
                f"{len(geom.fem_axis['z'])} axis pts, "
                f"{geom.fem_slice['Ve'].shape[0]} slices"
            )
            # F3.2 fix: when the user switches the active config in
            # the analysis chip, this restore reloads
            # geom.fiber_paths_Ve / geom.nerve_surface_Ve /
            # geom.fem_slice into geom for the NEW config — but
            # those arrays feed the 3D scene-state pipeline (Vₑ-on-
            # fibers tubes, Vₑ-on-surface heatmap, E-field
            # streamlines). Without an explicit render request,
            # the actors keep showing the PREVIOUS config's data
            # because `state.show_ve_fibers` / `_surface` /
            # `field_lines` didn't toggle and no watcher fired. The
            # FEM-solve completion path triggers a render via the
            # auto-enable assignment to show_*; the restore path
            # doesn't, so we do it explicitly here.
            _request_render()
            return True
        except Exception as ex:
            print(f"[project] FEM restore failed: {ex}", flush=True)
            state.fem_status = (
                f"⚠ failed to restore cached FEM: "
                f"{type(ex).__name__}: {ex}"
            )
            return False

    @state.change("selected_design_id")
    def _on_selected_design_mesh_swap(selected_design_id, **_kw):
        """F3.2-M1: under per-design simultaneous rendering, the
        viewport already shows every design's mesh — there's
        nothing to swap. We just (a) update the single-slot geom
        fields so legacy callers see the focused design's data,
        and (b) mirror selected_design_id → active_design_id so
        the analysis-side FEM / sim restores follow the focus.

        If the focused design has no mesh in `designs_meshes`
        (user just added it and hasn't meshed yet), we fall back
        to disk for THAT design only — preserves the "click new
        design then click Build Mesh" UX.

        Skipped when no project is open (project-load
        initialises selected_design_id mid-restore, and the
        mesh-restore step runs explicitly later in
        do_open_project)."""
        if not state.has_active_project:
            return
        if not selected_design_id:
            return
        # SceneCatalog Phase 4 fix — also skip when the source
        # nerve hasn't been loaded yet. Without geom.centroid +
        # R_global, `_restore_mesh_from_disk`'s back-transform
        # short-circuits and stashes design-local data into
        # `geom.designs_meshes`. This watcher fires DURING
        # project-load (when `_apply_ui_state` restores
        # `selected_design_id` before step 2 loads the nerve),
        # so without this guard the mesh ends up in the wrong
        # frame. do_open_project's explicit step 4 mesh-restore
        # runs after the nerve is loaded and does the right
        # thing.
        if geom.nerve is None or geom.centroid is None:
            return
        sid = str(selected_design_id)
        if str(state.active_design_id or "") != sid:
            state.active_design_id = sid
        # Cheap path: mesh already loaded into the per-design
        # dict. Just point single-slot fields at it + render.
        designs_meshes = geom.designs_meshes or {}
        if sid in designs_meshes:
            d = designs_meshes[sid]
            geom.msh_path = d.get("msh_path")
            geom.mesh_nodes = d.get("mesh_nodes")
            geom.mesh_elems = d.get("mesh_elems")
            geom.mesh_tags = d.get("mesh_tags")
            geom.mesh_q = d.get("mesh_q")
            geom.region_surfaces = d.get("region_surfaces")
            geom.region_surfaces_viz = d.get(
                "region_surfaces_viz",
            )
            geom.R_ci = float(d.get("R_ci") or 0.0)
            geom.R_co = float(d.get("R_co") or 0.0)
            state.has_mesh = True
            _request_render()
            return
        # Fall back: design isn't loaded yet (e.g. newly added
        # then user clicked another, then back). Re-run the full
        # restore which loads every design from disk.
        async def _swap():
            try:
                ok = await _restore_mesh_from_disk()
            except Exception as ex:
                print(
                    f"[design] mesh switch failed for "
                    f"'{selected_design_id}': {ex}",
                    flush=True,
                )
                return
            if not ok:
                state.has_mesh = False
                geom.msh_path = None
        try:
            asyncio.create_task(_swap())
        except RuntimeError:
            # No running loop (e.g. unit tests) — best-effort skip.
            pass
        # Phase 6b — re-evaluate pre-mesh previews + the catalog's
        # nerve fold when the focused design changes. The previews
        # only watch their own vis_* params, not
        # `selected_design_id`, so without this kick the user sees
        # stale visibility after switching from a meshed to an
        # unmeshed design (raw nerve / muscle bbox / epi shell
        # should reappear for the unmeshed design as a placement
        # backdrop).
        try:
            if (geom.nerve is not None
                    and bool(state.vis_muscle_preview)):
                _update_muscle_preview()
        except Exception:
            pass
        try:
            if (geom.nerve is not None
                    and bool(state.vis_epi_preview)):
                _update_epi_preview()
        except Exception:
            pass
        try:
            if (geom.nerve is None
                    or not _focused_design_has_mesh()):
                # Force off when focused design isn't meshed.
                # `_update_*_preview` short-circuits on
                # `_focused_design_has_mesh()` so a no-op call
                # also removes when needed.
                pass
        except Exception:
            pass

    @state.change("designs")
    def _on_designs_default_select(designs, **_kw):
        """M2.0.1: keep `selected_design_id` valid after design
        adds / removes. The legend's design-picker combobox is
        bound directly to `selected_design_id`; an empty or
        stale value would leave the legend tree empty (no
        v_for row matches `elec.eid === selected_design_id`).
        Auto-pick the first available design when the current
        selection is no longer valid."""
        if not state.has_active_project:
            return
        designs = list(designs or [])
        valid_eids = {
            str(d.get("eid", "")) for d in designs if d.get("eid")
        }
        current = str(state.selected_design_id or "")
        if current in valid_eids:
            return
        if valid_eids:
            state.selected_design_id = next(iter(valid_eids))

    # R1.4 — montage selector reactivity. Changing the dropdown
    # rebuilds the figure from the per-montage dict on geom
    # (populated by the fiber_sim / pop_sim drivers), without
    # re-running any simulation.

    @state.change("active_montage_single")
    def _on_active_montage_single_change(
        active_montage_single, **_kw,
    ):
        if not state.has_active_project:
            return
        cnap = getattr(geom, "cnap_single", None)
        if not cnap:
            return
        # Resolve montage meta from the active config so the
        # figure picks up the right color + label.
        montages: list = []
        cid = str(
            state.active_config_id
            or state.selected_config_id
            or "",
        )
        for c in (state.configs or []):
            if str(c.get("cid", "")) == cid:
                montages = list(
                    c.get("recording_montages") or [],
                )
                break
        fiber_label = ""
        try:
            sel = int(state.fiber_sel_idx)
            lab, _color = _fiber_label_and_color(sel)
            fiber_label = str(lab)
        except Exception:                                    # noqa: BLE001
            pass
        state.fiber_cnap_figure = build_fiber_cnap_figure(
            cnap_by_montage=cnap,
            montage_meta=montages,
            active_mid=str(active_montage_single or ""),
            fiber_label=fiber_label,
        )

    @state.change(
        "active_montage_pop", "cnap_decompose_by_type",
    )
    def _on_pop_cnap_view_change(
        active_montage_pop, cnap_decompose_by_type, **_kw,
    ):
        if not state.has_active_project:
            return
        cnap = getattr(geom, "cnap_pop", None)
        if not cnap:
            return
        montages: list = []
        cid = str(
            state.active_config_id
            or state.selected_config_id
            or "",
        )
        for c in (state.configs or []):
            if str(c.get("cid", "")) == cid:
                montages = list(
                    c.get("recording_montages") or [],
                )
                break
        state.pop_cnap_figure = build_pop_cnap_figure(
            cnap_by_montage=cnap,
            montage_meta=montages,
            active_mid=str(active_montage_pop or ""),
            decompose_by_type=bool(cnap_decompose_by_type),
        )

    @state.change("active_design_id")
    def _on_active_design_change(active_design_id, **_kw):
        """User picked a different design in the analysis-drawer
        switcher — re-load that design's cached FEM outputs into
        geom and the analysis plots. Skipped when no project is
        open (project-load also pokes this key as it initialises
        the switcher; _restore_fem_from_disk reads the new
        value).

        F3.2c: sim + pop restores moved to the
        `active_config_id` watcher (sims are now per-config, and
        a design swap cascades through selected_config_id →
        active_config_id → that watcher)."""
        if not state.has_active_project:
            return
        if not active_design_id:
            return
        try:
            _restore_fem_from_disk()
        except Exception as ex:
            print(
                f"[design] FEM switch failed for "
                f"'{active_design_id}': {ex}",
                flush=True,
            )

    @state.change("selected_config_id")
    def _on_selected_config_mirror(selected_config_id, **_kw):
        """F3.2c: keep `active_config_id` (analysis chip / FEM-
        restore target) in sync with `selected_config_id` (the
        config highlighted in the Designs-drawer configs panel).
        That way picking a different design → load_design_to_
        selected swaps selected_config_id → this watcher mirrors
        it to active_config_id → the FEM + sim restores fire."""
        if not state.has_active_project:
            return
        cid = str(selected_config_id or "")
        if cid and str(state.active_config_id or "") != cid:
            state.active_config_id = cid

    @state.change("configs", "designs", "fem_configs")
    def _on_config_items_rebuild(**_kw):
        """Precompute the three VSelect items lists (Solve
        multi-select, analysis chip, Compare multi-select)
        whenever configs / designs / fem_configs change.

        Why precompute rather than inline a `.map()` JS
        expression on the VSelect `items` prop: that expression
        produces a new array reference on every render, which
        confuses Vuetify's internal multi-select selection
        tracking and silently drops ticks. Reading from a stable
        state-bound list fixes that."""
        all_configs = list(state.configs or [])
        all_designs = list(state.designs or [])
        designs_by_eid = {
            str(d.get("eid", "")): d for d in all_designs
        }

        def _label(c: dict, design_name_field: str) -> str:
            cfg_name = str(c.get("name", c.get("cid", "")))
            d_name = str(c.get(design_name_field, "") or "")
            if not d_name:
                d_eid = str(c.get("design_id", ""))
                d_name = str(
                    designs_by_eid.get(d_eid, {})
                    .get("name", d_eid),
                )
            return f"{d_name} · {cfg_name}" if d_name else cfg_name

        # Solve drawer + Compare panel read from `state.configs`
        # (every config the user has saved, whether solved or
        # not — Solve drawer drives them through solve_nerve).
        state.solve_config_items = [
            {
                "value": str(c.get("cid", "")),
                "title": _label(c, "design_name"),
            }
            for c in all_configs
            if c.get("cid")
        ]
        # Analysis chip reads from `state.fem_configs` — only
        # configs that completed a solve and have outputs on
        # disk. The schema differs (uses `id`, not `cid`, and
        # has `design_name` populated by the FEM driver).
        fem_cfgs = list(state.fem_configs or [])
        state.fem_config_items = [
            {
                "value": str(c.get("id", "")),
                "title": _label(c, "design_name"),
            }
            for c in fem_cfgs
            if c.get("id")
        ]
        # Compare-view multi-select also reads from fem_configs
        # (you can only compare solved configs).
        state.compare_config_items = list(state.fem_config_items)
        # Unified design+config picker (replaces the analysis-grid
        # "Config" chip). Each design contributes ONE entry if it
        # has 0 or 1 solved configs (label is just the design
        # name); 2+ configs split into one entry per config so
        # the user can pick which solved result drives the
        # overlays. Value encoding: "<eid>" or "<eid>|<cid>".
        _fem_by_design: dict[str, list[dict]] = {}
        for c in fem_cfgs:
            did = str(c.get("design_id", ""))
            if c.get("id") and did:
                _fem_by_design.setdefault(did, []).append(c)
        _dc_items: list[dict] = []
        for d in all_designs:
            eid = str(d.get("eid", ""))
            if not eid:
                continue
            d_name = str(d.get("name", "") or eid)
            d_cfgs = _fem_by_design.get(eid, [])
            if len(d_cfgs) >= 2:
                for c in d_cfgs:
                    cid = str(c.get("id", ""))
                    cname = str(c.get("name", cid) or cid)
                    _dc_items.append({
                        "value": f"{eid}|{cid}",
                        "title": f"{d_name} · {cname}",
                    })
            else:
                _dc_items.append({
                    "value": eid,
                    "title": d_name,
                })
        state.design_config_items = _dc_items
        # M2.0: per-design has_fem flag — True iff at least one
        # solved config has this design as its parent. Gates the
        # Overlays sub-section in the per-design legend tree.
        # Mutate via new dicts so Vue picks up row-level changes.
        _eids_with_fem = {
            str(c.get("design_id", ""))
            for c in fem_cfgs
            if c.get("id") and c.get("design_id")
        }
        _ds_after = []
        _any_change = False
        for _d in all_designs:
            _eid = str(_d.get("eid", ""))
            _want = bool(_eid and _eid in _eids_with_fem)
            if bool(_d.get("has_fem", False)) != _want:
                _ds_after.append({**_d, "has_fem": _want})
                _any_change = True
            else:
                _ds_after.append(_d)
        if _any_change:
            state.designs = _ds_after

    @state.change(
        "compare_config_selection",
        "compare_slice_z_idx",
        "fem_configs",
        # F3.2 — selectivity inputs also drive the same watcher
        # so the user can adjust target / off-target / amplitude
        # without re-picking configs.
        "selectivity_target_branch",
        "selectivity_offtarget_branches",
        "selectivity_amplitude_mA",
        # I1 Phase A — impedance tile in the Compare panel.
        # `fem_impedance` updates after every FEM solve; the
        # watcher refreshes the bar charts from the cached
        # per-cid data without re-reading impedance.json.
        "fem_impedance",
    )
    def _on_compare_change(**_kw):
        """F3.2e: rebuild the Compare-view figures whenever the
        user changes which configs to overlay, the z-slice index,
        or the FEM-configs list (e.g., after a fresh solve).

        Pure read-back from <out>/configs/<cid>/ — no Python
        objects in geom needed, so the watcher is cheap even
        when toggling many configs."""
        if not state.has_active_project:
            return
        try:
            picked = list(state.compare_config_selection or [])
        except Exception:
            picked = []
        # Default to comparing every solved config when the user
        # hasn't picked any — surfaces the panel as "useful by
        # default" so first-time users see something immediately.
        if not picked:
            picked = [
                str(c.get("id", ""))
                for c in (state.fem_configs or [])
                if c.get("id")
            ]
        configs_meta = list(state.fem_configs or [])
        try:
            state.compare_axis_figure = (
                build_compare_axis_figure(
                    GOLGI_OUT, picked, configs_meta,
                )
            )
        except Exception as ex:
            print(
                f"[compare] axis figure rebuild failed: {ex}",
                flush=True,
            )
        try:
            state.compare_slice_grid_figure = (
                build_compare_slice_grid(
                    GOLGI_OUT, picked, configs_meta,
                    int(state.compare_slice_z_idx or 0),
                )
            )
        except Exception as ex:
            print(
                f"[compare] slice grid rebuild failed: {ex}",
                flush=True,
            )
        # F3.2 — selectivity. Load each picked config's per-cid
        # sweep result via `sweep_cache.load_latest_for_config`,
        # union the available branches into the picker, compute
        # SI bar + threshold ratio table.
        try:
            from golgi.projects import sweep_cache as _swc
            _per_cfg_sweeps: dict[str, object] = {}
            _all_branch_ids: set[int] = set()
            for _cid in picked:
                _sw = _swc.load_latest_for_config(
                    GOLGI_OUT, _cid,
                )
                if _sw is None:
                    continue
                _per_cfg_sweeps[_cid] = _sw
                try:
                    _all_branch_ids.update(
                        branch_ids_present(
                            _sw.fiber_branch_idx,
                        ),
                    )
                except Exception:
                    pass
            # Push branch items so the picker shows what's
            # available across the loaded sweeps. Labels use the
            # user-renamed branch names when set, else "Branch N".
            _branch_items: list[dict] = []
            for _bid in sorted(_all_branch_ids):
                _name = str(
                    getattr(
                        state,
                        f"fiber_branch_name_{_bid}",
                        "",
                    ) or ""
                ).strip()
                _branch_items.append({
                    "value": str(_bid),
                    "title": _name or f"Branch {_bid}",
                })
            state.selectivity_branch_items = _branch_items
            # Default target to branch 0 (or first available) if
            # the user hasn't picked. Off-target defaults to the
            # union of "all other branches" — handled below by
            # the empty-list short-circuit in the math.
            _target_raw = str(
                state.selectivity_target_branch or "",
            )
            if not _target_raw and _branch_items:
                _target_raw = _branch_items[0]["value"]
                state.selectivity_target_branch = _target_raw
            try:
                _target_branch = int(_target_raw)
            except (TypeError, ValueError):
                _target_branch = -1
            _off_picks = list(
                state.selectivity_offtarget_branches or [],
            )
            _off_branches: list[int] | None
            if not _off_picks:
                _off_branches = None  # = all-others sentinel
            else:
                _off_branches = []
                for _v in _off_picks:
                    try:
                        _off_branches.append(int(_v))
                    except (TypeError, ValueError):
                        pass
            # Per-config SI at the chosen amplitude (for bar).
            try:
                _amp_target = float(
                    state.selectivity_amplitude_mA or 0.0,
                )
            except (TypeError, ValueError):
                _amp_target = 0.0
            _per_cfg_si: dict[str, dict] = {}
            _per_cfg_thr: dict[str, dict] = {}
            _meta_by_cid = {
                str(c.get("id", "")): c
                for c in configs_meta if c.get("id")
            }
            for _cid, _sw in _per_cfg_sweeps.items():
                _label = (
                    str(_meta_by_cid.get(_cid, {})
                        .get("title")
                        or _meta_by_cid.get(_cid, {})
                        .get("design_name", _cid))
                )
                if _target_branch < 0:
                    continue
                # Recruitment-mode SI bar.
                if _sw.activated is not None:
                    # Amplitudes axis is implicit in the request.
                    try:
                        _amps = np.asarray(
                            _sw.request.amplitudes_mA,
                            dtype=np.float64,
                        )
                    except Exception:
                        _amps = np.zeros(0)
                    if _amps.size:
                        _si_curve = compute_veraart_si(
                            _sw.activated,
                            _sw.fiber_branch_idx,
                            target_branch=_target_branch,
                            offtarget_branches=_off_branches,
                        )
                        # Nearest-amp lookup.
                        _idx = int(
                            np.argmin(
                                np.abs(_amps - _amp_target),
                            ),
                        )
                        _si_val = float(_si_curve[_idx])
                        _per_branch = (
                            compute_branch_recruitment(
                                _sw.activated,
                                _sw.fiber_branch_idx,
                            )
                        )
                        _R_t = float(
                            _per_branch.get(
                                _target_branch,
                                np.zeros_like(_amps),
                            )[_idx],
                        )
                        _R_o_arr = (
                            _si_curve.copy()
                        )  # not used; recompute properly:
                        _bidx_arr = np.asarray(
                            _sw.fiber_branch_idx,
                            dtype=np.int64,
                        )
                        if _off_branches is None:
                            _off_mask = (
                                _bidx_arr != _target_branch
                            )
                        else:
                            _off_mask = np.isin(
                                _bidx_arr, _off_branches,
                            )
                        if _off_mask.any():
                            _R_o = float(
                                np.asarray(
                                    _sw.activated,
                                    dtype=np.bool_,
                                )[_off_mask]
                                .mean(axis=0)[_idx]
                            )
                        else:
                            _R_o = 0.0
                        _per_cfg_si[_cid] = {
                            "label": _label,
                            "si": _si_val,
                            "R_target": _R_t,
                            "R_offtarget": _R_o,
                        }
                # Threshold-mode ratio (works with thresholds_uA
                # if present, regardless of recruitment payload).
                if _sw.thresholds_uA is not None:
                    _stats = (
                        compute_threshold_stats_per_branch(
                            _sw.thresholds_uA,
                            _sw.fiber_branch_idx,
                        )
                    )
                    _t_stats = _stats.get(
                        _target_branch,
                        {},
                    )
                    _T_target = float(
                        _t_stats.get("median", float("nan")),
                    )
                    _n_target = int(
                        _t_stats.get("n_activated", 0),
                    )
                    _ratio = compute_threshold_ratio(
                        _sw.thresholds_uA,
                        _sw.fiber_branch_idx,
                        target_branch=_target_branch,
                        offtarget_branches=_off_branches,
                    )
                    # Off-target median for display.
                    _bidx_arr = np.asarray(
                        _sw.fiber_branch_idx, dtype=np.int64,
                    )
                    if _off_branches is None:
                        _off_mask = (
                            _bidx_arr != _target_branch
                        )
                    else:
                        _off_mask = np.isin(
                            _bidx_arr, _off_branches,
                        )
                    _thr_arr = np.asarray(
                        _sw.thresholds_uA, dtype=np.float64,
                    )
                    _off_pool = _thr_arr[_off_mask]
                    _off_good = np.isfinite(_off_pool)
                    if _off_good.any():
                        _T_off = float(
                            np.median(_off_pool[_off_good]),
                        )
                        _n_off = int(_off_good.sum())
                    else:
                        _T_off = float("inf")
                        _n_off = 0
                    _per_cfg_thr[_cid] = {
                        "label": _label,
                        "T_target_uA": _T_target,
                        "T_offtarget_uA": _T_off,
                        "ratio": _ratio,
                        "n_target": _n_target,
                        "n_offtarget": _n_off,
                    }
            # Render figures.
            _target_label = next(
                (it["title"] for it in _branch_items
                 if it["value"] == str(_target_branch)),
                f"branch {_target_branch}",
            )
            _off_label = (
                "all other branches"
                if _off_branches is None
                else "selected branches"
            )
            if _per_cfg_si:
                state.selectivity_bar_figure = (
                    build_selectivity_bar_figure(
                        _per_cfg_si,
                        target_branch_label=_target_label,
                        offtarget_label=_off_label,
                        amplitude_mA=_amp_target,
                    )
                )
            else:
                state.selectivity_bar_figure = {
                    "data": [], "layout": {},
                }
            if _per_cfg_thr:
                state.selectivity_table_html = (
                    build_threshold_ratio_table(
                        _per_cfg_thr,
                        target_branch_label=_target_label,
                        offtarget_label=_off_label,
                    )
                )
            else:
                state.selectivity_table_html = (
                    "<em style='color:#888; "
                    "font-size:12px;'>"
                    "Run a threshold-mode sweep on each "
                    "config to populate this table."
                    "</em>"
                )
            # Status line — counts of loaded configs + branches.
            _n_loaded = len(_per_cfg_sweeps)
            _n_picked = len(picked)
            if _n_loaded == 0 and _n_picked > 0:
                state.selectivity_status = (
                    f"⚠ No per-config sweep results found for "
                    f"the {_n_picked} picked config(s). Run "
                    f"the sweep tab on each one first."
                )
            elif _n_loaded < _n_picked:
                state.selectivity_status = (
                    f"Loaded sweeps for {_n_loaded} of "
                    f"{_n_picked} picked configs. Missing "
                    f"configs will be skipped in the metrics."
                )
            else:
                state.selectivity_status = ""
        except Exception as ex:
            print(
                f"[compare] selectivity rebuild failed: {ex}",
                flush=True,
            )
        # I1 Phase A — Impedance bars from cached state.fem_impedance.
        # Fast: no disk reads. Builds two figures (per-contact +
        # per-pair) keyed by the picked configs.
        try:
            _imp_all = dict(
                getattr(state, "fem_impedance", {}) or {},
            )
            _per_cfg_imp: dict[str, dict] = {}
            for _cid in picked:
                _imp = _imp_all.get(_cid)
                if not isinstance(_imp, dict):
                    continue
                _label = str(
                    _meta_by_cid.get(_cid, {}).get("title")
                    or _meta_by_cid.get(_cid, {}).get(
                        "design_name", _cid,
                    )
                )
                _per_cfg_imp[_cid] = {
                    "label": _label,
                    "per_contact": list(
                        _imp.get("per_contact", []) or [],
                    ),
                    "per_pair": list(
                        _imp.get("per_pair", []) or [],
                    ),
                }
            state.impedance_bar_figure = (
                build_impedance_bar_figure(_per_cfg_imp)
            )
            state.impedance_per_pair_figure = (
                build_impedance_per_pair_figure(_per_cfg_imp)
            )
        except Exception as ex:                          # noqa: BLE001
            print(
                f"[compare] impedance figure rebuild failed: "
                f"{ex}",
                flush=True,
            )

    @state.change("fem_impedance", "active_config_id")
    def _on_fem_impedance_chips_change(**_kw):
        """I1 Phase A — rebuild the Solve-drawer summary chips
        from `state.fem_impedance[active_config_id]`. The chips
        give the user a glanceable Z summary right under the FEM
        status banner; the full bar charts live in the Compare
        panel."""
        cid = str(getattr(state, "active_config_id", "") or "")
        _imp = (
            getattr(state, "fem_impedance", {}) or {}
        ).get(cid)
        if not isinstance(_imp, dict):
            state.fem_impedance_chips_contact = []
            state.fem_impedance_chips_pair = []
            state.fem_impedance_chips_meta = ""
            return
        # Role-tinted background so cathode/anode chips read at a
        # glance — same palette as the polarity dots in the
        # electrodes drawer.
        _role_color = {
            "cathode": "#fde9ec",
            "anode":   "#e6f0ff",
            "":        "#f1f3f5",
        }
        _contact_chips = []
        for _row in (_imp.get("per_contact", []) or []):
            try:
                _z = float(_row.get("Z_ohm", float("nan")))
            except (TypeError, ValueError):
                _z = float("nan")
            _role = str(_row.get("role", "") or "")
            _contact_chips.append({
                "id": int(_row.get("id", 0)),
                "role": _role,
                "z_fmt": _fmt_ohms_imp(_z),
                "bg": _role_color.get(_role, "#f1f3f5"),
            })
        _pair_chips = []
        for _row in (_imp.get("per_pair", []) or []):
            try:
                _z = float(
                    _row.get("Z_pair_ohm", float("nan")),
                )
            except (TypeError, ValueError):
                _z = float("nan")
            _pair_chips.append({
                "anode": int(_row.get("anode", 0)),
                "cathode": int(_row.get("cathode", 0)),
                "z_fmt": _fmt_ohms_imp(_z),
            })
        state.fem_impedance_chips_contact = _contact_chips
        state.fem_impedance_chips_pair = _pair_chips
        # Meta line — ground strategy + frequency, mostly for the
        # tooltip / footer. DC for Phase A; Phase B fills this
        # with the swept frequency.
        _f = _imp.get("frequency_hz", 0.0)
        _gs = str(_imp.get("ground_strategy", "") or "")
        try:
            _f_val = float(_f)
        except (TypeError, ValueError):
            _f_val = 0.0
        _freq_str = "DC" if _f_val == 0.0 else f"{_f_val:.1f} Hz"
        state.fem_impedance_chips_meta = (
            f"{_freq_str} · ground: {_gs}" if _gs else _freq_str
        )

    @state.change("design_config_key")
    def _on_design_config_key_change(design_config_key, **_kw):
        """Forward the unified picker's value into the underlying
        `selected_design_id` + `active_config_id` state vars.

        Encoding: "<eid>" (design-only) or "<eid>|<cid>"
        (design + specific solved config). Splitting it here lets
        the rest of the app keep using the two original state vars
        unchanged — only the legend toprow combobox writes to
        `design_config_key`."""
        key = str(design_config_key or "")
        if not key:
            return
        if "|" in key:
            eid, cid = key.split("|", 1)
        else:
            eid, cid = key, ""
        if eid and str(state.selected_design_id or "") != eid:
            state.selected_design_id = eid
        if cid and str(state.active_config_id or "") != cid:
            state.active_config_id = cid

    @state.change(
        "selected_design_id", "active_config_id",
        "design_config_items",
    )
    def _on_design_config_key_resync(**_kw):
        """Reverse-sync: when other code paths change
        `selected_design_id` or `active_config_id` (or the items
        list rebuilds because designs / fem_configs changed), pick
        the matching item value so the combobox highlight follows.

        Prefer the "<eid>|<cid>" entry when both are set and that
        compound value exists in the items list; fall back to the
        bare "<eid>" entry otherwise."""
        eid = str(state.selected_design_id or "")
        cid = str(state.active_config_id or "")
        items = state.design_config_items or []
        want = f"{eid}|{cid}" if (eid and cid) else eid
        # Try exact match first (compound or bare).
        for it in items:
            if str(it.get("value", "")) == want:
                if str(state.design_config_key or "") != want:
                    state.design_config_key = want
                return
        # Fall back to bare eid if the compound isn't in the list
        # (e.g., the active_config_id is stale or its design has
        # been collapsed back to the single-entry form).
        for it in items:
            if str(it.get("value", "")) == eid:
                if str(state.design_config_key or "") != eid:
                    state.design_config_key = eid
                return
        # No match — clear so the combobox shows blank rather
        # than a stale value Vuetify would silently drop.
        if state.design_config_key:
            state.design_config_key = ""

    @state.change("active_config_id")
    def _on_active_config_change(active_config_id, **_kw):
        """F3.2c: user picked a different config in the analysis
        chip — re-load that config's cached FEM outputs (axis,
        slice, paths_Ve, surface_Ve) into geom and refresh the
        analysis plots.

        Each config's outputs live under <out>/configs/<cid>/;
        switching is just a re-read of those files.

        F3.2c fix: if the new config's parent design differs
        from the currently-selected design, also flip
        selected_design_id — that cascades through the mesh-
        swap watcher so the viewport reloads the right mesh.
        Switching configs that share a parent → no remesh.
        Switching configs across parents → remesh."""
        if not state.has_active_project:
            return
        if not active_config_id:
            return
        # Find the parent design of the newly-active config.
        parent_eid = ""
        for c in (state.configs or []):
            if c.get("cid") == active_config_id:
                parent_eid = str(c.get("design_id", ""))
                break
        if (parent_eid
                and parent_eid != str(
                    state.selected_design_id or "",
                )):
            # Triggers _on_selected_design_mesh_swap which
            # reloads the parent design's nerve.msh into geom.
            state.selected_design_id = parent_eid
        try:
            _restore_fem_from_disk()
        except Exception as ex:
            print(
                f"[config] FEM switch failed for "
                f"'{active_config_id}': {ex}",
                flush=True,
            )
        # F3.2c: sim/pop caches are now per-CONFIG (live under
        # <out>/sims/<cid>/). Bring them along on the config
        # switch so the fiber + population panels stay in sync.
        try:
            _restore_fiber_sim_cache()
        except Exception as ex:
            print(
                f"[config] fiber-sim switch failed for "
                f"'{active_config_id}': {ex}",
                flush=True,
            )
        try:
            _restore_pop_state()
        except Exception as ex:
            print(
                f"[config] pop-state switch failed for "
                f"'{active_config_id}': {ex}",
                flush=True,
            )

    def _restore_fibers_from_disk() -> bool:
        """Reload nerve_paths_fibers.npz + reclassify branches +
        re-mount the polylines as actors. Same code-path as the
        success branch at the end of do_generate_fibers but
        without running the subprocess."""
        npz_path = GOLGI_OUT / "nerve_paths_fibers.npz"
        caps_json = GOLGI_OUT / "nerve_paths_caps.json"
        if not npz_path.exists():
            return False
        try:
            d = np.load(npz_path, allow_pickle=True)
            flat = np.asarray(d["paths_flat"])
            lens = np.asarray(d["path_lengths"])
            paths_raw: list[np.ndarray] = []
            off = 0
            for L in lens:
                paths_raw.append(flat[off:off + int(L)].copy())
                off += int(L)
            seed_end_key = (
                "low" if str(state.fiber_seed_end).startswith("trunk")
                else "high"
            )
            branch_idx, n_branches = _classify_fibers_by_branch(
                paths_raw, caps_json, seed_end_key,
            )
            geom.fiber_paths_raw = paths_raw
            geom.fiber_branch_idx = branch_idx
            geom.fiber_n_branches = n_branches
            # Same flag check as the post-generate path: if a
            # previous FEM solve already migrated the file to
            # cuff frame, we honour that instead of double-
            # transforming on display.
            geom.fibers_in_cuff_frame = bool(
                "frame_is_cuff" in d.files
                and int(d["frame_is_cuff"]) == 1
            )
            state.fiber_n_branches = n_branches
            state.has_fibers = True
            _refresh_fiber_sel_items()
            _refresh_pop_branches_meta()
            _request_render()
            state.fiber_branch_summary = (
                _compute_fiber_branch_summary(
                    paths_raw, caps_json, seed_end_key,
                    branch_labels={
                        _i: str(
                            state[f"fiber_branch_name_{_i}"]
                            or ""
                        )
                        for _i in range(MAX_FIBER_BRANCHES)
                    },
                )
            )
            state.fiber_stats_html = ""
            state.fiber_status = (
                f"✓ Restored {len(paths_raw)} cached trajectories"
            )
            return True
        except Exception as ex:
            print(
                f"[project] fiber restore failed: {ex}", flush=True,
            )
            state.fiber_status = (
                f"⚠ failed to restore cached fibers: "
                f"{type(ex).__name__}: {ex}"
            )
            return False

    # ----------------------------------------------------------------
    # Persistence for sim results + population state.
    #
    # Both flows produce dicts of numpy arrays + Python scalars
    # that don't survive the manifest's JSON `ui_state` channel.
    # We dump them to pickle files inside GOLGI_OUT so the
    # project carries them across close/open cycles. The
    # manifest itself stays JSON-only.
    #
    # F3.1: sims are scoped to the currently-active electrode
    # design — pickles live under `<out>/sims/<design_id>/` so
    # re-solving a different cuff doesn't clobber the prior
    # sim's results.
    # ----------------------------------------------------------------
    def _active_sim_dir() -> Path:
        """Resolve the per-config sim directory for the currently
        active contact configuration. F3.2c: sims are keyed by
        config id (each polarity wiring on a given design's mesh
        produces its own Ve field, so the activation patterns
        differ → each config needs its own sim cache). Falls back
        to active_design_id and then "default" + legacy flat
        root for back-compat with pre-F3.2c projects."""
        cid = str(
            state.active_config_id
            or state.selected_config_id
            or "",
        )
        if not cid:
            # Pre-F3.2c projects didn't have configs at all; fall
            # back to the active design id so the legacy flat-
            # root detection in sim_dir kicks in.
            cid = (
                str(state.active_design_id or "") or "default"
            )
        return _fem_layout.sim_dir(GOLGI_OUT, cid)

    def _save_fiber_sim_cache() -> None:
        """Persist `geom.fiber_sim_results` to a pickle in the
        project dir. Called at the tail of every successful
        do_run_fiber_sim. Skipped silently if no project is
        active (script-style run / migration paths)."""
        if not state.has_active_project:
            return
        sim_dir = _active_sim_dir()
        if not geom.fiber_sim_results:
            # No results to persist — make sure any stale cache
            # is cleared so reopening doesn't surface ghosts.
            stale = sim_dir / "fiber_sim_results.pkl"
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass
            return
        sim_dir.mkdir(parents=True, exist_ok=True)
        try:
            with open(
                sim_dir / "fiber_sim_results.pkl", "wb",
            ) as f:
                pickle.dump({
                    "version": 1,
                    "results": geom.fiber_sim_results,
                    "view_idx": int(state.fiber_sel_idx),
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception as ex:
            print(
                f"[persist] fiber sim cache write failed: {ex}",
                flush=True,
            )

    def _restore_fiber_sim_cache() -> bool:
        """Reload single-fiber sim results from the project's
        pickle cache. Rebuilds `fiber_sim_results_meta` from
        scratch (label / colour from `_fiber_label_and_color`,
        `activated` from `spike_t`). Returns True when a cache
        existed and was applied."""
        fp = _active_sim_dir() / "fiber_sim_results.pkl"
        if not fp.exists():
            return False
        try:
            with open(fp, "rb") as f:
                payload = pickle.load(f)
            results = payload.get("results") or {}
            if not results:
                return False
            geom.fiber_sim_results = results
            meta: list[dict] = []
            for idx, sim_data in results.items():
                label, color = _fiber_label_and_color(int(idx))
                spike_t = sim_data.get("spike_t", [])
                activated = False
                try:
                    if isinstance(spike_t, np.ndarray):
                        activated = bool(spike_t.size > 0)
                    else:
                        activated = any(
                            (len(s) if hasattr(s, "__len__")
                             else 0) > 0
                            for s in spike_t
                        )
                except (TypeError, ValueError):
                    activated = False
                meta.append({
                    "idx": int(idx),
                    "label": label,
                    "color": color,
                    "ok": True,
                    "activated": activated,
                    "summary": "restored from cache",
                })
            state.fiber_sim_results_meta = meta
            view = int(
                payload.get("view_idx")
                or min(results.keys())
            )
            if view not in results:
                view = int(min(results.keys()))
            state.fiber_sel_idx = view
            sim_data = results[view]
            geom.fiber_sim_data = sim_data
            state.fiber_propagation_figure = (
                _build_fiber_propagation_figure(sim_data)
            )
            state.fiber_waterfall_figure = (
                _build_fiber_waterfall_figure(sim_data)
            )
            state.fiber_sim_status = (
                f"✓ Restored {len(results)} fiber simulation"
                f"{'s' if len(results) != 1 else ''} from cache."
            )
            return True
        except Exception as ex:
            print(
                f"[persist] fiber sim cache read failed: {ex}",
                flush=True,
            )
            return False

    def _save_pop_state() -> None:
        """Persist population assignments + sim results to a
        pickle in the project dir. Called at the tail of every
        successful do_pop_generate AND do_pop_run_sim — the
        design (`pop_branch_types` / `pop_seed`) lives in the
        manifest's ui_state and persists separately."""
        if not state.has_active_project:
            return
        sim_dir = _active_sim_dir()
        # If there's no assignment, nuke the cache so reopen
        # doesn't restore stale data.
        if geom.fiber_pop_types is None:
            stale = sim_dir / "pop_state.pkl"
            try:
                if stale.exists():
                    stale.unlink()
            except Exception:
                pass
            return
        sim_dir.mkdir(parents=True, exist_ok=True)
        try:
            payload = {
                "version": 1,
                "pop_types": geom.fiber_pop_types,
                "pop_rows": geom.fiber_pop_rows,
                "pop_diams": geom.fiber_pop_diameters_um,
                "row_meta": dict(state.pop_row_meta or {}),
                "row_colors": dict(
                    state.pop_row_colors or {},
                ),
                "type_colors": dict(
                    state.pop_type_colors or {},
                ),
                "row_visible": dict(
                    state.pop_row_visible or {},
                ),
                "pop_sim_results": (
                    geom.fiber_pop_sim_results or {}
                ),
                "pop_sim_results_meta": list(
                    state.pop_sim_results_meta or [],
                ),
                "pop_activated_set": list(
                    state.pop_activated_set or [],
                ),
                "pop_view_idx": int(state.pop_view_idx or 0),
            }
            with open(
                sim_dir / "pop_state.pkl", "wb",
            ) as f:
                pickle.dump(
                    payload, f, protocol=pickle.HIGHEST_PROTOCOL,
                )
        except Exception as ex:
            print(
                f"[persist] pop state write failed: {ex}",
                flush=True,
            )

    def _restore_pop_state() -> bool:
        """Reload population assignments + sim results from the
        project's pickle cache. Requires `geom.fiber_paths_raw`
        to already be loaded (the population indexes parallel
        the fiber array). Returns True on a successful
        restore."""
        fp = _active_sim_dir() / "pop_state.pkl"
        if not fp.exists():
            return False
        if geom.fiber_paths_raw is None:
            return False
        try:
            with open(fp, "rb") as f:
                payload = pickle.load(f)
            pop_types = payload.get("pop_types")
            pop_rows = payload.get("pop_rows")
            pop_diams = payload.get("pop_diams")
            if pop_types is None or pop_rows is None:
                return False
            # Sanity-check the array lengths against the
            # current fiber count; mismatched lengths mean the
            # user regenerated fibers since the cache was
            # written — bail out and let the user regenerate
            # the population fresh.
            n_paths = len(geom.fiber_paths_raw)
            if (len(pop_types) != n_paths
                    or len(pop_rows) != n_paths):
                print(
                    "[persist] pop state cache stale "
                    "(fiber count changed) — skipping restore",
                    flush=True,
                )
                return False
            geom.fiber_pop_types = pop_types
            geom.fiber_pop_rows = pop_rows
            geom.fiber_pop_diameters_um = pop_diams
            state.pop_row_meta = (
                payload.get("row_meta") or {}
            )
            state.pop_row_colors = (
                payload.get("row_colors") or {}
            )
            state.pop_type_colors = (
                payload.get("type_colors") or {}
            )
            # Default every restored row to visible if the
            # cache didn't carry visibility (older cache
            # versions).
            row_visible = dict(payload.get("row_visible") or {})
            for rid in (state.pop_row_meta or {}):
                row_visible.setdefault(rid, True)
            state.pop_row_visible = row_visible
            state.pop_generated = True
            bidx_arr = np.asarray(
                geom.fiber_branch_idx, dtype=np.int32,
            )
            state.pop_kde_figure = _build_pop_kde_figure(
                bidx_arr,
                geom.fiber_pop_rows,
                geom.fiber_pop_diameters_um,
                state.pop_branches_meta,
                dict(state.pop_row_meta or {}),
            )
            # Rebuild the at-cuff-centre cross-section from
            # the restored population so it survives close /
            # reopen without needing a re-generate.
            _paths_display_restore = (
                _fiber_paths_display() or []
            )
            state.pop_xsec_cuff_figure = (
                _build_pop_xsec_at_cuff_figure(
                    paths_display=_paths_display_restore,
                    bidx=bidx_arr,
                    pop_rows=geom.fiber_pop_rows,
                    pop_diams=geom.fiber_pop_diameters_um,
                    row_meta=dict(
                        state.pop_row_meta or {}
                    ),
                    nerve_pts_cuff_m=geom.pts_cuff,
                )
            )
            # Sim results, if present.
            pop_sim = payload.get("pop_sim_results") or {}
            if pop_sim:
                geom.fiber_pop_sim_results = pop_sim
                state.pop_sim_done = True
                state.pop_sim_results_meta = list(
                    payload.get("pop_sim_results_meta") or [],
                )
                state.pop_activated_set = list(
                    payload.get("pop_activated_set") or [],
                )
                state.pop_view_idx = int(
                    payload.get("pop_view_idx") or 0,
                )
                paths_display = _fiber_paths_display() or []
                state.pop_xsec_figure = _build_pop_xsec_figure(
                    paths_display,
                    geom.fiber_pop_rows,
                    geom.fiber_pop_diameters_um,
                    dict(state.pop_row_meta or {}),
                    set(state.pop_activated_set),
                )
                sim_data = pop_sim.get(state.pop_view_idx)
                if sim_data is None and pop_sim:
                    state.pop_view_idx = int(
                        min(pop_sim.keys()),
                    )
                    sim_data = pop_sim[state.pop_view_idx]
                if sim_data is not None:
                    state.pop_propagation_figure = (
                        _build_fiber_propagation_figure(
                            sim_data,
                        )
                    )
                    state.pop_waterfall_figure = (
                        _build_fiber_waterfall_figure(sim_data)
                    )
            state.pop_status = (
                "✓ Restored population from cache."
            )
            _request_render()
            return True
        except Exception as ex:
            print(
                f"[persist] pop state read failed: {ex}",
                flush=True,
            )
            return False

    def _restore_sweep_from_disk() -> bool:
        """F2.1.d — read the most recent SweepResult from
        `<project>/sweeps/sweep_<sha>.npz` + `.json` and push its
        three figures + CSV paths into the Sweep panel's state.
        Returns True iff a sweep was found AND its fiber count
        still matches the live geometry (mismatched = different
        fibers were generated since cache was written; bail and
        let the user re-run)."""
        from golgi.figures.recruitment import (
            build_activation_heatmap_figure,
            build_recruitment_curve_figure,
            build_threshold_scatter_figure,
        )
        from golgi.projects import sweep_cache as _sweep_cache_mod
        result = _sweep_cache_mod.load_latest(
            Path(get_active().out_dir),
        )
        if result is None:
            return False
        # Sanity-check: the result's fiber_indices must all be in
        # range against the currently-loaded fibers. A mismatch
        # means the user regenerated fibers after the sweep cache
        # was written — skip the restore rather than show stale
        # results indexed against a vanished fiber set.
        n_paths = (
            int(len(geom.fiber_paths_raw))
            if geom.fiber_paths_raw is not None else 0
        )
        if n_paths == 0:
            return False
        max_idx = int(np.asarray(result.fiber_indices).max())
        if max_idx >= n_paths:
            print(
                "[sweep_cache] stale (fiber count changed) — "
                "skipping restore",
                flush=True,
            )
            return False
        # Push figures + CSV paths back to state.
        csvs = _sweep_cache_mod.csv_paths_for(
            Path(get_active().out_dir), str(result.sha),
        )
        if result.activated is not None:
            state.sweep_recruitment_figure = (
                build_recruitment_curve_figure(result)
            )
            state.sweep_heatmap_figure = (
                build_activation_heatmap_figure(result)
            )
            state.sweep_threshold_figure = (
                {"data": [], "layout": {}}
            )
            state.sweep_mode = "recruitment"
        elif result.thresholds_uA is not None:
            state.sweep_threshold_figure = (
                build_threshold_scatter_figure(result)
            )
            state.sweep_recruitment_figure = (
                {"data": [], "layout": {}}
            )
            state.sweep_heatmap_figure = {"data": [], "layout": {}}
            state.sweep_mode = "threshold"
        n_fibers = int(len(result.fiber_indices))
        if result.activated is not None:
            n_amps = int(result.activated.shape[1])
            summary = (
                f"Recruitment sweep · {n_fibers} fibers × "
                f"{n_amps} amplitudes · "
                f"{result.elapsed_s:.1f} s "
                f"({result.n_sims_total} sims) · restored"
            )
        else:
            n_activated = int(np.isfinite(
                np.asarray(result.thresholds_uA),
            ).sum())
            summary = (
                f"Threshold finder · {n_fibers} fibers · "
                f"{n_activated} activated · "
                f"{result.elapsed_s:.1f} s "
                f"({result.n_sims_total} sims) · restored"
            )
        state.sweep_result_summary = summary
        state.sweep_status = summary
        state.sweep_has_result = True
        state.sweep_cache_sha = str(result.sha or "")
        # Rebuild the browser-download data URIs from the restored
        # in-memory result (cheaper than reading the on-disk CSVs
        # back). Same shape as the live action-handler emit path.
        import base64 as _b64
        from golgi.figures.recruitment import (
            recruitment_to_csv as _rec_csv,
            threshold_to_csv as _thr_csv,
            activation_heatmap_to_csv as _hm_csv,
        )

        def _csv_uri(txt: str) -> str:
            return (
                "data:text/csv;base64,"
                + _b64.b64encode(txt.encode("utf-8")).decode("ascii")
            )

        if result.activated is not None:
            state.sweep_recruitment_csv_data_uri = _csv_uri(
                _rec_csv(result),
            )
            state.sweep_recruitment_csv_filename = (
                f"sweep_{result.sha}_recruitment.csv"
            )
            state.sweep_heatmap_csv_data_uri = _csv_uri(
                _hm_csv(result),
            )
            state.sweep_heatmap_csv_filename = (
                f"sweep_{result.sha}_activation_heatmap.csv"
            )
        else:
            state.sweep_recruitment_csv_data_uri = ""
            state.sweep_recruitment_csv_filename = ""
            state.sweep_heatmap_csv_data_uri = ""
            state.sweep_heatmap_csv_filename = ""
        if result.thresholds_uA is not None:
            state.sweep_threshold_csv_data_uri = _csv_uri(
                _thr_csv(result),
            )
            state.sweep_threshold_csv_filename = (
                f"sweep_{result.sha}_thresholds.csv"
            )
        else:
            state.sweep_threshold_csv_data_uri = ""
            state.sweep_threshold_csv_filename = ""
        # NPZ — read the disk file the cache wrote.
        npz_path = (
            Path(get_active().out_dir) / "sweeps"
            / f"sweep_{result.sha}.npz"
        )
        if npz_path.is_file():
            try:
                state.sweep_npz_data_uri = (
                    "data:application/octet-stream;base64,"
                    + _b64.b64encode(
                        npz_path.read_bytes(),
                    ).decode("ascii")
                )
                state.sweep_npz_filename = (
                    f"sweep_{result.sha}.npz"
                )
            except Exception as ex:                          # noqa: BLE001
                print(
                    f"[sweep_cache] npz data-uri build failed: "
                    f"{ex}", flush=True,
                )
                state.sweep_npz_data_uri = ""
                state.sweep_npz_filename = ""
        else:
            state.sweep_npz_data_uri = ""
            state.sweep_npz_filename = ""
        return True

    # ---- welcome-view click handlers ----
    def do_show_new_project_dialog():
        # Reset form fields so prior typed values don't leak
        # between dialog opens.
        state.new_project_name = ""
        state.new_project_error = ""
        state.show_new_project_dialog = True

    def do_cancel_new_project():
        state.show_new_project_dialog = False
        state.new_project_error = ""

    @gated("project_create")
    async def do_create_and_open_project():
        """Create an empty project (no source bundled yet) and
        open it. The user picks a nerve geometry from the Import
        drawer once the workspace lands. Owner stamped from the
        active auth session."""
        name = str(state.new_project_name or "").strip()
        if not name:
            state.new_project_error = "Project name is required."
            return
        owner_uid = _auth_session.get("user_id")
        try:
            pdir = _create_project(
                name, source_path=None,
                owner_user_id=owner_uid,
            )
        except Exception as ex:
            state.new_project_error = (
                f"Could not create project: "
                f"{type(ex).__name__}: {ex}"
            )
            return
        state.show_new_project_dialog = False
        await do_open_project(str(pdir))

    @gated("project_open")
    async def do_open_project(pdir_str: str):
        """Activate the project at `pdir_str` and restore every
        cached stage that exists on disk (source → mesh → FEM →
        fibers). Heavy steps (TetGen, FEM solve, RK4 integration)
        are NEVER rerun — they're loaded straight from cache.
        state.busy_open is the wrapper flag inner handlers
        (do_load_geometry, do_fit_cuff) check to skip their own
        busy=True/False lifecycle, so the lightbox stays up for
        the entire chain — including the post-mount cuff resync
        that used to fire AFTER busy went down."""
        pdir = Path(pdir_str)
        if not pdir.is_dir():
            return

        # Peek at project.json BEFORE _activate_project so we can
        # pre-count the steps that will actually run (some
        # projects only have source; others have the full
        # pipeline). The user wants a "(k/N)" counter in the
        # status line and no progress bar.
        try:
            _manifest_peek = json.loads(
                (pdir / "project.json").read_text(encoding="utf-8"),
            )
        except Exception:
            _manifest_peek = {}
        _src_rel_peek = str(
            _manifest_peek.get("source_file", "") or "",
        )
        _has_src_peek = bool(
            _src_rel_peek and (pdir / _src_rel_peek).is_file()
        )
        # SceneCatalog Phase 4 fix — multi-design projects (F3.2)
        # store nerve.msh per-design under designs/<eid>/nerve.msh,
        # not at the project root. The legacy root path stays
        # supported for pre-F3.2 projects. Without this, step 4
        # below was silently skipped for every multi-design
        # project, leaving the mesh-restore exclusively to the
        # early `_on_selected_design_mesh_swap` watcher — which
        # runs BEFORE step 2 loads the nerve, so the PCA back-
        # transform got skipped and the mesh stayed in
        # design-local frame.
        _designs_dir = pdir / "designs"
        _has_mesh_peek = (
            (pdir / "nerve.msh").exists()
            or (
                _designs_dir.is_dir()
                and any(
                    (d / "nerve.msh").is_file()
                    for d in _designs_dir.iterdir()
                    if d.is_dir()
                )
            )
        )
        # F3.1: FEM outputs may live at the flat root (legacy
        # layout) OR under fem/<design_id>/. We "have" FEM when
        # either form is detectable.
        _fem_designs_peek = _fem_layout.enumerate_designs(pdir)
        _has_fem_peek = bool(_fem_designs_peek) and any(
            (
                _fem_layout.fem_design_dir(pdir, _d["id"])
                / "axis_line.npz"
            ).exists()
            for _d in _fem_designs_peek
        )
        _has_fibers_peek = (
            pdir / "nerve_paths_fibers.npz"
        ).exists()
        # Single-fiber + population sim caches each live in their
        # own pickle in the project dir. The "Restoring
        # simulation cache" step only fires if at least one of
        # them is on disk — otherwise it would inflate the
        # denominator + show "(8/7)…" at the end of the run.
        def _any_sim_pkl_exists() -> bool:
            for _name in ("fiber_sim_results.pkl", "pop_state.pkl"):
                if (pdir / _name).is_file():
                    return True
            sims_root = pdir / "sims"
            if sims_root.is_dir():
                for _sub in sims_root.iterdir():
                    if _sub.is_dir() and (
                        (_sub / "fiber_sim_results.pkl").is_file()
                        or (_sub / "pop_state.pkl").is_file()
                    ):
                        return True
            return False
        _has_sim_cache_peek = (
            _has_fibers_peek and _any_sim_pkl_exists()
        )
        # F2.1.d: sweep cache peek — auto-restore the last sweep
        # for this project so the user sees the figures + CSV
        # paths on reopen without re-running.
        _sweeps_dir_peek = pdir / "sweeps"
        _has_sweep_cache_peek = (
            _has_fibers_peek
            and (_sweeps_dir_peek / "latest.txt").is_file()
        )
        steps: list[str] = ["Activating project"]
        if _has_src_peek:
            steps.append("Loading nerve")
            steps.append("Arranging electrodes")
        if _has_mesh_peek:
            steps.append("Restoring cached mesh")
        if _has_fem_peek:
            steps.append("Restoring FEM results")
        if _has_fibers_peek:
            steps.append("Restoring fiber trajectories")
        if _has_sim_cache_peek:
            steps.append("Restoring simulation cache")
        if _has_sweep_cache_peek:
            steps.append("Restoring sweep cache")
        steps.append("Finalising scene")
        n_steps = len(steps)

        log_lines: list[str] = []
        _step_idx = {"i": 0}

        def _push_log(line: str) -> None:
            log_lines.append(_stamp_user_line(line))
            # Last 8 lines fit comfortably in the lightbox log.
            state.busy_log = "\n".join(log_lines[-8:])
            state.flush()

        def _set_step(label: str) -> None:
            _step_idx["i"] += 1
            state.busy_msg = (
                f"{label} ({_step_idx['i']}/{n_steps})"
            )
            state.flush()

        state.busy_open = True
        state.busy = True
        state.busy_msg = (
            f"Opening project '{pdir.name}' (0/{n_steps})"
        )
        state.busy_log = ""
        state.flush()

        try:
            # 0) Wipe all actors + in-memory state from any
            #    previously-open project. Without this, a
            #    create-new-project (or tile-switch) inherits the
            #    prior project's nerve / cuff / mesh actors.
            _clear_plotter_actors()
            _reset_geom_and_state()

            # Per-stage safety settle — a hard sleep after each
            # phase finishes so any deferred work that wasn't
            # caught by the explicit pipeline (client-side mesh
            # uploads, vtk.js GPU buffer creation, late state
            # watchers) gets a chance to land before the next
            # phase starts. The final stage gets a much longer
            # buffer because that's where we kept seeing the
            # "lightbox closed but scene still appearing" lag.
            _STAGE_SETTLE_S = 5.0

            # 1) Activate + restore persisted UI state.
            _set_step("Activating project")
            manifest = _activate_project(pdir)
            state.has_active_project = True
            state.current_project_dir = str(pdir)
            state.current_project_name = str(
                manifest.get("name", pdir.name),
            )
            state.current_project_modified = _format_modified(
                manifest.get("last_modified", ""),
            )
            _apply_ui_state(manifest.get("ui_state", {}))
            # M17 — restore µCT-bundle import flags from the
            # top-level manifest. `_write_manifest` stores
            # `import_source_type` and `uct_bundle_id` at the
            # manifest top level (not under `ui_state`), and
            # they're not in `_PERSISTED_UI_KEYS`, so without
            # this explicit restore `do_load_geometry` always
            # took the STL path on reopen — losing the fascicle
            # structure and forcing the µCT epi.stl to be
            # labelled as endoneurium in the FEM region map.
            # Symptom the user saw: "after reopening there are
            # no fascicles anymore and the epineurium geometry
            # is mislabeled as endoneurium".
            _ist = str(manifest.get("import_source_type", "") or "")
            if _ist:
                state.import_source_type = _ist
            _bid = str(manifest.get("uct_bundle_id", "") or "")
            if _bid:
                # The manifest field is named `uct_bundle_id` for
                # historical reasons but now also stores the
                # histology-bundle id (M47). Route by source
                # type so each picker's selection is restored.
                if _ist == "histo_bundle":
                    state.selected_histo_bundle = _bid
                else:
                    state.selected_uct_bundle = _bid
            _apply_persisted_sigma()
            state.data_files = list_data_files()
            # V1 — refresh the µCT-bundle picker for the new
            # project before the user can open the import
            # wizard. Cheap (just a directory listing); fills
            # `state.uct_bundle_items` so the Step-1 source-
            # type tile shows the right "(N bundles available)"
            # text out of the gate. M47 — also refresh the
            # histology-bundle picker.
            do_refresh_uct_bundles()
            do_refresh_histo_bundles()
            state.view_mode = "workspace"
            state.flush()
            _push_log(f"✓ Activated project '{pdir.name}'")
            await asyncio.sleep(_STAGE_SETTLE_S)

            # 2) Restore source geometry (cheap: file read + PCA).
            if _has_src_peek:
                _set_step("Loading nerve")
                state.selected_file = str(pdir / _src_rel_peek)
                await do_load_geometry()
                if geom.nerve is not None:
                    _push_log(
                        f"✓ Loaded nerve: "
                        f"{Path(_src_rel_peek).name}  "
                        f"({geom.nerve['pts_raw'].shape[0]:,} pts, "
                        f"{geom.nerve['boundary_raw'].shape[0]:,} tris)"
                    )
                else:
                    _push_log(
                        "⚠ Source restore failed — "
                        "see Import drawer log"
                    )
                await asyncio.sleep(_STAGE_SETTLE_S)
                # 3) Cuff fit — prerequisite for the cached
                #    mesh actors to land in the right frame.
                if geom.nerve is not None:
                    _set_step("Arranging electrodes")
                    await do_fit_cuff(refit=True)
                    if geom.R_ci is not None:
                        _push_log(
                            f"✓ Cuff fitted  R_ci = "
                            f"{geom.R_ci * 1e3:.2f} mm, "
                            f"R_co = "
                            f"{(geom.R_co or 0.0) * 1e3:.2f} mm"
                        )
                    await asyncio.sleep(_STAGE_SETTLE_S)

            # 4) Restore cached mesh (skips slow TetGen rebuild).
            if _has_mesh_peek:
                _set_step("Restoring cached mesh")
                ok = await _restore_mesh_from_disk()
                if ok and geom.mesh_elems is not None:
                    _push_log(
                        f"✓ Mesh restored  "
                        f"({len(geom.mesh_elems):,} tets, "
                        f"{len(geom.region_surfaces or {}):,} "
                        f"regions)"
                    )
                else:
                    _push_log("⚠ Mesh restore failed — see logs")
                await asyncio.sleep(_STAGE_SETTLE_S)

            # 5) Restore cached FEM outputs.
            if _has_fem_peek:
                _set_step("Restoring FEM results")
                # _refresh_fem_plots regenerates three matplotlib
                # PNGs on the main thread — for a project with
                # 100+ fibers + SG-AF that's ~10-30 s. Push it
                # off the loop so we don't block state flushes
                # (e.g. busy_log updates).
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(
                    None, _restore_fem_from_disk,
                )
                if geom.fem_axis is not None:
                    _n_fibers = (
                        len(geom.fiber_paths_Ve)
                        if geom.fiber_paths_Ve is not None
                        else 0
                    )
                    _push_log(
                        f"✓ FEM restored  "
                        f"({len(geom.fem_axis['z']):,} axis pts, "
                        f"{geom.fem_slice['Ve'].shape[0]} slices"
                        + (f", {_n_fibers} per-fiber Vₑ traces)"
                           if _n_fibers > 0 else ")")
                    )
                else:
                    _push_log("⚠ FEM restore failed — see logs")
                await asyncio.sleep(_STAGE_SETTLE_S)

            # 6) Restore cached fibers.
            if _has_fibers_peek:
                _set_step("Restoring fiber trajectories")
                _restore_fibers_from_disk()
                if geom.fiber_paths_raw is not None:
                    _push_log(
                        f"✓ Fibers restored  "
                        f"({len(geom.fiber_paths_raw):,} "
                        f"trajectories, "
                        f"{geom.fiber_n_branches} branches)"
                    )
                else:
                    _push_log(
                        "⚠ Fiber restore failed — see logs"
                    )
                await asyncio.sleep(_STAGE_SETTLE_S)

            # 6b) Restore cached single-fiber sim results +
            #     population state, if present. Both depend on
            #     `geom.fiber_paths_raw` being populated (above)
            #     since they index parallel to the fiber array.
            #     We only enter this step (and consume a slot in
            #     the busy-overlay denominator) when at least
            #     one cache file is actually on disk — see the
            #     `_has_sim_cache_peek` gate above.
            if (geom.fiber_paths_raw is not None
                    and _has_sim_cache_peek):
                _set_step("Restoring simulation cache")
                if _restore_fiber_sim_cache():
                    _push_log(
                        f"✓ Single-fiber sim cache restored "
                        f"({len(geom.fiber_sim_results or {}):,} "
                        f"fiber result(s))"
                    )
                if _restore_pop_state():
                    n_sim = len(
                        geom.fiber_pop_sim_results or {},
                    )
                    if n_sim:
                        _push_log(
                            f"✓ Population restored with "
                            f"{n_sim:,} sim result(s)"
                        )
                    else:
                        _push_log(
                            "✓ Population assignments restored"
                        )
                await asyncio.sleep(_STAGE_SETTLE_S)

            # 6c) F2.1.d: restore the most recent sweep cache
            # (figures + CSV paths) so the Sweep tab is preloaded
            # with the last run on project reopen.
            if (geom.fiber_paths_raw is not None
                    and _has_sweep_cache_peek):
                _set_step("Restoring sweep cache")
                if _restore_sweep_from_disk():
                    _push_log(
                        f"✓ Sweep cache restored "
                        f"({state.sweep_cache_sha})"
                    )
                else:
                    _push_log(
                        "⚠ Sweep cache present but skipped "
                        "(stale fiber set)"
                    )
                await asyncio.sleep(_STAGE_SETTLE_S)

            # 7) Post-mount resync + final safety wait. The
            #    first do_fit_cuff ran while the workspace
            #    plotter was still being mounted on the client;
            #    the 0.35 s delay below gives the client a beat
            #    before the translate-only re-fit. The bigger
            #    2-minute hold afterwards is a deliberate "we
            #    know SOMETHING async is still landing on the
            #    client but didn't track it down" safety buffer.
            _set_step("Finalising scene")
            # Single scene-state render pass now that every cached
            # output (V_e on surface, fiber_paths_Ve, slice_volume)
            # has landed. The folder reads geom + state in one
            # consistent snapshot, so endo/epi V_e + fibers + the
            # field-lines kickoff happen atomically.
            _request_render()
            if (state.has_fem
                    and bool(state.show_field_lines)):
                _ensure_field_lines_async()
            # The final-settle wait is only meaningful when there
            # were actually cached actors to upload to the client.
            # A fresh project has nothing on disk, so skip the
            # sleep entirely — otherwise the user stares at the
            # busy lightbox for two minutes after clicking "create"
            # on an empty project. Detect "fresh" via the same
            # peek flags we used to decide which restore steps to
            # run.
            _has_any_cached = (
                _has_src_peek
                or _has_mesh_peek
                or _has_fem_peek
                or _has_fibers_peek
            )
            if _has_any_cached:
                _push_log("✓ Scene ready")
            else:
                _push_log("✓ New project ready")
        except Exception as ex:
            print(f"[project] open failed: {ex}", flush=True)
            _push_log(
                f"⚠ open failed: {type(ex).__name__}: {ex}"
            )
            state.fem_status = (
                f"⚠ open project failed: "
                f"{type(ex).__name__}: {ex}"
            )
        finally:
            state.busy_open = False
            state.busy = False
            state.busy_msg = ""
            state.busy_log = ""
            state.flush()
            safe_update()
            safe_reset_camera()


    # ---- close-project handlers ----
    def do_show_close_dialog():
        state.show_close_dialog = True

    def do_cancel_close():
        state.show_close_dialog = False

    @gated("project_close")
    async def do_confirm_close():
        """Autosave one last time + tear down all in-memory state +
        flip back to the welcome view."""
        state.show_close_dialog = False
        if state.has_active_project:
            try:
                _autosave(capture_thumb=True)
            except Exception:
                pass
        _clear_plotter_actors()
        _reset_geom_and_state()
        _deactivate_project()
        with state:
            state.has_active_project = False
            state.current_project_name = ""
            state.current_project_dir = ""
            state.current_project_modified = ""
            state.view_mode = "welcome"
            state.projects_list = _list_projects(owner_user_id=_auth_session.get("user_id"))
            state.data_files = list_data_files()
        safe_update()

    # ---- project-detail lightbox handlers ----
    def _refresh_detail_briefs() -> None:
        """Resolve owner + last-modifier briefs (avatar + name)
        for the currently-open detail_project and push them
        into state. Called whenever the dialog opens or the
        detail_project changes.

        Also populates the Status + Activity tabs:
          * `detail_status_rows` — 8-stage completion table
            derived from manifest + disk-file presence (works
            equally for closed projects on the welcome page).
          * `detail_activity_events` — recent audit-log rows
            for this project (newest first, ≤ 500).
        Both run synchronously here so they're ready by the
        time the tab is rendered; payload is small and the
        load uses cached briefs / a single SQL query.
        """
        proj = state.detail_project or {}
        owner_id = proj.get("owner_user_id")
        modifier_id = proj.get("last_modified_user_id")
        # If modifier wasn't stamped on the manifest yet
        # (legacy projects), default to the owner — they're
        # the only known actor for that history.
        if modifier_id is None:
            modifier_id = owner_id
        state.detail_project_owner = _user_brief_by_id(owner_id)
        state.detail_project_modifier = _user_brief_by_id(
            modifier_id,
        )
        # Refresh the users list too so the share picker has
        # any newly-registered users without requiring a logout
        # / login cycle.
        state.users_list = _list_users()
        # Stage + activity tabs. Empty project (no dir) =
        # empty rows / events.
        state.detail_status_rows = _compute_project_status(proj)
        state.detail_activity_events = (
            _load_audit_events_for_project(
                proj.get("dir", ""),
            )
        )
        # Drop any expanded-payload markers from a prior open
        # so the new dialog starts with everything collapsed.
        state.detail_activity_expanded = []
        # Default the tab back to Overview on open — the user
        # is most likely arriving for metadata; Status and
        # Activity are one click away.
        state.detail_tab = "overview"
        # Push everything we just wrote to the client in one go.
        # Without an explicit flush some clients render the dialog
        # before `detail_status_rows` / `detail_activity_events`
        # land — the user's "need to open a few times" symptom.
        state.flush()

    # W1.7d: detail-dialog open/refresh watcher moved to
    # watchers.project_detail (registered below, after
    # _persist_labels is also defined).

    def do_toggle_activity_payload(eid, *_args) -> None:
        """Flip whether `eid` sits in `detail_activity_expanded`.
        Server-side because the equivalent inline-JS (spread +
        arrow inside a Vue template attribute) tripped the
        compiler on some browsers."""
        try:
            eid_i = int(eid)
        except (TypeError, ValueError):
            return
        cur = list(state.detail_activity_expanded or [])
        if eid_i in cur:
            cur = [x for x in cur if x != eid_i]
        else:
            cur.append(eid_i)
        state.detail_activity_expanded = cur

    def do_save_shared_users(user_ids, *_args) -> None:
        """Persist the share-users list to the project's
        manifest. Filters out non-int / non-existing ids and
        the project owner (the owner is implicit and can't be
        'shared with themselves'). Refreshes projects_list so
        anyone newly added sees the tile when they next visit
        their welcome view."""
        proj = state.detail_project
        if not proj:
            return
        pdir = Path(proj.get("dir", ""))
        if not pdir.is_dir():
            return
        owner_id = proj.get("owner_user_id")
        cleaned: list[int] = []
        for v in (user_ids or []):
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if owner_id is not None and vi == int(owner_id):
                # Skip — owner is implicit.
                continue
            if vi not in cleaned:
                cleaned.append(vi)
        try:
            data = _write_manifest(pdir, shared_user_ids=cleaned)
        except Exception as ex:
            print(
                f"[share] write failed for {pdir}: {ex}",
                flush=True,
            )
            return
        # Mirror back into the open dialog so the chips render
        # from the saved list, not the transient pre-save one.
        updated = dict(proj)
        updated["shared_user_ids"] = sorted(cleaned)
        updated["last_modified"] = data.get(
            "last_modified", updated.get("last_modified"),
        )
        updated["last_modified_short"] = _format_modified(
            data.get("last_modified", ""),
        )
        if data.get("last_modified_user_id") is not None:
            updated["last_modified_user_id"] = int(
                data["last_modified_user_id"],
            )
        state.detail_project = updated
        if state.has_active_project and (
            str(state.current_project_dir) == str(pdir)
        ):
            state.current_project_modified = _format_modified(
                data.get("last_modified", ""),
            )
        _refresh_detail_briefs()
        state.projects_list = _list_projects(
            owner_user_id=_auth_session.get("user_id"),
        )

    def do_close_detail_dialog():
        state.show_detail_dialog = False
        state.edit_name_mode = False
        # Keep detail_project around for a tick so the close
        # transition doesn't blank the card mid-fade.

    def do_start_edit_name():
        """Enter inline rename mode for the open project tile.
        Seeds the input with the current name."""
        if not state.detail_project:
            return
        state.edit_name_value = str(
            state.detail_project.get("name", ""),
        )
        state.edit_name_mode = True

    def do_cancel_edit_name():
        state.edit_name_mode = False
        state.edit_name_value = ""

    def do_save_edit_name():
        """Persist the renamed value into the project's manifest +
        refresh the projects_list so the welcome tiles + the open
        details lightbox both reflect the new name."""
        proj = state.detail_project
        if not proj:
            state.edit_name_mode = False
            return
        new_name = str(state.edit_name_value or "").strip()
        if not new_name or new_name == proj.get("name", ""):
            # Empty / unchanged → just leave edit mode without
            # touching the manifest.
            state.edit_name_mode = False
            return
        pdir = Path(proj.get("dir", ""))
        if not pdir.is_dir():
            state.edit_name_mode = False
            return
        try:
            data = _write_manifest(pdir, name=new_name)
        except Exception as ex:
            print(
                f"[project] rename failed: {ex}", flush=True,
            )
            state.edit_name_mode = False
            return
        # Reflect immediately in the open lightbox + tile grid.
        # Mutating detail_project field-by-field keeps the
        # Vue reactivity happy (vs. reassigning the whole dict,
        # which would lose the v_if=detail_project transition).
        updated = dict(proj)
        updated["name"] = new_name
        updated["last_modified"] = data.get("last_modified", "")
        updated["last_modified_short"] = _format_modified(
            data.get("last_modified", ""),
        )
        state.detail_project = updated
        state.projects_list = _list_projects(owner_user_id=_auth_session.get("user_id"))
        # If the renamed project is the one currently OPEN in the
        # workspace, update the navbar pill too.
        if state.has_active_project and str(state.current_project_dir) == str(pdir):
            state.current_project_name = new_name
            state.current_project_modified = (
                updated["last_modified_short"]
            )
        state.edit_name_mode = False
        state.edit_name_value = ""

    # ---- label CRUD inside the details lightbox ----
    def do_start_add_label():
        if not state.detail_project:
            return
        state.add_label_value = ""
        state.add_label_mode = True

    def do_cancel_add_label():
        state.add_label_mode = False
        state.add_label_value = ""

    def _persist_labels(pdir: Path, labels: list[str]) -> None:
        """Write labels to manifest + push the updated list into
        detail_project + refresh the welcome tiles."""
        _write_manifest(pdir, labels=list(labels))
        proj = state.detail_project
        if proj is not None:
            updated = dict(proj)
            updated["labels"] = list(labels)
            state.detail_project = updated
        state.projects_list = _list_projects(owner_user_id=_auth_session.get("user_id"))

    def do_save_add_label():
        proj = state.detail_project
        if not proj:
            state.add_label_mode = False
            return
        new_label = str(state.add_label_value or "").strip()
        # Strip to a sensible length so a stray paste doesn't make
        # a 1k-char chip. Same uppercase-letter visual = same code
        # but kept case-sensitive for storage so the user can still
        # write e.g. "v1.2" verbatim.
        new_label = new_label[:48]
        if not new_label:
            state.add_label_mode = False
            return
        pdir = Path(proj.get("dir", ""))
        if not pdir.is_dir():
            state.add_label_mode = False
            return
        current = list(proj.get("labels", []))
        if new_label not in current:
            current.append(new_label)
            try:
                _persist_labels(pdir, current)
            except Exception as ex:
                print(
                    f"[project] add label failed: {ex}",
                    flush=True,
                )
        state.add_label_mode = False
        state.add_label_value = ""

    # W1.7d: detail-dialog open-refresh + remove-label watchers
    # moved to watchers.project_detail. Both closures
    # (_refresh_detail_briefs and _persist_labels) are defined above
    # this line, so kwarg eager-evaluation is safe (AST-verified
    # before commit, per the W1.7b fix lesson).
    _watchers.project_detail.register(
        state,
        refresh_detail_briefs=_refresh_detail_briefs,
        persist_labels=_persist_labels,
    )

    def do_open_from_detail():
        """Open button inside the details lightbox. Closes the
        lightbox and pokes the existing tile-click bridge — the
        @state.change('open_project_request') watcher launches
        the async open."""
        proj = state.detail_project
        if not proj:
            return
        state.show_detail_dialog = False
        state.open_project_request = str(proj.get("dir", ""))

    def do_request_delete_from_detail():
        """Delete button inside the details lightbox. Stages the
        target project + opens the confirm sub-dialog. We close
        the details lightbox first so it doesn't sit underneath
        the confirm — Vuetify renders both via teleport portals
        but stacking them looks busy."""
        proj = state.detail_project
        if not proj:
            return
        state.delete_project_dir = str(proj.get("dir", ""))
        state.delete_project_name = str(proj.get("name", ""))
        state.delete_error = ""
        state.show_detail_dialog = False
        state.show_delete_dialog = True

    def do_cancel_delete():
        state.show_delete_dialog = False
        state.delete_project_dir = ""
        state.delete_project_name = ""
        state.delete_error = ""

    @gated("project_delete")
    def do_confirm_delete():
        """Remove the staged project folder, refresh the tile
        list, and dismiss the dialog. Lives outside the
        details-lightbox path so the user can also bind it from
        elsewhere if needed."""
        pdir_str = str(state.delete_project_dir or "")
        if not pdir_str:
            state.show_delete_dialog = False
            return
        pdir = Path(pdir_str)
        # Safety check: must live under PROJECTS_ROOT — otherwise
        # we'd be deleting an arbitrary path the user typed into
        # state somehow.
        try:
            pdir.resolve().relative_to(PROJECTS_ROOT.resolve())
        except ValueError:
            state.delete_error = (
                "Refusing to delete: path is outside the projects "
                "root."
            )
            return
        try:
            shutil.rmtree(pdir)
        except Exception as ex:
            state.delete_error = (
                f"Delete failed: {type(ex).__name__}: {ex}"
            )
            return
        state.show_delete_dialog = False
        state.delete_project_dir = ""
        state.delete_project_name = ""
        state.delete_error = ""
        state.detail_project = None
        state.projects_list = _list_projects(owner_user_id=_auth_session.get("user_id"))

    # Expose key handlers on the controller so we can reference
    # them from inline Vue expressions (per-tile click that passes
    # the project's path).
    ctrl.do_open_project = do_open_project
    ctrl.do_create_and_open_project = do_create_and_open_project
    ctrl.do_confirm_close = do_confirm_close

    # W1.8b — Auth/profile/logout handlers extracted to
    # golgi/actions/auth.py. Registered HERE (not at the end of
    # build_app) because the ctrl.do_* bindings below need the
    # names in scope. All closure deps (_USERNAME_RE,
    # _decode_avatar_data_uri, _validate_avatar_bytes,
    # _push_auth_session, _clear_auth_session, do_confirm_close)
    # are defined above this line.
    _auth_actions = _actions.auth.register(
        state,
        username_re=_USERNAME_RE,
        decode_avatar_data_uri=_decode_avatar_data_uri,
        validate_avatar_bytes=_validate_avatar_bytes,
        push_auth_session=_push_auth_session,
        clear_auth_session=_clear_auth_session,
        do_confirm_close=do_confirm_close,
    )
    do_open_auth_dialog = _auth_actions["do_open_auth_dialog"]
    do_close_auth_dialog = _auth_actions["do_close_auth_dialog"]
    do_switch_auth_mode = _auth_actions["do_switch_auth_mode"]
    do_submit_login = _auth_actions["do_submit_login"]
    do_submit_register = _auth_actions["do_submit_register"]
    do_open_profile_dialog = _auth_actions["do_open_profile_dialog"]
    do_close_profile_dialog = _auth_actions["do_close_profile_dialog"]
    do_save_profile = _auth_actions["do_save_profile"]
    do_dismiss_logout_dialog = _auth_actions["do_dismiss_logout_dialog"]
    do_confirm_logout = _auth_actions["do_confirm_logout"]
    do_logout = _auth_actions["do_logout"]

    # Auth — registered so the login-card Vue `trigger(...)`
    # expressions (Enter key in password / confirm-password
    # fields) resolve to the right server handlers.
    ctrl.do_submit_login = do_submit_login
    ctrl.do_submit_register = do_submit_register
    ctrl.do_logout = do_logout
    ctrl.do_open_auth_dialog = do_open_auth_dialog
    ctrl.do_open_profile_dialog = do_open_profile_dialog
    ctrl.do_close_profile_dialog = do_close_profile_dialog
    ctrl.do_save_profile = do_save_profile

    # -----------------------------------------------------------------
    # Reactive callbacks
    # -----------------------------------------------------------------
    # Import-drawer scale_preset → scale_factor watcher extracted to
    # golgi.watchers.import_panel in step W1.7a.
    _watchers.import_panel.register(state)

    # Cuff + electrode watchers — extracted to
    # golgi.watchers.cuff in step 5.2.
    _watchers.cuff.register(
        state, geom=geom,
        elec_sync_guard=_elec_sync_guard,
        default_electrode=DEFAULT_ELECTRODE,
        save_selected_to_designs=_save_selected_to_designs,
        do_fit_cuff=do_fit_cuff,
        apply_electrode_visibility=_apply_electrode_visibility,
        safe_update=safe_update,
        load_design_to_selected=_load_design_to_selected,
        do_remove_design=do_remove_design,
        do_delete_mesh=do_delete_mesh,
        find_design=_find_design,
        find_cuff_origin_pca=find_cuff_origin_pca,
        local_pca_refine=local_pca_refine,
        autosize_R_ci=autosize_R_ci,
        compute_polarity_sums=_compute_polarity_sums,
        refit_design_geometry=_refit_design_geometry,
    )

    # _on_fiber_branch_names_change moved with fiber_panel.

    def do_start_branch_rename(idx, *_args) -> None:
        """Switch the Branch summary table into edit mode for
        branch `idx`. Seeds `branch_rename_value` with the
        current displayed label so the user is editing the
        existing text, not an empty field."""
        try:
            i = int(idx)
        except (TypeError, ValueError):
            return
        if i < 0 or i >= MAX_FIBER_BRANCHES:
            return
        state.branch_rename_active = i
        state.branch_rename_value = _branch_name(i)

    def do_apply_branch_rename(*_args) -> None:
        """Commit the inline-edited branch label into the
        persistent flat var. Empty value (or the literal
        default "Branch N") clears the rename so the default
        comes back. The flat-var watcher then propagates the
        new label to the summary table, the population tabs,
        the autocomplete chips, and the legend."""
        try:
            i = int(state.branch_rename_active)
        except (TypeError, ValueError):
            i = -1
        if i < 0 or i >= MAX_FIBER_BRANCHES:
            state.branch_rename_active = -1
            state.branch_rename_value = ""
            return
        val = str(state.branch_rename_value or "").strip()[:48]
        if val == f"Branch {i}":
            val = ""
        state[f"fiber_branch_name_{i}"] = val
        state.branch_rename_active = -1
        state.branch_rename_value = ""

    def do_cancel_branch_rename(*_args) -> None:
        """Abort the inline edit and drop any pending text."""
        state.branch_rename_active = -1
        state.branch_rename_value = ""

    # Render-toggle watchers — register() moved DOWN to after
    # _update_muscle_preview + _remove_muscle_overlay are defined
    # (W1.7b regression fix: passing those closures as kwargs forces
    # eager evaluation, which raised UnboundLocalError when the
    # `def`s were below this call).

    # FEM-panel watchers (slice slider, AF param sliders) —
    # extracted to golgi.watchers.fem_panel in step 5.2.
    _watchers.fem_panel.register(
        state, geom=geom,
        elec_sync_guard=_elec_sync_guard,
        refresh_fem_plots=_refresh_fem_plots,
    )

    # σ value + preset watchers — extracted to
    # golgi.watchers.sigma in step 5.2.
    _watchers.sigma.register(
        state,
        default_sigma=DEFAULT_SIGMA,
        sigma_match_label=_sigma_match_label,
        sigma_preset_lookup=_sigma_preset_lookup,
        active_project_out_dir=lambda: get_active().out_dir,
    )

    # Cole-Cole evaluator watchers — extracted to
    # golgi.watchers.cole_cole in step 5.2.
    _watchers.cole_cole.register(
        state,
        cole_cole_sigma=cole_cole_sigma,
        cole_cole_presets=COLE_COLE_PRESETS,
    )


    # ---- Electrode designer (ASCENT) handlers ----
    # Pipeline:
    #   user picks preset → _on_cuff_preset_change populates the
    #     cuff_p_<name> state vars from the resolved namespace
    #   user moves a slider → cuff_p_<name> changes → unified
    #     watcher calls _rebuild_cuff_preview() → designer plotter
    #     gets rebuilt actors via _update_designer_plotter()
    #   user clicks Apply → _apply_cuff_design_to_viewport() drops
    #     the designed parts into the main viewport at the cuff frame
    #   user clicks Clear (Electrodes drawer) → _clear_cuff_actors()

    # Guard used by the populate-state path to silence the watcher
    # while we're programmatically seeding cuff_p_<name> values
    # from a freshly-loaded preset.
    _cuff_designer_guard = {"populating": False}

    def _cuff_ns_extras(
        r_nerve_m: float | None = None,
    ) -> dict:
        """Seed values for the DUKE preset's expression namespace.
        Cuff is rendered in the cuff-local frame (z = cuff axis,
        origin at z=0), so z_nerve = 0 keeps the (z_nerve/2)-style
        expressions centred.

        `r_nerve_m` (F3.2c fix): the inner radius the preset
        should be evaluated against. Pass the DESIGN's refit
        R_ci_m so the rendered contact positions track the
        actual mesh that's being solved. When None, falls back
        to geom.R_ci (the currently-loaded mesh) — fine for the
        interactive designer-dialog preview, WRONG for multi-
        design FEM solves where the loaded mesh's R_ci may
        differ from the design under solve.

        Why this matters: the DUKE preset's contact placements
        are parametric in r_n / r_nerve (e.g. `z = r_n * 0.5`).
        If r_n doesn't match the silicone wall radius in the
        mesh, the patches' (z, phi) bounds end up off the
        cuff inner wall → solve_nerve reports 'No facets
        matched any patch'."""
        if r_nerve_m is not None and float(r_nerve_m) > 0.0:
            r_nerve = float(r_nerve_m)
        else:
            r_nerve = (
                float(geom.R_ci) if geom.R_ci else 1.5e-3
            )
        return {
            "z_nerve": 0.0,
            "r_nerve": r_nerve,
            "r_n": r_nerve,
        }

    def _cuff_current_visible() -> list:
        """Slider-row metadata for the currently-loaded preset
        (empty if no preset matches the code map)."""
        preset = _CUFF_PRESETS.get(str(state.cuff_preset_name), None)
        if preset is None:
            return []
        code = str(preset.get("code", ""))
        return cuff_designer.DESIGNER_VISIBLE_PARAMS.get(code, [])

    def _collect_cuff_overrides() -> dict:
        """Build the {name: SI_value} override dict from the
        currently-visible slider state vars. Display units are
        converted to SI via cuff_designer.DISPLAY_UNIT_TO_SI."""
        overrides: dict = {}
        for vp in _cuff_current_visible():
            name = vp["name"]
            try:
                val_disp = float(state[f"cuff_p_{name}"])
            except Exception:
                continue
            factor = cuff_designer.DISPLAY_UNIT_TO_SI.get(
                vp.get("unit", ""), 1.0,
            )
            overrides[name] = val_disp * factor
        return overrides

    def _populate_cuff_visible_state() -> None:
        """Seed the cuff_p_<name> sliders from the currently-selected
        preset's resolved namespace. Wrapped in the populating
        guard so the per-slider watcher doesn't trigger a redundant
        re-render during the batch write."""
        preset = _CUFF_PRESETS.get(str(state.cuff_preset_name), None)
        if preset is None:
            return
        all_params = (
            list(preset.get("params", []))
            + list(preset.get("local_params", []))
        )
        ns = cuff_designer.resolve_params(
            all_params, _cuff_ns_extras(),
        )
        _cuff_designer_guard["populating"] = True
        try:
            with state:
                state.cuff_preset_code = str(preset.get("code", ""))
                for vp in (cuff_designer.DESIGNER_VISIBLE_PARAMS
                                .get(state.cuff_preset_code, [])):
                    name = vp["name"]
                    val_si = float(ns.get(name, 0.0))
                    factor = cuff_designer.DISPLAY_UNIT_TO_SI.get(
                        vp.get("unit", ""), 1.0,
                    )
                    state[f"cuff_p_{name}"] = val_si / factor
        finally:
            _cuff_designer_guard["populating"] = False

    def _update_designer_plotter(parts: list) -> None:
        """Clear pl_cuff and mount the freshly-rendered design.
        Adds a translucent nerve cylinder for context."""
        try:
            pl_cuff.clear()
        except Exception:
            pass
        if not parts:
            try:
                ctrl.view_cuff_update()
            except Exception:
                pass
            return
        r_nerve = _cuff_ns_extras()["r_nerve"]
        if r_nerve > 0:
            z_min = min(m.bounds[4] for _, _, m, _ in parts)
            z_max = max(m.bounds[5] for _, _, m, _ in parts)
            pad = 0.25 * max(z_max - z_min, 1.0e-3)
            nerve_cyl = pv.Cylinder(
                center=(0.0, 0.0, 0.5 * (z_min + z_max)),
                direction=(0.0, 0.0, 1.0),
                radius=r_nerve,
                height=(z_max - z_min) + 2.0 * pad,
                resolution=64, capping=True,
            )
            pl_cuff.add_mesh(
                nerve_cyl, color="#1f1240", opacity=0.32,
                show_edges=False, smooth_shading=True,
                name="designer_nerve",
            )
        for idx, (inst_label, sub_label, mesh, role) in enumerate(
            parts,
        ):
            color = cuff_designer.ROLE_COLORS.get(
                role, (0.7, 0.7, 0.7),
            )
            opacity = cuff_designer.ROLE_OPACITIES.get(role, 1.0)
            pl_cuff.add_mesh(
                mesh, color=color, opacity=opacity,
                show_edges=False, smooth_shading=True,
                specular=0.40, specular_power=15.0,
                name=f"designer_part_{idx}",
            )
        try:
            pl_cuff.reset_camera()
            ctrl.view_cuff_reset_camera()
        except Exception:
            pass
        try:
            ctrl.view_cuff_update()
        except Exception:
            pass

    def _rebuild_cuff_preview() -> None:
        """Re-resolve overrides + re-render the designer plotter.
        Fired on preset change, on any slider edit, and on dialog
        open."""
        preset = _CUFF_PRESETS.get(str(state.cuff_preset_name), None)
        if preset is None:
            state.cuff_designer_status = "No preset selected."
            _update_designer_plotter([])
            return
        try:
            parts = cuff_designer.render_design(
                preset,
                param_overrides=_collect_cuff_overrides(),
                ns_extras=_cuff_ns_extras(),
            )
        except Exception as ex:
            state.cuff_designer_status = (
                f"⚠ render failed: {type(ex).__name__}: {ex}"
            )
            print(f"[cuff_designer] {ex}", flush=True)
            _update_designer_plotter([])
            return
        _update_designer_plotter(parts)
        geom._cuff_designer_parts = parts
        state.cuff_designer_status = (
            f"✓ {len(parts)} part(s) — drag to inspect. "
            f"Apply mounts the design in the workspace viewport."
        )

    def do_open_cuff_designer():
        """Open the ASCENT cuff designer scoped to the currently-
        selected electrode. Loads that electrode's stored
        duke_preset + duke_overrides (if any) into the dialog
        state, otherwise seeds with the first available preset
        so the user lands on a rendered cuff immediately."""
        sel_id = str(state.selected_design_id or "")
        target = _find_design(sel_id)
        if target is None:
            # No electrode selected → nothing to apply to. Just
            # don't open the dialog.
            return
        # Load preset + overrides from the electrode's dict. If
        # the electrode has no preset chosen yet, fall back to
        # the first one alphabetically.
        stored_preset = str(target.get("duke_preset", "") or "")
        if (stored_preset
                and stored_preset in _CUFF_PRESETS):
            state.cuff_preset_name = stored_preset
        elif _CUFF_PRESETS:
            state.cuff_preset_name = sorted(
                _CUFF_PRESETS.keys(),
            )[0]
        # Seed slider values from defaults first (via populate),
        # THEN overlay the per-electrode override dict so the
        # user sees this electrode's tuned values.
        _populate_cuff_visible_state()
        stored_over = dict(target.get("duke_overrides", {}) or {})
        if stored_over:
            _cuff_designer_guard["populating"] = True
            try:
                with state:
                    for k, v in stored_over.items():
                        state_key = f"cuff_p_{k}"
                        try:
                            state[state_key] = float(v)
                        except Exception:
                            pass
            finally:
                _cuff_designer_guard["populating"] = False
        state.show_cuff_designer_dialog = True
        _rebuild_cuff_preview()
        # The synchronous reset_camera inside _rebuild_cuff_preview
        # fires BEFORE the dialog has actually mounted on the
        # client (state changes batch + push after this function
        # returns), so the client-side WebGL canvas isn't there
        # yet to receive the camera-reset message. Schedule a
        # deferred reset so the user lands on a fitted view
        # instead of having to click reset-camera every time.
        async def _deferred_designer_reset():
            # Two short ticks: one for the dialog mount, another
            # for the plotter view to attach its WebGL context.
            await asyncio.sleep(0.20)
            try:
                ctrl.view_cuff_update()
            except Exception:
                pass
            try:
                ctrl.view_cuff_reset_camera()
            except Exception:
                pass
        asyncio.create_task(_deferred_designer_reset())

    def do_close_cuff_designer():
        state.show_cuff_designer_dialog = False

    # W1.7c: cuff-designer dialog watchers (preset picker +
    # per-param sliders) moved to golgi.watchers.cuff_designer.
    # Both closures (_populate_cuff_visible_state and
    # _rebuild_cuff_preview) are defined above this line, so the
    # kwarg pass resolves cleanly at call time.
    _watchers.cuff_designer.register(
        state,
        cuff_visible_names=_CUFF_ALL_VISIBLE_NAMES,
        populate_cuff_visible_state=_populate_cuff_visible_state,
        rebuild_cuff_preview=_rebuild_cuff_preview,
        cuff_designer_guard=_cuff_designer_guard,
    )

    def _clear_cuff_actors() -> None:
        """Remove every designer-created actor + the legacy preview
        actors from the main plotter. Used both when the user clicks
        Clear and when applying a new design to start from a clean
        slate."""
        pl.remove_actor("silicone_overlay", reset_camera=False)
        pl.remove_actor("saline_overlay", reset_camera=False)
        pl.remove_actor("muscle_overlay", reset_camera=False)
        for _i in range(64):
            pl.remove_actor(
                f"gold_overlay_{_i}", reset_camera=False,
            )
        for _i in range(256):
            pl.remove_actor(
                f"designer_part_{_i}", reset_camera=False,
            )
        state.has_designer_cuff = False

    def _save_design_to_selected_electrode() -> None:
        """Persist the current designer dialog state (preset name +
        slider overrides) onto the SELECTED electrode's dict so
        the next render pass picks it up. Also flips the
        electrode's type to "DUKE Cuff designer" so downstream
        logic routes it through the cuff_designer render path.

        Overrides are converted from DISPLAY units (mm / deg —
        what the slider holds) to SI (m / rad) before saving.
        The render path treats `duke_overrides` as SI directly;
        without this conversion a `1.5 mm` slider was being
        written as `1.5` and then re-fed as `1.5 m`, blowing
        the cuff up by ~1000× in the main viewport (the dialog
        worked because its render call routes through
        `_collect_cuff_overrides` which already does the unit
        conversion)."""
        sel_id = str(state.selected_design_id or "")
        if not sel_id:
            return
        electrodes = list(state.designs or [])
        target_e = None
        for idx, e in enumerate(electrodes):
            if e.get("eid") != sel_id:
                continue
            # Replace with a NEW dict so the client-side Vue list
            # sees a reference change for this row and re-renders
            # the `{{ elec.electrode_type }}` / `duke_preset`
            # binding immediately. (In-place mutation kept the
            # same object reference, so the list label stayed at
            # the previous value until project reopen.)
            new_e = dict(e)
            new_e["electrode_type"] = DUKE_ELECTRODE_TYPE
            new_e["duke_preset"] = str(state.cuff_preset_name or "")
            # Reuse the dialog's own collect-with-unit-conversion
            # helper so the values end up in the same SI shape
            # the renderer expects.
            new_e["duke_overrides"] = _collect_cuff_overrides()
            # Recompute polarities to match the new preset's
            # contact count — `_ensure_polarities` mutates the
            # dict in place when the list is empty / stale.
            _ensure_polarities(new_e)
            electrodes[idx] = new_e
            target_e = new_e
            break
        state.designs = electrodes
        # Force-mirror the polarity list + count onto the legacy
        # state vars BEFORE the watcher cascade. This way the
        # drawer's polarity table shows up on the same tick the
        # designer dialog closes, instead of waiting for a
        # follow-up open/close round trip. If electrode_type
        # was already DUKE (preset swap on a DUKE-typed
        # electrode), `_on_electrode_change` wouldn't fire and
        # the mirror would stay stale.
        if target_e is not None:
            _elec_sync_guard["loading"] = True
            try:
                pols = list(target_e.get(
                    "contact_polarities", [],
                ))
                state.contact_polarities = pols
                state.contact_count = len(pols)
            finally:
                _elec_sync_guard["loading"] = False
        # Also mirror to legacy state so the watcher cascade
        # doesn't fight us — set the active mirror to DUKE.
        state.electrode_type = DUKE_ELECTRODE_TYPE

    def do_apply_cuff_design():
        """Save the design into the selected electrode + trigger
        a refit so the new geometry lands in the viewport."""
        _save_design_to_selected_electrode()
        state.show_cuff_designer_dialog = False
        if geom.nerve is not None:
            asyncio.create_task(do_fit_cuff(refit=False))

    def do_clear_electrodes():
        """Remove every electrode + wipe every cuff-related actor
        from the viewport. Also closes the selected-electrode
        slot. Equivalent to clicking ✕ on each row, but in one
        action so the user can start the cuff design from a
        clean slate."""
        for actor_name in list(pl.actors.keys()):
            if actor_name.startswith("elec_"):
                pl.remove_actor(actor_name, reset_camera=False)
        # Legacy single-cuff overlays (in case any survive).
        pl.remove_actor("silicone_overlay", reset_camera=False)
        pl.remove_actor("saline_overlay", reset_camera=False)
        for _i in range(64):
            pl.remove_actor(
                f"gold_overlay_{_i}", reset_camera=False,
            )
        for _i in range(256):
            pl.remove_actor(
                f"designer_part_{_i}", reset_camera=False,
            )
        state.designs = []
        state.selected_design_id = ""
        state.has_designer_cuff = False
        # Scene-state pipeline: empty electrodes list → no
        # per-electrode groups; `_retire_unknown_actors` sweeps any
        # lingering `elec_*` actors on the next render pass.
        _request_render()
        safe_update()


    def _update_muscle_preview() -> None:
        """Translucent muscle cylinder shown while the Mesh drawer
        is open — gives the user a live preview of where the muscle
        bounding box will sit before they Build. Auto-fits to the
        full nerve bbox in cuff frame (NOT just the cuff axial
        window), so the preview matches what `assemble_multi_
        domain_plc` actually builds.

        F3.2-M2.1a — pre-fit fallback: when no design has been
        placed yet (`geom.pts_cuff` is None) we fit the bbox in
        PCA-aligned frame instead so the cylinder shows up
        immediately after Load Geometry. Same frame the raw
        nerve actor + epi preview use pre-fit, so the three
        stay visually aligned."""
        if geom.nerve is None:
            return
        # F3.2-M3 — gate on the one-way unlock flag. The flag
        # flips True the first time the user reaches Step 4 of
        # the import stepper, then stays True. Before that, no
        # cylinder is rendered — keeps the viewport tidy while
        # the user is still walking through earlier steps.
        if not bool(getattr(state, "muscle_preview_unlocked",
                            False)):
            return

        # Vuetify number-inputs send the empty string "" while the
        # user is mid-edit (cleared field). float("") raises and
        # crashes the watcher; treat any unparseable / empty value
        # as the field's stored default (or 0.0 fallback).
        def _f(key: str, fallback: float = 0.0) -> float:
            v = getattr(state, key, fallback)
            if v is None or v == "":
                return float(fallback)
            try:
                return float(v)
            except (TypeError, ValueError):
                return float(fallback)

        # F3.2-M2.1f — render in pure PCA frame to match the
        # nerve actor / fibers / meshes / electrodes. In PCA
        # frame the nerve's xy-centroid is at (0,0) by
        # construction, so `r_max = norm(pts[:, :2])` measures
        # the radius from the cylinder axis correctly without
        # any centroid subtraction.
        if (geom.centroid is not None
                and geom.R_global is not None):
            pts = (
                (np.asarray(
                    geom.nerve["pts_raw"], dtype=np.float64,
                ) - geom.centroid) @ geom.R_global
            )
            xy_centre = np.zeros(2, dtype=np.float64)
        else:
            # Pre-PCA fallback. Centre on nerve's xy centroid
            # so the cylinder still wraps the nerve.
            pts = geom.nerve["pts_raw"]
            xy_centre = np.asarray(
                pts[:, :2].mean(axis=0), dtype=np.float64,
            )
        r_max = float(np.linalg.norm(
            pts[:, :2] - xy_centre, axis=1,
        ).max())
        z_min = float(pts[:, 2].min())
        z_max = float(pts[:, 2].max())
        R_mus = r_max + _f("muscle_radial_pad_mm") * 1e-3
        z_lo = (z_min
                 - _f("muscle_axial_pad_mm") * 1e-3
                 + _f("muscle_dz_mm") * 1e-3)
        z_hi = (z_max
                 + _f("muscle_axial_pad_mm") * 1e-3
                 + _f("muscle_dz_mm") * 1e-3)
        L_mus = z_hi - z_lo
        z_centre = 0.5 * (z_lo + z_hi)
        cx = float(xy_centre[0]) + _f("muscle_dx_mm") * 1e-3
        cy = float(xy_centre[1]) + _f("muscle_dy_mm") * 1e-3
        render_muscle_preview(
            pl, R_mus * 1000.0, L_mus * 1000.0,
            cx * 1000.0, cy * 1000.0, z_centre * 1000.0,
        )
        safe_update()

    def _remove_muscle_overlay() -> None:
        """Strip the translucent muscle preview from the viewport.
        Originally invoked when the Mesh drawer closed; as of
        F3.2-M2.1a it's also called when `vis_muscle_preview`
        flips false or once any design has a meshed nerve.msh
        (the per-design Tissues > Muscle row then owns
        visibility)."""
        pl.remove_actor("muscle_overlay", reset_camera=False)
        safe_update()

    def _any_design_has_mesh() -> bool:
        """True if at least one design in `state.designs` has its
        `has_mesh` flag set. Kept for back-compat but the preview
        gates now use `_focused_design_has_mesh()` so a new
        unmeshed design can still see the preview as a placement
        backdrop even after another design is meshed."""
        try:
            for d in list(state.designs or []):
                if bool(d.get("has_mesh")):
                    return True
        except (TypeError, AttributeError):
            pass
        return False

    def _focused_design_has_mesh() -> bool:
        """True iff the design currently picked in the combobox
        (`selected_design_id`) has its `has_mesh` flag set. Used
        by the pre-mesh preview gates (raw nerve, epi shell,
        muscle bbox) so they stay visible whenever the FOCUSED
        design hasn't been meshed yet — even if another design
        already has a mesh. Returns False when no design is
        focused (legend pre-design state)."""
        focused_eid = str(state.selected_design_id or "")
        if not focused_eid:
            return False
        try:
            for d in list(state.designs or []):
                if str(d.get("eid", "")) == focused_eid:
                    return bool(d.get("has_mesh", False))
        except (TypeError, AttributeError):
            pass
        return False

    def _update_epi_preview() -> None:
        """Translucent inward-offset shell shown while the nerve is
        loaded so the user can see what `epi_thickness_um` is doing
        without having to build a mesh. Skipped silently when
        `use_epi` is false, the nerve isn't loaded, or any design
        already owns a meshed epi region (per-design Tissues > Epi
        row takes over). Pure geometric op — no pymeshfix, so the
        shell may have self-intersections in concave VN regions.
        Visually fine as a translucent preview; the real epi shell
        is built later with pymeshfix repair by the mesh pipeline."""
        if not state.use_epi:
            _remove_epi_overlay()
            return
        if geom.nerve is None or _focused_design_has_mesh():
            _remove_epi_overlay()
            return
        try:
            thickness_um = float(state.epi_thickness_um)
        except (TypeError, ValueError):
            return
        if thickness_um <= 0.0:
            return
        # F3.2-M2.1f — render in pure PCA frame to match the
        # nerve actor / fibers / meshes / electrodes. Falls back
        # to raw when PCA basis isn't ready yet (shouldn't
        # happen post-Load Geometry, but defensive).
        if (geom.centroid is not None
                and geom.R_global is not None):
            pts_for_render = (
                (np.asarray(
                    geom.nerve["pts_raw"], dtype=np.float64,
                ) - geom.centroid) @ geom.R_global
            )
        else:
            pts_for_render = geom.nerve["pts_raw"]
        pts_mm = np.asarray(
            pts_for_render, dtype=np.float64,
        ) * 1000.0
        tris = geom.nerve["boundary_raw"]
        n_tris = len(tris)
        faces = np.empty(n_tris * 4, dtype=np.int64)
        faces[0::4] = 3
        faces[1::4] = tris[:, 0]
        faces[2::4] = tris[:, 1]
        faces[3::4] = tris[:, 2]
        render_epi_preview(
            pl, pts_mm, faces, thickness_um * 1e-3,
        )
        safe_update()

    def _remove_epi_overlay() -> None:
        """Strip the translucent epi preview from the viewport.
        Called when `vis_epi_preview` flips false, `use_epi` flips
        false, or once any design has a meshed nerve.msh."""
        pl.remove_actor("epi_overlay", reset_camera=False)
        safe_update()

    # Render-toggle watchers — registered here (NOT earlier in
    # build_app) so the two muscle-preview closures above are
    # already defined by the time we pass them as kwargs.
    # See migration.md §B "Vue templates bind callables at template-
    # build time" — same issue, different angle: any name passed
    # as a kwarg is resolved at call time, so order matters.
    _watchers.render_toggles.register(
        state, geom=geom,
        elec_sync_guard=_elec_sync_guard,
        request_render=_request_render,
        ensure_field_lines_async=_ensure_field_lines_async,
        # W1.7a: per-region + per-fiber-branch visibility mirrors.
        region_tags=list(DEFAULTS.keys()) + [TAG_GOLD],
        max_fiber_branches=MAX_FIBER_BRANCHES,
        # W1.7b: mesh-panel show/edges/quality + muscle-pad
        # multi-key watchers.
        palette_edges_key=f"{pl._id_name}_edge_visibility",
        update_muscle_preview=_update_muscle_preview,
        remove_muscle_overlay=_remove_muscle_overlay,
        # F3.2-M2.1a — pre-mesh epi preview lifecycle.
        update_epi_preview=_update_epi_preview,
        remove_epi_overlay=_remove_epi_overlay,
    )

    # F3.2-M2.1a — top-level Nerve section toggle. Flipping
    # `vis_nerve_raw` from the legend re-folds the nerve group
    # so the raw endoneurium actor shows/hides accordingly.
    # Once any design has a meshed nerve.msh, _set_nerve_group
    # auto-suppresses the actor anyway, so this toggle becomes
    # a no-op at that point (the per-design Tissues row takes
    # over visibility control).
    @state.change("vis_nerve_raw", "use_epi", "vis_epi_preview")
    def _on_vis_nerve_raw_change(**_kwargs):
        # F3.2-M3: `use_epi` also triggers a refold because the
        # nerve fold below re-skins the raw nerve actor — opaque
        # indigo (endo) when no epi shell, cream semi-transparent
        # (epi outer) when use_epi=True — so the user can see the
        # endo through the shell during Step 2 of the wizard.
        # `vis_epi_preview` is added for the µCT-bundle path,
        # where the nerve fold renders the epi shell at low
        # opacity and gates its visibility on `vis_epi_preview`
        # (so the Epineurium legend row drives it independently
        # from the fascicle actors under `vis_nerve_raw`).
        _request_render()


    # Drawer mutual-exclusion + viewport-mode swap watcher +
    # do_close_all_tabs — extracted to golgi.watchers.drawer_exclusion
    # in step 5.2. register() returns the controller action so the
    # navbar can bind to it.
    _drawer_x = _watchers.drawer_exclusion.register(state)
    do_close_all_tabs = _drawer_x["do_close_all_tabs"]


    # ================================================================
    # Scene-state builders + the real _rebuild_scene_state.
    # ----------------------------------------------------------------
    # These pure functions read `geom` + `state` and produce
    # SceneGroups that the renderer consumes. They never touch `pl`.
    # ================================================================

    def _phong_style(spec: dict) -> dict:
        """Standard add_mesh kwargs for a phong-shaded surface from
        a DEFAULTS / GOLD / saline-style spec."""
        return dict(
            color=spec["color"],
            opacity=spec["opacity"],
            pbr=False,
            ambient=spec["ambient"],
            diffuse=spec["diffuse"],
            specular=spec["specular"],
            specular_power=spec["specular_power"],
            smooth_shading=True,
            show_edges=False,
        )

    def _polyline_polydata_from_paths(paths_mm: list) -> "pv.PolyData":
        """Concatenate a list of (N_i, 3) polylines into one PolyData
        with explicit line cells. Used by the fiber palette builder."""
        if not paths_mm:
            return pv.PolyData()
        pts_chunks = []
        cells_chunks = []
        offset = 0
        for p in paths_mm:
            n = int(p.shape[0])
            if n < 2:
                continue
            pts_chunks.append(p)
            cells_chunks.append(
                np.concatenate([[n], np.arange(n) + offset]),
            )
            offset += n
        if not pts_chunks:
            return pv.PolyData()
        return pv.PolyData(
            np.vstack(pts_chunks).astype(np.float64),
            lines=np.concatenate(cells_chunks).astype(np.int64),
        )

    # SceneCatalog Phase 2 (cut over) + Phase 6a (inline
    # retired) + Phase 6b (focused-design gate) — the nerve
    # entry. Build pure-PCA-frame PolyData for the raw
    # endoneurium surface, with quality-colouring on demand.
    #
    # Suppressed only when the FOCUSED design (combobox
    # selection) has a meshed nerve — its meshed endo region
    # (tag 1) takes over rendering for that design's view.
    # Other designs may also be meshed; their region actors
    # render in parallel. When the focused design is NOT yet
    # meshed (e.g. user just added cuff 2), the raw nerve
    # stays visible as a placement backdrop even though
    # cuff 1 might already be meshed.
    def _catalog_fold_nerve(geom, state, ctx) -> dict | None:
        if geom.nerve is None:
            return None
        focused_eid = str(state.selected_design_id or "")
        focused = next(
            (d for d in (state.designs or [])
             if str(d.get("eid", "")) == focused_eid),
            None,
        )
        if focused is not None and bool(
            focused.get("has_mesh", False),
        ):
            return None
        # Bundle preview: the combined viz buffer (pts_raw +
        # boundary_raw) mashes the epi shell together with every
        # fascicle into one mesh, which renders as a solid blob.
        # For the workspace viewport we want the epi semi-
        # transparent so the user sees the fascicles inside —
        # those are emitted as separate actors by the regions
        # fold below. Source ONLY the epi here.
        _is_bundle = (
            geom.nerve.get("kind") == "uct_bundle"
            and isinstance(geom.nerve.get("bundle"), dict)
            and isinstance(
                geom.nerve["bundle"].get("epi"), dict,
            )
        )
        if _is_bundle:
            _epi = geom.nerve["bundle"]["epi"]
            src_pts = np.asarray(_epi["verts_m"], dtype=np.float64)
            tris = np.asarray(_epi["faces"], dtype=np.int64)
        else:
            src_pts = np.asarray(
                geom.nerve["pts_raw"], dtype=np.float64,
            )
            tris = geom.nerve["boundary_raw"]
        pts_pca_mm = _to_pca_mm(
            src_pts,
            source_frame="raw",
            source_units="m",
            ctx=ctx,
        )
        n_tris = len(tris)
        faces = np.empty(n_tris * 4, dtype=np.int64)
        faces[0::4] = 3
        faces[1::4] = tris[:, 0]
        faces[2::4] = tris[:, 1]
        faces[3::4] = tris[:, 2]
        # Quality colouring only makes sense for the legacy STL
        # path where `nerve_q` was computed against the same
        # boundary_raw we're rendering. Skip it for bundles
        # (the per-triangle quality there refers to the combined
        # buffer, not the epi-only mesh we're rendering now).
        use_q = (bool(state.show_quality_color)
                 and not _is_bundle
                 and geom.nerve_q is not None
                 and len(geom.nerve_q) == n_tris)
        poly = pv.PolyData(pts_pca_mm, faces)
        if use_q:
            poly.cell_data["q_radius_ratio"] = np.asarray(
                geom.nerve_q, dtype=np.float32,
            )
            style = dict(
                scalars="q_radius_ratio",
                cmap="RdYlGn",
                clim=(0.0, 1.0),
                opacity=1.0,
                show_edges=False,
                smooth_shading=False,
                show_scalar_bar=False,
            )
        else:
            poly = poly.compute_normals(
                cell_normals=False, point_normals=True,
                consistent_normals=True,
                auto_orient_normals=True,
                non_manifold_traversal=False,
            )
            # F3.2-M3: when `use_epi=True`, the raw nerve actor's
            # outer surface IS the OUTER epi boundary in the
            # eventual mesh (the epi shell is an inward offset
            # carved out of this surface). Render it as the
            # epineurium (cream, semi-transparent at 0.7) so the
            # inward-offset preview actor — which represents the
            # endo outer boundary — shows through it as indigo
            # opaque. When `use_epi=False`, there's no shell, so
            # the raw nerve IS the endo: render indigo opaque.
            # Bundle imports always carry a real epi surface from
            # segmentation, so they get epi styling with a lower
            # opacity (0.35) so the fascicle actors inside read
            # clearly through the shell.
            if _is_bundle:
                style = _phong_style(DEFAULTS[5])
                style["opacity"] = 0.35
            elif bool(state.use_epi):
                style = _phong_style(DEFAULTS[5])
                style["opacity"] = 0.7
            else:
                style = _phong_style(DEFAULTS[1])
        # Bundle mode: the nerve actor IS the epi shell, so its
        # visibility belongs to the Epineurium legend row. Routing
        # the bundle nerve actor through `vis_epi_preview` lets the
        # user toggle the translucent epi shell independently from
        # the fascicle actors (which `_catalog_fold_regions` mounts
        # under `vis_nerve_raw`). Non-bundle mode keeps the legacy
        # `vis_nerve_raw` gate — that path renders the endoneurium
        # surface (or the cream epi-preview offset), not a separate
        # epi shell.
        _vis = bool(
            state.vis_epi_preview if _is_bundle
            else state.vis_nerve_raw
        )
        return _mkgrp(
            payload=poly,
            style=style,
            visible=_vis,
            signature=_next_sig(),
        )

    # M48 — label changed from "Endoneurium (raw)" to "Nerve".
    # For bundle imports this entry is the epi shell (rendered
    # semi-transparent so the per-fascicle endo actors show
    # through), so calling it "Endoneurium" was confusing —
    # the user saw "Endoneurium" in the legend and thought the
    # epi wasn't there. "Nerve" reads correctly for both
    # cases: in bundle mode it's the outer epi surface, in
    # STL mode it's the single closed nerve mesh.
    _scene_catalog.register(_SceneEntry(
        section="nerve",
        key="nerve",
        fold=_catalog_fold_nerve,
        label="Nerve",
    ))

    # SceneCatalog Phase 4 — register the regions entry. Whole-
    # section fold that mirrors `_set_region_groups` exactly:
    # reads `geom.designs_meshes[eid]["region_surfaces"]`
    # (already PCA-frame, mm), applies per-design visibility +
    # Vₑ overlay + mesh-quality overlay styling, builds one
    # SceneGroup per (eid, tag). Structural cut-over only — the
    # PCA-frame assumption on the source data is what the
    # accompanying disk-restore fix (anchor=0) enforces.
    def _catalog_fold_regions(geom, state, ctx) -> dict | None:
        electrodes = list(state.designs or [])
        designs_meshes = geom.designs_meshes or {}
        out_regions: dict = {}
        # Bundle preview — when a µCT bundle is loaded but no
        # design has been meshed yet, emit one preview actor per
        # fascicle (tag-1 endo styling) so the user can see each
        # fascicle through the semi-transparent epi shell that
        # the nerve fold rendered. These actors live in the
        # `regions` namespace under keys like `fascicle_<i>` —
        # safe because real region keys are `{eid}_{tag}`. As
        # soon as a design is meshed, the whole-section reset
        # in `apply_in_place` wipes these previews and the real
        # region surfaces take over.
        if (geom.nerve is not None
                and geom.nerve.get("kind") == "uct_bundle"
                and not designs_meshes
                and geom.region_surfaces is None):
            _bundle = geom.nerve.get("bundle") or {}
            _fascicles = _bundle.get("fascicles") or []
            _spec = DEFAULTS.get(1, {})
            for _fi, _fasc in enumerate(_fascicles):
                _fv_m = np.asarray(
                    _fasc.get("verts_m"), dtype=np.float64,
                )
                _ff = np.asarray(
                    _fasc.get("faces"), dtype=np.int64,
                )
                if _fv_m.size == 0 or _ff.size == 0:
                    continue
                _fv_pca = _to_pca_mm(
                    _fv_m,
                    source_frame="raw",
                    source_units="m",
                    ctx=ctx,
                )
                _nt = int(_ff.shape[0])
                _faces_flat = np.empty(_nt * 4, dtype=np.int64)
                _faces_flat[0::4] = 3
                _faces_flat[1::4] = _ff[:, 0]
                _faces_flat[2::4] = _ff[:, 1]
                _faces_flat[3::4] = _ff[:, 2]
                _poly = pv.PolyData(_fv_pca, _faces_flat)
                _poly = _poly.compute_normals(
                    cell_normals=False, point_normals=True,
                    consistent_normals=True,
                    auto_orient_normals=True,
                    non_manifold_traversal=False,
                )
                _style = _phong_style(_spec)
                _style["opacity"] = 1.0
                out_regions[f"fascicle_{_fi}"] = _mkgrp(
                    payload=_poly,
                    style=_style,
                    visible=bool(state.vis_nerve_raw),
                    signature=_next_sig(),
                )
        if (not designs_meshes
                and geom.region_surfaces is not None):
            _legacy_eid = (
                str(state.selected_design_id or "")
                or "default"
            )
            designs_meshes = {
                _legacy_eid: {
                    "region_surfaces": geom.region_surfaces,
                    "region_surfaces_viz": (
                        geom.region_surfaces_viz
                        or geom.region_surfaces
                    ),
                },
            }
            if not electrodes:
                electrodes = [{"eid": _legacy_eid}]
        if not designs_meshes:
            return out_regions
        active_id = str(state.selected_design_id or "")
        ve_on_surf = (
            bool(state.show_ve_surface)
            and geom.nerve_surface_Ve is not None
        )
        ve_clim = geom.ve_clim_mV
        for elec in electrodes:
            eid = str(elec.get("eid", ""))
            if not eid or eid not in designs_meshes:
                continue
            # Phase 6b — single-focus rendering. Only emit
            # region actors for the FOCUSED design (combobox
            # selection). Non-focused designs' meshes are
            # skipped entirely; they remount instantly when
            # the user switches back via the combobox. Matches
            # the legend's per-focused-design layout: viewport
            # shows what the legend is describing.
            if eid != active_id:
                continue
            mesh_data = designs_meshes[eid]
            region_surfaces = mesh_data.get(
                "region_surfaces",
            ) or {}
            viz_dict = (
                mesh_data.get("region_surfaces_viz")
                or region_surfaces
            )
            if not region_surfaces:
                continue
            v_master = (
                bool(elec.get("vis_master", True))
                and bool(elec.get("vis_mesh", True))
            )
            colour_by_q_mesh = bool(
                elec.get("vis_mesh_quality", False),
            )
            v_endo_e = v_master and bool(
                elec.get("vis_endo", True),
            )
            v_epi_e = v_master and bool(
                elec.get("vis_epi", True),
            )
            v_mus_e = v_master and bool(
                elec.get("vis_muscle", True),
            )
            v_sil_e = v_master and bool(
                elec.get("vis_silicone", True),
            )
            v_sal_e = v_master and bool(
                elec.get("vis_saline", True),
            )
            v_con_e = v_master and bool(
                elec.get("vis_contacts", True),
            )
            # F3.2-M3 — per-design scar / connective tissue
            # visibility. `vis_scar` defaults True; the row is
            # gated on `use_scar` in the legend so the toggle
            # only appears when the design actually has a scar
            # shell in its mesh.
            v_sca_e = v_master and bool(
                elec.get("vis_scar", True),
            )
            _is_active = (eid == active_id)
            _endo_surf = region_surfaces.get(1)
            ve_endo_ok = (
                _is_active and ve_on_surf
                and _endo_surf is not None
                and (len(geom.nerve_surface_Ve)
                     == _endo_surf.n_points)
            )
            _kdtree = None
            _ve_endo = None
            if ve_endo_ok:
                from scipy.spatial import cKDTree
                _ve_endo = np.asarray(
                    geom.nerve_surface_Ve, dtype=np.float32,
                ).copy()
            for tag in [
                t for t in TAG_ORDER
                if t in region_surfaces
            ]:
                surf_full = region_surfaces[tag]
                surf_viz = viz_dict.get(tag, surf_full)
                spec = DEFAULTS.get(tag, DEFAULTS[1])
                if ve_endo_ok and tag in (1, 5):
                    if tag == 1:
                        ve = _ve_endo.copy()
                    else:
                        if _kdtree is None:
                            _kdtree = cKDTree(
                                np.asarray(
                                    _endo_surf.points,
                                    dtype=np.float64,
                                ),
                            )
                        _, _nn = _kdtree.query(
                            np.asarray(
                                surf_full.points,
                                dtype=np.float64,
                            ),
                            k=1,
                        )
                        ve = _ve_endo[_nn].copy()
                    good = np.isfinite(ve)
                    if good.any():
                        ve[~good] = float(
                            np.median(ve[good]),
                        )
                    else:
                        ve[:] = 0.0
                    surf_ve = surf_full.copy()
                    surf_ve.point_data["Ve"] = ve * 1.0e3
                    surf_ve.GetPointData().SetActiveScalars(
                        "Ve",
                    )
                    if ve_clim is None:
                        _good = np.isfinite(ve)
                        if _good.any():
                            ve_mv = ve * 1.0e3
                            v_lo = float(np.percentile(
                                ve_mv[_good], 1.0,
                            ))
                            v_hi = float(np.percentile(
                                ve_mv[_good], 99.0,
                            ))
                        else:
                            v_lo, v_hi = -1.0, 1.0
                        if v_hi - v_lo < 1e-12:
                            v_hi = v_lo + 1.0
                        clim = (v_lo, v_hi)
                    else:
                        clim = ve_clim
                    opacity = 1.0 if tag == 1 else 0.45
                    style = dict(
                        scalars="Ve",
                        cmap="plasma",
                        clim=clim,
                        opacity=opacity,
                        pbr=False,
                        ambient=spec["ambient"],
                        diffuse=spec["diffuse"],
                        specular=spec["specular"],
                        specular_power=spec["specular_power"],
                        show_edges=False,
                        smooth_shading=True,
                        show_scalar_bar=False,
                    )
                    payload = surf_ve
                elif (colour_by_q_mesh
                      and "q_tet" in surf_full.cell_data):
                    style = dict(
                        scalars="q_tet",
                        cmap="RdYlGn", clim=(0.0, 1.0),
                        opacity=spec["opacity"],
                        pbr=False,
                        ambient=spec["ambient"],
                        diffuse=spec["diffuse"],
                        specular=spec["specular"],
                        specular_power=spec["specular_power"],
                        show_edges=False,
                        smooth_shading=True,
                        show_scalar_bar=False,
                    )
                    payload = surf_full
                else:
                    payload = surf_viz.compute_normals(
                        cell_normals=False,
                        point_normals=True,
                        consistent_normals=True,
                        auto_orient_normals=True,
                        non_manifold_traversal=False,
                    )
                    style = _phong_style(spec)
                if tag == 1:
                    v_eff = v_endo_e
                elif tag == 2:
                    v_eff = v_sal_e
                elif tag == 3:
                    v_eff = v_sil_e
                elif tag == 4:
                    v_eff = v_mus_e
                elif tag == 5:
                    v_eff = v_epi_e
                elif tag == TAG_SCAR:
                    v_eff = v_sca_e
                elif tag == TAG_GOLD:
                    v_eff = v_con_e
                else:
                    v_eff = v_master
                style["show_edges"] = bool(
                    state.show_mesh_edges,
                )
                key = f"{eid}_{tag}"
                out_regions[key] = _mkgrp(
                    payload=payload,
                    style=style,
                    visible=v_eff,
                    signature=_next_sig(),
                )
        return out_regions

    _scene_catalog.register(_SceneEntry(
        section="regions",
        key="*",
        fold=_catalog_fold_regions,
        label="Per-design mesh regions",
    ))

    def _fiber_paths_display() -> list | None:
        """Return fiber paths in the viewport's pure-PCA frame
        (metres). None when no fibers exist.

        F3.2-M2.1f — must match `_render_fibers_current_frame`
        and `_set_nerve_group` so fibers stay glued to the nerve
        regardless of cuff-fit state. Pre-FEM, fibers are in raw
        frame and get transformed via (p - centroid) @ R_global.
        Post-FEM (fibers_in_cuff_frame=True), they're already in
        cuff frame and get undone by adding cuff_origin_pca."""
        if geom.fiber_paths_raw is None:
            return None
        if (geom.fibers_in_cuff_frame
                and geom.cuff_origin_pca is not None):
            _off = np.asarray(
                geom.cuff_origin_pca, dtype=np.float64,
            )
            return [
                np.asarray(p, dtype=np.float64) + _off
                for p in geom.fiber_paths_raw
            ]
        if (geom.centroid is not None
                and geom.R_global is not None):
            _c = np.asarray(geom.centroid, dtype=np.float64)
            _R = np.asarray(geom.R_global, dtype=np.float64)
            return [
                (np.asarray(p, dtype=np.float64) - _c) @ _R
                for p in geom.fiber_paths_raw
            ]
        return geom.fiber_paths_raw

    def _catalog_fold_fibers(geom, state, ctx) -> dict | None:
        """Mode-aware fibers fold for the SceneCatalog. The mode rule:
          - "off"        when no paths or master vis_fibers is off
          - "population" when Population tab is active AND a
                         generate has run (overrides ve/palette)
          - "ve"         when show_ve_fibers and Vₑ data is
                         consistent
          - "palette"    otherwise.

        Returns a dict shaped `{"mode", "branches", "ve",
        "pop_types"}`. `_assign("*", "fibers")` merges it onto the
        existing `_scene_state["fibers"]` so `selected` (owned by
        `_apply_fiber_selection_highlight`) survives the rebuild.

        Actor namespace is mode-disjoint (no `fiber_branch_0`
        ambiguity) — empty branches in non-palette modes drop their
        actors via the renderer's `_retire_unknown_actors` sweep."""
        fg: dict = {
            "mode": "off",
            "branches": {
                i: _mkgrp() for i in range(MAX_FIBER_BRANCHES)
            },
            "ve": _mkgrp(),
            "pop_types": {},
        }
        master = bool(state.vis_fibers)
        paths_display = _fiber_paths_display()
        if paths_display is None or len(paths_display) == 0:
            return fg
        # Population mode — engaged when the Population tab is
        # open AND a generate has happened. One actor per
        # NAMED ROW (`fiber_pop_rows`), each with the row's
        # tab10 colour. Two rows that share a fiber model still
        # get DIFFERENT colours / actors so the viewport reads
        # the design split directly.
        is_pop = (
            str(state.active_analysis) == "population"
            and bool(state.pop_generated)
            and geom.fiber_pop_rows is not None
            and len(geom.fiber_pop_rows) == len(paths_display)
        )
        if is_pop:
            paths_mm = [
                np.asarray(p, dtype=np.float64) * 1000.0
                for p in paths_display
            ]
            pop_rows_arr = np.asarray(
                geom.fiber_pop_rows, dtype=object,
            )
            row_meta = dict(state.pop_row_meta or {})
            row_visible = dict(state.pop_row_visible or {})
            fg["mode"] = "population"
            pop_groups: dict = {}
            unique_rids = sorted({
                str(r) for r in pop_rows_arr
                if r and str(r).strip()
            })
            # Hidden rows render with a neutral grey + lower
            # opacity so the user can still see WHERE those
            # fibers run; only their type-colour gets stripped.
            # This matches the user's spec: "toggle the
            # coloring of fiber types on and off, otherwise
            # the fiber trajectories should be gray".
            grey_color = "#9aa3ad"
            for rid in unique_rids:
                r_mask = np.array(
                    [str(x) == rid for x in pop_rows_arr],
                    dtype=bool,
                )
                r_paths = [paths_mm[k]
                           for k in np.where(r_mask)[0]]
                poly = _polyline_polydata_from_paths(r_paths)
                meta = row_meta.get(rid, {})
                is_visible = bool(
                    row_visible.get(rid, True),
                )
                if is_visible:
                    color = str(meta.get("color", "#666"))
                    opacity = 0.95
                    line_width = 3
                else:
                    color = grey_color
                    opacity = 0.30
                    line_width = 2
                if poly.n_points == 0:
                    pop_groups[rid] = _mkgrp()
                    continue
                pop_groups[rid] = _mkgrp(
                    payload=poly,
                    style=dict(
                        color=color,
                        line_width=line_width,
                        opacity=opacity,
                        show_scalar_bar=False,
                        render_lines_as_tubes=True,
                    ),
                    visible=master,
                    signature=_next_sig(),
                )
            fg["pop_types"] = pop_groups
            return fg
        # (pop_types already initialised to {} above; no need to
        # clear stale entries — the fold starts fresh each call.)
        # Ve-mode preconditions: show_ve_fibers AND geom.fiber_paths_Ve
        # exists AND len matches.
        want_ve = (bool(state.show_ve_fibers)
                   and geom.fiber_paths_Ve is not None
                   and len(geom.fiber_paths_Ve) == len(paths_display))
        if want_ve:
            # Merge all paths into one tube mesh with per-point Vₑ
            # scalar. NaNs (off-mesh samples) → median.
            pts_chunks = []
            cells_chunks = []
            ve_chunks = []
            offset = 0
            for p, ve in zip(paths_display, geom.fiber_paths_Ve):
                p_mm = np.asarray(p, dtype=np.float64) * 1000.0
                n = int(p_mm.shape[0])
                if n < 2 or len(ve) != n:
                    continue
                pts_chunks.append(p_mm)
                cells_chunks.append(
                    np.concatenate([[n], np.arange(n) + offset]),
                )
                ve_chunks.append(np.asarray(ve, dtype=np.float32))
                offset += n
            if not pts_chunks:
                fg["mode"] = "off"
                return fg
            ve_pts = np.concatenate(ve_chunks).astype(np.float32)
            good = np.isfinite(ve_pts)
            if good.any():
                ve_pts[~good] = np.float32(
                    float(np.median(ve_pts[good])),
                )
            else:
                ve_pts[:] = 0.0
            poly = pv.PolyData(
                np.vstack(pts_chunks).astype(np.float64),
                lines=np.concatenate(cells_chunks).astype(np.int64),
            )
            poly.point_data["Ve"] = ve_pts
            poly.GetPointData().SetActiveScalars("Ve")
            tube = poly.tube(radius=0.03, n_sides=10, capping=False)
            tube.point_data["Ve"] = (
                tube.point_data["Ve"].astype(np.float32) * 1.0e3
            )
            tube.GetPointData().SetActiveScalars("Ve")
            if geom.ve_clim_mV is not None:
                clim = geom.ve_clim_mV
            else:
                ve_mv = ve_pts.astype(np.float32) * 1.0e3
                _good = np.isfinite(ve_mv)
                if _good.any():
                    v_lo = float(np.percentile(ve_mv[_good], 1.0))
                    v_hi = float(np.percentile(ve_mv[_good], 99.0))
                else:
                    v_lo, v_hi = -1.0, 1.0
                if v_hi - v_lo < 1e-12:
                    v_hi = v_lo + 1.0
                clim = (v_lo, v_hi)
            fg["mode"] = "ve"
            # Per-branch toggles are inert in Vₑ mode (the per-
            # branch swatches in the UI go to the greyed-out
            # `is-off-locked` class — only `vis_fibers` master
            # controls visibility).
            fg["ve"] = _mkgrp(
                payload=tube,
                style=dict(
                    scalars="Ve",
                    cmap="plasma",
                    clim=clim,
                    opacity=1.0,
                    show_scalar_bar=False,
                    smooth_shading=True,
                    lighting=True,
                ),
                visible=master,
                signature=_next_sig(),
            )
            return fg
        # Palette mode — one actor per branch, hidden by
        # `vis_fibers AND vis_fiber_branch_<i>`.
        paths_mm = [
            np.asarray(p, dtype=np.float64) * 1000.0
            for p in paths_display
        ]
        n_branches = int(geom.fiber_n_branches or 0)
        bidx = geom.fiber_branch_idx
        fg["mode"] = "palette"
        if n_branches <= 1 or bidx is None:
            # Single bundle — mount under branch 0 only.
            poly = _polyline_polydata_from_paths(paths_mm)
            if poly.n_points == 0:
                return fg
            per_branch_0 = bool(state["vis_fiber_branch_0"])
            fg["branches"][0] = _mkgrp(
                payload=poly,
                style=dict(
                    color=FIBERS_MASTER_COLOUR,
                    line_width=2, opacity=0.9,
                    show_scalar_bar=False,
                ),
                visible=master and per_branch_0,
                signature=_next_sig(),
            )
            return fg
        for bi in range(n_branches):
            mask = bidx == bi
            if not mask.any():
                continue
            bpaths_mm = [paths_mm[k] for k in np.where(mask)[0]]
            poly = _polyline_polydata_from_paths(bpaths_mm)
            if poly.n_points == 0:
                continue
            per_branch = bool(state[f"vis_fiber_branch_{bi}"])
            fg["branches"][bi] = _mkgrp(
                payload=poly,
                style=dict(
                    color=BRANCH_PALETTE[bi % len(BRANCH_PALETTE)],
                    line_width=2, opacity=0.9,
                    show_scalar_bar=False,
                ),
                visible=master and per_branch,
                signature=_next_sig(),
            )
        return fg

    def _apply_fiber_selection_highlight() -> None:
        """When the user is in the Fiber tab AND `fiber_sel_idx`
        points at a valid trajectory:
          - Force-hide every NON-fiber actor (nerve, regions,
            electrodes, field lines + arrows) so the 3D viewport
            shows trajectories only.
          - Dim every other fiber actor to ~0.10 opacity (very
            translucent — the bundle reads as 'context').
          - Mount a separate `fiber_selected` tube for the chosen
            path: bright yellow, opaque, ~2× radius vs the legacy
            highlight so the picked fiber is unambiguous against
            the dim bundle.

        Called once at the tail of `_rebuild_scene_state` (AFTER
        the per-group builders have populated their groups).
        Mutates the already-built SceneGroups' `style.opacity` /
        `visible` in place and writes the `selected` group."""
        fg = _scene_state["fibers"]
        # Always reset the selected slot first so it correctly
        # disappears the moment the user leaves the Fiber tab
        # or toggles to an out-of-range index. `selected` is now
        # a DICT keyed by fiber-index → group (multi-fiber
        # selection), since the user can pick several
        # trajectories in the combobox and each gets its own
        # tab10 colour in the viewport.
        fg["selected"] = {}
        active = str(state.active_analysis)
        # Both the Single-fiber and Population tabs want the
        # viewport stripped down to "trajectories + cuff only" —
        # nerve / regions / field-lines are distractions in
        # either context. Fiber-specific dimming + multi-pick
        # highlight only runs in the 'fiber' tab.
        if active not in ("fiber", "population"):
            return

        # ---- Force-hide every other group in the scene. ----
        # The Single-fiber tab is about ONE trajectory in
        # isolation; nerve / regions / electrodes / field lines
        # are distractions here. We flip their `visible` flag
        # (and bump the signature so the renderer re-applies it)
        # rather than nulling `payload` — that way leaving the
        # tab restores the previous visibility cheaply, with no
        # mesh/tube rebuild required.
        def _force_hide(g: dict) -> None:
            if not isinstance(g, dict) or "payload" not in g:
                return
            if g.get("payload") is None:
                return
            if not g.get("visible", True):
                return
            g["visible"] = False
            g["signature"] = _next_sig()

        # Hide everything that distracts from the trajectory
        # inspection: nerve surface, region tags, FEM streamlines.
        # The CUFF ELECTRODE stays visible — it's the anatomical
        # landmark the user needs to see relative to the selected
        # fibers ("which trajectory passes under the active
        # contact?"). Electrodes are left alone here on purpose.
        _force_hide(_scene_state["nerve"])
        for tag, g in _scene_state["regions"].items():
            _force_hide(g)
        _force_hide(_scene_state["field"]["tubes"])
        _force_hide(_scene_state["field"]["arrows"])

        # Population tab: hide is enough. The per-row fiber
        # actors built by `_catalog_fold_fibers` carry their own
        # colours from `pop_row_meta`. No multi-pick highlight.
        if active != "fiber":
            return

        # ---- Trajectory dim + multi-fiber highlight ----
        paths_display = _fiber_paths_display()
        if paths_display is None or not paths_display:
            return
        # Coalesce the multi-select set into a clean int list,
        # filter out anything out of range (defensive — the
        # combobox is supposed to enforce this but
        # post-regenerate state can have stale indices).
        n_paths = int(len(paths_display))
        sel_set: list[int] = []
        for v in (state.fiber_sel_indices or []):
            try:
                vi = int(v)
            except (TypeError, ValueError):
                continue
            if 0 <= vi < n_paths and vi not in sel_set:
                sel_set.append(vi)
        # Always include the single "viewed" fiber index too —
        # so dragging the result-picker after a run keeps that
        # fiber highlighted even if the user deselected it from
        # the multi-pick combobox.
        view_i = int(state.fiber_sel_idx)
        if (0 <= view_i < n_paths
                and view_i not in sel_set):
            sel_set.append(view_i)
        if not sel_set:
            return
        # Dim every other fiber actor — palette branches AND the
        # ve tube. The branches lose their per-branch colour and
        # get a flat neutral grey so the selected red filaments
        # really pop. 0.07 opacity reads as a faint ghost cloud
        # against white.
        dim_opacity = 0.07
        dim_color = "#9aa3ad"  # cool neutral grey
        if fg["mode"] == "palette":
            for i, g in fg["branches"].items():
                if g.get("payload") is None:
                    continue
                g["style"]["opacity"] = dim_opacity
                g["style"]["color"] = dim_color
                # Bump the sig so the renderer notices the style
                # change and re-mounts with the new opacity +
                # colour.
                g["signature"] = _next_sig()
        elif fg["mode"] == "ve":
            g = fg["ve"]
            if g.get("payload") is not None:
                g["style"]["opacity"] = dim_opacity
                g["signature"] = _next_sig()
        # Render ONE actor per selected fiber so each gets its
        # own tab10 colour — matches the colour on its chip in
        # the combobox above, so the chip ↔ trajectory mapping
        # is unambiguous. Using `render_lines_as_tubes=True`
        # plus a moderate line_width=8 (down from the previous
        # 14: too thick when multiple fibers are selected and
        # they start crowding each other in the viewport).
        # Pixel-based width keeps the highlight visible at any
        # camera zoom.
        sel_groups: dict = {}
        for i in sel_set:
            path = paths_display[i]
            if path is None or len(path) < 2:
                continue
            path_mm = np.asarray(path, dtype=np.float64) * 1000.0
            poly = _polyline_polydata_from_paths([path_mm])
            if poly.n_points == 0:
                continue
            color = TAB10_PALETTE[int(i) % len(TAB10_PALETTE)]
            sel_groups[int(i)] = _mkgrp(
                payload=poly,
                style=dict(
                    color=color,
                    line_width=8,
                    render_lines_as_tubes=True,
                    opacity=1.0,
                    show_scalar_bar=False,
                    lighting=True,
                ),
                visible=True,
                signature=_next_sig(),
            )
        fg["selected"] = sel_groups

    # SceneCatalog Phase 5 — register the field section. The
    # streamlines polydata in `geom.field_lines_poly` lives in
    # cuff-local frame (the FEM solve's frame for whatever
    # parent design the active config belongs to). The viewport
    # is in pure PCA — without a back-transform, the streamlines
    # render at the wrong place. The catalog fold looks up the
    # active config's parent design, computes its cuff_origin +
    # M_design, and routes the streamline polyline points
    # through `to_pca_mm` with `source_frame="cuff_local"`. Same
    # treatment for the arrow glyphs (they share the parent
    # polydata's frame).
    def _active_fem_cuff_xform() -> (
        "tuple[np.ndarray, np.ndarray] | None"
    ):
        """Return (M_design, cuff_origin_pca_m) for the active
        FEM config's parent design, or None if not resolvable.
        Used by the field-section catalog fold for the
        cuff_local → PCA back-transform."""
        if (geom.nerve is None
                or geom.centroid is None
                or geom.R_global is None):
            return None
        active_cid = str(
            getattr(state, "active_config_id", "") or "",
        )
        configs = list(state.fem_configs or [])
        parent_eid = ""
        if active_cid:
            parent_eid = next(
                (str(c.get("design_id", ""))
                 for c in configs
                 if str(c.get("id", "")) == active_cid),
                "",
            )
        if not parent_eid:
            parent_eid = str(
                getattr(state, "active_design_id", "")
                or state.selected_design_id or "",
            )
        if not parent_eid:
            return None
        design = next(
            (d for d in (state.designs or [])
             if str(d.get("eid", "")) == parent_eid),
            None,
        )
        if design is None:
            return None
        from golgi.scene.cuff_fit import (
            _design_M, find_cuff_origin_pca,
        )
        pts_pca = (
            (np.asarray(
                geom.nerve["pts_raw"], dtype=np.float64,
            ) - geom.centroid) @ geom.R_global
        )
        cuff_origin_m = find_cuff_origin_pca(
            pts_pca, state.cuff_anchor,
            float(design.get("cuff_offset_mm", 0.0)),
            float(design.get("cuff_dx_mm", 0.0)),
            float(design.get("cuff_dy_mm", 0.0)),
        )
        M_D = _design_M(design)
        return (M_D, np.asarray(cuff_origin_m, dtype=np.float64))

    def _catalog_fold_field(geom, state, ctx) -> dict | None:
        # Whole-section fold returning {"tubes": SceneGroup,
        # "arrows": SceneGroup, ...}. None / empty groups when
        # streamlines aren't on or aren't computed yet.
        if (not bool(state.show_field_lines)
                or geom.field_lines_poly is None):
            return {
                "tubes": _mkgrp(),
                "arrows": _mkgrp(),
            }
        xform = _active_fem_cuff_xform()
        poly = geom.field_lines_poly
        if xform is None:
            # No cuff context — render as-is, in whatever frame
            # the polydata happens to be in. Better than not
            # rendering at all; matches pre-catalog behaviour.
            poly_view = poly
        else:
            M_D, cuff_origin_m = xform
            poly_view = poly.copy(deep=True)
            poly_view.points = _to_pca_mm(
                np.asarray(poly.points, dtype=np.float64),
                source_frame="cuff_local",
                source_units="mm",
                ctx=ctx,
                cuff_origin_pca_m=cuff_origin_m,
                M_design=M_D,
            )
        _em = np.asarray(
            poly_view.point_data.get("E_mag", []),
            dtype=np.float64,
        )
        _good = np.isfinite(_em) & (_em > 0)
        if _good.any():
            _clo = float(np.percentile(_em[_good], 1.0))
            _chi = float(np.percentile(_em[_good], 99.0))
        else:
            _clo, _chi = 0.0, 1.0
        if _chi - _clo < 1e-9:
            _chi = _clo + 1.0
        arrow_mesh = None
        try:
            n_pts = int(poly_view.n_points)
            stride = max(1, int(round(n_pts / 1500)))
            sampled = poly_view.extract_points(
                np.arange(0, n_pts, stride),
                include_cells=False,
                adjacent_cells=False,
            )
            arrow_mesh = sampled.glyph(
                orient="E",
                scale=False,
                factor=1.20,
                geom=pv.Arrow(
                    tip_length=0.45,
                    tip_radius=0.32,
                    shaft_radius=0.11,
                ),
            )
        except Exception as ex:
            print(
                f"[field-lines] glyph build failed: {ex}",
                flush=True,
            )
        tubes_poly = None
        try:
            tubes_poly = poly_view.tube(
                radius=0.10, n_sides=8, capping=False,
            )
        except Exception:
            pass
        out: dict = {}
        if arrow_mesh is not None and arrow_mesh.n_points > 0:
            out["arrows"] = _mkgrp(
                payload=arrow_mesh,
                style=dict(
                    scalars="E_mag",
                    cmap="viridis", clim=(_clo, _chi),
                    opacity=0.95,
                    show_edges=False, smooth_shading=True,
                    show_scalar_bar=False,
                    lighting=True,
                ),
                visible=True,
                signature=_next_sig(),
            )
        else:
            out["arrows"] = _mkgrp()
        if tubes_poly is not None:
            out["tubes"] = _mkgrp(
                payload=tubes_poly,
                style=dict(
                    scalars="E_mag",
                    cmap="viridis", clim=(_clo, _chi),
                    opacity=0.60,
                    show_edges=False, smooth_shading=True,
                    show_scalar_bar=False,
                    lighting=True,
                ),
                visible=True,
                signature=_next_sig(),
            )
        else:
            out["tubes"] = _mkgrp()
        return out

    _scene_catalog.register(_SceneEntry(
        section="field",
        key="*",
        fold=_catalog_fold_field,
        label="E-field streamlines",
    ))

    def _build_one_cuff_parts(
        eid: str,
        L_cuff_m: float,
        R_ci_m: float,
        R_co_m: float,
        patches: list,
        show_saline: bool,
        offset_xyz_m: tuple,
        R_local_in_frame: "np.ndarray | None",
        polarities: "list | None",
        is_selected: bool,
        elec_visible: dict,
        use_scar: bool = False,
        scar_thickness_m: float = 0.0,
        cuff_clearance_m: float = 0.0,
        recording_montages: "list | None" = None,
    ) -> dict:
        """Build the SceneGroup map for ONE standard parametric cuff.
        Returns a dict keyed by sub-name (silicone/saline/contacts/
        halo) → SceneGroup or sub-dict-of-SceneGroups. `elec_visible`
        carries the effective per-sub visibility (`vis_master & sub`
        already folded). When the global tag toggle decoupling rule
        from G6/Q1 kicks in, the global override is folded by the
        caller into `elec_visible`."""
        L_mm = L_cuff_m * 1000.0
        R_ci_mm = R_ci_m * 1000.0
        R_co_mm = R_co_m * 1000.0
        dx_mm = offset_xyz_m[0] * 1000.0
        dy_mm = offset_xyz_m[1] * 1000.0
        dz_mm = offset_xyz_m[2] * 1000.0
        R = (np.asarray(R_local_in_frame, dtype=np.float64)
             if R_local_in_frame is not None
             else np.eye(3, dtype=np.float64))
        M = np.eye(4, dtype=np.float64)
        M[:3, :3] = R
        M[:3, 3] = (dx_mm, dy_mm, dz_mm)

        def _to_frame(mesh):
            return mesh.transform(M, inplace=False)

        outer = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=R_co_mm, height=L_mm,
            resolution=96, capping=False,
        )
        inner = pv.Cylinder(
            center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
            radius=R_ci_mm, height=L_mm,
            resolution=96, capping=False,
        )
        cap_top = pv.Disc(
            center=(0.0, 0.0, L_mm / 2.0),
            inner=R_ci_mm, outer=R_co_mm,
            normal=(0.0, 0.0, 1.0), r_res=2, c_res=96,
        )
        cap_bot = pv.Disc(
            center=(0.0, 0.0, -L_mm / 2.0),
            inner=R_ci_mm, outer=R_co_mm,
            normal=(0.0, 0.0, -1.0), r_res=2, c_res=96,
        )
        wall = outer.merge([inner, cap_top, cap_bot])
        wall = _to_frame(wall)
        wall = wall.compute_normals(
            cell_normals=False, point_normals=True,
            consistent_normals=True, auto_orient_normals=True,
            non_manifold_traversal=False,
        )
        out: dict = {
            "silicone": _mkgrp(
                payload=wall,
                style=_phong_style(DEFAULTS[3]),
                visible=elec_visible["silicone"],
                signature=_next_sig(),
            ),
        }
        if show_saline:
            sal_R = R_ci_mm * 0.995
            sal_L = L_mm * 0.999
            saline = pv.Cylinder(
                center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                radius=sal_R, height=sal_L,
                resolution=96, capping=True,
            )
            saline = _to_frame(saline)
            sstyle = SALINE_OVERLAY_STYLE
            out["saline"] = _mkgrp(
                payload=saline,
                style=dict(
                    color=sstyle["color"],
                    opacity=sstyle["opacity"],
                    pbr=False,
                    ambient=sstyle["ambient"],
                    diffuse=sstyle["diffuse"],
                    specular=sstyle["specular"],
                    specular_power=sstyle["specular_power"],
                    smooth_shading=True,
                    show_edges=False,
                    culling=False,
                ),
                visible=elec_visible["saline"],
                signature=_next_sig(),
            )
        else:
            out["saline"] = _mkgrp()
        # F3.2-M3 — pre-mesh scar preview. Translucent salmon
        # cylinder at R_scar = r_nerve_outer + scar_thickness,
        # so bigger thickness = bigger scar shell wrapping the
        # nerve (intuitive direction). We use R_ci − cuff_clearance
        # as the proxy for r_nerve_outer (matches what autosize_R_ci
        # uses to set R_ci in the first place). Clamped to R_ci − ε
        # so the preview never crosses the silicone wall.
        if use_scar and scar_thickness_m > 0.0:
            _r_nerve_proxy_m = max(
                float(R_ci_m) - float(cuff_clearance_m),
                1.0e-6,
            )
            _R_scar_m = min(
                _r_nerve_proxy_m + float(scar_thickness_m),
                float(R_ci_m) - 1.0e-6,
            )
            R_scar_mm = max(_R_scar_m * 1000.0, 0.05)
            sca_R = R_scar_mm * 0.995
            sca_L = L_mm * 0.999
            scar = pv.Cylinder(
                center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                radius=sca_R, height=sca_L,
                resolution=96, capping=True,
            )
            scar = _to_frame(scar)
            _scar_spec = DEFAULTS[TAG_SCAR]
            out["scar"] = _mkgrp(
                payload=scar,
                style=dict(
                    color=_scar_spec["color"],
                    opacity=_scar_spec["opacity"],
                    pbr=False,
                    ambient=_scar_spec["ambient"],
                    diffuse=_scar_spec["diffuse"],
                    specular=_scar_spec["specular"],
                    specular_power=_scar_spec["specular_power"],
                    smooth_shading=True,
                    show_edges=False,
                    culling=False,
                ),
                visible=elec_visible.get("scar", False),
                signature=_next_sig(),
            )
        else:
            out["scar"] = _mkgrp()
        contacts: dict = {}
        for _i, _patch in enumerate(patches):
            _patch_mm = _patch.copy()
            _patch_mm.points = _patch_mm.points * 1000.0
            _patch_mm = _to_frame(_patch_mm)
            _patch_mm = _patch_mm.compute_normals(
                cell_normals=False, point_normals=True,
                consistent_normals=True, auto_orient_normals=True,
                non_manifold_traversal=False,
            )
            if polarities is not None and _i < len(polarities):
                _pol = polarities[_i]
            else:
                _pol = "off"
            if _pol == "anode":
                _style = ANODE_STYLE
            elif _pol == "cathode":
                _style = CATHODE_STYLE
            else:
                _style = GOLD_STYLE
            contacts[_i] = _mkgrp(
                payload=_patch_mm,
                style=_phong_style(_style),
                visible=elec_visible["contacts"],
                signature=_next_sig(),
            )
        out["contacts"] = contacts
        # R1.1 — Recording-montage arcs. One straight 3D line per
        # montage, between the centroids of its + and − contacts.
        # Skip silently when a montage references a contact id
        # out of range (defensive — the editor blocks this, but
        # legacy projects might carry stale ids).
        #
        # KILL SWITCH: setting GOLGI_DISABLE_RECORDING_ARCS=1
        # disables this whole block. Use it to A/B-test whether a
        # crash is from the arc rendering or from something else.
        #
        # R1.4 fix-up #3: dropped `render_lines_as_tubes=True`.
        # Empirically the VTK tube filter on macOS can crash when
        # invoked alongside other tube-rendered actors during a
        # rapid scene rebuild (fiber re-colouring after pop-
        # generate). Plain thick polylines (line_width=6) render
        # via the OpenGL rasteriser directly and look ~the same.
        # Added explicit finite-coord + min-distance checks so a
        # degenerate line never reaches VTK.
        rec_arcs: dict = {}
        if os.environ.get("GOLGI_DISABLE_RECORDING_ARCS", "0") != "1":
            for _m in (recording_montages or []):
                try:
                    _mid = str(_m.get("mid", ""))
                    _plus = int(_m.get("plus_contact", -1))
                    _minus = int(_m.get("minus_contact", -1))
                    _color = str(_m.get("color") or "#22c55e")
                except (TypeError, ValueError):
                    continue
                if not _mid:
                    continue
                _g_plus = contacts.get(_plus)
                _g_minus = contacts.get(_minus)
                if _g_plus is None or _g_minus is None:
                    continue
                _pl_payload = _g_plus.get("payload")
                _mn_payload = _g_minus.get("payload")
                if _pl_payload is None or _mn_payload is None:
                    continue
                try:
                    _p0 = np.asarray(
                        _pl_payload.center, dtype=np.float64,
                    )
                    _p1 = np.asarray(
                        _mn_payload.center, dtype=np.float64,
                    )
                except Exception:
                    continue
                if (_p0.shape != (3,) or _p1.shape != (3,)
                        or not np.all(np.isfinite(_p0))
                        or not np.all(np.isfinite(_p1))):
                    continue
                if float(np.linalg.norm(_p1 - _p0)) < 1.0e-6:
                    # Degenerate line; vtkTubeFilter-style
                    # filters segfault on zero-length input.
                    continue
                try:
                    _line = pv.Line(
                        tuple(_p0), tuple(_p1), resolution=1,
                    )
                except Exception as _ex:                       # noqa: BLE001
                    print(
                        f"[recording-arc] pv.Line failed for "
                        f"montage {_mid!r}: "
                        f"{type(_ex).__name__}: {_ex} — skipping",
                        flush=True,
                    )
                    continue
                rec_arcs[_mid] = _mkgrp(
                    payload=_line,
                    style=dict(
                        color=_color,
                        line_width=6,
                        opacity=0.9,
                    ),
                    visible=elec_visible.get("contacts", True),
                    signature=_next_sig(),
                )
        out["recording_arcs"] = rec_arcs
        # Halo (only when selected & drawer open & master visible)
        if is_selected and elec_visible.get("master", True):
            _halo_R_mm = float(R_co_m) * 1000.0 * 1.15
            _halo_L_mm = float(L_cuff_m) * 1000.0 * 1.18
            _halo = pv.Cylinder(
                center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                radius=_halo_R_mm, height=_halo_L_mm,
                resolution=96, capping=False,
            )
            _halo = _to_frame(_halo)
            out["halo"] = _mkgrp(
                payload=_halo,
                style=dict(
                    color="#e24b4a",
                    opacity=0.32,
                    smooth_shading=True,
                    show_edges=False,
                    culling=False,
                    ambient=0.45,
                    diffuse=0.55,
                    specular=0.10,
                ),
                visible=True,
                signature=_next_sig(),
            )
        else:
            out["halo"] = _mkgrp()
        return out

    def _build_one_duke_parts(
        eid: str,
        elec: dict,
        offset_xyz_m: tuple,
        R_local_in_frame: "np.ndarray | None",
        polarities: "list | None",
        is_selected: bool,
        elec_visible: dict,
    ) -> dict:
        """Build the SceneGroup map for ONE DUKE-designer cuff.
        Designer parts are routed into silicone/saline/contacts
        visibility buckets by role."""
        preset_name = str(elec.get("duke_preset", "") or "")
        preset = _CUFF_PRESETS.get(preset_name, None)
        if preset is None:
            return {"silicone": _mkgrp(), "saline": _mkgrp(),
                    "contacts": {}, "halo": _mkgrp(),
                    "designer": {}}
        overrides_si = {
            str(k): float(v)
            for k, v in (
                elec.get("duke_overrides", {}) or {}
            ).items()
        }
        try:
            parts = cuff_designer.render_design(
                preset,
                param_overrides=overrides_si,
                ns_extras=_cuff_ns_extras(),
            )
        except Exception as ex:
            print(f"[cuff_designer] eid={eid} render failed: {ex}",
                  flush=True)
            parts = []
        dx_mm = offset_xyz_m[0] * 1000.0
        dy_mm = offset_xyz_m[1] * 1000.0
        dz_mm = offset_xyz_m[2] * 1000.0
        R_for_duke = (R_local_in_frame
                      if R_local_in_frame is not None
                      else np.eye(3, dtype=np.float64))
        t_for_duke = np.array(
            [dx_mm, dy_mm, dz_mm], dtype=np.float64,
        )
        designer_groups: dict = {}
        _conductor_idx = 0
        for idx, (_inst, _sub, mesh, role) in enumerate(parts):
            mesh_mm = mesh.copy()
            pts_scaled = (
                np.asarray(mesh_mm.points, dtype=np.float64)
                * 1000.0
            )
            mesh_mm.points = pts_scaled @ R_for_duke.T + t_for_duke
            color = cuff_designer.ROLE_COLORS.get(
                role, (0.7, 0.7, 0.7),
            )
            opacity = cuff_designer.ROLE_OPACITIES.get(role, 1.0)
            if role == "conductor":
                if (polarities is not None
                        and _conductor_idx < len(polarities)):
                    _pol = polarities[_conductor_idx]
                    if _pol == "anode":
                        color = ANODE_STYLE["color"]
                    elif _pol == "cathode":
                        color = CATHODE_STYLE["color"]
                    elif _pol == "ground":
                        color = (0.55, 0.55, 0.55)
                _conductor_idx += 1
            # Role-based visibility routing.
            if role == "conductor":
                v = elec_visible["contacts"]
            elif role == "fill":
                v = elec_visible["saline"]
            elif role in ("insulator", "recess"):
                v = elec_visible["silicone"]
            else:
                v = elec_visible.get("master", True)
            designer_groups[f"{role}_{idx}"] = _mkgrp(
                payload=mesh_mm,
                style=dict(
                    color=color, opacity=opacity,
                    show_edges=False, smooth_shading=True,
                    specular=0.40, specular_power=15.0,
                ),
                visible=v,
                signature=_next_sig(),
            )
        return {
            "silicone": _mkgrp(),
            "saline": _mkgrp(),
            "contacts": {},
            "halo": _mkgrp(),
            "designer": designer_groups,
        }

    # SceneCatalog Phase 3 — register the electrodes entry. The
    # existing `_set_electrode_groups` already builds all the
    # per-design parts via `_build_one_cuff_parts` /
    # `_build_one_duke_parts` and returns a `{eid: {part:
    # SceneGroup}}` dict. The frame logic is already correct
    # post-M2.1f (each electrode positioned at its own absolute
    # PCA cuff origin, not a delta from the currently-fit one),
    # so the catalog wrap is structural — it doesn't change the
    # transform, just routes the output through the catalog so
    # the dual-run can prove parity and Phase 6 can delete the
    # inline.
    def _catalog_fold_electrodes(
        geom, state, ctx,
    ) -> dict | None:
        if (geom.nerve is None
                or geom.pts_cuff is None
                or ctx.centroid is None
                or ctx.R_global is None
                or geom.cuff_origin_pca is None
                or geom.R_ci is None
                or geom.R_co is None):
            return {}
        R_ci_def = float(geom.R_ci)
        R_co_def = float(geom.R_co)
        L_def = float(state.L_cuff_mm) * 1e-3

        def _f(d, k, default):
            # `dict.get(k, default)` only swaps in the default when
            # the key is MISSING, not when the value is "" — and an
            # empty VTextField leaves the value as "". Without this
            # coercion `float("")` raises and the whole fold bails,
            # making every cuff vanish from the viewport.
            v = d.get(k, default)
            if v is None or v == "":
                return default
            return float(v)

        pts_pca = (
            (np.asarray(
                geom.nerve["pts_raw"], dtype=np.float64,
            ) - ctx.centroid) @ ctx.R_global
        )
        electrodes = list(state.designs or [])
        selected_eid = str(state.selected_design_id or "")
        show_halo = (bool(state.show_cuff)
                     and selected_eid != "")
        new_electrodes: dict = {}
        for elec in electrodes:
            eid = str(elec.get("eid", ""))
            if not eid:
                continue
            # Phase 6b — single-focus rendering. Only emit
            # cuff geometry for the FOCUSED design. Other
            # designs' cuff parts are skipped entirely; they
            # remount on combobox switch.
            if eid != selected_eid:
                continue
            has_mesh_e = bool(elec.get("has_mesh", False))
            vm = (
                bool(elec.get("vis_master", True))
                and not has_mesh_e
            )
            elec_visible = {
                "master": vm,
                "silicone": vm and bool(
                    elec.get("vis_silicone", True),
                ),
                "saline": vm and bool(
                    elec.get("vis_saline", True),
                ),
                "contacts": vm and bool(
                    elec.get("vis_contacts", True),
                ),
                # F3.2-M3 — scar preview is gated on use_scar
                # AND vis_scar AND master. Without use_scar the
                # design has no scar shell at all, so the actor
                # shouldn't render even if vis_scar is True.
                "scar": (
                    vm
                    and bool(elec.get("use_scar", False))
                    and bool(elec.get("vis_scar", True))
                ),
            }
            elec_L = _f(elec, "L_cuff_mm", L_def * 1000.0) * 1e-3
            elec_R_ci = _f(elec, "R_ci_m", R_ci_def)
            elec_R_co = _f(elec, "R_co_m", R_co_def)
            elec_kind = str(
                elec.get(
                    "electrode_type", "bipolar ring-pair",
                ),
            )
            elec_origin_pca = find_cuff_origin_pca(
                pts_pca, state.cuff_anchor,
                _f(elec, "cuff_offset_mm", 0.0),
                _f(elec, "cuff_dx_mm", 0.0),
                _f(elec, "cuff_dy_mm", 0.0),
            )
            rel = np.asarray(
                elec_origin_pca, dtype=np.float64,
            )
            ddx = float(rel[0])
            ddy = float(rel[1])
            dz_off = float(rel[2])
            stored_R = elec.get("R_local_elec", None)
            if stored_R is not None and len(stored_R) == 9:
                R_elec_pca = np.asarray(
                    stored_R, dtype=np.float64,
                ).reshape(3, 3)
                R_elec_in_frame = R_elec_pca.T
            else:
                R_elec_in_frame = None
            _rot_x_deg = _f(elec, "cuff_rot_x_deg", 0.0)
            _rot_y_deg = _f(elec, "cuff_rot_y_deg", 0.0)
            _rot_z_deg = _f(elec, "cuff_rot_z_deg", 0.0)
            if _rot_x_deg or _rot_y_deg or _rot_z_deg:
                _R_local_tilt = np.eye(3, dtype=np.float64)
                if _rot_x_deg:
                    _t = float(np.deg2rad(_rot_x_deg))
                    _c, _s = (
                        float(np.cos(_t)), float(np.sin(_t)),
                    )
                    _Rx = np.array(
                        [[1.0, 0.0, 0.0],
                         [0.0,  _c, -_s],
                         [0.0,  _s,  _c]],
                        dtype=np.float64,
                    )
                    _R_local_tilt = _R_local_tilt @ _Rx
                if _rot_y_deg:
                    _t = float(np.deg2rad(_rot_y_deg))
                    _c, _s = (
                        float(np.cos(_t)), float(np.sin(_t)),
                    )
                    _Ry = np.array(
                        [[ _c, 0.0,  _s],
                         [0.0, 1.0, 0.0],
                         [-_s, 0.0,  _c]],
                        dtype=np.float64,
                    )
                    _R_local_tilt = _R_local_tilt @ _Ry
                if _rot_z_deg:
                    _t = float(np.deg2rad(_rot_z_deg))
                    _c, _s = (
                        float(np.cos(_t)), float(np.sin(_t)),
                    )
                    _Rz = np.array(
                        [[ _c, -_s, 0.0],
                         [ _s,  _c, 0.0],
                         [0.0, 0.0, 1.0]],
                        dtype=np.float64,
                    )
                    _R_local_tilt = _R_local_tilt @ _Rz
                if R_elec_in_frame is None:
                    R_elec_in_frame = _R_local_tilt
                else:
                    R_elec_in_frame = (
                        R_elec_in_frame @ _R_local_tilt
                    )
            is_selected = (
                show_halo and eid == selected_eid
            )
            elec_polarities = (
                _ensure_polarities(elec)
                if is_selected else None
            )
            if elec_kind == DUKE_ELECTRODE_TYPE:
                parts = _build_one_duke_parts(
                    eid=eid,
                    elec=elec,
                    offset_xyz_m=(ddx, ddy, dz_off),
                    R_local_in_frame=R_elec_in_frame,
                    polarities=elec_polarities,
                    is_selected=is_selected,
                    elec_visible=elec_visible,
                )
            else:
                elec_cfg = {
                    k: elec.get(k, DEFAULT_ELECTRODE[k])
                    for k in DEFAULT_ELECTRODE
                    if k != "electrode_type"
                }
                elec_patches = build_electrode_patches(
                    elec_L, elec_R_ci,
                    kind=elec_kind, cfg=elec_cfg,
                )
                # F3.2-M3 — per-design scar params for the
                # pre-mesh preview cylinder at R_scar.
                _use_scar = bool(elec.get("use_scar", False))
                _scar_thickness_m = (
                    _f(elec, "scar_thickness_um", 0.0) * 1e-6
                    if _use_scar else 0.0
                )
                _cuff_clearance_m = (
                    _f(elec, "cuff_clearance_mm", 0.2) * 1e-3
                )
                parts = _build_one_cuff_parts(
                    eid=eid,
                    L_cuff_m=elec_L,
                    R_ci_m=elec_R_ci,
                    R_co_m=elec_R_co,
                    patches=elec_patches,
                    show_saline=bool(
                        elec.get("show_saline", True),
                    ),
                    offset_xyz_m=(ddx, ddy, dz_off),
                    R_local_in_frame=R_elec_in_frame,
                    polarities=elec_polarities,
                    is_selected=is_selected,
                    elec_visible=elec_visible,
                    use_scar=_use_scar,
                    scar_thickness_m=_scar_thickness_m,
                    cuff_clearance_m=_cuff_clearance_m,
                    recording_montages=list(
                        elec.get("recording_montages") or [],
                    ),
                )
            new_electrodes[eid] = parts
        return new_electrodes

    _scene_catalog.register(_SceneEntry(
        section="electrodes",
        key="*",  # whole-section fill
        fold=_catalog_fold_electrodes,
        label="Cuff electrodes",
    ))
    _scene_catalog.register(_SceneEntry(
        section="fibers",
        key="*",  # whole-section fill
        fold=_catalog_fold_fibers,
        label="Fiber bundles",
    ))
    _catalog_sections_active = (
        "nerve", "electrodes", "regions", "field", "fibers",
    )

    def _rebuild_scene_state_real() -> None:  # noqa: E306
        """The real scene-state folder. Reads geom + state and
        populates every group in `_scene_state`. Pure CPU — never
        touches `pl`.

        SceneCatalog Phase 6b-ii — the catalog is the sole writer
        for nerve / regions / field / electrodes / fibers (their
        folds live as `_catalog_fold_*` closures in this module,
        registered against `scene/catalog.py`). The inline
        `_set_*_group` functions are retired. `selected` (per-
        fiber highlight tubes) is still owned by
        `_apply_fiber_selection_highlight` below — the fibers
        fold returns a dict without that key so the highlighter's
        writes survive the `.update()` merge.
        """
        try:
            _scene_catalog.apply_in_place(
                _scene_state, geom, state, _mkgrp,
            )
        except Exception as ex:
            print(f"[scene] catalog apply failed: {ex}",
                  flush=True)
        # `_apply_fiber_selection_highlight` MUST run LAST — it
        # force-hides nerve / regions / field tubes / arrows when
        # the Single-fiber tab is active. If it ran before the
        # catalog, the catalog would rebuild those groups with
        # `visible=True` and silently clobber the force-hide.
        try:
            _apply_fiber_selection_highlight()
        except Exception as ex:
            print(f"[scene] fiber highlight fold failed: {ex}",
                  flush=True)

    # Override the forward-declared no-op with the real folder.
    _rebuild_scene_state = _rebuild_scene_state_real

    # ----------------------------------------------------------------
    # Visibility-watcher shims. These keep the legacy function names
    # for any external caller, but the body is just a render request.
    # The actual visibility folding happens in _set_*_groups above.
    # ----------------------------------------------------------------
    def _apply_visibility(tag: int, visible: bool) -> None:
        """Per-tag visibility flip. State already updated by trame
        before this fires; we just request a render pass."""
        _request_render()

    def _apply_fiber_visibility() -> None:
        """Per-branch / master fiber visibility flip."""
        _request_render()

    # W1.7a: per-region vis_<tag> + master vis_fibers + per-branch
    # vis_fiber_branch_<i> watchers extracted into
    # golgi.watchers.render_toggles.register (registered above).

    # -----------------------------------------------------------------
    # Avatar upload handlers — VFileInput's v_model carries a list
    # of {name, content, size, type} dicts where `content` is the
    # raw file bytes (trame transcodes the browser File for us).
    # We base64-encode here and stash the data URI in the matching
    # *_image_data_uri var so the dialog preview <img> can show
    # it AND the submit handlers can decode it back via
    # `_decode_avatar_data_uri`. Validation (size + mime) runs
    # twice — once on upload (so the user sees the error
    # immediately) and again on submit (defence in depth).
    # -----------------------------------------------------------------
    def _ingest_avatar_upload(
        info, error_var: str, target_var: str,
        also_clear_remove: str | None = None,
    ) -> None:
        """Shared body for the auth + profile avatar watchers."""
        if not info:
            return
        entries = info if isinstance(info, list) else [info]
        entry = entries[0] if entries else None
        if entry is None or "content" not in entry:
            return
        content = entry.get("content") or b""
        if not isinstance(content, (bytes, bytearray)):
            return
        err = _validate_avatar_bytes(bytes(content))
        if err:
            setattr(state, error_var, err)
            setattr(state, target_var, "")
            return
        mime = _sniff_image_mime(bytes(content)) or "png"
        data_uri = (
            f"data:image/{mime};base64,"
            + base64.b64encode(bytes(content)).decode("ascii")
        )
        setattr(state, target_var, data_uri)
        setattr(state, error_var, "")
        if also_clear_remove:
            setattr(state, also_clear_remove, False)

    # Auth-image + profile-image + upload-file + open-project
    # watchers — extracted to golgi.watchers.auth_upload in
    # step 5.2.
    _watchers.auth_upload.register(
        state,
        ingest_avatar_upload=_ingest_avatar_upload,
        list_data_files=list_data_files,
        do_open_project=do_open_project,
        active_upload_dir=lambda: get_active().upload_dir,
    )

    # -----------------------------------------------------------------
    # UI — manual VAppLayout (no SinglePageLayout) so we have full
    # control over the toolbar: just our GOLGI logo + menu items,
    # both left-aligned. No hamburger, no lowercase title, no camera
    # icon.
    # -----------------------------------------------------------------
    # Register our magma-loader CSS as a static asset so trame
    # injects a <link rel="stylesheet"> into the document <head>.
    # NOTE: html.Style(...) inside a Vue template is stripped by
    # Vue 3's compiler, so we have to go through the module
    # mechanism to get the CSS into the page <head>.
    _setup_golgi_assets(server)

    # Browser tab title — trame propagates `state.trame__title`
    # into the document's <title> tag. `VAppLayout` (unlike the
    # bundled SinglePageLayout) doesn't expose a `.title` slot,
    # so we set it via the state shortcut instead.
    server.state.trame__title = "GOLGI.IO"

    # Construct the PipelineContext now — every closure + helper
    # it bundles is defined above this point. The `do_*` handlers
    # already in scope reference `_pipeline_ctx` via late binding,
    # so this rebind activates them.
    _pipeline_helpers = SimpleNamespace(
        # mesh-pipeline deps
        assemble_multi_domain_plc=assemble_multi_domain_plc,
        write_msh22=write_msh22,
        tet_shape_quality=_tet_shape_quality,
        compute_mesh_stats_html=_compute_mesh_stats_html,
        build_quality_histogram_figure=_build_quality_histogram_figure,
        extract_region_surfaces=_extract_region_surfaces_mm,
        build_viz_surfaces=_build_viz_surfaces,
        defaults_by_tag=DEFAULTS,
        # F3.2 — per-design refit helper. The mesh + FEM drivers
        # compute (cuff_offset, R) directly via
        # `scene.cuff_fit.design_cuff_transform`; the previous
        # `_compute_design_pts_cuff` / `_design_local_to_anchor_frame`
        # round-trip is gone because the mesh is built in the
        # shared canonical frame (no per-design transforms).
        refit_design_geometry=_refit_design_geometry,
        # fem-pipeline deps
        transform_to_cuff_frame=transform_to_cuff_frame,
        build_electrode_patches_dicts=build_electrode_patches_dicts,
        cuff_designer=cuff_designer,
        _CUFF_PRESETS=_CUFF_PRESETS,
        DUKE_ELECTRODE_TYPE=DUKE_ELECTRODE_TYPE,
        DEFAULT_ELECTRODE=DEFAULT_ELECTRODE,
        cuff_ns_extras=_cuff_ns_extras,
        ensure_polarities=_ensure_polarities,
        refresh_fem_plots=_refresh_fem_plots,
        script_cwd=HERE,
        # fibers-pipeline deps
        classify_fibers_by_branch=_classify_fibers_by_branch,
        compute_fiber_branch_summary=_compute_fiber_branch_summary,
        MAX_FIBER_BRANCHES=MAX_FIBER_BRANCHES,
        refresh_fiber_sel_items=_refresh_fiber_sel_items,
        refresh_pop_branches_meta=_refresh_pop_branches_meta,
        # fiber-sim-pipeline deps
        axonml_run_single=axonml_run_single,
        build_pulse_waveform=build_pulse_waveform,
        build_pulse_breakpoints=build_pulse_breakpoints,
        MYELINATED_MODELS=MYELINATED_MODELS,
        UNMYELINATED_MODELS=UNMYELINATED_MODELS,
        fiber_pulse_params=_fiber_pulse_params,
        fiber_label_and_color=_fiber_label_and_color,
        save_fiber_sim_cache=_save_fiber_sim_cache,
        # pop-sim-pipeline deps
        TAB10_PALETTE=TAB10_PALETTE,
        fiber_paths_display=_fiber_paths_display,
        save_pop_state=_save_pop_state,
        # shared
        active_project=get_active,
    )
    _pipeline_ctx = PipelineContext(
        state=state, geom=geom, scene=scene,
        stamp_user_line=_stamp_user_line,
        autosave=_autosave,
        safe_update=safe_update,
        safe_reset_camera=safe_reset_camera,
        register_subprocess=_cancel.arm,
        clear_subprocess=_cancel.clear,
        was_cancelled=_cancel.was_requested,
        helpers=_pipeline_helpers,
    )

    # ------------------------------------------------------------------
    # W1.8a — Action handlers (do_*) extracted into golgi/actions/.
    # Registered here, BEFORE the VAppLayout template is built, so
    # the names re-bound below are in scope at template-construction
    # time. Each register() call returns a dict of {handler_name:
    # callable}; we unpack into local names so existing UI bindings
    # (click=do_xyz) resolve unchanged.
    # ------------------------------------------------------------------
    _cond_actions = _actions.conductivity.register(
        state,
        default_sigma=DEFAULT_SIGMA,
        autosave=_autosave,
        get_out_dir=lambda: get_active().out_dir,
    )
    do_reset_sigma = _cond_actions["do_reset_sigma"]
    do_update_sigma = _cond_actions["do_update_sigma"]
    do_cole_cole_apply = _cond_actions["do_cole_cole_apply"]
    do_cole_cole_cancel = _cond_actions["do_cole_cole_cancel"]

    _cancel_actions = _actions.cancel_busy.register(
        state, cancel_token=_cancel,
    )
    do_request_cancel = _cancel_actions["do_request_cancel"]
    do_dismiss_cancel = _cancel_actions["do_dismiss_cancel"]
    do_confirm_cancel = _cancel_actions["do_confirm_cancel"]

    _compute_actions = _actions.compute.register(
        gated=gated,
        pipeline_ctx=_pipeline_ctx,
        pipeline_mesh=_pipeline_mesh,
        pipeline_fem=_pipeline_fem,
        pipeline_fibers=_pipeline_fibers,
        pipeline_fiber_sim=_pipeline_fiber_sim,
    )
    do_build_mesh = _compute_actions["do_build_mesh"]
    do_solve_fem = _compute_actions["do_solve_fem"]
    do_generate_fibers = _compute_actions["do_generate_fibers"]
    do_run_fiber_sim = _compute_actions["do_run_fiber_sim"]

    # F3.2-M3 — Import-stepper handlers, split into:
    #   `do_stepper_action`: runs the step's underlying action
    #     (load nerve / generate fibers / set focus mode). No
    #     auto-advance — the user explicitly clicks Continue to
    #     move on.
    #   `do_stepper_next`: pure step advance, no actions.
    # The previous single-handler `do_stepper_next` did BOTH and
    # the user wanted explicit control over each transition, so
    # the wizard now exposes two buttons (action + advance) and
    # Continue is gated on the step's action having completed
    # (e.g. Step 1's Continue needs has_geometry; Step 3's needs
    # has_fibers).
    async def do_stepper_action(*_args):
        step = str(state.import_stepper_step or "1")
        if step == "1":
            await do_load_geometry()
            # Focus mode = default — nothing else to hide on the
            # cold-loaded nerve. (vis_* flags untouched.)
        elif step == "2":
            # Endoneurium / epineurium "generate". No async work —
            # the epi shell + thickness are mesh-time params and
            # the preview is already live. The action commits the
            # focus mode: only nerve (endo) + epi visible, hide
            # the rest so the user can compare endo vs epi without
            # the muscle bbox or fibers crowding the viewport.
            state.vis_nerve_raw = True
            state.vis_epi_preview = True
            state.vis_muscle_preview = False
            state.vis_fibers = False
        elif step == "3":
            await do_generate_fibers()
            if state.has_fibers:
                # Focus mode = fibers only. Hide every other
                # actor so the user can inspect trajectories
                # against an empty viewport.
                state.vis_nerve_raw = False
                state.vis_epi_preview = False
                state.vis_muscle_preview = False
                state.vis_fibers = True
        elif step == "4":
            # Muscle "generate" — no async work; the bbox preview
            # is already live (muscle_preview_unlocked flipped on
            # step arrival). The action restores full visibility
            # of every prior actor so the user can review the
            # whole assembly before closing the wizard.
            # M48 — for bundle imports the nerve actor IS the
            # epi shell (the catalog fold routes its visibility
            # through `vis_epi_preview` for bundles). For STL
            # imports `vis_epi_preview` controls the optional
            # inward-offset shell that `use_epi` enables.
            # Same flag, two meanings — gate on source type so
            # the bundle's real epi shows up after the muscle
            # step instead of getting hidden because the user
            # left `use_epi=False` (which is the right default
            # for bundles: they already carry an epi).
            _is_bundle = str(
                getattr(state, "import_source_type", "stl") or "stl",
            ) in ("uct_bundle", "histo_bundle")
            state.vis_nerve_raw = True
            state.vis_epi_preview = (
                True if _is_bundle else bool(state.use_epi)
            )
            state.vis_muscle_preview = True
            state.vis_fibers = True

    def do_stepper_next(*_args):
        """Advance the import-nerve wizard's stepper.

        For µCT bundle imports we SKIP step 2 entirely:
        the bundle already carries an epi shell + per-
        fascicle endoneurium surfaces, so there's nothing
        to generate via the inward-offset workflow that
        step 2 normally drives. Step 1 → 3 directly; the
        Back handler in the dialog mirrors this by jumping
        3 → 1.
        """
        is_bundle = (
            str(
                getattr(state, "import_source_type", "stl"),
            ) in ("uct_bundle", "histo_bundle")
        )
        step = str(state.import_stepper_step or "1")
        if step == "1":
            state.import_stepper_step = (
                "3" if is_bundle else "2"
            )
        elif step == "2":
            state.import_stepper_step = "3"
        elif step == "3":
            state.import_stepper_step = "4"
        elif step == "4":
            state.show_import_stepper = False

    # F2.1.c — Sweep actions. fiber_pulse_params is the closure
    # _watchers.fiber_panel.register returned at startup (sweep
    # mode-A: uses the same pulse design as the single-fiber sim).
    # save_sweep (F2.1.d) wraps golgi.projects.sweep_cache.save_sweep
    # with the active-project's sweeps directory; returns None when
    # no project is active so the action handler can skip caching.
    from golgi.projects import sweep_cache as _sweep_cache  # noqa: E402

    def _save_sweep_to_project(result):
        if not state.has_active_project:
            return None
        # F3.2 — tag the sweep with the currently-active config_id
        # so per-config sweeps coexist in `<project>/sweeps/`.
        # Cross-config selectivity comparison reads these via
        # `sweep_cache.load_latest_for_config(cid)`. Falls back to
        # untagged when no config is active (e.g. user ran a sweep
        # before adding a config — unusual but harmless).
        active_cid = (
            str(getattr(state, "active_config_id", "") or "")
        )
        return _sweep_cache.save_sweep(
            result, Path(get_active().out_dir),
            write_csvs=True,
            cid=active_cid or None,
        )

    _sweep_actions = _actions.sweep.register(
        state,
        pipeline_ctx=_pipeline_ctx,
        pipeline_sweep=_pipeline_sweep,
        fiber_pulse_params=_fiber_pulse_params,
        save_sweep=_save_sweep_to_project,
    )
    do_run_amplitude_sweep = _sweep_actions["do_run_amplitude_sweep"]
    do_find_thresholds = _sweep_actions["do_find_thresholds"]
    do_toggle_sweep_advanced = _sweep_actions[
        "do_toggle_sweep_advanced"
    ]

    # F2.3.a — per-panel publication-grade figure export. One async
    # handler routed by `fig_id` (passed as the first click arg from
    # the figure_export_btn popover); writes the rendered bytes as a
    # base64 data URI on state.export_pending_* so the popover's
    # Download anchor activates. `viewports` maps the floating
    # camera buttons (workspace 3D scene + the cuff designer
    # dialog's plotter) to the underlying PyVista plotters so the
    # do_export_viewport_screenshot handler can call .screenshot().
    _export_actions = _actions.figure_export.register(
        state,
        pipeline_ctx=_pipeline_ctx,
        viewports={
            "main": pl,
            "cuff_designer": pl_cuff,
        },
        # F2.3.c Phase 2 — off-screen render variants need the
        # styling constants from app.py. Threaded through here so
        # golgi.figures.render3d stays free of an app.py import
        # (which would be circular: app.py → figures → app.py).
        render3d_kwargs={
            "region_defaults": DEFAULTS,
            "gold_style": GOLD_STYLE,
            "branch_palette": BRANCH_PALETTE,
        },
    )
    do_export_single_figure = _export_actions[
        "do_export_single_figure"
    ]
    do_export_viewport_screenshot = _export_actions[
        "do_export_viewport_screenshot"
    ]
    do_bulk_export = _export_actions["do_bulk_export"]
    do_bulk_export_select_all_available = _export_actions[
        "do_bulk_export_select_all_available"
    ]
    do_bulk_export_clear = _export_actions["do_bulk_export_clear"]
    do_generate_report = _export_actions["do_generate_report"]
    do_open_generate_report_dialog = _export_actions[
        "do_open_generate_report_dialog"
    ]
    do_close_generate_report_dialog = _export_actions[
        "do_close_generate_report_dialog"
    ]

    # F2.2 — study bundle (export / import / replay). Hands the
    # action handler PROJECTS_ROOT (so import_study can pick a
    # fresh dir under the user's workspace) + the do_open_project
    # callback (so the imported project opens immediately).
    # `refresh_projects_list` is the local closure that re-runs
    # `_list_projects` for the welcome-view tile grid; we capture
    # it here so the imported project appears without a page
    # reload.
    _study_actions = _actions.study_bundle.register(
        state,
        pipeline_ctx=_pipeline_ctx,
        projects_root=PROJECTS_ROOT,
        downloads_dir=_DOWNLOADS_DIR,
        downloads_endpoint="_downloads",
        do_open_project=do_open_project,
        refresh_projects_list=_refresh_projects_list,
    )
    do_export_study = _study_actions["do_export_study"]
    do_export_study_dialog_close = _study_actions[
        "do_export_study_dialog_close"
    ]
    do_import_study_open = _study_actions["do_import_study_open"]
    do_import_study_close = _study_actions[
        "do_import_study_close"
    ]
    do_import_study_upload = _study_actions[
        "do_import_study_upload"
    ]
    do_import_study_run = _study_actions["do_import_study_run"]
    do_import_study_load_from_disk = _study_actions[
        "do_import_study_load_from_disk"
    ]
    _ingest_uploaded_bundle = _study_actions[
        "ingest_uploaded_bundle"
    ]

    # V1 Phase A — µCT segmentation dialog handlers. The
    # closure inside register() owns the heavy NumPy arrays +
    # the resolved Segmenter so they stay off the Trame state
    # proxy (no msgpack-over-WebSocket overhead per slice).
    # `get_active_project_dir` lets the Phase A.4 persistence
    # write into <project>/uct/ on save.
    _segment_uct_actions = _actions.segment_uct.register(
        state,
        get_active_project_dir=lambda: get_active().out_dir,
        on_recon_meshes_ready=_update_recon_viewport,
        cancel_token=_cancel,
    )
    _do_open_segment_uct_dialog_raw = _segment_uct_actions[
        "do_open_segment_uct_dialog"
    ]
    _do_close_segment_uct_dialog_raw = _segment_uct_actions[
        "do_close_segment_uct_dialog"
    ]

    def do_open_segment_uct_dialog(*args, **kwargs):
        """Wrapper around the segment-µCT open handler. Also
        refreshes the bundle list at open time so the user sees
        any newly-saved bundles without needing a project
        reload."""
        do_refresh_uct_bundles()
        return _do_open_segment_uct_dialog_raw(*args, **kwargs)

    def do_close_segment_uct_dialog(*args, **kwargs):
        """Wrapper around the segment-µCT close handler. Bumps
        `state.uct_bundle_items` after close so a freshly-
        generated bundle shows up in the import-wizard picker
        on the user's NEXT step (when they click File → Import
        Nerve)."""
        ret = _do_close_segment_uct_dialog_raw(*args, **kwargs)
        do_refresh_uct_bundles()
        return ret
    do_load_uct_stack = _segment_uct_actions["do_load_uct_stack"]
    do_clear_uct_stack = _segment_uct_actions[
        "do_clear_uct_stack"
    ]
    do_run_uct_segmentation = _segment_uct_actions[
        "do_run_uct_segmentation"
    ]
    do_label_uct_proposal = _segment_uct_actions[
        "do_label_uct_proposal"
    ]
    do_generate_epi = _segment_uct_actions["do_generate_epi"]
    do_refine_masks = _segment_uct_actions["do_refine_masks"]
    do_save_uct_segmentation = _segment_uct_actions[
        "do_save_uct_segmentation"
    ]
    do_finalize_segmentation = _segment_uct_actions[
        "do_finalize_segmentation"
    ]
    # SAM2 video keyframe-propagation handlers.
    do_toggle_keyframe = _segment_uct_actions[
        "do_toggle_keyframe"
    ]
    do_propagate_from_keyframes = _segment_uct_actions[
        "do_propagate_from_keyframes"
    ]
    # V1 Phase B — Step-2 3D-reconstruction handlers.
    do_recon_next = _segment_uct_actions["do_recon_next"]
    do_recon_back = _segment_uct_actions["do_recon_back"]
    do_run_reconstruction = _segment_uct_actions[
        "do_run_reconstruction"
    ]
    do_run_reconstruction_preview = _segment_uct_actions[
        "do_run_reconstruction_preview"
    ]

    def do_finish_recon(*_args) -> None:
        """Hand the just-generated µCT bundle off to the nerve-
        import wizard.

        Steps:
          1. Close the Segment-µCT dialog.
          2. Rescan <project>/uct/nerve_3d/ so the wizard
             picker sees the freshly-written bundle.
          3. Flip `import_source_type` to "uct_bundle" + set
             `selected_uct_bundle` to the timestamp id stashed
             by `do_run_reconstruction`.
          4. Reset the wizard to Step 1 + open it.
        The user lands on the wizard's Step 1 with the bundle
        tile + bundle picker pre-selected; from there they
        step through to fiber-trajectory + muscle setup. The
        bundle's epi.stl + fascicle_*.stl are wired into the
        multi-region PLC (see `assemble_multi_domain_plc`'s
        `inner_surfaces` kwarg), and the fiber seeder reads
        the per-fascicle sidecar npz to constrain seed points
        to fascicle interiors — that pipeline was wired in
        Phase 5 and Phase 6 of the bundle import work and
        does not need re-implementing here.
        """
        bid = str(
            getattr(state, "uct_last_bundle_id", "") or "",
        )
        if not bid:
            with state:
                state.uct_recon_status = (
                    "Generate the 3D nerve first — there's "
                    "no bundle to hand over yet."
                )
            return
        # Close the segment dialog (and refresh the bundle
        # list via the wrapped close handler).
        try:
            do_close_segment_uct_dialog()
        except Exception:                                # noqa: BLE001
            with state:
                state.show_segment_uct_dialog = False
        # Make absolutely sure the bundle list is current.
        do_refresh_uct_bundles()
        with state:
            state.import_source_type = "uct_bundle"
            state.selected_uct_bundle = bid
            state.import_stepper_step = "1"
            state.show_import_stepper = True

    # M47 — histology bundle import dialog. Compact entry point
    # for the "I already have masks, skip SAM2" workflow.
    # Writes STLs into the same uct/nerve_3d/<ts>/ layout the
    # µCT recon path uses, so the existing import wizard picks
    # the new bundle up without extra plumbing.
    def _open_import_stepper_after_bundle(bundle_id: str):
        """M47 — open the import wizard pre-selecting the
        newly-extruded histology bundle. Uses the dedicated
        `histo_bundle` source type (third tile) so the
        existing µCT path stays untouched."""
        print(
            f"[handoff] opening wizard with histo bundle "
            f"{bundle_id!r}",
            flush=True,
        )
        do_refresh_histo_bundles()
        with state:
            state.import_source_type = "histo_bundle"
            state.selected_histo_bundle = bundle_id
            state.import_stepper_step = "1"
            state.show_import_stepper = True
        print(
            f"[handoff] state after: "
            f"import_source_type={state.import_source_type!r}, "
            f"selected_histo_bundle="
            f"{state.selected_histo_bundle!r}, "
            f"histo_bundle_items count="
            f"{len(state.histo_bundle_items or [])}",
            flush=True,
        )

    # M47 — source-type tile click handlers. Bound via the
    # widgets' `click=` kwarg in import_stepper.py so the
    # server-side state change is unambiguous; previously the
    # tiles used inline `@click="import_source_type = '...'"`
    # in raw_attrs which didn't reliably trigger Vue's two-way
    # state sync to the server in this trame-client build.
    # The µCT + histology tiles also short-circuit when no
    # bundles are available so a click on the greyed-out
    # tile doesn't flip the source type into an unusable state.
    def do_select_source_stl(*_args) -> None:
        print(
            "[tile-click] STL "
            f"(before: {state.import_source_type!r})",
            flush=True,
        )
        with state:
            state.import_source_type = "stl"

    def do_select_source_uct_bundle(*_args) -> None:
        items = list(
            getattr(state, "uct_bundle_items", []) or [],
        )
        print(
            f"[tile-click] uct_bundle "
            f"(items={len(items)}, "
            f"before: {state.import_source_type!r})",
            flush=True,
        )
        if not items:
            return
        with state:
            state.import_source_type = "uct_bundle"

    def do_select_source_histo_bundle(*_args) -> None:
        items = list(
            getattr(state, "histo_bundle_items", []) or [],
        )
        print(
            f"[tile-click] histo_bundle "
            f"(items={len(items)}, "
            f"before: {state.import_source_type!r})",
            flush=True,
        )
        if not items:
            return
        with state:
            state.import_source_type = "histo_bundle"
        print(
            f"[tile-click] histo_bundle done "
            f"(after: {state.import_source_type!r})",
            flush=True,
        )

    # M47 — keep the show_picker_* booleans in sync with the
    # current source type. The wizard's v_show bindings use
    # these single-state-var booleans (NOT complex JS
    # expressions on `import_source_type`) because complex
    # `v_show="a === 'x' && b !== 'y'"` expressions are
    # observed not to reactively re-evaluate in this trame /
    # Vuetify build, while plain `v_show="show_picker_xxx"` is
    # rock-solid. Also fires once at registration time so the
    # initial render matches the default `import_source_type`.
    def _recompute_load_nerve_blocked() -> None:
        """Set `load_nerve_blocked` based on whether the
        currently-selected source has a value. Called from the
        watcher below — any of the four state vars it reads can
        change and require a recompute."""
        ist = str(
            getattr(state, "import_source_type", "stl") or "stl",
        )
        if ist == "uct_bundle":
            blocked = not bool(
                getattr(state, "selected_uct_bundle", "") or "",
            )
        elif ist == "histo_bundle":
            blocked = not bool(
                getattr(state, "selected_histo_bundle", "") or "",
            )
        else:
            blocked = not bool(
                getattr(state, "selected_file", "") or "",
            )
        state.load_nerve_blocked = blocked

    @state.change(
        "import_source_type",
        "selected_uct_bundle",
        "selected_histo_bundle",
        "selected_file",
        "selected_epi_file",
    )
    def _on_import_source_type_change(
        import_source_type=None, **_kw,
    ):
        ist = str(
            getattr(state, "import_source_type", "stl") or "stl",
        )
        state.show_picker_stl = (
            ist not in ("uct_bundle", "histo_bundle")
        )
        state.show_picker_uct_bundle = (ist == "uct_bundle")
        state.show_picker_histo_bundle = (ist == "histo_bundle")
        # STL flow: an optional explicit epineurium surface turns the
        # import into a multi-region epi+endo build (the same
        # uct_bundle shape the µCT/histology bundles use). The step-2
        # offset generator + the epi note are gated on dedicated
        # booleans — compound v_show expressions are unreliable in
        # this trame/Vuetify build (see M47).
        _has_epi = bool(
            state.show_picker_stl
            and (getattr(state, "selected_epi_file", "") or "")
        )
        state.stl_has_epi = _has_epi
        state.show_stl_offset = (
            bool(state.show_picker_stl) and not _has_epi
        )
        state.show_stl_epi_note = _has_epi
        if _has_epi:
            # Mesh the real epi region — matches fig 8's
            # new_human_mesh.py recipe (use_epi=True + bundle epi);
            # the inward offset is skipped at PLC time because inner
            # fascicle surfaces are present (pipeline/plc.py).
            state.use_epi = True
        _recompute_load_nerve_blocked()
        print(
            f"[picker-vis] ist={ist!r} → "
            f"stl={state.show_picker_stl}, "
            f"uct={state.show_picker_uct_bundle}, "
            f"histo={state.show_picker_histo_bundle}, "
            f"load_nerve_blocked={state.load_nerve_blocked}",
            flush=True,
        )

    @state.change("epi_upload_file")
    def _on_epi_upload(**_kw):
        """Persist an uploaded epineurium surface into the project's
        upload dir and select it. Parallels the generic `upload_file`
        sink (watchers/auth_upload.py) but targets `selected_epi_file`,
        which the epi+endo STL build in do_load_geometry reads."""
        info = getattr(state, "epi_upload_file", None)
        if not info:
            return
        entries = info if isinstance(info, list) else [info]
        saved = None
        for entry in entries:
            if not entry or "name" not in entry:
                continue
            content = entry.get("content")
            if not content:
                continue
            target = Path(get_active().upload_dir) / entry["name"]
            with open(target, "wb") as fh:
                fh.write(content)
            saved = target
        if saved is None:
            return
        state.data_files = list_data_files()
        try:
            state.selected_epi_file = str(saved.relative_to(HERE))
        except ValueError:
            state.selected_epi_file = str(saved)

    _bundle_actions = _actions.bundle_import.register(
        state,
        get_active_project_dir=lambda: get_active().out_dir,
        do_open_import_stepper=(
            _open_import_stepper_after_bundle
        ),
    )
    do_open_bundle_import_dialog = _bundle_actions[
        "do_open_bundle_import_dialog"
    ]
    do_close_bundle_import_dialog = _bundle_actions[
        "do_close_bundle_import_dialog"
    ]
    do_detect_bundle_files = _bundle_actions[
        "do_detect_bundle_files"
    ]
    do_run_bundle_import = _bundle_actions[
        "do_run_bundle_import"
    ]

    # Slice-scrubber watcher — re-renders the preview when the
    # user releases the VSlider thumb. Bound here (not in the
    # actions module) because @state.change is a decorator on
    # the live state instance.
    _on_uct_slice_change_cb = _segment_uct_actions[
        "_on_uct_slice_change"
    ]

    @state.change("uct_slice_idx")
    def _on_uct_slice_change(uct_slice_idx, **_kw):
        _on_uct_slice_change_cb(uct_slice_idx, **_kw)

    _on_uct_crop_change_cb = _segment_uct_actions[
        "_on_uct_crop_change"
    ]

    @state.change("uct_crop_x_range", "uct_crop_y_range")
    def _on_uct_crop_change(
        uct_crop_x_range, uct_crop_y_range, **_kw,
    ):
        _on_uct_crop_change_cb(
            uct_crop_x_range, uct_crop_y_range, **_kw,
        )

    _on_uct_backend_change_cb = _segment_uct_actions[
        "_on_uct_backend_change"
    ]

    @state.change("uct_backend_choice")
    def _on_uct_backend_change(uct_backend_choice, **_kw):
        _on_uct_backend_change_cb(uct_backend_choice, **_kw)

    _on_uct_clahe_change_cb = _segment_uct_actions[
        "_on_uct_clahe_change"
    ]

    @state.change("uct_clahe")
    def _on_uct_clahe_change(uct_clahe, **_kw):
        _on_uct_clahe_change_cb(uct_clahe, **_kw)

    _on_uct_click_payload_cb = _segment_uct_actions[
        "_on_uct_click_payload"
    ]

    @state.change("uct_click_payload")
    def _on_uct_click_payload(uct_click_payload, **_kw):
        _on_uct_click_payload_cb(uct_click_payload, **_kw)

    _on_uct_zoom_change_cb = _segment_uct_actions[
        "_on_uct_zoom_change"
    ]

    @state.change("uct_zoom_x_range", "uct_zoom_y_range")
    def _on_uct_zoom_change(
        uct_zoom_x_range, uct_zoom_y_range, **_kw,
    ):
        _on_uct_zoom_change_cb(
            uct_zoom_x_range, uct_zoom_y_range, **_kw,
        )

    _on_uct_paint_payload_cb = _segment_uct_actions[
        "_on_uct_paint_payload"
    ]

    @state.change("uct_paint_payload")
    def _on_uct_paint_payload(uct_paint_payload, **_kw):
        _on_uct_paint_payload_cb(uct_paint_payload, **_kw)

    # V1 Phase B — slice-range watcher for Step-2 coverage
    # readout. Refires _refresh_recon_coverage so the
    # "N / M slices annotated · K ZOH-filled" line stays
    # accurate as the user tweaks the range.
    _on_uct_recon_range_change_cb = _segment_uct_actions[
        "_on_uct_recon_range_change"
    ]

    @state.change(
        "uct_recon_slice_start", "uct_recon_slice_end",
    )
    def _on_uct_recon_range_change(
        uct_recon_slice_start, uct_recon_slice_end, **_kw,
    ):
        _on_uct_recon_range_change_cb(
            uct_recon_slice_start,
            uct_recon_slice_end,
            **_kw,
        )

    # V1 — view-style watchers for the Step-3 pyvista plotter.
    # Edges + quality colormap require rebuilding actors (the
    # PyVista API doesn't expose "swap scalars on existing
    # actor" cleanly). The meshes cache makes this cheap —
    # marching cubes isn't re-run, just the actor styling.
    @state.change(
        "uct_recon_show_edges", "uct_recon_color_by_quality",
    )
    def _on_recon_view_style_change(
        uct_recon_show_edges, uct_recon_color_by_quality,
        **_kw,
    ):
        if not _uct_recon_meshes_cache:
            return
        _update_recon_viewport(
            _uct_recon_meshes_cache,
            keep_camera=True,
        )

    # Per-mesh visibility: the legend renders one checkbox per
    # entry in `uct_recon_mesh_items`; flipping a checkbox
    # updates the list element's `.visible` flag, which fires
    # this watcher. We push the flag through to the actor
    # directly (cheaper than rebuilding) and re-render.
    @state.change("uct_recon_mesh_items")
    def _on_recon_mesh_visibility_change(
        uct_recon_mesh_items, **_kw,
    ):
        if not uct_recon_mesh_items:
            return
        try:
            renderer = pl_uct_recon.renderer
            for entry in uct_recon_mesh_items:
                name = str(entry.get("name", ""))
                visible = bool(entry.get("visible", True))
                actor = renderer.actors.get(name)
                if actor is not None:
                    actor.SetVisibility(int(visible))
            ctrl.view_uct_recon_update()
        except Exception:                                # noqa: BLE001
            pass

    # F2.2 — HTTP upload route. Streams multipart POST bodies
    # straight to disk under _DOWNLOADS_DIR, bypassing the
    # browser msgpack + wslink WS caps that wreck big bundle
    # uploads. The callback runs the manifest-peek action on
    # the asyncio main loop once the upload completes, so the
    # dialog auto-advances to "ready to import" without
    # needing a second user click.
    def _on_study_uploaded(server_path: str) -> None:
        with state:
            state.study_import_path_on_disk = server_path
        state.flush()
        do_import_study_load_from_disk()

    from golgi.projects import upload_route as _upload_route
    _upload_route.register(
        server,
        downloads_dir=_DOWNLOADS_DIR,
        on_upload_complete=_on_study_uploaded,
    )

    # V1 Phase A.6 — µCT upload route. Streams multipart POST
    # bodies into <active project>/uct/uploads/ (per-project so
    # the source image colocates with the segmentation that
    # consumes it). Lookup of the destination dir happens at
    # upload time via get_active().out_dir so project switches
    # don't need re-registration.
    def _on_uct_uploaded(server_path: str) -> None:
        with state:
            state.uct_file_path = server_path
            state.uct_status = (
                f"Uploaded → {server_path}. Loading…"
            )
        state.flush()
        # Trigger the load action so the user gets metadata +
        # slice preview without a second click.
        do_load_uct_stack()

    def _on_uct_upload_progress(
        idx: int, name: str,
        file_bytes: int, total_bytes_so_far: int,
    ) -> None:
        """Per-file progress callback from the upload route.
        Pushes a `received file N · <name>` line into the busy
        status so the in-drop-zone overlay shows file-level
        granularity on multi-file (DICOM series) uploads. The
        XHR upload progress bar continues to advance via the
        JS-side xhr.upload.onprogress event — this just adds
        file boundaries to the readout."""
        mb = float(file_bytes) / (1024.0 * 1024.0)
        with state:
            state.uct_upload_status = (
                f"Received file {idx} · {name} "
                f"({mb:.2f} MB)"
            )
            state.busy_log = (
                f"file {idx} · {name} ({mb:.2f} MB)"
            )
        state.flush()

    from golgi.projects import uct_route as _uct_route
    _uct_route.register(
        server,
        get_active_project_dir=lambda: get_active().out_dir,
        on_upload_complete=_on_uct_uploaded,
        on_upload_progress=_on_uct_upload_progress,
    )

    # F2.2 — File-pick watcher for the Import Study dialog. The
    # VFileInput's @change DOM event doesn't fire reliably across
    # trame builds (state.change on the v-model does), so this
    # watcher is the real "user picked a file" entry point.
    @state.change("study_import_upload")
    def _on_study_import_upload(study_import_upload, **_kwargs):
        # Diagnostic — confirms the watcher is firing. Pair with
        # the print inside _ingest_uploaded_bundle to trace
        # the full path.
        print(
            f"[study-import] state.change fired · "
            f"v-model type="
            f"{type(study_import_upload).__name__} · "
            f"truthy={bool(study_import_upload)}",
            flush=True,
        )
        _ingest_uploaded_bundle(study_import_upload)

    # F2.3.b — push the figure-registry meta into state so the
    # bulk Exports drawer can iterate via v-for. Categories are
    # listed in registry order so the drawer reads top-to-bottom
    # in the same direction as the analysis tabs.
    from golgi.figures import registry as _figures_registry
    state.exports_registry_meta = [
        {
            "id": spec.id,
            "title": spec.title,
            "category": spec.category,
        }
        for spec in _figures_registry.REGISTRY
    ]
    _seen_cats: list[str] = []
    for spec in _figures_registry.REGISTRY:
        if spec.category not in _seen_cats:
            _seen_cats.append(spec.category)
    state.exports_registry_categories = _seen_cats
    # Pre-grouped variant of the same data, shaped for a single-
    # level v-for in the bulk Exports drawer. The original
    # nested `<template v-for>` + `.filter()` pattern in v1 lost
    # rows for some categories on first paint (Vue 3 nested
    # template v-for needs explicit :key on each level — easier
    # to flatten the data and iterate once).
    _grouped: list[dict] = []
    for cat in _seen_cats:
        items = [
            {"id": s.id, "title": s.title}
            for s in _figures_registry.REGISTRY
            if s.category == cat
        ]
        _grouped.append({"category": cat, "items": items})
    state.exports_registry_grouped = _grouped

    def _export_btn(fig_id: str) -> None:
        """Tiny shim so each figure tile spells the button as
        `_export_btn("fem.axis_line")` instead of a 3-line call to
        the component. The component takes care of placement +
        popover + the Download anchor."""
        _ui.components.figure_export_btn.render(
            fig_id=fig_id,
            do_export_single_figure=do_export_single_figure,
        )

    with VAppLayout(server, full_height=True) as layout:
        # v_model on VAppBar uses Vuetify's built-in slide-in/out
        # animation and reclaims the layout space when false — so
        # in welcome mode VMain fills the full viewport and the
        # big wordmark logo + tiles get the page to themselves.
        _ui.navbar.render(
            logo_url=_LOGO_URL,
            do_close_all_tabs=do_close_all_tabs,
            do_save_project=do_save_project,
            do_show_close_dialog=do_show_close_dialog,
            do_open_profile_dialog=do_open_profile_dialog,
            do_open_generate_report_dialog=(
                do_open_generate_report_dialog
            ),
            do_open_import_study_dialog=do_import_study_open,
            do_export_study=do_export_study,
            do_open_segment_uct_dialog=(
                do_open_segment_uct_dialog
            ),
            do_open_bundle_import_dialog=(
                do_open_bundle_import_dialog
            ),
        )

        # ----- Import drawer -----
        _ui.drawers.import_drawer.render(
            do_load_geometry=do_load_geometry,
            export_btn=_export_btn,
        )

        # ----- Cuff & Electrodes drawer -----
        _ui.drawers.cuff_electrodes.render(
            duke_electrode_type=DUKE_ELECTRODE_TYPE,
            electrode_types=ELECTRODE_TYPES,
            edit_icon_url=_EDIT_ICON_URL,
            do_add_design=do_add_design,
            do_save_rename_eid=do_save_rename_eid,
            do_cancel_rename_eid=do_cancel_rename_eid,
            do_open_cuff_designer=do_open_cuff_designer,
        )

        # ----- Mesh drawer -----
        _ui.drawers.mesh.render(
            do_build_mesh=do_build_mesh,
            export_btn=_export_btn,
        )

        # ----- Conductivities drawer -----
        _ui.drawers.conductivities.render(
            sigma_tag_map=SIGMA_TAG_MAP,
            sigma_label_map=SIGMA_LABEL_MAP,
            do_reset_sigma=do_reset_sigma,
            do_update_sigma=do_update_sigma,
        )

        # F3.2-M2.1b — Fibers drawer dropped from build_app.
        # Trajectory generation params + branch summary + rename
        # all live in the import stepper (Step 3). The drawer
        # source file stays in `golgi/ui/drawers/fibers.py` for
        # the moment in case we resurrect it as a separate
        # branch-stats dialog later (M2.1d), but it isn't
        # mounted by any UI surface today.

        # ===== ANALYSIS DRAWERS =====
        # Only the Solve (FEM) tab still renders a side drawer;
        # the Single-fiber and Population tabs moved their
        # controls into their respective analysis panels (the
        # show_fiber / show_pop state vars still drive the
        # tab-active watcher that flips active_analysis).
        _ui.drawers.analysis.render(do_solve_fem=do_solve_fem)

        # ----- Bulk Exports drawer (F2.3.b) -----
        _ui.drawers.exports.render(
            do_bulk_export=do_bulk_export,
            do_bulk_export_select_all_available=(
                do_bulk_export_select_all_available
            ),
            do_bulk_export_clear=do_bulk_export_clear,
        )

        # (the Render drawer was replaced by an always-visible
        # floating legend overlay — see below, inside VMain.)

        # ----- Busy lightbox -----
        _ui.busy_lightbox.render(do_request_cancel=do_request_cancel)

        # ----- Center: 3D viewport + analysis area (flex column) -----
        with v3.VMain():
            with v3.VContainer(
                fluid=True,
                classes="pa-0 fill-height golgi-central",
                style="background-color: white; position:relative;",
            ):
                # ----- Welcome view (project picker) -----
                # Absolute-positioned overlay shown when view_mode
                # === 'welcome'. The plotter_ui below stays mounted
                # (display: none would not unmount it; the welcome
                # view paints over it). Tiles: one "+ New project"
                # + one per existing project under PROJECTS_ROOT.
                _ui.welcome.render(
                    logo_text_url=_LOGO_TEXT_URL,
                    ext_site_url=_EXT_SITE_URL,
                    ext_link_url=_EXT_LINK_URL,
                    login_icon_url=_LOGIN_ICON_URL,
                    do_open_profile_dialog=do_open_profile_dialog,
                    do_logout=do_logout,
                    do_show_new_project_dialog=do_show_new_project_dialog,
                    do_open_auth_dialog=do_open_auth_dialog,
                )

                # Solve tab — 2x2 grid overlay. The 3D viewport
                # repositions itself into the top-left cell via
                # `.golgi-viewport.mode-grid-tl`; the heatmap /
                # Vₑ-E_z ribbon / Vₑ-AF tiles fill the other 3
                # cells. Pointer events are off on the grid
                # container so the viewport stays interactive;
                # each tile re-enables them on itself.
                with html.Div(
                    v_show=(
                        "viewport_mode === 'analysis' "
                        "&& active_analysis === 'solve'",
                    ),
                    classes="golgi-solve-grid",
                ):
                    # F3.2c Config chip retired — the legend toprow
                    # combobox (top-right of the viewport) now
                    # carries both design + solved-config selection
                    # in one adaptive picker (`design_config_key`).
                    # Empty top-left frame (the viewport
                    # actually occupies this cell via its own
                    # absolute positioning).
                    html.Div(
                        classes="golgi-solve-tile tile-render",
                    )
                    # --- Heatmap tile (top-right) ---
                    with html.Div(
                        id="tile-fem-heatmap",
                        classes="golgi-solve-tile tile-heatmap",
                    ):
                        with html.Div(
                            classes="golgi-tile-header",
                        ):
                            html.Span(
                                "Vₑ slice heatmap",
                                classes="golgi-tile-title",
                            )
                            with html.Div(
                                classes="golgi-tile-actions",
                            ):
                                html.Button(
                                    "PNG",
                                    type="button",
                                    classes=(
                                        "golgi-tile-export-btn"
                                    ),
                                    click=(
                                        "window.golgi_export_plot"
                                        "('tile-fem-heatmap', "
                                        " 'png')"
                                    ),
                                )
                                html.Button(
                                    "SVG",
                                    type="button",
                                    classes=(
                                        "golgi-tile-export-btn"
                                    ),
                                    click=(
                                        "window.golgi_export_plot"
                                        "('tile-fem-heatmap', "
                                        " 'svg')"
                                    ),
                                )
                        with html.Div(
                            classes="golgi-tile-body",
                            v_show=("has_fem",),
                        ):
                            _export_btn("fem.slice_volume")
                            if twp is not None:
                                twp.Figure(
                                    state_variable_name=(
                                        "fem_slice_figure"
                                    ),
                                    display_logo=False,
                                    display_mode_bar=True,
                                )
                            else:
                                html.Div(
                                    "Install `trame-plotly` to "
                                    "enable interactive plots.",
                                    style=("padding: 20px; "
                                            "color:#888; "
                                            "font-size:12px;"),
                                )
                        html.Div(
                            "Run a FEM solve to see the heatmap.",
                            v_show=("!has_fem",),
                            style=("padding: 18px; color:#888; "
                                    "font-size:12px; "
                                    "font-style:italic;"),
                        )
                    # --- Vₑ + E_z ribbon tile (bottom-left) ---
                    with html.Div(
                        id="tile-fem-veez",
                        classes="golgi-solve-tile tile-veez",
                    ):
                        with html.Div(
                            classes="golgi-tile-header",
                        ):
                            html.Span(
                                "Vₑ and E_z along fibers",
                                classes="golgi-tile-title",
                            )
                            with html.Div(
                                classes="golgi-tile-actions",
                            ):
                                html.Button(
                                    "PNG",
                                    type="button",
                                    classes=(
                                        "golgi-tile-export-btn"
                                    ),
                                    click=(
                                        "window.golgi_export_plot"
                                        "('tile-fem-veez', "
                                        " 'png')"
                                    ),
                                )
                                html.Button(
                                    "SVG",
                                    type="button",
                                    classes=(
                                        "golgi-tile-export-btn"
                                    ),
                                    click=(
                                        "window.golgi_export_plot"
                                        "('tile-fem-veez', "
                                        " 'svg')"
                                    ),
                                )
                        with html.Div(
                            classes="golgi-tile-body",
                            v_show=("has_fem",),
                        ):
                            _export_btn("fem.axis_line")
                            if twp is not None:
                                twp.Figure(
                                    state_variable_name=(
                                        "fem_axis_figure"
                                    ),
                                    display_logo=False,
                                    display_mode_bar=True,
                                )
                        html.Div(
                            "Run a FEM solve with fibers to see "
                            "Vₑ(s) and E_z(s) ribbons.",
                            v_show=("!has_fem",),
                            style=("padding: 18px; color:#888; "
                                    "font-size:12px; "
                                    "font-style:italic;"),
                        )
                    # --- Vₑ + AF tile (bottom-right) ---
                    with html.Div(
                        id="tile-fem-af",
                        classes="golgi-solve-tile tile-af",
                    ):
                        with html.Div(
                            classes="golgi-tile-header",
                        ):
                            html.Span(
                                "Vₑ and activation function",
                                classes="golgi-tile-title",
                            )
                            with html.Div(
                                classes="golgi-tile-actions",
                            ):
                                html.Button(
                                    "PNG",
                                    type="button",
                                    classes=(
                                        "golgi-tile-export-btn"
                                    ),
                                    click=(
                                        "window.golgi_export_plot"
                                        "('tile-fem-af', 'png')"
                                    ),
                                )
                                html.Button(
                                    "SVG",
                                    type="button",
                                    classes=(
                                        "golgi-tile-export-btn"
                                    ),
                                    click=(
                                        "window.golgi_export_plot"
                                        "('tile-fem-af', 'svg')"
                                    ),
                                )
                        with html.Div(
                            classes="golgi-tile-body",
                            v_show=("has_fem",),
                        ):
                            _export_btn("fem.activation_fn")
                            if twp is not None:
                                twp.Figure(
                                    state_variable_name=(
                                        "fem_af_figure"
                                    ),
                                    display_logo=False,
                                    display_mode_bar=True,
                                )
                        # Per-tile interactive controls — match
                        # nerve_studio's §10 sliders. Moving them
                        # in-tile (vs the sidebar) means the
                        # user can adjust the AF smoothing and the
                        # selected fiber while looking at the
                        # plot.
                        with html.Div(
                            classes="golgi-tile-controls",
                            v_show=("has_fem",),
                        ):
                            with html.Div(classes="slider-row"):
                                html.Label("select fiber")
                                v3.VSlider(
                                    v_model=("fem_fiber_sel",),
                                    min=0, max=999, step=1,
                                    density="compact",
                                    hide_details=True,
                                    thumb_label=True,
                                )
                            with html.Div(classes="slider-row"):
                                html.Label("AF smoothing")
                                v3.VSlider(
                                    v_model=("fem_sg_window",),
                                    min=5, max=301, step=2,
                                    density="compact",
                                    hide_details=True,
                                    thumb_label=True,
                                )
                        html.Div(
                            "Run a FEM solve with fibers to see "
                            "the activation function.",
                            v_show=("!has_fem",),
                            style=("padding: 18px; color:#888; "
                                    "font-size:12px; "
                                    "font-style:italic;"),
                        )

                # NON-Solve panels (Fiber / Population). Solve
                # has its own 2x2 grid overlay above. Single-fiber
                # mode toggles `is-fiber-mode` so the area gets
                # extra top padding to clear the absolute-positioned
                # viewport band (`.golgi-viewport.mode-fiber-band`).
                with html.Div(
                    v_show=(
                        "viewport_mode === 'analysis' "
                        "&& active_analysis !== 'solve'",
                    ),
                    classes=(
                        "['golgi-analysis-area', "
                        "active_analysis === 'fiber' "
                        "  ? 'is-fiber-mode' "
                        "  : (active_analysis === 'population' "
                        "      ? 'is-pop-mode' : '')]",
                    ),
                ):
                    # ---- Fiber panel (§10-12) ----
                    # Single unified panel: header, subtitle,
                    # trajectory/backend/model/diameter row, pulse
                    # design, run button, output tiles, sim log.
                    # The viewport sits as a plain tile ABOVE this
                    # panel (see `.golgi-viewport.mode-fiber-band`
                    # CSS) so the user can keep an eye on the
                    # highlighted trajectory while editing inputs.
                    with html.Div(
                        v_show=("active_analysis === 'fiber'",),
                        classes="golgi-analysis-panel "
                                "golgi-fiber-panel",
                    ):
                        # Title row — H2 + an info icon whose
                        # tooltip carries the longer help text
                        # the panel used to surface as a
                        # subtitle paragraph. Cleaner first-
                        # screen layout; users who need the
                        # explanation can hover the (i).
                        with html.Div(
                            classes="golgi-fiber-title-row",
                        ):
                            html.H2(
                                "Single fiber — playground"
                            )
                            with v3.VTooltip(
                                location="top",
                                max_width=420,
                            ):
                                with v3.Template(
                                    v_slot_activator=(
                                        "{ props }",
                                    ),
                                ):
                                    with v3.VBtn(
                                        v_bind="props",
                                        icon=True,
                                        size="small",
                                        variant="text",
                                        density="compact",
                                        classes=(
                                            "golgi-fiber-info-btn"
                                        ),
                                    ):
                                        v3.VIcon(
                                            "mdi-information-outline",
                                            size="20",
                                            color="grey-darken-1",
                                        )
                                html.Span(
                                    "Pick one or more "
                                    "trajectories, choose a "
                                    "backend + fiber model, "
                                    "design a stim pulse, and "
                                    "run the simulation. All "
                                    "selected trajectories are "
                                    "highlighted in the 3D "
                                    "viewport above; the sim "
                                    "loops over them and you "
                                    "can browse each fiber's "
                                    "V_m output after the run "
                                    "using the result picker."
                                )
                        # ---- Trajectory / backend / model /
                        #      diameter row ----
                        with html.Div(
                            classes="golgi-fiber-controls",
                        ):
                            # VAutocomplete — multi-select with
                            # chips + clearable. Was VCombobox
                            # previously, but VCombobox lets the
                            # user TYPE arbitrary text + press
                            # Enter to create a free-text chip
                            # (that's the design difference
                            # between Combobox and Autocomplete).
                            # Free-text entries get a string value
                            # that the chip-slot title-lookup
                            # can't resolve back to a
                            # "Branch X · Fiber N" string, so the
                            # chip silently loses its prefix and
                            # shows "Fiber {garbage}". Switching
                            # to VAutocomplete restricts the
                            # selection to the items list, which
                            # is what we want here: free text on
                            # a fiber index is never useful.
                            #
                            # The prepend-item slot holds a VTabs
                            # row, ONE tab per detected branch
                            # ("Branch 0 / Branch 1 / ..."). The
                            # Vue `:items` expression filters
                            # fiber_sel_items to just the active
                            # branch's fibers so the dropdown is
                            # never a flat 500-item list. Each
                            # chip carries a tab10-coloured circle
                            # that matches the fiber's tint in
                            # the 3-D viewport so the chip ↔
                            # trajectory mapping is unambiguous.
                            # Selecting a fiber adds its index to
                            # `fiber_sel_indices`; on Run,
                            # do_run_fiber_sim loops over the set.
                            with v3.VAutocomplete(
                                v_model=("fiber_sel_indices",),
                                items=(
                                    "fiber_sel_items.filter("
                                    "  it => "
                                    "    it.branch.toString() "
                                    "    === fiber_sel_tab"
                                    ")",
                                ),
                                item_title="title",
                                item_value="value",
                                label="trajectories",
                                multiple=True,
                                chips=True,
                                clearable=True,
                                closable_chips=True,
                                return_object=False,
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                classes="golgi-fiber-sel-traj",
                                disabled=("fiber_sim_busy",),
                                menu_props=(
                                    "{ maxHeight: '420px' }",
                                ),
                            ):
                                # `chip` slot — custom chip
                                # rendering for each selected
                                # fiber. Prepends a coloured
                                # circle (tab10 by fiber index)
                                # before the title text so the
                                # user can spot which chip maps
                                # to which trajectory in the 3-D
                                # viewport. Lookup goes through
                                # fiber_sel_items by value
                                # because the chip slot's `item`
                                # only carries the bare value
                                # when `return-object=false`.
                                with v3.Template(
                                    v_slot_chip=(
                                        "{ props, item }",
                                    ),
                                ):
                                    with v3.VChip(
                                        v_bind="props",
                                        size="small",
                                        closable=True,
                                    ):
                                        # `Number(...)` on BOTH
                                        # sides of the comparison
                                        # so number/string drift
                                        # between renders (which
                                        # happened intermittently
                                        # under the old VCombobox)
                                        # can't break the lookup.
                                        # Falls back to a neutral
                                        # grey + "Fiber N" if the
                                        # value can't be found at
                                        # all.
                                        html.Span(
                                            classes=(
                                                "golgi-fiber-chip-dot"
                                            ),
                                            style=(
                                                "'background:' + ("
                                                "  (fiber_sel_items."
                                                "    find(it => "
                                                "      Number(it.value) "
                                                "      === Number(item.value)"
                                                "    ) || {})"
                                                "  .color || '#888'"
                                                ")",
                                            ),
                                        )
                                        # Look up the title in the
                                        # FULL fiber_sel_items list
                                        # (not the filtered set
                                        # passed as `items=`) so
                                        # chips with values from a
                                        # different branch tab
                                        # still display the full
                                        # "Branch X · Fiber N"
                                        # string.
                                        html.Span(
                                            "{{ ("
                                            "  fiber_sel_items.find("
                                            "    it => "
                                            "    Number(it.value) "
                                            "    === Number(item.value)"
                                            "  ) || {}"
                                            ").title || "
                                            "('Fiber ' + item.value) }}"
                                        )
                                # `prepend-item` slot — renders
                                # ABOVE the items inside the
                                # dropdown menu. One tab per
                                # branch detected in fiber data.
                                with v3.Template(
                                    v_slot_prepend_item=True,
                                ):
                                    with v3.VTabs(
                                        v_model=("fiber_sel_tab",),
                                        density="compact",
                                        grow=True,
                                        slider_color="primary",
                                        classes="golgi-fiber-tab-strip",
                                    ):
                                        v3.VTab(
                                            "{{ m.label }}",
                                            v_for=(
                                                "m in pop_branches_meta"
                                            ),
                                            key="m.idx",
                                            value=(
                                                "m.idx.toString()",
                                            ),
                                        )
                                    v3.VDivider()
                            # backend / fiber-model / diameter
                            # moved down next to the Run button —
                            # see the `.golgi-fiber-sim-row`
                            # block further below. Keeps the
                            # trajectory selector at the top of
                            # the panel uncluttered (just the
                            # one full-width combobox).
                        # ---- Pulse design (2-col: params + preview) ----
                        # Outer wrapper is a 2-col grid: existing
                        # pulse-design block (params) on the left,
                        # designed-pulse Plotly preview on the
                        # right. The preview used to live in the
                        # output grid below alongside the sim
                        # outputs; that conflated "input you're
                        # editing" with "sim result", so it's
                        # moved here next to the params that drive
                        # it.
                        with html.Div(
                            classes="golgi-fiber-pulse-2col",
                        ):
                            with html.Div(
                                classes="golgi-fiber-pulse-design",
                            ):
                                html.H3("Pulse Design")
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                ):
                                    v3.VSelect(
                                        v_model=("fiber_pulse_type",),
                                        items=("['monophasic', 'biphasic']",),
                                        label="Pulse Type",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-sel",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_onset_ms",),
                                        label="Onset (ms)",
                                        type="number",
                                        step=0.1, min=0.1, max=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    # tstop moved down next to the
                                    # Run button — it's more a
                                    # simulator-control than a
                                    # pulse parameter.
                                # Monophasic widgets
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === "
                                        "'monophasic'",
                                    ),
                                ):
                                    v3.VSelect(
                                        v_model=("fiber_mono_polarity",),
                                        items=("['cathodic', 'anodic']",),
                                        label="Polarity",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-sel",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_mono_amp_mA",),
                                        label="Amplitude (mA)",
                                        type="number",
                                        step=0.05, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_mono_pw_us",),
                                        label="Pulse Width (µs)",
                                        type="number",
                                        step=10.0, min=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                # Biphasic widgets
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === 'biphasic'",
                                    ),
                                ):
                                    v3.VSelect(
                                        v_model=("fiber_bi_order",),
                                        items=(
                                            "['cathodic-first', "
                                            " 'anodic-first']",
                                        ),
                                        label="Phase Order",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-sel",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VCheckbox(
                                        v_model=(
                                            "fiber_bi_charge_balanced",
                                        ),
                                        label="Charge-Balanced",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-chk",
                                        disabled=("fiber_sim_busy",),
                                    )
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === 'biphasic'",
                                    ),
                                ):
                                    v3.VTextField(
                                        v_model=(
                                            "fiber_bi_phase1_amp_mA",
                                        ),
                                        label="Phase 1 Amplitude (mA)",
                                        type="number",
                                        step=0.05, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_bi_phase1_pw_us",),
                                        label="Phase 1 Pulse Width (µs)",
                                        type="number",
                                        step=10.0, min=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_bi_gap_us",),
                                        label="Inter-Phase Gap (µs)",
                                        type="number",
                                        step=10.0, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === 'biphasic'",
                                    ),
                                ):
                                    v3.VTextField(
                                        v_model=(
                                            "fiber_bi_phase2_amp_mA",
                                        ),
                                        label="Phase 2 Amplitude (mA)",
                                        type="number",
                                        step=0.05, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("fiber_sim_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_bi_phase2_pw_us",),
                                        label="Phase 2 Pulse Width (µs)",
                                        type="number",
                                        step=10.0, min=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=(
                                            "fiber_sim_busy "
                                            "|| fiber_bi_charge_balanced",
                                        ),
                                    )
                            # Right column — designed-pulse preview.
                            with html.Div(
                                classes="golgi-fiber-pulse-preview",
                            ):
                                _export_btn("fiber.pulse")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "fiber_pulse_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=False,
                                    )
                                else:
                                    html.Div(
                                        "Install `trame-plotly` "
                                        "to see the pulse "
                                        "preview.",
                                        classes="placeholder",
                                    )
                        # ---- Simulator settings (row 1) ----
                        # Backend + Fiber Model + Diameter
                        # (slider + wide numeric spinner).
                        with html.Div(
                            classes="golgi-fiber-sim-row",
                        ):
                            v3.VSelect(
                                v_model=("fiber_backend",),
                                items=("['pyfibers', 'axonml']",),
                                label="Backend",
                                density="compact",
                                hide_details=True,
                                classes="golgi-fiber-sel",
                                disabled=("fiber_sim_busy",),
                            )
                            v3.VSelect(
                                v_model=("fiber_model",),
                                items=(
                                    "fiber_backend === 'axonml' "
                                    "? ['MRG_INTERPOLATION'] "
                                    f": {MYELINATED_MODELS + UNMYELINATED_MODELS!r}",
                                ),
                                label="Fiber Model",
                                density="compact",
                                hide_details=True,
                                classes="golgi-fiber-sel",
                                disabled=("fiber_sim_busy",),
                            )
                            # Diameter slider + numeric input
                            # pair. Slider range tracks the
                            # active fibre model; ticks render
                            # at the permitted discrete points
                            # when fiber_diameter_ticks is
                            # non-empty (MRG_DISCRETE).
                            with html.Div(
                                classes=(
                                    "golgi-fiber-slider-pair"
                                ),
                            ):
                                v3.VSlider(
                                    v_model=("fiber_diameter_um",),
                                    min=("fiber_diameter_min",),
                                    max=("fiber_diameter_max",),
                                    step=("fiber_diameter_step",),
                                    # `ticks` accepts an array
                                    # of explicit positions —
                                    # for MRG_DISCRETE this is
                                    # the list of tabulated
                                    # diameters so the slider
                                    # shows them as visible
                                    # tick marks. Falls back to
                                    # `false` (no ticks) for
                                    # the continuous models.
                                    ticks=(
                                        "fiber_diameter_ticks"
                                        ".length > 0 "
                                        "? fiber_diameter_ticks "
                                        ": false",
                                    ),
                                    show_ticks=(
                                        "fiber_diameter_ticks"
                                        ".length > 0 "
                                        "? 'always' : false",
                                    ),
                                    label="Diameter (µm)",
                                    density="compact",
                                    hide_details=True,
                                    thumb_label=True,
                                    classes=(
                                        "golgi-fiber-slider"
                                    ),
                                    disabled=("fiber_sim_busy",),
                                )
                                v3.VTextField(
                                    v_model=("fiber_diameter_um",),
                                    type="number",
                                    step=("fiber_diameter_step",),
                                    min=("fiber_diameter_min",),
                                    max=("fiber_diameter_max",),
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    classes=(
                                        "golgi-fiber-slider-num"
                                    ),
                                    disabled=("fiber_sim_busy",),
                                )
                        # ---- Simulator settings (row 2) ----
                        # Wider Simulation Duration slider + the
                        # Run CTA on the right. Block-style CTA
                        # so it matches the FEM "▶ Run FEM solve"
                        # button on the Solve tab.
                        with html.Div(
                            classes=(
                                "golgi-fiber-sim-row "
                                "golgi-fiber-sim-row-2"
                            ),
                        ):
                            with html.Div(
                                classes=(
                                    "golgi-fiber-slider-pair "
                                    "golgi-fiber-slider-pair-wide"
                                ),
                            ):
                                v3.VSlider(
                                    v_model=("fiber_tstop_ms",),
                                    min=FIBER_TSTOP_MIN_MS,
                                    max=FIBER_TSTOP_MAX_MS,
                                    step=FIBER_TSTOP_STEP_MS,
                                    label="Simulation Duration (ms)",
                                    density="compact",
                                    hide_details=True,
                                    thumb_label=True,
                                    classes=(
                                        "golgi-fiber-slider"
                                    ),
                                    disabled=("fiber_sim_busy",),
                                )
                                v3.VTextField(
                                    v_model=("fiber_tstop_ms",),
                                    type="number",
                                    step=FIBER_TSTOP_STEP_MS,
                                    min=FIBER_TSTOP_MIN_MS,
                                    max=FIBER_TSTOP_MAX_MS,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    classes=(
                                        "golgi-fiber-slider-num"
                                    ),
                                    disabled=("fiber_sim_busy",),
                                )
                            # Run CTA — same block-CTA style as
                            # the Solve tab's "▶ Run FEM solve"
                            # so all three Run buttons (Solve,
                            # Single fiber, Population) feel
                            # consistent.
                            with html.Button(
                                type="button",
                                classes=(
                                    "golgi-cta-wrapper "
                                    "golgi-cta-wrapper-block "
                                    "golgi-fiber-run-cta"
                                ),
                                disabled=("fiber_sim_busy",),
                                click=do_run_fiber_sim,
                            ):
                                html.Span(
                                    classes="golgi-cta-spinner",
                                )
                                with html.Span(
                                    classes="golgi-cta-inner",
                                ):
                                    html.Span(
                                        "▶ Run fiber simulation"
                                    )
                        html.Div(
                            "{{ fiber_sim_status }}",
                            classes="golgi-fiber-status",
                        )
                        # ---- Result picker ----
                        # Only visible after a batch sim that
                        # produced 2+ results. Each entry has
                        # `ok` so failed fibers show a ⚠ in the
                        # label — they're picked too (their plots
                        # just stay blank). Selecting a result
                        # writes `fiber_sel_idx`, which the
                        # `_on_fiber_sel_change` watcher uses to
                        # swap the figure state vars.
                        with html.Div(
                            classes="golgi-fiber-result-picker",
                            v_show=(
                                "fiber_sim_results_meta.length > 1",
                            ),
                        ):
                            html.Div(
                                "Viewing",
                                classes=(
                                    "golgi-fiber-result-label"
                                ),
                            )
                            # Items include `color` so the
                            # custom item + selection slots can
                            # paint a tab10-coloured dot that
                            # matches the chip in the trajectory
                            # combobox above + the actor tint in
                            # the 3-D viewport.
                            with v3.VSelect(
                                v_model=("fiber_sel_idx",),
                                items=(
                                    "fiber_sim_results_meta.map("
                                    "  m => ({"
                                    "    title: (m.ok ? '✓ ' : '⚠ ')"
                                    "      + m.label,"
                                    "    value: m.idx,"
                                    "    color: m.color"
                                    "  })"
                                    ")",
                                ),
                                item_title="title",
                                item_value="value",
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                classes=(
                                    "golgi-fiber-result-select"
                                ),
                            ):
                                # Dropdown item: dot + label.
                                # Look up the colour from
                                # fiber_sim_results_meta by idx
                                # rather than via `item.raw.color`
                                # — `item.raw` isn't reliably
                                # populated in some trame/Vuetify
                                # combinations when items is a
                                # computed expression rather than
                                # a plain state ref.
                                with v3.Template(
                                    v_slot_item=(
                                        "{ props, item }",
                                    ),
                                ):
                                    with v3.VListItem(
                                        v_bind="props",
                                    ):
                                        with v3.Template(
                                            v_slot_prepend=True,
                                        ):
                                            html.Span(
                                                classes=(
                                                    "golgi-fiber-chip-dot"
                                                ),
                                                style=(
                                                    "'background:' + ("
                                                    "  (fiber_sim_results_meta."
                                                    "    find(m => "
                                                    "      m.idx === item.value"
                                                    "    ) || {})"
                                                    "  .color || '#888'"
                                                    ")",
                                                ),
                                            )
                                # Currently-selected display: dot
                                # + title in the closed select.
                                with v3.Template(
                                    v_slot_selection=(
                                        "{ item }",
                                    ),
                                ):
                                    html.Span(
                                        classes=(
                                            "golgi-fiber-chip-dot"
                                        ),
                                        style=(
                                            "'background:' + ("
                                            "  (fiber_sim_results_meta."
                                            "    find(m => "
                                            "      m.idx === item.value"
                                            "    ) || {})"
                                            "  .color || '#888'"
                                            ")",
                                        ),
                                    )
                                    html.Span("{{ item.title }}")
                        # ---- Output grid ----
                        # Two stacked tiles: V_m propagation
                        # heatmap on top, waterfall below.
                        # Hidden until at least one sim has
                        # produced a result — empty figure tiles
                        # are visual noise. The result picker
                        # above already gates on > 1 results;
                        # this v_show gates the tile rows on
                        # >= 1 result (anything we can plot).
                        with html.Div(
                            classes="golgi-fiber-output-grid",
                            v_show=(
                                "fiber_sim_results_meta.length "
                                "> 0",
                            ),
                        ):
                            # Propagation heatmap
                            with html.Div(
                                classes=(
                                    "golgi-fiber-tile "
                                    "golgi-fiber-tile-heat"
                                ),
                            ):
                                _export_btn("fiber.propagation")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "fiber_propagation_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=True,
                                    )
                            # Waterfall (bottom, full-width)
                            with html.Div(
                                classes=(
                                    "golgi-fiber-tile "
                                    "golgi-fiber-tile-water"
                                ),
                            ):
                                _export_btn("fiber.waterfall")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "fiber_waterfall_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=True,
                                    )
                            # R1.4 — single-fiber cNAP tile.
                            # Header row: title + montage selector +
                            # status chip. Figure below.
                            with html.Div(
                                classes=(
                                    "golgi-fiber-tile "
                                    "golgi-fiber-tile-cnap"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "d-flex align-center "
                                        "justify-space-between "
                                        "mb-2"
                                    ),
                                    style="gap: 12px;",
                                ):
                                    html.H4(
                                        "Single-fiber "
                                        "contribution to cNAP",
                                        style=(
                                            "margin: 0; "
                                            "font-size: 14px;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center"
                                        ),
                                        style="gap: 8px;",
                                    ):
                                        v3.VSelect(
                                            v_model=(
                                                "active_montage_single",
                                            ),
                                            items=(
                                                "(recording_montages "
                                                "  || []).map("
                                                "    m => ({"
                                                "      title: m.label, "
                                                "      value: m.mid"
                                                "    }))",
                                            ),
                                            item_title="title",
                                            item_value="value",
                                            label="Montage",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "min-width: "
                                                "160px;"
                                            ),
                                            v_show=(
                                                "(recording_montages "
                                                "  || []).length > 0",
                                            ),
                                        )
                                # Status / placeholder hint.
                                html.Div(
                                    "{{ fiber_cnap_status }}",
                                    v_show=(
                                        "fiber_cnap_status",
                                    ),
                                    style=(
                                        "font-size: 11px; "
                                        "color: #666; "
                                        "margin-bottom: 4px;"
                                    ),
                                )
                                _export_btn("fiber.cnap")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "fiber_cnap_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=True,
                                    )
                        # (Sim log removed — per-fiber summaries
                        # surface in the busy lightbox during the
                        # run and in fiber_sim_status afterwards;
                        # the full log was redundant in the panel
                        # and ate vertical space below the plots.)
                    # ---- Population panel (§13-15) ----
                    # Dynamic per-branch fiber-type mixture. Each
                    # branch has its own list of (model, mean, σ,
                    # fraction) rows; "Generate" samples a type +
                    # diameter for every fiber, then re-colours
                    # the 3-D viewport by type and renders the
                    # per-branch + overall KDE figure.
                    with html.Div(
                        v_show=("active_analysis === 'population'",),
                        classes="golgi-analysis-panel "
                                "golgi-pop-panel",
                    ):
                        # Title row matches the Single-fiber tab —
                        # H2 + info icon whose tooltip carries the
                        # long-form explanation. Same
                        # `golgi-fiber-title-row` class for layout
                        # consistency.
                        with html.Div(
                            classes="golgi-fiber-title-row",
                        ):
                            html.H2(
                                "Population — fiber-type mixture"
                            )
                            with v3.VTooltip(
                                location="top",
                                max_width=420,
                            ):
                                with v3.Template(
                                    v_slot_activator=(
                                        "{ props }",
                                    ),
                                ):
                                    with v3.VBtn(
                                        v_bind="props",
                                        icon=True,
                                        size="small",
                                        variant="text",
                                        density="compact",
                                        classes=(
                                            "golgi-fiber-info-btn"
                                        ),
                                    ):
                                        v3.VIcon(
                                            "mdi-information-outline",
                                            size="20",
                                            color="grey-darken-1",
                                        )
                                html.Span(
                                    "For each detected fiber "
                                    "branch, add fiber types "
                                    "with their relative amount "
                                    "and diameter distribution "
                                    "(mean ± σ). Click Generate "
                                    "to assign each trajectory a "
                                    "(type, diameter) — the "
                                    "viewport above re-colours "
                                    "by type and the KDE plot "
                                    "shows the resulting "
                                    "diameter distribution per "
                                    "branch + overall."
                                )
                        # Empty state — no fibers yet.
                        html.Div(
                            "No fiber trajectories loaded — open a "
                            "project with fiber data, or generate "
                            "fibers in the Fiber trajectories tab "
                            "first.",
                            classes="placeholder",
                            v_show=(
                                "pop_branches_meta.length === 0",
                            ),
                        )
                        # ---- F1.1: curated preset picker ----
                        # Dropdown + citation/notes + preview KDE +
                        # Apply button. Selecting a preset only
                        # rebuilds the preview (server-side watcher);
                        # the row-overwrite happens on Apply.
                        with html.Div(
                            v_show=(
                                "pop_branches_meta.length > 0",
                            ),
                            classes=(
                                "golgi-pop-preset-block "
                                "mb-4 pa-3"
                            ),
                            style=(
                                "border: 1px solid #e0e0e6; "
                                "border-radius: 6px; "
                                "background: #fafafb;"
                            ),
                        ):
                            html.Div(
                                "FIBER POPULATION PRESET",
                                style=(
                                    "font-size: 10px; "
                                    "letter-spacing: 0.04em; "
                                    "color: #888a90; "
                                    "text-transform: uppercase; "
                                    "margin-bottom: 6px;"
                                ),
                            )
                            with html.Div(
                                classes="d-flex align-center",
                                style="gap: 8px;",
                            ):
                                v3.VSelect(
                                    v_model=("pop_preset_choice",),
                                    items=("pop_preset_items",),
                                    item_title="title",
                                    item_value="value",
                                    label="Curated preset",
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    style="flex: 1 1 auto;",
                                )
                                html.Button(
                                    "Apply preset",
                                    type="button",
                                    classes=(
                                        "golgi-btn-primary "
                                        "golgi-btn-sm"
                                    ),
                                    disabled=(
                                        "!pop_preset_choice",
                                    ),
                                    click=do_pop_apply_preset,
                                )
                            # Citation + notes — only when a preset
                            # is picked. Uses the v_show on the
                            # parent div instead of nested conds
                            # so the empty-state lays out cleanly.
                            with html.Div(
                                v_show=("pop_preset_choice",),
                                style="margin-top: 8px;",
                            ):
                                html.Div(
                                    "{{ pop_preset_meta.species }} "
                                    "· {{ pop_preset_meta.nerve }}",
                                    style=(
                                        "font-size: 11px; "
                                        "color: #4a4a52; "
                                        "font-weight: 600;"
                                    ),
                                )
                                html.Div(
                                    "Citation: "
                                    "{{ pop_preset_meta.citation }}",
                                    style=(
                                        "font-size: 11px; "
                                        "color: #4a4a52; "
                                        "margin-top: 2px;"
                                    ),
                                )
                                html.Div(
                                    "{{ pop_preset_meta.notes }}",
                                    style=(
                                        "font-size: 11px; "
                                        "color: #888a90; "
                                        "margin-top: 4px; "
                                        "font-style: italic; "
                                        "line-height: 1.4;"
                                    ),
                                )
                                # Preview KDE tile — rebuilt server-
                                # side on every dropdown change. No
                                # interaction with pop_branch_types.
                                with html.Div(
                                    style=(
                                        "width: 100%; "
                                        "height: 220px; "
                                        "margin-top: 8px; "
                                        "position: relative;"
                                    ),
                                ):
                                    _export_btn("pop.preset_preview")
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "pop_preset_"
                                                "preview_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=False,
                                        )
                                    else:
                                        html.Div(
                                            "Install `trame-plotly` "
                                            "to see the preview.",
                                            style=(
                                                "padding: 20px; "
                                                "color:#888; "
                                                "font-size:12px;"
                                            ),
                                        )
                        # ---- Per-branch sections (outer v-for) ----
                        with html.Div(
                            v_show=(
                                "pop_branches_meta.length > 0",
                            ),
                            classes="golgi-pop-branches",
                        ):
                            with html.Div(
                                v_for=(
                                    "(meta, bidx) in pop_branches_meta"
                                ),
                                key="meta.idx",
                                classes="golgi-pop-branch",
                            ):
                                # Branch header — swatch + label +
                                # live sum-of-fractions indicator
                                # + Add button. Sum is computed
                                # client-side from
                                # pop_branch_types[branch_idx],
                                # so it updates live as the user
                                # edits the per-row fraction
                                # inputs (no server round-trip).
                                with html.Div(
                                    classes="golgi-pop-branch-head",
                                ):
                                    # (Branch swatch removed.
                                    # Per-row colour swatches now
                                    # do all the colour work, and
                                    # the branch header reads as
                                    # plain text "Branch X · N
                                    # fibers".)
                                    html.Span(
                                        "{{ meta.label }} · "
                                        "{{ meta.n_fibers }} fibers",
                                        classes=(
                                            "golgi-pop-branch-title"
                                        ),
                                    )
                                    # Sum-of-fractions indicator:
                                    # green when exactly 100 %,
                                    # red otherwise. Empty branches
                                    # (no types yet) hide the chip.
                                    html.Span(
                                        "sum: {{ ("
                                        "  pop_branch_types["
                                        "    meta.idx.toString()"
                                        "  ] || []"
                                        ").reduce("
                                        "  (s, r) => s + ("
                                        "    parseFloat(r.frac) "
                                        "    || 0), 0"
                                        ").toFixed(0) }} %",
                                        v_show=(
                                            "(pop_branch_types["
                                            "  meta.idx.toString()"
                                            "] || []).length > 0",
                                        ),
                                        classes=(
                                            "['golgi-pop-sum', "
                                            "Math.abs(("
                                            "  pop_branch_types["
                                            "    meta.idx.toString()"
                                            "  ] || []"
                                            ").reduce("
                                            "  (s, r) => s + ("
                                            "    parseFloat(r.frac)"
                                            "    || 0), 0"
                                            ") - 100) < 0.5 "
                                            "? 'is-ok' : 'is-warn']"
                                        ),
                                    )
                                    # + Add fiber type — inline JS
                                    # appends a new row with
                                    # default frac=100 if it's the
                                    # first row in this branch,
                                    # else frac=0 so the new row
                                    # doesn't blow the sum past
                                    # 100 % automatically. Each
                                    # row also gets a `name`
                                    # ("Type N" by default) and a
                                    # tab10 `color` indexed by
                                    # the total row count so
                                    # consecutively-added rows
                                    # cycle through the palette
                                    # in a stable order. The
                                    # colour follows the row's
                                    # fibers in the 3-D viewport
                                    # + the KDE plot + the chip.
                                    html.Button(
                                        "+ Add fiber type",
                                        type="button",
                                        classes=(
                                            "golgi-btn-secondary "
                                            "golgi-btn-sm"
                                        ),
                                        click=(
                                            "pop_branch_types = {"
                                            "  ...pop_branch_types,"
                                            "  [meta.idx.toString()]: ["
                                            "    ...(pop_branch_types["
                                            "      meta.idx.toString()"
                                            "    ] || []),"
                                            "    {"
                                            "      id: Math.random()"
                                            "        .toString(36)"
                                            "        .slice(2, 10),"
                                            "      name: 'Type ' + ("
                                            "        Object.values("
                                            "          pop_branch_types"
                                            "        ).reduce("
                                            "          (s, rs) => "
                                            "          s + rs.length, 0"
                                            "        ) + 1"
                                            "      ),"
                                            "      backend: 'pyfibers',"
                                            "      model: 'MRG_INTERPOLATION',"
                                            "      mean_um: 10,"
                                            "      std_um: 1.5,"
                                            "      frac: ("
                                            "        pop_branch_types["
                                            "          meta.idx.toString()"
                                            "        ] || []"
                                            "      ).length === 0 "
                                            "        ? 100 : 0,"
                                            "      color: "
                                            "        fiber_tab10_palette["
                                            "          Object.values("
                                            "            pop_branch_types"
                                            "          ).reduce("
                                            "            (s, rs) => "
                                            "            s + rs.length, 0"
                                            "          ) "
                                            "          % fiber_tab10_palette.length"
                                            "        ]"
                                            "    }"
                                            "  ]"
                                            "}; "
                                            "pop_generated = false"
                                        ),
                                    )
                                # Inner v-for over type rows for
                                # this branch.
                                with html.Div(
                                    classes="golgi-pop-types",
                                ):
                                    # "No types yet" hint.
                                    html.Div(
                                        "No fiber types assigned "
                                        "yet — click + Add fiber "
                                        "type.",
                                        classes=(
                                            "golgi-pop-no-types"
                                        ),
                                        v_show=(
                                            "!(pop_branch_types["
                                            "  meta.idx.toString()"
                                            "] && "
                                            "pop_branch_types["
                                            "  meta.idx.toString()"
                                            "].length)",
                                        ),
                                    )
                                    with html.Div(
                                        v_for=(
                                            "(row, ridx) in "
                                            "(pop_branch_types["
                                            "  meta.idx.toString()"
                                            "] || [])"
                                        ),
                                        key="row.id",
                                        classes=(
                                            "golgi-pop-type-row"
                                        ),
                                    ):
                                        # All inputs use INLINE JS
                                        # for `update:model-value`
                                        # so the mutation is purely
                                        # client-side. No server
                                        # round-trip during typing
                                        # = no re-render flicker
                                        # that would otherwise drop
                                        # the user's keystrokes
                                        # (the bug the previous
                                        # tuple-bound
                                        # `do_pop_update_type`
                                        # version had on every
                                        # char). Trame syncs the
                                        # full state back to the
                                        # server on the next event
                                        # (Generate click).
                                        # Per-row colour dot —
                                        # tab10 colour assigned at
                                        # row-add time. Matches the
                                        # row's 3-D actor + KDE
                                        # trace + per-row subplot
                                        # title so every visual
                                        # cue agrees.
                                        html.Span(
                                            classes=(
                                                "golgi-pop-row-dot"
                                            ),
                                            style=(
                                                "'background:' "
                                                "+ (row.color "
                                                "  || '#888')",
                                            ),
                                        )
                                        # Name input — free-text
                                        # label for the row. Drives
                                        # the per-row KDE subplot
                                        # title and the chip-style
                                        # marker further down.
                                        v3.VTextField(
                                            model_value=("row.name",),
                                            label="Name",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            classes=(
                                                "golgi-pop-row-name"
                                            ),
                                            update_modelValue=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types,"
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).map(r => "
                                                "    r.id === row.id "
                                                "      ? {...r, "
                                                "          name: $event} "
                                                "      : r"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                                        # Backend selector — per
                                        # row so the user can mix
                                        # pyfibers (NEURON) and
                                        # axonml (surrogate) inside
                                        # a single population. The
                                        # subsequent model dropdown
                                        # filters to MRG_INTERPOLATION
                                        # only when the row's
                                        # backend is axonml (matches
                                        # the gating used in the
                                        # Single-fiber tab).
                                        v3.VSelect(
                                            model_value=(
                                                "row.backend",
                                            ),
                                            items=(
                                                "['pyfibers', "
                                                " 'axonml']",
                                            ),
                                            label="Backend",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            classes=(
                                                "golgi-pop-row-backend"
                                            ),
                                            update_modelValue=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types,"
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).map(r => "
                                                "    r.id === row.id "
                                                "      ? {...r,"
                                                "          backend: $event,"
                                                "          model: $event "
                                                "            === 'axonml'"
                                                "            ? 'MRG_INTERPOLATION'"
                                                "            : r.model}"
                                                "      : r"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                                        v3.VSelect(
                                            model_value=("row.model",),
                                            items=(
                                                "row.backend === 'axonml'"
                                                "  ? ['MRG_INTERPOLATION']"
                                                f"  : {MYELINATED_MODELS + UNMYELINATED_MODELS!r}",
                                            ),
                                            label="Fiber Model",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            classes=(
                                                "golgi-pop-row-model"
                                            ),
                                            update_modelValue=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types, "
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).map(r => "
                                                "    r.id === row.id "
                                                "      ? {...r, model: $event} "
                                                "      : r"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                                        v3.VTextField(
                                            model_value=(
                                                "row.mean_um",
                                            ),
                                            label="Mean Diameter (µm)",
                                            type="number",
                                            step=0.1,
                                            min=0.1, max=20.0,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            classes=(
                                                "golgi-pop-row-num"
                                            ),
                                            update_modelValue=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types, "
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).map(r => "
                                                "    r.id === row.id "
                                                "      ? {...r, mean_um: "
                                                "          parseFloat($event) "
                                                "          || 0} "
                                                "      : r"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                                        v3.VTextField(
                                            model_value=(
                                                "row.std_um",
                                            ),
                                            label="Standard Deviation (µm)",
                                            type="number",
                                            step=0.05,
                                            min=0.0, max=8.0,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            classes=(
                                                "golgi-pop-row-num"
                                            ),
                                            update_modelValue=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types, "
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).map(r => "
                                                "    r.id === row.id "
                                                "      ? {...r, std_um: "
                                                "          parseFloat($event) "
                                                "          || 0} "
                                                "      : r"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                                        v3.VTextField(
                                            model_value=("row.frac",),
                                            label="Fraction (%)",
                                            type="number",
                                            step=1,
                                            min=0, max=100,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            classes=(
                                                "golgi-pop-row-num"
                                            ),
                                            update_modelValue=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types, "
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).map(r => "
                                                "    r.id === row.id "
                                                "      ? {...r, frac: "
                                                "          parseFloat($event) "
                                                "          || 0} "
                                                "      : r"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                                        # × delete button — inline
                                        # JS filters out this row's
                                        # id from the branch's list.
                                        html.Button(
                                            "×",
                                            type="button",
                                            classes=(
                                                "golgi-pop-row-del"
                                            ),
                                            click=(
                                                "pop_branch_types = {"
                                                "  ...pop_branch_types, "
                                                "  [meta.idx.toString()]: ("
                                                "    pop_branch_types["
                                                "      meta.idx.toString()"
                                                "    ] || []"
                                                "  ).filter(r => "
                                                "    r.id !== row.id"
                                                "  )"
                                                "}; "
                                                "pop_generated = false"
                                            ),
                                        )
                        # ---- Generate row (seed + button + status) ----
                        with html.Div(
                            v_show=(
                                "pop_branches_meta.length > 0",
                            ),
                            classes="golgi-pop-controls",
                        ):
                            v3.VTextField(
                                v_model=("pop_seed",),
                                label="Random Seed",
                                type="number",
                                step=1, min=0,
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                classes="golgi-pop-seed",
                            )
                            html.Button(
                                "⚡ Generate population",
                                type="button",
                                classes=(
                                    "golgi-btn-primary "
                                    "golgi-btn-sm"
                                ),
                                click=do_pop_generate,
                            )
                            html.Div(
                                "{{ pop_status }}",
                                classes="golgi-pop-status",
                            )
                        # ---- Cross-section at cuff centre ----
                        # New tile (above the KDE) showing every
                        # generated fiber as a dot at its
                        # trajectory's nearest-to-z=0 (x, y) in
                        # cuff frame, coloured by named
                        # subpopulation. The nerve outline at
                        # z=0 plus a per-branch convex hull
                        # (thin dotted black) give the cluster
                        # context. Gated on pop_generated.
                        with html.Div(
                            v_show=("pop_generated",),
                            classes="golgi-pop-xsec-cuff",
                        ):
                            html.H3(
                                "Cross-section at cuff centre"
                            )
                            _export_btn("pop.xsec_cuff")
                            if twp is not None:
                                twp.Figure(
                                    state_variable_name=(
                                        "pop_xsec_cuff_figure"
                                    ),
                                    display_logo=False,
                                    display_mode_bar=True,
                                )
                            else:
                                html.Div(
                                    "Install `trame-plotly` to "
                                    "see the cross-section.",
                                    classes="placeholder",
                                )
                        # ---- KDE plot ----
                        with html.Div(
                            v_show=("pop_generated",),
                            classes="golgi-pop-kde",
                        ):
                            html.H3(
                                "Diameter distribution by named "
                                "subpopulation"
                            )
                            _export_btn("pop.kde")
                            if twp is not None:
                                twp.Figure(
                                    state_variable_name=(
                                        "pop_kde_figure"
                                    ),
                                    display_logo=False,
                                    display_mode_bar=True,
                                )
                            else:
                                html.Div(
                                    "Install `trame-plotly` to "
                                    "see the KDE figure.",
                                    classes="placeholder",
                                )
                        # ---- Pulse designer (2-col: params +
                        # preview) — same widget as the Single
                        # fiber tab, reusing the SAME state vars
                        # (fiber_pulse_*) so the user designs ONE
                        # pulse and it applies across analyses.
                        # Gated on pop_generated so the user
                        # focuses on the design step first.
                        with html.Div(
                            v_show=("pop_generated",),
                            classes="golgi-fiber-pulse-2col "
                                    "golgi-pop-pulse-section",
                        ):
                            with html.Div(
                                classes="golgi-fiber-pulse-design",
                            ):
                                html.H3("Pulse Design")
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                ):
                                    v3.VSelect(
                                        v_model=("fiber_pulse_type",),
                                        items=(
                                            "['monophasic', 'biphasic']",
                                        ),
                                        label="Pulse Type",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-sel",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_onset_ms",),
                                        label="Onset (ms)",
                                        type="number",
                                        step=0.1, min=0.1,
                                        max=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === "
                                        "'monophasic'",
                                    ),
                                ):
                                    v3.VSelect(
                                        v_model=(
                                            "fiber_mono_polarity",
                                        ),
                                        items=(
                                            "['cathodic', 'anodic']",
                                        ),
                                        label="Polarity",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-sel",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_mono_amp_mA",),
                                        label="Amplitude (mA)",
                                        type="number",
                                        step=0.05, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_mono_pw_us",),
                                        label="Pulse Width (µs)",
                                        type="number",
                                        step=10.0, min=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === 'biphasic'",
                                    ),
                                ):
                                    v3.VSelect(
                                        v_model=("fiber_bi_order",),
                                        items=(
                                            "['cathodic-first', "
                                            " 'anodic-first']",
                                        ),
                                        label="Phase Order",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-sel",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VCheckbox(
                                        v_model=(
                                            "fiber_bi_charge_balanced",
                                        ),
                                        label="Charge-Balanced",
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-chk",
                                        disabled=("pop_busy",),
                                    )
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === 'biphasic'",
                                    ),
                                ):
                                    v3.VTextField(
                                        v_model=(
                                            "fiber_bi_phase1_amp_mA",
                                        ),
                                        label="Phase 1 Amplitude (mA)",
                                        type="number",
                                        step=0.05, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=(
                                            "fiber_bi_phase1_pw_us",
                                        ),
                                        label="Phase 1 Pulse Width (µs)",
                                        type="number",
                                        step=10.0, min=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=("fiber_bi_gap_us",),
                                        label="Inter-Phase Gap (µs)",
                                        type="number",
                                        step=10.0, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                with html.Div(
                                    classes="golgi-fiber-pulse-row",
                                    v_show=(
                                        "fiber_pulse_type === 'biphasic'",
                                    ),
                                ):
                                    v3.VTextField(
                                        v_model=(
                                            "fiber_bi_phase2_amp_mA",
                                        ),
                                        label="Phase 2 Amplitude (mA)",
                                        type="number",
                                        step=0.05, min=0.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=("pop_busy",),
                                    )
                                    v3.VTextField(
                                        v_model=(
                                            "fiber_bi_phase2_pw_us",
                                        ),
                                        label="Phase 2 Pulse Width (µs)",
                                        type="number",
                                        step=10.0, min=10.0,
                                        density="compact",
                                        hide_details=True,
                                        classes="golgi-fiber-num",
                                        disabled=(
                                            "pop_busy "
                                            "|| fiber_bi_charge_balanced",
                                        ),
                                    )
                            with html.Div(
                                classes="golgi-fiber-pulse-preview",
                            ):
                                _export_btn("fiber.pulse")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "fiber_pulse_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=False,
                                    )
                                else:
                                    html.Div(
                                        "Install `trame-plotly`"
                                        " to see the pulse "
                                        "preview.",
                                        classes="placeholder",
                                    )
                        # ---- Sim-row (tstop) + Run CTA ----
                        # Same pattern as the Single-fiber sim
                        # row + FEM Run button: tstop sits next
                        # to the trigger because they're both
                        # simulator-controls, and the CTA gets
                        # the same animated conic-gradient
                        # wrapper as the Solve / Single-fiber
                        # buttons so all three Run actions feel
                        # consistent.
                        with html.Div(
                            v_show=("pop_generated",),
                            classes="golgi-pop-sim-row",
                        ):
                            v3.VTextField(
                                v_model=("fiber_tstop_ms",),
                                label="Simulation Duration (ms)",
                                type="number",
                                step=0.5, min=2.0, max=30.0,
                                density="compact",
                                hide_details=True,
                                classes="golgi-fiber-num",
                                disabled=("pop_busy",),
                            )
                            with html.Button(
                                type="button",
                                classes=(
                                    "golgi-cta-wrapper "
                                    "golgi-cta-wrapper-block "
                                    "golgi-pop-run-cta"
                                ),
                                disabled=(
                                    "!pop_generated || pop_busy",
                                ),
                                click=do_pop_run_sim,
                            ):
                                html.Span(
                                    classes="golgi-cta-spinner",
                                )
                                with html.Span(
                                    classes="golgi-cta-inner",
                                ):
                                    html.Span(
                                        "▶ Run population "
                                        "simulation"
                                    )
                        # ---- Population sim outputs ----
                        # Cross-section overview (full-width
                        # tile) + per-fiber result picker +
                        # heatmap + waterfall — appear once
                        # `pop_sim_done` flips true.
                        with html.Div(
                            v_show=("pop_sim_done",),
                            classes="golgi-pop-results",
                        ):
                            # Cross-section: each fiber as a
                            # point at its (x,y) centroid, the
                            # activated ones in their row's
                            # colour, the quiescent ones light
                            # grey.
                            with html.Div(
                                classes="golgi-pop-xsec-tile",
                            ):
                                html.H3(
                                    "Cross-section overview"
                                )
                                _export_btn("pop.xsec_activated")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "pop_xsec_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=True,
                                    )
                                else:
                                    html.Div(
                                        "Install `trame-plotly`"
                                        " to see the cross-"
                                        "section overview.",
                                        classes="placeholder",
                                    )
                            # Result picker — same UX as Single
                            # Fiber's. Each entry carries the
                            # row's tab10 colour so the dot in
                            # the picker matches the 3-D actor
                            # + cross-section dot.
                            with html.Div(
                                classes="golgi-fiber-result-picker",
                            ):
                                html.Div(
                                    "Viewing",
                                    classes=(
                                        "golgi-fiber-result-label"
                                    ),
                                )
                                with v3.VSelect(
                                    v_model=("pop_view_idx",),
                                    items=(
                                        "pop_sim_results_meta.map("
                                        "  m => ({"
                                        "    title: (m.ok "
                                        "      ? (m.activated "
                                        "          ? '⚡ ' "
                                        "          : '· ') "
                                        "      : '⚠ ')"
                                        "      + m.label,"
                                        "    value: m.idx,"
                                        "    color: m.color"
                                        "  })"
                                        ")",
                                    ),
                                    item_title="title",
                                    item_value="value",
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    classes=(
                                        "golgi-fiber-result-select"
                                    ),
                                ):
                                    with v3.Template(
                                        v_slot_item=(
                                            "{ props, item }",
                                        ),
                                    ):
                                        with v3.VListItem(
                                            v_bind="props",
                                        ):
                                            with v3.Template(
                                                v_slot_prepend=True,
                                            ):
                                                html.Span(
                                                    classes=(
                                                        "golgi-fiber-chip-dot"
                                                    ),
                                                    style=(
                                                        "'background:' + ("
                                                        "  (pop_sim_results_meta."
                                                        "    find(m => "
                                                        "      m.idx === item.value"
                                                        "    ) || {})"
                                                        "  .color || '#888'"
                                                        ")",
                                                    ),
                                                )
                                    with v3.Template(
                                        v_slot_selection=(
                                            "{ item }",
                                        ),
                                    ):
                                        html.Span(
                                            classes=(
                                                "golgi-fiber-chip-dot"
                                            ),
                                            style=(
                                                "'background:' + ("
                                                "  (pop_sim_results_meta."
                                                "    find(m => "
                                                "      m.idx === item.value"
                                                "    ) || {})"
                                                "  .color || '#888'"
                                                ")",
                                            ),
                                        )
                                        html.Span(
                                            "{{ item.title }}"
                                        )
                            # Heatmap + waterfall stacked,
                            # full-width, same component layout
                            # as the Single-fiber output grid.
                            with html.Div(
                                classes=(
                                    "golgi-fiber-output-grid"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-fiber-tile "
                                        "golgi-fiber-tile-heat"
                                    ),
                                ):
                                    _export_btn("pop.propagation")
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "pop_propagation_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=True,
                                        )
                                with html.Div(
                                    classes=(
                                        "golgi-fiber-tile "
                                        "golgi-fiber-tile-water"
                                    ),
                                ):
                                    _export_btn("pop.waterfall")
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "pop_waterfall_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=True,
                                        )
                            # R1.4 — population cNAP tile. Sum
                            # across all simulated fibers per
                            # montage, optionally decomposed by
                            # fiber type. Multi-montage panels
                            # would overlay; for R1.4 we render
                            # one montage at a time with a
                            # dropdown switcher.
                            with html.Div(
                                classes=(
                                    "golgi-fiber-tile "
                                    "golgi-fiber-tile-cnap"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "d-flex align-center "
                                        "justify-space-between "
                                        "mb-2"
                                    ),
                                    style="gap: 12px;",
                                ):
                                    html.H4(
                                        "Population cNAP",
                                        style=(
                                            "margin: 0; "
                                            "font-size: 14px;"
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "d-flex align-center"
                                        ),
                                        style="gap: 8px;",
                                    ):
                                        v3.VSelect(
                                            v_model=(
                                                "active_montage_pop",
                                            ),
                                            items=(
                                                "(recording_montages "
                                                "  || []).map("
                                                "    m => ({"
                                                "      title: m.label, "
                                                "      value: m.mid"
                                                "    }))",
                                            ),
                                            item_title="title",
                                            item_value="value",
                                            label="Montage",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "min-width: "
                                                "160px;"
                                            ),
                                            v_show=(
                                                "(recording_montages "
                                                "  || []).length > 0",
                                            ),
                                        )
                                        v3.VSwitch(
                                            v_model=(
                                                "cnap_decompose_by_type",
                                            ),
                                            label=(
                                                "Decompose by "
                                                "fiber type"
                                            ),
                                            density="compact",
                                            hide_details=True,
                                            color="primary",
                                            style=(
                                                "transform: "
                                                "scale(0.85);"
                                            ),
                                        )
                                html.Div(
                                    "{{ pop_cnap_status }}",
                                    v_show=(
                                        "pop_cnap_status",
                                    ),
                                    style=(
                                        "font-size: 11px; "
                                        "color: #666; "
                                        "margin-bottom: 4px;"
                                    ),
                                )
                                _export_btn("pop.cnap")
                                if twp is not None:
                                    twp.Figure(
                                        state_variable_name=(
                                            "pop_cnap_figure"
                                        ),
                                        display_logo=False,
                                        display_mode_bar=True,
                                    )

                    # ---- F2.1 Sweep panel (recruitment / threshold) ----
                    # Sibling to the Fiber / Population panels;
                    # gated on active_analysis === 'sweep'.
                    with html.Div(
                        v_show=("active_analysis === 'sweep'",),
                        classes=(
                            "golgi-analysis-panel "
                            "golgi-sweep-panel"
                        ),
                    ):
                        # Title row.
                        with html.Div(
                            classes="golgi-fiber-title-row",
                        ):
                            html.H2(
                                "Parameter sweep "
                                "(recruitment / threshold)"
                            )
                            with v3.VTooltip(
                                location="top",
                                max_width=420,
                            ):
                                with v3.Template(
                                    v_slot_activator=(
                                        "{ props }",
                                    ),
                                ):
                                    with v3.VBtn(
                                        v_bind="props",
                                        icon=True,
                                        size="small",
                                        variant="text",
                                        density="compact",
                                    ):
                                        v3.VIcon(
                                            "mdi-information-outline",
                                            size="20",
                                            color="grey-darken-1",
                                        )
                                html.Span(
                                    "Sweep the stim amplitude to "
                                    "see the recruitment curve "
                                    "(% fibers activated vs I_stim), "
                                    "or bisect per fiber to find "
                                    "each fiber's activation "
                                    "threshold. Uses the cached "
                                    "Vₑ on fibers from the last "
                                    "FEM solve — no re-solve."
                                )

                        # Empty / preflight state.
                        html.Div(
                            "Run a FEM solve with fibers loaded "
                            "to enable the sweep panel.",
                            classes="placeholder",
                            v_show=("!has_fem || !has_fibers",),
                        )

                        # Mode toggle + controls + run row.
                        with html.Div(
                            v_show=("has_fem && has_fibers",),
                        ):
                            # Mode toggle (recruitment | threshold).
                            html.Div(
                                "MODE",
                                style=(
                                    "font-size: 10px; "
                                    "letter-spacing: 0.04em; "
                                    "color: #888a90; "
                                    "text-transform: uppercase; "
                                    "margin-top: 12px; "
                                    "margin-bottom: 4px;"
                                ),
                            )
                            with v3.VBtnToggle(
                                v_model=("sweep_mode",),
                                mandatory=True,
                                density="compact",
                                color="primary",
                                variant="outlined",
                                divided=True,
                                classes="mb-3",
                            ):
                                v3.VBtn(
                                    "Recruitment curve",
                                    value="recruitment",
                                    size="small",
                                )
                                v3.VBtn(
                                    "Threshold finder",
                                    value="threshold",
                                    size="small",
                                )

                            # Model source — pick per-fiber from
                            # the Population (each row's model +
                            # backend) or one model from the
                            # Single-fiber tab applied to every
                            # fiber.
                            html.Div(
                                "FIBER MODEL SOURCE",
                                style=(
                                    "font-size: 10px; "
                                    "letter-spacing: 0.04em; "
                                    "color: #888a90; "
                                    "text-transform: uppercase; "
                                    "margin-bottom: 4px;"
                                ),
                            )
                            with v3.VBtnToggle(
                                v_model=("sweep_model_source",),
                                mandatory=True,
                                density="compact",
                                color="primary",
                                variant="outlined",
                                divided=True,
                                classes="mb-1",
                            ):
                                v3.VBtn(
                                    "Population (per-fiber)",
                                    value="population",
                                    size="small",
                                )
                                v3.VBtn(
                                    "Single-fiber tab",
                                    value="single_fiber",
                                    size="small",
                                )
                            # ---- Recruitment controls ----
                            with html.Div(
                                v_show=(
                                    "sweep_mode === 'recruitment'",
                                ),
                            ):
                                html.Div(
                                    "AMPLITUDE RANGE (mA)",
                                    style=(
                                        "font-size: 10px; "
                                        "letter-spacing: 0.04em; "
                                        "color: #888a90; "
                                        "text-transform: uppercase; "
                                        "margin-bottom: 4px;"
                                    ),
                                )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px;",
                                ):
                                    v3.VTextField(
                                        v_model=(
                                            "sweep_amp_min_mA",
                                        ),
                                        type="number", step=0.01,
                                        min=0.001, max=50.0,
                                        label="min",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )
                                    v3.VTextField(
                                        v_model=(
                                            "sweep_amp_max_mA",
                                        ),
                                        type="number", step=0.01,
                                        min=0.001, max=50.0,
                                        label="max",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )
                                    v3.VTextField(
                                        v_model=(
                                            "sweep_amp_n_points",
                                        ),
                                        type="number", step=1,
                                        min=2, max=200,
                                        label="n",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 90px;",
                                    )
                                    v3.VSelect(
                                        v_model=(
                                            "sweep_amp_spacing",
                                        ),
                                        items=(["lin", "log"],),
                                        label="spacing",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )

                            # ---- Threshold-finder controls ----
                            with html.Div(
                                v_show=(
                                    "sweep_mode === 'threshold'",
                                ),
                            ):
                                html.Div(
                                    "BISECT RANGE (mA) + TOL (µA)",
                                    style=(
                                        "font-size: 10px; "
                                        "letter-spacing: 0.04em; "
                                        "color: #888a90; "
                                        "text-transform: uppercase; "
                                        "margin-bottom: 4px;"
                                    ),
                                )
                                with html.Div(
                                    classes="d-flex align-center",
                                    style="gap: 8px;",
                                ):
                                    v3.VTextField(
                                        v_model=(
                                            "sweep_bisect_lo_mA",
                                        ),
                                        type="number", step=0.01,
                                        min=0.001, max=50.0,
                                        label="lo (mA)",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )
                                    v3.VTextField(
                                        v_model=(
                                            "sweep_bisect_hi_mA",
                                        ),
                                        type="number", step=0.01,
                                        min=0.001, max=50.0,
                                        label="hi (mA)",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 110px;",
                                    )
                                    v3.VTextField(
                                        v_model=(
                                            "sweep_bisect_tol_uA",
                                        ),
                                        type="number", step=1,
                                        min=1, max=1000,
                                        label="tol (µA)",
                                        density="compact",
                                        hide_details=True,
                                        variant="outlined",
                                        style="max-width: 130px;",
                                    )

                            # ---- Fiber selection ----
                            # Reuses the Single-fiber tab's
                            # `fiber_sel_indices` + `fiber_sel_items`
                            # so the user picks fibers from the same
                            # combobox UI they already know. Empty
                            # selection = sweep ALL fibers (the
                            # common default).
                            html.Div(
                                "FIBERS TO SWEEP",
                                style=(
                                    "font-size: 10px; "
                                    "letter-spacing: 0.04em; "
                                    "color: #888a90; "
                                    "text-transform: uppercase; "
                                    "margin-top: 14px; "
                                    "margin-bottom: 4px;"
                                ),
                            )
                            html.Div(
                                "Leave empty to sweep every fiber. "
                                "Otherwise pick specific "
                                "trajectories — same picker as "
                                "the Single-fiber tab.",
                                style=(
                                    "font-size: 10px; "
                                    "color: #888a90; "
                                    "margin-bottom: 6px; "
                                    "line-height: 1.4;"
                                ),
                            )
                            with v3.VAutocomplete(
                                v_model=("fiber_sel_indices",),
                                # Same filter as the Single-fiber
                                # tab: items shown in the dropdown
                                # are limited to the currently-
                                # active branch tab. Selected chips
                                # (in v_model) stay visible even
                                # when their branch isn't the
                                # active tab — so the user can
                                # mix-and-match across branches.
                                items=(
                                    "fiber_sel_items.filter("
                                    "  it => "
                                    "    it.branch.toString() "
                                    "    === fiber_sel_tab"
                                    ")",
                                ),
                                item_title="title",
                                item_value="value",
                                label="trajectories (empty = all)",
                                multiple=True,
                                chips=True,
                                clearable=True,
                                closable_chips=True,
                                return_object=False,
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                disabled=("sweep_busy",),
                                menu_props=(
                                    "{ maxHeight: '420px' }",
                                ),
                            ):
                                # ---- prepend-item slot ----
                                # Branch tabs at the TOP of the
                                # dropdown menu, one VTab per branch
                                # detected (same UI as the Single-
                                # fiber tab's combobox). Switching
                                # tabs filters the visible items via
                                # the `items=` filter expression
                                # above; selected chips persist
                                # across tab switches.
                                with v3.Template(
                                    v_slot_prepend_item=True,
                                ):
                                    with v3.VTabs(
                                        v_model=("fiber_sel_tab",),
                                        density="compact",
                                        grow=True,
                                        slider_color="primary",
                                        classes=(
                                            "golgi-fiber-tab-strip"
                                        ),
                                    ):
                                        v3.VTab(
                                            "{{ m.label }}",
                                            v_for=(
                                                "m in "
                                                "pop_branches_meta"
                                            ),
                                            key="m.idx",
                                            value=(
                                                "m.idx.toString()",
                                            ),
                                        )
                                    v3.VDivider()
                                # ---- chip slot ----
                                with v3.Template(
                                    v_slot_chip=(
                                        "{ props, item }",
                                    ),
                                ):
                                    with v3.VChip(
                                        v_bind="props",
                                        size="small",
                                        closable=True,
                                    ):
                                        html.Span(
                                            classes=(
                                                "golgi-fiber-chip-dot"
                                            ),
                                            style=(
                                                "'background:' + ("
                                                "  (fiber_sel_items."
                                                "    find(it => "
                                                "      Number(it.value) "
                                                "      === Number(item.value)"
                                                "    ) || {})"
                                                "  .color || '#888'"
                                                ")",
                                            ),
                                        )
                                        html.Span(
                                            "{{ "
                                            "  (fiber_sel_items"
                                            "    .find(it => "
                                            "      Number(it.value) "
                                            "      === Number(item.value)"
                                            "    ) || {}).title "
                                            "  || ('Fiber ' + item.value) "
                                            "}}"
                                        )

                            # ---- Run buttons ----
                            with html.Div(
                                classes="d-flex align-center mt-3",
                                style="gap: 8px;",
                            ):
                                html.Button(
                                    "▶ Run amplitude sweep",
                                    type="button",
                                    classes=(
                                        "golgi-btn-primary"
                                    ),
                                    disabled=(
                                        "sweep_busy || "
                                        "sweep_mode !== "
                                        "'recruitment'",
                                    ),
                                    click=do_run_amplitude_sweep,
                                )
                                html.Button(
                                    "▶ Find thresholds",
                                    type="button",
                                    classes=(
                                        "golgi-btn-primary"
                                    ),
                                    disabled=(
                                        "sweep_busy || "
                                        "sweep_mode !== "
                                        "'threshold'",
                                    ),
                                    click=do_find_thresholds,
                                )

                            # Status line.
                            html.Div(
                                "{{ sweep_status }}",
                                style=(
                                    "font-size: 11px; "
                                    "color: #444; "
                                    "background: #f6f6f7; "
                                    "padding: 6px 10px; "
                                    "border-radius: 4px; "
                                    "margin-top: 12px; "
                                    "font-family: monospace;"
                                ),
                            )

                            # ---- Result figures ----
                            with html.Div(
                                v_show=("sweep_has_result",),
                                classes="mt-4",
                            ):
                                html.Div(
                                    "{{ sweep_result_summary }}",
                                    style=(
                                        "font-size: 11px; "
                                        "color: #146e3a; "
                                        "font-weight: 500; "
                                        "margin-bottom: 8px;"
                                    ),
                                )
                                # Recruitment curve.
                                with html.Div(
                                    classes="golgi-fiber-tile",
                                    style=(
                                        "height: 360px; "
                                        "margin-bottom: 6px;"
                                    ),
                                ):
                                    _export_btn("sweep.recruitment")
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "sweep_"
                                                "recruitment_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=True,
                                        )
                                # CSV download button — plain <a>
                                # with golgi-btn-secondary (outlined
                                # pill, same family as the welcome-
                                # view "Documentation" CTA). Anchor
                                # carries the data URI as href +
                                # the download attr, so the browser
                                # handles the save dialog natively
                                # — no client-side JS, and avoids
                                # Vuetify VBtn's own styling that
                                # conflicts with the golgi pill
                                # buttons used elsewhere.
                                with html.Div(
                                    v_show=(
                                        "sweep_recruitment_csv_"
                                        "data_uri",
                                    ),
                                    classes="mb-3",
                                ):
                                    with html.A(
                                        href=(
                                            "sweep_recruitment_"
                                            "csv_data_uri",
                                        ),
                                        download=(
                                            "sweep_recruitment_"
                                            "csv_filename",
                                        ),
                                        classes=(
                                            "golgi-btn-secondary "
                                            "golgi-btn-sm"
                                        ),
                                    ):
                                        html.I(
                                            classes=(
                                                "mdi mdi-"
                                                "file-download-"
                                                "outline"
                                            ),
                                            style=(
                                                "font-size: 16px;"
                                            ),
                                        )
                                        html.Span(
                                            "Recruitment CSV",
                                        )

                                # Threshold scatter.
                                with html.Div(
                                    classes="golgi-fiber-tile",
                                    style=(
                                        "height: 360px; "
                                        "margin-bottom: 6px;"
                                    ),
                                ):
                                    _export_btn(
                                        "sweep.threshold_scatter"
                                    )
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "sweep_"
                                                "threshold_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=True,
                                        )
                                with html.Div(
                                    v_show=(
                                        "sweep_threshold_csv_"
                                        "data_uri",
                                    ),
                                    classes="mb-3",
                                ):
                                    with html.A(
                                        href=(
                                            "sweep_threshold_"
                                            "csv_data_uri",
                                        ),
                                        download=(
                                            "sweep_threshold_"
                                            "csv_filename",
                                        ),
                                        classes=(
                                            "golgi-btn-secondary "
                                            "golgi-btn-sm"
                                        ),
                                    ):
                                        html.I(
                                            classes=(
                                                "mdi mdi-"
                                                "file-download-"
                                                "outline"
                                            ),
                                            style=(
                                                "font-size: 16px;"
                                            ),
                                        )
                                        html.Span(
                                            "Threshold CSV",
                                        )

                                # Activation heatmap.
                                with html.Div(
                                    classes="golgi-fiber-tile",
                                    style=(
                                        "height: 400px; "
                                        "margin-bottom: 6px;"
                                    ),
                                ):
                                    _export_btn(
                                        "sweep.activation_heatmap"
                                    )
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "sweep_"
                                                "heatmap_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=True,
                                        )
                                with html.Div(
                                    v_show=(
                                        "sweep_heatmap_csv_"
                                        "data_uri",
                                    ),
                                    classes="mb-3",
                                ):
                                    with html.A(
                                        href=(
                                            "sweep_heatmap_"
                                            "csv_data_uri",
                                        ),
                                        download=(
                                            "sweep_heatmap_"
                                            "csv_filename",
                                        ),
                                        classes=(
                                            "golgi-btn-secondary "
                                            "golgi-btn-sm"
                                        ),
                                    ):
                                        html.I(
                                            classes=(
                                                "mdi mdi-"
                                                "file-download-"
                                                "outline"
                                            ),
                                            style=(
                                                "font-size: 16px;"
                                            ),
                                        )
                                        html.Span(
                                            "Activation map CSV",
                                        )

                                # Full SweepResult binary (.npz).
                                # Smaller than the heatmap CSV
                                # usually; useful for sharing the
                                # whole result with a collaborator
                                # who can re-load via the F4.1
                                # headless API.
                                with html.Div(
                                    v_show=(
                                        "sweep_npz_data_uri",
                                    ),
                                    classes="mb-2",
                                ):
                                    with html.A(
                                        href=(
                                            "sweep_npz_data_uri",
                                        ),
                                        download=(
                                            "sweep_npz_filename",
                                        ),
                                        classes=(
                                            "golgi-btn-secondary "
                                            "golgi-btn-sm"
                                        ),
                                    ):
                                        html.I(
                                            classes=(
                                                "mdi mdi-"
                                                "package-down"
                                            ),
                                            style=(
                                                "font-size: 16px;"
                                            ),
                                        )
                                        html.Span(
                                            "Full result (.npz)",
                                        )

                    # ---- F3.2e Compare panel ----
                    # Side-by-side overlays of solved configs.
                    # Sibling to the Fiber / Population / Sweep
                    # panels; gated on
                    # active_analysis === 'compare'.
                    with html.Div(
                        v_show=(
                            "active_analysis === 'compare'",
                        ),
                        classes=(
                            "golgi-analysis-panel "
                            "golgi-compare-panel"
                        ),
                    ):
                        with html.Div(
                            classes="golgi-fiber-title-row",
                        ):
                            html.H2(
                                "Compare configurations"
                            )
                            with v3.VTooltip(
                                location="top",
                                max_width=420,
                            ):
                                with v3.Template(
                                    v_slot_activator=(
                                        "{ props }",
                                    ),
                                ):
                                    with v3.VBtn(
                                        v_bind="props",
                                        icon=True,
                                        size="small",
                                        variant="text",
                                        density="compact",
                                    ):
                                        v3.VIcon(
                                            "mdi-information-"
                                            "outline",
                                            size="18",
                                            color=(
                                                "grey-darken-1"
                                            ),
                                        )
                                html.Span(
                                    "Overlay solved "
                                    "configurations side-by-"
                                    "side. The axis-overlay "
                                    "plot shows V_e along the "
                                    "cuff axis (one line per "
                                    "config). The slice grid "
                                    "shows the V_e slice at a "
                                    "chosen z-index — one "
                                    "subplot per config, "
                                    "common colour scale so "
                                    "magnitudes are "
                                    "comparable."
                                )
                        # Multi-select: which configs to overlay.
                        with html.Div(
                            classes="d-flex align-center",
                            style=(
                                "gap: 12px; "
                                "margin: 8px 0 16px 0; "
                                "flex-wrap: wrap;"
                            ),
                        ):
                            html.Span(
                                "Configs:",
                                style=(
                                    "font-size: 11px; "
                                    "color: #555; "
                                    "min-width: 60px;"
                                ),
                            )
                            v3.VSelect(
                                v_model=(
                                    "compare_config_selection",
                                ),
                                items=("compare_config_items",),
                                item_title="title",
                                item_value="value",
                                multiple=True,
                                chips=True,
                                closable_chips=True,
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                style="flex: 1 1 320px;",
                            )
                            html.Span(
                                "z-slice idx:",
                                style=(
                                    "font-size: 11px; "
                                    "color: #555;"
                                ),
                            )
                            v3.VTextField(
                                v_model_number=(
                                    "compare_slice_z_idx",
                                ),
                                type="number",
                                min=0, step=1,
                                density="compact",
                                hide_details=True,
                                variant="outlined",
                                style=(
                                    "flex: 0 0 110px;"
                                ),
                            )
                        # Empty-state hint when no configs solved.
                        html.Div(
                            "No FEM-solved configurations on "
                            "disk yet. Run the FEM solver on at "
                            "least two configs first.",
                            v_show=(
                                "!(fem_configs "
                                "  && fem_configs.length >= 2)",
                            ),
                            style=(
                                "padding: 12px; "
                                "color: #888; "
                                "font-size: 12px; "
                                "font-style: italic; "
                                "border: 1px dashed #e3e3e6; "
                                "border-radius: 8px;"
                            ),
                        )
                        # Two stacked figures.
                        with html.Div(
                            v_show=(
                                "fem_configs "
                                "&& fem_configs.length >= 2",
                            ),
                            style=(
                                "display: flex; "
                                "flex-direction: column; "
                                "gap: 16px;"
                            ),
                        ):
                            with html.Div(
                                id="tile-compare-axis",
                                classes=(
                                    "golgi-compare-tile"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-tile-header"
                                    ),
                                ):
                                    html.Span(
                                        "V_e along the "
                                        "cuff axis",
                                        classes=(
                                            "golgi-tile-title"
                                        ),
                                    )
                                with html.Div(
                                    classes=(
                                        "golgi-tile-body"
                                    ),
                                ):
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "compare_axis_"
                                                "figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=(
                                                True
                                            ),
                                        )
                            with html.Div(
                                id="tile-compare-slice-grid",
                                classes=(
                                    "golgi-compare-tile"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-tile-header"
                                    ),
                                ):
                                    html.Span(
                                        "V_e slice grid",
                                        classes=(
                                            "golgi-tile-title"
                                        ),
                                    )
                                with html.Div(
                                    classes=(
                                        "golgi-tile-body"
                                    ),
                                ):
                                    if twp is not None:
                                        twp.Figure(
                                            state_variable_name=(
                                                "compare_slice_"
                                                "grid_figure"
                                            ),
                                            display_logo=False,
                                            display_mode_bar=(
                                                True
                                            ),
                                        )
                            # F3.2 — Selectivity tile. Picks a
                            # target branch (and optional off-
                            # target multi-select), reads each
                            # picked config's cid-tagged sweep
                            # result, renders the Veraart SI bar
                            # chart at the chosen amplitude + a
                            # threshold-ratio table. Inputs are
                            # gated on having any sweep results
                            # at all; an empty-state hint shows
                            # otherwise.
                            with html.Div(
                                id="tile-compare-selectivity",
                                classes=(
                                    "golgi-compare-tile"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-tile-header"
                                    ),
                                ):
                                    html.Span(
                                        "Selectivity",
                                        classes=(
                                            "golgi-tile-title"
                                        ),
                                    )
                                with html.Div(
                                    classes=(
                                        "golgi-tile-body"
                                    ),
                                ):
                                    # ---- Controls row ----
                                    with html.Div(
                                        classes=(
                                            "d-flex "
                                            "align-center"
                                        ),
                                        style=(
                                            "gap: 12px; "
                                            "flex-wrap: wrap; "
                                            "margin-bottom: "
                                            "10px;"
                                        ),
                                    ):
                                        html.Span(
                                            "Target:",
                                            style=(
                                                "font-size: "
                                                "11px; "
                                                "color: #555;"
                                            ),
                                        )
                                        v3.VSelect(
                                            v_model=(
                                                "selectivity_"
                                                "target_branch",
                                            ),
                                            items=(
                                                "selectivity_"
                                                "branch_items",
                                            ),
                                            item_title="title",
                                            item_value="value",
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "flex: 0 0 "
                                                "180px;"
                                            ),
                                        )
                                        html.Span(
                                            "Off-target:",
                                            style=(
                                                "font-size: "
                                                "11px; "
                                                "color: #555;"
                                            ),
                                        )
                                        v3.VSelect(
                                            v_model=(
                                                "selectivity_"
                                                "offtarget_"
                                                "branches",
                                            ),
                                            items=(
                                                "selectivity_"
                                                "branch_items",
                                            ),
                                            item_title="title",
                                            item_value="value",
                                            multiple=True,
                                            chips=True,
                                            closable_chips=(
                                                True
                                            ),
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            placeholder=(
                                                "all others"
                                            ),
                                            persistent_placeholder=(
                                                True
                                            ),
                                            style=(
                                                "flex: 1 1 "
                                                "220px;"
                                            ),
                                        )
                                        html.Span(
                                            "Amp (mA):",
                                            style=(
                                                "font-size: "
                                                "11px; "
                                                "color: #555;"
                                            ),
                                        )
                                        v3.VTextField(
                                            v_model_number=(
                                                "selectivity_"
                                                "amplitude_mA",
                                            ),
                                            type="number",
                                            min=0, step=0.1,
                                            density="compact",
                                            hide_details=True,
                                            variant="outlined",
                                            style=(
                                                "flex: 0 0 "
                                                "100px;"
                                            ),
                                        )
                                    # ---- Status line ----
                                    html.Div(
                                        "{{ selectivity_status "
                                        "}}",
                                        v_show=(
                                            "selectivity_"
                                            "status",
                                        ),
                                        style=(
                                            "font-size: 11px; "
                                            "color: #b45309; "
                                            "background: "
                                            "#fff7ed; "
                                            "border: 1px "
                                            "solid #fed7aa; "
                                            "border-radius: "
                                            "4px; padding: "
                                            "6px 10px; "
                                            "margin-bottom: "
                                            "10px;"
                                        ),
                                    )
                                    # ---- SI bar chart ----
                                    with html.Div(
                                        style=(
                                            "width: 100%; "
                                            "height: 320px; "
                                            "border: 1px "
                                            "solid #e6e6e8; "
                                            "border-radius: "
                                            "6px; background: "
                                            "white; "
                                            "margin-bottom: "
                                            "12px;"
                                        ),
                                    ):
                                        if twp is not None:
                                            twp.Figure(
                                                state_variable_name=(
                                                    "selectivity_"
                                                    "bar_figure"
                                                ),
                                                display_logo=(
                                                    False
                                                ),
                                                display_mode_bar=(
                                                    True
                                                ),
                                            )
                                    # ---- Threshold ratio
                                    # table (HTML, populated
                                    # from threshold-mode
                                    # sweeps). ----
                                    html.Div(
                                        v_html=(
                                            "selectivity_"
                                            "table_html",
                                        ),
                                    )
                            # I1 Phase A — Impedance tile.
                            # Sibling to Selectivity. Reads
                            # `state.fem_impedance` (populated
                            # by the FEM solve when
                            # `emit_impedance=True`). Two
                            # stacked bar charts: per-contact
                            # access impedance + per-pair
                            # stimulation impedance.
                            with html.Div(
                                id="tile-compare-impedance",
                                classes=(
                                    "golgi-compare-tile"
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-tile-header"
                                    ),
                                ):
                                    html.Span(
                                        "Access impedance",
                                        classes=(
                                            "golgi-tile-title"
                                        ),
                                    )
                                with html.Div(
                                    classes=(
                                        "golgi-tile-body"
                                    ),
                                ):
                                    # Per-contact bar.
                                    with html.Div(
                                        style=(
                                            "width: 100%; "
                                            "height: 280px; "
                                            "border: 1px "
                                            "solid #e6e6e8; "
                                            "border-radius: "
                                            "6px; "
                                            "background: "
                                            "white; "
                                            "margin-bottom: "
                                            "12px;"
                                        ),
                                    ):
                                        if twp is not None:
                                            twp.Figure(
                                                state_variable_name=(
                                                    "impedance_"
                                                    "bar_figure"
                                                ),
                                                display_logo=(
                                                    False
                                                ),
                                                display_mode_bar=(
                                                    True
                                                ),
                                            )
                                    # Per-pair bar.
                                    with html.Div(
                                        style=(
                                            "width: 100%; "
                                            "height: 280px; "
                                            "border: 1px "
                                            "solid #e6e6e8; "
                                            "border-radius: "
                                            "6px; "
                                            "background: "
                                            "white;"
                                        ),
                                    ):
                                        if twp is not None:
                                            twp.Figure(
                                                state_variable_name=(
                                                    "impedance_"
                                                    "per_pair_"
                                                    "figure"
                                                ),
                                                display_logo=(
                                                    False
                                                ),
                                                display_mode_bar=(
                                                    True
                                                ),
                                            )

                # 3D viewport — flex sibling to the analysis area.
                # `mode-full` (3d) → fills available height;
                # `mode-grid-tl` (Solve) → top-left of 2×2 grid;
                # `mode-fiber-band` (Single fiber) → standalone
                #   card-style tile above the unified playground
                #   panel, aligned with its width so the page
                #   reads as viewport-on-top, controls-below;
                # `mode-pop-band` (Population) → same as
                #   mode-fiber-band; mirrors the Single-fiber
                #   layout so the two tabs feel consistent;
                # `mode-panel` (fallback) → fixed 52vh top panel.
                # Scale bar + legend live INSIDE so they overlay
                # the 3D scene regardless of mode.
                with html.Div(
                    classes=(
                        "['golgi-viewport', "
                        "viewport_mode === 'analysis' "
                        "  ? (active_analysis === 'solve' "
                        "      ? 'mode-grid-tl' "
                        "      : (active_analysis === 'fiber' "
                        "          ? 'mode-fiber-band' "
                        "          : (active_analysis === 'population' "
                        "              ? 'mode-pop-band' "
                        "              : 'mode-panel'))) "
                        "  : 'mode-full']",
                    ),
                    # Double-click selects the cuff under the
                    # cursor. The args expression normalises the
                    # offset into [0, 1] and flips the Y axis
                    # (DOM is top-down, VTK is bottom-up) so the
                    # server-side picker can rescale to its own
                    # render-window dimensions.
                    dblclick=(
                        do_pick_electrode_at,
                        "[$event.offsetX / "
                        "    $event.currentTarget.clientWidth, "
                        " 1 - $event.offsetY / "
                        "    $event.currentTarget.clientHeight]",
                    ),
                ):
                    view = plotter_ui(
                        pl,
                        interactive_ratio=1,
                        mode="trame",
                        default_server_rendering=False,
                    )
                    ctrl.view_update = view.update
                    ctrl.view_reset_camera = view.reset_camera
                    ctrl.view_push_camera = view.push_camera

                    # F2.3.a — floating camera button anchored to
                    # the workspace viewport (bottom-right, away
                    # from the legend FAB in the top-right).
                    # Triggers do_export_viewport_screenshot('main')
                    # which screenshots the live PyVista plotter
                    # and pushes the PNG bytes as a data URI to
                    # state.export_pending_*.
                    _ui.components.viewport_export_btn.render(
                        viewport_id="main",
                        do_export_viewport_screenshot=(
                            do_export_viewport_screenshot
                        ),
                        label="3D viewport",
                    )

                    # (Static "5 mm" scale-bar removed — it was
                    # a fixed 150 × 10 px overlay that didn't
                    # actually scale with the viewport's zoom,
                    # so it became misleading at non-default
                    # zoom levels. The matching `.golgi-scalebar`
                    # / `.golgi-scalebar-bar` /
                    # `.golgi-scalebar-label` CSS classes stay
                    # in golgi.css as harmless dead style.)

                    # Horizontal Vₑ colourbar — only visible when a
                    # FEM solve has produced a Vₑ field AND at least
                    # one of the Vₑ overlays is enabled (fibers OR
                    # surface). Gives a shared mV scale for the
                    # endo/epi plasma overlays + the fiber tubes.
                    with html.Div(
                        classes="golgi-ve-cbar",
                        v_show=(
                            "view_mode === 'workspace' "
                            "&& fem_ve_cbar_b64 "
                            "&& (show_ve_fibers "
                            "    || show_ve_surface)",
                        ),
                    ):
                        html.Img(src=("fem_ve_cbar_b64",))

                    # ====================================================
                    # SceneCatalog Phase 6b — viewport top-right overlay.
                    # Combobox (design picker) sits LEFT of the eye-icon
                    # FAB. Combobox appears once a nerve is loaded; FAB
                    # always visible in workspace mode. The legend popup
                    # hangs below this row when the FAB is toggled.
                    # ====================================================
                    with html.Div(
                        classes="golgi-legend-toprow",
                        v_show=("view_mode === 'workspace'",),
                    ):
                        # Design picker — left of FAB. Visible once
                        # geometry is loaded; empty designs list →
                        # disabled placeholder.
                        with html.Div(
                            classes="golgi-legend-design-outer",
                            v_show=("has_geometry",),
                        ):
                            v3.VSelect(
                                v_model=("design_config_key",),
                                items=("design_config_items",),
                                item_value="value",
                                item_title="title",
                                # When the user has no designs
                                # yet, swap the floating label
                                # from "Design" to a clearer
                                # "No designs yet" hint and pin
                                # the placeholder so the
                                # "create one in the Designs
                                # tab" pointer is always visible
                                # (not just on focus).
                                label=(
                                    "designs && designs.length "
                                    "  ? 'Design' "
                                    "  : 'No designs yet'",
                                ),
                                density="compact",
                                variant="outlined",
                                hide_details=True,
                                disabled=(
                                    "!designs "
                                    "|| designs.length === 0",
                                ),
                                placeholder=(
                                    "Add one in the "
                                    "Designs tab"
                                ),
                                persistent_placeholder=True,
                                no_data_text=(
                                    "No designs yet — add one "
                                    "in the Designs tab"
                                ),
                                classes=(
                                    "golgi-legend-design-select"
                                ),
                            )
                        # Eye-icon FAB
                        v3.VBtn(
                            icon=(
                                "legend_visible "
                                "? 'mdi-eye-off' : 'mdi-eye'",
                            ),
                            size="small",
                            density="comfortable",
                            variant="elevated",
                            classes="golgi-legend-fab",
                            click=(
                                "legend_visible = !legend_visible"
                            ),
                        )

                    # ====================================================
                    # SceneCatalog Phase 6b — floating visibility legend.
                    # Hidden by default; revealed by the FAB above.
                    # Per-design view scoped to the focused design
                    # (combobox above). Six foldable supersections, each
                    # gated on the workflow stage it represents:
                    #   Nerve       — after has_geometry
                    #   Fibers      — after has_fibers
                    #   Muscle      — after has_geometry
                    #   Cuff Elec   — after a design exists (focused)
                    #   Mesh        — after that design's has_mesh
                    #   Overlays    — after has_mesh (Mesh Quality)
                    #                 + has_fem (Vₑ, E-field)
                    # ====================================================

                    # ---- Precompute swatch colour strings ----
                    _endo_r, _endo_g, _endo_b = DEFAULTS[1]["color"]
                    _endo_swatch = (
                        f"background: rgb("
                        f"{int(_endo_r*255)},"
                        f"{int(_endo_g*255)},"
                        f"{int(_endo_b*255)});"
                    )
                    _epi_r, _epi_g, _epi_b = DEFAULTS[5]["color"]
                    _epi_swatch = (
                        f"background: rgb("
                        f"{int(_epi_r*255)},"
                        f"{int(_epi_g*255)},"
                        f"{int(_epi_b*255)});"
                    )
                    _mus_r, _mus_g, _mus_b = DEFAULTS[4]["color"]
                    _mus_swatch = (
                        f"background: rgb("
                        f"{int(_mus_r*255)},"
                        f"{int(_mus_g*255)},"
                        f"{int(_mus_b*255)});"
                    )
                    _sil_r, _sil_g, _sil_b = DEFAULTS[3]["color"]
                    _sil_swatch = (
                        f"background: rgb("
                        f"{int(_sil_r*255)},"
                        f"{int(_sil_g*255)},"
                        f"{int(_sil_b*255)});"
                    )
                    _sal_r, _sal_g, _sal_b = SALINE_OVERLAY_STYLE["color"]
                    _sal_swatch = (
                        f"background: rgb("
                        f"{int(_sal_r*255)},"
                        f"{int(_sal_g*255)},"
                        f"{int(_sal_b*255)});"
                    )
                    _g_r, _g_g, _g_b = GOLD_STYLE["color"]
                    _g_swatch = (
                        f"background: rgb("
                        f"{int(_g_r*255)},"
                        f"{int(_g_g*255)},"
                        f"{int(_g_b*255)});"
                    )
                    # F3.2-M3 — scar / connective tissue swatch.
                    _sca_r, _sca_g, _sca_b = DEFAULTS[7]["color"]
                    _sca_swatch = (
                        f"background: rgb("
                        f"{int(_sca_r*255)},"
                        f"{int(_sca_g*255)},"
                        f"{int(_sca_b*255)});"
                    )
                    _q_swatch = (
                        "background: linear-gradient(90deg, "
                        "#a50026, #d73027, #fdae61, #fee08b, "
                        "#d9ef8b, #66bd63, #1a9850);"
                    )
                    _ve_swatch = (
                        "background: linear-gradient(90deg, "
                        "#0d0887, #6a00a8, #b12a90, #e16462, "
                        "#fca636, #fcfdbf);"
                    )
                    _e_swatch = (
                        "background: linear-gradient(90deg, "
                        "#440154, #3b528b, #21918c, #5ec962, "
                        "#fde725);"
                    )

                    with html.Div(
                        classes="golgi-legend",
                        v_show=(
                            "view_mode === 'workspace' "
                            "&& legend_visible",
                        ),
                    ):
                        # ===== Pre-design fallback view =====
                        # When no design has been placed yet
                        # (just imported the nerve), the per-
                        # design v_for body has nothing to
                        # iterate. Show a minimal global view
                        # (Nerve / Fibers / Muscle) using the
                        # pre-design state vars so the user
                        # still sees something useful between
                        # "load nerve" and "add design".
                        with html.Div(
                            v_show=(
                                "has_geometry "
                                "&& (!designs "
                                "|| designs.length === 0)",
                            ),
                        ):
                            # ----- Nerve (pre-design) -----
                            with html.Div(
                                classes="golgi-legend-supheader",
                                click=(
                                    "legend_nerve_open = "
                                    "!legend_nerve_open"
                                ),
                            ):
                                html.Span(
                                    "{{ legend_nerve_open "
                                    "? '▾' : '▸' }}",
                                    classes=(
                                        "golgi-legend-chevron"
                                    ),
                                )
                                html.Span("Nerve")
                            with html.Div(
                                v_show=("legend_nerve_open",),
                            ):
                                # Endoneurium (raw STL surface).
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!vis_nerve_raw}]",
                                    ),
                                    click=(
                                        "vis_nerve_raw = "
                                        "!vis_nerve_raw"
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_endo_swatch,
                                    )
                                    html.Span(
                                        "Endoneurium",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                # Epineurium (preview, gated on
                                # use_epi for the legacy inward-
                                # offset shell OR `import_source_-
                                # type === 'uct_bundle'` for the
                                # bundle path — in bundle mode the
                                # imported epi surface is rendered
                                # as the translucent nerve actor
                                # and this row drives its
                                # visibility independently from
                                # the per-fascicle endoneurium
                                # actors).
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!vis_epi_preview}]",
                                    ),
                                    v_show=(
                                        "use_epi "
                                        "|| import_source_type "
                                        "=== 'uct_bundle' "
                                        "|| import_source_type "
                                        "=== 'histo_bundle'",
                                    ),
                                    click=(
                                        "vis_epi_preview = "
                                        "!vis_epi_preview"
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_epi_swatch,
                                    )
                                    html.Span(
                                        "Epineurium",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                            # ----- Fibers (pre-design) -----
                            with html.Div(
                                v_show=("has_fibers",),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-legend-supheader"
                                    ),
                                    click=(
                                        "legend_fibers_open = "
                                        "!legend_fibers_open"
                                    ),
                                ):
                                    html.Span(
                                        "{{ legend_fibers_open "
                                        "? '▾' : '▸' }}",
                                        classes=(
                                            "golgi-legend-chevron"
                                        ),
                                    )
                                    html.Span("Fibers")
                                with html.Div(
                                    v_show=(
                                        "legend_fibers_open",
                                    ),
                                ):
                                    # Trajectories master.
                                    with html.Div(
                                        classes=(
                                            "['golgi-legend-row', "
                                            "'golgi-legend-sub', "
                                            "{'is-off': "
                                            "!vis_fibers}]",
                                        ),
                                        click=(
                                            "vis_fibers = "
                                            "!vis_fibers"
                                        ),
                                    ):
                                        html.Div(
                                            classes=(
                                                "golgi-legend-swatch"
                                            ),
                                            style=(
                                                f"background: "
                                                f"{FIBERS_MASTER_COLOUR};"
                                            ),
                                        )
                                        html.Span(
                                            "Trajectories",
                                            classes=(
                                                "golgi-legend-label"
                                            ),
                                        )
                                    # Per-branch rows.
                                    for _i, _color in enumerate(
                                        BRANCH_PALETTE,
                                    ):
                                        _vis_key = (
                                            f"vis_fiber_branch_{_i}"
                                        )
                                        with html.Div(
                                            classes=(
                                                f"['golgi-legend-row', "
                                                f"'golgi-legend-sub', "
                                                f"{{'is-off': "
                                                f"!{_vis_key}}}]",
                                            ),
                                            style=(
                                                "padding-left: "
                                                "38px;"
                                            ),
                                            v_show=(
                                                f"fiber_n_branches "
                                                f"> 1 "
                                                f"&& {_i} < "
                                                f"fiber_n_branches",
                                            ),
                                            click=(
                                                f"{_vis_key} = "
                                                f"!{_vis_key}"
                                            ),
                                        ):
                                            html.Div(
                                                classes=(
                                                    "golgi-legend-swatch"
                                                ),
                                                style=(
                                                    f"background: "
                                                    f"{_color};"
                                                ),
                                            )
                                            html.Span(
                                                f"{{{{ "
                                                f"fiber_branch_name_{_i} "
                                                f"|| 'Branch {_i}' }}}}",
                                                classes=(
                                                    "golgi-legend-label"
                                                ),
                                            )
                            # ----- Muscle (pre-design) -----
                            # F3.2-M3: muscle row hidden until
                            # the user reaches Step 4 of the
                            # import stepper at least once (see
                            # `muscle_preview_unlocked` watcher).
                            with html.Div(
                                classes="golgi-legend-supheader",
                                v_show=(
                                    "muscle_preview_unlocked",
                                ),
                                click=(
                                    "legend_muscle_open = "
                                    "!legend_muscle_open"
                                ),
                            ):
                                html.Span(
                                    "{{ legend_muscle_open "
                                    "? '▾' : '▸' }}",
                                    classes=(
                                        "golgi-legend-chevron"
                                    ),
                                )
                                html.Span("Muscle")
                            with html.Div(
                                v_show=(
                                    "legend_muscle_open "
                                    "&& muscle_preview_unlocked",
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!vis_muscle_preview}]",
                                    ),
                                    click=(
                                        "vis_muscle_preview = "
                                        "!vis_muscle_preview"
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_mus_swatch,
                                    )
                                    html.Span(
                                        "Muscle",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )

                        # ===== Per-design view =====
                        # Iterates state.designs but filters to
                        # the focused one — combobox above sets
                        # `selected_design_id`. Each design's
                        # six supersections appear/disappear
                        # based on that design's progress (cuff
                        # placed → cuff section; meshed → mesh +
                        # mesh quality; FEM solved → Vₑ + field).
                        with html.Div(
                            v_for=("elec in designs",),
                            key="elec.eid",
                            v_show=(
                                "elec.eid === selected_design_id",
                            ),
                        ):
                            # ----- Nerve -----
                            with html.Div(
                                classes="golgi-legend-supheader",
                                click=(
                                    "legend_nerve_open = "
                                    "!legend_nerve_open"
                                ),
                            ):
                                html.Span(
                                    "{{ legend_nerve_open "
                                    "? '▾' : '▸' }}",
                                    classes=(
                                        "golgi-legend-chevron"
                                    ),
                                )
                                html.Span("Nerve")
                            with html.Div(
                                v_show=("legend_nerve_open",),
                            ):
                                # Endoneurium — pre-mesh row
                                # uses global `vis_nerve_raw`,
                                # post-mesh row uses the per-
                                # design `vis_endo` (controls
                                # the meshed tag 1 region).
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!vis_nerve_raw}]",
                                    ),
                                    v_show=("!elec.has_mesh",),
                                    click=(
                                        "vis_nerve_raw = "
                                        "!vis_nerve_raw"
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_endo_swatch,
                                    )
                                    html.Span(
                                        "Endoneurium",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_endo}]",
                                    ),
                                    v_show=("elec.has_mesh",),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_endo']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_endo_swatch,
                                    )
                                    html.Span(
                                        "Endoneurium",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                # Epineurium — pre-mesh preview
                                # (gated on use_epi for the
                                # legacy inward-offset shell OR
                                # `import_source_type === 'uct_-
                                # bundle'` for the µCT bundle
                                # path); post-mesh meshed tag 5.
                                # Without the uct_bundle branch
                                # this row would silently vanish
                                # once the user added a design,
                                # because µCT bundles use
                                # use_epi=False.
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!vis_epi_preview}]",
                                    ),
                                    v_show=(
                                        "!elec.has_mesh "
                                        "&& (use_epi "
                                        "    || import_source_type "
                                        "       === 'uct_bundle' "
                                        "    || import_source_type "
                                        "       === 'histo_bundle')",
                                    ),
                                    click=(
                                        "vis_epi_preview = "
                                        "!vis_epi_preview"
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_epi_swatch,
                                    )
                                    html.Span(
                                        "Epineurium",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_epi}]",
                                    ),
                                    v_show=("elec.has_mesh",),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_epi']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_epi_swatch,
                                    )
                                    html.Span(
                                        "Epineurium",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )

                            # ----- Fibers -----
                            with html.Div(
                                v_show=("has_fibers",),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-legend-supheader"
                                    ),
                                    click=(
                                        "legend_fibers_open = "
                                        "!legend_fibers_open"
                                    ),
                                ):
                                    html.Span(
                                        "{{ legend_fibers_open "
                                        "? '▾' : '▸' }}",
                                        classes=(
                                            "golgi-legend-chevron"
                                        ),
                                    )
                                    html.Span("Fibers")
                                with html.Div(
                                    v_show=(
                                        "legend_fibers_open",
                                    ),
                                ):
                                    # Trajectories master.
                                    with html.Div(
                                        classes=(
                                            "['golgi-legend-row', "
                                            "'golgi-legend-sub', "
                                            "{'is-off': "
                                            "!vis_fibers}]",
                                        ),
                                        click=(
                                            "vis_fibers = "
                                            "!vis_fibers"
                                        ),
                                    ):
                                        html.Div(
                                            classes=(
                                                "golgi-legend-swatch"
                                            ),
                                            style=(
                                                f"background: "
                                                f"{FIBERS_MASTER_COLOUR};"
                                            ),
                                        )
                                        html.Span(
                                            "Trajectories",
                                            classes=(
                                                "golgi-legend-label"
                                            ),
                                        )
                                    for _i, _color in enumerate(
                                        BRANCH_PALETTE,
                                    ):
                                        _vis_key = (
                                            f"vis_fiber_branch_{_i}"
                                        )
                                        with html.Div(
                                            classes=(
                                                f"['golgi-legend-row', "
                                                f"'golgi-legend-sub', "
                                                f"{{'is-off': "
                                                f"!{_vis_key}}}]",
                                            ),
                                            style=(
                                                "padding-left: "
                                                "38px;"
                                            ),
                                            v_show=(
                                                f"fiber_n_branches "
                                                f"> 1 "
                                                f"&& {_i} < "
                                                f"fiber_n_branches",
                                            ),
                                            click=(
                                                f"{_vis_key} = "
                                                f"!{_vis_key}"
                                            ),
                                        ):
                                            html.Div(
                                                classes=(
                                                    "golgi-legend-swatch"
                                                ),
                                                style=(
                                                    f"background: "
                                                    f"{_color};"
                                                ),
                                            )
                                            html.Span(
                                                f"{{{{ "
                                                f"fiber_branch_name_{_i} "
                                                f"|| 'Branch {_i}' }}}}",
                                                classes=(
                                                    "golgi-legend-label"
                                                ),
                                            )

                            # ----- Muscle -----
                            # F3.2-M3: gated on the one-way
                            # `muscle_preview_unlocked` flag, same
                            # as the pre-design muscle row above.
                            with html.Div(
                                classes="golgi-legend-supheader",
                                v_show=(
                                    "muscle_preview_unlocked",
                                ),
                                click=(
                                    "legend_muscle_open = "
                                    "!legend_muscle_open"
                                ),
                            ):
                                html.Span(
                                    "{{ legend_muscle_open "
                                    "? '▾' : '▸' }}",
                                    classes=(
                                        "golgi-legend-chevron"
                                    ),
                                )
                                html.Span("Muscle")
                            with html.Div(
                                v_show=(
                                    "legend_muscle_open "
                                    "&& muscle_preview_unlocked",
                                ),
                            ):
                                # Pre-mesh: muscle bbox preview.
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!vis_muscle_preview}]",
                                    ),
                                    v_show=("!elec.has_mesh",),
                                    click=(
                                        "vis_muscle_preview = "
                                        "!vis_muscle_preview"
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_mus_swatch,
                                    )
                                    html.Span(
                                        "Muscle",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                # Post-mesh: meshed muscle (tag 4).
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_muscle}]",
                                    ),
                                    v_show=("elec.has_mesh",),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_muscle']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_mus_swatch,
                                    )
                                    html.Span(
                                        "Muscle",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )

                            # ----- Cuff Electrode -----
                            # Per-design — sub-rows cover the
                            # designer-built cuff parts pre-mesh
                            # AND the meshed cuff regions post-
                            # mesh (same vis_* flags drive both
                            # via _catalog_fold_electrodes /
                            # _catalog_fold_regions).
                            with html.Div(
                                classes="golgi-legend-supheader",
                                click=(
                                    "legend_cuff_open = "
                                    "!legend_cuff_open"
                                ),
                            ):
                                html.Span(
                                    "{{ legend_cuff_open "
                                    "? '▾' : '▸' }}",
                                    classes=(
                                        "golgi-legend-chevron"
                                    ),
                                )
                                html.Span("Cuff Electrode")
                            with html.Div(
                                v_show=("legend_cuff_open",),
                            ):
                                # Silicone.
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_silicone}]",
                                    ),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_silicone']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_sil_swatch,
                                    )
                                    html.Span(
                                        "Silicone Cuff",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                # Contacts.
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_contacts}]",
                                    ),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_contacts']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_g_swatch,
                                    )
                                    html.Span(
                                        "Contacts",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                # Saline Infill.
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_saline}]",
                                    ),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_saline']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_sal_swatch,
                                    )
                                    html.Span(
                                        "Saline Infill",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )
                                # F3.2-M3 — Scar / connective
                                # tissue row, only when this
                                # design has scar enabled.
                                with html.Div(
                                    classes=(
                                        "['golgi-legend-row', "
                                        "'golgi-legend-sub', "
                                        "{'is-off': "
                                        "!elec.vis_scar}]",
                                    ),
                                    v_show=("elec.use_scar",),
                                    click=(
                                        do_toggle_elec_vis,
                                        "[elec.eid, 'vis_scar']",
                                    ),
                                ):
                                    html.Div(
                                        classes=(
                                            "golgi-legend-swatch"
                                        ),
                                        style=_sca_swatch,
                                    )
                                    html.Span(
                                        "Scar Tissue",
                                        classes=(
                                            "golgi-legend-label"
                                        ),
                                    )

                            # ----- Mesh -----
                            # Single whole-mesh master toggle.
                            # Section + body appear once the
                            # design is meshed.
                            with html.Div(
                                v_show=("elec.has_mesh",),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-legend-supheader"
                                    ),
                                    click=(
                                        "legend_mesh_open = "
                                        "!legend_mesh_open"
                                    ),
                                ):
                                    html.Span(
                                        "{{ legend_mesh_open "
                                        "? '▾' : '▸' }}",
                                        classes=(
                                            "golgi-legend-chevron"
                                        ),
                                    )
                                    html.Span("Mesh")
                                with html.Div(
                                    v_show=(
                                        "legend_mesh_open",
                                    ),
                                ):
                                    with html.Div(
                                        classes=(
                                            "['golgi-legend-row', "
                                            "'golgi-legend-sub', "
                                            "{'is-off': "
                                            "!elec.vis_mesh}]",
                                        ),
                                        click=(
                                            do_toggle_elec_vis,
                                            "[elec.eid, 'vis_mesh']",
                                        ),
                                    ):
                                        html.Div(
                                            classes=(
                                                "golgi-legend-swatch"
                                            ),
                                            style=_endo_swatch,
                                        )
                                        html.Span(
                                            "{{ elec.name || "
                                            "elec.eid }} mesh",
                                            classes=(
                                                "golgi-legend-label"
                                            ),
                                        )

                            # ----- Overlays -----
                            # Section gated on has_mesh OR
                            # has_fem so it appears as soon as
                            # any overlay applies.
                            with html.Div(
                                v_show=(
                                    "elec.has_mesh "
                                    "|| elec.has_fem",
                                ),
                            ):
                                with html.Div(
                                    classes=(
                                        "golgi-legend-supheader"
                                    ),
                                    click=(
                                        "legend_overlays_open = "
                                        "!legend_overlays_open"
                                    ),
                                ):
                                    html.Span(
                                        "{{ legend_overlays_open "
                                        "? '▾' : '▸' }}",
                                        classes=(
                                            "golgi-legend-chevron"
                                        ),
                                    )
                                    html.Span("Overlays")
                                with html.Div(
                                    v_show=(
                                        "legend_overlays_open",
                                    ),
                                ):
                                    # Mesh Quality (per-design).
                                    with html.Div(
                                        classes=(
                                            "['golgi-legend-row', "
                                            "'golgi-legend-sub', "
                                            "{'is-off': "
                                            "!elec.vis_mesh_quality"
                                            "}]",
                                        ),
                                        v_show=(
                                            "elec.has_mesh",
                                        ),
                                        click=(
                                            do_toggle_elec_vis,
                                            "[elec.eid, "
                                            "'vis_mesh_quality']",
                                        ),
                                    ):
                                        html.Div(
                                            classes=(
                                                "golgi-legend-swatch"
                                            ),
                                            style=_q_swatch,
                                        )
                                        html.Span(
                                            "Mesh Quality",
                                            classes=(
                                                "golgi-legend-label"
                                            ),
                                        )
                                    # Vₑ — single toggle that
                                    # flips both surface and
                                    # fiber overlays together
                                    # (mirrored). Gated on
                                    # has_fem.
                                    with html.Div(
                                        classes=(
                                            "['golgi-legend-row', "
                                            "'golgi-legend-sub', "
                                            "{'is-off': "
                                            "!show_ve_surface}]",
                                        ),
                                        v_show=("elec.has_fem",),
                                        click=(
                                            "show_ve_surface = "
                                            "!show_ve_surface; "
                                            "show_ve_fibers = "
                                            "show_ve_surface"
                                        ),
                                    ):
                                        html.Div(
                                            classes=(
                                                "golgi-legend-swatch"
                                            ),
                                            style=_ve_swatch,
                                        )
                                        html.Span(
                                            "Vₑ",
                                            classes=(
                                                "golgi-legend-label"
                                            ),
                                        )
                                    # E-field streamlines.
                                    with html.Div(
                                        classes=(
                                            "['golgi-legend-row', "
                                            "'golgi-legend-sub', "
                                            "{'is-off': "
                                            "!show_field_lines}]",
                                        ),
                                        v_show=("elec.has_fem",),
                                        click=(
                                            "show_field_lines = "
                                            "!show_field_lines"
                                        ),
                                    ):
                                        html.Div(
                                            classes=(
                                                "golgi-legend-swatch"
                                            ),
                                            style=_e_swatch,
                                        )
                                        html.Span(
                                            "E-field",
                                            classes=(
                                                "golgi-legend-label"
                                            ),
                                        )


        # Login / Register dialog — extracted to
        # golgi.ui.dialogs.auth in step 5.3.
        _ui.dialogs.auth.render(
            do_close_auth_dialog=do_close_auth_dialog,
            do_submit_login=do_submit_login,
            do_submit_register=do_submit_register,
            do_switch_auth_mode=do_switch_auth_mode,
        )

        # Profile dialog — extracted to golgi.ui.dialogs.profile
        # in step 5.3.
        _ui.dialogs.profile.render(
            do_close_profile_dialog=do_close_profile_dialog,
            do_save_profile=do_save_profile,
        )

        # New-project + Close-project dialogs — extracted to
        # golgi.ui.dialogs in step 5.3.
        _ui.dialogs.new_project.render(
            do_cancel_new_project=do_cancel_new_project,
            do_create_and_open_project=do_create_and_open_project,
        )
        _ui.dialogs.close_project.render(
            do_cancel_close=do_cancel_close,
            do_confirm_close=do_confirm_close,
        )

        # Logout-confirmation dialog — extracted to
        # golgi.ui.dialogs.logout in step 5.3.
        _ui.dialogs.logout.render(
            do_dismiss_logout_dialog=do_dismiss_logout_dialog,
            do_confirm_logout=do_confirm_logout,
        )

        # Confirm-remove-geometry dialog — extracted to
        # golgi.ui.dialogs.confirm_remove_geometry in step 5.3.
        _ui.dialogs.confirm_remove_geometry.render(
            do_remove_geometry=do_remove_geometry,
        )

        # Confirm-delete-electrode dialog — extracted to
        # golgi.ui.dialogs.confirm_delete_electrode in step 5.3.
        _ui.dialogs.confirm_delete_electrode.render()
        # Confirm-delete-mesh dialog — same pattern: driven by
        # confirm_delete_mesh_eid / _name, posts the eid to
        # remove_mesh_request on confirm. Watcher in app.py
        # routes to do_delete_mesh.
        _ui.dialogs.confirm_delete_mesh.render()

        # F3.2b: contact-config sweep dialogs (random + manual)
        # + design sweep wizard. All render unconditionally —
        # visibility is gated by show_sweep_*_dialog flags
        # flipped from the drawer buttons.
        _ui.dialogs.sweep_random.render()
        _ui.dialogs.sweep_manual.render()
        _ui.dialogs.sweep_designs.render()

        # Project-detail lightbox — extracted to
        # golgi.ui.dialogs.project_detail in step 5.3.
        _ui.dialogs.project_detail.render(
            edit_icon_url=_EDIT_ICON_URL,
            do_close_detail_dialog=do_close_detail_dialog,
            do_start_edit_name=do_start_edit_name,
            do_cancel_edit_name=do_cancel_edit_name,
            do_save_edit_name=do_save_edit_name,
            do_start_add_label=do_start_add_label,
            do_cancel_add_label=do_cancel_add_label,
            do_save_add_label=do_save_add_label,
            do_open_from_detail=do_open_from_detail,
            do_request_delete_from_detail=(
                do_request_delete_from_detail
            ),
            do_toggle_activity_payload=(
                do_toggle_activity_payload
            ),
            do_save_shared_users=do_save_shared_users,
            do_export_study=do_export_study,
        )

        # F3.2-M2.1a — Import nerve stepper wizard. Replaces the
        # old `show_import` drawer for the nerve-level setup
        # flow (load → endoneurium → fibers → muscle). Opened
        # from the navbar's File → Import Nerve entry.
        _ui.dialogs.import_stepper.render(
            do_stepper_next=do_stepper_next,
            do_stepper_action=do_stepper_action,
            do_start_branch_rename=do_start_branch_rename,
            do_apply_branch_rename=do_apply_branch_rename,
            do_cancel_branch_rename=do_cancel_branch_rename,
            export_btn=_export_btn,
            do_select_source_stl=do_select_source_stl,
            do_select_source_uct_bundle=(
                do_select_source_uct_bundle
            ),
            do_select_source_histo_bundle=(
                do_select_source_histo_bundle
            ),
            do_delete_source_file=do_delete_source_file,
            do_delete_epi_file=do_delete_epi_file,
            do_delete_uct_bundle=do_delete_uct_bundle,
            do_delete_histo_bundle=do_delete_histo_bundle,
        )

        # F2.2 — Import Study dialog (Phase 2). Opened via the
        # navbar File menu's "Import study" entry; the dialog
        # handles the upload + manifest peek + the
        # Reproduction Run flow.
        _ui.dialogs.import_study.render(
            do_import_study_close=do_import_study_close,
            do_import_study_run=do_import_study_run,
            # do_import_study_upload is legacy — the file-pick
            # path now goes through the @state.change watcher
            # on `study_import_upload` registered above.
            do_import_study_upload=None,
            # Phase 3's Reproduction Run button is wired here
            # later. For now leave None so the button doesn't
            # render — the unpack-and-open Import flow works
            # standalone.
            do_import_study_check_only=None,
            # Big-bundle path: read directly from a path on
            # disk, bypasses the browser msgpack/ArrayBuffer cap.
            do_import_study_load_from_disk=(
                do_import_study_load_from_disk
            ),
        )

        # F2.2 — Export Study progress dialog. Auto-opens when
        # the export action fires; carries the spinner + the
        # Download anchor so the navbar's File → Export study
        # entry produces visible UI even with no project-detail
        # dialog open.
        _ui.dialogs.export_study.render(
            do_close=do_export_study_dialog_close,
        )

        # Delete-project confirmation dialog — extracted to
        # golgi.ui.dialogs.confirm_delete_project in step 5.3.
        # (Was inadvertently bundled into project_detail.py by
        # 5.3e's bulk sed; restored here in the hotfix.)
        _ui.dialogs.confirm_delete_project.render(
            do_cancel_delete=do_cancel_delete,
            do_confirm_delete=do_confirm_delete,
        )

        # Cole-Cole evaluator dialog — extracted to
        # golgi.ui.dialogs.cole_cole in step 5.3.
        _ui.dialogs.cole_cole.render(
            do_cole_cole_cancel=do_cole_cole_cancel,
            do_cole_cole_apply=do_cole_cole_apply,
            export_btn=_export_btn,
        )

        # Electrode designer dialog — extracted to
        # golgi.ui.dialogs.cuff_designer in step 5.3. Carries
        # the second pyvista plotter (pl_cuff) into the dialog
        # render so the plotter_ui mount lands inside it.
        _ui.dialogs.cuff_designer.render(
            pl_cuff=pl_cuff,
            ctrl=ctrl,
            do_close_cuff_designer=do_close_cuff_designer,
            do_apply_cuff_design=do_apply_cuff_design,
            do_export_viewport_screenshot=(
                do_export_viewport_screenshot
            ),
        )

        # F2.3.c — Generate Report dialog
        _ui.dialogs.generate_report.render(
            do_generate_report=do_generate_report,
            do_close_generate_report_dialog=(
                do_close_generate_report_dialog
            ),
        )

        # M47 — Histology bundle import dialog.
        _ui.dialogs.bundle_import.render(
            do_close_bundle_import_dialog=(
                do_close_bundle_import_dialog
            ),
            do_detect_bundle_files=do_detect_bundle_files,
            do_run_bundle_import=do_run_bundle_import,
        )

        # V1 Phase A.3 — µCT segmentation dialog. Opened via
        # the Import drawer's "Segment µCT slice → extrude"
        # button (temporary entry point; Phase D moves it into
        # the Import wizard as a third nerve-source tile).
        _ui.dialogs.segment_uct.render(
            do_close_segment_uct_dialog=(
                do_close_segment_uct_dialog
            ),
            do_run_uct_segmentation=do_run_uct_segmentation,
            do_clear_uct_stack=do_clear_uct_stack,
            do_label_uct_proposal=do_label_uct_proposal,
            do_generate_epi=do_generate_epi,
            do_refine_masks=do_refine_masks,
            do_save_uct_segmentation=do_save_uct_segmentation,
            do_finalize_segmentation=do_finalize_segmentation,
            do_toggle_keyframe=do_toggle_keyframe,
            do_propagate_from_keyframes=(
                do_propagate_from_keyframes
            ),
            do_recon_next=do_recon_next,
            do_recon_back=do_recon_back,
            do_run_reconstruction=do_run_reconstruction,
            do_run_reconstruction_preview=(
                do_run_reconstruction_preview
            ),
            do_finish_recon=do_finish_recon,
            pl_uct_recon=pl_uct_recon,
            ctrl=ctrl,
            plotly_module=twp,
        )

        # Cancel-busy confirmation dialog — extracted to
        # golgi.ui.dialogs.cancel_busy in step 5.3.
        _ui.dialogs.cancel_busy.render(
            do_dismiss_cancel=do_dismiss_cancel,
            do_confirm_cancel=do_confirm_cancel,
        )

    # Silence "Task exception was never retrieved" tracebacks
    # from wslink/aiohttp when the browser tab disconnects mid-
    # push. They're harmless — the client is gone, the bytes have
    # nowhere to go — but the stack-traces from un-awaited tasks
    # clutter the console.
    #
    # IMPORTANT: install the handler INSIDE on_server_ready, not
    # here. In Python 3.14, `asyncio.get_event_loop()` called
    # outside a running loop returns the policy default (or
    # auto-creates) — but `server.start()` then runs wslink on a
    # different loop, so the handler attaches to an orphan loop
    # and never fires. on_server_ready runs after the trame
    # server is up, on the loop wslink actually uses.
    _CLIENT_GONE_EXC_NAMES = frozenset({
        "ClientConnectionResetError",
        "ConnectionResetError",
        "ConnectionAbortedError",
        "BrokenPipeError",
    })

    def _quiet_disconnect_handler(loop, context):
        exc = context.get("exception")
        if (exc is not None
                and exc.__class__.__name__
                in _CLIENT_GONE_EXC_NAMES):
            return  # swallow — browser is gone, nothing to do
        loop.default_exception_handler(context)

    @ctrl.on_server_ready.add
    def _install_quiet_handler(**_kw):
        try:
            asyncio.get_running_loop().set_exception_handler(
                _quiet_disconnect_handler,
            )
        except Exception:
            pass

    # ---- WS lifecycle telemetry --------------------------------
    # Logs connect/exit timestamps + a 60s "still alive" heartbeat
    # showing the active session count, so silent disconnects
    # (heartbeat / network drop, with no browser beforeunload)
    # show up as the session count dropping mid-run instead of
    # leaking unobserved.
    #
    # Limitations: trame's life_cycle_update fires these hooks
    # with no arguments (see trame_server/protocol.py), so we
    # can't tag individual clients. The count is server-wide.
    # `on_client_exited` is tied to the browser `beforeunload`
    # event, so it only fires on user-initiated close — NOT on
    # a heartbeat timeout or network drop. If `connected` count
    # goes up without a matching `exited`, and the heartbeat
    # tick later shows the count dropped, the disconnect was
    # silent.
    import time as _time
    _ws_state = {"active": 0, "total_connects": 0}

    @ctrl.on_client_connected.add
    def _ws_connect_log(*_a, **_kw):
        _ws_state["active"] += 1
        _ws_state["total_connects"] += 1
        ts = _time.strftime("%H:%M:%S")
        print(
            f"[ws] {ts} CONNECT  active={_ws_state['active']} "
            f"total_connects={_ws_state['total_connects']}",
            flush=True,
        )

    @ctrl.on_client_exited.add
    def _ws_exit_log(*_a, **_kw):
        _ws_state["active"] = max(0, _ws_state["active"] - 1)
        ts = _time.strftime("%H:%M:%S")
        print(
            f"[ws] {ts} EXIT     active={_ws_state['active']} "
            f"(beforeunload — user closed tab)",
            flush=True,
        )

    # Enable wslink's INFO-level logger so we see the actual
    # server-side WS lifecycle (different from trame's beforeunload-
    # tied on_client_exited). wslink prints "client X connected"
    # and "client X disconnected" — the latter fires whenever the
    # `async for msg in current_ws` loop exits, regardless of
    # reason (heartbeat timeout, browser close, network drop).
    # That's the layer we've been missing visibility into.
    import logging as _logging
    _wslink_logger = _logging.getLogger("wslink")
    _wslink_logger.setLevel(_logging.INFO)
    if not _wslink_logger.handlers:
        _wsh = _logging.StreamHandler()
        _wsh.setFormatter(
            _logging.Formatter("[wslink] %(asctime)s %(message)s",
                                 datefmt="%H:%M:%S"),
        )
        _wslink_logger.addHandler(_wsh)

    # 60s "still alive" tick. The trame `on_client_exited` is
    # tied to browser beforeunload, so a heartbeat/network drop
    # leaves active=1 even when nobody's there. The TICK lets
    # us see that gap until wslink's own "disconnected" line
    # above fires (which can take up to 2x WSLINK_HEART_BEAT).
    async def _ws_heartbeat_loop():
        while True:
            await asyncio.sleep(60)
            ts = _time.strftime("%H:%M:%S")
            print(
                f"[ws] {ts} TICK     active={_ws_state['active']}",
                flush=True,
            )

    @ctrl.on_server_ready.add
    def _start_ws_heartbeat(**_kw):
        try:
            asyncio.get_running_loop().create_task(
                _ws_heartbeat_loop(),
            )
        except Exception:
            pass

    # `timeout=0` disables wslink's "auto-shutdown after N seconds
    # of no clients" timer. Default is 300s — but every transient
    # WS hiccup (browser tab being slow to PONG, OS network blip,
    # long-running subprocess starving the loop) flips the client
    # count to 0 briefly, schedules a shutdown, and if anything
    # delays reconnect past 5 minutes the server stops. The user
    # then sees the trame "Connection closed" screen with no way
    # back except re-launching the script. We don't want any auto-
    # shutdown — the user closes the terminal when they're done.
    server.start(
        port=port, exec_mode="main", open_browser=True,
        timeout=0,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # F2.2 — study-bundle CLI subcommands take precedence over the
    # default server-start path. dispatch() returns an int exit
    # code when it recognised + ran a subcommand, or None to fall
    # through to the legacy `--port` server-mode.
    from golgi.cli import dispatch as _study_cli_dispatch
    rc = _study_cli_dispatch(sys.argv[1:])
    if rc is not None:
        sys.exit(int(rc))

    _ensure_initialized()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    build_app(args.port)


if __name__ == "__main__":
    main()
