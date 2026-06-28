# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Cuff parameter state defaults (shared frame + selected-electrode mirror)."""
from __future__ import annotations


def register(state, *, default_cuff: dict) -> None:
    """Seed cuff params from the DEFAULT_CUFF dict.

    `default_cuff` is the module-level DEFAULT_CUFF mapping in
    golgi.py — passed by build_app to avoid this module having
    to reach back into golgi for it."""
    for _k, _v in default_cuff.items():
        state[_k] = _v
