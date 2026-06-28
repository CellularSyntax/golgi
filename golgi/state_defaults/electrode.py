# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Electrode parameter state defaults (selected-electrode mirror)."""
from __future__ import annotations


def register(state, *, default_electrode: dict) -> None:
    """Seed electrode params from the DEFAULT_ELECTRODE dict."""
    for _k, _v in default_electrode.items():
        state[_k] = _v
