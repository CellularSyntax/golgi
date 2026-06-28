# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""F4.1 — `NullScene`: a Scene that satisfies the public API
with no-op rendering, for the headless `golgi.Study` path.

The real `Scene` (in `golgi/scene/renderer.py`) owns a PyVista
plotter, mounts actors per-frame, and pushes renders through the
trame WebSocket. None of that makes sense in a headless script
or notebook — but the pipeline drivers still reach for
`ctx.scene.request_render()`, `ctx.scene.state_dict[...]`, etc.,
so the drivers need *something* with the right shape behind that
attribute.

`NullScene` is that shape. Every mutator is a no-op; the
`state_dict` is initialised with the same nested skeleton the
real Scene uses, so pipeline writes like
`scene.state_dict["regions"][tag] = group` don't crash on missing
keys. `mkgrp()` returns the same dict shape the real Scene's
`mkgrp` does so the pipeline keeps producing valid groups (even
if no one ever renders them).

Used by `golgi.api.Study` and any other headless caller. Not
used by `build_app`; the GUI path keeps the real Scene.
"""
from __future__ import annotations

from typing import Iterable


class NullScene:
    """Drop-in replacement for `scene.Scene` in headless mode."""

    def __init__(
        self,
        *,
        region_tags: Iterable[int] = (1, 2, 3, 4, 5, 6, 7),
        max_fiber_branches: int = 6,
    ):
        # Same nested structure the real Scene maintains so
        # pipeline drivers can read/write without KeyError. None
        # of these payloads ever render; the dict is just a
        # legal target for the writes.
        self._sig: int = 1
        self.state_dict: dict = {
            "nerve": self.mkgrp(),
            "regions": {
                int(tag): self.mkgrp() for tag in region_tags
            },
            "fibers": {
                "mode": "off",
                "branches": {
                    i: self.mkgrp()
                    for i in range(int(max_fiber_branches))
                },
                "ve": self.mkgrp(),
                "pop_types": {},
                "selected": {},
            },
            "field": {
                "tubes": self.mkgrp(),
                "arrows": self.mkgrp(),
            },
            "electrodes": {},
        }

    # ----- Group factory -------------------------------------------------

    def mkgrp(
        self,
        *,
        payload=None,
        style: dict | None = None,
        visible: bool = True,
        signature: int = 0,
    ) -> dict:
        """Same shape the real Scene returns: ready to drop into
        `state_dict[...]`. `signature` is auto-bumped when omitted
        so the headless writes still produce monotonically
        increasing sigs (the real renderer remounts on sig
        change; headless never remounts, so this is purely
        cosmetic for log inspection)."""
        if signature == 0:
            self._sig += 1
            signature = self._sig
        return {
            "payload": payload,
            "style": dict(style or {}),
            "visible": bool(visible),
            "signature": int(signature),
        }

    # ----- No-op renderer surface ---------------------------------------

    def apply_group(self, name: str, g: dict) -> None:  # noqa: ARG002
        """Real Scene: mount/update an actor named `name`. Here:
        no-op. Pipeline writes still land in `state_dict` so a
        post-run inspector can see what would have rendered."""
        return None

    def retire_unknown(
        self, known_names: set,                          # noqa: ARG002
    ) -> None:
        """Real Scene: remove actors no longer claimed. Here:
        no-op."""
        return None

    def set_actor_visible(self, actor, vis: bool) -> None:  # noqa: ARG002
        """Real Scene: flip an actor's visibility. Here: no-op."""
        return None

    def render_scene(self) -> None:
        """Real Scene: rebuild + push a render frame. Here:
        no-op. Pipeline drivers call this at end-of-stage; the
        no-op keeps them happy without spinning up VTK."""
        return None

    def request_render(self) -> None:
        """Real Scene: schedule a coalesced render on the next
        tick. Here: no-op (no event loop, no tick)."""
        return None
