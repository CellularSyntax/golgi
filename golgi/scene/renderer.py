# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""3D scene renderer — PyVista plotter factory + Scene class.

Scene owns:
  pl              — the off-screen pyvista.Plotter
  state_dict      — declarative target ("what should be visible"),
                    keyed by actor group name
  rendered_sigs   — what's actually mounted right now, keyed by name
  main_loop_ref   — captured asyncio main loop (single mutable holder
                    so external code can also read/write into it)

Workflow:
  Actor builders mutate state_dict.
  request_render() coalesces N updates per tick onto the main loop.
  render_scene() mounts new actors (apply_group), updates visibilities,
    retires actors no longer claimed by state_dict.

Step 3.2 of migration.md: build_app's old in-closure renderer
(`_scene_state`, `_apply_group`, `_render_scene`,
`_request_render`, …) becomes a thin set of name aliases over
the Scene instance, leaving every call site unchanged.
"""
from __future__ import annotations

from typing import Callable

import pyvista as pv

BG_COLOR = "#ffffff"


def build_plotter(bg_color: str = BG_COLOR) -> pv.Plotter:
    """Off-screen pyvista plotter with cinematic-cool three-point
    lighting. Returned plotter is ready for add_mesh / remove_actor
    / render calls."""
    pl = pv.Plotter(off_screen=True, window_size=(1400, 900),
                    lighting="none")
    pl.background_color = bg_color
    pl.enable_anti_aliasing("ssaa")

    # Cinematic Cool lighting — dramatic three-point with a
    # strong cool key, a warm rim for separation, and a small
    # neutral fill so deep shadows stay legible. Designed to
    # accent geometric detail (nerve branching, cuff contacts)
    # via shape-from-shading at the cost of some colour
    # neutrality in the shadow side. Tuned against the white
    # workspace background.
    #
    # Light placement convention (data is centred around the
    # origin; nerve typically aligned along the z-axis):
    #   key  → front-upper-left   (-x, -y, +z)
    #   rim  → behind-upper-right (+x, +y, +z)
    #   fill → low front          ( 0, -y, -z)
    #
    # Colour temperature → RGB approximations (D65-ish for cool,
    # tungsten-ish for warm; values picked by eye so the cuff
    # silicone reads as off-white not pink/cyan):
    #   cool key  ≈ 5800 K   (0.92, 0.95, 1.00)
    #   warm rim  ≈ 3200 K   (1.00, 0.75, 0.50)
    #   fill      ≈ neutral  (0.95, 0.95, 0.95)
    pl.add_light(pv.Light(
        position=(-3.0, -2.5, 4.0), focal_point=(0, 0, 0),
        color=(0.92, 0.95, 1.00), intensity=1.00,
        light_type="scenelight",
    ))
    pl.add_light(pv.Light(
        position=(2.5, 2.5, 3.0), focal_point=(0, 0, 0),
        color=(1.00, 0.75, 0.50), intensity=0.70,
        light_type="scenelight",
    ))
    pl.add_light(pv.Light(
        position=(0.0, -1.8, -1.5), focal_point=(0, 0, 0),
        color=(0.95, 0.95, 0.95), intensity=0.15,
        light_type="scenelight",
    ))
    return pl


class Scene:
    """Owner of the 3D viewport actor lifecycle.

    One instance per build_app() call. The cuff-designer dialog
    uses its own pyvista.Plotter built via `build_plotter()`
    directly (not a second Scene) — the dialog doesn't need the
    declarative state_dict, just an offscreen render target.

    A "group" dict carries:
      payload:   the polydata to render, or None for "no actor"
      style:     add_mesh kwargs (color/scalar/cmap/clim/opacity/...)
      visible:   effective visibility
      signature: int that bumps when payload or style changes; the
                 renderer remounts only when the cached sig differs
                 (cheap visibility flips otherwise).
    """

    def __init__(
        self,
        *,
        geom,
        state,
        ctrl,
        loop_factory: Callable,
        region_tags,
        max_fiber_branches: int,
        bg_color: str = BG_COLOR,
    ):
        self.geom = geom
        self.state = state
        self.ctrl = ctrl
        self._loop_factory = loop_factory
        self.pl = build_plotter(bg_color=bg_color)

        self._sig_counter: dict = {"value": 1}
        self._render_pending: dict = {"value": False}

        # Mutable holder so external code (e.g. async helpers that
        # capture the loop themselves) can read/write the same slot
        # the renderer sees. Exposed as the `main_loop_ref` attr;
        # build_app aliases it back as `_main_loop_ref`.
        self.main_loop_ref: dict = {"loop": None}

        self.state_dict: dict = {
            "nerve": self.mkgrp(),
            "regions": {tag: self.mkgrp() for tag in region_tags},
            # mode = "palette" | "ve" | "population" | "off".
            # render_scene() mounts under disjoint actor names so
            # the modes never share IDs.
            "fibers": {
                "mode": "off",
                "branches": {i: self.mkgrp()
                             for i in range(max_fiber_branches)},
                "ve": self.mkgrp(),
                # Population-mode: dict of model-name → group,
                # built by the fiber actor builder when
                # mode == "population". Each entry mounts under
                # `fiber_pop_<model>`.
                "pop_types": {},
                # Per-fiber highlight tubes. Map of fiber-index →
                # group. Each entry mounts under
                # `fiber_selected_<idx>` so the renderer can retire
                # actors that drop out of selection.
                "selected": {},
            },
            "field": {"tubes": self.mkgrp(), "arrows": self.mkgrp()},
            # `electrodes` is rebuilt every pass keyed by eid → dict
            # of sub-groups (silicone / saline / contacts / halo /
            # designer). The renderer also retires actors whose eid
            # is no longer present.
            "electrodes": {},
        }
        self.rendered_sigs: dict[str, int] = {}

        # Set by the caller once the rebuild-state function is
        # defined (the actor builders / state-folder live further
        # down in build_app and need to be defined before this can
        # be bound). Called once per request_render dispatch,
        # before render_scene.
        self.rebuild_callback: Callable[[], None] = lambda: None

    # ------------------------------------------------------------------
    # Group factory + signature counter
    # ------------------------------------------------------------------

    def mkgrp(
        self, payload=None, style=None, visible=True, signature=0,
    ) -> dict:
        return {
            "payload": payload,
            "style": dict(style) if style else {},
            "visible": bool(visible),
            "signature": int(signature),
        }

    def next_sig(self) -> int:
        self._sig_counter["value"] += 1
        return self._sig_counter["value"]

    # ------------------------------------------------------------------
    # Actor lifecycle
    # ------------------------------------------------------------------

    def apply_group(self, name: str, g: dict) -> None:
        payload = g.get("payload")
        sig = int(g.get("signature", 0))
        cached_sig = self.rendered_sigs.get(name)
        if payload is None:
            if cached_sig is not None:
                try:
                    self.pl.remove_actor(name, reset_camera=False)
                except Exception:
                    pass
                self.rendered_sigs.pop(name, None)
            return
        if cached_sig != sig:
            # `pl.add_mesh(name=name)` atomically replaces a
            # same-named actor; the explicit `remove_actor`
            # round-trip we used to do triggers the documented
            # trame eviction race (the source of the historical
            # "two nerves" bug). Skip it.
            style = dict(g.get("style") or {})
            try:
                self.pl.add_mesh(payload, name=name, **style)
            except Exception as ex:
                print(f"[scene] add_mesh {name} failed: {ex}",
                      flush=True)
                return
            self.rendered_sigs[name] = sig
        # Belt-and-suspenders visibility — bare property writes do
        # not always propagate through trame's vue3 serializer; we
        # use the same helper everywhere so visibility changes are
        # uniformly applied.
        actor = (self.pl.actors.get(name)
                  if hasattr(self.pl, "actors") else None)
        v_int = 1 if g.get("visible", True) else 0
        if actor is not None:
            try:
                actor.SetVisibility(v_int)
            except Exception:
                pass
            try:
                actor.visibility = bool(v_int)
            except Exception:
                pass

    def retire_unknown(self, known_names: set) -> None:
        """Drop any actor that the scene state no longer owns.
        Names currently rendered are tracked in rendered_sigs;
        anything in there but not in `known_names` gets removed."""
        for nm in list(self.rendered_sigs.keys()):
            if nm in known_names:
                continue
            try:
                self.pl.remove_actor(nm, reset_camera=False)
            except Exception:
                pass
            self.rendered_sigs.pop(nm, None)

    def set_actor_visible(self, actor, vis: bool) -> None:
        """Belt-and-suspenders visibility setter. Some pyvista
        versions ship `pl.actors[name]` as a wrapper whose
        `.visibility` property doesn't always propagate through
        trame's serializer; falling back to the underlying
        `SetVisibility(int)` covers that case."""
        if actor is None:
            return
        v_int = 1 if vis else 0
        try:
            actor.SetVisibility(v_int)
        except Exception:
            pass
        try:
            actor.visibility = bool(vis)
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Scene pass + render dispatch
    # ------------------------------------------------------------------

    def render_scene(self) -> None:
        """ONLY function that mutates `pl` after init. MUST run on
        the main asyncio loop / main thread."""
        known: set = set()
        sd = self.state_dict

        # nerve
        self.apply_group("nerve", sd["nerve"])
        if sd["nerve"].get("payload") is not None:
            known.add("nerve")

        # regions
        for tag, g in sd["regions"].items():
            nm = f"region_{tag}"
            self.apply_group(nm, g)
            if g.get("payload") is not None:
                known.add(nm)

        # fibers (mode-aware actor namespace)
        fmode = sd["fibers"]["mode"]
        pop_types_groups = sd["fibers"].get("pop_types", {})
        if fmode == "palette":
            for i, g in sd["fibers"]["branches"].items():
                nm = f"fiber_branch_{i}"
                self.apply_group(nm, g)
                if g.get("payload") is not None:
                    known.add(nm)
            self.apply_group("fiber_ve", self.mkgrp())
            for tname in list(self.rendered_sigs.keys()):
                if tname.startswith("fiber_pop_"):
                    self.apply_group(tname, self.mkgrp())
        elif fmode == "ve":
            for i in sd["fibers"]["branches"]:
                self.apply_group(f"fiber_branch_{i}", self.mkgrp())
            self.apply_group("fiber_ve", sd["fibers"]["ve"])
            if sd["fibers"]["ve"].get("payload") is not None:
                known.add("fiber_ve")
            for tname in list(self.rendered_sigs.keys()):
                if tname.startswith("fiber_pop_"):
                    self.apply_group(tname, self.mkgrp())
        elif fmode == "population":
            # Population mode: one actor per fiber TYPE,
            # mounted under `fiber_pop_<safe-type-name>`. Retire
            # palette branches + the ve actor.
            for i in sd["fibers"]["branches"]:
                self.apply_group(f"fiber_branch_{i}", self.mkgrp())
            self.apply_group("fiber_ve", self.mkgrp())
            for tname, g in pop_types_groups.items():
                # Actor names must be filesystem-safe-ish (no
                # spaces). Type strings are model identifiers
                # like "MRG_INTERPOLATION" so we just prefix.
                nm = f"fiber_pop_{tname}"
                self.apply_group(nm, g)
                if g.get("payload") is not None:
                    known.add(nm)
            # Retire pop actors that disappeared from the current
            # type set (e.g., user removed a type).
            current = {f"fiber_pop_{t}" for t in pop_types_groups}
            for tname in list(self.rendered_sigs.keys()):
                if (tname.startswith("fiber_pop_")
                        and tname not in current):
                    self.apply_group(tname, self.mkgrp())
        else:  # "off"
            for i in sd["fibers"]["branches"]:
                self.apply_group(f"fiber_branch_{i}", self.mkgrp())
            self.apply_group("fiber_ve", self.mkgrp())
            for tname in list(self.rendered_sigs.keys()):
                if tname.startswith("fiber_pop_"):
                    self.apply_group(tname, self.mkgrp())

        # Single-fiber multi-highlight — one actor per selected
        # fiber, mounted under `fiber_selected_<idx>`. The dict
        # `sd["fibers"]["selected"]` is keyed by fiber index →
        # group. Disjoint actor namespace so the same slot works
        # regardless of palette / ve / population mode.
        sel_groups = sd["fibers"]["selected"]
        if not isinstance(sel_groups, dict):
            sel_groups = {}
        current_sel = {f"fiber_selected_{i}" for i in sel_groups}
        for i, g in sel_groups.items():
            nm = f"fiber_selected_{i}"
            self.apply_group(nm, g)
            if g.get("payload") is not None:
                known.add(nm)
        # Retire highlight actors that fell out of the selection
        # (user closed a chip in the combobox).
        for tname in list(self.rendered_sigs.keys()):
            if (tname.startswith("fiber_selected_")
                    and tname not in current_sel):
                self.apply_group(tname, self.mkgrp())

        # field lines + arrows
        self.apply_group("field_lines", sd["field"]["tubes"])
        self.apply_group("field_lines_arrows", sd["field"]["arrows"])
        if sd["field"]["tubes"].get("payload") is not None:
            known.add("field_lines")
        if sd["field"]["arrows"].get("payload") is not None:
            known.add("field_lines_arrows")

        # per-electrode
        for eid, parts in sd["electrodes"].items():
            for part_name, g in parts.items():
                if isinstance(g, dict) and "payload" in g:
                    nm = f"{eid}_{part_name}"
                    self.apply_group(nm, g)
                    if g.get("payload") is not None:
                        known.add(nm)
                elif isinstance(g, dict):
                    # nested dict for sub-indexed parts (contacts
                    # / designer parts) — keys are int indices or
                    # role-suffix strings, values are groups.
                    for sub_key, sub_g in g.items():
                        nm = f"{eid}_{part_name}_{sub_key}"
                        self.apply_group(nm, sub_g)
                        if sub_g.get("payload") is not None:
                            known.add(nm)

        # Retire any actor that's still in the renderer but no
        # longer claimed by the scene state. This is what kills
        # the phantom-cuff / phantom-nerve "stale actor never
        # removed because no one explicitly named it" path.
        self.retire_unknown(known)

        # One-shot camera fit. Set by `do_load_geometry` (fresh
        # source loaded) or any caller that wants the next frame
        # to refit. By doing the reset AFTER mounting, the
        # bounding box reflects the actual scene — fixes the
        # "camera was set to empty bounds because actors hadn't
        # landed yet" race the old per-handler `pl.reset_camera()`
        # had.
        if getattr(self.geom, "_needs_camera_reset", False):
            try:
                self.pl.reset_camera()
            except Exception:
                pass
            self.geom._needs_camera_reset = False
            try:
                self.ctrl.view_push_camera()
            except Exception:
                pass

        try:
            self.pl.render()
        except Exception:
            pass
        try:
            self.ctrl.view_update()
        except Exception:
            pass

    def request_render(self) -> None:
        """Coalesce many state mutations into one render pass per
        tick. Always lands on the main asyncio loop, so it's safe
        to call from executor threads too. Strict rule: `pl.*` is
        mutated ONLY from the captured main loop's thread."""
        # Capture the main loop on the first main-thread call.
        # `asyncio.get_running_loop()` raises if there's no loop
        # in the current thread; the first watcher / coroutine
        # invocation comes from trame's main loop and seeds this.
        if self.main_loop_ref["loop"] is None:
            try:
                self.main_loop_ref["loop"] = self._loop_factory()
            except RuntimeError:
                pass
        if self._render_pending["value"]:
            return
        loop = self.main_loop_ref["loop"]

        def _run():
            self._render_pending["value"] = False
            try:
                self.rebuild_callback()
                self.render_scene()
            except Exception as ex:
                print(f"[scene] render pass failed: {ex}",
                      flush=True)

        if loop is not None and loop.is_running():
            self._render_pending["value"] = True
            loop.call_soon_threadsafe(_run)
            return
        # No captured main loop yet — only happens during the
        # very first synchronous call sites before trame starts
        # serving. Safe to run inline on the main thread; never
        # reached from executor threads in practice because the
        # executor work is always scheduled AFTER the main loop
        # is running (and therefore captured).
        self._render_pending["value"] = True
        _run()
