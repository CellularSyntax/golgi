# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Welcome view — full-viewport overlay shown when
view_mode === 'welcome'. Hosts: top-right user avatar +
dropdown, hero (logo + Create / Sign-in CTA + Documentation
link), greeting, Projects tile grid, version footer.

Painted on top of the always-mounted plotter_ui inside VMain —
the welcome <div> is z-index:10 so the WebGL view stays mounted
underneath but invisible. Call inside the
`with v3.VContainer(...)` directly under VMain."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    logo_text_url: str,
    ext_site_url: str,
    ext_link_url: str,
    login_icon_url: str,
    do_open_profile_dialog: Callable,
    do_logout: Callable,
    do_show_new_project_dialog: Callable,
    do_open_auth_dialog: Callable,
) -> None:
    """Build the welcome overlay inside the surrounding VMain /
    VContainer."""
    with html.Div(
        v_show=("view_mode === 'welcome'",),
        # Vue class-binding: when the user is logged out the
        # Projects section is hidden, leaving only the hero —
        # flip the flex layout to `justify-content: center` via
        # the modifier so the logo + Sign-in CTA sit in the
        # middle of the screen instead of pinned to the top.
        classes=(
            "['golgi-welcome', "
            "{'is-signed-out': !authenticated}]",
        ),
    ):
        # ----- Top-right user chip + dropdown -----
        # Mirrors the navbar avatar+VMenu used in the workspace,
        # but with a direct logout (no project to warn about).
        # v_if (not v_show) so this button + its child VMenu
        # unmount entirely when we're in workspace mode — both
        # menus share the `show_user_menu` v_model, so leaving
        # both mounted causes a double-open.
        with html.Button(
            type="button",
            classes="golgi-welcome-userchip",
            v_if=(
                "authenticated && "
                "view_mode === 'welcome'",
            ),
            title=(
                "current_user_username "
                "|| current_user_email "
                "|| 'User'",
            ),
        ):
            html.Span(
                "{{ current_user_username "
                "|| current_user_email "
                "|| 'user' }}",
                classes=(
                    "golgi-welcome-userchip-name"
                ),
            )
            html.Img(
                src=("current_user_avatar",),
                v_show=("current_user_avatar",),
                alt="user avatar",
                classes=(
                    "golgi-welcome-userchip-img"
                ),
            )
            with v3.VMenu(
                v_model=("show_user_menu",),
                activator="parent",
                close_on_content_click=False,
                location="bottom end",
                offset="8",
            ):
                with html.Div(
                    classes=(
                        "golgi-user-menu-card"
                    ),
                ):
                    with html.Div(
                        classes=(
                            "golgi-user-menu-header"
                        ),
                    ):
                        html.Img(
                            src=(
                                "current_user_avatar",
                            ),
                            v_show=(
                                "current_user_avatar",
                            ),
                            alt="",
                            classes=(
                                "golgi-user-menu-avatar-img"
                            ),
                        )
                        html.Div(
                            "{{ "
                            "((current_user_first_name"
                            " || '') + ' ' + "
                            "(current_user_last_name "
                            "|| '')).trim() "
                            "|| current_user_username "
                            "|| current_user_email }}",
                            classes=(
                                "golgi-user-menu-name"
                            ),
                        )
                        html.Div(
                            "{{ current_user_email }}",
                            classes=(
                                "golgi-user-menu-email"
                            ),
                        )
                    with html.Div(
                        classes=(
                            "golgi-user-menu-items"
                        ),
                    ):
                        with html.Button(
                            type="button",
                            classes=(
                                "golgi-user-menu-item"
                            ),
                            click=do_open_profile_dialog,
                        ):
                            v3.VIcon(
                                "mdi-account-cog",
                                size="20",
                                classes=(
                                    "golgi-user-menu-icon"
                                ),
                            )
                            html.Span("Profile Settings")
                    with html.Div(
                        classes=(
                            "golgi-user-menu-footer"
                        ),
                    ):
                        # Direct logout — no project to warn
                        # about on the welcome view.
                        html.Button(
                            "Sign out",
                            type="button",
                            classes=(
                                "golgi-user-menu-signout"
                            ),
                            click=do_logout,
                        )

        # ----- Hero: wordmark logo + Create/Sign-in + Docs -----
        with html.Div(classes="golgi-welcome-hero"):
            html.Img(
                src=logo_text_url,
                alt="GOLGI",
                classes="golgi-welcome-logo",
            )
            with html.Div(
                classes="golgi-welcome-actions",
            ):
                # AUTHENTICATED → "Create new project" CTA.
                with html.Button(
                    type="button",
                    classes="golgi-cta-wrapper",
                    v_show=("authenticated",),
                    click=do_show_new_project_dialog,
                ):
                    html.Span(
                        classes="golgi-cta-spinner",
                    )
                    with html.Span(
                        classes="golgi-cta-inner",
                    ):
                        html.Span("Create new project")
                        html.Img(
                            src=ext_site_url,
                            alt="",
                            classes="golgi-btn-icon",
                        )
                # NOT AUTHENTICATED → "Sign in" CTA in the
                # same slot — same visual treatment so the
                # landing page layout doesn't reflow on
                # login/logout.
                with html.Button(
                    type="button",
                    classes="golgi-cta-wrapper",
                    v_show=("!authenticated",),
                    click=do_open_auth_dialog,
                ):
                    html.Span(
                        classes="golgi-cta-spinner",
                    )
                    with html.Span(
                        classes="golgi-cta-inner",
                    ):
                        html.Span("Sign in")
                        html.Img(
                            src=login_icon_url,
                            alt="",
                            classes="golgi-btn-icon",
                        )
                # Anchor (not <button>) so the browser handles
                # the new-tab navigation + middle-click natively.
                with html.A(
                    href="https://github.com/CellularSyntax/golgi/wiki",
                    target="_blank",
                    rel="noopener noreferrer",
                    classes="golgi-btn-secondary",
                ):
                    html.Span("Documentation")
                    html.Img(
                        src=ext_link_url,
                        alt="",
                        classes="golgi-btn-icon",
                    )

        # ----- Personal greeting -----
        with html.Div(
            v_show=("authenticated",),
            classes="golgi-welcome-greeting",
        ):
            html.Span("Welcome back, ")
            html.Span(
                "{{ "
                "current_user_first_name "
                "|| current_user_username "
                "|| 'friend' }}",
                classes=(
                    "golgi-welcome-greeting-name"
                ),
            )
            html.Span("!")

        # ----- Projects section (hidden when logged-out) -----
        with html.Div(
            classes="golgi-welcome-section",
            v_show=("authenticated",),
        ):
            with html.Div(
                classes="golgi-welcome-section-head",
            ):
                html.Span(
                    "Projects",
                    classes=(
                        "golgi-welcome-section-title"
                    ),
                )
                html.Span(
                    classes=(
                        "golgi-welcome-section-rule"
                    ),
                )
            with html.Div(
                classes="golgi-welcome-grid",
            ):
                # v-for over the projects_list. Click stages
                # the selected project + opens the details
                # lightbox. NOTE: pass click as a bare string
                # (a Vue expression), not as a tuple — the
                # tuple form is reserved for (callable, args).
                with html.Div(
                    classes="golgi-welcome-tile",
                    v_for="project in projects_list",
                    key="project.dir",
                    click=(
                        "detail_project = project; "
                        "detail_dialog_source = 'tile'; "
                        "show_detail_dialog = true"
                    ),
                ):
                    with html.Div(
                        classes=(
                            "golgi-welcome-tile-thumb"
                        ),
                    ):
                        html.Img(
                            src=(
                                "project.thumbnail_data_uri",
                            ),
                            v_show=(
                                "project.thumbnail_data_uri",
                            ),
                            alt=("project.name",),
                        )
                        # Placeholder when no thumbnail yet.
                        html.Div(
                            "◇",
                            v_show=(
                                "!project.thumbnail_data_uri",
                            ),
                        )
                    with html.Div(
                        classes=(
                            "golgi-welcome-tile-body"
                        ),
                    ):
                        html.Div(
                            "{{ project.name }}",
                            classes=(
                                "golgi-welcome-tile-name"
                            ),
                        )
                        html.Div(
                            "saved "
                            "{{ project.last_modified_short }}"
                            " · {{ project.size_short }}",
                            classes=(
                                "golgi-welcome-tile-mod"
                            ),
                            v_show=(
                                "project.last_modified_short",
                            ),
                        )
                        # Stage badges (mesh / fem / fibers).
                        with html.Div(
                            classes=(
                                "golgi-welcome-tile-stages"
                            ),
                            v_show=(
                                "project.labels && "
                                "project.labels.length",
                            ),
                        ):
                            html.Span(
                                "{{ label }}",
                                classes=(
                                    "golgi-welcome-tile-stage"
                                ),
                                v_for=(
                                    "label in project.labels"
                                ),
                                key="label",
                            )
            html.Div(
                classes="golgi-welcome-section-foot",
            )

        # ----- Version footer -----
        html.Div(
            "Version 1.0.0 · "
            "Copyright 2026 Medical "
            "University of Vienna",
            classes="golgi-welcome-version",
        )
