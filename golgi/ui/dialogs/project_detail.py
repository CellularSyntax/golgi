# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Project-detail lightbox — the largest dialog in the app.

Two-column card: thumbnail | metadata. Embeds the activity log,
the rename inline editor, the labels CRUD UI, and the share-
with-users autocomplete. ~700 LOC verbatim from build_app.
"""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    edit_icon_url: str,
    do_close_detail_dialog: Callable,
    do_start_edit_name: Callable,
    do_cancel_edit_name: Callable,
    do_save_edit_name: Callable,
    do_start_add_label: Callable,
    do_cancel_add_label: Callable,
    do_save_add_label: Callable,
    do_open_from_detail: Callable,
    do_request_delete_from_detail: Callable,
    do_toggle_activity_payload: Callable,
    do_save_shared_users: Callable,
    do_export_study: Callable | None = None,
) -> None:
    """Build the project-detail VDialog inside the current
    VAppLayout context. `edit_icon_url` is _EDIT_ICON_URL from
    golgi.py (the path to the pencil SVG asset)."""
    _EDIT_ICON_URL = edit_icon_url  # noqa: N806 (legacy name)
    # ----- Project-detail lightbox -----
    # Two-column card: thumbnail | metadata. Open + Delete
    # actions sit in the top-right corner so the most-likely
    # next action is right where Fitts' law wants it.
    with v3.VDialog(
        v_model=("show_detail_dialog",),
        max_width=880,
    ):
        with v3.VCard(classes="golgi-detail-card"):
            # Title row — project name + rename pencil + Open
            # & Delete actions (tile-entry only) + the close
            # X (always). All on a single line. Open + Delete
            # are nonsensical when the dialog was opened from
            # the navbar (the project is already open), so
            # they're v_show=false there.
            with html.Div(classes="golgi-detail-header"):
                # Display mode
                html.Span(
                    "{{ detail_project ? detail_project.name "
                    ": '' }}",
                    v_show=("!edit_name_mode",),
                    classes="golgi-detail-title",
                )
                with html.Button(
                    type="button",
                    v_show=("!edit_name_mode",),
                    classes="golgi-detail-edit-btn",
                    title="Rename project",
                    click=do_start_edit_name,
                ):
                    html.Img(
                        src=_EDIT_ICON_URL,
                        alt="",
                        classes="golgi-detail-edit-icon",
                    )
                # Open + Delete — inline in the title row,
                # right after the pencil. Wrapped so we can
                # margin-left: auto the wrapper to push the
                # action group to the right of the title.
                with html.Div(
                    classes="golgi-detail-actions-inline",
                    v_show=(
                        "detail_dialog_source !== 'navbar'"
                        " && !edit_name_mode",
                    ),
                ):
                    html.Button(
                        "Open",
                        type="button",
                        classes=(
                            "golgi-btn-primary golgi-btn-sm"
                        ),
                        click=do_open_from_detail,
                    )
                    html.Button(
                        "Delete",
                        type="button",
                        classes=(
                            "golgi-btn-secondary "
                            "golgi-btn-sm"
                        ),
                        click=do_request_delete_from_detail,
                    )
                    # F2.2 — Export study. Visible only when the
                    # dialog was opened on an existing project
                    # (matches Open + Delete visibility). The
                    # button passes detail_project.dir so the
                    # handler exports whichever project the
                    # dialog is showing — open OR closed. The
                    # busy spinner + download anchor live in the
                    # status strip below the title row.
                    if do_export_study is not None:
                        html.Button(
                            "Export study",
                            type="button",
                            classes=(
                                "golgi-btn-secondary "
                                "golgi-btn-sm"
                            ),
                            disabled=(
                                "study_export_pending_busy",
                            ),
                            click=(
                                do_export_study,
                                "[detail_project.dir]",
                            ),
                        )
                # Close X — sits at the very right of the
                # title row. `margin-left: auto` in the CSS
                # only fires when the inline-actions group is
                # hidden (navbar entry); otherwise the
                # actions-inline wrapper carries that margin
                # so the layout reads: title • pencil •
                # [auto-gap] • Open + Delete • close-X.
                html.Button(
                    "✕",
                    type="button",
                    classes=(
                        "golgi-btn-secondary "
                        "golgi-btn-round golgi-btn-sm "
                        "golgi-detail-close-btn"
                    ),
                    title="Close",
                    click=do_close_detail_dialog,
                )

                # Edit mode — input, save, cancel
                v3.VTextField(
                    v_model=("edit_name_value",),
                    v_show=("edit_name_mode",),
                    density="compact",
                    hide_details=True,
                    autofocus=True,
                    variant="outlined",
                    keydown_enter=do_save_edit_name,
                    keydown_escape=do_cancel_edit_name,
                    style="max-width: 360px;",
                )
                html.Button(
                    "✓",
                    type="button",
                    v_show=("edit_name_mode",),
                    classes="golgi-detail-edit-confirm",
                    title="Save",
                    click=do_save_edit_name,
                )
                html.Button(
                    "✕",
                    type="button",
                    v_show=("edit_name_mode",),
                    classes="golgi-detail-edit-cancel",
                    title="Cancel",
                    click=do_cancel_edit_name,
                )
            # Tabbed body — Overview / Status / Activity.
            # `v_if=detail_project` on the wrapper guards
            # against Vue dereferencing null while the
            # lightbox closes. The Status + Activity tabs
            # are populated by `_refresh_detail_briefs` on
            # F2.2 — Study export status strip. Visible only
            # while an export is in flight OR after one has just
            # completed (the Download anchor activates once the
            # data URI is populated). Sits between the title row
            # and the tabbed body so the user sees it as soon as
            # they kick off the export from the title-row button.
            with html.Div(
                v_show=(
                    "study_export_pending_busy "
                    "|| study_export_pending_data_uri "
                    "|| study_export_pending_error",
                ),
                style=(
                    "margin: 0 16px 12px 16px; "
                    "padding: 8px 12px; "
                    "background: #f6f6f7; "
                    "border-radius: 6px; "
                    "font-size: 11px;"
                ),
            ):
                html.Div(
                    "{{ study_export_pending_status }}",
                    v_show=("study_export_pending_status",),
                    style="color: #146e3a; margin-bottom: 4px;",
                )
                html.Div(
                    "{{ study_export_pending_error }}",
                    v_show=("study_export_pending_error",),
                    style="color: #c0392b;",
                )
                with html.A(
                    href=("study_export_pending_data_uri",),
                    download=("study_export_pending_filename",),
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    style="margin-top: 6px;",
                    v_show=("study_export_pending_data_uri",),
                ):
                    html.I(
                        classes="mdi mdi-folder-zip-outline",
                        style="font-size: 16px;",
                    )
                    html.Span("Download study .zip")

            # open (closed projects render from the
            # manifest + disk-file presence; live projects
            # use the same code path so the two views stay
            # consistent).
            with html.Div(
                v_if=("detail_project",),
                classes="golgi-detail-tabs-wrapper",
            ):
                with v3.VTabs(
                    v_model=("detail_tab",),
                    density="compact",
                    slider_color="primary",
                    classes="golgi-detail-tabs",
                ):
                    v3.VTab(
                        "Overview",
                        value="overview",
                    )
                    v3.VTab(
                        "Status",
                        value="status",
                    )
                    v3.VTab(
                        "Activity",
                        value="activity",
                    )
                v3.VDivider()
                with v3.VWindow(
                    v_model=("detail_tab",),
                    classes="golgi-detail-window",
                ):
                    # ---- Overview tab ----
                    with v3.VWindowItem(value="overview"):
                        with html.Div(
                            classes="golgi-detail-body",
                        ):
                            with html.Div(classes="golgi-detail-thumb"):
                                html.Img(
                                    src=("detail_project.thumbnail_data_uri",),
                                    v_show=(
                                        "detail_project.thumbnail_data_uri",
                                    ),
                                    alt=("detail_project.name",),
                                )
                                html.Span(
                                    "◇",
                                    v_show=(
                                        "!detail_project.thumbnail_data_uri",
                                    ),
                                )
                            with html.Div(classes="golgi-detail-meta"):
                                html.Span(
                                    "Created",
                                    classes="golgi-detail-meta-label",
                                )
                                html.Span(
                                    "{{ detail_project.created_short "
                                    "|| '—' }}",
                                    classes="golgi-detail-meta-value",
                                )
                                html.Span(
                                    "Created by",
                                    classes="golgi-detail-meta-label",
                                )
                                with html.Div(
                                    classes="golgi-detail-userchip",
                                ):
                                    html.Img(
                                        src=(
                                            "detail_project_owner."
                                            "avatar_data_uri",
                                        ),
                                        v_show=(
                                            "detail_project_owner."
                                            "avatar_data_uri",
                                        ),
                                        alt="",
                                        classes=(
                                            "golgi-detail-userchip-img"
                                        ),
                                    )
                                    html.Span(
                                        "{{ detail_project_owner.username"
                                        " || '—' }}",
                                        classes=(
                                            "golgi-detail-userchip-name"
                                        ),
                                    )
                                html.Span(
                                    "Modified",
                                    classes="golgi-detail-meta-label",
                                )
                                # Modified value combines the date + the
                                # last-modifier user chip so the user can
                                # see WHO did the last change at a glance.
                                with html.Div(
                                    classes=(
                                        "golgi-detail-meta-modified"
                                    ),
                                ):
                                    html.Span(
                                        "{{ detail_project."
                                        "last_modified_short || '—' }}",
                                        classes=(
                                            "golgi-detail-meta-value"
                                        ),
                                    )
                                    html.Span(
                                        "by",
                                        classes=(
                                            "golgi-detail-meta-by"
                                        ),
                                        v_show=(
                                            "detail_project_modifier."
                                            "username",
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "golgi-detail-userchip"
                                        ),
                                        v_show=(
                                            "detail_project_modifier."
                                            "username",
                                        ),
                                    ):
                                        html.Img(
                                            src=(
                                                "detail_project_modifier."
                                                "avatar_data_uri",
                                            ),
                                            v_show=(
                                                "detail_project_modifier"
                                                ".avatar_data_uri",
                                            ),
                                            alt="",
                                            classes=(
                                                "golgi-detail-userchip-img"
                                            ),
                                        )
                                        html.Span(
                                            "{{ "
                                            "detail_project_modifier."
                                            "username || '' }}",
                                            classes=(
                                                "golgi-detail-userchip-name"
                                            ),
                                        )
                                html.Span(
                                    "Shared with",
                                    classes="golgi-detail-meta-label",
                                )
                                # Share picker — VAutocomplete with chips
                                # showing avatar + name. v_model is bound
                                # to `detail_project.shared_user_ids` (so
                                # the chip set is live with the current
                                # state); update_modelValue calls the
                                # server handler to persist + refresh
                                # `projects_list` so newly-shared users
                                # see the tile on their next welcome
                                # visit. The OWNER is hidden from the
                                # items list so the picker can't add the
                                # owner to themselves.
                                with v3.VAutocomplete(
                                    model_value=(
                                        "(detail_project && "
                                        "detail_project.shared_user_ids)"
                                        " || []",
                                    ),
                                    items=(
                                        "users_list.filter(u => "
                                        "  !detail_project "
                                        "  || u.id !== "
                                        "     detail_project.owner_user_id"
                                        ")",
                                    ),
                                    item_title="username",
                                    item_value="id",
                                    label="add users …",
                                    multiple=True,
                                    chips=True,
                                    clearable=True,
                                    closable_chips=True,
                                    return_object=False,
                                    density="compact",
                                    hide_details=True,
                                    variant="outlined",
                                    classes=(
                                        "golgi-detail-share-picker"
                                    ),
                                    menu_props=(
                                        "{ maxHeight: '320px' }",
                                    ),
                                    update_modelValue=(
                                        do_save_shared_users,
                                        "[$event]",
                                    ),
                                ):
                                    # Dropdown item slot — avatar + name +
                                    # secondary email so the picker is
                                    # easy to scan.
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
                                                html.Img(
                                                    src=(
                                                        "item.raw."
                                                        "avatar_data_uri",
                                                    ),
                                                    classes=(
                                                        "golgi-detail-share-avatar"
                                                    ),
                                                )
                                    # Chip slot — avatar + name. Looked
                                    # up via Number() comparison to
                                    # bridge any string-vs-int drift in
                                    # the value.
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
                                            html.Img(
                                                src=(
                                                    "(users_list.find("
                                                    "  u => Number(u.id) "
                                                    "  === Number(item.value)"
                                                    ") || {})"
                                                    ".avatar_data_uri",
                                                ),
                                                classes=(
                                                    "golgi-detail-share-avatar"
                                                ),
                                            )
                                            html.Span(
                                                "{{ ("
                                                "  users_list.find(u => "
                                                "    Number(u.id) === "
                                                "    Number(item.value)"
                                                "  ) || {}"
                                                ").username || "
                                                "('user ' + item.value) }}"
                                            )
                                html.Span(
                                    "Size",
                                    classes="golgi-detail-meta-label",
                                )
                                html.Span(
                                    "{{ detail_project.size_short || '—' }}",
                                    classes="golgi-detail-meta-value",
                                )
                                html.Span(
                                    "Source",
                                    classes="golgi-detail-meta-label",
                                )
                                html.Span(
                                    "{{ detail_project.source_file || "
                                    "'(none bundled yet)' }}",
                                    classes=("golgi-detail-meta-value "
                                              "golgi-detail-path"),
                                )
                                html.Span(
                                    "Labels",
                                    classes="golgi-detail-meta-label",
                                )
                                # Labels row — removable chips + an "add"
                                # affordance that toggles between a dashed
                                # pill button and an inline input.
                                with html.Div(classes="golgi-detail-labels"):
                                    with html.Span(
                                        classes="golgi-detail-label-chip",
                                        v_for=(
                                            "label in detail_project.labels"
                                        ),
                                        key="label",
                                    ):
                                        html.Span("{{ label }}")
                                        html.Button(
                                            "✕",
                                            type="button",
                                            classes=(
                                                "golgi-detail-label-remove"
                                            ),
                                            title="Remove label",
                                            click=(
                                                "remove_label_request "
                                                "= label"
                                            ),
                                        )
                                    # "+ add label" pill — visible when not
                                    # in input mode.
                                    html.Button(
                                        "+ add label",
                                        type="button",
                                        v_show=("!add_label_mode",),
                                        classes="golgi-detail-label-add",
                                        click=do_start_add_label,
                                    )
                                    # Inline input — visible in add-label
                                    # mode. Enter saves, Esc cancels.
                                    with html.Span(
                                        classes="golgi-detail-label-input",
                                        v_show=("add_label_mode",),
                                    ):
                                        html.Input(
                                            type="text",
                                            v_model=("add_label_value",),
                                            placeholder="new label",
                                            keydown_enter=do_save_add_label,
                                            keydown_escape=(
                                                do_cancel_add_label
                                            ),
                                        )
                                        html.Button(
                                            "✓",
                                            type="button",
                                            classes=(
                                                "golgi-detail-edit-confirm"
                                            ),
                                            title="Save",
                                            click=do_save_add_label,
                                        )
                                        html.Button(
                                            "✕",
                                            type="button",
                                            classes=(
                                                "golgi-detail-edit-cancel"
                                            ),
                                            title="Cancel",
                                            click=do_cancel_add_label,
                                        )
                                html.Span(
                                    "Path",
                                    classes="golgi-detail-meta-label",
                                )
                                html.Span(
                                    "{{ detail_project.dir }}",
                                    classes=("golgi-detail-meta-value "
                                              "golgi-detail-path"),
                                )

                    # ---- Status tab ----
                    # 8-stage completion table. Each row =
                    # status icon + stage label + secondary
                    # details (path / count / size).
                    with v3.VWindowItem(value="status"):
                        with html.Div(
                            classes="golgi-detail-status",
                        ):
                            html.Div(
                                "No status available.",
                                v_show=(
                                    "detail_status_rows"
                                    ".length === 0",
                                ),
                                classes=(
                                    "golgi-detail-status-empty"
                                ),
                            )
                            with html.Div(
                                v_show=(
                                    "detail_status_rows"
                                    ".length > 0",
                                ),
                                classes=(
                                    "golgi-detail-status-list"
                                ),
                            ):
                                with html.Div(
                                    v_for=(
                                        "row in "
                                        "detail_status_rows",
                                    ),
                                    key="row.id",
                                    classes=(
                                        "['golgi-detail-status-row', "
                                        "row.done "
                                        "? 'is-done' "
                                        ": 'is-pending']",
                                    ),
                                ):
                                    # Status icon as a plain
                                    # text glyph — avoids any
                                    # Vuetify icon-prop
                                    # ambiguity.
                                    html.Span(
                                        "{{ row.done "
                                        "? '✓' "
                                        ": '○' }}",
                                        classes=(
                                            "['golgi-detail-status-icon', "
                                            "row.done "
                                            "? 'is-done-icon' "
                                            ": 'is-pending-icon']",
                                        ),
                                    )
                                    with html.Div(
                                        classes=(
                                            "golgi-detail-status-text"
                                        ),
                                    ):
                                        html.Span(
                                            "{{ row.label }}",
                                            classes=(
                                                "golgi-detail-status-label"
                                            ),
                                        )
                                        html.Span(
                                            "{{ row.details }}",
                                            classes=(
                                                "golgi-detail-status-details"
                                            ),
                                        )

                    # ---- Activity tab ----
                    # Audit-log scroller. Each row carries a
                    # timestamp (relative), the user chip, a
                    # pretty action name, and a status pill.
                    # The chevron toggles a JSON payload
                    # panel via a server-side handler (the
                    # equivalent inline-JS — spread + arrow
                    # function — tripped Vue's compiler).
                    with v3.VWindowItem(value="activity"):
                        with html.Div(
                            classes="golgi-detail-activity",
                        ):
                            html.Div(
                                "No activity yet.",
                                v_show=(
                                    "detail_activity_events"
                                    ".length === 0",
                                ),
                                classes=(
                                    "golgi-detail-activity-empty"
                                ),
                            )
                            with html.Div(
                                v_show=(
                                    "detail_activity_events"
                                    ".length > 0",
                                ),
                                classes=(
                                    "golgi-detail-activity-list"
                                ),
                            ):
                                with html.Div(
                                    v_for=(
                                        "evt in "
                                        "detail_activity_events",
                                    ),
                                    key="evt.id",
                                    classes=(
                                        "golgi-detail-activity-row"
                                    ),
                                ):
                                    with html.Div(
                                        classes=(
                                            "golgi-detail-activity-line"
                                        ),
                                    ):
                                        html.Span(
                                            "{{ evt.ts_relative }}",
                                            classes=(
                                                "golgi-detail-activity-time"
                                            ),
                                        )
                                        with html.Div(
                                            classes=(
                                                "golgi-detail-userchip "
                                                "golgi-detail-activity-user"
                                            ),
                                            v_show=(
                                                "evt.username",
                                            ),
                                        ):
                                            html.Img(
                                                src=(
                                                    "evt"
                                                    ".avatar_data_uri",
                                                ),
                                                v_show=(
                                                    "evt"
                                                    ".avatar_data_uri",
                                                ),
                                                alt="",
                                                classes=(
                                                    "golgi-detail-userchip-img"
                                                ),
                                            )
                                            html.Span(
                                                "{{ evt.username }}",
                                                classes=(
                                                    "golgi-detail-userchip-name"
                                                ),
                                            )
                                        html.Span(
                                            "—",
                                            v_show=(
                                                "!evt.username",
                                            ),
                                            classes=(
                                                "golgi-detail-activity-user-empty"
                                            ),
                                        )
                                        html.Span(
                                            "{{ evt.action_pretty }}",
                                            classes=(
                                                "golgi-detail-activity-action"
                                            ),
                                        )
                                        html.Span(
                                            "{{ evt.status }}",
                                            classes=(
                                                "['golgi-detail-activity-status', "
                                                "'status-' + evt.status]",
                                            ),
                                        )
                                        with html.Button(
                                            type="button",
                                            v_show=(
                                                "evt.has_payload",
                                            ),
                                            classes=(
                                                "golgi-detail-activity-chevron"
                                            ),
                                            click=(
                                                do_toggle_activity_payload,
                                                "[evt.id]",
                                            ),
                                        ):
                                            html.Span(
                                                "−",
                                                v_show=(
                                                    "detail_activity_expanded"
                                                    ".indexOf(evt.id) "
                                                    ">= 0",
                                                ),
                                            )
                                            html.Span(
                                                "+",
                                                v_show=(
                                                    "detail_activity_expanded"
                                                    ".indexOf(evt.id) "
                                                    "< 0",
                                                ),
                                            )
                                    # Payload panel — JSON
                                    # pre-formatted server-
                                    # side so we just display
                                    # the string.
                                    html.Pre(
                                        "{{ evt.payload_pretty }}",
                                        v_show=(
                                            "evt.has_payload "
                                            "&& "
                                            "detail_activity_expanded"
                                            ".indexOf(evt.id) "
                                            ">= 0",
                                        ),
                                        classes=(
                                            "golgi-detail-activity-payload"
                                        ),
                                    )

