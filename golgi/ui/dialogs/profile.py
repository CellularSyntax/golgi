# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""User profile dialog (navbar avatar dropdown → "Profile")."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_close_profile_dialog: Callable,
    do_save_profile: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_profile_dialog",),
        max_width=620,
        persistent=False,
    ):
        with v3.VCard(classes="golgi-profile-card"):
            with v3.VCardText(classes="golgi-profile-body"):
                # Title row: H3-style heading + info-icon tooltip.
                with html.Div(
                    classes=(
                        "d-flex align-center "
                        "golgi-profile-title-row"
                    ),
                ):
                    html.Div(
                        "Profile Settings",
                        classes="golgi-dialog-title",
                    )
                    with v3.VTooltip(
                        location="bottom",
                        max_width=360,
                    ):
                        with v3.Template(
                            v_slot_activator=("{ props }",),
                        ):
                            with v3.VBtn(
                                v_bind="props",
                                icon=True,
                                size="small",
                                variant="text",
                                density="compact",
                                classes="ml-1",
                            ):
                                v3.VIcon(
                                    "mdi-information-outline",
                                    size="18",
                                    color="grey-darken-1",
                                )
                        html.Span(
                            "Update your account details "
                            "and profile image. Sign-in "
                            "keeps working with either "
                            "email or username."
                        )
                # Account section.
                html.Div(
                    "Account",
                    classes="golgi-profile-section",
                )
                with html.Div(
                    classes="golgi-profile-grid",
                ):
                    v3.VTextField(
                        v_model=("profile_email",),
                        label="Email",
                        type="email",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("profile_busy",),
                    )
                    v3.VTextField(
                        v_model=("profile_username",),
                        label="Username (3–64 chars)",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("profile_busy",),
                    )
                # Personal section.
                html.Div(
                    "Personal",
                    classes="golgi-profile-section",
                )
                with html.Div(
                    classes="golgi-profile-grid",
                ):
                    v3.VTextField(
                        v_model=("profile_first_name",),
                        label="First Name (optional)",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("profile_busy",),
                    )
                    v3.VTextField(
                        v_model=("profile_last_name",),
                        label="Last Name (optional)",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("profile_busy",),
                    )
                # Affiliation section.
                html.Div(
                    "Affiliation",
                    classes="golgi-profile-section",
                )
                with html.Div(
                    classes="golgi-profile-grid",
                ):
                    v3.VAutocomplete(
                        v_model=("profile_country",),
                        items=("country_options",),
                        label="Country (optional)",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("profile_busy",),
                        menu_props=(
                            "{ maxHeight: '320px' }",
                        ),
                    )
                    v3.VTextField(
                        v_model=("profile_institution",),
                        label="Institution (optional)",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("profile_busy",),
                    )
                # Position spans the full grid width.
                v3.VAutocomplete(
                    v_model=("profile_position",),
                    items=("position_options",),
                    label="Position (optional)",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    disabled=("profile_busy",),
                    menu_props=("{ maxHeight: '320px' }",),
                    classes="golgi-profile-position",
                )
                # Image section.
                html.Div(
                    "Profile Image",
                    classes="golgi-profile-section",
                )
                with html.Div(
                    classes=(
                        "mb-2 golgi-profile-avatar-row"
                    ),
                ):
                    html.Img(
                        src=("profile_image_data_uri",),
                        v_show=(
                            "profile_image_data_uri "
                            "&& !profile_remove_image",
                        ),
                        classes="golgi-profile-avatar",
                        alt="current avatar",
                    )
                    html.Div(
                        "no image",
                        v_show=(
                            "!profile_image_data_uri "
                            "|| profile_remove_image",
                        ),
                        classes=(
                            "golgi-profile-avatar "
                            "golgi-profile-avatar-empty"
                        ),
                    )
                    with html.Div(
                        classes=(
                            "golgi-profile-avatar-controls"
                        ),
                    ):
                        v3.VFileInput(
                            v_model=("profile_image_file",),
                            label="upload (optional)",
                            density="compact",
                            hide_details=True,
                            accept=(
                                "image/png,image/jpeg,"
                                "image/webp"
                            ),
                            prepend_icon="mdi-camera",
                            disabled=("profile_busy",),
                            classes="golgi-auth-file",
                        )
                        v3.VCheckbox(
                            v_model=("profile_remove_image",),
                            label="remove current image",
                            density="compact",
                            hide_details=True,
                            disabled=("profile_busy",),
                        )
                html.Div(
                    "{{ profile_error }}",
                    v_show=("profile_error",),
                    style=(
                        "color: #e24b4a; font-size: 12px; "
                        "margin-top: 6px;"
                    ),
                )
                html.Div(
                    "{{ profile_status }}",
                    v_show=(
                        "profile_status && !profile_error",
                    ),
                    style=(
                        "color: #1a8a3a; font-size: 12px; "
                        "margin-top: 6px;"
                    ),
                )
            with v3.VCardActions(classes="px-6 pb-4"):
                v3.VSpacer()
                html.Button(
                    "Close",
                    type="button",
                    classes=(
                        "golgi-btn-secondary "
                        "golgi-btn-sm"
                    ),
                    disabled=("profile_busy",),
                    click=do_close_profile_dialog,
                )
                html.Button(
                    "Save",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    disabled=("profile_busy",),
                    click=do_save_profile,
                )
