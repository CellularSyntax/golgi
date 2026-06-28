# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Login / Register dialog. Single VDialog with `auth_mode`
toggling between login and register layouts."""
from __future__ import annotations

from typing import Callable

from trame.widgets import html
from trame.widgets import vuetify3 as v3


def render(
    *,
    do_close_auth_dialog: Callable,
    do_submit_login: Callable,
    do_submit_register: Callable,
    do_switch_auth_mode: Callable,
) -> None:
    with v3.VDialog(
        v_model=("show_auth_dialog",),
        max_width=("auth_mode === 'login' ? 460 : 620",),
        persistent=False,
    ):
        with v3.VCard(classes="golgi-auth-card"):
            # ============================================
            # LOGIN MODE — centred "Welcome back!" greeting,
            # two underline-styled inputs, pill Login button,
            # bottom row with Forgot / Cancel / Register links.
            # ============================================
            with html.Div(
                v_show=("auth_mode === 'login'",),
                classes="golgi-auth-login",
            ):
                html.Div(
                    "Welcome back!",
                    classes="golgi-auth-greeting",
                )
                html.Div(
                    "User Login",
                    classes="golgi-auth-subtitle",
                )
                v3.VTextField(
                    v_model=("auth_login_id",),
                    label="Email",
                    autofocus=True,
                    density="comfortable",
                    hide_details=True,
                    variant="underlined",
                    classes="golgi-auth-input",
                    disabled=("auth_busy",),
                )
                v3.VTextField(
                    v_model=("auth_password",),
                    label="Password",
                    type="password",
                    density="comfortable",
                    hide_details=True,
                    variant="underlined",
                    classes="golgi-auth-input",
                    disabled=("auth_busy",),
                    keydown_enter=(
                        "trigger('do_submit_login')"
                    ),
                )
                html.Div(
                    "{{ auth_error }}",
                    v_show=("auth_error",),
                    classes="golgi-auth-error",
                )
                with html.Div(
                    classes="golgi-auth-login-actions",
                ):
                    html.Button(
                        "Cancel",
                        type="button",
                        classes=(
                            "golgi-btn-secondary "
                            "golgi-btn-sm"
                        ),
                        disabled=("auth_busy",),
                        click=do_close_auth_dialog,
                    )
                    html.Button(
                        "Login",
                        type="button",
                        classes=(
                            "golgi-btn-primary "
                            "golgi-btn-sm"
                        ),
                        disabled=("auth_busy",),
                        click=do_submit_login,
                    )
                with html.Div(
                    classes="golgi-auth-login-links",
                ):
                    # Forgot Password — placeholder for now;
                    # not yet implemented.
                    html.Button(
                        "Forgot Password",
                        type="button",
                        classes="golgi-auth-link",
                        click=(
                            "auth_error = 'Password reset is "
                            "not available yet — contact your "
                            "admin to reset your password.'"
                        ),
                    )
                    html.Button(
                        "Register",
                        type="button",
                        classes=(
                            "golgi-auth-link "
                            "golgi-auth-link-accent"
                        ),
                        disabled=("auth_busy",),
                        click=(
                            do_switch_auth_mode,
                            "['register']",
                        ),
                    )

            # ============================================
            # REGISTER MODE — dense form layout.
            # ============================================
            with v3.VCardText(
                classes="golgi-register-body",
                v_show=("auth_mode === 'register'",),
            ):
                # Header.
                html.Div(
                    "Create Account",
                    classes="golgi-dialog-title",
                )
                html.Div(
                    "All fields are required except the "
                    "profile image.",
                    classes=(
                        "golgi-dialog-body "
                        "golgi-register-intro"
                    ),
                )

                # Account section.
                html.Div(
                    "Account",
                    classes="golgi-profile-section",
                )
                v3.VTextField(
                    v_model=("auth_email",),
                    label="Email",
                    type="email",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    classes="mb-3",
                    disabled=("auth_busy",),
                )
                v3.VTextField(
                    v_model=("auth_username",),
                    label="Username (3–64 chars)",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    classes="mb-3",
                    disabled=("auth_busy",),
                )
                with html.Div(classes="golgi-auth-row mb-1"):
                    v3.VTextField(
                        v_model=("auth_password",),
                        label="Password",
                        type="password",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("auth_busy",),
                        keydown_enter=(
                            "trigger('do_submit_register')"
                        ),
                    )
                    v3.VTextField(
                        v_model=(
                            "auth_password_confirm",
                        ),
                        label="Confirm Password",
                        type="password",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("auth_busy",),
                        keydown_enter=(
                            "trigger('do_submit_register')"
                        ),
                    )

                v3.VDivider(classes="golgi-register-divider")

                # Personal section.
                html.Div(
                    "Personal",
                    classes="golgi-profile-section",
                )
                with html.Div(classes="golgi-auth-row mb-1"):
                    v3.VTextField(
                        v_model=("auth_first_name",),
                        label="First Name",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("auth_busy",),
                    )
                    v3.VTextField(
                        v_model=("auth_last_name",),
                        label="Last Name",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("auth_busy",),
                    )

                v3.VDivider(classes="golgi-register-divider")

                # Affiliation section.
                html.Div(
                    "Affiliation",
                    classes="golgi-profile-section",
                )
                with html.Div(classes="golgi-auth-row mb-3"):
                    v3.VAutocomplete(
                        v_model=("auth_country",),
                        items=("country_options",),
                        label="Country",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("auth_busy",),
                        menu_props=(
                            "{ maxHeight: '320px' }",
                        ),
                    )
                    v3.VTextField(
                        v_model=("auth_institution",),
                        label="Institution",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        disabled=("auth_busy",),
                    )
                v3.VAutocomplete(
                    v_model=("auth_position",),
                    items=("position_options",),
                    label="Position",
                    density="compact",
                    hide_details=True,
                    variant="outlined",
                    classes="mb-1",
                    disabled=("auth_busy",),
                    menu_props=("{ maxHeight: '320px' }",),
                )

                v3.VDivider(classes="golgi-register-divider")

                # Profile Image section.
                html.Div(
                    "Profile Image",
                    classes="golgi-profile-section",
                )
                with html.Div(
                    classes="mb-1 golgi-auth-avatar-row",
                ):
                    v3.VFileInput(
                        v_model=("auth_image_file",),
                        label="Upload (optional)",
                        density="compact",
                        hide_details=True,
                        variant="outlined",
                        accept=(
                            "image/png,image/jpeg,image/webp"
                        ),
                        prepend_icon="mdi-camera",
                        disabled=("auth_busy",),
                        classes="golgi-auth-file",
                    )
                    html.Img(
                        src=("auth_image_data_uri",),
                        v_show=("auth_image_data_uri",),
                        classes="golgi-auth-avatar-preview",
                        alt="avatar preview",
                    )

                # Error line.
                html.Div(
                    "{{ auth_error }}",
                    v_show=("auth_error",),
                    classes="golgi-register-error",
                )
            # Register-mode action row — Sign-in link on the
            # left, Cancel + Create account on the right.
            with v3.VCardActions(
                classes="px-6 pb-4",
                v_show=("auth_mode === 'register'",),
            ):
                html.Button(
                    "Already have an account? Sign in",
                    type="button",
                    classes="golgi-auth-link",
                    click=(
                        do_switch_auth_mode, "['login']",
                    ),
                )
                v3.VSpacer()
                html.Button(
                    "Cancel",
                    type="button",
                    classes=(
                        "golgi-btn-secondary golgi-btn-sm"
                    ),
                    disabled=("auth_busy",),
                    click=do_close_auth_dialog,
                )
                html.Button(
                    "Create account",
                    type="button",
                    classes="golgi-btn-primary golgi-btn-sm",
                    disabled=("auth_busy",),
                    click=do_submit_register,
                )
