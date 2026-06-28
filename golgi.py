# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Backwards-compatible launcher shim.

The real entrypoint lives in `golgi/app.py` (moved there in
migration Step 6.1). Run either:

    python golgi.py [--port 8080]    # via this shim
    python -m golgi.app [--port 8080]  # direct

Both invoke `golgi.app.main()`.

Python's import resolution finds the `golgi/` package (regular
packages take precedence over same-named top-level modules), so
`from golgi.app import main` here resolves to golgi/app.py, not
to this file recursively.
"""
from golgi.app import main

if __name__ == "__main__":
    main()
