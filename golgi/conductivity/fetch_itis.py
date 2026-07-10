# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""On-demand download of the IT'IS tissue-properties database.

golgi does **not** redistribute the IT'IS Material Parameter Database. This
helper fetches it directly from the IT'IS Foundation — on the user's own machine,
with an attribution/consent notice — and installs the SQLite file where golgi
expects it (``resources/tissue_db/``), so the Cole-Cole tissue presets light up.

Attribution — the downloaded database is:

    Tissue Properties Database V4-1 · IT'IS Foundation · 2022-02-22
    DOI: 10.13099/VIP21000-04-1 · https://itis.swiss/virtual-population/tissue-properties/

Its use is governed by the IT'IS Foundation's own license/terms, not golgi's.

Run:  ``python -m golgi.conductivity.fetch_itis``   (or ``golgi fetch-tissue-db``)
"""
from __future__ import annotations

import io
import sys
import zipfile
import urllib.request
from pathlib import Path

# Public IT'IS download (no auth). The Sim4Life dielectric SQLite inside the
# archive carries the Gabriel/Cole-Cole parameters golgi consumes.
ITIS_URL = "https://itis.swiss/assets/Downloads/TissueDb/Database-V4-1.zip"
ZIP_MEMBER = (
    "Database-V4.1/Thermal_dielectric_acoustic_MR properties_database "
    "V4.1(h5-Sim4Life_v2.0).db"
)
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = _REPO_ROOT / "resources" / "tissue_db"
DEST = DEST_DIR / "IT'IS_Material_database_V4.1.db"

CITATION = (
    "IT'IS Foundation — Tissue Properties Database V4-1 (2022-02-22), "
    "DOI 10.13099/VIP21000-04-1."
)
_NOTICE = (
    "[golgi] The IT'IS tissue-properties database is provided by the IT'IS\n"
    "        Foundation, not by golgi. This downloads it directly from\n"
    "        itis.swiss; your use is subject to the IT'IS license/terms.\n"
    f"        Source: {CITATION}\n"
    "        https://itis.swiss/virtual-population/tissue-properties/"
)


def _pick_member(names: list[str]) -> str | None:
    if ZIP_MEMBER in names:
        return ZIP_MEMBER
    # tolerate minor archive-name drift: the dielectric Sim4Life .db
    return next(
        (n for n in names if n.lower().endswith(".db")
         and "thermal_dielectric" in n.lower() and "sim4life" in n.lower()),
        None,
    )


def fetch_itis_db(dest: Path = DEST, url: str = ITIS_URL, *,
                  force: bool = False, quiet: bool = False) -> Path:
    """Download + install the IT'IS database, returning the installed path.

    No-op (returns the existing path) if a DB is already present unless
    ``force``. Raises on network/archive/validation failure."""
    if dest.exists() and not force:
        if not quiet:
            print(f"[golgi] IT'IS database already present at {dest}")
        return dest
    if not quiet:
        print(_NOTICE)
        print(f"[golgi] downloading {url} ...", flush=True)
    req = urllib.request.Request(url, headers={"User-Agent": "golgi/1.0"})
    with urllib.request.urlopen(req) as resp:            # noqa: S310 (trusted IT'IS URL)
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        member = _pick_member(zf.namelist())
        if member is None:
            raise RuntimeError(
                f"dielectric .db not found in {url}; archive holds: {zf.namelist()}")
        data = zf.read(member)
    if not data.startswith(b"SQLite format 3"):
        raise RuntimeError("downloaded file is not a SQLite database")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)
    if not quiet:
        print(f"[golgi] installed IT'IS database → {dest}")
        print(f"[golgi] please cite: {CITATION}")
    return dest


def main(argv: "list[str] | None" = None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        prog="golgi fetch-tissue-db",
        description="Download the IT'IS tissue-properties database from itis.swiss.")
    ap.add_argument("--force", action="store_true",
                    help="Re-download even if a database is already installed.")
    ap.add_argument("--dest", type=Path, default=DEST,
                    help=f"Destination path (default: {DEST}).")
    args = ap.parse_args(argv)
    try:
        fetch_itis_db(dest=args.dest, force=args.force)
    except Exception as ex:                              # noqa: BLE001
        print(f"[golgi] IT'IS download failed: {ex}\n"
              "        Download it manually instead — see "
              "resources/tissue_db/README.md.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
