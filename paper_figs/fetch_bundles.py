# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Download the golgi reproduction *study bundles* from Zenodo.

The Zenodo record ships replay-verified `.golgi.zip` study bundles (plus their
BUNDLES.json + CHECKSUMS.sha256 manifest and a README). Each bundle drives:

    golgi replay <bundle.golgi.zip>    # re-hash / integrity-verify the study
    golgi figure <bundle.golgi.zip>    # render its quick-look panels

This is the third-party reproduction path. It is distinct from
`make_figures.py`, which regenerates the full multi-panel paper figures from the
authors' complete out/_intermediate + out/data working tree (not on Zenodo).

Usage:
    python paper_figs/fetch_bundles.py                 # all files -> default dir
    python paper_figs/fetch_bundles.py --dest ./repro  # choose destination
    python paper_figs/fetch_bundles.py --only fig06    # substring filter
    python paper_figs/fetch_bundles.py --list          # list, don't download

Stdlib only (urllib) — no extra dependencies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.parse import quote
from urllib.request import Request, urlopen

# Concept DOI 10.5281/zenodo.21300944 — always resolves to the LATEST version,
# so the fetcher keeps working when a new bundle version is published. The API
# returns the latest version's files (with version-specific download links) for
# this concept record id.
RECORD_ID = "21300944"                     # concept recid (10.5281/zenodo.21300944)
API = "https://zenodo.org/api/records/{rid}"
DL = "https://zenodo.org/records/{rid}/files/{key}?download=1"   # fallback only
HERE = Path(__file__).resolve().parent
DEFAULT_DEST = HERE / "out" / "zenodo_bundles"


def _get(url: str):
    req = Request(url, headers={"User-Agent": "golgi-fetch-bundles/1.0"})
    with urlopen(req) as r:                # noqa: S310 (trusted Zenodo host)
        return r.read()


def _list_files(rid: str) -> list[dict]:
    meta = json.loads(_get(API.format(rid=rid)).decode("utf-8"))
    out = []
    for f in meta.get("files", []):
        key = f.get("key") or f.get("filename")
        chk = f.get("checksum", "")        # "md5:<hex>"
        algo, _, digest = chk.partition(":") if ":" in chk else ("", "", "")
        # version-specific download link straight from the API (robust when the
        # concept recid resolves to whichever version is latest)
        link = (f.get("links", {}) or {}).get("self", "")
        out.append(dict(key=key, size=int(f.get("size", 0)),
                        algo=algo or "md5", digest=digest, link=link))
    return sorted(out, key=lambda d: d["key"])


def _digest(path: Path, algo: str) -> str:
    h = hashlib.new(algo)
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, size: int) -> None:
    req = Request(url, headers={"User-Agent": "golgi-fetch-bundles/1.0"})
    done = 0
    with urlopen(req) as r, dest.open("wb") as out:   # noqa: S310
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            out.write(chunk); done += len(chunk)
            if size:
                pct = 100 * done / size
                print(f"\r    {dest.name}: {done/1e6:7.1f}/{size/1e6:.1f} MB "
                      f"({pct:4.0f}%)", end="", flush=True)
    print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--record", default=RECORD_ID, help="Zenodo record id")
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST,
                    help=f"download directory (default: {DEFAULT_DEST})")
    ap.add_argument("--only", default="", help="substring filter on filename")
    ap.add_argument("--list", action="store_true", help="list files and exit")
    a = ap.parse_args()

    try:
        files = _list_files(a.record)
    except Exception as ex:                 # noqa: BLE001
        print(f"✗ could not reach Zenodo record {a.record}: {ex}", file=sys.stderr)
        print("  Check the DOI (10.5281/zenodo.<record>) and your connection.",
              file=sys.stderr)
        return 1
    if a.only:
        files = [f for f in files if a.only in f["key"]]
    if not files:
        print("no matching files.", file=sys.stderr); return 1

    total = sum(f["size"] for f in files)
    print(f"record {a.record}: {len(files)} file(s), {total/1e6:.0f} MB total")
    if a.list:
        for f in files:
            print(f"  {f['size']/1e6:8.1f} MB  {f['key']}")
        return 0

    a.dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for f in files:
        key, dest = f["key"], a.dest / f["key"]
        if dest.exists() and f["digest"] and _digest(dest, f["algo"]) == f["digest"]:
            print(f"  ✓ {key} (already present, checksum ok)"); ok += 1
            continue
        print(f"  ↓ {key} ({f['size']/1e6:.1f} MB)")
        url = f["link"] or DL.format(rid=a.record, key=quote(key))
        try:
            _download(url, dest, f["size"])
        except Exception as ex:             # noqa: BLE001
            print(f"    ✗ download failed: {ex}", file=sys.stderr); continue
        if f["digest"]:
            got = _digest(dest, f["algo"])
            if got != f["digest"]:
                print(f"    ✗ checksum mismatch ({f['algo']}): "
                      f"{got} != {f['digest']}", file=sys.stderr); continue
            print(f"    ✓ {f['algo']} verified")
        ok += 1

    print(f"\n{ok}/{len(files)} file(s) in {a.dest}")
    print("Next: golgi replay <bundle.golgi.zip>  |  golgi figure <bundle.golgi.zip>")
    return 0 if ok == len(files) else 1


if __name__ == "__main__":
    raise SystemExit(main())
