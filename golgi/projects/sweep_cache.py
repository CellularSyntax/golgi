# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (C) 2026 the golgi authors.

"""Sweep-result on-disk cache (F2.1.d).

Each sweep run persists to `<project>/sweeps/sweep_<sha>.npz` plus a
sibling JSON manifest and three CSV exports (recruitment / threshold
/ activation_heatmap). The SHA is `SweepRequest`-derived so repeating
the same sweep hits the cache instantly.

Layout per sweep:
  <project>/sweeps/
    sweep_<sha>.npz                       binary SweepResult payload
    sweep_<sha>.json                      request + summary metadata
    sweep_<sha>_recruitment.csv           when mode = recruitment
    sweep_<sha>_thresholds.csv            when mode = threshold
    sweep_<sha>_activation_heatmap.csv    when mode = recruitment

The .json manifest carries the SweepRequest dict + a human-friendly
summary, and an `is_latest` flag is maintained via a sibling
`latest.txt` that names the most recently-written sha. Auto-restore
on project open reads `latest.txt` and surfaces the matching sweep
in the UI.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from golgi.figures.recruitment import (
    activation_heatmap_to_csv,
    recruitment_to_csv,
    threshold_to_csv,
)
from golgi.jobs.schemas import SweepRequest, SweepResult


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def sweeps_dir(out_dir: Path) -> Path:
    """Return (and lazily create) `<project>/sweeps/`."""
    p = Path(out_dir) / "sweeps"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_sweep(
    result: SweepResult,
    out_dir: Path,
    *,
    write_csvs: bool = True,
    cid: Optional[str] = None,
) -> dict[str, Path]:
    """Persist a SweepResult to disk.

    Filename: `sweep_<sha>.npz` when `cid` is None (legacy /
    project-wide sweep); `sweep_<cid>_<sha>.npz` when `cid` is
    provided. The cid-tagged form is what F3.2 selectivity reads —
    it lets multiple configs co-exist in `<project>/sweeps/`
    without overwriting each other.

    Returns a dict {kind: path} for the files written, e.g.
        {"npz": .../sweep_abc123.npz,
         "json": .../sweep_abc123.json,
         "recruitment_csv": .../sweep_abc123_recruitment.csv,
         "heatmap_csv": .../sweep_abc123_activation_heatmap.csv}

    Idempotent: re-saving the same SweepResult overwrites the
    existing files. Bumps `latest.txt` to point at this run
    (cid-prefixed when applicable).
    """
    sha = str(result.sha or _short_sha(result.request))
    out = sweeps_dir(out_dir)
    paths: dict[str, Path] = {}

    # F3.2 — optional cid prefix so per-config sweeps coexist.
    # The cid is sanitised (no path separators / dots) so it can't
    # escape the sweeps/ directory.
    _cid_safe = (
        "".join(
            c if (c.isalnum() or c in ("-", "_")) else "_"
            for c in str(cid)
        )
        if cid else ""
    )
    stem = (
        f"sweep_{_cid_safe}_{sha}" if _cid_safe
        else f"sweep_{sha}"
    )

    # Binary payload — np.savez_compressed handles every field type
    # in to_npz_payload() (arrays, scalars, object arrays for strs).
    npz_path = out / f"{stem}.npz"
    payload = result.to_npz_payload()
    np.savez_compressed(npz_path, **payload)
    paths["npz"] = npz_path

    # JSON manifest — the request schema + a human-readable summary.
    json_path = out / f"{stem}.json"
    manifest = {
        "sha": sha,
        "cid": str(cid) if cid else "",
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "mode": str(result.request.mode),
        "elapsed_s": float(result.elapsed_s),
        "n_fibers": int(len(result.fiber_indices)),
        "n_sims_total": int(result.n_sims_total),
        "request": result.request.serialize(),
    }
    json_path.write_text(json.dumps(manifest, indent=2))
    paths["json"] = json_path

    if write_csvs:
        if result.activated is not None:
            rec_csv = out / f"{stem}_recruitment.csv"
            rec_csv.write_text(recruitment_to_csv(result))
            paths["recruitment_csv"] = rec_csv
            hm_csv = out / f"{stem}_activation_heatmap.csv"
            hm_csv.write_text(activation_heatmap_to_csv(result))
            paths["heatmap_csv"] = hm_csv
        if result.thresholds_uA is not None:
            thr_csv = out / f"{stem}_thresholds.csv"
            thr_csv.write_text(threshold_to_csv(result))
            paths["threshold_csv"] = thr_csv

    # Mark this as the latest run. When tagged with a cid, write
    # BOTH the global latest.txt (for backward-compat callers that
    # only know about the project-wide latest) AND a per-cid
    # `latest_<cid>.txt` for F3.2 selectivity loading.
    (out / "latest.txt").write_text(sha)
    if _cid_safe:
        (out / f"latest_{_cid_safe}.txt").write_text(sha)

    return paths


def list_sweeps(out_dir: Path) -> list[dict]:
    """Enumerate every cached sweep with summary metadata. Sorted
    by saved_at descending (newest first). Missing manifest files
    are skipped silently."""
    out = sweeps_dir(out_dir)
    rows: list[dict] = []
    for json_path in out.glob("sweep_*.json"):
        try:
            m = json.loads(json_path.read_text())
        except Exception:                                    # noqa: BLE001
            continue
        m["json_path"] = str(json_path)
        m["npz_path"] = str(
            json_path.with_suffix(".npz"),
        )
        rows.append(m)
    rows.sort(
        key=lambda r: r.get("saved_at", ""), reverse=True,
    )
    return rows


def load_sweep(
    out_dir: Path, sha: str,
) -> Optional[SweepResult]:
    """Reconstitute a SweepResult from `<project>/sweeps/sweep_<sha>.npz`
    + `.json`. Returns None when either file is missing or unreadable
    (caller logs + treats as cache miss)."""
    out = sweeps_dir(out_dir)
    npz_path = out / f"sweep_{sha}.npz"
    json_path = out / f"sweep_{sha}.json"
    if not (npz_path.exists() and json_path.exists()):
        return None
    try:
        manifest = json.loads(json_path.read_text())
        req = SweepRequest.deserialize(manifest.get("request", {}))
        with np.load(npz_path, allow_pickle=True) as z:
            kw: dict = {
                "request": req,
                "fiber_indices": np.asarray(
                    z["fiber_indices"], dtype=np.int64,
                ),
                "fiber_diameters_um": np.asarray(
                    z["fiber_diameters_um"], dtype=np.float64,
                ),
                "fiber_branch_idx": np.asarray(
                    z["fiber_branch_idx"], dtype=np.int32,
                ),
                "fiber_type_labels": [
                    str(s) for s in z["fiber_type_labels"]
                ],
                "elapsed_s": float(z["elapsed_s"]),
                "sha": str(z["sha"]),
                "n_sims_total": int(z["n_sims_total"]),
            }
            if "activated" in z.files:
                kw["activated"] = np.asarray(
                    z["activated"], dtype=np.bool_,
                )
            if "thresholds_uA" in z.files:
                kw["thresholds_uA"] = np.asarray(
                    z["thresholds_uA"], dtype=np.float64,
                )
            if "bisect_iters" in z.files:
                kw["bisect_iters"] = np.asarray(
                    z["bisect_iters"], dtype=np.int32,
                )
        return SweepResult(**kw)
    except Exception as ex:                                  # noqa: BLE001
        print(
            f"[sweep_cache] failed to load {sha}: {ex}",
            flush=True,
        )
        return None


def load_latest(out_dir: Path) -> Optional[SweepResult]:
    """Read `<project>/sweeps/latest.txt` for the most recent sha
    and return the matching SweepResult, or None if no sweeps
    have ever been cached for this project."""
    out = sweeps_dir(out_dir)
    latest = out / "latest.txt"
    if not latest.is_file():
        return None
    sha = latest.read_text().strip()
    if not sha:
        return None
    return load_sweep(out_dir, sha)


def load_latest_for_config(
    out_dir: Path, cid: str,
) -> Optional[SweepResult]:
    """F3.2 — load the most recent SweepResult tagged with `cid`.

    Reads `<project>/sweeps/latest_<cid>.txt` (written by
    `save_sweep(... cid=<cid>)`) for the sha, then locates the
    file as `sweep_<cid>_<sha>.npz`. Returns None when the cid
    has never had a sweep saved against it. Used by the Compare
    panel's selectivity loader.
    """
    if not cid:
        return None
    _cid_safe = "".join(
        c if (c.isalnum() or c in ("-", "_")) else "_"
        for c in str(cid)
    )
    out = sweeps_dir(out_dir)
    marker = out / f"latest_{_cid_safe}.txt"
    if not marker.is_file():
        return None
    sha = marker.read_text().strip()
    if not sha:
        return None
    # Reconstruct via the same loader path but with the cid-tagged
    # stem.
    stem = f"sweep_{_cid_safe}_{sha}"
    npz_path = out / f"{stem}.npz"
    json_path = out / f"{stem}.json"
    if not (npz_path.exists() and json_path.exists()):
        return None
    try:
        manifest = json.loads(json_path.read_text())
        req = SweepRequest.deserialize(
            manifest.get("request", {}),
        )
        with np.load(npz_path, allow_pickle=True) as z:
            kw: dict = {
                "request": req,
                "fiber_indices": np.asarray(
                    z["fiber_indices"], dtype=np.int64,
                ),
                "fiber_diameters_um": np.asarray(
                    z["fiber_diameters_um"], dtype=np.float64,
                ),
                "fiber_branch_idx": np.asarray(
                    z["fiber_branch_idx"], dtype=np.int32,
                ),
                "fiber_type_labels": [
                    str(s) for s in z["fiber_type_labels"]
                ],
                "elapsed_s": float(z["elapsed_s"]),
                "sha": str(z["sha"]),
                "n_sims_total": int(z["n_sims_total"]),
            }
            if "activated" in z.files:
                kw["activated"] = np.asarray(
                    z["activated"], dtype=np.bool_,
                )
            if "thresholds_uA" in z.files:
                kw["thresholds_uA"] = np.asarray(
                    z["thresholds_uA"], dtype=np.float64,
                )
            if "bisect_iters" in z.files:
                kw["bisect_iters"] = np.asarray(
                    z["bisect_iters"], dtype=np.int32,
                )
        return SweepResult(**kw)
    except Exception as ex:                                  # noqa: BLE001
        print(
            f"[sweep_cache] failed to load cid={cid}: {ex}",
            flush=True,
        )
        return None


def csv_paths_for(
    out_dir: Path, sha: str,
) -> dict[str, Optional[Path]]:
    """Return the expected CSV paths for a given sha (whether or
    not the files exist). Used by the UI to surface "open this
    file" affordances after a sweep completes."""
    out = sweeps_dir(out_dir)
    return {
        "recruitment_csv": out / f"sweep_{sha}_recruitment.csv",
        "threshold_csv": out / f"sweep_{sha}_thresholds.csv",
        "heatmap_csv": (
            out / f"sweep_{sha}_activation_heatmap.csv"
        ),
    }


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _short_sha(req: SweepRequest) -> str:
    """Fallback sha computation in case SweepResult.sha wasn't
    populated (older driver versions). Mirrors
    pipeline/sweep.py:_sha_for_request — keep the formula in sync
    if either changes."""
    import hashlib
    blob = json.dumps(
        req.serialize(), sort_keys=True, separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]
