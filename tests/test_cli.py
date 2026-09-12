# SPDX-License-Identifier: AGPL-3.0-or-later
"""`golgi export | import | replay` command-line interface (solver-free)."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from golgi.cli import dispatch
from golgi.projects import bundle as _bundle


def _fake_project(root: Path) -> Path:
    proj = root / "cli_study"
    (proj / "designs" / "elec_01").mkdir(parents=True)
    (proj / "project.json").write_text(json.dumps({"name": "cli_study"}))
    (proj / "designs" / "elec_01" / "nerve.msh").write_bytes(b"mesh" * 64)
    return proj


def test_dispatch_falls_through_without_subcommand():
    # No recognised subcommand → None so golgi.app starts the GUI server.
    assert dispatch([]) is None
    assert dispatch(["--port", "8080"]) is None


def test_export_import_replay_roundtrip(tmp_path, capsys):
    proj = _fake_project(tmp_path)
    out = tmp_path / "cli_study.golgi.zip"

    rc = dispatch(["export", str(proj), str(out), "--user", "pytest"])
    assert rc == 0, capsys.readouterr()
    assert out.is_file() and out.stat().st_size > 0

    capsys.readouterr()                      # drop export progress output
    rc = dispatch(["replay", str(out), "--json"])
    assert rc == 0
    report = json.loads(capsys.readouterr().out)
    assert report["files"] and all(f["matched"] for f in report["files"])
    assert report["ok"] is True
    assert report["n_files_mismatched"] == 0

    target = tmp_path / "imported"
    rc = dispatch(["import", str(out), str(target)])
    assert rc == 0, capsys.readouterr()
    assert any(target.rglob("project.json")) or (target / "project.json").is_file()


def test_replay_missing_bundle_is_an_error(tmp_path, capsys):
    rc = dispatch(["replay", str(tmp_path / "nope.zip")])
    assert rc == 1
    assert "not found" in capsys.readouterr().err


def test_replay_reports_tamper(tmp_path, capsys):
    proj = _fake_project(tmp_path)
    blob = _bundle.export_study(proj, exported_by_user="pytest")
    out = tmp_path / "ok.zip"
    out.write_bytes(blob)
    # corrupt: rewrite the mesh inside the zip
    import zipfile
    bad = tmp_path / "bad.zip"
    with zipfile.ZipFile(out) as zi, zipfile.ZipFile(bad, "w") as zo:
        for it in zi.infolist():
            data = zi.read(it.filename)
            if it.filename.endswith("nerve.msh"):
                data = b"HSEM" * 64
            zo.writestr(it, data)
    rc = dispatch(["replay", str(bad)])
    assert rc == 1
    assert "nerve.msh" in capsys.readouterr().out
