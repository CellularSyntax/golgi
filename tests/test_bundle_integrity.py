# SPDX-License-Identifier: AGPL-3.0-or-later
"""Study-bundle export → import → integrity verification.

Solver-free: these tests build a small fake project directory, export it
as a bundle, verify it with `golgi.projects.replay.replay_study`, and
check that a single flipped byte is detected. They exercise the exact
code paths behind `golgi export` / `golgi import` / `golgi replay`.
"""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from golgi.projects import bundle as _bundle
from golgi.projects import replay as _replay


def _fake_project(root: Path) -> Path:
    """A minimal project directory with the files the bundle DAG knows."""
    proj = root / "fake_study"
    (proj / "designs" / "elec_01").mkdir(parents=True)
    (proj / "configs" / "cfg_01").mkdir(parents=True)
    (proj / "project.json").write_text(json.dumps({
        "name": "fake_study", "created_by": "pytest", "version": 1,
    }))
    (proj / "ui_state.json").write_text(json.dumps({"n_fibers": 4}))
    (proj / "designs" / "elec_01" / "nerve.msh").write_bytes(
        b"$MeshFormat\n4.1 0 8\n$EndMeshFormat\n" + bytes(range(256)) * 8)
    (proj / "configs" / "cfg_01" / "lead_field.npz").write_bytes(
        b"PK\x03\x04" + b"\x00" * 512)
    return proj


def _sha256(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


@pytest.fixture()
def bundle_zip(tmp_path) -> tuple[Path, Path]:
    proj = _fake_project(tmp_path)
    blob = _bundle.export_study(proj, exported_by_user="pytest")
    out = tmp_path / "fake_study.golgi.zip"
    out.write_bytes(blob)
    return proj, out


def test_export_writes_manifest_with_hashes(bundle_zip):
    proj, out = bundle_zip
    with zipfile.ZipFile(out) as zf:
        names = set(zf.namelist())
        assert "MANIFEST.json" in names
        assert "project.json" in names
        manifest = json.loads(zf.read("MANIFEST.json"))
    files = manifest.get("files") or []
    assert files, "manifest lists no files"
    by_name = {f["name"]: f["sha256"] for f in files}
    # every hash in the manifest is the sha256 of the project file
    # (audit/audit_excerpt.json is generated at export time and has no
    # on-disk counterpart in the project directory)
    checked = 0
    for name, sha in by_name.items():
        src = proj / name
        if not src.is_file():
            assert name.startswith(("audit/", "env/")), \
                f"unexpected generated file {name}"
            continue
        assert sha == _sha256(src), name
        checked += 1
    assert checked >= 3   # project.json, nerve.msh, lead_field.npz
    # the pipeline DAG and environment metadata travel with the bundle
    assert manifest.get("dag"), "no stage DAG"
    assert {"mesh", "fem"} <= {st["stage"] for st in manifest["dag"]}
    assert manifest.get("golgi_version")


def test_export_is_deterministic(tmp_path):
    proj = _fake_project(tmp_path)
    a = _bundle.export_study(proj, exported_by_user="pytest")
    b = _bundle.export_study(proj, exported_by_user="pytest")
    # file payload hashes are identical run-to-run (timestamps live only
    # in MANIFEST.json's metadata block)
    with zipfile.ZipFile(_bytes_io(a)) as za, zipfile.ZipFile(_bytes_io(b)) as zb:
        ma = json.loads(za.read("MANIFEST.json"))
        mb = json.loads(zb.read("MANIFEST.json"))
    assert {f["name"]: f["sha256"] for f in ma["files"]} == \
           {f["name"]: f["sha256"] for f in mb["files"]}


def _bytes_io(blob: bytes):
    import io
    return io.BytesIO(blob)


def test_replay_verifies_intact_bundle(bundle_zip):
    _, out = bundle_zip
    report = _replay.replay_study(out, check_only=True)
    assert report.ok, report.short_summary()
    assert report.n_files_mismatched == 0
    assert report.n_files_matched == report.n_files_total > 0


def test_replay_detects_single_byte_tamper(bundle_zip, tmp_path):
    _, out = bundle_zip
    # Re-write the zip with one payload byte flipped in the mesh file.
    tampered = tmp_path / "tampered.zip"
    with zipfile.ZipFile(out) as zin, zipfile.ZipFile(tampered, "w") as zout:
        for item in zin.infolist():
            data = zin.read(item.filename)
            if item.filename.endswith("nerve.msh"):
                data = bytearray(data)
                data[-1] ^= 0xFF
                data = bytes(data)
            zout.writestr(item, data)
    report = _replay.replay_study(tampered, check_only=True)
    assert not report.ok
    assert report.n_files_mismatched == 1
    bad = report.mismatched_files()
    assert [f.name for f in bad] == ["designs/elec_01/nerve.msh"]
    assert bad[0].note == "(sha mismatch)"


def test_import_roundtrip(bundle_zip, tmp_path):
    proj, out = bundle_zip
    target = tmp_path / "imported"          # must not exist yet
    manifest = _bundle.import_study(out, target)
    assert manifest.get("project", {}).get("name") == "fake_study"
    imported = target                        # bundle root == project root
    assert (imported / "project.json").is_file()
    assert (imported / "env" / "golgi_version.txt").is_file()
    for rel in ("designs/elec_01/nerve.msh", "configs/cfg_01/lead_field.npz"):
        assert _sha256(imported / rel) == _sha256(proj / rel)
