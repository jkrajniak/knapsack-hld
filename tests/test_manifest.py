"""Build → write → verify cycle for the archive manifest."""

from __future__ import annotations

from pathlib import Path

import pytest

from instances import (
    build_manifest,
    generate_instance,
    instance_id,
    save_instance,
    verify_manifest,
    write_manifest,
)


@pytest.fixture
def tiny_archive(tmp_path: Path) -> Path:
    for corr in ("uncorrelated", "strongly"):
        for f in (0.25, 0.75):
            for seed in (0, 1, 2):
                inst = generate_instance(N=50, M=5, correlation=corr, f=f, seed=seed)
                stem = instance_id(N=50, M=5, correlation=corr, f=f, seed=seed)
                target = tmp_path / corr / "N50_M5" / f"{stem}.json.gz"
                save_instance(inst, target)
    return tmp_path


def test_build_and_verify_clean(tiny_archive: Path) -> None:
    manifest = build_manifest(tiny_archive)
    write_manifest(manifest, tiny_archive)
    assert len(manifest["files"]) == 2 * 2 * 3
    ok, errors = verify_manifest(tiny_archive)
    assert ok, errors


def test_verify_detects_tampering(tiny_archive: Path) -> None:
    manifest = build_manifest(tiny_archive)
    write_manifest(manifest, tiny_archive)
    victim = tiny_archive / manifest["files"][0]["path"]
    victim.write_bytes(victim.read_bytes() + b"\x00")
    ok, errors = verify_manifest(tiny_archive)
    assert not ok
    assert any("size mismatch" in e or "sha256 mismatch" in e for e in errors)


def test_verify_detects_missing_file(tiny_archive: Path) -> None:
    manifest = build_manifest(tiny_archive)
    write_manifest(manifest, tiny_archive)
    victim = tiny_archive / manifest["files"][0]["path"]
    victim.unlink()
    ok, errors = verify_manifest(tiny_archive)
    assert not ok
    assert any("missing file" in e for e in errors)


def test_manifest_subset_labels_consistent(tiny_archive: Path) -> None:
    manifest = build_manifest(tiny_archive)
    subsets = {entry["subset"] for entry in manifest["files"]}
    assert subsets <= {"tuning", "test"}
