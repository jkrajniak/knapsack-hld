"""SMAC run artifact archiver stores completed campaigns outside git."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "archive_smac_run.sh"


def test_archive_smac_run_dry_run_uses_private_artifact_dir() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--run-dir",
            "results/smac_run/full_20260508T192425Z",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "artifact_dir: ../artifacts/smac" in completed.stdout
    assert "log_file: logs/smac_full_20260508T192425Z.log" in completed.stdout
    assert "tar -czf ../artifacts/smac/smac_full_20260508T192425Z.tar.gz" in completed.stdout
    assert "shasum -a 256 ../artifacts/smac/smac_full_20260508T192425Z.tar.gz" in completed.stdout


def test_archive_smac_run_writes_tarball_and_checksum(tmp_path: Path) -> None:
    run_dir = tmp_path / "results" / "smac_run" / "full_20260508T192425Z"
    log_file = tmp_path / "logs" / "smac_full_20260508T192425Z.log"
    artifact_dir = tmp_path / "artifacts"
    (run_dir / "hld_smac").mkdir(parents=True)
    log_file.parent.mkdir(parents=True)
    for name in ("incumbent.json", "evaluations.csv", "reference_profits.json"):
        (run_dir / name).write_text("ok\n")
    log_file.write_text("log\n")

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--run-dir",
            str(run_dir),
            "--log-file",
            str(log_file),
            "--artifact-dir",
            str(artifact_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    archive = artifact_dir / "smac_full_20260508T192425Z.tar.gz"
    checksum = artifact_dir / "smac_full_20260508T192425Z.tar.gz.sha256"
    assert completed.returncode == 0
    assert archive.exists()
    assert checksum.exists()
    assert archive.name in checksum.read_text()
