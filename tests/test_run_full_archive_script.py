"""Big-machine archive wrapper plans the full generation safely."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_full_archive.sh"


def test_full_archive_script_dry_run_shows_generation_and_verification() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--out",
            "instances_full_candidate",
            "--jobs",
            "16",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "uv sync" in completed.stdout
    assert "scripts/generate_instances.py" in completed.stdout
    assert "--config scripts/configs/archive_full.yaml" in completed.stdout
    assert "--out instances_full_candidate" in completed.stdout
    assert "--jobs 16" in completed.stdout
    assert "scripts/verify_instances.py --archive instances_full_candidate" in completed.stdout
    assert "logs/full_archive_" in completed.stdout
