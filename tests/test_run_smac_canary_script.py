"""SMAC canary wrapper exposes a safe dry run."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_smac_canary.sh"


def test_smac_canary_script_dry_run_shows_archive_and_tuning_steps() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--archive",
            "instances",
            "--out-dir",
            "tuning/smac_run/full_canary",
            "--budget",
            "5",
            "--max-instances",
            "12",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scripts/finalize_full_archive.sh --archive instances --expected-files 9000" in completed.stdout
    assert "du -sh instances" in completed.stdout
    assert "uv run python code/tuning/smac_run.py" in completed.stdout
    assert "--archive instances" in completed.stdout
    assert "--out-dir tuning/smac_run/full_canary" in completed.stdout
    assert "--budget 5" in completed.stdout
    assert "--max-instances 12" in completed.stdout
    assert "--jobs 4" in completed.stdout
    assert "--ref-time-limit-s 60" in completed.stdout
    assert "--eval-time-limit-s 60" in completed.stdout
    assert "logs/smac_canary_" in completed.stdout
