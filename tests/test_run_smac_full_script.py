"""Full SMAC campaign wrapper exposes a safe dry run."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_smac_full.sh"


def test_smac_full_script_dry_run_shows_unique_full_campaign_steps() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--archive",
            "instances",
            "--budget",
            "5000",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert (
        "scripts/finalize_full_archive.sh --archive instances --expected-files 9000"
        in completed.stdout
    )
    assert "du -sh instances" in completed.stdout
    assert "uv run python code/tuning/smac_run.py" in completed.stdout
    assert "--archive instances" in completed.stdout
    assert "--out-dir results/smac_run/full_" in completed.stdout
    assert "--budget 5000" in completed.stdout
    assert "--max-N 10000" in completed.stdout
    assert "--jobs 8" in completed.stdout
    assert "--seed 7" in completed.stdout
    assert "--ref-time-limit-s 60" in completed.stdout
    assert "--eval-time-limit-s 60" in completed.stdout
    assert "logs/smac_full_" in completed.stdout
