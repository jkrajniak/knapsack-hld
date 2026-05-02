"""Post-generation archive finalizer exposes a safe dry run."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "finalize_full_archive.sh"


def test_finalize_script_dry_run_shows_verify_summary_and_promote_steps() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--archive",
            "instances_full_candidate",
            "--promote",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "scripts/verify_instances.py --archive instances_full_candidate" in completed.stdout
    assert "expected files: 9000" in completed.stdout
    assert "summary path: logs/full_archive_summary_" in completed.stdout
    assert "mv instances instances_backup_" in completed.stdout
    assert "mv instances_full_candidate instances" in completed.stdout
