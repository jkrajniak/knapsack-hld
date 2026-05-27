"""Wrapper script for the class-ordering ablation (Task 3.3.2)."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_class_ordering_ablation.sh"


def test_run_class_ordering_ablation_dry_run_lists_plan() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    assert "class-ordering ablation plan" in out
    assert "orderings:    sequential random adversarial" in out
    assert "cells:        6" in out
    assert "time_limit_s: 60" in out
    assert "100000,20,inversely_strongly,0.5" in out
    assert "100000,5,strongly,0.75" in out


def test_run_class_ordering_ablation_respects_env_overrides() -> None:
    """ORDERINGS / TIME_LIMIT_S env overrides reach the printed plan."""
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--dry-run"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
        env={
            "PATH": "/usr/bin:/bin",
            "ORDERINGS": "sequential adversarial",
            "TIME_LIMIT_S": "300",
            "JOBS": "6",
        },
    )

    assert completed.returncode == 0, completed.stderr
    out = completed.stdout
    assert "orderings:    sequential adversarial" in out
    assert "time_limit_s: 300" in out
    assert "jobs:         6" in out
