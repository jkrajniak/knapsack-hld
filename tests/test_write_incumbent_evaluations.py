"""Post-process a completed SMAC run into per-instance incumbent rows."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "write_incumbent_evaluations.py"


def test_write_incumbent_evaluations_dry_run_reports_resolved_paths(tmp_path: Path) -> None:
    run_dir = tmp_path / "full_20260508T192425Z"
    run_dir.mkdir()
    (run_dir / "incumbent.json").write_text(
        json.dumps(
            {
                "config": {
                    "N_iter": 35,
                    "alpha": 0.998,
                    "K": 58,
                    "lambda_max": 80.745,
                },
                "smac": {"seed": 7},
            }
        )
    )

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--dry-run",
            "--run-dir",
            str(run_dir),
            "--archive",
            "instances",
            "--jobs",
            "8",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert f"run_dir: {run_dir}" in completed.stdout
    assert f"incumbent_json: {run_dir / 'incumbent.json'}" in completed.stdout
    assert f"reference_cache: {run_dir / 'reference_profits.json'}" in completed.stdout
    assert f"out_csv: {run_dir / 'incumbent_evaluations.csv'}" in completed.stdout
    assert "config: n_iter=35 alpha=0.998 k=58 lambda_max=80.745" in completed.stdout
    assert "jobs: 8" in completed.stdout
