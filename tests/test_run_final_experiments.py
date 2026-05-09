"""Final experiment runner CLI and manifest filtering."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_final_experiments.py"


def test_final_experiments_dry_run_reports_plan() -> None:
    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--dry-run",
            "--archive",
            "instances",
            "--config",
            "configs/hld_smac_best.json",
            "--solvers",
            "hld",
            "--jobs",
            "8",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "archive: instances" in completed.stdout
    assert "config: configs/hld_smac_best.json" in completed.stdout
    assert "out_csv: results/final_experiments/results.csv" in completed.stdout
    assert "subset: test" in completed.stdout
    assert "solvers: hld" in completed.stdout
    assert "jobs: 8" in completed.stdout


def test_final_experiments_manifest_filter_counts_test_subset(tmp_path: Path) -> None:
    manifest = {
        "files": [
            {
                "path": "weakly/N1000_M5/a.json.gz",
                "seed": 1,
                "subset": "test",
                "cell": {"N": 1000, "M": 5, "correlation": "weakly", "f": 0.1},
            },
            {
                "path": "weakly/N1000_M5/b.json.gz",
                "seed": 2,
                "subset": "tuning",
                "cell": {"N": 1000, "M": 5, "correlation": "weakly", "f": 0.1},
            },
            {
                "path": "weakly/N100000_M5/c.json.gz",
                "seed": 3,
                "subset": "test",
                "cell": {"N": 100000, "M": 5, "correlation": "weakly", "f": 0.1},
            },
        ]
    }
    manifest_path = tmp_path / "MANIFEST.json"
    manifest_path.write_text(json.dumps(manifest))

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--dry-run",
            "--archive",
            str(tmp_path),
            "--manifest",
            str(manifest_path),
            "--max-N",
            "1000",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "eligible_instances: 1" in completed.stdout
