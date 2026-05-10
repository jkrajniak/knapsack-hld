"""Time-limit sensitivity runner CLI and cell filtering."""

from __future__ import annotations

import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_time_limit_sensitivity.py"


def test_time_limit_sensitivity_dry_run_filters_selected_cells(tmp_path: Path) -> None:
    manifest = {
        "files": [
            {
                "path": "inversely_strongly/N100000_M20/a.json.gz",
                "seed": 1,
                "subset": "test",
                "cell": {
                    "N": 100000,
                    "M": 20,
                    "correlation": "inversely_strongly",
                    "f": 0.5,
                },
            },
            {
                "path": "weakly/N100000_M20/b.json.gz",
                "seed": 2,
                "subset": "test",
                "cell": {"N": 100000, "M": 20, "correlation": "weakly", "f": 0.5},
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
            "--time-limits-s",
            "30",
            "120",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "out_csv: results/final_experiments/time_limit_sensitivity.csv" in completed.stdout
    assert "selected_cells: 6" in completed.stdout
    assert "eligible_instances: 1" in completed.stdout
    assert "time_limits_s: 30 120" in completed.stdout
    assert "planned_rows: 2" in completed.stdout


def test_time_limit_sensitivity_completed_keys_include_time_limit(tmp_path: Path) -> None:
    module = _load_script_module()
    out_csv = tmp_path / "sensitivity.csv"
    with out_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["time_limit_s", "solver", "instance_id", "status"],
        )
        writer.writeheader()
        writer.writerow(
            {
                "time_limit_s": "30",
                "solver": "hld",
                "instance_id": "a.json.gz",
                "status": "timeout",
            }
        )

    assert module.completed_keys(out_csv) == {("30", "hld", "a.json.gz")}


def _load_script_module() -> object:
    spec = importlib.util.spec_from_file_location("run_time_limit_sensitivity", SCRIPT)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["run_time_limit_sensitivity"] = module
    spec.loader.exec_module(module)
    return module
