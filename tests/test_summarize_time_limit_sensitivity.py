"""Summary CLI for time-limit sensitivity result CSVs."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_time_limit_sensitivity.py"


def test_summarize_time_limit_sensitivity_writes_gain_summaries(tmp_path: Path) -> None:
    sensitivity_csv = tmp_path / "time_limit_sensitivity.csv"
    out_dir = tmp_path / "summary"
    rows = [
        _row(instance_id="a.json.gz", time_limit_s="30", profit="100", status="timeout"),
        _row(instance_id="a.json.gz", time_limit_s="60", profit="110", status="timeout"),
        _row(instance_id="a.json.gz", time_limit_s="120", profit="121", status="feasible"),
        _row(instance_id="a.json.gz", time_limit_s="300", profit="133", status="feasible"),
        _row(instance_id="b.json.gz", time_limit_s="30", profit="200", status="timeout"),
        _row(instance_id="b.json.gz", time_limit_s="60", profit="200", status="timeout"),
        _row(instance_id="b.json.gz", time_limit_s="120", profit="220", status="timeout"),
        _row(instance_id="b.json.gz", time_limit_s="300", profit="242", status="feasible"),
    ]
    with sensitivity_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--sensitivity-csv",
            str(sensitivity_csv),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert f"sensitivity_csv: {sensitivity_csv}" in completed.stdout
    assert f"out_dir: {out_dir}" in completed.stdout
    assert "total_rows: 8" in completed.stdout
    assert "60->300: n=2 mean=20.9545% median=20.9545% max=21.0000%" in completed.stdout

    overall = json.loads((out_dir / "overall.json").read_text())
    assert overall["total_rows"] == 8
    assert overall["status_counts"] == {"feasible": 3, "timeout": 5}

    status_rows = list(csv.DictReader((out_dir / "time_limit_status.csv").open(newline="")))
    assert status_rows[0] == {
        "time_limit_s": "30",
        "total_rows": "2",
        "status_counts": "timeout=2",
        "feasible_count": "0",
        "timeout_count": "2",
        "timeout_rate": "1.000000",
    }

    gain_rows = list(csv.DictReader((out_dir / "profit_gains.csv").open(newline="")))
    assert gain_rows[-1]["comparison"] == "60->300"
    assert gain_rows[-1]["n"] == "2"
    assert gain_rows[-1]["mean_gain"] == "0.209545"
    assert gain_rows[-1]["median_gain"] == "0.209545"
    assert gain_rows[-1]["max_gain"] == "0.210000"


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "time_limit_s": "30",
        "instance_id": "a.json.gz",
        "N": "100000",
        "M": "20",
        "correlation": "strongly",
        "f": "0.5",
        "solver": "hld",
        "status": "timeout",
        "profit": "100",
        "wall_time_s": "30.1",
    }
    row.update(overrides)
    return row
