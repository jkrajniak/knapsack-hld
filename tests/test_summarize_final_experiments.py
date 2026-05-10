"""Summary CLI for final experiment result CSVs."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_final_experiments.py"


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "N": "1000",
        "M": "5",
        "correlation": "weakly",
        "f": "0.1",
        "solver": "hld",
        "status": "feasible",
        "wall_time_s": "12.0",
    }
    row.update(overrides)
    return row


def test_summarize_final_experiments_writes_status_and_timeout_summaries(
    tmp_path: Path,
) -> None:
    results_csv = tmp_path / "results.csv"
    out_dir = tmp_path / "summary"
    rows = [
        _row(),
        _row(
            status="timeout",
            wall_time_s="60.1",
        ),
        _row(
            M="10",
            correlation="strongly",
            f="0.9",
            status="error",
            wall_time_s="1.5",
        ),
    ]
    with results_csv.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--results-csv",
            str(results_csv),
            "--out-dir",
            str(out_dir),
            "--top-timeout-cells",
            "5",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert f"results_csv: {results_csv}" in completed.stdout
    assert f"out_dir: {out_dir}" in completed.stdout
    assert "total_rows: 3" in completed.stdout

    overall = json.loads((out_dir / "overall.json").read_text())
    assert overall["total_rows"] == 3
    assert overall["status_counts"] == {"error": 1, "feasible": 1, "timeout": 1}
    assert overall["solvers"] == ["hld"]

    cell_rows = list(csv.DictReader((out_dir / "cell_summary.csv").open(newline="")))
    weakly_row = next(row for row in cell_rows if row["correlation"] == "weakly")
    assert weakly_row["total_rows"] == "2"
    assert weakly_row["timeout_count"] == "1"
    assert weakly_row["timeout_rate"] == "0.500000"
    assert weakly_row["status_counts"] == "feasible=1;timeout=1"

    timeout_rows = list(csv.DictReader((out_dir / "top_timeout_cells.csv").open(newline="")))
    assert timeout_rows[0]["correlation"] == "weakly"
    assert timeout_rows[0]["timeout_rate"] == "0.500000"
