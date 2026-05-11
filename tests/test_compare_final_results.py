"""Paired HLD-vs-baseline comparison summaries."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "compare_final_results.py"


def test_compare_final_results_writes_paired_and_aggregate_summaries(tmp_path: Path) -> None:
    hld_csv = tmp_path / "hld.csv"
    baseline_csv = tmp_path / "baseline.csv"
    out_dir = tmp_path / "comparison_summary"
    _write_csv(
        hld_csv,
        [
            _row(instance_id="a.json.gz", solver="hld", profit="100", wall_time_s="10"),
            _row(instance_id="b.json.gz", solver="hld", profit="80", wall_time_s="20"),
        ],
    )
    _write_csv(
        baseline_csv,
        [
            _row(instance_id="a.json.gz", solver="greedy_max_ratio", profit="90", wall_time_s="1"),
            _row(
                instance_id="b.json.gz", solver="greedy_max_ratio", profit="100", wall_time_s="1.5"
            ),
            _row(
                instance_id="a.json.gz",
                solver="highs",
                profit="120",
                status="optimal",
                wall_time_s="60",
            ),
        ],
    )

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--hld-csv",
            str(hld_csv),
            "--baseline-csv",
            str(baseline_csv),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert f"out_dir: {out_dir}" in completed.stdout
    assert "paired_profit_gaps.csv: 3 rows" in completed.stdout
    assert "aggregate_profit_gaps.csv: 2 rows" in completed.stdout

    paired_rows = list(csv.DictReader((out_dir / "paired_profit_gaps.csv").open(newline="")))
    greedy_a = next(
        row
        for row in paired_rows
        if row["baseline_solver"] == "greedy_max_ratio" and row["instance_id"] == "a.json.gz"
    )
    assert greedy_a["hld_profit"] == "100"
    assert greedy_a["baseline_profit"] == "90"
    assert greedy_a["hld_vs_baseline_gain_pct"] == "11.111111"
    assert greedy_a["winner"] == "hld"

    aggregate_rows = list(csv.DictReader((out_dir / "aggregate_profit_gaps.csv").open(newline="")))
    greedy_aggregate = next(
        row for row in aggregate_rows if row["baseline_solver"] == "greedy_max_ratio"
    )
    assert greedy_aggregate["paired_rows"] == "2"
    assert greedy_aggregate["hld_wins"] == "1"
    assert greedy_aggregate["baseline_wins"] == "1"
    assert greedy_aggregate["mean_hld_vs_baseline_gain_pct"] == "-4.444445"

    status_rows = list(csv.DictReader((out_dir / "solver_status_runtime.csv").open(newline="")))
    highs_status = next(row for row in status_rows if row["solver"] == "highs")
    assert highs_status["status_counts"] == "optimal=1"
    assert highs_status["median_wall_time_s"] == "60.000000"

    latex = (out_dir / "aggregate_profit_gaps.tex").read_text()
    assert "greedy\\_max\\_ratio" in latex


def _row(**overrides: str) -> dict[str, str]:
    row = {
        "instance_id": "a.json.gz",
        "subset": "test",
        "N": "1000",
        "M": "5",
        "correlation": "weakly",
        "f": "0.5",
        "seed": "1",
        "solver": "hld",
        "status": "feasible",
        "profit": "100",
        "total_cost": "80",
        "n_classes_selected": "5",
        "wall_time_s": "10",
        "n_iter": "",
        "alpha": "",
        "k": "",
        "lambda_max": "",
        "error_message": "",
    }
    row.update(overrides)
    return row


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
