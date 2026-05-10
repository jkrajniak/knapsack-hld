"""Paper-table exporter for final experiment summaries."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_paper_tables.py"


def test_make_paper_tables_exports_compact_csv_and_latex(tmp_path: Path) -> None:
    final_summary = tmp_path / "summary"
    sensitivity_summary = tmp_path / "time_limit_sensitivity_summary"
    out_dir = tmp_path / "paper_tables"
    final_summary.mkdir()
    sensitivity_summary.mkdir()

    _write_csv(
        final_summary / "cell_summary.csv",
        [
            {
                "solver": "hld",
                "N": "100000",
                "M": "20",
                "correlation": "inversely_strongly",
                "f": "0.5",
                "total_rows": "35",
                "status_counts": "timeout=35",
                "feasible_count": "0",
                "timeout_count": "35",
                "error_count": "0",
                "timeout_rate": "1.000000",
                "mean_wall_time_s": "64.972311",
                "median_wall_time_s": "64.974029",
                "max_wall_time_s": "65.468343",
            }
        ],
    )
    _write_csv(
        sensitivity_summary / "time_limit_status.csv",
        [
            {
                "time_limit_s": "60",
                "total_rows": "210",
                "status_counts": "feasible=1;timeout=209",
                "feasible_count": "1",
                "timeout_count": "209",
                "timeout_rate": "0.995238",
            }
        ],
    )
    _write_csv(
        sensitivity_summary / "profit_gains.csv",
        [
            {
                "comparison": "60->300",
                "n": "210",
                "mean_gain": "1.848913",
                "median_gain": "1.418501",
                "max_gain": "6.651345",
            }
        ],
    )

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--final-summary-dir",
            str(final_summary),
            "--sensitivity-summary-dir",
            str(sensitivity_summary),
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
    assert "final_cell_status_runtime.csv: 1 rows" in completed.stdout
    assert "sensitivity_profit_gains.csv: 1 rows" in completed.stdout

    final_rows = list(csv.DictReader((out_dir / "final_cell_status_runtime.csv").open()))
    assert final_rows[0] == {
        "N": "100000",
        "M": "20",
        "correlation": "inversely_strongly",
        "f": "0.5",
        "total_rows": "35",
        "feasible_count": "0",
        "timeout_count": "35",
        "timeout_rate_pct": "100.0",
        "median_wall_time_s": "64.97",
        "max_wall_time_s": "65.47",
    }

    gain_rows = list(csv.DictReader((out_dir / "sensitivity_profit_gains.csv").open()))
    assert gain_rows[0]["median_gain_pct"] == "141.9"
    assert "60 to 300" in (out_dir / "sensitivity_profit_gains.tex").read_text()
    assert "99.5" in (out_dir / "sensitivity_time_limit_status.tex").read_text()


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
