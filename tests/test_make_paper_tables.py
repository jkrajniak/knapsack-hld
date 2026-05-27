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
    comparison_summary = tmp_path / "comparison_summary"
    out_dir = tmp_path / "paper_tables"
    final_summary.mkdir()
    sensitivity_summary.mkdir()
    comparison_summary.mkdir()

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
    _write_csv(
        comparison_summary / "paired_profit_gaps.csv",
        _synthetic_paired_rows(),
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
            "--comparison-summary-dir",
            str(comparison_summary),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert f"out_dir: {out_dir}" in completed.stdout
    assert "final_cell_status_runtime.csv: 1 rows" in completed.stdout
    assert "sensitivity_profit_gains.csv: 1 rows" in completed.stdout
    assert "hld_vs_partition_summary.csv: 2 rows" in completed.stdout
    assert "hld_vs_highs_summary.csv: 2 rows" in completed.stdout

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

    po_rows = list(csv.DictReader((out_dir / "hld_vs_partition_summary.csv").open()))
    po_by_n = {row["N"]: row for row in po_rows}
    assert set(po_by_n) == {"1000", "100000"}
    assert po_by_n["100000"]["paired_rows"] == "4"
    assert po_by_n["100000"]["hld_wins"] == "3"
    assert po_by_n["100000"]["baseline_wins"] == "1"
    assert po_by_n["100000"]["ties"] == "0"
    # Fixture gains at N=100000 are (17.19, 30.00, -10.00, 25.50);
    # sorted (-10, 17.19, 25.5, 30), median = (17.19 + 25.50) / 2 = 21.345 -> +21.34.
    assert po_by_n["100000"]["median_gain_pct"] == "+21.34"
    assert po_by_n["100000"]["median_hld_wall_time_s"] == "60.00"
    # N=1000 PO gains are (0.0, 0.1, -0.05, 0.02): one tie, three non-zero.
    assert po_by_n["1000"]["ties"] == "1"
    # |median| < 1, so 4 dp output.
    assert po_by_n["1000"]["median_gain_pct"].startswith("+0.01") or po_by_n["1000"][
        "median_gain_pct"
    ].startswith("-0.01")

    highs_rows = list(csv.DictReader((out_dir / "hld_vs_highs_summary.csv").open()))
    highs_by_n = {row["N"]: row for row in highs_rows}
    assert set(highs_by_n) == {"1000", "10000"}
    assert highs_by_n["10000"]["paired_rows"] == "3"
    # Fixture gains at N=10000 for HiGHS are (-0.003, -15.0, 0.001);
    # sorted (-15.0, -0.003, 0.001), median = -0.003 -> 4 dp output -0.0030.
    assert highs_by_n["10000"]["median_gain_pct"] == "-0.0030"
    # N=1000 HiGHS gains (-0.09, -0.50, -0.01, 0.0); median (-0.09 + -0.01)/2 = -0.05.
    assert highs_by_n["1000"]["median_gain_pct"] == "-0.0500"

    po_tex = (out_dir / "hld_vs_partition_summary.tex").read_text()
    assert "+21.34" in po_tex
    assert "Paired rows" in po_tex
    highs_tex = (out_dir / "hld_vs_highs_summary.tex").read_text()
    assert "-0.0030" in highs_tex


def _synthetic_paired_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    rows.extend(
        _row("partition_optimal", "1000", gain, hld_wall=0.5, baseline_wall=0.3)
        for gain in ("0.00", "0.10", "-0.05", "0.02")
    )
    rows.extend(
        _row("partition_optimal", "100000", gain, hld_wall=60.0, baseline_wall=45.0)
        for gain in ("17.19", "30.00", "-10.00", "25.50")
    )
    rows.extend(
        _row("highs", "1000", gain, hld_wall=0.4, baseline_wall=0.2)
        for gain in ("-0.09", "-0.50", "-0.01", "0.00")
    )
    rows.extend(
        _row("highs", "10000", gain, hld_wall=10.0, baseline_wall=5.0)
        for gain in ("-0.003", "-15.0", "0.0010")
    )
    return rows


def _row(
    baseline: str,
    n: str,
    gain_pct: str,
    *,
    hld_wall: float,
    baseline_wall: float,
) -> dict[str, str]:
    gain_value = float(gain_pct)
    if gain_value > 0:
        winner = "hld"
    elif gain_value < 0:
        winner = baseline
    else:
        winner = "tie"
    return {
        "baseline_solver": baseline,
        "instance_id": f"{baseline}/N{n}/seed.json.gz",
        "N": n,
        "M": "5",
        "correlation": "uncorrelated",
        "f": "0.5",
        "seed": "0",
        "hld_status": "feasible",
        "baseline_status": "feasible",
        "hld_profit": "1000",
        "baseline_profit": "1000",
        "hld_vs_baseline_gain_pct": gain_pct,
        "hld_wall_time_s": f"{hld_wall:.6f}",
        "baseline_wall_time_s": f"{baseline_wall:.6f}",
        "winner": winner,
    }


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
