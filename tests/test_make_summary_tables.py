"""Smoke tests for `scripts/make_summary_tables.py`.

Verify the per-N aggregation logic, the LaTeX-safe escaping
(`\\,` thousands separator, no double-escape), and the fixed set
of baselines required by §3.9 of the manuscript.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "make_summary_tables.py"

FIELDS = [
    "baseline_solver",
    "instance_id",
    "N",
    "M",
    "correlation",
    "f",
    "seed",
    "hld_status",
    "baseline_status",
    "hld_profit",
    "baseline_profit",
    "hld_vs_baseline_gain_pct",
    "hld_wall_time_s",
    "baseline_wall_time_s",
    "winner",
]


def _make_row(
    baseline: str, N: int, gain: float, seed: int = 0
) -> dict[str, object]:
    return {
        "baseline_solver": baseline,
        "instance_id": f"foo/N{N}/inst_seed{seed}.json.gz",
        "N": str(N),
        "M": "5",
        "correlation": "uncorrelated",
        "f": "0.1",
        "seed": str(seed),
        "hld_status": "feasible",
        "baseline_status": "feasible",
        "hld_profit": "100",
        "baseline_profit": "100",
        "hld_vs_baseline_gain_pct": str(gain),
        "hld_wall_time_s": "0.1",
        "baseline_wall_time_s": "0.1",
        "winner": "hld" if gain > 0 else "baseline" if gain < 0 else "tie",
    }


def _write_paired(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def test_partition_summary_aggregates_per_N(tmp_path: Path) -> None:
    paired = tmp_path / "paired.csv"
    rows: list[dict[str, object]] = []
    for N in (1_000, 10_000, 100_000):
        for seed in range(3):
            rows.append(_make_row("partition_optimal", N, gain=10.0 + seed))
    rows.append(_make_row("highs", 1_000, gain=-0.5))
    rows.append(_make_row("highs", 10_000, gain=-0.1))
    _write_paired(paired, rows)

    out = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--paired-csv", str(paired), "--out-dir", str(out)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    pt = (out / "hld_vs_partition_summary.tex").read_text()
    assert r"\begin{tabular}" in pt
    assert r"$1\,000$" in pt
    assert r"$10\,000$" in pt
    assert r"$100\,000$" in pt
    assert r"\\," not in pt, "thousands separator must not be double-escaped"

    ht = (out / "hld_vs_highs_summary.tex").read_text()
    assert r"$1\,000$" in ht
    assert r"$10\,000$" in ht
    assert "$100" not in ht, "HiGHS table must not include N=100k row"


def test_missing_baseline_raises(tmp_path: Path) -> None:
    paired = tmp_path / "paired.csv"
    _write_paired(paired, [_make_row("bissa", 1_000, gain=1.0)])
    out = tmp_path / "out"
    completed = subprocess.run(
        [sys.executable, str(SCRIPT), "--paired-csv", str(paired), "--out-dir", str(out)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode != 0
    assert "no rows for baseline" in completed.stderr
