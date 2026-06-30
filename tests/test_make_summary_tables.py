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


def _make_row(baseline: str, N: int, gain: float, seed: int = 0) -> dict[str, object]:
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


def _write_sensitivity(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "time_limit_s",
        "instance_id",
        "subset",
        "N",
        "M",
        "correlation",
        "f",
        "seed",
        "solver",
        "status",
        "profit",
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def test_hld_300s_table_recomputes_gains(tmp_path: Path) -> None:
    """HLD@300s table swaps in 300s profit and recomputes paired gains."""
    paired = tmp_path / "paired.csv"
    rows: list[dict[str, object]] = []
    # Three heuristics at N=100k, each with a baseline_profit of 100 (60s gain irrelevant).
    for baseline in ("greedy_max_ratio", "trs2008", "bissa"):
        for seed in range(2):
            r = _make_row(baseline, 100_000, gain=-5.0, seed=seed)
            r["baseline_profit"] = "100"
            rows.append(r)
    # A small-N row that must be ignored by the 300s table (only N=100k in scope).
    rows.append(_make_row("bissa", 1_000, gain=1.0))
    # main() also emits the existing per-baseline tables, which require these.
    for N in (1_000, 10_000, 100_000):
        rows.append(_make_row("partition_optimal", N, gain=5.0))
    for N in (1_000, 10_000):
        rows.append(_make_row("highs", N, gain=-0.1))
    _write_paired(paired, rows)

    sens = tmp_path / "sensitivity.csv"
    sens_rows: list[dict[str, object]] = []
    for _baseline in ("greedy_max_ratio", "trs2008", "bissa"):
        for seed in range(2):
            # HLD@300s profit 110 -> +10% gain vs baseline_profit 100.
            sens_rows.append(
                {
                    "time_limit_s": "300",
                    "instance_id": f"foo/N100000/inst_seed{seed}.json.gz",
                    "solver": "hld",
                    "status": "feasible",
                    "profit": "110",
                }
            )
            # 60s rows for the same instances must be ignored.
            sens_rows.append(
                {
                    "time_limit_s": "60",
                    "instance_id": f"foo/N100000/inst_seed{seed}.json.gz",
                    "solver": "hld",
                    "status": "feasible",
                    "profit": "999",
                }
            )
    _write_sensitivity(sens, sens_rows)

    out = tmp_path / "out"
    completed = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--paired-csv",
            str(paired),
            "--out-dir",
            str(out),
            "--hld-300s-csv",
            str(sens),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr

    tex = (out / "hld_vs_baselines_300s.tex").read_text()
    assert r"\begin{tabular}" in tex
    assert "Greedy-MaxRatio" in tex and "TRS-2008" in tex and "BISSA" in tex
    assert "+10.00" in tex, "300s gain must be recomputed from 300s profit"
    assert "$1\\,000$" not in tex, "300s table is N=100k only"


def test_hld_300s_infeasible_counts_as_loss(tmp_path: Path) -> None:
    """An HLD@300s timeout against a feasible heuristic is a -100% loss."""
    from importlib import util as _util

    spec = _util.spec_from_file_location("mst", SCRIPT)
    mst = _util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mst)

    paired_rows = [
        {
            "baseline_solver": "bissa",
            "instance_id": "x/inst.json.gz",
            "N": "100000",
            "baseline_status": "feasible",
            "baseline_profit": "100",
            "hld_status": "feasible",
            "hld_profit": "100",
            "hld_vs_baseline_gain_pct": "0",
            "winner": "tie",
        }
    ]
    hld_300s = {"x/inst.json.gz": ("timeout", 0.0)}
    synth, unmatched = mst.build_300s_rows(paired_rows, hld_300s)
    assert unmatched == 0
    assert len(synth) == 1
    assert float(synth[0]["hld_vs_baseline_gain_pct"]) == -100.0
    assert synth[0]["winner"] == "baseline"


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
