"""Aggregator for HLD `fallback_equal_split` (Task 3.4.1)."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_fallback_stats.py"

FIELDS = [
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
    "total_cost",
    "n_classes_selected",
    "wall_time_s",
    "n_iter",
    "alpha",
    "k",
    "lambda_max",
    "class_ordering",
    "fallback_equal_split",
    "error_message",
]


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "instance_id": "strongly/N1000_M5/x.json.gz",
        "subset": "test",
        "N": "1000",
        "M": "5",
        "correlation": "strongly",
        "f": "0.5",
        "seed": "1",
        "solver": "hld",
        "status": "feasible",
        "profit": "1000",
        "total_cost": "100",
        "n_classes_selected": "10",
        "wall_time_s": "0.5",
        "n_iter": "20",
        "alpha": "0.9",
        "k": "8",
        "lambda_max": "10",
        "class_ordering": "sequential",
        "fallback_equal_split": "0",
        "error_message": "",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run(results_csv: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--results-csv",
            str(results_csv),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_summarize_fallback_stats_counts_per_cell(tmp_path: Path) -> None:
    """Per-cell aggregation reports rows / fallback count / rate."""
    csv_path = tmp_path / "results.csv"
    rows = [
        _row(seed="1", fallback_equal_split="0"),
        _row(seed="2", fallback_equal_split="1"),
        _row(seed="3", fallback_equal_split="0"),
        _row(seed="4", correlation="uncorrelated", fallback_equal_split="1"),
        _row(seed="5", correlation="uncorrelated", fallback_equal_split="1"),
    ]
    _write_csv(csv_path, rows)

    out = tmp_path / "out"
    completed = _run(csv_path, out)
    assert completed.returncode == 0, completed.stderr

    overall = json.loads((out / "fallback_overall.json").read_text())
    assert overall["n_rows"] == 5
    assert overall["n_fallbacks"] == 3
    assert overall["fallback_rate"] == 0.6

    with (out / "fallback_stats.csv").open() as fh:
        per_cell = {(r["correlation"], r["f"]): r for r in csv.DictReader(fh)}

    assert len(per_cell) == 2
    s = per_cell[("strongly", "0.5")]
    assert int(s["n_rows"]) == 3 and int(s["n_fallbacks"]) == 1
    u = per_cell[("uncorrelated", "0.5")]
    assert int(u["n_rows"]) == 2 and int(u["n_fallbacks"]) == 2


def test_summarize_fallback_stats_ignores_non_hld_and_missing_values(tmp_path: Path) -> None:
    csv_path = tmp_path / "results.csv"
    rows = [
        _row(seed="1", solver="hld", fallback_equal_split="1"),
        _row(seed="2", solver="highs", fallback_equal_split=""),  # ignored
        _row(seed="3", solver="hld", fallback_equal_split=""),  # ignored (no signal)
    ]
    _write_csv(csv_path, rows)

    out = tmp_path / "out"
    completed = _run(csv_path, out)
    assert completed.returncode == 0, completed.stderr

    overall = json.loads((out / "fallback_overall.json").read_text())
    # Only the one row with hld + concrete 0/1 fallback contributes.
    assert overall["n_rows"] == 1
    assert overall["n_fallbacks"] == 1
    assert overall["fallback_rate"] == 1.0


def test_summarize_fallback_stats_per_n_summary(tmp_path: Path) -> None:
    """The script also breaks down counts by N scale."""
    csv_path = tmp_path / "results.csv"
    rows = [
        _row(seed="1", N="1000", fallback_equal_split="0"),
        _row(seed="2", N="1000", fallback_equal_split="0"),
        _row(seed="3", N="100000", fallback_equal_split="1"),
        _row(seed="4", N="100000", fallback_equal_split="0"),
    ]
    _write_csv(csv_path, rows)

    out = tmp_path / "out"
    completed = _run(csv_path, out)
    assert completed.returncode == 0, completed.stderr

    overall = json.loads((out / "fallback_overall.json").read_text())
    by_n = overall["by_N"]
    assert by_n["1000"]["n_fallbacks"] == 0
    assert by_n["1000"]["n_rows"] == 2
    assert by_n["100000"]["n_fallbacks"] == 1
    assert by_n["100000"]["n_rows"] == 2


def test_summarize_fallback_stats_handles_pre_schema_csv(tmp_path: Path) -> None:
    """Old CSVs (no fallback_equal_split column) must not crash; they yield n_rows=0."""
    csv_path = tmp_path / "results.csv"
    # Write a CSV that simply omits the new column (mimics pre-3.4.1 archives).
    old_fields = [f for f in FIELDS if f != "fallback_equal_split"]
    with csv_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=old_fields)
        writer.writeheader()
        writer.writerow({k: _row()[k] for k in old_fields})

    out = tmp_path / "out"
    completed = _run(csv_path, out)
    assert completed.returncode == 0, completed.stderr
    overall = json.loads((out / "fallback_overall.json").read_text())
    assert overall["n_rows"] == 0
    assert overall["n_fallbacks"] == 0
