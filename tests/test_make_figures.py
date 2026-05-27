"""Tests for the figure-regeneration scaffold."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"


def test_make_figures_large_scale_scaling_ab(tmp_path: Path) -> None:
    results_csv = _write_results_csv(tmp_path / "results.csv")
    out_dir = tmp_path / "figs"
    archive_id = "final_experiments_TEST.tar.gz"
    archive_sha = "b" * 64

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
            "--archive-id",
            archive_id,
            "--archive-sha256",
            archive_sha,
            "--only",
            "large_scale_scaling_ab",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    pdf_path = out_dir / "large_scale_scaling_ab.pdf"
    meta_path = out_dir / "large_scale_scaling_ab.meta.json"
    assert pdf_path.exists(), completed.stderr
    assert pdf_path.stat().st_size > 1024
    assert meta_path.exists()

    meta = json.loads(meta_path.read_text())
    assert meta["figure"] == "large_scale_scaling_ab"
    assert meta["archive"] == {"id": archive_id, "sha256": archive_sha}
    assert set(meta["solvers"]) == {"hld", "partition_optimal", "highs"}
    # Stats keyed by solver -> N -> {n, median_wall_time_s, median_profit}.
    stats = meta["stats"]
    assert "hld" in stats and "1000" in stats["hld"]
    assert stats["hld"]["1000"]["n"] == 2
    assert stats["hld"]["1000"]["median_wall_time_s"] == 0.5
    assert "highs" not in {n for n in stats.get("highs", {})} or "100000" not in stats["highs"]


def _write_results_csv(path: Path) -> Path:
    fieldnames = [
        "solver",
        "instance_id",
        "N",
        "M",
        "correlation",
        "f",
        "seed",
        "status",
        "profit",
        "wall_time_s",
    ]
    rows: list[dict[str, str]] = []
    for solver, walls, profits in [
        ("hld", [0.4, 0.6], [1000, 1100]),
        ("hld", [6.0, 8.0], [10000, 11000]),
        ("hld", [60.0, 60.0], [100000, 110000]),
        ("partition_optimal", [1.0, 1.5], [950, 1050]),
        ("partition_optimal", [18.0, 20.0], [9000, 9500]),
        ("partition_optimal", [65.0, 70.0], [60000, 70000]),
        ("highs", [0.2, 0.3], [1010, 1110]),
        ("highs", [5.0, 6.0], [10100, 11100]),
        # No HiGHS rows at N=100k by design.
        ("bissa", [0.1, 0.15], [900, 950]),  # other solver — should NOT appear in plot
    ]:
        n = {("hld", 0.4): 1000, ("hld", 6.0): 10000, ("hld", 60.0): 100000}.get(
            (solver, walls[0])
        )
        if n is None:
            # second-batch derivations
            n_lookup = {
                ("partition_optimal", 1.0): 1000,
                ("partition_optimal", 18.0): 10000,
                ("partition_optimal", 65.0): 100000,
                ("highs", 0.2): 1000,
                ("highs", 5.0): 10000,
                ("bissa", 0.1): 1000,
            }
            n = n_lookup[(solver, walls[0])]
        for wall, profit in zip(walls, profits, strict=True):
            rows.append(
                {
                    "solver": solver,
                    "instance_id": f"{solver}/N{n}/seed.json.gz",
                    "N": str(n),
                    "M": "5",
                    "correlation": "uncorrelated",
                    "f": "0.5",
                    "seed": "0",
                    "status": "feasible",
                    "profit": str(profit),
                    "wall_time_s": str(wall),
                }
            )
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def test_make_figures_emits_pivot_figure_with_meta_sidecar(tmp_path: Path) -> None:
    paired_csv = _write_paired_csv(tmp_path / "paired_profit_gaps.csv")
    out_dir = tmp_path / "figs"
    archive_id = "final_experiments_TEST.tar.gz"
    archive_sha = "a" * 64

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--paired-csv",
            str(paired_csv),
            "--out-dir",
            str(out_dir),
            "--archive-id",
            archive_id,
            "--archive-sha256",
            archive_sha,
            "--only",
            "hld_vs_partition_paired_gains",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    pdf_path = out_dir / "hld_vs_partition_paired_gains.pdf"
    meta_path = out_dir / "hld_vs_partition_paired_gains.meta.json"
    assert pdf_path.exists(), f"missing {pdf_path}\nstdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
    assert pdf_path.stat().st_size > 1024, "PDF too small to be a real figure"
    assert meta_path.exists(), f"missing {meta_path}"

    meta = json.loads(meta_path.read_text())
    assert meta["figure"] == "hld_vs_partition_paired_gains"
    assert meta["archive"] == {"id": archive_id, "sha256": archive_sha}
    assert meta["source_csv"].endswith("paired_profit_gaps.csv")
    assert meta["script"].endswith("make_figures.py")
    assert "command" in meta and "--only" in meta["command"]
    assert "generated_at" in meta
    assert meta["n_paired"] == 4  # only the N=100000 partition_optimal rows
    assert "median_gain_pct" in meta["stats"]


def _write_paired_csv(path: Path) -> Path:
    fieldnames = [
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
    rows = [
        _row("partition_optimal", "100000", "17.19", "hld"),
        _row("partition_optimal", "100000", "30.00", "hld"),
        _row("partition_optimal", "100000", "-10.00", "partition_optimal"),
        _row("partition_optimal", "100000", "25.50", "hld"),
        _row("partition_optimal", "1000", "0.10", "hld"),  # N=1000 row, should be excluded
        _row("highs", "100000", "5.00", "hld"),  # baseline=highs, should be excluded
    ]
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return path


def _row(baseline: str, n: str, gain_pct: str, winner: str) -> dict[str, str]:
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
        "hld_wall_time_s": "60.0",
        "baseline_wall_time_s": "45.0",
        "winner": winner,
    }
