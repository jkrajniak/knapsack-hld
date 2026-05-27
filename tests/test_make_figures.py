"""Tests for the figure-regeneration scaffold."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"


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
