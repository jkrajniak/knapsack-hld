"""Final experiment artifact archiver stores completed outputs outside git."""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "archive_final_experiments.sh"


def test_archive_final_experiments_dry_run_uses_private_artifact_dir() -> None:
    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--dry-run",
            "--run-id",
            "20260510T104515Z",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "result_dir: results/final_experiments" in completed.stdout
    assert "artifact_dir: ../knapsack-artifacts/final_experiments" in completed.stdout
    assert (
        "tar -czf ../knapsack-artifacts/final_experiments/final_experiments_20260510T104515Z.tar.gz"
        in completed.stdout
    )
    assert "results/final_experiments/comparison_summary" in completed.stdout
    assert "results/final_experiments/heuristic_baselines_refreshed.csv" in completed.stdout
    assert "results/final_experiments/partition_optimal_refreshed.csv" in completed.stdout


def test_archive_final_experiments_writes_tarball_and_checksum(tmp_path: Path) -> None:
    result_dir = tmp_path / "results" / "final_experiments"
    summary_dir = result_dir / "summary"
    artifact_dir = tmp_path / "artifacts"
    summary_dir.mkdir(parents=True)
    (result_dir / "results.csv").write_text("status\nfeasible\n")
    (summary_dir / "overall.json").write_text("{}\n")
    (result_dir / "time_limit_sensitivity.csv").write_text("status\ntimeout\n")
    (result_dir / "heuristic_baselines.csv").write_text("status\nfeasible\n")
    (result_dir / "heuristic_baselines_refreshed.csv").write_text("status\nfeasible\n")
    (result_dir / "partition_optimal_refreshed.csv").write_text("status\nfeasible\n")
    (result_dir / "highs_baseline_maxN10000.csv").write_text("status\noptimal\n")
    (result_dir / "comparison_summary").mkdir()
    (result_dir / "comparison_summary" / "aggregate_profit_gaps.csv").write_text("ok\n")

    completed = subprocess.run(
        [
            "bash",
            str(SCRIPT),
            "--run-id",
            "test_run",
            "--result-dir",
            str(result_dir),
            "--artifact-dir",
            str(artifact_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    archive = artifact_dir / "final_experiments_test_run.tar.gz"
    checksum = artifact_dir / "final_experiments_test_run.tar.gz.sha256"
    assert completed.returncode == 0
    assert archive.exists()
    assert checksum.exists()
    assert archive.name in checksum.read_text()
