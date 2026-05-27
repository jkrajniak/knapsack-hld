"""Per-instance appendix tables (Task 3.6.1)."""

from __future__ import annotations

import csv
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_appendix_tables.py"

PAIRED_FIELDS = [
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


def _paired_row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "baseline_solver": "highs",
        "instance_id": "strongly/N1000_M5/x.json.gz",
        "N": "1000",
        "M": "5",
        "correlation": "strongly",
        "f": "0.5",
        "seed": "1",
        "hld_status": "feasible",
        "baseline_status": "optimal",
        "hld_profit": "1000",
        "baseline_profit": "1000",
        "hld_vs_baseline_gain_pct": "0.0",
        "hld_wall_time_s": "0.5",
        "baseline_wall_time_s": "1.5",
        "winner": "tie",
    }
    base.update(overrides)
    return base


def _write_paired(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=PAIRED_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _run(paired_csv: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--paired-csv",
            str(paired_csv),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_appendix_uses_highs_for_low_n_and_partition_optimal_for_n100k(tmp_path: Path) -> None:
    """Reference solver is HiGHS at N ≤ 10000, Partition-Optimal at N = 100000."""
    paired = tmp_path / "paired.csv"

    def make_rows(n: str, baseline: str, count: int) -> list[dict[str, object]]:
        return [
            _paired_row(
                instance_id=f"strongly/N{n}_M5/inst{i}.json.gz",
                N=n,
                seed=str(i),
                baseline_solver=baseline,
            )
            for i in range(count)
        ]

    rows: list[dict[str, object]] = []
    rows += make_rows("1000", "highs", 3)
    rows += make_rows("1000", "bissa", 3)  # noise — wrong baseline
    rows += make_rows("10000", "highs", 3)
    rows += make_rows("100000", "partition_optimal", 3)
    rows += make_rows("100000", "highs", 3)  # noise — at N=100k HiGHS DNF, ignore
    _write_paired(paired, rows)

    out = tmp_path / "out"
    completed = _run(paired, out)
    assert completed.returncode == 0, completed.stderr

    for n, expected_baseline in (("1000", "highs"), ("10000", "highs"), ("100000", "partition_optimal")):
        csv_path = out / f"appendix_N{n}.csv"
        assert csv_path.exists()
        with csv_path.open() as fh:
            data = list(csv.DictReader(fh))
        baselines = {row["reference_solver"] for row in data}
        assert baselines == {expected_baseline}, (n, baselines)


def test_appendix_aggregates_by_cell(tmp_path: Path) -> None:
    """One row per (M, correlation, f) cell at each scale, with median/worst gap."""
    paired = tmp_path / "paired.csv"
    # Two cells × 3 seeds each at N=1000, baseline HiGHS, varying gain.
    rows = []
    for cell_idx, (corr, f) in enumerate([("strongly", "0.5"), ("weakly", "0.1")]):
        for seed in range(3):
            rows.append(
                _paired_row(
                    instance_id=f"{corr}/N1000_M5/i{cell_idx}_{seed}.json.gz",
                    correlation=corr,
                    f=f,
                    seed=str(seed),
                    # gain in {-0.5, 0.0, +0.5} for cell 0; {+1, +2, +3} for cell 1.
                    hld_vs_baseline_gain_pct=str([-0.5, 0.0, 0.5, 1.0, 2.0, 3.0][cell_idx * 3 + seed]),
                    hld_wall_time_s=str(0.1 * (seed + 1)),
                    baseline_wall_time_s=str(0.5 * (seed + 1)),
                )
            )
    _write_paired(paired, rows)

    out = tmp_path / "out"
    completed = _run(paired, out)
    assert completed.returncode == 0, completed.stderr

    with (out / "appendix_N1000.csv").open() as fh:
        data = {(r["correlation"], r["f"]): r for r in csv.DictReader(fh)}

    assert len(data) == 2
    cell0 = data[("strongly", "0.5")]
    cell1 = data[("weakly", "0.1")]

    assert int(cell0["n_paired"]) == 3
    assert float(cell0["median_gain_pct"]) == 0.0
    assert float(cell0["worst_loss_pct"]) == -0.5  # min gain
    assert float(cell0["worst_gain_pct"]) == 0.5  # max gain

    assert int(cell1["n_paired"]) == 3
    assert float(cell1["median_gain_pct"]) == 2.0
    assert float(cell1["worst_loss_pct"]) == 1.0
    assert float(cell1["worst_gain_pct"]) == 3.0


def test_appendix_emits_latex_per_scale(tmp_path: Path) -> None:
    paired = tmp_path / "paired.csv"
    rows = [
        _paired_row(
            instance_id=f"strongly/N100000_M5/i{i}.json.gz",
            N="100000",
            seed=str(i),
            baseline_solver="partition_optimal",
            hld_vs_baseline_gain_pct=str(10.0 + i),
        )
        for i in range(3)
    ]
    _write_paired(paired, rows)

    out = tmp_path / "out"
    completed = _run(paired, out)
    assert completed.returncode == 0, completed.stderr

    tex = (out / "appendix_N100000.tex").read_text()
    assert "\\begin{tabular}" in tex
    assert "partition_optimal" in tex
    assert "strongly" in tex


def test_appendix_handles_missing_n_gracefully(tmp_path: Path) -> None:
    """If a scale has no paired rows, the script must not crash; it just skips it."""
    paired = tmp_path / "paired.csv"
    rows = [
        _paired_row(
            instance_id="strongly/N1000_M5/a.json.gz",
            N="1000",
            baseline_solver="highs",
        ),
    ]
    _write_paired(paired, rows)

    out = tmp_path / "out"
    completed = _run(paired, out)
    assert completed.returncode == 0, completed.stderr
    assert (out / "appendix_N1000.csv").exists()
    assert not (out / "appendix_N10000.csv").exists()
    assert not (out / "appendix_N100000.csv").exists()
