"""Regression tests for the K-sweep summarizer CLI."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_k_sweep.py"


def _row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "instance_id": "inversely_strongly/N100000_M10/inst.json.gz",
        "subset": "test",
        "N": 100000,
        "M": 10,
        "correlation": "inversely_strongly",
        "f": 0.5,
        "seed": 0,
        "solver": "hld",
        "k": 16,
        "status": "timeout",
        "profit": 1000,
        "total_cost": 999,
        "n_classes_selected": 10,
        "wall_time_s": 60.0,
        "solver_seed": 7,
        "time_limit_s": 60.0,
    }
    row.update(overrides)
    return row


def test_summarize_k_sweep_classifies_regimes(tmp_path: Path) -> None:
    sweep = tmp_path / "sweep.jsonl"
    out_dir = tmp_path / "summary"
    rows = [
        # K=2: infeasible for both solvers (profit 0 on every seed).
        _row(seed=0, solver="hld", k=2, profit=0),
        _row(seed=1, solver="hld", k=2, profit=0),
        _row(seed=0, solver="partition_optimal", k=2, profit=0),
        _row(seed=1, solver="partition_optimal", k=2, profit=0),
        # K=16: HLD sweet spot, beats Partition-Optimal on both seeds.
        _row(seed=0, solver="hld", k=16, profit=1000),
        _row(seed=1, solver="hld", k=16, profit=1200),
        _row(seed=0, solver="partition_optimal", k=16, profit=800),
        _row(seed=1, solver="partition_optimal", k=16, profit=900),
        # K=32: HLD quality declines above the sweet spot.
        _row(seed=0, solver="hld", k=32, profit=500),
        _row(seed=1, solver="hld", k=32, profit=600),
        _row(seed=0, solver="partition_optimal", k=32, profit=700),
        _row(seed=1, solver="partition_optimal", k=32, profit=650),
    ]
    sweep.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    completed = subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--sweep-jsonl",
            str(sweep),
            "--out-dir",
            str(out_dir),
            "--no-plots",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "total_rows: 12" in completed.stdout
    assert "f=0.5: best_k=16 feasible_ks=[16, 32] infeasible_ks=[2]" in completed.stdout

    summary = json.loads((out_dir / "analyses.json").read_text())
    info = summary["per_f"]["0.5"]
    assert info["best_k"] == 16
    assert info["feasible_ks"] == [16, 32]
    assert info["infeasible_ks"] == [2]

    by_k = {cell["k"]: cell for cell in summary["cells"]}
    # K=2 infeasible: no feasible instances, no paired gain.
    assert by_k[2]["hld_feasible"] == 0
    assert by_k[2]["po_feasible"] == 0
    assert by_k[2]["paired_median_gain"] is None
    # K=16 sweet spot: HLD wins both seeds; median gain = median(0.25, 0.333).
    assert by_k[16]["paired_win"] == 2
    assert by_k[16]["paired_lose"] == 0
    assert by_k[16]["hld_median_profit"] == 1100
    assert abs(by_k[16]["paired_median_gain"] - 0.2916666666666667) < 1e-9
    # K=32 decline: HLD below its K=16 profit and loses to Partition-Optimal.
    assert by_k[32]["hld_median_profit"] == 550
    assert by_k[32]["paired_lose"] == 2

    report = (out_dir / "REPORT.md").read_text()
    assert "Small-K is infeasible" in report
    assert "Sweet spot at moderate K" in report
