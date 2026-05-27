"""Summariser for the class-ordering ablation (Task 3.3.2)."""

from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "summarize_class_ordering.py"

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
    "error_message",
]


def _row(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "instance_id": "strongly/N100000_M5/x.json.gz",
        "subset": "test",
        "N": "100000",
        "M": "5",
        "correlation": "strongly",
        "f": "0.75",
        "seed": "1",
        "solver": "hld",
        "status": "feasible",
        "profit": "1000",
        "total_cost": "100",
        "n_classes_selected": "10",
        "wall_time_s": "12.3",
        "n_iter": "35",
        "alpha": "0.998",
        "k": "58",
        "lambda_max": "80.745",
        "class_ordering": "sequential",
        "error_message": "",
    }
    base.update(overrides)
    return base


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _run(results_dir: Path, out_dir: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            "uv",
            "run",
            "python",
            str(SCRIPT),
            "--results-dir",
            str(results_dir),
            "--out-dir",
            str(out_dir),
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )


def test_summarize_class_ordering_identical_csvs_yield_zero_gap(tmp_path: Path) -> None:
    """When all three orderings produce identical profits, every gap is exactly zero."""
    results = tmp_path / "in"
    rows = [
        _row(instance_id="a.json.gz", seed="1", profit="1000"),
        _row(instance_id="b.json.gz", seed="2", profit="2000"),
        _row(instance_id="c.json.gz", seed="3", profit="3000"),
    ]
    for ordering in ("sequential", "random", "adversarial"):
        _write_csv(
            results / f"{ordering}.csv",
            [{**row, "class_ordering": ordering} for row in rows],
        )

    out = tmp_path / "out"
    completed = _run(results, out)
    assert completed.returncode == 0, completed.stderr

    overall = json.loads((out / "class_ordering_overall.json").read_text())
    for ordering in ("sequential", "random", "adversarial"):
        block = overall[ordering]
        assert block["n_instances"] == 3
        assert block["mean_gap_pct"] == 0.0
        assert block["std_gap_pct"] == 0.0
        assert block["median_gap_pct"] == 0.0
        assert block["win_rate"] == 1.0  # all three tie -> all three "win"
        assert block["timeout_count"] == 0


def test_summarize_class_ordering_one_winner_yields_positive_gap(tmp_path: Path) -> None:
    """If `random` always beats the others, it wins every instance."""
    results = tmp_path / "in"

    def make(ordering: str, profits: list[int]) -> list[dict[str, object]]:
        return [
            _row(
                instance_id=f"inst{i}.json.gz",
                seed=str(i),
                profit=str(p),
                class_ordering=ordering,
            )
            for i, p in enumerate(profits, start=1)
        ]

    _write_csv(results / "sequential.csv", make("sequential", [100, 200, 300]))
    _write_csv(results / "random.csv", make("random", [110, 220, 330]))
    _write_csv(results / "adversarial.csv", make("adversarial", [90, 180, 270]))

    out = tmp_path / "out"
    completed = _run(results, out)
    assert completed.returncode == 0, completed.stderr

    overall = json.loads((out / "class_ordering_overall.json").read_text())
    assert overall["random"]["win_rate"] == 1.0
    assert overall["sequential"]["win_rate"] == 0.0
    assert overall["adversarial"]["win_rate"] == 0.0
    # All three sequential profits are 10/11 of best -> gap = ~9.09 %
    assert abs(overall["sequential"]["mean_gap_pct"] - (10 / 110 * 100)) < 1e-6
    # All three adversarial profits are 90/110 of best -> gap = ~18.18 %
    assert abs(overall["adversarial"]["mean_gap_pct"] - (20 / 110 * 100)) < 1e-6


def test_summarize_class_ordering_per_cell_csv_groups_by_cell(tmp_path: Path) -> None:
    """`class_ordering_per_cell.csv` aggregates by (N, M, correlation, f, ordering)."""
    results = tmp_path / "in"
    cells = [
        ("strongly", "0.5"),
        ("uncorrelated", "0.5"),
    ]
    for ordering, bonus in (("sequential", 0), ("random", 10), ("adversarial", -5)):
        rows = []
        for cell_idx, (corr, f) in enumerate(cells):
            for seed in range(2):
                rows.append(
                    _row(
                        instance_id=f"{corr}/{cell_idx}_{seed}.json.gz",
                        correlation=corr,
                        f=f,
                        seed=str(seed),
                        profit=str(1000 + bonus + cell_idx * 100),
                        class_ordering=ordering,
                    )
                )
        _write_csv(results / f"{ordering}.csv", rows)

    out = tmp_path / "out"
    completed = _run(results, out)
    assert completed.returncode == 0, completed.stderr

    per_cell_path = out / "class_ordering_per_cell.csv"
    with per_cell_path.open() as fh:
        per_cell_rows = list(csv.DictReader(fh))
    assert len(per_cell_rows) == 2 * 3  # 2 cells * 3 orderings
    keys = {(r["correlation"], r["f"], r["class_ordering"]) for r in per_cell_rows}
    assert ("strongly", "0.5", "random") in keys
    assert ("uncorrelated", "0.5", "adversarial") in keys


def test_summarize_class_ordering_requires_matching_instance_sets(tmp_path: Path) -> None:
    """If one ordering is missing an instance the summariser must fail loudly."""
    results = tmp_path / "in"
    _write_csv(
        results / "sequential.csv",
        [_row(instance_id="a.json.gz", class_ordering="sequential")],
    )
    _write_csv(
        results / "random.csv",
        [_row(instance_id="b.json.gz", class_ordering="random")],
    )
    _write_csv(
        results / "adversarial.csv",
        [_row(instance_id="a.json.gz", class_ordering="adversarial")],
    )

    out = tmp_path / "out"
    completed = _run(results, out)
    assert completed.returncode != 0
    assert "instance set mismatch" in completed.stderr.lower()


def test_summarize_class_ordering_writes_latex_table(tmp_path: Path) -> None:
    """`class_ordering_summary.tex` is project-cadence formatted."""
    results = tmp_path / "in"
    for ordering in ("sequential", "random", "adversarial"):
        _write_csv(
            results / f"{ordering}.csv",
            [
                _row(instance_id="a.json.gz", profit="1000", class_ordering=ordering),
                _row(instance_id="b.json.gz", profit="2000", class_ordering=ordering),
            ],
        )

    out = tmp_path / "out"
    completed = _run(results, out)
    assert completed.returncode == 0, completed.stderr

    tex = (out / "class_ordering_summary.tex").read_text()
    assert "\\begin{tabular}" in tex
    assert "sequential" in tex
    assert "random" in tex
    assert "adversarial" in tex
