"""Smoke test for the Pisinger validation runner — types 1-6 grid (Task B6).

Exercises ``scripts/run_pisinger_validation.py`` on a deliberately tiny grid
that still covers all six Pisinger 1995 §6 instance types. Confirms:
  - Default ``--types`` covers types 1-6.
  - Each type generates a valid instance and solves under mcknap.
  - Resume logic (``(instance_id, solver)`` dedup) skips rows already in the
    output CSV.
  - The ``correlation`` column carries the right ``CorrelationKind`` value
    for each of the six types.

Kept tiny on purpose (k=2, n=2, r=100, single seed, single solver) so the
full test stays sub-second on the laptop. The full 4 800-instance grid is
launched separately on the VM via the runner CLI.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_pisinger_validation.py"

EXPECTED_CORRELATIONS = {
    1: "uncorrelated",
    2: "weakly",
    3: "strongly",
    4: "subset_sum",
    5: "similar_weights",
    6: "uncorrelated_with_skip",
}


def _run(out_csv: Path, types: list[int] | None = None) -> subprocess.CompletedProcess[str]:
    cmd = [
        sys.executable,
        str(SCRIPT),
        "--out-csv",
        str(out_csv),
        "--solvers",
        "mcknap",
        "--ks",
        "2",
        "--ns",
        "2",
        "--rs",
        "100",
        "--seeds",
        "1",
        "--time-limit-s",
        "10",
    ]
    if types is not None:
        cmd += ["--types", *(str(t) for t in types)]
    return subprocess.run(cmd, cwd=ROOT, check=False, capture_output=True, text=True)


def _read_rows(out_csv: Path) -> list[dict[str, str]]:
    with out_csv.open("r", newline="") as fh:
        return list(csv.DictReader(fh))


def test_default_grid_covers_all_six_pisinger_types(tmp_path: Path) -> None:
    out_csv = tmp_path / "pisinger_smoke.csv"

    completed = _run(out_csv)

    assert completed.returncode == 0, completed.stderr
    assert out_csv.exists()

    rows = _read_rows(out_csv)
    assert len(rows) == 6, f"expected 6 rows (one per type), got {len(rows)}"

    seen_types: dict[int, dict[str, str]] = {int(r["type_id"]): r for r in rows}
    assert set(seen_types) == set(EXPECTED_CORRELATIONS), (
        f"default --types should cover the six Pisinger 1995 types; got {sorted(seen_types)}"
    )

    for type_id, expected in EXPECTED_CORRELATIONS.items():
        row = seen_types[type_id]
        assert row["correlation"] == expected, (
            f"type {type_id}: expected correlation={expected!r}, got {row['correlation']!r}"
        )
        assert row["solver"] == "mcknap"
        assert row["status"] in {"optimal", "feasible"}, (
            f"type {type_id}: mcknap reference solver should reach an integer "
            f"solution on a 2×2 instance; got status={row['status']!r}"
        )
        assert int(row["profit"]) >= 0


def test_resume_skips_existing_rows(tmp_path: Path) -> None:
    out_csv = tmp_path / "pisinger_resume.csv"

    first = _run(out_csv, types=[1, 2, 3])
    assert first.returncode == 0, first.stderr
    rows_after_first = _read_rows(out_csv)
    assert {int(r["type_id"]) for r in rows_after_first} == {1, 2, 3}

    second = _run(out_csv, types=[1, 2, 3, 4, 5, 6])
    assert second.returncode == 0, second.stderr

    rows_after_second = _read_rows(out_csv)
    assert {int(r["type_id"]) for r in rows_after_second} == {1, 2, 3, 4, 5, 6}
    assert len(rows_after_second) == 6, (
        f"resume should add only types 4-6, total should be 6 not {len(rows_after_second)}"
    )
