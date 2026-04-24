"""Canonical results schema: round-trip + schema stability."""

from __future__ import annotations

from pathlib import Path

from utils.results_schema import (
    RUN_RESULT_FIELDS,
    RunResult,
    host_metadata,
    read_results_csv_gz,
    write_host_metadata,
    write_results_csv_gz,
)


def _make_row(**overrides: object) -> RunResult:
    base = {
        "experiment": "test",
        "instance_id": "mckp_N4_M2_uncorrelated_f0.500_seed0",
        "N": 4,
        "M": 2,
        "correlation": "uncorrelated",
        "f": 0.5,
        "instance_seed": 0,
        "solver": "highs",
        "solver_seed": 0,
        "wall_time_s": 0.123,
        "profit": 42,
        "total_cost": 17,
        "n_classes_selected": 3,
        "status": "OPTIMAL",
        "time_limit_s": 60.0,
    }
    base.update(overrides)
    return RunResult(**base)


def test_field_order_is_stable() -> None:
    assert RUN_RESULT_FIELDS[0] == "experiment"
    assert RUN_RESULT_FIELDS[-1] == "solver_metadata_json"
    assert "optimality_gap_pct" in RUN_RESULT_FIELDS
    assert len(RUN_RESULT_FIELDS) == len(set(RUN_RESULT_FIELDS))


def test_round_trip_csv_gz(tmp_path: Path) -> None:
    rows = [
        _make_row(),
        _make_row(
            solver="hld", optimality_gap_pct=2.5, reference_solver="highs", reference_profit=42
        ),
    ]
    out = tmp_path / "run.csv.gz"
    write_results_csv_gz(out, rows)
    loaded = read_results_csv_gz(out)
    assert len(loaded) == 2
    assert loaded[0]["experiment"] == "test"
    assert loaded[1]["solver"] == "hld"
    assert loaded[1]["optimality_gap_pct"] == "2.5"


def test_none_values_serialize_as_empty(tmp_path: Path) -> None:
    rows = [_make_row(reference_solver=None, optimality_gap_pct=None)]
    out = tmp_path / "run.csv.gz"
    write_results_csv_gz(out, rows)
    loaded = read_results_csv_gz(out)
    assert loaded[0]["reference_solver"] == ""
    assert loaded[0]["optimality_gap_pct"] == ""


def test_host_metadata_contains_required_keys() -> None:
    md = host_metadata()
    for key in ("hostname", "platform", "machine", "python", "captured_utc"):
        assert key in md
        assert isinstance(md[key], str)
        assert md[key]


def test_write_host_metadata_is_valid_json(tmp_path: Path) -> None:
    import json

    out = tmp_path / "host.json"
    write_host_metadata(out)
    md = json.loads(out.read_text())
    assert "hostname" in md
