"""Unit tests for the §4.3.4 alpha sweep harness."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import pytest
from anomalies.sweep import (
    DEFAULT_ALPHA_GRID,
    DEFAULT_ALPHA_NITER,
    AnomalyInstance,
    SweepRecord,
    reference_cache_from_tight_validation,
    run_alpha_sweep,
    run_one,
)
from instances.schema import GENERATOR_VERSION, CorrelationKind, InstanceModel


def test_default_alpha_grid_spans_unit_interval() -> None:
    """Spec §4.3.4: alpha ∈ {0.0, 0.1, …, 1.0}."""
    assert len(DEFAULT_ALPHA_GRID) == 11
    assert DEFAULT_ALPHA_GRID[0] == 0.0
    assert DEFAULT_ALPHA_GRID[-1] == 1.0
    assert 0.5 in DEFAULT_ALPHA_GRID  # bitwise-exact mid-point matters for §4.3.4


def test_default_alpha_n_iter_is_twenty() -> None:
    """Spec §4.3.4: fix N_iter=20 across the alpha sweep."""
    assert DEFAULT_ALPHA_NITER == 20


def test_sweep_record_carries_alpha_default() -> None:
    """Records emitted by the original N_iter sweep must keep the HLD default alpha."""
    rec = SweepRecord(
        inst_id="x",
        n_iter=5,
        hld_profit=10,
        opt_profit=10,
        optimality_gap=0.0,
        hld_wall_s=0.1,
        opt_wall_s=0.1,
        solver_metadata={},
        budget=100,
    )
    assert rec.alpha == 0.9
    j = rec.as_json()
    assert j["alpha"] == 0.9


def test_sweep_record_alpha_round_trip_json() -> None:
    rec = SweepRecord(
        inst_id="x",
        n_iter=20,
        hld_profit=10,
        opt_profit=10,
        optimality_gap=0.0,
        hld_wall_s=0.1,
        opt_wall_s=0.1,
        solver_metadata={},
        budget=100,
        alpha=0.5,
    )
    j = rec.as_json()
    assert j["alpha"] == 0.5
    assert j["n_iter"] == 20


def test_run_one_threads_alpha_into_hld_metadata(tmp_path) -> None:
    """`run_one(..., alpha=...)` must propagate alpha to HLD's solver_metadata."""
    inst = _tiny_instance()
    item = AnomalyInstance(inst=inst, path=tmp_path / "x.json", inst_id="tiny")
    rec = run_one(
        item=item,
        n_iter=5,
        opt_profit=int(inst.items[0][0][0]),
        opt_wall_s=0.0,
        sub_solver="highs",
        alpha=0.42,
    )
    assert rec.alpha == pytest.approx(0.42)
    assert rec.solver_metadata["params"]["alpha"] == pytest.approx(0.42)


def test_reference_cache_from_tight_validation(tmp_path) -> None:
    """The §4.3.5 validation JSON must round-trip into a usable cache."""
    payload = [
        {
            "inst_id": "iid_a",
            "default": {
                "label": "default",
                "profit": 100,
                "status": "optimal",
                "wall_s": 1.0,
                "highs_status": "kOptimal",
                "mip_gap": 1e-5,
                "mip_rel_gap_set": None,
            },
            "tight": {
                "label": "tight_1e-9",
                "profit": 110,
                "status": "optimal",
                "wall_s": 5.0,
                "highs_status": "kOptimal",
                "mip_gap": 0.0,
                "mip_rel_gap_set": 1e-9,
            },
        }
    ]
    p = tmp_path / "v.json"
    p.write_text(json.dumps(payload))

    tight = reference_cache_from_tight_validation(p)
    assert "iid_a" in tight
    profit, wall, status, meta = tight["iid_a"]
    assert profit == 110
    assert wall == 5.0
    assert status == "optimal"
    assert meta["mip_rel_gap_set"] == 1e-9
    assert meta["_source"] == "tight_gap_validation.tight"

    default = reference_cache_from_tight_validation(p, prefer="default")
    assert default["iid_a"][0] == 100
    assert default["iid_a"][3]["_source"] == "tight_gap_validation.default"


def test_reference_cache_rejects_unknown_prefer(tmp_path) -> None:
    p = tmp_path / "v.json"
    p.write_text("[]")
    with pytest.raises(ValueError):
        reference_cache_from_tight_validation(p, prefer="nope")


def test_run_alpha_sweep_uses_cache_and_writes_jsonl(tmp_path) -> None:
    """End-to-end: cached reference, alpha grid {0.0, 1.0}, JSONL streamed."""
    inst = _tiny_instance()
    item = AnomalyInstance(inst=inst, path=tmp_path / "tiny.json", inst_id="tiny")
    cache = {"tiny": (int(inst.items[0][0][0]), 0.0, "optimal", {"_source": "test"})}
    out_path = tmp_path / "alpha.jsonl"

    recs = run_alpha_sweep(
        items=[item],
        alpha_grid=(0.0, 1.0),
        n_iter=3,
        sub_solver="highs",
        reference_cache=cache,
        out_path=out_path,
    )
    assert len(recs) == 2
    assert {r.alpha for r in recs} == {0.0, 1.0}
    assert all(r.n_iter == 3 for r in recs)

    lines = [json.loads(line) for line in out_path.read_text().splitlines() if line]
    assert len(lines) == 2
    assert {entry["alpha"] for entry in lines} == {0.0, 1.0}


def _tiny_instance() -> InstanceModel:
    """Trivial 1-class, 1-item instance: profit=10, cost=5, B=10."""
    return InstanceModel(
        N=1,
        M=1,
        B=10,
        items=[[(10, 5)]],
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        generator_version=GENERATOR_VERSION,
    )
