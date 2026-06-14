"""Determinism + round-trip tests for the instance generator."""

from __future__ import annotations

import pytest

from instances import (
    CorrelationKind,
    generate_instance,
    instance_id,
    load_instance,
    save_instance,
)

_NATIVE_CORRELATIONS = [
    CorrelationKind.UNCORRELATED,
    CorrelationKind.WEAKLY,
    CorrelationKind.STRONGLY,
    CorrelationKind.INVERSELY_STRONGLY,
]


@pytest.mark.parametrize("correlation", _NATIVE_CORRELATIONS)
def test_same_seed_same_instance(correlation: CorrelationKind) -> None:
    a = generate_instance(N=20, M=5, correlation=correlation, f=0.5, seed=42)
    b = generate_instance(N=20, M=5, correlation=correlation, f=0.5, seed=42)
    assert a.model_dump() == b.model_dump()


@pytest.mark.parametrize(
    "correlation",
    [
        CorrelationKind.SUBSET_SUM,
        CorrelationKind.SIMILAR_WEIGHTS,
        CorrelationKind.UNCORRELATED_WITH_SKIP,
    ],
)
def test_native_generator_rejects_pisinger_only_kinds(correlation: CorrelationKind) -> None:
    """The project's own generator does not support Pisinger-only kinds.
    These are produced exclusively by `instances.pisinger_generator`.
    """
    with pytest.raises(ValueError, match="unknown correlation"):
        generate_instance(N=4, M=4, correlation=correlation, f=0.5, seed=0)


def test_different_seeds_differ() -> None:
    a = generate_instance(N=20, M=5, correlation="uncorrelated", f=0.5, seed=1)
    b = generate_instance(N=20, M=5, correlation="uncorrelated", f=0.5, seed=2)
    assert a.items != b.items


def test_roundtrip_json(tmp_path) -> None:
    inst = generate_instance(N=15, M=4, correlation="weakly", f=0.25, seed=7)
    path = tmp_path / (instance_id(N=15, M=4, correlation="weakly", f=0.25, seed=7) + ".json")
    save_instance(inst, path)
    reloaded = load_instance(path)
    assert reloaded.model_dump() == inst.model_dump()


def test_roundtrip_jsongz(tmp_path) -> None:
    inst = generate_instance(N=15, M=4, correlation="strongly", f=0.5, seed=9)
    path = tmp_path / "inst.json.gz"
    save_instance(inst, path)
    reloaded = load_instance(path)
    assert reloaded.model_dump() == inst.model_dump()


def test_budget_formula() -> None:
    inst = generate_instance(N=100, M=5, correlation="uncorrelated", f=0.5, seed=11)
    mean_cost = sum(c for cls in inst.items for _p, c in cls) / (inst.N * inst.M)
    expected = max(1, round(0.5 * inst.N * mean_cost))
    assert expected == inst.B


def test_instance_id_format() -> None:
    name = instance_id(N=1000, M=10, correlation="strongly", f=0.5, seed=42)
    assert name == "mckp_N1000_M10_strongly_f0.500_seed42"
