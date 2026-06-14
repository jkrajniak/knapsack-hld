"""Tests for the Pisinger 1995 MCKP generator (Python port of `mcknap.c`)."""

from __future__ import annotations

import pytest
from instances.pisinger_generator import (
    PISINGER_TYPES,
    Drand48,
    _capacity_pisinger,
    emit_pisinger_in_text,
    generate_pisinger_instance,
)
from instances.pisinger_loader import parse_pisinger_text
from instances.schema import CorrelationKind

# ---------------------------------------------------------------------------
# Drand48: POSIX bit-exactness + determinism
# ---------------------------------------------------------------------------


def test_drand48_matches_posix_seed_zero() -> None:
    """Canonical POSIX/glibc value: srand48(0); lrand48() == 366850414."""
    rng = Drand48(0)
    assert rng.lrand48() == 366850414


def test_drand48_pinned_sequences() -> None:
    """Regression fixture for three seeds; locks LCG output bit-by-bit."""
    rng = Drand48(0)
    assert [rng.lrand48() for _ in range(5)] == [
        366850414,
        1610402240,
        206956554,
        1869309841,
        1239749840,
    ]
    rng = Drand48(1)
    assert [rng.lrand48() for _ in range(5)] == [
        89400484,
        976015093,
        1792756325,
        721524505,
        1214379247,
    ]
    rng = Drand48(42)
    assert [rng.lrand48() for _ in range(3)] == [1598855263, 735945821, 238553827]


def test_drand48_is_deterministic() -> None:
    a = Drand48(7)
    b = Drand48(7)
    assert [a.lrand48() for _ in range(20)] == [b.lrand48() for _ in range(20)]


def test_random_below_respects_range() -> None:
    rng = Drand48(123)
    for _ in range(1000):
        v = rng.random_below(10)
        assert 0 <= v < 10


def test_random_below_rejects_nonpositive_range() -> None:
    rng = Drand48(0)
    with pytest.raises(ValueError, match="random_below requires r > 0"):
        rng.random_below(0)


# ---------------------------------------------------------------------------
# Per-type structural invariants
# ---------------------------------------------------------------------------


def test_type_1_uncorrelated_has_independent_values_in_range() -> None:
    inst = generate_pisinger_instance(n_classes=4, n_items=50, r=1000, type_id=1, seed=11)
    assert inst.correlation == CorrelationKind.UNCORRELATED
    for cls in inst.items:
        for p, c in cls:
            assert 1 <= p <= 1000
            assert 1 <= c <= 1000


def test_type_2_weakly_correlated_keeps_profit_within_pm10_window() -> None:
    inst = generate_pisinger_instance(n_classes=3, n_items=200, r=1000, type_id=2, seed=22)
    assert inst.correlation == CorrelationKind.WEAKLY
    for cls in inst.items:
        for p, c in cls:
            assert 1 <= c <= 1000
            assert p >= 1
            assert p == 1 or abs(p - c) <= 10


def test_type_3_strongly_correlated_is_cumulative_within_class() -> None:
    inst = generate_pisinger_instance(n_classes=2, n_items=20, r=1000, type_id=3, seed=33)
    assert inst.correlation == CorrelationKind.STRONGLY
    for cls in inst.items:
        for j in range(1, len(cls)):
            assert cls[j][0] > cls[j - 1][0]
            assert cls[j][1] > cls[j - 1][1]


def test_type_4_subset_sum_has_profit_equal_cost() -> None:
    inst = generate_pisinger_instance(n_classes=4, n_items=50, r=1000, type_id=4, seed=44)
    assert inst.correlation == CorrelationKind.SUBSET_SUM
    for cls in inst.items:
        for p, c in cls:
            assert 1 <= c <= 1000
            assert p == c


def test_type_5_similar_weights_sorted_pairs_within_class() -> None:
    inst = generate_pisinger_instance(n_classes=3, n_items=30, r=1000, type_id=5, seed=55)
    assert inst.correlation == CorrelationKind.SIMILAR_WEIGHTS
    for cls in inst.items:
        for j in range(1, len(cls)):
            assert cls[j][0] >= cls[j - 1][0]
            assert cls[j][1] >= cls[j - 1][1]
        for p, c in cls:
            assert 1 <= p <= 1000
            assert 1 <= c <= 1000


def test_type_6_first_item_is_zero_skip_in_every_class() -> None:
    inst = generate_pisinger_instance(n_classes=5, n_items=20, r=1000, type_id=6, seed=66)
    assert inst.correlation == CorrelationKind.UNCORRELATED_WITH_SKIP
    for cls in inst.items:
        assert cls[0] == [0, 0]
        for p, c in cls[1:]:
            assert 1 <= p <= 1000
            assert 1 <= c <= 1000


def test_type_6_consumes_no_lcg_draws_for_first_item() -> None:
    """Type 6 first item is hard-coded (0,0); LCG stream after one class
    must equal `2*(n-1)` calls of `random_below(r)+1`."""
    n = 5
    r = 100
    rng_a = Drand48(7)
    inst = generate_pisinger_instance(n_classes=1, n_items=n, r=r, type_id=6, seed=7)
    rng_b = Drand48(7)
    expected_draws = [(rng_b.random_below(r) + 1) for _ in range(2 * (n - 1))]
    actual_draws: list[int] = []
    for p, c in inst.items[0][1:]:
        actual_draws.extend([c, p])
    assert actual_draws == expected_draws
    _ = rng_a


# ---------------------------------------------------------------------------
# Capacity: hand-checked formula
# ---------------------------------------------------------------------------


def test_capacity_matches_formula_on_handcrafted_classes() -> None:
    items = [
        [[10, 5], [20, 8], [3, 2]],
        [[7, 1], [50, 7], [50, 4]],
    ]
    expected = (2 + 1 + 8 + 4) // 2
    assert _capacity_pisinger(items) == expected


def test_capacity_breaks_max_profit_ties_with_smaller_weight() -> None:
    items = [[[5, 9], [5, 3], [5, 6]]]
    assert _capacity_pisinger(items) == (3 + 3) // 2


# ---------------------------------------------------------------------------
# Determinism + supported-type coverage
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("type_id", sorted(PISINGER_TYPES))
def test_same_seed_yields_identical_instance(type_id: int) -> None:
    a = generate_pisinger_instance(n_classes=5, n_items=10, r=500, type_id=type_id, seed=99)
    b = generate_pisinger_instance(n_classes=5, n_items=10, r=500, type_id=type_id, seed=99)
    assert a.items == b.items
    assert a.B == b.B


def test_different_seeds_yield_different_instances() -> None:
    a = generate_pisinger_instance(n_classes=5, n_items=10, r=500, type_id=1, seed=1)
    b = generate_pisinger_instance(n_classes=5, n_items=10, r=500, type_id=1, seed=2)
    assert a.items != b.items


# ---------------------------------------------------------------------------
# Round-trip via pisinger_loader
# ---------------------------------------------------------------------------


def test_emit_then_parse_round_trips() -> None:
    inst = generate_pisinger_instance(n_classes=4, n_items=8, r=200, type_id=2, seed=2026)
    text = emit_pisinger_in_text(inst)
    parsed = parse_pisinger_text(text)
    assert parsed.N == inst.N
    assert parsed.M == inst.M
    assert parsed.B == inst.B
    assert parsed.items == inst.items


# ---------------------------------------------------------------------------
# Out-of-scope types are explicit
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("type_id", [0, 7, 8, -1])
def test_unsupported_types_raise_not_implemented(type_id: int) -> None:
    with pytest.raises(NotImplementedError, match="Pisinger type"):
        generate_pisinger_instance(n_classes=2, n_items=4, r=100, type_id=type_id, seed=0)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(n_classes=0, n_items=4, r=100, type_id=1, seed=0),
        dict(n_classes=4, n_items=0, r=100, type_id=1, seed=0),
        dict(n_classes=4, n_items=4, r=0, type_id=1, seed=0),
    ],
)
def test_invalid_dimensions_rejected(kwargs: dict) -> None:
    with pytest.raises(ValueError):
        generate_pisinger_instance(**kwargs)
