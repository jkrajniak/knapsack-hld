"""Tuning/test split is disjoint, deterministic, and cell-independent."""

from __future__ import annotations

import pytest

from instances import (
    CellKey,
    CorrelationKind,
    assert_test_only,
    assert_tuning_only,
    generate_instance,
    split_seeds,
)


@pytest.fixture
def cell() -> CellKey:
    return CellKey(N=1000, M=10, correlation=CorrelationKind.STRONGLY, f=0.5)


@pytest.fixture
def seeds() -> list[int]:
    return list(range(50))


def test_split_disjoint(cell: CellKey, seeds: list[int]) -> None:
    s = split_seeds(seeds, cell=cell)
    assert set(s.tuning).isdisjoint(s.test)
    assert sorted([*s.tuning, *s.test]) == sorted(seeds)


def test_split_ratio(cell: CellKey, seeds: list[int]) -> None:
    s = split_seeds(seeds, cell=cell, tuning_ratio=0.30)
    assert len(s.tuning) == 15
    assert len(s.test) == 35


def test_split_deterministic(cell: CellKey, seeds: list[int]) -> None:
    a = split_seeds(seeds, cell=cell)
    b = split_seeds(seeds, cell=cell)
    assert a == b


def test_split_cell_independent(seeds: list[int]) -> None:
    a = split_seeds(seeds, cell=CellKey(N=1000, M=10, correlation=CorrelationKind.STRONGLY, f=0.5))
    b = split_seeds(seeds, cell=CellKey(N=1000, M=10, correlation=CorrelationKind.STRONGLY, f=0.9))
    assert a.tuning != b.tuning


def test_disjoint_across_every_cell() -> None:
    seeds = list(range(50))
    for N in (1000, 10000):
        for M in (5, 10, 20):
            for corr in CorrelationKind:
                for f in (0.1, 0.5, 0.9):
                    cell = CellKey(N=N, M=M, correlation=corr, f=f)
                    s = split_seeds(seeds, cell=cell)
                    assert set(s.tuning).isdisjoint(s.test)


def test_assert_test_only_passes_for_test_seed() -> None:
    seeds = list(range(50))
    inst = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=0)
    s = split_seeds(seeds, cell=CellKey.from_instance(inst))
    test_seed = next(iter(s.test))
    inst_test = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=test_seed)
    assert_test_only(inst_test, cell_seeds=seeds)


def test_assert_test_only_blocks_tuning_seed() -> None:
    seeds = list(range(50))
    inst_probe = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=0)
    s = split_seeds(seeds, cell=CellKey.from_instance(inst_probe))
    tune_seed = next(iter(s.tuning))
    inst_tune = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=tune_seed)
    with pytest.raises(AssertionError, match="TUNING subset"):
        assert_test_only(inst_tune, cell_seeds=seeds)


def test_assert_tuning_only_passes_for_tuning_seed() -> None:
    seeds = list(range(50))
    inst_probe = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=0)
    s = split_seeds(seeds, cell=CellKey.from_instance(inst_probe))
    tune_seed = next(iter(s.tuning))
    inst_tune = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=tune_seed)
    assert_tuning_only(inst_tune, cell_seeds=seeds)


def test_assert_tuning_only_blocks_test_seed() -> None:
    seeds = list(range(50))
    inst_probe = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=0)
    s = split_seeds(seeds, cell=CellKey.from_instance(inst_probe))
    test_seed = next(iter(s.test))
    inst_test = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=test_seed)
    with pytest.raises(AssertionError, match="TEST subset"):
        assert_tuning_only(inst_test, cell_seeds=seeds)


def test_assert_helpers_reject_unknown_seed() -> None:
    seeds = list(range(50))
    inst = generate_instance(N=200, M=5, correlation="uncorrelated", f=0.5, seed=999)
    with pytest.raises(AssertionError, match="not present in archive cell"):
        assert_tuning_only(inst, cell_seeds=seeds)
    with pytest.raises(AssertionError, match="not present in archive cell"):
        assert_test_only(inst, cell_seeds=seeds)
