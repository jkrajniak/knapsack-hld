"""Sanity checks on the correlation classes themselves.

Strong/inverse-strong instances obey a deterministic relationship between
profit and cost; uncorrelated/weakly instances should show low/medium
empirical Pearson correlation.
"""

from __future__ import annotations

import math

from instances import generate_instance


def _flat(inst):
    profits = [p for cls in inst.items for p, _c in cls]
    costs = [c for cls in inst.items for _p, c in cls]
    return profits, costs


def _pearson(xs: list[int], ys: list[int]) -> float:
    n = len(xs)
    mx, my = sum(xs) / n, sum(ys) / n
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else 0.0


def test_strongly_correlated_relation() -> None:
    inst = generate_instance(N=200, M=10, correlation="strongly", f=0.5, seed=1, R=1000)
    spread = 1000 // 10
    for cls in inst.items:
        for p, c in cls:
            assert p == min(1000, max(1, c + spread))


def test_inversely_strongly_relation() -> None:
    inst = generate_instance(N=200, M=10, correlation="inversely_strongly", f=0.5, seed=2, R=1000)
    spread = 1000 // 10
    for cls in inst.items:
        for p, c in cls:
            assert c == min(1000, max(1, p + spread))


def test_uncorrelated_low_pearson() -> None:
    inst = generate_instance(N=500, M=10, correlation="uncorrelated", f=0.5, seed=3, R=1000)
    p, c = _flat(inst)
    assert abs(_pearson(p, c)) < 0.1


def test_weakly_moderate_pearson() -> None:
    inst = generate_instance(N=500, M=10, correlation="weakly", f=0.5, seed=4, R=1000)
    p, c = _flat(inst)
    rho = _pearson(p, c)
    assert 0.85 < rho < 1.0
