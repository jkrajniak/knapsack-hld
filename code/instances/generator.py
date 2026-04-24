"""Deterministic pure-Python Selective-MCKP instance generator.

The four correlation classes follow Pisinger (1995) and Martello & Toth
(1990). For each class i = 1..N we draw M items independently, then group
them into the class. The budget is B = round(f · N · mean(c)) where the
mean is taken over all N·M generated items, matching the manuscript's
§2.6 specification (B = f · N · c̄).

The generator is fully deterministic: identical `(N, M, correlation, f,
seed)` tuples produce bit-identical instances on any platform supporting
NumPy ≥ 1.17 (PCG64 is platform-independent).
"""

from __future__ import annotations

import numpy as np

from instances.schema import CorrelationKind, InstanceModel

DEFAULT_AMPLITUDE: int = 1000


def generate_instance(
    *,
    N: int,
    M: int,
    correlation: CorrelationKind | str,
    f: float,
    seed: int,
    R: int = DEFAULT_AMPLITUDE,
) -> InstanceModel:
    """Generate one Selective-MCKP instance.

    Parameters
    ----------
    N: number of classes.
    M: items per class (uniform across classes).
    correlation: one of `CorrelationKind`.
    f: budget tightness factor (B = round(f · N · mean(c))).
    seed: master seed.
    R: amplitude (max value of profits and costs); Pisinger default is 1000.

    Returns
    -------
    `InstanceModel` with `items[i][j] = [profit, cost]`.
    """
    if N <= 0 or M <= 0:
        raise ValueError(f"N and M must be positive (got N={N}, M={M})")
    if f <= 0:
        raise ValueError(f"f must be positive (got {f})")
    if R <= 1:
        raise ValueError(f"R must be > 1 (got {R})")

    correlation = CorrelationKind(correlation)
    rng = np.random.default_rng(seed)
    p_arr, c_arr = _draw_items(rng, N * M, correlation, R)

    p_grid = p_arr.reshape(N, M)
    c_grid = c_arr.reshape(N, M)

    mean_cost = float(c_arr.mean())
    B = max(1, int(round(f * N * mean_cost)))

    items = [
        [[int(p_grid[i, j]), int(c_grid[i, j])] for j in range(M)] for i in range(N)
    ]

    return InstanceModel(
        N=N,
        M=M,
        correlation=correlation,
        f=f,
        seed=seed,
        B=B,
        items=items,
    )


def _draw_items(
    rng: np.random.Generator,
    n: int,
    correlation: CorrelationKind,
    R: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Draw `n` (profit, cost) pairs of the requested correlation.

    Returns two int64 arrays of length `n` clipped to `[1, R]`.
    """
    spread = max(1, R // 10)

    if correlation is CorrelationKind.UNCORRELATED:
        p = rng.integers(1, R + 1, size=n)
        c = rng.integers(1, R + 1, size=n)

    elif correlation is CorrelationKind.WEAKLY:
        c = rng.integers(1, R + 1, size=n)
        p = c + rng.integers(-spread, spread + 1, size=n)
        p = np.clip(p, 1, R)

    elif correlation is CorrelationKind.STRONGLY:
        c = rng.integers(1, R + 1, size=n)
        p = np.clip(c + spread, 1, R)

    elif correlation is CorrelationKind.INVERSELY_STRONGLY:
        p = rng.integers(1, R + 1, size=n)
        c = np.clip(p + spread, 1, R)

    else:  # pragma: no cover — exhausted by the enum
        raise ValueError(f"unknown correlation: {correlation!r}")

    return p.astype(np.int64), c.astype(np.int64)
