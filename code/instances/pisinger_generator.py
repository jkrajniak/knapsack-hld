"""Pisinger 1995 MCKP instance generator (Python port of `mcknap.c` L740-870).

This module is a faithful port of the inline test-instance generator
embedded in David Pisinger's 1993/1994 reference implementation
(`mcknap.c`). The original C source is preserved at
`instances/pisinger_1995/mcknap.c` (SHA-256 documented in
`instances/pisinger_1995/CHECKSUMS.txt`).

Why a port and not a download
-----------------------------
There is no `test_mcknap.tgz` archive on Pisinger's upstream site; the
"Pisinger benchmarks" cited in the literature are *regenerated from
`mcknap.c`* at the parameters published in §6 of the 1995 paper. See
`instances/pisinger_1995/FINDING_2026_06_05.md` for the acquisition
finding that motivated this port.

PRNG faithfulness
-----------------
The generator emulates POSIX `srand48` / `lrand48` exactly so that
instance generation is bit-deterministic given a seed and matches
the algorithmic semantics of the upstream C code. Seeds are *not*
guaranteed to reproduce specific numbers from the 1995 paper because
that paper does not publish its seeds; what is reproducible is the
algorithm, the parameter grid, and our own seed schedule (which we
publish alongside the manuscript).

Scope
-----
All six Pisinger 1995 instance types are implemented:

    1. uncorrelated                 (CorrelationKind.UNCORRELATED)
    2. weakly correlated            (CorrelationKind.WEAKLY)
    3. strongly correlated cumul.   (CorrelationKind.STRONGLY)
    4. subset-sum (p == w)          (CorrelationKind.SUBSET_SUM)
    5. similar-weights (sorted)     (CorrelationKind.SIMILAR_WEIGHTS)
    6. uncorrelated with skip       (CorrelationKind.UNCORRELATED_WITH_SKIP)

Each type is a faithful port of the corresponding `case` in
`mcknap.c::maketest` (L740-789). The Drand48 LCG and the
`_capacity_pisinger` capacity formula are shared across all types,
matching the C code exactly.
"""

from __future__ import annotations

from typing import Final

from instances.schema import GENERATOR_VERSION, CorrelationKind, InstanceModel

_LCG_MULT: Final[int] = 0x5DEECE66D
_LCG_INC: Final[int] = 0xB
_LCG_MASK: Final[int] = (1 << 48) - 1
_LCG_SEED_MASK: Final[int] = 0x330E

PISINGER_TYPES: Final[dict[int, CorrelationKind]] = {
    1: CorrelationKind.UNCORRELATED,
    2: CorrelationKind.WEAKLY,
    3: CorrelationKind.STRONGLY,
    4: CorrelationKind.SUBSET_SUM,
    5: CorrelationKind.SIMILAR_WEIGHTS,
    6: CorrelationKind.UNCORRELATED_WITH_SKIP,
}


class Drand48:
    """POSIX `drand48`/`lrand48` linear congruential generator.

    State update: `s_{n+1} = (a · s_n + c) mod 2^48` with
    `a = 0x5DEECE66D`, `c = 0xB`. `srand48(seed)` sets
    `state = (seed << 16) | 0x330E`. `lrand48()` returns the top
    31 bits of state (i.e. `state >> 17`).
    """

    __slots__ = ("_state",)

    def __init__(self, seed: int) -> None:
        self.srand48(seed)

    def srand48(self, seed: int) -> None:
        self._state = ((int(seed) & 0xFFFFFFFF) << 16) | _LCG_SEED_MASK

    def lrand48(self) -> int:
        self._state = (self._state * _LCG_MULT + _LCG_INC) & _LCG_MASK
        return self._state >> 17

    def random_below(self, r: int) -> int:
        """Match `random(r)` in `mcknap.c` (= `lrand48() % r`, in `[0, r)`)."""
        if r <= 0:
            raise ValueError(f"random_below requires r > 0, got {r}")
        return self.lrand48() % r


def _make_class_items(rng: Drand48, n_items: int, r: int, type_id: int) -> list[list[int]]:
    """Generate one class' items per `mcknap.c::maketest` for the given type."""
    if type_id == 1:
        return _items_type_1(rng, n_items, r)
    if type_id == 2:
        return _items_type_2(rng, n_items, r)
    if type_id == 3:
        return _items_type_3(rng, n_items, r)
    if type_id == 4:
        return _items_type_4(rng, n_items, r)
    if type_id == 5:
        return _items_type_5(rng, n_items, r)
    if type_id == 6:
        return _items_type_6(rng, n_items, r)
    raise NotImplementedError(
        f"Pisinger type {type_id} not supported; valid types are {sorted(PISINGER_TYPES)}."
    )


def _items_type_1(rng: Drand48, n: int, r: int) -> list[list[int]]:
    return [[rng.random_below(r) + 1, rng.random_below(r) + 1] for _ in range(n)]


def _items_type_2(rng: Drand48, n: int, r: int) -> list[list[int]]:
    items: list[list[int]] = []
    for _ in range(n):
        wsum = rng.random_below(r) + 1
        psum = rng.random_below(21) + wsum - 10
        if psum <= 0:
            psum = 1
        items.append([psum, wsum])
    return items


def _items_type_3(rng: Drand48, n: int, r: int) -> list[list[int]]:
    """Strongly correlated cumulative class (mcknap.c L749-789, type=3)."""
    r_eff = (2 * r) // n
    if r_eff < 1:
        raise ValueError(
            f"Pisinger type 3 requires (2*r)/n >= 1; got r={r}, n={n} -> r_eff={r_eff}"
        )
    w_raw = [rng.random_below(r_eff) + 1 for _ in range(n)]
    p_raw = [w + 10 for w in w_raw]
    w_sorted = sorted(w_raw)
    p_sorted = sorted(p_raw)
    items: list[list[int]] = []
    ws_acc = 0
    ps_acc = 0
    for i in range(n):
        ws_acc += w_sorted[i]
        ps_acc += p_sorted[i]
        items.append([ps_acc, ws_acc])
    return items


def _items_type_4(rng: Drand48, n: int, r: int) -> list[list[int]]:
    """Subset-sum class (mcknap.c case 4): p == w == random(r)+1."""
    items: list[list[int]] = []
    for _ in range(n):
        w = rng.random_below(r) + 1
        items.append([w, w])
    return items


def _items_type_5(rng: Drand48, n: int, r: int) -> list[list[int]]:
    """Similar-weights class (mcknap.c case 5).

    Pre-loop: draw n weights and n profits independently in [1, r]; sort
    each ascending; assign in sorted-pair order. This induces strong
    rank correlation without the cumulative growth of type 3.

    Note: matches C code's pre-allocation order — w[k] is drawn before
    p[k] for each k, so the LCG stream is interleaved (w0, p0, w1, p1,
    ...). This preserves bit-exact reproduction relative to mcknap.c.
    """
    w_raw: list[int] = []
    p_raw: list[int] = []
    for _ in range(n):
        w_raw.append(rng.random_below(r) + 1)
        p_raw.append(rng.random_below(r) + 1)
    w_sorted = sorted(w_raw)
    p_sorted = sorted(p_raw)
    return [[p_sorted[i], w_sorted[i]] for i in range(n)]


def _items_type_6(rng: Drand48, n: int, r: int) -> list[list[int]]:
    """Uncorrelated with skip class (mcknap.c case 6).

    First item in each class is forced to (p=0, w=0), making it a free
    "skip" option. Remaining items follow the type-1 uncorrelated
    distribution. Note: the C code's `if (i == j->fset) { wsum=0; psum=0;
    break; }` short-circuits via the switch's `break`, so the first
    item consumes NO LCG draws.
    """
    items: list[list[int]] = [[0, 0]]
    for _ in range(n - 1):
        w = rng.random_below(r) + 1
        p = rng.random_below(r) + 1
        items.append([p, w])
    return items


def _capacity_pisinger(items: list[list[list[int]]]) -> int:
    """Reproduce mcknap.c L823-837: B = (Σ min_w + Σ max_p_break_min_w) // 2."""
    wsum1 = 0
    wsum2 = 0
    for cls in items:
        mi_w = cls[0][1]
        ma_p, ma_w = cls[0][0], cls[0][1]
        for p, w in cls[1:]:
            if w < mi_w:
                mi_w = w
            if p > ma_p or (p == ma_p and w < ma_w):
                ma_p, ma_w = p, w
        wsum1 += mi_w
        wsum2 += ma_w
    return (wsum1 + wsum2) // 2


def generate_pisinger_instance(
    *,
    n_classes: int,
    n_items: int,
    r: int,
    type_id: int,
    seed: int,
) -> InstanceModel:
    """Generate one Pisinger 1995 MCKP instance (`mcknap k n r type` equivalent).

    Args:
        n_classes: Number of classes (`k` in the C code).
        n_items: Items per class (`n` in the C code; uniform across classes).
        r: Coefficient range (`r` in the C code).
        type_id: 1=uncorrelated, 2=weakly correlated, 3=strongly correlated.
        seed: Master seed mapped 1:1 to `srand48(seed)`.

    Returns:
        An `InstanceModel` whose `[profit, cost]` pairs and capacity match
        what `mcknap k n r type_id` would emit on its `seed`-th test pass.
    """
    if n_classes <= 0 or n_items <= 0 or r <= 0:
        raise ValueError(f"n_classes, n_items, r must all be > 0; got {n_classes=}, {n_items=}, {r=}")
    if type_id not in PISINGER_TYPES:
        raise NotImplementedError(
            f"Pisinger type {type_id} not supported; supported: {sorted(PISINGER_TYPES)}"
        )

    rng = Drand48(seed)
    items: list[list[list[int]]] = [
        _make_class_items(rng, n_items, r, type_id) for _ in range(n_classes)
    ]
    B = _capacity_pisinger(items)

    mean_cost = sum(c for cls in items for _p, c in cls) / (n_classes * n_items)
    f = max(1e-6, B / (n_classes * mean_cost)) if mean_cost > 0 else 0.5

    return InstanceModel(
        N=n_classes,
        M=n_items,
        correlation=PISINGER_TYPES[type_id],
        f=f,
        seed=seed,
        B=B,
        items=items,
        generator_version=f"pisinger_generator+{GENERATOR_VERSION}",
    )


def emit_pisinger_in_text(instance: InstanceModel) -> str:
    """Serialise an instance into Pisinger `.in` text format.

    Layout matches what `instances.pisinger_loader.parse_pisinger_text`
    expects so that `parse(emit(inst))` round-trips. Format:

        N
        M_1
        p_{1,1} c_{1,1}
        ...
        M_N
        p_{N,M} c_{N,M}
        B
    """
    lines: list[str] = [str(instance.N)]
    for cls in instance.items:
        lines.append(str(len(cls)))
        for p, c in cls:
            lines.append(f"{p} {c}")
    lines.append(str(instance.B))
    return "\n".join(lines) + "\n"
