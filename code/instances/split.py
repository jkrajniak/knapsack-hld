"""Deterministic tuning/test split of generated instances.

The full archive is partitioned per `(N, M, correlation, f)` cell into
disjoint **tuning** and **test** subsets at a fixed ratio (default 30 %
tuning, 70 % test). The split is reproducible from a single master seed
recorded in the archive manifest.

`assert_test_only` is the safety hook that all result-producing scripts
in `scripts/run_experiments/` MUST call before reporting any number.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from instances.schema import CorrelationKind, InstanceModel

DEFAULT_TUNING_RATIO: float = 0.30
DEFAULT_MASTER_SEED: int = 20260424


@dataclass(frozen=True)
class CellKey:
    """Hashable identifier for one cell of the archive grid."""

    N: int
    M: int
    correlation: CorrelationKind
    f: float

    @classmethod
    def from_instance(cls, inst: InstanceModel) -> "CellKey":
        return cls(N=inst.N, M=inst.M, correlation=inst.correlation, f=inst.f)


@dataclass(frozen=True)
class Split:
    """Tuning/test partition of seeds for a single cell."""

    tuning: tuple[int, ...]
    test: tuple[int, ...]

    def __post_init__(self) -> None:
        if set(self.tuning) & set(self.test):
            raise ValueError("tuning and test seed sets overlap")


def split_seeds(
    seeds: Sequence[int],
    *,
    cell: CellKey,
    tuning_ratio: float = DEFAULT_TUNING_RATIO,
    master_seed: int = DEFAULT_MASTER_SEED,
) -> Split:
    """Deterministically partition `seeds` into tuning/test subsets.

    The cell key is mixed into the master seed so each cell gets an
    independent permutation, and the same call twice yields the same
    partition.
    """
    if not 0.0 < tuning_ratio < 1.0:
        raise ValueError(f"tuning_ratio must be in (0, 1) (got {tuning_ratio})")
    if not seeds:
        return Split(tuning=(), test=())

    rng = np.random.default_rng(_mix_seed(master_seed, cell))
    ordered = np.array(sorted(set(int(s) for s in seeds)), dtype=np.int64)
    perm = rng.permutation(len(ordered))
    n_tune = max(1, int(round(tuning_ratio * len(ordered))))
    tune_idx = sorted(perm[:n_tune].tolist())
    test_idx = sorted(perm[n_tune:].tolist())
    return Split(
        tuning=tuple(int(ordered[i]) for i in tune_idx),
        test=tuple(int(ordered[i]) for i in test_idx),
    )


def assert_test_only(
    instance: InstanceModel,
    *,
    tuning_ratio: float = DEFAULT_TUNING_RATIO,
    master_seed: int = DEFAULT_MASTER_SEED,
    cell_seeds: Iterable[int] | None = None,
) -> None:
    """Raise `AssertionError` if `instance.seed` belongs to the tuning subset.

    Result-producing scripts call this on every loaded instance. The
    caller MAY pass the explicit list of seeds present in the archive
    for the instance's cell; otherwise this function falls back to the
    canonical 50-seed grid `[0, 50)`.
    """
    cell = CellKey.from_instance(instance)
    seeds = list(cell_seeds) if cell_seeds is not None else list(range(50))
    if instance.seed not in seeds:
        raise AssertionError(
            f"seed {instance.seed} not present in archive cell {cell}; "
            "cannot determine tuning/test membership"
        )
    split = split_seeds(seeds, cell=cell, tuning_ratio=tuning_ratio, master_seed=master_seed)
    if instance.seed in split.tuning:
        raise AssertionError(
            f"instance seed={instance.seed} is in the TUNING subset for cell {cell}; "
            "result-producing scripts must use the test subset only"
        )


def _mix_seed(master_seed: int, cell: CellKey) -> int:
    """Hash the cell into a 63-bit seed independent across cells."""
    payload = (
        f"{master_seed}|{cell.N}|{cell.M}|{cell.correlation.value}|{cell.f:0.6f}"
    ).encode()
    h = np.frombuffer(_blake2b_8(payload), dtype=np.int64)[0]
    return int(h & 0x7FFFFFFFFFFFFFFF)


def _blake2b_8(payload: bytes) -> bytes:
    import hashlib

    return hashlib.blake2b(payload, digest_size=8).digest()
