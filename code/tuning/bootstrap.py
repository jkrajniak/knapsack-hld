"""Non-parametric bootstrap mean + percentile CI (parameter-tuning §4.2.2).

We use the simple percentile bootstrap: draw ``n_resamples`` samples of
size ``len(values)`` with replacement, take the mean of each, and report
the empirical 2.5 / 97.5 percentiles as the 95 % CI. This is the
standard reporting tool for SMAC incumbent evaluations (cf. Hutter et
al. 2011, design D5) and is sufficient for the small tuning batches we
work with (≤ 80 instances in the preview, ≤ 600 in the full run).

Returns a typed result rather than a tuple so callers can extend the
report (e.g. add bias-corrected variants) without breaking the API.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_N_RESAMPLES = 1000
DEFAULT_CI_LEVEL = 0.95


@dataclass(frozen=True)
class BootstrapCI:
    """Percentile-bootstrap mean estimate with two-sided CI."""

    mean: float
    ci_low: float
    ci_high: float
    ci_level: float
    n_samples: int
    n_resamples: int

    def as_dict(self) -> dict[str, float | int]:
        return {k: v for k, v in asdict(self).items()}


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    n_resamples: int = DEFAULT_N_RESAMPLES,
    ci_level: float = DEFAULT_CI_LEVEL,
    random_seed: int = 0,
) -> BootstrapCI:
    """Percentile-bootstrap mean and ``ci_level`` CI."""
    if len(values) == 0:
        raise ValueError("values must be non-empty")
    if not 0.0 < ci_level < 1.0:
        raise ValueError(f"ci_level must be in (0, 1) (got {ci_level})")
    if n_resamples < 100:
        raise ValueError(f"n_resamples must be >= 100 (got {n_resamples})")

    arr = np.asarray(values, dtype=float)
    rng = np.random.default_rng(random_seed)
    idx = rng.integers(0, len(arr), size=(n_resamples, len(arr)))
    means = arr[idx].mean(axis=1)

    alpha = (1.0 - ci_level) / 2.0
    lo = float(np.quantile(means, alpha))
    hi = float(np.quantile(means, 1.0 - alpha))
    return BootstrapCI(
        mean=float(arr.mean()),
        ci_low=lo,
        ci_high=hi,
        ci_level=ci_level,
        n_samples=int(arr.size),
        n_resamples=int(n_resamples),
    )
