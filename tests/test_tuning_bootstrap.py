"""Bootstrap CI helper used by the SMAC harness (parameter-tuning §4.2.2)."""

from __future__ import annotations

import numpy as np
import pytest
from tuning.bootstrap import BootstrapCI, bootstrap_mean_ci


def test_bootstrap_mean_matches_arithmetic_mean() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5]
    ci = bootstrap_mean_ci(values, random_seed=0)
    assert ci.mean == pytest.approx(0.3)


def test_bootstrap_ci_brackets_mean() -> None:
    rng = np.random.default_rng(0)
    sample = rng.normal(loc=2.5, scale=0.4, size=200).tolist()
    ci = bootstrap_mean_ci(sample, n_resamples=2000, random_seed=42)
    assert ci.ci_low <= ci.mean <= ci.ci_high


def test_bootstrap_is_deterministic_for_seed() -> None:
    values = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    a = bootstrap_mean_ci(values, n_resamples=500, random_seed=1)
    b = bootstrap_mean_ci(values, n_resamples=500, random_seed=1)
    assert a == b


def test_bootstrap_ci_shrinks_with_sample_size() -> None:
    rng = np.random.default_rng(0)
    small = rng.normal(loc=0.0, scale=1.0, size=20).tolist()
    large = rng.normal(loc=0.0, scale=1.0, size=2000).tolist()
    ci_small = bootstrap_mean_ci(small, n_resamples=1000, random_seed=0)
    ci_large = bootstrap_mean_ci(large, n_resamples=1000, random_seed=0)
    assert (ci_large.ci_high - ci_large.ci_low) < (ci_small.ci_high - ci_small.ci_low)


def test_bootstrap_rejects_invalid_inputs() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        bootstrap_mean_ci([], random_seed=0)
    with pytest.raises(ValueError, match="ci_level"):
        bootstrap_mean_ci([1.0, 2.0], ci_level=1.0, random_seed=0)
    with pytest.raises(ValueError, match="n_resamples"):
        bootstrap_mean_ci([1.0, 2.0], n_resamples=10, random_seed=0)


def test_bootstrap_as_dict_serialisable() -> None:
    ci = bootstrap_mean_ci([1.0, 2.0, 3.0], n_resamples=200, random_seed=0)
    d = ci.as_dict()
    assert isinstance(ci, BootstrapCI)
    assert set(d) == {"mean", "ci_low", "ci_high", "ci_level", "n_samples", "n_resamples"}
    assert all(isinstance(v, (int, float)) for v in d.values())
