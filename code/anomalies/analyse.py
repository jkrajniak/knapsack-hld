"""Pure-function anomaly diagnostics for HLD Phase-1 / Phase-3 logs.

Inputs are the dicts that the HLD adapter records in
``SolveResult.solver_metadata`` (``phase1_trajectory`` and
``phase3_batches``). Both functions are total — they accept empty or
single-element inputs and return well-defined fallback values.

Hypothesis tests
----------------
**H1 — degenerate dual basis.** A converging Phase-1 binary search
should drive the relative spread of ``lambda_mid`` over the final
window to ~0 and stop flipping the sign of ``total_cost - B``.
We declare H1 = True if either:

- the relative spread of the last ``window`` ``lambda_mid`` values is
  above ``H1_LAMBDA_REL_SPREAD_THRESHOLD`` (default 1 %), **or**
- the sign of ``total_cost - B`` flips at least
  ``H1_SIGN_FLIP_THRESHOLD`` times (default 3) in the last ``window``
  iterations.

**H2 — sub-MILP straggler.** Equal/proportional Phase-2 allocation
cannot help if one batch dominates the wall time. We declare
H2 = True if the slowest batch consumes at least
``H2_MAX_BATCH_SHARE_THRESHOLD`` (default 50 %) of the total Phase-3
wall time.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Any

H1_LAMBDA_REL_SPREAD_THRESHOLD: float = 0.01
H1_SIGN_FLIP_THRESHOLD: int = 3
H1_DEFAULT_WINDOW: int = 5

H2_MAX_BATCH_SHARE_THRESHOLD: float = 0.5


@dataclass(frozen=True)
class Phase1Metrics:
    """Summary statistics of a Phase-1 binary-search trajectory."""

    n_iter: int
    final_lambda: float
    final_gap_to_budget: int
    lambda_rel_spread: float
    sign_flips_in_window: int
    window: int

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Phase3Metrics:
    """Summary statistics of the Phase-3 batched sub-MILP wall times."""

    n_batches: int
    total_wall_s: float
    max_batch_wall_s: float
    max_batch_share: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnomalyVerdicts:
    """Per-record output of :func:`analyse_record`."""

    phase1: Phase1Metrics
    phase3: Phase3Metrics
    h1_degenerate_dual: bool
    h2_straggler: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "phase1": self.phase1.as_dict(),
            "phase3": self.phase3.as_dict(),
            "h1_degenerate_dual": self.h1_degenerate_dual,
            "h2_straggler": self.h2_straggler,
        }


def phase1_metrics(
    trajectory: list[dict[str, Any]],
    *,
    budget: int,
    window: int = H1_DEFAULT_WINDOW,
) -> Phase1Metrics:
    """Summarise a Phase-1 trajectory.

    ``trajectory`` is the ``phase1_trajectory`` list from
    ``SolveResult.solver_metadata``: each entry carries
    ``lambda_lo, lambda_mid, lambda_hi, total_cost``. ``budget`` is the
    instance budget ``B``; we use it to compute sign flips of
    ``total_cost - B``.
    """
    if not trajectory:
        return Phase1Metrics(
            n_iter=0,
            final_lambda=0.0,
            final_gap_to_budget=0,
            lambda_rel_spread=0.0,
            sign_flips_in_window=0,
            window=window,
        )

    n_iter = len(trajectory)
    final_lambda = float(trajectory[-1]["lambda_mid"])
    final_cost = int(trajectory[-1]["total_cost"])
    final_gap = final_cost - int(budget)

    eff_window = min(window, n_iter)
    tail = trajectory[-eff_window:]
    lambdas = [float(e["lambda_mid"]) for e in tail]
    span = max(lambdas) - min(lambdas)
    denom = max(abs(v) for v in lambdas) or 1.0
    rel_spread = span / denom

    signs = [_sign(int(e["total_cost"]) - int(budget)) for e in tail]
    flips = sum(1 for a, b in pairwise(signs) if a * b < 0)

    return Phase1Metrics(
        n_iter=n_iter,
        final_lambda=final_lambda,
        final_gap_to_budget=final_gap,
        lambda_rel_spread=rel_spread,
        sign_flips_in_window=flips,
        window=eff_window,
    )


def phase3_metrics(phase3_batches: list[dict[str, Any]]) -> Phase3Metrics:
    """Summarise per-batch Phase-3 wall times."""
    if not phase3_batches:
        return Phase3Metrics(
            n_batches=0,
            total_wall_s=0.0,
            max_batch_wall_s=0.0,
            max_batch_share=0.0,
        )
    walls = [float(b["sub_milp_wall_s"]) for b in phase3_batches]
    total = sum(walls)
    mx = max(walls)
    share = mx / total if total > 0 else 0.0
    return Phase3Metrics(
        n_batches=len(walls),
        total_wall_s=total,
        max_batch_wall_s=mx,
        max_batch_share=share,
    )


def is_h1_degenerate_dual(
    metrics: Phase1Metrics,
    *,
    rel_spread_threshold: float = H1_LAMBDA_REL_SPREAD_THRESHOLD,
    sign_flip_threshold: int = H1_SIGN_FLIP_THRESHOLD,
) -> bool:
    """Return True iff Phase-1 looks degenerate (oscillating dual)."""
    if metrics.n_iter == 0:
        return False
    return (
        metrics.lambda_rel_spread > rel_spread_threshold
        or metrics.sign_flips_in_window >= sign_flip_threshold
    )


def is_h2_straggler(
    metrics: Phase3Metrics,
    *,
    max_batch_share_threshold: float = H2_MAX_BATCH_SHARE_THRESHOLD,
) -> bool:
    """Return True iff one Phase-3 batch dominates total wall time."""
    if metrics.n_batches < 2:
        return False
    return metrics.max_batch_share >= max_batch_share_threshold


def analyse_record(
    *,
    phase1_trajectory: list[dict[str, Any]],
    phase3_batches: list[dict[str, Any]],
    budget: int,
    window: int = H1_DEFAULT_WINDOW,
    rel_spread_threshold: float = H1_LAMBDA_REL_SPREAD_THRESHOLD,
    sign_flip_threshold: int = H1_SIGN_FLIP_THRESHOLD,
    max_batch_share_threshold: float = H2_MAX_BATCH_SHARE_THRESHOLD,
) -> AnomalyVerdicts:
    """End-to-end analysis of one HLD ``solver_metadata`` record."""
    p1 = phase1_metrics(phase1_trajectory, budget=budget, window=window)
    p3 = phase3_metrics(phase3_batches)
    return AnomalyVerdicts(
        phase1=p1,
        phase3=p3,
        h1_degenerate_dual=is_h1_degenerate_dual(
            p1,
            rel_spread_threshold=rel_spread_threshold,
            sign_flip_threshold=sign_flip_threshold,
        ),
        h2_straggler=is_h2_straggler(p3, max_batch_share_threshold=max_batch_share_threshold),
    )


def _sign(x: int) -> int:
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
