"""Phase-D §4.3 anomaly investigation utilities.

The HLD adapter already emits the per-iteration Phase-1 trajectory and
the per-batch Phase-3 wall times in ``SolveResult.solver_metadata``
(see ``solvers.hld``). This package adds:

- :mod:`anomalies.sweep` — runs HLD across a grid of ``N_iter`` values
  on a fixed deterministic anomaly subset, captures the metadata, and
  records a HiGHS reference profit.
- :mod:`anomalies.analyse` — pure functions that turn one sweep record
  into hypothesis-test verdicts:

  - **H1 — degenerate dual basis.** The Phase-1 binary search fails to
    converge (lambda oscillates and the cost--budget gap repeatedly
    flips sign), so the data-driven budget allocation is driven by a
    noisy ``lambda_est``.
  - **H2 — sub-MILP straggler.** One Phase-3 batch dominates the total
    wall time, masking any benefit from the budget split.

Each verdict is computed from the recorded metadata only, so the
analysis is deterministic and replayable without re-running HLD.
"""

from anomalies.analyse import (
    H1_LAMBDA_REL_SPREAD_THRESHOLD,
    H1_SIGN_FLIP_THRESHOLD,
    H2_MAX_BATCH_SHARE_THRESHOLD,
    AnomalyVerdicts,
    Phase1Metrics,
    Phase3Metrics,
    analyse_record,
    is_h1_degenerate_dual,
    is_h2_straggler,
    phase1_metrics,
    phase3_metrics,
)

__all__ = [
    "H1_LAMBDA_REL_SPREAD_THRESHOLD",
    "H1_SIGN_FLIP_THRESHOLD",
    "H2_MAX_BATCH_SHARE_THRESHOLD",
    "AnomalyVerdicts",
    "Phase1Metrics",
    "Phase3Metrics",
    "analyse_record",
    "is_h1_degenerate_dual",
    "is_h2_straggler",
    "phase1_metrics",
    "phase3_metrics",
]
