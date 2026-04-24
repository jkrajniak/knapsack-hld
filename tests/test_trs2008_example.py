"""Phase C §3.4.2: placeholder for the §5 worked example from TRS-2008.

The original Tsesmetzis-Roussaki-Sykas (2008) paper is closed-access
(EJOR / Elsevier; no preprint or open-access mirror was available at
implementation time). The §5 numerical example from that paper has
therefore not been transcribed into this test file yet.

This test is marked `xfail(strict=True)` so that:

- it reminds the maintainer that the canonical paper-fixture is still
  missing, and
- it WILL flip to `XPASS` (a hard failure under `strict=True`) the
  moment the example is transcribed and the implementation reproduces
  it correctly. That way the gate cannot be silently forgotten.

When the §5 example is to hand:

1. Replace `_TODO_PAPER_INSTANCE` with the exact (N, M, profits,
   costs, B) from the paper.
2. Replace `_TODO_PAPER_EXPECTED_PROFIT` with the published optimum.
3. Remove the `xfail` marker.

Tracked in `openspec/changes/itor-major-revision-2026/tasks.md` §3.4.2
as PENDING (paper-access gated).
"""

from __future__ import annotations

import pytest
from solvers import get_solver, validate_solution


@pytest.mark.xfail(
    strict=True,
    reason=(
        "TRS-2008 §5 worked example not transcribed yet "
        "(closed-access EJOR paper, see tasks.md §3.4.2)."
    ),
)
def test_trs2008_reproduces_paper_section5_example() -> None:
    """Reproduce the §5 worked numerical example from TRS-2008.

    Pending: paper access. See module docstring.
    """
    pytest.importorskip("instances.schema")
    from instances.schema import CorrelationKind, InstanceModel

    instance = InstanceModel(
        N=1,
        M=1,
        correlation=CorrelationKind.UNCORRELATED,
        f=0.5,
        seed=0,
        B=1,
        items=[[[1, 1]]],
    )
    expected_profit = -1

    result = get_solver("trs2008").solve(instance)
    validate_solution(instance, result)
    assert result.profit == expected_profit
