"""Regression test: Greedy-MaxProfit must NOT be in the registry.

Reviewer R1 (M5) flagged Greedy-MaxProfit as a redundant baseline. Per
the revision plan (Q6) we drop it. This test fails loudly if anyone
re-registers it.
"""

from __future__ import annotations

import pytest
from solvers import get_solver, list_solvers


def test_greedy_max_profit_is_not_registered() -> None:
    assert "greedy_max_profit" not in list_solvers()


def test_get_solver_raises_for_greedy_max_profit() -> None:
    with pytest.raises(KeyError, match="greedy_max_profit"):
        get_solver("greedy_max_profit")
