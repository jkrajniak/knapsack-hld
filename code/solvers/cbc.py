"""COIN-OR CBC exact MILP adapter for Selective-MCKP via PuLP."""

from __future__ import annotations

import time
from typing import Any

import pulp
from instances.schema import InstanceModel

from solvers._milp_common import extract_selection, iter_pairs, make_result
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register


class CbcAdapter:
    """COIN-OR CBC exact MILP solver for Selective-MCKP."""

    name: str = "cbc"

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        prob = pulp.LpProblem("selective_mckp", pulp.LpMaximize)
        x: dict[tuple[int, int], pulp.LpVariable] = {}
        for i, j, _p, _c in iter_pairs(instance):
            x[i, j] = pulp.LpVariable(f"x_{i}_{j}", cat=pulp.LpBinary)

        prob += pulp.lpSum(instance.items[i][j][0] * x[i, j] for (i, j) in x)
        for i in range(instance.N):
            prob += pulp.lpSum(x[i, j] for j in range(instance.M)) <= 1
        prob += pulp.lpSum(instance.items[i][j][1] * x[i, j] for (i, j) in x) <= instance.B

        kwargs: dict[str, Any] = {"msg": 0}
        if time_limit_s is not None:
            kwargs["timeLimit"] = float(time_limit_s)
        if random_seed is not None:
            kwargs["options"] = [f"randomSeed {int(random_seed) % (2**31 - 1)}"]
        cbc = pulp.PULP_CBC_CMD(**kwargs)

        t0 = time.perf_counter()
        cbc.solve(prob)
        wall = time.perf_counter() - t0

        pulp_status = pulp.LpStatus[prob.status]
        status = _translate_status(pulp_status, time_limit_s, wall)

        if status is SolverStatus.INFEASIBLE:
            return make_result(
                profit=0,
                items_selected={i: None for i in range(instance.N)},
                total_cost=0,
                wall_time_s=wall,
                status=status,
                metadata={"cbc_status": pulp_status},
            )

        def _value(i: int, j: int) -> float:
            v = x[i, j].value()
            return 0.0 if v is None else float(v)

        sel, profit, cost = extract_selection(instance, _value)

        objective = pulp.value(prob.objective)
        meta: dict[str, Any] = {
            "cbc_status": pulp_status,
            "objective_value": float(objective) if objective is not None else float(profit),
            "mip_gap": None,  # PuLP/CBC does not surface the gap by default
        }
        return make_result(
            profit=profit,
            items_selected=sel,
            total_cost=cost,
            wall_time_s=wall,
            status=status,
            metadata=meta,
        )


def _translate_status(pulp_status: str, time_limit_s: float | None, wall: float) -> SolverStatus:
    s = (pulp_status or "").lower()
    if s == "optimal":
        return SolverStatus.OPTIMAL
    if s == "infeasible":
        return SolverStatus.INFEASIBLE
    if s in ("not solved", "undefined"):
        if time_limit_s is not None and wall >= 0.95 * time_limit_s:
            return SolverStatus.TIMEOUT
        return SolverStatus.ERROR
    return SolverStatus.FEASIBLE


@register("cbc")
def _factory() -> CbcAdapter:
    return CbcAdapter()
