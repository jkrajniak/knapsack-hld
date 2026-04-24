"""SCIP exact MILP adapter for Selective-MCKP via PySCIPOpt."""

from __future__ import annotations

import time
from typing import Any

import pyscipopt as scip
from instances.schema import InstanceModel

from solvers._milp_common import extract_selection, iter_pairs, make_result
from solvers.base import SolveResult, SolverStatus
from solvers.registry import register


class ScipAdapter:
    """SCIP exact MILP solver for Selective-MCKP."""

    name: str = "scip"

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        model = scip.Model("selective_mckp")
        model.hideOutput()
        if time_limit_s is not None:
            model.setParam("limits/time", float(time_limit_s))
        if random_seed is not None:
            model.setParam("randomization/randomseedshift", int(random_seed) % (2**30))

        x = self._build(model, instance)

        t0 = time.perf_counter()
        model.optimize()
        wall = time.perf_counter() - t0

        scip_status = model.getStatus()
        status = _translate_status(scip_status)

        if model.getNSols() == 0:
            return make_result(
                profit=0,
                items_selected={i: None for i in range(instance.N)},
                total_cost=0,
                wall_time_s=wall,
                status=status,
                metadata={"scip_status": scip_status, "mip_gap": None},
            )

        def _value(i: int, j: int) -> float:
            return float(model.getVal(x[i, j]))

        sel, profit, cost = extract_selection(instance, _value)

        meta: dict[str, Any] = {
            "scip_status": scip_status,
            "mip_gap": float(model.getGap()),
            "objective_value": float(model.getObjVal()),
            "n_nodes": int(model.getNNodes()),
            "n_sols": int(model.getNSols()),
        }
        return make_result(
            profit=profit,
            items_selected=sel,
            total_cost=cost,
            wall_time_s=wall,
            status=status,
            metadata=meta,
        )

    @staticmethod
    def _build(model, instance: InstanceModel):
        x: dict[tuple[int, int], Any] = {}
        for i, j, _p, _c in iter_pairs(instance):
            x[i, j] = model.addVar(name=f"x_{i}_{j}", vtype="B")

        for i in range(instance.N):
            model.addCons(scip.quicksum(x[i, j] for j in range(instance.M)) <= 1)

        budget_expr = scip.quicksum(instance.items[i][j][1] * x[i, j] for (i, j) in x)
        model.addCons(budget_expr <= instance.B)

        model.setObjective(
            scip.quicksum(instance.items[i][j][0] * x[i, j] for (i, j) in x),
            sense="maximize",
        )
        return x


def _translate_status(s: str) -> SolverStatus:
    s = (s or "").lower()
    if s == "optimal":
        return SolverStatus.OPTIMAL
    if s == "timelimit":
        return SolverStatus.TIMEOUT
    if s == "infeasible":
        return SolverStatus.INFEASIBLE
    if s in ("gaplimit", "sollimit", "userinterrupt", "bestsollimit"):
        return SolverStatus.FEASIBLE
    return SolverStatus.ERROR


@register("scip")
def _factory() -> ScipAdapter:
    return ScipAdapter()
