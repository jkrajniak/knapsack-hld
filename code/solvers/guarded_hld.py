"""Guarded Hybrid Lagrangian-Decomposition (guarded-HLD) for Selective-MCKP.

Composition of Partition-Optimal (PO, equal-split) and HLD that removes HLD's
structural downside at the cost of one extra small-sub-MILP pass.

Why a guard is needed
---------------------
HLD's Lagrangian budget reallocation helps on heterogeneous instances at small
batch size but *harms* on homogeneous + loose-budget instances (e.g.
`strongly`/f=0.9), where equal-split is already near-optimal and the
cost-proportional reallocation moves away from a good split. The harm is up to
+5pp and is not predictable from any cheap instance-observable scalar (PO gap,
fill fraction, shadow price) because the success and failure regimes overlap on
all of them. The only safe guarantee is therefore a **never-lose acceptance
rule**: return HLD only when it beats PO, else return PO.

Algorithm
---------
1. Solve PO (equal-split, K batches) -> ``po_res``.
2. Compute HLD Phase-1 (binary search for ``lambda_est``, pure arithmetic) and
   the Lagrangian upper bound
       U_L = sum_i max(0, max_j (p_ij - lambda_est * c_ij)) + lambda_est * B
   which is a free byproduct of Phase-1 (no extra MILP).
3. **Skip pre-check.** If ``(U_L - po_res.profit) / U_L < tau_skip`` then PO is
   already within ``tau_skip`` of the optimum ceiling, so there is no
   recoverable allocation error; return PO and skip HLD's Phase-3 entirely.
4. **Never-lose backstop.** Otherwise run full HLD -> ``hld_res`` and return
   whichever of ``po_res`` / ``hld_res`` has higher profit (ties go to PO, the
   cheaper solution). This guarantees the guarded profit is never below PO's.

Cost
----
Worst case ~2x a single decomposition's wall time (both PO and HLD Phase-3
solved). The skip pre-check avoids the HLD pass on cells where PO is already
near-optimal. Per-batch sub-MILPs stay small (that is the point of
decomposition), so the doubling is on small solves, not a monolithic one.

Solver-metadata schema
----------------------
- ``decision``      -- "skip" | "po_wins" | "hld_wins"
- ``po_profit``     -- PO achieved profit
- ``hld_profit``    -- HLD achieved profit, or None if skipped
- ``lagrangian_ub`` -- U_L (free Phase-1 upper bound)
- ``lambda_est``    -- Phase-1 shadow price used for U_L and HLD
- ``po_gap_to_ul``  -- (U_L - po_profit) / U_L
- ``tau_skip``      -- the skip threshold applied
- ``wall_po_s``     -- PO wall time
- ``wall_hld_s``    -- HLD wall time (0.0 if skipped)
- ``sub``           -- {"po": po_res.solver_metadata, "hld": hld_res.solver_metadata|None}
"""

from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

from heuristics.partition_optimal import PartitionOptimalAdapter
from instances.schema import InstanceModel

from solvers.base import SolveResult, SolverStatus
from solvers.hld import (
    HldAdapter,
    _instance_dependent_lambda_max,
    _phase1_binary_search,
)
from solvers.registry import register

DEFAULT_TAU_SKIP = 0.005  # 0.5% of U_L: skip HLD when PO is already this close to ceiling


def _lagrangian_upper_bound(instance: InstanceModel, lam: float) -> float:
    """U(lam) = sum_i max(0, max_j (p_ij - lam*c_ij)) + lam * B  (valid UB for lam>=0)."""
    value = 0.0
    for i in range(instance.N):
        best = 0.0
        for j in range(instance.M):
            p, c = instance.items[i][j]
            v = float(p) - lam * float(c)
            if v > best:
                best = v
        value += best
    return value + lam * float(instance.B)


class GuardedHldAdapter:
    """Never-lose composition of equal-split (PO) and HLD with a Lagrangian-UB skip."""

    name: str = "guarded_hld"

    def __init__(
        self,
        *,
        n_iter: int = 35,
        alpha: float = 0.998,
        k: int = 8,
        sub_solver: str = "highs",
        sub_solver_threads: int | None = None,
        batch_jobs: int | None = None,
        lambda_max_override: float | None = None,
        class_ordering: str = "sequential",
        rebalance_rounds: int = 0,
        tau_skip: float = DEFAULT_TAU_SKIP,
    ) -> None:
        if not 0.0 <= tau_skip <= 1.0:
            raise ValueError(f"tau_skip must be in [0, 1], got {tau_skip}")
        self.tau_skip = float(tau_skip)
        self._hld_params = dict(
            n_iter=n_iter, alpha=alpha, k=k, sub_solver=sub_solver,
            sub_solver_threads=sub_solver_threads, batch_jobs=batch_jobs,
            lambda_max_override=lambda_max_override,
            class_ordering=class_ordering, rebalance_rounds=rebalance_rounds,
        )
        self._po_params = dict(
            k=k, sub_solver=sub_solver, sub_solver_threads=sub_solver_threads,
            batch_jobs=batch_jobs,
        )

    def _build_po(self) -> PartitionOptimalAdapter:
        return PartitionOptimalAdapter(**self._po_params)

    def _build_hld(self) -> HldAdapter:
        return HldAdapter(**self._hld_params)

    def _lambda_max(self, instance: InstanceModel) -> float:
        override = self._hld_params["lambda_max_override"]
        if override is not None:
            return float(override)
        return _instance_dependent_lambda_max(instance)

    def solve(
        self,
        instance: InstanceModel,
        *,
        time_limit_s: float | None = None,
        random_seed: int | None = None,
    ) -> SolveResult:
        t0 = time.perf_counter()

        po_res = self._build_po().solve(instance, time_limit_s=time_limit_s, random_seed=random_seed)
        wall_po = time.perf_counter() - t0

        # Phase-1 is pure arithmetic; reuse HLD's machinery so lambda_est matches
        # exactly what a full HLD run would use.
        lambda_max = self._lambda_max(instance)
        lambda_est, _traj = _phase1_binary_search(
            instance, lambda_max=lambda_max, n_iter=self._hld_params["n_iter"]
        )
        ub = _lagrangian_upper_bound(instance, lambda_est)
        po_gap_to_ul = (ub - po_res.profit) / ub if ub > 0 else 0.0

        if po_gap_to_ul < self.tau_skip:
            return self._finalize(
                instance, decision="skip", po_res=po_res, hld_res=None,
                lambda_est=lambda_est, ub=ub, po_gap_to_ul=po_gap_to_ul,
                wall_po=wall_po, wall_hld=0.0, t0=t0,
            )

        hld_t0 = time.perf_counter()
        remaining = self._remaining_time(time_limit_s, t0)
        hld_res = self._build_hld().solve(instance, time_limit_s=remaining, random_seed=random_seed)
        wall_hld = time.perf_counter() - hld_t0

        if hld_res.profit > po_res.profit:
            decision = "hld_wins"
        else:
            decision = "po_wins"
        return self._finalize(
            instance, decision=decision, po_res=po_res, hld_res=hld_res,
            lambda_est=lambda_est, ub=ub, po_gap_to_ul=po_gap_to_ul,
            wall_po=wall_po, wall_hld=wall_hld, t0=t0,
            winner=hld_res if decision == "hld_wins" else po_res,
        )

    def _remaining_time(self, time_limit_s: float | None, t0: float) -> float | None:
        if time_limit_s is None:
            return None
        return max(0.001, float(time_limit_s) - (time.perf_counter() - t0))

    def _finalize(
        self,
        instance: InstanceModel,
        *,
        decision: str,
        po_res: SolveResult,
        hld_res: SolveResult | None,
        lambda_est: float,
        ub: float,
        po_gap_to_ul: float,
        wall_po: float,
        wall_hld: float,
        t0: float,
        winner: SolveResult | None = None,
    ) -> SolveResult:
        base = winner if winner is not None else po_res
        meta: dict[str, Any] = {
            "decision": decision,
            "po_profit": po_res.profit,
            "hld_profit": hld_res.profit if hld_res is not None else None,
            "lagrangian_ub": ub,
            "lambda_est": lambda_est,
            "po_gap_to_ul": po_gap_to_ul,
            "tau_skip": self.tau_skip,
            "wall_po_s": wall_po,
            "wall_hld_s": wall_hld,
            "sub": {
                "po": po_res.solver_metadata,
                "hld": hld_res.solver_metadata if hld_res is not None else None,
            },
        }
        return replace(
            base,
            wall_time_s=time.perf_counter() - t0,
            status=SolverStatus.FEASIBLE,
            solver_metadata=meta,
        )


@register("guarded_hld")
def _factory() -> GuardedHldAdapter:
    return GuardedHldAdapter()
