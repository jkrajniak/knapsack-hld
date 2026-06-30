"""Is equal-split allocation error governed by N or by batch granularity?

Decisive test of the concentration hypothesis. Equal-split forces every batch
the same budget B/K; the penalty depends on cross-batch heterogeneity, which
self-averages as classes-per-batch (the batch size, bs = N/K) grows. The
prediction: PO error is a function of bs ALONE, not N -- so error-vs-bs curves
at different N should overlap, and error should fall as bs grows.

For each (N, seed) we solve an exact oracle (HiGHS, tight dual bound) once, then
sweep batch size bs by setting K = round(N/bs). Sub-problems are kept small so
they solve to optimality (n_timeout must be 0; otherwise the row is Error-B /
timeout-confounded, not a clean Error-A / allocation measurement). We log gap
vs the oracle dual upper bound and HLD recovery.

Resumable: skips (N, f, seed, bs) rows already in --out-csv.

Usage:
    PYTHONPATH=code uv run python scripts/check_batch_granularity.py \
        --out-csv results/batch_granularity/inversely_strongly.csv \
        --correlation inversely_strongly --m 10 \
        --ns 200 1000 3000 10000 --fs 0.500 0.900 --seeds 0 1 2 \
        --batch-sizes 2 4 8 16 32
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from heuristics.partition_optimal import PartitionOptimalAdapter
from instances.io import load_instance
from solvers.guarded_hld import DEFAULT_TAU_SKIP, GuardedHldAdapter
from solvers.highs import HighsAdapter
from solvers.hld import HldAdapter

CFG = json.loads(Path("configs/hld_smac_best.json").read_text())

OUT_FIELDS = (
    "correlation",
    "f",
    "seed",
    "n",
    "m",
    "bs_target",
    "k",
    "batch_size",
    "oracle_profit",
    "oracle_status",
    "oracle_dual_ub",
    "oracle_gap_pct",
    "method",
    "profit",
    "fill_pct",
    "gap_oracle_pct",
    "n_timeout",
    "n_batches",
    "wall_s",
    # guarded_hld only (empty for po/hld rows)
    "decision",
    "wall_po_s",
    "wall_hld_s",
    "po_gap_to_ul",
    "tau_skip",
)


def _n_timeout(batches: list[dict[str, Any]]) -> int:
    return sum(1 for d in batches if "timeout" in str(d["status"]).lower())


def _solve_methods(
    inst,
    k: int,
    time_limit: float,
    *,
    methods: frozenset[str],
    tau_skip: float,
) -> dict[str, tuple[Any, list[dict[str, Any]], dict[str, Any]]]:
    """Return {method: (result, batches, extra)} for requested methods at K=k."""
    common = dict(sub_solver_threads=1, batch_jobs=8)
    out: dict[str, tuple[Any, list[dict[str, Any]], dict[str, Any]]] = {}

    if "po" in methods:
        po = PartitionOptimalAdapter(k=k, **common).solve(
            inst, time_limit_s=time_limit, random_seed=0
        )
        out["po"] = (po, po.solver_metadata["batches"], {})

    if "hld" in methods:
        hld = HldAdapter(
            n_iter=CFG["n_iter"],
            alpha=CFG["alpha"],
            k=k,
            lambda_max_override=CFG["lambda_max"],
            rebalance_rounds=0,
            **common,
        ).solve(inst, time_limit_s=time_limit, random_seed=0)
        out["hld"] = (hld, hld.solver_metadata["phase3_batches"], {})

    if "guarded_hld" in methods:
        gd = GuardedHldAdapter(
            n_iter=CFG["n_iter"],
            alpha=CFG["alpha"],
            k=k,
            lambda_max_override=CFG["lambda_max"],
            rebalance_rounds=0,
            tau_skip=tau_skip,
            **common,
        ).solve(inst, time_limit_s=time_limit * 2, random_seed=0)
        meta = gd.solver_metadata
        po_batches = meta["sub"]["po"]["batches"]
        hld_meta = meta["sub"]["hld"]
        hld_batches = hld_meta["phase3_batches"] if hld_meta else []
        batches = po_batches + hld_batches
        extra = {
            "decision": meta["decision"],
            "wall_po_s": round(meta["wall_po_s"], 3),
            "wall_hld_s": round(meta["wall_hld_s"], 3),
            "po_gap_to_ul": round(meta["po_gap_to_ul"], 6),
            "tau_skip": meta["tau_skip"],
        }
        out["guarded_hld"] = (gd, batches, extra)

    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-csv", type=Path, required=True)
    parser.add_argument("--correlation", default="inversely_strongly")
    parser.add_argument("--m", type=int, default=10)
    parser.add_argument("--ns", type=int, nargs="+", default=[200, 1000, 3000, 10000])
    parser.add_argument("--fs", nargs="+", default=["0.500", "0.900"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[2, 4, 8, 16, 32])
    parser.add_argument("--oracle-time", type=float, default=300.0)
    parser.add_argument("--sub-time", type=float, default=120.0)
    parser.add_argument(
        "--methods",
        default="po,hld",
        help="Comma-separated: po, hld, guarded_hld (default: po,hld)",
    )
    parser.add_argument(
        "--tau-skip",
        type=float,
        default=DEFAULT_TAU_SKIP,
        help="Guarded-HLD Lagrangian-UB skip threshold (default 0.005)",
    )
    args = parser.parse_args()
    methods = frozenset(m.strip() for m in args.methods.split(",") if m.strip())
    unknown = methods - {"po", "hld", "guarded_hld"}
    if unknown:
        raise SystemExit(f"unknown --methods: {sorted(unknown)}")

    done: set[tuple[str, float, int, int, int, str]] = set()
    if args.out_csv.exists():
        with args.out_csv.open(newline="") as fh:
            for r in csv.DictReader(fh):
                done.add(
                    (
                        r["correlation"],
                        float(r["f"]),
                        int(r["seed"]),
                        int(r["n"]),
                        int(r["bs_target"]),
                        r["method"],
                    )
                )

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    write_header = not args.out_csv.exists()
    fh = args.out_csv.open("a", newline="")
    writer = csv.DictWriter(fh, fieldnames=OUT_FIELDS)
    if write_header:
        writer.writeheader()
        fh.flush()

    corr = args.correlation
    for n in args.ns:
        for f in args.fs:
            for seed in args.seeds:
                bss = [bs for bs in args.batch_sizes if bs <= n]
                if all((corr, float(f), seed, n, bs, m) in done for bs in bss for m in methods):
                    print(f"skip N{n} f{f} seed{seed} (all bs x methods done)", flush=True)
                    continue
                path = Path(
                    f"instances/{corr}/N{n}_M{args.m}/"
                    f"mckp_N{n}_M{args.m}_{corr}_f{f}_seed{seed}.json.gz"
                )
                if not path.exists():
                    print(f"MISSING {path}", flush=True)
                    continue
                inst = load_instance(str(path))
                orc = HighsAdapter(mip_rel_gap=1e-9, threads=8).solve(
                    inst, time_limit_s=args.oracle_time, random_seed=0
                )
                dual_ub = orc.solver_metadata.get("mip_dual_bound") or orc.profit
                ub = max(orc.profit, dual_ub)
                orc_gap = (dual_ub - orc.profit) / ub * 100 if ub else None

                for bs in bss:
                    if all((corr, float(f), seed, n, bs, m) in done for m in methods):
                        continue
                    k = max(1, min(n, round(n / bs)))
                    solved = _solve_methods(
                        inst,
                        k,
                        args.sub_time,
                        methods=methods,
                        tau_skip=args.tau_skip,
                    )
                    for method, (res, batches, extra) in solved.items():
                        if (corr, float(f), seed, n, bs, method) in done:
                            continue
                        gap = (ub - res.profit) / ub * 100 if ub else None
                        row = {
                            "correlation": corr,
                            "f": inst.f,
                            "seed": seed,
                            "n": n,
                            "m": args.m,
                            "bs_target": bs,
                            "k": k,
                            "batch_size": round(n / k, 1),
                            "oracle_profit": orc.profit,
                            "oracle_status": str(orc.status),
                            "oracle_dual_ub": round(dual_ub, 1),
                            "oracle_gap_pct": round(orc_gap, 4) if orc_gap is not None else None,
                            "method": method,
                            "profit": res.profit,
                            "fill_pct": round(res.total_cost / inst.B * 100, 3),
                            "gap_oracle_pct": round(gap, 4) if gap is not None else None,
                            "n_timeout": _n_timeout(batches),
                            "n_batches": len(batches),
                            "wall_s": round(res.wall_time_s, 1),
                            "decision": extra.get("decision", ""),
                            "wall_po_s": extra.get("wall_po_s", ""),
                            "wall_hld_s": extra.get("wall_hld_s", ""),
                            "po_gap_to_ul": extra.get("po_gap_to_ul", ""),
                            "tau_skip": extra.get("tau_skip", ""),
                        }
                        writer.writerow(row)
                    fh.flush()
                    parts = []
                    for m in ("po", "hld", "guarded_hld"):
                        if m not in solved:
                            continue
                        res, batches, extra = solved[m]
                        g = (ub - res.profit) / ub * 100 if ub else 0.0
                        tag = m.upper() if m != "guarded_hld" else "GD"
                        suffix = f" [{extra['decision']}]" if m == "guarded_hld" else ""
                        parts.append(f"{tag} gap={g:7.3f}% (TO {_n_timeout(batches)}){suffix}")
                    print(
                        f"N{n:6d} f{f} seed{seed} bs={bs:3d} k={k:4d} "
                        f"bs_eff={n / k:6.1f} | {' '.join(parts)} "
                        f"orc {orc.status} gap={orc_gap:.4f}%",
                        flush=True,
                    )
    fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
