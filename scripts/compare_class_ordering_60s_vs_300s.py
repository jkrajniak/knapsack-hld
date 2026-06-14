# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Paired 60s-vs-300s comparison for the class-ordering ablation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ORDERINGS = ("sequential", "random", "adversarial")
JOIN_KEYS = ("instance_id", "class_ordering")


def load_run(root: Path) -> pd.DataFrame:
    frames = []
    for ordering in ORDERINGS:
        df = pd.read_csv(root / f"{ordering}.csv")
        df["class_ordering"] = ordering
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def compare(primary: pd.DataFrame, sub: pd.DataFrame) -> pd.DataFrame:
    cols = [*JOIN_KEYS, "status", "profit", "wall_time_s"]
    merged = primary[cols].merge(
        sub[cols],
        on=list(JOIN_KEYS),
        suffixes=("_60s", "_300s"),
        how="inner",
    )

    rows = []
    for ordering, sub_df in merged.groupby("class_ordering"):
        seeds = len(sub_df)
        delta = sub_df["profit_300s"] - sub_df["profit_60s"]

        win_300 = int((delta > 0).sum())
        win_60 = int((delta < 0).sum())
        tied = int((delta == 0).sum())

        denom = sub_df["profit_60s"].replace(0, pd.NA)
        pct = (delta / denom) * 100
        median_pct = float(pct.dropna().median()) if pct.notna().any() else float("nan")
        mean_pct = float(pct.dropna().mean()) if pct.notna().any() else float("nan")

        timeout_60 = sub_df["status_60s"].eq("timeout").sum()
        escaped = int(
            ((sub_df["status_60s"] == "timeout") & (sub_df["status_300s"] != "timeout")).sum()
        )
        escape_rate = (escaped / timeout_60 * 100) if timeout_60 else float("nan")

        rows.append(
            {
                "ordering": ordering,
                "seeds": seeds,
                "60s_wins": win_60,
                "300s_wins": win_300,
                "tied": tied,
                "median_dprofit_pct": median_pct,
                "mean_dprofit_pct": mean_pct,
                "timeouts_60s": int(timeout_60),
                "timeout_escape_rate_pct": escape_rate,
            }
        )

    return pd.DataFrame(rows).set_index("ordering")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-dir", type=Path, required=True)
    parser.add_argument("--sub-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    primary = load_run(args.primary_dir)
    sub = load_run(args.sub_dir)

    table = compare(primary, sub)
    pd.options.display.float_format = "{:.4f}".format
    print(table.to_string())
    print()
    print("Markdown:")
    print()
    print(
        "| Ordering    | seeds | 60s win | 300s win | tied | median Δprofit (%) | timeout escape rate |"
    )
    print(
        "|-------------|------:|--------:|---------:|-----:|-------------------:|--------------------:|"
    )
    for ordering, row in table.iterrows():
        esc = (
            f"{row['timeout_escape_rate_pct']:.1f}%"
            if pd.notna(row["timeout_escape_rate_pct"])
            else "—"
        )
        median = (
            f"{row['median_dprofit_pct']:+.3f}%" if pd.notna(row["median_dprofit_pct"]) else "—"
        )
        print(
            f"| {ordering:<11} | {int(row['seeds']):>5} | "
            f"{int(row['60s_wins']):>7} | {int(row['300s_wins']):>8} | "
            f"{int(row['tied']):>4} | {median:>18} | {esc:>19} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
