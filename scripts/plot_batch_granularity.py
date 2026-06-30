#!/usr/bin/env python3
"""Headline figure for the batch-granularity / allocation-error result.

One panel per budget tightness f. Each panel plots equal-split (PO) and HLD
allocation error vs the oracle as a function of batch size (classes/batch),
overlaying two correlation families. The center line is the mean over N; the
shaded band spans min..max over N -- a thin band is the visual signature of the
N-invariance (the concentration mechanism). The story in one figure:

  * curves collapse across the N range (thin band)  -> error is bs-governed;
  * inversely_strongly sits far above strongly       -> magnitude is heterogeneity;
  * HLD dips below PO only for inversely_strongly at  -> benefit is conditional
    small bs, and rises above PO on strongly.            (heterogeneity x granularity).

Only clean rows feed the figure: n_timeout == 0 and a tight oracle
(|oracle_gap_pct| <= --max-oracle-gap), matching analyze_batch_granularity.py.

Emits a PDF plus a sibling <name>.meta.json provenance sidecar.

Usage:
    PYTHONPATH=code uv run python scripts/plot_batch_granularity.py \
        --csv results/batch_granularity/inversely_strongly.csv \
        --csv results/batch_granularity/strongly.csv \
        --out-dir figures
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette; one hue per correlation family.
FAMILY_COLOR = {"inversely_strongly": "#0072B2", "strongly": "#D55E00"}
FAMILY_LABEL = {"inversely_strongly": "inversely_strongly", "strongly": "strongly"}
# Method = line style + marker (reinforcing); family = colour.
METHOD_STYLE = {"po": ("-", "o", "equal-split (PO)"), "hld": ("--", "s", "HLD")}
Y_FLOOR = 1e-2  # symlog linear threshold; also keeps exact-0 gaps visible
SUPTITLE = "Allocation error is set by batch size, not N"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        action="append",
        required=True,
        help="batch-granularity CSV (repeat for each family)",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures"))
    parser.add_argument("--name", default="batch_granularity_allocation_error")
    parser.add_argument(
        "--max-oracle-gap",
        type=float,
        default=0.5,
        help="drop rows whose oracle dual gap exceeds this (pct)",
    )
    return parser.parse_args(argv)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_clean(csv_paths: list[Path], max_oracle_gap: float) -> tuple[dict, int, int]:
    """Return mean-over-seeds error keyed by (family, f, method, bs, n)."""
    cells: dict[tuple, list[float]] = defaultdict(list)
    total = 0
    dropped = 0
    for path in csv_paths:
        for r in csv.DictReader(path.open(newline="")):
            total += 1
            if int(r["n_timeout"]) > 0:
                dropped += 1
                continue
            og = r["oracle_gap_pct"]
            if og not in ("", None) and abs(float(og)) > max_oracle_gap:
                dropped += 1
                continue
            key = (r["correlation"], float(r["f"]), r["method"], int(r["bs_target"]), int(r["n"]))
            cells[key].append(float(r["gap_oracle_pct"]))
    agg = {k: mean(v) for k, v in cells.items()}
    return agg, total, dropped


def _series(agg: dict, family: str, f: float, method: str, bss: list[int]):
    """Center (mean over N) and band (min..max over N) per batch size."""
    xs, center, lo, hi = [], [], [], []
    for bs in bss:
        per_n = [
            v
            for (fam, ff, m, b, _n), v in agg.items()
            if fam == family and ff == f and m == method and b == bs
        ]
        if not per_n:
            continue
        xs.append(bs)
        center.append(max(mean(per_n), 0.0))
        lo.append(max(min(per_n), 0.0))
        hi.append(max(max(per_n), 0.0))
    return xs, center, lo, hi


def make_figure(agg: dict, name: str, out_dir: Path) -> Path:
    fs = sorted({f for (_fam, f, _m, _b, _n) in agg})
    bss = sorted({b for (_fam, _f, _m, b, _n) in agg})
    families = sorted({fam for (fam, _f, _m, _b, _n) in agg})

    fig, axes = plt.subplots(1, len(fs), figsize=(4.6 * len(fs), 4.2), sharey=True)
    if len(fs) == 1:
        axes = [axes]

    for ax, f in zip(axes, fs, strict=True):
        for family in families:
            color = FAMILY_COLOR.get(family, "0.4")
            for method, (ls, marker, _lbl) in METHOD_STYLE.items():
                xs, center, lo, hi = _series(agg, family, f, method, bss)
                if not xs:
                    continue
                ax.plot(
                    xs, center, ls=ls, marker=marker, color=color, markersize=3.5, linewidth=1.6
                )
                ax.fill_between(xs, lo, hi, color=color, alpha=0.12, linewidth=0)
        ax.set_xscale("log", base=2)
        ax.set_yscale("symlog", linthresh=Y_FLOOR)
        ax.set_xticks(bss)
        ax.set_xticklabels([str(b) for b in bss])
        ax.set_xlabel("batch size (classes / batch)")
        ax.set_title(f"f = {f:g}  (budget tightness)")
        ax.grid(True, which="major", alpha=0.3)
    axes[0].set_ylabel("allocation error vs oracle (%)")

    # One shared legend (family x method = 4 fully-specified entries) outside both panels.
    handles = [
        plt.Line2D(
            [],
            [],
            color=FAMILY_COLOR.get(fam, "0.4"),
            ls=ls,
            marker=marker,
            markersize=3.5,
            label=f"{FAMILY_LABEL.get(fam, fam)} \u00b7 {lbl}",
        )
        for fam in families
        for _m, (ls, marker, lbl) in METHOD_STYLE.items()
    ]
    fig.suptitle(SUPTITLE, fontsize=11, y=0.98)
    fig.tight_layout(rect=(0, 0, 1.0, 0.93))
    legend = fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        frameon=True,
        framealpha=0.9,
        title="band = min..max over N",
        title_fontsize=8,
    )
    fig._bg_extra_artists = [legend]  # captured by bbox_inches="tight" on save

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = out_dir / f"{name}.pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def write_meta(pdf_path: Path, csv_paths: list[Path], total: int, dropped: int) -> None:
    meta = {
        "figure": pdf_path.stem,
        "script": "scripts/plot_batch_granularity.py",
        "command": "scripts/plot_batch_granularity.py "
        + " ".join(shlex.quote(a) for a in sys.argv[1:]),
        "generated_at": dt.datetime.now(dt.UTC).isoformat(),
        "source_csv": [{"path": str(p), "sha256": _sha256(p)} for p in csv_paths],
        "rows_total": total,
        "rows_clean": total - dropped,
        "rows_dropped": dropped,
        "notes": "Clean rows only (n_timeout==0, tight oracle). Center=mean over N; "
        "band=min..max over N (thin band => N-invariance).",
    }
    meta_path = pdf_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    agg, total, dropped = load_clean(args.csv, args.max_oracle_gap)
    if not agg:
        print("no clean rows; nothing to plot", file=sys.stderr)
        return 1
    pdf_path = make_figure(agg, args.name, args.out_dir)
    write_meta(pdf_path, args.csv, total, dropped)
    print(f"generated: {pdf_path} (+ .meta.json) | clean {total - dropped}/{total} rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
