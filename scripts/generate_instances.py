"""Generate the Selective-MCKP instance archive from a YAML grid spec.

Usage:

    uv run python scripts/generate_instances.py \
        --config scripts/configs/archive_smoke.yaml \
        --out instances

Each grid cell `(N, M, correlation, f)` is generated for every seed in
the grid, files are written under
`instances/{correlation}/N{N}_M{M}/mckp_..json.gz`, and a SHA-256
`MANIFEST.json` is rebuilt.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "code"))

import yaml
from joblib import Parallel, delayed

from instances import (
    CorrelationKind,
    build_manifest,
    generate_instance,
    instance_id,
    save_instance,
    write_manifest,
)

LOG = logging.getLogger("generate_instances")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--jobs", type=int, default=-1, help="parallel workers (-1 = all CPUs)")
    p.add_argument("--force", action="store_true", help="overwrite existing files")
    p.add_argument("--no-manifest", action="store_true", help="skip manifest rebuild")
    return p.parse_args()


def _load_config(path: Path) -> dict:
    with path.open() as fh:
        return yaml.safe_load(fh)


def _generate_one(
    *,
    N: int,
    M: int,
    correlation: str,
    f: float,
    seed: int,
    out_root: Path,
    force: bool,
) -> Path:
    stem = instance_id(N=N, M=M, correlation=correlation, f=f, seed=seed)
    rel_dir = Path(correlation) / f"N{N}_M{M}"
    target = out_root / rel_dir / f"{stem}.json.gz"
    if target.exists() and not force:
        return target
    inst = generate_instance(N=N, M=M, correlation=correlation, f=f, seed=seed)
    return save_instance(inst, target)


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    cfg = _load_config(args.config)

    Ns = cfg["N"]
    Ms = cfg["M"]
    correlations = cfg["correlation"]
    fs = cfg["f"]
    seeds = cfg["seeds"]

    for c in correlations:
        CorrelationKind(c)  # raises if unknown

    jobs = [
        dict(N=N, M=M, correlation=c, f=f, seed=seed, out_root=args.out, force=args.force)
        for N in Ns
        for M in Ms
        for c in correlations
        for f in fs
        for seed in seeds
    ]
    LOG.info("planning %d instances over %d cells", len(jobs), len(jobs) // len(seeds))

    t0 = time.perf_counter()
    Parallel(n_jobs=args.jobs, verbose=10)(delayed(_generate_one)(**j) for j in jobs)
    LOG.info("generation done in %.1fs", time.perf_counter() - t0)

    if not args.no_manifest:
        LOG.info("building manifest…")
        manifest = build_manifest(args.out)
        write_manifest(manifest, args.out)
        LOG.info("manifest contains %d files", len(manifest["files"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
