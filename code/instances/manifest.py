"""SHA-256 manifest for the instance archive.

The manifest records, for every instance file under `instances/`:

    {
      "schema_version": "1",
      "generator_version": "0.1.0",
      "tuning_ratio": 0.30,
      "master_seed": 20260424,
      "files": [
        {"path": "uncorrelated/N1000_M5/...json.gz",
         "sha256": "...",
         "bytes": 12345,
         "cell": {"N": 1000, "M": 5, "correlation": "...", "f": 0.5},
         "seed": 0,
         "subset": "test"}
      ]
    }

The verifier re-hashes every file and confirms that every cell has at
least the expected number of seeds.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from instances.io import load_instance
from instances.split import (
    DEFAULT_MASTER_SEED,
    DEFAULT_TUNING_RATIO,
    CellKey,
    split_seeds,
)

MANIFEST_SCHEMA_VERSION = "1"
MANIFEST_FILENAME = "MANIFEST.json"


@dataclass(frozen=True)
class FileEntry:
    path: str
    sha256: str
    bytes: int
    cell: dict
    seed: int
    subset: str


def build_manifest(
    archive_root: Path,
    *,
    tuning_ratio: float = DEFAULT_TUNING_RATIO,
    master_seed: int = DEFAULT_MASTER_SEED,
    instance_glob: str = "**/mckp_*.json*",
) -> dict:
    """Walk `archive_root` and build a SHA-256 manifest.

    Loads every matching file once to extract its `(cell, seed)` and
    label it `tuning` or `test` according to the deterministic split.
    """
    archive_root = Path(archive_root)
    files: list[FileEntry] = []
    seeds_per_cell: dict[CellKey, list[int]] = defaultdict(list)
    paths_per_cell: dict[CellKey, list[tuple[int, Path]]] = defaultdict(list)

    for p in sorted(archive_root.glob(instance_glob)):
        inst = load_instance(p)
        cell = CellKey.from_instance(inst)
        seeds_per_cell[cell].append(inst.seed)
        paths_per_cell[cell].append((inst.seed, p))

    for cell, items in paths_per_cell.items():
        split = split_seeds(
            seeds_per_cell[cell],
            cell=cell,
            tuning_ratio=tuning_ratio,
            master_seed=master_seed,
        )
        for seed, p in sorted(items):
            subset = "tuning" if seed in split.tuning else "test"
            files.append(
                FileEntry(
                    path=str(p.relative_to(archive_root)),
                    sha256=_sha256(p),
                    bytes=p.stat().st_size,
                    cell={
                        "N": cell.N,
                        "M": cell.M,
                        "correlation": cell.correlation.value,
                        "f": cell.f,
                    },
                    seed=seed,
                    subset=subset,
                )
            )

    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "tuning_ratio": tuning_ratio,
        "master_seed": master_seed,
        "files": [asdict(fe) for fe in files],
    }


def write_manifest(manifest: dict, archive_root: Path) -> Path:
    path = Path(archive_root) / MANIFEST_FILENAME
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return path


def verify_manifest(archive_root: Path) -> tuple[bool, list[str]]:
    """Re-hash every recorded file and return (ok, list-of-errors)."""
    archive_root = Path(archive_root)
    manifest_path = archive_root / MANIFEST_FILENAME
    if not manifest_path.exists():
        return False, [f"missing manifest at {manifest_path}"]

    manifest = json.loads(manifest_path.read_text())
    errors: list[str] = []
    for entry in manifest["files"]:
        p = archive_root / entry["path"]
        if not p.exists():
            errors.append(f"missing file: {entry['path']}")
            continue
        if p.stat().st_size != entry["bytes"]:
            errors.append(
                f"size mismatch: {entry['path']} ({p.stat().st_size} vs {entry['bytes']})"
            )
        digest = _sha256(p)
        if digest != entry["sha256"]:
            errors.append(f"sha256 mismatch: {entry['path']}")
    return (not errors), errors


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
