"""Read/write Selective-MCKP instances as JSON (plain or gzipped).

Filename convention:

    mckp_N{N}_M{M}_{correlation}_f{f:0.3f}_seed{seed}.json[.gz]

This deterministic name is the canonical instance ID used by the manifest
and the tuning/test split.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

from instances.schema import CorrelationKind, InstanceModel


def instance_id(
    *,
    N: int,
    M: int,
    correlation: CorrelationKind | str,
    f: float,
    seed: int,
) -> str:
    """Canonical filename stem (no extension)."""
    correlation = CorrelationKind(correlation)
    return f"mckp_N{N}_M{M}_{correlation.value}_f{f:0.3f}_seed{seed}"


def save_instance(instance: InstanceModel, path: str | Path) -> Path:
    """Serialise `instance` to `path`. Compress if path ends with `.gz`."""
    path = Path(path)
    payload = instance.model_dump(mode="json")
    encoded = json.dumps(payload, separators=(",", ":")).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".gz":
        with gzip.open(path, "wb") as fh:
            fh.write(encoded)
    else:
        path.write_bytes(encoded)
    return path


def load_instance(path: str | Path) -> InstanceModel:
    """Load and validate an instance from `path`."""
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as fh:
            raw = fh.read()
    else:
        raw = path.read_bytes()
    return InstanceModel.model_validate_json(raw)
