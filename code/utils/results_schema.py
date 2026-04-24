"""Canonical schema for one experimental row.

Every benchmark script writes one row per `(instance_id, solver, seed)`
to a gzipped CSV. The column order is fixed in `RUN_RESULT_FIELDS` so
the figure/table scripts can rely on positional access.

The schema is intentionally narrow: derived quantities (optimality gap,
speed-up, normalised time) are computed downstream from this raw output.
This keeps the raw record reproducible and audit-friendly.
"""

from __future__ import annotations

import csv
import gzip
import json
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RUN_RESULT_FIELDS: tuple[str, ...] = (
    "experiment",
    "instance_id",
    "N",
    "M",
    "correlation",
    "f",
    "instance_seed",
    "solver",
    "solver_seed",
    "wall_time_s",
    "profit",
    "total_cost",
    "n_classes_selected",
    "status",
    "time_limit_s",
    "reference_solver",
    "reference_profit",
    "optimality_gap_pct",
    "host",
    "timestamp_utc",
    "solver_metadata_json",
)


@dataclass(frozen=True)
class RunResult:
    """One row in the canonical results CSV."""

    experiment: str
    instance_id: str
    N: int
    M: int
    correlation: str
    f: float
    instance_seed: int
    solver: str
    solver_seed: int | None
    wall_time_s: float
    profit: int
    total_cost: int
    n_classes_selected: int
    status: str
    time_limit_s: float | None
    reference_solver: str | None = None
    reference_profit: int | None = None
    optimality_gap_pct: float | None = None
    host: str = field(default_factory=lambda: socket.gethostname())
    timestamp_utc: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    solver_metadata_json: str = "{}"

    def as_row(self) -> dict[str, Any]:
        d = asdict(self)
        return {k: ("" if v is None else v) for k, v in d.items()}


def write_results_csv_gz(path: str | Path, rows: list[RunResult]) -> Path:
    """Write `rows` to a gzipped CSV with the canonical header."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(RUN_RESULT_FIELDS))
        writer.writeheader()
        for r in rows:
            writer.writerow(r.as_row())
    return path


def read_results_csv_gz(path: str | Path) -> list[dict[str, str]]:
    """Read a results CSV back as raw string-valued dicts (no coercion)."""
    path = Path(path)
    with gzip.open(path, "rt", newline="") as fh:
        return list(csv.DictReader(fh))


def host_metadata() -> dict[str, str]:
    """Capture host/CPU/Python info; pair with the CSV for reproducibility."""
    return {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
        "python": sys.version.split()[0],
        "captured_utc": datetime.now(UTC).isoformat(),
    }


def write_host_metadata(path: str | Path) -> Path:
    """Pretty-print `host_metadata()` next to a results CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(host_metadata(), indent=2, sort_keys=True))
    return path
