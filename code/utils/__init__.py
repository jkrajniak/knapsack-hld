"""Cross-cutting helpers: result schemas, IO utilities, run metadata.

Public API:

- `RUN_RESULT_FIELDS`     — canonical CSV column order
- `RunResult`             — dataclass for one (instance, solver, seed) row
- `write_results_csv_gz`  — append/write the canonical gzipped CSV
- `read_results_csv_gz`   — load a results CSV back as a list of dicts
- `host_metadata`         — capture host/CPU/Python info for reproducibility
"""

from utils.results_schema import (
    RUN_RESULT_FIELDS,
    RunResult,
    host_metadata,
    read_results_csv_gz,
    write_results_csv_gz,
)

__all__ = [
    "RUN_RESULT_FIELDS",
    "RunResult",
    "host_metadata",
    "read_results_csv_gz",
    "write_results_csv_gz",
]
