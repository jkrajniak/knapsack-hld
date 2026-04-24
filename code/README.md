# `code/` — Python implementation

All algorithmic and tooling code lives here. Sub-packages:

| Sub-package    | Purpose                                                                  |
| -------------- | ------------------------------------------------------------------------ |
| `instances/`   | Pure-Python MCKP instance generator and the `InstanceModel` schema       |
| `solvers/`     | Unified wrapper for HiGHS, SCIP, COIN-OR CBC, and Pisinger `mcknap`      |
| `heuristics/`  | Greedy-MaxRatio, BISSA, TRS-2008 (Tsesmetzis et al. 2008), Partition-Optimal |
| `hld/`         | The Hybrid Lagrangian-Decomposition algorithm (Phase 1–3 + instrumentation) |
| `tuning/`      | SMAC3 tuning campaign and incumbent loader                               |
| `utils/`       | Metrics, IO, structured logging, parallel primitives                     |

Each sub-package exposes a small public API in its `__init__.py`.

The package is intentionally framework-light: no class hierarchies unless
state encapsulation genuinely helps; functions take and return plain
dataclasses or `pydantic` models.

## Import convention

`code/` is on `pythonpath` (configured in `pyproject.toml`), so its
sub-packages are imported as if they were top-level packages — for
example `from instances import generate_instance`, not
`from code.instances import generate_instance`. The `code/` name is
reserved for the directory layout only (it shadows the Python stdlib
`code` module if used as a package name).

For ad-hoc scripts outside `pytest`, prepend `code/` to `PYTHONPATH`:

```bash
PYTHONPATH=code python -c "from instances import generate_instance"
```

`scripts/` entry points (added in Phase B) handle this automatically.
