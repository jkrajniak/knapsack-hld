# `tests/` — Unit and integration tests

Run via:

```bash
uv run pytest -q
```

Test layout (filled in across phases):

| File                              | Phase | Coverage                                          |
| --------------------------------- | ----- | ------------------------------------------------- |
| `test_smoke.py`                   | A     | Imports + Python version sanity check             |
| `test_generator_determinism.py`   | B     | `(N, M, correlation, f, seed)` round-trips bit-exactly |
| `test_solver_interface.py`        | C     | All solver wrappers respect the unified contract |
| `test_hld_correctness.py`         | C     | HLD optimum ≤ global IP optimum                   |
| `test_pisinger_mcknap.py`         | C     | Python `mcknap` ≡ C reference on Pisinger archive |
| `test_smac_setup.py`              | D     | ConfigSpace and scenario load without error       |

Performance regressions are measured separately under `tests/bench/` (not
run on every push) once Phase B lands.
