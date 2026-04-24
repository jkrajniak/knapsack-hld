# `scripts/` — End-user CLI entry points

These scripts are the **public** reproduction surface; everything else under
`code/` is library code that they call.

| Script                       | Purpose                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `generate_instances.py`      | Build the full benchmark archive from `configs/instances.yaml` |
| `verify_instances.py`        | Re-hash the archive and compare with `MANIFEST.json`           |
| `run_baselines.py`           | Run all baselines (HiGHS/SCIP/CBC/mcknap/heuristics) on the test set |
| `run_hld.py`                 | Run the HLD algorithm on the test set                          |
| `run_tuning.py`              | Launch the SMAC3 parameter-tuning campaign on the tuning set   |
| `make_figures.py`            | Build every figure that appears in the paper                   |
| `make_tables.py`             | Build every table that appears in the paper                    |

The top-level `Makefile` (added in a later phase) wires these together as
`make reproduce`, `make reproduce-quick`, and `make tables figures`.
