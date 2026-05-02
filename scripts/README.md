# `scripts/` — End-user CLI entry points

These scripts are the **public** reproduction surface; everything else under
`code/` is library code that they call.

| Script                       | Purpose                                                       |
| ---------------------------- | ------------------------------------------------------------- |
| `generate_instances.py`      | Build the full benchmark archive from `configs/instances.yaml` |
| `verify_instances.py`        | Re-hash the archive and compare with `MANIFEST.json`           |
| `run_full_archive.sh`        | Big-machine wrapper: sync deps, generate the full archive, verify it, and log the run |
| `finalize_full_archive.sh`   | Verify, summarize, and optionally promote a completed full archive |
| `run_baselines.py`           | Run all baselines (HiGHS/SCIP/CBC/mcknap/heuristics) on the test set |
| `run_hld.py`                 | Run the HLD algorithm on the test set                          |
| `run_tuning.py`              | Launch the SMAC3 parameter-tuning campaign on the tuning set   |
| `make_figures.py`            | Build every figure that appears in the paper                   |
| `make_tables.py`             | Build every table that appears in the paper                    |

The top-level `Makefile` (added in a later phase) wires these together as
`make reproduce`, `make reproduce-quick`, and `make tables figures`.

## Full archive on a large machine

Use the wrapper rather than invoking the generator by hand. It writes a
timestamped log under `logs/`, keeps the candidate archive separate from the
small committed smoke archive, and verifies `MANIFEST.json` after generation.

```bash
scripts/run_full_archive.sh --out instances_full_candidate --jobs 16
```

Check the command sequence without generating files:

```bash
scripts/run_full_archive.sh --dry-run --out instances_full_candidate --jobs 16
```

After generation finishes, verify and summarize the candidate archive:

```bash
scripts/finalize_full_archive.sh --archive instances_full_candidate
```

If the summary reports 9,000 files and the manifest verification passes,
promote the candidate archive to the canonical `instances/` path:

```bash
scripts/finalize_full_archive.sh --archive instances_full_candidate --promote
```
