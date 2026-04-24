# `tuning/` — SMAC3 parameter-tuning artefacts

This directory holds the **inputs** and **outputs** of the SMAC3 campaign
that selects the HLD parameters $(\lambda_{\max}, N_{\text{iter}}, \alpha, K)$
on a *disjoint* tuning slice of the benchmark archive.

| File                           | Purpose                                                    |
| ------------------------------ | ---------------------------------------------------------- |
| `config_space.json`            | SMAC3 ConfigSpace definition (parameter ranges + types)    |
| `scenario.yaml`                | SMAC3 scenario file (budget, seeds, instance file)         |
| `incumbent.json`               | Best configuration found, used by `run_hld.py` by default  |
| `runhistory.json` *(gitignored)* | Full SMAC3 trace; reproducible from `scenario.yaml`     |

The tuning set never overlaps with the test set; this is enforced by
`scripts/run_tuning.py`, which refuses to start if any tuning instance ID
appears in `instances/test/MANIFEST.json`.
