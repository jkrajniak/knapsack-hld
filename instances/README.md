# `instances/` — Benchmark instance archive

Generated synthetic Selective-MCKP instances live here. Layout:

```
instances/
├── MANIFEST.json          # SHA-256 of every instance file (verified in CI)
├── schema.json            # JSON-Schema for the InstanceModel
├── tuning/                # Disjoint slice used only for SMAC3 tuning
├── test/                  # Held-out evaluation slice
└── pisinger/              # Mirror of Pisinger 1995 archive (small/medium sanity checks)
```

Generation grid (Phase B):

| Axis           | Values                                                              |
| -------------- | ------------------------------------------------------------------- |
| `N` (classes)  | 1 000, 10 000, 100 000                                              |
| `M` (items/cls)| 5, 10, 20                                                           |
| Correlation    | uncorrelated, weakly, strongly, inversely strongly                  |
| `f` (tightness)| 0.1, 0.25, 0.5, 0.75, 0.9 (with `B = f · N · c̄`)                  |
| Seeds          | ≥ 50 per cell                                                       |

All instances are **deterministic** in `(N, M, correlation, f, seed)` and
verified bit-exactly by `scripts/verify_instances.py`.

Large instance archives may be hosted via Git LFS or as a Zenodo release
asset; see the project README and `MANIFEST.json` for the canonical location
of the version used in the paper.
