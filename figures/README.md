# `figures/` — Paper figures

PDF figures used by the manuscript live here. Files are produced by
`scripts/make_figures.py` from the gzipped CSVs under `results/`.

This directory is git-ignored except for `.gitkeep`, because every figure
is regenerable from raw results. The camera-ready `figures/` will be
re-created by `make figures` against the tagged release, ensuring 1:1
correspondence with the published PDF.
