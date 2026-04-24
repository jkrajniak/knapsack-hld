# `results/` — Raw experimental output

Every run writes one row per `(instance_id, algorithm, seed)` triple to a
gzipped CSV file (`results/<experiment>/<date>.csv.gz`). Schema is
documented in `code/utils/results_schema.py`.

Raw CSVs are checked into the repository so that figures and tables can be
regenerated *without* re-running the experiments. The script
`make_figures.py` and `make_tables.py` consume these files only.

Total uncompressed size is large; gzipped CSV keeps the diff hostile-to-noise
while staying under typical Git LFS thresholds.
