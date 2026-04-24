# Pisinger 1995 `mcknap` instance archive

Provenance for the classic MCKP benchmark archive used in §3.8 of the
manuscript as a correctness/optimality sanity check.

## Citation

> Pisinger, D. (1995). *A minimal algorithm for the multiple-choice
> knapsack problem*. European Journal of Operational Research,
> 83(2): 394–410. https://doi.org/10.1016/0377-2217(95)00015-I

## Source

Originally distributed from David Pisinger's web page:

> http://hjemmesider.diku.dk/~pisinger/codes.html

The archive (`test_mcknap.tgz`) bundles `.in` files grouped by
correlation type (`uncorr`, `weakcorr`, `strongcorr`, `inverse`) and
size (`n10`, `n50`, ...).

## Acquisition (manual step)

The archive is **not** redistributed in this repository. To populate
this directory:

```bash
# 1. Download the archive (URL may change; verify in the citation
#    above).
curl -fsSLO http://hjemmesider.diku.dk/~pisinger/test_mcknap.tgz

# 2. Verify the SHA-256 of the downloaded archive against the value
#    recorded in CHECKSUMS.txt (this file is updated when a new
#    upstream archive is observed).

# 3. Extract into this directory.
tar -xzf test_mcknap.tgz -C instances/pisinger_1995/
```

After extraction, files live under
`instances/pisinger_1995/<correlation>/<size>/*.in`.

## Loader

```python
from pathlib import Path
from instances.pisinger_loader import load_pisinger_file

inst = load_pisinger_file("instances/pisinger_1995/uncorr/n10/inst1.in")
```

The loader emits the same `InstanceModel` schema as the synthetic
generator, so the rest of the pipeline (solvers, manifest,
result-collection) treats Pisinger instances uniformly.

The Pisinger MCKP form requires every class to contribute exactly one
item. We feed these instances to BISSA via the explicit
MCKP→Selective-MCKP transformation documented in `code/solvers/bissa/`
(adds a dummy zero-cost item per class).

## CHECKSUMS

Recorded the day the archive is first installed locally. Update the
file alongside any re-download. As of this writing the upstream archive
has not been added to the repository workflow; see the project README
for the current status.

```
# format: <sha256>  <relative path inside extracted tarball>
```
