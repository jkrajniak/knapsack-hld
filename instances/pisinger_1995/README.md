# Pisinger 1995 `mcknap` instance archive

Provenance for the classic MCKP benchmark archive used in §3.8 of the
manuscript as a correctness/optimality sanity check.

## Citation

> Pisinger, D. (1995). *A minimal algorithm for the multiple-choice
> knapsack problem*. European Journal of Operational Research,
> 83(2): 394–410. https://doi.org/10.1016/0377-2217(95)00015-I

## Source

> http://hjemmesider.diku.dk/~pisinger/codes.html

**IMPORTANT (correction filed 2026-06-05 in `FINDING_2026_06_05.md`):**
the upstream Pisinger codes page does **not** host a separate test-
instance archive for MCKP. Only the solver source `mcknap.c` is
distributed; it embeds the instance generator inline (`srand48`-seeded
RNG, CLI `mcknap k n r type`). All "Pisinger 1995 instances" in the
literature are regenerated from this code at the parameters published
in §6 of the 1995 paper. The earlier wording of this README that
described a `test_mcknap.tgz` archive was speculative and does not
match upstream reality; the historical retrieval attempts are
documented in `FINDING_2026_06_05.md`.

## Acquisition (current procedure)

```bash
# 1. Download mcknap.c (solver + generator in one file).
curl -fsSLO http://hjemmesider.diku.dk/~pisinger/mcknap.c

# 2. Verify SHA-256 against CHECKSUMS.txt.
shasum -a 256 mcknap.c
# Expect: 60c6647341f4794cced2278a87d587df4f25a1793ca0114a1d3f454129961e75
```

Local `mcknap.c` is already present in this directory (checked in
2026-06-05). What happens next depends on the PI's decision in
`FINDING_2026_06_05.md` (P-port to Python / C-compile / D-descope R.7).

## Loader

**Not yet implemented.** The previously-planned
`instances/pisinger_loader.py` assumed `.in` files extracted from an
upstream archive. Since the archive does not exist (see above), the
loader will instead be one of:

- a Python port of the `mcknap.c` generator emitting `InstanceModel`
  records directly (option P-port in `FINDING_2026_06_05.md`), or
- a thin wrapper around a compiled `mcknap` binary that captures its
  printed instances (option C-compile).

The Pisinger MCKP form requires every class to contribute exactly one
item. Whichever loader path lands, we feed these instances to BISSA
via the explicit MCKP→Selective-MCKP transformation documented in
`code/solvers/bissa/` (adds a dummy zero-cost item per class).

## CHECKSUMS

See `CHECKSUMS.txt`. Currently records only `mcknap.c` retrieved
2026-06-05; new entries land when the P-port or C-compile path lands.
