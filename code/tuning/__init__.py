"""SMAC3-based parameter tuning for HLD (parameter-tuning spec §4.1).

Modules
-------
- ``smac_run``  — CLI entrypoint + AlgorithmConfigurationFacade wiring.
- ``bootstrap`` — bootstrap 95 % CI for incumbent evaluation.

The harness exposes HLD's parameter space ``(N_iter, alpha, K, lambda_max)``
as a SMAC target. Every loaded instance is asserted to belong to the
**tuning** subset of the archive manifest, so the parameter campaign can
never leak signal from the held-out test partition.
"""

from tuning.bootstrap import bootstrap_mean_ci

__all__ = ["bootstrap_mean_ci"]
