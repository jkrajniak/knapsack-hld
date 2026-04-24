"""Smoke tests verifying the dev environment is wired up correctly.

These tests intentionally do nothing useful beyond importing the third-party
solvers and helpers we depend on. They are replaced by real unit tests in
Phase B (benchmark suite) and Phase C (baselines).
"""

from __future__ import annotations


def test_python_version_supported() -> None:
    import sys

    assert sys.version_info >= (3, 12)


def test_milp_solver_imports() -> None:
    import highspy
    import pulp
    import pyscipopt

    assert highspy is not None
    assert pulp is not None
    assert pyscipopt is not None


def test_scientific_stack_imports() -> None:
    import joblib
    import matplotlib
    import numpy
    import pandas
    import pydantic
    import yaml

    assert all(m is not None for m in (joblib, matplotlib, numpy, pandas, pydantic, yaml))


def test_smac_imports() -> None:
    import smac

    assert smac is not None
