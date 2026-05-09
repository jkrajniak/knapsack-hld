"""Checked-in HLD calibration config stays machine-readable."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_hld_smac_best_config_has_expected_parameters() -> None:
    config = json.loads((ROOT / "configs" / "hld_smac_best.json").read_text())

    assert config == {
        "source_run": "results/smac_run/full_20260508T192425Z",
        "n_iter": 35,
        "alpha": 0.998,
        "k": 58,
        "lambda_max": 80.745,
    }
