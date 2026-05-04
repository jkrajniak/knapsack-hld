"""SMAC harness: parameter space, manifest filtering, target evaluation, CLI.

These tests exercise the SMAC-independent parts of ``tuning.smac_run``
(loading, evaluation, CLI parsing). The end-to-end SMAC campaign itself
is exercised separately by ``scripts``-level preview runs.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tuning.smac_run import (
    DEFAULT_BUDGET,
    DEFAULT_PREVIEW_BUDGET,
    PARAM_SPACE,
    HldConfig,
    _entry_passes_filters,
    _resolve_out_dir,
    build_configspace,
    evaluate_hld,
    load_tuning_archive,
    parse_args,
)

ARCHIVE_ROOT = Path(__file__).resolve().parent.parent / "instances"


def test_param_space_matches_spec() -> None:
    """N_iter in [5,50], alpha in [0,1], K in [4,64], lambda_max in [1,100] (spec 4.1.1)."""
    assert PARAM_SPACE["n_iter"]["low"] == 5
    assert PARAM_SPACE["n_iter"]["high"] == 50
    assert PARAM_SPACE["alpha"]["low"] == 0.0
    assert PARAM_SPACE["alpha"]["high"] == 1.0
    assert PARAM_SPACE["k"]["low"] == 4
    assert PARAM_SPACE["k"]["high"] == 64
    assert PARAM_SPACE["lambda_max"]["low"] == 1.0
    assert PARAM_SPACE["lambda_max"]["high"] == 100.0


def test_build_configspace_has_four_hyperparams() -> None:
    cs = build_configspace()
    names = sorted(cs.keys())
    assert names == ["alpha", "k", "lambda_max", "n_iter"]


def test_hld_config_from_mapping_round_trip() -> None:
    cfg = HldConfig.from_mapping({"n_iter": 12, "alpha": 0.7, "k": 5, "lambda_max": 7.5})
    assert cfg == HldConfig(n_iter=12, alpha=0.7, k=5, lambda_max=7.5)


def test_parse_args_defaults() -> None:
    ns = parse_args([])
    assert ns.budget == DEFAULT_BUDGET
    assert ns.preview is False
    assert ns.bootstrap_resamples == 1000
    assert ns.max_n is None
    assert ns.jobs == 1


def test_manifest_entry_filter_excludes_large_n() -> None:
    small = {"cell": {"N": 10000, "M": 10, "correlation": "weakly", "f": 0.5}}
    large = {"cell": {"N": 100000, "M": 10, "correlation": "weakly", "f": 0.5}}
    assert _entry_passes_filters(small, max_n=10000)
    assert not _entry_passes_filters(large, max_n=10000)


def test_resolve_out_dir_preview_swaps_budget_and_subdir() -> None:
    ns = parse_args(["--preview"])
    out, budget = _resolve_out_dir(ns)
    assert out.name == "preview"
    assert budget == DEFAULT_PREVIEW_BUDGET


def test_resolve_out_dir_explicit_budget_preserved_in_preview() -> None:
    ns = parse_args(["--preview", "--budget", "5"])
    out, budget = _resolve_out_dir(ns)
    assert out.name == "preview"
    assert budget == 5


def test_resolve_out_dir_full_run_uses_default_budget_dir() -> None:
    ns = parse_args([])
    out, budget = _resolve_out_dir(ns)
    assert out.name != "preview"
    assert budget == DEFAULT_BUDGET


@pytest.mark.skipif(
    not (ARCHIVE_ROOT / "MANIFEST.json").exists(),
    reason="instance archive not present (preview not generated)",
)
def test_load_tuning_archive_filters_to_tuning_subset() -> None:
    """Loader sees only `subset == 'tuning'` entries and asserts each one."""
    archive = load_tuning_archive(
        archive_root=ARCHIVE_ROOT,
        max_instances=1,
        max_n=1000,
        reference_cache=None,
    )
    assert len(archive.items) == 1
    for item in archive.items:
        assert item.ref_profit > 0
        assert item.inst.N > 0


@pytest.mark.skipif(
    not (ARCHIVE_ROOT / "MANIFEST.json").exists(),
    reason="instance archive not present (preview not generated)",
)
def test_load_tuning_archive_skips_non_positive_cached_references(tmp_path: Path) -> None:
    manifest = json.loads((ARCHIVE_ROOT / "MANIFEST.json").read_text())
    entries = sorted(
        [
            entry
            for entry in manifest["files"]
            if entry.get("subset") == "tuning" and _entry_passes_filters(entry, max_n=1000)
        ],
        key=lambda entry: entry["path"],
    )[:2]
    cache_path = tmp_path / "reference_profits.json"
    cache_path.write_text(
        json.dumps(
            {
                entries[0]["path"]: {"profit": 0, "time_s": 1.0},
                entries[1]["path"]: {"profit": 123, "time_s": 1.0},
            }
        )
    )

    archive = load_tuning_archive(
        archive_root=ARCHIVE_ROOT,
        max_instances=2,
        max_n=1000,
        reference_cache=cache_path,
    )

    assert archive.instance_ids == [entries[1]["path"]]
    assert archive.items[0].ref_profit == 123


@pytest.mark.skipif(
    not (ARCHIVE_ROOT / "MANIFEST.json").exists(),
    reason="instance archive not present (preview not generated)",
)
def test_evaluate_hld_returns_non_negative_gap() -> None:
    archive = load_tuning_archive(
        archive_root=ARCHIVE_ROOT,
        max_instances=1,
        max_n=1000,
        reference_cache=None,
    )
    cfg = HldConfig(n_iter=20, alpha=0.9, k=8, lambda_max=10.0)
    for item in archive.items:
        ev = evaluate_hld(item, cfg, seed=0)
        assert ev.optimality_gap >= 0.0
        assert ev.wall_time_s >= 0.0
        assert ev.profit <= ev.ref_profit
