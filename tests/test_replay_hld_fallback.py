"""End-to-end smoke test for `scripts/replay_hld_fallback.py` (Task 3.4.3)."""

from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest
from instances.generator import generate_instance
from instances.io import save_instance
from instances.schema import CorrelationKind

ROOT = Path(__file__).resolve().parents[1]
REPLAY_SCRIPT = ROOT / "scripts" / "replay_hld_fallback.py"
SUMMARISER_SCRIPT = ROOT / "scripts" / "summarize_fallback_stats.py"

FIELDNAMES = {
    "instance_id",
    "subset",
    "N",
    "M",
    "correlation",
    "f",
    "seed",
    "solver",
    "class_ordering",
    "n_iter",
    "alpha",
    "k",
    "lambda_max",
    "lambda_est",
    "fallback_equal_split",
    "phase1_wall_s",
    "phase2_wall_s",
}


def _build_archive(tmp_path: Path, n_instances: int = 4) -> tuple[Path, Path]:
    """Generate `n_instances` tiny instances + a manifest matching the runner schema."""
    archive = tmp_path / "instances"
    archive.mkdir()
    rel_dir = Path("uncorrelated") / "N8_M3"
    (archive / rel_dir).mkdir(parents=True)
    files = []
    for seed in range(n_instances):
        inst = generate_instance(
            N=8, M=3, correlation=CorrelationKind.UNCORRELATED, f=0.5, seed=seed
        )
        name = f"mckp_N8_M3_uncorrelated_f0.500_seed{seed}.json"
        save_instance(inst, archive / rel_dir / name)
        files.append(
            {
                "path": str(rel_dir / name),
                "subset": "test",
                "seed": seed,
                "cell": {
                    "N": 8,
                    "M": 3,
                    "correlation": "uncorrelated",
                    "f": 0.5,
                },
            }
        )
    manifest = archive / "MANIFEST.json"
    manifest.write_text(json.dumps({"files": files}, indent=2))
    return archive, manifest


def _write_config(tmp_path: Path) -> Path:
    """SMAC-style HLD config with K large enough to encourage Phase-2 fallbacks on a tiny instance."""
    payload = {"n_iter": 35, "alpha": 0.998, "k": 58, "lambda_max": 80.745}
    cfg = tmp_path / "hld.json"
    cfg.write_text(json.dumps(payload))
    return cfg


def _run_replay(*, archive: Path, manifest: Path, config: Path, out_csv: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [
            sys.executable,
            str(REPLAY_SCRIPT),
            "--archive",
            str(archive),
            "--manifest",
            str(manifest),
            "--subset",
            "test",
            "--config",
            str(config),
            "--out-csv",
            str(out_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )


def test_replay_emits_one_row_per_instance(tmp_path: Path) -> None:
    archive, manifest = _build_archive(tmp_path, n_instances=4)
    config = _write_config(tmp_path)
    out_csv = tmp_path / "fallback_pinned.csv"

    proc = _run_replay(archive=archive, manifest=manifest, config=config, out_csv=out_csv)
    assert "n_instances: 4" in proc.stdout, proc.stdout
    rows = list(csv.DictReader(out_csv.open(newline="")))
    assert len(rows) == 4
    assert set(rows[0].keys()) == FIELDNAMES
    for row in rows:
        assert row["solver"] == "hld"
        assert row["class_ordering"] == "sequential"
        assert row["fallback_equal_split"] in {"0", "1"}
        assert int(row["n_iter"]) == 35
        assert float(row["lambda_max"]) == pytest.approx(80.745)


def test_replay_output_feeds_existing_summariser(tmp_path: Path) -> None:
    archive, manifest = _build_archive(tmp_path, n_instances=4)
    config = _write_config(tmp_path)
    out_csv = tmp_path / "fallback_pinned.csv"
    _run_replay(archive=archive, manifest=manifest, config=config, out_csv=out_csv)

    out_dir = tmp_path / "summary"
    subprocess.run(
        [
            sys.executable,
            str(SUMMARISER_SCRIPT),
            "--results-csv",
            str(out_csv),
            "--out-dir",
            str(out_dir),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    overall = json.loads((out_dir / "fallback_overall.json").read_text())
    assert overall["n_rows"] == 4
    assert "8" in overall["by_N"]
    assert overall["by_N"]["8"]["n_rows"] == 4


def test_replay_respects_max_instances_smoke_cap(tmp_path: Path) -> None:
    archive, manifest = _build_archive(tmp_path, n_instances=4)
    config = _write_config(tmp_path)
    out_csv = tmp_path / "fallback_pinned.csv"
    proc = subprocess.run(
        [
            sys.executable,
            str(REPLAY_SCRIPT),
            "--archive",
            str(archive),
            "--manifest",
            str(manifest),
            "--subset",
            "test",
            "--config",
            str(config),
            "--max-instances",
            "2",
            "--out-csv",
            str(out_csv),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert "n_instances: 2" in proc.stdout
    rows = list(csv.DictReader(out_csv.open(newline="")))
    assert len(rows) == 2
