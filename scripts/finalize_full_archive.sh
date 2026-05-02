#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/finalize_full_archive.sh [options]

Verifies a generated full archive, summarizes MANIFEST.json, writes a
timestamped evidence file, and optionally promotes the candidate archive
to the canonical `instances/` path.

Options:
  --archive PATH          Candidate archive directory (default: instances_full_candidate)
  --target PATH           Canonical archive directory for promotion (default: instances)
  --expected-files N      Expected manifest file count (default: 9000)
  --log-dir PATH          Directory for summary evidence (default: logs)
  --promote               Move current target aside and promote archive to target
  --dry-run               Print commands without executing them
  -h, --help              Show this help

Typical remote-machine flow:
  scripts/finalize_full_archive.sh --archive instances_full_candidate
  scripts/finalize_full_archive.sh --archive instances_full_candidate --promote
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

archive="instances_full_candidate"
target="instances"
expected_files="9000"
log_dir="logs"
promote=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive)
            archive="$2"
            shift 2
            ;;
        --target)
            target="$2"
            shift 2
            ;;
        --expected-files)
            expected_files="$2"
            shift 2
            ;;
        --log-dir)
            log_dir="$2"
            shift 2
            ;;
        --promote)
            promote=1
            shift
            ;;
        --dry-run)
            dry_run=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

cd "${repo_root}"

timestamp="$(date -u +%Y%m%dT%H%M%SZ)"
summary_path="${log_dir}/full_archive_summary_${timestamp}.json"
backup_path="${target}_backup_${timestamp}"

print_step() {
    printf '+ %s\n' "$*"
}

run_cmd() {
    print_step "$*"
    if [[ "${dry_run}" -eq 1 ]]; then
        return 0
    fi
    "$@"
}

if [[ "${dry_run}" -eq 1 ]]; then
    echo "DRY RUN: commands will not be executed."
fi
echo "archive: ${archive}"
echo "target: ${target}"
echo "expected files: ${expected_files}"
echo "summary path: ${summary_path}"

if [[ "${dry_run}" -eq 0 ]]; then
    if [[ ! -d "${archive}" ]]; then
        echo "Archive directory does not exist: ${archive}" >&2
        exit 1
    fi
    mkdir -p "${log_dir}"
fi

run_cmd uv run python scripts/verify_instances.py --archive "${archive}"

print_step python summarize_manifest
if [[ "${dry_run}" -eq 0 ]]; then
    ARCHIVE="${archive}" EXPECTED_FILES="${expected_files}" SUMMARY_PATH="${summary_path}" python - <<'PY'
import json
import os
from pathlib import Path

archive = Path(os.environ["ARCHIVE"])
expected_files = int(os.environ["EXPECTED_FILES"])
summary_path = Path(os.environ["SUMMARY_PATH"])
manifest_path = archive / "MANIFEST.json"
manifest = json.loads(manifest_path.read_text())
files = manifest["files"]

summary = {
    "archive": str(archive),
    "manifest": str(manifest_path),
    "file_count": len(files),
    "expected_files": expected_files,
    "total_bytes": sum(entry["bytes"] for entry in files),
    "N": sorted({entry["cell"]["N"] for entry in files}),
    "M": sorted({entry["cell"]["M"] for entry in files}),
    "correlation": sorted({entry["cell"]["correlation"] for entry in files}),
    "f": sorted({entry["cell"]["f"] for entry in files}),
    "subset_counts": {
        subset: sum(1 for entry in files if entry["subset"] == subset)
        for subset in sorted({entry["subset"] for entry in files})
    },
}

print(json.dumps(summary, indent=2))
summary_path.write_text(json.dumps(summary, indent=2) + "\n")

if summary["file_count"] != expected_files:
    raise SystemExit(
        f"expected {expected_files} files, found {summary['file_count']}; "
        f"summary written to {summary_path}"
    )
PY
fi

if [[ "${promote}" -eq 1 ]]; then
    if [[ "${archive}" = "${target}" ]]; then
        echo "--archive and --target are the same; nothing to promote." >&2
        exit 1
    fi

    if [[ "${dry_run}" -eq 0 && -e "${backup_path}" ]]; then
        echo "Backup path already exists: ${backup_path}" >&2
        exit 1
    fi

    if [[ -e "${target}" || "${dry_run}" -eq 1 ]]; then
        run_cmd mv "${target}" "${backup_path}"
    fi
    run_cmd mv "${archive}" "${target}"
    echo "Promoted ${archive} -> ${target}"
    echo "Previous ${target} saved as ${backup_path}"
else
    echo "Promotion skipped. Re-run with --promote after reviewing ${summary_path}."
fi
