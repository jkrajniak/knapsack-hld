#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_full_archive.sh [options]

Options:
  --config PATH     Archive grid YAML (default: scripts/configs/archive_full.yaml)
  --out PATH        Output archive directory (default: instances_full_candidate)
  --jobs N          Parallel workers (default: HLD_ARCHIVE_JOBS or detected CPU count)
  --log-dir PATH    Directory for run logs (default: logs)
  --force           Overwrite existing instance files
  --skip-sync       Do not run uv sync before generation
  --no-verify       Skip manifest verification after generation
  --dry-run         Print commands without executing them
  -h, --help        Show this help

Recommended big-machine run:
  scripts/run_full_archive.sh --out instances_full_candidate --jobs 16
EOF
}

detect_jobs() {
    if command -v nproc >/dev/null 2>&1; then
        nproc
        return
    fi
    if command -v sysctl >/dev/null 2>&1; then
        sysctl -n hw.ncpu
        return
    fi
    echo 4
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

config="scripts/configs/archive_full.yaml"
out_dir="instances_full_candidate"
jobs="${HLD_ARCHIVE_JOBS:-$(detect_jobs)}"
log_dir="logs"
force=0
skip_sync=0
verify=1
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            config="$2"
            shift 2
            ;;
        --out)
            out_dir="$2"
            shift 2
            ;;
        --jobs)
            jobs="$2"
            shift 2
            ;;
        --log-dir)
            log_dir="$2"
            shift 2
            ;;
        --force)
            force=1
            shift
            ;;
        --skip-sync)
            skip_sync=1
            shift
            ;;
        --no-verify)
            verify=0
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
log_file="${log_dir}/full_archive_${timestamp}.log"

print_step() {
    printf '+ %s\n' "$*"
}

run_cmd() {
    print_step "$*"
    if [[ "${dry_run}" -eq 1 ]]; then
        return 0
    fi
    "$@" 2>&1 | tee -a "${log_file}"
    return "${PIPESTATUS[0]}"
}

if [[ "${dry_run}" -eq 1 ]]; then
    echo "DRY RUN: commands will not be executed."
    echo "Log file would be: ${log_file}"
else
    mkdir -p "${log_dir}"
    {
        echo "Full archive generation"
        echo "started_utc=${timestamp}"
        echo "config=${config}"
        echo "out=${out_dir}"
        echo "jobs=${jobs}"
    } | tee "${log_file}"
fi

if [[ "${skip_sync}" -eq 0 ]]; then
    run_cmd uv sync
fi

generate_cmd=(
    uv run python scripts/generate_instances.py
    --config "${config}"
    --out "${out_dir}"
    --jobs "${jobs}"
)

if [[ "${force}" -eq 1 ]]; then
    generate_cmd+=(--force)
fi

run_cmd "${generate_cmd[@]}"

if [[ "${verify}" -eq 1 ]]; then
    run_cmd uv run python scripts/verify_instances.py --archive "${out_dir}"
fi

if [[ "${dry_run}" -eq 0 ]]; then
    echo "Finished. Log: ${log_file}" | tee -a "${log_file}"
fi
