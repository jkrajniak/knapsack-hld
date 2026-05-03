#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/run_smac_full.sh [options]

Verifies the promoted full archive and launches the full SMAC3 tuning
campaign in a fresh timestamped output directory.

Options:
  --archive PATH             Archive root (default: instances)
  --out-root PATH            Parent output directory (default: results/smac_run)
  --out-dir PATH             Exact output directory (default: <out-root>/full_<timestamp>)
  --expected-files N         Expected manifest file count (default: 9000)
  --budget N                 SMAC trial budget (default: 5000)
  --max-N N                  Exclude tuning instances with larger N (default: 10000)
  --jobs N                   Parallel workers for references/final CI (default: 8)
  --seed N                   SMAC seed (default: 7)
  --ref-time-limit-s SECONDS HiGHS reference cap per instance (default: 60)
  --eval-time-limit-s SEC    HLD evaluation cap per trial (default: 60)
  --log-dir PATH             Directory for logs (default: logs)
  --skip-sync                Do not run uv sync before the campaign
  --dry-run                  Print commands without executing them
  -h, --help                 Show this help

Typical remote-machine run:
  scripts/run_smac_full.sh --archive instances
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

archive="instances"
out_root="results/smac_run"
out_dir=""
expected_files="9000"
budget="5000"
max_n="10000"
jobs="8"
seed="7"
ref_time_limit_s="60"
eval_time_limit_s="60"
log_dir="logs"
skip_sync=0
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --archive)
            archive="$2"
            shift 2
            ;;
        --out-root)
            out_root="$2"
            shift 2
            ;;
        --out-dir)
            out_dir="$2"
            shift 2
            ;;
        --expected-files)
            expected_files="$2"
            shift 2
            ;;
        --budget)
            budget="$2"
            shift 2
            ;;
        --max-N)
            max_n="$2"
            shift 2
            ;;
        --jobs)
            jobs="$2"
            shift 2
            ;;
        --seed)
            seed="$2"
            shift 2
            ;;
        --ref-time-limit-s)
            ref_time_limit_s="$2"
            shift 2
            ;;
        --eval-time-limit-s)
            eval_time_limit_s="$2"
            shift 2
            ;;
        --log-dir)
            log_dir="$2"
            shift 2
            ;;
        --skip-sync)
            skip_sync=1
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
if [[ -z "${out_dir}" ]]; then
    out_dir="${out_root}/full_${timestamp}"
fi
log_file="${log_dir}/smac_full_${timestamp}.log"

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
    if [[ -e "${out_dir}" ]]; then
        echo "Output directory already exists: ${out_dir}" >&2
        echo "Choose --out-dir explicitly or remove the existing directory." >&2
        exit 1
    fi
    {
        echo "SMAC full campaign"
        echo "started_utc=${timestamp}"
        echo "archive=${archive}"
        echo "out_dir=${out_dir}"
        echo "budget=${budget}"
        echo "max_N=${max_n}"
        echo "jobs=${jobs}"
        echo "seed=${seed}"
    } | tee "${log_file}"
fi

if [[ "${skip_sync}" -eq 0 ]]; then
    run_cmd uv sync
fi

run_cmd scripts/finalize_full_archive.sh --archive "${archive}" --expected-files "${expected_files}"
run_cmd du -sh "${archive}"
run_cmd uv run python code/tuning/smac_run.py \
    --archive "${archive}" \
    --out-dir "${out_dir}" \
    --budget "${budget}" \
    --max-N "${max_n}" \
    --jobs "${jobs}" \
    --seed "${seed}" \
    --ref-time-limit-s "${ref_time_limit_s}" \
    --eval-time-limit-s "${eval_time_limit_s}"

if [[ "${dry_run}" -eq 0 ]]; then
    echo "Finished. Log: ${log_file}" | tee -a "${log_file}"
    echo "Full SMAC output: ${out_dir}" | tee -a "${log_file}"
fi
