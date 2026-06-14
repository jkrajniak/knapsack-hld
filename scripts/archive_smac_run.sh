#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/archive_smac_run.sh --run-dir PATH [options]

Packages a completed SMAC run directory and its log into a private artifact
directory outside the git repository, then writes a SHA-256 checksum.

Options:
  --run-dir PATH        Completed SMAC output directory, e.g. results/smac_run/full_YYYYMMDDTHHMMSSZ
  --log-file PATH       Matching log file (default: logs/smac_full_<run-id>.log)
  --artifact-dir PATH   Destination directory (default: ${KNAPSACK_ARTIFACTS_DIR:-../artifacts}/smac)
  --dry-run             Print commands without executing them
  -h, --help            Show this help

Example:
  scripts/archive_smac_run.sh --run-dir results/smac_run/full_20260508T192425Z
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

run_dir=""
log_file=""
artifact_dir="${KNAPSACK_ARTIFACTS_DIR:-../artifacts}/smac"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-dir)
            run_dir="$2"
            shift 2
            ;;
        --log-file)
            log_file="$2"
            shift 2
            ;;
        --artifact-dir)
            artifact_dir="$2"
            shift 2
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

if [[ -z "${run_dir}" ]]; then
    echo "--run-dir is required." >&2
    usage >&2
    exit 2
fi

run_id="$(basename "${run_dir}")"
if [[ "${run_id}" == full_* ]]; then
    run_id="${run_id#full_}"
fi

if [[ -z "${log_file}" ]]; then
    log_file="logs/smac_full_${run_id}.log"
fi

archive_path="${artifact_dir}/smac_full_${run_id}.tar.gz"
checksum_path="${archive_path}.sha256"

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

echo "run_dir: ${run_dir}"
echo "log_file: ${log_file}"
echo "artifact_dir: ${artifact_dir}"
echo "archive_path: ${archive_path}"

if [[ "${dry_run}" -eq 0 ]]; then
    if [[ ! -d "${run_dir}" ]]; then
        echo "Run directory does not exist: ${run_dir}" >&2
        exit 1
    fi
    if [[ ! -f "${log_file}" ]]; then
        echo "Log file does not exist: ${log_file}" >&2
        exit 1
    fi
    for required in incumbent.json evaluations.csv reference_profits.json; do
        if [[ ! -f "${run_dir}/${required}" ]]; then
            echo "Missing expected SMAC output: ${run_dir}/${required}" >&2
            exit 1
        fi
    done
    mkdir -p "${artifact_dir}"
fi

run_cmd tar -czf "${archive_path}" "${run_dir}" "${log_file}"
run_cmd shasum -a 256 "${archive_path}"

if [[ "${dry_run}" -eq 0 ]]; then
    shasum -a 256 "${archive_path}" > "${checksum_path}"
    ls -lh "${archive_path}" "${checksum_path}"
fi
