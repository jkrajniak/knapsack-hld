#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<'EOF'
Usage:
  scripts/archive_final_experiments.sh --run-id ID [options]

Packages completed final experiment outputs into a private artifact directory
outside the git repository, then writes a SHA-256 checksum.

Options:
  --run-id ID          Archive identifier, e.g. 20260510T104515Z
  --result-dir PATH    Final experiment directory (default: results/final_experiments)
  --artifact-dir PATH  Destination directory (default: ../knapsack-artifacts/final_experiments)
  --dry-run            Print commands without executing them
  -h, --help           Show this help

Example:
  scripts/archive_final_experiments.sh --run-id 20260510T104515Z
EOF
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

run_id=""
result_dir="results/final_experiments"
artifact_dir="../knapsack-artifacts/final_experiments"
dry_run=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-id)
            run_id="$2"
            shift 2
            ;;
        --result-dir)
            result_dir="$2"
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

if [[ -z "${run_id}" ]]; then
    echo "--run-id is required." >&2
    usage >&2
    exit 2
fi

archive_path="${artifact_dir}/final_experiments_${run_id}.tar.gz"
checksum_path="${archive_path}.sha256"

tar_inputs=("${result_dir}/results.csv" "${result_dir}/summary")
if [[ "${dry_run}" -eq 1 || -f "${result_dir}/time_limit_sensitivity.csv" ]]; then
    tar_inputs+=("${result_dir}/time_limit_sensitivity.csv")
fi
if [[ "${dry_run}" -eq 1 || -d "${result_dir}/time_limit_sensitivity_summary" ]]; then
    tar_inputs+=("${result_dir}/time_limit_sensitivity_summary")
fi
if [[ "${dry_run}" -eq 1 || -f "${result_dir}/heuristic_baselines.csv" ]]; then
    tar_inputs+=("${result_dir}/heuristic_baselines.csv")
fi
if [[ "${dry_run}" -eq 1 || -f "${result_dir}/heuristic_baselines_refreshed.csv" ]]; then
    tar_inputs+=("${result_dir}/heuristic_baselines_refreshed.csv")
fi
if [[ "${dry_run}" -eq 1 || -f "${result_dir}/partition_optimal_refreshed.csv" ]]; then
    tar_inputs+=("${result_dir}/partition_optimal_refreshed.csv")
fi
if [[ "${dry_run}" -eq 1 || -f "${result_dir}/highs_baseline_maxN10000.csv" ]]; then
    tar_inputs+=("${result_dir}/highs_baseline_maxN10000.csv")
fi
if [[ "${dry_run}" -eq 1 || -d "${result_dir}/comparison_summary" ]]; then
    tar_inputs+=("${result_dir}/comparison_summary")
fi
if [[ "${dry_run}" -eq 1 || -d "${result_dir}/paper_tables" ]]; then
    tar_inputs+=("${result_dir}/paper_tables")
fi

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

echo "result_dir: ${result_dir}"
echo "artifact_dir: ${artifact_dir}"
echo "archive_path: ${archive_path}"

if [[ "${dry_run}" -eq 0 ]]; then
    if [[ ! -f "${result_dir}/results.csv" ]]; then
        echo "Missing final results CSV: ${result_dir}/results.csv" >&2
        exit 1
    fi
    if [[ ! -d "${result_dir}/summary" ]]; then
        echo "Missing final summary directory: ${result_dir}/summary" >&2
        exit 1
    fi
    mkdir -p "${artifact_dir}"
fi

run_cmd tar -czf "${archive_path}" "${tar_inputs[@]}"
run_cmd shasum -a 256 "${archive_path}"

if [[ "${dry_run}" -eq 0 ]]; then
    shasum -a 256 "${archive_path}" > "${checksum_path}"
    ls -lh "${archive_path}" "${checksum_path}"
fi
