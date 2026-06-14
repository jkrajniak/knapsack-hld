#!/usr/bin/env bash
# Post-G2 pipeline for the Optional class-ordering 300s sub-run.
#
# Orchestrates the five steps that close Optional G2:
#   1. Summarise the three per-ordering CSVs into headline tables.
#   2. Sync the raw + summarised results down from the remote VM to
#      the local artifacts directory.
#   3. Tar + sha256 the bundle into the artifacts directory.
#   4. Print an evidence-index pin block to paste into your
#      experiment-tracking notebook.
#   5. Print the 60s-vs-300s comparison-writeup template to drop
#      into your experiment-tracking notebook.
#
# Designed to run AFTER the remote tmux session `class_ord_g2:run`
# finishes (all three of sequential.csv / random.csv /
# adversarial.csv have 210 data rows + 1 header = 211 lines each).
#
# Idempotent: every step can be re-run; existing files are not
# clobbered without --force.
#
# Usage:
#   scripts/run_post_g2_pipeline.sh \
#       [--remote remote.example] \
#       [--remote-port 2222] \
#       [--remote-user "$USER"] \
#       [--remote-root '~/knapsack-hld'] \
#       [--results-subdir results/class_ordering_300s] \
#       [--run-id 20260528T1620Z_g2_300s] \
#       [--artifacts-dir ../artifacts/class_ordering] \
#       [--summary-subdir results/class_ordering_300s_summary] \
#       [--force] [--skip-step N[,N...]] [--dry-run]
#
# Example:
#   scripts/run_post_g2_pipeline.sh --run-id 20260528T1620Z_g2_300s

set -euo pipefail

remote_host="${KNAPSACK_REMOTE_HOST:-remote.example}"
remote_port="2222"
remote_user="${KNAPSACK_REMOTE_USER:-$USER}"
remote_root='~/knapsack-hld'
results_subdir="results/class_ordering_300s"
summary_subdir="results/class_ordering_300s_summary"
artifacts_dir="${KNAPSACK_ARTIFACTS_DIR:-../artifacts}/class_ordering"
run_id=""
force=0
dry_run=0
skip_steps=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote) remote_host="$2"; shift 2 ;;
        --remote-port) remote_port="$2"; shift 2 ;;
        --remote-user) remote_user="$2"; shift 2 ;;
        --remote-root) remote_root="$2"; shift 2 ;;
        --results-subdir) results_subdir="$2"; shift 2 ;;
        --summary-subdir) summary_subdir="$2"; shift 2 ;;
        --artifacts-dir) artifacts_dir="$2"; shift 2 ;;
        --run-id) run_id="$2"; shift 2 ;;
        --force) force=1; shift ;;
        --dry-run) dry_run=1; shift ;;
        --skip-step) skip_steps="$2"; shift 2 ;;
        -h|--help) sed -n '2,40p' "$0"; exit 0 ;;
        *) echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

if [[ -z "${run_id}" ]]; then
    run_id="$(date -u +%Y%m%dT%H%MZ)_g2_300s"
fi

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"
cd "${repo_root}"

step_skipped() {
    [[ ",${skip_steps}," == *",$1,"* ]]
}

run_cmd() {
    printf '+ %s\n' "$*"
    if [[ "${dry_run}" -eq 1 ]]; then return 0; fi
    "$@"
}

ssh_cmd=(ssh -p "${remote_port}" -i ~/.ssh/id_rsa "${remote_user}@${remote_host}")

echo "==> run_id: ${run_id}"
echo "==> remote: ${remote_user}@${remote_host}:${remote_port}"
echo "==> results_subdir (remote + local): ${results_subdir}"
echo "==> summary_subdir (remote + local): ${summary_subdir}"
echo "==> artifacts_dir (local): ${artifacts_dir}"
echo

# ---------- Step 1: remote summariser ----------
if step_skipped 1; then
    echo "[1/5] SKIPPED — summariser"
else
    echo "[1/5] Summarising on remote (uv run python scripts/summarize_class_ordering.py)"
    remote_cmd="cd ${remote_root} && \
mkdir -p ${summary_subdir} && \
uv run python scripts/summarize_class_ordering.py \
    --results-dir ${results_subdir} \
    --out-dir ${summary_subdir}"
    run_cmd "${ssh_cmd[@]}" zsh -lc "'${remote_cmd}'"
fi

# ---------- Step 2: sync down ----------
if step_skipped 2; then
    echo "[2/5] SKIPPED — sync"
else
    echo "[2/5] Syncing ${results_subdir} + ${summary_subdir} from remote"
    mkdir -p "${results_subdir}" "${summary_subdir}"
    run_cmd rsync -avh --progress \
        -e "ssh -p ${remote_port} -i ${HOME}/.ssh/id_rsa" \
        "${remote_user}@${remote_host}:${remote_root}/${results_subdir}/" \
        "${results_subdir}/"
    run_cmd rsync -avh --progress \
        -e "ssh -p ${remote_port} -i ${HOME}/.ssh/id_rsa" \
        "${remote_user}@${remote_host}:${remote_root}/${summary_subdir}/" \
        "${summary_subdir}/"
fi

# ---------- Step 3: archive + checksum ----------
archive_path="${artifacts_dir}/class_ordering_300s_${run_id}.tar.gz"
checksum_path="${archive_path}.sha256"
if step_skipped 3; then
    echo "[3/5] SKIPPED — archive"
elif [[ -f "${archive_path}" && "${force}" -eq 0 ]]; then
    echo "[3/5] Archive already exists at ${archive_path}; use --force to overwrite."
else
    echo "[3/5] Archiving to ${archive_path}"
    mkdir -p "${artifacts_dir}"
    run_cmd tar -czf "${archive_path}" "${results_subdir}" "${summary_subdir}"
    if [[ "${dry_run}" -eq 0 ]]; then
        shasum -a 256 "${archive_path}" | tee "${checksum_path}"
        ls -lh "${archive_path}" "${checksum_path}"
    fi
fi

# ---------- Step 4: EVIDENCE_BASE.md pin block ----------
sha256=""
if [[ -f "${checksum_path}" ]]; then
    sha256="$(awk '{print $1}' "${checksum_path}")"
fi
short_sha="${sha256:0:8}…${sha256: -6}"
cat <<EOF

[4/5] Evidence-index pin block — paste into your experiment-tracking
      notebook under "Class-ordering ablation archives":

------------------------------------------------------------------------
- **Class-ordering ablation — Optional G2 (300s sub-run)** — \`${archive_path##*/}\`
  (SHA-256 \`${short_sha:-PENDING_SHA256}\`). 6 hardest \$N = 100\,000\$ cells from the
  primary 60s class-ordering ablation, sequential / random / adversarial
  orderings, 35 seeds per (cell × ordering), 210 instances per ordering.
  Generated 2026-05-28 by \`scripts/run_class_ordering_ablation.sh\` with
  \`--time-limit 300\` on an Apple M4 Pro (12 cores).
  Summariser: \`scripts/summarize_class_ordering.py\`. Paper sections
  affected: §3.6 (class-ordering ablation; 60s primary is the headline,
  this run is the sensitivity).
------------------------------------------------------------------------
EOF

# ---------- Step 5: 60s-vs-300s comparison writeup template ----------
cat <<EOF

[5/5] Comparison writeup template — save as
      20260528_class_ordering_g2_results.md in your experiment notebook:

------------------------------------------------------------------------
# Class-ordering ablation — G2 (300s sub-run) results vs primary (60s)

**Status:** results landed $(date -u +%Y-%m-%d) (\`${run_id}\`). Companion of
the 60s primary class-ordering writeup (default ordering selected).

## Sources

- 300s sub-run archive: \`${archive_path##*/}\` (SHA-256 \`${short_sha:-PENDING_SHA256}\`).
- 60s primary archive: see your evidence index, "Class-ordering ablation".

## Comparison method

Pair instances by \`(cell, seed)\` across the 60s and 300s runs.
For each ordering, report:
- median paired \$\\Delta\$profit (300s − 60s) in absolute units;
- median paired \$\\Delta\$profit in % of the 60s incumbent;
- \$N\$-cell win count where the 300s incumbent strictly exceeds the
  60s incumbent;
- \$N\$-cell timeout-rate (the \$N = 100\,000\$ rows for which
  \`status=timeout\` in the 60s primary but \`status=feasible\` in 300s).

## Headline numbers (FILL IN AFTER STEP 1)

| Ordering   | seeds | 60s win | 300s win | tied | median Δprofit (%) | timeout escape rate |
|------------|-------|---------|----------|------|---------------------|---------------------|
| sequential |       |         |          |      |                     |                     |
| random     |       |         |          |      |                     |                     |
| adversarial|       |         |          |      |                     |                     |

## Manuscript implications

- §3.6: confirm the default-ordering framing (sequential as the default)
  holds at 300s, OR flag if random / adversarial overtake on the hardest cells.
- §3.9 / §3.11 sensitivity sentence: cross-check that the +141.85%
  median gain at 300s reported in §3.9 (which is HLD vs PO, not
  class-ordering vs class-ordering) is unaffected by this run.

## Decisions

- [ ] Sign off the headline numbers.
- [ ] If §3.6 framing changes, propose a sentence delta to §3.6.
- [ ] Pin the archive in your evidence index (already templated in
      this script's step 4 output).
------------------------------------------------------------------------

==> Pipeline complete. Remaining manual steps:
    1. Verify the headline numbers in the template against the
       summariser output (results/class_ordering_300s_summary/).
    2. Commit the writeup + evidence-index pin in your notebook.
    3. If §3.6 needs amending, draft the diff and apply it to the
       manuscript.
EOF
