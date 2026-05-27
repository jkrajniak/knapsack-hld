#!/usr/bin/env bash
# Run the HLD class-ordering ablation (Task 3.3.2 of revision-finalization-2026).
#
# Sequentially runs the three orderings (sequential, random, adversarial)
# on the six hard cells defined in run_time_limit_sensitivity.py, writing
# one CSV per ordering into results/class_ordering/.
#
# Sequential, NOT parallel: each ordering uses --jobs 12 by itself; running
# the three concurrently would oversubscribe the M4 Pro the same way the
# 2026-05-13 HiGHS incident did.
#
# Usage (from the knapsack-hld repo root):
#   scripts/run_class_ordering_ablation.sh                # primary 60s run
#   scripts/run_class_ordering_ablation.sh --dry-run      # print plan, no compute
#   TIME_LIMIT_S=300 scripts/run_class_ordering_ablation.sh  # override budget
#
# Optional env overrides:
#   JOBS               (default: 12)            joblib worker count
#   TIME_LIMIT_S       (default: 60)            per-instance solver cap
#   SEED               (default: 7)             random seed passed to HLD
#   OUT_DIR            (default: results/class_ordering)
#   ARCHIVE            (default: instances)     instance archive root
#   MANIFEST           (default: instances/MANIFEST.json)
#   CONFIG             (default: configs/hld_smac_best.json)
#   ORDERINGS          (default: "sequential random adversarial")
#
set -euo pipefail

DRY_RUN=0
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN=1
fi

JOBS="${JOBS:-12}"
TIME_LIMIT_S="${TIME_LIMIT_S:-60}"
SEED="${SEED:-7}"
OUT_DIR="${OUT_DIR:-results/class_ordering}"
ARCHIVE="${ARCHIVE:-instances}"
MANIFEST="${MANIFEST:-instances/MANIFEST.json}"
CONFIG="${CONFIG:-configs/hld_smac_best.json}"
ORDERINGS="${ORDERINGS:-sequential random adversarial}"

# Hard cells — must stay in sync with
# scripts/run_time_limit_sensitivity.py::DEFAULT_CELLS.
HARD_CELLS=(
    "100000,20,inversely_strongly,0.5"
    "100000,20,strongly,0.5"
    "100000,20,uncorrelated,0.5"
    "100000,10,strongly,0.5"
    "100000,5,strongly,0.75"
    "100000,10,inversely_strongly,0.1"
)

mkdir -p "${OUT_DIR}/logs"

echo "class-ordering ablation plan"
echo "  out_dir:      ${OUT_DIR}"
echo "  archive:      ${ARCHIVE}"
echo "  manifest:     ${MANIFEST}"
echo "  config:       ${CONFIG}"
echo "  jobs:         ${JOBS}"
echo "  time_limit_s: ${TIME_LIMIT_S}"
echo "  seed:         ${SEED}"
echo "  orderings:    ${ORDERINGS}"
echo "  cells:        ${#HARD_CELLS[@]}"
for cell in "${HARD_CELLS[@]}"; do
    echo "    - ${cell}"
done

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "dry-run: not invoking run_final_experiments.py"
    exit 0
fi

# Compose the --cell flags once; reused for every ordering.
CELL_FLAGS=()
for cell in "${HARD_CELLS[@]}"; do
    CELL_FLAGS+=("--cell" "${cell}")
done

for ordering in ${ORDERINGS}; do
    out_csv="${OUT_DIR}/${ordering}.csv"
    log_path="${OUT_DIR}/logs/${ordering}.log"
    echo
    echo "==> ordering=${ordering}  out_csv=${out_csv}"
    uv run python scripts/run_final_experiments.py \
        --archive "${ARCHIVE}" \
        --manifest "${MANIFEST}" \
        --subset test \
        --solvers hld \
        --class-ordering "${ordering}" \
        "${CELL_FLAGS[@]}" \
        --jobs "${JOBS}" \
        --highs-threads 1 \
        --seed "${SEED}" \
        --time-limit-s "${TIME_LIMIT_S}" \
        --out-csv "${out_csv}" \
        2>&1 | tee "${log_path}"
done

echo
echo "All ordering runs complete. Suggested next step:"
echo "  uv run python scripts/summarize_class_ordering.py \\"
echo "    --results-dir ${OUT_DIR} \\"
echo "    --out-dir ${OUT_DIR}/summary"
