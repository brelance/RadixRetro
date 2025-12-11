#!/usr/bin/env bash
# Run mmlu_offline.py for every MMLU subject and aggregate MoE selections.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Paths and knobs (override via env if needed)
DATA_DIR="${DATA_DIR:-"${SCRIPT_DIR}/../datasets/MMLU"}"
LOG_ROOT="${LOG_ROOT:-"${SCRIPT_DIR}/../traces/radix_moe/mmlu"}"
AGG_ROOT="${AGG_ROOT:-"${SCRIPT_DIR}/../traces/radix_moe/agg"}"
RESULT_FILE="${RESULT_FILE:-"${SCRIPT_DIR}/analyzeMoe/mmlu_results.jsonl"}"
RAW_RESULT_DIR="${RAW_RESULT_DIR:-""}"  # optional dump of per-prompt outputs
# Extra args for mmlu_offline (e.g., model flags); space-separated string is fine.
MMLU_EXTRA_ARGS="${MMLU_EXTRA_ARGS:-""}"

MMLU_SCRIPT="${SCRIPT_DIR}/mmlu_offline.py"
AGG_SCRIPT="${SCRIPT_DIR}/analyzeMoe/aggregate_moe_selections.py"

if [[ ! -d "${DATA_DIR}/test" ]]; then
  echo "Missing MMLU data directory: ${DATA_DIR}/test" >&2
  exit 1
fi

mkdir -p "${LOG_ROOT}" "${AGG_ROOT}"

# Expand user-provided extra args safely.
# shellcheck disable=SC2206
EXTRA_ARR=(${MMLU_EXTRA_ARGS})

# Iterate over all *_test.csv under MMLU/test (57 subjects expected).
mapfile -t SUBJECTS < <(find "${DATA_DIR}/test" -maxdepth 1 -type f -name "*_test.csv" -print | sort)

for test_csv in "${SUBJECTS[@]}"; do
  base_name="$(basename "${test_csv}")"
  subject="${base_name%_test.csv}"

  log_dir="${LOG_ROOT}/${subject}"
  agg_json="${AGG_ROOT}/${subject}_aggregated.json"
  agg_txt="${AGG_ROOT}/${subject}_aggregated.txt"
  mkdir -p "${log_dir}"

  # Configure MoE logger to capture selections for this subject.
  export SGLANG_RADIX_MOE_LOG_DIR="${log_dir}"

  raw_args=()
  if [[ -n "${RAW_RESULT_DIR}" ]]; then
    mkdir -p "${RAW_RESULT_DIR}"
    raw_args=(--raw-result-file "${RAW_RESULT_DIR}/${subject}.jsonl")
  fi

  echo "Running subject: ${subject}"
  python3 "${MMLU_SCRIPT}" \
    --data-dir "${DATA_DIR}" \
    --subjects "${subject}" \
    --result-file "${RESULT_FILE}" \
    "${EXTRA_ARR[@]}" \
    "${raw_args[@]}"

  input_jsonl="${log_dir}/moe_selections.jsonl"
  if [[ ! -s "${input_jsonl}" ]]; then
    echo "Warning: missing MoE log for ${subject} at ${input_jsonl}, skipping aggregation." >&2
    continue
  fi

  python3 "${AGG_SCRIPT}" \
    --input "${input_jsonl}" \
    --output "${agg_json}" \
    --text-output "${agg_txt}"
done

echo "Done. Aggregated outputs in ${AGG_ROOT}, MoE logs in ${LOG_ROOT}."
