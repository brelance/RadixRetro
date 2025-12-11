#!/usr/bin/env bash
set -euo pipefail

# Run lmsys-chat-1m.py with varying NUM_SAMPLES and visualize the resulting radix trees.
# Optionally set MODEL_PATH or DATASET_PATH env vars to override defaults in the Python script.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRACE_DIR="${TRACE_DIR:-"$ROOT_DIR/traces/radix_dump_whit"}"
mkdir -p "$TRACE_DIR"

MODEL_ARG=()
DATASET_ARG=()
DISABLE_RADIX_ARG=()

if [[ -n "${MODEL_PATH:-}" ]]; then
  MODEL_ARG=(--model-path "$MODEL_PATH")
fi

if [[ -n "${DATASET_PATH:-}" ]]; then
  DATASET_ARG=(--dataset-path "$DATASET_PATH")
fi

if [[ "${DISABLE_RADIX:-0}" != "0" ]]; then
  DISABLE_RADIX_ARG=(--disable-radix)
fi

for n in {10..100..10}; do
# for n in 10; do
  trace_path="$TRACE_DIR/tree_node_trace_${n}.json"
  img_path="$TRACE_DIR/radix_tree_${n}.png"

  echo "Running NUM_SAMPLES=$n -> $trace_path"
  python3 "$ROOT_DIR/benchmarks/lmsys-chat-1m.py" \
    --num-samples "$n" \
    --trace-path "$trace_path" \
    "${MODEL_ARG[@]}" \
    "${DATASET_ARG[@]}" \
    "${DISABLE_RADIX_ARG[@]}"

  echo "Visualizing $trace_path -> $img_path"
  python3 "$ROOT_DIR/benchmarks/visualize_radix_tree.py" \
    --trace "$trace_path" \
    --output "$img_path"
done

echo "All runs completed. Traces and images are under $TRACE_DIR"
