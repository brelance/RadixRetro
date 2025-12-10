#!/bin/bash
# path = "0,4,7,94,96,95"
path="0,4,7,94,96,97"
trace_file="/HOME/nju_mli/nju_mli_1/junjie/RadixRetro/traces/tree_node_trace_30.json"
tokenizer_path="/HOME/nju_mli/nju_mli_1/HDD_POOL/dpskv2"

python3 benchmarks/extract_radix_path.py \
  --trace "$trace_file" \
  --path "$path" \
  --print-detokenized \
  --tokenizer-path "$tokenizer_path" \
  --detokenize-mode auto \
  --skip-special-tokens \
  --clean-up-spaces