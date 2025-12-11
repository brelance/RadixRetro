# node_ids="96,95,97"
node_ids="2"
trace_file="/HOME/nju_mli/nju_mli_1/junjie/RadixRetro/traces/radix_dump/tree_node_trace_30.json"
tokenizer_path="/HOME/nju_mli/nju_mli_1/HDD_POOL/dpskv2"

python3 benchmarks/query_radix_nodes.py \
  --trace "$trace_file" \
  --node-ids "$node_ids" \
  --detokenize \
  --tokenizer-path "$tokenizer_path" \
  --skip-special-tokens
