TRACE_ID=90
python benchmarks/analyze_radix_tree.py \
    --input ./traces/tree_node_trace_${TRACE_ID}.json \
    --output $HOME/nju_mli/nju_mli_1/junjie/RadixRetro/traces/results/analyze_radix_tree/tree_node_trace_${TRACE_ID}.json