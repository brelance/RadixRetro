TRACE_ID=100

python benchmarks/analyze_radix_tree.py \
    --input ./traces/radix_dump_whit/tree_node_trace_${TRACE_ID}.json \
    --output $HOME/junjie/RadixRetro/traces/results/analyze_radix_tree/tree_node_trace_whit_${TRACE_ID}.json \
    --hits-output $HOME/junjie/RadixRetro/traces/results/analyze_radix_tree/nodes_by_hit.json
