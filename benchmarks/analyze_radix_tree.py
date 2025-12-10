#!/usr/bin/env python3
"""Utility script to summarize radix tree node snapshots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze a tree_node_trace.json snapshot and summarize "
            "overall node statistics."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("tree_node_trace.json"),
        help="Path to the tree_node_trace.json file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/radix_tree_multi_child_nodes.json"),
        help="Where to write the multi-child node information JSON.",
    )
    parser.add_argument(
        "--hits-output",
        type=Path,
        default=Path("benchmarks/radix_tree_nodes_by_hit.json"),
        help="Where to write nodes sorted by hit_count in descending order.",
    )
    return parser.parse_args()


def load_nodes(tree_trace_path: Path) -> List[Dict[str, Any]]:
    with tree_trace_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    nodes = data.get("nodes")
    if nodes is None:
        raise ValueError(f"'nodes' entry missing in {tree_trace_path}")
    return nodes


def collect_multi_child_nodes(nodes: List[Dict[str, Any]]) -> Dict[str, Any]:
    multi_child_nodes: List[Dict[str, Any]] = []
    total_tokens_length = 0
    multi_child_tokens_length = 0
    for node in nodes:
        children_count = int(node.get("children_count") or 0)
        tokens = node.get("tokens") or []
        token_len = len(tokens)
        total_tokens_length += token_len
        if children_count > 1:
            multi_child_nodes.append(
                {
                    "node_id": node.get("node_id"),
                    "hit_count": node.get("hit_count"),
                    "tokens_length": token_len,
                    "children_count": children_count,
                }
            )
            multi_child_tokens_length += token_len
    return {
        "multi_child_nodes": multi_child_nodes,
        "total_tokens_length": total_tokens_length,
        "multi_child_tokens_length": multi_child_tokens_length,
    }


def dump_nodes_sorted_by_hit(nodes: List[Dict[str, Any]], output_path: Path) -> None:
    sorted_nodes: List[Dict[str, Any]] = []
    for node in nodes:
        tokens = node.get("tokens") or []
        hit_count = node.get("hit_count")
        hit_value = int(hit_count or 0)
        sorted_nodes.append(
            {
                "node_id": node.get("node_id"),
                "tokens": tokens,
                "hit_count": hit_value,
                "tokens_length": len(tokens),
            }
        )
    sorted_nodes.sort(key=lambda item: item["hit_count"], reverse=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(sorted_nodes, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    nodes = load_nodes(args.input)
    total_nodes = len(nodes)
    stats = collect_multi_child_nodes(nodes)

    payload = {
        "source": str(args.input),
        "total_nodes": total_nodes,
        "total_tokens_length": stats["total_tokens_length"],
        "multi_child_nodes_count": len(stats["multi_child_nodes"]),
        "multi_child_tokens_length": stats["multi_child_tokens_length"],
        "multi_child_nodes": stats["multi_child_nodes"],
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    dump_nodes_sorted_by_hit(nodes, args.hits_output)

    print(
        f"Total nodes: {total_nodes}. "
        f"Nodes with >1 child: {len(stats['multi_child_nodes'])}. "
        f"Details saved to {args.output}."
    )


if __name__ == "__main__":
    main()
