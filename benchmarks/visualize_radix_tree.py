import argparse
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover - optional dependency
    raise SystemExit(
        "matplotlib is required for visualization. Please install it first."
    ) from exc


def _load_snapshot(path: str, index: int):
    with open(path, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    if not lines:
        raise ValueError(f"No snapshots found in {path}")

    idx = index if index >= 0 else len(lines) + index
    idx = max(0, min(idx, len(lines) - 1))
    return json.loads(lines[idx])


def _compute_positions(
    nodes: Dict[int, Dict], edges: List[Tuple[int, int]]
) -> Dict[int, Tuple[float, float]]:
    children = defaultdict(list)
    parents = set()
    for parent_id, child_id in edges:
        children[parent_id].append(child_id)
        parents.add(child_id)

    roots = [n["node_id"] for n in nodes.values() if n.get("parent_id") is None]
    if not roots and nodes:
        roots = [nid for nid in nodes if nid not in parents]

    depth_map: Dict[int, int] = {}

    def dfs(node_id: int, depth: int):
        if node_id in depth_map:
            return
        depth_map[node_id] = depth
        for child in children.get(node_id, []):
            dfs(child, depth + 1)

    for root in roots:
        dfs(root, 0)

    positions: Dict[int, Tuple[float, float]] = {}
    level_to_nodes: Dict[int, List[int]] = defaultdict(list)
    for nid, depth in depth_map.items():
        level_to_nodes[depth].append(nid)

    for depth, level_nodes in level_to_nodes.items():
        for idx, nid in enumerate(level_nodes):
            offset = idx - (len(level_nodes) - 1) / 2
            positions[nid] = (offset, -depth)

    return positions


def _format_label(node: Dict) -> str:
    tokens = node.get("tokens", [])
    extra = node.get("extra_key")
    token_preview = (
        str(tokens[:4]) + ("..." if len(tokens) > 4 else "")
        if isinstance(tokens, list)
        else str(tokens)
    )
    prefix = f"{node.get('node_id')}"
    if extra:
        prefix += f" | {extra}"
    return f"{prefix}\n{token_preview}"


def visualize(snapshot: Dict, output_path: str):
    nodes = {n["node_id"]: n for n in snapshot.get("nodes", [])}
    edges = snapshot.get("edges", [])
    positions = _compute_positions(nodes, edges)

    plt.figure(figsize=(10, 6))

    # Draw edges
    for parent_id, child_id in edges:
        if parent_id not in positions or child_id not in positions:
            continue
        x0, y0 = positions[parent_id]
        x1, y1 = positions[child_id]
        plt.plot([x0, x1], [y0, y1], color="gray", linewidth=1.0, zorder=1)

    # Draw nodes
    for node_id, (x, y) in positions.items():
        node = nodes.get(node_id, {})
        plt.scatter([x], [y], s=120, color="#4e79a7", zorder=2)
        plt.text(
            x,
            y + 0.05,
            _format_label(node),
            ha="center",
            va="bottom",
            fontsize=8,
            color="black",
        )

    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    print(f"Saved RadixTree visualization to {os.path.abspath(output_path)}")


def main():
    parser = argparse.ArgumentParser(
        description="Visualize RadixTree from tree_node_trace.json"
    )
    parser.add_argument(
        "--trace",
        default="tree_node_trace.json",
        help="Path to tree_node_trace.json (JSONL with snapshots)",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Snapshot index to draw (default: last)",
    )
    parser.add_argument(
        "--output",
        default="radix_tree.png",
        help="Output image path",
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args.trace, args.index)
    visualize(snapshot, args.output)


if __name__ == "__main__":
    main()
