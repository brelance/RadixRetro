#!/usr/bin/env python3
"""Aggregate MoE expert selections by layer across requests."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate experts and counts for each layer in a MoE selection trace."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("traces/radix_moe/moe_selections.jsonl"),
        help="Path to the input JSONL file produced by the MoE trace.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmarks/analyzeMoe/aggregated_moe_selections.json"),
        help="Where to write the aggregated per-layer results.",
    )
    parser.add_argument(
        "--sort-by",
        choices=["count", "expert"],
        default="count",
        help="Sort experts by aggregated count (desc) or expert id (asc).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=None,
        help="If set, only keep the top-k experts per layer after sorting.",
    )
    return parser.parse_args()


def parse_line(
    line: str, line_no: int, source: Path
) -> Tuple[int, List[int], List[int]]:
    try:
        record = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Failed to parse line {line_no} in {source}: {exc}") from exc

    if "layer" not in record:
        raise ValueError(f"Missing 'layer' in line {line_no} of {source}")
    layer = int(record["layer"])

    experts = record.get("experts")
    counts = record.get("counts")
    if not isinstance(experts, list) or not isinstance(counts, list):
        raise ValueError(
            f"'experts' and 'counts' must be lists in line {line_no} of {source}"
        )
    if len(experts) != len(counts):
        raise ValueError(
            f"Length mismatch between experts ({len(experts)}) and counts ({len(counts)}) "
            f"in line {line_no} of {source}"
        )

    try:
        expert_ids = [int(e) for e in experts]
        expert_counts = [int(c) for c in counts]
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Non-integer expert or count value in line {line_no} of {source}"
        ) from exc

    return layer, expert_ids, expert_counts


def aggregate_by_layer(
    records: Iterable[Tuple[int, List[int], List[int]]]
) -> Tuple[Dict[int, Dict[int, int]], Dict[int, int]]:
    layer_expert_counts: DefaultDict[int, DefaultDict[int, int]] = defaultdict(
        lambda: defaultdict(int)
    )
    layer_request_counts: DefaultDict[int, int] = defaultdict(int)

    for layer, experts, counts in records:
        layer_request_counts[layer] += 1
        for expert, count in zip(experts, counts):
            layer_expert_counts[layer][expert] += count

    return layer_expert_counts, layer_request_counts


def load_records(input_path: Path) -> List[Tuple[int, List[int], List[int]]]:
    records: List[Tuple[int, List[int], List[int]]] = []
    with input_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            records.append(parse_line(line, line_no, input_path))
    return records


def build_layer_payload(
    layer_expert_counts: Dict[int, Dict[int, int]],
    layer_request_counts: Dict[int, int],
    sort_by: str,
    top_k: int | None,
) -> List[Dict[str, object]]:
    layers: List[Dict[str, object]] = []
    for layer in sorted(layer_expert_counts.keys()):
        counts_by_expert = layer_expert_counts[layer]
        if sort_by == "count":
            sorted_items = sorted(
                counts_by_expert.items(), key=lambda item: (-item[1], item[0])
            )
        else:
            sorted_items = sorted(counts_by_expert.items(), key=lambda item: item[0])
        if top_k is not None:
            sorted_items = sorted_items[:top_k]
        experts = [expert for expert, _ in sorted_items]
        counts = [count for _, count in sorted_items]
        layers.append(
            {
                "layer": layer,
                "experts": experts,
                "counts": counts,
                "total_count": int(sum(counts_by_expert.values())),
                "request_count": layer_request_counts.get(layer, 0),
            }
        )
    return layers


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")

    records = load_records(args.input)
    layer_expert_counts, layer_request_counts = aggregate_by_layer(records)
    layers = build_layer_payload(
        layer_expert_counts, layer_request_counts, args.sort_by, args.top_k
    )

    payload = {
        "source": str(args.input),
        "total_layers": len(layers),
        "layers": layers,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(
        f"Aggregated {len(records)} records across {len(layers)} layers. "
        f"Results saved to {args.output}."
    )


if __name__ == "__main__":
    main()
