import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load_snapshot(path: str, index: int):
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()

    if not content:
        raise ValueError(f"No snapshots found in {path}")

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list) and parsed:
            idx = index if index >= 0 else len(parsed) + index
            idx = max(0, min(idx, len(parsed) - 1))
            return parsed[idx]
    except json.JSONDecodeError:
        pass

    lines = [ln.strip() for ln in content.splitlines() if ln.strip()]
    if not lines:
        raise ValueError(f"No snapshots found in {path}")
    idx = index if index >= 0 else len(lines) + index
    idx = max(0, min(idx, len(lines) - 1))
    return json.loads(lines[idx])


def _parse_ids(ids_str: str) -> List[int]:
    parts = [p.strip() for p in ids_str.replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Please provide at least one node id.")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid node id in '{ids_str}'") from exc


def flatten_tokens(tokens: List[Any]) -> List[int]:
    if not tokens:
        return []
    first = tokens[0]
    if isinstance(first, (tuple, list)):
        flat: List[int] = []
        for chunk in tokens:
            vals = list(chunk)
            if not vals:
                continue
            if not flat:
                flat.extend(int(x) for x in vals)
            else:
                flat.extend(int(x) for x in vals[1:])
        return flat
    return [int(tok) for tok in tokens]


def main():
    parser = argparse.ArgumentParser(
        description="Query nodes by id from tree_node_trace.json and show tokens."
    )
    parser.add_argument(
        "--trace",
        default="tree_node_trace.json",
        help="Path to tree_node_trace.json",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=-1,
        help="Snapshot index (default: last)",
    )
    parser.add_argument(
        "--node-ids",
        required=True,
        help="Comma/space separated node ids to inspect (e.g., '0,3,7')",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Optional HuggingFace tokenizer for detokenization",
    )
    parser.add_argument(
        "--detokenize",
        action="store_true",
        help="Detokenize tokens using the provided tokenizer",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )
    parser.add_argument(
        "--revision",
        help="Specific tokenizer revision to load",
    )
    parser.add_argument(
        "--skip-special-tokens",
        action="store_true",
        help="Skip special tokens during detokenization",
    )
    parser.add_argument(
        "--clean-up-spaces",
        action="store_true",
        help="Clean up tokenization spaces during detokenization",
    )
    args = parser.parse_args()

    snapshot = _load_snapshot(args.trace, args.index)
    node_ids = _parse_ids(args.node_ids)
    nodes = {node["node_id"]: node for node in snapshot.get("nodes", [])}

    tokenizer = None
    if args.detokenize:
        if not args.tokenizer_path:
            raise ValueError("--tokenizer-path is required when --detokenize is set")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("transformers is required for detokenization") from exc
        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
        )

    for nid in node_ids:
        node = nodes.get(nid)
        if node is None:
            print(f"[WARN] Node {nid} not found in snapshot.")
            continue
        tokens = node.get("tokens", [])
        print(f"Node {nid}:")
        print(f"  tokens: {tokens}")
        if tokenizer is not None:
            token_ids = flatten_tokens(tokens)
            text = tokenizer.decode(
                token_ids,
                skip_special_tokens=args.skip_special_tokens,
                clean_up_tokenization_spaces=args.clean_up_spaces,
            )
            print("  detokenized:")
            print(f"    {text}")


if __name__ == "__main__":
    main()
