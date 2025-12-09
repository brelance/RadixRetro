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


def _parse_path(path_str: str) -> List[int]:
    parts = [p.strip() for p in path_str.replace(" ", ",").split(",") if p.strip()]
    if not parts:
        raise ValueError("Path must contain at least one node id, e.g. '0,3,7'")
    try:
        return [int(p) for p in parts]
    except ValueError as exc:
        raise ValueError(f"Invalid path component in '{path_str}'") from exc


def _normalize_tokens(tokens: List[Any]) -> List[Any]:
    normalized = []
    for tok in tokens or []:
        if isinstance(tok, list):
            normalized.append(tuple(tok))
        else:
            normalized.append(tok)
    return normalized


def _validate_path(nodes: Dict[int, Dict], path: List[int]):
    for nid in path:
        if nid not in nodes:
            raise ValueError(f"Node id {nid} not found in snapshot.")
    for prev, nxt in zip(path, path[1:]):
        parent_id = nodes[nxt].get("parent_id")
        if parent_id != prev:
            raise ValueError(
                f"Invalid path segment {prev}->{nxt}: "
                f"node {nxt} has parent {parent_id}."
            )


def extract_tokens(snapshot: Dict, path_ids: List[int], mode: str, show_extra: bool):
    nodes = {node["node_id"]: node for node in snapshot.get("nodes", [])}
    _validate_path(nodes, path_ids)

    if mode == "concat":
        collected = []
        for nid in path_ids:
            collected.extend(_normalize_tokens(nodes[nid].get("tokens", [])))
        return collected

    details = []
    for nid in path_ids:
        node = nodes[nid]
        entry = {
            "node_id": nid,
            "tokens": _normalize_tokens(node.get("tokens", [])),
        }
        if show_extra:
            entry["parent_id"] = node.get("parent_id")
            entry["extra_key"] = node.get("extra_key")
            entry["key_len"] = node.get("key_len")
            entry["value_len"] = node.get("value_len")
            entry["lock_ref"] = node.get("lock_ref")
        details.append(entry)
    return details


def flatten_for_detokenize(tokens: List[Any]) -> List[int]:
    """Convert stored tokens (ints or tuples) into a linear token id sequence."""
    if not tokens:
        return []
    first = tokens[0]
    if isinstance(first, (tuple, list)):
        flat: List[int] = []
        for chunk in tokens:
            chunk_vals = list(chunk)
            if not chunk_vals:
                continue
            if not flat:
                flat.extend(int(x) for x in chunk_vals)
            else:
                flat.extend(int(x) for x in chunk_vals[1:])
        return flat
    return [int(tok) for tok in tokens]


def detokenize_sequence(tokenizer, token_ids: List[int], skip_special_tokens: bool, clean_up_spaces: bool):
    if not token_ids:
        return ""
    return tokenizer.decode(
        token_ids,
        skip_special_tokens=skip_special_tokens,
        clean_up_tokenization_spaces=clean_up_spaces,
    )


def main():
    parser = argparse.ArgumentParser(
        description="Extract token sequences for a path in tree_node_trace.json"
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
        "--path",
        required=True,
        help="Comma/space separated node ids representing the path (e.g., '0,3,7')",
    )
    parser.add_argument(
        "--output-mode",
        choices=["concat", "per-node"],
        default="concat",
        help="Output tokens concatenated or per node",
    )
    parser.add_argument(
        "--show-extra",
        action="store_true",
        help="Show extra node metadata when using per-node mode",
    )
    parser.add_argument(
        "--tokenizer-path",
        help="Path/name of HuggingFace tokenizer for detokenization",
    )
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Trust remote code when loading tokenizer",
    )
    parser.add_argument(
        "--revision",
        help="Specific revision of tokenizer to load",
    )
    parser.add_argument(
        "--print-detokenized",
        action="store_true",
        help="Print detokenized text (requires --tokenizer-path)",
    )
    parser.add_argument(
        "--detokenize-mode",
        choices=["auto", "raw"],
        default="auto",
        help="auto: reconstruct from tuples/bigrams; raw: treat tokens as-is",
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
    parser.add_argument(
        "--max-detokenize-length",
        type=int,
        default=0,
        help="Limit number of tokens when detokenizing (0 = no limit)",
    )
    args = parser.parse_args()

    trace_path = Path(args.trace)
    snapshot = _load_snapshot(str(trace_path), args.index)
    node_path = _parse_path(args.path)
    result = extract_tokens(
        snapshot,
        node_path,
        mode=args.output_mode,
        show_extra=args.show_extra,
    )

    print(f"Trace file: {trace_path.resolve()}")
    print(f"Snapshot index: {args.index}")
    print(f"Path: {node_path}")

    tokenizer = None
    if args.print_detokenized:
        if not args.tokenizer_path:
            raise ValueError("Please specify --tokenizer-path when using --print-detokenized")
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers is required for detokenization. Please install it via pip."
            ) from exc

        tokenizer = AutoTokenizer.from_pretrained(
            args.tokenizer_path,
            trust_remote_code=args.trust_remote_code,
            revision=args.revision,
        )

    def prepare_for_detok(tokens: List[Any]) -> List[int]:
        if args.detokenize_mode == "auto":
            return flatten_for_detokenize(tokens)
        return [int(tok) for tok in tokens if isinstance(tok, (int, float))]

    def maybe_clip(ids: List[int]) -> List[int]:
        if args.max_detokenize_length and args.max_detokenize_length > 0:
            return ids[: args.max_detokenize_length]
        return ids

    if args.output_mode == "concat":
        print("Concatenated tokens:")
        print(result)
        print(f"Total tokens: {len(result)}")
        if tokenizer is not None:
            token_ids = maybe_clip(prepare_for_detok(result))
            text = detokenize_sequence(
                tokenizer,
                token_ids,
                skip_special_tokens=args.skip_special_tokens,
                clean_up_spaces=args.clean_up_spaces,
            )
            print("Detokenized text:\n---")
            print(text)
            print("---")
    else:
        for entry in result:
            print(f"Node {entry['node_id']}:")
            print(f"  tokens: {entry['tokens']}")
            if args.show_extra:
                print(
                    f"  parent_id: {entry.get('parent_id')} | extra_key: {entry.get('extra_key')}"
                )
                print(
                    f"  key_len: {entry.get('key_len')} | value_len: {entry.get('value_len')} | lock_ref: {entry.get('lock_ref')}"
                )
            if tokenizer is not None:
                token_ids = maybe_clip(prepare_for_detok(entry["tokens"]))
                text = detokenize_sequence(
                    tokenizer,
                    token_ids,
                    skip_special_tokens=args.skip_special_tokens,
                    clean_up_spaces=args.clean_up_spaces,
                )
                print("  detokenized:")
                print(f"    {text}")


if __name__ == "__main__":
    main()
