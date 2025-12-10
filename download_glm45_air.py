"""Download zai-org/GLM-4.5-Air with ModelScope into a target directory."""

import argparse
import os
import sys

try:
    from modelscope.hub.snapshot_download import snapshot_download
except ModuleNotFoundError:  # pragma: no cover - import guard
    sys.stderr.write(
        "ModelScope is not installed. Install it with `pip install modelscope`.\n"
    )
    sys.exit(1)


MODEL_ID = "zai-org/GLM-4.5-Air"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download GLM-4.5-Air from ModelScope to a local directory."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Destination directory where the model files will be stored.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional model revision/branch/tag. Defaults to the latest.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = os.path.abspath(args.target)
    os.makedirs(target_dir, exist_ok=True)

    snapshot_download(
        model_id=MODEL_ID,
        cache_dir=target_dir,
        revision=args.revision,
        local_files_only=False,
    )
    print(f"Model downloaded to {target_dir}")


if __name__ == "__main__":
    main()
