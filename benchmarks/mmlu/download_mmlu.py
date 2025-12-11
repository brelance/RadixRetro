#!/usr/bin/env python3
"""Download the public MMLU release into ``./datasets/MMLU``.

The upstream project hosts a single ``data.tar`` archive that contains the
``dev`` and ``test`` CSV splits (along with a few auxiliary folders).  This
script fetches the archive, extracts it into a temporary directory, and then
copies the contents into ``datasets/MMLU`` inside the repository (or a custom
location selected via ``--output-dir``).
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Iterable
from urllib.request import urlopen

MMLU_URL = "https://people.eecs.berkeley.edu/~hendrycks/data.tar"


def download_file(url: str, destination: Path) -> None:
    """Download ``url`` into ``destination`` with a simple progress indicator."""
    with urlopen(url) as response, destination.open("wb") as outfile:
        total = int(response.headers.get("Content-Length", "0"))
        downloaded = 0
        chunk_size = 1024 * 1024
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            outfile.write(chunk)
            downloaded += len(chunk)
            if total:
                percent = downloaded * 100 // total
                print(f"\rDownloading {url} [{percent}%]", end="", flush=True)
        print("\rDownload complete".ljust(80))


def ensure_empty_dir(path: Path, force: bool) -> None:
    """Create ``path`` (optionally clearing it first when ``force`` is True)."""
    if path.exists() and any(path.iterdir()):
        if not force:
            raise RuntimeError(
                f"{path} already exists and is not empty. Use --force to overwrite it."
            )
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def safe_extract(tar_path: Path, destination: Path) -> None:
    """Safely extract ``tar_path`` into ``destination``."""
    with tarfile.open(tar_path) as tar:
        base = destination.resolve()
        members = tar.getmembers()
        for member in members:
            member_path = (base / member.name).resolve()
            if not str(member_path).startswith(str(base)):
                raise RuntimeError(
                    f"Blocked extraction outside of {base}: {member.name}"
                )
        tar.extractall(destination)


def locate_data_root(extraction_dir: Path) -> Path:
    """Locate the folder that contains the split directories."""
    required = {"dev", "test"}
    for candidate in [extraction_dir] + list(extraction_dir.rglob("*")):
        if not candidate.is_dir():
            continue
        children = {child.name for child in candidate.iterdir() if child.is_dir()}
        if required.issubset(children):
            return candidate
    raise FileNotFoundError(
        f"Could not find a folder with {', '.join(sorted(required))} under {extraction_dir}"
    )


def copy_contents(src_root: Path, dst_root: Path) -> None:
    """Copy the contents (files + directories) from ``src_root`` to ``dst_root``."""
    for item in src_root.iterdir():
        target = dst_root / item.name
        if item.is_dir():
            shutil.copytree(item, target)
        else:
            shutil.copy2(item, target)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    repo_root = Path(__file__).resolve().parents[2]
    default_output = repo_root / "datasets" / "MMLU"

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=MMLU_URL,
        help=f"Location of the tar archive (default: {MMLU_URL})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output,
        help=f"Destination directory (default: {default_output})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite --output-dir if it already contains files.",
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir.expanduser().resolve()

    ensure_empty_dir(output_dir, args.force)

    with tempfile.TemporaryDirectory() as tmp:
        tmpdir = Path(tmp)
        archive_path = tmpdir / "mmlu.tar"
        print(f"Downloading MMLU from {args.url}")
        download_file(args.url, archive_path)

        extraction_dir = tmpdir / "extracted"
        extraction_dir.mkdir()
        print("Extracting archive...")
        safe_extract(archive_path, extraction_dir)

        data_root = locate_data_root(extraction_dir)
        print(f"Copying dataset contents from {data_root} to {output_dir}")
        copy_contents(data_root, output_dir)

    print(f"MMLU dataset is ready at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
