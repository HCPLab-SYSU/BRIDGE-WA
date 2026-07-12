#!/usr/bin/env python3
"""Restore small VLABench assets omitted from the current download bundle."""

from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import urllib.request
from pathlib import Path


# The upstream repository tracked both directories at this revision. Later
# revisions ignored all of VLABench/assets, while download_assets.py continued
# to fetch only the large obj and scenes archives.
ASSET_COMMIT = "d4cde3a87ddbc1e96db1b41b3165cab6dc4f3afa"
ARCHIVE_URL = f"https://github.com/OpenMOSS/VLABench/archive/{ASSET_COMMIT}.tar.gz"
ARCHIVE_ROOT = f"VLABench-{ASSET_COMMIT}/VLABench/assets/"
INCLUDED_PREFIXES = (
    "base/",
    "robots/franka_emika_panda/",
)
REQUIRED_FILES = (
    "base/default.xml",
    "base/camera.xml",
    "robots/franka_emika_panda/panda.xml",
    "robots/franka_emika_panda/assets/link0.stl",
)


def is_complete(asset_root: Path) -> bool:
    return all((asset_root / relative).is_file() for relative in REQUIRED_FILES)


def extract_selected(archive_path: Path, asset_root: Path) -> None:
    asset_root.mkdir(parents=True, exist_ok=True)
    root_resolved = asset_root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith(ARCHIVE_ROOT):
                continue
            relative = member.name.removeprefix(ARCHIVE_ROOT)
            if not any(relative.startswith(prefix) for prefix in INCLUDED_PREFIXES):
                continue

            destination = asset_root / relative
            if not destination.resolve().is_relative_to(root_resolved):
                raise RuntimeError(f"Unsafe archive member: {member.name}")
            if member.isdir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            if not member.isfile():
                continue

            source = archive.extractfile(member)
            if source is None:
                raise RuntimeError(f"Failed to read archive member: {member.name}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            with source, destination.open("wb") as output:
                shutil.copyfileobj(source, output)
            destination.chmod(member.mode & 0o777)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    args = parser.parse_args()
    asset_root = args.asset_root.resolve()

    if is_complete(asset_root):
        print(f"[vlabench-assets] core assets already present: {asset_root}")
        return 0

    print(f"[vlabench-assets] downloading official core assets from {ASSET_COMMIT}")
    with tempfile.TemporaryDirectory(prefix="bridge-wa-vlabench-assets-") as temp_dir:
        archive_path = Path(temp_dir) / "vlabench-core-assets.tar.gz"
        urllib.request.urlretrieve(ARCHIVE_URL, archive_path)
        extract_selected(archive_path, asset_root)

    missing = [relative for relative in REQUIRED_FILES if not (asset_root / relative).is_file()]
    if missing:
        raise RuntimeError(f"Core asset archive is incomplete; missing: {missing}")
    print(f"[vlabench-assets] restored base and simulated Franka assets: {asset_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
