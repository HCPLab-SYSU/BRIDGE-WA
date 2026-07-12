#!/usr/bin/env python3
"""Validate the Python environment used by the Bridge-WA VLABench client."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path


os.environ.setdefault("MUJOCO_GL", "egl")

MODULES = {
    "colorama": "colorama",
    "colorlog": "colorlog",
    "cv2": "opencv-python",
    "dm_control": "dm-control",
    "gdown": "gdown",
    "json_numpy": "json-numpy",
    "mediapy": "mediapy",
    "mujoco": "mujoco",
    "networkx": "networkx",
    "numpy": "numpy",
    "open3d": "open3d",
    "openai": "openai",
    "PIL": "pillow",
    "requests": "requests",
    "scipy": "scipy",
    "sklearn": "scikit-learn",
    "tqdm": "tqdm",
    "yaml": "PyYAML",
}

ASSET_FILES = (
    "base/default.xml",
    "base/camera.xml",
    "obj/meshes/table/table.xml",
    "robots/franka_emika_panda/panda.xml",
    "scenes/default/empty.xml",
)


def main() -> int:
    errors: list[tuple[str, str, BaseException]] = []
    for module, distribution in MODULES.items():
        try:
            importlib.import_module(module)
        except BaseException as exc:  # Binary-extension load errors matter too.
            errors.append((module, distribution, exc))

    if not errors:
        try:
            importlib.import_module("vlabench_client")
        except BaseException as exc:
            errors.append(("vlabench_client", "VLABench editable install", exc))

    asset_root = Path(__file__).with_name("VLABench") / "VLABench" / "assets"
    missing_assets = [relative for relative in ASSET_FILES if not (asset_root / relative).is_file()]

    if errors or missing_assets:
        print("[vlabench-env-check] environment validation failed:", file=sys.stderr)
        for module, distribution, exc in errors:
            print(
                f"  - import {module!r} ({distribution}) failed: "
                f"{type(exc).__name__}: {exc}",
                file=sys.stderr,
            )
        for relative in missing_assets:
            print(f"  - missing simulator asset: {asset_root / relative}", file=sys.stderr)
        if errors:
            requirements = Path(__file__).with_name("requirements-eval.txt")
            vendored = Path(__file__).with_name("VLABench")
            print("[vlabench-env-check] repair dependencies with:", file=sys.stderr)
            print(
                f"  {sys.executable} -m pip install -r {requirements}",
                file=sys.stderr,
            )
            print(
                f"  {sys.executable} -m pip install --no-deps -e {vendored}",
                file=sys.stderr,
            )
        if missing_assets:
            downloader = Path(__file__).parents[1] / ".." / "scripts" / "download_vlabench_assets.sh"
            print("[vlabench-env-check] repair assets with:", file=sys.stderr)
            print(f"  bash {downloader.resolve()}", file=sys.stderr)
        return 1

    print(f"[vlabench-env-check] ok: python={sys.executable}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
