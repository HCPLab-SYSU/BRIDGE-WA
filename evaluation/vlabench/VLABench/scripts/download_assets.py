import argparse
import os
import shutil
import tarfile
import tempfile
import urllib.request
import zipfile
from pathlib import Path

import gdown
import VLABench

asset_url = "https://drive.google.com/file/d/1ldEMZua2OzXHJTYTCP0IGVU1aFYBCMu-/view?usp=sharing"
scene_url = "https://drive.google.com/file/d/1KdReRkibJClBHHD32jz_wTkaBzhEJ9Kw/view?usp=drive_link"

core_asset_commit = "d4cde3a87ddbc1e96db1b41b3165cab6dc4f3afa"
core_asset_url = f"https://github.com/OpenMOSS/VLABench/archive/{core_asset_commit}.tar.gz"
core_archive_root = f"VLABench-{core_asset_commit}/VLABench/assets/"
core_asset_prefixes = (
    "base/",
    "robots/franka_emika_panda/",
)
required_core_assets = (
    "base/default.xml",
    "base/camera.xml",
    "robots/franka_emika_panda/panda.xml",
    "robots/franka_emika_panda/assets/link0.stl",
)

asset_id = asset_url.split("/d/")[1].split("/")[0]
scene_id = scene_url.split("/d/")[1].split("/")[0]

def download_assets():
    target_path = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")
    zip_path = os.path.join(target_path, "obj.zip")
    gdown.download(f"https://drive.google.com/uc?id={asset_id}", zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    os.remove(zip_path)
    print(f"Asset data has been downloaded, extracted to {target_path}, and the zip file has been deleted.")
    
def download_scene():
    target_path = os.path.join(os.getenv("VLABENCH_ROOT"), "assets")
    zip_path = os.path.join(target_path, "scene.zip")
    gdown.download(f"https://drive.google.com/uc?id={scene_id}", zip_path, quiet=False)
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(target_path)
    os.remove(zip_path)
    print(f"Scene data has been downloaded, extracted to {target_path}, and the zip file has been deleted.")


def core_assets_complete(asset_root):
    return all((asset_root / relative).is_file() for relative in required_core_assets)


def extract_core_assets(archive_path, asset_root):
    asset_root.mkdir(parents=True, exist_ok=True)
    root_resolved = asset_root.resolve()
    with tarfile.open(archive_path, "r:gz") as archive:
        for member in archive.getmembers():
            if not member.name.startswith(core_archive_root):
                continue
            relative = member.name.removeprefix(core_archive_root)
            if not any(relative.startswith(prefix) for prefix in core_asset_prefixes):
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


def download_core_assets():
    asset_root = Path(os.environ["VLABENCH_ROOT"]) / "assets"
    if core_assets_complete(asset_root):
        print(f"Core simulator assets are already present: {asset_root}")
        return

    print(f"Downloading official core simulator assets from {core_asset_commit}")
    with tempfile.TemporaryDirectory(prefix="vlabench-assets-") as temp_dir:
        archive_path = Path(temp_dir) / "vlabench-core-assets.tar.gz"
        urllib.request.urlretrieve(core_asset_url, archive_path)
        extract_core_assets(archive_path, asset_root)

    missing = [
        relative for relative in required_core_assets
        if not (asset_root / relative).is_file()
    ]
    if missing:
        raise RuntimeError(f"Core asset archive is incomplete; missing: {missing}")
    print(f"Core simulator assets have been restored to {asset_root}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Download assets and scene data.')
    parser.add_argument(
        '--choice',
        default="all",
        choices=['all', 'asset', 'scene', 'core'],
        help='Download assets',
    )
    args = parser.parse_args()
    
    if args.choice == "asset":
        download_assets()
    elif args.choice == "scene":
        download_scene()
    elif args.choice == "core":
        download_core_assets()
    elif args.choice == "all":
        download_assets()
        download_scene()
        download_core_assets()
    else:
        raise ValueError("Invalid choice")
