#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from tqdm import tqdm

from datasets import create_dataloader
from datasets.dataset import _collect_meta_file_paths, _dataset_tag, _load_metas_into_dict
from scripts.train_bridge_wa import (
    _SimpleLogger,
    _episode_key_to_cache_path,
    _sample_key_to_cache_path,
    _sample_key_to_episode_cache_path,
    _sample_key_to_episode_key,
    build_teacher_future_latents,
    build_tokenizer,
    denorm_views,
    load_world_teacher,
    maybe_resize,
    maybe_get_wrist,
    _compute_optical_flow_maps_batch,
    _decode_future_images_batch,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Precompute Bridge-WA world-guidance cache")
    parser.add_argument("--world_teacher_path", type=str, required=True)
    parser.add_argument("--train_metas_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--teacher_future_steps", type=int, default=50)
    parser.add_argument("--tokenizer_path", type=str, default="google/umt5-xxl")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--num_actions", type=int, default=10)
    parser.add_argument("--action_mode", type=str, default="libero")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--flow_workers", type=int, default=0)
    parser.add_argument("--cache_granularity", type=str, default="sample", choices=("sample", "episode"))
    parser.add_argument("--vlabench_camera_order", type=str, default="")
    parser.add_argument("--vlabench_future_view", type=str, default="")
    parser.add_argument("--dobot_camera_order", type=str, default="")
    parser.add_argument("--dobot_future_view", type=str, default="")
    parser.add_argument("--dobot_action_offset", type=int, default=None)
    parser.add_argument("--dobot_ee6d_arm_slot", type=str, default="")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--sample_key_file",
        type=str,
        default="",
        help="Optional text file of sample keys to precompute, one per line.",
    )
    parser.add_argument(
        "--keep_static_frames",
        action="store_true",
        default=False,
        help="Keep near-static action windows instead of dropping them in dataset handlers that support it.",
    )
    parser.add_argument(
        "--no_pooled_future_cache",
        action="store_false",
        dest="save_pooled_future_cache",
        default=True,
        help="Do not save the legacy 2x2 future_latents_pooled tensor; keep only full-resolution cache tensors.",
    )
    parser.add_argument(
        "--pooled_future_cache_only",
        action="store_true",
        default=False,
        help="Save future_latents_pooled/change_map/flow_map only; omit full-resolution future_latents.",
    )
    return parser.parse_args()


def build_front_inputs(world_teacher, batch, device: torch.device):
    image_input = denorm_views(batch["image_input"].to(device))
    image_mask = batch.get("image_mask")
    if image_mask is not None:
        image_mask = image_mask.to(device).bool()

    front = maybe_resize(image_input[:, 0], world_teacher.config.wan_height, world_teacher.config.wan_width)
    wrist, wrist_valid_mask = maybe_get_wrist(
        image_input=image_input,
        image_mask=image_mask,
        height=world_teacher.config.wan_height,
        width=world_teacher.config.wan_width,
    )
    return front, wrist, wrist_valid_mask


def build_change_maps(front_latents: torch.Tensor, future_latents: torch.Tensor) -> torch.Tensor:
    if future_latents.dim() == 5:
        future_latents = future_latents[:, :, 0]
    delta = future_latents.float() - front_latents.float()
    change_map = torch.linalg.vector_norm(delta, dim=1, keepdim=True)
    change_map = change_map / (change_map.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    return change_map


def build_flow_maps(world_teacher, front: torch.Tensor, future_latents: torch.Tensor, flow_workers: int) -> torch.Tensor:
    future_images = _decode_future_images_batch(world_teacher, future_latents)
    target_hw = future_latents.shape[-2:]
    return _compute_optical_flow_maps_batch(front, future_images, target_hw, flow_workers=flow_workers)


def _subset_batch(batch: dict, indices: list[int]) -> dict:
    if not indices:
        return {}
    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value):
            index = torch.tensor(indices, device=value.device, dtype=torch.long)
            out[key] = value.index_select(0, index)
        elif isinstance(value, list):
            out[key] = [value[i] for i in indices]
        elif isinstance(value, tuple):
            out[key] = tuple(value[i] for i in indices)
        else:
            out[key] = value
    return out


def _hash_mod(key: str, num_shards: int) -> int:
    h = hashlib.md5(key.encode("utf-8")).digest()
    return int.from_bytes(h[:8], byteorder="big", signed=False) % num_shards


def _sample_on_shard(sample_key: str, *, cache_granularity: str, num_shards: int, shard_id: int) -> bool:
    if num_shards <= 1:
        return True
    key = _sample_key_to_episode_key(sample_key) if cache_granularity == "episode" else sample_key
    return _hash_mod(key, num_shards) == shard_id


def _flush_episode_payload(output_dir: Path, episode_buffer: dict, args, manifest_fp) -> tuple[int, int]:
    sample_keys = episode_buffer["sample_keys"]
    if not sample_keys:
        return 0, 0
    cache_path = _sample_key_to_episode_cache_path(output_dir, sample_keys[0])
    if cache_path.exists() and not args.overwrite:
        return 0, len(sample_keys)

    sample_to_index = {str(k): i for i, k in enumerate(sample_keys)}
    payload = {
        "episode_key": episode_buffer["episode_key"],
        "sample_keys": [str(k) for k in sample_keys],
        "sample_to_index": sample_to_index,
        "change_map": torch.stack(episode_buffer["change_map"], dim=0).clone(),
        "flow_map": torch.stack(episode_buffer["flow_map"], dim=0).clone(),
        "teacher_future_steps": int(args.teacher_future_steps),
        "world_teacher_path": str(args.world_teacher_path),
        "cache_version": "bridge_wa_world_guidance_v1",
        "cache_granularity": "episode",
        "save_pooled_future_cache": bool(args.save_pooled_future_cache),
        "save_full_future_cache": not bool(args.pooled_future_cache_only),
    }
    if not args.pooled_future_cache_only:
        payload["future_latents"] = torch.stack(episode_buffer["future_latents"], dim=0).clone()
    if args.save_pooled_future_cache:
        payload["future_latents_pooled"] = torch.stack(episode_buffer["future_latents_pooled"], dim=0).clone()
    torch.save(payload, str(cache_path))
    for sample_key in payload["sample_keys"]:
        meta = {
            "sample_key": sample_key,
            "episode_key": payload["episode_key"],
            "teacher_future_steps": payload["teacher_future_steps"],
            "world_teacher_path": payload["world_teacher_path"],
            "cache_version": payload["cache_version"],
            "cache_granularity": payload["cache_granularity"],
            "save_pooled_future_cache": payload["save_pooled_future_cache"],
        }
        manifest_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
    return len(sample_keys), 0


def _new_episode_buffer() -> dict:
    return {
        "episode_key": None,
        "sample_keys": [],
        "future_latents": [],
        "future_latents_pooled": [],
        "change_map": [],
        "flow_map": [],
    }


def _update_progress(progress: tqdm, delta: int) -> None:
    if delta <= 0:
        return
    if progress.total is None:
        progress.update(delta)
        return
    remaining = max(0, int(progress.total - progress.n))
    progress.update(min(delta, remaining))


def _count_episode_caches(
    metas_path: str,
    output_dir: Path,
    overwrite: bool,
    *,
    num_shards: int,
    shard_id: int,
) -> tuple[int, int]:
    metas: dict[str, dict] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        _load_metas_into_dict(_collect_meta_file_paths(metas_path), metas)

    total_episodes = 0
    completed_episodes = 0
    for dataset_name, meta in metas.items():
        datalist = meta.get("datalist", [])
        dataset_tag = _dataset_tag(dataset_name)
        for traj_idx, item in enumerate(datalist):
            if isinstance(item, dict) and "episode_index" in item:
                episode_index = int(item["episode_index"])
            else:
                episode_index = int(traj_idx)
            episode_key = f"{dataset_tag}_ep{episode_index:06d}"
            if not _sample_on_shard(
                f"{episode_key}_frame000000",
                cache_granularity="episode",
                num_shards=num_shards,
                shard_id=shard_id,
            ):
                continue
            total_episodes += 1
            if overwrite:
                continue
            if _episode_key_to_cache_path(output_dir, episode_key).exists():
                completed_episodes += 1
    return total_episodes, completed_episodes


def _estimate_sample_caches(
    metas_path: str,
    output_dir: Path,
    num_actions: int,
    overwrite: bool,
    *,
    num_shards: int,
) -> tuple[int | None, int]:
    if num_shards > 1:
        return None, 0
    metas: dict[str, dict] = {}
    with contextlib.redirect_stdout(io.StringIO()):
        _load_metas_into_dict(_collect_meta_file_paths(metas_path), metas)

    estimated_total = 0
    for meta in metas.values():
        total_frames = meta.get("total_frames")
        total_episodes = meta.get("total_episodes")
        if total_frames is None or total_episodes is None:
            return None, 0
        estimated_total += max(0, int(total_frames) - int(total_episodes) * int(num_actions))

    if overwrite:
        return estimated_total, 0

    existing_samples = 0
    for path in output_dir.glob("*.pt"):
        if "_frame" in path.stem:
            existing_samples += 1
    return estimated_total, min(existing_samples, estimated_total)


def _set_progress_postfix(
    progress: tqdm,
    *,
    progress_unit: str,
    saved_samples: int,
    skipped_samples: int,
    saved_episodes: int,
) -> None:
    if progress_unit == "ep":
        progress.set_postfix(
            done_ep=int(progress.n),
            saved_ep=saved_episodes,
            sample_saved=saved_samples,
            sample_skipped=skipped_samples,
        )
        return
    progress.set_postfix(saved=saved_samples, skipped=skipped_samples)


def _load_sample_key_filter(sample_key_file: str) -> set[str] | None:
    if not sample_key_file:
        return None
    path = Path(sample_key_file)
    keys = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    return keys


def main() -> None:
    args = parse_args()
    if args.pooled_future_cache_only:
        args.save_pooled_future_cache = True
    if args.num_shards < 1:
        raise ValueError("--num_shards must be >= 1")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError("--shard_id must satisfy 0 <= shard_id < num_shards")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = _SimpleLogger(None)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    world_teacher = load_world_teacher(args.world_teacher_path, device, logger)
    world_teacher_tokenizer = build_tokenizer(args.tokenizer_path)
    num_workers = int(args.num_workers)
    sample_key_filter = _load_sample_key_filter(args.sample_key_file)
    handler_kwargs = {}
    if args.keep_static_frames:
        handler_kwargs["filter_static_frames"] = False
    if args.vlabench_camera_order:
        handler_kwargs["vlabench_camera_order"] = args.vlabench_camera_order
    if args.vlabench_future_view:
        handler_kwargs["vlabench_future_view"] = args.vlabench_future_view
    if args.dobot_camera_order:
        handler_kwargs["dobot_camera_order"] = args.dobot_camera_order
    if args.dobot_future_view:
        handler_kwargs["dobot_future_view"] = args.dobot_future_view
    if args.dobot_action_offset is not None:
        handler_kwargs["dobot_action_offset"] = int(args.dobot_action_offset)
    if args.dobot_ee6d_arm_slot:
        handler_kwargs["dobot_ee6d_arm_slot"] = args.dobot_ee6d_arm_slot

    # Silence dataset construction prints so the terminal stays readable for the main tqdm bar.
    with contextlib.redirect_stdout(io.StringIO()):
        dataloader = create_dataloader(
            batch_size=args.batch_size,
            metas_path=args.train_metas_path,
            num_actions=args.num_actions,
            training=False,
            action_mode=args.action_mode,
            num_workers=num_workers,
            include_producer_id=(args.cache_granularity == "episode"),
            future_index=0,
            return_future=False,
            infinite=False,
            image_height=world_teacher.config.wan_height,
            image_width=world_teacher.config.wan_width,
            handler_kwargs=handler_kwargs or None,
        )

    saved = 0
    skipped = 0
    saved_episodes = 0
    manifest_path = output_dir / "manifest.jsonl"
    manifest_fp = manifest_path.open("a", encoding="utf-8")
    episode_buffers: dict[int, dict] = {}
    stop_requested = False

    progress_total = int(args.limit) if args.limit > 0 else None
    progress_unit = "sample"
    progress_initial = 0
    if sample_key_filter is not None and args.cache_granularity != "sample":
        raise ValueError("--sample_key_file currently supports only --cache_granularity sample")
    if sample_key_filter is not None and args.limit <= 0:
        progress_total = len(sample_key_filter)
    if args.limit <= 0:
        if sample_key_filter is not None:
            progress_initial = 0
            if not args.overwrite:
                for key in sample_key_filter:
                    if _sample_key_to_cache_path(output_dir, str(key)).exists():
                        progress_initial += 1
        elif args.cache_granularity == "episode":
            progress_total, progress_initial = _count_episode_caches(
                args.train_metas_path,
                output_dir,
                overwrite=bool(args.overwrite),
                num_shards=int(args.num_shards),
                shard_id=int(args.shard_id),
            )
            progress_unit = "ep"
        else:
            progress_total, progress_initial = _estimate_sample_caches(
                args.train_metas_path,
                output_dir,
                num_actions=int(args.num_actions),
                overwrite=bool(args.overwrite),
                num_shards=int(args.num_shards),
            )
    progress = tqdm(
        total=progress_total,
        initial=progress_initial,
        desc=f"precompute_all_cache[{args.shard_id}/{args.num_shards}]",
        unit=progress_unit,
        dynamic_ncols=True,
    )
    with torch.no_grad():
        for batch in dataloader:
            batch_saved_before = saved
            batch_skipped_before = skipped
            batch_saved_episodes_before = saved_episodes
            sample_keys = batch.get("sample_key")
            if sample_keys is None:
                raise KeyError("sample_key missing from batch; update dataset handler first.")

            active_indices = [
                i for i, key in enumerate(sample_keys)
                if _sample_on_shard(
                    str(key),
                    cache_granularity=str(args.cache_granularity),
                    num_shards=int(args.num_shards),
                    shard_id=int(args.shard_id),
                )
            ]
            if sample_key_filter is not None:
                active_indices = [i for i in active_indices if str(sample_keys[i]) in sample_key_filter]
            if not active_indices:
                continue
            if len(active_indices) != len(sample_keys):
                batch = _subset_batch(batch, active_indices)
                sample_keys = batch["sample_key"]

            if args.cache_granularity == "sample":
                cache_paths = [_sample_key_to_cache_path(output_dir, str(k)) for k in sample_keys]
            else:
                cache_paths = [_sample_key_to_episode_cache_path(output_dir, str(k)) for k in sample_keys]
            if args.overwrite:
                active_indices = list(range(len(sample_keys)))
            else:
                active_indices = [i for i, path in enumerate(cache_paths) if not path.exists()]
                skipped += len(cache_paths) - len(active_indices)
            if not active_indices:
                if progress_unit == "sample":
                    _update_progress(progress, skipped - batch_skipped_before)
                _set_progress_postfix(
                    progress,
                    progress_unit=progress_unit,
                    saved_samples=saved,
                    skipped_samples=skipped,
                    saved_episodes=saved_episodes,
                )
                if args.limit and (saved + skipped) >= args.limit:
                    break
                continue

            cache_paths = [cache_paths[i] for i in active_indices]
            batch = _subset_batch(batch, active_indices)
            sample_keys = batch["sample_key"]
            producer_ids = batch.get("producer_id")
            if producer_ids is None:
                producer_ids = [0] * len(sample_keys)
            front, _, _ = build_front_inputs(world_teacher, batch, device)
            future_latents, front_latents = build_teacher_future_latents(
                world_teacher,
                world_teacher_tokenizer,
                batch,
                device,
                args.teacher_future_steps,
                return_front_latents=True,
            )
            if future_latents.dim() == 5:
                future_latents_4d = future_latents[:, :, 0]
            else:
                future_latents_4d = future_latents

            future_latents_cache = None
            if not args.pooled_future_cache_only:
                future_latents_cache = future_latents_4d.detach().cpu().to(torch.float16)
            future_latents_pooled = None
            if args.save_pooled_future_cache:
                future_latents_pooled = F.adaptive_avg_pool2d(future_latents_4d, (2, 2)).detach().cpu().to(torch.float16)
            change_map = build_change_maps(front_latents, future_latents_4d).detach().cpu().to(torch.float16)
            flow_map = build_flow_maps(
                world_teacher,
                front,
                future_latents_4d,
                flow_workers=int(args.flow_workers),
            ).detach().cpu().to(torch.float16)

            for i, sample_key in enumerate(sample_keys):
                if args.cache_granularity == "sample":
                    cache_path = cache_paths[i]
                    if cache_path.exists() and not args.overwrite:
                        skipped += 1
                        continue
                    payload = {
                        "sample_key": str(sample_key),
                        "change_map": change_map[i].clone(),
                        "flow_map": flow_map[i].clone(),
                        "teacher_future_steps": int(args.teacher_future_steps),
                        "world_teacher_path": str(args.world_teacher_path),
                        "cache_version": "bridge_wa_world_guidance_v1",
                        "cache_granularity": "sample",
                        "save_pooled_future_cache": bool(args.save_pooled_future_cache),
                        "save_full_future_cache": not bool(args.pooled_future_cache_only),
                    }
                    if not args.pooled_future_cache_only:
                        payload["future_latents"] = future_latents_cache[i].clone()
                    if args.save_pooled_future_cache:
                        payload["future_latents_pooled"] = future_latents_pooled[i].clone()
                    torch.save(payload, str(cache_path))
                    meta = {
                        "sample_key": payload["sample_key"],
                        "teacher_future_steps": payload["teacher_future_steps"],
                        "world_teacher_path": payload["world_teacher_path"],
                        "cache_version": payload["cache_version"],
                        "cache_granularity": payload["cache_granularity"],
                        "save_pooled_future_cache": payload["save_pooled_future_cache"],
                    }
                    manifest_fp.write(json.dumps(meta, ensure_ascii=False) + "\n")
                    saved += 1
                    if args.limit and saved >= args.limit:
                        stop_requested = True
                        break
                else:
                    episode_key = _sample_key_to_episode_key(str(sample_key))
                    producer_id = int(producer_ids[i])
                    episode_buffer = episode_buffers.setdefault(producer_id, _new_episode_buffer())
                    if episode_buffer["episode_key"] is None:
                        episode_buffer["episode_key"] = episode_key
                    elif episode_key != episode_buffer["episode_key"]:
                        saved_delta, skipped_delta = _flush_episode_payload(output_dir, episode_buffer, args, manifest_fp)
                        saved += saved_delta
                        skipped += skipped_delta
                        if saved_delta > 0:
                            saved_episodes += 1
                        episode_buffer = _new_episode_buffer()
                        episode_buffer["episode_key"] = episode_key
                        episode_buffers[producer_id] = episode_buffer
                        if args.limit and saved >= args.limit:
                            stop_requested = True
                            break
                    episode_buffer["sample_keys"].append(str(sample_key))
                    if not args.pooled_future_cache_only:
                        episode_buffer["future_latents"].append(future_latents_cache[i].clone())
                    if args.save_pooled_future_cache:
                        episode_buffer["future_latents_pooled"].append(future_latents_pooled[i].clone())
                    episode_buffer["change_map"].append(change_map[i].clone())
                    episode_buffer["flow_map"].append(flow_map[i].clone())
            if progress_unit == "sample":
                batch_delta = (saved - batch_saved_before) + (skipped - batch_skipped_before)
                _update_progress(progress, batch_delta)
            else:
                batch_delta = saved_episodes - batch_saved_episodes_before
                _update_progress(progress, batch_delta)
            manifest_fp.flush()
            _set_progress_postfix(
                progress,
                progress_unit=progress_unit,
                saved_samples=saved,
                skipped_samples=skipped,
                saved_episodes=saved_episodes,
            )
            if stop_requested:
                break

        if args.cache_granularity == "episode":
            for episode_buffer in episode_buffers.values():
                if not episode_buffer["sample_keys"]:
                    continue
                batch_saved_before = saved
                batch_skipped_before = skipped
                batch_saved_episodes_before = saved_episodes
                saved_delta, skipped_delta = _flush_episode_payload(output_dir, episode_buffer, args, manifest_fp)
                saved += saved_delta
                skipped += skipped_delta
                if saved_delta > 0:
                    saved_episodes += 1
                if progress_unit == "sample":
                    batch_delta = (saved - batch_saved_before) + (skipped - batch_skipped_before)
                else:
                    batch_delta = saved_episodes - batch_saved_episodes_before
                _update_progress(progress, batch_delta)
                _set_progress_postfix(
                    progress,
                    progress_unit=progress_unit,
                    saved_samples=saved,
                    skipped_samples=skipped,
                    saved_episodes=saved_episodes,
                )

    manifest_fp.close()
    if progress_unit == "sample" and args.limit <= 0 and progress.total is not None and progress.n < progress.total:
        progress.total = progress.n
        progress.refresh()
    progress.close()
    if progress_unit == "ep":
        print(f"Saved={saved} skipped={skipped} saved_ep={saved_episodes} cache_dir={output_dir}")
    else:
        print(f"Saved={saved} skipped={skipped} cache_dir={output_dir}")


if __name__ == "__main__":
    main()
