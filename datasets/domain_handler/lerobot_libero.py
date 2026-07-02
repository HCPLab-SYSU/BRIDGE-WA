# ------------------------------------------------------------------------------
# Copyright 2025 2toINF (https://github.com/2toINF)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------------

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from mmengine import fileio
from PIL import Image

from ..utils import read_video_to_frames, read_parquet, decode_image_from_bytes, read_bytes, rotvec_to_rotate6d
from .base import DomainHandler


class LeRobotLiberoHandler(DomainHandler):
    """
    LeRobot v2.1 Libero dataset (openpi convert script).

    Expected features:
      - image / wrist_image videos
      - actions: [T,7] (delta ee + gripper, from Libero)
      - state: [T,8] (robot state, e.g. joints)
      - tasks: list[str] in episodes.jsonl
    """

    CAMERA_VIEW = ["image", "wrist_image"]

    @staticmethod
    def _load_abs_cache(cache_dir: str | None, *, episode_chunk: int, episode_index: int) -> dict | None:
        if not cache_dir:
            return None
        cache_path = Path(cache_dir) / f"chunk-{episode_chunk:03d}" / f"episode_{episode_index:06d}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing Libero abs cache: {cache_path}")
        payload = torch.load(cache_path, map_location="cpu")
        if not isinstance(payload, dict):
            raise TypeError(f"Unsupported Libero abs cache payload type {type(payload)} at {cache_path}")
        return payload

    def iter_episode(
        self,
        traj_idx: int,
        *,
        num_actions: int,
        training: bool,
        image_aug,
        lang_aug_map: dict | None,
        **kwargs,
    ) -> Iterable[dict]:
        future_index = int(kwargs.pop("future_index", 0) or 0)
        return_future = bool(kwargs.pop("return_future", False))
        action_mode = str(kwargs.get("action_mode", "libero")).lower()
        use_libero_space = action_mode in ("libero", "bridge_libero", "libero_abs", "bridge_libero_abs")
        use_ee6d20_space = action_mode in ("ee6d", "ee6d_abs", "libero_ee6d", "libero_ee6d20", "libero_abs_ee6d", "libero_abs_ee6d20")
        use_ee6d10_space = action_mode in ("ee6d10", "ee6d_10", "ee6d10_abs", "libero_ee6d10", "libero_ee6d_10", "libero_abs_ee6d10", "libero_abs_ee6d_10")
        use_abs_cache = action_mode in (
            "libero_abs",
            "bridge_libero_abs",
            "ee6d_abs",
            "ee6d10_abs",
            "libero_abs_ee6d",
            "libero_abs_ee6d20",
            "libero_abs_ee6d10",
            "libero_abs_ee6d_10",
        )
        libero_abs_cache_dir = str(kwargs.get("libero_abs_cache_dir") or os.environ.get("LIBERO_ABS_CACHE_DIR", "")).strip()
        filter_static_frames = bool(kwargs.get("filter_static_frames", True))

        item = self.meta["datalist"][traj_idx]
        episode_index = int(item["episode_index"])
        episode_chunk = episode_index // int(self.meta["chunks_size"])
        data_path = fileio.join_path(self.meta["root_path"], self.meta["data_path"]).format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        video_paths = [
            fileio.join_path(self.meta["root_path"], self.meta["video_path"]).format(
                episode_chunk=episode_chunk, episode_index=episode_index, video_key=vkey
            )
            for vkey in self.CAMERA_VIEW
        ]

        data = read_parquet(data_path)
        use_video = all(os.path.exists(p) for p in video_paths)
        if use_video:
            videos = [read_video_to_frames(p) for p in video_paths]
        else:
            def _decode_image_item(item):
                if isinstance(item, dict):
                    if item.get("bytes") is not None:
                        return decode_image_from_bytes(item["bytes"])
                    if item.get("path"):
                        return decode_image_from_bytes(read_bytes(item["path"]))
                if isinstance(item, (bytes, bytearray)):
                    return decode_image_from_bytes(item)
                if isinstance(item, np.ndarray):
                    return Image.fromarray(item)
                return Image.fromarray(np.asarray(item))

            videos = []
            for key in self.CAMERA_VIEW:
                if key not in data:
                    videos.append(np.zeros((0, 256, 256, 3), dtype=np.uint8))
                    continue
                frames = [_decode_image_item(x) for x in data[key]]
                videos.append(np.stack([np.array(f) for f in frames], axis=0))

        actions = np.asarray(data["actions"], dtype=np.float32)
        state = np.asarray(data["state"], dtype=np.float32)
        if actions.ndim == 1:
            actions = np.stack(actions, axis=0)
        if state.ndim == 1:
            state = np.stack(state, axis=0)

        length = min(actions.shape[0], state.shape[0], *[v.shape[0] for v in videos])
        actions = actions[:length]
        state = state[:length]
        videos = [v[:length] for v in videos]

        def _fit_last_dim(x: np.ndarray, target_dim: int) -> np.ndarray:
            cur_dim = int(x.shape[-1])
            if cur_dim == target_dim:
                return x
            if cur_dim < target_dim:
                pad_shape = list(x.shape)
                pad_shape[-1] = target_dim - cur_dim
                pad = np.zeros(pad_shape, dtype=x.dtype)
                return np.concatenate([x, pad], axis=-1)
            return x[..., :target_dim]

        raw_actions = actions[:, :7]
        raw_state = state[:, :7]
        raw_state_libero = state[:, :8]

        if use_abs_cache:
            payload = self._load_abs_cache(libero_abs_cache_dir, episode_chunk=episode_chunk, episode_index=episode_index)
            if use_ee6d10_space:
                actions = np.asarray(payload["action_abs_ee6d10"], dtype=np.float32)
                state = np.asarray(payload["state_abs_ee6d10"], dtype=np.float32)
            elif use_ee6d20_space:
                actions = np.asarray(payload["action_abs_ee6d20"], dtype=np.float32)
                state = np.asarray(payload["state_abs_ee6d20"], dtype=np.float32)
            elif use_libero_space:
                actions = np.asarray(payload["action_abs_axis_angle"], dtype=np.float32)
                state = np.asarray(payload["state_abs_axis_angle"], dtype=np.float32)
            else:
                raise ValueError(f"Unsupported Libero abs action_mode: {action_mode}")
            actions = actions[:length]
            state = state[:length]
        else:
            actions = raw_actions
            if use_ee6d20_space or use_ee6d10_space:
                state = raw_state
                pos = actions[:, :3]
                rotvec = actions[:, 3:6]
                grip = actions[:, 6:7]
                state_pos = state[:, :3]
                state_rotvec = state[:, 3:6]
                state_grip = state[:, 6:7]
                rot6d = rotvec_to_rotate6d(rotvec)
                state_rot6d = rotvec_to_rotate6d(state_rotvec)
                left = np.concatenate([pos, rot6d, grip], axis=-1)
                state_left = np.concatenate([state_pos, state_rot6d, state_grip], axis=-1)
                if use_ee6d10_space:
                    actions = left
                    state = state_left
                else:
                    right = np.zeros_like(left)
                    state_right = np.zeros_like(state_left)
                    actions = np.concatenate([left, right], axis=-1)
                    state = np.concatenate([state_left, state_right], axis=-1)
            else:
                # Libero source actions are 7D, but LiberoActionSpace is model-facing 8D.
                # Keep the dataset's 8th state channel when available and pad actions below.
                state = raw_state_libero

        if use_libero_space:
            actions = _fit_last_dim(actions, 8)
            state = _fit_last_dim(state, 8)
        elif not (use_ee6d20_space or use_ee6d10_space or use_abs_cache):
            action_dim = actions.shape[1]
            proprio_dim = state.shape[1]
            target_dim = max(action_dim, proprio_dim)
            if action_dim < target_dim:
                pad = np.zeros((actions.shape[0], target_dim - action_dim), dtype=actions.dtype)
                actions = np.concatenate([actions, pad], axis=-1)
            elif action_dim > target_dim:
                actions = actions[:, :target_dim]
            if proprio_dim < target_dim:
                pad = np.zeros((state.shape[0], target_dim - proprio_dim), dtype=state.dtype)
                state = np.concatenate([state, pad], axis=-1)
            elif proprio_dim > target_dim:
                state = state[:, :target_dim]

        base_image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        base_image_mask[: min(self.num_views, len(videos))] = True

        wrist_valid = np.ones(length, dtype=np.bool_)
        if "wrist_valid" in data:
            wrist_valid_raw = np.asarray(data["wrist_valid"])
            if wrist_valid_raw.ndim > 1:
                wrist_valid_raw = wrist_valid_raw.reshape(wrist_valid_raw.shape[0], -1)[:, 0]
            wrist_valid_raw = wrist_valid_raw.reshape(-1)
            valid_len = min(length, wrist_valid_raw.shape[0])
            if valid_len > 0:
                wrist_valid[:valid_len] = wrist_valid_raw[:valid_len].astype(np.float32) > 0.5

        idxs = list(range(0, max(0, len(videos[0]) - 1)))
        if training:
            random.shuffle(idxs)

        ins = item["tasks"][0] if item.get("tasks") else ""
        for idx in idxs:
            # Use only original integer-frame slicing (no interpolation).
            if idx + num_actions >= length:
                continue
            if return_future and future_index > 0 and idx + future_index >= videos[0].shape[0]:
                continue

            image_mask = base_image_mask.clone()
            if image_mask.shape[0] > 1 and not bool(wrist_valid[idx]):
                image_mask[1] = False

            imgs = []
            for v in range(min(self.num_views, len(videos))):
                if v == 1 and not bool(image_mask[1]):
                    # Keep tensor shape stable while explicitly masking out invalid wrist view.
                    frame = np.zeros_like(videos[0][idx])
                else:
                    frame = videos[v][idx]
                imgs.append(image_aug(Image.fromarray(frame)))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            state_cur = torch.tensor(state[idx], dtype=torch.float32).view(1, -1)
            action_seq = torch.tensor(actions[idx + 1 : idx + 1 + num_actions], dtype=torch.float32)
            if filter_static_frames and (action_seq[0] - action_seq[-1]).abs().max() < 1e-5:
                continue

            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            dataset_tag = os.path.basename(str(self.meta.get("root_path", "libero")).rstrip("/")) or "libero"
            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([state_cur, action_seq], dim=0).float(),
                "sample_key": f"{dataset_tag}_ep{episode_index:06d}_frame{idx:06d}",
                "episode_index": int(episode_index),
                "frame_index": int(idx),
            }
            if use_libero_space:
                sample["idx_for_gripper"] = [6]
                sample["gripper_deadzone"] = 0.05
                sample["gripper_to_signed"] = True
            elif use_ee6d10_space:
                sample["idx_for_gripper"] = [9]
            elif use_ee6d20_space:
                sample["idx_for_gripper"] = [9, 19]
            if return_future and future_index > 0:
                future_img = image_aug(Image.fromarray(videos[0][idx + future_index]))
                sample["future_image"] = future_img
            yield sample
