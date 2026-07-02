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
from typing import Iterable

import numpy as np
import torch
from mmengine import fileio
from scipy.spatial.transform import Rotation as R
from PIL import Image

from ..utils import read_video_to_frames, read_parquet, quat_to_rotate6d, decode_image_from_bytes, read_bytes
from .base import DomainHandler


class LeRobotRLBenchHandler(DomainHandler):
    """
    LeRobot v2.1 RLBench dataset (openpi convert script).

    Expected features:
      - image / wrist_image / left_shoulder_image videos
      - actions: [T,9] = xyz(3) + quat(xyzw)(4) + gripper(1) + ignore_collision(1)
      - tasks: list[str] in episodes.jsonl
    """

    CAMERA_VIEW = ["image", "wrist_image"]

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
        action_mode = str(kwargs.get("action_mode", "ee6d")).lower()
        single_arm = action_mode in ("ee6d10", "ee6d_10", "rlbench10", "rlbench_10")
        use_libero_space = action_mode in ("libero", "bridge_libero", "libero_abs", "bridge_libero_abs")
        apply_libero_delta = action_mode in ("libero", "bridge_libero")
        libero_delta_idx = [0, 1, 2, 3, 4, 5] if apply_libero_delta else []

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
        if actions.ndim == 1:
            actions = np.stack(actions, axis=0)

        length = min(actions.shape[0], *[v.shape[0] for v in videos])
        actions = actions[:length]
        videos = [v[:length] for v in videos]

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

        # Pad the first action to align with image index
        actions = np.concatenate([actions[:1], actions], axis=0)

        pos = actions[:, :3]
        quat = actions[:, 3:7]
        grip = actions[:, 7:8]
        if use_libero_space:
            # Bridge/Libero-style unified action: xyz + rotvec + gripper (+1 zero pad) => 8D.
            rotvec = R.from_quat(quat).as_rotvec().astype(np.float32)
            left = np.concatenate([pos, rotvec, grip], axis=-1)
            pad = np.zeros((left.shape[0], 1), dtype=np.float32)
            abs_traj = np.concatenate([left, pad], axis=-1)
        else:
            rot6d = quat_to_rotate6d(quat, scalar_first=False)
            left = np.concatenate([pos, rot6d, grip], axis=-1)
            if single_arm:
                abs_traj = left
            else:
                right = np.zeros_like(left)
                abs_traj = np.concatenate([left, right], axis=-1)

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

            cur_action = torch.tensor(abs_traj[idx : idx + num_actions + 1], dtype=torch.float32)
            if (cur_action[1] - cur_action[0]).abs().max() < 1e-5:
                continue

            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": cur_action.float(),
            }
            if libero_delta_idx:
                # Align RLBench absolute pose targets to Bridge/Libero-style delta xyz+rotvec targets.
                sample["idx_for_delta"] = libero_delta_idx
                sample["idx_for_rot_delta"] = [3, 4, 5]
            if use_libero_space:
                # Unify mixed-source gripper semantics with hysteresis around 0.
                sample["idx_for_gripper"] = [6]
                sample["gripper_deadzone"] = 0.05
                sample["gripper_to_signed"] = True
            if return_future and future_index > 0:
                future_img = image_aug(Image.fromarray(videos[0][idx + future_index]))
                sample["future_image"] = future_img
            yield sample
