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
from PIL import Image

from ..utils import read_video_to_frames, read_parquet, decode_image_from_bytes, read_bytes
from .base import DomainHandler


class LeRobotDroidHandler(DomainHandler):
    """
    LeRobot-style DROID dataset for WorldTeacher.

    Expected features (aligned with WorldTeacher libero-style columns):
      - image / wrist_image
      - state: [T, 8] = xyz+rot(6) + gripper(1) + pad(1)
      - actions: [T, 7] (or [T, 8]) = cartesian target + gripper
      - optional wrist_valid: [T]

    Training target is anchor-delta cartesian action via idx_for_delta=[0..5].
    """

    CAMERA_VIEW = ["image", "wrist_image"]

    @staticmethod
    def _column_to_array(data: dict, keys: list[str], dtype=np.float32) -> np.ndarray:
        for key in keys:
            if key not in data:
                continue
            arr = np.asarray(data[key], dtype=dtype)
            if arr.ndim == 1:
                arr = np.stack(arr, axis=0)
            return arr
        raise KeyError(f"Missing columns. Tried keys: {keys}")

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

        actions = self._column_to_array(data, ["actions", "action"], dtype=np.float32)
        state = self._column_to_array(data, ["state", "observation.state"], dtype=np.float32)

        length = min(actions.shape[0], state.shape[0], *[v.shape[0] for v in videos])
        actions = actions[:length]
        state = state[:length]
        videos = [v[:length] for v in videos]

        # Keep mixed-dataset shape stable: use at least 8 dims (libero-compatible).
        action_dim = actions.shape[1]
        proprio_dim = state.shape[1]
        target_dim = max(8, action_dim, proprio_dim)
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
                    frame = np.zeros_like(videos[0][idx])
                else:
                    frame = videos[v][idx]
                imgs.append(image_aug(Image.fromarray(frame)))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            state_cur = torch.tensor(state[idx], dtype=torch.float32).view(1, -1)
            action_seq = torch.tensor(actions[idx + 1 : idx + 1 + num_actions], dtype=torch.float32)
            if (action_seq[0] - action_seq[-1]).abs().max() < 1e-5:
                continue

            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            sample = {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([state_cur, action_seq], dim=0).float(),
                # Convert cartesian xyz+rot(6) to anchor-delta on the fly.
                "idx_for_delta": [0, 1, 2, 3, 4, 5],
                # Rotation (axis-angle) uses proper relative composition, not direct subtraction.
                "idx_for_rot_delta": [3, 4, 5],
                # Unify mixed-source gripper semantics with hysteresis around 0.
                "idx_for_gripper": [6],
                "gripper_deadzone": 0.05,
                "gripper_to_signed": True,
            }
            if return_future and future_index > 0:
                future_img = image_aug(Image.fromarray(videos[0][idx + future_index]))
                sample["future_image"] = future_img
            yield sample
