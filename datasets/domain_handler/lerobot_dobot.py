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

from ..utils import decode_image_from_bytes, euler_to_rotate6d, read_bytes, read_parquet, read_video_to_frames
from .base import DomainHandler


class LeRobotDobotNova2Handler(DomainHandler):
    """
    Dobot Nova2 right-arm LeRobot v2.1 dataset collected by
    realrobot/dobot-sdk-clawkep/experiments/run_control_lerobot21.py.

    Expected features:
      - observation.images.third_person / observation.images.wrist videos
      - observation.state: [T, 7] = xyz(m) + euler_xyz(rad) + gripper
      - action:            [T, 7] = absolute Cartesian command in the same layout

    The collection loop records image/state/action before env.step(action), so
    the default training alignment is observation t -> command action t.

    ACTION_MODE=dobot_ee6d uses absolute Cartesian targets with continuous
    gripper regression:
      - first 10 dims: Dobot active arm xyz + rot6d + gripper
      - last 10 dims: zeros for the missing second arm
    Set dobot_ee6d_arm_slot=second to put the active arm in dims 10:20 instead.

    ACTION_MODE=libero keeps xyz/Euler/gripper and lets action_slice subtract
    the current proprio on xyz/Euler channels, yielding Bridge/Libero-style
    relative targets.
    """

    CAMERA_VIEW = ["observation.images.third_person", "observation.images.wrist"]
    FRONT_VIEW = "observation.images.third_person"

    @classmethod
    def _resolve_camera_view_order(cls, requested_order) -> list[str]:
        if requested_order is None:
            return list(cls.CAMERA_VIEW)
        if isinstance(requested_order, str):
            requested_order = [x.strip() for x in requested_order.split(",") if x.strip()]
        else:
            requested_order = [str(x).strip() for x in requested_order if str(x).strip()]
        if not requested_order:
            return list(cls.CAMERA_VIEW)
        expected = list(cls.CAMERA_VIEW)
        if sorted(requested_order) != sorted(expected):
            raise ValueError(
                f"Invalid Dobot camera view order {requested_order}. Expected a permutation of {expected}."
            )
        return requested_order

    @staticmethod
    def _as_2d_float(data, key: str, width: int = 7) -> np.ndarray:
        arr = np.asarray(data[key], dtype=np.float32)
        if arr.ndim == 1:
            arr = np.stack(arr, axis=0)
        if arr.ndim != 2:
            arr = arr.reshape(arr.shape[0], -1)
        return arr[:, :width]

    @staticmethod
    def _align_euler_to_reference(euler: np.ndarray, reference: np.ndarray) -> np.ndarray:
        """Return equivalent Euler angles whose elementwise delta to reference is in [-pi, pi]."""
        return reference + (euler - reference + np.pi) % (2.0 * np.pi) - np.pi

    @staticmethod
    def _decode_image_item(obj):
        if isinstance(obj, dict):
            if obj.get("bytes") is not None:
                return decode_image_from_bytes(obj["bytes"])
            if obj.get("path"):
                return decode_image_from_bytes(read_bytes(obj["path"]))
        if isinstance(obj, (bytes, bytearray)):
            return decode_image_from_bytes(obj)
        if isinstance(obj, np.ndarray):
            return Image.fromarray(obj)
        return Image.fromarray(np.asarray(obj))

    @classmethod
    def _prepare_image_frame(cls, frame: np.ndarray, view_key: str) -> np.ndarray:
        frame = np.asarray(frame)
        if view_key != cls.FRONT_VIEW or frame.ndim < 2:
            return frame

        height, width = frame.shape[:2]
        if width <= height:
            return frame

        # Dobot front camera is 640x480. Keep the right 480x480 crop and drop
        # the left strip, matching the table-centered region used for training.
        left = width - height
        return frame[:, left : left + height].copy()

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
        camera_view = self._resolve_camera_view_order(
            kwargs.get("camera_view_order", kwargs.get("dobot_camera_order"))
        )
        future_view = str(kwargs.get("dobot_future_view", "") or "").strip()
        if future_view:
            if future_view not in camera_view:
                raise ValueError(f"Invalid Dobot future view {future_view!r}. Expected one of {camera_view}.")
            future_view_index = camera_view.index(future_view)
        else:
            future_view_index = 0

        action_mode = str(kwargs.get("action_mode", "libero")).lower()
        single_arm_ee6d = action_mode in (
            "ee6d10",
            "ee6d_10",
            "dobot_ee6d10",
            "dobot_ee6d_10",
        )
        use_libero_space = action_mode in ("libero", "bridge_libero", "libero_abs", "bridge_libero_abs")
        apply_libero_delta = action_mode in ("libero", "bridge_libero")
        libero_delta_idx = [0, 1, 2, 3, 4, 5] if apply_libero_delta else []
        ee6d_arm_slot = str(kwargs.get("dobot_ee6d_arm_slot", "first") or "first").lower()
        if ee6d_arm_slot in ("0", "left", "active", "single"):
            ee6d_arm_slot = "first"
        elif ee6d_arm_slot in ("1", "right"):
            ee6d_arm_slot = "second"
        if ee6d_arm_slot not in ("first", "second"):
            raise ValueError(f"dobot_ee6d_arm_slot must be first or second, got {ee6d_arm_slot!r}.")
        action_offset = int(kwargs.get("dobot_action_offset", kwargs.get("action_offset", 0)) or 0)
        if action_offset < 0:
            raise ValueError(f"dobot_action_offset must be >= 0, got {action_offset}.")
        filter_static_frames = bool(kwargs.get("filter_static_frames", True))

        item = self.meta["datalist"][traj_idx]
        episode_index = int(item["episode_index"])
        episode_chunk = episode_index // int(self.meta["chunks_size"])
        data_path = fileio.join_path(self.meta["root_path"], self.meta["data_path"]).format(
            episode_chunk=episode_chunk,
            episode_index=episode_index,
        )
        video_paths = [
            fileio.join_path(self.meta["root_path"], self.meta["video_path"]).format(
                episode_chunk=episode_chunk,
                episode_index=episode_index,
                video_key=vkey,
            )
            for vkey in camera_view
        ]

        data = read_parquet(data_path)
        if all(os.path.exists(p) for p in video_paths):
            videos = [read_video_to_frames(p) for p in video_paths]
        else:
            videos = []
            for key in camera_view:
                if key not in data:
                    videos.append(np.zeros((0, 256, 256, 3), dtype=np.uint8))
                    continue
                frames = [self._decode_image_item(x) for x in data[key]]
                videos.append(np.stack([np.asarray(frame) for frame in frames], axis=0))

        actions_raw = self._as_2d_float(data, "action", width=7)
        state_raw = self._as_2d_float(data, "observation.state", width=7)

        valid_video_lengths = [v.shape[0] for v in videos if v.shape[0] > 0]
        length = min(actions_raw.shape[0], state_raw.shape[0], *valid_video_lengths)
        actions_raw = actions_raw[:length]
        state_raw = state_raw[:length]
        videos = [v[:length] for v in videos]

        if use_libero_space:
            pad = np.zeros((length, 1), dtype=np.float32)
            abs_action = np.concatenate([actions_raw[:, :7], pad], axis=-1)
            abs_state = np.concatenate([state_raw[:, :7], pad], axis=-1)
        else:
            pos = actions_raw[:, :3]
            euler = actions_raw[:, 3:6]
            grip = actions_raw[:, 6:7]
            state_pos = state_raw[:, :3]
            state_euler = state_raw[:, 3:6]
            state_grip = state_raw[:, 6:7]
            left = np.concatenate([pos, euler_to_rotate6d(euler, "xyz").astype(np.float32), grip], axis=-1)
            state_left = np.concatenate(
                [state_pos, euler_to_rotate6d(state_euler, "xyz").astype(np.float32), state_grip],
                axis=-1,
            )
            if single_arm_ee6d:
                abs_action = left
                abs_state = state_left
            else:
                zeros = np.zeros_like(left)
                if ee6d_arm_slot == "second":
                    abs_action = np.concatenate([zeros, left], axis=-1)
                    abs_state = np.concatenate([zeros, state_left], axis=-1)
                else:
                    abs_action = np.concatenate([left, zeros], axis=-1)
                    abs_state = np.concatenate([state_left, zeros], axis=-1)

        base_image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        base_image_mask[: min(self.num_views, len(videos))] = True

        idxs = list(range(0, max(0, length - action_offset)))
        if training:
            random.shuffle(idxs)

        ins = item["tasks"][0] if item.get("tasks") else ""
        dataset_tag = os.path.basename(str(self.meta.get("root_path", "dobot_nova2")).rstrip("/")) or "dobot_nova2"
        for idx in idxs:
            action_start = idx + action_offset
            action_end = action_start + num_actions
            if action_end > length:
                continue
            if return_future and future_index > 0 and idx + future_index >= videos[future_view_index].shape[0]:
                continue

            imgs = []
            for v in range(min(self.num_views, len(videos))):
                frame = videos[v][idx] if videos[v].shape[0] > 0 else np.zeros_like(videos[0][idx])
                frame = self._prepare_image_frame(frame, camera_view[v])
                imgs.append(image_aug(Image.fromarray(frame)))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            state_cur_np = abs_state[idx].copy()
            action_seq_np = abs_action[action_start:action_end].copy()
            if apply_libero_delta:
                action_seq_np[:, 3:6] = self._align_euler_to_reference(action_seq_np[:, 3:6], state_cur_np[3:6])

            state_cur = torch.tensor(state_cur_np, dtype=torch.float32).view(1, -1)
            action_seq = torch.tensor(action_seq_np, dtype=torch.float32)
            if filter_static_frames:
                if use_libero_space:
                    motion_probe = action_seq[:, :7]
                elif single_arm_ee6d:
                    motion_probe = action_seq[:, :10]
                elif ee6d_arm_slot == "second":
                    motion_probe = action_seq[:, 10:20]
                else:
                    motion_probe = action_seq[:, :10]
                if (motion_probe[0] - motion_probe[-1]).abs().max() < 1e-5:
                    continue

            aug_ins = ins
            if training and lang_aug_map and aug_ins in lang_aug_map:
                aug_ins = random.choice(lang_aug_map[aug_ins])

            image_mask = base_image_mask.clone()
            sample = {
                "language_instruction": aug_ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": torch.cat([state_cur, action_seq], dim=0).float(),
                "sample_key": f"{dataset_tag}_ep{episode_index:06d}_frame{idx:06d}",
                "episode_index": int(episode_index),
                "frame_index": int(idx),
                "dataset_tag": dataset_tag,
            }
            if libero_delta_idx:
                sample["idx_for_delta"] = libero_delta_idx
            if use_libero_space:
                sample["idx_for_gripper"] = [6]
                sample["gripper_deadzone"] = 0.05
                sample["gripper_to_signed"] = True
            elif single_arm_ee6d:
                sample["idx_for_gripper"] = [9]
            else:
                sample["idx_for_gripper"] = [9, 19]
            if return_future and future_index > 0:
                future_frame = self._prepare_image_frame(
                    videos[future_view_index][idx + future_index],
                    camera_view[future_view_index],
                )
                future_img = image_aug(Image.fromarray(future_frame))
                sample["future_image"] = future_img
            yield sample
