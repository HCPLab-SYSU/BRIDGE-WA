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

from ..utils import (
    decode_image_from_bytes,
    euler_to_rotate6d,
    read_bytes,
    read_parquet,
    read_video_to_frames,
)
from .base import DomainHandler


class LeRobotVLABenchHandler(DomainHandler):
    """
    LeRobot VLABench primitive dataset from HF.

    Expected features:
      - image / second_image / wrist_image
      - actions: [T, 7] = xyz(3) + euler_xyz(3) + gripper(1)
      - state:   [T, 7] = xyz(3) + euler_xyz(3) + gripper(1)
      - tasks: list[str] in episodes.jsonl

    Default VisionAction/VLABench action space is ee6d. We convert euler xyz to 6D rotation
    and pad the second arm with zeros, matching the deployment client which only uses
    the first 10 dims.
    """

    CAMERA_VIEW = ["image", "wrist_image", "second_image"]

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
                f"Invalid VLABench camera view order {requested_order}. Expected a permutation of {expected}."
            )
        return requested_order

    def _warn_skip_episode(self, *, kind: str, episode_index: int, path: str, exc: Exception) -> None:
        reported = getattr(self, "_reported_bad_episode_paths", None)
        if reported is None:
            reported = set()
            self._reported_bad_episode_paths = reported
        key = (kind, path)
        if key in reported:
            return
        reported.add(key)
        print(
            f"[VLABench] Skip episode {episode_index} due to unreadable {kind}: {path} "
            f"({type(exc).__name__}: {exc})"
        )

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
            kwargs.get("camera_view_order", kwargs.get("vlabench_camera_order"))
        )
        future_view = str(kwargs.get("vlabench_future_view", "") or "").strip()
        if future_view:
            if future_view not in camera_view:
                raise ValueError(
                    f"Invalid VLABench future view {future_view!r}. Expected one of {camera_view}."
                )
            future_view_index = camera_view.index(future_view)
        else:
            future_view_index = 0
        wrist_view_index = camera_view.index("wrist_image") if "wrist_image" in camera_view else None
        action_mode = str(kwargs.get("action_mode", "ee6d")).lower()
        single_arm = action_mode in ("ee6d10", "ee6d_10", "vlabench10", "vlabench_10")
        use_libero_space = action_mode in ("libero", "bridge_libero", "libero_abs", "bridge_libero_abs")
        apply_libero_delta = action_mode in ("libero", "bridge_libero")
        libero_delta_idx = [0, 1, 2, 3, 4, 5] if apply_libero_delta else []
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
            for vkey in camera_view
        ]

        try:
            data = read_parquet(data_path)
        except Exception as exc:
            self._warn_skip_episode(kind="parquet", episode_index=episode_index, path=data_path, exc=exc)
            return
        use_video = all(os.path.exists(p) for p in video_paths)
        if use_video:
            try:
                videos = [read_video_to_frames(p) for p in video_paths]
            except Exception as exc:
                bad_path = next((p for p in video_paths if os.path.exists(p)), video_paths[0])
                self._warn_skip_episode(kind="video", episode_index=episode_index, path=bad_path, exc=exc)
                return
        else:
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

            videos = []
            for key in camera_view:
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

        length = min(actions.shape[0], state.shape[0], *[v.shape[0] for v in videos if v.shape[0] > 0])
        actions = actions[:length, :7]
        state = state[:length, :7]
        videos = [v[:length] for v in videos]

        pos = actions[:, :3]
        euler = actions[:, 3:6]
        grip = actions[:, 6:7]
        state_pos = state[:, :3]
        state_euler = state[:, 3:6]
        state_grip = state[:, 6:7]

        if use_libero_space:
            # Keep raw xyz-Euler channels in libero-mode to stay aligned with the
            # existing BridgeDataV2 pretrained checkpoint distribution.
            left = np.concatenate([pos, euler, grip], axis=-1)
            state_left = np.concatenate([state_pos, state_euler, state_grip], axis=-1)
            pad = np.zeros((left.shape[0], 1), dtype=np.float32)
            state_pad = np.zeros((state_left.shape[0], 1), dtype=np.float32)
            abs_action = np.concatenate([left, pad], axis=-1)
            abs_state = np.concatenate([state_left, state_pad], axis=-1)
        else:
            rot6d = euler_to_rotate6d(euler, "xyz").astype(np.float32)
            state_rot6d = euler_to_rotate6d(state_euler, "xyz").astype(np.float32)
            left = np.concatenate([pos, rot6d, grip], axis=-1)
            state_left = np.concatenate([state_pos, state_rot6d, state_grip], axis=-1)
            if single_arm:
                abs_action = left
                abs_state = state_left
            else:
                right = np.zeros_like(left)
                state_right = np.zeros_like(state_left)
                abs_action = np.concatenate([left, right], axis=-1)
                abs_state = np.concatenate([state_left, state_right], axis=-1)

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

        idxs = list(range(0, max(0, length - 1)))
        if training:
            random.shuffle(idxs)

        ins = item["tasks"][0] if item.get("tasks") else ""
        dataset_tag = os.path.basename(str(self.meta.get("root_path", "vlabench")).rstrip("/")) or "vlabench"
        for idx in idxs:
            if idx + num_actions >= length:
                continue
            if return_future and future_index > 0 and idx + future_index >= videos[future_view_index].shape[0]:
                continue

            image_mask = base_image_mask.clone()
            if wrist_view_index is not None and wrist_view_index < image_mask.shape[0] and not bool(wrist_valid[idx]):
                image_mask[wrist_view_index] = False

            imgs = []
            for v in range(min(self.num_views, len(videos))):
                if videos[v].shape[0] == 0:
                    frame = np.zeros_like(videos[0][idx])
                elif wrist_view_index is not None and v == wrist_view_index and not bool(image_mask[wrist_view_index]):
                    frame = np.zeros_like(videos[0][idx])
                else:
                    frame = videos[v][idx]
                imgs.append(image_aug(Image.fromarray(frame)))
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, dim=0)

            state_cur = torch.tensor(abs_state[idx], dtype=torch.float32).view(1, -1)
            action_seq = torch.tensor(abs_action[idx + 1 : idx + 1 + num_actions], dtype=torch.float32)
            if filter_static_frames and (action_seq[0] - action_seq[-1]).abs().max() < 1e-5:
                continue

            aug_ins = ins
            if training and lang_aug_map and aug_ins in lang_aug_map:
                aug_ins = random.choice(lang_aug_map[aug_ins])

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
            else:
                sample["idx_for_gripper"] = [9] if single_arm else [9, 19]
            if return_future and future_index > 0:
                future_img = image_aug(Image.fromarray(videos[future_view_index][idx + future_index]))
                sample["future_image"] = future_img
            yield sample
