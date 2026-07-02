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

from typing import Optional, Tuple, Iterable, Sequence, Any
import json
import os
import random
import numpy as np
import h5py
import torch
from mmengine import fileio
from PIL import Image
from scipy.interpolate import interp1d

from ..utils import euler_to_rotate6d, quat_to_rotate6d
from .base import BaseHDF5Handler, _open_h5


# ------------------------------- Calvin --------------------------------------
class CalvinHandler(BaseHDF5Handler):
    """Calvin (sim): proprio [T,7] -> xyz(3)+euler_xyz(3)+grip(1). Right is zeros."""
    dataset_name = "Calvin"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        proprio = f["proprio"][()]  # [T,7]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:] < 0.],
            axis=-1,
        )  # [T,10]
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 20))


# --------------------------------- RT1 ---------------------------------------
class RT1Handler(BaseHDF5Handler):
    """RT1 (sim-like packaging): eef_quat_orientation [T,7], gripper [T,1]."""
    dataset_name = "RT1"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 3.0, 10.0
        eefq = f["eef_quat_orientation"][()]  # [T,7] pos3 + quat4
        grip = f["gripper"][()]               # [T,1] or [T]
        if grip.ndim == 1:
            grip = grip[:, None]
        left = np.concatenate([eefq[:, :3], quat_to_rotate6d(eefq[:, 3:]), grip], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 6))


# ------------------------------- Bridge --------------------------------------
class BridgeHandler(BaseHDF5Handler):
    """
    Bridge (sim). HDF5:
      /proprio [T, >=6] -> xyz(3) + euler_xyz(3) + ...
      /action  [T, ...] -> last channel is gripper (1=open), we convert to (1=closed)
    Output left/right: [T,10] = xyz(3)+rot6d(6)+grip(1). Single arm → right zeros.
    """
    dataset_name = "Bridge"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 5.0, 5.0
        proprio = f["proprio"][()]                     # [T, >=6]
        action  = f["action"][()]                      # [T, ...]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), 1 - action[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------- LIBERO --------------------------------------
class LiberoHandler(BaseHDF5Handler):
    """
    LIBERO (sim). HDF5:
      /abs_action_6d [T,10] = xyz(3)+rot6d(6)+grip_raw(1). Single arm.
    Also drops first frame for images (matches original pipeline behavior).
    """
    dataset_name = "libero"

    def get_image_datasets(self, f: h5py.File) -> Sequence[Any]:
        keys = self.meta["observation_key"]
        images = [f[k] for k in keys]
        # Drop the first frame (image desync quirk in original data)
        return [img[1:] for img in images]

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        a = f["abs_action_6d"][()]                             # [T,10]
        left = np.concatenate([a[:, :9], (a[:, 9:] > 0.0)], axis=-1)
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


# ------------------------------ VLABench -------------------------------------
class VLABenchHandler(BaseHDF5Handler):
    """
    VLABench (sim). HDF5:
      /proprio [T, >=7] -> xyz(3) + euler_xyz(3) + grip(1).
    Single arm → right zeros.
    """
    dataset_name = "VLABench"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        proprio = f["proprio"][()]
        left = np.concatenate(
            [proprio[:, :3], euler_to_rotate6d(proprio[:, 3:6], "xyz"), proprio[:, -1:]],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 15))


# ------------------------------ RobotWin2 ------------------------------------
class RobotWin2Handler(BaseHDF5Handler):
    """
    robotwin2_abs_ee / robotwin2_clean (sim). HDF5:
      /endpose/left_endpose   [T,7]  xyz(3)+quat(4)
      /endpose/right_endpose  [T,7]
      /endpose/left_gripper   [T]    1=open  -> convert to 1=closed
      /endpose/right_gripper  [T]
    Output both arms. freq≈30Hz, qdur=1s.
    """
    dataset_name = "robotwin2-*"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        l = f["endpose/left_endpose"][()]                      # [T,7]
        r = f["endpose/right_endpose"][()]                     # [T,7]
        lg = (1 - f["endpose/left_gripper"][()][:, None])      # [T,1] 1=closed
        rg = (1 - f["endpose/right_gripper"][()][:, None])
        left  = np.concatenate([l[:, :3], quat_to_rotate6d(l[:, 3:]), lg], axis=-1)
        right = np.concatenate([r[:, :3], quat_to_rotate6d(r[:, 3:]), rg], axis=-1)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 10))


class RobotWin2Clean50HDF5Handler(BaseHDF5Handler):
    """
    TianxingChen/RoboTwin2.0 aloha-agilex_clean_50 HDF5.

    Images:
      - front/current: observation/head_camera/rgb
      - future: observation/head_camera/rgb at future_index
      - wrist token: horizontal composition of observation/left_camera/rgb and
        observation/right_camera/rgb so both arms are visible through the single
        WorldTeacher wrist-token path.

    Actions/proprio are absolute Cartesian EE6D for both arms:
      [left xyz(3), left rot6d(6), left closed_grip(1),
       right xyz(3), right rot6d(6), right closed_grip(1)].
    """
    dataset_name = "robotwin2_clean50_hdf5"

    HEAD_KEY = "observation/head_camera/rgb"
    LEFT_WRIST_KEY = "observation/left_camera/rgb"
    RIGHT_WRIST_KEY = "observation/right_camera/rgb"

    def _resolve_item(self, item: Any) -> dict:
        if isinstance(item, dict):
            out = dict(item)
        elif isinstance(item, (list, tuple)):
            out = {"path": item[0]}
        else:
            out = {"path": item}
        root = str(self.meta.get("root_path", "") or "")
        for key in ("path", "instruction_path"):
            value = out.get(key)
            if value and root and not os.path.isabs(str(value)):
                out[key] = fileio.join_path(root, str(value))
        return out

    def _instruction_pool(self, item: dict) -> list[str]:
        if item.get("instruction"):
            return [str(item["instruction"])]
        path = item.get("instruction_path")
        if path:
            try:
                payload = json.loads(fileio.get(path).decode("utf-8"))
                split = str(self.meta.get("instruction_split", "seen"))
                if split == "all":
                    pool = []
                    for value in payload.values():
                        if isinstance(value, list):
                            pool.extend(str(x) for x in value)
                    if pool:
                        return pool
                value = payload.get(split) or payload.get("seen") or payload.get("unseen")
                if isinstance(value, list) and value:
                    return [str(x) for x in value]
            except Exception as exc:
                print(f"[RobotWin2Clean50] failed to read instruction {path}: {type(exc).__name__}: {exc}")
        task_name = str(item.get("task_name") or self.meta.get("task_name") or "robotwin task")
        return [task_name.replace("_", " ")]

    @staticmethod
    def _compose_wrist(left: Image.Image, right: Image.Image) -> Image.Image:
        left = left.convert("RGB")
        right = right.convert("RGB")
        if right.size[1] != left.size[1]:
            scale = left.size[1] / max(1, right.size[1])
            right = right.resize((max(1, int(right.size[0] * scale)), left.size[1]))
        canvas = Image.new("RGB", (left.size[0] + right.size[0], left.size[1]))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.size[0], 0))
        return canvas

    @staticmethod
    def _closed_gripper(f: h5py.File, key: str, length: int) -> np.ndarray:
        g = np.asarray(f[key][()], dtype=np.float32)
        if g.ndim == 1:
            g = g[:, None]
        # RoboTwin stores 1=open, 0=closed. WorldTeacher EE6D targets use 1=closed.
        return (1.0 - g[:length, :1]).astype(np.float32)

    def _build_ee6d(self, f: h5py.File) -> np.ndarray:
        left_pose = np.asarray(f["endpose/left_endpose"][()], dtype=np.float32)
        right_pose = np.asarray(f["endpose/right_endpose"][()], dtype=np.float32)
        length = min(left_pose.shape[0], right_pose.shape[0])
        left_pose = left_pose[:length]
        right_pose = right_pose[:length]
        left = np.concatenate(
            [
                left_pose[:, :3],
                quat_to_rotate6d(left_pose[:, 3:7]).astype(np.float32),
                self._closed_gripper(f, "endpose/left_gripper", length),
            ],
            axis=-1,
        )
        right = np.concatenate(
            [
                right_pose[:, :3],
                quat_to_rotate6d(right_pose[:, 3:7]).astype(np.float32),
                self._closed_gripper(f, "endpose/right_gripper", length),
            ],
            axis=-1,
        )
        return np.concatenate([left, right], axis=-1).astype(np.float32)

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
        item = self._resolve_item(self.meta["datalist"][traj_idx])
        datapath = str(item["path"])
        episode_index = int(item.get("episode_index", traj_idx))

        with _open_h5(datapath) as f:
            head_images = f[self.HEAD_KEY][()]
            left_images = f[self.LEFT_WRIST_KEY][()]
            right_images = f[self.RIGHT_WRIST_KEY][()]
            ee = self._build_ee6d(f)

        length = min(ee.shape[0], head_images.shape[0], left_images.shape[0], right_images.shape[0])
        ee = ee[:length]
        head_images = head_images[:length]
        left_images = left_images[:length]
        right_images = right_images[:length]

        freq = float(self.meta.get("fps", 30.0))
        qdur = float(self.meta.get("query_duration", 1.0))
        t = np.arange(length, dtype=np.float64) / freq
        interp = interp1d(t, ee, axis=0, bounds_error=False, fill_value=(ee[0], ee[-1]))

        idxs = list(range(0, max(0, length - 1)))
        if training:
            random.shuffle(idxs)
        instruction_pool = self._instruction_pool(item)
        dataset_tag = str(item.get("task_name") or self.meta.get("dataset_name") or "robotwin2_clean50")

        image_mask = torch.zeros(self.num_views, dtype=torch.bool)
        image_mask[: min(self.num_views, 2)] = True

        for idx in idxs:
            if idx + num_actions >= length:
                continue
            if return_future and future_index > 0 and idx + future_index >= length:
                continue

            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            traj = torch.tensor(interp(q), dtype=torch.float32)
            if (traj[1] - traj[-1]).abs().max() < 1e-5:
                continue

            front_img = self._pil_from_arr(head_images[idx])
            left_wrist = self._pil_from_arr(left_images[idx])
            right_wrist = self._pil_from_arr(right_images[idx])
            wrist_img = self._compose_wrist(left_wrist, right_wrist)

            imgs = [image_aug(front_img), image_aug(wrist_img)]
            while len(imgs) < self.num_views:
                imgs.append(torch.zeros_like(imgs[0]))

            ins = random.choice(instruction_pool) if training and instruction_pool else instruction_pool[0]
            if training and lang_aug_map and ins in lang_aug_map:
                ins = random.choice(lang_aug_map[ins])

            sample = {
                "language_instruction": ins,
                "image_input": torch.stack(imgs, dim=0),
                "image_mask": image_mask,
                "abs_trajectory": traj.float(),
                "sample_key": f"{dataset_tag}_ep{episode_index:06d}_frame{idx:06d}",
                "episode_index": episode_index,
                "frame_index": int(idx),
                "idx_for_gripper": [9, 19],
            }
            if return_future and future_index > 0:
                sample["future_image"] = image_aug(self._pil_from_arr(head_images[idx + future_index]))
            yield sample


# ---------------------------- Robocasa-Human ---------------------------------
class RobocasaHumanHandler(BaseHDF5Handler):
    """
    robocasa-human (teleop in sim). HDF5:
      /action_dict/abs_pos     [T,3]
      /action_dict/abs_rot_6d  [T,6]
      /action_dict/gripper     [T,1]  ( >0 => closed )
    Single arm → right zeros.
    """
    dataset_name = "robocasa-human"

    def build_left_right(
        self, f: h5py.File
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], float, float]:
        freq, qdur = 30.0, 1.0
        left = np.concatenate(
            [
                f["action_dict/abs_pos"][()],
                f["action_dict/abs_rot_6d"][()],
                (f["action_dict/gripper"][()] > 0.0).astype(np.float32),
            ],
            axis=-1,
        )
        right = np.zeros_like(left)
        return left, right, None, None, freq, qdur

    def index_candidates(self, T_left: int, training: bool) -> Iterable[int]:
        return range(0, max(0, T_left - 30))
