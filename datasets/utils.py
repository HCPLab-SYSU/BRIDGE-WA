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
import io, numpy as np, pyarrow.parquet as pq, av, cv2
from mmengine import fileio
from PIL import Image
from scipy.spatial.transform import Rotation as R
import h5py
from typing import Sequence, Dict
import torch

def read_bytes(path: str) -> bytes:
    return fileio.get(path)

def open_h5(path: str) -> h5py.File:
    try: return h5py.File(path, "r")
    except OSError: return h5py.File(io.BytesIO(read_bytes(path)), "r")

def read_video_to_frames(path: str) -> np.ndarray:
    buf = io.BytesIO(read_bytes(path)); container = av.open(buf, options={'threads': '2'})
    frames = []
    for packet in container.demux(video=0):
        for f in packet.decode(): frames.append(f.to_ndarray(format="rgb24"))
    container.close()
    return np.stack(frames, axis=0)

def read_parquet(path: str) -> dict:
    buf = io.BytesIO(read_bytes(path))
    return pq.read_table(buf).to_pydict()

def decode_image_from_bytes(x) -> Image.Image:
    if isinstance(x, (bytes, bytearray)): x = np.frombuffer(x, dtype=np.uint8)
    rgb = cv2.imdecode(x, cv2.IMREAD_COLOR)
    if rgb is None:
        rgb = np.frombuffer(x, dtype=np.uint8)
        if rgb.size == 2764800: rgb = rgb.reshape(720, 1280, 3)
        elif rgb.size == 921600: rgb = rgb.reshape(480, 640, 3)
    else:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

def _quat_to_scipy(q: np.ndarray, scalar_first: bool) -> np.ndarray:
    q = np.asarray(q)
    if scalar_first:
        # Convert wxyz -> xyzw for SciPy without scalar_first support
        q = np.concatenate([q[..., 1:4], q[..., 0:1]], axis=-1)
    return q


def _quat_from_scipy(q: np.ndarray, scalar_first: bool) -> np.ndarray:
    q = np.asarray(q)
    if scalar_first:
        # Convert xyzw -> wxyz
        q = np.concatenate([q[..., 3:4], q[..., 0:3]], axis=-1)
    return q


def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    try:
        rot = R.from_quat(q, scalar_first=scalar_first)
    except TypeError:
        rot = R.from_quat(_quat_to_scipy(q, scalar_first))
    return rot.as_matrix()[..., :, :2].reshape(np.asarray(q).shape[:-1] + (6,))

def euler_to_rotate6d(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    return R.from_euler(pattern, q, degrees=False).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def euler_to_rotvec(q: np.ndarray, pattern: str = "xyz") -> np.ndarray:
    q = np.asarray(q)
    return R.from_euler(pattern, q, degrees=False).as_rotvec()


def rotvec_to_rotate6d(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    return R.from_rotvec(q).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))


def rotate6d_to_xyz(v6: np.ndarray) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    return R.from_matrix(rot_mats).as_euler('xyz')

def rotate6d_to_quat(v6: np.ndarray, scalar_first = False) -> np.ndarray:
    v6 = np.asarray(v6)
    if v6.shape[-1] != 6:
        raise ValueError("Last dimension must be 6 (got %s)" % (v6.shape[-1],))
    a1 = v6[..., 0:5:2]
    a2 = v6[..., 1:6:2]
    b1 = a1 / np.linalg.norm(a1, axis=-1, keepdims=True)
    proj = np.sum(b1 * a2, axis=-1, keepdims=True) * b1
    b2 = a2 - proj
    b2 = b2 / np.linalg.norm(b2, axis=-1, keepdims=True)
    b3 = np.cross(b1, b2)
    rot_mats = np.stack((b1, b2, b3), axis=-1)      # shape (..., 3, 3)
    rot = R.from_matrix(rot_mats)
    try:
        return rot.as_quat(scalar_first=scalar_first)
    except TypeError:
        return _quat_from_scipy(rot.as_quat(), scalar_first)


def _rotvec_to_quat_torch(rotvec: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle (rotvec) to quaternion in xyzw order."""
    angle = torch.linalg.norm(rotvec, dim=-1, keepdim=True)
    half = 0.5 * angle
    eps = 1e-8
    # sin(theta/2)/theta, stable around zero
    scale = torch.where(
        angle > eps,
        torch.sin(half) / angle,
        0.5 - (angle * angle) / 48.0,
    )
    xyz = rotvec * scale
    w = torch.where(angle > eps, torch.cos(half), 1.0 - (angle * angle) / 8.0)
    return torch.cat([xyz, w], dim=-1)


def _quat_mul_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """Quaternion multiplication in xyzw order."""
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    x2, y2, z2, w2 = q2.unbind(dim=-1)
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    return torch.stack([x, y, z, w], dim=-1)


def _quat_to_rotvec_torch(quat: torch.Tensor) -> torch.Tensor:
    """Convert quaternion in xyzw order to axis-angle (rotvec)."""
    eps = 1e-8
    quat = quat / torch.linalg.norm(quat, dim=-1, keepdim=True).clamp_min(eps)
    # Canonicalize to shortest-path rotation.
    sign = torch.where(quat[..., 3:4] < 0, -1.0, 1.0)
    quat = quat * sign
    xyz = quat[..., :3]
    w = quat[..., 3:4]
    xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, w.clamp_min(eps))
    scale = torch.where(xyz_norm > eps, angle / xyz_norm, torch.full_like(xyz_norm, 2.0))
    return xyz * scale


def _rotvec_delta_torch(target: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """
    Relative rotation from anchor to target in rotvec form:
        R_delta = R_anchor^{-1} * R_target
    """
    q_target = _rotvec_to_quat_torch(target)
    q_anchor = _rotvec_to_quat_torch(anchor)
    q_anchor_inv = torch.cat([-q_anchor[..., :3], q_anchor[..., 3:4]], dim=-1)
    q_delta = _quat_mul_torch(q_anchor_inv, q_target)
    return _quat_to_rotvec_torch(q_delta)


def action_slice(abs_traj: torch.Tensor,
                 idx_for_delta: Sequence[int] = (),
                 idx_for_mask_proprio: Sequence[int] = (),
                 idx_for_gripper: Sequence[int] = (),
                 gripper_deadzone: float = 0.0,
                 gripper_to_signed: bool = False,
                 idx_for_rot_delta: Sequence[int] = (),
                ) -> Dict[str, torch.Tensor]:
    if not isinstance(abs_traj, torch.Tensor):
        raise TypeError("abs_traj must be a torch.Tensor")
    if abs_traj.ndim != 2 or abs_traj.size(0) < 2:
        raise ValueError("abs_traj must be [H+1, D] with H>=1")

    proprio = abs_traj[0].clone() # [D]
    action = abs_traj[1:].clone() # [H, D]

    if idx_for_gripper:
        dz = max(0.0, float(gripper_deadzone))
        for gi in idx_for_gripper:
            if gi < 0 or gi >= action.shape[1]:
                continue
            g = action[:, gi].clone()

            # Unify 0/1 and -1/1 conventions into a signed representation when requested.
            if gripper_to_signed:
                finite_g = g[torch.isfinite(g)]
                if finite_g.numel() > 0:
                    g_min = float(finite_g.min())
                    g_max = float(finite_g.max())
                    if g_min >= -dz and g_max <= 1.0 + dz:
                        g = g * 2.0 - 1.0

            # Apply dead-zone hysteresis to suppress tiny interpolation noise around 0.
            if dz > 0.0:
                prev = None
                for t in range(g.shape[0]):
                    vt = float(g[t])
                    if vt > dz:
                        prev = 1.0
                        break
                    if vt < -dz:
                        prev = -1.0
                        break
                if prev is None:
                    prev = 1.0 if float(g[0]) >= 0.0 else -1.0
                for t in range(g.shape[0]):
                    vt = float(g[t])
                    if vt > dz:
                        prev = 1.0
                    elif vt < -dz:
                        prev = -1.0
                    g[t] = prev
            action[:, gi] = g

    rot_idx_set = set(int(i) for i in idx_for_rot_delta)
    if idx_for_delta:
        lin_idx = [int(i) for i in idx_for_delta if int(i) not in rot_idx_set]
        if lin_idx:
            idx = torch.as_tensor(lin_idx, dtype=torch.long, device=abs_traj.device)
            action[:, idx] -= proprio[idx]

    if idx_for_rot_delta:
        # Treat every 3 dims as one axis-angle rotation block.
        rot_idx = [int(i) for i in idx_for_rot_delta]
        if len(rot_idx) >= 3:
            for base in range(0, len(rot_idx) - 2, 3):
                tri = rot_idx[base:base + 3]
                if len(tri) < 3:
                    break
                tri_t = torch.as_tensor(tri, dtype=torch.long, device=abs_traj.device)
                target_rot = action[:, tri_t]      # [H, 3]
                anchor_rot = proprio[tri_t][None]  # [1, 3]
                action[:, tri_t] = _rotvec_delta_torch(target_rot, anchor_rot)

    if idx_for_mask_proprio:
        idx = torch.as_tensor(idx_for_mask_proprio, dtype=torch.long, device=abs_traj.device)
        proprio[idx] = 0.0
    return {"proprio": proprio, "action": action}
