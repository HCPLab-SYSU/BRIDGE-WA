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
from typing import Any, Dict, Iterable, List, Optional
import io, json, math, os, random, numpy as np, torch
from torch.utils.data import IterableDataset, get_worker_info
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from mmengine import fileio
from .utils import action_slice
from .domain_config import DATA_WEIGHTS, DATA_DOMAIN_ID
from .domain_handler.registry import get_handler_cls, infer_robot_type

def _collect_meta_file_paths(metas_path: str) -> List[str]:
    """Resolve one or multiple meta inputs to concrete json file paths.

    Supported formats:
      - single file path
      - single directory (scan *.json recursively)
      - comma-separated list of files/directories
    """
    parts = [p.strip() for p in str(metas_path).split(",") if p.strip()]
    if not parts:
        raise ValueError("metas_path is empty.")

    resolved: List[str] = []
    for part in parts:
        if fileio.isdir(part):
            files = fileio.list_dir_or_file(part, suffix=".json", recursive=True, list_dir=False)
            resolved.extend([fileio.join_path(part, f) for f in files])
        else:
            resolved.append(part)

    # Keep order while removing duplicates.
    uniq: List[str] = []
    seen = set()
    for p in resolved:
        if p in seen:
            continue
        uniq.append(p)
        seen.add(p)
    return uniq


def _load_metas_into_dict(meta_paths: List[str], metas: Dict[str, dict]) -> None:
    recognized = 0
    for file_path in meta_paths:
        with io.BytesIO(fileio.get(file_path)) as f:
            meta = json.load(f)
        # General style meta.
        if "dataset_name" in meta.keys() and "datalist" in meta.keys():
            print(f"== dataset {meta['dataset_name']} with {len(meta['datalist'])} trajs")
            metas[meta["dataset_name"]] = meta
            recognized += 1
            continue
        # LeRobot v2.x style meta (official HF datasets use v2.0 / v2.1).
        if "codebase_version" in meta.keys() and str(meta["codebase_version"]).startswith("v2."):
            meta["datalist"] = []
            if "root_path" not in meta.keys():
                meta["root_path"] = "/".join(file_path.split("/")[:-2])
            with io.BytesIO(fileio.get(fileio.join_path("/".join(file_path.split("/")[:-1]), "episodes.jsonl"))) as f:
                for line in f:
                    meta["datalist"].append(json.loads(line.decode("utf-8")))
            metas[meta["root_path"]] = meta
            print(
                f"== lerobot dataset {meta.get('robot_type', 'unknown')} ({meta['codebase_version']}) with {meta['total_episodes']} "
                f"trajs at {meta['root_path']}===="
            )
            recognized += 1
            continue
        # Ignore unrelated json (e.g., convert args / misc configs) to enable mixed-root inputs.
        print(f"== skip non-meta json: {file_path}")

    if recognized == 0:
        raise RuntimeError(f"No valid dataset meta found in: {meta_paths}")


def _normalize_data_proportions(
    data_proportions: Optional[List[float]],
    dataset_names: List[str],
) -> Optional[Dict[str, float]]:
    if data_proportions is None:
        return None
    if len(data_proportions) != len(dataset_names):
        raise ValueError(
            f"Length mismatch: {len(data_proportions)=} but {len(dataset_names)=}. "
            "Please provide one proportion for each meta in --train_metas_path."
        )
    out: Dict[str, float] = {}
    for name, p in zip(dataset_names, data_proportions):
        p = float(p)
        if p < 0.0 or p > 1.0:
            raise ValueError(f"Invalid data proportion {p} for dataset {name}. Expected in [0, 1].")
        out[name] = p
    return out


def _dataset_tag(name: str) -> str:
    n = str(name).rstrip("/")
    if not n:
        return "unknown"
    return os.path.basename(n) or n


class InfiniteDataReader(IterableDataset):
    """
    Output sample:
      {
        'domain_id': LongTensor[],    # domain id
        'language_instruction': str,
        'image_input': FloatTensor[V, C, H, W],
        'image_mask': BoolTensor[V],
        'proprio': FloatTensor[dim_proprio],
        'action': FloatTensor[T, dim_action]
      }
    """
    def __init__(self, 
                 metas_path: str, 
                 num_actions: int = 10, 
                 num_views: int = 3, 
                 training: bool = True,
                 action_mode: str = "ee6d",
                 include_producer_id: bool = False,
                 data_proportions: Optional[List[float]] = None,
                 lang_aug: str = None,
                 future_index: int = 0,
                 return_future: bool = False,
                 image_height: int = 224,
                 image_width: int = 224,
                 handler_kwargs: Optional[Dict[str, Any]] = None,
                 ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.include_producer_id = include_producer_id
        self.future_index = future_index
        self.return_future = return_future
        self.handler_kwargs = dict(handler_kwargs or {})
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        _load_metas_into_dict(_collect_meta_file_paths(metas_path), self.metas)
        dataset_names = list(self.metas.keys())
        self.data_proportion_map = _normalize_data_proportions(data_proportions, dataset_names)
        self.sampled_traj_num: Dict[str, int] = {}
        for name in dataset_names:
            total_traj = len(self.metas[name]["datalist"])
            p = self.data_proportion_map.get(name, 1.0) if self.data_proportion_map else 1.0
            if p <= 0.0:
                sampled = 0
            elif p >= 1.0:
                sampled = total_traj
            else:
                sampled = min(total_traj, int(math.ceil(total_traj * p)))
            self.sampled_traj_num[name] = sampled
        if self.data_proportion_map:
            print("== data proportions enabled:", self.data_proportion_map)
            print("== sampled trajectories:", self.sampled_traj_num)

        self.image_aug = [
            transforms.Resize((image_height, image_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.) \
                if training else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        proportion = self.data_proportion_map.get(dataset_name, 1.0) if self.data_proportion_map else 1.0
        if self.training and proportion <= 0.0:
            return
        if self.training: random.shuffle(traj_indices)
        if self.training and proportion < 1.0:
            keep = min(len(traj_indices), int(math.ceil(len(traj_indices) * proportion)))
            traj_indices = traj_indices[:keep]
        
        if 'robot_type' in meta.keys():
            robot_type = infer_robot_type(meta['robot_type'], meta)
        else:
            robot_type = dataset_name
        Handler = get_handler_cls(robot_type, meta)
        handler = Handler(meta=meta, num_views=self.num_views)
        for traj_idx in traj_indices:
                for sample in handler.iter_episode(
                    traj_idx,
                    num_actions=self.num_actions,
                    training=self.training,
                    image_aug=self.image_aug,
                    lang_aug_map= meta["lang_aug_map"] if "lang_aug_map" in meta.keys() else None,
                    action_mode=self.action_mode,
                    future_index=self.future_index,
                    return_future=self.return_future,
                    **self.handler_kwargs,
                ):
                    sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))
                    sample["dataset_name"] = str(dataset_name)
                    sample["dataset_tag"] = _dataset_tag(dataset_name)
                    idx_for_delta = sample.pop("idx_for_delta", [])
                    idx_for_rot_delta = sample.pop("idx_for_rot_delta", [])
                    idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
                    idx_for_gripper = sample.pop("idx_for_gripper", [])
                    gripper_deadzone = sample.pop("gripper_deadzone", 0.0)
                    gripper_to_signed = sample.pop("gripper_to_signed", False)
                    sample.update(
                        action_slice(
                            sample.pop("abs_trajectory", None),
                            idx_for_delta,
                            idx_for_mask_proprio,
                            idx_for_gripper,
                            gripper_deadzone,
                            gripper_to_signed,
                            idx_for_rot_delta,
                        )
                    )
                    if self.include_producer_id:
                        sample["producer_id"] = torch.tensor(0, dtype=torch.long)
                    yield sample
        if self.training: yield from self._iter_one_dataset(dataset_name)


    def __iter__(self):
        names = list(self.metas.keys())
        if self.data_proportion_map:
            names = [n for n in names if self.sampled_traj_num.get(n, 0) > 0]
            if not names:
                raise RuntimeError("All datasets are disabled by --data_proportions (all are 0).")
        if not self.training: 
            for n in names: yield from self._iter_one_dataset(n)
        else:
            #names = names * 2 # increase the dataset sampling frequency
            gens = [iter(self._iter_one_dataset(n)) for n in names]
            if self.data_proportion_map:
                ws = [float(self.sampled_traj_num.get(n, 1)) for n in names]
            else:
                ws = [DATA_WEIGHTS.get(n, 1.0) for n in names]
            s = sum(ws); ws = [w / s for w in ws]
            while True:
                i = random.choices(range(len(names)), weights=ws, k=1)[0]
                yield next(gens[i])


class EpochDataReader(IterableDataset):
    """
    Finite iterable dataset: one full pass over all data constitutes one epoch.
    """

    def __init__(
        self,
        metas_path: str,
        num_actions: int = 10,
        num_views: int = 3,
        training: bool = True,
        action_mode: str = "ee6d",
        include_producer_id: bool = False,
        data_proportions: Optional[List[float]] = None,
        lang_aug: str = None,
        future_index: int = 0,
        return_future: bool = False,
        image_height: int = 224,
        image_width: int = 224,
        handler_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.num_views = num_views
        self.training = training
        self.num_actions = num_actions
        self.action_mode = action_mode
        self.include_producer_id = include_producer_id
        self.future_index = future_index
        self.return_future = return_future
        self.handler_kwargs = dict(handler_kwargs or {})
        self.metas: Dict[str, dict] = {}
        print("use action mode:", action_mode)
        _load_metas_into_dict(_collect_meta_file_paths(metas_path), self.metas)
        dataset_names = list(self.metas.keys())
        self.data_proportion_map = _normalize_data_proportions(data_proportions, dataset_names)
        if self.data_proportion_map:
            print("== data proportions enabled:", self.data_proportion_map)

        self.image_aug = [
            transforms.Resize((image_height, image_width), interpolation=InterpolationMode.BICUBIC),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.0)
            if training
            else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225), inplace=True),
        ]
        self.image_aug = transforms.Compose(self.image_aug)

    def set_epoch(self, epoch: int):
        self._epoch = epoch

    def _get_shard_info(self):
        worker_info = get_worker_info()
        if worker_info is None:
            worker_id, num_workers = 0, 1
        else:
            worker_id, num_workers = worker_info.id, worker_info.num_workers
        try:
            if torch.distributed.is_available() and torch.distributed.is_initialized():
                rank = torch.distributed.get_rank()
                world_size = torch.distributed.get_world_size()
            else:
                rank, world_size = 0, 1
        except Exception:
            rank, world_size = 0, 1
        shard_id = rank * num_workers + worker_id
        num_shards = world_size * num_workers
        return shard_id, num_shards

    def _iter_one_dataset(self, dataset_name: str) -> Iterable[dict]:
        meta = self.metas[dataset_name]
        traj_indices = list(range(len(meta["datalist"])))
        proportion = self.data_proportion_map.get(dataset_name, 1.0) if self.data_proportion_map else 1.0
        if self.training and proportion <= 0.0:
            return
        if self.training:
            random.shuffle(traj_indices)
            if proportion < 1.0:
                keep = min(len(traj_indices), int(math.ceil(len(traj_indices) * proportion)))
                traj_indices = traj_indices[:keep]
        shard_id, num_shards = self._get_shard_info()
        if num_shards > 1:
            traj_indices = [idx for idx in traj_indices if idx % num_shards == shard_id]

        if "robot_type" in meta.keys():
            robot_type = infer_robot_type(meta["robot_type"], meta)
        else:
            robot_type = dataset_name
        Handler = get_handler_cls(robot_type, meta)
        handler = Handler(meta=meta, num_views=self.num_views)
        for traj_idx in traj_indices:
            for sample in handler.iter_episode(
                traj_idx,
                num_actions=self.num_actions,
                training=self.training,
                image_aug=self.image_aug,
                lang_aug_map=meta["lang_aug_map"] if "lang_aug_map" in meta.keys() else None,
                action_mode=self.action_mode,
                future_index=self.future_index,
                return_future=self.return_future,
                **self.handler_kwargs,
            ):
                sample["domain_id"] = torch.tensor(DATA_DOMAIN_ID.get(robot_type, 0))
                sample["dataset_name"] = str(dataset_name)
                sample["dataset_tag"] = _dataset_tag(dataset_name)
                idx_for_delta = sample.pop("idx_for_delta", [])
                idx_for_rot_delta = sample.pop("idx_for_rot_delta", [])
                idx_for_mask_proprio = sample.pop("idx_for_mask_proprio", [])
                idx_for_gripper = sample.pop("idx_for_gripper", [])
                gripper_deadzone = sample.pop("gripper_deadzone", 0.0)
                gripper_to_signed = sample.pop("gripper_to_signed", False)
                sample.update(
                    action_slice(
                        sample.pop("abs_trajectory", None),
                        idx_for_delta,
                        idx_for_mask_proprio,
                        idx_for_gripper,
                        gripper_deadzone,
                        gripper_to_signed,
                        idx_for_rot_delta,
                    )
                )
                if self.include_producer_id:
                    sample["producer_id"] = torch.tensor(shard_id, dtype=torch.long)
                yield sample

    def __iter__(self):
        names = list(self.metas.keys())
        if self.data_proportion_map:
            names = [n for n in names if self.data_proportion_map.get(n, 1.0) > 0.0]
            if not names:
                raise RuntimeError("All datasets are disabled by --data_proportions (all are 0).")
        if self.training:
            random.shuffle(names)
        for n in names:
            yield from self._iter_one_dataset(n)
