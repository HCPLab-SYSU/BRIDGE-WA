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
import numpy as np, torch, random
from mmengine import fileio
from scipy.interpolate import interp1d
from ..utils import read_video_to_frames, read_parquet
from PIL import Image
from .base import DomainHandler

class LeRobotV21Handler(DomainHandler):

    # adjust this hyper-parameters according to your need
    CAMERA_VIEW = ["video.top_camera_view", "video.left_camera_view", "video.right_camera_view"]
    ACTION_KEY = ["action.joints", "action.gripper", "action.base_delta"] # 12 + 2 + 3
    idx_for_delta = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
    idx_for_mask_proprio = [12, 13, 14, 15, 16]
    ###########################################
    
    
    def iter_episode(self, traj_idx: int, *, num_actions: int, training: bool,
                     image_aug, lang_aug_map: dict | None, **kwargs):
        item = self.meta["datalist"][traj_idx]
        
        episode_index = item["episode_index"]
        episode_chunk = episode_index // self.meta["chunks_size"]
        data_path = fileio.join_path(self.meta["root_path"], self.meta["data_path"]).format(
            episode_chunk=episode_chunk, episode_index=episode_index
        )
        images = [read_video_to_frames(
            fileio.join_path(self.meta["root_path"], self.meta["video_path"]).format(
            episode_chunk=episode_chunk, episode_index=episode_index, video_key = vkey
        ))
        for vkey in self.CAMERA_VIEW]

        image_mask = torch.ones(self.num_views, dtype=torch.bool)
        data = read_parquet(data_path)
        
        # do prefix sum for action.base_delta
        # data['action.base'] = np.cumsum(np.asarray(data['action.base_delta']), axis=0)
        
        all_action = np.concatenate(
            [np.asarray(data[action_key]) for action_key in self.ACTION_KEY], axis=-1)
        all_action = np.concatenate(
            [all_action[:1], all_action[:-1]], axis=0
        ) # pad the first action
        
        freq = 30.0; qdur = 1.0; t = np.arange(all_action.shape[0], dtype=np.float64) / freq
        idxs = list(range(1, all_action.shape[0] - 30))
        
        if training: random.shuffle(idxs)
        all_action = interp1d(t, all_action,  axis=0, bounds_error=False, 
                              fill_value=(all_action[0], all_action[-1]))
        
        ins = item["tasks"][0]
        for idx in idxs:
            imgs = [] 
            for v in range(min(self.num_views, len(images))):
                imgs.append(image_aug(Image.fromarray(images[v][idx])))
            while len(imgs) < self.num_views: imgs.append(torch.zeros_like(imgs[0]))
            image_input = torch.stack(imgs, 0)
            cur = t[idx]
            q = np.linspace(cur, min(cur + qdur, float(t.max())), num_actions + 1, dtype=np.float32)
            cur_action = torch.tensor(all_action(q))
            if (cur_action[1]-cur_action[0]).abs().max() < 1e-5: continue
            
            ### pad the action to max_action_dim
            cur_action = torch.cat([
                cur_action,
                torch.zeros((cur_action.shape[0], 20 - cur_action.shape[1]))
            ], dim=-1)
            
            if lang_aug_map is not None and ins in lang_aug_map: ins = random.choice(lang_aug_map[ins])
            
            yield {
                "language_instruction": ins,
                "image_input": image_input,
                "image_mask": image_mask,
                "abs_trajectory": cur_action.float(),
                "idx_for_delta": self.idx_for_delta,
                "idx_for_mask_proprio": self.idx_for_mask_proprio
            }
