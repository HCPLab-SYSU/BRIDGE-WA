from .base import DomainHandler
from ..utils import read_video_to_frames
from mmengine import fileio
import json
import torch
import numpy as np
from scipy.interpolate import interp1d
import random
from PIL import Image

class X2RobotHandler(DomainHandler):
    """
    X2Robot HDF5 handler.
    """

    dataset_name = "x2robot"
    CAMERA_VIEW = ["faceImg", "leftImg", "rightImg"]
    ACTION_KEY = ["follow_left_joint_pos", 
                  "follow_right_joint_pos",
                  "follow_left_gripper", 
                  "follow_right_gripper"]
    idx_for_delta = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
    idx_for_mask_proprio = [14, 15]
    def iter_episode(self, traj_idx, 
                     *, num_actions, 
                     training, 
                     image_aug, 
                     lang_aug_map, 
                     **kwargs):
        
        item = self.meta["datalist"][traj_idx]
        if "top_path" in self.meta: item["path"] = fileio.join_path(self.meta["top_path"], item["path"])
        data_dir = fileio.join_path(item["path"], item["name"])
        ins = item["instruction"]

        action_file = fileio.join_path(data_dir, item["name"] + ".json")
        with open(action_file, "r") as f:  action_data = json.load(f)["data"]
        images = [read_video_to_frames(
            fileio.join_path(data_dir, f"{vkey}.mp4"))
            for vkey in self.CAMERA_VIEW]
        image_mask = torch.ones(self.num_views, dtype=torch.bool)

        all_action = {key: [] for key in self.ACTION_KEY}
        for frame_data in action_data:
            for key in self.ACTION_KEY:
                if not isinstance(frame_data[key], list): 
                    frame_data[key] = [frame_data[key]]
                all_action[key].append(np.array(frame_data[key]))
        all_action = {key: np.stack(all_action[key]) for key in self.ACTION_KEY}
        all_action = np.concatenate(
            [np.asarray(all_action[key]) for key in self.ACTION_KEY], axis=-1)
        
        assert all_action.shape[0] == len(images[0]), \
            f"action length {all_action.shape[0]} != image length {len(images[0])}"
        freq = 30.0; qdur = 1.0; t = np.arange(all_action.shape[0], dtype=np.float64) / freq
        idxs = list(range(0, all_action.shape[0] - 30))
        if training: random.shuffle(idxs)
        all_action = interp1d(t, all_action,  axis=0, bounds_error=False, 
                              fill_value=(all_action[0], all_action[-1]))
        
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