import os
import argparse
from pathlib import Path
os.environ.setdefault("MUJOCO_GL", "egl")

if "VLABENCH_ROOT" not in os.environ:
    os.environ["VLABENCH_ROOT"] = str(
        Path(__file__).resolve().parent / "VLABench" / "VLABench"
    )

from VLABench.evaluation.evaluator import Evaluator
from VLABench.evaluation.model.policy.base import RandomPolicy
from VLABench.tasks import *
from VLABench.robots import *

import json_numpy
import collections
import requests
import PIL.Image as Image
import json
from scipy.spatial.transform import Rotation as R
import numpy as np
from typing import Deque, Dict, Iterable, List, Optional, Tuple

try:
    from scripts.summarize_vlabench5_track_breakdown import summarize_eval_dir
except Exception as exc:
    summarize_eval_dir = None
    _SUMMARIZE_IMPORT_ERROR = exc

def quat_to_rotate6d(q: np.ndarray, scalar_first = False) -> np.ndarray:
    return R.from_quat(q, scalar_first = scalar_first).as_matrix()[..., :, :2].reshape(q.shape[:-1] + (6,))

# def fix_pitch_positive(euler):
#     roll = euler[..., 0]
#     pitch = euler[..., 1]
#     yaw = euler[..., 2]

#     mask_flip = pitch < 0
#     pitch[mask_flip] = -pitch[mask_flip]
#     roll[mask_flip] = roll[mask_flip] + np.pi
#     yaw[mask_flip] = yaw[mask_flip] + np.pi

#     roll = (roll + np.pi) % (2 * np.pi) - np.pi
#     yaw  = (yaw  + np.pi) % (2 * np.pi) - np.pi

#     return np.stack([roll, pitch, yaw], axis=-1)

def quat2euler(quat, is_degree=False):
    r = R.from_quat([quat[1], quat[2], quat[3], quat[0]])
    euler_angles = r.as_euler('xyz', degrees=is_degree)  
    return euler_angles

def rotate6D_to_euler(v6: np.ndarray) -> np.ndarray:
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
    euler = R.from_matrix(rot_mats).as_euler('xyz', degrees=False)
    return euler


class ClientModel():
    def __init__(self,
                 host,
                 port,
                 control_mode = 'ee'):

        self.url = f"http://{host}:{port}/act"
        assert control_mode in ['ee', 'joint']
        self.control_mode = control_mode
        self.name = 'hdp'
        self.reset()
        
    def reset(self):
        """
        This is called
        """
        # currently, we dont use historical observation, so we dont need this fc
        
        self.action_plan = collections.deque()
        return None
    
    def _post(self, payload: Dict) -> np.ndarray:
        resp = requests.post(self.url, json=payload)
        resp.raise_for_status()
        data = resp.json()
        
        # try:
        #     resp = requests.post(self.url, json=payload)
        #     resp.raise_for_status()
        #     data = resp.json()
        # except Exception as e:
        #     raise RuntimeError(f"Policy server request failed: {e}") from e

        action = np.array(data["action"])  # shape (T, 10) expected: [pos3, rot6d, grip1]
        if action.ndim != 2 or action.shape[1] < 10:
            raise RuntimeError(f"Unexpected action shape from server: {action.shape}")
        return action

    def predict(self, obs, **kwargs):

        """
        Args:
            obs: (dict) environment observations
        Returns:
            action: (np.array) predicted action
        """
        # Avoid dumping the full cached action plan during long evaluations.
        if not self.action_plan:
            multiview = obs['rgb']  # # np.ndarray with shape (4, 480, 480, 3)
            
            # Match VLABench LeRobot training order: image, second_image, wrist_image.
            # In raw VLABench observations, image/front is camera index 2 and wrist is last.
            main_view = multiview[2]     # LeRobot key: image
            second_view = multiview[0]   # LeRobot key: second_image
            wrist_view = multiview[-1]   # LeRobot key: wrist_image
            
            # proprio
            proprio = obs['ee_state'] # np.ndarray with shape (1, 8)
            ee_pos, ee_quat, gripper = proprio[:3], proprio[3:7], proprio[7:8]
            # VLABench ee_state quaternions are stored as (w, x, y, z).
            ee_6d = np.array(quat_to_rotate6d(ee_quat, scalar_first=True))
            ee_pos -= np.array([0, -0.4, 0.78])
            ee_state = np.concatenate([ee_pos, ee_6d, gripper], axis=0)
            proprio = np.concatenate([ee_state, np.zeros_like(ee_state)], axis=0).copy()

            query = {
                "proprio": json_numpy.dumps(proprio),
                "language_instruction": obs['instruction'],
                "image0": json_numpy.dumps(main_view),
                "image1": json_numpy.dumps(second_view),
                "image2": json_numpy.dumps(wrist_view),
                "domain_id": 8,
                "steps": 10,
            }

            action = self._post(query)

            target_eef = action[:, :3]
            target_euler = rotate6D_to_euler(action[:, 3:9])
            target_act = action[:, 9:10]
            final_action = np.concatenate([target_eef, target_euler, target_act], axis=-1)

            # Queue up the plan
            for row in final_action.tolist():
                self.action_plan.append(row)

        action_predict = np.array(self.action_plan.popleft())
       
        pos, euler, open_close = action_predict[:3], action_predict[3:-1], action_predict[-1]
        open_close = float(open_close) 
        
        # Training labels map 1 -> open (trajectory gripper qpos > 0.03), 0 -> closed.
        if open_close > 0.5:
            gripper_state = np.ones(2) * 0.04
        else:
            gripper_state = np.zeros(2)

        pos = np.array(pos) + np.array([0, -0.4, 0.78])  # transform from world cordinates to robot cordinates
        euler = np.array(euler)
        return pos, euler, gripper_state
    
def get_args():
    parser = argparse.ArgumentParser()
    # parser.add_argument('--tasks', nargs='+', default=None, help="Specific tasks to run, work when eval-track is None")
    parser.add_argument(
        '--eval-track',
        nargs='+',
        default=["track_1_in_distribution"],
        type=str,
        help=(
            "Evaluation track name(s) under VLABench/configs/evaluation/tracks, "
            "or explicit JSON path(s)."
        ),
    )
    parser.add_argument(
        "--n-episode",
        default=10,
        type=int,
        help="Number of episodes to evaluate for every task.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip completed tasks and continue a partially saved task.",
    )
    parser.add_argument('--visulization', action="store_true", default=True, help="Whether to save the visualized episodes")
    parser.add_argument('--no-visulization', action="store_false", dest="visulization")
    parser.add_argument('--metrics', nargs='+', default=["success_rate"], choices=["success_rate", "intention_score", "progress_score"], help="The metrics to evaluate")
    parser.add_argument(
        "--track-source-json",
        default=None,
        type=str,
        help=(
            "Optional sidecar JSON that maps each episode in a mixed track to "
            "its original VLABench track for breakdown metrics."
        ),
    )
    
    parser.add_argument("--host", default='0.0.0.0', help="Your client host ip")
    parser.add_argument("--port", default=8000, type=int, help="Your client port")
    parser.add_argument("--eval_log_dir", default='results/test', type=str, help="Where to log the evaluation results.")
    args = parser.parse_args()
    return args

def resolve_track_path(eval_track: str) -> Tuple[str, str]:
    track_path = Path(eval_track)
    if track_path.suffix == ".json":
        if not track_path.is_absolute():
            track_path = Path.cwd() / track_path
        return track_path.stem, str(track_path)
    track_name = track_path.stem
    return track_name, os.path.join(
        "./VLABench/VLABench",
        "configs/evaluation/tracks",
        f"{track_name}.json",
    )

def resolve_track_source_path(track_json: str, requested: Optional[str]) -> Optional[str]:
    if requested:
        path = Path(requested)
        if not path.is_absolute():
            path = Path.cwd() / path
        return str(path)

    track_path = Path(track_json)
    if not track_path.is_absolute():
        track_path = Path.cwd() / track_path
    candidates = [
        track_path.with_suffix(".sources.json"),
        track_path.with_name(f"{track_path.stem}_sources.json"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    return None

def evaluate(args):
    kwargs = vars(args)
    episode_config = None
    
    for eval_track in args.eval_track:
        track_name, track_json = resolve_track_path(eval_track)
        save_dir = os.path.join(args.eval_log_dir, track_name)
        with open(track_json, "r") as f:
            episode_config = json.load(f)
            tasks = list(episode_config.keys())

        assert isinstance(tasks, list)
        if args.n_episode <= 0:
            raise ValueError("--n-episode must be greater than zero")
        episodes_per_task = args.n_episode
        total_episodes = episodes_per_task * len(tasks)
        print(
            f"[vlabench-client] track={track_name} tasks={len(tasks)} "
            f"episodes_per_task={episodes_per_task} total_episodes={total_episodes}"
        )

        evaluator = Evaluator(
            tasks=tasks,
            n_episodes=episodes_per_task,
            episode_config=episode_config,
            max_substeps=20,
            tolerance=1e-2,
            save_dir=save_dir,
            visulization=args.visulization,
            metrics=args.metrics,
            resume=args.resume,
        )

        policy = ClientModel(host=kwargs['host'], port=kwargs['port'])
        # policy = RandomPolicy(None)

        result = evaluator.evaluate(policy)
        

        # average score
        totals = {
            "success_rate": 0.0,
            "intention_score": 0.0,
            "progress_score": 0.0
        }
        count = len(result)
        for item in result.values():
            for key in totals:
                totals[key] += item.get(key, 0.0)

        averages = {key: total / count for key, total in totals.items()}

        print("average:")
        for key, avg in averages.items():
            print(f"{key}: {avg:.4f}")
        
        # save
        result["averages"] = averages
        result_path = os.path.join(save_dir, "evaluation_result.json")
        with open(result_path, "w") as f:
            json.dump(result, f)

        source_json = resolve_track_source_path(track_json, args.track_source_json)
        if source_json:
            if summarize_eval_dir is None:
                print(f"track breakdown skipped: failed to import summarizer: {_SUMMARIZE_IMPORT_ERROR}")
                continue
            try:
                breakdown = summarize_eval_dir(
                    save_dir,
                    source_json,
                    run_name=track_name,
                    metrics=args.metrics,
                    output_json=os.path.join(save_dir, "track_breakdown.json"),
                    output_csv=os.path.join(save_dir, "track_breakdown.csv"),
                )
                result["track_breakdown"] = breakdown
                with open(result_path, "w") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                print(f"track breakdown: {os.path.join(save_dir, 'track_breakdown.json')}")
            except Exception as exc:
                print(f"track breakdown skipped: {exc}")

if __name__ == "__main__":
    args = get_args()
    evaluate(args)
