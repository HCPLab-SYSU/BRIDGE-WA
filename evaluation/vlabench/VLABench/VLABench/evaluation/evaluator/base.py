import ctypes
import gc
import json
import os
import numpy as np
import random
import mediapy
import traceback
from tqdm import tqdm
from VLABench.envs import load_env
from VLABench.configs import name2config
from VLABench.utils.utils import euler_to_quaternion, quaternion_to_euler, find_key_by_value


def _release_native_memory():
    """Release Python cycles and return free glibc arenas to the OS."""
    gc.collect()
    try:
        ctypes.CDLL(None).malloc_trim(0)
    except (AttributeError, OSError):
        pass


class Evaluator:
    def __init__(self, 
                 tasks,
                 n_episodes,
                 episode_config=None,
                 max_substeps=1,
                 tolerance=1e-2,
                 metrics=["success_rate"],
                 save_dir=None,
                 visulization=False,
                 eval_unseen=False,
                 unnorm_key='primitive',
                 resume=False,
                 **kwargs
                 ):
        """
        Basic evaluator of policy
        params:
            tasks: list of task names to evaluate, e.g. ["task1", "task2"]
            n_episodes: number of episodes to evaluate in each task
            episode_config: dict or path of config file for episode generation
            max_substeps: maximum number of substeps for env.step
            metrics: list of metrics to evaluate
            save_dir: directory to save the evaluation results
            visulization: whether to visualize the evaluation progress as videos
            eval_unseen: whether to evaluate the unseen object categories
            unnorm_key: the dataset statistics name of the task suite
        """
        if isinstance(episode_config, str):
            with open(episode_config, "r") as f:
                self.episode_config = json.load(f)
        else:self.episode_config = episode_config
        if self.episode_config is None:
            print("Load the task episodes by seeds, instead of episodes")
        else:
            for task in tasks:
                assert len(self.episode_config[task]) >= n_episodes, "The number of episodes should be less than the number of configurations"
        self.eval_tasks = tasks
        self.n_episodes = n_episodes 
        
        self.max_substeps = max_substeps
        self.tolerance = tolerance
        self.target_metrics = metrics
        self.intention_score_threshold = kwargs.get("intention_score_threshold", 0.1)
        self.eval_unseen = eval_unseen
        self.unnorm_key = unnorm_key
        # log, store and visualization
        self.save_dir = save_dir
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
        self.visulization = visulization
        self.resume = resume
        with open(os.path.join(os.getenv("VLABENCH_ROOT"), "configs/task_config.json"), "r") as f:
           self.task_configs = json.load(f)
        
    def evaluate(self, agent):
        """
        Evaluate the agent on all tasks defined in the evaluator.
        """   
        metrics = {}
        for task in self.eval_tasks:
            task_infos = []
            start_episode = 0
            detail_path = None
            if self.save_dir is not None:
                detail_path = os.path.join(self.save_dir, task, "detail_info.json")
            if self.resume and detail_path is not None and os.path.isfile(detail_path):
                with open(detail_path, "r") as f:
                    previous_infos = json.load(f)
                if len(previous_infos) == self.n_episodes:
                    task_infos = previous_infos
                    start_episode = self.n_episodes
                    print(
                        f"[vlabench-evaluator] resume: skipping completed task "
                        f"{task} ({self.n_episodes}/{self.n_episodes})"
                    )
                elif all(
                    info.get("episode_id") == index
                    for index, info in enumerate(previous_infos)
                ):
                    task_infos = previous_infos
                    start_episode = len(previous_infos)
                    print(
                        f"[vlabench-evaluator] resume: continuing task {task} "
                        f"at episode {start_episode}/{self.n_episodes}"
                    )
                else:
                    print(
                        f"[vlabench-evaluator] resume: rerunning incomplete legacy "
                        f"task {task} ({len(previous_infos)}/{self.n_episodes})"
                    )
            max_episode_length = 200
            if self.task_configs.get(find_key_by_value(name2config, task), None):
                if self.task_configs[find_key_by_value(name2config, task)].get("evaluation", {}).get("max_episode_length", None):
                    max_episode_length = self.task_configs[find_key_by_value(name2config, task)]["evaluation"]["max_episode_length"]
                
            for i in tqdm(
                range(start_episode, self.n_episodes),
                initial=start_episode,
                total=self.n_episodes,
                desc=f"Evaluating {task} of {agent.name}",
            ):
                agent.reset()
                kwargs = {
                    "unnorm_key": 'primitive',
                    "max_episode_length": max_episode_length
                }
                if self.episode_config is None:
                    info = self.evaluate_single_episode(agent, task, i, None, seed=42+i, **kwargs)
                else:
                    info = self.evaluate_single_episode(agent, task, i, self.episode_config[task][i], **kwargs)
                task_infos.append(info)
                if detail_path is not None:
                    self._write_json(detail_path, task_infos)
                    
            metric_score = self.compute_metric(task_infos)       
            metrics[task] = metric_score
            
            if self.save_dir is not None:
                if os.path.exists(os.path.join(self.save_dir, "metrics.json")):
                    with open(os.path.join(self.save_dir, "metrics.json"), "r") as f:
                        previous_metrics = json.load(f)
                else:
                    previous_metrics = {}
                previous_metrics[task] = metric_score
                self._write_json(
                    os.path.join(self.save_dir, "metrics.json"),
                    previous_metrics,
                )
                self._write_json(detail_path, task_infos)
            _release_native_memory()
        return metrics

    @staticmethod
    def _write_json(path, value):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary_path = f"{path}.tmp"
        with open(temporary_path, "w") as f:
            json.dump(value, f, indent=4)
        os.replace(temporary_path, path)
        
    def evaluate_single_episode(self, agent, task_name, episode_id, episode_config, seed=42, max_episode_length=200, **kwargs):
        """
        If episode_config is given, the task and scene will load deterministically.
        params:
            agent: policy to evaluate
            task_name: name of the task
            episode_id: id of the episode
            episode_config: configuration of the task
            seed: seed for the random number generator, if episode_config is None
            max_episode_length: maximum length of the episode
        """
        env = None
        video_writer = None
        partial_video_path = None
        success = False
        info = {
            "task": task_name,
            "episode_id": episode_id,
            "success": False,
            "consumed_step": 0,
            "intention_score": 0.0,
            "progress_score": 0.0,
        }
        last_action = None
        i = 0
        try:
            if episode_config is None: # use random seed to ditermine the task
                np.random.seed(seed)
                random.seed(seed)
                env = load_env(task_name, random_init=True, eval=self.eval_unseen, run_mode="eval")
            else:
                env = load_env(task_name, episode_config=episode_config, random_init=False, eval=self.eval_unseen, run_mode="eval")
            env.reset()
            robot_frame = env.get_robot_frame_position()
            while i < max_episode_length:
                observation = env.get_observation(require_pcd=False)
                observation["instruction"] = env.task.get_instruction()
                ee_state = observation["ee_state"]
                observation['robot_frame'] = robot_frame
                if last_action is None:
                    last_action = np.concatenate([ee_state[:3], quaternion_to_euler(ee_state[3:7])])
                observation["last_action"] = last_action
                if self.save_dir is not None and self.visulization:
                    frame = self._video_frame(observation["rgb"])
                    if video_writer is None:
                        video_dir = os.path.join(self.save_dir, task_name, "videos")
                        os.makedirs(video_dir, exist_ok=True)
                        partial_video_path = os.path.join(video_dir, f"{episode_id}.partial.mp4")
                        video_writer = mediapy.VideoWriter(
                            partial_video_path,
                            shape=frame.shape[:2],
                            fps=10,
                        )
                        video_writer.__enter__()
                    video_writer.add_image(frame)
                if agent.control_mode == "ee":
                    pos, euler, gripper_state = agent.predict(observation, **kwargs)
                    last_action = np.concatenate([pos, euler])
                    quat = euler_to_quaternion(*euler)
                    _, action = env.robot.get_qpos_from_ee_pos(physics=env.physics, pos=pos, quat=quat)
                    action = np.concatenate([action, gripper_state])
                elif agent.control_mode == "joint":
                    qpos, gripper_state = agent.predict(observation, **kwargs)
                    action = np.concatenate([qpos, gripper_state])
                else:
                    raise NotImplementedError(f"Control mode {agent.control_mode} is not implemented")
                for _ in range(self.max_substeps):
                    timestep = env.step(action)
                    if timestep.last():
                        success=True
                        break
                    current_qpos = np.array(env.task.robot.get_qpos(env.physics)).reshape(-1)
                    if np.max(current_qpos-np.array(action)[:7]) < self.tolerance \
                        and np.min(current_qpos - np.array(action)[:7]) > -self.tolerance:
                        break
                if success:
                    break
                i += 1
            info["success"] = bool(success)
            info["consumed_step"] = i
            info["intention_score"] = float(env.get_intention_score(threshold=self.intention_score_threshold))
            info["progress_score"] = float(env.get_task_progress())
        except Exception as exc:
            print(f"[vlabench-evaluator] episode failed: task={task_name} episode={episode_id}: {exc}")
            traceback.print_exc()
            info["consumed_step"] = i
            info["error_type"] = type(exc).__name__
            info["error_message"] = str(exc)
        finally:
            if video_writer is not None:
                try:
                    video_writer.close()
                except Exception as exc:
                    print(f"[vlabench-evaluator] failed to close video writer: {exc}")
            if env is not None:
                try:
                    env.close()
                except Exception as exc:
                    print(f"[vlabench-evaluator] failed to close environment: {exc}")

        if partial_video_path is not None and os.path.isfile(partial_video_path):
            if "error_type" in info:
                outcome = f"error_{info['error_type']}"
            else:
                outcome = f"success_{info['success']}"
            final_video_path = os.path.join(
                os.path.dirname(partial_video_path),
                f"{episode_id}_{outcome}_progress_{info['progress_score']:.2f}.mp4",
            )
            os.replace(partial_video_path, final_video_path)
        _release_native_memory()
        return info
        
    def compute_metric(self, infos):
        """
        Compute the metric scores for the evaluation
        param:
            infos: list of episode information
        """
        metric = {}
        for key in self.target_metrics:
            if key == "success_rate": # compute the success rate
                success = [info["success"] for info in infos]
                sucess_rate = np.mean(success)
                metric["success_rate"] = sucess_rate
            elif key == "intention_score":
                intention_score = [info["intention_score"] for info in infos]
                avg_intention_score = np.mean(intention_score)
                metric["intention_score"] = avg_intention_score
            elif key == "progress_score":
                progress_score = [info["progress_score"] for info in infos]
                avg_progress_score = np.mean(progress_score)
                metric["progress_score"] = avg_progress_score
            else:
                raise NotImplementedError(f"Metric {key} is not implemented")
        return metric
    
    @staticmethod
    def _video_frame(frame):
        return np.vstack([np.hstack(frame[:2]), np.hstack(frame[2:4])])
