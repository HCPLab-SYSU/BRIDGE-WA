import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

os.environ.setdefault(
    "VLABENCH_ROOT",
    str(Path(__file__).resolve().parent / "VLABench" / "VLABench"),
)

from VLABench.evaluation.evaluator import base as evaluator_module


class _Task:
    @staticmethod
    def get_instruction():
        return "test instruction"


class _FailingEnv:
    def __init__(self):
        self.task = _Task()
        self.closed = False

    def reset(self):
        return None

    def get_robot_frame_position(self):
        return np.zeros(3)

    def get_observation(self, require_pcd=False):
        del require_pcd
        return {
            "rgb": np.zeros((4, 2, 2, 3), dtype=np.uint8),
            "ee_state": np.array([0, 0, 0, 1, 0, 0, 0, 0], dtype=float),
        }

    def step(self, action):
        del action
        raise RuntimeError("simulated unstable physics")

    def close(self):
        self.closed = True


class _Agent:
    name = "test-agent"
    control_mode = "joint"

    def reset(self):
        return None

    def predict(self, observation, **kwargs):
        del observation, kwargs
        return np.zeros(7), np.zeros(2)


class _Writer:
    def __init__(self, path, shape, fps):
        del shape, fps
        self.path = Path(path)

    def __enter__(self):
        self.path.write_bytes(b"partial video")
        return self

    def add_image(self, image):
        del image

    def close(self):
        return None


def _bare_evaluator(save_dir, *, resume=False, n_episodes=1):
    evaluator = evaluator_module.Evaluator.__new__(evaluator_module.Evaluator)
    evaluator.eval_unseen = False
    evaluator.max_substeps = 1
    evaluator.tolerance = 1e-2
    evaluator.intention_score_threshold = 0.1
    evaluator.save_dir = str(save_dir)
    evaluator.visulization = True
    evaluator.resume = resume
    evaluator.n_episodes = n_episodes
    evaluator.target_metrics = ["success_rate", "intention_score", "progress_score"]
    evaluator.task_configs = {}
    return evaluator


class EvaluatorLifecycleTest(unittest.TestCase):
    def test_physics_error_is_counted_and_environment_is_closed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            env = _FailingEnv()
            evaluator = _bare_evaluator(temp_dir)
            with (
                mock.patch.object(evaluator_module, "load_env", return_value=env),
                mock.patch.object(evaluator_module.mediapy, "VideoWriter", _Writer),
            ):
                info = evaluator.evaluate_single_episode(
                    _Agent(), "task", 0, episode_config={}
                )

            self.assertTrue(env.closed)
            self.assertFalse(info["success"])
            self.assertEqual(info["episode_id"], 0)
            self.assertEqual(info["error_type"], "RuntimeError")
            self.assertEqual(info["intention_score"], 0.0)
            self.assertEqual(info["progress_score"], 0.0)
            videos = list((Path(temp_dir) / "task" / "videos").glob("*.mp4"))
            self.assertEqual(len(videos), 1)
            self.assertIn("error_RuntimeError", videos[0].name)

    def test_resume_skips_a_completed_task(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            save_dir = Path(temp_dir)
            task_dir = save_dir / "task"
            task_dir.mkdir()
            infos = [
                {
                    "task": "task",
                    "success": value,
                    "consumed_step": 1,
                    "intention_score": float(value),
                    "progress_score": float(value),
                }
                for value in (True, False)
            ]
            (task_dir / "detail_info.json").write_text(json.dumps(infos))

            evaluator = _bare_evaluator(save_dir, resume=True, n_episodes=2)
            evaluator.eval_tasks = ["task"]
            evaluator.episode_config = {"task": [{}, {}]}
            evaluator.evaluate_single_episode = mock.Mock(
                side_effect=AssertionError("completed task must not rerun")
            )
            with mock.patch.object(evaluator_module, "find_key_by_value", return_value="task"):
                metrics = evaluator.evaluate(_Agent())

            evaluator.evaluate_single_episode.assert_not_called()
            self.assertEqual(metrics["task"]["success_rate"], 0.5)
            self.assertEqual(metrics["task"]["progress_score"], 0.5)


if __name__ == "__main__":
    unittest.main()
