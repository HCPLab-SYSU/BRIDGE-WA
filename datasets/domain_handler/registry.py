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
from typing import Dict, Type
from .base import DomainHandler

# Handlers
from .lerobot_agibot import AGIBOTLeRobotHandler
from .agiworld import AGIWolrdHandler
from .robomind import RobomindHandler
from .droid import DroidHandler
from .real_world import AIRAgilexHandler, AIRAgilexHQHandler, AIRBotHandler, WidowxAirHandler
from .simulations import (
    BridgeHandler,
    LiberoHandler,
    VLABenchHandler,
    RobotWin2Handler,
    RobotWin2Clean50HDF5Handler,
    RobocasaHumanHandler,
    CalvinHandler,
    RT1Handler,
)
from .lerobotv21 import LeRobotV21Handler
from .lerobot_rlbench import LeRobotRLBenchHandler
from .lerobot_libero import LeRobotLiberoHandler
from .lerobot_vlabench import LeRobotVLABenchHandler
from .lerobot_droid import LeRobotDroidHandler
from .lerobot_dobot import LeRobotDobotNova2Handler
from .lerobot_bridge_wa_franka import LeRobotBridgeWAFrankaHandler
from .x2robot import X2RobotHandler

# 1) Exact registry only (no heuristics)
_REGISTRY: Dict[str, Type[DomainHandler]] = {
    
    # X2Robot
    "x2robot": X2RobotHandler,

    # Lerobot (v2.1 - sim)
    "lift2": LeRobotV21Handler,
    "panda": LeRobotRLBenchHandler,
    "libero-panda": LeRobotLiberoHandler,
    "vlabench-panda": LeRobotVLABenchHandler,
    "droid": LeRobotDroidHandler,
    "dobot-nova2": LeRobotDobotNova2Handler,
    "bridge_wa-franka": LeRobotBridgeWAFrankaHandler,
    
    # LeRobot (parquet)
    "AGIBOT": AGIBOTLeRobotHandler,
    "AGIBOT-challenge": AGIBOTLeRobotHandler,

    # HDF5 (exact)
    "Calvin": CalvinHandler,
    "RT1": RT1Handler,

    # AIR family
    "AIR-AGILEX": AIRAgilexHandler,
    "AIR-AGILEX-HQ": AIRAgilexHQHandler,
    "AIRBOT": AIRBotHandler,
    "widowx-air": WidowxAirHandler,

    # Sim/others
    "Bridge": BridgeHandler,
    "libero": LiberoHandler,
    "VLABench": VLABenchHandler,
    "robotwin2_abs_ee": RobotWin2Handler,
    "robotwin2_clean": RobotWin2Handler,
    "robotwin2_clean50_hdf5": RobotWin2Clean50HDF5Handler,
    "robocasa-human": RobocasaHumanHandler,

    # Robomind
    "robomind-franka": RobomindHandler,
    "robomind-ur": RobomindHandler,
    "robomind-agilex": RobomindHandler,
    "robomind-franka-dual": RobomindHandler,

    # Droid
    "Droid-Left": DroidHandler,
    "Droid-Right": DroidHandler,
    
    
    "agiworld-on-site-pack": AGIWolrdHandler ,
    "agiworld-on-site-pack-extra": AGIWolrdHandler ,
    "agiworld-on-site-conveyor": AGIWolrdHandler ,
    "agiworld-on-site-conveyor-extra": AGIWolrdHandler ,
    "agiworld-on-site-restock": AGIWolrdHandler ,
    "agiworld-on-site-pour": AGIWolrdHandler ,
    "agiworld-on-site-microwave": AGIWolrdHandler ,
    "agiworld-on-site-cloth": AGIWolrdHandler,
    "agiworld-on-site-cloth-2": AGIWolrdHandler
}

def infer_robot_type(dataset_name: str, meta: dict | None = None) -> str:
    robot_name = str(dataset_name or "")
    if robot_name.lower().startswith("dobot xtrainer right arm"):
        return "dobot-nova2"
    if robot_name.lower().startswith("franka fr3 bridge_wa"):
        return "bridge_wa-franka"
    if dataset_name != "panda":
        return dataset_name
    if not meta:
        return "panda"
    feats = meta.get("features", {}) or {}
    action_shape = feats.get("actions", {}).get("shape")
    feature_keys = set(feats.keys())
    root_path = str(meta.get("root_path", "")).lower()
    # Prefer explicit dataset hints before brittle feature-count heuristics.
    if "libero" in root_path:
        return "libero-panda"
    if "vlabench" in root_path:
        return "vlabench-panda"
    if "second_image" in feature_keys:
        return "vlabench-panda"
    if action_shape in ([7], (7,), 7):
        return "libero-panda"
    if "task_index" in feature_keys and int(meta.get("total_tasks", 0) or 0) >= 100:
        return "vlabench-panda"
    return "panda"


def _infer_panda_handler(meta: dict | None) -> Type[DomainHandler]:
    return _REGISTRY[infer_robot_type("panda", meta)]


def get_handler_cls(dataset_name: str, meta: dict | None = None) -> Type[DomainHandler]:
    """Lookup handler. For panda, infer RLBench vs Libero vs VLABench from metadata."""
    resolved = infer_robot_type(dataset_name, meta)
    try:
        return _REGISTRY[resolved]
    except KeyError:
        raise KeyError(
            f"No handler registered for dataset '{dataset_name}' (resolved='{resolved}'). "
            f"Add it to _REGISTRY in datasets/domains/registry.py."
        )
