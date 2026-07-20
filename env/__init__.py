"""
env/
====
Two modules that define the robot-environment interface:

  action_space.py  — discrete token definitions + continuous action encoding
  isaac_sim.py     — Isaac Sim RGB-D observation pipeline + action executor

Import the environment directly for use in train/inference scripts:

    from env import IsaacSimEnv, ActionSpace, ACTION_DIM
"""

from env.action_space import ActionSpace, ACTION_DIM, GRIPPER_OPEN, GRIPPER_CLOSE
from env.isaac_sim import IsaacSimEnv

__all__ = [
    "ActionSpace",
    "ACTION_DIM",
    "GRIPPER_OPEN",
    "GRIPPER_CLOSE",
    "IsaacSimEnv",
]
