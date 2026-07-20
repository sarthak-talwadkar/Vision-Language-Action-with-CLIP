"""
data/collect_demos.py
=====================
Collect expert demonstration trajectories in Isaac Sim for training the
CLIP-RT policy.

WHAT ARE DEMONSTRATIONS?
--------------------------
A demonstration is a complete trajectory of (observation, action) pairs
recorded while an expert (human teleoperator or scripted policy) solves a task.
The policy is trained to *imitate* these demonstrations via behaviour cloning.

WHY DO WE NEED DEMONSTRATIONS IF CLIP KNOWS LANGUAGE?
-------------------------------------------------------
CLIP's pretraining teaches semantic alignment between images and text, but it
does NOT know how to output robot joint commands.  The action decoder (the
transformer decoder in policy.py) must be trained to map CLIP embeddings to
continuous motor commands.  This requires labelled (observation, action) pairs.

The CLIP-RT pretrain phase uses Open-X Embodiment (860k robot trajectories)
to train the action decoder broadly.  The fine-tune phase (finetune/) then
adapts the decoder to the specific task distribution using a smaller set of
demonstrations collected in Isaac Sim (typically 100–500 per task).

DATA FORMAT: HDF5
------------------
Each trajectory is stored as a group in an HDF5 file:
  trajectory_0000/
    rgb         : [T, H, W, 3]   uint8    — raw camera frames
    depth       : [T, H, W]      float32  — depth in metres
    state       : [T, STATE_DIM] float32  — EEF pose + gripper
    actions     : [T, ACTION_DIM] float32 — expert actions
    instruction : str            UTF-8    — task language description

HDF5 was chosen because:
  1. Random-access: any timestep can be loaded without reading the whole file.
  2. Compression: lossless gzip reduces storage by 3–5× for depth images.
  3. Torch-compatible: h5py arrays can be sliced directly into numpy arrays,
     then converted to tensors without extra copies.

COLLECTION METHODS
------------------
Two modes are supported:

  1. Scripted policy (default for reproducible data collection):
     A hand-coded controller solves each task deterministically.  This gives
     clean, low-noise demonstrations but only works for simple tasks where
     the solution can be scripted (e.g. "pick up the red cube").

  2. Teleoperation (for complex, contact-rich tasks):
     A human operates the robot via keyboard/SpaceMouse in real time.
     Isaac Sim's teleop interface captures the operator's 6-DOF input and
     converts it to EEF delta actions at 30 Hz.

Usage
-----
    # Collect 500 demos for a pick-and-place task
    python data/collect_demos.py \\
        --task "pick up the red cube and place it in the bin" \\
        --n-demos 500 \\
        --output data/demos/pick_place.hdf5 \\
        --method scripted

    # Collect 50 demos via teleoperation
    python data/collect_demos.py \\
        --task "open the drawer" \\
        --n-demos 50 \\
        --output data/demos/open_drawer.hdf5 \\
        --method teleop
"""

from __future__ import annotations

import argparse
import os
import numpy as np
import h5py
from tqdm import tqdm

from env.isaac_sim import IsaacSimEnv
from env.action_space import ActionSpace, ACTION_DIM, GRIPPER_OPEN, GRIPPER_CLOSE


def collect_demos(
    task_instruction: str,
    n_demos: int,
    output_path: str,
    method: str = "scripted",
    scene_usd: str = "assets/tabletop.usd",
    resolution: tuple[int, int] = (640, 480),
    max_steps_per_episode: int = 300,
):
    """Collect and save expert demonstrations.

    Parameters
    ----------
    task_instruction : str
        Natural language description of the task.  Stored with each trajectory
        and used as the conditioning instruction during training.
    n_demos : int
        Number of successful trajectories to collect.
    output_path : str
        Path to output HDF5 file.
    method : str
        "scripted" or "teleop".
    scene_usd : str
        Isaac Sim USD scene file path.
    resolution : tuple[int, int]
        Camera resolution (width, height).
    max_steps_per_episode : int
        Maximum steps before declaring the episode a failure.
    """
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    env = IsaacSimEnv(scene_usd=scene_usd, resolution=resolution)

    if method == "scripted":
        controller = ScriptedController(task_instruction)
    elif method == "teleop":
        controller = TeleopController()
    else:
        raise ValueError(f"Unknown collection method: {method}")

    # We will compute action statistics over all collected actions for later
    # normalisation in the Dataset.
    all_actions: list[np.ndarray] = []

    collected = 0
    attempts  = 0

    # Open the HDF5 file for writing.  We open it once and append trajectories
    # as groups, so the file stays consistent even if collection is interrupted.
    with h5py.File(output_path, "w") as f:
        # Store metadata at the file level
        f.attrs["task"] = task_instruction
        f.attrs["action_dim"] = ACTION_DIM

        pbar = tqdm(total=n_demos, desc="Collecting demos")

        while collected < n_demos:
            obs = env.reset()
            controller.reset()

            # Buffers for this episode
            rgbs:    list[np.ndarray] = []
            depths:  list[np.ndarray] = []
            states:  list[np.ndarray] = []
            actions: list[np.ndarray] = []

            done = False
            for _ in range(max_steps_per_episode):
                # Get expert action
                action = controller.get_action(obs, task_instruction)

                # Record before stepping (so obs[t] is paired with action[t])
                rgbs.append(obs.rgb.copy())
                depths.append(obs.depth.copy())
                states.append(obs.state.copy())
                actions.append(action.copy())

                obs, done = env.step(action)

                if done:
                    break

            attempts += 1

            if done:
                # Only save successful demonstrations.
                # This is important: failed trajectories teach the policy to
                # reproduce failures, which is harmful for imitation learning.
                traj_name = f"trajectory_{collected:04d}"
                grp = f.create_group(traj_name)

                # Store as arrays.  Use gzip compression for depth images
                # (they have lots of structure and compress well).
                T = len(rgbs)
                H, W = resolution[1], resolution[0]

                grp.create_dataset("rgb",    data=np.stack(rgbs),    dtype="uint8",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("depth",  data=np.stack(depths),  dtype="float32",
                                   compression="gzip", compression_opts=4)
                grp.create_dataset("state",  data=np.stack(states),  dtype="float32")
                grp.create_dataset("actions",data=np.stack(actions), dtype="float32")
                grp.attrs["instruction"] = task_instruction
                grp.attrs["T"] = T

                all_actions.extend(actions)

                collected += 1
                pbar.update(1)
                pbar.set_postfix({"attempts": attempts, "success_rate": f"{collected/attempts:.1%}"})

        pbar.close()

        # Compute and save normalisation statistics.
        # The Dataset will use these to zero-centre and scale-to-unit-variance
        # actions before feeding them to the policy.
        all_actions_arr = np.stack(all_actions)  # [N_total_steps, ACTION_DIM]
        action_mean = all_actions_arr.mean(axis=0)
        action_std  = all_actions_arr.std(axis=0)
        f.attrs["action_mean"] = action_mean
        f.attrs["action_std"]  = action_std

        print(f"\nCollected {collected}/{attempts} successful demos ({collected/attempts:.1%} success rate)")
        print(f"Total timesteps: {len(all_actions)}")
        print(f"Action mean: {action_mean}")
        print(f"Action std:  {action_std}")

    env.close()


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------

class ScriptedController:
    """Hard-coded pick-and-place policy for common Isaac Sim tasks.

    This controller assumes a simple table-top workspace where:
      - The target object's position is known from the scene state.
      - The goal position is fixed (e.g. a bin at a known location).

    For more complex tasks use TeleopController or train an RL policy.

    Parameters
    ----------
    task_description : str
        Used to parse the target object and goal from the task string.
        Simple keyword matching: "pick up the red cube" → target = red cube.
    """

    def __init__(self, task_description: str = ""):
        self.task_description = task_description
        self._phase = "approach"  # approach → grasp → lift → transport → release
        self._step_in_phase = 0

    def reset(self):
        self._phase = "approach"
        self._step_in_phase = 0

    def get_action(self, obs, instruction: str) -> np.ndarray:
        """Return the next scripted action given the current observation.

        The scripted policy works in phases:
          1. Approach: move EEF toward the target object
          2. Grasp:    lower and close gripper
          3. Lift:     raise EEF
          4. Transport: move toward goal
          5. Release:  open gripper

        In a real deployment you would use the object pose from Isaac Sim's
        object tracker rather than hardcoded positions.  For demonstration
        collection this simple open-loop controller suffices.
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if self._phase == "approach":
            action[0] = 0.02  # move forward
            action[2] = -0.01 # lower slightly
            action[6] = GRIPPER_OPEN
            if self._step_in_phase > 20:
                self._phase = "grasp"
                self._step_in_phase = 0

        elif self._phase == "grasp":
            action[2] = -0.02  # lower more
            action[6] = GRIPPER_CLOSE
            if self._step_in_phase > 15:
                self._phase = "lift"
                self._step_in_phase = 0

        elif self._phase == "lift":
            action[2] = 0.03   # lift up
            action[6] = GRIPPER_CLOSE
            if self._step_in_phase > 20:
                self._phase = "transport"
                self._step_in_phase = 0

        elif self._phase == "transport":
            action[1] = 0.02   # move laterally toward goal
            action[6] = GRIPPER_CLOSE
            if self._step_in_phase > 25:
                self._phase = "release"
                self._step_in_phase = 0

        elif self._phase == "release":
            action[2] = -0.01  # lower onto goal surface
            action[6] = GRIPPER_OPEN
            if self._step_in_phase > 10:
                # No more phases — episode should end
                action[6] = GRIPPER_OPEN

        self._step_in_phase += 1
        return action


class TeleopController:
    """Keyboard / SpaceMouse teleoperation controller.

    Reads 6-DOF input from a connected SpaceMouse (or keyboard as fallback)
    and converts it to EEF delta actions.  Requires pyspacemouse or
    keyboard library to be installed.

    The operator presses SPACE to toggle gripper open/close.

    Key bindings (keyboard fallback):
      W/S: +/- dx (forward/backward)
      A/D: +/- dy (left/right)
      Q/E: +/- dz (up/down)
      SPACE: toggle gripper
    """

    def __init__(self, translation_scale: float = 0.02, rotation_scale: float = 0.05):
        self.translation_scale = translation_scale
        self.rotation_scale = rotation_scale
        self._gripper_state = GRIPPER_OPEN

        try:
            import pyspacemouse
            self._use_spacemouse = pyspacemouse.open()
        except ImportError:
            self._use_spacemouse = False
            print("[TeleopController] pyspacemouse not found, using keyboard.")

    def reset(self):
        self._gripper_state = GRIPPER_OPEN

    def get_action(self, obs, instruction: str) -> np.ndarray:
        """Read current device state and return a 7-dim action."""
        action = np.zeros(ACTION_DIM, dtype=np.float32)

        if self._use_spacemouse:
            import pyspacemouse
            state = pyspacemouse.read()
            action[0] =  state.x * self.translation_scale
            action[1] =  state.y * self.translation_scale
            action[2] =  state.z * self.translation_scale
            action[3] =  state.roll  * self.rotation_scale
            action[4] =  state.pitch * self.rotation_scale
            action[5] =  state.yaw   * self.rotation_scale
            # Button 0 toggles gripper
            if state.buttons[0]:
                self._gripper_state = (
                    GRIPPER_CLOSE if self._gripper_state == GRIPPER_OPEN else GRIPPER_OPEN
                )
        else:
            # Keyboard fallback (requires 'keyboard' package)
            try:
                import keyboard
                if keyboard.is_pressed("w"): action[0] = self.translation_scale
                if keyboard.is_pressed("s"): action[0] = -self.translation_scale
                if keyboard.is_pressed("a"): action[1] = self.translation_scale
                if keyboard.is_pressed("d"): action[1] = -self.translation_scale
                if keyboard.is_pressed("q"): action[2] = self.translation_scale
                if keyboard.is_pressed("e"): action[2] = -self.translation_scale
                if keyboard.is_pressed("space"):
                    self._gripper_state = (
                        GRIPPER_CLOSE if self._gripper_state == GRIPPER_OPEN else GRIPPER_OPEN
                    )
            except ImportError:
                pass  # No input — returns zero action

        action[6] = self._gripper_state
        return action


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Collect expert demonstrations in Isaac Sim.")
    parser.add_argument("--task",    type=str, required=True, help="Task instruction string.")
    parser.add_argument("--n-demos", type=int, default=500,   help="Number of demos to collect.")
    parser.add_argument("--output",  type=str, default="data/demos/demos.hdf5", help="HDF5 output path.")
    parser.add_argument("--method",  type=str, default="scripted", choices=["scripted", "teleop"])
    parser.add_argument("--scene",   type=str, default="assets/tabletop.usd", help="Isaac Sim USD scene.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    collect_demos(
        task_instruction=args.task,
        n_demos=args.n_demos,
        output_path=args.output,
        method=args.method,
        scene_usd=args.scene,
    )
