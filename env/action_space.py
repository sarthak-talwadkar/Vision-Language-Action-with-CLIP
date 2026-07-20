"""
env/action_space.py
===================
Action space definition and normalisation for the CLIP-RT policy.

ACTION REPRESENTATION
---------------------
CLIP-RT uses *continuous* end-effector delta actions, NOT discrete tokens.
(This is different from RT-2, which discretises actions into text-token bins.)

Each 7-dimensional action vector means:
  [0] dx          : end-effector x translation delta (metres, positive = forward)
  [1] dy          : end-effector y translation delta (metres, positive = left)
  [2] dz          : end-effector z translation delta (metres, positive = up)
  [3] droll       : roll rotation delta (radians)
  [4] dpitch      : pitch rotation delta (radians)
  [5] dyaw        : yaw rotation delta (radians)
  [6] gripper     : gripper command, continuous in [-1, 1]
                    values < 0 → close, values > 0 → open
                    In LIBERO the dummy action uses -1 (open)

This EEF_POS encoding is the most common in Open-X Embodiment and is used by
every dataset in the pretrain mixture (see pretrain/configs.py).

WHY EEF DELTAS OVER JOINT ANGLES?
-----------------------------------
End-effector (Cartesian) deltas generalise better across robot morphologies.
A policy trained on a Franka can be adapted to a UR5 because both express
"move 5cm forward" the same way, even though their joint configurations differ.
Joint deltas would require completely separate policies per robot.

ACTION CHUNK FORMAT
-------------------
The full output of CLIPRTPolicy is shape [8, 7]:
  - 8 consecutive timesteps (action chunk)
  - 7 values per timestep

During execution in run_libero_eval_clip_rt.py, each of the 8 sub-actions is
fed to env.step() in sequence, collecting a new observation after each one.

RELATIONSHIP TO action_to_language.json
-----------------------------------------
The docs/action_to_language.json maps 8-dimensional "primitive" action vectors
to natural language descriptions used during CLIP-RT *pretraining*.  The 8th
dimension in that file represents the gripper state as a 0/1 binary flag
(for the pretraining label, not the runtime action).

At inference time the model outputs 7-dim actions (matching the robot API).
The pretraining labels are only used to construct the contrastive pairs that
teach CLIP to align action descriptions to visual motion.

NORMALISATION
-------------
Actions are normalised to [-1, 1] before training and unnormalised for execution.
Statistics (mean, std) are computed per dataset during preprocessing
(see pretrain/preprocess.py and OpenVLA's dataset.py).

The LIBERO fine-tune uses the dataset-specific statistics from the LIBERO
training demonstrations.  Isaac Sim uses custom statistics collected from
the demonstration corpus (see data/collect_demos.py).
"""

from __future__ import annotations

import numpy as np
# NOTE: torch is intentionally NOT imported at module level.  env/__init__.py
# imports this module, and so do scripts that run inside the *Isaac Sim* Python
# environment (e.g. scripts/capture_frame.py) — and Isaac Sim's bundled Python
# does not ship torch.  The only torch touch-point is ActionSpace.unnormalise(),
# which duck-types tensor inputs (see below) instead of importing torch.

# ---------------------------------------------------------------------------
# Dimension constants
# ---------------------------------------------------------------------------

ACTION_DIM   = 7    # [dx, dy, dz, droll, dpitch, dyaw, gripper]
TRANS_DIMS   = 3    # indices 0-2: translation
ROT_DIMS     = 3    # indices 3-5: rotation (roll, pitch, yaw)
GRIPPER_DIM  = 1    # index 6: gripper open/close

# Gripper convention matching LIBERO and OpenVLA
GRIPPER_OPEN  = 1.0
GRIPPER_CLOSE = -1.0

# Safety clamps: actions beyond these values are physically infeasible
# for a Franka Panda and would cause joint-limit violations in Isaac Sim.
TRANS_CLIP   = 0.25   # max translation delta per step, metres
ROT_CLIP     = np.pi  # max rotation delta per step, radians


class ActionSpace:
    """Utility class for action normalisation, clipping, and execution dispatch.

    Parameters
    ----------
    action_mean : np.ndarray, shape [ACTION_DIM]
        Per-dimension mean of actions in the training dataset.
        Used to zero-centre actions before the policy sees them.
    action_std : np.ndarray, shape [ACTION_DIM]
        Per-dimension standard deviation.
        Used to scale actions to unit variance.

    Both statistics are computed by data/collect_demos.py over the full
    demonstration corpus and saved alongside the dataset.
    """

    def __init__(
        self,
        action_mean: np.ndarray | None = None,
        action_std:  np.ndarray | None = None,
    ):
        # Default statistics: zero mean, unit std (no normalisation).
        # In practice these should be computed from your demonstration data.
        self.mean = action_mean if action_mean is not None else np.zeros(ACTION_DIM)
        self.std  = action_std  if action_std  is not None else np.ones(ACTION_DIM)

        # Avoid division by zero for any constant dimension
        self.std = np.where(self.std < 1e-6, 1.0, self.std)

    # ------------------------------------------------------------------
    # Normalise / unnormalise
    # ------------------------------------------------------------------

    def normalise(self, action: np.ndarray) -> np.ndarray:
        """Map raw action to zero-mean unit-variance space.

        Called during dataset construction so the policy always sees
        actions in a consistent scale regardless of joint range.

        Parameters
        ----------
        action : np.ndarray, shape [..., ACTION_DIM]
        """
        return (action - self.mean) / self.std

    def unnormalise(self, normalised: "np.ndarray | torch.Tensor") -> np.ndarray:
        """Invert normalisation before passing actions to the robot API.

        Called in inference.py after the policy predicts normalised actions.

        Accepts either a NumPy array or a torch.Tensor.  We deliberately avoid
        importing torch here (see the module-level note) so this file stays
        importable inside Isaac Sim's Python.  A torch.Tensor is detected by
        duck-typing: only tensors expose ``.detach()``.

        Parameters
        ----------
        normalised : np.ndarray or torch.Tensor, shape [..., ACTION_DIM]
        """
        if hasattr(normalised, "detach"):          # torch.Tensor → NumPy
            normalised = normalised.detach().cpu().numpy()
        normalised = np.asarray(normalised)
        return normalised * self.std + self.mean

    # ------------------------------------------------------------------
    # Safety clipping
    # ------------------------------------------------------------------

    def clip(self, action: np.ndarray) -> np.ndarray:
        """Hard-clip actions to physically safe ranges.

        Translation: max 25 cm per step (Franka max speed ≈ 1.7 m/s at 7 Hz)
        Rotation:    max π rad per step (avoids singularity flip)
        Gripper:     clamp to [-1, 1]
        """
        action = action.copy()
        action[:3] = np.clip(action[:3], -TRANS_CLIP, TRANS_CLIP)
        action[3:6] = np.clip(action[3:6], -ROT_CLIP, ROT_CLIP)
        action[6]   = np.clip(action[6], -1.0, 1.0)
        return action

    # ------------------------------------------------------------------
    # Gripper utilities
    # ------------------------------------------------------------------

    @staticmethod
    def gripper_is_open(action: np.ndarray | list) -> bool:
        """Return True if the gripper command indicates opening."""
        return float(action[6]) > 0.0

    @staticmethod
    def gripper_is_close(action: np.ndarray | list) -> bool:
        """Return True if the gripper command indicates closing."""
        return float(action[6]) <= 0.0

    # ------------------------------------------------------------------
    # No-op / dummy action
    # ------------------------------------------------------------------

    @staticmethod
    def noop() -> np.ndarray:
        """Return a do-nothing action.

        Used during the initial stabilisation steps in each episode:
        the LIBERO simulator drops objects when the episode starts, and
        the robot must wait (~10 steps) for physics to settle before
        acting.  The noop keeps the arm stationary and the gripper open.
        """
        action = np.zeros(ACTION_DIM, dtype=np.float32)
        action[6] = GRIPPER_OPEN  # keep gripper open during wait
        return action

    # ------------------------------------------------------------------
    # String representation for logging
    # ------------------------------------------------------------------

    @staticmethod
    def action_to_str(action: np.ndarray | list) -> str:
        """Human-readable description of an action for logging."""
        a = list(action)
        parts = [
            f"dx={a[0]:+.3f}",
            f"dy={a[1]:+.3f}",
            f"dz={a[2]:+.3f}",
            f"roll={a[3]:+.3f}",
            f"pitch={a[4]:+.3f}",
            f"yaw={a[5]:+.3f}",
            f"grip={'open' if a[6] > 0 else 'close'}({a[6]:+.2f})",
        ]
        return "  ".join(parts)
