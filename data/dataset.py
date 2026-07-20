"""
data/dataset.py
===============
PyTorch Dataset that loads HDF5 demonstration files and yields batches of
(preprocessed_image, instruction_tokens, action_chunk) for training.

DATASET DESIGN DECISIONS
--------------------------

1. Action Chunking — Why slice the trajectory into overlapping chunks?
   During training, the policy must learn to predict 8 *consecutive* actions
   from a single observation.  So for a trajectory of T steps, we can create
   T - CHUNK_SIZE + 1 training examples by sliding a window of size 8 over
   the action sequence.

   Example: T=50, CHUNK_SIZE=8
     Window at t=0 → actions[0:8], obs=frame[0]
     Window at t=1 → actions[1:9], obs=frame[1]
     ...
     Window at t=42 → actions[42:50], obs=frame[42]

   This gives many training examples per trajectory and trains the policy
   to predict reasonable futures from any point in the trajectory.

2. Action Normalisation — Why normalise actions?
   Raw EEF delta actions have very different scales:
     translations: O(0.01 – 0.2)  metres
     rotations:    O(0.08 – 1.57) radians
     gripper:      O(-1 – 1)      (binary-ish)

   Without normalisation the MSE loss would be dominated by the largest-
   magnitude dimensions (rotations), causing the policy to underfittranslations.
   Normalising to zero-mean unit-variance balances the loss across all dims.

3. Data Augmentation — Why augment?
   With 500 demonstrations, the model can overfit to specific lighting,
   textures, or camera positions seen in the training data.  Colour jitter,
   random crop, and noise augmentation reduce this overfitting and improve
   transfer to slightly different real-world conditions.

4. Multi-task — Why support multiple HDF5 files?
   CLIP-RT can be trained on multiple tasks simultaneously.  Each task has
   its own instruction, but they share the same CLIP encoder and action
   decoder.  Multi-task training encourages the policy to generalise across
   tasks via the shared CLIP embedding space.

References
----------
- Zhao et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware." RSS 2023. (ACT action chunking)
- OpenVLA dataset.py (Open-X data pipeline, which this file is modelled on)
"""

from __future__ import annotations

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import h5py
from pathlib import Path
from PIL import Image

from env.action_space import ActionSpace, ACTION_DIM


# Number of consecutive actions predicted per forward pass.
# Must match CLIPRTPolicy.action_chunk.
CHUNK_SIZE = 8


class DemoDataset(Dataset):
    """PyTorch Dataset for CLIP-RT action chunking training.

    Each sample is a tuple:
      (image_tensor, instruction_tokens, action_chunk, action_mask)

    image_tensor    : [3, H, W]           float32  — preprocessed for CLIP
    instruction_tokens : [context_length] int64    — tokenised CLIP-RT prompt
    action_chunk    : [CHUNK_SIZE, ACTION_DIM] float32 — normalised actions
    action_mask     : [CHUNK_SIZE]        bool     — valid steps (False = padding)

    The action_mask is needed because the last few windows in a trajectory are
    shorter than CHUNK_SIZE (e.g. the last window might only have 3 steps before
    the episode ends).  Padded steps are masked out of the training loss.

    Parameters
    ----------
    hdf5_paths : list[str | Path]
        Paths to HDF5 demo files produced by collect_demos.py.
        Each file may contain multiple trajectories (tasks) as groups.
    preprocess : callable
        CLIP image preprocessing transform (from CLIPRTEncoder.preprocess).
        Converts a PIL image to a normalised [3, H, W] tensor.
    tokenizer : callable
        CLIP-RT tokeniser (from CLIPRTEncoder.tokenizer).
        Converts a string to an integer token tensor.
    chunk_size : int
        Number of action steps per training example.
    augment : bool
        Whether to apply data augmentation to images during training.
    use_3d : bool
        If True, each sample additionally yields:
          depth      : [H, W] float32   — depth frame (metres) at t_start,
                       used by train.py to build the Scene3DFeatureField;
          raw_tokens : [context_length] int64 — the instruction tokenised
                       WITHOUT the motion-question prompt.  The 3D field is
                       queried with object-phrase embeddings (what the gate
                       experiment validated); the motion prompt is only for
                       the action decoder.
        Requires demos collected with depth (collect_demos.py always saves it).
    """

    def __init__(
        self,
        hdf5_paths: list[str | Path],
        preprocess,
        tokenizer,
        chunk_size: int = CHUNK_SIZE,
        augment: bool = True,
        use_3d: bool = False,
    ):
        self.chunk_size = chunk_size
        self.preprocess = preprocess
        self.tokenizer  = tokenizer
        self.augment    = augment
        self.use_3d     = use_3d

        # Index: list of (hdf5_path, group_name, start_timestep) tuples.
        # One entry per valid sliding-window position in each trajectory.
        self._index: list[tuple[str, str, int]] = []

        # Action normalisation statistics (pooled across all files/tasks)
        all_means: list[np.ndarray] = []
        all_stds:  list[np.ndarray] = []

        for p in hdf5_paths:
            p = str(p)
            with h5py.File(p, "r") as f:
                mean = f.attrs.get("action_mean", np.zeros(ACTION_DIM))
                std  = f.attrs.get("action_std",  np.ones(ACTION_DIM))
                all_means.append(mean)
                all_stds.append(std)

                for group_name in f.keys():
                    grp = f[group_name]
                    T   = int(grp.attrs["T"])
                    # Create one window starting at each valid timestep.
                    # The last valid start is T-1 (1-step window, padded to chunk_size).
                    for t in range(T):
                        self._index.append((p, group_name, t))

        # Pool statistics: simple mean across files.  In production you would
        # compute these over the full concatenated dataset.
        self._action_space = ActionSpace(
            action_mean=np.mean(all_means, axis=0),
            action_std =np.mean(all_stds,  axis=0),
        )

        # Augmentation pipeline: photometric only (colour jitter).
        #
        # NO GEOMETRIC AUGMENTATION.  A horizontal flip mirrors the image but
        # NOT the action labels: a demo moving the arm right (+dy) would be
        # paired with a mirrored image where the target is on the LEFT —
        # teaching the policy the opposite lateral motion on every flipped
        # sample.  It also breaks lateralised instructions ("left of the
        # bowl") and would desynchronise RGB from depth in the 3D branch.
        # Flip augmentation is only sound if dy/droll/dyaw are negated and
        # the instruction is mirrored too — not worth the complexity here.
        self._aug_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ]) if augment else None

        print(
            f"DemoDataset: {len(self._index)} training samples from "
            f"{len(hdf5_paths)} file(s) across "
            f"{sum(1 for p,g,t in self._index if t == 0)} trajectories."
        )

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> tuple:
        """Load and return one training sample.

        We open the HDF5 file inside __getitem__ rather than in __init__
        because HDF5 file handles are not picklable, and DataLoader uses
        multiple worker processes (each needs its own file handle).
        """
        hdf5_path, group_name, t_start = self._index[idx]

        with h5py.File(hdf5_path, "r") as f:
            grp = f[group_name]
            T   = int(grp.attrs["T"])

            # ----------------------------------------------------------------
            # 1. Load the RGB frame at the start of this window
            # ----------------------------------------------------------------
            rgb_np = grp["rgb"][t_start]  # [H, W, 3] uint8
            pil_image = Image.fromarray(rgb_np).convert("RGB")

            # Apply augmentation before CLIP preprocessing.
            if self._aug_transform is not None:
                pil_image = self._aug_transform(pil_image)

            # CLIP preprocessing: resize → centre-crop → normalise
            image_tensor = self.preprocess(pil_image)  # [3, H, W] float32

            # ----------------------------------------------------------------
            # 2. Load the task instruction and tokenise it
            # ----------------------------------------------------------------
            instruction = grp.attrs["instruction"]
            # h5py returns str for attrs written from Python str, but bytes
            # for files written with older h5py / fixed-length dtypes.
            if isinstance(instruction, bytes):
                instruction = instruction.decode("utf-8")

            # Wrap in the CLIP-RT motion-question prompt
            prompted = (
                f"what motion should the robot arm perform to complete "
                f"the instruction '{instruction}'?"
            )
            tokens = self.tokenizer(prompted).squeeze(0)  # [context_length]

            # ----------------------------------------------------------------
            # 3. Load the action chunk (sliding window of size CHUNK_SIZE)
            # ----------------------------------------------------------------
            t_end = min(t_start + self.chunk_size, T)
            actual_len = t_end - t_start

            # Raw actions from the HDF5 file
            raw_actions = grp["actions"][t_start:t_end]  # [actual_len, ACTION_DIM]

            # Normalise to zero-mean unit-variance
            norm_actions = self._action_space.normalise(raw_actions)  # [actual_len, ACTION_DIM]

            # Pad to CHUNK_SIZE if near the end of the trajectory.
            # Padding with zeros is safe because the action_mask will prevent
            # these from contributing to the training loss.
            action_chunk = np.zeros((self.chunk_size, ACTION_DIM), dtype=np.float32)
            action_chunk[:actual_len] = norm_actions

            # Boolean mask: True for valid steps, False for padding
            action_mask = np.zeros(self.chunk_size, dtype=bool)
            action_mask[:actual_len] = True

            # ----------------------------------------------------------------
            # 4. (use_3d only) depth frame + raw instruction tokens
            # ----------------------------------------------------------------
            if self.use_3d:
                depth_np = np.asarray(grp["depth"][t_start], dtype=np.float32)  # [H, W]
                raw_tokens = self.tokenizer(instruction).squeeze(0)             # [ctx_len]

        base = (
            image_tensor,                               # [3, H, W]   float32
            tokens,                                     # [ctx_len]   int64
            torch.tensor(action_chunk, dtype=torch.float32),  # [C, 7]
            torch.tensor(action_mask,  dtype=torch.bool),     # [C]
        )
        if not self.use_3d:
            return base
        return base + (
            torch.from_numpy(depth_np),                 # [H, W]      float32
            raw_tokens,                                 # [ctx_len]   int64
        )

    @property
    def action_space(self) -> ActionSpace:
        """Return the action normalisation statistics for the dataset."""
        return self._action_space


def make_dataloader(
    hdf5_paths: list[str | Path],
    preprocess,
    tokenizer,
    batch_size: int = 32,
    num_workers: int = 4,
    augment: bool = True,
    shuffle: bool = True,
    use_3d: bool = False,
) -> tuple[DataLoader, ActionSpace]:
    """Convenience function to build a DataLoader from demo HDF5 files.

    Parameters
    ----------
    hdf5_paths : list[str]
        Paths to HDF5 files from collect_demos.py.
    preprocess : callable
        CLIP preprocessing transform.
    tokenizer : callable
        CLIP-RT tokeniser.
    batch_size : int
        Training batch size.  32 fits in ~20 GB GPU memory for ViT-H.
    num_workers : int
        DataLoader worker processes.  4 is safe for most systems.
        Use 0 for debugging (no subprocesses → easier to catch errors).
    augment : bool
        Whether to apply image augmentation.
    shuffle : bool
        Shuffle the training set each epoch.

    Returns
    -------
    dataloader : DataLoader
    action_space : ActionSpace
        Normalisation statistics for use in inference.py (to unnormalise
        model outputs before sending to the robot).
    """
    dataset = DemoDataset(hdf5_paths, preprocess, tokenizer, augment=augment, use_3d=use_3d)
    loader  = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,           # speeds up CPU→GPU transfer
        drop_last=True,            # ensures all batches are full size
        persistent_workers=(num_workers > 0),
    )
    return loader, dataset.action_space
