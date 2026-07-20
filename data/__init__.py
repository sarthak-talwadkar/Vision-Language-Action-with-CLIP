"""
data/
=====
Two scripts that handle demonstration data:

  collect_demos.py  — collect expert trajectories in Isaac Sim via
                      teleoperation or a scripted policy, save to HDF5
  dataset.py        — PyTorch Dataset that loads the HDF5 files and
                      yields (image, instruction_tokens, action_chunk) tuples

Import the dataset directly for training:

    from data.dataset import DemoDataset, make_dataloader
"""

from data.dataset import DemoDataset, make_dataloader

__all__ = ["DemoDataset", "make_dataloader"]
