r"""
scripts/make_synthetic_demos.py
===============================
Write a tiny synthetic HDF5 demonstration file in the exact schema that
data/dataset.py::DemoDataset expects, so train.py can be smoke-tested without
collecting real demonstrations.

Schema produced (matches collect_demos.py):
    file.attrs["action_mean"]  [7] float
    file.attrs["action_std"]   [7] float
    <group "traj_XXXX">
        .attrs["T"]           int   — trajectory length
        .attrs["instruction"] str
        ["rgb"]     [T, H, W, 3] uint8
        ["actions"] [T, 7]       float32
        ["depth"]   [T, H, W]    float32   (needed for --use-3d)

The RGB/depth frames are tiled from the real Isaac capture
(outputs/capture/{frame.png, depth.npy}) so the images/depth are realistic;
the ACTIONS are random — this file is only for exercising the training loop
(shapes, loss, backprop), not for learning anything meaningful.

Run:
    .venv\Scripts\python.exe scripts\make_synthetic_demos.py
"""

import sys
from pathlib import Path

import numpy as np
import h5py
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
CAP = REPO_ROOT / "outputs" / "capture"
OUT = REPO_ROOT / "outputs" / "synthetic_demos"

N_TRAJ = 2
T = 10          # steps per trajectory
INSTRUCTION = "pick up the red cube"


def main() -> int:
    if not (CAP / "frame.png").is_file():
        print(f"Need {CAP/'frame.png'} + depth.npy — run capture_frame.py first.")
        return 1

    rgb = np.asarray(Image.open(CAP / "frame.png").convert("RGB"), dtype=np.uint8)  # [H,W,3]
    depth = np.load(CAP / "depth.npy").astype(np.float32)                            # [H,W]
    H, W = depth.shape
    print(f"Base frame: rgb {rgb.shape}, depth {depth.shape}")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "demo.hdf5"
    rng = np.random.default_rng(0)

    with h5py.File(path, "w") as f:
        # Zero-mean / unit-std → dataset normalisation is a no-op (fine for a test).
        f.attrs["action_mean"] = np.zeros(7, dtype=np.float32)
        f.attrs["action_std"] = np.ones(7, dtype=np.float32)

        for i in range(N_TRAJ):
            g = f.create_group(f"traj_{i:04d}")
            g.attrs["T"] = T
            g.attrs["instruction"] = INSTRUCTION
            # Tile the single real frame across all timesteps.
            g.create_dataset("rgb", data=np.broadcast_to(rgb, (T, H, W, 3)).copy())
            g.create_dataset("depth", data=np.broadcast_to(depth, (T, H, W)).copy())
            # Random small EEF deltas + gripper in [-1, 1].
            actions = np.zeros((T, 7), dtype=np.float32)
            actions[:, :3] = rng.uniform(-0.05, 0.05, size=(T, 3))   # translations (m)
            actions[:, 3:6] = rng.uniform(-0.2, 0.2, size=(T, 3))    # rotations (rad)
            actions[:, 6] = rng.choice([-1.0, 1.0], size=T)          # gripper
            g.create_dataset("actions", data=actions)

    size_mb = path.stat().st_size / 1e6
    print(f"Wrote {path}  ({N_TRAJ} trajectories x {T} steps, {size_mb:.1f} MB)")
    print(f"  -> train with:  --demos {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
