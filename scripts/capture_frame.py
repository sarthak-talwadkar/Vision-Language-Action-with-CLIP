r"""
scripts/capture_frame.py
=========================
Capture one RGB-D frame from the Isaac Sim scene for the alignment gate test.

This is the bridge between Isaac Sim and scripts/smoke_alignment_test.py:
the synthetic-scene gate validated the dense-feature pipeline (clearclip mode,
ViT-H), but the decisive question is whether it transfers to YOUR rendered
scenes — ray-traced lighting, textured objects, the robot arm in frame.

MUST run inside the Isaac Sim Python environment (omni.isaac.* imports):

    /path/to/isaac-sim/python.sh scripts/capture_frame.py --scene assets/tabletop.usd
    # Windows: <isaac-sim>\python.bat scripts\capture_frame.py ...

Outputs (default outputs/capture/):
    frame.png      — RGB frame, input for --image
    depth.npy      — float32 metres, input for --depth
    depth_vis.png  — normalised depth visualisation for eyeballing
    (camera intrinsics printed to console if the sensor exposes them)

Then, from the regular training venv:

    python scripts/smoke_alignment_test.py \
        --image outputs/capture/frame.png \
        --depth outputs/capture/depth.npy \
        --model ViT-H-14-378-quickgelu --pretrained dfn5b \
        --queries "a red cube,a robot gripper,a wooden table"

No ground-truth boxes exist for real frames — judge the saved overlays by
eye: the hot region should sit ON the queried object for clearclip/csa.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from env.isaac_sim import IsaacSimEnv  # noqa: E402
from env.action_space import ActionSpace  # noqa: E402


def main():
    p = argparse.ArgumentParser(description="Capture one RGB-D frame from Isaac Sim.")
    p.add_argument("--scene", type=str, default="assets/tabletop.usd")
    p.add_argument("--out", type=str, default="outputs/capture")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--settle-steps", type=int, default=30,
                   help="No-op steps after reset so objects finish settling "
                        "and the render pipeline produces a clean frame.")
    p.add_argument("--gui", action="store_true",
                   help="Run with the Isaac Sim viewport (default headless).")
    args = p.parse_args()

    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    env = IsaacSimEnv(
        scene_usd=args.scene,
        resolution=(args.width, args.height),
        headless=not args.gui,
        # reset() advances physics this many ticks (no render) so objects settle
        # and the RTX denoiser warms up, THEN renders one clean frame.  Doing the
        # settle inside reset() avoids an expensive orchestrator render per step.
        settle_steps=args.settle_steps,
    )
    obs = env.reset()

    # ---- save RGB ----
    from PIL import Image
    rgb = np.asarray(obs.rgb, dtype=np.uint8)
    if rgb.shape[-1] == 4:  # some Isaac Sim versions return RGBA
        rgb = rgb[..., :3]
    Image.fromarray(rgb).save(out_dir / "frame.png")

    # ---- save depth ----
    depth = np.asarray(obs.depth, dtype=np.float32)
    np.save(out_dir / "depth.npy", depth)

    # Visualisation: normalise valid depths to 0..255 for a quick sanity look
    # (is the depth aligned with the RGB?  any all-zero/inf regions?)
    valid = np.isfinite(depth) & (depth > 0)
    vis = np.zeros_like(depth)
    if valid.any():
        lo, hi = depth[valid].min(), depth[valid].max()
        vis[valid] = (depth[valid] - lo) / max(hi - lo, 1e-6)
    Image.fromarray((vis * 255).astype(np.uint8)).save(out_dir / "depth_vis.png")

    print(f"\nSaved to {out_dir}:")
    print(f"  frame.png      {rgb.shape}")
    print(f"  depth.npy      {depth.shape}  "
          f"range [{depth[valid].min():.3f}, {depth[valid].max():.3f}] m, "
          f"{100 * (~valid).mean():.1f}% invalid px" if valid.any()
          else "  depth.npy      WARNING: no valid depth values!")
    print("  depth_vis.png  (eyeball: should look like a grayscale render "
          "aligned with frame.png)")

    # ---- camera intrinsics ----
    # Real pinhole intrinsics extracted from the camera prim, saved next to the
    # depth.  smoke_alignment_test.py auto-loads this intrinsics.json when you
    # pass --depth from this folder, so its 3D feature field is metrically
    # correct (no more fx=fy=500 guess).
    import json
    K = env.get_intrinsics()
    with open(out_dir / "intrinsics.json", "w") as f:
        json.dump(K, f, indent=2)
    print(f"\nCamera intrinsics (source: {K.get('source')}):")
    print(f"  fx={K['fx']:.1f}  fy={K['fy']:.1f}  cx={K['cx']:.1f}  cy={K['cy']:.1f}"
          f"  ({K['width']}x{K['height']})")
    print(f"  saved -> {out_dir / 'intrinsics.json'}")

    env.close()


if __name__ == "__main__":
    main()
