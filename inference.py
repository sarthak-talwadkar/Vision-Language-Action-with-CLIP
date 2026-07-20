"""
inference.py
============
Closed-loop zero-shot inference: run the trained CLIP-RT policy in Isaac Sim.

CLOSED-LOOP CONTROL
--------------------
"Closed-loop" means the policy re-observes the environment after every set of
actions and re-predicts the next chunk.  This is essential for manipulation:

  - The world changes as the robot acts (objects move, gripper state changes).
  - Small errors accumulate; re-observing lets the policy correct course.
  - The 8-step chunk is executed open-loop, then a new observation triggers
    the next inference call.

Compare to open-loop: execute a pre-planned sequence without re-observing.
Open-loop fails for contact-rich tasks (grasping) because contact introduces
state-dependent uncertainty that only closed-loop feedback can correct.

ZERO-SHOT GENERALISATION
--------------------------
At test time the policy receives instructions NOT seen during training.  For
example, if trained on "pick up the red cube", it should generalise to:
  - "grasp the crimson block"   (synonym generalisation)
  - "pick up the small red object"  (attribute generalisation)
  - "lift the cube from the table"  (paraphrase generalisation)

This works because CLIP's embedding space places these semantically similar
strings close together.  The action decoder, which has learned to map regions
of CLIP space to motion patterns, naturally generalises.

WHAT CLIP-RT CANNOT ZERO-SHOT
-------------------------------
  - Novel tasks with completely different motion primitives (e.g. if trained
    only on pick-and-place but asked to "pour water from the bottle").
  - Complex spatial reasoning ("place the cube to the LEFT of the bowl") if
    the training data has no left/right examples.
  - Tasks requiring reasoning chains ("first open the drawer, THEN pick up
    the object inside") — the MLP/single-chunk policy handles one motion
    primitive per inference call, not multi-step plans.

Usage
-----
    python inference.py \\
        --checkpoint checkpoints/pick_place/best.pt \\
        --instruction "grasp the crimson block" \\
        --scene assets/tabletop.usd \\
        --max-steps 300
"""

from __future__ import annotations

import argparse
import time
import numpy as np
import torch
from PIL import Image

from models.clip_encoder import CLIPRTEncoder
from models.fusion import FeatureFusion
from models.policy import CLIPRTPolicy
from models.scene_3d import Scene3DFeatureField, CameraIntrinsics
from env.isaac_sim import IsaacSimEnv
from env.action_space import ActionSpace, ACTION_DIM


def run_inference(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------------------
    # 1. Load checkpoint
    # -------------------------------------------------------------------------
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)

    # Recover action normalisation statistics from the checkpoint.
    # These were saved during training and are needed to unnormalise the
    # model's predictions before sending them to the robot.
    action_mean = np.array(ckpt.get("action_mean", np.zeros(ACTION_DIM)))
    action_std  = np.array(ckpt.get("action_std",  np.ones(ACTION_DIM)))
    action_space = ActionSpace(action_mean=action_mean, action_std=action_std)

    # -------------------------------------------------------------------------
    # 2. Build model
    # -------------------------------------------------------------------------
    encoder = CLIPRTEncoder(
        model_path="",  # base weights; fine-tuned weights loaded below
        use_action_decoder=True,
        device=device,
    )

    fusion = FeatureFusion(embed_dim=encoder.embed_dim, use_3d=args.use_3d)

    # Build the policy with use_3d matching how the checkpoint was TRAINED
    # (recorded by train.py), so its proj_3d weights have somewhere to load.
    # The --use-3d CLI flag controls whether depth is actually used at runtime.
    ckpt_use_3d = bool(ckpt.get("use_3d", False))
    if ckpt_use_3d and not args.use_3d:
        print("WARNING: checkpoint was trained WITH the 3D injection but "
              "--use-3d is off — the policy will run without the 3D signal "
              "it was trained to use.")
    policy = CLIPRTPolicy(encoder=encoder, fusion=fusion,
                          use_3d=ckpt_use_3d or args.use_3d)

    # Load the fine-tuned policy weights.  train.py saves policy.state_dict()
    # (keys: _model.* + proj_3d.*).  strict=False tolerates partial
    # checkpoints, leaving unmatched parameters at their pretrained values.
    policy.load_state_dict(ckpt["model_state"], strict=False)
    policy.eval()

    # -------------------------------------------------------------------------
    # 3. Build 3D scene processor (if depth is available)
    # -------------------------------------------------------------------------
    # Camera intrinsics must match the Isaac Sim camera configuration.
    scene_builder = None
    if args.use_3d:
        intrinsics = CameraIntrinsics(
            fx=args.fx, fy=args.fy,
            cx=args.cx, cy=args.cy,
            width=args.cam_width, height=args.cam_height,
        )
        scene_builder = Scene3DFeatureField(
            intrinsics=intrinsics,
            patch_grid_size=encoder.patch_grid_size,
            device=device,
            preprocess_mode=encoder.preprocess_mode,
        )

    # -------------------------------------------------------------------------
    # 4. Initialise the environment
    # -------------------------------------------------------------------------
    print(f"Initialising Isaac Sim scene: {args.scene}")
    env = IsaacSimEnv(
        scene_usd=args.scene,
        resolution=(args.cam_width, args.cam_height),
        action_space=action_space,
        headless=args.headless,
    )

    # -------------------------------------------------------------------------
    # 5. Closed-loop control loop
    # -------------------------------------------------------------------------
    print(f"\nInstruction: '{args.instruction}'")
    print("Starting closed-loop inference...\n")

    obs = env.reset()
    done = False
    total_steps = 0
    inference_times = []

    # Wait for physics to stabilise (matches LIBERO evaluation protocol)
    print("Waiting for physics stabilisation...")
    for _ in range(10):
        obs, _ = env.step(action_space.noop())

    while total_steps < args.max_steps and not done:
        # -----------------------------------------------------------------
        # Convert observation to model inputs
        # -----------------------------------------------------------------
        pil_image = Image.fromarray(obs.rgb).convert("RGB")

        # Build 3D feature field if depth is available.  maskclip dense
        # features are the gate-validated choice; pixel_stride trades field
        # resolution for latency in the control loop.  Rebuilding via the
        # same scene_builder object just overwrites its point/feature buffers.
        scene = None
        if scene_builder is not None and obs.depth is not None:
            _, patch_features = encoder.encode_image_dense(
                pil_image, mode=args.patch_mode
            )
            scene = scene_builder.build(
                obs.depth, patch_features, pixel_stride=args.pixel_stride
            )

        # -----------------------------------------------------------------
        # Policy inference: image + instruction → action chunk [8 × 7]
        # -----------------------------------------------------------------
        t0 = time.time()
        action_chunk = policy.predict(
            image=pil_image,
            instruction=args.instruction,
            scene=scene,
        )
        inference_time = time.time() - t0
        inference_times.append(inference_time)

        # -----------------------------------------------------------------
        # Execute the action chunk step-by-step
        # -----------------------------------------------------------------
        # Each chunk has 8 steps.  We execute all of them, collecting a new
        # observation after each step (closed-loop within the chunk).
        for step_action in action_chunk:
            # Unnormalise: model predicts in normalised space, robot needs raw values
            raw_action = action_space.unnormalise(np.array(step_action))
            raw_action = action_space.clip(raw_action)

            obs, done = env.step(raw_action)
            total_steps += 1

            if done:
                break

        print(
            f"Step {total_steps:3d}/{args.max_steps} | "
            f"Inference: {inference_time*1000:.1f}ms | "
            f"{'DONE ✓' if done else 'running...'}"
        )

    # -------------------------------------------------------------------------
    # 6. Report results
    # -------------------------------------------------------------------------
    print(f"\n{'='*50}")
    print(f"Task: {args.instruction}")
    print(f"Result: {'SUCCESS' if done else 'FAILURE'}")
    print(f"Steps taken: {total_steps}")
    if inference_times:
        avg_ms = np.mean(inference_times) * 1000
        print(f"Avg inference latency: {avg_ms:.1f} ms ({1000/avg_ms:.1f} Hz)")
    print(f"{'='*50}\n")

    env.close()
    return done


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CLIP-RT closed-loop inference.")

    p.add_argument("--checkpoint",   type=str, required=True,
                   help="Path to trained checkpoint (.pt from train.py).")
    p.add_argument("--instruction",  type=str, required=True,
                   help="Task instruction (can be zero-shot).")
    p.add_argument("--scene",        type=str, default="assets/tabletop.usd",
                   help="Isaac Sim USD scene file.")
    p.add_argument("--max-steps",    type=int, default=300,
                   help="Maximum environment steps before declaring failure.")
    p.add_argument("--headless",     action="store_true",
                   help="Run Isaac Sim without GUI.")
    p.add_argument("--use-3d",       action="store_true",
                   help="Use RGB-D + 3D CLIP feature field (requires depth sensor).")
    p.add_argument("--pixel-stride", type=int, default=2,
                   help="Depth subsampling for per-step field builds "
                        "(2 → ~77k points; higher = faster control loop).")
    p.add_argument("--patch-mode", type=str, default="clearclip",
                   choices=["clearclip", "csa", "maskclip", "proj"],
                   help="Dense feature mode for the 3D field — must match "
                        "what training used (smoke_alignment_test winner).")

    # Camera intrinsics (only needed with --use-3d)
    p.add_argument("--fx",         type=float, default=500.0)
    p.add_argument("--fy",         type=float, default=500.0)
    p.add_argument("--cx",         type=float, default=320.0)
    p.add_argument("--cy",         type=float, default=240.0)
    p.add_argument("--cam-width",  type=int,   default=640)
    p.add_argument("--cam-height", type=int,   default=480)

    return p.parse_args()


if __name__ == "__main__":
    run_inference(parse_args())
