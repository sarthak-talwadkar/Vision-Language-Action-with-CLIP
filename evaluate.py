"""
evaluate.py
===========
Task success rate evaluation across multiple tasks and instructions.

EVALUATION PROTOCOL
--------------------
For each (task, instruction) combination:
  1. Reset the scene to its canonical initial state (fixed random seed).
  2. Run the policy for up to max_steps.
  3. Check the task success condition.
  4. Repeat for N trials (default 50, matching LIBERO protocol).

Report per-task and aggregate success rate.

THREE EVALUATION MODES
-----------------------
1. SEEN INSTRUCTIONS  (in-distribution):
   Instructions used during training.  Tests if the policy learned the task.
   Expected success rate: >80% for a well-trained model.

2. PARAPHRASE INSTRUCTIONS  (synonym generalisation):
   Semantically equivalent but lexically different instructions.
   e.g. train on "pick up the red cube", test on "grasp the red block".
   Tests CLIP's language generalisation within the semantic neighbourhood.

3. NOVEL INSTRUCTIONS  (zero-shot):
   Instructions describing tasks NOT in the training set.
   e.g. trained on pick-and-place, tested on "push the cube to the right".
   Tests whether the CLIP embedding enables genuine zero-shot transfer.

KEY METRICS
-----------
- Task success rate: fraction of episodes where the robot achieves the goal.
- Generalisation gap: (seen_rate - novel_rate).  Small gap = good generalisation.
- Latency: average inference time per chunk (ms).

EXPECTED RESULTS (from CLIP-RT paper on LIBERO benchmark)
-----------------------------------------------------------
  libero_spatial:  ~74% (CLIP-RT) vs ~53% (BC baseline)
  libero_object:   ~82%
  libero_goal:     ~68%

Usage
-----
    python evaluate.py \\
        --checkpoint checkpoints/pick_place/best.pt \\
        --task-suite seen \\
        --tasks-file configs/eval_tasks.json \\
        --n-trials 50 \\
        --scene assets/tabletop.usd
"""

from __future__ import annotations

import argparse
import json
import os
import time
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
from typing import Optional

from models.clip_encoder import CLIPRTEncoder
from models.fusion import FeatureFusion
from models.policy import CLIPRTPolicy
from models.scene_3d import Scene3DFeatureField, CameraIntrinsics
from env.isaac_sim import IsaacSimEnv
from env.action_space import ActionSpace, ACTION_DIM


# ---------------------------------------------------------------------------
# Task definitions
# ---------------------------------------------------------------------------

# Format: list of dicts with keys "instruction", "suite" (seen/paraphrase/novel)
# These would normally live in configs/eval_tasks.json
DEFAULT_EVAL_TASKS = [
    # Seen instructions (same as training)
    {"instruction": "pick up the red cube",                 "suite": "seen"},
    {"instruction": "place the blue cube in the bin",       "suite": "seen"},
    # Paraphrase (synonym generalisation)
    {"instruction": "grasp the crimson block",              "suite": "paraphrase"},
    {"instruction": "put the blue object into the box",     "suite": "paraphrase"},
    # Novel zero-shot
    {"instruction": "move the red object to the left",      "suite": "novel"},
    {"instruction": "push the cube away from you",          "suite": "novel"},
]


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate(
    policy: CLIPRTPolicy,
    encoder: CLIPRTEncoder,
    action_space: ActionSpace,
    tasks: list[dict],
    env: IsaacSimEnv,
    n_trials: int = 50,
    max_steps: int = 300,
    scene_builder: Optional[Scene3DFeatureField] = None,
    device: torch.device = torch.device("cpu"),
    log_file=None,
) -> dict:
    """Run evaluation across all tasks.

    Parameters
    ----------
    policy : CLIPRTPolicy
    encoder : CLIPRTEncoder
    action_space : ActionSpace
        For unnormalising model outputs.
    tasks : list[dict]
        Each dict must have "instruction" and "suite" keys.
    env : IsaacSimEnv
    n_trials : int
        Episodes per task.
    max_steps : int
        Max steps per episode.
    scene_builder : Scene3DFeatureField | None
        Pre-built 3D scene processor (if using depth).
    device : torch.device
    log_file : file-like, optional
        Write results to this log file in addition to stdout.

    Returns
    -------
    results : dict
        Keys: suite names.  Values: {"success_rate": float, "n_tasks": int}.
    """
    suite_successes: dict[str, list[float]] = {}
    total_episodes = 0
    total_successes = 0

    for task in tqdm(tasks, desc="Tasks"):
        instruction = task["instruction"]
        suite       = task["suite"]

        task_successes = 0

        for trial_idx in range(n_trials):
            # Reset with a fixed seed for reproducibility.
            # Using trial_idx as the seed ensures each trial starts from a
            # unique but deterministic initial state.
            torch.manual_seed(trial_idx)
            np.random.seed(trial_idx)

            obs = env.reset()

            # Wait for physics stabilisation (matches LIBERO protocol)
            for _ in range(10):
                obs, _ = env.step(action_space.noop())

            done = False
            steps = 0

            while steps < max_steps and not done:
                # Get observation
                pil_image = Image.fromarray(obs.rgb).convert("RGB")

                # Build 3D scene if depth available
                scene = None
                if scene_builder is not None and obs.depth is not None:
                    with torch.no_grad():
                        _, patch_feats = encoder.encode_image(
                            pil_image, return_patch_features=True
                        )
                    scene = Scene3DFeatureField(
                        intrinsics=scene_builder.K,
                        patch_grid_size=scene_builder.patch_grid_size,
                        device=device,
                    ).build(obs.depth, patch_feats)

                # Predict action chunk
                action_chunk = policy.predict(pil_image, instruction, scene)

                # Execute chunk
                for step_action in action_chunk:
                    raw = action_space.unnormalise(np.array(step_action))
                    raw = action_space.clip(raw)
                    obs, done = env.step(raw)
                    steps += 1
                    if done:
                        break

            if done:
                task_successes += 1
                total_successes += 1
            total_episodes += 1

        task_rate = task_successes / n_trials

        if suite not in suite_successes:
            suite_successes[suite] = []
        suite_successes[suite].append(task_rate)

        msg = (
            f"[{suite:12s}] '{instruction[:50]}' → "
            f"{task_successes}/{n_trials} = {task_rate:.1%}"
        )
        print(msg)
        if log_file:
            log_file.write(msg + "\n")
            log_file.flush()

    # Aggregate per suite
    results = {}
    for suite, rates in suite_successes.items():
        results[suite] = {
            "success_rate": float(np.mean(rates)),
            "n_tasks":      len(rates),
        }

    # Overall
    results["overall"] = {
        "success_rate": total_successes / max(1, total_episodes),
        "n_tasks":      len(tasks),
    }

    # Generalisation gap: how much success drops from seen to novel
    if "seen" in results and "novel" in results:
        results["generalisation_gap"] = (
            results["seen"]["success_rate"] - results["novel"]["success_rate"]
        )

    return results


def print_results(results: dict, log_file=None):
    """Print a formatted summary table."""
    lines = [
        "\n" + "=" * 60,
        "EVALUATION RESULTS",
        "=" * 60,
    ]
    for suite, info in results.items():
        if suite == "generalisation_gap":
            lines.append(f"  Generalisation gap (seen−novel): {info:.1%}")
        else:
            rate = info.get("success_rate", 0.0)
            n    = info.get("n_tasks", 0)
            lines.append(f"  {suite:20s}: {rate:.1%}  ({n} tasks)")
    lines.append("=" * 60 + "\n")

    for line in lines:
        print(line)
        if log_file:
            log_file.write(line + "\n")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    action_mean = np.array(ckpt.get("action_mean", np.zeros(ACTION_DIM)))
    action_std  = np.array(ckpt.get("action_std",  np.ones(ACTION_DIM)))
    action_space = ActionSpace(action_mean=action_mean, action_std=action_std)

    # Build model
    encoder = CLIPRTEncoder(model_path="", use_action_decoder=True, device=device)
    encoder.raw_model.load_state_dict(ckpt["model_state"], strict=False)
    fusion = FeatureFusion(embed_dim=encoder.embed_dim, use_3d=args.use_3d)
    policy = CLIPRTPolicy(encoder=encoder, fusion=fusion)
    policy.eval()

    # Load tasks
    if args.tasks_file and os.path.exists(args.tasks_file):
        with open(args.tasks_file) as f:
            tasks = json.load(f)
    else:
        print("Using default eval tasks.")
        tasks = DEFAULT_EVAL_TASKS

    # Filter by suite if requested
    if args.task_suite != "all":
        tasks = [t for t in tasks if t["suite"] == args.task_suite]
    print(f"Evaluating {len(tasks)} tasks (suite: {args.task_suite}).")

    # Build environment
    env = IsaacSimEnv(
        scene_usd=args.scene,
        resolution=(640, 480),
        action_space=action_space,
        headless=args.headless,
    )

    # 3D scene builder
    scene_builder = None
    if args.use_3d:
        scene_builder = Scene3DFeatureField(
            intrinsics=CameraIntrinsics(),
            patch_grid_size=encoder.patch_grid_size,
            device=device,
        )

    # Logging
    os.makedirs("experiments/eval", exist_ok=True)
    log_path = f"experiments/eval/{time.strftime('%Y%m%d_%H%M%S')}_eval.txt"
    log_file = open(log_path, "w")
    print(f"Logging to: {log_path}")

    # Run evaluation
    results = evaluate(
        policy=policy,
        encoder=encoder,
        action_space=action_space,
        tasks=tasks,
        env=env,
        n_trials=args.n_trials,
        max_steps=args.max_steps,
        scene_builder=scene_builder,
        device=device,
        log_file=log_file,
    )

    print_results(results, log_file)
    log_file.close()
    env.close()

    # Save results JSON
    results_path = log_path.replace(".txt", ".json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved: {results_path}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate CLIP-RT policy.")
    p.add_argument("--checkpoint",  type=str, required=True)
    p.add_argument("--scene",       type=str, default="assets/tabletop.usd")
    p.add_argument("--tasks-file",  type=str, default="configs/eval_tasks.json")
    p.add_argument("--task-suite",  type=str, default="all",
                   choices=["all", "seen", "paraphrase", "novel"],
                   help="Evaluation suite to run.")
    p.add_argument("--n-trials",    type=int, default=50)
    p.add_argument("--max-steps",   type=int, default=300)
    p.add_argument("--headless",    action="store_true")
    p.add_argument("--use-3d",      action="store_true")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
