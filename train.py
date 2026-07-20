"""
train.py
========
Policy training entry point for the CLIP-RT VLA system.

TRAINING STRATEGY
-----------------
We use *behaviour cloning* (BC): the policy is trained to imitate the expert
actions in the demonstration dataset via supervised learning.  The loss is the
mean absolute error (L1) between predicted and expert action chunks.

WHY L1 LOSS OVER MSE (L2)?
----------------------------
MSE penalises large errors quadratically, causing the model to hedge toward
the mean of multiple modes in the data.  For manipulation, the policy must
commit to one solution (e.g. grasp from the LEFT side of the cup, not average
between left and right).

L1 loss is more robust to outliers and encourages sharper predictions.  This
matches the CLIP-RT paper and OpenVLA training configuration.

FINE-TUNING REGIME (FREEZE ENCODER)
-------------------------------------
By default, the CLIP encoder (630M parameters) is frozen and only the action
decoder + head (~20M parameters) are trained.  This is because:
  1. We have at most 500 demonstrations (vs 400M image-text pairs for CLIP).
  2. Fine-tuning the encoder would destroy the zero-shot generalisation that
     makes CLIP-RT useful for novel instructions.
  3. Training time: fine-tuning the encoder requires more memory and steps.

WHEN TO UNFREEZE THE ENCODER (full fine-tune)
----------------------------------------------
If your Isaac Sim scene is visually very different from the CLIP training data
(unusual lighting, non-photorealistic textures, synthetic objects), you may
benefit from end-to-end fine-tuning.  Use --train-full-model to enable this,
but increase dataset size to ≥2000 demos and use a lower learning rate (1e-6).

LEARNING RATE SCHEDULE
-----------------------
We use cosine annealing with warmup (matching CLIP-RT pretraining):
  - Warmup (first `warmup_steps` steps): linear ramp from 0 to `lr`.
    Avoids destabilising the pretrained weights with large early gradients.
  - Cosine decay (remaining steps): smooth reduction to `lr * min_lr_ratio`.
    Ensures the model converges without oscillating at the end of training.

Usage
-----
    python train.py \\
        --demos data/demos/pick_place.hdf5 \\
        --checkpoint path/to/clip_rt.pt \\
        --epochs 50 \\
        --batch-size 32 \\
        --lr 1e-4 \\
        --save-dir checkpoints/pick_place/
"""

from __future__ import annotations

import argparse
import os
import math
import time
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from pathlib import Path

from models.clip_encoder import CLIPRTEncoder
from models.fusion import FeatureFusion
from models.policy import CLIPRTPolicy, MLPPolicy
from models.scene_3d import CameraIntrinsics, Scene3DFeatureField
from data.dataset import make_dataloader


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def cosine_lr_with_warmup(
    step: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float,
    min_lr_ratio: float = 0.1,
) -> float:
    """Compute the learning rate at `step` using cosine decay with warmup.

    Returns a multiplicative factor applied to the base_lr via LambdaLR.
    """
    if step < warmup_steps:
        # Linear warmup from 0 → base_lr
        return step / max(1, warmup_steps)
    # Cosine decay from base_lr → min_lr_ratio * base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    cosine   = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


def action_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 loss over the action chunk.

    Parameters
    ----------
    pred   : [B, chunk_size, action_dim]   — policy predictions
    target : [B, chunk_size, action_dim]   — expert (normalised) actions
    mask   : [B, chunk_size]   bool        — True for valid (non-padded) steps

    Returns
    -------
    loss : scalar tensor
    """
    # L1 element-wise: [B, chunk_size, action_dim]
    loss_per_elem = F.l1_loss(pred, target, reduction="none")

    # Average over action_dim: [B, chunk_size]
    loss_per_step = loss_per_elem.mean(dim=-1)

    # Apply mask: only count valid steps
    # mask: [B, chunk_size] bool → [B, chunk_size] float
    masked_loss = loss_per_step * mask.float()

    # Average over valid steps (sum / n_valid)
    n_valid = mask.float().sum().clamp(min=1.0)
    return masked_loss.sum() / n_valid


def compute_f3d_batch(
    encoder: CLIPRTEncoder,
    images: torch.Tensor,
    depths: torch.Tensor,
    raw_tokens: torch.Tensor,
    intrinsics: CameraIntrinsics,
    pixel_stride: int,
    top_k: int,
    device: torch.device,
    patch_mode: str = "clearclip",
) -> torch.Tensor:
    """Per-sample 3D conditioning vectors [f_3d ‖ centroid_xyz] for a batch.

    Everything here is under no_grad because the CLIP encoder is FROZEN —
    f_3d is a deterministic function of (image, depth, instruction), i.e. an
    input feature, not something we backpropagate through.  Gradients only
    flow into the policy's zero-init proj_3d, which consumes the result.

    Cost breakdown per batch:
      - one batched dense encode (maskclip): ~1 extra ViT block over the
        normal image encode the policy does anyway;
      - one batched raw-instruction text encode;
      - B small per-sample field builds on stride-subsampled depth
        (stride 4 → ~19k points instead of 307k).

    The dense pass reuses the SAME preprocessed (colour-jittered) image
    tensor the policy sees, so the 3D features are consistent with the
    augmented observation.
    """
    with torch.no_grad():
        # [B, N_patches, D] — batched dense features.  patch_mode must be the
        # smoke-test winner for THIS backbone (maskclip for ViT-B; the
        # residual-free modes clearclip/csa for ViT-H — see clip_encoder.py).
        _, patch_feats = encoder.encode_image_dense(images, mode=patch_mode)

        # Object-phrase embeddings for the field query (NOT motion-prompted).
        e_t_obj = encoder.model.encode_text(raw_tokens.to(device), normalize=True)
        e_t_obj = e_t_obj.float()

        vecs = []
        for i in range(images.shape[0]):
            field = Scene3DFeatureField(
                intrinsics=intrinsics,
                device=device,
                preprocess_mode=encoder.preprocess_mode,
            ).build(depths[i].numpy(), patch_feats[i], pixel_stride=pixel_stride)
            f_3d, centroid = field.localize(e_t_obj[i], top_k=top_k)
            vecs.append(torch.cat([f_3d, centroid], dim=-1))

    return torch.stack(vecs)  # [B, D+3]


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    # -------------------------------------------------------------------------
    # 1. Load the CLIP-RT encoder and build the policy
    # -------------------------------------------------------------------------
    # The encoder loads the pretrained CLIP-RT checkpoint.  The checkpoint
    # contains both the dual-encoder weights AND the pretrained action decoder
    # weights from the Open-X Embodiment pretraining phase.
    print(f"Loading CLIP-RT encoder ({args.model_name}) from: {args.checkpoint or '<pretrained tag>'}")
    encoder = CLIPRTEncoder(
        model_name=args.model_name,
        model_path=args.checkpoint,
        use_action_decoder=True,
        device=device,
    )

    # FeatureFusion here only carries the top_k default for 3D queries; the
    # CLIPRTPolicy consumes precomputed f_3d vectors (see compute_f3d_batch).
    fusion = FeatureFusion(embed_dim=encoder.embed_dim, top_k=args.top_k,
                           use_3d=args.use_3d)

    # CLIPRTPolicy wraps the full pipeline.
    # train_policy_only=True freezes the CLIP encoder (recommended).
    # use_3d=True adds the zero-init 3D injection (policy.py docstring) —
    # at initialisation the model is exactly vanilla CLIP-RT either way.
    policy = CLIPRTPolicy(
        encoder=encoder,
        fusion=fusion,
        train_policy_only=not args.train_full_model,
        use_3d=args.use_3d,
    )
    policy.to(device)

    # Camera intrinsics for the training-time field builds.  Only the depth
    # range filter and the centroid SCALE depend on these (f_3d features are
    # coordinate-free), so the collect_demos defaults are fine unless you
    # changed the Isaac Sim camera.
    intrinsics = CameraIntrinsics()

    # -------------------------------------------------------------------------
    # 2. Build the data pipeline
    # -------------------------------------------------------------------------
    hdf5_paths = args.demos  # list of paths from command line
    loader, action_space = make_dataloader(
        hdf5_paths=hdf5_paths,
        preprocess=encoder.preprocess,
        tokenizer=encoder.tokenizer,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        augment=True,
        shuffle=True,
        use_3d=args.use_3d,
    )
    print(f"Training samples: {len(loader.dataset)}")

    # -------------------------------------------------------------------------
    # 3. Optimiser
    # -------------------------------------------------------------------------
    # Only optimise trainable parameters.  If train_policy_only=True, this
    # is just the action decoder + action_head (~20M params).
    trainable_params = [p for p in policy.parameters() if p.requires_grad]
    print(f"Trainable parameters: {sum(p.numel() for p in trainable_params):,}")

    # AdamW: Adam + L2 weight decay.  Weight decay regularises the action
    # decoder and prevents overfitting on small demo datasets.
    # β₁=0.9, β₂=0.95 (slightly lower β₂ than default 0.999) for more
    # stable training on noisy robotic action data.
    optimizer = AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )

    # LR schedule: cosine decay with warmup
    total_steps = len(loader) * args.epochs
    warmup_steps = min(args.warmup_steps, total_steps // 10)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_lr_with_warmup(
            step, warmup_steps, total_steps, args.lr
        ),
    )

    # -------------------------------------------------------------------------
    # 4. Training loop
    # -------------------------------------------------------------------------
    os.makedirs(args.save_dir, exist_ok=True)
    global_step = 0
    best_loss   = float("inf")

    # Mixed precision: reduces memory by 2× and speeds up training on GPUs
    # with Tensor Cores.  enabled=False on CPU keeps debugging runs working.
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for epoch in range(args.epochs):
        policy.train()
        epoch_loss = 0.0
        epoch_start = time.time()

        for batch_idx, batch in enumerate(loader):
            if args.use_3d:
                images, tokens, action_chunks, action_masks, depths, raw_tokens = batch
            else:
                images, tokens, action_chunks, action_masks = batch

            images        = images.to(device, non_blocking=True)
            tokens        = tokens.to(device, non_blocking=True)
            action_chunks = action_chunks.to(device, non_blocking=True)
            action_masks  = action_masks.to(device, non_blocking=True)

            # 3D conditioning vectors (no_grad — the encoder is frozen, so
            # f_3d is an input feature, not part of the computation graph).
            f_3d = None
            if args.use_3d:
                f_3d = compute_f3d_batch(
                    encoder, images, depths, raw_tokens,
                    intrinsics=intrinsics,
                    pixel_stride=args.pixel_stride,
                    top_k=args.top_k,
                    device=device,
                    patch_mode=args.patch_mode,
                )

            # Forward pass under autocast for fp16/bf16 computation.
            with torch.autocast(device_type=device.type, enabled=use_amp):
                pred_actions = policy(images, tokens, f_3d=f_3d)  # [B, chunk, act_dim]
                loss = action_l1_loss(pred_actions, action_chunks, action_masks)

            # Backward pass: scale loss to prevent fp16 underflow.
            optimizer.zero_grad()
            scaler.scale(loss).backward()

            # Gradient clipping: prevents exploding gradients in the transformer.
            # max_norm=1.0 is the standard value used in CLIP-RT and OpenVLA.
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            global_step += 1

            if batch_idx % args.log_interval == 0:
                lr = scheduler.get_last_lr()[0]
                print(
                    f"Epoch {epoch+1}/{args.epochs} | "
                    f"Batch {batch_idx}/{len(loader)} | "
                    f"Loss {loss.item():.4f} | "
                    f"LR {lr:.2e}"
                )

        # Epoch summary
        avg_loss = epoch_loss / len(loader)
        elapsed  = time.time() - epoch_start
        print(f"Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}  ({elapsed:.1f}s)")

        # Save checkpoint after every epoch
        ckpt_path = os.path.join(args.save_dir, f"epoch_{epoch+1:03d}.pt")
        torch.save({
            "epoch":       epoch + 1,
            "global_step": global_step,
            # Save the POLICY state dict (keys: _model.* + proj_3d.*), not
            # just the inner model — otherwise the trained 3D injection
            # weights would silently be dropped from every checkpoint.
            # inference.py loads with policy.load_state_dict(...).
            "model_state": policy.state_dict(),
            "use_3d":      args.use_3d,
            "optim_state": optimizer.state_dict(),
            "loss":        avg_loss,
            "action_mean": action_space.mean.tolist(),
            "action_std":  action_space.std.tolist(),
        }, ckpt_path)

        # Track the best checkpoint
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_path = os.path.join(args.save_dir, "best.pt")
            torch.save(torch.load(ckpt_path), best_path)
            print(f"  ** New best checkpoint saved: {best_path}")

    print(f"\nTraining complete. Best loss: {best_loss:.4f}")
    print(f"Best checkpoint: {best_path}")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train CLIP-RT VLA policy.")

    p.add_argument("--demos",       nargs="+", required=True,
                   help="Path(s) to HDF5 demonstration files.")
    p.add_argument("--model-name",  type=str, default="ViT-H-14-378-quickgelu",
                   help="OpenCLIP backbone architecture.  Default matches the "
                        "released CLIP-RT checkpoints; use 'ViT-B-16' for a "
                        "fast quick-test on limited hardware.")
    p.add_argument("--checkpoint",  type=str, default="",
                   help="Path to pretrained CLIP-RT checkpoint (.pt), or an "
                        "OpenCLIP tag (e.g. 'openai', 'dfn5b').  Empty = "
                        "auto-select a public tag for the architecture.")
    p.add_argument("--save-dir",    type=str, default="checkpoints/",
                   help="Directory to save training checkpoints.")
    p.add_argument("--epochs",      type=int, default=50,
                   help="Number of training epochs.")
    p.add_argument("--batch-size",  type=int, default=32)
    p.add_argument("--lr",          type=float, default=1e-4,
                   help="Peak learning rate (after warmup).")
    p.add_argument("--warmup-steps",type=int, default=500,
                   help="Number of LR warmup steps.")
    p.add_argument("--num-workers", type=int, default=4,
                   help="DataLoader worker processes.")
    p.add_argument("--log-interval",type=int, default=10,
                   help="Print loss every N batches.")
    p.add_argument("--train-full-model", action="store_true",
                   help="Fine-tune the full CLIP encoder (not just the decoder).  "
                        "Use only with large datasets (≥2000 demos).")
    p.add_argument("--use-3d", action="store_true",
                   help="Enable the zero-init 3D injection: per-sample "
                        "Scene3DFeatureField built from demo depth frames "
                        "(maskclip dense features), text-queried, injected "
                        "into the decoder's image token.  Demos must contain "
                        "depth (collect_demos.py saves it by default).")
    p.add_argument("--pixel-stride", type=int, default=4,
                   help="Depth subsampling for training-time field builds "
                        "(4 → ~19k points per 640×480 frame).")
    p.add_argument("--top-k", type=int, default=64,
                   help="Number of top-matching 3D points pooled into f_3d.")
    p.add_argument("--patch-mode", type=str, default="clearclip",
                   choices=["clearclip", "csa", "maskclip", "proj"],
                   help="Dense feature mode for the 3D field.  Use the "
                        "smoke_alignment_test winner for your backbone.")

    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
