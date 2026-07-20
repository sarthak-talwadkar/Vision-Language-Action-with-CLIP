"""
models/fusion.py
================
Fuse image embedding, text embedding, and 3D scene feature into a joint
representation for the action policy.

WHY FUSION?
-----------
CLIP produces two embeddings:
  e_v ∈ R^D  — "what does the scene look like?"
  e_t ∈ R^D  — "what action should be performed?"

The dot product e_v · e_t is a single scalar — it tells us HOW WELL the scene
matches the instruction, but not WHAT ACTION to take.  To predict actions we
need the full vector information from both.

Concatenating [e_v || e_t] gives a 2D-dimensional joint representation that
carries both visual and linguistic context.  Adding the 3D aggregate f_3d
appends spatial geometry: not just "there is a cup" but "the cup is at this
depth and this offset from camera centre".

FUSION STRATEGY: WHY CONCATENATION OVER CROSS-ATTENTION?
---------------------------------------------------------
RT-2 (Brohan et al. 2023) and OpenVLA (Kim et al. 2024) use transformer
cross-attention between visual tokens and language tokens, which is more
expressive but computationally heavier (quadratic in sequence length).

CLIP-RT's MLP policy head uses simple concatenation.  This is sufficient
because:
  1. CLIP's contrastive pretraining has already aligned the two embedding
     spaces — they speak the same "language", so concatenation suffices.
  2. The MLP has enough capacity to learn which dimensions of [e_v || e_t]
     are jointly informative for each action token.

The CLIP-RT transformer decoder (in policy.py) still uses cross-attention
for the autoregressive action generation, but the *input* to the decoder is
the concatenated (or separately provided) embeddings, not raw tokens.

WHAT IS f_3d AND WHY ADD IT?
-----------------------------
f_3d is the mean of the top-k 3D CLIP features most similar to e_t.
It answers: "Given the instruction, what does the task-relevant region of the
scene look like in 3D?"

Ablation from the README:
  2D CLIP only → policy must infer gripper approach angle from 2D appearance.
  2D + 3D      → policy also knows the depth and 3D orientation of the target.

Example:
  Instruction: "grasp the mug by the handle"
  e_v: knows there is a mug
  e_t: knows to look for the handle
  f_3d: the mean of top-64 points near the handle gives the 3D position and
        approach vector of the handle → policy knows to approach from the side.

References
----------
- Brohan et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge
  to Robotic Control." CoRL 2023.
- Kim et al. "CLIP-RT." 2024.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from models.scene_3d import Scene3DFeatureField


class FeatureFusion:
    """Produce the fused embedding passed to the policy head.

    Parameters
    ----------
    embed_dim : int
        Dimension D of each individual embedding (512 for ViT-B, 1024 for ViT-H).
    top_k : int
        Number of top-matching 3D points used for the 3D aggregate feature.
    use_3d : bool
        Whether to include the 3D aggregate feature.  Set False for a 2D-only
        ablation (no depth sensor required).
    """

    def __init__(
        self,
        embed_dim: int = 1024,
        top_k: int = 64,
        use_3d: bool = True,
    ):
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.use_3d = use_3d

    @property
    def output_dim(self) -> int:
        """Dimensionality of the fused vector output.

        With 3D:    [e_v || e_t || f_3d] = 3 × D
        Without 3D: [e_v || e_t]         = 2 × D
        """
        return 3 * self.embed_dim if self.use_3d else 2 * self.embed_dim

    def fuse(
        self,
        e_v: torch.Tensor,
        e_t: torch.Tensor,
        scene: Scene3DFeatureField | None = None,
    ) -> torch.Tensor:
        """Produce the joint fused embedding.

        Parameters
        ----------
        e_v : torch.Tensor, shape [D] or [B, D]
            Global image embedding from CLIPRTEncoder.
        e_t : torch.Tensor, shape [D] or [B, D]
            Text embedding from CLIPRTEncoder.
        scene : Scene3DFeatureField | None
            Pre-built 3D feature field.  Required when use_3d=True.

        Returns
        -------
        fused : torch.Tensor, shape [3D] or [B, 3D]
            Concatenated joint representation.
        """
        batched = e_v.dim() == 2
        if not batched:
            # Add batch dimension for uniform handling.
            e_v = e_v.unsqueeze(0)
            e_t = e_t.unsqueeze(0)

        parts = [e_v, e_t]

        if self.use_3d:
            if scene is None:
                raise ValueError(
                    "Scene3DFeatureField required when use_3d=True. "
                    "Either pass a built scene or set use_3d=False."
                )
            # Aggregate once per batch element.  In a batch the same scene is
            # usually shared (one observation per env step), so we broadcast.
            f_3d = scene.aggregate(e_t[0], top_k=self.top_k)  # [D]
            f_3d = f_3d.unsqueeze(0).expand(e_v.shape[0], -1)  # [B, D]
            parts.append(f_3d)

        # Concatenate along feature dimension: [B, 2D] or [B, 3D]
        fused = torch.cat(parts, dim=-1)

        return fused if batched else fused.squeeze(0)

    def similarity(self, e_v: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """Cosine similarity between image and text embeddings.

        This is the zero-shot alignment score: high value means the current
        scene is well-matched to the instruction.  Useful for sanity checks
        during evaluation.

        Returns
        -------
        sim : torch.Tensor, scalar
            Value in [-1, 1].
        """
        return F.cosine_similarity(e_v.unsqueeze(0), e_t.unsqueeze(0)).squeeze()
