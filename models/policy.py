"""
models/policy.py
================
Two policy architectures: MLPPolicy and CLIPRTPolicy.

ARCHITECTURE OVERVIEW
---------------------

MLPPolicy (simple baseline, as in the README)
  fused ∈ R^{3D}  →  FC(3D→512) → ReLU → FC(512→256) → ReLU → FC(256→7)
  Output: one 7-dim continuous action [dx, dy, dz, droll, dpitch, dyaw, gripper]

  This is sufficient if the action space is small and the task distribution is
  narrow.  It cannot generate multi-step action chunks, so it must be called at
  every environment step.

CLIPRTPolicy (full architecture, matches the CLIP-RT paper)
  Input: (image, instruction) → CLIPRTEncoder → e_v, e_t
  Transformer action decoder cross-attends to e_v and e_t to autoregressively
  decode a *chunk* of 8 consecutive actions.
  Output: 8 × 7 continuous actions

WHY ACTION CHUNKING?
---------------------
Action chunking (from ACT, Zhao et al. 2023) predicts multiple future actions
at once rather than one action per observation.  Benefits:

  1. Reduces compounding error: small prediction mistakes don't propagate
     through 8 individual closed-loop steps; the chunk is executed open-loop.
  2. Produces smoother, more temporally consistent trajectories.
  3. Enables the policy to reason about a short horizon: "I need to move left
     THEN down THEN close gripper" can be expressed in one chunk.

CLIP-RT uses chunks of size 8: each inference call returns 8 consecutive
7-dim action vectors to execute in the environment.

WHY A TRANSFORMER DECODER (NOT JUST MLP)?
------------------------------------------
The transformer decoder uses cross-attention:
  Query  : learnable action tokens (one per timestep in the chunk)
  Key/Val: image and text patch tokens from the CLIP encoder

Cross-attention lets each action token in the chunk "look back" at the visual
and language evidence to decide its value.  An MLP flattens all context into
one vector and loses the spatial structure of the patch tokens.

The decoder architecture matches OpenAI's GPT-2 (causal, decoder-only) with
the cross-attention keys/values coming from the CLIP encoder outputs.

HOW THE CLIP-RT DECODER WORKS (step by step)
----------------------------------------------
  1. Concatenate [image_features; text_features] as the cross-attention
     context (K, V).
  2. Initialise 56 learnable "action query" tokens (dummy_tokens filled with
     pad_id).  These will be decoded into 56 per-step features.
  3. Run the causal transformer decoder: each query token attends causally to
     previous query tokens (for temporal coherence) and cross-attends to the
     image+text context (for conditional action generation).
  4. Take output tokens [2:] (skip 2 context-conditioning prefix tokens →
     length 56).
  5. Reshape [56, hidden_dim] → [8, 7, hidden_dim/7] → linear projection
     via action_head → [8, 7] continuous actions.

HOW CONTINUOUS ACTIONS DIFFER FROM RT-2
-----------------------------------------
RT-2 tokenises continuous actions into discrete bins (256 bins per dimension)
and generates them as text tokens using the LLM vocabulary.  This allows using
the LLM's seq2seq machinery but requires careful bin quantisation.

CLIP-RT keeps actions continuous (float32) and uses a learned regression head
rather than token-class classification.  This avoids quantisation error and
produces smoother trajectories.

References
----------
- Zhao et al. "Learning Fine-Grained Bimanual Manipulation with Low-Cost
  Hardware." RSS 2023. (ACT, action chunking)
- Brohan et al. "RT-2: Vision-Language-Action Models Transfer Web Knowledge
  to Robotic Control." CoRL 2023.
- Kim et al. "CLIP-RT." 2024.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.clip_encoder import CLIPRTEncoder
from models.fusion import FeatureFusion
from models.scene_3d import Scene3DFeatureField


# ---------------------------------------------------------------------------
# Constants matching CLIP-RT paper and the existing clip_rt_utils.py
# ---------------------------------------------------------------------------
ACTION_DIM      = 7    # [dx, dy, dz, droll, dpitch, dyaw, gripper_open]
ACTION_CHUNK    = 8    # Predict 8 consecutive steps per inference call
DECODER_TOKENS  = 56   # = ACTION_CHUNK × (some multiple); matches model.pad_id usage


# ===========================================================================
# 1. MLP Policy  (simple baseline described in the README)
# ===========================================================================

class MLPPolicy(nn.Module):
    """Simple feed-forward policy: fused_embedding → single action vector.

    Use this when:
      - You don't need action chunking (e.g. high-frequency control loops).
      - You want a quick baseline to verify data pipeline and training loop.
      - You only have a 2D camera (no depth → use_3d=False in FeatureFusion).

    Parameters
    ----------
    input_dim : int
        Dimension of the fused embedding.  Must match FeatureFusion.output_dim.
        Default 3072 = 3 × 1024 (ViT-H, with 3D feature).
    hidden_dims : list[int]
        Hidden layer sizes.  Deeper = more expressive but slower to train.
    action_dim : int
        Output dimension.  7 for standard 6-DOF arm + gripper.
    dropout : float
        Dropout probability.  Helps prevent overfitting on small demo datasets.
    """

    def __init__(
        self,
        input_dim: int = 3 * 1024,
        hidden_dims: list[int] = [512, 256],
        action_dim: int = ACTION_DIM,
        dropout: float = 0.2,
    ):
        super().__init__()

        layers: list[nn.Module] = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h_dim

        # Final linear layer projects to action space.
        # NO activation on the output: actions are unbounded continuous values
        # (clipped later by the env/action_space normalisation layer).
        layers.append(nn.Linear(in_dim, action_dim))

        self.net = nn.Sequential(*layers)

        # Initialise weights with Xavier uniform to prevent gradient vanishing
        # in the early training steps (important when the encoder is frozen).
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, fused: torch.Tensor) -> torch.Tensor:
        """Predict a single action vector from the fused embedding.

        Parameters
        ----------
        fused : torch.Tensor, shape [B, input_dim] or [input_dim]

        Returns
        -------
        action : torch.Tensor, shape [B, action_dim] or [action_dim]
        """
        single = fused.dim() == 1
        if single:
            fused = fused.unsqueeze(0)
        action = self.net(fused)
        return action.squeeze(0) if single else action


# ===========================================================================
# 2. CLIP-RT Policy  (full architecture)
# ===========================================================================

class CLIPRTPolicy(nn.Module):
    """Full CLIP-RT policy: image + instruction → 8-step action chunk.

    This class orchestrates the complete inference pipeline:
      CLIPRTEncoder → Scene3DFeatureField → FeatureFusion → action decoder

    Training mode (train_policy_only=True, default for fine-tuning):
      The CLIP encoder is frozen — only the action decoder and head are trained.
      This is the standard fine-tuning regime because:
        - The CLIP encoder already has rich representations from pretraining.
        - Fine-tuning the full 630M parameter encoder on ~500 demos would
          severely overfit and destroy the zero-shot generalisation.
        - The action decoder is ~20M parameters and can adapt quickly.

    Full training mode (train_policy_only=False):
      All parameters are trained end-to-end.  Use this when starting from a
      randomly-initialised model (no pretrained CLIP checkpoint).

    3D INJECTION (use_3d=True) — DESIGN RATIONALE
    ----------------------------------------------
    The CLIP-RT decoder's positional embedding is sized for EXACTLY 58 input
    tokens: [img, inst, 56 action queries].  Appending a 3D scene token would
    change that size and break loading of every pretrained CLIP-RT checkpoint.

    Instead we use a ZERO-INITIALISED RESIDUAL INJECTION into the image
    context token:

        img_token  ←  img_token + W_3d · [f_3d ‖ centroid_xyz]

    where f_3d ∈ R^D is the text-queried 3D feature summary (mean of the
    top-k point features most similar to the instruction), centroid_xyz ∈ R³
    is the mean 3D position of those points (camera frame, metres), and
    W_3d ∈ R^{D×(D+3)} starts at ZERO.  Consequences:

      1. Checkpoint compatible: no architecture change visible to the
         pretrained weights.
      2. Identity at init: with W_3d = 0 the policy is bit-for-bit vanilla
         CLIP-RT, so 3D can only help if gradients say so — a clean 2D vs
         2D+3D ablation with all else equal.
      3. Geometry actually enters: the centroid tells the decoder WHERE the
         instruction-relevant region is in 3D, not just what it looks like.

    We deliberately do NOT re-normalise the image token after injection:
    at init the norm is unchanged anyway (zero injection), and later the
    learned magnitude of the 3D term is itself signal.

    Parameters
    ----------
    encoder : CLIPRTEncoder
        Pre-initialised CLIP encoder (see clip_encoder.py).
    fusion : FeatureFusion
        Pre-initialised fusion module (see fusion.py).  In this policy it
        only supplies the top_k default for 3D queries; the MLP baseline
        uses its fuse() method.
    action_dim : int
        Per-timestep action dimensionality.
    action_chunk : int
        Number of timesteps predicted per forward pass.
    train_policy_only : bool
        If True, freeze the CLIP encoder parameters during training.
    use_3d : bool
        Enable the zero-init 3D injection described above.  Requires depth
        (a Scene3DFeatureField at inference, per-sample f_3d in training).
    """

    def __init__(
        self,
        encoder: CLIPRTEncoder,
        fusion: FeatureFusion,
        action_dim: int = ACTION_DIM,
        action_chunk: int = ACTION_CHUNK,
        train_policy_only: bool = True,
        use_3d: bool = False,
    ):
        super().__init__()

        self.encoder = encoder
        self.fusion = fusion
        self.action_dim = action_dim
        self.action_chunk = action_chunk
        self.use_3d = use_3d

        # The underlying OpenCLIP model with the action decoder.
        # This is what decode_action() and action_head live on.
        self._model = encoder.raw_model

        if use_3d:
            # Zero-init injection: [f_3d (D) ‖ centroid_xyz (3)] → D.
            # Zero weights ⇒ exactly vanilla CLIP-RT at initialisation.
            self.proj_3d = nn.Linear(encoder.embed_dim + 3, encoder.embed_dim, bias=False)
            nn.init.zeros_(self.proj_3d.weight)
        else:
            self.proj_3d = None

        if train_policy_only:
            # Freeze the CLIP encoder.  Only the action decoder (text2/
            # transformer2), action_head, and proj_3d remain trainable.
            #
            # NB: we freeze the WHOLE text tower (self._model.text), not just
            # its transformer.  The fork aliases self.transformer to
            # self.text.transformer, so freezing only that would leave the
            # text token embedding, ln_final and text_projection trainable —
            # letting the "frozen" encoder drift and silently eroding the
            # zero-shot property the freeze exists to protect.
            for param in self._model.visual.parameters():
                param.requires_grad = False
            for param in self._model.text.parameters():
                param.requires_grad = False

    # ------------------------------------------------------------------
    # Forward: full inference path
    # ------------------------------------------------------------------

    def forward(
        self,
        image: torch.Tensor,
        instruction_tokens: torch.Tensor,
        f_3d: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict an action chunk from a preprocessed image + tokenised instruction.

        This method is called during TRAINING.  For inference from raw PIL
        images use `predict()` instead.

        Parameters
        ----------
        image : torch.Tensor, shape [B, 3, H, W]
            Preprocessed image tensor (output of preprocess pipeline).
        instruction_tokens : torch.Tensor, shape [B, context_length]
            Tokenised (and prompted) instruction.
        f_3d : torch.Tensor | None, shape [B, D+3]
            Per-sample 3D conditioning vector [f_3d ‖ centroid_xyz] from
            Scene3DFeatureField.localize().  Ignored unless use_3d=True.
            If None, the 3D branch contributes nothing (pure 2D CLIP-RT).

        Returns
        -------
        actions : torch.Tensor, shape [B, action_chunk, action_dim]
            Predicted continuous action chunk.
        """
        B = image.shape[0]

        # Step 1: Encode image and text.
        # normalize=True applies L2 normalisation so embeddings live on the
        # unit hypersphere — cosine similarity is then just a dot product.
        image_features = self._model.encode_image(image, normalize=True)
        text_features  = self._model.encode_text(instruction_tokens, normalize=True)

        # Step 1b: 3D injection (see class docstring).  Zero-init proj_3d
        # means this is exactly a no-op at the start of training.
        if self.use_3d and f_3d is not None:
            image_features = image_features + self.proj_3d(f_3d.to(image_features.dtype))

        # Step 2: Initialise the decoder with pad tokens.
        # The transformer decoder uses these as initial "query" tokens.
        # They will be updated by the causal self-attention and cross-attention
        # layers in the decoder to become action-specific features.
        dummy_tokens = torch.full(
            (B, DECODER_TOKENS),
            self._model.pad_id,
            dtype=torch.long,
            device=image.device,
        )

        # Step 3: Run the transformer action decoder.
        # decode_action cross-attends dummy_tokens to image_features and
        # text_features, producing one output feature per decoder token.
        # out_features: [B, DECODER_TOKENS + 2, hidden_dim]
        out_features = self._model.decode_action(
            dummy_tokens, image_features, text_features
        )

        # Step 4: Discard the first 2 tokens (prefix / context tokens used by
        # the decoder's cross-attention setup).
        # After slicing: [B, DECODER_TOKENS, hidden_dim] = [B, 56, hidden_dim]
        out_features = out_features[:, 2:, :]

        # Step 5: Reshape for action chunking.
        # 56 tokens / 8 chunks = 7 tokens per timestep.  Each timestep's 7
        # tokens are flattened into a single vector and projected by action_head.
        out_features = out_features.reshape(B, self.action_chunk, -1)
        # out_features: [B, 8, 7 * hidden_dim]

        # Step 6: Project each chunk-step's feature to ACTION_DIM continuous values.
        # action_head is a learned linear layer: [7*hidden_dim → action_dim]
        actions = self._model.action_head(out_features)
        # actions: [B, 8, 7]

        return actions

    # ------------------------------------------------------------------
    # Inference: from raw PIL image + string instruction
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self,
        image,   # PIL.Image
        instruction: str,
        scene: Scene3DFeatureField | None = None,
    ) -> list[list[float]]:
        """Run closed-loop inference from a raw image and instruction string.

        This is the method called by inference.py at each environment step.

        Parameters
        ----------
        image : PIL.Image
            Current RGB frame from the robot camera.
        instruction : str
            Task instruction, e.g. "pick up the red cup".
        scene : Scene3DFeatureField | None
            Optional 3D feature field built from the current RGB-D observation
            (see inference.py).  Used only when the policy was built with
            use_3d=True.

        Returns
        -------
        action_chunk : list[list[float]]
            8 action vectors, each of length 7.
            Format: [[dx, dy, dz, droll, dpitch, dyaw, gripper], ...]
        """
        device = next(self._model.parameters()).device

        # Preprocess image: PIL → normalised tensor
        img_tensor = self.encoder.preprocess(image).unsqueeze(0).to(device)

        # Tokenise the prompted instruction
        prompted = f"what motion should the robot arm perform to complete the instruction '{instruction}'?"
        inst_tokens = self.encoder.tokenizer(prompted).to(device)

        # 3D conditioning: query the feature field with the RAW instruction
        # (use_motion_prompt=False).  The motion-question prompt deliberately
        # shifts the embedding toward motion semantics, which is right for the
        # action decoder but WRONG for localisation — patch features align
        # with object phrases ("the red cube"), not motion questions.
        f_3d = None
        if self.use_3d and scene is not None:
            e_t_obj = self.encoder.encode_text(instruction, use_motion_prompt=False)
            top_k = self.fusion.top_k if self.fusion is not None else 64
            feat, centroid = scene.localize(e_t_obj.to(scene.device), top_k=top_k)
            f_3d = torch.cat([feat, centroid], dim=-1).unsqueeze(0).to(device)

        # Run the forward pass (mixed precision on GPU, fp32 on CPU)
        with torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
            actions = self.forward(img_tensor, inst_tokens, f_3d=f_3d)

        # actions: [1, 8, 7] → convert to plain Python list [[float×7]×8]
        chunk = actions.squeeze(0).float().cpu().numpy().tolist()
        return chunk
