"""
models/clip_encoder.py
======================
CLIP-RT visual and language encoder + dense (per-patch) feature extraction.

WHY CLIP AS THE BACKBONE?
--------------------------
CLIP (Contrastive Language-Image Pretraining, Radford et al. 2021) was trained
on 400 million (image, alt-text) pairs scraped from the web.  The contrastive
objective pushes matched (image, text) pairs to have high cosine similarity and
mismatched pairs to have low similarity in a shared 512-1024 dimensional
embedding space.

The critical property for robotics: CLIP's embedding space generalises to
concepts that were *never* seen together during training.  "crimson block" maps
close to "red cube" because both co-occur with similar images online.  This is
what enables zero-shot generalisation — the robot policy need not be retrained
for every paraphrase of an instruction.

WHY ViT-H-14-378-quickgelu?
-----------------------------
- ViT-H  : Huge ViT, ~630M parameters.  Larger model → richer per-patch
           features → better spatial grounding for manipulation.
- 14     : 14×14 pixel patches.  Finer patch grid → more spatial resolution
           in the feature map (27×27 patches for a 378px image).
- 378    : Input resolution 378×378.  Higher than the standard 224 gives the
           encoder more pixel detail to distinguish objects at close range.
- quickgelu: A numerically-identical but slightly faster variant of GELU used
             in the original OpenAI CLIP training.  We keep it to stay
             compatible with released CLIP-RT checkpoints.

NOTE ON PRETRAINED WEIGHTS: OpenAI never released a ViT-H checkpoint.  The
only public weights for ViT-H-14-378-quickgelu are Apple's DFN5B
(pretrained="dfn5b").  CLIP-RT's released checkpoints are further trained
from DFN5B.  Passing pretrained="openai" for this architecture fails.

WHY PATCH FEATURES NEED SPECIAL HANDLING (READ THIS BEFORE TOUCHING scene_3d)
------------------------------------------------------------------------------
CLIP's contrastive objective only ever aligns ONE vector per image with ONE
vector per text: the pooled [CLS] embedding, *after* the final projection
matrix `visual.proj`.  The per-patch tokens inside the transformer:

  1. live in a DIFFERENT vector space than the text embeddings.  Tokens come
     out at transformer width (1280 for ViT-H); text embeddings are at
     embed_dim (1024).  In the standard forward pass only the pooled [CLS]
     token is multiplied by `visual.proj` ([1280 → 1024]) — the patch tokens
     are returned unprojected (see open_clip/transformer.py, ~line 541).
  2. were never directly optimised to align with text.  Worse, the LAST
     self-attention layer mixes every patch with every other patch to serve
     the [CLS] pooling, which empirically scrambles patch-level semantics
     (MaskCLIP, Zhou et al. ECCV 2022).

Naively comparing raw patch tokens against text embeddings therefore fails
twice: dimensionally (1280 vs 1024 → matmul crash) and semantically (weak,
noisy alignment even after projection).

`encode_image_dense()` implements the two standard remedies:

  mode="proj"     : run the tower normally, then apply ln_post + visual.proj
                    to the patch tokens.  Correct vector space; alignment
                    quality varies by model — often noisy.
  mode="maskclip" : re-run ONLY the last transformer block with the query-key
                    attention replaced by identity — each patch keeps its own
                    VALUE vector v(x_i) instead of a global attention mixture —
                    then out_proj → residual → MLP → ln_post → visual.proj.
                    This is the MaskCLIP trick (Zhou et al. ECCV 2022), the
                    standard way to extract dense text-aligned CLIP features
                    without any retraining.  Costs one extra transformer block
                    per frame.

Which mode actually localises objects in OUR setup is exactly what
scripts/smoke_alignment_test.py (the gate experiment) measures.

HOW CLIP-RT EXTENDS STANDARD CLIP
-----------------------------------
CLIP-RT (Kang et al. 2024) adds a *transformer action decoder* on top of the
standard dual-encoder: a second (non-causal) text tower whose input sequence is
[image_embedding, text_embedding, 56 pad tokens]; the 56 output features are
reshaped to 8 timesteps × 7 tokens and projected by an MLPResNet head to an
8-step chunk of 7-DoF continuous actions.  See open_clip/src/open_clip/model.py
(decode_action) and models/policy.py.

References
----------
- Radford et al. "Learning Transferable Visual Models From Natural Language
  Supervision." ICML 2021.  https://arxiv.org/abs/2103.00020
- Zhou et al. "Extract Free Dense Labels from CLIP." ECCV 2022. (MaskCLIP)
  https://arxiv.org/abs/2112.01071
- Kang et al. "CLIP-RT: Learning Language-Conditioned Robotic Policies from
  Natural Language Supervision." 2024.
- Ilharco et al. "OpenCLIP." https://github.com/mlfoundations/open_clip
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import open_clip
from open_clip.transformer import _expand_token
from PIL import Image
from torchvision import transforms as _tvt


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------
# CLIP-RT frames action prediction as a *language grounding* problem:
# the text query is not the raw task instruction but a motion question.
# This phrasing steers the text encoder toward motion semantics rather than
# object-classification semantics.
CLIP_RT_PROMPT = "what motion should the robot arm perform to complete the instruction '{}'?"

# Default model used throughout.  All checkpoints released with CLIP-RT use
# this architecture.  Changing it requires retraining from scratch.
DEFAULT_MODEL_NAME = "ViT-H-14-378-quickgelu"

# Valid fallback pretrained tags per architecture when no CLIP-RT checkpoint
# is given.  "openai" only exists for ViT-B/32, ViT-B/16, ViT-L/14 (+336).
_DEFAULT_PRETRAINED_TAGS = {
    "ViT-H-14-378-quickgelu": "dfn5b",  # Apple DFN5B — the only public ViT-H-378
    "ViT-H-14-quickgelu": "dfn5b",
}


def _load_openai_weights_nonstrict(model, model_name: str) -> None:
    """Load OpenAI CLIP weights into a CLIP-RT-fork model without strict checks.

    open_clip's stock OpenAI loader (``build_model_from_openai_state_dict``)
    does a *strict* ``load_state_dict``.  That fails on this fork because its
    ``CLIP`` class registers the text tower twice: once as the submodule
    ``self.text`` (→ ``text.*`` keys) and once via flat aliases that point at
    the *same* tensors (``self.transformer = self.text.transformer`` etc.).
    OpenAI's checkpoint only carries the flat names, so strict loading reports
    every ``text.*`` alias as "missing".

    We replicate only the download + TorchScript-extract steps that the stock
    loader performs, then load NON-strict.  The flat OpenAI keys populate the
    aliased tensors; because the ``text.*`` duplicates share storage with those
    aliases, the entire text tower ends up correctly initialised anyway.
    """
    from open_clip.pretrained import get_pretrained_url, download_pretrained_from_url

    url = get_pretrained_url(model_name, "openai")
    if not url:
        raise RuntimeError(
            f"No public OpenAI weights for '{model_name}'. OpenAI only released "
            "RN50/RN101/RN50x{4,16,64}, ViT-B-32, ViT-B-16, ViT-L-14 and "
            "ViT-L-14-336.  For ViT-H use pretrained='dfn5b' or a CLIP-RT "
            "checkpoint (models/clip_encoder.py:_DEFAULT_PRETRAINED_TAGS)."
        )

    ckpt_path = download_pretrained_from_url(url)

    # OpenAI checkpoints ship as TorchScript archives; fall back to a plain
    # state_dict for the rare non-JIT mirror.
    try:
        state_dict = torch.jit.load(ckpt_path, map_location="cpu").state_dict()
    except RuntimeError:
        state_dict = torch.load(ckpt_path, map_location="cpu")

    # Bookkeeping tensors baked into the JIT archive that are not parameters.
    for k in ("input_resolution", "context_length", "vocab_size"):
        state_dict.pop(k, None)

    # OpenAI weights are stored in fp16; upcast so they match the fp32 / AMP
    # path used everywhere else in this project.
    state_dict = {
        k: (v.float() if torch.is_floating_point(v) else v)
        for k, v in state_dict.items()
    }

    incompatible = model.load_state_dict(state_dict, strict=False)

    # Every OpenAI weight must find a home.  Unexpected keys mean the requested
    # architecture does not match the checkpoint (wrong model_name).  Missing
    # keys are EXPECTED here: the duplicated `text.*` aliases plus, when
    # use_action_decoder=True, the action decoder (`text2.*`, `action_head.*`)
    # which OpenAI never trained — those are meant to start from init.
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"OpenAI checkpoint for '{model_name}' has keys the model does not "
            f"expect (architecture mismatch): {incompatible.unexpected_keys[:8]}"
        )


class CLIPRTEncoder:
    """Wraps OpenCLIP's dual-encoder to produce L2-normalised embeddings.

    Parameters
    ----------
    model_name : str
        OpenCLIP architecture name.  Must match the pretrained checkpoint.
    model_path : str
        Path to a pretrained .pt checkpoint, or an OpenCLIP pretrained tag
        (e.g. "dfn5b", "openai").  Pass "" to auto-select a sensible public
        tag for the architecture (dfn5b for ViT-H, openai otherwise).
    use_action_decoder : bool
        Whether to attach the transformer action decoder head.  Must be True
        to run full CLIP-RT inference; False gives a standard dual-encoder
        useful for embedding-only tasks like the alignment smoke test.
    device : torch.device | str | None
        Target device.  None = "cuda" if available, else "cpu".
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        model_path: str = "",
        use_action_decoder: bool = True,
        device: torch.device | str | None = None,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = model_name

        # Resolve the pretrained source: explicit checkpoint/tag wins;
        # otherwise fall back to a tag that actually exists for this arch.
        pretrained = model_path if model_path else _DEFAULT_PRETRAINED_TAGS.get(
            model_name, "openai"
        )

        # open_clip.create_model_and_transforms returns:
        #   model      - the CLIP model (visual tower + text tower + optional decoder)
        #   _          - training transforms (we don't use these at inference)
        #   preprocess - inference-time image transforms
        # use_action_decoder=True is the CLIP-RT fork's flag that attaches the
        # transformer action decoder and action_head on top of the dual-encoder.
        #
        # The "openai" tag needs a detour: open_clip's OpenAI loader does a
        # STRICT state_dict load, which this fork's duplicated text-tower layout
        # (self.text.* aliased by flat self.transformer/etc.) makes impossible.
        # For every other tag (dfn5b, laion*, real CLIP-RT .pt) the normal path
        # loads non-strict and works.  So we branch: build the architecture with
        # pretrained=None, then fill OpenAI weights ourselves, non-strict.
        # See _load_openai_weights_nonstrict above for the full rationale.
        if isinstance(pretrained, str) and pretrained.lower() == "openai":
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=None,
                use_action_decoder=use_action_decoder,
            )
            _load_openai_weights_nonstrict(self.model, model_name)
        else:
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                model_name=model_name,
                pretrained=pretrained,
                use_action_decoder=use_action_decoder,
            )
        self.model.eval()
        self.model.to(self.device)

        # Tokeniser converts raw strings to integer token ids (BPE, 49408
        # tokens, max context 77) — same vocabulary as OpenAI CLIP.
        self.tokenizer = open_clip.get_tokenizer(model_name)

        # D: the shared embedding dimension.  1024 for ViT-H, 512 for ViT-B.
        self.embed_dim: int = self.model.visual.output_dim

        # Spatial resolution of the patch feature grid.  The visual tower
        # already computes this correctly for us (transformer.py sets
        # grid_size = image_size // patch_size per axis) — 27 for ViT-H-378,
        # 14 for ViT-B-16.  NB: visual.image_size / visual.patch_size are
        # TUPLES, so never divide them directly.
        gh, gw = self.model.visual.grid_size
        assert gh == gw, f"non-square patch grid {gh}x{gw} not supported"
        self.patch_grid_size: int = gh

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _autocast(self):
        """Mixed precision on GPU, plain fp32 on CPU (autocast('cuda') would
        crash on a CPU-only machine)."""
        return torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        )

    @property
    def preprocess_mode(self) -> str:
        """How the preprocess pipeline maps the camera frame onto the ViT input.

        "squash"      — the whole frame is resized to a square (DFN5B/CLIP-RT
                        checkpoints use resize_mode="squash").  Every pixel of
                        the frame is visible to the ViT.
        "center_crop" — shortest side resized then central square cropped
                        (OpenAI-style).  Pixels outside the central square are
                        NEVER seen by the ViT.

        scene_3d.Scene3DFeatureField needs this to map depth-image pixels to
        the correct patch — getting it wrong stretches the lookup by W/H
        (a 33% horizontal error on a 640×480 camera).
        """
        tfms = getattr(self.preprocess, "transforms", [])
        has_crop = any(isinstance(t, _tvt.CenterCrop) for t in tfms)
        return "center_crop" if has_crop else "squash"

    def _tower_trunk(self, img_tensor: torch.Tensor) -> torch.Tensor:
        """Run the visual tower up to (but excluding) the LAST transformer
        block.  Mirrors VisionTransformer.forward exactly — see
        open_clip/src/open_clip/transformer.py:502.

        Returns x in LND layout: [1 + G*G tokens, B, width].
        """
        v = self.model.visual
        assert v.attn_pool is None, (
            "dense feature extraction assumes standard CLS pooling, "
            "not attentional pooling"
        )

        x = v.conv1(img_tensor)                      # [B, width, G, G]
        x = x.reshape(x.shape[0], x.shape[1], -1)    # [B, width, G*G]
        x = x.permute(0, 2, 1)                       # [B, G*G, width]
        x = torch.cat(
            [_expand_token(v.class_embedding, x.shape[0]).to(x.dtype), x], dim=1
        )                                            # [B, 1+G*G, width]
        x = x + v.positional_embedding.to(x.dtype)
        x = v.patch_dropout(x)                       # Identity in eval configs
        x = v.ln_pre(x)
        x = x.permute(1, 0, 2)                       # NLD -> LND

        for block in v.transformer.resblocks[:-1]:
            x = block(x)
        return x

    def _last_block_dense(self, x: torch.Tensor, mode: str) -> torch.Tensor:
        """Run the LAST transformer block in a dense-friendly variant.

        Standard attention computes  out_i = out_proj( Σ_j softmax(q_i·k_j) v_j )
        — every output token is a *global mixture*, which is what destroys
        per-patch semantics in the final layer.  Three published remedies,
        differing in what they keep:

        mode="maskclip"  (Zhou et al. ECCV 2022)
            Identity attention on values: out_i = out_proj(v_i), KEEPING the
            residual stream and FFN.  Works for small ViTs (validated on our
            ViT-B-16 gate run) but fails for large ones: big ViTs repurpose
            low-information background patches as "register" tokens carrying
            GLOBAL image summaries (Darcet et al. 2023, "Vision Transformers
            Need Registers").  Those globalised features travel through the
            RESIDUAL stream — so background patches end up matching the text
            query better than the object itself, INVERTING the similarity
            map.  Observed exactly this on ViT-H-378/dfn5b: objects were the
            similarity minimum, all top-k mass on background.

        mode="clearclip"  (Lan et al. ECCV 2024)
            Identity attention on values, but DROP the residual and the FFN:
            the output is out_proj(v_i) alone.  Cutting the residual severs
            the globalised register signal; the value pathway keeps the
            local, text-aligned content.  The recipe ClearCLIP shows scales
            to large ViTs.

        mode="csa"  (SCLIP, Wang et al. 2024 — correlative self-attention)
            Attention weights from softmax(q·qᵀ) + softmax(k·kᵀ): each patch
            attends to patches SIMILAR to itself (self-correlation) instead
            of the trained q·k pattern, then values are mixed as usual.
            Also without residual/FFN here.  Slightly smooths features over
            each object region vs clearclip's purely per-patch output.
        """
        block = self.model.visual.transformer.resblocks[-1]
        attn = block.attn  # nn.MultiheadAttention
        x_ln = block.ln_1(x)
        L, B, E = x_ln.shape

        # in_proj packs [W_q; W_k; W_v] along dim 0 — compute all three.
        qkv = F.linear(x_ln, attn.in_proj_weight, attn.in_proj_bias)
        q, k, v = qkv.chunk(3, dim=-1)  # each [L, B, E]

        if mode == "csa":
            H = attn.num_heads
            dh = E // H
            scale = dh ** -0.5

            def to_heads(t: torch.Tensor) -> torch.Tensor:
                return t.reshape(L, B, H, dh).permute(1, 2, 0, 3)  # [B,H,L,dh]

            qh, kh, vh = to_heads(q), to_heads(k), to_heads(v)
            w = (
                torch.softmax(qh @ qh.transpose(-1, -2) * scale, dim=-1)
                + torch.softmax(kh @ kh.transpose(-1, -2) * scale, dim=-1)
            )
            out = (w @ vh).permute(2, 0, 1, 3).reshape(L, B, E)
        else:  # "maskclip" / "clearclip": identity attention on values
            out = v

        out = F.linear(out, attn.out_proj.weight, attn.out_proj.bias)

        if mode == "maskclip":
            # Original recipe: keep residual + FFN (ls_* are Identity here).
            x = x + block.ls_1(out)
            x = x + block.ls_2(block.mlp(block.ln_2(x)))
            return x
        # "clearclip" / "csa": attention output ONLY — no residual, no FFN.
        return out

    def _pool_and_project(self, x_lnd: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """ln_post + CLS/patch split + final projection into the shared space.

        Replicates the tail of VisionTransformer.forward, but — unlike the
        stock forward — projects the PATCH tokens through visual.proj too,
        so they land in the same embed_dim space as the text embeddings.

        Returns (pooled [B, D], tokens [B, G*G, D]), unnormalised.
        """
        v = self.model.visual
        assert v.pool_type == "tok", f"unsupported pool_type {v.pool_type!r}"

        x = x_lnd.permute(1, 0, 2)  # LND -> NLD: [B, 1+G*G, width]

        if v.final_ln_after_pool:
            pooled, tokens = x[:, 0], x[:, 1:]
            pooled = v.ln_post(pooled)
            tokens = v.ln_post(tokens)
        else:
            x = v.ln_post(x)
            pooled, tokens = x[:, 0], x[:, 1:]

        if v.proj is not None:
            pooled = pooled @ v.proj    # [B, D]
            tokens = tokens @ v.proj    # [B, G*G, D] — the crucial extra step
        return pooled, tokens

    # ------------------------------------------------------------------
    # Image encoding
    # ------------------------------------------------------------------

    def encode_image_dense(
        self,
        image: Image.Image,
        mode: str = "maskclip",
        normalize: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode an image into a global embedding AND per-patch features,
        both living in CLIP's shared image-text embedding space.

        The trunk (all blocks except the last) runs once; the last block runs
        twice — once normally (for the true global [CLS] embedding) and once
        in the requested dense mode (for the patch features).  Overhead over
        a plain encode_image call: one transformer block (~3% for ViT-H).

        Parameters
        ----------
        image : PIL.Image | torch.Tensor
            Raw RGB frame, OR an already-preprocessed tensor: [3, H, W] for a
            single image, [B, 3, H, W] for a batch (the training loop uses
            the batched form so the dense pass runs once per batch, not once
            per sample).
        mode : str
            "proj"      — stock patch tokens, ln_post + proj applied.
            "maskclip"  — value-identity attention, residual/FFN kept.
                          Validated on ViT-B-16; INVERTS on ViT-H (register
                          tokens — see _last_block_dense docstring).
            "clearclip" — value-identity attention, NO residual, NO FFN.
                          The large-ViT recipe.
            "csa"       — SCLIP correlative self-attention, no residual/FFN.
            Run scripts/smoke_alignment_test.py to pick per backbone.
        normalize : bool
            L2-normalise outputs (do this if you plan cosine similarities).

        Returns
        -------
        e_v : torch.Tensor
            Global image embedding — identical to encode_image(image).
            Shape [D] for single input, [B, D] for batched tensor input.
        patch_features : torch.Tensor
            One text-space feature per ViT patch, row-major from top-left.
            Shape [G*G, D] for single input, [B, G*G, D] for batched input.
        """
        assert mode in ("maskclip", "proj", "clearclip", "csa"), mode
        if isinstance(image, torch.Tensor):
            batched = image.dim() == 4
            img_tensor = (image if batched else image.unsqueeze(0)).to(self.device)
        else:
            batched = False
            img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

        with torch.no_grad(), self._autocast():
            trunk = self._tower_trunk(img_tensor)

            # Global embedding: normal last block, CLS token.
            last_block = self.model.visual.transformer.resblocks[-1]
            pooled, tokens_proj = self._pool_and_project(last_block(trunk))

            if mode == "proj":
                tokens = tokens_proj
            else:
                _, tokens = self._pool_and_project(self._last_block_dense(trunk, mode))

        e_v = pooled.float()
        patch_features = tokens.float()
        if normalize:
            e_v = F.normalize(e_v, dim=-1)
            patch_features = F.normalize(patch_features, dim=-1)
        if not batched:
            e_v = e_v.squeeze(0)
            patch_features = patch_features.squeeze(0)
        return e_v, patch_features

    def encode_image(
        self,
        image: Image.Image,
        return_patch_features: bool = False,
        patch_mode: str = "maskclip",
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """Encode a single PIL image to a global L2-normalised embedding.

        Parameters
        ----------
        image : PIL.Image
            Raw RGB image from the robot's camera.
        return_patch_features : bool
            If True, also return per-patch features (see encode_image_dense).
            NB: the underlying CLIP.encode_image has no per-call token flag —
            this delegates to the manual dense path.
        patch_mode : str
            Dense extraction mode, "maskclip" or "proj".

        Returns
        -------
        e_v : torch.Tensor, shape [D]
        patch_features : torch.Tensor, shape [G*G, D]   (only if requested)
        """
        if return_patch_features:
            return self.encode_image_dense(image, mode=patch_mode)

        img_tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        with torch.no_grad(), self._autocast():
            e_v = self.model.encode_image(img_tensor, normalize=True)
        return e_v.squeeze(0).float()

    # ------------------------------------------------------------------
    # Text encoding
    # ------------------------------------------------------------------

    def encode_text(self, instruction: str, use_motion_prompt: bool = True) -> torch.Tensor:
        """Encode a natural-language string to an L2-normalised embedding.

        Parameters
        ----------
        instruction : str
            Task instruction, e.g. "pick up the red cube".
        use_motion_prompt : bool
            If True (default), wrap the instruction in the CLIP-RT motion
            question first — the phrasing the action decoder was trained
            against.  Set False for plain grounding queries ("a photo of a
            red cube"), e.g. when querying the 3D feature field for OBJECT
            locations rather than motions — object phrases align with visual
            patches; motion questions do not.

        Returns
        -------
        e_t : torch.Tensor, shape [D]
        """
        text = CLIP_RT_PROMPT.format(instruction) if use_motion_prompt else instruction
        tokens = self.tokenizer(text).to(self.device)

        with torch.no_grad(), self._autocast():
            e_t = self.model.encode_text(tokens, normalize=True)
        return e_t.squeeze(0).float()

    # ------------------------------------------------------------------
    # Convenience: encode both at once
    # ------------------------------------------------------------------

    def encode(
        self,
        image: Image.Image,
        instruction: str,
        return_patch_features: bool = False,
        patch_mode: str = "maskclip",
    ) -> dict[str, torch.Tensor]:
        """Encode image and instruction together and return a named dict.

        Returns
        -------
        dict with keys:
          "e_v"            : global image embedding [D]
          "e_t"            : text embedding [D] (motion-prompted)
          "similarity"     : scalar cosine similarity in [-1, 1]
          "patch_features" : patch feature grid [G*G, D]  (if requested)
        """
        if return_patch_features:
            e_v, patch_features = self.encode_image_dense(image, mode=patch_mode)
        else:
            e_v = self.encode_image(image)

        e_t = self.encode_text(instruction)
        similarity = (e_v * e_t).sum()

        result = {"e_v": e_v, "e_t": e_t, "similarity": similarity}
        if return_patch_features:
            result["patch_features"] = patch_features
        return result

    # ------------------------------------------------------------------
    # Expose underlying model for the action decoder (used in policy.py)
    # ------------------------------------------------------------------

    @property
    def raw_model(self) -> torch.nn.Module:
        """Return the underlying OpenCLIP model with action decoder attached."""
        return self.model
