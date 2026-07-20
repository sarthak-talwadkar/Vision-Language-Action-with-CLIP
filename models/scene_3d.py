"""
models/scene_3d.py
==================
Lift 2D CLIP patch features into a 3D per-point feature field.

WHY 3D FEATURE LIFTING?
------------------------
Pure 2D CLIP embeddings are excellent at identifying *what* is in a scene but
are blind to *where* objects are in 3D space.  A manipulation policy needs
both: it must know that the cup is on the table AND that it is 35 cm in front
and 10 cm to the right at the current depth.

This module implements the OpenScene-style pipeline (Peng et al. CVPR 2023):
  1. Backproject the depth map to a 3D point cloud in camera coordinates.
  2. Each 3D point P_i already knows its pixel origin (u_i, v_i) because
     backprojection is a deterministic invertible mapping.
  3. Sample the CLIP patch feature at (u_i, v_i) from the ViT patch grid.
  4. Assign: F_3D[i] = CLIP_patch_feature(u_i, v_i).

Result: every 3D point carries a semantically rich CLIP feature vector.  We
can now query "which 3D points are semantically close to the phrase 'cup
handle'?" and get back a *physical location* in 3D space.

HOW CLIP PATCH FEATURES WORK
------------------------------
A ViT splits the image into a grid of non-overlapping patches (14×14 px for
ViT-H).  Each patch is linearly projected to a token, then transformer layers
mix information across patches via self-attention.  After N transformer layers,
each patch token carries a contextualised feature vector that encodes both the
local appearance of that patch AND global context from the rest of the image.

For a 378-px image with 14-px patches: grid is 27×27 = 729 patch tokens.
Each token is a vector of dimension D=1024 for ViT-H.

HOW BACKPROJECTION WORKS
--------------------------
Given:
  - pixel (u, v) in image coordinates
  - depth d metres
  - camera intrinsics K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

3D point in camera frame:
  X = (u - cx) * d / fx
  Y = (v - cy) * d / fy
  Z = d

This is the pinhole camera model.  Isaac Sim provides fx, fy, cx, cy as part
of the camera calibration.  The default Franka camera in Isaac Sim uses
fx=fy≈500 for a 640×480 image.

References
----------
- Peng et al. "OpenScene: 3D Scene Understanding with Open Vocabularies."
  CVPR 2023.  https://arxiv.org/abs/2211.15654
- He et al. "Masked Autoencoders Are Scalable Vision Learners." CVPR 2022.
  (Background on ViT patch tokenisation.)
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass


@dataclass
class CameraIntrinsics:
    """Pinhole camera calibration parameters.

    These come from Isaac Sim's Camera sensor calibration.  The defaults match
    a 640×480 Realsense-style sensor mounted on the Franka end-effector in the
    standard Isaac Sim manipulation scene.

    Attributes
    ----------
    fx, fy : float
        Focal lengths in pixels.  Higher = more zoom = narrower FOV.
    cx, cy : float
        Principal point (image centre in pixels).  Usually ≈ W/2, H/2.
    width, height : int
        Image resolution in pixels.
    """
    fx: float = 500.0
    fy: float = 500.0
    cx: float = 320.0
    cy: float = 240.0
    width: int = 640
    height: int = 480


class Scene3DFeatureField:
    """Builds a per-point 3D CLIP feature field from an RGB-D observation.

    Workflow
    --------
    1. Call `build(depth, patch_features)` with:
       - depth  : H×W float32 numpy array (metres, from Isaac Sim camera)
       - patch_features : N_patches×D tensor (from CLIPRTEncoder.encode_image)
    2. The field F_3D ∈ R^{N×D} is stored in `self.points_3d` and
       `self.features`.
    3. Query the field with `query(text_embedding)` to get a spatially-grounded
       feature that is aligned with the instruction.

    Parameters
    ----------
    intrinsics : CameraIntrinsics
        Camera calibration for the scene.  Must match the camera used to
        capture the depth map.
    patch_grid_size : int
        Number of patches along one side of the ViT patch grid.
        For ViT-H-14-378: 378 // 14 = 27.  This is only a default — build()
        re-infers it from the actual token count, so ViT-B (14×14 grid)
        features work without reconfiguration.
    max_depth : float
        Discard points farther than this distance (metres).  Points beyond
        arm reach are irrelevant for manipulation.
    min_depth : float
        Discard points closer than this (usually the robot body or table edge
        artefacts in depth sensors).
    preprocess_mode : str
        How CLIP's preprocess pipeline mapped the camera frame onto the
        square ViT input — REQUIRED to look up the right patch for a pixel:
          "squash"      : whole frame resized to a square (DFN5B / CLIP-RT).
          "center_crop" : central square cropped (OpenAI-style checkpoints);
                          pixels outside the square were never seen by the
                          ViT and are clamped to the border patch.
        Use CLIPRTEncoder.preprocess_mode to get the correct value — a
        mismatch stretches the feature lookup horizontally by W/H (33% on a
        640×480 camera), assigning features to the wrong objects.
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics | None = None,
        patch_grid_size: int = 27,
        max_depth: float = 2.0,
        min_depth: float = 0.05,
        device: torch.device | str | None = None,
        preprocess_mode: str = "squash",
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        assert preprocess_mode in ("squash", "center_crop"), preprocess_mode
        self.K = intrinsics or CameraIntrinsics()
        self.patch_grid_size = patch_grid_size
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.preprocess_mode = preprocess_mode
        self.device = torch.device(device)

        # Set after calling build()
        self.points_3d: torch.Tensor | None = None   # [N, 3]  XYZ in camera frame
        self.features: torch.Tensor | None = None    # [N, D]  CLIP feature per point

    # ------------------------------------------------------------------
    # Step 1: backproject depth → 3D point cloud
    # ------------------------------------------------------------------

    def _backproject(
        self, depth: np.ndarray, stride: int = 1
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert a depth image to a set of 3D points and their pixel locations.

        Parameters
        ----------
        depth : np.ndarray, shape [H, W], dtype float32
            Depth in metres.  0 or NaN = no measurement.
        stride : int
            Take every `stride`-th pixel along both axes.  stride=4 turns a
            640×480 frame (307k points) into ~19k points — plenty for a
            feature field whose native resolution is a 27×27 patch grid,
            and ~16× cheaper to build.  Pixel COORDINATES stay in the
            original image frame, so the feature lookup is unaffected.

        Returns
        -------
        points_3d : torch.Tensor, shape [N, 3]
            3D coordinates (X, Y, Z) in camera frame.
        pixel_coords : torch.Tensor, shape [N, 2]
            Corresponding (u, v) pixel coordinates (float, for bilinear sampling).
        """
        H, W = depth.shape
        K = self.K

        # Build pixel coordinate grids (subsampled by `stride`, but holding
        # ORIGINAL pixel indices).  v = row = y direction, u = col = x.
        v_grid, u_grid = np.meshgrid(
            np.arange(0, H, stride, dtype=np.float32),
            np.arange(0, W, stride, dtype=np.float32),
            indexing="ij",
        )

        # Flatten everything.  Shape: [(H/stride)*(W/stride)]
        u_flat = u_grid.flatten()
        v_flat = v_grid.flatten()
        d_flat = depth[::stride, ::stride].flatten()

        # Filter: keep only pixels with valid, in-range depth.
        valid = (d_flat > self.min_depth) & (d_flat < self.max_depth)
        u_valid = u_flat[valid]
        v_valid = v_flat[valid]
        d_valid = d_flat[valid]

        # Pinhole backprojection (vectorised).
        X = (u_valid - K.cx) * d_valid / K.fx
        Y = (v_valid - K.cy) * d_valid / K.fy
        Z = d_valid

        points_3d = torch.tensor(
            np.stack([X, Y, Z], axis=1), dtype=torch.float32, device=self.device
        )
        pixel_coords = torch.tensor(
            np.stack([u_valid, v_valid], axis=1), dtype=torch.float32, device=self.device
        )
        return points_3d, pixel_coords

    # ------------------------------------------------------------------
    # Step 2: sample CLIP patch features at each pixel location
    # ------------------------------------------------------------------

    def _sample_patch_features(
        self,
        patch_features: torch.Tensor,
        pixel_coords: torch.Tensor,
    ) -> torch.Tensor:
        """Bilinearly sample the patch feature grid at arbitrary pixel locations.

        The ViT patch grid is coarser than the depth image.  For a 27×27 patch
        grid on a 640×480 depth image, each patch covers ~23×17 pixels.  We
        use bilinear interpolation to assign a smooth feature to each pixel.

        Parameters
        ----------
        patch_features : torch.Tensor, shape [N_patches, D]
            Flattened patch tokens from the ViT (row-major, top-left first).
        pixel_coords : torch.Tensor, shape [N_points, 2]
            (u, v) pixel coordinates in the *depth image* space (640×480).

        Returns
        -------
        sampled : torch.Tensor, shape [N_points, D]
            CLIP feature vector assigned to each 3D point.
        """
        N_patches, D = patch_features.shape
        # Infer the grid side from the token count rather than trusting the
        # constructor default: 729 tokens → 27×27 (ViT-H-378), 196 → 14×14
        # (ViT-B-16).  A non-square count means the [CLS] token was passed in
        # by mistake — fail loudly rather than silently mis-reshaping.
        G = int(round(N_patches ** 0.5))
        assert G * G == N_patches, (
            f"got {N_patches} patch tokens, not a square grid — did the "
            f"[CLS] token slip in?  Expected e.g. 729 (27x27) or 196 (14x14)."
        )
        self.patch_grid_size = G

        # Reshape flat patch tokens back to a spatial grid: [1, D, G, G]
        # Unsqueeze batch dim and move channels first for F.grid_sample.
        patch_grid = patch_features.view(G, G, D).permute(2, 0, 1).unsqueeze(0)
        # patch_grid: [1, D, G, G]

        # ------------------------------------------------------------------
        # Map depth-image pixels into the coordinate frame the ViT ACTUALLY
        # saw.  The preprocess pipeline does not feed the raw (e.g. 640×480)
        # frame to the ViT — two geometries exist, depending on checkpoint:
        #
        #   "squash"      (DFN5B / CLIP-RT): whole frame resized to 378×378.
        #                 A pixel at fraction (u/W, v/H) of the frame lands at
        #                 the same fraction of the patch grid.
        #   "center_crop" (OpenAI-style): shortest side resized, then central
        #                 square cropped.  Only the central square exists in
        #                 the patch grid; outside pixels clamp to the border
        #                 patch via padding_mode="border".
        #
        # Using raw u/(W-1) for a non-square camera stretches the horizontal
        # lookup by W/H — features would land on the wrong object.
        # ------------------------------------------------------------------
        K = self.K
        if self.preprocess_mode == "center_crop":
            side = float(min(K.width, K.height))
            u_frac = (pixel_coords[:, 0] - (K.width - side) / 2.0) / side
            v_frac = (pixel_coords[:, 1] - (K.height - side) / 2.0) / side
        else:  # "squash"
            u_frac = pixel_coords[:, 0] / float(K.width)
            v_frac = pixel_coords[:, 1] / float(K.height)

        # With align_corners=False, grid coordinate -1 is the LEFT EDGE of
        # the first patch and +1 the RIGHT EDGE of the last, so patch centres
        # sit at fractions (i + 0.5)/G — exactly the geometry of a ViT patch
        # grid.  (align_corners=True would wrongly pin patch centres to the
        # image corners, shifting every lookup by half a patch.)
        u_norm = u_frac * 2 - 1
        v_norm = v_frac * 2 - 1

        # F.grid_sample expects a grid of shape [1, N, 1, 2] for N query points.
        # grid[..., 0] = x (horizontal = u), grid[..., 1] = y (vertical = v).
        sample_grid = torch.stack([u_norm, v_norm], dim=1).unsqueeze(0).unsqueeze(2)
        # sample_grid: [1, N_points, 1, 2]

        # Bilinear sampling.  Output: [1, D, N_points, 1]
        sampled = F.grid_sample(
            patch_grid,
            sample_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        # Reshape to [N_points, D]
        sampled = sampled.squeeze(0).squeeze(-1).permute(1, 0)
        return sampled

    # ------------------------------------------------------------------
    # Public: build the full 3D feature field
    # ------------------------------------------------------------------

    def build(
        self,
        depth: np.ndarray,
        patch_features: torch.Tensor,
        pixel_stride: int = 1,
    ) -> "Scene3DFeatureField":
        """Construct the 3D CLIP feature field.

        Parameters
        ----------
        depth : np.ndarray, shape [H, W]
            Depth image from Isaac Sim's Camera sensor.  Values in metres.
        patch_features : torch.Tensor, shape [N_patches, D]
            Patch-level feature tokens from CLIPRTEncoder.encode_image_dense()
            (mode="maskclip" — validated by the gate experiment).
        pixel_stride : int
            Subsample the depth image by this factor (see _backproject).
            Use 1 for visualisation, 2 for closed-loop inference, 4 for
            per-sample field building inside the training loop.

        Returns
        -------
        self (for method chaining)
        """
        # Step 1: depth → 3D points + pixel locations
        self.points_3d, pixel_coords = self._backproject(depth, stride=pixel_stride)

        # Step 2: sample CLIP features at each pixel → assign to 3D points
        # L2-normalise so cosine similarity queries work correctly later.
        raw_features = self._sample_patch_features(patch_features, pixel_coords)
        self.features = F.normalize(raw_features, dim=-1)

        return self

    # ------------------------------------------------------------------
    # Public: spatial query
    # ------------------------------------------------------------------

    def query(
        self,
        text_embedding: torch.Tensor,
        top_k: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Find the 3D points most semantically aligned with the instruction.

        This is the key query used by fusion.py:
          "Which points in 3D space are most relevant to the text instruction?"

        For "pick up the cup", this will return points near the cup.
        For "place on the blue plate", this will return points near the plate.

        Parameters
        ----------
        text_embedding : torch.Tensor, shape [D]
            L2-normalised text embedding from CLIPRTEncoder.encode_text().
        top_k : int
            Number of top-matching points to return.  64 is a good default:
            enough to cover an object surface without including too much
            background noise.

        Returns
        -------
        top_features : torch.Tensor, shape [top_k, D]
            CLIP features of the k most relevant 3D points.
        top_points : torch.Tensor, shape [top_k, 3]
            XYZ coordinates of those points (camera frame, metres).
        """
        assert self.features is not None, "Call build() before query()."

        # Cosine similarity between all N point features and the text embedding.
        # text_embedding: [D] → unsqueeze to [1, D] for matrix multiply.
        # similarities: [N]
        similarities = (self.features @ text_embedding.unsqueeze(1)).squeeze(1)

        # Pick the top-k most similar points.
        k = min(top_k, self.features.shape[0])
        top_indices = torch.topk(similarities, k=k).indices

        top_features = self.features[top_indices]  # [k, D]
        top_points   = self.points_3d[top_indices]  # [k, 3]

        return top_features, top_points

    def aggregate(
        self,
        text_embedding: torch.Tensor,
        top_k: int = 64,
    ) -> torch.Tensor:
        """Mean-pool the top-k point features into a single scene summary.

        Aggregation compresses the variable-size point cloud into a fixed-size
        vector suitable for concatenation in fusion.py.

        Returns
        -------
        f_3d : torch.Tensor, shape [D]
            L2-normalised aggregate 3D feature.
        """
        top_features, _ = self.query(text_embedding, top_k=top_k)
        f_3d = top_features.mean(dim=0)
        return F.normalize(f_3d, dim=0)

    def localize(
        self,
        text_embedding: torch.Tensor,
        top_k: int = 64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Text-conditioned 3D summary: WHAT the target looks like + WHERE it is.

        This is what the policy's 3D injection consumes (policy.py, use_3d).
        aggregate() alone returns only appearance features — the mean of
        top-k point FEATURES contains no coordinates, so a policy fed only
        f_3d would receive zero geometric information.  The centroid is the
        actual geometry: the mean 3D position (camera frame, metres) of the
        instruction-relevant region, e.g. "the red cube is 0.4 m ahead,
        0.1 m left, at table height".

        Query with an OBJECT-phrase embedding (encode_text(...,
        use_motion_prompt=False)) — the gate experiment validated alignment
        for object phrases, not motion questions.

        Returns
        -------
        f_3d : torch.Tensor, shape [D]
            L2-normalised mean feature of the top-k matching points.
        centroid : torch.Tensor, shape [3]
            Mean XYZ of those points (camera frame, metres).
        """
        top_features, top_points = self.query(text_embedding, top_k=top_k)
        f_3d = F.normalize(top_features.mean(dim=0), dim=0)
        centroid = top_points.mean(dim=0)
        return f_3d, centroid
