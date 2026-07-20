"""
models/
=======
Four-module stack that turns a (RGB-D frame, language instruction) pair
into a predicted action chunk:

  clip_encoder  →  scene_3d  →  fusion  →  policy
       ↓               ↓           ↓           ↓
  e_v, e_t, patch  F_3D field   fused      action
  features        (per-point)   R^{3D}     chunk

Import the high-level policy directly for inference:

    from models import CLIPRTPolicy, MLPPolicy
"""

from models.clip_encoder import CLIPRTEncoder
from models.scene_3d import Scene3DFeatureField
from models.fusion import FeatureFusion
from models.policy import MLPPolicy, CLIPRTPolicy

__all__ = [
    "CLIPRTEncoder",
    "Scene3DFeatureField",
    "FeatureFusion",
    "MLPPolicy",
    "CLIPRTPolicy",
]
