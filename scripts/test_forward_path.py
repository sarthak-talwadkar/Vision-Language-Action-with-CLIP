r"""
scripts/test_forward_path.py
============================
End-to-end SHAPE + wiring check for the model forward path (Stage 2 → 3):

    CLIPRTEncoder  →  Scene3DFeatureField  →  FeatureFusion  →  policy

Runs entirely in the training .venv (no Isaac Sim) on GPU if available, using
the real RGB-D frame + intrinsics captured from Isaac Sim
(outputs/capture/{frame.png, depth.npy, intrinsics.json}).  It exercises BOTH
policy heads:

  * MLPPolicy         : fused [3D] -> single action [7]
  * CLIPRTPolicy      : (image, instruction) -> action chunk [8, 7]
                        with use_3d=False (pure 2D) and use_3d=True (3D inject)

Weights are random/pretrained-CLIP (no CLIP-RT action checkpoint), so the
NUMBERS are meaningless — this verifies the tensors flow and shapes line up,
which is exactly what we need before writing the training loop.

Run:
    .venv\Scripts\python.exe scripts\test_forward_path.py
"""

import sys
import json
from pathlib import Path

# Windows cp1252 console can't encode Unicode arrows; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
try:
    import open_clip  # noqa: F401
    _ = open_clip.create_model_and_transforms
except (ImportError, AttributeError):
    sys.path.insert(0, str(REPO_ROOT / "open_clip" / "src"))
    import open_clip  # noqa: F401

from models.clip_encoder import CLIPRTEncoder
from models.scene_3d import CameraIntrinsics, Scene3DFeatureField
from models.fusion import FeatureFusion
from models.policy import MLPPolicy, CLIPRTPolicy, ACTION_DIM, ACTION_CHUNK


def check(name, got, expected):
    ok = tuple(got) == tuple(expected)
    print(f"  {'OK ' if ok else 'XX '} {name:28s} {tuple(got)}  (expected {tuple(expected)})")
    assert ok, f"{name}: got {tuple(got)}, expected {tuple(expected)}"


def main() -> int:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    instr = "pick up the red cube"
    print(f"device={device}  instruction='{instr}'")

    # ---- real inputs from the Isaac capture ----
    cap = REPO_ROOT / "outputs" / "capture"
    image = Image.open(cap / "frame.png").convert("RGB")
    depth = np.load(cap / "depth.npy")
    K = json.loads((cap / "intrinsics.json").read_text())
    H, W = depth.shape

    # ---- encoder (ViT-B/16 for speed; action decoder ON for CLIPRTPolicy) ----
    print("\n[1] Encoder")
    enc = CLIPRTEncoder(model_name="ViT-B-16", model_path="openai",
                        use_action_decoder=True, device=device)
    D = enc.embed_dim
    print(f"  embed_dim={D}  patch_grid={enc.patch_grid_size}  preprocess={enc.preprocess_mode}")

    # ---- encode ----
    print("\n[2] Encode image + text")
    e_v = enc.encode_image(image)
    e_t = enc.encode_text(instr)                       # motion prompt (for the policy)
    e_t_obj = enc.encode_text(instr, use_motion_prompt=False)  # object phrase (for 3D localise)
    _, patch = enc.encode_image_dense(image, mode="maskclip")
    check("e_v", e_v.shape, (D,))
    check("e_t", e_t.shape, (D,))
    check("patch_features", patch.shape, (enc.patch_grid_size ** 2, D))

    # ---- 3D feature field with the REAL intrinsics ----
    print("\n[3] Scene3DFeatureField")
    intr = CameraIntrinsics(fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"], width=W, height=H)
    scene = Scene3DFeatureField(
        intrinsics=intr, device=device, preprocess_mode=enc.preprocess_mode,
    ).build(depth, patch.to(device))
    print(f"  points={tuple(scene.points_3d.shape)}  features={tuple(scene.features.shape)}")
    assert scene.features.shape[1] == D

    # ---- fusion + MLP policy ----
    print("\n[4] FeatureFusion -> MLPPolicy")
    fusion = FeatureFusion(embed_dim=D, top_k=64, use_3d=True)
    fused = fusion.fuse(e_v.to(device), e_t.to(device), scene)
    check("fused (3D)", fused.shape, (fusion.output_dim,))
    check("fusion.output_dim", (fusion.output_dim,), (3 * D,))
    mlp = MLPPolicy(input_dim=fusion.output_dim).to(device)
    action = mlp(fused)
    check("MLPPolicy action", action.shape, (ACTION_DIM,))

    # ---- CLIP-RT policy, 2D (use_3d=False) ----
    print("\n[5] CLIPRTPolicy.predict  (use_3d=False)")
    pol2d = CLIPRTPolicy(enc, fusion, use_3d=False).to(device)
    chunk2d = np.array(pol2d.predict(image, instr), dtype=np.float32)
    check("action chunk (2D)", chunk2d.shape, (ACTION_CHUNK, ACTION_DIM))

    # ---- CLIP-RT policy, 3D injection (use_3d=True) ----
    print("\n[6] CLIPRTPolicy.predict  (use_3d=True, 3D injection)")
    pol3d = CLIPRTPolicy(enc, fusion, use_3d=True).to(device)
    chunk3d = np.array(pol3d.predict(image, instr, scene=scene), dtype=np.float32)
    check("action chunk (3D)", chunk3d.shape, (ACTION_CHUNK, ACTION_DIM))

    # zero-init proj_3d ⇒ 3D path must equal 2D path at initialisation
    same = np.allclose(chunk2d, chunk3d, atol=1e-4)
    print(f"  zero-init 3D injection is identity vs 2D at init: {same}")

    print("\n==================================================")
    print("FORWARD PATH OK -- encoder -> scene3d -> fusion -> policy all wired, shapes correct.")
    print(f"  MLPPolicy:    fused[{3*D}] -> action[{ACTION_DIM}]")
    print(f"  CLIPRTPolicy: (image, text) -> chunk[{ACTION_CHUNK}, {ACTION_DIM}]")
    print("==================================================")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
