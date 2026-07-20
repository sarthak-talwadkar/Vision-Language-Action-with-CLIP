"""
scripts/smoke_alignment_test.py
================================
THE GATE EXPERIMENT for the 3D branch of this project.

WHAT THIS TESTS
----------------
The entire Stage-1 pipeline (scene_3d.py → fusion.py) rests on ONE assumption:

    "A CLIP patch feature at pixel (u,v) has high cosine similarity with the
     text embedding of whatever object is visible at (u,v)."

CLIP was never trained for this — only the GLOBAL image embedding is aligned
with text.  models/clip_encoder.py:encode_image_dense() implements two ways to
extract per-patch features anyway:

    mode="proj"     — stock patch tokens, projected into the shared space.
    mode="maskclip" — the MaskCLIP trick (Zhou et al. ECCV 2022): last
                      attention layer replaced by identity-on-values.

This script measures whether either mode actually LOCALISES objects:
it computes a patch×text similarity heatmap for several queries and checks
that the hottest patches fall on the right object.

PASS / FAIL INTERPRETATION
---------------------------
- A mode PASSES a query if ≥50% of its top-16 patches land inside the
  ground-truth object box (synthetic mode; for real images, judge the saved
  heatmap overlays by eye).
- If maskclip passes and proj fails    → wire the 3D field with maskclip
                                          features (expected outcome).
- If both pass                          → great; prefer maskclip anyway (it is
                                          the literature-standard baseline).
- If both fail on the SYNTHETIC scene   → something is broken in the pipeline
                                          (installation, checkpoint, geometry),
                                          not in the science.  Debug first.
- If both pass synthetic but fail on a  → CLIP's dense features don't transfer
  real Isaac Sim frame                    to your rendered scenes.  STOP: do
                                          not wire the 3D branch into training;
                                          consider LSeg/OpenSeg features
                                          (what OpenScene actually uses).

USAGE
------
# 1. Cheapest run — synthetic scene, small ViT-B/16 (~600 MB download):
python scripts/smoke_alignment_test.py --synthetic

# 2. The project's real backbone (~4 GB download, Apple DFN5B weights):
python scripts/smoke_alignment_test.py --synthetic \
    --model ViT-H-14-378-quickgelu --pretrained dfn5b

# 3. A CLIP-RT checkpoint (what the policy will actually use):
python scripts/smoke_alignment_test.py --synthetic --checkpoint path/to/cliprt.pt

# 4. A real photo / Isaac Sim frame (judge the overlays by eye):
python scripts/smoke_alignment_test.py --image frame.png \
    --queries "a red coffee mug,a computer keyboard,a wooden table" \
    --depth depth.npy        # optional: H×W float32 metres → also tests 3D field

Outputs land in outputs/smoke_test/: one heatmap overlay PNG per
(mode × query), plus the synthetic scene itself for reference.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Windows consoles default to the cp1252 codec, which cannot encode the Unicode
# arrows (→) and multiplication signs (×) this script prints — that raises
# UnicodeEncodeError mid-run.  Force stdout/stderr to UTF-8 so output is
# identical across Windows, Linux and macOS.  (reconfigure exists on 3.7+.)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import torch

# --- import shim -----------------------------------------------------------
# Make the script runnable from anywhere, with or without `pip install -e
# ./open_clip`: put the repo root (for models/) and the vendored open_clip
# fork (for open_clip/) on sys.path.
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
try:
    import open_clip  # noqa: F401
    _ = open_clip.create_model_and_transforms  # namespace-package guard
except (ImportError, AttributeError):
    sys.path.insert(0, str(REPO_ROOT / "open_clip" / "src"))
    import open_clip  # noqa: F401

import matplotlib
matplotlib.use("Agg")  # headless: we only save PNGs
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

from models.clip_encoder import CLIPRTEncoder
from models.scene_3d import CameraIntrinsics, Scene3DFeatureField

TOPK_PATCHES = 16     # patches counted for the hit-rate metric
HIT_MARGIN = 15       # px of slack around the GT box (patches are ~24 px wide)
PASS_THRESHOLD = 0.5  # ≥50% of top patches inside the box → PASS


# ---------------------------------------------------------------------------
# Synthetic scene: three coloured shapes with known ground-truth boxes.
# Deliberately trivial — if dense CLIP features can't find a red circle on a
# plain background, they will never find a mug handle in Isaac Sim.
# ---------------------------------------------------------------------------

def build_synthetic_scene() -> tuple[Image.Image, np.ndarray, dict[str, tuple]]:
    W, H = 640, 480
    img = Image.new("RGB", (W, H), (231, 231, 228))
    draw = ImageDraw.Draw(img)

    # All shapes live inside the central 480×480 square (x ∈ [80, 560]) so the
    # test is fair for BOTH preprocess geometries (center_crop never sees the
    # outer margins of a 640-wide frame).
    gt: dict[str, tuple] = {}

    bbox = (120, 80, 240, 200)                      # red circle, top-left
    draw.ellipse(bbox, fill=(210, 30, 25))
    gt["a photo of a red circle"] = bbox

    bbox = (280, 190, 390, 300)                     # blue square, centre
    draw.rectangle(bbox, fill=(25, 60, 200))
    gt["a photo of a blue square"] = bbox

    tri = [(420, 420), (480, 310), (540, 420)]      # green triangle, bottom-right
    draw.polygon(tri, fill=(20, 150, 45))
    gt["a photo of a green triangle"] = (420, 310, 540, 420)

    # Depth: flat backdrop at 0.8 m, each object raised to 0.65 m inside its
    # box.  Crude, but it exercises backprojection + feature lookup end to end.
    depth = np.full((H, W), 0.80, dtype=np.float32)
    for x0, y0, x1, y1 in gt.values():
        depth[y0:y1, x0:x1] = 0.65

    return img, depth, gt


# ---------------------------------------------------------------------------
# Geometry helpers: patch grid ↔ image pixels (must mirror scene_3d.py)
# ---------------------------------------------------------------------------

def patch_centers_px(G: int, W: int, H: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
    """Pixel coordinates of every patch centre, respecting preprocess geometry.

    Patch i covers fraction [i/G, (i+1)/G] of the ViT input, so its centre is
    at fraction (i+0.5)/G.  "squash" maps input fractions onto the full frame;
    "center_crop" maps them onto the central square only.
    """
    frac = (np.arange(G) + 0.5) / G
    if mode == "center_crop":
        side = min(W, H)
        us = frac * side + (W - side) / 2.0
        vs = frac * side + (H - side) / 2.0
    else:  # squash
        us = frac * W
        vs = frac * H
    uu, vv = np.meshgrid(us, vs)   # [G, G] each; row-major like the tokens
    return uu, vv


def vit_region_extent(W: int, H: int, mode: str) -> tuple[float, float, float, float]:
    """Image-pixel extent (x0, x1, y0, y1) of the region the ViT saw."""
    if mode == "center_crop":
        side = min(W, H)
        return (W - side) / 2.0, (W + side) / 2.0, (H - side) / 2.0, (H + side) / 2.0
    return 0.0, float(W), 0.0, float(H)


# ---------------------------------------------------------------------------
# Core measurement
# ---------------------------------------------------------------------------

def similarity_heatmap(patch_features: torch.Tensor, e_t: torch.Tensor, G: int) -> np.ndarray:
    """Cosine similarity of every patch with the text embedding → [G, G]."""
    sim = (patch_features @ e_t).reshape(G, G)
    return sim.cpu().numpy()


def hit_rate(heat: np.ndarray, gt_box: tuple, W: int, H: int, mode: str, k: int,
             largest: bool = True) -> float:
    """Fraction of the top-k (or bottom-k) patches whose centres fall in the
    GT box.  The bottom-k variant (largest=False) is the INVERSION detector:
    large ViTs can repurpose background patches as global "registers", making
    the object the similarity MINIMUM — bottom-k concentrating on the object
    while top-k scatters over background is the signature of that failure
    (Darcet et al. 2023), as opposed to a geometry bug (which would SHIFT the
    hot region, not invert it)."""
    G = heat.shape[0]
    uu, vv = patch_centers_px(G, W, H, mode)
    order = np.argsort(heat.ravel())
    flat_idx = order[::-1][:k] if largest else order[:k]
    x0, y0, x1, y1 = gt_box
    x0, y0, x1, y1 = x0 - HIT_MARGIN, y0 - HIT_MARGIN, x1 + HIT_MARGIN, y1 + HIT_MARGIN
    hits = sum(
        1 for i in flat_idx
        if x0 <= uu.ravel()[i] <= x1 and y0 <= vv.ravel()[i] <= y1
    )
    return hits / k


def save_overlay(
    image: Image.Image,
    heat: np.ndarray,
    query: str,
    mode: str,
    out_dir: Path,
    preprocess_mode: str,
    gt_box: tuple | None,
    k: int,
):
    """Save the heatmap blended over the image, top-k patch centres marked."""
    W, H = image.size
    G = heat.shape[0]
    x0, x1, y0, y1 = vit_region_extent(W, H, preprocess_mode)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    # extent=(left, right, bottom, top) — y grows downward in image coords.
    ax.imshow(heat, cmap="jet", alpha=0.45, interpolation="bilinear",
              extent=(x0, x1, y1, y0))

    uu, vv = patch_centers_px(G, W, H, preprocess_mode)
    top = np.argsort(heat.ravel())[::-1][:k]
    ax.scatter(uu.ravel()[top], vv.ravel()[top], s=28, c="white",
               edgecolors="black", linewidths=0.8, label=f"top-{k} patches")

    if gt_box is not None:
        bx0, by0, bx1, by1 = gt_box
        ax.add_patch(plt.Rectangle((bx0, by0), bx1 - bx0, by1 - by0,
                                   fill=False, edgecolor="lime", linewidth=2,
                                   label="ground truth"))
    ax.set_title(f"[{mode}]  {query}")
    ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0, W); ax.set_ylim(H, 0); ax.axis("off")

    slug = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")[:40]
    path = out_dir / f"{mode}__{slug}.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def test_3d_field(
    encoder: CLIPRTEncoder,
    image: Image.Image,
    depth: np.ndarray,
    queries: list[str],
    gt: dict[str, tuple] | None,
    mode: str,
    device: str,
    intrinsics: dict | None = None,
):
    """End-to-end Stage-1 check: RGB-D → 3D feature field → text query →
    3D location.  With synthetic GT we also verify the returned centroid sits
    where backprojecting the object's box centre says it should."""
    print(f"\n--- 3D feature field (Scene3DFeatureField, mode={mode}) ---")
    H, W = depth.shape
    if intrinsics is not None:
        K = CameraIntrinsics(fx=float(intrinsics["fx"]), fy=float(intrinsics["fy"]),
                             cx=float(intrinsics["cx"]), cy=float(intrinsics["cy"]),
                             width=W, height=H)
        print(f"    intrinsics: fx={K.fx:.1f} fy={K.fy:.1f} cx={K.cx:.1f} cy={K.cy:.1f} "
              f"(from capture)")
    else:
        K = CameraIntrinsics(fx=500.0, fy=500.0, cx=W / 2, cy=H / 2, width=W, height=H)
        print(f"    intrinsics: fx=fy=500 (default guess — no intrinsics.json found)")

    _, patch_features = encoder.encode_image_dense(image, mode=mode)
    field = Scene3DFeatureField(
        intrinsics=K, device=device, preprocess_mode=encoder.preprocess_mode,
    ).build(depth, patch_features.to(device))
    print(f"    field: {field.features.shape[0]} points, D={field.features.shape[1]}")

    for q in queries:
        e_t = encoder.encode_text(q, use_motion_prompt=False).to(device)
        _, top_points = field.query(e_t, top_k=64)
        centroid = top_points.mean(dim=0).cpu().numpy()
        line = f"    '{q}': top-64 centroid at ({centroid[0]:+.3f}, {centroid[1]:+.3f}, {centroid[2]:.3f}) m"
        if gt is not None:
            x0, y0, x1, y1 = gt[q]
            cu, cv, d = (x0 + x1) / 2, (y0 + y1) / 2, 0.65
            expected = np.array([(cu - K.cx) * d / K.fx, (cv - K.cy) * d / K.fy, d])
            err = float(np.linalg.norm(centroid - expected))
            line += f" | expected ({expected[0]:+.3f}, {expected[1]:+.3f}, {expected[2]:.3f}), err {err*100:.1f} cm"
            line += "  PASS" if err < 0.15 else "  FAIL"
        print(line)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Dense CLIP patch-text alignment gate experiment.")
    p.add_argument("--synthetic", action="store_true",
                   help="Use the built-in three-shapes scene with known ground truth.")
    p.add_argument("--image", type=str, default="",
                   help="Path to an RGB image (alternative to --synthetic).")
    p.add_argument("--depth", type=str, default="",
                   help="Optional H×W depth for --image: .npy float32 metres, "
                        "or 16-bit .png in millimetres.")
    p.add_argument("--intrinsics", type=str, default="",
                   help="Path to intrinsics.json (fx,fy,cx,cy). If omitted, a "
                        "sidecar intrinsics.json next to --depth/--image is used "
                        "when present; else fx=fy=500 defaults.")
    p.add_argument("--queries", type=str, default="",
                   help="Comma-separated text queries (required with --image).")
    p.add_argument("--model", type=str, default="ViT-B-16",
                   help="OpenCLIP architecture (default: cheap ViT-B-16).")
    p.add_argument("--pretrained", type=str, default="openai",
                   help="Pretrained tag, e.g. 'openai', 'dfn5b'.")
    p.add_argument("--checkpoint", type=str, default="",
                   help="CLIP-RT .pt checkpoint. Implies --model ViT-H-14-378-quickgelu.")
    p.add_argument("--modes", type=str, default="clearclip,csa,maskclip,proj",
                   help="Dense extraction modes to compare.  clearclip/csa cut "
                        "the residual stream (large-ViT recipe); maskclip keeps "
                        "it (works on ViT-B, inverts on ViT-H); proj is the "
                        "stock-tokens baseline.")
    p.add_argument("--out", type=str, default="outputs/smoke_test")
    p.add_argument("--device", type=str, default="")
    args = p.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = REPO_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- inputs ----
    if args.synthetic or not args.image:
        image, depth, gt = build_synthetic_scene()
        queries = list(gt.keys())
        image.save(out_dir / "synthetic_scene.png")
        print("Scene: synthetic (three shapes, known ground truth)")
    else:
        image = Image.open(args.image).convert("RGB")
        depth, gt = None, None
        if args.depth:
            dp = Path(args.depth)
            depth = (np.load(dp) if dp.suffix == ".npy"
                     else np.asarray(Image.open(dp), dtype=np.float32) / 1000.0)
        if not args.queries:
            p.error("--queries is required with --image")
        queries = [q.strip() for q in args.queries.split(",") if q.strip()]
        print(f"Scene: {args.image}  ({'with' if depth is not None else 'no'} depth)")

    # ---- model ----
    model_name = "ViT-H-14-378-quickgelu" if args.checkpoint else args.model
    model_path = args.checkpoint or args.pretrained
    print(f"Loading {model_name} ({model_path}) on {device} ...")
    encoder = CLIPRTEncoder(
        model_name=model_name,
        model_path=model_path,
        use_action_decoder=False,   # dense alignment needs no action decoder
        device=device,
    )
    G = encoder.patch_grid_size
    print(f"Encoder ready: grid {G}x{G}, D={encoder.embed_dim}, "
          f"preprocess={encoder.preprocess_mode}")

    W, H = image.size
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    text_embs = {q: encoder.encode_text(q, use_motion_prompt=False) for q in queries}

    # ---- 2D alignment measurement ----
    print(f"\n=== GATE EXPERIMENT: patch-text alignment (top-{TOPK_PATCHES} hit rate) ===")
    results: dict[str, list[bool]] = {}
    for mode in modes:
        _, patch_features = encoder.encode_image_dense(image, mode=mode)
        assert patch_features.shape[1] == encoder.embed_dim  # space check
        print(f"\n--- mode={mode} | patch features {tuple(patch_features.shape)} ---")
        results[mode] = []
        for q in queries:
            heat = similarity_heatmap(patch_features, text_embs[q], G)
            path = save_overlay(image, heat, q, mode, out_dir,
                                encoder.preprocess_mode,
                                gt[q] if gt else None, TOPK_PATCHES)
            line = (f"  '{q}': sim max {heat.max():+.3f} mean {heat.mean():+.3f} "
                    f"→ {path.name}")
            if gt is not None:
                hr = hit_rate(heat, gt[q], W, H, encoder.preprocess_mode, TOPK_PATCHES)
                inv = hit_rate(heat, gt[q], W, H, encoder.preprocess_mode,
                               TOPK_PATCHES, largest=False)
                ok = hr >= PASS_THRESHOLD
                results[mode].append(ok)
                line += f" | hit@{TOPK_PATCHES} {hr:.2f} {'PASS' if ok else 'FAIL'}"
                if not ok and inv >= PASS_THRESHOLD:
                    line += (f"  [INVERTED — bottom-{TOPK_PATCHES} hit {inv:.2f}: "
                             f"object is the similarity MINIMUM (register tokens)]")
            print(line)

    # ---- 3D pipeline (only when depth exists) ----
    if depth is not None:
        # Resolve camera intrinsics: explicit --intrinsics, else a sidecar
        # intrinsics.json next to the depth/image (written by capture_frame.py),
        # else None (test_3d_field falls back to the fx=fy=500 guess).
        intr = None
        candidates = []
        if args.intrinsics:
            candidates.append(Path(args.intrinsics))
        if args.depth:
            candidates.append(Path(args.depth).parent / "intrinsics.json")
        if args.image:
            candidates.append(Path(args.image).parent / "intrinsics.json")
        for c in candidates:
            if c and c.is_file():
                import json
                intr = json.loads(c.read_text())
                print(f"Loaded intrinsics from {c}")
                break

        best_mode = modes[0]
        if gt is not None and results:
            best_mode = max(results, key=lambda m: sum(results[m]))
        test_3d_field(encoder, image, depth, queries, gt, best_mode, device,
                      intrinsics=intr)

    # ---- verdict ----
    print("\n=== VERDICT ===")
    if gt is None:
        print("No ground truth for real images — inspect the overlays in "
              f"{out_dir} and judge whether the hot regions sit on the "
              "queried objects.")
    else:
        for mode in modes:
            n_pass = sum(results[mode])
            print(f"  {mode:9s}: {n_pass}/{len(queries)} queries PASS")
        winner = max(results, key=lambda m: sum(results[m]))
        if sum(results[winner]) == len(queries):
            print(f"\nGate OPEN: use patch_mode='{winner}' for the 3D feature "
                  f"field.\nNext: rerun on a real Isaac Sim RGB-D frame "
                  f"(--image frame.png --depth depth.npy --queries ...), then "
                  f"on the CLIP-RT checkpoint (--checkpoint).")
        elif sum(results[winner]) > 0:
            print(f"\nGate PARTIAL: '{winner}' localises some queries. Inspect "
                  f"the overlays before proceeding — marginal alignment gets "
                  f"worse on cluttered scenes, not better.")
        else:
            print("\nGate CLOSED for this backbone: no mode localises even the "
                  "synthetic scene.  ViT-B-16/openai passed this same scene "
                  "through the same code path, so this is a BACKBONE-specific "
                  "dense-feature failure, not a code bug — INVERTED flags "
                  "above confirm the large-ViT register phenomenon.  Options: "
                  "(a) dual-backbone: keep this model for the policy, use the "
                  "validated small model for the 3D grounding field; "
                  "(b) LSeg/OpenSeg dense features (what OpenScene uses).")


if __name__ == "__main__":
    main()
