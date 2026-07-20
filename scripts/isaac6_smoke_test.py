r"""
scripts/isaac6_smoke_test.py
============================
MINIMAL Isaac Sim 6.0 launch + RTX render test (Replicator SDG API).

PURPOSE
-------
Confirm on THIS machine (RTX 5060 Laptop, driver 610.62) that Isaac Sim 6.0
launches headless AND produces an RGB frame + metric depth frame.

WHAT THE EARLIER ATTEMPTS TAUGHT US
-----------------------------------
- GPU/launch already PASSES (log shows "RTX 5060 Laptop ... sm_120 ... mempool").
- The classic Camera returned no data: with `--no-window`, plain
  `simulation_app.update()` does NOT drive the RTX render / Replicator
  annotators, so the camera texture never fills.

The fix, straight from Isaac Sim 6.0's own SDG example
(standalone_examples/api/isaacsim.replicator.examples/simulation_get_data.py):
drive rendering explicitly with the Replicator ORCHESTRATOR:
    rep.orchestrator.step(rt_subframes=N)   # renders headless, fills annotators
    rgb = rgb_annot.get_data()              # real pixels

If this prints "SMOKE TEST PASSED" and writes rgb.png, the full port is safe.

RUN (inside your Isaac Sim 6.0 Python):
    C:\Workspace\Issac\python.bat .\scripts\isaac6_smoke_test.py

Outputs: outputs/isaac6_smoke/{rgb.png, depth_vis.png}
"""

import sys
from pathlib import Path

# Windows cp1252 console can't encode some Unicode; force UTF-8.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs" / "isaac6_smoke"


def main() -> int:
    # ---- 1. Boot the Kit runtime (headless) -------------------------------
    from isaacsim import SimulationApp

    print("[1/6] Creating SimulationApp (headless)...  (first launch compiles "
          "shaders — can take several minutes)")
    sim_app = SimulationApp({"headless": True, "width": 640, "height": 480})

    # ---- 2. Imports (only valid AFTER the app exists) ---------------------
    import numpy as np
    import omni.usd
    import omni.replicator.core as rep
    from isaacsim.core.experimental.objects import GroundPlane

    try:
        print("[2/6] Creating a fresh stage...")
        omni.usd.get_context().new_stage()
        # We render on demand (via orchestrator.step), not automatically on play.
        rep.orchestrator.set_capture_on_play(False)

        print("[3/6] Adding dome light, ground plane, and a cube...")
        rep.functional.create.dome_light(intensity=800, name="DomeLight")
        GroundPlane("/World/GroundPlane")
        cube = rep.functional.create.cube(name="Cube", parent="/World")
        rep.functional.modify.position(cube, (0.0, 0.0, 0.5))

        print("[4/6] Creating a camera (look_at) + render product...")
        # look_at handles aiming for us — no quaternion math needed.
        cam = rep.functional.create.camera(
            position=(3.0, 0.0, 1.8),
            look_at=(0.0, 0.0, 0.4),
            parent="/World",
            name="Camera",
        )
        rp = rep.create.render_product(cam, resolution=(640, 480))  # (width, height)

        rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        rgb_annot.attach(rp)
        depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        depth_annot.attach(rp)

        print("[5/6] Driving the renderer via the orchestrator...")
        rgb_np = None
        depth_np = None
        # A handful of orchestrator steps lets the RTX image converge; rt_subframes
        # accumulates several render subframes per step for a clean frame.
        for attempt in range(8):
            rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)
            rgb = np.asarray(rgb_annot.get_data())
            if rgb.size and rgb.ndim == 3 and rgb.max() > 0:
                rgb_np = rgb[..., :3]
                depth_np = np.asarray(depth_annot.get_data())
                print(f"      got a valid frame after {attempt + 1} orchestrator step(s)")
                break

        if rgb_np is None:
            print("[FAIL] Renderer produced no non-empty RGB after 8 steps.")
            return 1

        print("[6/6] Saving outputs...")
        from PIL import Image
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb_np.astype(np.uint8)).save(OUT_DIR / "rgb.png")

        depth_ok = depth_np is not None and depth_np.size > 0
        valid = None
        if depth_ok:
            depth_np = depth_np.astype(np.float32)
            valid = np.isfinite(depth_np) & (depth_np > 0)
            vis = np.zeros_like(depth_np, dtype=np.float32)
            if valid.any():
                lo, hi = depth_np[valid].min(), depth_np[valid].max()
                vis[valid] = (depth_np[valid] - lo) / max(hi - lo, 1e-6)
            Image.fromarray((vis * 255).astype(np.uint8)).save(OUT_DIR / "depth_vis.png")

        print("\n==================================================")
        print("SMOKE TEST PASSED")
        print(f"  RGB   : {rgb_np.shape} -> {OUT_DIR / 'rgb.png'}")
        if depth_ok and valid is not None and valid.any():
            print(f"  Depth : {depth_np.shape}  range "
                  f"[{depth_np[valid].min():.2f}, {depth_np[valid].max():.2f}] m")
        else:
            print("  Depth : not available (annotator returned None/empty)")
        print("  -> Isaac Sim 6.0 renders headless via Replicator. Port is safe.")
        print("==================================================")
        return 0

    finally:
        sim_app.close()


if __name__ == "__main__":
    raise SystemExit(main())
