"""
env/isaac_sim.py
================
NVIDIA Isaac Sim observation + action interface — ported to the Isaac Sim
6.x ``isaacsim.*`` API (verified against Isaac Sim 6.0.1-rc).

WHY THIS FILE LOOKS DIFFERENT FROM THE README's PSEUDO-CODE
----------------------------------------------------------
Isaac Sim 5.0 renamed the whole Python API from ``omni.isaac.*`` to
``isaacsim.*`` and removed the high-level ``World``/``Camera``/``Franka``
convenience classes.  Isaac Sim 4.2 (the last ``omni.isaac.*`` release) does not
support Blackwell / RTX 50-series GPUs, so on an RTX 5060 we MUST use 6.x and
therefore the new API.  The key differences this file is built around, each
verified on-device or in the bundled standalone examples:

  * App:        ``from isaacsim import SimulationApp`` (must be created before
                any other isaacsim/omni import).
  * Stage:      ``isaacsim.core.experimental.utils.stage`` (create_new_stage,
                open_stage, add_reference_to_stage).
  * Objects:    ``isaacsim.core.experimental.objects`` (GroundPlane, DomeLight,
                Cube) + ``.prims`` (GeomPrim, RigidPrim, Articulation) +
                ``.materials`` (OmniPbrMaterial).
  * Robot:      Franka loaded by referencing the asset USD and wrapping it in
                ``Articulation`` (see control_frankas.py).
  * Rendering:  There is NO ``world.step(render=True)``.  Headless rendering is
                driven by the REPLICATOR ORCHESTRATOR: create a render product
                over a camera, attach "rgb" / "distance_to_image_plane"
                annotators, then ``rep.orchestrator.step()`` renders and
                ``annot.get_data()`` returns the pixels.  Plain
                ``simulation_app.update()` does NOT fill camera buffers headless.
  * Physics:    ``simulation_app.update()`` advances physics once the timeline
                is playing (``isaacsim.core.experimental.utils.app.play()``).

WHAT IS AND ISN'T IMPLEMENTED HERE
----------------------------------
IMPLEMENTED (proven): headless launch, procedural tabletop scene, RGB-D capture
via the Replicator annotator path, physics settling, reset/close.  This is
everything capture_frame.py needs for the CLIP dense-feature gate test.

DEFERRED: closed-loop Cartesian EEF control (inverse kinematics) and gripper
actuation for inference.py.  ``_apply_eef_delta`` accepts a zero/no-op action
(what capture + settling send) but raises NotImplementedError for a real motion
command — the 6.x IK path (``isaacsim.robot_motion.motion_generation`` /
Lula / RMPFlow) is unverified and is the next task, not a capture blocker.

Usage
-----
    env = IsaacSimEnv(scene_usd="assets/tabletop.usd")   # missing file -> default scene
    obs = env.reset()          # settles physics, returns first RGB-D observation
    # obs.rgb   : HxWx3 uint8
    # obs.depth : HxW float32 metres
    env.close()
"""

from __future__ import annotations

import os
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from env.action_space import ActionSpace, ACTION_DIM, GRIPPER_OPEN


# ---------------------------------------------------------------------------
# Observation type
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """A single RGB-D observation from the Isaac Sim camera.

    rgb   : np.ndarray [H, W, 3] uint8   — colour image (CLIP input).
    depth : np.ndarray [H, W]    float32 — metric depth in metres (scene_3d input).
    state : np.ndarray [8]       float32 — proprioception (unused by the CLIP-RT
            policy; currently zeros — see _get_observation).
    """
    rgb:   np.ndarray
    depth: np.ndarray
    state: np.ndarray = field(default_factory=lambda: np.zeros(8, dtype=np.float32))


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class IsaacSimEnv:
    """Interface between the CLIP-RT policy and NVIDIA Isaac Sim 6.x.

    Parameters mirror the original design; ``settle_steps`` is new and controls
    how many physics ticks reset() runs (without rendering) before it grabs the
    first frame.
    """

    # Isaac's on-disk asset path for the Franka Panda (verified in
    # control_frankas.py).  Loaded relative to get_assets_root_path().
    _FRANKA_ASSET_SUBPATH = "/Isaac/Robots/FrankaRobotics/FrankaPanda/franka.usd"

    def __init__(
        self,
        scene_usd: str = "assets/tabletop.usd",
        robot_prim_path: str = "/World/franka",
        camera_prim_path: str = "/World/capture_camera",
        resolution: tuple[int, int] = (640, 480),
        physics_dt: float = 1.0 / 60.0,
        rendering_dt: float = 1.0 / 30.0,
        headless: bool = True,
        settle_steps: int = 30,
        action_space: Optional[ActionSpace] = None,
    ):
        self.scene_usd = scene_usd
        self.robot_prim_path = robot_prim_path
        self.camera_prim_path = camera_prim_path
        self.resolution = resolution
        self.physics_dt = physics_dt
        self.rendering_dt = rendering_dt
        self.headless = headless
        self.settle_steps = settle_steps
        self.action_space = action_space or ActionSpace()

        # Set in _initialise_isaac()
        self._app = None
        self._rep = None            # omni.replicator.core module handle
        self._robot = None          # Articulation wrapper (or None if load failed)
        self._camera = None         # kept None: 6.x uses a render product, not a Camera obj
        self._render_product = None
        self._rgb_annot = None
        self._depth_annot = None
        self._is_initialised = False

    # ------------------------------------------------------------------
    # Initialisation (deferred to first reset() call)
    # ------------------------------------------------------------------

    def _initialise_isaac(self):
        """Boot the Kit runtime, build/open the scene, wire up the camera.

        Deferred so importing this module (e.g. in train.py) does not require
        Isaac Sim.  All isaacsim/omni imports happen AFTER SimulationApp().
        """
        from isaacsim import SimulationApp

        w, h = self.resolution
        self._app = SimulationApp({"headless": self.headless, "width": w, "height": h})

        # --- imports valid only after the app exists ---
        import omni.replicator.core as rep
        import isaacsim.core.experimental.utils.stage as stage_utils
        import isaacsim.core.experimental.utils.app as app_utils
        from isaacsim.core.experimental.prims import Articulation

        self._rep = rep
        self._app_utils = app_utils

        # Render on demand (via orchestrator.step), not automatically each frame.
        rep.orchestrator.set_capture_on_play(False)

        # Build or open the scene.  In both branches we end with self._robot set
        # (or None) and self._camera_handle pointing at a camera prim/path.
        if self.scene_usd and os.path.isfile(self.scene_usd):
            stage_utils.open_stage(self.scene_usd)
            try:
                self._robot = Articulation(self.robot_prim_path)
            except Exception as e:  # noqa: BLE001
                print(f"[IsaacSimEnv] Could not wrap robot at {self.robot_prim_path}: {e}")
                self._robot = None
            self._camera_handle = self.camera_prim_path
        else:
            print(
                f"[IsaacSimEnv] Scene USD '{self.scene_usd}' not found — "
                "building the default in-code tabletop scene."
            )
            self._build_default_scene(stage_utils, Articulation)

        # --- camera render product + annotators (the PROVEN capture path) ---
        self._render_product = rep.create.render_product(
            self._camera_handle, resolution=(w, h)
        )
        self._rgb_annot = rep.AnnotatorRegistry.get_annotator("rgb")
        self._rgb_annot.attach(self._render_product)
        self._depth_annot = rep.AnnotatorRegistry.get_annotator("distance_to_image_plane")
        self._depth_annot.attach(self._render_product)

        # Start the timeline so simulation_app.update() advances physics.
        app_utils.play()
        self._app.update()
        self._is_initialised = True

    def _build_default_scene(self, stage_utils, Articulation):
        """Construct a minimal tabletop scene procedurally (no USD required).

        Ground + dome light + a static table + three colour-coded cubes (red /
        green / blue) for language grounding + a Franka Panda (best-effort) + a
        fixed camera framing the table.  Every call here uses a pattern verified
        in the bundled examples (camera.py for Cube/material, control_frankas.py
        for the robot, simulation_get_data.py for the camera+render product).

        Poses are a sensible starting point — after your first capture, eyeball
        outputs/capture/{frame,depth_vis}.png and nudge the cube positions or
        the camera position/look_at below if the framing is off.
        """
        rep = self._rep
        from isaacsim.core.experimental.objects import GroundPlane, DomeLight, Cube
        from isaacsim.core.experimental.prims import GeomPrim, RigidPrim
        from isaacsim.core.experimental.materials import OmniPbrMaterial
        from isaacsim.storage.native import get_assets_root_path

        stage_utils.create_new_stage()

        # Lighting + floor.
        GroundPlane("/World/GroundPlane")
        dome = DomeLight("/World/DomeLight")
        dome.set_intensities(800.0)

        # Static table: a Cube with collision but NO rigid body (stays put).
        Cube(
            "/World/table",
            sizes=1.0,
            positions=np.array([0.55, 0.0, 0.20]),
            scales=np.array([0.70, 1.20, 0.40]),
        )
        GeomPrim("/World/table", apply_collision_apis=True)
        table_mat = OmniPbrMaterial("/World/Materials/table")
        table_mat.set_input_values("diffuse_color_constant", [0.62, 0.42, 0.24])
        GeomPrim("/World/table").apply_visual_materials(table_mat)

        # Colour-coded cubes as rigid bodies resting on the table top (z≈0.45).
        cubes = [
            ("red_cube",   [0.50, -0.18, 0.48], [0.90, 0.10, 0.10]),
            ("green_cube", [0.55,  0.00, 0.48], [0.10, 0.70, 0.20]),
            ("blue_cube",  [0.50,  0.18, 0.48], [0.15, 0.25, 0.85]),
        ]
        for name, pos, col in cubes:
            path = f"/World/{name}"
            Cube(path, sizes=1.0, positions=np.array(pos), scales=np.array([0.05, 0.05, 0.05]))
            GeomPrim(path, apply_collision_apis=True)
            rb = RigidPrim(path)
            mat = OmniPbrMaterial(f"/World/Materials/{name}")
            mat.set_input_values("diffuse_color_constant", col)
            rb.apply_visual_materials(mat)

        # Franka Panda — best effort.  Loading references a USD from the Isaac
        # asset server (can be slow on first run, or unreachable offline); the
        # capture works without it, so failure is non-fatal.
        self._robot = None
        assets_root = get_assets_root_path()
        if assets_root is not None:
            try:
                stage_utils.add_reference_to_stage(
                    usd_path=assets_root + self._FRANKA_ASSET_SUBPATH,
                    path=self.robot_prim_path,
                    variants=[("Gripper", "AlternateFinger"), ("Mesh", "Quality")],
                )
                self._robot = Articulation(self.robot_prim_path)
            except Exception as e:  # noqa: BLE001
                print(f"[IsaacSimEnv] Franka load skipped ({e}). Scene has no robot.")
        else:
            print("[IsaacSimEnv] No asset root (offline?) — scene has no robot.")

        # Fixed camera framing the table.  look_at removes the need for any
        # quaternion math (verified in simulation_get_data.py).
        self._camera_handle = rep.functional.create.camera(
            position=(1.6, 0.0, 1.05),
            look_at=(0.55, 0.0, 0.42),
            parent="/World",
            name=self.camera_prim_path.rsplit("/", 1)[-1],
        )

    # ------------------------------------------------------------------
    # Reset / step
    # ------------------------------------------------------------------

    def reset(self) -> Observation:
        """Reset to the initial state and return the first RGB-D observation."""
        if not self._is_initialised:
            self._initialise_isaac()

        if self._robot is not None:
            try:
                self._robot.reset_to_default_state()
            except Exception:  # noqa: BLE001
                pass

        # Let physics settle WITHOUT rendering (renders are expensive; we only
        # render once, in the _get_observation() below).
        for _ in range(self.settle_steps):
            self._app.update()

        return self._get_observation()

    def step(self, action: np.ndarray) -> tuple[Observation, bool]:
        """Execute one action, advance physics, return (observation, done).

        For a no-op action (what capture / settling send) this just advances
        physics and re-renders.  Real Cartesian motion is deferred — see
        _apply_eef_delta.
        """
        assert self._is_initialised, "Call reset() before step()."
        assert len(action) == ACTION_DIM, f"Expected {ACTION_DIM}-dim action."

        action = self.action_space.clip(action)
        self._apply_eef_delta(action[:6])   # no-op for zero delta; deferred otherwise
        self._apply_gripper(action[6])      # deferred (no-op) for now

        self._app.update()                  # advance physics one tick
        obs = self._get_observation()       # renders via the orchestrator
        return obs, self._check_success()

    # ------------------------------------------------------------------
    # Observation retrieval
    # ------------------------------------------------------------------

    def _get_observation(self) -> Observation:
        """Render one frame via the Replicator orchestrator and read RGB-D."""
        # This is the step that ACTUALLY renders headless (rt_subframes lets the
        # RTX image accumulate for a clean frame).  Without it the annotators
        # return empty buffers.
        self._rep.orchestrator.step(rt_subframes=8, delta_time=0.0, pause_timeline=False)

        rgb = np.asarray(self._rgb_annot.get_data())
        if rgb.ndim == 3 and rgb.shape[-1] == 4:      # RGBA -> RGB
            rgb = rgb[..., :3]
        rgb = rgb.astype(np.uint8)

        depth = np.asarray(self._depth_annot.get_data(), dtype=np.float32)

        # Proprioception is unused by the CLIP-RT policy; return zeros for now.
        # (Best-effort joint read is left for the inference/control work, where
        # the Articulation DOF layout matters.)
        state = np.zeros(8, dtype=np.float32)

        return Observation(rgb=rgb, depth=depth, state=state)

    # ------------------------------------------------------------------
    # Low-level action application  (control path — DEFERRED)
    # ------------------------------------------------------------------

    def _apply_eef_delta(self, eef_delta: np.ndarray):
        """Apply a Cartesian EEF delta.  DEFERRED for non-zero deltas.

        A zero/near-zero delta means "hold position" and is a no-op (this is all
        capture_frame + the settle loop ever send).  A real motion command
        raises NotImplementedError: the Isaac Sim 6.x inverse-kinematics path
        (isaacsim.robot_motion.motion_generation — Lula/RMPFlow) has not been
        wired up yet and is the next task (needed only by inference.py).
        """
        if float(np.linalg.norm(eef_delta[:6])) < 1e-6:
            return
        raise NotImplementedError(
            "Cartesian EEF control is not yet ported to the Isaac Sim 6.x API. "
            "capture_frame.py only sends no-op actions, so this is not needed "
            "for RGB-D capture. Implement IK here via "
            "isaacsim.robot_motion.motion_generation (Lula/RMPFlow) for inference."
        )

    def _apply_gripper(self, gripper_cmd: float):
        """Set the gripper open/closed.  DEFERRED (no-op) for now.

        Wired alongside the EEF IK control for inference; capture does not need
        gripper motion, so this intentionally does nothing yet.
        """
        return

    def _check_success(self) -> bool:
        """Task-success detector.  Override per task; default False."""
        return False

    # ------------------------------------------------------------------
    # Camera intrinsics
    # ------------------------------------------------------------------

    def get_intrinsics(self) -> dict:
        """Pinhole intrinsics (fx, fy, cx, cy) computed from the camera prim.

        USD cameras store focalLength / horizontalAperture / verticalAperture in
        millimetres (tenths of a scene unit); the pixel focal lengths follow from
        the pinhole model:
            fx = focalLength / horizontalAperture * width
            fy = focalLength / verticalAperture   * height
        and the principal point is the image centre.  This replaces the
        CameraIntrinsics fx=fy=500 guess so scene_3d.py backprojects depth into a
        metrically-correct point cloud.  Falls back to the defaults if the prim
        cannot be read.
        """
        w, h = self.resolution
        default = {"fx": 500.0, "fy": 500.0, "cx": w / 2.0, "cy": h / 2.0,
                   "width": w, "height": h, "source": "default"}
        try:
            import omni.usd
            from pxr import UsdGeom

            stage = omni.usd.get_context().get_stage()
            prim = stage.GetPrimAtPath(self.camera_prim_path)
            if not (prim and prim.IsValid() and prim.IsA(UsdGeom.Camera)):
                # Fall back to the first camera prim anywhere in the stage.
                prim = next((p for p in stage.Traverse() if p.IsA(UsdGeom.Camera)), None)
            if prim is None:
                return default

            cam = UsdGeom.Camera(prim)
            focal = float(cam.GetFocalLengthAttr().Get())
            h_ap = float(cam.GetHorizontalApertureAttr().Get())
            v_ap = float(cam.GetVerticalApertureAttr().Get())
            return {
                "fx": focal / h_ap * w,
                "fy": focal / v_ap * h,
                "cx": w / 2.0,
                "cy": h / 2.0,
                "width": w,
                "height": h,
                "source": str(prim.GetPath()),
            }
        except Exception as e:  # noqa: BLE001
            print(f"[IsaacSimEnv] intrinsics extraction failed ({e}); using defaults.")
            return default

    # ------------------------------------------------------------------
    # Shutdown
    # ------------------------------------------------------------------

    def close(self):
        """Shut down Isaac Sim cleanly."""
        if self._app is not None:
            self._app.close()


# ---------------------------------------------------------------------------
# Quaternion / axis-angle helpers
# (Pure NumPy — kept for the deferred IK/proprio work.  Same convention as
#  LIBERO's quat2axisangle and the action_to_language.json rotation entries.)
# ---------------------------------------------------------------------------

def _quat_to_axisangle(quat: np.ndarray) -> np.ndarray:
    """Convert quaternion (x, y, z, w) to axis-angle (ax, ay, az)."""
    import math
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(max(0.0, 1.0 - quat[3] ** 2))
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _axisangle_to_quat(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle to quaternion (x, y, z, w)."""
    angle = np.linalg.norm(aa)
    if angle < 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0])
    axis = aa / angle
    s = np.sin(angle / 2)
    c = np.cos(angle / 2)
    return np.array([axis[0] * s, axis[1] * s, axis[2] * s, c])
