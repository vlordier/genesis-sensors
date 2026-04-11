"""Small, self-contained demo scene builders for Genesis Sensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

from .rigs import SensorRig, make_drone_navigation_rig, make_drone_perception_rig, make_franka_wrist_rig, make_go2_rig

_HOVER_RPM = 14_468.43
_SCENE_SUBSTEPS = 2
_DRONE_START_POS = (0.0, 0.0, 0.5)
_PERCEPTION_START_POS = (0.0, 0.0, 0.7)
_GO2_START_POS = (0.0, 0.0, 0.35)
_FRANKA_ARM_DOF_INDICES = tuple(range(7))
_GO2_DOF_INDICES = tuple(range(12))
_GO2_STAND_POSE = np.array([0.0, 0.7, -1.4] * 4, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ViewerCameraPreset:
    """Viewer camera settings reused by the demo scene builders."""

    camera_pos: tuple[float, float, float]
    camera_lookat: tuple[float, float, float]
    camera_fov: int


@dataclass(frozen=True, slots=True)
class DroneMotionProfile:
    """Oscillation gains used by the drone demo controllers."""

    tilt_x_amplitude: float
    tilt_x_frequency: float
    tilt_y_amplitude: float
    tilt_y_frequency: float


_DRONE_VIEW = ViewerCameraPreset((2.5, 0.0, 2.5), (0.0, 0.0, 0.8), 35)
_PERCEPTION_VIEW = ViewerCameraPreset((3.0, -1.2, 2.2), (0.0, 0.0, 0.8), 40)
_FRANKA_VIEW = ViewerCameraPreset((3.2, 0.0, 2.0), (0.0, 0.0, 0.5), 40)
_GO2_VIEW = ViewerCameraPreset((2.8, 0.0, 1.5), (0.0, 0.0, 0.35), 45)
_DRONE_MOTION = DroneMotionProfile(0.012, 0.4, 0.012, 0.4)
_PERCEPTION_MOTION = DroneMotionProfile(0.02, 0.55, 0.015, 0.35)


def _make_viewer_options(gs: Any, preset: ViewerCameraPreset) -> Any:
    """Build Genesis viewer options from a named camera preset."""
    return gs.options.ViewerOptions(
        camera_pos=preset.camera_pos,
        camera_lookat=preset.camera_lookat,
        camera_fov=preset.camera_fov,
    )


@dataclass
class DemoScene:
    """Container bundling a Genesis scene, its main entity, and the attached sensor rig.

    Examples
    --------
    >>> demo = build_drone_demo(show_viewer=False)
    >>> demo.rig.reset()
    >>> demo.scene.step()
    >>> _ = demo.rig.step(0.0)
    """

    name: str
    scene: Any
    entity: Any
    rig: SensorRig
    controller: Callable[[int], None] | None = None


def _get_genesis() -> Any:
    """Import Genesis lazily so module import remains lightweight without Torch."""
    try:
        return import_module("genesis")
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise ImportError(
            "Genesis demo scenes require a working Genesis + PyTorch runtime. Install torch in the target environment first."
        ) from exc


def _init_genesis(*, use_gpu: bool, logging_level: str = "warning") -> Any:
    """Initialize Genesis with a CPU/GPU backend selected at runtime."""
    gs = _get_genesis()
    backend = getattr(gs, "gpu" if use_gpu else "cpu", None)
    gs.init(backend=backend, logging_level=logging_level)
    return gs


def build_drone_demo(*, dt: float = 0.01, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0) -> DemoScene:
    """Build a small drone scene plus a navigation sensor rig."""
    gs = _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=_SCENE_SUBSTEPS),
        viewer_options=_make_viewer_options(gs, _DRONE_VIEW),
        rigid_options=gs.options.RigidOptions(dt=dt, enable_collision=True, enable_joint_limit=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=_DRONE_START_POS))
    scene.build()

    def _controller(step: int) -> None:
        t = step * dt
        tilt_x = _DRONE_MOTION.tilt_x_amplitude * np.sin(_DRONE_MOTION.tilt_x_frequency * t)
        tilt_y = _DRONE_MOTION.tilt_y_amplitude * np.cos(_DRONE_MOTION.tilt_y_frequency * t)
        rpms = (
            np.array(
                [
                    1.0 + tilt_x - tilt_y,
                    1.0 + tilt_x + tilt_y,
                    1.0 - tilt_x - tilt_y,
                    1.0 - tilt_x + tilt_y,
                ],
                dtype=np.float32,
            )
            * _HOVER_RPM
        )
        drone.set_propellels_rpm(rpms)

    return DemoScene(
        name="drone",
        scene=scene,
        entity=drone,
        rig=make_drone_navigation_rig(drone, dt=dt, seed=seed),
        controller=_controller,
    )


def build_perception_demo(
    *, dt: float = 0.01, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
) -> DemoScene:
    """Build a drone scene showcasing the richer multimodal perception stack."""
    gs = _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=_SCENE_SUBSTEPS),
        viewer_options=_make_viewer_options(gs, _PERCEPTION_VIEW),
        rigid_options=gs.options.RigidOptions(dt=dt, enable_collision=True, enable_joint_limit=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=_PERCEPTION_START_POS))
    scene.build()

    def _controller(step: int) -> None:
        t = step * dt
        tilt_x = _PERCEPTION_MOTION.tilt_x_amplitude * np.sin(_PERCEPTION_MOTION.tilt_x_frequency * t)
        tilt_y = _PERCEPTION_MOTION.tilt_y_amplitude * np.cos(_PERCEPTION_MOTION.tilt_y_frequency * t)
        rpms = (
            np.array(
                [
                    1.0 + tilt_x - tilt_y,
                    1.0 + tilt_x + tilt_y,
                    1.0 - tilt_x - tilt_y,
                    1.0 - tilt_x + tilt_y,
                ],
                dtype=np.float32,
            )
            * _HOVER_RPM
        )
        drone.set_propellels_rpm(rpms)

    return DemoScene(
        name="perception",
        scene=scene,
        entity=drone,
        rig=make_drone_perception_rig(drone, dt=dt, seed=seed),
        controller=_controller,
    )


def build_franka_demo(
    *, dt: float = 0.01, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
) -> DemoScene:
    """Build a Franka arm scene plus a wrist/proprioception sensor rig."""
    gs = _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        viewer_options=_make_viewer_options(gs, _FRANKA_VIEW),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.build()

    franka.set_dofs_kp(np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0]))
    franka.set_dofs_kv(np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0]))
    franka.set_dofs_force_range(
        np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0]),
        np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0]),
    )

    arm_dofs = list(_FRANKA_ARM_DOF_INDICES)
    home_q = np.array([0.0, 0.35, 0.0, -2.0, 0.0, 2.35, 0.7], dtype=np.float32)

    def _controller(step: int) -> None:
        phase = step * dt
        target_q = home_q + 0.18 * np.array(
            [
                np.sin(0.9 * phase),
                np.sin(0.7 * phase + 0.2),
                np.sin(0.8 * phase + 0.4),
                np.sin(0.6 * phase + 0.6),
                np.sin(1.0 * phase + 0.8),
                np.sin(0.5 * phase + 1.0),
                np.sin(0.75 * phase + 1.2),
            ],
            dtype=np.float32,
        )
        franka.control_dofs_position(target_q, arm_dofs)

    return DemoScene(
        name="franka",
        scene=scene,
        entity=franka,
        rig=make_franka_wrist_rig(franka, dt=dt, seed=seed),
        controller=_controller,
    )


def build_go2_demo(*, dt: float = 0.01, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0) -> DemoScene:
    """Build a Go2 quadruped scene plus a proprioception/contact rig."""
    gs = _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        viewer_options=_make_viewer_options(gs, _GO2_VIEW),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    go2 = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=_GO2_START_POS))
    scene.build()

    stand = _GO2_STAND_POSE

    def _controller(step: int) -> None:
        try:
            go2.control_dofs_position(stand, list(_GO2_DOF_INDICES))
        except Exception:
            pass

    return DemoScene(
        name="go2", scene=scene, entity=go2, rig=make_go2_rig(go2, dt=dt, seed=seed), controller=_controller
    )


__all__ = [
    "DemoScene",
    "build_drone_demo",
    "build_perception_demo",
    "build_franka_demo",
    "build_go2_demo",
]
