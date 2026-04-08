"""Small, self-contained demo scene builders for Genesis Sensors."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import genesis as gs

from .rigs import SensorRig, make_drone_navigation_rig, make_franka_wrist_rig, make_go2_rig

_HOVER_RPM = 14_468.43


@dataclass
class DemoScene:
    """Container bundling a Genesis scene, its main entity, and the attached sensor rig."""

    name: str
    scene: Any
    entity: Any
    rig: SensorRig
    controller: Callable[[int], None] | None = None


def _init_genesis(*, use_gpu: bool, logging_level: str = "warning") -> None:
    """Initialize Genesis with a CPU/GPU backend selected at runtime."""
    backend = getattr(gs, "gpu" if use_gpu else "cpu", None)
    gs.init(backend=backend, logging_level=logging_level)


def build_drone_demo(*, dt: float = 0.01, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0) -> DemoScene:
    """Build a small drone scene plus a navigation sensor rig."""
    _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt, substeps=2),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.5, 0.0, 2.5), camera_lookat=(0.0, 0.0, 0.8), camera_fov=35),
        rigid_options=gs.options.RigidOptions(dt=dt, enable_collision=True, enable_joint_limit=True),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.5)))
    scene.build()

    def _controller(step: int) -> None:
        t = step * dt
        tilt_x = 0.012 * np.sin(0.4 * t)
        tilt_y = 0.012 * np.cos(0.4 * t)
        rpms = np.array(
            [
                1.0 + tilt_x - tilt_y,
                1.0 + tilt_x + tilt_y,
                1.0 - tilt_x - tilt_y,
                1.0 - tilt_x + tilt_y,
            ],
            dtype=np.float32,
        ) * _HOVER_RPM
        drone.set_propellels_rpm(rpms)

    return DemoScene(
        name="drone",
        scene=scene,
        entity=drone,
        rig=make_drone_navigation_rig(drone, dt=dt, seed=seed),
        controller=_controller,
    )


def build_franka_demo(*, dt: float = 0.01, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0) -> DemoScene:
    """Build a Franka arm scene plus a wrist/proprioception sensor rig."""
    _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        viewer_options=gs.options.ViewerOptions(camera_pos=(3.2, 0.0, 2.0), camera_lookat=(0.0, 0.0, 0.5), camera_fov=40),
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

    arm_dofs = list(range(7))
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
    _init_genesis(use_gpu=use_gpu)

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=dt),
        viewer_options=gs.options.ViewerOptions(camera_pos=(2.8, 0.0, 1.5), camera_lookat=(0.0, 0.0, 0.35), camera_fov=45),
        vis_options=gs.options.VisOptions(show_world_frame=True),
        profiling_options=gs.options.ProfilingOptions(show_FPS=False),
        show_viewer=show_viewer,
    )
    scene.add_entity(gs.morphs.Plane())
    go2 = scene.add_entity(gs.morphs.URDF(file="urdf/go2/urdf/go2.urdf", pos=(0.0, 0.0, 0.35)))
    scene.build()

    stand = np.array([0.0, 0.7, -1.4] * 4, dtype=np.float32)

    def _controller(step: int) -> None:
        try:
            go2.control_dofs_position(stand, list(range(12)))
        except Exception:
            pass

    return DemoScene(name="go2", scene=scene, entity=go2, rig=make_go2_rig(go2, dt=dt, seed=seed), controller=_controller)
