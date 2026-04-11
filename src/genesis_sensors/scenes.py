"""Small, self-contained demo scene builders for Genesis Sensors."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import Any

import numpy as np

from .rigs import (
    RigProfile,
    SensorRig,
    make_drone_navigation_rig,
    make_drone_perception_rig,
    make_franka_wrist_rig,
    make_go2_rig,
    make_synthetic_multimodal_rig,
)

_HOVER_RPM = 14_468.43
_DEFAULT_DEMO_DT = 0.01
_DEFAULT_SYNTHETIC_DT = 0.05
_DEFAULT_RUNTIME_STEPS = 200
_SCENE_SUBSTEPS = 2
_DRONE_START_POS = (0.0, 0.0, 0.5)
_PERCEPTION_START_POS = (0.0, 0.0, 0.7)
_GO2_START_POS = (0.0, 0.0, 0.35)
_FRANKA_ARM_DOF_INDICES = tuple(range(7))
_GO2_DOF_INDICES = tuple(range(12))
_GO2_STAND_POSE = np.array([0.0, 0.7, -1.4] * 4, dtype=np.float32)
_FRANKA_JOINT_KP = np.array([4500.0, 4500.0, 3500.0, 3500.0, 2000.0, 2000.0, 2000.0, 100.0, 100.0])
_FRANKA_JOINT_KV = np.array([450.0, 450.0, 350.0, 350.0, 200.0, 200.0, 200.0, 10.0, 10.0])
_FRANKA_FORCE_RANGE_LOW = np.array([-87.0, -87.0, -87.0, -87.0, -12.0, -12.0, -12.0, -100.0, -100.0])
_FRANKA_FORCE_RANGE_HIGH = np.array([87.0, 87.0, 87.0, 87.0, 12.0, 12.0, 12.0, 100.0, 100.0])
_FRANKA_HOME_Q = np.array([0.0, 0.35, 0.0, -2.0, 0.0, 2.35, 0.7], dtype=np.float32)


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

ObservationCallback = Callable[[int, dict[str, Any]], None]


class SceneRuntimeMode(str, Enum):
    """Runtime category for a built-in demo scene."""

    RUNTIME = "runtime"
    HEADLESS = "headless"

    @classmethod
    def values(cls) -> tuple[str, ...]:
        """Return the serialized runtime-mode values in declaration order."""
        return tuple(mode.value for mode in cls)

    @classmethod
    def from_requires_runtime(cls, requires_runtime: bool) -> "SceneRuntimeMode":
        """Map a legacy boolean runtime flag to the matching runtime mode."""
        return cls.RUNTIME if requires_runtime else cls.HEADLESS


@dataclass(frozen=True, slots=True)
class SceneCatalogFilter:
    """Typed filter bundle used by the demo catalog helpers and CLI."""

    profile: RigProfile | None = None
    runtime_mode: SceneRuntimeMode | None = None
    query: str | None = None

    @property
    def normalized_query(self) -> str | None:
        """Return the stripped case-folded query, or ``None`` if it is empty."""
        if self.query is None:
            return None
        query = self.query.strip().casefold()
        return query or None


@dataclass(frozen=True, slots=True)
class DemoSceneSpec:
    """Metadata describing the built-in demo scenes exposed by the package."""

    name: str
    description: str
    profile: RigProfile
    requires_runtime: bool
    default_dt: float

    @property
    def runtime_mode(self) -> SceneRuntimeMode:
        """Return whether the scene needs the Genesis runtime or can run headlessly."""
        return SceneRuntimeMode.from_requires_runtime(self.requires_runtime)

    @property
    def is_headless(self) -> bool:
        """Return ``True`` when the scene can run without Genesis."""
        return not self.requires_runtime

    def matches(
        self,
        *,
        profile: RigProfile | None = None,
        requires_runtime: bool | None = None,
        runtime_mode: SceneRuntimeMode | None = None,
        query: str | None = None,
    ) -> bool:
        """Check whether the scene metadata matches a catalog filter."""
        if profile is not None and self.profile != profile:
            return False
        if requires_runtime is not None and self.requires_runtime != requires_runtime:
            return False
        if runtime_mode is not None and self.runtime_mode != runtime_mode:
            return False
        if query is None:
            return True
        normalized_query = query.strip().casefold()
        if not normalized_query:
            return True
        haystack = f"{self.name} {self.description} {self.profile.value}".casefold()
        return normalized_query in haystack

    def recommended_command(self) -> str:
        """Return a small copy-pasteable CLI command for this scene."""
        if self.runtime_mode == SceneRuntimeMode.HEADLESS:
            return f"genesis-sensors {self.name} --dry-run --summary-format json"
        return f"genesis-sensors {self.name} --steps {_DEFAULT_RUNTIME_STEPS}"

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation of the scene metadata."""
        return {
            "name": self.name,
            "description": self.description,
            "profile": self.profile.value,
            "requires_runtime": self.requires_runtime,
            "runtime_mode": self.runtime_mode.value,
            "is_headless": self.is_headless,
            "default_dt": self.default_dt,
            "recommended_command": self.recommended_command(),
        }


@dataclass(frozen=True, slots=True)
class DemoCatalogSummary:
    """Compact summary of the built-in demo catalog for automation and docs."""

    scene_count: int
    profile_counts: dict[str, int]
    runtime_counts: dict[str, int]
    scene_names: tuple[str, ...]
    headless_scenes: tuple[str, ...]
    runtime_scenes: tuple[str, ...]
    recommended_commands: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly summary payload."""
        return {
            "scene_count": self.scene_count,
            "profile_counts": dict(self.profile_counts),
            "runtime_counts": dict(self.runtime_counts),
            "scene_names": list(self.scene_names),
            "headless_scenes": list(self.headless_scenes),
            "runtime_scenes": list(self.runtime_scenes),
            "recommended_commands": list(self.recommended_commands),
        }

    def headless_scene_names(self) -> tuple[str, ...]:
        """Return the names of headless scenes in the summarized catalog."""
        return self.headless_scenes

    def runtime_scene_names(self) -> tuple[str, ...]:
        """Return the names of runtime-backed scenes in the summarized catalog."""
        return self.runtime_scenes

    def preview_commands(self, limit: int = 3) -> tuple[str, ...]:
        """Return a small preview of recommended CLI commands from the catalog."""
        if limit <= 0:
            return ()
        return self.recommended_commands[:limit]


_DEMO_SCENE_SPECS: tuple[DemoSceneSpec, ...] = (
    DemoSceneSpec("drone", "Navigation-focused quadrotor demo", RigProfile.NAVIGATION, True, _DEFAULT_DEMO_DT),
    DemoSceneSpec("perception", "Multimodal drone perception stack", RigProfile.PERCEPTION, True, _DEFAULT_DEMO_DT),
    DemoSceneSpec("franka", "Manipulation and wrist-sensing demo", RigProfile.MANIPULATION, True, _DEFAULT_DEMO_DT),
    DemoSceneSpec("go2", "Quadruped proprioception and contact demo", RigProfile.QUADRUPED, True, _DEFAULT_DEMO_DT),
    DemoSceneSpec(
        "synthetic",
        "Headless multimodal smoke test without Genesis runtime",
        RigProfile.SYNTHETIC_MULTIMODAL,
        False,
        _DEFAULT_SYNTHETIC_DT,
    ),
)


def list_demo_scenes() -> tuple[DemoSceneSpec, ...]:
    """Return the built-in demo catalog for CLI help, docs, and automation."""
    return _DEMO_SCENE_SPECS


def _coerce_catalog_filter(
    *,
    profile: RigProfile | None = None,
    requires_runtime: bool | None = None,
    runtime_mode: SceneRuntimeMode | None = None,
    query: str | None = None,
    filters: SceneCatalogFilter | None = None,
) -> SceneCatalogFilter:
    """Merge explicit filter arguments with an optional filter dataclass."""
    if filters is not None:
        if profile is None:
            profile = filters.profile
        if runtime_mode is None:
            runtime_mode = filters.runtime_mode
        if query is None:
            query = filters.normalized_query
    if runtime_mode is None and requires_runtime is not None:
        runtime_mode = SceneRuntimeMode.from_requires_runtime(requires_runtime)
    return SceneCatalogFilter(profile=profile, runtime_mode=runtime_mode, query=query)


def list_demo_scene_names(
    *,
    profile: RigProfile | None = None,
    requires_runtime: bool | None = None,
    runtime_mode: SceneRuntimeMode | None = None,
    query: str | None = None,
    filters: SceneCatalogFilter | None = None,
) -> tuple[str, ...]:
    """Return only the scene names for a filtered view of the built-in catalog."""
    return tuple(
        spec.name
        for spec in filter_demo_scenes(
            profile=profile,
            requires_runtime=requires_runtime,
            runtime_mode=runtime_mode,
            query=query,
            filters=filters,
        )
    )


def list_demo_scene_commands(
    *,
    profile: RigProfile | None = None,
    requires_runtime: bool | None = None,
    runtime_mode: SceneRuntimeMode | None = None,
    query: str | None = None,
    filters: SceneCatalogFilter | None = None,
) -> tuple[str, ...]:
    """Return recommended CLI commands for a filtered view of the built-in catalog."""
    return tuple(
        spec.recommended_command()
        for spec in filter_demo_scenes(
            profile=profile,
            requires_runtime=requires_runtime,
            runtime_mode=runtime_mode,
            query=query,
            filters=filters,
        )
    )


def filter_demo_scenes(
    *,
    profile: RigProfile | None = None,
    requires_runtime: bool | None = None,
    runtime_mode: SceneRuntimeMode | None = None,
    query: str | None = None,
    filters: SceneCatalogFilter | None = None,
) -> tuple[DemoSceneSpec, ...]:
    """Filter the built-in demo catalog by rig profile, runtime mode, or text query."""
    catalog_filter = _coerce_catalog_filter(
        profile=profile,
        requires_runtime=requires_runtime,
        runtime_mode=runtime_mode,
        query=query,
        filters=filters,
    )
    return tuple(
        spec
        for spec in _DEMO_SCENE_SPECS
        if spec.matches(
            profile=catalog_filter.profile,
            runtime_mode=catalog_filter.runtime_mode,
            query=catalog_filter.normalized_query,
        )
    )


def describe_demo_catalog(
    *,
    profile: RigProfile | None = None,
    requires_runtime: bool | None = None,
    runtime_mode: SceneRuntimeMode | None = None,
    query: str | None = None,
    filters: SceneCatalogFilter | None = None,
) -> DemoCatalogSummary:
    """Summarize the built-in demo catalog for docs, CLIs, and automation."""
    specs = filter_demo_scenes(
        profile=profile,
        requires_runtime=requires_runtime,
        runtime_mode=runtime_mode,
        query=query,
        filters=filters,
    )
    scene_names = tuple(spec.name for spec in specs)
    headless_scenes = tuple(spec.name for spec in specs if spec.is_headless)
    runtime_scenes = tuple(spec.name for spec in specs if spec.requires_runtime)
    recommended_commands = tuple(spec.recommended_command() for spec in specs)
    profile_counts = dict(sorted(Counter(spec.profile.value for spec in specs).items()))
    runtime_counts = dict(sorted(Counter(spec.runtime_mode.value for spec in specs).items()))
    return DemoCatalogSummary(
        scene_count=len(specs),
        profile_counts=profile_counts,
        runtime_counts=runtime_counts,
        scene_names=scene_names,
        headless_scenes=headless_scenes,
        runtime_scenes=runtime_scenes,
        recommended_commands=recommended_commands,
    )


def get_demo_scene_spec(name: str) -> DemoSceneSpec:
    """Resolve a built-in demo scene by name."""
    for spec in _DEMO_SCENE_SPECS:
        if spec.name == name:
            return spec
    raise KeyError(f"unknown demo scene {name!r}")


def _make_viewer_options(gs: Any, preset: ViewerCameraPreset) -> Any:
    """Build Genesis viewer options from a named camera preset."""
    return gs.options.ViewerOptions(
        camera_pos=preset.camera_pos,
        camera_lookat=preset.camera_lookat,
        camera_fov=preset.camera_fov,
    )


def _hold_go2_stance(robot: Any, stand_pose: np.ndarray) -> None:
    """Keep the quadruped near its nominal stance when the runtime exposes joint control."""
    try:
        robot.control_dofs_position(stand_pose, list(_GO2_DOF_INDICES))
    except (AttributeError, RuntimeError, TypeError, ValueError):
        return


@dataclass(slots=True)
class HeadlessScene:
    """Minimal scene-like container for headless synthetic demos."""

    step_count: int = 0

    def reset(self) -> None:
        self.step_count = 0

    def step(self) -> None:
        self.step_count += 1


@dataclass
class DemoScene:
    """Container bundling a Genesis scene, its main entity, and the attached sensor rig.

    Examples
    --------
    >>> demo = build_drone_demo(show_viewer=False)
    >>> observations = demo.run(steps=1, dt=0.01)
    >>> len(observations) == 1
    True
    """

    name: str
    scene: Any
    entity: Any
    rig: SensorRig
    controller: Callable[[int], None] | None = None

    def reset(self) -> None:
        """Reset the scene if possible and always reset the sensor rig."""
        reset_scene = getattr(self.scene, "reset", None)
        if callable(reset_scene):
            reset_scene()
        self.rig.reset()

    def step_once(self, step: int, *, dt: float) -> dict[str, Any]:
        """Advance the controller, scene, and rig by one timestep."""
        if self.controller is not None:
            self.controller(step)
        step_scene = getattr(self.scene, "step", None)
        if callable(step_scene):
            step_scene()
        return self.rig.step(step * dt)

    def run(
        self,
        *,
        steps: int,
        dt: float,
        on_step: ObservationCallback | None = None,
    ) -> list[dict[str, Any]]:
        """Run the demo for a fixed number of steps and collect sensor observations."""
        self.reset()
        observations: list[dict[str, Any]] = []
        for step in range(steps):
            observation = self.step_once(step, dt=dt)
            observations.append(observation)
            if on_step is not None:
                on_step(step, observation)
        return observations

    def describe(self) -> dict[str, Any]:
        """Return a structured summary of the demo scene and its sensor rig."""
        try:
            scene_spec = get_demo_scene_spec(self.name).as_dict()
        except KeyError:
            scene_spec = {"name": self.name}

        return {
            "scene": self.name,
            "entity_present": self.entity is not None,
            "has_controller": self.controller is not None,
            "scene_spec": scene_spec,
            "rig": self.rig.describe().as_dict(),
        }


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


def build_drone_demo(
    *, dt: float = _DEFAULT_DEMO_DT, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
) -> DemoScene:
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
    *, dt: float = _DEFAULT_DEMO_DT, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
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
    *, dt: float = _DEFAULT_DEMO_DT, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
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

    franka.set_dofs_kp(_FRANKA_JOINT_KP)
    franka.set_dofs_kv(_FRANKA_JOINT_KV)
    franka.set_dofs_force_range(_FRANKA_FORCE_RANGE_LOW, _FRANKA_FORCE_RANGE_HIGH)

    arm_dofs = list(_FRANKA_ARM_DOF_INDICES)
    home_q = _FRANKA_HOME_Q

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


def build_go2_demo(
    *, dt: float = _DEFAULT_DEMO_DT, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
) -> DemoScene:
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
        _ = step
        _hold_go2_stance(go2, stand)

    return DemoScene(
        name="go2", scene=scene, entity=go2, rig=make_go2_rig(go2, dt=dt, seed=seed), controller=_controller
    )


def build_synthetic_demo(
    *, dt: float = _DEFAULT_SYNTHETIC_DT, show_viewer: bool = False, use_gpu: bool = False, seed: int = 0
) -> DemoScene:
    """Build a headless multimodal rig demo that does not require a Genesis runtime."""
    del show_viewer, use_gpu
    return DemoScene(
        name="synthetic",
        scene=HeadlessScene(),
        entity=None,
        rig=make_synthetic_multimodal_rig(dt=dt, seed=seed),
        controller=None,
    )


__all__ = [
    "DemoCatalogSummary",
    "DemoScene",
    "DemoSceneSpec",
    "HeadlessScene",
    "SceneCatalogFilter",
    "SceneRuntimeMode",
    "build_drone_demo",
    "build_perception_demo",
    "build_franka_demo",
    "build_go2_demo",
    "build_synthetic_demo",
    "describe_demo_catalog",
    "filter_demo_scenes",
    "get_demo_scene_spec",
    "list_demo_scene_commands",
    "list_demo_scene_names",
    "list_demo_scenes",
]
