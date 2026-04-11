"""Standalone sensor rigs, preset helpers, and demos for Genesis."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

from . import _compat as _sensor_backend
from . import synthetic as _synthetic

SENSOR_BACKEND = _sensor_backend.SENSOR_BACKEND
SensorScheduler = _sensor_backend.SensorScheduler
SensorSuite = _sensor_backend.SensorSuite
UPSTREAM_SENSORS_AVAILABLE = _sensor_backend.UPSTREAM_SENSORS_AVAILABLE
get_preset = _sensor_backend.get_preset
has_upstream_sensors = _sensor_backend.has_upstream_sensors
list_presets = _sensor_backend.list_presets
require_upstream_sensors = _sensor_backend.require_upstream_sensors
upstream_sensors_error = _sensor_backend.upstream_sensors_error

GNSS_ORIGIN_LLH = _synthetic.GNSS_ORIGIN_LLH
ScenarioPhase = _synthetic.ScenarioPhase
SyntheticRolloutSummary = _synthetic.SyntheticRolloutSummary
SyntheticScenarioConfig = _synthetic.SyntheticScenarioConfig
get_scenario_phase = _synthetic.get_scenario_phase
list_scenario_windows = _synthetic.list_scenario_windows
make_synthetic_rollout = _synthetic.make_synthetic_rollout
make_synthetic_sensor_state = _synthetic.make_synthetic_sensor_state
summarize_synthetic_rollout = _synthetic.summarize_synthetic_rollout

_COMPAT_EXPORTS = set(getattr(_sensor_backend, "__all__", []))

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DemoCatalogSummary": (".scenes", "DemoCatalogSummary"),
    "DemoScene": (".scenes", "DemoScene"),
    "DemoSceneSpec": (".scenes", "DemoSceneSpec"),
    "HeadlessScene": (".scenes", "HeadlessScene"),
    "NamedContactSensor": (".rigs", "NamedContactSensor"),
    "ObservationMetadata": (".robustness", "ObservationMetadata"),
    "ObservationStatus": (".robustness", "ObservationStatus"),
    "RigProfile": (".rigs", "RigProfile"),
    "SceneRuntimeMode": (".scenes", "SceneRuntimeMode"),
    "RobustSensorWrapper": (".robustness", "RobustSensorWrapper"),
    "SensorFaultConfig": (".robustness", "SensorFaultConfig"),
    "SensorRig": (".rigs", "SensorRig"),
    "SensorRigSummary": (".rigs", "SensorRigSummary"),
    "ScenarioPhase": (".synthetic", "ScenarioPhase"),
    "build_drone_demo": (".scenes", "build_drone_demo"),
    "build_franka_demo": (".scenes", "build_franka_demo"),
    "build_go2_demo": (".scenes", "build_go2_demo"),
    "build_perception_demo": (".scenes", "build_perception_demo"),
    "build_synthetic_demo": (".scenes", "build_synthetic_demo"),
    "describe_demo_catalog": (".scenes", "describe_demo_catalog"),
    "filter_demo_scenes": (".scenes", "filter_demo_scenes"),
    "get_demo_scene_spec": (".scenes", "get_demo_scene_spec"),
    "list_demo_scene_names": (".scenes", "list_demo_scene_names"),
    "list_demo_scenes": (".scenes", "list_demo_scenes"),
    "make_drone_navigation_rig": (".rigs", "make_drone_navigation_rig"),
    "make_drone_perception_rig": (".rigs", "make_drone_perception_rig"),
    "make_franka_wrist_rig": (".rigs", "make_franka_wrist_rig"),
    "make_go2_rig": (".rigs", "make_go2_rig"),
    "make_synthetic_multimodal_rig": (".rigs", "make_synthetic_multimodal_rig"),
    "wrap_rig_with_faults": (".robustness", "wrap_rig_with_faults"),
    "wrap_scheduler_with_faults": (".robustness", "wrap_scheduler_with_faults"),
    "wrap_suite_with_faults": (".robustness", "wrap_suite_with_faults"),
}


def __getattr__(name: str) -> Any:
    """Lazily expose both the selected sensor backend symbols and the high-level scene helpers."""
    if name in _COMPAT_EXPORTS and hasattr(_sensor_backend, name):
        value = getattr(_sensor_backend, name)
        globals()[name] = value
        return value

    if name not in _LAZY_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_name, attr_name = _LAZY_EXPORTS[name]
    module = import_module(module_name, __name__)
    value = getattr(module, attr_name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_EXPORTS) | _COMPAT_EXPORTS)


__all__ = sorted(
    {
        "__version__",
        "DemoCatalogSummary",
        "DemoScene",
        "DemoSceneSpec",
        "GNSS_ORIGIN_LLH",
        "HeadlessScene",
        "NamedContactSensor",
        "ObservationMetadata",
        "ObservationStatus",
        "RigProfile",
        "RobustSensorWrapper",
        "SceneRuntimeMode",
        "SENSOR_BACKEND",
        "ScenarioPhase",
        "SensorFaultConfig",
        "SyntheticRolloutSummary",
        "SyntheticScenarioConfig",
        "SensorRig",
        "SensorRigSummary",
        "SensorScheduler",
        "SensorSuite",
        "UPSTREAM_SENSORS_AVAILABLE",
        "build_drone_demo",
        "build_franka_demo",
        "build_go2_demo",
        "build_perception_demo",
        "build_synthetic_demo",
        "describe_demo_catalog",
        "filter_demo_scenes",
        "get_demo_scene_spec",
        "get_scenario_phase",
        "has_upstream_sensors",
        "list_demo_scene_names",
        "list_demo_scenes",
        "list_scenario_windows",
        "make_drone_navigation_rig",
        "make_drone_perception_rig",
        "make_franka_wrist_rig",
        "make_go2_rig",
        "make_synthetic_multimodal_rig",
        "make_synthetic_rollout",
        "make_synthetic_sensor_state",
        "require_upstream_sensors",
        "summarize_synthetic_rollout",
        "wrap_rig_with_faults",
        "wrap_scheduler_with_faults",
        "wrap_suite_with_faults",
        "upstream_sensors_error",
    }
    | _COMPAT_EXPORTS
)
