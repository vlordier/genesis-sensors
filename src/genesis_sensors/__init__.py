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
get_scenario_phase = _synthetic.get_scenario_phase
make_synthetic_sensor_state = _synthetic.make_synthetic_sensor_state

_COMPAT_EXPORTS = set(getattr(_sensor_backend, "__all__", []))

_LAZY_EXPORTS: dict[str, tuple[str, str]] = {
    "DemoScene": (".scenes", "DemoScene"),
    "GenesisLiDAR": ("._runtime_sensors", "GenesisLiDAR"),
    "NamedContactSensor": (".rigs", "NamedContactSensor"),
    "RobustSensorWrapper": (".robustness", "RobustSensorWrapper"),
    "SensorFaultConfig": (".robustness", "SensorFaultConfig"),
    "SensorRig": (".rigs", "SensorRig"),
    "build_drone_demo": (".scenes", "build_drone_demo"),
    "build_franka_demo": (".scenes", "build_franka_demo"),
    "build_go2_demo": (".scenes", "build_go2_demo"),
    "build_perception_demo": (".scenes", "build_perception_demo"),
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
        "DemoScene",
        "GNSS_ORIGIN_LLH",
        "NamedContactSensor",
        "RobustSensorWrapper",
        "SENSOR_BACKEND",
        "SensorFaultConfig",
        "SensorRig",
        "SensorScheduler",
        "SensorSuite",
        "UPSTREAM_SENSORS_AVAILABLE",
        "build_drone_demo",
        "build_franka_demo",
        "build_go2_demo",
        "build_perception_demo",
        "get_scenario_phase",
        "has_upstream_sensors",
        "make_drone_navigation_rig",
        "make_drone_perception_rig",
        "make_franka_wrist_rig",
        "make_go2_rig",
        "make_synthetic_multimodal_rig",
        "make_synthetic_sensor_state",
        "require_upstream_sensors",
        "wrap_rig_with_faults",
        "wrap_scheduler_with_faults",
        "wrap_suite_with_faults",
        "upstream_sensors_error",
    }
    | _COMPAT_EXPORTS
)
