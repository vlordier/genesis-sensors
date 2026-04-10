"""Compatibility helpers that prefer upstream `genesis.sensors` and fall back to a bundled runtime."""

from __future__ import annotations

import inspect
from importlib import import_module
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - for static analysis only
    from ._runtime_sensors import *  # noqa: F401,F403
    from ._runtime_sensors.genesis_bridge import (  # noqa: F401
        extract_joint_state,
        extract_link_contact_force_n,
        extract_link_ft_state,
        extract_link_imu_state,
        extract_rigid_body_state,
    )

_UPSTREAM_IMPORT_ERROR: Exception | None = None


def _upstream_missing_features(backend: Any, presets_module: Any) -> list[str]:
    """Return the companion-runtime features missing from an installed upstream backend."""
    missing: list[str] = []

    base_sensor = getattr(backend, "BaseSensor", None)
    if base_sensor is None or not hasattr(base_sensor, "configure_noise_model"):
        missing.append("BaseSensor.configure_noise_model")

    suite_cls = getattr(backend, "SensorSuite", None)
    required_suite_kwargs = {
        "uwb",
        "radar",
        "thermometer",
        "hygrometer",
        "light_sensor",
        "gas_sensor",
        "anemometer",
        "ultrasonic",
        "imaging_sonar",
        "side_scan",
        "dvl",
        "current_profiler",
    }
    if suite_cls is None:
        missing.append("SensorSuite")
    else:
        if not hasattr(suite_cls, "configure_noise_models"):
            missing.append("SensorSuite.configure_noise_models")
        try:
            suite_params = set(inspect.signature(suite_cls.__init__).parameters)
        except (TypeError, ValueError):
            suite_params = set()
        missing_kwargs = sorted(required_suite_kwargs - suite_params)
        if missing_kwargs:
            missing.append(f"SensorSuite kwargs ({', '.join(missing_kwargs)})")

    list_presets_fn = getattr(presets_module, "list_presets", None)
    if callable(list_presets_fn):
        for kind in (
            "thermometer",
            "anemometer",
            "uwb",
            "radar",
            "ultrasonic",
            "imaging_sonar",
            "side_scan_sonar",
            "dvl",
            "current_profiler",
        ):
            try:
                list_presets_fn(kind=kind)
            except Exception:
                missing.append(f"list_presets(kind={kind!r})")
                break
    else:
        missing.append("list_presets()")

    return missing


try:
    _candidate_backend = import_module("genesis.sensors")
    _candidate_config_module = import_module("genesis.sensors.config")
    _candidate_bridge_module = import_module("genesis.sensors.genesis_bridge")
    _candidate_presets_module = import_module("genesis.sensors.presets")
    UPSTREAM_SENSORS_AVAILABLE = True

    missing_features = _upstream_missing_features(_candidate_backend, _candidate_presets_module)
    if missing_features:
        _UPSTREAM_IMPORT_ERROR = RuntimeError(
            "Installed `genesis.sensors` is present but missing companion-surface features: "
            + ", ".join(missing_features)
        )
        _backend = import_module("._runtime_sensors", __package__)
        _config_module = import_module("._runtime_sensors.config", __package__)
        _bridge_module = import_module("._runtime_sensors.genesis_bridge", __package__)
        _presets_module = import_module("._runtime_sensors.presets", __package__)
        SENSOR_BACKEND = "bundled"
    else:
        _backend = _candidate_backend
        _config_module = _candidate_config_module
        _bridge_module = _candidate_bridge_module
        _presets_module = _candidate_presets_module
        SENSOR_BACKEND = "upstream"
except ImportError as exc:  # pragma: no cover - exercised when upstream sensors are unavailable
    _UPSTREAM_IMPORT_ERROR = exc
    _backend = import_module("._runtime_sensors", __package__)
    _config_module = import_module("._runtime_sensors.config", __package__)
    _bridge_module = import_module("._runtime_sensors.genesis_bridge", __package__)
    _presets_module = import_module("._runtime_sensors.presets", __package__)
    SENSOR_BACKEND = "bundled"
    UPSTREAM_SENSORS_AVAILABLE = False

_bundled_backend = import_module("._runtime_sensors", __package__)
_bundled_config_module = import_module("._runtime_sensors.config", __package__)
_bundled_bridge_module = import_module("._runtime_sensors.genesis_bridge", __package__)
_bundled_presets_module = import_module("._runtime_sensors.presets", __package__)

# Re-export the selected backend's public symbols at module scope, then fill in
# any symbols that only exist in the bundled extension layer.
for _module in (
    _backend,
    _config_module,
    _presets_module,
    _bundled_backend,
    _bundled_config_module,
    _bundled_presets_module,
):
    for _name in getattr(_module, "__all__", []):
        if hasattr(_module, _name) and _name not in globals():
            globals()[_name] = getattr(_module, _name)

for _name in (
    "extract_joint_state",
    "extract_link_contact_force_n",
    "extract_link_ft_state",
    "extract_link_imu_state",
    "extract_rigid_body_state",
):
    if hasattr(_bridge_module, _name):
        globals()[_name] = getattr(_bridge_module, _name)
    elif hasattr(_bundled_bridge_module, _name):
        globals()[_name] = getattr(_bundled_bridge_module, _name)


def has_upstream_sensors() -> bool:
    """Return whether the installed Genesis build exposes the native `genesis.sensors` package."""
    return UPSTREAM_SENSORS_AVAILABLE


def upstream_sensors_error() -> Exception | None:
    """Return the original upstream import error when the bundled backend is being used."""
    return _UPSTREAM_IMPORT_ERROR


def require_upstream_sensors(feature: str = "This feature") -> None:
    """Require the native upstream sensor package, even though a bundled fallback exists."""
    if SENSOR_BACKEND != "upstream":
        detail = (
            f" Original import error: {type(_UPSTREAM_IMPORT_ERROR).__name__}: {_UPSTREAM_IMPORT_ERROR}"
            if _UPSTREAM_IMPORT_ERROR
            else ""
        )
        raise ImportError(
            f"{feature} requires the upstream `genesis.sensors` module specifically; the current setup is using the bundled fallback backend.{detail}"
        ) from _UPSTREAM_IMPORT_ERROR


__all__ = sorted(
    set(getattr(_backend, "__all__", []))
    | set(getattr(_config_module, "__all__", []))
    | set(getattr(_presets_module, "__all__", []))
    | set(getattr(_bundled_backend, "__all__", []))
    | set(getattr(_bundled_config_module, "__all__", []))
    | set(getattr(_bundled_presets_module, "__all__", []))
    | {
        "SENSOR_BACKEND",
        "UPSTREAM_SENSORS_AVAILABLE",
        "extract_joint_state",
        "extract_link_contact_force_n",
        "extract_link_ft_state",
        "extract_link_imu_state",
        "extract_rigid_body_state",
        "has_upstream_sensors",
        "require_upstream_sensors",
        "upstream_sensors_error",
    }
)
