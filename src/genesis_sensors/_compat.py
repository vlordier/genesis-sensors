"""Compatibility helpers that prefer upstream `genesis.sensors` and fall back to a bundled runtime."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

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

try:
    _backend = import_module("genesis.sensors")
    _config_module = import_module("genesis.sensors.config")
    _bridge_module = import_module("genesis.sensors.genesis_bridge")
    _presets_module = import_module("genesis.sensors.presets")
    SENSOR_BACKEND = "upstream"
    UPSTREAM_SENSORS_AVAILABLE = True
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
    if not UPSTREAM_SENSORS_AVAILABLE:
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
