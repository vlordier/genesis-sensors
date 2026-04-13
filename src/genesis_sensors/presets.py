"""Preset re-exports for the active Genesis Sensors backend."""

from __future__ import annotations

from typing import Any

from . import _compat as _sensor_backend

_PRESET_EXPORTS = sorted(getattr(_sensor_backend, "__all__", []))


def __getattr__(name: str) -> Any:
    """Lazily expose preset helpers and built-in preset constants."""
    if name not in _PRESET_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_sensor_backend, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_PRESET_EXPORTS))


__all__ = _PRESET_EXPORTS
