"""Config-model re-exports for the active Genesis Sensors backend."""

from __future__ import annotations

from typing import Any

from . import _compat as _sensor_backend

_CONFIG_EXPORTS = sorted(name for name in getattr(_sensor_backend, "__all__", []) if name.endswith("Config"))


def __getattr__(name: str) -> Any:
    """Lazily resolve validated config models from the selected backend."""
    if name not in _CONFIG_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_sensor_backend, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_CONFIG_EXPORTS))


__all__ = _CONFIG_EXPORTS
