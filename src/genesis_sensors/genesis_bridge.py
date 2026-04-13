"""Genesis state-bridge helper re-exports for the active sensor backend."""

from __future__ import annotations

from typing import Any

from . import _compat as _sensor_backend

_BRIDGE_EXPORTS = sorted(name for name in getattr(_sensor_backend, "__all__", []) if name.startswith("extract_"))


def __getattr__(name: str) -> Any:
    """Lazily expose Genesis state-extraction helpers from the selected backend."""
    if name not in _BRIDGE_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(_sensor_backend, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_BRIDGE_EXPORTS))


__all__ = _BRIDGE_EXPORTS
