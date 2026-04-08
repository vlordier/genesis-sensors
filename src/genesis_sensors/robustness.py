"""Robustness helpers for injecting latency/dropouts and surfacing sensor health metadata."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

import numpy as np

from . import SENSOR_BACKEND, SensorScheduler, SensorSuite


@dataclass(slots=True)
class SensorFaultConfig:
    """Configuration for latency/dropout injection on a wrapped sensor."""

    latency_s: float = 0.0
    dropout_prob: float = 0.0
    hold_last_on_dropout: bool = True
    include_metadata: bool = True
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.latency_s < 0.0:
            raise ValueError(f"latency_s must be >= 0, got {self.latency_s}")
        if not 0.0 <= self.dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {self.dropout_prob}")


class RobustSensorWrapper:
    """Wrap a sensor-like object with latency, dropout, and health metadata.

    The wrapper is intentionally backend-agnostic: it only assumes the wrapped
    sensor exposes ``name``, ``update_rate_hz``, ``reset()``, ``step()``,
    ``get_observation()``, and optionally ``is_due()``.
    """

    def __init__(
        self,
        sensor: Any,
        *,
        name: str | None = None,
        latency_s: float = 0.0,
        dropout_prob: float = 0.0,
        hold_last_on_dropout: bool = True,
        include_metadata: bool = True,
        seed: int | None = None,
    ) -> None:
        if latency_s < 0.0:
            raise ValueError(f"latency_s must be >= 0, got {latency_s}")
        if not 0.0 <= dropout_prob <= 1.0:
            raise ValueError(f"dropout_prob must be in [0, 1], got {dropout_prob}")

        self.sensor = sensor
        self.name = name or getattr(sensor, "name", type(sensor).__name__)
        self.update_rate_hz = float(getattr(sensor, "update_rate_hz", 1.0))
        self.latency_s = float(latency_s)
        self.dropout_prob = float(dropout_prob)
        self.hold_last_on_dropout = bool(hold_last_on_dropout)
        self.include_metadata = bool(include_metadata)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._history: deque[tuple[float, dict[str, Any]]] = deque()
        self._last_payload: dict[str, Any] = {}
        self._last_obs: dict[str, Any] = {}
        self._last_source_time: float | None = None
        self._last_update_time: float = -1.0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.sensor, name)

    def reset(self, env_id: int = 0) -> None:
        self.sensor.reset(env_id=env_id)
        self._history.clear()
        self._last_payload = {}
        self._last_obs = {}
        self._last_source_time = None
        self._last_update_time = -1.0

    def is_due(self, sim_time: float) -> bool:
        sensor_is_due = getattr(self.sensor, "is_due", None)
        if callable(sensor_is_due):
            return bool(sensor_is_due(sim_time))
        if self._last_update_time < 0.0:
            return True
        return (sim_time - self._last_update_time) >= 1.0 / self.update_rate_hz

    def step(self, sim_time: float, state: Mapping[str, Any]) -> dict[str, Any]:
        raw_obs = self._copy_mapping(self.sensor.step(sim_time=sim_time, state=state))
        self._last_update_time = float(sim_time)
        self._history.append((float(sim_time), raw_obs))

        ready_obs: dict[str, Any] | None = None
        ready_source_time: float | None = None
        if self.latency_s <= 0.0:
            ready_obs = self._copy_mapping(raw_obs)
            ready_source_time = float(sim_time)
        else:
            release_time = float(sim_time) - self.latency_s + 1e-12
            while self._history and self._history[0][0] <= release_time:
                ready_source_time, ready_obs = self._history.popleft()

        if ready_obs is None:
            if self._last_payload:
                return self._annotate(
                    self._last_payload,
                    sim_time=sim_time,
                    source_time=self._last_source_time,
                    status="warming_up",
                    dropped=False,
                )
            return self._annotate({}, sim_time=sim_time, source_time=None, status="warming_up", dropped=False)

        if self.dropout_prob > 0.0 and float(self._rng.random()) < self.dropout_prob:
            if self.hold_last_on_dropout and self._last_payload:
                return self._annotate(
                    self._last_payload,
                    sim_time=sim_time,
                    source_time=self._last_source_time,
                    status="dropout_hold",
                    dropped=True,
                )
            return self._annotate(
                {},
                sim_time=sim_time,
                source_time=ready_source_time,
                status="dropout",
                dropped=True,
            )

        self._last_payload = self._copy_mapping(ready_obs)
        self._last_source_time = ready_source_time
        status = "ok" if ready_source_time is not None and abs(sim_time - ready_source_time) < 1e-9 else "delayed"
        return self._annotate(ready_obs, sim_time=sim_time, source_time=ready_source_time, status=status, dropped=False)

    def get_observation(self) -> dict[str, Any]:
        return self._copy_mapping(self._last_obs)

    def _annotate(
        self,
        payload: Mapping[str, Any],
        *,
        sim_time: float,
        source_time: float | None,
        status: str,
        dropped: bool,
    ) -> dict[str, Any]:
        obs = self._copy_mapping(payload)
        if self.include_metadata:
            obs["_meta"] = {
                "sensor_name": self.name,
                "wrapped_sensor_type": type(self.sensor).__name__,
                "backend": SENSOR_BACKEND,
                "sim_time": float(sim_time),
                "source_time": None if source_time is None else float(source_time),
                "age_s": None if source_time is None else float(max(0.0, sim_time - source_time)),
                "status": status,
                "dropped": bool(dropped),
                "latency_s": self.latency_s,
                "dropout_prob": self.dropout_prob,
            }
        self._last_obs = obs
        return self._copy_mapping(obs)

    @staticmethod
    def _copy_mapping(payload: Mapping[str, Any] | None) -> dict[str, Any]:
        if payload is None:
            return {}
        return deepcopy(dict(payload))

    def __repr__(self) -> str:
        return (
            f"RobustSensorWrapper(name={self.name!r}, sensor={type(self.sensor).__name__}, "
            f"latency_s={self.latency_s}, dropout_prob={self.dropout_prob})"
        )


def _value_for_sensor(value: float | Mapping[str, float], name: str, default: float = 0.0) -> float:
    if isinstance(value, Mapping):
        return float(value.get(name, default))
    return float(value)


def wrap_scheduler_with_faults(
    scheduler: SensorScheduler,
    *,
    latency_s: float | Mapping[str, float] = 0.0,
    dropout_prob: float | Mapping[str, float] = 0.0,
    hold_last_on_dropout: bool = True,
    include_metadata: bool = True,
    seed: int | None = None,
) -> SensorScheduler:
    """Return a new scheduler whose sensors are wrapped with fault injection."""
    wrapped = SensorScheduler()
    names = scheduler.sensor_names()
    child_seeds = np.random.SeedSequence(seed).spawn(len(names)) if seed is not None else [None] * len(names)

    for idx, name in enumerate(names):
        child_seed = None if child_seeds[idx] is None else int(child_seeds[idx].generate_state(1)[0])
        sensor = scheduler.get_sensor(name)
        wrapped.add(
            RobustSensorWrapper(
                sensor,
                name=name,
                latency_s=_value_for_sensor(latency_s, name),
                dropout_prob=_value_for_sensor(dropout_prob, name),
                hold_last_on_dropout=hold_last_on_dropout,
                include_metadata=include_metadata,
                seed=child_seed,
            ),
            name=name,
        )
    return wrapped


def wrap_suite_with_faults(
    suite: SensorSuite,
    *,
    latency_s: float | Mapping[str, float] = 0.0,
    dropout_prob: float | Mapping[str, float] = 0.0,
    hold_last_on_dropout: bool = True,
    include_metadata: bool = True,
    seed: int | None = None,
) -> SensorSuite:
    """Clone a :class:`SensorSuite` while wrapping its sensors with robustness metadata/faults."""
    wrapped = SensorSuite()
    wrapped._scheduler = wrap_scheduler_with_faults(
        suite.scheduler,
        latency_s=latency_s,
        dropout_prob=dropout_prob,
        hold_last_on_dropout=hold_last_on_dropout,
        include_metadata=include_metadata,
        seed=seed,
    )
    return wrapped


def wrap_rig_with_faults(
    rig: Any,
    *,
    latency_s: float | Mapping[str, float] = 0.0,
    dropout_prob: float | Mapping[str, float] = 0.0,
    hold_last_on_dropout: bool = True,
    include_metadata: bool = True,
    seed: int | None = None,
) -> Any:
    """Clone a high-level :class:`~genesis_sensors.rigs.SensorRig` with wrapped sensors."""
    from .rigs import SensorRig

    return SensorRig(
        name=f"{rig.name}_robust",
        suite=wrap_suite_with_faults(
            rig.suite,
            latency_s=latency_s,
            dropout_prob=dropout_prob,
            hold_last_on_dropout=hold_last_on_dropout,
            include_metadata=include_metadata,
            seed=seed,
        ),
        state_fn=rig.state_fn,
        metadata={**getattr(rig, "metadata", {}), "fault_injection": True},
        reset_fn=getattr(rig, "reset_fn", None),
    )


__all__ = [
    "RobustSensorWrapper",
    "SensorFaultConfig",
    "wrap_rig_with_faults",
    "wrap_scheduler_with_faults",
    "wrap_suite_with_faults",
]
