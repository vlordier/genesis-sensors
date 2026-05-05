"""
Multi-rate sensor scheduler.

Maintains a collection of :class:`~genesis.sensors.base.BaseSensor`
instances and drives them at their individual update rates.  At each
simulation step the caller passes the current time and an ideal-state
dict; the scheduler determines which sensors are *due* for an update,
calls their ``step()`` methods, and returns only the observations that
were refreshed.

Sensors that are not yet due return their *cached* last observation so
that consumers always have a valid (possibly stale) measurement.

Example
-------
::

    scheduler = SensorScheduler()
    scheduler.add(gnss, name="gnss")
    scheduler.add(imu_model, name="imu")

    while sim.running:
        state = collect_genesis_state(scene)
        obs = scheduler.update(sim_time=scene.cur_t, state=state)
        print(obs["gnss"])   # freshly updated (if due) or cached
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from .base import BaseSensor


class SensorScheduler:
    """
    Multi-rate sensor scheduler.

    Parameters
    ----------
    sensors:
        Optional initial list of ``(name, sensor)`` pairs.
    jitter_sigma_s:
        Standard deviation of Gaussian timing jitter applied to the
        simulated measurement timestamp.  Models the fact that real
        sensor hardware does not sample at perfectly uniform intervals.
        ``0`` = no jitter (default).
    add_timestamps:
        When ``True``, each sensor observation dict is augmented with a
        ``"timestamp_s"`` key containing the (possibly jittered) simulation
        time at which the measurement was taken.
    """

    def __init__(
        self,
        sensors: list[tuple[str, BaseSensor]] | None = None,
        jitter_sigma_s: float = 0.0,
        add_timestamps: bool = False,
        seed: int | None = None,
    ) -> None:
        self._sensors: dict[str, BaseSensor] = {}
        self.jitter_sigma_s = float(max(jitter_sigma_s, 0.0))
        self.add_timestamps = bool(add_timestamps)
        self._rng = np.random.default_rng(seed=seed)
        for name, sensor in sensors or []:
            self.add(sensor, name=name)

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def add(self, sensor: BaseSensor, name: str | None = None) -> None:
        """Register a sensor.  *name* defaults to ``sensor.name``."""
        key = name or sensor.name
        if key in self._sensors:
            raise ValueError(f"A sensor named {key!r} is already registered.")
        self._sensors[key] = sensor

    def remove(self, name: str) -> None:
        """Unregister a sensor by name.

        Raises
        ------
        KeyError
            If *name* is not registered, preventing silent no-ops from hiding typos.
        """
        if name not in self._sensors:
            raise KeyError(f"No sensor named {name!r} is registered; registered names: {list(self._sensors)}")
        del self._sensors[name]

    def __contains__(self, name: str) -> bool:
        return name in self._sensors

    def __len__(self) -> int:
        return len(self._sensors)

    # ------------------------------------------------------------------
    # Scheduling
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Reset all registered sensors."""
        for sensor in self._sensors.values():
            sensor.reset(env_id=env_id)

    def update(self, sim_time: float, state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        """
        Advance all due sensors and return a merged observation dict.

        Parameters
        ----------
        sim_time:
            Current simulation time in seconds.
        state:
            Ideal Genesis measurements shared across all sensors.

        Returns
        -------
        dict
            Mapping ``sensor_name -> observation_dict``.  Sensors that
            were not due for an update contribute their cached
            ``get_observation()`` result.

        Raises
        ------
        RuntimeError
            If any sensor raises during its ``step()`` call.  The original
            exception is chained so that the full traceback is preserved.
        """
        obs: dict[str, Mapping[str, Any]] = {}
        for name, sensor in self._sensors.items():
            try:
                if sensor.is_due(sim_time):
                    sensor_obs = sensor.step(sim_time=sim_time, state=state)
                    if self.add_timestamps:
                        ts = sim_time
                        if self.jitter_sigma_s > 0.0:
                            ts += float(self._rng.normal(0.0, self.jitter_sigma_s))
                        sensor_obs["timestamp_s"] = ts
                    obs[name] = sensor_obs
                else:
                    obs[name] = sensor.get_observation()
            except Exception as exc:
                raise RuntimeError(
                    f"Sensor {name!r} ({type(sensor).__name__}) raised during update at sim_time={sim_time:.6f}: {exc}"
                ) from exc
        return obs

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def sensor_names(self) -> list[str]:
        """Return registered sensor names."""
        return list(self._sensors.keys())

    def get_sensor(self, name: str) -> BaseSensor:
        """Return the sensor registered under *name*."""
        try:
            return self._sensors[name]
        except KeyError:
            registered = list(self._sensors)
            raise KeyError(f"No sensor named {name!r}. Registered sensors: {registered}") from None

    def __repr__(self) -> str:
        entries = ", ".join(f"{name}@{s.update_rate_hz}Hz" for name, s in self._sensors.items())
        return f"SensorScheduler([{entries}])"
