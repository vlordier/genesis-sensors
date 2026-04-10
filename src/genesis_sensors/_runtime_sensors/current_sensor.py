"""
Current and power monitor sensor model.

Models a shunt-resistor or Hall-effect current sensor attached to a motor
drive or battery bus.  Adds Gaussian noise, a fixed DC offset bias, and
clips the reading to the sensor's measurement range.

State keys consumed
-------------------
``"current_a"``
    Ideal current draw in Amperes.  Defaults to ``0.0`` when absent.
``"voltage_v"``
    Supply voltage in Volts used to compute instantaneous power.  Falls
    back to ``voltage_nominal_v`` when absent.

Observation keys
----------------
``"current_a"``
    ``float32`` scalar — noisy measured current (A).
``"power_w"``
    ``float32`` scalar — instantaneous power estimate (W).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor
from .types import CurrentObservation

if TYPE_CHECKING:
    from .config import CurrentSensorConfig


class CurrentSensor(BaseSensor[CurrentObservation]):
    """
    Shunt-resistor / Hall-effect current and power monitor.

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Sampling rate (Hz).
    noise_sigma_a:
        Gaussian current noise 1-σ (A).
    offset_a:
        Fixed DC offset bias added to every reading (A).
    range_a:
        Maximum measurable current magnitude (A); readings are clipped to
        ``[-range_a, range_a]``.
    voltage_nominal_v:
        Fallback supply voltage (V) used for power computation when
        ``"voltage_v"`` is absent from the state dict.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "current",
        update_rate_hz: float = 200.0,
        noise_sigma_a: float = 0.05,
        offset_a: float = 0.0,
        range_a: float = 50.0,
        voltage_nominal_v: float = 24.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_a = float(noise_sigma_a)
        self.offset_a = float(offset_a)
        self.range_a = float(range_a)
        self.voltage_nominal_v = float(voltage_nominal_v)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._last_obs: CurrentObservation | dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "CurrentSensorConfig") -> "CurrentSensor":
        """Construct from a :class:`~genesis.sensors.config.CurrentSensorConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "CurrentSensorConfig":
        """Serialise parameters back to a :class:`~genesis.sensors.config.CurrentSensorConfig`."""
        from .config import CurrentSensorConfig

        return CurrentSensorConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_a=self.noise_sigma_a,
            offset_a=self.offset_a,
            range_a=self.range_a,
            voltage_nominal_v=self.voltage_nominal_v,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _measure_current(self, ideal_current_a: float) -> float:
        """Apply sensor bias, white noise, and range clipping to the ideal current."""
        if self.noise_sigma_a > 0.0:
            measured = ideal_current_a + self.offset_a + float(self._rng.normal(0.0, self.noise_sigma_a))
        else:
            measured = ideal_current_a + self.offset_a
        return float(np.clip(measured, -self.range_a, self.range_a))

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Clear the cached observation and scheduler timestamp."""
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: Mapping[str, Any]) -> CurrentObservation:
        """Compute a noisy current and power observation."""
        ideal_current = float(state.get("current_a", 0.0))
        voltage = float(state.get("voltage_v", self.voltage_nominal_v))
        measured_current = self._measure_current(ideal_current)

        obs: CurrentObservation = {
            "current_a": measured_current,
            "power_w": measured_current * voltage,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> CurrentObservation | dict[str, Any]:
        """Return the most recent observation without triggering a new step."""
        return self._last_obs

    def __repr__(self) -> str:
        return (
            f"CurrentSensor(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"range=±{self.range_a} A, noise={self.noise_sigma_a} A)"
        )


__all__ = ["CurrentSensor"]
