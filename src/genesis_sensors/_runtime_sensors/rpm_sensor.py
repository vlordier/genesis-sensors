"""
Motor / rotor RPM sensor model.

Models a quadrature encoder or magnetic angle sensor mounted on a shaft.
Adds Gaussian noise and quantisation based on the encoder resolution
(counts-per-revolution).

State keys consumed
-------------------
``"rpm"``
    Ideal shaft speed in revolutions per minute.  Takes priority over
    ``"ang_vel_rads"`` when present.
``"ang_vel_rads"``
    Ideal angular velocity in rad/s.  Converted to RPM when ``"rpm"``
    is absent.

Observation keys
----------------
``"rpm"``
    ``float32`` scalar — noisy, quantised shaft speed in RPM.
``"speed_rads"``
    ``float32`` scalar — equivalent angular velocity in rad/s.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor
from .types import RPMObservation

if TYPE_CHECKING:
    from .config import RPMSensorConfig

_TWO_PI: float = 2.0 * math.pi


class RPMSensor(BaseSensor[RPMObservation]):
    """
    Quadrature encoder / magnetic angle sensor for rotor or motor speed.

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Measurement rate (Hz).
    cpr:
        Counts per revolution.  Determines quantisation resolution:
        ``rpm_lsb = 60 × update_rate_hz / cpr``.  Set to ``0`` to
        disable quantisation entirely.
    noise_sigma_rpm:
        Gaussian noise 1-σ in RPM.
    rpm_range:
        Maximum measurable speed (RPM); readings are clipped to
        ``[-rpm_range, rpm_range]``.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "rpm",
        update_rate_hz: float = 500.0,
        cpr: int = 1024,
        noise_sigma_rpm: float = 5.0,
        rpm_range: float = 20_000.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.cpr = int(cpr)
        self.noise_sigma_rpm = float(noise_sigma_rpm)
        self.rpm_range = float(rpm_range)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._last_obs: RPMObservation | dict[str, Any] = {}

        # Pre-compute quantisation step (RPM per LSB)
        if self.cpr > 0 and self.update_rate_hz > 0:
            self._rpm_lsb: float = 60.0 * self.update_rate_hz / self.cpr
        else:
            self._rpm_lsb = 0.0

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "RPMSensorConfig") -> "RPMSensor":
        """Construct from a :class:`~genesis.sensors.config.RPMSensorConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "RPMSensorConfig":
        """Serialise parameters back to a :class:`~genesis.sensors.config.RPMSensorConfig`."""
        from .config import RPMSensorConfig

        return RPMSensorConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            cpr=self.cpr,
            noise_sigma_rpm=self.noise_sigma_rpm,
            rpm_range=self.rpm_range,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_ideal_rpm(self, state: Mapping[str, Any]) -> float:
        """Read the ideal shaft speed from the shared state."""
        if "rpm" in state:
            return float(state["rpm"])
        if "ang_vel_rads" in state:
            return float(state["ang_vel_rads"]) * 60.0 / _TWO_PI
        return 0.0

    def _measure_rpm(self, ideal_rpm: float) -> float:
        """Apply noise, quantisation, and sensor saturation to the ideal RPM."""
        if self.noise_sigma_rpm > 0.0:
            measured = ideal_rpm + float(self._rng.normal(0.0, self.noise_sigma_rpm))
        else:
            measured = ideal_rpm

        if self._rpm_lsb > 0.0:
            measured = round(measured / self._rpm_lsb) * self._rpm_lsb
        return float(np.clip(measured, -self.rpm_range, self.rpm_range))

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Clear the cached observation and scheduler timestamp."""
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: Mapping[str, Any]) -> RPMObservation:
        """Compute a noisy, quantised RPM observation."""
        measured_rpm = self._measure_rpm(self._extract_ideal_rpm(state))
        obs: RPMObservation = {
            "rpm": measured_rpm,
            "speed_rads": measured_rpm * _TWO_PI / 60.0,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> RPMObservation | dict[str, Any]:
        """Return the most recent observation without triggering a new step."""
        return self._last_obs

    def __repr__(self) -> str:
        return (
            f"RPMSensor(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"cpr={self.cpr}, noise={self.noise_sigma_rpm} rpm, "
            f"range={self.rpm_range} rpm)"
        )


__all__ = ["RPMSensor"]
