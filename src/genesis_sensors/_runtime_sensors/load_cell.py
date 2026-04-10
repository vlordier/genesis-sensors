"""Load cell / weight sensor model.

Single-axis force measurement for payload weighing, hopper monitoring,
and crane operations.  Distinct from the 6-axis F/T sensor: higher
range, lower resolution, simpler interface.

State keys consumed
-------------------
``"load_force_n"``
    Force along the measurement axis (N).  Positive = compression.
``"temperature_c"``
    Ambient temperature (°C).  Default 25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import LoadCellConfig


class LoadCellModel(BaseSensor):
    """Single-axis load cell / weight sensor.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    capacity_n:
        Rated capacity (N).  Readings are clipped to ±capacity.
    noise_sigma_n:
        1-σ Gaussian noise (N).
    tare_offset_n:
        Constant offset / tare error (N).
    creep_rate_per_s:
        Fractional creep rate under sustained load (per second).
        Output drifts by ``creep_rate * load * dt`` each step.  0 = off.
    temp_coeff_n_per_c:
        Temperature coefficient for zero drift (N/°C, ref 25 °C).
    resolution_n:
        ADC quantisation step (N).  0 = continuous.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "load_cell",
        update_rate_hz: float = 50.0,
        capacity_n: float = 5000.0,
        noise_sigma_n: float = 0.5,
        tare_offset_n: float = 0.0,
        creep_rate_per_s: float = 0.0,
        temp_coeff_n_per_c: float = 0.0,
        resolution_n: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.capacity_n = float(max(0.0, capacity_n))
        self.noise_sigma_n = float(max(0.0, noise_sigma_n))
        self.tare_offset_n = float(tare_offset_n)
        self.creep_rate_per_s = float(max(0.0, creep_rate_per_s))
        self.temp_coeff_n_per_c = float(temp_coeff_n_per_c)
        self.resolution_n = float(max(0.0, resolution_n))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._creep_accumulator: float = 0.0
        self._prev_time: float | None = None
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "LoadCellConfig") -> "LoadCellModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "LoadCellConfig":
        from .config import LoadCellConfig

        return LoadCellConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            capacity_n=self.capacity_n,
            noise_sigma_n=self.noise_sigma_n,
            tare_offset_n=self.tare_offset_n,
            creep_rate_per_s=self.creep_rate_per_s,
            temp_coeff_n_per_c=self.temp_coeff_n_per_c,
            resolution_n=self.resolution_n,
            seed=self._seed,
        )

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        true_force = float(state.get("load_force_n", 0.0))
        temp_c = float(state.get("temperature_c", 25.0))

        # Creep accumulation under sustained load
        if self.creep_rate_per_s > 0.0 and self._prev_time is not None:
            dt = sim_time - self._prev_time
            if dt > 0.0:
                self._creep_accumulator += self.creep_rate_per_s * true_force * dt
        self._prev_time = sim_time

        # Build reading
        reading = true_force + self.tare_offset_n + self._creep_accumulator

        # Temperature drift
        if self.temp_coeff_n_per_c != 0.0:
            reading += self.temp_coeff_n_per_c * (temp_c - 25.0)

        # Noise
        if self.noise_sigma_n > 0.0:
            reading += float(self._rng.normal(0.0, self.noise_sigma_n))

        # Clip to capacity
        reading = float(np.clip(reading, -self.capacity_n, self.capacity_n))

        # Quantisation
        if self.resolution_n > 0.0:
            reading = round(reading / self.resolution_n) * self.resolution_n

        overload = abs(true_force) >= self.capacity_n * 0.95

        self._last_obs = {
            "force_n": reading,
            "weight_kg": reading / 9.80665,
            "overload": overload,
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._creep_accumulator = 0.0
        self._prev_time = None
        self._last_obs = {}
        self._last_update_time = -1.0
