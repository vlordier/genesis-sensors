"""Wire encoder / draw-wire / cable-extension sensor model.

Linear position measurement for actuators, boom arms, and telescopic
mechanisms using a cable wound on a spring-loaded spool.

State keys consumed
-------------------
``"extension_m"``
    True linear extension (m).
``"temperature_c"``
    Ambient temperature (°C).  Default 25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import WireEncoderConfig


class WireEncoderModel(BaseSensor):
    """Cable-extension / draw-wire position sensor.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    stroke_m:
        Full stroke / maximum extension (m).
    resolution_m:
        Position quantisation step (m).  0 = continuous.
    noise_sigma_m:
        1-σ Gaussian noise (m).
    linearity_pct:
        Non-linearity as % of full stroke.  0 = perfectly linear.
    temp_coeff_m_per_c:
        Thermal expansion coefficient for cable (m/°C, ref 25 °C).
    hysteresis_m:
        Mechanical hysteresis band (m).  Retraction reading differs
        from extension reading by up to this amount.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "wire_encoder",
        update_rate_hz: float = 100.0,
        stroke_m: float = 2.0,
        resolution_m: float = 0.0,
        noise_sigma_m: float = 0.0005,
        linearity_pct: float = 0.0,
        temp_coeff_m_per_c: float = 0.0,
        hysteresis_m: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.stroke_m = float(max(0.01, stroke_m))
        self.resolution_m = float(max(0.0, resolution_m))
        self.noise_sigma_m = float(max(0.0, noise_sigma_m))
        self.linearity_pct = float(max(0.0, linearity_pct))
        self.temp_coeff_m_per_c = float(temp_coeff_m_per_c)
        self.hysteresis_m = float(max(0.0, hysteresis_m))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._prev_extension: float = 0.0
        self._hysteresis_offset: float = 0.0
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "WireEncoderConfig") -> "WireEncoderModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "WireEncoderConfig":
        from .config import WireEncoderConfig

        return WireEncoderConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            stroke_m=self.stroke_m,
            resolution_m=self.resolution_m,
            noise_sigma_m=self.noise_sigma_m,
            linearity_pct=self.linearity_pct,
            temp_coeff_m_per_c=self.temp_coeff_m_per_c,
            hysteresis_m=self.hysteresis_m,
            seed=self._seed,
        )

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        true_ext = float(state.get("extension_m", 0.0))
        true_ext = max(0.0, min(true_ext, self.stroke_m))
        temp_c = float(state.get("temperature_c", 25.0))

        reading = true_ext

        # Non-linearity error (sinusoidal model)
        if self.linearity_pct > 0.0:
            import math

            nl_amplitude = self.linearity_pct / 100.0 * self.stroke_m
            reading += nl_amplitude * math.sin(math.pi * true_ext / self.stroke_m)

        # Temperature drift
        if self.temp_coeff_m_per_c != 0.0:
            reading += self.temp_coeff_m_per_c * (temp_c - 25.0)

        # Hysteresis
        if self.hysteresis_m > 0.0:
            direction = true_ext - self._prev_extension
            if direction > 0:
                self._hysteresis_offset = min(self._hysteresis_offset + self.hysteresis_m * 0.1, self.hysteresis_m / 2)
            elif direction < 0:
                self._hysteresis_offset = max(self._hysteresis_offset - self.hysteresis_m * 0.1, -self.hysteresis_m / 2)
            reading += self._hysteresis_offset
        self._prev_extension = true_ext

        # Noise
        if self.noise_sigma_m > 0.0:
            reading += float(self._rng.normal(0.0, self.noise_sigma_m))

        # Clip
        reading = max(0.0, min(reading, self.stroke_m))

        # Quantisation
        if self.resolution_m > 0.0:
            reading = round(reading / self.resolution_m) * self.resolution_m

        self._last_obs = {
            "extension_m": reading,
            "extension_pct": reading / self.stroke_m * 100.0,
            "at_min": reading <= (self.resolution_m if self.resolution_m > 0 else 0.001),
            "at_max": reading >= self.stroke_m - (self.resolution_m if self.resolution_m > 0 else 0.001),
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._prev_extension = 0.0
        self._hysteresis_offset = 0.0
        self._last_obs = {}
        self._last_update_time = -1.0
