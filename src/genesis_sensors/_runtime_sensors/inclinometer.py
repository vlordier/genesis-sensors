"""Dual-axis inclinometer / tilt sensor model.

Separate from IMU: inclinometers are dedicated low-noise, narrow-range
tilt sensors commonly used on heavy machinery, construction robots,
and agricultural equipment.  Typically foam-damped with ±0.01° accuracy.

State keys consumed
-------------------
``"orientation"``
    Body orientation as a quaternion [w, x, y, z] or rotation matrix (3×3).
``"gravity_body"``
    Gravity vector in body frame (3-vec, m/s²).  Preferred if available.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import InclinometerConfig


class InclinometerModel(BaseSensor):
    """Dual-axis inclinometer with foam-damped response.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    noise_sigma_deg:
        1-σ angular noise (degrees).  Typical: 0.001–0.01°.
    bias_deg:
        Constant angular bias (degrees).
    range_deg:
        Maximum measurable tilt (degrees).  Readings beyond are clipped.
    response_tau_s:
        First-order response time constant (s).  Models fluid damping.
        0 = instantaneous response.
    resolution_deg:
        Quantisation step (degrees).  0 = no quantisation.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "inclinometer",
        update_rate_hz: float = 50.0,
        noise_sigma_deg: float = 0.005,
        bias_deg: float = 0.0,
        range_deg: float = 60.0,
        response_tau_s: float = 0.0,
        resolution_deg: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_deg = float(max(0.0, noise_sigma_deg))
        self.bias_deg = float(bias_deg)
        self.range_deg = float(max(0.0, range_deg))
        self.response_tau_s = float(max(0.0, response_tau_s))
        self.resolution_deg = float(max(0.0, resolution_deg))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._filtered_roll_deg: float = 0.0
        self._filtered_pitch_deg: float = 0.0
        self._prev_time: float | None = None
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "InclinometerConfig") -> "InclinometerModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "InclinometerConfig":
        from .config import InclinometerConfig

        return InclinometerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_deg=self.noise_sigma_deg,
            bias_deg=self.bias_deg,
            range_deg=self.range_deg,
            response_tau_s=self.response_tau_s,
            resolution_deg=self.resolution_deg,
            seed=self._seed,
        )

    @staticmethod
    def _gravity_to_tilt(g_body: np.ndarray) -> tuple[float, float]:
        """Convert gravity vector in body frame to roll/pitch (degrees)."""
        gx, gy, gz = float(g_body[0]), float(g_body[1]), float(g_body[2])
        g_mag = math.sqrt(gx**2 + gy**2 + gz**2)
        if g_mag < 1e-6:
            return 0.0, 0.0
        # Roll = atan2(gy, gz), Pitch = atan2(-gx, sqrt(gy²+gz²))
        roll = math.degrees(math.atan2(gy, gz))
        pitch = math.degrees(math.atan2(-gx, math.sqrt(gy**2 + gz**2)))
        return roll, pitch

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        # Get true tilt from gravity in body frame
        g_body = state.get("gravity_body")
        if g_body is not None:
            g_body = np.asarray(g_body, dtype=np.float64)[:3]
        else:
            # Fall back to default downward gravity
            g_body = np.array([0.0, 0.0, -9.80665], dtype=np.float64)

        true_roll, true_pitch = self._gravity_to_tilt(g_body)

        # First-order response lag
        if self.response_tau_s > 0.0 and self._prev_time is not None:
            dt = sim_time - self._prev_time
            if dt > 0.0:
                alpha = min(1.0, dt / (self.response_tau_s + dt))
                self._filtered_roll_deg += alpha * (true_roll - self._filtered_roll_deg)
                self._filtered_pitch_deg += alpha * (true_pitch - self._filtered_pitch_deg)
            roll_out = self._filtered_roll_deg
            pitch_out = self._filtered_pitch_deg
        else:
            self._filtered_roll_deg = true_roll
            self._filtered_pitch_deg = true_pitch
            roll_out = true_roll
            pitch_out = true_pitch

        self._prev_time = sim_time

        # Add bias and noise
        roll_out += self.bias_deg
        pitch_out += self.bias_deg
        if self.noise_sigma_deg > 0.0:
            roll_out += float(self._rng.normal(0.0, self.noise_sigma_deg))
            pitch_out += float(self._rng.normal(0.0, self.noise_sigma_deg))

        # Clip to range
        roll_out = float(np.clip(roll_out, -self.range_deg, self.range_deg))
        pitch_out = float(np.clip(pitch_out, -self.range_deg, self.range_deg))

        # Quantisation
        if self.resolution_deg > 0.0:
            roll_out = round(roll_out / self.resolution_deg) * self.resolution_deg
            pitch_out = round(pitch_out / self.resolution_deg) * self.resolution_deg

        self._last_obs = {
            "roll_deg": roll_out,
            "pitch_deg": pitch_out,
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._filtered_roll_deg = 0.0
        self._filtered_pitch_deg = 0.0
        self._prev_time = None
        self._last_obs = {}
        self._last_update_time = -1.0
