"""
Wheel odometry sensor model for ground robots.

Applies encoder quantisation noise, wheel slip, and Gaussian dead-reckoning
noise to ideal velocity / angular-velocity state to simulate a hardware
odometry encoder stack.

State keys consumed
-------------------
``"vel"``
    World-frame linear velocity, shape ``(3,)`` in m/s.  Only the XY
    plane (first two components) is used.
``"ang_vel"``
    Body-frame angular velocity, shape ``(3,)`` in rad/s.  Only the
    Z-axis (third component, yaw rate) is used.

Both keys are optional; they default to zero when absent.

Observation keys
----------------
``"delta_pos_m"``
    ``float32`` shape ``(2,)`` — world-frame XY position increment this
    step in metres.
``"delta_heading_rad"``
    ``float32`` — heading (yaw) change this step in radians.
``"linear_vel_ms"``
    ``float32`` — estimated ground-plane speed (m/s).
``"angular_vel_rads"``
    ``float32`` — estimated yaw rate (rad/s).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor

if TYPE_CHECKING:
    from .config import WheelOdometryConfig


class WheelOdometryModel(BaseSensor):
    """
    Wheel odometry model for differential-drive and similar ground robots.

    Parameters
    ----------
    name:
        Sensor identifier used in :class:`~genesis.sensors.SensorSuite`.
    update_rate_hz:
        Odometry publish rate (Hz).
    pos_noise_sigma_m:
        1-σ Gaussian noise on each axis of the delta-position step (m).
    heading_noise_sigma_rad:
        1-σ Gaussian noise on the heading change per step (rad).
    slip_sigma:
        Wheel-slip standard deviation as a fraction of instantaneous speed.
        A random multiplicative scale drawn from N(1, slip_sigma) is applied
        to the velocity before dead-reckoning integration.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "wheel_odometry",
        update_rate_hz: float = 50.0,
        pos_noise_sigma_m: float = 0.002,
        heading_noise_sigma_rad: float = 0.001,
        slip_sigma: float = 0.005,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.pos_noise_sigma_m = float(pos_noise_sigma_m)
        self.heading_noise_sigma_rad = float(heading_noise_sigma_rad)
        self.slip_sigma = float(slip_sigma)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "WheelOdometryConfig") -> "WheelOdometryModel":
        """Construct from a :class:`~genesis.sensors.config.WheelOdometryConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "WheelOdometryConfig":
        """Serialise current parameters to a :class:`~genesis.sensors.config.WheelOdometryConfig`."""
        from .config import WheelOdometryConfig

        return WheelOdometryConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            pos_noise_sigma_m=self.pos_noise_sigma_m,
            heading_noise_sigma_rad=self.heading_noise_sigma_rad,
            slip_sigma=self.slip_sigma,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Compute noisy wheel-odometry increments for one update tick.

        Parameters
        ----------
        sim_time:
            Current simulation time (s).
        state:
            Should contain ``"vel"`` (3,) m/s and/or ``"ang_vel"`` (3,) rad/s.

        Returns
        -------
        dict
            Keys: ``delta_pos_m``, ``delta_heading_rad``,
            ``linear_vel_ms``, ``angular_vel_rads``.
        """
        dt = 1.0 / self.update_rate_hz

        # Ideal world-frame XY velocity
        vel_raw = state.get("vel")
        if vel_raw is not None:
            vel = np.asarray(vel_raw, dtype=np.float64).flatten()
            vx = float(vel[0]) if len(vel) > 0 else 0.0
            vy = float(vel[1]) if len(vel) > 1 else 0.0
        else:
            vx, vy = 0.0, 0.0

        # Ideal yaw rate (body-frame Z, index 2)
        ang_raw = state.get("ang_vel")
        if ang_raw is not None:
            ang = np.asarray(ang_raw, dtype=np.float32).flatten()
            omega = float(ang[2]) if len(ang) >= 3 else 0.0
        else:
            omega = 0.0

        # Wheel slip: multiplicative scale on velocity magnitude
        v_linear = float(np.sqrt(vx**2 + vy**2))
        if v_linear > 1e-9 and self.slip_sigma > 0.0:
            slip_factor = float(self._rng.normal(1.0, self.slip_sigma))
        else:
            slip_factor = 1.0

        # Dead-reckoning increments with additive noise
        delta_x = vx * slip_factor * dt + float(self._rng.normal(0.0, self.pos_noise_sigma_m))
        delta_y = vy * slip_factor * dt + float(self._rng.normal(0.0, self.pos_noise_sigma_m * 0.5))
        delta_heading = omega * dt + float(self._rng.normal(0.0, self.heading_noise_sigma_rad))

        linear_vel_noisy = float(np.sqrt((vx * slip_factor) ** 2 + (vy * slip_factor) ** 2))
        angular_vel_noisy = omega + float(self._rng.normal(0.0, self.heading_noise_sigma_rad / dt))

        obs: dict[str, Any] = {
            "delta_pos_m": np.array([delta_x, delta_y], dtype=np.float32),
            "delta_heading_rad": np.float32(delta_heading),
            "linear_vel_ms": np.float32(linear_vel_noisy),
            "angular_vel_rads": np.float32(angular_vel_noisy),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    def __repr__(self) -> str:
        return (
            f"WheelOdometryModel(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"pos_noise={self.pos_noise_sigma_m} m, heading_noise={self.heading_noise_sigma_rad} rad)"
        )
