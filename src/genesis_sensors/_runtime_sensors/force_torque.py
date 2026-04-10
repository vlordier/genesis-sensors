"""
Six-axis wrist force/torque sensor model.

Applies per-axis Gaussian noise, constant bias offsets, and saturation
clipping to ideal force and torque measurements from Genesis.

State keys consumed
-------------------
``"force"``
    Ideal 3-axis force in sensor frame (N), shape ``(3,)``.
``"torque"``
    Ideal 3-axis torque in sensor frame (Nm), shape ``(3,)``.

Both keys are optional; they default to zero when absent.

Observation keys
----------------
``"force_n"``
    ``float32`` shape ``(3,)`` — noisy, biased, saturated force
    ``[Fx, Fy, Fz]`` in N.
``"torque_nm"``
    ``float32`` shape ``(3,)`` — noisy, biased, saturated torque
    ``[Tx, Ty, Tz]`` in Nm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor

if TYPE_CHECKING:
    from .config import ForceTorqueConfig


class ForceTorqueSensorModel(BaseSensor):
    """
    Six-axis wrist force/torque sensor model.

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Output rate (Hz).
    force_noise_sigma_n:
        Per-axis Gaussian force noise 1-σ (N).
    torque_noise_sigma_nm:
        Per-axis Gaussian torque noise 1-σ (Nm).
    force_bias_n:
        Constant per-axis force bias (N), shape (3,).  Defaults to zero.
    torque_bias_nm:
        Constant per-axis torque bias (Nm), shape (3,).  Defaults to zero.
    force_range_n:
        Force saturation threshold per axis (N).  Output is clipped to
        ``[−force_range_n, +force_range_n]``.
    torque_range_nm:
        Torque saturation threshold per axis (Nm).
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "force_torque",
        update_rate_hz: float = 1000.0,
        force_noise_sigma_n: float = 0.05,
        torque_noise_sigma_nm: float = 0.002,
        force_bias_n: tuple[float, float, float] | None = None,
        torque_bias_nm: tuple[float, float, float] | None = None,
        force_range_n: float = 200.0,
        torque_range_nm: float = 10.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.force_noise_sigma_n = float(force_noise_sigma_n)
        self.torque_noise_sigma_nm = float(torque_noise_sigma_nm)
        _fb = force_bias_n if force_bias_n is not None else (0.0, 0.0, 0.0)
        _tb = torque_bias_nm if torque_bias_nm is not None else (0.0, 0.0, 0.0)
        self._force_bias = np.asarray(_fb, dtype=np.float32)
        self._torque_bias = np.asarray(_tb, dtype=np.float32)
        self.force_range_n = float(force_range_n)
        self.torque_range_nm = float(torque_range_nm)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}
        # Store originals for get_config serialisation
        self._force_bias_cfg: tuple[float, float, float] = (
            float(self._force_bias[0]),
            float(self._force_bias[1]),
            float(self._force_bias[2]),
        )
        self._torque_bias_cfg: tuple[float, float, float] = (
            float(self._torque_bias[0]),
            float(self._torque_bias[1]),
            float(self._torque_bias[2]),
        )

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "ForceTorqueConfig") -> "ForceTorqueSensorModel":
        """Construct from a :class:`~genesis.sensors.config.ForceTorqueConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "ForceTorqueConfig":
        """Serialise current parameters to a :class:`~genesis.sensors.config.ForceTorqueConfig`."""
        from .config import ForceTorqueConfig

        return ForceTorqueConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            force_noise_sigma_n=self.force_noise_sigma_n,
            torque_noise_sigma_nm=self.torque_noise_sigma_nm,
            force_bias_n=self._force_bias_cfg,
            torque_bias_nm=self._torque_bias_cfg,
            force_range_n=self.force_range_n,
            torque_range_nm=self.torque_range_nm,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """Apply noise, bias, and saturation to ideal force/torque inputs."""
        force_raw = state.get("force")
        if force_raw is not None:
            force = np.asarray(force_raw, dtype=np.float32).flatten()
            if len(force) < 3:
                force = np.pad(force, (0, 3 - len(force)))
            else:
                force = force[:3]
        else:
            force = np.zeros(3, dtype=np.float32)

        torque_raw = state.get("torque")
        if torque_raw is not None:
            torque = np.asarray(torque_raw, dtype=np.float32).flatten()
            if len(torque) < 3:
                torque = np.pad(torque, (0, 3 - len(torque)))
            else:
                torque = torque[:3]
        else:
            torque = np.zeros(3, dtype=np.float32)

        force_noise = self._rng.normal(0.0, self.force_noise_sigma_n, 3).astype(np.float32)
        torque_noise = self._rng.normal(0.0, self.torque_noise_sigma_nm, 3).astype(np.float32)

        force_out = np.clip(
            force + self._force_bias + force_noise,
            -self.force_range_n,
            self.force_range_n,
        ).astype(np.float32)
        torque_out = np.clip(
            torque + self._torque_bias + torque_noise,
            -self.torque_range_nm,
            self.torque_range_nm,
        ).astype(np.float32)

        obs: dict[str, Any] = {"force_n": force_out, "torque_nm": torque_out}
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    def __repr__(self) -> str:
        return (
            f"ForceTorqueSensorModel(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"force_noise={self.force_noise_sigma_n} N, torque_noise={self.torque_noise_sigma_nm} Nm)"
        )
