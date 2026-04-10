"""
Joint state sensor model for robotic arms and legged systems.

Reads ideal joint positions, velocities, and torques from Genesis and adds
per-axis Gaussian noise to simulate encoder-based proprioception.  An
optional first-order low-pass filter can be applied to the velocity channel
to mimic the differentiation filtering used in many hardware drivers.

State keys consumed
-------------------
``"joint_pos"``
    Ideal joint positions, shape ``(N,)`` in radians.  Defaults to zeros
    when absent.
``"joint_vel"``
    Ideal joint velocities, shape ``(N,)`` in rad/s.  Defaults to zeros
    when absent.
``"joint_torque"``
    Ideal joint torques, shape ``(N,)`` in Nm.  Defaults to zeros when
    absent.

Observation keys
----------------
``"joint_pos_rad"``
    ``float32`` shape ``(N,)`` — noisy joint positions in radians.
``"joint_vel_rads"``
    ``float32`` shape ``(N,)`` — noisy (optionally filtered) joint
    velocities in rad/s.
``"joint_torque_nm"``
    ``float32`` shape ``(N,)`` — noisy joint torques in Nm.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor

if TYPE_CHECKING:
    from .config import JointStateConfig


class JointStateSensor(BaseSensor):
    """
    Proprioceptive joint state sensor (position / velocity / torque).

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Output rate (Hz).
    pos_noise_sigma_rad:
        Per-joint Gaussian position noise 1-σ (rad).
    vel_noise_sigma_rads:
        Per-joint Gaussian velocity noise 1-σ (rad/s).
    torque_noise_sigma_nm:
        Per-joint Gaussian torque noise 1-σ (Nm).
    velocity_filter_alpha:
        First-order low-pass smoothing factor for velocity (0 = off,
        values > 0 apply  ``v_filt = (1-α) * v_filt_prev + α * v_raw``).
        Typical hardware values are in [0.1, 0.5].
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "joint_state",
        update_rate_hz: float = 1000.0,
        pos_noise_sigma_rad: float = 0.0001,
        vel_noise_sigma_rads: float = 0.001,
        torque_noise_sigma_nm: float = 0.01,
        velocity_filter_alpha: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.pos_noise_sigma_rad = float(pos_noise_sigma_rad)
        self.vel_noise_sigma_rads = float(vel_noise_sigma_rads)
        self.torque_noise_sigma_nm = float(torque_noise_sigma_nm)
        self.velocity_filter_alpha = float(velocity_filter_alpha)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._vel_filtered: np.ndarray | None = None  # lazy-initialised on first step
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "JointStateConfig") -> "JointStateSensor":
        """Construct from a :class:`~genesis.sensors.config.JointStateConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "JointStateConfig":
        """Serialise parameters back to a :class:`~genesis.sensors.config.JointStateConfig`."""
        from .config import JointStateConfig

        return JointStateConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            pos_noise_sigma_rad=self.pos_noise_sigma_rad,
            vel_noise_sigma_rads=self.vel_noise_sigma_rads,
            torque_noise_sigma_nm=self.torque_noise_sigma_nm,
            velocity_filter_alpha=self.velocity_filter_alpha,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Reset velocity filter state, cached observation, and scheduling state."""
        self._vel_filtered = None
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Compute a noisy joint-state observation.

        Parameters
        ----------
        sim_time:
            Current simulation time (s) — not used internally but kept for
            API consistency.
        state:
            Dict with optional keys ``"joint_pos"``, ``"joint_vel"``,
            ``"joint_torque"`` (all shape ``(N,)``).

        Returns
        -------
        dict
            ``"joint_pos_rad"``, ``"joint_vel_rads"``, ``"joint_torque_nm"``
            as ``float32`` arrays.
        """
        raw_pos = np.asarray(state.get("joint_pos", [0.0]), dtype=np.float32)
        raw_vel = np.asarray(state.get("joint_vel", np.zeros_like(raw_pos)), dtype=np.float32)
        raw_torque = np.asarray(state.get("joint_torque", np.zeros_like(raw_pos)), dtype=np.float32)

        n = raw_pos.shape[0]

        # --- Position noise ---
        noisy_pos = raw_pos + self._rng.normal(0.0, self.pos_noise_sigma_rad, size=n).astype(np.float32)

        # --- Velocity noise + optional LP filter ---
        noisy_vel = raw_vel + self._rng.normal(0.0, self.vel_noise_sigma_rads, size=n).astype(np.float32)
        if self.velocity_filter_alpha > 0.0:
            alpha = self.velocity_filter_alpha
            if self._vel_filtered is None or self._vel_filtered.shape[0] != n:
                self._vel_filtered = noisy_vel.copy()
            else:
                self._vel_filtered = (1.0 - alpha) * self._vel_filtered + alpha * noisy_vel
            noisy_vel = self._vel_filtered.astype(np.float32)

        # --- Torque noise ---
        noisy_torque = raw_torque + self._rng.normal(0.0, self.torque_noise_sigma_nm, size=n).astype(np.float32)

        obs: dict[str, Any] = {
            "joint_pos_rad": noisy_pos,
            "joint_vel_rads": noisy_vel,
            "joint_torque_nm": noisy_torque,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        """Return the most recent observation without triggering a new step."""
        return self._last_obs

    def __repr__(self) -> str:
        return (
            f"JointStateSensor(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"n_pos_noise={self.pos_noise_sigma_rad:.4f} rad)"
        )
