"""
Binary and analog contact / tactile sensor model.

Reads an ideal contact-force magnitude from Genesis, adds Gaussian noise,
and applies a configurable threshold to produce a binary contact detection
signal.  An optional software debounce filter holds the contact state for
a given number of consecutive steps before toggling.

State keys consumed
-------------------
``"contact_force_n"``
    Scalar ideal contact force magnitude in Newtons.  Defaults to 0 when
    absent.

Observation keys
----------------
``"contact_detected"``
    ``bool`` — ``True`` when filtered noisy force exceeds *force_threshold_n*.
``"force_n"``
    ``float32`` — noisy contact force magnitude (N), clipped to
    ``[0, force_range_n]``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor

if TYPE_CHECKING:
    from .config import ContactSensorConfig


class ContactSensor(BaseSensor):
    """
    Binary + analog contact sensor model.

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Output rate (Hz).
    force_threshold_n:
        Force magnitude above which contact is declared (N).
    noise_sigma_n:
        Gaussian noise 1-σ on the measured force (N).
    force_range_n:
        Maximum measurable force (N); output is clipped to this value.
    debounce_steps:
        Minimum number of consecutive steps a new state must persist before
        the output toggles.  0 = no debounce (instantaneous).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "contact",
        update_rate_hz: float = 200.0,
        force_threshold_n: float = 0.5,
        noise_sigma_n: float = 0.02,
        force_range_n: float = 50.0,
        debounce_steps: int = 0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.force_threshold_n = float(force_threshold_n)
        self.noise_sigma_n = float(noise_sigma_n)
        self.force_range_n = float(force_range_n)
        self.debounce_steps = int(debounce_steps)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        # Debounce state
        self._pending_state: bool = False
        self._pending_count: int = 0
        self._confirmed_state: bool = False
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "ContactSensorConfig") -> "ContactSensor":
        """Construct from a :class:`~genesis.sensors.config.ContactSensorConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "ContactSensorConfig":
        """Serialise parameters back to a :class:`~genesis.sensors.config.ContactSensorConfig`."""
        from .config import ContactSensorConfig

        return ContactSensorConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            force_threshold_n=self.force_threshold_n,
            noise_sigma_n=self.noise_sigma_n,
            force_range_n=self.force_range_n,
            debounce_steps=self.debounce_steps,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Reset debounce state, cached observation, and scheduling state."""
        self._pending_state = False
        self._pending_count = 0
        self._confirmed_state = False
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Compute a noisy contact observation.

        Parameters
        ----------
        sim_time:
            Current simulation time (s).
        state:
            Dict with optional key ``"contact_force_n"`` (scalar, N).

        Returns
        -------
        dict
            ``"contact_detected"`` (bool), ``"force_n"`` (float32).
        """
        ideal_force = float(state.get("contact_force_n", 0.0))
        noisy_force = ideal_force + float(self._rng.normal(0.0, self.noise_sigma_n))
        noisy_force = float(np.clip(noisy_force, 0.0, self.force_range_n))

        # Raw detection before debounce
        raw_contact = noisy_force >= self.force_threshold_n

        # --- Debounce logic ---
        if self.debounce_steps <= 0:
            contact_detected = raw_contact
        else:
            if raw_contact == self._pending_state:
                self._pending_count += 1
            else:
                self._pending_state = raw_contact
                self._pending_count = 1

            if self._pending_count >= self.debounce_steps:
                self._confirmed_state = self._pending_state
            contact_detected = self._confirmed_state

        obs: dict[str, Any] = {
            "contact_detected": bool(contact_detected),
            "force_n": np.float32(noisy_force),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        """Return the most recent observation without triggering a new step."""
        return self._last_obs

    def __repr__(self) -> str:
        return (
            f"ContactSensor(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"threshold={self.force_threshold_n} N, debounce={self.debounce_steps})"
        )
