"""
Rangefinder / laser altimeter distance sensor model.

Simulates a 1-D ranging sensor (e.g., TeraRanger One, Garmin LiDAR-Lite,
TFmini Plus, HC-SR04 ultrasonic) directed perpendicular to the flight surface.
The model adds:

* **Range-dependent Gaussian noise** — sigma = ``noise_floor_m`` +
  ``noise_slope * range`` (characterises both fixed measurement noise and
  proportional ranging error at longer distances).
* **Minimum range blind zone** — below ``min_range_m`` the sensor cannot
  distinguish a close return from direct RF/optical leakage; the sensor
  returns ``no_hit_value`` (typically 0).
* **Maximum range dropout** — beyond ``max_range_m`` no reliable return is
  detected; returns ``no_hit_value``.
* **Random dropout** — per-step probability of a missed measurement (simulates
  surface reflectivity failures, sun saturation, or motion blur in ultrasonics).
* **Quantisation** — models the integer LSB count of low-cost sensors.

Usage
-----
::

    rf = RangefinderModel(
        max_range_m=8.0,
        min_range_m=0.1,
        noise_floor_m=0.02,
        noise_slope=0.01,
    )
    obs = rf.step(sim_time, {"range_m": 1.5})
    print(obs["range_m"])    # noisy range measurement (m) or no_hit_value
    print(obs["in_range"])   # True when a valid return was received

The caller is responsible for providing ``state["range_m"]``, which is the
true perpendicular distance from the sensor to the nearest surface.  In a
Genesis simulation this can be obtained from a downward-facing raycaster.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .base import BaseSensor
from .types import RangefinderObservation

if TYPE_CHECKING:
    from .config import RangefinderConfig

# Default no-hit return value.
_DEFAULT_NO_HIT: Final[float] = 0.0
# Minimum allowable time-constant for Gauss-Markov bias processes.
_MIN_BIAS_TAU_S: Final[float] = 1e-3


class RangefinderModel(BaseSensor):
    """
    Single-axis laser / sonar rangefinder sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Sensor output rate in Hz.  Typical values: 10 Hz (ultrasonic) to
        200 Hz (solid-state laser).
    min_range_m:
        Minimum measurable range (m).  Closer targets return ``no_hit_value``.
        Typical value: 0.05–0.3 m.
    max_range_m:
        Maximum measurable range (m).  Farther targets return ``no_hit_value``.
        Typical value: 5–40 m (laser), 2–4 m (ultrasonic).
    noise_floor_m:
        Fixed Gaussian noise sigma (m) independent of range.
        Represents electronic quantisation / timing jitter.
    noise_slope:
        Range-proportional noise coefficient (dimensionless, typically 0–0.02).
        ``accuracy_mode`` controls how floor and slope are combined.
    accuracy_mode:
        How ``noise_floor_m`` and ``noise_slope * range`` are combined:

        * ``"additive"`` *(default)* — ``σ = floor + slope·range``.
          Good for sensors with independent fixed + proportional error sources.
        * ``"max"`` — ``σ = max(floor, slope·range)``.
          Correct for sensors where the spec reads “±X cm **or** ±Y% (whichever
          is greater)” (e.g. Benewake TFmini, Garmin Lidar-Lite).
    dropout_prob:
        Per-step probability [0, 1] of a missed return (output → ``no_hit_value``).
        Models specular reflections, dark surfaces, or saturation by sunlight.
    resolution_m:
        Output quantisation step (m).  0 = disabled.  Typical: 0.001–0.01 m
        for laser sensors.
    no_hit_value:
        Value returned when the range is outside ``[min_range_m, max_range_m]``
        or a dropout occurs.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "rangefinder",
        update_rate_hz: float = 20.0,
        min_range_m: float = 0.05,
        max_range_m: float = 12.0,
        noise_floor_m: float = 0.02,
        noise_slope: float = 0.01,
        accuracy_mode: Literal["additive", "max"] = "additive",
        dropout_prob: float = 0.001,
        resolution_m: float = 0.001,
        no_hit_value: float = _DEFAULT_NO_HIT,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if min_range_m >= max_range_m:
            raise ValueError(f"min_range_m ({min_range_m}) must be less than max_range_m ({max_range_m})")
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.noise_floor_m = float(max(0.0, noise_floor_m))
        self.noise_slope = float(max(0.0, noise_slope))
        if accuracy_mode not in ("additive", "max"):
            raise ValueError(f"accuracy_mode must be 'additive' or 'max', got {accuracy_mode!r}")
        self.accuracy_mode: Literal["additive", "max"] = accuracy_mode
        self.dropout_prob = float(dropout_prob)
        self.resolution_m = float(max(0.0, resolution_m))
        self.no_hit_value = float(no_hit_value)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "RangefinderConfig") -> "RangefinderModel":
        """Construct from a :class:`~genesis.sensors.config.RangefinderConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "RangefinderConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.RangefinderConfig`."""
        from .config import RangefinderConfig

        return RangefinderConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            noise_floor_m=self.noise_floor_m,
            noise_slope=self.noise_slope,
            accuracy_mode=self.accuracy_mode,
            dropout_prob=self.dropout_prob,
            resolution_m=self.resolution_m,
            no_hit_value=self.no_hit_value,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> RangefinderObservation | dict[str, Any]:
        """
        Produce a realistic range measurement.

        Expected keys in *state*:

        - ``"range_m"`` (float) — true perpendicular distance to the nearest
          surface along the sensor beam axis (m).  When absent, returns ``{}``.
        """
        raw = state.get("range_m")
        if raw is None:
            self._last_obs = {}
            return self._last_obs

        true_range = float(raw)

        # Guard against NaN/Inf from upstream simulation blow-up
        if not math.isfinite(true_range):
            obs: RangefinderObservation = {"range_m": self.no_hit_value, "in_range": False}
            self._last_obs = obs
            self._mark_updated(sim_time)
            return obs

        # ------------------------------------------------------------------
        # Out-of-range → no hit
        # ------------------------------------------------------------------
        if true_range < self.min_range_m or true_range > self.max_range_m:
            obs: RangefinderObservation = {"range_m": self.no_hit_value, "in_range": False}
            self._last_obs = obs
            self._mark_updated(sim_time)
            return obs

        # ------------------------------------------------------------------
        # Random dropout
        # ------------------------------------------------------------------
        if self.dropout_prob > 0.0 and float(self._rng.random()) < self.dropout_prob:
            obs = {"range_m": self.no_hit_value, "in_range": False}
            self._last_obs = obs
            self._mark_updated(sim_time)
            return obs

        # ------------------------------------------------------------------
        # Range-dependent Gaussian noise
        # ------------------------------------------------------------------
        if self.accuracy_mode == "max":
            sigma = max(self.noise_floor_m, self.noise_slope * true_range)
        else:  # "additive" (default)
            sigma = self.noise_floor_m + self.noise_slope * true_range
        noisy = true_range + float(self._rng.normal(0.0, sigma))

        # Clamp to physical limits after noise addition.
        # If noise pushes the reading beyond the sensor range it is still a
        # valid return (the beam hit something); clamp rather than reject.
        noisy = max(self.min_range_m, min(noisy, self.max_range_m))

        # ------------------------------------------------------------------
        # Quantisation
        # ------------------------------------------------------------------
        if self.resolution_m > 0.0:
            noisy = round(noisy / self.resolution_m) * self.resolution_m

        obs = {"range_m": noisy, "in_range": True}
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["RangefinderModel"]
