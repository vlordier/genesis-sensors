"""Multi-zone time-of-flight proximity array sensor model.

Models sensors like VL53L5CX (8×8 zones) that provide a structured
depth map from a compact solid-state emitter.  Used on manipulator
end-effectors, mobile robot bumpers, and drone obstacle avoidance.

State keys consumed
-------------------
``"tof_ranges_m"``
    Ground-truth per-zone range array, shape (rows, cols).
    Missing zones default to max_range_m (no hit).
``"tof_range_image"``
    Alternative key; same shape and semantics.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import ProximityToFArrayConfig


class ProximityToFArrayModel(BaseSensor):
    """Multi-zone time-of-flight proximity sensor.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    rows:
        Number of zone rows.
    cols:
        Number of zone columns.
    min_range_m:
        Minimum detectable range (m).
    max_range_m:
        Maximum detectable range (m).
    noise_sigma_base_m:
        Per-zone baseline 1-σ Gaussian range noise (m).
    noise_sigma_scale:
        Range-dependent noise coefficient: σ = σ_base + scale × z².
    fov_deg:
        Total field of view (degrees).
    crosstalk_fraction:
        Fractional optical crosstalk blended in from neighboring zones.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "proximity_tof",
        update_rate_hz: float = 15.0,
        rows: int = 8,
        cols: int = 8,
        min_range_m: float = 0.02,
        max_range_m: float = 4.0,
        noise_sigma_base_m: float = 0.005,
        noise_sigma_scale: float = 0.001,
        fov_deg: float = 63.0,
        crosstalk_fraction: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.rows = int(max(1, rows))
        self.cols = int(max(1, cols))
        self.min_range_m = float(max(0.0, min_range_m))
        self.max_range_m = float(max(self.min_range_m + 0.01, max_range_m))
        self.noise_sigma_base_m = float(max(0.0, noise_sigma_base_m))
        self.noise_sigma_scale = float(max(0.0, noise_sigma_scale))
        self.fov_deg = float(max(1.0, fov_deg))
        self.crosstalk_fraction = float(np.clip(crosstalk_fraction, 0.0, 1.0))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "ProximityToFArrayConfig") -> "ProximityToFArrayModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "ProximityToFArrayConfig":
        from .config import ProximityToFArrayConfig

        return ProximityToFArrayConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            rows=self.rows,
            cols=self.cols,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            noise_sigma_base_m=self.noise_sigma_base_m,
            noise_sigma_scale=self.noise_sigma_scale,
            fov_deg=self.fov_deg,
            crosstalk_fraction=self.crosstalk_fraction,
            seed=self._seed,
        )

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._mark_updated(sim_time)

        raw = state.get("tof_ranges_m", state.get("tof_range_image"))
        if raw is not None:
            ranges = np.asarray(raw, dtype=np.float32).reshape(self.rows, self.cols)
            valid = np.isfinite(ranges) & (ranges >= self.min_range_m) & (ranges < self.max_range_m)
        else:
            ranges = np.full((self.rows, self.cols), self.max_range_m, dtype=np.float32)
            valid = np.zeros((self.rows, self.cols), dtype=bool)

        # Range-dependent noise
        sigma = self.noise_sigma_base_m + self.noise_sigma_scale * ranges**2
        noisy = ranges + self._rng.normal(0.0, 1.0, size=ranges.shape).astype(np.float32) * sigma

        # Crosstalk between neighboring zones
        if self.crosstalk_fraction > 0.0:
            neighbor_mean = (
                np.roll(noisy, 1, axis=0)
                + np.roll(noisy, -1, axis=0)
                + np.roll(noisy, 1, axis=1)
                + np.roll(noisy, -1, axis=1)
            ) / 4.0
            noisy = (1.0 - self.crosstalk_fraction) * noisy + self.crosstalk_fraction * neighbor_mean

        # Clip to valid range
        max_output_range = float(np.nextafter(np.float32(self.max_range_m), np.float32(-np.inf)))
        noisy = np.clip(noisy, self.min_range_m, max_output_range)
        noisy[~valid] = 0.0

        self._last_obs = {
            "range_image": noisy,
            "valid_mask": valid,
            "min_range_m": float(np.min(noisy[valid])) if np.any(valid) else self.max_range_m,
            "n_valid_zones": int(np.sum(valid)),
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0
