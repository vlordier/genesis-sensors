"""
Tactile array sensor model for robotic fingertips and skin patches.

Models a 2-D grid of pressure-sensitive taxels with per-taxel Gaussian
noise, a permanently-disabled dead-zone mask, and pressure saturation.
The sensor is suitable for fingertip tactile pads, palm arrays, and
wrist skins.

State keys consumed
-------------------
``"pressure_map"``
    Ideal contact pressure, shape ``(H, W)`` in Pascals.  Defaults to an
    all-zero array (no contact) when absent.

Observation keys
----------------
``"pressure_pa"``
    ``float32`` shape ``(H, W)`` — noisy, saturated per-taxel readings.
``"contact_mask"``
    ``bool`` shape ``(H, W)`` — True where pressure ≥ threshold.
``"cop_xy"``
    ``float32`` shape ``(2,)`` — centre-of-pressure in normalised image
    coordinates ``[0, 1]²`` (column, row).  Returns ``[0.5, 0.5]`` when
    no contact is detected.
``"total_force_n"``
    ``float32`` scalar — total normal force integrated over all contact
    taxels (N).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor
from .types import TactileArrayObservation

if TYPE_CHECKING:
    from .config import TactileArrayConfig


class TactileArraySensor(BaseSensor[TactileArrayObservation]):
    """
    2-D pressure taxel array (fingertip / skin tactile sensor).

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Output rate (Hz).
    resolution:
        ``(width, height)`` taxel grid dimensions.
    max_pressure_pa:
        Saturation pressure in Pa; readings are clipped to
        ``[0, max_pressure_pa]``.
    noise_sigma_pa:
        Per-taxel Gaussian noise 1-σ in Pa.
    contact_threshold_pa:
        Minimum pressure (Pa) for a taxel to be marked as "in contact".
    taxel_area_mm2:
        Physical area of each taxel in mm².  Used to convert pressure to
        force: ``total_force_n = Σ(pressure_pa × taxel_area_mm2 × 1e-6)``.
    dead_zone_fraction:
        Fraction of taxels permanently disabled at construction time
        (simulates wear / defective units).  In ``[0.0, 1.0]``.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "tactile_array",
        update_rate_hz: float = 200.0,
        resolution: tuple[int, int] = (4, 4),
        max_pressure_pa: float = 500_000.0,
        noise_sigma_pa: float = 1000.0,
        contact_threshold_pa: float = 5000.0,
        taxel_area_mm2: float = 4.0,
        dead_zone_fraction: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.width = int(resolution[0])
        self.height = int(resolution[1])
        self.resolution = (self.width, self.height)
        self._shape_hw = (self.height, self.width)
        self.max_pressure_pa = float(max_pressure_pa)
        self.noise_sigma_pa = float(noise_sigma_pa)
        self.contact_threshold_pa = float(contact_threshold_pa)
        self.taxel_area_mm2 = float(taxel_area_mm2)
        self.dead_zone_fraction = float(dead_zone_fraction)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._default_cop = np.array([0.5, 0.5], dtype=np.float32)
        self._last_obs: TactileArrayObservation | dict[str, Any] = {}
        self._dead_mask = self._build_dead_mask()

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "TactileArrayConfig") -> "TactileArraySensor":
        """Construct from a :class:`~genesis.sensors.config.TactileArrayConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "TactileArrayConfig":
        """Serialise parameters back to a :class:`~genesis.sensors.config.TactileArrayConfig`."""
        from .config import TactileArrayConfig

        return TactileArrayConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            resolution=self.resolution,
            max_pressure_pa=self.max_pressure_pa,
            noise_sigma_pa=self.noise_sigma_pa,
            contact_threshold_pa=self.contact_threshold_pa,
            taxel_area_mm2=self.taxel_area_mm2,
            dead_zone_fraction=self.dead_zone_fraction,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_dead_mask(self) -> np.ndarray:
        """Create the fixed per-taxel dead-zone mask used for all measurements."""
        n_taxels = self.width * self.height
        flat_mask = np.zeros(n_taxels, dtype=bool)
        if self.dead_zone_fraction > 0.0:
            n_dead = max(0, int(round(self.dead_zone_fraction * n_taxels)))
            dead_idx = self._rng.choice(n_taxels, size=n_dead, replace=False)
            flat_mask[dead_idx] = True
        return flat_mask.reshape(self._shape_hw)

    def _coerce_pressure_map(self, state: Mapping[str, Any]) -> np.ndarray:
        """Read the ideal pressure map from the shared state, falling back to zeros."""
        ideal = np.asarray(state.get("pressure_map", np.zeros(self._shape_hw, dtype=np.float32)), dtype=np.float32)
        if ideal.shape != self._shape_hw:
            return np.zeros(self._shape_hw, dtype=np.float32)
        return ideal

    def _compute_cop(self, pressure: np.ndarray, contact_mask: np.ndarray) -> np.ndarray:
        """Compute the normalised center-of-pressure in image coordinates."""
        total_weight = float(pressure[contact_mask].sum())
        if total_weight <= 0.0:
            return self._default_cop.copy()

        ys, xs = np.where(contact_mask)
        p_vals = pressure[contact_mask]
        cop_x = float((xs.astype(np.float32) * p_vals).sum() / total_weight) / max(self.width - 1, 1)
        cop_y = float((ys.astype(np.float32) * p_vals).sum() / total_weight) / max(self.height - 1, 1)
        return np.array([cop_x, cop_y], dtype=np.float32)

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Clear the cached observation and scheduler timestamp."""
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: Mapping[str, Any]) -> TactileArrayObservation:
        """Process the ideal pressure map through noise and dead-zone models."""
        ideal = self._coerce_pressure_map(state)

        if self.noise_sigma_pa > 0.0:
            pressure = ideal + self._rng.normal(0.0, self.noise_sigma_pa, size=self._shape_hw).astype(np.float32)
        else:
            pressure = ideal.copy()

        pressure[self._dead_mask] = 0.0
        pressure = np.clip(pressure, 0.0, self.max_pressure_pa).astype(np.float32)
        contact_mask = pressure >= self.contact_threshold_pa
        cop_xy = self._compute_cop(pressure, contact_mask)

        taxel_area_m2 = self.taxel_area_mm2 * 1e-6
        obs: TactileArrayObservation = {
            "pressure_pa": pressure,
            "contact_mask": contact_mask,
            "cop_xy": cop_xy,
            "total_force_n": float(pressure.sum()) * taxel_area_m2,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> TactileArrayObservation | dict[str, Any]:
        """Return the most recent observation without triggering a new step."""
        return self._last_obs

    def __repr__(self) -> str:
        W, H = self.resolution
        return (
            f"TactileArraySensor(name={self.name!r}, resolution={W}×{H}, "
            f"rate={self.update_rate_hz} Hz, noise={self.noise_sigma_pa:.0f} Pa, "
            f"dead={self.dead_zone_fraction:.1%})"
        )


__all__ = ["TactileArraySensor"]
