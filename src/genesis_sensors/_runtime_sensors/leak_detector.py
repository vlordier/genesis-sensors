"""Leak detector sensor model for underwater vehicles.

Simple binary/analog conductivity-based leak sensor.  Returns whether
water has been detected inside the hull and the estimated conductivity.

State keys consumed
-------------------
``"hull_breach"``
    Boolean indicating an active hull breach.
``"water_ingress_ml"``
    Volume of water inside the hull (mL).  0 = dry.
``"water_salinity_ppt"``
    Water salinity (‰).  Affects conductivity reading.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import LeakDetectorConfig


class LeakDetectorModel(BaseSensor):
    """Hull leak / water ingress detector.

    Models a conductivity-based leak sensor mounted inside the pressure
    hull of an AUV/ROV.  Triggers an alarm when water volume exceeds
    a configurable threshold.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    volume_threshold_ml:
        Water volume threshold for alarm activation (mL).
    conductivity_noise_sigma:
        1-σ Gaussian noise on conductivity reading (µS/cm).
    false_alarm_prob:
        Per-step probability of a spurious alarm (condensation etc.).
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "leak_detector",
        update_rate_hz: float = 1.0,
        volume_threshold_ml: float = 0.5,
        conductivity_noise_sigma: float = 10.0,
        false_alarm_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.volume_threshold_ml = float(max(0.0, volume_threshold_ml))
        self.conductivity_noise_sigma = float(max(0.0, conductivity_noise_sigma))
        self.false_alarm_prob = float(np.clip(false_alarm_prob, 0.0, 1.0))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "LeakDetectorConfig") -> "LeakDetectorModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "LeakDetectorConfig":
        from .config import LeakDetectorConfig

        return LeakDetectorConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            volume_threshold_ml=self.volume_threshold_ml,
            conductivity_noise_sigma=self.conductivity_noise_sigma,
            false_alarm_prob=self.false_alarm_prob,
            seed=self._seed,
        )

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        hull_breach = bool(state.get("hull_breach", False))
        water_ml = float(state.get("water_ingress_ml", 0.0))
        salinity = float(state.get("water_salinity_ppt", 35.0))

        # Conductivity: seawater ~50000 µS/cm, freshwater ~500, dry air ~0
        if water_ml > 0.0:
            base_conductivity = salinity * 1428.0  # rough scaling
            conductivity = max(0.0, base_conductivity + float(self._rng.normal(0.0, self.conductivity_noise_sigma)))
        else:
            conductivity = max(0.0, float(self._rng.normal(0.0, self.conductivity_noise_sigma * 0.01)))

        alarm = water_ml >= self.volume_threshold_ml or hull_breach
        if not alarm and self.false_alarm_prob > 0.0:
            alarm = float(self._rng.random()) < self.false_alarm_prob

        self._last_obs = {
            "alarm": alarm,
            "conductivity_us_cm": conductivity,
            "water_ingress_ml": water_ml,
            "hull_breach": hull_breach,
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0
