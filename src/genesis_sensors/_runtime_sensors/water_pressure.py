"""Water pressure / depth gauge sensor model for underwater vehicles.

Converts true depth (from ``state["depth_m"]`` or ``state["pos"][2]``)
into a pressure reading with noise, quantisation, and temperature
cross-sensitivity.  Fundamental navigation sensor for AUV/ROV.

State keys consumed
-------------------
``"depth_m"``
    True depth below surface (m, positive downward).
    Falls back to ``-state["pos"][2]`` if absent.
``"water_salinity_ppt"``
    Water salinity (‰).  Default 35 (ocean).
``"temperature_c"``
    Ambient water temperature (°C).  Default 15.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import WaterPressureConfig


class WaterPressureModel(BaseSensor):
    """Submersible pressure sensor / depth gauge.

    Models a piezoresistive or ceramic pressure cell measuring absolute
    hydrostatic pressure and converting to depth using water density
    derived from temperature and salinity (UNESCO EOS-80 simplified).

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    noise_sigma_m:
        1-σ Gaussian noise on depth reading (m).
    bias_m:
        Constant depth offset / calibration error (m).
    resolution_m:
        Quantisation step (m).  0 = no quantisation.
    max_depth_m:
        Maximum rated depth (m).  Readings above are clipped.
    temp_coeff_m_per_c:
        Temperature cross-sensitivity on depth (m/°C relative to 15 °C).
    surface_pressure_pa:
        Assumed atmospheric pressure at surface (Pa).  101325 = 1 atm.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "water_pressure",
        update_rate_hz: float = 20.0,
        noise_sigma_m: float = 0.01,
        bias_m: float = 0.0,
        resolution_m: float = 0.0,
        max_depth_m: float = 300.0,
        temp_coeff_m_per_c: float = 0.0,
        surface_pressure_pa: float = 101_325.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_m = float(max(0.0, noise_sigma_m))
        self.bias_m = float(bias_m)
        self.resolution_m = float(max(0.0, resolution_m))
        self.max_depth_m = float(max(0.0, max_depth_m))
        self.temp_coeff_m_per_c = float(temp_coeff_m_per_c)
        self.surface_pressure_pa = float(surface_pressure_pa)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "WaterPressureConfig") -> "WaterPressureModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "WaterPressureConfig":
        from .config import WaterPressureConfig

        return WaterPressureConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_m=self.noise_sigma_m,
            bias_m=self.bias_m,
            resolution_m=self.resolution_m,
            max_depth_m=self.max_depth_m,
            temp_coeff_m_per_c=self.temp_coeff_m_per_c,
            surface_pressure_pa=self.surface_pressure_pa,
            seed=self._seed,
        )

    @staticmethod
    def _water_density(temp_c: float, salinity_ppt: float) -> float:
        """Simplified UNESCO EOS-80 seawater density (kg/m³)."""
        t = temp_c
        s = salinity_ppt
        rho_fw = 999.842594 + 6.793952e-2 * t - 9.09529e-3 * t**2 + 1.001685e-4 * t**3
        rho = rho_fw + s * (0.824493 - 4.0899e-3 * t + 7.6438e-5 * t**2)
        return float(rho)

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        # True depth
        if "depth_m" in state:
            true_depth = float(state["depth_m"])
        else:
            pos = state.get("pos")
            true_depth = -float(pos[2]) if pos is not None else 0.0

        true_depth = max(0.0, true_depth)

        temp_c = float(state.get("temperature_c", 15.0))
        salinity_ppt = float(state.get("water_salinity_ppt", 35.0))

        # Water density for pressure conversion
        rho = self._water_density(temp_c, salinity_ppt)
        g = 9.80665

        # Noisy depth
        depth = true_depth + self.bias_m
        if self.temp_coeff_m_per_c != 0.0:
            depth += self.temp_coeff_m_per_c * (temp_c - 15.0)
        if self.noise_sigma_m > 0.0:
            depth += float(self._rng.normal(0.0, self.noise_sigma_m))

        depth = max(0.0, min(depth, self.max_depth_m))

        # Quantisation
        if self.resolution_m > 0.0:
            depth = round(depth / self.resolution_m) * self.resolution_m

        # Noisy pressure from noisy depth
        noisy_pressure_pa = self.surface_pressure_pa + rho * g * depth

        self._last_obs = {
            "depth_m": depth,
            "pressure_pa": noisy_pressure_pa,
            "temperature_c": temp_c,
            "water_density_kg_m3": rho,
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0
