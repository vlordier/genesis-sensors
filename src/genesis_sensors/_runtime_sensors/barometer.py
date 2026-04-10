"""
Barometric altimeter / pressure sensor model.

Converts the ideal world-frame altitude (``state["pos"][2]``, the z-component
of the ENU position vector) into a realistic barometric measurement using the
International Standard Atmosphere (ISA) troposphere model with:

* **ISA pressure conversion** — true altitude mapped to absolute pressure via
  the standard troposphere formula (valid 0 – 11 km MSL).
* **Gaussian white noise** on the altitude output.
* **Gauss-Markov bias drift** representing slow temperature-induced pressure
  sensor creep (dominates longer time-scales in real MEMS barometers).
* **Optional quantisation** to model the discrete output resolution of
  low-cost MEMS barometers.

Both noisy altitude (metres) and noisy absolute pressure (Pa) are reported so
callers can feed raw pressure into their own altitude filters if desired.

Usage
-----
::

    baro = BarometerModel(noise_sigma_m=0.3, bias_sigma_m=1.5)
    obs = baro.step(sim_time, {"pos": np.array([x, y, z])})
    print(obs["altitude_m"])   # noisy barometric altitude (m)
    print(obs["pressure_pa"])  # corresponding ISA absolute pressure (Pa)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ._gauss_markov import GaussMarkovProcess
from .base import BaseSensor
from .types import BarometerObservation

if TYPE_CHECKING:
    from .config import BarometerConfig

# ---------------------------------------------------------------------------
# ISA troposphere constants (ICAO / ISO 2533)
# ---------------------------------------------------------------------------

_ISA_T0: Final[float] = 288.15  # Sea-level temperature (K)
_ISA_P0: Final[float] = 101_325.0  # Sea-level pressure (Pa)
_ISA_L: Final[float] = 0.0065  # Temperature lapse rate (K/m)
_ISA_G: Final[float] = 9.80665  # Standard gravity (m/s²)
_ISA_M: Final[float] = 0.028_9644  # Molar mass of dry air (kg/mol)
_ISA_R: Final[float] = 8.314_46  # Universal gas constant (J/(mol·K))
# Pressure-altitude exponent: gM/(RL) ≈ 5.2561
_ISA_EXPONENT: Final[float] = (_ISA_G * _ISA_M) / (_ISA_R * _ISA_L)
# 1/exponent, pre-computed for the inverse conversion
_ISA_INV_EXPONENT: Final[float] = 1.0 / _ISA_EXPONENT

# Minimum Gauss-Markov time constant to avoid numerical underflow.
_MIN_BIAS_TAU_S: Final[float] = 1e-3


# ---------------------------------------------------------------------------
# ISA helper functions
# ---------------------------------------------------------------------------


def _altitude_to_pressure(altitude_m: float) -> float:
    """Convert geometric altitude (m MSL) to ISA absolute pressure (Pa).

    Uses the troposphere formula (valid 0–11 km).  Altitudes below sea level
    are accepted (negative *altitude_m* increases pressure above P₀).
    """
    t = _ISA_T0 - _ISA_L * altitude_m
    t = max(t, 1.0)  # guard against unphysical altitudes
    return _ISA_P0 * (t / _ISA_T0) ** _ISA_EXPONENT


def _pressure_to_altitude(pressure_pa: float) -> float:
    """Convert ISA absolute pressure (Pa) back to altitude (m MSL)."""
    pressure_pa = max(pressure_pa, 1.0)  # guard against non-positive pressure
    return (_ISA_T0 / _ISA_L) * (1.0 - (pressure_pa / _ISA_P0) ** _ISA_INV_EXPONENT)


class BarometerModel(BaseSensor):
    """
    ISA barometric altimeter / pressure sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Sensor output rate in Hz (50–200 Hz for typical MEMS barometers).
    noise_sigma_m:
        Standard deviation of Gaussian white noise on altitude (m).
        Corresponds to short-term measurement noise; typical MEMS value: 0.1–0.5 m.
    bias_tau_s:
        Gauss-Markov correlation time for the slow pressure-sensor drift (s).
        Represents temperature-induced creep; typical value: 100–600 s.
    bias_sigma_m:
        Steady-state standard deviation of the altitude bias random walk (m).
        Represents absolute accuracy after environmental drift; typical: 1–5 m.
    ground_alt_m:
        Reference ground altitude in metres MSL (used to shift the ENU world-
        frame z-axis to a meaningful MSL altitude when the world origin is not
        at sea level).
    resolution_m:
        Quantisation step for the altitude output in metres (0 = disabled).
        Models the discrete ADC / pressure resolution of MEMS sensors.
    qnh_pa:
        Sea-level reference pressure (Pa).  Default: ISA 101325.
        Models time-varying weather pressure by departing from ISA.
        Set to the actual QNH to simulate realistic baro-altitude errors.
    temp_cross_sensitivity_m_per_c:
        Temperature cross-sensitivity on altitude (m/°C, ref 25 °C).
        Models the fact that MEMS piezoresistive barometers have residual
        temperature dependence after factory calibration.  Typical: 0.01–0.1.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "barometer",
        update_rate_hz: float = 50.0,
        noise_sigma_m: float = 0.3,
        bias_tau_s: float = 300.0,
        bias_sigma_m: float = 1.5,
        ground_alt_m: float = 0.0,
        resolution_m: float = 0.0,
        qnh_pa: float = 101_325.0,
        temp_cross_sensitivity_m_per_c: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_m = float(noise_sigma_m)
        self.bias_tau_s = float(max(bias_tau_s, _MIN_BIAS_TAU_S))
        self.bias_sigma_m = float(bias_sigma_m)
        self.ground_alt_m = float(ground_alt_m)
        self.resolution_m = float(max(0.0, resolution_m))
        self.qnh_pa = float(qnh_pa)
        self.temp_cross_sensitivity_m_per_c = float(temp_cross_sensitivity_m_per_c)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed

        dt = 1.0 / self.update_rate_hz
        self._bias_process = GaussMarkovProcess(
            tau_s=self.bias_tau_s,
            sigma=self.bias_sigma_m,
            dt=dt,
            rng=self._rng,
        )
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "BarometerConfig") -> "BarometerModel":
        """Construct a :class:`BarometerModel` from a :class:`~genesis.sensors.config.BarometerConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "BarometerConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.BarometerConfig`."""
        from .config import BarometerConfig

        return BarometerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_m=self.noise_sigma_m,
            bias_tau_s=self.bias_tau_s,
            bias_sigma_m=self.bias_sigma_m,
            ground_alt_m=self.ground_alt_m,
            resolution_m=self.resolution_m,
            qnh_pa=self.qnh_pa,
            temp_cross_sensitivity_m_per_c=self.temp_cross_sensitivity_m_per_c,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bias_m(self) -> float:
        """Current altitude bias (m).  Useful for debugging or ground truth."""
        return float(self._bias_process.value)

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        # Draw a fresh steady-state bias each episode; this represents the
        # (unknown) initial pressure sensor offset at power-on.
        self._bias_process.reset(self.bias_sigma_m)
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> BarometerObservation | dict[str, Any]:
        """
        Produce a realistic barometric altitude + pressure measurement.

        Expected keys in *state*:
        - ``"pos"`` -- ``np.ndarray[3]`` world-frame ENU position (m).  The
          z-component (index 2) is used as the true AGL/MSL altitude.
        """
        pos = state.get("pos")
        if pos is None:
            self._last_obs = {}
            return self._last_obs

        true_alt_m = float(np.asarray(pos, dtype=np.float64)[2]) + self.ground_alt_m

        # QNH weather offset: difference from ISA gives an altitude error
        qnh_alt_error = _pressure_to_altitude(self.qnh_pa)

        # Temperature cross-sensitivity
        temp_c = float(state.get("temperature_c", 25.0))
        temp_error = self.temp_cross_sensitivity_m_per_c * (temp_c - 25.0)

        # Gauss-Markov bias drift (temperature-induced pressure creep)
        bias_m = float(self._bias_process.step())

        # White noise
        white_m = float(self._rng.normal(0.0, self.noise_sigma_m))

        noisy_alt = true_alt_m + bias_m + white_m + qnh_alt_error + temp_error

        # Quantise if a resolution is specified
        if self.resolution_m > 0.0:
            noisy_alt = round(noisy_alt / self.resolution_m) * self.resolution_m

        # Derive ISA pressure from the noisy altitude
        pressure_pa = _altitude_to_pressure(noisy_alt)

        obs: BarometerObservation = {
            "altitude_m": noisy_alt,
            "pressure_pa": pressure_pa,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["BarometerModel", "_altitude_to_pressure", "_pressure_to_altitude"]
