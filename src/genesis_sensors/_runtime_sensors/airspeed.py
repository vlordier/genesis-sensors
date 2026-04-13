"""
Airspeed / pitot-tube differential pressure sensor model.

Simulates a Prandtl pitot tube as used on fixed-wing UAVs and high-speed
multi-rotors.  The model converts the true airspeed (TAS) — derived from
the vehicle velocity relative to the local wind — into an indicated airspeed
(IAS) measurement with realistic sensor characteristics:

* **Bernoulli pressure conversion** — TAS → dynamic pressure ``q = ½ρV²``,
  then back to indicated airspeed corrected to sea-level ISA density.
* **ISA air density correction** — density decreases with altitude, so an
  uncorrected tube over-reads IAS at altitude.  Callers interested in TAS
  should apply the standard density-altitude correction themselves.
* **White-noise** on the differential pressure (converted back to m/s units).
* **Slow Gauss-Markov bias** representing ice accretion, contamination,
  or temperature-induced zero-offset drift.
* **Minimum detectable speed** (port-pressure dead-band below ~2–3 m/s typical).
* **Tube blockage** simulation — rare random events that persistently clamp
  output to zero until the sensor is reset.

Usage
-----
::

    airspeed = AirspeedModel(noise_sigma_ms=0.3, min_detectable_ms=2.0)
    obs = airspeed.step(sim_time, {
        "vel":  np.array([vx, vy, vz]),   # body or world-frame velocity (m/s)
        "wind": np.array([wx, wy, wz]),   # optional wind vector (same frame, m/s)
        "pos":  np.array([x,  y,  z]),    # optional ENU position for density correction
    })
    print(obs["airspeed_ms"])          # indicated airspeed (m/s)
    print(obs["dynamic_pressure_pa"])  # raw differential pressure (Pa)

If ``state["airspeed_ms"]`` is present it is used directly as the true
airspeed (bypassing the vel/wind computation), which is convenient for
testing.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ._gauss_markov import GaussMarkovProcess
from .base import BaseSensor
from .types import AirspeedObservation

if TYPE_CHECKING:
    from .config import AirspeedConfig

# ---------------------------------------------------------------------------
# ISA troposphere constants (shared with barometer — redefined here to avoid
# a circular dependency on barometer.py)
# ---------------------------------------------------------------------------

_ISA_T0: Final[float] = 288.15  # Sea-level temperature (K)
_ISA_P0: Final[float] = 101_325.0  # Sea-level pressure (Pa)
_ISA_RHO0: Final[float] = 1.225  # Sea-level air density (kg/m³)
_ISA_L: Final[float] = 0.0065  # Lapse rate (K/m)
_ISA_G: Final[float] = 9.80665  # Standard gravity (m/s²)
_ISA_M: Final[float] = 0.028_9644  # Molar mass of dry air (kg/mol)
_ISA_R: Final[float] = 8.314_46  # Gas constant (J/(mol·K))
_ISA_EXPONENT: Final[float] = (_ISA_G * _ISA_M) / (_ISA_R * _ISA_L)  # ≈ 5.2561

# Minimum admissible air density to guard against extreme altitudes.
_MIN_RHO_KG_M3: Final[float] = 0.01
# Minimum Gauss-Markov time constant.
_MIN_BIAS_TAU_S: Final[float] = 1e-3


# ISA stratosphere (11 km ≤ h < 20 km): isothermal at 216.65 K.
_ISA_H_TROP: Final[float] = 11_000.0  # tropopause height (m)
_ISA_T_STRAT: Final[float] = 216.65  # stratosphere temperature (K)
# Pressure at the tropopause (Pa), pre-computed from the troposphere formula.
_ISA_P_TROP: Final[float] = 101_325.0 * (_ISA_T_STRAT / _ISA_T0) ** (9.80665 / (287.05 * _ISA_L))
# Density at the tropopause (kg/m³).
_ISA_RHO_TROP: Final[float] = _ISA_RHO0 * (_ISA_T_STRAT / _ISA_T0) ** (_ISA_EXPONENT - 1.0)


def _isa_density(altitude_m: float) -> float:
    """Return ISA air density (kg/m³) at the given geometric altitude (m MSL).

    Covers the troposphere (0–11 km) and lower stratosphere (11–20 km);
    altitudes below 0 are clamped to sea level.
    """
    h = max(0.0, altitude_m)  # clamp below MSL to sea level
    if h <= _ISA_H_TROP:
        t = _ISA_T0 - _ISA_L * h
        return _ISA_RHO0 * (t / _ISA_T0) ** (_ISA_EXPONENT - 1.0)
    # Stratosphere: isothermal exponential decay, T = 216.65 K
    # rho = rho_trop * exp(-g/(R*T) * (h - h_trop))
    dh = h - _ISA_H_TROP
    return _ISA_RHO_TROP * math.exp(-9.80665 / (287.05 * _ISA_T_STRAT) * dh)


class AirspeedModel(BaseSensor):
    """
    Pitot-tube airspeed / differential pressure sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Sensor output rate in Hz (50–100 Hz for typical flight-controller use).
    noise_sigma_ms:
        Standard deviation of Gaussian white noise on the airspeed output (m/s).
        Noise is applied in the pressure domain and converted back to m/s.
        Typical MEMS pitot-static range: 0.2–0.5 m/s.
    bias_tau_s:
        Gauss-Markov correlation time for the slow pressure-zero drift (s).
        Represents temperature / icing offset; typical: 100–600 s.
    bias_sigma_ms:
        Steady-state standard deviation of the bias random walk (m/s).
        Typical value: 0.5–1.5 m/s for uncalibrated consumer tubes.
    min_detectable_ms:
        Dead-band around zero: airspeeds below this threshold are reported as 0.
        Typical value: 1–3 m/s.
    max_speed_ms:
        Tube saturation speed (m/s).  Exceeding this clips the measurement.
        Typical industrial range: 50–300 m/s.
    tube_blockage_prob:
        Per-step probability [0, 1] that the tube becomes permanently blocked
        (until reset).  Once triggered the output is clamped to 0.
        Models ice build-up or debris obstructing the pitot port.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "airspeed",
        update_rate_hz: float = 50.0,
        noise_sigma_ms: float = 0.3,
        bias_tau_s: float = 300.0,
        bias_sigma_ms: float = 0.8,
        min_detectable_ms: float = 2.0,
        max_speed_ms: float = 200.0,
        tube_blockage_prob: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_ms = float(noise_sigma_ms)
        self.bias_tau_s = float(max(bias_tau_s, _MIN_BIAS_TAU_S))
        self.bias_sigma_ms = float(bias_sigma_ms)
        self.min_detectable_ms = float(min_detectable_ms)
        self.max_speed_ms = float(max_speed_ms)
        self.tube_blockage_prob = float(tube_blockage_prob)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._tube_blocked: bool = False

        dt = 1.0 / self.update_rate_hz
        self._bias_process = GaussMarkovProcess(
            tau_s=self.bias_tau_s,
            sigma=self.bias_sigma_ms,
            dt=dt,
            rng=self._rng,
        )
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "AirspeedConfig") -> "AirspeedModel":
        """Construct from an :class:`~genesis.sensors.config.AirspeedConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "AirspeedConfig":
        """Return the current parameters as an :class:`~genesis.sensors.config.AirspeedConfig`."""
        from .config import AirspeedConfig

        return AirspeedConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_ms=self.noise_sigma_ms,
            bias_tau_s=self.bias_tau_s,
            bias_sigma_ms=self.bias_sigma_ms,
            min_detectable_ms=self.min_detectable_ms,
            max_speed_ms=self.max_speed_ms,
            tube_blockage_prob=self.tube_blockage_prob,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bias_ms(self) -> float:
        """Current airspeed bias (m/s).  Useful for debugging."""
        return float(self._bias_process.value)

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._bias_process.reset(self.bias_sigma_ms)
        self._tube_blocked = False
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> AirspeedObservation | dict[str, Any]:
        """
        Produce a realistic airspeed measurement.

        Expected keys in *state*:

        - ``"airspeed_ms"`` (float) — true airspeed directly.  **Preferred**
          for testing or when the caller has already computed TAS.
        - **or** ``"vel"`` (np.ndarray, shape (3,)) — vehicle velocity in any
          consistent frame (m/s).  Optionally combined with:
          - ``"wind"`` (np.ndarray, shape (3,)) — ambient wind vector in the
            same frame (m/s); default ``[0, 0, 0]``.
          - ``"pos"`` (np.ndarray, shape (3,)) — ENU position whose z-component
            gives altitude (m MSL) for ISA density correction; default: use
            sea-level density.
        """
        # ------------------------------------------------------------------
        # 1. Compute true airspeed
        # ------------------------------------------------------------------
        tas_ms: float
        if "airspeed_ms" in state:
            tas_ms = abs(float(state["airspeed_ms"]))  # magnitude only; direction not defined
        elif "vel" in state:
            vel = np.asarray(state["vel"], dtype=np.float64)
            wind = np.asarray(state.get("wind", np.zeros(3)), dtype=np.float64)
            relative_vel = vel - wind
            tas_ms = float(np.linalg.norm(relative_vel))
        else:
            self._last_obs = {}
            self._mark_updated(sim_time)
            return self._last_obs

        # Guard against NaN / Inf from upstream (e.g. simulation blow-up)
        if not math.isfinite(tas_ms):
            tas_ms = 0.0

        # ------------------------------------------------------------------
        # 2. ISA density correction: convert TAS → differential pressure → IAS
        #    q_dyn = 0.5 * rho_actual * TAS²  (Pa)
        #    IAS   = sqrt(2 * q_dyn / rho_SL)
        # ------------------------------------------------------------------
        altitude_m = 0.0
        if "pos" in state:
            altitude_m = float(np.asarray(state["pos"], dtype=np.float64)[2])
        rho = _isa_density(altitude_m)
        rho_clamped = max(rho, _MIN_RHO_KG_M3)

        q_dynamic_pa = 0.5 * rho_clamped * tas_ms**2  # dynamic pressure (Pa)

        # Indicated airspeed: what an uncorrected pitot tube measures
        # (referenced to sea-level density)
        ias_ms = math.sqrt(max(0.0, 2.0 * q_dynamic_pa / _ISA_RHO0))

        # ------------------------------------------------------------------
        # 3. Bias drift (Gauss-Markov) — always advances regardless of blockage
        # ------------------------------------------------------------------
        bias_ms = float(self._bias_process.step())

        # ------------------------------------------------------------------
        # 4. Tube blockage — persistent: once triggered, stays blocked until
        #    reset().  Models ice build-up or debris obstructing the pitot port.
        # ------------------------------------------------------------------
        if not self._tube_blocked and self.tube_blockage_prob > 0.0:
            if float(self._rng.random()) < self.tube_blockage_prob:
                self._tube_blocked = True

        if self._tube_blocked:
            obs: AirspeedObservation = {"airspeed_ms": 0.0, "dynamic_pressure_pa": 0.0}
            self._last_obs = obs
            self._mark_updated(sim_time)
            return obs

        # ------------------------------------------------------------------
        # 5. Gaussian white noise (applied in airspeed domain)
        # ------------------------------------------------------------------
        white_ms = float(self._rng.normal(0.0, self.noise_sigma_ms))

        noisy_ias = ias_ms + bias_ms + white_ms

        # ------------------------------------------------------------------
        # 6. Dead-band and saturation clamps
        # ------------------------------------------------------------------
        if noisy_ias < self.min_detectable_ms:
            noisy_ias = 0.0
        noisy_ias = min(noisy_ias, self.max_speed_ms)
        noisy_ias = max(0.0, noisy_ias)

        # Convert final IAS back to differential pressure for the raw output
        noisy_q = 0.5 * _ISA_RHO0 * noisy_ias**2

        obs = {"airspeed_ms": noisy_ias, "dynamic_pressure_pa": noisy_q}
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["AirspeedModel"]
