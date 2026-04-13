"""
GNSS / GPS sensor model.

Converts a perfect world-frame position and velocity (from Genesis) into
a realistic GNSS measurement with:

* Gaussian white noise on position and velocity.
* First-order Gauss-Markov bias drift (random walk) on position.
* Multipath error correlated with nearby obstacle geometry.
* Satellite constellation availability mask based on altitude and
  urban-canyon heuristics.
* Configurable fix quality modes (no fix / autonomous / RTK).

Usage
-----
::

    gnss = GNSSModel(update_rate_hz=10.0, noise_m=1.5)
    obs = gnss.step(sim_time, {
        "pos":  np.array([x, y, z]),   # world-frame metres (ENU: x=East, y=North)
        "vel":  np.array([vx, vy, vz]),
    })
    print(obs["pos_llh"])  # latitude, longitude, height (degrees, metres)
"""

from __future__ import annotations

import math
from enum import IntEnum
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ._gauss_markov import GaussMarkovProcess
from .base import BaseSensor
from .types import Float64Array, GnssObservation, JammerZone

if TYPE_CHECKING:
    from .config import GNSSConfig

# Physical constants
_EARTH_RADIUS_M: Final[float] = 6_378_137.0
# Minimum metres-per-degree-longitude to guard against division by zero at the poles
# (cos(90°) ≈ 0 → _m_per_deg_lon → 0 → exploding lon values in flat-earth conversion).
_MIN_M_PER_DEG_LON_AT_POLE: Final[float] = 1.0
# Fallback HDOP reported when there is no fix or inside a jammer zone.
_HDOP_NO_FIX: Final[float] = 99.9
# Nominal satellites visible at zero obstruction.
_BASE_SATELLITES: Final[int] = 8
# Number of satellites lost per unit obstruction fraction.
_SAT_LOSS_PER_OBSTRUCTION: Final[int] = 6
# Maximum satellite count (clamp upper bound).
_MAX_SATELLITES: Final[int] = 12
# HDOP slope as a function of obstruction fraction.
_HDOP_OBSTRUCTION_SLOPE: Final[float] = 2.5
# Minimum HDOP (ideal open sky).
_HDOP_MIN: Final[float] = 1.0
# Maximum HDOP (very degraded).
_HDOP_MAX: Final[float] = 10.0
# VDOP to HDOP ratio for a standard single-constellation receiver.
# Vertical geometry is always worse than horizontal; empirical ratio ≈1.5.
_VDOP_HDOP_RATIO: Final[float] = 1.5
# Position noise below this threshold is treated as RTK-grade accuracy.
# Represents the 0.012 m 1-σ of a working RTK solution.
_RTK_NOISE_THRESHOLD_M: Final[float] = 0.05


class GnssFixQuality(IntEnum):
    """NMEA GGA fix-quality indicator values."""

    NO_FIX = 0
    AUTONOMOUS = 1
    RTK = 4


class GNSSModel(BaseSensor):
    """
    Realistic GNSS / GPS sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Output rate in Hz (typically 1-10 Hz for GPS, up to ~50 Hz for RTK).
    noise_m:
        1-sigma Gaussian position noise in metres (horizontal and vertical).
    vel_noise_ms:
        1-sigma Gaussian velocity noise in m/s.
    bias_tau_s:
        Time constant for the first-order Gauss-Markov position bias (s).
    bias_sigma_m:
        Steady-state standard deviation of the bias random walk (m).
    multipath_sigma_m:
        Standard deviation of multipath error (m).  Scaled by the
        obstruction fraction provided in ``state["obstruction"]``.
    min_fix_altitude_m:
        Altitude (metres above ground) below which fix quality degrades.
    jammer_zones:
        List of ``(centre_xyz, radius_m)`` tuples; if the drone is inside
        any zone the output ``fix_quality`` is set to ``GnssFixQuality.NO_FIX``.
    origin_llh:
        ``(lat_deg, lon_deg, alt_m)`` of the simulation world origin.
        Used to convert XYZ to lat/lon/height.  The coordinate convention
        is ENU: world-frame x = East, y = North, z = Up.
    ionospheric_delay_m:
        Zenith ionospheric delay in metres.  Scaled by an obliquity factor
        ``1 / cos(elev)`` where ``elev`` is derived from the satellite
        geometry (a heuristic elevation of 60° is assumed when no explicit
        satellite model is available).  Typical single-frequency L1 delay:
        1–10 m at zenith.  ``0`` = disabled.
    tropospheric_delay_m:
        Zenith tropospheric (wet + dry) delay in metres.  Same obliquity
        scaling as ionospheric delay.  Typical dry-component delay: ~2.3 m;
        wet component: 0.1–0.5 m.  ``0`` = disabled.
    seed:
        Optional seed for the random-number generator (reproducibility).
    """

    def __init__(
        self,
        name: str = "gnss",
        update_rate_hz: float = 10.0,
        noise_m: float = 1.5,
        vel_noise_ms: float = 0.05,
        bias_tau_s: float = 60.0,
        bias_sigma_m: float = 0.5,
        multipath_sigma_m: float = 1.0,
        min_fix_altitude_m: float = 0.5,
        jammer_zones: list[JammerZone] | None = None,
        origin_llh: tuple[float, float, float] = (0.0, 0.0, 0.0),
        ionospheric_delay_m: float = 0.0,
        tropospheric_delay_m: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_m = float(noise_m)
        self.vel_noise_ms = float(vel_noise_ms)
        self.bias_tau_s = float(max(bias_tau_s, 1e-3))
        self.bias_sigma_m = float(bias_sigma_m)
        self.multipath_sigma_m = float(multipath_sigma_m)
        self.min_fix_altitude_m = float(min_fix_altitude_m)
        self.jammer_zones: list[JammerZone] = jammer_zones or []
        self.origin_llh = tuple(origin_llh)
        self.ionospheric_delay_m = float(max(ionospheric_delay_m, 0.0))
        self.tropospheric_delay_m = float(max(tropospheric_delay_m, 0.0))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed

        # Earth radius and degrees-per-metre factors (flat-earth ENU approximation).
        # Clamp m_per_deg_lon with a minimum value to prevent division by zero
        # when the simulation origin is at or very near the geographic poles.
        self._m_per_deg_lat = math.pi * _EARTH_RADIUS_M / 180.0
        self._m_per_deg_lon = max(
            self._m_per_deg_lat * math.cos(math.radians(self.origin_llh[0])),
            _MIN_M_PER_DEG_LON_AT_POLE,
        )

        # ------------------------------------------------------------------
        # Pre-computed Gauss-Markov coefficients (constant for fixed update rate)
        # ------------------------------------------------------------------
        self._dt: float = 1.0 / self.update_rate_hz
        self._bias_process = GaussMarkovProcess(
            tau_s=self.bias_tau_s,
            sigma=self.bias_sigma_m,
            dt=self._dt,
            rng=self._rng,
            shape=(3,),
        )

        # Pre-convert jammer zone centres to numpy arrays to avoid repeated
        # np.asarray() calls inside the hot step() loop.
        self._jammer_centres: list[Float64Array] = [
            np.asarray(centre, dtype=np.float64) for centre, _ in self.jammer_zones
        ]
        self._jammer_radii: list[float] = [float(r) for _, r in self.jammer_zones]

        # Initialise from steady-state distribution so position bias is
        # physically realistic from t=0 instead of ramping up from zero.
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "GNSSConfig") -> "GNSSModel":
        """Construct a :class:`GNSSModel` from a :class:`~genesis.sensors.config.GNSSConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "GNSSConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.GNSSConfig`."""
        from .config import GNSSConfig

        return GNSSConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_m=self.noise_m,
            vel_noise_ms=self.vel_noise_ms,
            bias_tau_s=self.bias_tau_s,
            bias_sigma_m=self.bias_sigma_m,
            multipath_sigma_m=self.multipath_sigma_m,
            min_fix_altitude_m=self.min_fix_altitude_m,
            jammer_zones=[
                (centre.tolist(), radius) for centre, radius in zip(self._jammer_centres, self._jammer_radii)
            ],
            origin_llh=self.origin_llh,
            ionospheric_delay_m=self.ionospheric_delay_m,
            tropospheric_delay_m=self.tropospheric_delay_m,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def bias(self) -> Float64Array:
        """Current position bias vector (metres, world frame)."""
        v = self._bias_process.value
        return np.asarray(v, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._bias_process.reset(self.bias_sigma_m)
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> GnssObservation | dict[str, Any]:
        """
        Produce a realistic GNSS measurement.

        Expected keys in *state*:
        - ``"pos"`` -- ``np.ndarray[3]`` world-frame position in metres (ENU).
        - ``"vel"`` -- ``np.ndarray[3]`` world-frame velocity in m/s.
        - ``"obstruction"`` *(optional)* -- float 0-1 representing the
          fraction of sky hemisphere blocked by obstacles (used to scale
          multipath error).
        """
        true_pos: Float64Array = np.asarray(state.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        true_vel: Float64Array = np.asarray(state.get("vel", [0.0, 0.0, 0.0]), dtype=np.float64)
        if true_pos.shape != (3,):
            raise ValueError(f"state['pos'] must be a 3-element array, got shape {true_pos.shape}")
        if true_vel.shape != (3,):
            raise ValueError(f"state['vel'] must be a 3-element array, got shape {true_vel.shape}")
        obstruction = float(state.get("obstruction", 0.0))

        # Check jammer zones using pre-converted centre arrays
        for centre_arr, radius in zip(self._jammer_centres, self._jammer_radii):
            if float(np.linalg.norm(true_pos - centre_arr)) <= radius:
                # Return stale last-known position/velocity when jammed so that
                # ground truth is never exposed through the sensor output.
                stale_pos = (
                    np.asarray(self._last_obs["pos"], dtype=np.float64)
                    if self._last_obs and "pos" in self._last_obs
                    else np.zeros(3, dtype=np.float64)
                )
                stale_vel = (
                    np.asarray(self._last_obs["vel"], dtype=np.float64)
                    if self._last_obs and "vel" in self._last_obs
                    else np.zeros(3, dtype=np.float64)
                )
                # jammed pos_llh: convert the stale ENU position to LLH so users
                # can correlate pos and pos_llh without seeing ground truth.
                result: GnssObservation = {
                    "pos": stale_pos,
                    "vel": stale_vel,
                    "pos_llh": self._enu_to_llh(stale_pos),
                    "fix_quality": GnssFixQuality.NO_FIX,
                    "n_satellites": 0,
                    "hdop": _HDOP_NO_FIX,
                    "vdop": _HDOP_NO_FIX,
                }
                self._last_obs = result
                self._mark_updated(sim_time)
                return result

        # Update bias random walk (Gauss-Markov) using the shared process
        bias = self._bias_process.step()

        # Position error: white noise + Gauss-Markov bias + multipath
        white: Float64Array = self._rng.normal(0.0, self.noise_m, 3)
        # Guard against zero-sigma when obstruction=0 to keep semantics clear
        if obstruction > 0.0:
            multipath: Float64Array = self._rng.normal(0.0, self.multipath_sigma_m * obstruction, 3)
        else:
            multipath = np.zeros(3, dtype=np.float64)

        # Atmospheric propagation delays (ionospheric + tropospheric).
        # Modelled as a zenith delay scaled by 1/cos(elev); we use a
        # heuristic average satellite elevation of 60° (obliquity ≈ 2.0)
        # and spread the delay randomly across XYZ with a slight vertical
        # bias (ratio 1:1:1.5) matching real GNSS error distributions.
        atmo_delay = np.zeros(3, dtype=np.float64)
        total_zenith_delay = self.ionospheric_delay_m + self.tropospheric_delay_m
        if total_zenith_delay > 0.0:
            obliquity = 2.0  # 1/cos(60°)
            slant_delay = total_zenith_delay * obliquity
            # Distribute with random sign per axis; vertical component larger
            atmo_delay = self._rng.normal(0.0, slant_delay * 0.3, 3)
            atmo_delay[2] += slant_delay * 0.5  # systematic vertical bias

        noisy_pos: Float64Array = true_pos + bias + white + multipath + atmo_delay

        # Velocity error
        noisy_vel: Float64Array = true_vel + self._rng.normal(0.0, self.vel_noise_ms, 3)

        # Fix quality and satellite count
        # Use noisy altitude (not ground truth) to decide fix quality.
        noisy_alt_fix_check = float(noisy_pos[2])
        fix_quality = (
            GnssFixQuality.AUTONOMOUS if noisy_alt_fix_check >= self.min_fix_altitude_m else GnssFixQuality.NO_FIX
        )
        n_sat = int(
            min(
                max(_BASE_SATELLITES - round(obstruction * _SAT_LOSS_PER_OBSTRUCTION), 0),
                _MAX_SATELLITES,
            )
        )
        hdop = min(max(_HDOP_MIN + obstruction * _HDOP_OBSTRUCTION_SLOPE, _HDOP_MIN), _HDOP_MAX)
        vdop = min(hdop * _VDOP_HDOP_RATIO, _HDOP_MAX)
        # GPS requires at least 4 satellites for a valid 3-D position fix.
        if n_sat < 4:
            fix_quality = GnssFixQuality.NO_FIX
        # Upgrade to RTK when the sensor is configured with RTK-grade accuracy
        # and conditions are sufficient for a differential solution.
        # RTK requires adequate satellite geometry (n_sat ≥ 5) and clear sky.
        if (
            fix_quality == GnssFixQuality.AUTONOMOUS
            and self.noise_m <= _RTK_NOISE_THRESHOLD_M
            and n_sat >= 5
            and obstruction < 0.5
        ):
            fix_quality = GnssFixQuality.RTK

        obs: GnssObservation = {
            "pos": noisy_pos,
            "vel": noisy_vel,
            "pos_llh": self._enu_to_llh(noisy_pos),
            "fix_quality": fix_quality,
            "n_satellites": n_sat,
            "hdop": hdop,
            "vdop": vdop,
            "pdop": math.sqrt(hdop**2 + vdop**2),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    def _enu_to_llh(self, enu_pos: Float64Array) -> Float64Array:
        """Convert ENU XYZ (metres) to LLH (lat_deg, lon_deg, alt_m)."""
        lat = self.origin_llh[0] + enu_pos[1] / self._m_per_deg_lat
        lon = self.origin_llh[1] + enu_pos[0] / self._m_per_deg_lon
        alt_m = self.origin_llh[2] + enu_pos[2]
        return np.array([lat, lon, alt_m], dtype=np.float64)


__all__ = ["GNSSModel", "GnssFixQuality"]
