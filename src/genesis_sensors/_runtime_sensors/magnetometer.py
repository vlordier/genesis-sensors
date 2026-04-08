"""
3-axis magnetometer / digital compass sensor model.

Simulates a solid-state magnetometer measuring Earth's magnetic field in the
sensor (body) frame.  The error model covers the dominant real-world effects:

* **Hard-iron distortion** — constant per-axis magnetic bias caused by
  permanent magnets or steady DC currents near the sensor (motor ESCs, battery).
  Represented as an additive offset vector in body frame (µT).
* **Soft-iron distortion** — induced magnetisation from ferromagnetic material;
  modelled as a per-axis scale factor applied before the hard-iron offset.
* **Gaussian white noise** on each axis.

The sensor needs a rotation to express the world-frame Earth field in the
body frame.  It accepts either:

* ``state["attitude_mat"]`` — a 3×3 body-to-world rotation matrix ``R_bw``
  (e.g., from ``genesis.entity.get_quat().as_matrix()``).
* ``state["attitude_quat"]`` — a 4-element body-to-world quaternion
  ``(w, x, y, z)`` (note: **not** the Genesis ``[x, y, z, w]`` convention;
  callers must reorder accordingly).

If neither is present the world-frame Earth field is returned unchanged (useful
for stationary debugging / unit tests).

Usage
-----
::

    mag = MagnetometerModel(noise_sigma_ut=0.5, declination_deg=-3.0)
    obs = mag.step(sim_time, {
        "attitude_mat": R_bw,       # (3, 3) body-to-world rotation
    })
    print(obs["mag_field_ut"])      # (3,) noisy field in body frame (µT)
    print(obs["field_norm_ut"])     # scalar — should be near 50 µT

Reference
---------
World Magnetic Model (WMM) 2020: https://www.ngdc.noaa.gov/geomag/WMM/
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import Float64Array, MagnetometerObservation

if TYPE_CHECKING:
    from .config import MagnetometerConfig

# Default Earth field magnitude (µT) representative of mid-latitude surface.
# WMM-2020 reports ~48–52 µT across most populated land areas.
_EARTH_FIELD_DEFAULT_UT: Final[float] = 50.0
# Minimum field magnitude guard: if computed norm < this value the unit-vector
# normalisation is skipped to avoid division by near-zero.
_MIN_FIELD_NORM_UT: Final[float] = 1e-3


# ---------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------


def _earth_field_enu(amplitude_ut: float, declination_deg: float, inclination_deg: float) -> Float64Array:
    """Compute Earth's magnetic field vector in the ENU world frame (µT).

    ENU convention: x = East, y = North, z = Up.

    Parameters
    ----------
    amplitude_ut:
        Total field magnitude (µT).  Typical surface value: 25–65 µT.
    declination_deg:
        Magnetic declination (degrees east of geographic north).
        Positive = east.
    inclination_deg:
        Magnetic dip angle (degrees below horizontal).
        Positive = downward (into Earth).  Northern Hemisphere: positive.

    Returns
    -------
    np.ndarray, shape (3,), dtype float64
        [East, North, Up] field components in µT.
    """
    d = math.radians(declination_deg)
    inc = math.radians(inclination_deg)
    # Horizontal component H = B * cos(I); vertical component Z = B * sin(I)
    h = amplitude_ut * math.cos(inc)
    return np.array(
        [
            h * math.sin(d),  # East  (+x in ENU)
            h * math.cos(d),  # North (+y in ENU)
            -amplitude_ut * math.sin(inc),  # Up (+z in ENU); field points down → negative
        ],
        dtype=np.float64,
    )


def _quat_wxyz_to_rotmat(q: Float64Array) -> Float64Array:
    """Build a 3×3 rotation matrix from a body-to-world quaternion ``(w, x, y, z)``.

    The resulting matrix ``R`` satisfies ``v_world = R @ v_body``, i.e.,
    ``R`` rotates body-frame vectors to world-frame vectors.
    """
    w, x, y, z = float(q[0]), float(q[1]), float(q[2]), float(q[3])
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


class MagnetometerModel(BaseSensor):
    """
    3-axis solid-state magnetometer model with hard/soft-iron distortion.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Sensor output rate in Hz (50–200 Hz for typical MEMS magnetometers).
    noise_sigma_ut:
        Isotropic Gaussian white noise standard deviation on each axis (µT).
        Typical MEMS value: 0.1–1.0 µT.
    field_amplitude_ut:
        Magnitude of Earth's magnetic field (µT).
        Use the WMM value for your location (typically 25–65 µT).
    declination_deg:
        Magnetic declination at the simulation origin (degrees east of north).
    inclination_deg:
        Magnetic dip angle at the simulation origin (degrees below horizontal).
        Positive in the Northern Hemisphere; ~60-70° for mid-latitudes.
    hard_iron_ut:
        3-element list/array of per-axis hard-iron bias offsets (µT).
        Represents constant magnetic interference from nearby ferrous material
        or permanent magnets.  Default: [0, 0, 0] (no distortion).
    soft_iron_scale:
        3-element list/array of per-axis soft-iron scale factors (dimensionless,
        all > 0).  Models asymmetric permeability of nearby soft-iron material.
        Default: [1, 1, 1] (no distortion).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "magnetometer",
        update_rate_hz: float = 100.0,
        noise_sigma_ut: float = 0.5,
        field_amplitude_ut: float = _EARTH_FIELD_DEFAULT_UT,
        declination_deg: float = 0.0,
        inclination_deg: float = 60.0,
        hard_iron_ut: list[float] | None = None,
        soft_iron_scale: list[float] | None = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_ut = float(noise_sigma_ut)
        self.field_amplitude_ut = float(field_amplitude_ut)
        self.declination_deg = float(declination_deg)
        self.inclination_deg = float(inclination_deg)
        self.hard_iron_ut: Float64Array = np.asarray(
            hard_iron_ut if hard_iron_ut is not None else [0.0, 0.0, 0.0],
            dtype=np.float64,
        )
        if self.hard_iron_ut.shape != (3,):
            raise ValueError(f"hard_iron_ut must be a 3-element array, got shape {self.hard_iron_ut.shape}")
        self.soft_iron_scale: Float64Array = np.asarray(
            soft_iron_scale if soft_iron_scale is not None else [1.0, 1.0, 1.0],
            dtype=np.float64,
        )
        if self.soft_iron_scale.shape != (3,):
            raise ValueError(f"soft_iron_scale must be a 3-element array, got shape {self.soft_iron_scale.shape}")
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed

        # Pre-compute the constant world-frame Earth field vector (µT, ENU frame).
        self._earth_field: Float64Array = _earth_field_enu(
            self.field_amplitude_ut,
            self.declination_deg,
            self.inclination_deg,
        )
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "MagnetometerConfig") -> "MagnetometerModel":
        """Construct a :class:`MagnetometerModel` from a :class:`~genesis.sensors.config.MagnetometerConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "MagnetometerConfig":
        """Return the current parameters as a :class:`~genesis.sensors.config.MagnetometerConfig`."""
        from .config import MagnetometerConfig

        return MagnetometerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_ut=self.noise_sigma_ut,
            field_amplitude_ut=self.field_amplitude_ut,
            declination_deg=self.declination_deg,
            inclination_deg=self.inclination_deg,
            hard_iron_ut=self.hard_iron_ut.tolist(),
            soft_iron_scale=self.soft_iron_scale.tolist(),
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> MagnetometerObservation | dict[str, Any]:
        """
        Produce a realistic magnetometer reading in the body frame.

        The sensor looks for a body-to-world rotation in *state* to project
        the world-frame Earth field into body coordinates.

        Expected keys in *state* (at least one required for a rotated output):
        - ``"attitude_mat"`` -- ``np.ndarray`` shape ``(3, 3)``, body-to-world
          rotation matrix ``R_bw`` (columns are body-frame axes expressed in
          world coordinates).  ``v_world = R_bw @ v_body``.
        - ``"attitude_quat"`` -- ``np.ndarray`` shape ``(4,)``, body-to-world
          quaternion in ``(w, x, y, z)`` order.  Used when
          ``"attitude_mat"`` is absent.

        When neither key is present, the world-frame Earth field is used
        directly (equivalent to the sensor being aligned with the world frame).
        """
        # ------------------------------------------------------------------
        # Rotate Earth field into body frame
        # ------------------------------------------------------------------
        R_bw = state.get("attitude_mat")
        if R_bw is not None:
            R = np.asarray(R_bw, dtype=np.float64)
            if R.shape != (3, 3):
                raise ValueError(f"state['attitude_mat'] must be shape (3, 3), got {R.shape}")
            # R_bw maps body→world; R_bw.T maps world→body
            field_body: Float64Array = R.T @ self._earth_field
        else:
            q = state.get("attitude_quat")
            if q is not None:
                q_arr = np.asarray(q, dtype=np.float64)
                if q_arr.shape != (4,):
                    raise ValueError(f"state['attitude_quat'] must be shape (4,), got {q_arr.shape}")
                # Normalise before converting to avoid numerical rot-mat skew.
                q_norm = float(np.linalg.norm(q_arr))
                if q_norm < _MIN_FIELD_NORM_UT:  # essentially zero quaternion
                    field_body = self._earth_field.copy()
                else:
                    q_arr = q_arr / q_norm
                    R = _quat_wxyz_to_rotmat(q_arr)
                    # R maps body→world; R.T maps world→body
                    field_body = R.T @ self._earth_field
            else:
                # No attitude — return world-frame field (frame = aligned with world)
                field_body = self._earth_field.copy()

        # ------------------------------------------------------------------
        # Soft-iron distortion (per-axis scale)
        # ------------------------------------------------------------------
        field_distorted: Float64Array = field_body * self.soft_iron_scale

        # ------------------------------------------------------------------
        # Hard-iron bias (constant per-axis offset)
        # ------------------------------------------------------------------
        field_biased: Float64Array = field_distorted + self.hard_iron_ut

        # ------------------------------------------------------------------
        # Gaussian white noise
        # ------------------------------------------------------------------
        noise: Float64Array = self._rng.normal(0.0, self.noise_sigma_ut, 3).astype(np.float64)
        field_noisy: Float64Array = field_biased + noise

        obs: MagnetometerObservation = {
            "mag_field_ut": field_noisy,
            "field_norm_ut": float(np.linalg.norm(field_noisy)),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["MagnetometerModel"]
