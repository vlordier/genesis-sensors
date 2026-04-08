"""
Downward-facing optical flow / visual-odometry sensor model.

Models a camera-based optical flow sensor (e.g. PX4FLOW, MatekSys 3901-L0X,
PixArt PAA3905) that estimates horizontal body velocity by tracking visual
features on the ground below the vehicle.

The core physical relationship for a pinhole camera looking straight down::

    omega_x = v_y / h          # angular flow in camera X = forward velocity / height
    omega_y = -v_x / h         # angular flow in camera Y = rightward velocity / height

where ``v_x``, ``v_y`` are ENU body-frame horizontal velocities (m/s) and
``h`` is the height above ground (m).  The signs follow the MAVLink optical
flow convention (frame = FRD camera body frame).

Noise model
-----------
* **Shot / read noise** — fixed noise floor ``noise_floor_rad_s`` independent
  of velocity.  Represents quantisation, fixed-pattern noise, and dark current.
* **Motion-blur noise** — noise proportional to the flow *speed*
  (``noise_slope * |omega|``).  Models pixel smearing at high angular rates.
* **Quality** — integer 0–255 track quality, reduced by altitude and velocity:

  * *Height penalty*: quality is maximum below ``nominal_quality_height_m``
    and drops to zero at ``max_quality_height_m`` (features become too small
    to track).
  * *Velocity penalty*: quality degrades above ``max_detection_rad_s``
    (motion blur saturates detectors).

Usage
-----
::

    flow = OpticalFlowModel(update_rate_hz=100.0)
    obs = flow.step(sim_time, {
        "vel":  np.array([0.5, 0.3, -0.1]),   # world-frame ENU velocity (m/s)
        "pos":  np.array([10.0, 5.0, 3.0]),   # world-frame ENU pos; pos[2] = height AGL
    })
    print(obs["flow_x_rad"])      # integrated horizontal flow (rad)
    print(obs["quality"])         # tracking quality 0–255
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .types import Float64Array, OpticalFlowObservation

if TYPE_CHECKING:
    from .config import OpticalFlowConfig

# Minimum height to avoid division-by-zero in flow computation.
_MIN_HEIGHT_M: Final[float] = 0.01
# Quality range for MAVLink OPTICAL_FLOW message.
_MAX_QUALITY: Final[int] = 255


class OpticalFlowModel(BaseSensor):
    """
    Downward-facing optical flow sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Sensor output rate in Hz.  Typical: 100–400 Hz (MEMS cameras),
        30–100 Hz (USB cameras with onboard processing).
    noise_floor_rad_s:
        Fixed Gaussian noise floor on flow rate (rad/s per axis).
        Represents read noise, ADC quantisation, and tracker jitter.
        Typical MEMS optical flow: 0.002–0.01 rad/s.
    noise_slope:
        Motion-blur noise coefficient (dimensionless).  Additional sigma per
        (rad/s) of true flow speed.  Typical: 0.02–0.1.
    max_detection_rad_s:
        Maximum detectable flow rate (rad/s) before the sensor saturates.
        Above this the sensor reports ``quality=0`` and invalid flow.
        Typical: 2.5–5 rad/s.
    nominal_quality_height_m:
        Height below which tracking quality is at its peak (m).
        At lower altitudes, ground features subtend more pixels → good tracking.
    max_quality_height_m:
        Height above which quality drops to zero (features too small to track).
        Typical: 20–50 m.
    base_quality:
        Maximum achievable tracking quality (0–255) under ideal conditions.
        Reduces this to account for texture-poor environments or long focal lengths.
    low_quality_threshold:
        Quality values at or below this threshold are considered unreliable.
        Observations with ``quality <= low_quality_threshold`` still return
        valid TypedDict keys but the flow values should be treated as invalid
        by the consumer.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "optical_flow",
        update_rate_hz: float = 100.0,
        noise_floor_rad_s: float = 0.005,
        noise_slope: float = 0.05,
        max_detection_rad_s: float = 4.0,
        nominal_quality_height_m: float = 2.0,
        max_quality_height_m: float = 30.0,
        base_quality: int = 220,
        low_quality_threshold: int = 25,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_floor_rad_s = float(noise_floor_rad_s)
        self.noise_slope = float(noise_slope)
        self.max_detection_rad_s = float(max_detection_rad_s)
        self.nominal_quality_height_m = float(nominal_quality_height_m)
        if max_quality_height_m <= nominal_quality_height_m:
            raise ValueError(
                f"max_quality_height_m ({max_quality_height_m}) must be greater than "
                f"nominal_quality_height_m ({nominal_quality_height_m})"
            )
        self.max_quality_height_m = float(max_quality_height_m)
        if not 0 <= base_quality <= _MAX_QUALITY:
            raise ValueError(f"base_quality must be in [0, 255], got {base_quality}")
        self.base_quality = int(base_quality)
        self.low_quality_threshold = int(low_quality_threshold)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}
        # Accumulated integrated flow (rad) — reset each step after reporting
        self._dt: float = 1.0 / self.update_rate_hz

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "OpticalFlowConfig") -> "OpticalFlowModel":
        """Construct from an :class:`~genesis.sensors.config.OpticalFlowConfig`."""
        return cls(**config.model_dump())

    def get_config(self) -> "OpticalFlowConfig":
        """Return the current parameters as an :class:`~genesis.sensors.config.OpticalFlowConfig`."""
        from .config import OpticalFlowConfig

        return OpticalFlowConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_floor_rad_s=self.noise_floor_rad_s,
            noise_slope=self.noise_slope,
            max_detection_rad_s=self.max_detection_rad_s,
            nominal_quality_height_m=self.nominal_quality_height_m,
            max_quality_height_m=self.max_quality_height_m,
            base_quality=self.base_quality,
            low_quality_threshold=self.low_quality_threshold,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> OpticalFlowObservation | dict[str, Any]:
        """
        Produce a realistic optical flow measurement.

        Expected keys in *state*:

        - ``"vel"`` -- ``np.ndarray[3]`` world-frame ENU velocity (m/s).
          Horizontal components ``vel[0]`` (East) and ``vel[1]`` (North) drive
          the flow calculation.
        - ``"pos"`` -- ``np.ndarray[3]`` world-frame ENU position (m).
          The z-component (``pos[2]``) is used as height above ground.
          Can be omitted if ``"optical_flow_height_m"`` is provided directly.
        - ``"optical_flow_height_m"`` *(optional)* -- explicit height above
          ground (m), overrides ``pos[2]``.  Use this when the world-frame
          origin is not at ground level.

        When ``"vel"`` is absent the observation dict is empty.
        """
        vel_raw = state.get("vel")
        if vel_raw is None:
            self._last_obs = {}
            return self._last_obs

        vel: Float64Array = np.asarray(vel_raw, dtype=np.float64)
        if vel.shape != (3,):
            raise ValueError(f"state['vel'] must be a 3-element array, got shape {vel.shape}")

        # Guard NaN/Inf
        if not np.all(np.isfinite(vel)):
            vel = np.zeros(3, dtype=np.float64)

        # Height above ground: prefer explicit override, else pos[2]
        if "optical_flow_height_m" in state:
            height_m = float(state["optical_flow_height_m"])
        elif "pos" in state:
            height_m = float(np.asarray(state["pos"], dtype=np.float64)[2])
        else:
            height_m = self.nominal_quality_height_m  # fallback; quality will be nominal

        height_m = max(height_m, _MIN_HEIGHT_M)

        # ------------------------------------------------------------------
        # True flow rates (rad/s) — pinhole projection for a nadir-pointing cam.
        # ENU convention: +X East, +Y North. Following the MAVLink/PX4 optical-
        # flow convention, rotation about one image axis is induced by velocity
        # along the orthogonal horizontal axis:
        #   omega_x = v_north / h
        #   omega_y = -v_east / h
        # ------------------------------------------------------------------
        true_omega_x = vel[1] / height_m  # rad/s
        true_omega_y = -vel[0] / height_m  # rad/s
        flow_speed = math.hypot(true_omega_x, true_omega_y)

        # ------------------------------------------------------------------
        # Quality estimation
        # ------------------------------------------------------------------
        # Height factor: peak at low altitude, zero above max_quality_height
        if height_m >= self.max_quality_height_m:
            height_qual_factor = 0.0
        elif height_m <= self.nominal_quality_height_m:
            height_qual_factor = 1.0
        else:
            height_qual_factor = 1.0 - (height_m - self.nominal_quality_height_m) / (
                self.max_quality_height_m - self.nominal_quality_height_m
            )

        # Velocity factor: degrades proportionally above 50% of max detectable rate
        half_max = 0.5 * self.max_detection_rad_s
        if flow_speed >= self.max_detection_rad_s:
            vel_qual_factor = 0.0
        elif flow_speed <= half_max:
            vel_qual_factor = 1.0
        else:
            vel_qual_factor = 1.0 - (flow_speed - half_max) / half_max

        quality_raw = self.base_quality * height_qual_factor * vel_qual_factor
        quality = int(round(max(0.0, min(_MAX_QUALITY, quality_raw))))

        # ------------------------------------------------------------------
        # Sensor saturation: flow beyond max_detection_rad_s → invalid
        # ------------------------------------------------------------------
        if flow_speed > self.max_detection_rad_s:
            quality = 0
            obs: OpticalFlowObservation = {
                "flow_x_rad": 0.0,
                "flow_y_rad": 0.0,
                "flow_rate_x_rad_s": 0.0,
                "flow_rate_y_rad_s": 0.0,
                "quality": quality,
                "ground_distance_m": height_m,
            }
            self._last_obs = obs
            self._mark_updated(sim_time)
            return obs

        # ------------------------------------------------------------------
        # Gaussian noise (rad/s per axis)
        # ------------------------------------------------------------------
        sigma_x = self.noise_floor_rad_s + self.noise_slope * abs(true_omega_x)
        sigma_y = self.noise_floor_rad_s + self.noise_slope * abs(true_omega_y)
        noisy_omega_x = true_omega_x + float(self._rng.normal(0.0, sigma_x))
        noisy_omega_y = true_omega_y + float(self._rng.normal(0.0, sigma_y))

        # Integrated flow over the update interval (rad)
        flow_x = noisy_omega_x * self._dt
        flow_y = noisy_omega_y * self._dt

        obs = {
            "flow_x_rad": flow_x,
            "flow_y_rad": flow_y,
            "flow_rate_x_rad_s": noisy_omega_x,
            "flow_rate_y_rad_s": noisy_omega_y,
            "quality": quality,
            "ground_distance_m": height_m,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["OpticalFlowModel"]
