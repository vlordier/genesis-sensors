"""
Shared type aliases, TypedDicts, and numpy array type helpers for
the Genesis external sensor realism layer.

These types serve two purposes:

1. **Static type checking** — IDEs and ``mypy`` can infer precise
   types for array shapes and observation dict keys.
2. **Documentation** — TypedDicts make the expected shape of every
   sensor's state input and observation output explicit.

All TypedDicts use ``total=False`` for *state* dicts (callers may omit
any field) and ``total=True`` (the default) for *observation* dicts
(every field is always present in a successful update).
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, TypedDict

import numpy as np
import numpy.typing as npt

# ---------------------------------------------------------------------------
# Generic NDArray aliases
# ---------------------------------------------------------------------------

#: A float32 NumPy array of arbitrary shape.
FloatArray = npt.NDArray[np.float32]
#: A float64 NumPy array of arbitrary shape.
Float64Array = npt.NDArray[np.float64]
#: A uint8 NumPy array of arbitrary shape.
UInt8Array = npt.NDArray[np.uint8]
#: A uint16 NumPy array of arbitrary shape.
UInt16Array = npt.NDArray[np.uint16]
#: An int32 NumPy array of arbitrary shape.
Int32Array = npt.NDArray[np.int32]
#: A boolean NumPy array of arbitrary shape.
BoolArray = npt.NDArray[np.bool_]
#: Re-export numpy's ArrayLike for convenience.
ArrayLike = npt.ArrayLike
#: A jammer zone is a (centre_xyz, radius_m) pair.
JammerZone = tuple[ArrayLike, float]

# ---------------------------------------------------------------------------
# Polarity
# ---------------------------------------------------------------------------


class Polarity(IntEnum):
    """
    Event polarity for a Dynamic Vision Sensor (DVS).

    ``POSITIVE`` (+1) indicates a log-intensity increase, ``NEGATIVE``
    (-1) indicates a decrease.  The ``IntEnum`` base ensures arithmetic
    with plain integers still works (``Polarity.POSITIVE + 0 == 1``).
    """

    NEGATIVE = -1
    POSITIVE = 1


# ---------------------------------------------------------------------------
# Shared sensor input state
# ---------------------------------------------------------------------------


class SensorState(TypedDict, total=False):
    """
    Combined ideal-state dict consumed by the sensor layer.

    All fields are optional (``total=False``); individual sensors only
    read the keys they need.  Building this dict from Genesis outputs::

        state: SensorState = {
            "rgb":   cam.render(rgb=True),
            "depth": cam.render(depth=True),
            "seg":   cam.render(segmentation=True),
            "normal": cam.render(normal=True),
            "gray":  gray_from_rgb(rgb),
            "pose":  drone.get_pos_quat(),
            "pos":   drone.get_pos().numpy(),
            "vel":   drone.get_vel().numpy(),
            "ang_vel": drone.get_ang_vel().numpy(),
            "range_image": raycaster.read().numpy(),
            "intensity_image": intensity.numpy(),
            "temperature_map": {e.id: e.temp_c for e in scene.entities},
            "obstruction": sky_obstruction_fraction,
            "weather": {"rain_rate_mm_h": 5.0},
        }
    """

    # Visual — shape annotations are in comments; TypedDict fields must be
    # plain types, so we use the closest NDArray alias available.
    rgb: UInt8Array  # shape (H, W, 3) — may also be float32 [0, 1]
    rgb_right: UInt8Array  # shape (H, W, 3) — right-eye image for stereo; synthesised when absent
    depth: FloatArray  # shape (H, W), metres
    seg: Int32Array  # shape (H, W), entity IDs
    normal: FloatArray  # shape (H, W, 3)
    gray: FloatArray  # shape (H, W), [0, 1]

    # Pose / velocity
    pose: tuple[FloatArray, FloatArray]  # (pos (3,), quat (4,))
    pos: Float64Array  # shape (3,) metres ENU
    vel: Float64Array  # shape (3,) m/s ENU
    ang_vel: FloatArray  # shape (3,) rad/s

    # IMU
    lin_acc: Float64Array  # shape (3,) body-frame linear acceleration (m/s²), no gravity
    gravity_body: Float64Array  # shape (3,) gravity vector in body frame (m/s²)

    # Magnetometer — body-to-world rotation (only one is needed)
    attitude_mat: FloatArray  # (3, 3) body-to-world rotation matrix R_bw
    attitude_quat: FloatArray  # (4,) body-to-world quaternion (w, x, y, z)

    # LiDAR
    range_image: FloatArray  # shape (n_channels, h_resolution), metres
    intensity_image: FloatArray  # shape (n_channels, h_resolution), [0, 1]

    # Thermal
    temperature_map: dict[int, float]  # entity_id → temperature °C

    # Environment
    obstruction: float  # 0–1 sky-hemisphere obstruction fraction
    weather: dict[str, float]  # e.g. {"rain_rate_mm_h": 5.0}
    ambient_temp_c: float  # ambient air temperature (°C)
    relative_humidity_pct: float  # ambient relative humidity (%)
    illuminance_lux: float  # ambient light level (lux)
    gas_ppm: float  # directly supplied gas concentration (ppm)
    gas_background_ppm: float  # background gas concentration (ppm)
    gas_sources: list[dict[str, Any]]  # optional plume source descriptors

    # Airspeed / wind
    airspeed_ms: float  # true airspeed (m/s); used directly when present
    wind: Float64Array  # shape (3,) world/body-frame wind vector (m/s)
    wind_ms: Float64Array  # alias for wind vector (m/s)

    # Rangefinder
    range_m: float  # true perpendicular range to nearest surface (m)

    # Optical flow (downward-facing camera)
    # optical_flow_vel_xy: world-frame horizontal velocity used when "vel" absent from flow calculation
    optical_flow_height_m: float  # ground distance (m) overriding pos[2] for flow conversion

    # Battery monitor
    current_a: float  # total current drawn from the battery pack (A)

    # Wheel odometry
    # vel / ang_vel keys (above) are shared — no additional keys required.

    # Force/torque sensor
    force: FloatArray  # shape (3,) — ideal force in sensor frame (N)
    torque: FloatArray  # shape (3,) — ideal torque in sensor frame (Nm)

    # Joint state sensor
    joint_pos: FloatArray  # shape (N,) — ideal joint positions (rad)
    joint_vel: FloatArray  # shape (N,) — ideal joint velocities (rad/s)
    joint_torque: FloatArray  # shape (N,) — ideal joint torques (Nm)

    # Contact sensor
    contact_force_n: float  # scalar ideal contact force magnitude (N)

    # Tactile array sensor
    pressure_map: FloatArray  # shape (H, W) — ideal taxel pressure map (Pa)

    # Current / power sensor
    voltage_v: float  # bus voltage (V)

    # RPM sensor
    rpm: float  # scalar shaft or rotor speed (revolutions per minute)


# ---------------------------------------------------------------------------
# Airspeed model
# ---------------------------------------------------------------------------


class AirspeedObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.AirspeedModel`."""

    airspeed_ms: float  # indicated airspeed (m/s); 0 when below dead-band or blocked
    dynamic_pressure_pa: float  # raw Bernoulli differential pressure (Pa)


# ---------------------------------------------------------------------------
# Rangefinder model
# ---------------------------------------------------------------------------


class RangefinderObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.RangefinderModel`."""

    range_m: float  # noisy range measurement (m), or no_hit_value when out of range / dropout
    in_range: bool  # True when a valid return was detected


# ---------------------------------------------------------------------------
# Barometer model
# ---------------------------------------------------------------------------


class BarometerObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.BarometerModel`."""

    altitude_m: float  # noisy barometric altitude (m MSL)
    pressure_pa: float  # corresponding ISA absolute pressure (Pa)


# ---------------------------------------------------------------------------
# Magnetometer model
# ---------------------------------------------------------------------------


class MagnetometerObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.MagnetometerModel`."""

    mag_field_ut: Float64Array  # shape (3,) — noisy magnetic field in body frame (µT)
    field_norm_ut: float  # scalar total field magnitude (µT)


# ---------------------------------------------------------------------------
# Environmental sensing models
# ---------------------------------------------------------------------------


class ThermometerObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.ThermometerModel`."""

    temperature_c: float  # ambient temperature (°C)
    temperature_f: float  # ambient temperature (°F)


class HygrometerObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.HygrometerModel`."""

    relative_humidity_pct: float  # measured relative humidity (%)
    dew_point_c: float  # dew-point estimate (°C)


class LightSensorObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.LightSensorModel`."""

    illuminance_lux: float  # ambient illuminance (lux)
    is_saturated: bool  # True when the reading clips at the configured maximum


class GasObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.GasSensorModel`."""

    concentration_ppm: float  # measured gas concentration (ppm)
    alarm: bool  # True when concentration exceeds the configured threshold


class AnemometerObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.AnemometerModel`."""

    wind_vector_ms: Float64Array  # shape (3,) — measured wind vector (m/s)
    wind_speed_ms: float  # scalar wind speed magnitude (m/s)
    wind_direction_deg: float  # heading of the measured wind vector in the XY plane (deg)


# ---------------------------------------------------------------------------
# IMU model
# ---------------------------------------------------------------------------


class ImuObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.IMUModel`."""

    lin_acc: Float64Array  # shape (3,) — noisy specific force (m/s²); includes gravity when add_gravity=True
    ang_vel: Float64Array  # shape (3,) — noisy angular velocity (rad/s)


# ---------------------------------------------------------------------------
# Camera model
# ---------------------------------------------------------------------------


class CameraObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.CameraModel`."""

    rgb: UInt8Array  # shape (H, W, 3)


# ---------------------------------------------------------------------------
# Stereo camera model
# ---------------------------------------------------------------------------


class StereoCameraObservation(TypedDict, total=False):
    """Observation emitted by :class:`~genesis.sensors.StereoCameraModel`.

    The ``disparity``, ``depth``, ``valid_mask``, and ``point_cloud`` keys are
    present only when ``state["depth"]`` (the ideal depth map) was provided.
    """

    rgb_left: UInt8Array  # shape (H, W, 3) — corrupted left-eye image
    rgb_right: UInt8Array  # shape (H, W, 3) — corrupted right-eye image
    disparity: FloatArray  # shape (H, W) float32 — noisy disparity in pixels
    depth: FloatArray  # shape (H, W) float32 — reconstructed depth in metres
    valid_mask: BoolArray  # shape (H, W) bool — True where depth is valid
    point_cloud: FloatArray  # shape (N, 3) float32 — points in left-camera frame
    baseline_m: float  # configured baseline (metres)
    focal_px: float  # focal length derived from resolution + HFOV (pixels)


# ---------------------------------------------------------------------------
# Event camera model
# ---------------------------------------------------------------------------


class EventCameraObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.EventCameraModel`."""

    events: list[Any]  # list[Event]; forward-ref avoids circular import
    events_array: FloatArray  # shape (N, 4): columns x, y, polarity (+1/-1), timestamp


# ---------------------------------------------------------------------------
# Thermal camera model
# ---------------------------------------------------------------------------


class ThermalObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.ThermalCameraModel`."""

    thermal: UInt8Array | UInt16Array  # shape (H, W); dtype depends on bit_depth
    temperature_c: FloatArray  # shape (H, W) — pre-quantisation temperature


# ---------------------------------------------------------------------------
# LiDAR model
# ---------------------------------------------------------------------------


class LidarObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.LidarModel`."""

    points: FloatArray  # shape (N, 4) — x, y, z, intensity
    range_image: FloatArray  # shape (n_channels, h_resolution) — processed ranges


# ---------------------------------------------------------------------------
# GNSS model
# ---------------------------------------------------------------------------


class GnssObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.GNSSModel`."""

    pos: Float64Array  # shape (3,) — noisy world-frame position (m)
    vel: Float64Array  # shape (3,) — noisy world-frame velocity (m/s)
    pos_llh: Float64Array  # shape (3,) — latitude (deg), longitude (deg), altitude (m)
    fix_quality: int  # GnssFixQuality value
    n_satellites: int
    hdop: float
    vdop: float  # vertical DOP; typically ~1.5× HDOP for single-constellation


# ---------------------------------------------------------------------------
# Radio link model
# ---------------------------------------------------------------------------


class RadioObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.RadioLinkModel`."""

    delivered: list[Any]  # list[ScheduledPacket]; forward-ref avoids circular import
    queue_depth: int


# ---------------------------------------------------------------------------
# Optical flow model
# ---------------------------------------------------------------------------


class OpticalFlowObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.OpticalFlowModel`."""

    flow_x_rad: float  # integrated X-axis angular flow since last update (rad), positive = rightward
    flow_y_rad: float  # integrated Y-axis angular flow since last update (rad), positive = forward
    flow_rate_x_rad_s: float  # instantaneous X-axis angular flow rate (rad/s)
    flow_rate_y_rad_s: float  # instantaneous Y-axis angular flow rate (rad/s)
    quality: int  # tracking quality 0 (lost) – 255 (perfect)
    ground_distance_m: float  # estimated height above ground used for flow conversion (m)


# ---------------------------------------------------------------------------
# Battery monitor model
# ---------------------------------------------------------------------------


class BatteryObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.BatteryModel`."""

    voltage_v: float  # measured terminal voltage of the full pack (V)
    voltage_per_cell_v: float  # per-cell terminal voltage (V)
    current_a: float  # measured current draw (A; positive = discharge)
    power_w: float  # instantaneous power draw (W)
    soc: float  # estimated state of charge (0 = empty … 1 = full)
    capacity_used_mah: float  # cumulative charge consumed since last reset (mAh)
    is_low: bool  # True when per-cell voltage < low_cell_voltage_v


# ---------------------------------------------------------------------------
# Wheel odometry model
# ---------------------------------------------------------------------------


class WheelOdometryObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.WheelOdometryModel`."""

    delta_pos_m: FloatArray  # shape (2,) float32 — world-frame XY position increment (m)
    delta_heading_rad: float  # yaw change this step (rad)
    linear_vel_ms: float  # estimated ground-plane speed (m/s)
    angular_vel_rads: float  # estimated yaw rate (rad/s)


# ---------------------------------------------------------------------------
# Force / torque sensor model
# ---------------------------------------------------------------------------


class ForceTorqueObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.ForceTorqueSensorModel`."""

    force_n: FloatArray  # shape (3,) float32 — noisy, biased force [Fx, Fy, Fz] (N)
    torque_nm: FloatArray  # shape (3,) float32 — noisy, biased torque [Tx, Ty, Tz] (Nm)


# ---------------------------------------------------------------------------
# Joint state sensor model
# ---------------------------------------------------------------------------


class JointStateObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.JointStateSensor`."""

    joint_pos_rad: FloatArray  # shape (N,) float32 — noisy joint positions (rad)
    joint_vel_rads: FloatArray  # shape (N,) float32 — noisy joint velocities (rad/s)
    joint_torque_nm: FloatArray  # shape (N,) float32 — noisy joint torques (Nm)


# ---------------------------------------------------------------------------
# Contact sensor model
# ---------------------------------------------------------------------------


class ContactObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.ContactSensor`."""

    contact_detected: bool  # True when filtered force exceeds threshold
    force_n: float  # noisy contact force magnitude (N), clipped to force_range_n


# ---------------------------------------------------------------------------
# Depth camera model
# ---------------------------------------------------------------------------


class DepthCameraObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.DepthCameraModel`."""

    depth_m: FloatArray  # shape (H, W) float32 — noisy depth in metres; 0 = invalid
    valid_mask: BoolArray  # shape (H, W) bool — True at pixels with a valid depth return


# ---------------------------------------------------------------------------
# Tactile array sensor model
# ---------------------------------------------------------------------------


class TactileArrayObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.TactileArraySensor`."""

    pressure_pa: FloatArray  # shape (H, W) float32 — noisy per-taxel pressures
    contact_mask: BoolArray  # shape (H, W) bool — taxels above contact threshold
    cop_xy: FloatArray  # shape (2,) float32 — normalised centre-of-pressure [0,1]^2
    total_force_n: float  # integrated normal force over all taxels (N)


# ---------------------------------------------------------------------------
# Current sensor model
# ---------------------------------------------------------------------------


class CurrentObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.CurrentSensor`."""

    current_a: float  # measured bus current (A)
    power_w: float  # instantaneous electrical power (W)


# ---------------------------------------------------------------------------
# RPM sensor model
# ---------------------------------------------------------------------------


class RPMObservation(TypedDict):
    """Observation emitted by :class:`~genesis.sensors.RPMSensor`."""

    rpm: float  # measured shaft speed (RPM)
    speed_rads: float  # equivalent angular velocity (rad/s)


__all__ = [
    # Array type aliases
    "ArrayLike",
    "Float64Array",
    "FloatArray",
    "Int32Array",
    "JammerZone",
    "Polarity",
    "UInt16Array",
    "UInt8Array",
    # Input state
    "SensorState",
    # Observation TypedDicts
    "AirspeedObservation",
    "AnemometerObservation",
    "BarometerObservation",
    "BatteryObservation",
    "BoolArray",
    "CameraObservation",
    "ContactObservation",
    "CurrentObservation",
    "DepthCameraObservation",
    "EventCameraObservation",
    "ForceTorqueObservation",
    "GasObservation",
    "GnssObservation",
    "HygrometerObservation",
    "ImuObservation",
    "JointStateObservation",
    "LidarObservation",
    "LightSensorObservation",
    "MagnetometerObservation",
    "OpticalFlowObservation",
    "RPMObservation",
    "RadioObservation",
    "RangefinderObservation",
    "StereoCameraObservation",
    "TactileArrayObservation",
    "ThermalObservation",
    "ThermometerObservation",
    "WheelOdometryObservation",
]
