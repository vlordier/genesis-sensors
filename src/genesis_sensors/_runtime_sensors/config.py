"""
Pydantic v2 configuration models for every Genesis sensor.

Each ``*Config`` class is a ``pydantic.BaseModel`` with field-level
validation (range checks, type coercion).  Sensors can be constructed
directly from these models via their ``from_config()`` class-method or
serialised / de-serialised for experiment logging::

    import json
    from genesis.sensors.config import CameraConfig, SensorSuiteConfig

    cfg = CameraConfig(iso=800, jpeg_quality=70)
    print(cfg.model_dump_json(indent=2))     # JSON export
    cfg2 = CameraConfig.model_validate_json(json_str)  # JSON import

    suite_cfg = SensorSuiteConfig(rgb=cfg)
    suite = SensorSuite.from_config(suite_cfg)

All fields mirror the constructor parameters of the corresponding sensor
class exactly, so ``sensor_class(**config.model_dump())`` always works.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

# ---------------------------------------------------------------------------
# CameraConfig
# ---------------------------------------------------------------------------


class CameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.CameraModel`."""

    name: str = "rgb_camera"
    update_rate_hz: float = Field(default=30.0, gt=0, description="Frame rate (Hz).")
    resolution: tuple[int, int] = Field(default=(640, 480), description="(width, height) in pixels.")
    distortion_coeffs: tuple[float, ...] | None = Field(
        default=None,
        description="OpenCV-style (k1, k2, p1, p2[, k3]).  None = no distortion.",
    )
    focal_length_px: float | None = Field(
        default=None,
        gt=0,
        description="Optional focal length in pixels for distortion / rolling-shutter geometry. Defaults to max(width, height).",
    )
    rolling_shutter_fraction: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="0 = global shutter, 1 = full rolling shutter.",
    )
    motion_blur_kernel: int = Field(default=0, ge=0, description="Half-length of 1-D motion-blur kernel. 0 = off.")
    base_iso: float = Field(default=100.0, gt=0, description="Reference ISO.")
    iso: float = Field(default=100.0, gt=0, description="Effective ISO.")
    read_noise_sigma: float = Field(default=1.5, ge=0, description="Gaussian read-noise sigma (electrons).")
    dead_pixel_fraction: float = Field(default=0.0001, ge=0.0, le=1.0, description="Fraction of dead pixels.")
    hot_pixel_fraction: float = Field(default=0.00005, ge=0.0, le=1.0, description="Fraction of hot pixels.")
    jpeg_quality: int = Field(default=0, ge=0, le=100, description="JPEG quality (0 = disabled).")
    full_well_electrons: float = Field(default=3500.0, gt=0, description="Full-well capacity at base_iso (electrons).")
    vignetting_strength: float = Field(
        default=0.0,
        ge=0.0,
        description="Radial vignetting strength (0 = off, 0.5 = moderate, 1.0 = strong).",
    )
    chromatic_aberration_px: float = Field(
        default=0.0,
        ge=0.0,
        description="Lateral chromatic aberration: max channel shift in pixels at image corner (0 = off).",
    )
    auto_exposure: bool = Field(default=False, description="Enable automatic exposure control.")
    ae_setpoint: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Target mean frame brightness for AE controller [0, 1].",
    )
    ae_speed: float = Field(
        default=0.15,
        ge=0.0,
        le=1.0,
        description="AE convergence rate — exponential smoothing factor applied each frame [0, 1].",
    )
    ae_min_iso: float = Field(default=100.0, gt=0, description="Minimum ISO the AE controller may set.")
    ae_max_iso: float = Field(default=6400.0, gt=0, description="Maximum ISO the AE controller may set.")
    seed: int | None = Field(default=None, description="RNG seed for reproducibility.")

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


# ---------------------------------------------------------------------------
# EventCameraConfig
# ---------------------------------------------------------------------------


class EventCameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.EventCameraModel`."""

    name: str = "event_camera"
    update_rate_hz: float = Field(default=1000.0, gt=0)
    threshold_pos: float = Field(default=0.2, gt=0, description="Positive contrast threshold (log-intensity).")
    threshold_neg: float = Field(default=0.2, gt=0, description="Negative contrast threshold (log-intensity).")
    refractory_period_s: float = Field(default=0.0, ge=0.0, description="Per-pixel minimum inter-event interval (s).")
    threshold_variation: float = Field(
        default=0.0,
        ge=0.0,
        description="Relative per-pixel threshold spread (sigma / nominal).",
    )
    background_activity_rate_hz: float = Field(
        default=0.0, ge=0.0, description="Mean spontaneous noise event rate per pixel per second."
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# ThermalCameraConfig
# ---------------------------------------------------------------------------


class ThermalCameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.ThermalCameraModel`."""

    name: str = "thermal_camera"
    update_rate_hz: float = Field(default=9.0, gt=0)
    resolution: tuple[int, int] = Field(default=(320, 240))
    temp_ambient_c: float = Field(default=20.0, description="Background pixel temperature (°C).")
    temp_sky_c: float = Field(default=-30.0, description="Sky pixel temperature (°C).")
    psf_sigma: float = Field(default=1.0, ge=0.0, description="Gaussian PSF sigma (pixels).")
    nuc_sigma: float = Field(default=0.5, ge=0.0, description="NUC residual offset sigma (°C).")
    noise_sigma: float = Field(default=0.05, ge=0.0, description="Per-frame detector noise sigma (°C).")
    bit_depth: int = Field(default=14, ge=1, le=32, description="Output quantisation bit depth.")
    fog_density: float = Field(default=0.0, ge=0.0, description="Fog extinction coefficient (1/m).")
    temp_range_c: tuple[float, float] = Field(
        default=(-20.0, 140.0), description="(t_min, t_max) for quantisation clipping."
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _temp_range_ordered(self) -> "ThermalCameraConfig":
        t_min, t_max = self.temp_range_c
        if t_min >= t_max:
            raise ValueError(f"temp_range_c must satisfy t_min < t_max, got {self.temp_range_c}")
        return self

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


# ---------------------------------------------------------------------------
# LidarConfig
# ---------------------------------------------------------------------------


class LidarConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.LidarModel`."""

    name: str = "lidar"
    update_rate_hz: float = Field(default=10.0, gt=0)
    n_channels: int = Field(default=16, ge=1, description="Number of vertical scan beams.")
    v_fov_deg: tuple[float, float] = Field(default=(-15.0, 15.0), description="(min_elevation_deg, max_elevation_deg).")
    h_resolution: int = Field(default=1800, ge=1, description="Azimuth steps per revolution.")
    max_range_m: float = Field(default=100.0, gt=0, description="Maximum measurable range (m).")
    no_hit_value: float = Field(default=0.0, description="Value written for beams with no return.")
    range_noise_sigma_m: float = Field(default=0.02, ge=0.0, description="Gaussian range noise sigma (m).")
    intensity_noise_sigma: float = Field(default=0.01, ge=0.0, description="Gaussian intensity noise sigma (0-1).")
    dropout_prob: float = Field(default=0.0, ge=0.0, le=1.0, description="Per-beam random dropout probability.")
    rain_rate_mm_h: float = Field(default=0.0, ge=0.0, description="Rain rate for two-way attenuation (mm/h).")
    fog_density: float = Field(default=0.0, ge=0.0, description="Fog extinction coefficient (1/m).")
    channel_offsets_m: list[float] | None = Field(default=None, description="Per-channel calibration offsets (m).")
    beam_divergence_mrad: float = Field(
        default=0.0,
        ge=0.0,
        description="Half-angle beam divergence (mrad).  0 = off; typical spinning LiDAR: 1.5–3.0 mrad.",
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _channel_offsets_length(self) -> "LidarConfig":
        if self.channel_offsets_m is not None and len(self.channel_offsets_m) != self.n_channels:
            raise ValueError(
                f"channel_offsets_m must have exactly n_channels ({self.n_channels}) elements, "
                f"got {len(self.channel_offsets_m)}"
            )
        return self

    @model_validator(mode="after")
    def _v_fov_ordered(self) -> "LidarConfig":
        lo, hi = self.v_fov_deg
        if lo >= hi:
            raise ValueError(f"v_fov_deg must satisfy min < max, got {self.v_fov_deg}")
        return self


# ---------------------------------------------------------------------------
# GNSSConfig
# ---------------------------------------------------------------------------


class GNSSConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.GNSSModel`."""

    name: str = "gnss"
    update_rate_hz: float = Field(default=10.0, gt=0)
    noise_m: float = Field(default=1.5, ge=0.0, description="1-sigma position noise (m).")
    vel_noise_ms: float = Field(default=0.05, ge=0.0, description="1-sigma velocity noise (m/s).")
    bias_tau_s: float = Field(
        default=60.0,
        gt=0,
        description="Gauss-Markov bias time constant (s).",
    )
    bias_sigma_m: float = Field(default=0.5, ge=0.0, description="Steady-state bias sigma (m).")
    multipath_sigma_m: float = Field(default=1.0, ge=0.0, description="Multipath error sigma (m).")
    min_fix_altitude_m: float = Field(default=0.5, description="Altitude below which fix degrades (m).")
    jammer_zones: list[tuple[list[float], float]] = Field(
        default_factory=list,
        description="List of (centre_xyz, radius_m) jammer zones.  "
        "``centre_xyz`` is a list of 3 floats [x, y, z] in world-frame metres.",
    )
    origin_llh: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0),
        description="(lat_deg, lon_deg, alt_m) of world origin.",
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# RadioConfig
# ---------------------------------------------------------------------------


class RadioConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.RadioLinkModel`."""

    name: str = "radio"
    update_rate_hz: float = Field(default=100.0, gt=0)
    tx_power_dbm: float = Field(
        default=20.0,
        ge=-30.0,
        le=43.0,
        description="Transmit power (dBm). Range: −30 to +43 dBm (regulatory maximum in most regions).",
    )
    frequency_ghz: float = Field(default=2.4, gt=0, description="Carrier frequency (GHz).")
    noise_figure_db: float = Field(default=6.0, ge=0.0, description="Receiver noise figure (dB).")
    path_loss_exponent: float = Field(default=2.5, ge=2.0, description="Log-distance path-loss exponent.")
    shadowing_sigma_db: float = Field(default=4.0, ge=0.0, description="Log-normal shadow fading sigma (dB).")
    min_snr_db: float = Field(default=-5.0, description="SNR below which PER → 1 (dB).")
    snr_transition_db: float = Field(default=10.0, gt=0, description="SNR range for PER sigmoid transition (dB).")
    base_latency_s: float = Field(default=0.001, ge=0.0, description="Minimum delivery latency (s).")
    jitter_sigma_s: float = Field(default=0.0005, ge=0.0, description="Latency jitter sigma (s).")
    nlos_excess_loss_db: float = Field(default=20.0, ge=0.0, description="Extra path loss when no LoS (dB).")
    los_required: bool = Field(default=False, description="Drop packets immediately when no LoS.")
    seed: int | None = None


class UWBRangeConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.UWBRangingModel`."""

    name: str = "uwb"
    update_rate_hz: float = Field(default=20.0, gt=0, description="Measurement rate (Hz).")
    range_noise_sigma_m: float = Field(default=0.05, ge=0.0, description="Gaussian range noise 1-σ (m).")
    dropout_prob: float = Field(default=0.02, ge=0.0, le=1.0, description="Per-anchor ranging dropout probability.")
    max_range_m: float = Field(default=60.0, gt=0.0, description="Maximum measurable anchor range (m).")
    nlos_bias_m: float = Field(default=0.3, ge=0.0, description="Positive bias applied to NLoS measurements (m).")
    tx_power_dbm: float = Field(default=0.0, description="Reference transmit power used for RSSI estimation (dBm).")
    estimate_position: bool = Field(
        default=True, description="Estimate the platform position from valid anchor ranges."
    )
    seed: int | None = None


class RadarConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.RadarModel`."""

    name: str = "radar"
    update_rate_hz: float = Field(default=15.0, gt=0, description="Measurement rate (Hz).")
    max_range_m: float = Field(default=120.0, gt=0.0, description="Maximum measurable range (m).")
    min_range_m: float = Field(default=0.5, ge=0.0, description="Minimum measurable range (m).")
    azimuth_fov_deg: float = Field(default=120.0, gt=0.0, le=360.0, description="Horizontal field of view (deg).")
    elevation_fov_deg: float = Field(default=40.0, gt=0.0, le=180.0, description="Vertical field of view (deg).")
    range_noise_sigma_m: float = Field(default=0.15, ge=0.0, description="Gaussian range noise 1-σ (m).")
    velocity_noise_sigma_ms: float = Field(
        default=0.08, ge=0.0, description="Gaussian radial-velocity noise 1-σ (m/s)."
    )
    azimuth_noise_deg: float = Field(default=0.4, ge=0.0, description="Gaussian azimuth noise 1-σ (deg).")
    elevation_noise_deg: float = Field(default=0.25, ge=0.0, description="Gaussian elevation noise 1-σ (deg).")
    detection_prob: float = Field(default=0.97, ge=0.0, le=1.0, description="Nominal per-target detection probability.")
    false_alarm_rate: float = Field(default=0.05, ge=0.0, description="Poisson clutter detections per scan.")
    rain_attenuation_db_per_mm_h: float = Field(
        default=0.12,
        ge=0.0,
        description="Additional SNR loss per mm/h of rain rate (dB per mm/h).",
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _range_ordered(self) -> "RadarConfig":
        if self.min_range_m >= self.max_range_m:
            raise ValueError(f"min_range_m ({self.min_range_m}) must be less than max_range_m ({self.max_range_m})")
        return self


# ---------------------------------------------------------------------------
# IMUConfig
# ---------------------------------------------------------------------------


class IMUConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.IMUModel`."""

    name: str = "imu"
    update_rate_hz: float = Field(default=200.0, gt=0, description="IMU output rate (Hz).")
    noise_density_acc: float = Field(
        default=2.0e-3,
        gt=0,
        description="Accelerometer white-noise density (m/s²/√Hz).  Typical MEMS: 2–5 × 10⁻³.",
    )
    noise_density_gyr: float = Field(
        default=1.7e-4,
        gt=0,
        description="Gyroscope white-noise density (rad/s/√Hz).  Typical MEMS: 1–5 × 10⁻⁴.",
    )
    bias_tau_acc_s: float = Field(default=300.0, gt=0, description="Accelerometer bias correlation time (s).")
    bias_sigma_acc: float = Field(default=5.0e-3, ge=0.0, description="Steady-state accelerometer bias sigma (m/s²).")
    bias_tau_gyr_s: float = Field(default=300.0, gt=0, description="Gyroscope bias correlation time (s).")
    bias_sigma_gyr: float = Field(default=1.0e-4, ge=0.0, description="Steady-state gyroscope bias sigma (rad/s).")
    scale_factor_acc: float = Field(
        default=0.0, ge=-1.0, description="Relative accelerometer scale-factor error (≥ −1)."
    )
    scale_factor_gyr: float = Field(default=0.0, ge=-1.0, description="Relative gyroscope scale-factor error (≥ −1).")
    cross_axis_sensitivity_acc: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Accelerometer off-diagonal coupling coefficient (dimensionless). "
            "0.01 = 1 % per-axis crosstalk.  Typical MEMS: 0.005–0.03."
        ),
    )
    cross_axis_sensitivity_gyr: float = Field(
        default=0.0,
        ge=0.0,
        description="Gyroscope off-diagonal coupling coefficient (dimensionless).  Same units as acc.",
    )
    add_gravity: bool = Field(
        default=True,
        description="Add gravity vector (from state['gravity_body']) to acceleration, mimicking specific-force output.",
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# AirspeedConfig
# ---------------------------------------------------------------------------


class AirspeedConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.AirspeedModel`."""

    name: str = "airspeed"
    update_rate_hz: float = Field(default=50.0, gt=0)
    noise_sigma_ms: float = Field(default=0.3, ge=0.0, description="Gaussian white noise sigma on airspeed (m/s).")
    bias_tau_s: float = Field(default=300.0, gt=0, description="Gauss-Markov bias time constant (s).")
    bias_sigma_ms: float = Field(default=0.8, ge=0.0, description="Steady-state bias sigma (m/s).")
    min_detectable_ms: float = Field(
        default=2.0, ge=0.0, description="Dead-band: airspeeds below this are reported as 0 (m/s)."
    )
    max_speed_ms: float = Field(default=200.0, gt=0, description="Saturation speed; output is clamped here (m/s).")
    tube_blockage_prob: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Per-step probability of total tube blockage."
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _speed_range_ordered(self) -> "AirspeedConfig":
        if self.min_detectable_ms >= self.max_speed_ms:
            raise ValueError(
                f"min_detectable_ms ({self.min_detectable_ms}) must be less than max_speed_ms ({self.max_speed_ms})"
            )
        return self


# ---------------------------------------------------------------------------
# RangefinderConfig
# ---------------------------------------------------------------------------


class RangefinderConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.RangefinderModel`."""

    name: str = "rangefinder"
    update_rate_hz: float = Field(default=20.0, gt=0)
    min_range_m: float = Field(default=0.05, ge=0.0, description="Minimum measurable range (m).")
    max_range_m: float = Field(default=12.0, gt=0, description="Maximum measurable range (m).")
    noise_floor_m: float = Field(default=0.02, ge=0.0, description="Fixed Gaussian noise sigma (m).")
    noise_slope: float = Field(
        default=0.01, ge=0.0, description="Range-proportional noise coefficient; total sigma += noise_slope * range."
    )
    accuracy_mode: Literal["additive", "max"] = Field(
        default="additive",
        description=(
            "How floor and slope are combined. 'additive': σ = floor + slope·range. "
            "'max': σ = max(floor, slope·range). Use 'max' for specs that say '±X cm or ±Y% whichever greater'."
        ),
    )
    dropout_prob: float = Field(default=0.001, ge=0.0, le=1.0, description="Per-step missed-measurement probability.")
    resolution_m: float = Field(default=0.001, ge=0.0, description="Quantisation step (m). 0 = off.")
    no_hit_value: float = Field(default=0.0, description="Value returned when out of range or dropout.")
    seed: int | None = None

    @model_validator(mode="after")
    def _range_ordered(self) -> "RangefinderConfig":
        if self.min_range_m >= self.max_range_m:
            raise ValueError(f"min_range_m ({self.min_range_m}) must be less than max_range_m ({self.max_range_m})")
        return self


class UltrasonicArrayConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.UltrasonicArrayModel`."""

    name: str = "ultrasonic"
    update_rate_hz: float = Field(default=15.0, gt=0, description="Measurement rate (Hz).")
    n_beams: int = Field(default=4, ge=1, description="Number of sonar transducers in the array.")
    beam_span_deg: float = Field(default=120.0, ge=0.0, le=360.0, description="Angular coverage of the beam fan (deg).")
    beam_angles_deg: list[float] | None = Field(
        default=None,
        description="Optional explicit per-beam pointing angles (deg). Must match `n_beams` when provided.",
    )
    min_range_m: float = Field(default=0.02, ge=0.0, description="Minimum measurable range (m).")
    max_range_m: float = Field(default=4.5, gt=0.0, description="Maximum measurable range (m).")
    noise_floor_m: float = Field(default=0.005, ge=0.0, description="Fixed Gaussian timing/noise floor (m).")
    noise_slope: float = Field(default=0.01, ge=0.0, description="Range-proportional noise coefficient.")
    dropout_prob: float = Field(default=0.02, ge=0.0, le=1.0, description="Per-beam missed-echo probability.")
    cross_talk_prob: float = Field(
        default=0.03, ge=0.0, le=1.0, description="Probability of echo bleed between adjacent beams."
    )
    beam_width_deg: float = Field(default=25.0, ge=0.0, description="Approximate -3 dB beam width (deg).")
    temperature_compensation: bool = Field(default=True, description="Compensate for ambient speed-of-sound changes.")
    no_hit_value: float = Field(default=0.0, description="Value returned when no valid echo is observed.")
    seed: int | None = None

    @model_validator(mode="after")
    def _range_ordered(self) -> "UltrasonicArrayConfig":
        if self.min_range_m >= self.max_range_m:
            raise ValueError(f"min_range_m ({self.min_range_m}) must be less than max_range_m ({self.max_range_m})")
        return self

    @model_validator(mode="after")
    def _angles_match(self) -> "UltrasonicArrayConfig":
        if self.beam_angles_deg is not None and len(self.beam_angles_deg) != self.n_beams:
            raise ValueError(
                f"beam_angles_deg must have exactly n_beams ({self.n_beams}) values, got {len(self.beam_angles_deg)}"
            )
        return self


class ImagingSonarConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.ImagingSonarModel`."""

    name: str = "imaging_sonar"
    update_rate_hz: float = Field(default=8.0, gt=0, description="Ping / frame rate (Hz).")
    azimuth_bins: int = Field(default=96, ge=1, description="Number of azimuth columns in the acoustic image.")
    range_bins: int = Field(default=128, ge=1, description="Number of range bins / rows in the acoustic image.")
    azimuth_fov_deg: float = Field(default=120.0, gt=0.0, le=360.0, description="Horizontal field of view (deg).")
    min_range_m: float = Field(default=0.5, ge=0.0, description="Minimum measurable range (m).")
    max_range_m: float = Field(default=30.0, gt=0.0, description="Maximum measurable range (m).")
    range_noise_sigma_m: float = Field(default=0.05, ge=0.0, description="Gaussian range noise 1-σ (m).")
    azimuth_noise_deg: float = Field(default=1.0, ge=0.0, description="Gaussian azimuth noise 1-σ (deg).")
    speckle_sigma: float = Field(default=0.04, ge=0.0, description="Additive speckle-like intensity noise amplitude.")
    attenuation_db_per_m: float = Field(default=0.12, ge=0.0, description="Water-column attenuation coefficient.")
    false_alarm_rate: float = Field(default=0.02, ge=0.0, description="Poisson clutter blobs per frame.")
    seed: int | None = None

    @model_validator(mode="after")
    def _range_ordered(self) -> "ImagingSonarConfig":
        if self.min_range_m >= self.max_range_m:
            raise ValueError(f"min_range_m ({self.min_range_m}) must be less than max_range_m ({self.max_range_m})")
        return self


class SideScanSonarConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.SideScanSonarModel`."""

    name: str = "side_scan"
    update_rate_hz: float = Field(default=4.0, gt=0, description="Ping / line rate (Hz).")
    range_bins: int = Field(default=128, ge=1, description="Number of slant-range bins per side.")
    min_range_m: float = Field(default=0.5, ge=0.0, description="Minimum measurable range (m).")
    max_range_m: float = Field(default=40.0, gt=0.0, description="Maximum measurable range (m).")
    range_noise_sigma_m: float = Field(default=0.06, ge=0.0, description="Gaussian slant-range noise 1-σ (m).")
    speckle_sigma: float = Field(default=0.04, ge=0.0, description="Additive speckle-like strip noise amplitude.")
    attenuation_db_per_m: float = Field(default=0.10, ge=0.0, description="Water-column attenuation coefficient.")
    false_alarm_rate: float = Field(default=0.02, ge=0.0, description="Poisson clutter spikes per ping.")
    seed: int | None = None

    @model_validator(mode="after")
    def _range_ordered(self) -> "SideScanSonarConfig":
        if self.min_range_m >= self.max_range_m:
            raise ValueError(f"min_range_m ({self.min_range_m}) must be less than max_range_m ({self.max_range_m})")
        return self


class DVLConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.DVLModel`."""

    name: str = "dvl"
    update_rate_hz: float = Field(default=5.0, gt=0, description="Bottom-track update rate (Hz).")
    n_beams: int = Field(default=4, ge=1, description="Number of DVL transducer beams.")
    beam_angle_deg: float = Field(default=30.0, ge=0.0, lt=90.0, description="Off-nadir beam angle (deg).")
    min_altitude_m: float = Field(default=0.2, ge=0.0, description="Minimum valid altitude above bottom (m).")
    max_altitude_m: float = Field(default=80.0, gt=0.0, description="Maximum bottom-lock altitude (m).")
    velocity_noise_sigma_ms: float = Field(default=0.01, ge=0.0, description="Gaussian velocity noise 1-σ (m/s).")
    range_noise_sigma_m: float = Field(default=0.02, ge=0.0, description="Gaussian beam-range noise 1-σ (m).")
    dropout_prob: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Probability of losing bottom lock on a ping."
    )
    water_track_blend: float = Field(
        default=0.20, ge=0.0, le=1.0, description="Blend factor for water-track fallback when bottom lock is lost."
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _range_ordered(self) -> "DVLConfig":
        if self.min_altitude_m >= self.max_altitude_m:
            raise ValueError(
                f"min_altitude_m ({self.min_altitude_m}) must be less than max_altitude_m ({self.max_altitude_m})"
            )
        return self


class AcousticCurrentProfilerConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.AcousticCurrentProfilerModel`."""

    name: str = "current_profiler"
    update_rate_hz: float = Field(default=2.0, gt=0, description="Water-column profile rate (Hz).")
    n_cells: int = Field(default=8, ge=1, description="Number of depth bins / cells in the current profile.")
    min_depth_m: float = Field(default=1.0, ge=0.0, description="Near-field depth of the first cell (m).")
    max_depth_m: float = Field(default=20.0, gt=0.0, description="Maximum sampled depth in the water column (m).")
    velocity_noise_sigma_ms: float = Field(
        default=0.02, ge=0.0, description="Gaussian per-axis current noise 1-σ (m/s)."
    )
    attenuation_per_m: float = Field(default=0.03, ge=0.0, description="Depth-dependent attenuation factor per metre.")
    false_bin_rate: float = Field(
        default=0.01, ge=0.0, le=1.0, description="Probability that a depth bin returns no valid estimate."
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _depth_ordered(self) -> "AcousticCurrentProfilerConfig":
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError(f"min_depth_m ({self.min_depth_m}) must be less than max_depth_m ({self.max_depth_m})")
        return self


# ---------------------------------------------------------------------------
# Environmental sensing configs
# ---------------------------------------------------------------------------


class ThermometerConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.ThermometerModel`."""

    name: str = "thermometer"
    update_rate_hz: float = Field(default=4.0, gt=0, description="Measurement rate (Hz).")
    noise_sigma_c: float = Field(default=0.08, ge=0.0, description="Gaussian temperature noise 1-σ (°C).")
    bias_tau_s: float = Field(default=300.0, gt=0, description="Gauss-Markov bias correlation time (s).")
    bias_sigma_c: float = Field(default=0.2, ge=0.0, description="Steady-state bias sigma (°C).")
    response_tau_s: float = Field(default=2.0, gt=0, description="Thermal response time constant (s).")
    seed: int | None = None


class HygrometerConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.HygrometerModel`."""

    name: str = "hygrometer"
    update_rate_hz: float = Field(default=2.0, gt=0, description="Measurement rate (Hz).")
    noise_sigma_pct: float = Field(default=1.5, ge=0.0, description="Gaussian relative humidity noise 1-σ (%RH).")
    bias_tau_s: float = Field(default=400.0, gt=0, description="Gauss-Markov bias correlation time (s).")
    bias_sigma_pct: float = Field(default=3.0, ge=0.0, description="Steady-state humidity bias sigma (%RH).")
    response_tau_s: float = Field(default=4.0, gt=0, description="Humidity response time constant (s).")
    seed: int | None = None


class LightSensorConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.LightSensorModel`."""

    name: str = "light_sensor"
    update_rate_hz: float = Field(default=10.0, gt=0, description="Measurement rate (Hz).")
    noise_sigma_ratio: float = Field(
        default=0.04,
        ge=0.0,
        description="Relative Gaussian noise sigma as a fraction of the illuminance reading.",
    )
    min_lux: float = Field(default=0.0, ge=0.0, description="Minimum reportable illuminance (lux).")
    max_lux: float = Field(default=120_000.0, gt=0.0, description="Maximum reportable illuminance (lux).")
    response_tau_s: float = Field(default=0.4, gt=0, description="Photodiode response time constant (s).")
    seed: int | None = None


class GasSensorConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.GasSensorModel`."""

    name: str = "gas_sensor"
    update_rate_hz: float = Field(default=5.0, gt=0, description="Measurement rate (Hz).")
    background_ppm: float = Field(default=420.0, ge=0.0, description="Background concentration (ppm).")
    noise_sigma_ppm: float = Field(default=8.0, ge=0.0, description="Gaussian concentration noise 1-σ (ppm).")
    response_tau_s: float = Field(default=3.0, gt=0, description="Sensor response time constant (s).")
    alarm_threshold_ppm: float = Field(default=900.0, ge=0.0, description="Alarm threshold (ppm).")
    max_ppm: float = Field(default=10_000.0, gt=0.0, description="Maximum reportable concentration (ppm).")
    plume_sigma_m: float = Field(default=0.8, gt=0.0, description="Default lateral plume width when sources are used.")
    seed: int | None = None


class AnemometerConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.AnemometerModel`."""

    name: str = "anemometer"
    update_rate_hz: float = Field(default=10.0, gt=0, description="Measurement rate (Hz).")
    noise_sigma_ms: float = Field(default=0.15, ge=0.0, description="Per-axis Gaussian wind noise 1-σ (m/s).")
    direction_noise_deg: float = Field(default=2.0, ge=0.0, description="Heading noise 1-σ (degrees).")
    measure_relative_wind: bool = Field(
        default=False,
        description="Measure relative airflow by subtracting platform velocity instead of ambient wind only.",
    )
    max_speed_ms: float = Field(default=80.0, gt=0.0, description="Maximum reportable wind speed (m/s).")
    seed: int | None = None


# ---------------------------------------------------------------------------
# BarometerConfig
# ---------------------------------------------------------------------------


class BarometerConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.BarometerModel`."""

    name: str = "barometer"
    update_rate_hz: float = Field(default=50.0, gt=0)
    noise_sigma_m: float = Field(default=0.3, ge=0.0, description="Gaussian altitude noise sigma (m).")
    bias_tau_s: float = Field(default=300.0, gt=0, description="Gauss-Markov bias time constant (s).")
    bias_sigma_m: float = Field(default=1.5, ge=0.0, description="Steady-state altitude bias sigma (m).")
    ground_alt_m: float = Field(
        default=0.0,
        description="Reference ground altitude (m MSL) added to the ENU z-axis to obtain MSL altitude.",
    )
    resolution_m: float = Field(default=0.0, ge=0.0, description="Quantisation step (m). 0 = off.")
    seed: int | None = None


# ---------------------------------------------------------------------------
# MagnetometerConfig
# ---------------------------------------------------------------------------


class MagnetometerConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.MagnetometerModel`."""

    name: str = "magnetometer"
    update_rate_hz: float = Field(default=100.0, gt=0)
    noise_sigma_ut: float = Field(default=0.5, ge=0.0, description="Per-axis Gaussian white noise sigma (\u00b5T).")
    field_amplitude_ut: float = Field(
        default=50.0, gt=0, description="Earth field magnitude (\u00b5T). Typical: 25\u201365 \u00b5T."
    )
    declination_deg: float = Field(
        default=0.0,
        ge=-180.0,
        le=180.0,
        description="Magnetic declination at the simulation origin (degrees east of north).",
    )
    inclination_deg: float = Field(
        default=60.0,
        ge=-90.0,
        le=90.0,
        description="Magnetic dip angle at the simulation origin (degrees below horizontal).",
    )
    hard_iron_ut: list[float] = Field(
        default=[0.0, 0.0, 0.0],
        description="Per-axis hard-iron bias (\u00b5T). Must have exactly 3 elements.",
    )
    soft_iron_scale: list[float] = Field(
        default=[1.0, 1.0, 1.0],
        description="Per-axis soft-iron scale factors (> 0). Must have exactly 3 elements.",
    )
    seed: int | None = None

    @field_validator("hard_iron_ut", "soft_iron_scale")
    @classmethod
    def _three_elements(cls, v: list[float]) -> list[float]:
        if len(v) != 3:
            raise ValueError(f"must have exactly 3 elements, got {len(v)}")
        return v

    @field_validator("soft_iron_scale")
    @classmethod
    def _positive_scale(cls, v: list[float]) -> list[float]:
        if any(s <= 0.0 for s in v):
            raise ValueError("all soft_iron_scale elements must be positive")
        return v


# ---------------------------------------------------------------------------
# OpticalFlowConfig
# ---------------------------------------------------------------------------


class OpticalFlowConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.OpticalFlowModel`."""

    name: str = "optical_flow"
    update_rate_hz: float = Field(default=100.0, gt=0, description="Output rate (Hz).")
    noise_floor_rad_s: float = Field(
        default=0.005, ge=0.0, description="Fixed Gaussian noise floor on flow rate (rad/s per axis)."
    )
    noise_slope: float = Field(
        default=0.05, ge=0.0, description="Motion-blur noise coefficient (sigma increase per rad/s of flow)."
    )
    max_detection_rad_s: float = Field(
        default=4.0,
        gt=0,
        description="Maximum detectable flow rate (rad/s); above this quality=0 and flow is invalid.",
    )
    nominal_quality_height_m: float = Field(
        default=2.0,
        gt=0,
        description="Height below which tracking quality is at its peak (m).",
    )
    max_quality_height_m: float = Field(
        default=30.0,
        gt=0,
        description="Height above which quality drops to zero (m).",
    )
    base_quality: int = Field(default=220, ge=0, le=255, description="Maximum quality under ideal conditions (0–255).")
    low_quality_threshold: int = Field(
        default=25,
        ge=0,
        le=255,
        description="Quality ≤ this is considered unreliable (consumer should discard the measurement).",
    )
    seed: int | None = None

    @model_validator(mode="after")
    def _height_range_ordered(self) -> "OpticalFlowConfig":
        if self.max_quality_height_m <= self.nominal_quality_height_m:
            raise ValueError(
                f"max_quality_height_m ({self.max_quality_height_m}) must be greater than "
                f"nominal_quality_height_m ({self.nominal_quality_height_m})"
            )
        return self


# ---------------------------------------------------------------------------
# BatteryConfig
# ---------------------------------------------------------------------------


class BatteryConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.BatteryModel`."""

    name: str = "battery"
    update_rate_hz: float = Field(default=10.0, gt=0, description="Observation output rate (Hz).")
    n_cells: int = Field(default=4, ge=1, description="Number of cells in series (e.g. 4 for 4S).")
    capacity_mah: float = Field(default=5000.0, gt=0, description="Rated capacity (mAh).")
    cell_chemistry: Literal["lipo", "lihv"] = Field(
        default="lipo",
        description="Cell chemistry: 'lipo' (max 4.20 V/cell) or 'lihv' (max 4.35 V/cell).",
    )
    internal_resistance_ohm: float = Field(default=0.012, ge=0.0, description="Pack-level DC internal resistance (Ω).")
    v_warn_per_cell_v: float = Field(default=3.65, gt=0, description="Per-cell voltage threshold for is_low alert (V).")
    current_noise_a: float = Field(default=0.05, ge=0.0, description="Gaussian current measurement noise sigma (A).")
    voltage_noise_v: float = Field(default=0.01, ge=0.0, description="Gaussian voltage measurement noise sigma (V).")
    initial_soc: float = Field(default=1.0, ge=0.0, le=1.0, description="Starting state of charge (0–1).")
    seed: int | None = None


# ---------------------------------------------------------------------------
# StereoCameraConfig
# ---------------------------------------------------------------------------


class StereoCameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.StereoCameraModel`."""

    name: str = "stereo_camera"
    update_rate_hz: float = Field(default=30.0, gt=0, description="Frame rate (Hz), shared by both eyes.")
    resolution: tuple[int, int] = Field(default=(640, 480), description="(width, height) in pixels per eye.")
    baseline_m: float = Field(
        default=0.06,
        gt=0,
        description="Physical separation between optical centres (m). Typical: 0.06 (ZED 2), 0.12 (MYNT EYE D).",
    )
    hfov_deg: float = Field(
        default=90.0,
        gt=0.0,
        lt=180.0,
        description="Horizontal field of view (degrees). Used to compute focal length: f = W / (2·tan(hfov/2)).",
    )
    disparity_noise_sigma_px: float = Field(
        default=0.5,
        ge=0.0,
        description="Gaussian noise sigma on disparity (pixels). Typical SGBM: 0.3–1.0 px.",
    )
    min_depth_m: float = Field(default=0.05, gt=0, description="Minimum valid depth — closer pixels are invalid.")
    max_depth_m: float = Field(default=20.0, gt=0, description="Maximum measurable depth — farther pixels are invalid.")
    # Per-eye camera corruption can be tuned via iso / vignetting_strength etc.
    # These mirror the most commonly varied CameraModel parameters.
    iso: float = Field(default=100.0, gt=0, description="Effective ISO applied to both eyes.")
    read_noise_sigma: float = Field(default=1.5, ge=0, description="Read-noise sigma (electrons) per eye.")
    vignetting_strength: float = Field(default=0.0, ge=0.0, description="Radial vignetting (0 = off).")
    chromatic_aberration_px: float = Field(default=0.0, ge=0.0, description="CA chromatic shift at corner (px).")
    seed: int | None = Field(default=None, description="RNG seed; each eye gets an independent child seed.")

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v

    @model_validator(mode="after")
    def _depth_range_order(self) -> "StereoCameraConfig":
        if self.min_depth_m >= self.max_depth_m:
            raise ValueError(f"min_depth_m ({self.min_depth_m}) must be < max_depth_m ({self.max_depth_m})")
        return self

    def to_camera_kwargs(self) -> dict:
        """Return kwargs suitable for constructing a :class:`~genesis.sensors.CameraModel`."""
        return {
            "update_rate_hz": self.update_rate_hz,
            "resolution": self.resolution,
            "iso": self.iso,
            "read_noise_sigma": self.read_noise_sigma,
            "vignetting_strength": self.vignetting_strength,
            "chromatic_aberration_px": self.chromatic_aberration_px,
        }


# ---------------------------------------------------------------------------
# WheelOdometryConfig
# ---------------------------------------------------------------------------


class WheelOdometryConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.WheelOdometryModel`."""

    name: str = "wheel_odometry"
    update_rate_hz: float = Field(default=50.0, gt=0, description="Odometry output rate (Hz).")
    pos_noise_sigma_m: float = Field(
        default=0.002, ge=0.0, description="1-\u03c3 Gaussian noise on each position-delta axis per step (m)."
    )
    heading_noise_sigma_rad: float = Field(
        default=0.001, ge=0.0, description="1-\u03c3 Gaussian noise on heading change per step (rad)."
    )
    slip_sigma: float = Field(
        default=0.005,
        ge=0.0,
        description="Wheel-slip standard deviation as a fraction of instantaneous speed (dimensionless).",
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# ForceTorqueConfig
# ---------------------------------------------------------------------------


class ForceTorqueConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.ForceTorqueSensorModel`."""

    name: str = "force_torque"
    update_rate_hz: float = Field(default=1000.0, gt=0, description="Output rate (Hz).")
    force_noise_sigma_n: float = Field(default=0.05, ge=0.0, description="Per-axis Gaussian force noise 1-\u03c3 (N).")
    torque_noise_sigma_nm: float = Field(
        default=0.002, ge=0.0, description="Per-axis Gaussian torque noise 1-\u03c3 (Nm)."
    )
    force_bias_n: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Constant per-axis force bias (N)."
    )
    torque_bias_nm: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Constant per-axis torque bias (Nm)."
    )
    force_range_n: float = Field(default=200.0, gt=0, description="Force saturation threshold per axis (N).")
    torque_range_nm: float = Field(default=10.0, gt=0, description="Torque saturation threshold per axis (Nm).")
    seed: int | None = None


# ---------------------------------------------------------------------------
# JointStateConfig
# ---------------------------------------------------------------------------


class JointStateConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.JointStateSensor`."""

    name: str = "joint_state"
    update_rate_hz: float = Field(default=1000.0, gt=0, description="Joint encoder output rate (Hz).")
    pos_noise_sigma_rad: float = Field(
        default=0.0001, ge=0.0, description="Per-joint Gaussian position noise 1-σ (rad)."
    )
    vel_noise_sigma_rads: float = Field(
        default=0.001, ge=0.0, description="Per-joint Gaussian velocity noise 1-σ (rad/s)."
    )
    torque_noise_sigma_nm: float = Field(default=0.01, ge=0.0, description="Per-joint Gaussian torque noise 1-σ (Nm).")
    velocity_filter_alpha: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="First-order LP smoothing factor for velocity (0 = off).",
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# ContactSensorConfig
# ---------------------------------------------------------------------------


class ContactSensorConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.ContactSensor`."""

    name: str = "contact"
    update_rate_hz: float = Field(default=200.0, gt=0, description="Contact sensor output rate (Hz).")
    force_threshold_n: float = Field(
        default=0.5, ge=0.0, description="Force magnitude above which contact is declared (N)."
    )
    noise_sigma_n: float = Field(
        default=0.02, ge=0.0, description="Gaussian noise 1-σ on measured force magnitude (N)."
    )
    force_range_n: float = Field(
        default=50.0, gt=0, description="Maximum measurable force; output clipped to this value (N)."
    )
    debounce_steps: int = Field(
        default=0, ge=0, description="Minimum consecutive steps new state must persist before toggling."
    )
    seed: int | None = None


# ---------------------------------------------------------------------------
# DepthCameraConfig
# ---------------------------------------------------------------------------


class DepthCameraConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.DepthCameraModel`."""

    name: str = "depth_camera"
    update_rate_hz: float = Field(default=30.0, gt=0, description="Depth frame rate (Hz).")
    resolution: tuple[int, int] = Field(
        default=(640, 480), description="(width, height) in pixels; used as fall-back when no depth image provided."
    )
    depth_noise_sigma_m: float = Field(default=0.002, ge=0.0, description="Constant Gaussian depth noise 1-σ (m).")
    depth_noise_scale_z: float = Field(
        default=0.0005,
        ge=0.0,
        description="Range-dependent noise coefficient; additional sigma = scale × z² (m).",
    )
    missing_edge_px: int = Field(default=2, ge=0, description="Width of invalid-depth border in pixels (0 = disabled).")
    min_depth_m: float = Field(default=0.2, gt=0, description="Minimum measurable range (m).")
    max_depth_m: float = Field(default=10.0, gt=0, description="Maximum measurable range (m).")
    seed: int | None = None

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


class TactileArrayConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.TactileArraySensor`."""

    name: str = "tactile_array"
    update_rate_hz: float = Field(default=200.0, gt=0, description="Sensor output rate (Hz).")
    resolution: tuple[int, int] = Field(default=(4, 4), description="(width, height) taxel grid dimensions.")
    max_pressure_pa: float = Field(default=500_000.0, gt=0, description="Saturation pressure (Pa).")
    noise_sigma_pa: float = Field(default=1000.0, ge=0.0, description="Per-taxel Gaussian noise 1-σ (Pa).")
    contact_threshold_pa: float = Field(
        default=5000.0, ge=0.0, description="Minimum pressure (Pa) to mark a taxel as in contact."
    )
    taxel_area_mm2: float = Field(default=4.0, gt=0, description="Physical taxel area (mm²) for force integration.")
    dead_zone_fraction: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fraction of taxels permanently disabled [0, 1]."
    )
    seed: int | None = None

    @field_validator("resolution")
    @classmethod
    def _positive_resolution(cls, v: tuple[int, int]) -> tuple[int, int]:
        w, h = v
        if w <= 0 or h <= 0:
            raise ValueError(f"resolution must be positive, got {v}")
        return v


class CurrentSensorConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.CurrentSensor`."""

    name: str = "current"
    update_rate_hz: float = Field(default=200.0, gt=0, description="Sampling rate (Hz).")
    noise_sigma_a: float = Field(default=0.05, ge=0.0, description="Gaussian current noise 1-σ (A).")
    offset_a: float = Field(default=0.0, description="Fixed DC offset bias (A).")
    range_a: float = Field(default=50.0, gt=0, description="Maximum measurable current (A).")
    voltage_nominal_v: float = Field(
        default=24.0, gt=0, description="Fallback supply voltage (V) when state lacks 'voltage_v'."
    )
    seed: int | None = None


class RPMSensorConfig(BaseModel):
    """Configuration for :class:`~genesis.sensors.RPMSensor`."""

    name: str = "rpm"
    update_rate_hz: float = Field(default=500.0, gt=0, description="Measurement rate (Hz).")
    cpr: int = Field(default=1024, ge=0, description="Counts per revolution (0 = no quantisation).")
    noise_sigma_rpm: float = Field(default=5.0, ge=0.0, description="Gaussian speed noise 1-σ (RPM).")
    rpm_range: float = Field(default=20_000.0, gt=0, description="Maximum measurable speed (RPM).")
    seed: int | None = None


# ---------------------------------------------------------------------------
# SensorSuiteConfig
# ---------------------------------------------------------------------------


class SensorSuiteConfig(BaseModel):
    """
    Top-level configuration for a complete :class:`~genesis.sensors.SensorSuite`.

    Each field corresponds to one sensor in the suite.  Set a field to
    ``None`` to disable that sensor entirely.

    Example
    -------
    ::

        cfg = SensorSuiteConfig(
            rgb=CameraConfig(iso=400, jpeg_quality=60),
            gnss=GNSSConfig(noise_m=0.5),
            lidar=None,     # disabled
            event=None,
            thermal=None,
            radio=None,
        )
        suite = SensorSuite.from_config(cfg)
    """

    rgb: CameraConfig | None = Field(default_factory=CameraConfig, description="RGB camera (None = disabled).")
    stereo_camera: StereoCameraConfig | None = Field(default=None, description="Stereo camera pair (None = disabled).")
    event: EventCameraConfig | None = Field(default=None, description="Event camera (None = disabled).")
    thermal: ThermalCameraConfig | None = Field(default=None, description="Thermal camera (None = disabled).")
    lidar: LidarConfig | None = Field(default_factory=LidarConfig, description="LiDAR (None = disabled).")
    gnss: GNSSConfig | None = Field(default_factory=GNSSConfig, description="GNSS (None = disabled).")
    radio: RadioConfig | None = Field(default_factory=RadioConfig, description="Radio link (None = disabled).")
    uwb: UWBRangeConfig | None = Field(default=None, description="UWB anchor-ranging sensor (None = disabled).")
    radar: RadarConfig | None = Field(default=None, description="Radar detection sensor (None = disabled).")
    imu: IMUConfig | None = Field(default_factory=IMUConfig, description="IMU (None = disabled).")
    barometer: BarometerConfig | None = Field(default=None, description="Barometer (None = disabled).")
    magnetometer: MagnetometerConfig | None = Field(default=None, description="Magnetometer (None = disabled).")
    thermometer: ThermometerConfig | None = Field(
        default=None, description="Ambient temperature sensor (None = disabled)."
    )
    hygrometer: HygrometerConfig | None = Field(default=None, description="Relative humidity sensor (None = disabled).")
    light_sensor: LightSensorConfig | None = Field(
        default=None, description="Ambient illuminance sensor (None = disabled)."
    )
    gas_sensor: GasSensorConfig | None = Field(default=None, description="Gas concentration sensor (None = disabled).")
    anemometer: AnemometerConfig | None = Field(default=None, description="Wind sensor (None = disabled).")
    airspeed: AirspeedConfig | None = Field(default=None, description="Pitot airspeed sensor (None = disabled).")
    rangefinder: RangefinderConfig | None = Field(default=None, description="1-D rangefinder (None = disabled).")
    ultrasonic: UltrasonicArrayConfig | None = Field(
        default=None, description="Ultrasonic proximity array (None = disabled)."
    )
    imaging_sonar: ImagingSonarConfig | None = Field(
        default=None, description="Forward-looking imaging sonar (None = disabled)."
    )
    side_scan: SideScanSonarConfig | None = Field(
        default=None, description="Side-scan sonar strip imager (None = disabled)."
    )
    dvl: DVLConfig | None = Field(default=None, description="Doppler velocity log (None = disabled).")
    current_profiler: AcousticCurrentProfilerConfig | None = Field(
        default=None, description="Water-column acoustic current profiler (None = disabled)."
    )
    optical_flow: OpticalFlowConfig | None = Field(
        default=None, description="Downward-facing optical flow sensor (None = disabled)."
    )
    battery: BatteryConfig | None = Field(
        default=None, description="Battery voltage/current monitor (None = disabled)."
    )
    wheel_odometry: WheelOdometryConfig | None = Field(
        default=None, description="Wheel odometry encoder model (None = disabled)."
    )
    force_torque: ForceTorqueConfig | None = Field(
        default=None, description="6-axis wrist force/torque sensor (None = disabled)."
    )
    joint_state: JointStateConfig | None = Field(
        default=None, description="Joint position/velocity/torque encoder (None = disabled)."
    )
    contact: ContactSensorConfig | None = Field(
        default=None, description="Binary + analog contact/tactile sensor (None = disabled)."
    )
    depth_camera: DepthCameraConfig | None = Field(
        default=None, description="Depth camera / RGBD sensor (None = disabled)."
    )
    tactile_array: TactileArrayConfig | None = Field(
        default=None, description="2-D pressure taxel array sensor (None = disabled)."
    )
    current: CurrentSensorConfig | None = Field(
        default=None, description="Current and power monitor (None = disabled)."
    )
    rpm: RPMSensorConfig | None = Field(default=None, description="Motor / rotor RPM sensor (None = disabled).")

    @classmethod
    def minimal(cls) -> "SensorSuiteConfig":
        """Return a config with only GNSS enabled (lightest default)."""
        return cls(
            rgb=None,
            stereo_camera=None,
            event=None,
            thermal=None,
            lidar=None,
            radio=None,
            uwb=None,
            radar=None,
            imu=None,
            barometer=None,
            magnetometer=None,
            thermometer=None,
            hygrometer=None,
            light_sensor=None,
            gas_sensor=None,
            anemometer=None,
            airspeed=None,
            rangefinder=None,
            ultrasonic=None,
            imaging_sonar=None,
            side_scan=None,
            dvl=None,
            current_profiler=None,
            optical_flow=None,
            battery=None,
            wheel_odometry=None,
            force_torque=None,
            joint_state=None,
            contact=None,
            depth_camera=None,
            tactile_array=None,
            current=None,
            rpm=None,
        )

    @classmethod
    def all_disabled(cls) -> "SensorSuiteConfig":
        """Return a config with every sensor disabled."""
        return cls(
            rgb=None,
            stereo_camera=None,
            event=None,
            thermal=None,
            lidar=None,
            gnss=None,
            radio=None,
            uwb=None,
            radar=None,
            imu=None,
            barometer=None,
            magnetometer=None,
            thermometer=None,
            hygrometer=None,
            light_sensor=None,
            gas_sensor=None,
            anemometer=None,
            airspeed=None,
            rangefinder=None,
            ultrasonic=None,
            imaging_sonar=None,
            side_scan=None,
            dvl=None,
            current_profiler=None,
            optical_flow=None,
            battery=None,
            wheel_odometry=None,
            force_torque=None,
            joint_state=None,
            contact=None,
            depth_camera=None,
            tactile_array=None,
            current=None,
            rpm=None,
        )

    @classmethod
    def full(cls) -> "SensorSuiteConfig":
        """Return a config with every sensor enabled at default settings."""
        return cls(
            rgb=CameraConfig(),
            stereo_camera=StereoCameraConfig(),
            event=EventCameraConfig(),
            thermal=ThermalCameraConfig(),
            lidar=LidarConfig(),
            gnss=GNSSConfig(),
            radio=RadioConfig(),
            uwb=UWBRangeConfig(),
            radar=RadarConfig(),
            imu=IMUConfig(),
            barometer=BarometerConfig(),
            magnetometer=MagnetometerConfig(),
            thermometer=ThermometerConfig(),
            hygrometer=HygrometerConfig(),
            light_sensor=LightSensorConfig(),
            gas_sensor=GasSensorConfig(),
            anemometer=AnemometerConfig(),
            airspeed=AirspeedConfig(),
            rangefinder=RangefinderConfig(),
            ultrasonic=UltrasonicArrayConfig(),
            imaging_sonar=ImagingSonarConfig(),
            side_scan=SideScanSonarConfig(),
            dvl=DVLConfig(),
            current_profiler=AcousticCurrentProfilerConfig(),
            optical_flow=OpticalFlowConfig(),
            battery=BatteryConfig(),
            wheel_odometry=WheelOdometryConfig(),
            force_torque=ForceTorqueConfig(),
            joint_state=JointStateConfig(),
            contact=ContactSensorConfig(),
            depth_camera=DepthCameraConfig(),
            tactile_array=TactileArrayConfig(),
            current=CurrentSensorConfig(),
            rpm=RPMSensorConfig(),
        )


__all__ = [
    "AcousticCurrentProfilerConfig",
    "AirspeedConfig",
    "AnemometerConfig",
    "BarometerConfig",
    "BatteryConfig",
    "CameraConfig",
    "ContactSensorConfig",
    "CurrentSensorConfig",
    "DVLConfig",
    "DepthCameraConfig",
    "EventCameraConfig",
    "ForceTorqueConfig",
    "GasSensorConfig",
    "GNSSConfig",
    "HygrometerConfig",
    "IMUConfig",
    "ImagingSonarConfig",
    "JointStateConfig",
    "LidarConfig",
    "LightSensorConfig",
    "MagnetometerConfig",
    "OpticalFlowConfig",
    "RadarConfig",
    "RPMSensorConfig",
    "RadioConfig",
    "RangefinderConfig",
    "SensorSuiteConfig",
    "SideScanSonarConfig",
    "StereoCameraConfig",
    "UltrasonicArrayConfig",
    "TactileArrayConfig",
    "ThermalCameraConfig",
    "ThermometerConfig",
    "UWBRangeConfig",
    "WheelOdometryConfig",
]
