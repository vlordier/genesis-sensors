"""Selectable sensor fidelity levels for CPU/GPU tradeoffs."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field


class FidelityLevel(enum.IntEnum):
    """Sensor fidelity levels — higher = more effects, more compute.

    - LOW: Minimal noise, no physical effects. Fastest. Use for CI/smoke tests.
    - STANDARD: Datasheet-level noise, basic effects. Current default.
    - HIGH: All physical effects (vibration, motion distortion, misalignment).
    - ULTRA: Ray-tracing-level physics (carrier-phase, specular, multi-constellation).
    """

    LOW = 0
    STANDARD = 1
    HIGH = 2
    ULTRA = 3


# Per-feature activation levels.
# feature_name → minimum FidelityLevel required
DEFAULT_FIDELITY_MAP: dict[str, FidelityLevel] = {
    # Global features
    "gpu_acceleration": FidelityLevel.STANDARD,
    # IMU
    "imu_bias_drift": FidelityLevel.STANDARD,
    "imu_scale_factor": FidelityLevel.STANDARD,
    "imu_cross_axis": FidelityLevel.STANDARD,
    "imu_g_sensitivity": FidelityLevel.HIGH,
    "imu_vibration": FidelityLevel.HIGH,
    "imu_misalignment": FidelityLevel.HIGH,
    "imu_coning_sculling": FidelityLevel.ULTRA,
    "imu_temperature_bias": FidelityLevel.STANDARD,
    "imu_adc_quantization": FidelityLevel.STANDARD,
    # GNSS
    "gnss_multipath": FidelityLevel.STANDARD,
    "gnss_ionospheric": FidelityLevel.STANDARD,
    "gnss_tropospheric": FidelityLevel.STANDARD,
    "gnss_sbas_dgps": FidelityLevel.HIGH,
    "gnss_carrier_phase": FidelityLevel.ULTRA,
    "gnss_multi_constellation": FidelityLevel.ULTRA,
    # LiDAR
    "lidar_divergence": FidelityLevel.STANDARD,
    "lidar_range_noise": FidelityLevel.STANDARD,
    "lidar_reflectance_noise": FidelityLevel.STANDARD,
    "lidar_multi_return": FidelityLevel.STANDARD,
    "lidar_motion_distortion": FidelityLevel.HIGH,
    "lidar_specular_reflection": FidelityLevel.ULTRA,
    "lidar_ambient_interference": FidelityLevel.ULTRA,
    # Camera
    "camera_shot_noise": FidelityLevel.STANDARD,
    "camera_read_noise": FidelityLevel.STANDARD,
    "camera_vignetting": FidelityLevel.STANDARD,
    "camera_lens_distortion": FidelityLevel.STANDARD,
    "camera_chromatic_aberration": FidelityLevel.STANDARD,
    "camera_rolling_shutter": FidelityLevel.HIGH,
    "camera_motion_blur": FidelityLevel.HIGH,
    "camera_blooming": FidelityLevel.HIGH,
    "camera_dead_pixels": FidelityLevel.STANDARD,
    "camera_bayer_pattern": FidelityLevel.ULTRA,
    "camera_dark_current": FidelityLevel.ULTRA,
    "camera_lens_flare": FidelityLevel.ULTRA,
    # Wheel Odometry
    "odom_slip": FidelityLevel.STANDARD,
    "odom_position_noise": FidelityLevel.STANDARD,
    "odom_surface_friction": FidelityLevel.HIGH,
}


@dataclass
class FidelityConfig:
    """Per-sensor fidelity configuration."""

    level: FidelityLevel = FidelityLevel.STANDARD
    overrides: dict[str, FidelityLevel] = field(default_factory=dict)
    feature_map: dict[str, FidelityLevel] = field(default_factory=lambda: dict(DEFAULT_FIDELITY_MAP))

    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled at the current fidelity level."""
        required = self.overrides.get(feature, self.feature_map.get(feature, FidelityLevel.STANDARD))
        return self.level >= required

    def configure_sensor(self, features: dict[str, bool | None]) -> dict[str, bool]:
        """Resolve per-sensor feature flags from fidelity level + overrides."""
        result: dict[str, bool] = {}
        for feat, manual in features.items():
            if manual is not None:
                result[feat] = manual
            else:
                result[feat] = self.is_enabled(feat)
        return result


def get_fidelity_config(level: FidelityLevel | str = "standard") -> FidelityConfig:
    """Create a FidelityConfig from a level name or enum."""
    if isinstance(level, str):
        level = FidelityLevel[level.upper()]
    return FidelityConfig(level=level)
