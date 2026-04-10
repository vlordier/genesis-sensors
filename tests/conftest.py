"""Test configuration and shared fixtures for genesis-sensors."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genesis_sensors import make_synthetic_sensor_state  # noqa: E402
from genesis_sensors._runtime_sensors import (  # noqa: E402
    AcousticCurrentProfilerModel,
    AirspeedModel,
    AnemometerModel,
    BarometerModel,
    BatteryModel,
    CameraModel,
    ContactSensor,
    CurrentSensor,
    DepthCameraModel,
    DVLModel,
    EventCameraModel,
    ForceTorqueSensorModel,
    GasSensorModel,
    GNSSModel,
    HygrometerModel,
    IMUModel,
    ImagingSonarModel,
    JointStateSensor,
    LidarModel,
    LightSensorModel,
    MagnetometerModel,
    OpticalFlowModel,
    RadarModel,
    RadioLinkModel,
    RangefinderModel,
    RPMSensor,
    SensorScheduler,
    SensorSuite,
    SideScanSonarModel,
    StereoCameraModel,
    TactileArraySensor,
    ThermalCameraModel,
    ThermometerModel,
    UltrasonicArrayModel,
    UWBRangingModel,
    WheelOdometryModel,
)
from genesis_sensors._runtime_sensors.config import (  # noqa: E402
    AcousticCurrentProfilerConfig,
    AirspeedConfig,
    AnemometerConfig,
    BarometerConfig,
    BatteryConfig,
    CameraConfig,
    ContactSensorConfig,
    CurrentSensorConfig,
    DepthCameraConfig,
    DVLConfig,
    EventCameraConfig,
    ForceTorqueConfig,
    GasSensorConfig,
    GNSSConfig,
    HygrometerConfig,
    IMUConfig,
    ImagingSonarConfig,
    JointStateConfig,
    LidarConfig,
    LightSensorConfig,
    MagnetometerConfig,
    OpticalFlowConfig,
    RadarConfig,
    RadioConfig,
    RangefinderConfig,
    RPMSensorConfig,
    SideScanSonarConfig,
    StereoCameraConfig,
    TactileArrayConfig,
    ThermalCameraConfig,
    ThermometerConfig,
    UltrasonicArrayConfig,
    UWBRangeConfig,
    WheelOdometryConfig,
)
from genesis_sensors._runtime_sensors.presets import list_presets  # noqa: E402

# ---------------------------------------------------------------------------
# All (model_class, config_class) pairs — the single source of truth for
# parametrised tests that must cover every sensor.
# ---------------------------------------------------------------------------

ALL_SENSOR_PAIRS: list[tuple[type, type]] = [
    (IMUModel, IMUConfig),
    (GNSSModel, GNSSConfig),
    (BarometerModel, BarometerConfig),
    (MagnetometerModel, MagnetometerConfig),
    (CameraModel, CameraConfig),
    (EventCameraModel, EventCameraConfig),
    (ThermalCameraModel, ThermalCameraConfig),
    (DepthCameraModel, DepthCameraConfig),
    (StereoCameraModel, StereoCameraConfig),
    (LidarModel, LidarConfig),
    (RangefinderModel, RangefinderConfig),
    (UltrasonicArrayModel, UltrasonicArrayConfig),
    (ImagingSonarModel, ImagingSonarConfig),
    (RadioLinkModel, RadioConfig),
    (UWBRangingModel, UWBRangeConfig),
    (RadarModel, RadarConfig),
    (AirspeedModel, AirspeedConfig),
    (OpticalFlowModel, OpticalFlowConfig),
    (WheelOdometryModel, WheelOdometryConfig),
    (BatteryModel, BatteryConfig),
    (ThermometerModel, ThermometerConfig),
    (HygrometerModel, HygrometerConfig),
    (LightSensorModel, LightSensorConfig),
    (GasSensorModel, GasSensorConfig),
    (AnemometerModel, AnemometerConfig),
    (ForceTorqueSensorModel, ForceTorqueConfig),
    (JointStateSensor, JointStateConfig),
    (ContactSensor, ContactSensorConfig),
    (TactileArraySensor, TactileArrayConfig),
    (CurrentSensor, CurrentSensorConfig),
    (RPMSensor, RPMSensorConfig),
    (DVLModel, DVLConfig),
    (AcousticCurrentProfilerModel, AcousticCurrentProfilerConfig),
    (SideScanSonarModel, SideScanSonarConfig),
]

# Sensors that can step with a basic synthetic state dict.
# Excludes radio/UWB/radar/sonar which need specialised state keys.
STEPPABLE_SENSORS: list[tuple[type, str]] = [
    (IMUModel, "imu"),
    (GNSSModel, "gnss"),
    (BarometerModel, "barometer"),
    (MagnetometerModel, "magnetometer"),
    (CameraModel, "camera"),
    (DepthCameraModel, "depth_camera"),
    (LidarModel, "lidar"),
    (RangefinderModel, "rangefinder"),
    (AirspeedModel, "airspeed"),
    (OpticalFlowModel, "optical_flow"),
    (WheelOdometryModel, "wheel_odometry"),
    (BatteryModel, "battery"),
    (ThermometerModel, "thermometer"),
    (HygrometerModel, "hygrometer"),
    (LightSensorModel, "light_sensor"),
    (GasSensorModel, "gas_sensor"),
    (AnemometerModel, "anemometer"),
    (ForceTorqueSensorModel, "force_torque"),
    (JointStateSensor, "joint_state"),
    (ContactSensor, "contact"),
    (TactileArraySensor, "tactile_array"),
    (CurrentSensor, "current"),
    (RPMSensor, "rpm"),
    (DVLModel, "dvl"),
    (AcousticCurrentProfilerModel, "current_profiler"),
]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def synthetic_state() -> dict:
    """A rich synthetic state dict that satisfies all sensors."""
    return make_synthetic_sensor_state(seed=42)


@pytest.fixture()
def scheduler() -> SensorScheduler:
    """A fresh SensorScheduler instance."""
    return SensorScheduler()


@pytest.fixture()
def default_suite() -> SensorSuite:
    """A SensorSuite with IMU + barometer enabled, everything else off."""
    return SensorSuite.default(
        imu_rate_hz=200.0,
        baro_rate_hz=50.0,
        gnss_rate_hz=0.0,
        rgb_rate_hz=0.0,
        event_rate_hz=0.0,
        thermal_rate_hz=0.0,
        lidar_rate_hz=0.0,
        radio_rate_hz=0.0,
        mag_rate_hz=0.0,
        seed=0,
    )


@pytest.fixture()
def all_presets() -> list[str]:
    """Sorted list of every registered preset name."""
    return list_presets()
