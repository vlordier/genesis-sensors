"""
External sensor realism layer for Genesis.

This package provides a multi-rate sensor stack that sits **on top** of
Genesis rather than inside it.  Genesis produces ideal, noise-free
measurements; this layer converts them into realistic device outputs.

Public API
----------
.. autosummary::

    BaseSensor
    CameraModel
    EventCameraModel
    IMUModel
    ThermalCameraModel
    LidarModel
    GNSSModel
    RadioLinkModel
    SensorScheduler
    SensorSuite
    SensorSuiteConfig

Quick-start (keyword args)
--------------------------
::

    from genesis.sensors import SensorSuite

    suite = SensorSuite.default()
    suite.reset()
    obs = suite.step(scene.cur_t, state)

Quick-start (config-driven)
---------------------------
::

    from genesis.sensors import SensorSuite
    from genesis.sensors.config import CameraConfig, GNSSConfig, SensorSuiteConfig

    cfg = SensorSuiteConfig(
        rgb=CameraConfig(iso=800, jpeg_quality=60),
        gnss=GNSSConfig(noise_m=0.5),
        lidar=None,
    )
    suite = SensorSuite.from_config(cfg)
    print(cfg.model_dump_json(indent=2))  # serialise to JSON

Quick-start (device presets)
----------------------------
::

    from genesis.sensors import get_preset, list_presets, LidarModel

    print(list_presets())                       # sorted list of all preset names
    vlp16 = get_preset("VELODYNE_VLP16")        # returns LidarConfig
    lidar = LidarModel.from_config(vlp16)

    # Presets are also importable by name:
    from genesis.sensors import RASPBERRY_PI_V2, VELODYNE_VLP16, PIXHAWK_ICM20689, UBLOX_F9P_RTK
    from genesis.sensors import FLIR_BOSON_320, DAVIS_346, BMP388, IST8310, SDP33, TFMINI_PLUS

Examples
--------
See the following runnable examples for progressively deeper dives:

* ``examples/sensors/external_sensors.py`` -- headless walkthrough of every
  sensor model (RGB, event, thermal, LiDAR, GNSS, radio, IMU) using synthetic
  NumPy state; the recommended first stop.
* ``examples/sensors/sensor_usage_patterns.py`` -- shows four distinct API
  styles: direct stepping, preset construction, config-driven suites, and
  manual schedulers.
* ``examples/sensors/camera_as_sensor.py`` -- scene-backed bridge from
  ``scene.add_sensor(...)`` cameras into ``CameraModel`` / ``EventCameraModel``
  / ``ThermalCameraModel`` post-processing, with saved comparison panels.
* ``examples/sensors/external_sensors_rerun.py`` -- Rerun viewer dashboard
  visualising the full synthetic scenario in real time.
"""

from . import presets
from ._gauss_markov import GaussMarkovProcess
from .acoustic_navigation import AcousticCurrentProfilerModel, DVLModel
from .airspeed import AirspeedModel
from .barometer import BarometerModel
from .base import BaseSensor, SensorInput, SensorObservation
from .battery import BatteryModel
from .camera_model import CameraModel
from .config import (
    AcousticCurrentProfilerConfig,
    AirspeedConfig,
    AnemometerConfig,
    BarometerConfig,
    BatteryConfig,
    CameraConfig,
    ContactSensorConfig,
    CurrentSensorConfig,
    DVLConfig,
    DepthCameraConfig,
    EventCameraConfig,
    ForceTorqueConfig,
    GasSensorConfig,
    GNSSConfig,
    HydrophoneConfig,
    HygrometerConfig,
    IMUConfig,
    ImagingSonarConfig,
    InclinometerConfig,
    JointStateConfig,
    LeakDetectorConfig,
    LidarConfig,
    LightSensorConfig,
    LoadCellConfig,
    MagnetometerConfig,
    MotorTemperatureConfig,
    OpticalFlowConfig,
    ProximityToFArrayConfig,
    RadarConfig,
    RadioConfig,
    RangefinderConfig,
    RPMSensorConfig,
    SensorSuiteConfig,
    SideScanSonarConfig,
    StereoCameraConfig,
    TactileArrayConfig,
    ThermalCameraConfig,
    ThermometerConfig,
    UnderwaterModemConfig,
    UWBRangeConfig,
    UltrasonicArrayConfig,
    WaterPressureConfig,
    WheelOdometryConfig,
    WireEncoderConfig,
)
from .contact_sensor import ContactSensor
from .current_sensor import CurrentSensor
from .depth_camera import DepthCameraModel
from .event_camera import Event, EventCameraModel
from .environmental import AnemometerModel, GasSensorModel, HygrometerModel, LightSensorModel, ThermometerModel
from .force_torque import ForceTorqueSensorModel
from .genesis_camera import GenesisCamera
from .genesis_imu import GenesisIMU
from .genesis_lidar import GenesisLiDAR
from .gnss import GNSSModel, GnssFixQuality
from .hydrophone import HydrophoneModel
from .imu import IMUModel
from .inclinometer import InclinometerModel
from .joint_state import JointStateSensor
from .leak_detector import LeakDetectorModel
from .lidar import LidarModel, LidarPoint
from .load_cell import LoadCellModel
from .magnetometer import MagnetometerModel
from .motor_temperature import MotorTemperatureModel
from .optical_flow import OpticalFlowModel
from .proximity_tof import ProximityToFArrayModel
from .radio import RadioLinkModel, ScheduledPacket
from .rangefinder import RangefinderModel
from .rpm_sensor import RPMSensor
from .tactile_array import TactileArraySensor
from .underwater_modem import UnderwaterModemModel
from .water_pressure import WaterPressureModel
from .wheel_odometry import WheelOdometryModel
from .wire_encoder import WireEncoderModel
from .presets import (
    ACS712_5A,
    AS5048A_MAG_ENC,
    ATI_MINI45,
    BMP280,
    BMP388,
    BOSCH_BMI088,
    NORTEK_DVL1000,
    BUMPER_50HZ,
    DAVIS_346,
    DAVIS_6410_ANEMOMETER,
    DIFF_DRIVE_ENCODER_50HZ,
    DS18B20_PROBE,
    DPS310,
    EMLID_REACH_RS2,
    BLUEVIEW_P900_130,
    EDGETECH_4125,
    FINGERTIP_TACTILE_200HZ,
    TRITECH_GEMINI_720IK,
    FINGERTIP_TACTILE_4X4,
    FLIR_BOSON_320,
    FLIR_LEPTON_35,
    FRANKA_JOINT_ENCODER,
    GARMIN_LIDAR_LITE_V3,
    GENERIC_SERVO_ENCODER,
    GOPRO_HERO11_4K30,
    HC_SR04_ARRAY4,
    HESAI_XT32,
    HMC5883L,
    INTEL_D415,
    INTEL_D435,
    INTEL_D435_RGB,
    INTEL_D435_STEREO,
    INTEL_L515,
    LSM6DSO,
    LPS22HH,
    INVENSENSE_MPU9250,
    IST8310,
    ICM42688P,
    ICM20948_BARO,
    LIPO_3S_2200MAH,
    LIPO_4S_5000MAH,
    LIPO_6S_10000MAH,
    LIVOX_AVIA,
    LIVOX_MID360,
    INA226_10A,
    MATEKSYS_3901_L0X,
    MAUCH_HS_200,
    MAXBOTIX_MB1242_RING8,
    MECANUM_DRIVE_ENCODER_100HZ,
    MS4525DO,
    MS5611,
    MYNT_EYE_D_120,
    NOVATEL_OEM7,
    NOVATEL_OEM7_SBAS,
    OAK_D_LITE,
    OPTOFORCE_OMD,
    OPTICAL_ENC_1024,
    OUSTER_OS1_64,
    TELEDYNE_WORKHORSE_600,
    PALM_TACTILE_8X8,
    PIXHAWK_ICM20689,
    ADIS16470,
    PROPHESEE_EVK4,
    PresetConfig,
    PX4FLOW,
    QORVO_DWM3001C,
    RASPBERRY_PI_V2,
    RADIO_FREESPACE,
    RADIO_INDOOR,
    RADIO_URBAN,
    REALSENSE_D455,
    RM3100,
    ROKUBI_FT300,
    SDP33,
    SGP30_AIR_QUALITY,
    SHT31_HUMIDITY,
    SICK_TIM571,
    SEEK_THERMAL_COMPACT_PRO,
    SPL06_001,
    T_MOTOR_HALL_6P,
    TERARANGER_ONE,
    TFMINI_PLUS,
    TI_IWR6843AOP,
    TSL2591_LIGHT,
    TRIMBLE_BD992,
    NAVTECH_CTS350X,
    UBLOX_F9P_RTK,
    UBLOX_F9P_FLOAT,
    UBLOX_F9P_AUTONOMOUS,
    UBLOX_M8N,
    UR5_JOINT_ENCODER,
    VELODYNE_HDL64E,
    VELODYNE_VLP16,
    VECTORNAV_VN100,
    WRIST_TACTILE_8X2,
    XSENS_MTI_3,
    ZED2_LEFT,
    ZED2_RIGHT,
    ZED2_STEREO,
    get_preset,
    list_presets,
)
from .scheduler import SensorScheduler
from .stereo_camera import StereoCameraModel
from .suite import SensorSuite
from .sonar import ImagingSonarModel, SideScanSonarModel
from .thermal_camera import ThermalCameraModel
from .ultrasonic import UltrasonicArrayModel
from .wireless import RadarModel, UWBRangingModel
from .types import (
    AcousticCurrentProfilerObservation,
    AirspeedObservation,
    AnemometerObservation,
    ArrayLike,
    BarometerObservation,
    BatteryObservation,
    BoolArray,
    CameraObservation,
    ContactObservation,
    CurrentObservation,
    DVLObservation,
    DepthCameraObservation,
    EventCameraObservation,
    Float64Array,
    FloatArray,
    ForceTorqueObservation,
    GasObservation,
    GnssObservation,
    HydrophoneObservation,
    HygrometerObservation,
    ImagingSonarObservation,
    ImuObservation,
    InclinometerObservation,
    Int32Array,
    JammerZone,
    JointStateObservation,
    LeakDetectorObservation,
    LidarObservation,
    LightSensorObservation,
    LoadCellObservation,
    MagnetometerObservation,
    MotorTemperatureObservation,
    OpticalFlowObservation,
    Polarity,
    ProximityToFObservation,
    RadarObservation,
    RadioObservation,
    RangefinderObservation,
    RPMObservation,
    SensorState,
    SideScanSonarObservation,
    StereoCameraObservation,
    TactileArrayObservation,
    ThermalObservation,
    ThermometerObservation,
    UInt16Array,
    UWBObservation,
    UInt8Array,
    UltrasonicObservation,
    UnderwaterModemObservation,
    WaterPressureObservation,
    WheelOdometryObservation,
    WireEncoderObservation,
)

__all__ = [
    # Sensor presets module
    "presets",
    # Preset constants — cameras
    "GOPRO_HERO11_4K30",
    "INTEL_D435_RGB",
    "RASPBERRY_PI_V2",
    "ZED2_LEFT",
    "ZED2_RIGHT",
    # Preset constants — stereo cameras
    "INTEL_D435_STEREO",
    "MYNT_EYE_D_120",
    "ZED2_STEREO",
    # Preset constants — LiDAR
    "HESAI_XT32",
    "LIVOX_AVIA",
    "LIVOX_MID360",
    "OUSTER_OS1_64",
    "SICK_TIM571",
    "VELODYNE_HDL64E",
    "VELODYNE_VLP16",
    # Preset constants — IMU
    "BOSCH_BMI088",
    "INVENSENSE_MPU9250",
    "PIXHAWK_ICM20689",
    "VECTORNAV_VN100",
    "XSENS_MTI_3",
    "ADIS16470",
    "LSM6DSO",
    "ICM42688P",
    # Preset constants — GNSS
    "EMLID_REACH_RS2",
    "NOVATEL_OEM7",
    "NOVATEL_OEM7_SBAS",
    "TRIMBLE_BD992",
    "UBLOX_F9P_RTK",
    "UBLOX_F9P_FLOAT",
    "UBLOX_F9P_AUTONOMOUS",
    "UBLOX_M8N",
    # Preset constants — thermal
    "FLIR_BOSON_320",
    "FLIR_LEPTON_35",
    "SEEK_THERMAL_COMPACT_PRO",
    # Preset constants — event camera
    "DAVIS_346",
    "PROPHESEE_EVK4",
    # Preset constants — barometer
    "BMP280",
    "BMP388",
    "DPS310",
    "LPS22HH",
    "ICM20948_BARO",
    "MS5611",
    "SPL06_001",
    # Preset constants — environmental
    "DAVIS_6410_ANEMOMETER",
    "DS18B20_PROBE",
    "SGP30_AIR_QUALITY",
    "SHT31_HUMIDITY",
    "TSL2591_LIGHT",
    # Preset constants — wireless
    "NAVTECH_CTS350X",
    "QORVO_DWM3001C",
    "TI_IWR6843AOP",
    "RADIO_FREESPACE",
    "RADIO_URBAN",
    "RADIO_INDOOR",
    # Preset constants — ultrasonic
    "HC_SR04_ARRAY4",
    "MAXBOTIX_MB1242_RING8",
    # Preset constants — underwater navigation
    "NORTEK_DVL1000",
    "TELEDYNE_WORKHORSE_600",
    # Preset constants — sonar
    "BLUEVIEW_P900_130",
    "EDGETECH_4125",
    "TRITECH_GEMINI_720IK",
    # Preset constants — magnetometer
    "HMC5883L",
    "IST8310",
    "RM3100",
    # Preset constants — airspeed
    "MS4525DO",
    "SDP33",
    # Preset constants — rangefinder
    "GARMIN_LIDAR_LITE_V3",
    "TERARANGER_ONE",
    "TFMINI_PLUS",
    # Preset constants — optical flow
    "MATEKSYS_3901_L0X",
    "PX4FLOW",
    # Preset constants — battery
    "LIPO_3S_2200MAH",
    "LIPO_4S_5000MAH",
    "LIPO_6S_10000MAH",
    # Preset constants — wheel odometry
    "DIFF_DRIVE_ENCODER_50HZ",
    "MECANUM_DRIVE_ENCODER_100HZ",
    # Preset constants — force/torque
    "ATI_MINI45",
    "ROKUBI_FT300",
    "OPTOFORCE_OMD",
    # Preset constants — tactile array
    "FINGERTIP_TACTILE_4X4",
    "WRIST_TACTILE_8X2",
    "PALM_TACTILE_8X8",
    # Preset constants — current
    "INA226_10A",
    "MAUCH_HS_200",
    "ACS712_5A",
    # Preset constants — rpm
    "AS5048A_MAG_ENC",
    "T_MOTOR_HALL_6P",
    "OPTICAL_ENC_1024",
    # Preset constants — joint state
    "FRANKA_JOINT_ENCODER",
    "GENERIC_SERVO_ENCODER",
    "UR5_JOINT_ENCODER",
    # Preset constants — contact
    "BUMPER_50HZ",
    "FINGERTIP_TACTILE_200HZ",
    # Preset constants — depth camera
    "INTEL_D415",
    "INTEL_D435",
    "INTEL_L515",
    "OAK_D_LITE",
    "REALSENSE_D455",
    # Preset helpers
    "get_preset",
    "list_presets",
    # Preset type alias
    "PresetConfig",
    # Utilities
    "GaussMarkovProcess",
    # Sensor classes
    "AcousticCurrentProfilerModel",
    "AirspeedModel",
    "AnemometerModel",
    "BarometerModel",
    "BaseSensor",
    "BatteryModel",
    "CameraModel",
    "ContactSensor",
    "CurrentSensor",
    "DVLModel",
    "DepthCameraModel",
    "Event",
    "EventCameraModel",
    "ForceTorqueSensorModel",
    "GasSensorModel",
    "GenesisCamera",
    "GenesisIMU",
    "GenesisLiDAR",
    "GNSSModel",
    "GnssFixQuality",
    "HydrophoneModel",
    "HygrometerModel",
    "IMUModel",
    "ImagingSonarModel",
    "InclinometerModel",
    "JointStateSensor",
    "LeakDetectorModel",
    "LidarModel",
    "LidarPoint",
    "LightSensorModel",
    "LoadCellModel",
    "MagnetometerModel",
    "MotorTemperatureModel",
    "OpticalFlowModel",
    "ProximityToFArrayModel",
    "RadarModel",
    "RadioLinkModel",
    "RangefinderModel",
    "RPMSensor",
    "ScheduledPacket",
    "SensorScheduler",
    "SensorSuite",
    "SideScanSonarModel",
    "StereoCameraModel",
    "TactileArraySensor",
    "ThermalCameraModel",
    "ThermometerModel",
    "UltrasonicArrayModel",
    "UnderwaterModemModel",
    "UWBRangingModel",
    "WaterPressureModel",
    "WheelOdometryModel",
    "WireEncoderModel",
    # Config classes
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
    "HydrophoneConfig",
    "HygrometerConfig",
    "IMUConfig",
    "ImagingSonarConfig",
    "InclinometerConfig",
    "JointStateConfig",
    "LeakDetectorConfig",
    "LidarConfig",
    "LightSensorConfig",
    "LoadCellConfig",
    "MagnetometerConfig",
    "MotorTemperatureConfig",
    "OpticalFlowConfig",
    "ProximityToFArrayConfig",
    "RadarConfig",
    "RadioConfig",
    "RangefinderConfig",
    "RPMSensorConfig",
    "SensorSuiteConfig",
    "SideScanSonarConfig",
    "StereoCameraConfig",
    "TactileArrayConfig",
    "ThermalCameraConfig",
    "ThermometerConfig",
    "UnderwaterModemConfig",
    "UWBRangeConfig",
    "UltrasonicArrayConfig",
    "WaterPressureConfig",
    "WheelOdometryConfig",
    "WireEncoderConfig",
    # Type aliases and TypedDicts
    "AcousticCurrentProfilerObservation",
    "AirspeedObservation",
    "AnemometerObservation",
    "ArrayLike",
    "BarometerObservation",
    "BatteryObservation",
    "BoolArray",
    "CameraObservation",
    "ContactObservation",
    "CurrentObservation",
    "DVLObservation",
    "DepthCameraObservation",
    "EventCameraObservation",
    "Float64Array",
    "FloatArray",
    "ForceTorqueObservation",
    "GasObservation",
    "GnssObservation",
    "HydrophoneObservation",
    "HygrometerObservation",
    "ImagingSonarObservation",
    "ImuObservation",
    "InclinometerObservation",
    "Int32Array",
    "JammerZone",
    "JointStateObservation",
    "LeakDetectorObservation",
    "LidarObservation",
    "LightSensorObservation",
    "LoadCellObservation",
    "MagnetometerObservation",
    "MotorTemperatureObservation",
    "OpticalFlowObservation",
    "Polarity",
    "ProximityToFObservation",
    "RadarObservation",
    "RadioObservation",
    "RangefinderObservation",
    "RPMObservation",
    "SensorInput",
    "SideScanSonarObservation",
    "SensorObservation",
    "SensorState",
    "StereoCameraObservation",
    "TactileArrayObservation",
    "ThermalObservation",
    "ThermometerObservation",
    "UInt16Array",
    "UInt8Array",
    "UWBObservation",
    "UltrasonicObservation",
    "UnderwaterModemObservation",
    "WaterPressureObservation",
    "WheelOdometryObservation",
    "WireEncoderObservation",
]
