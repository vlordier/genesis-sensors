"""
High-level ``SensorSuite`` wrapper.

Bundles all sensor models into a single object that mirrors the structure
described in the problem statement.  Internally it uses a
:class:`~genesis.sensors.scheduler.SensorScheduler` to drive sensors at
their individual rates.

The suite expects an *ideal state* dict that consumers build from Genesis
outputs at every simulation step::

    state = {
        # From scene / entity queries
        "pose":    drone.get_pos_quat(),        # (pos, quat)
        "vel":     drone.get_vel(),             # (3,)
        "ang_vel": drone.get_ang_vel(),         # (3,)

        # From cam.render(...)
        "rgb":     cam.render(rgb=True),
        "depth":   cam.render(depth=True),
        "seg":     cam.render(segmentation=True),
        "normal":  cam.render(normal=True),

        # Domain-specific
        "temperature_map": {entity_id: temp_c, ...},
        "obstruction":     0.3,   # 0–1 sky-obstruction fraction
        "weather":         {"rain_rate_mm_h": 5.0},
    }

    obs = suite.step(sim_time, state)
    print(obs["rgb"]["rgb"])          # corrupted RGB uint8
    print(obs["events"]["events"])    # list of Event objects
    print(obs["gnss"]["pos_llh"])     # lat, lon, alt

Example
-------
::

    import genesis as gs
    from genesis.sensors import SensorSuite

    suite = SensorSuite.default()
    suite.reset()

    # inside the simulation loop
    obs = suite.step(scene.cur_t, state)
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from .acoustic_navigation import AcousticCurrentProfilerModel, DVLModel
from .airspeed import AirspeedModel
from .barometer import BarometerModel
from .battery import BatteryModel
from .camera_model import CameraModel
from .contact_sensor import ContactSensor
from .current_sensor import CurrentSensor
from .depth_camera import DepthCameraModel
from .event_camera import EventCameraModel
from .environmental import AnemometerModel, GasSensorModel, HygrometerModel, LightSensorModel, ThermometerModel
from .force_torque import ForceTorqueSensorModel
from .gnss import GNSSModel
from .imu import IMUModel
from .joint_state import JointStateSensor
from .lidar import LidarModel
from .magnetometer import MagnetometerModel
from .optical_flow import OpticalFlowModel
from .radio import RadioLinkModel
from .rangefinder import RangefinderModel
from .rpm_sensor import RPMSensor
from .scheduler import SensorScheduler
from .sonar import ImagingSonarModel, SideScanSonarModel
from .wireless import RadarModel, UWBRangingModel
from .stereo_camera import StereoCameraModel
from .ultrasonic import UltrasonicArrayModel
from .tactile_array import TactileArraySensor
from .thermal_camera import ThermalCameraModel
from .wheel_odometry import WheelOdometryModel

if TYPE_CHECKING:
    from .base import BaseSensor
    from .config import SensorSuiteConfig


class SensorSuite:
    """
    Convenience wrapper that instantiates and drives a full sensor stack.

    Parameters
    ----------
    rgb_camera:
        :class:`~genesis.sensors.camera_model.CameraModel` instance.
    event_camera:
        :class:`~genesis.sensors.event_camera.EventCameraModel` instance.
    thermal_camera:
        :class:`~genesis.sensors.thermal_camera.ThermalCameraModel` instance.
    lidar:
        :class:`~genesis.sensors.lidar.LidarModel` instance.
    gnss:
        :class:`~genesis.sensors.gnss.GNSSModel` instance.
    radio:
        :class:`~genesis.sensors.radio.RadioLinkModel` instance.
    imu:
        :class:`~genesis.sensors.imu.IMUModel` instance.
    barometer:
        :class:`~genesis.sensors.barometer.BarometerModel` instance.
    magnetometer:
        :class:`~genesis.sensors.magnetometer.MagnetometerModel` instance.
    airspeed:
        :class:`~genesis.sensors.airspeed.AirspeedModel` instance.
    rangefinder:
        :class:`~genesis.sensors.rangefinder.RangefinderModel` instance.
    stereo_camera:
        :class:`~genesis.sensors.stereo_camera.StereoCameraModel` instance.
    wheel_odometry:
        :class:`~genesis.sensors.wheel_odometry.WheelOdometryModel` instance.
    force_torque:
        :class:`~genesis.sensors.force_torque.ForceTorqueSensorModel` instance.
    joint_state:
        :class:`~genesis.sensors.joint_state.JointStateSensor` instance.
    contact:
        :class:`~genesis.sensors.contact_sensor.ContactSensor` instance.
    depth_camera:
        :class:`~genesis.sensors.depth_camera.DepthCameraModel` instance.
    tactile_array:
        :class:`~genesis.sensors.tactile_array.TactileArraySensor` instance.
    current:
        :class:`~genesis.sensors.current_sensor.CurrentSensor` instance.
    rpm:
        :class:`~genesis.sensors.rpm_sensor.RPMSensor` instance.
    extra_sensors:
        Additional ``(name, BaseSensor)`` pairs to register.
    """

    def __init__(
        self,
        rgb_camera: CameraModel | None = None,
        event_camera: EventCameraModel | None = None,
        thermal_camera: ThermalCameraModel | None = None,
        lidar: LidarModel | None = None,
        gnss: GNSSModel | None = None,
        radio: RadioLinkModel | None = None,
        uwb: UWBRangingModel | None = None,
        radar: RadarModel | None = None,
        imu: IMUModel | None = None,
        barometer: BarometerModel | None = None,
        magnetometer: MagnetometerModel | None = None,
        thermometer: ThermometerModel | None = None,
        hygrometer: HygrometerModel | None = None,
        light_sensor: LightSensorModel | None = None,
        gas_sensor: GasSensorModel | None = None,
        anemometer: AnemometerModel | None = None,
        airspeed: AirspeedModel | None = None,
        rangefinder: RangefinderModel | None = None,
        ultrasonic: UltrasonicArrayModel | None = None,
        imaging_sonar: ImagingSonarModel | None = None,
        side_scan: SideScanSonarModel | None = None,
        dvl: DVLModel | None = None,
        current_profiler: AcousticCurrentProfilerModel | None = None,
        optical_flow: OpticalFlowModel | None = None,
        battery: BatteryModel | None = None,
        stereo_camera: StereoCameraModel | None = None,
        wheel_odometry: WheelOdometryModel | None = None,
        force_torque: ForceTorqueSensorModel | None = None,
        joint_state: JointStateSensor | None = None,
        contact: ContactSensor | None = None,
        depth_camera: DepthCameraModel | None = None,
        tactile_array: TactileArraySensor | None = None,
        current: CurrentSensor | None = None,
        rpm: RPMSensor | None = None,
        extra_sensors: list[tuple[str, BaseSensor]] | None = None,
    ) -> None:
        self._scheduler = SensorScheduler()

        if rgb_camera is not None:
            self._scheduler.add(rgb_camera, name="rgb")
        if event_camera is not None:
            self._scheduler.add(event_camera, name="events")
        if thermal_camera is not None:
            self._scheduler.add(thermal_camera, name="thermal")
        if lidar is not None:
            self._scheduler.add(lidar, name="lidar")
        if gnss is not None:
            self._scheduler.add(gnss, name="gnss")
        if radio is not None:
            self._scheduler.add(radio, name="radio")
        if uwb is not None:
            self._scheduler.add(uwb, name="uwb")
        if radar is not None:
            self._scheduler.add(radar, name="radar")
        if imu is not None:
            self._scheduler.add(imu, name="imu")
        if barometer is not None:
            self._scheduler.add(barometer, name="barometer")
        if magnetometer is not None:
            self._scheduler.add(magnetometer, name="magnetometer")
        if thermometer is not None:
            self._scheduler.add(thermometer, name="thermometer")
        if hygrometer is not None:
            self._scheduler.add(hygrometer, name="hygrometer")
        if light_sensor is not None:
            self._scheduler.add(light_sensor, name="light_sensor")
        if gas_sensor is not None:
            self._scheduler.add(gas_sensor, name="gas_sensor")
        if anemometer is not None:
            self._scheduler.add(anemometer, name="anemometer")
        if airspeed is not None:
            self._scheduler.add(airspeed, name="airspeed")
        if rangefinder is not None:
            self._scheduler.add(rangefinder, name="rangefinder")
        if ultrasonic is not None:
            self._scheduler.add(ultrasonic, name="ultrasonic")
        if imaging_sonar is not None:
            self._scheduler.add(imaging_sonar, name="imaging_sonar")
        if side_scan is not None:
            self._scheduler.add(side_scan, name="side_scan")
        if dvl is not None:
            self._scheduler.add(dvl, name="dvl")
        if current_profiler is not None:
            self._scheduler.add(current_profiler, name="current_profiler")
        if optical_flow is not None:
            self._scheduler.add(optical_flow, name="optical_flow")
        if battery is not None:
            self._scheduler.add(battery, name="battery")
        if stereo_camera is not None:
            self._scheduler.add(stereo_camera, name="stereo")
        if wheel_odometry is not None:
            self._scheduler.add(wheel_odometry, name="wheel_odometry")
        if force_torque is not None:
            self._scheduler.add(force_torque, name="force_torque")
        if joint_state is not None:
            self._scheduler.add(joint_state, name="joint_state")
        if contact is not None:
            self._scheduler.add(contact, name="contact")
        if depth_camera is not None:
            self._scheduler.add(depth_camera, name="depth_camera")
        if tactile_array is not None:
            self._scheduler.add(tactile_array, name="tactile_array")
        if current is not None:
            self._scheduler.add(current, name="current")
        if rpm is not None:
            self._scheduler.add(rpm, name="rpm")

        for name, sensor in extra_sensors or []:
            self._scheduler.add(sensor, name=name)

    # ------------------------------------------------------------------
    # Class-level factory
    # ------------------------------------------------------------------

    @classmethod
    def default(
        cls,
        rgb_rate_hz: float = 30.0,
        event_rate_hz: float = 1000.0,
        thermal_rate_hz: float = 9.0,
        lidar_rate_hz: float = 10.0,
        gnss_rate_hz: float = 10.0,
        radio_rate_hz: float = 100.0,
        uwb_rate_hz: float = 0.0,
        radar_rate_hz: float = 0.0,
        imu_rate_hz: float = 200.0,
        baro_rate_hz: float = 50.0,
        mag_rate_hz: float = 100.0,
        thermometer_rate_hz: float = 0.0,
        hygrometer_rate_hz: float = 0.0,
        light_sensor_rate_hz: float = 0.0,
        gas_sensor_rate_hz: float = 0.0,
        anemometer_rate_hz: float = 0.0,
        airspeed_rate_hz: float = 0.0,
        rangefinder_rate_hz: float = 0.0,
        ultrasonic_rate_hz: float = 0.0,
        imaging_sonar_rate_hz: float = 0.0,
        side_scan_rate_hz: float = 0.0,
        dvl_rate_hz: float = 0.0,
        current_profiler_rate_hz: float = 0.0,
        optical_flow_rate_hz: float = 0.0,
        battery_rate_hz: float = 0.0,
        stereo_rate_hz: float = 0.0,
        wheel_odometry_rate_hz: float = 0.0,
        force_torque_rate_hz: float = 0.0,
        joint_state_rate_hz: float = 0.0,
        contact_rate_hz: float = 0.0,
        depth_camera_rate_hz: float = 0.0,
        tactile_array_rate_hz: float = 0.0,
        current_rate_hz: float = 0.0,
        rpm_rate_hz: float = 0.0,
        seed: int | None = None,
    ) -> SensorSuite:
        """
        Create a :class:`SensorSuite` with default sensor configurations.

        All parameters are optional; pass ``0`` to
        disable a specific sensor.

        Parameters
        ----------
        rgb_rate_hz:
            RGB camera frame rate.
        event_rate_hz:
            Event camera update rate.
        thermal_rate_hz:
            Thermal camera frame rate.
        lidar_rate_hz:
            LiDAR rotation rate.
        gnss_rate_hz:
            GNSS output rate.
        radio_rate_hz:
            Radio link scheduler poll rate.
        imu_rate_hz:
            IMU output rate.
        baro_rate_hz:
            Barometer output rate.
        mag_rate_hz:
            Magnetometer output rate.
        airspeed_rate_hz:
            Pitot airspeed sensor rate (0 = disabled by default).
        rangefinder_rate_hz:
            Rangefinder rate (0 = disabled by default).
        seed:
            Optional base seed for reproducibility.  Each sensor receives an
            independent child seed derived via :class:`numpy.random.SeedSequence`
            so that sensor RNG streams are statistically uncorrelated regardless
            of which sensors are enabled.
        """
        # Derive N independent, deterministic seeds via SeedSequence so that
        # close base seeds (e.g. 0 vs 1) don't produce correlated sensor noise.
        _N_SENSORS = 34
        if seed is not None:
            child_seeds = np.random.SeedSequence(seed).spawn(_N_SENSORS)
            _seeds: list[int | None] = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        else:
            _seeds = [None] * _N_SENSORS

        def _seed(offset: int) -> int | None:
            return _seeds[offset]

        return cls(
            rgb_camera=CameraModel(update_rate_hz=rgb_rate_hz, seed=_seed(0)) if rgb_rate_hz > 0 else None,
            event_camera=EventCameraModel(update_rate_hz=event_rate_hz, seed=_seed(1)) if event_rate_hz > 0 else None,
            thermal_camera=(
                ThermalCameraModel(update_rate_hz=thermal_rate_hz, seed=_seed(2)) if thermal_rate_hz > 0 else None
            ),
            lidar=LidarModel(update_rate_hz=lidar_rate_hz, seed=_seed(3)) if lidar_rate_hz > 0 else None,
            gnss=GNSSModel(update_rate_hz=gnss_rate_hz, seed=_seed(4)) if gnss_rate_hz > 0 else None,
            radio=RadioLinkModel(update_rate_hz=radio_rate_hz, seed=_seed(5)) if radio_rate_hz > 0 else None,
            uwb=UWBRangingModel(update_rate_hz=uwb_rate_hz, seed=_seed(6)) if uwb_rate_hz > 0 else None,
            radar=RadarModel(update_rate_hz=radar_rate_hz, seed=_seed(7)) if radar_rate_hz > 0 else None,
            imu=IMUModel(update_rate_hz=imu_rate_hz, seed=_seed(8)) if imu_rate_hz > 0 else None,
            barometer=BarometerModel(update_rate_hz=baro_rate_hz, seed=_seed(9)) if baro_rate_hz > 0 else None,
            magnetometer=MagnetometerModel(update_rate_hz=mag_rate_hz, seed=_seed(10)) if mag_rate_hz > 0 else None,
            thermometer=(
                ThermometerModel(update_rate_hz=thermometer_rate_hz, seed=_seed(11))
                if thermometer_rate_hz > 0
                else None
            ),
            hygrometer=(
                HygrometerModel(update_rate_hz=hygrometer_rate_hz, seed=_seed(12)) if hygrometer_rate_hz > 0 else None
            ),
            light_sensor=(
                LightSensorModel(update_rate_hz=light_sensor_rate_hz, seed=_seed(13))
                if light_sensor_rate_hz > 0
                else None
            ),
            gas_sensor=(
                GasSensorModel(update_rate_hz=gas_sensor_rate_hz, seed=_seed(14)) if gas_sensor_rate_hz > 0 else None
            ),
            anemometer=(
                AnemometerModel(update_rate_hz=anemometer_rate_hz, seed=_seed(15)) if anemometer_rate_hz > 0 else None
            ),
            airspeed=AirspeedModel(update_rate_hz=airspeed_rate_hz, seed=_seed(16)) if airspeed_rate_hz > 0 else None,
            rangefinder=(
                RangefinderModel(update_rate_hz=rangefinder_rate_hz, seed=_seed(17))
                if rangefinder_rate_hz > 0
                else None
            ),
            ultrasonic=(
                UltrasonicArrayModel(update_rate_hz=ultrasonic_rate_hz, seed=_seed(18))
                if ultrasonic_rate_hz > 0
                else None
            ),
            imaging_sonar=(
                ImagingSonarModel(update_rate_hz=imaging_sonar_rate_hz, seed=_seed(19))
                if imaging_sonar_rate_hz > 0
                else None
            ),
            side_scan=(
                SideScanSonarModel(update_rate_hz=side_scan_rate_hz, seed=_seed(20)) if side_scan_rate_hz > 0 else None
            ),
            dvl=(DVLModel(update_rate_hz=dvl_rate_hz, seed=_seed(21)) if dvl_rate_hz > 0 else None),
            current_profiler=(
                AcousticCurrentProfilerModel(update_rate_hz=current_profiler_rate_hz, seed=_seed(22))
                if current_profiler_rate_hz > 0
                else None
            ),
            optical_flow=(
                OpticalFlowModel(update_rate_hz=optical_flow_rate_hz, seed=_seed(23))
                if optical_flow_rate_hz > 0
                else None
            ),
            battery=(BatteryModel(update_rate_hz=battery_rate_hz, seed=_seed(24)) if battery_rate_hz > 0 else None),
            stereo_camera=(
                StereoCameraModel(update_rate_hz=stereo_rate_hz, seed=_seed(25)) if stereo_rate_hz > 0 else None
            ),
            wheel_odometry=(
                WheelOdometryModel(update_rate_hz=wheel_odometry_rate_hz, seed=_seed(26))
                if wheel_odometry_rate_hz > 0
                else None
            ),
            force_torque=(
                ForceTorqueSensorModel(update_rate_hz=force_torque_rate_hz, seed=_seed(27))
                if force_torque_rate_hz > 0
                else None
            ),
            joint_state=(
                JointStateSensor(update_rate_hz=joint_state_rate_hz, seed=_seed(28))
                if joint_state_rate_hz > 0
                else None
            ),
            contact=(ContactSensor(update_rate_hz=contact_rate_hz, seed=_seed(29)) if contact_rate_hz > 0 else None),
            depth_camera=(
                DepthCameraModel(update_rate_hz=depth_camera_rate_hz, seed=_seed(30))
                if depth_camera_rate_hz > 0
                else None
            ),
            tactile_array=(
                TactileArraySensor(update_rate_hz=tactile_array_rate_hz, seed=_seed(31))
                if tactile_array_rate_hz > 0
                else None
            ),
            current=(CurrentSensor(update_rate_hz=current_rate_hz, seed=_seed(32)) if current_rate_hz > 0 else None),
            rpm=(RPMSensor(update_rate_hz=rpm_rate_hz, seed=_seed(33)) if rpm_rate_hz > 0 else None),
        )

    @classmethod
    def from_config(cls, config: "SensorSuiteConfig") -> SensorSuite:
        """
        Create a :class:`SensorSuite` from a :class:`~genesis.sensors.config.SensorSuiteConfig`.

        Example
        -------
        ::

            from genesis.sensors.config import SensorSuiteConfig, CameraConfig, GNSSConfig
            cfg = SensorSuiteConfig(
                rgb=CameraConfig(iso=400),
                gnss=GNSSConfig(noise_m=0.3),
                lidar=None,
            )
            suite = SensorSuite.from_config(cfg)
        """
        return cls(
            rgb_camera=CameraModel.from_config(config.rgb) if config.rgb is not None else None,
            event_camera=(EventCameraModel.from_config(config.event) if config.event is not None else None),
            thermal_camera=(ThermalCameraModel.from_config(config.thermal) if config.thermal is not None else None),
            lidar=LidarModel.from_config(config.lidar) if config.lidar is not None else None,
            gnss=GNSSModel.from_config(config.gnss) if config.gnss is not None else None,
            radio=RadioLinkModel.from_config(config.radio) if config.radio is not None else None,
            uwb=UWBRangingModel.from_config(config.uwb) if config.uwb is not None else None,
            radar=RadarModel.from_config(config.radar) if config.radar is not None else None,
            imu=IMUModel.from_config(config.imu) if config.imu is not None else None,
            barometer=BarometerModel.from_config(config.barometer) if config.barometer is not None else None,
            magnetometer=(
                MagnetometerModel.from_config(config.magnetometer) if config.magnetometer is not None else None
            ),
            thermometer=(ThermometerModel.from_config(config.thermometer) if config.thermometer is not None else None),
            hygrometer=(HygrometerModel.from_config(config.hygrometer) if config.hygrometer is not None else None),
            light_sensor=(
                LightSensorModel.from_config(config.light_sensor) if config.light_sensor is not None else None
            ),
            gas_sensor=(GasSensorModel.from_config(config.gas_sensor) if config.gas_sensor is not None else None),
            anemometer=(AnemometerModel.from_config(config.anemometer) if config.anemometer is not None else None),
            airspeed=AirspeedModel.from_config(config.airspeed) if config.airspeed is not None else None,
            rangefinder=(RangefinderModel.from_config(config.rangefinder) if config.rangefinder is not None else None),
            ultrasonic=(UltrasonicArrayModel.from_config(config.ultrasonic) if config.ultrasonic is not None else None),
            imaging_sonar=(
                ImagingSonarModel.from_config(config.imaging_sonar) if config.imaging_sonar is not None else None
            ),
            side_scan=(SideScanSonarModel.from_config(config.side_scan) if config.side_scan is not None else None),
            dvl=(DVLModel.from_config(config.dvl) if config.dvl is not None else None),
            current_profiler=(
                AcousticCurrentProfilerModel.from_config(config.current_profiler)
                if config.current_profiler is not None
                else None
            ),
            optical_flow=(
                OpticalFlowModel.from_config(config.optical_flow) if config.optical_flow is not None else None
            ),
            battery=(BatteryModel.from_config(config.battery) if config.battery is not None else None),
            stereo_camera=(
                StereoCameraModel.from_config(config.stereo_camera) if config.stereo_camera is not None else None
            ),
            wheel_odometry=(
                WheelOdometryModel.from_config(config.wheel_odometry) if config.wheel_odometry is not None else None
            ),
            force_torque=(
                ForceTorqueSensorModel.from_config(config.force_torque) if config.force_torque is not None else None
            ),
            joint_state=(JointStateSensor.from_config(config.joint_state) if config.joint_state is not None else None),
            contact=(ContactSensor.from_config(config.contact) if config.contact is not None else None),
            depth_camera=(
                DepthCameraModel.from_config(config.depth_camera) if config.depth_camera is not None else None
            ),
            tactile_array=(
                TactileArraySensor.from_config(config.tactile_array) if config.tactile_array is not None else None
            ),
            current=(CurrentSensor.from_config(config.current) if config.current is not None else None),
            rpm=(RPMSensor.from_config(config.rpm) if config.rpm is not None else None),
        )

    # ------------------------------------------------------------------
    # Sensor lifecycle
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Reset all sensor models (call at the start of each episode)."""
        self._scheduler.reset(env_id=env_id)

    def step(self, sim_time: float, state: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
        """
        Advance all due sensors and return their observations.

        Parameters
        ----------
        sim_time:
            Current simulation time in seconds (e.g., ``scene.cur_t``).
        state:
            Ideal measurements from Genesis.  See module docstring for the
            expected layout.

        Returns
        -------
        dict
            Mapping ``sensor_name → observation_dict``.
        """
        return self._scheduler.update(sim_time=sim_time, state=state)

    # ------------------------------------------------------------------
    # Direct sensor access
    # ------------------------------------------------------------------

    def get_sensor(self, name: str) -> BaseSensor:
        """Return a registered sensor by name."""
        return self._scheduler.get_sensor(name)

    def sensor_names(self) -> list[str]:
        """Return all registered sensor names."""
        return self._scheduler.sensor_names()

    @property
    def scheduler(self) -> SensorScheduler:
        """The underlying :class:`SensorScheduler`."""
        return self._scheduler

    def __repr__(self) -> str:
        names = self.sensor_names()
        return f"SensorSuite(sensors={names})"
