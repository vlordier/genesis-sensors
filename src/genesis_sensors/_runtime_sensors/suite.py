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

from ._gauss_markov import GaussMarkovProcess
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

# ---------------------------------------------------------------------------
# Registry that maps *constructor kwarg name* → *(display_name, model_class,
# rate_param, default_rate, config_attr)*.
#
# This single table drives ``__init__``, ``default()``, and ``from_config()``
# so that adding a new sensor type is a one-line change.
# ---------------------------------------------------------------------------
_SENSOR_SLOTS: tuple[tuple[str, str, type, str, float, str], ...] = (
    # (kwarg,             display,           model_class,                  rate_param,              default_rate, cfg_attr)
    ("rgb_camera", "rgb", CameraModel, "rgb_rate_hz", 30.0, "rgb"),
    ("event_camera", "events", EventCameraModel, "event_rate_hz", 1000.0, "event"),
    ("thermal_camera", "thermal", ThermalCameraModel, "thermal_rate_hz", 9.0, "thermal"),
    ("lidar", "lidar", LidarModel, "lidar_rate_hz", 10.0, "lidar"),
    ("gnss", "gnss", GNSSModel, "gnss_rate_hz", 10.0, "gnss"),
    ("radio", "radio", RadioLinkModel, "radio_rate_hz", 100.0, "radio"),
    ("uwb", "uwb", UWBRangingModel, "uwb_rate_hz", 0.0, "uwb"),
    ("radar", "radar", RadarModel, "radar_rate_hz", 0.0, "radar"),
    ("imu", "imu", IMUModel, "imu_rate_hz", 200.0, "imu"),
    ("barometer", "barometer", BarometerModel, "baro_rate_hz", 50.0, "barometer"),
    ("magnetometer", "magnetometer", MagnetometerModel, "mag_rate_hz", 100.0, "magnetometer"),
    ("thermometer", "thermometer", ThermometerModel, "thermometer_rate_hz", 0.0, "thermometer"),
    ("hygrometer", "hygrometer", HygrometerModel, "hygrometer_rate_hz", 0.0, "hygrometer"),
    ("light_sensor", "light_sensor", LightSensorModel, "light_sensor_rate_hz", 0.0, "light_sensor"),
    ("gas_sensor", "gas_sensor", GasSensorModel, "gas_sensor_rate_hz", 0.0, "gas_sensor"),
    ("anemometer", "anemometer", AnemometerModel, "anemometer_rate_hz", 0.0, "anemometer"),
    ("airspeed", "airspeed", AirspeedModel, "airspeed_rate_hz", 0.0, "airspeed"),
    ("rangefinder", "rangefinder", RangefinderModel, "rangefinder_rate_hz", 0.0, "rangefinder"),
    ("ultrasonic", "ultrasonic", UltrasonicArrayModel, "ultrasonic_rate_hz", 0.0, "ultrasonic"),
    ("imaging_sonar", "imaging_sonar", ImagingSonarModel, "imaging_sonar_rate_hz", 0.0, "imaging_sonar"),
    ("side_scan", "side_scan", SideScanSonarModel, "side_scan_rate_hz", 0.0, "side_scan"),
    ("dvl", "dvl", DVLModel, "dvl_rate_hz", 0.0, "dvl"),
    (
        "current_profiler",
        "current_profiler",
        AcousticCurrentProfilerModel,
        "current_profiler_rate_hz",
        0.0,
        "current_profiler",
    ),
    ("optical_flow", "optical_flow", OpticalFlowModel, "optical_flow_rate_hz", 0.0, "optical_flow"),
    ("battery", "battery", BatteryModel, "battery_rate_hz", 0.0, "battery"),
    ("stereo_camera", "stereo", StereoCameraModel, "stereo_rate_hz", 0.0, "stereo_camera"),
    ("wheel_odometry", "wheel_odometry", WheelOdometryModel, "wheel_odometry_rate_hz", 0.0, "wheel_odometry"),
    ("force_torque", "force_torque", ForceTorqueSensorModel, "force_torque_rate_hz", 0.0, "force_torque"),
    ("joint_state", "joint_state", JointStateSensor, "joint_state_rate_hz", 0.0, "joint_state"),
    ("contact", "contact", ContactSensor, "contact_rate_hz", 0.0, "contact"),
    ("depth_camera", "depth_camera", DepthCameraModel, "depth_camera_rate_hz", 0.0, "depth_camera"),
    ("tactile_array", "tactile_array", TactileArraySensor, "tactile_array_rate_hz", 0.0, "tactile_array"),
    ("current", "current", CurrentSensor, "current_rate_hz", 0.0, "current"),
    ("rpm", "rpm", RPMSensor, "rpm_rate_hz", 0.0, "rpm"),
)


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

        # Register each non-None sensor under its canonical display name.
        _locals = {
            "rgb_camera": rgb_camera,
            "event_camera": event_camera,
            "thermal_camera": thermal_camera,
            "lidar": lidar,
            "gnss": gnss,
            "radio": radio,
            "uwb": uwb,
            "radar": radar,
            "imu": imu,
            "barometer": barometer,
            "magnetometer": magnetometer,
            "thermometer": thermometer,
            "hygrometer": hygrometer,
            "light_sensor": light_sensor,
            "gas_sensor": gas_sensor,
            "anemometer": anemometer,
            "airspeed": airspeed,
            "rangefinder": rangefinder,
            "ultrasonic": ultrasonic,
            "imaging_sonar": imaging_sonar,
            "side_scan": side_scan,
            "dvl": dvl,
            "current_profiler": current_profiler,
            "optical_flow": optical_flow,
            "battery": battery,
            "stereo_camera": stereo_camera,
            "wheel_odometry": wheel_odometry,
            "force_torque": force_torque,
            "joint_state": joint_state,
            "contact": contact,
            "depth_camera": depth_camera,
            "tactile_array": tactile_array,
            "current": current,
            "rpm": rpm,
        }
        for kwarg, display, _cls, _rate, _dflt, _cfg in _SENSOR_SLOTS:
            sensor = _locals.get(kwarg)
            if sensor is not None:
                self._scheduler.add(sensor, name=display)

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
        n_slots = len(_SENSOR_SLOTS)
        if seed is not None:
            child_seeds = np.random.SeedSequence(seed).spawn(n_slots)
            _seeds: list[int | None] = [int(cs.generate_state(1)[0]) for cs in child_seeds]
        else:
            _seeds = [None] * n_slots

        # Build sensors from the registry using matched rates and seeds.
        rate_overrides = {
            "rgb_rate_hz": rgb_rate_hz,
            "event_rate_hz": event_rate_hz,
            "thermal_rate_hz": thermal_rate_hz,
            "lidar_rate_hz": lidar_rate_hz,
            "gnss_rate_hz": gnss_rate_hz,
            "radio_rate_hz": radio_rate_hz,
            "uwb_rate_hz": uwb_rate_hz,
            "radar_rate_hz": radar_rate_hz,
            "imu_rate_hz": imu_rate_hz,
            "baro_rate_hz": baro_rate_hz,
            "mag_rate_hz": mag_rate_hz,
            "thermometer_rate_hz": thermometer_rate_hz,
            "hygrometer_rate_hz": hygrometer_rate_hz,
            "light_sensor_rate_hz": light_sensor_rate_hz,
            "gas_sensor_rate_hz": gas_sensor_rate_hz,
            "anemometer_rate_hz": anemometer_rate_hz,
            "airspeed_rate_hz": airspeed_rate_hz,
            "rangefinder_rate_hz": rangefinder_rate_hz,
            "ultrasonic_rate_hz": ultrasonic_rate_hz,
            "imaging_sonar_rate_hz": imaging_sonar_rate_hz,
            "side_scan_rate_hz": side_scan_rate_hz,
            "dvl_rate_hz": dvl_rate_hz,
            "current_profiler_rate_hz": current_profiler_rate_hz,
            "optical_flow_rate_hz": optical_flow_rate_hz,
            "battery_rate_hz": battery_rate_hz,
            "stereo_rate_hz": stereo_rate_hz,
            "wheel_odometry_rate_hz": wheel_odometry_rate_hz,
            "force_torque_rate_hz": force_torque_rate_hz,
            "joint_state_rate_hz": joint_state_rate_hz,
            "contact_rate_hz": contact_rate_hz,
            "depth_camera_rate_hz": depth_camera_rate_hz,
            "tactile_array_rate_hz": tactile_array_rate_hz,
            "current_rate_hz": current_rate_hz,
            "rpm_rate_hz": rpm_rate_hz,
        }

        kwargs: dict[str, Any] = {}
        for idx, (_kwarg, _display, model_cls, rate_param, _dflt, _cfg) in enumerate(_SENSOR_SLOTS):
            rate = rate_overrides[rate_param]
            if rate > 0:
                kwargs[_kwarg] = model_cls(update_rate_hz=rate, seed=_seeds[idx])

        return cls(**kwargs)

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
        kwargs: dict[str, Any] = {}
        for kwarg, _display, model_cls, _rate, _dflt, cfg_attr in _SENSOR_SLOTS:
            cfg = getattr(config, cfg_attr, None)
            if cfg is not None:
                kwargs[kwarg] = model_cls.from_config(cfg)

        return cls(**kwargs)

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

    def configure_noise_models(
        self,
        noise_model: str | Mapping[str, str],
        *,
        outlier_prob: float = 0.0,
        outlier_scale: float = 6.0,
    ) -> None:
        """Configure the shared white-noise model for one or all registered sensors."""
        if isinstance(noise_model, Mapping):
            for name, model in noise_model.items():
                self.get_sensor(name).configure_noise_model(
                    model, outlier_prob=outlier_prob, outlier_scale=outlier_scale
                )
            return
        for name in self.sensor_names():
            self.get_sensor(name).configure_noise_model(
                noise_model,
                outlier_prob=outlier_prob,
                outlier_scale=outlier_scale,
            )

    def set_seed(self, base_seed: int) -> None:
        """Re-seed every registered sensor deterministically.

        Each sensor receives an independent child seed derived via
        :class:`numpy.random.SeedSequence` so that streams are
        statistically uncorrelated.

        Parameters
        ----------
        base_seed:
            Integer seed from which all per-sensor seeds are derived.
        """
        names = self._scheduler.sensor_names()
        child_seeds = np.random.SeedSequence(base_seed).spawn(len(names))
        for name, cs in zip(names, child_seeds):
            sensor = self._scheduler.get_sensor(name)
            new_seed = int(cs.generate_state(1)[0])
            new_rng = sensor._make_rng(new_seed) if hasattr(sensor, "_make_rng") else np.random.default_rng(new_seed)
            if hasattr(sensor, "_rng"):
                sensor._rng = new_rng  # type: ignore[assignment]
            if hasattr(sensor, "_seed"):
                sensor._seed = new_seed
            # Also update any GaussMarkovProcess instances that hold
            # their own reference to the sensor's rng.
            for attr_name in dir(sensor):
                attr = getattr(sensor, attr_name, None)
                if isinstance(attr, GaussMarkovProcess):
                    attr._rng = new_rng  # type: ignore[assignment]

    @property
    def scheduler(self) -> SensorScheduler:
        """The underlying :class:`SensorScheduler`."""
        return self._scheduler

    def __repr__(self) -> str:
        names = self.sensor_names()
        return f"SensorSuite(sensors={names})"
