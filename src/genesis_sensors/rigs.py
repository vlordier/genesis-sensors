"""Reusable, self-contained sensor rig factories for Genesis scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

from ._compat import (
    ATI_MINI45,
    BLUEVIEW_P900_130,
    BUMPER_50HZ,
    DAVIS_346,
    NORTEK_DVL1000,
    DAVIS_6410_ANEMOMETER,
    DIFF_DRIVE_ENCODER_50HZ,
    DS18B20_PROBE,
    EDGETECH_4125,
    FLIR_BOSON_320,
    FRANKA_JOINT_ENCODER,
    FINGERTIP_TACTILE_4X4,
    FINGERTIP_TACTILE_200HZ,
    GENERIC_SERVO_ENCODER,
    HC_SR04_ARRAY4,
    INA226_10A,
    INTEL_D435,
    OPTICAL_ENC_1024,
    QORVO_DWM3001C,
    RASPBERRY_PI_V2,
    T_MOTOR_HALL_6P,
    TI_IWR6843AOP,
    VELODYNE_VLP16,
    ZED2_STEREO,
    AcousticCurrentProfilerModel,
    AirspeedModel,
    BarometerModel,
    BatteryModel,
    CameraModel,
    ContactSensor,
    CurrentSensor,
    DVLModel,
    DepthCameraModel,
    EventCameraModel,
    ForceTorqueSensorModel,
    GNSSModel,
    HydrophoneModel,
    IMUModel,
    ImagingSonarModel,
    InclinometerModel,
    JointStateSensor,
    LeakDetectorModel,
    LidarModel,
    LoadCellModel,
    MagnetometerModel,
    MotorTemperatureModel,
    OpticalFlowModel,
    ProximityToFArrayModel,
    RadarModel,
    RPMSensor,
    RadioLinkModel,
    SGP30_AIR_QUALITY,
    SHT31_HUMIDITY,
    RangefinderModel,
    SensorSuite,
    SideScanSonarModel,
    StereoCameraModel,
    TELEDYNE_WORKHORSE_600,
    TSL2591_LIGHT,
    TactileArraySensor,
    ThermalCameraModel,
    ThermometerModel,
    UltrasonicArrayModel,
    UnderwaterModemModel,
    HygrometerModel,
    LightSensorModel,
    GasSensorModel,
    AnemometerModel,
    UWBRangingModel,
    WaterPressureModel,
    WheelOdometryModel,
    WireEncoderModel,
    extract_joint_state,
    extract_link_contact_force_n,
    extract_link_ft_state,
    extract_link_imu_state,
    extract_rigid_body_state,
)

StateBuilder: TypeAlias = Callable[[], dict[str, Any]]
ObservationMap: TypeAlias = dict[str, Mapping[str, Any]]

_DEFAULT_GO2_FEET = ("FR_calf", "FL_calf", "RR_calf", "RL_calf")


@dataclass
class VelocityCache:
    """Track motion state used by higher-level demo rig builders."""

    prev_world_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    frame_idx: int = 0


@dataclass
class SensorRig:
    """Small wrapper around a `SensorSuite` plus a state-building callback."""

    name: str
    suite: SensorSuite
    state_fn: StateBuilder
    metadata: dict[str, Any] = field(default_factory=dict)
    reset_fn: Callable[[], None] | None = None

    def reset(self) -> None:
        self.suite.reset()
        if self.reset_fn is not None:
            self.reset_fn()

    def sensor_names(self) -> list[str]:
        return self.suite.sensor_names()

    def step(self, sim_time: float, extra_state: Mapping[str, Any] | None = None) -> ObservationMap:
        state = self.state_fn()
        if extra_state:
            state.update(dict(extra_state))
        return self.suite.step(sim_time, state)


class NamedContactSensor(ContactSensor):
    """A contact sensor that reads its force from `state['contact_forces'][link_name]`."""

    def __init__(self, link_name: str, **kwargs: Any) -> None:
        kwargs.setdefault("name", f"{link_name}_contact")
        super().__init__(**kwargs)
        self.link_name = link_name

    def step(self, sim_time: float, state: Mapping[str, Any]) -> dict[str, Any]:
        contact_forces = state.get("contact_forces", {})
        force_n = 0.0
        if isinstance(contact_forces, Mapping):
            force_n = float(contact_forces.get(self.link_name, 0.0))
        return super().step(sim_time, {"contact_force_n": force_n})


def _seed_getter(base_seed: int | None, n_children: int = 48) -> Callable[[int], int | None]:
    """Derive deterministic child seeds so individual sensors stay decorrelated."""
    if base_seed is None:
        return lambda _: None

    children = np.random.SeedSequence(base_seed).spawn(n_children)

    def _seed(index: int) -> int | None:
        return int(children[index].generate_state(1)[0])

    return _seed


def _pressure_patch(force_n: float, resolution: tuple[int, int]) -> np.ndarray:
    """Create a small synthetic tactile pressure map from a scalar contact force."""
    width, height = resolution
    pressure = np.zeros((height, width), dtype=np.float32)
    peak = min(120_000.0, max(0.0, force_n) * 18_000.0)
    cy, cx = height // 2, width // 2
    pressure[max(0, cy - 1) : min(height, cy + 1), max(0, cx - 1) : min(width, cx + 1)] = peak
    return pressure


def _reset_motion_cache(cache: VelocityCache) -> None:
    """Reset cached motion state when a rig is reset."""
    cache.prev_world_vel = np.zeros(3, dtype=np.float64)
    cache.frame_idx = 0


def _make_perception_state(
    *,
    sim_time: float,
    frame_idx: int,
    pos: np.ndarray,
    vel: np.ndarray,
    resolution: tuple[int, int] = (96, 72),
    lidar_shape: tuple[int, int] = (8, 64),
) -> dict[str, Any]:
    """Generate a compact synthetic visual/ranging state for richer demo rigs."""
    width, height = resolution
    lidar_channels, lidar_res = lidar_shape

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    speed = float(np.linalg.norm(vel))
    phase = float(0.45 * sim_time + 0.17 * frame_idx)
    glow = np.clip(1.0 - 2.2 * np.sqrt((xx - 0.52) ** 2 + (yy - 0.48) ** 2), 0.0, 1.0)
    stripes = 0.18 * np.sin(6.0 * np.pi * xx + phase) + 0.12 * np.cos(5.0 * np.pi * yy - 0.8 * phase)
    gray = np.clip(0.12 + 0.32 * xx + 0.24 * yy + 0.38 * glow + stripes, 0.0, 1.0).astype(np.float32)

    box_w = max(6, width // 7)
    box_h = max(6, height // 6)
    box_x = int((0.18 * width + 0.28 * width * np.sin(0.31 * phase)) % max(width - box_w, 1))
    box_y = int(np.clip(0.22 * height + 0.16 * height * np.cos(0.23 * phase), 0, max(height - box_h, 0)))
    gray[box_y : box_y + box_h, box_x : box_x + box_w] = 1.0

    rgb = np.stack(
        [
            np.clip(gray + 0.16 * np.sin(2.0 * np.pi * yy + 0.35 * phase), 0.0, 1.0),
            np.clip(0.25 + 0.7 * np.flipud(gray), 0.0, 1.0),
            np.clip(1.0 - 0.55 * gray + 0.18 * np.cos(2.0 * np.pi * xx - 0.2 * phase), 0.0, 1.0),
        ],
        axis=-1,
    )
    rgb_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    depth = np.clip(max(0.35, float(pos[2])) + 1.4 * xx + 0.35 * yy - 0.28 * glow, 0.15, 8.0).astype(np.float32)
    disparity_shift = max(1, min(4, int(round(2.0 + speed))))
    rgb_right = np.roll(rgb_uint8, -disparity_shift, axis=1)
    rgb_right[:, -disparity_shift:] = rgb_uint8[:, -1:]

    seg = np.full((height, width), ThermalCameraModel.SKY_ENTITY_ID, dtype=np.int32)
    seg[2:-2, 2:-2] = 1
    seg[box_y : box_y + box_h, box_x : box_x + box_w] = 2
    hot_mask = ((xx - (0.25 + 0.08 * np.sin(phase))) ** 2 + (yy - 0.72) ** 2) < 0.01
    seg[hot_mask] = 3

    az = np.linspace(-1.0, 1.0, lidar_res, dtype=np.float32)[None, :]
    channel = np.linspace(-1.0, 1.0, lidar_channels, dtype=np.float32).reshape(-1, 1)
    moving_obstacle = np.exp(-((az - 0.55 * np.sin(0.7 * sim_time + 0.2)) ** 2) / 0.03)
    range_image = 5.5 + 1.2 * az + 0.7 * channel - 1.3 * moving_obstacle
    range_image += 0.15 * np.sin(phase + np.pi * channel)
    range_image[:, :3] = 0.0
    range_image = np.clip(range_image, 0.0, 12.0).astype(np.float32)
    intensity_image = np.clip(1.1 - range_image / 12.0 + 0.05 * moving_obstacle, 0.0, 1.0).astype(np.float32)

    obstruction = float(np.clip(0.15 + 0.3 * (0.5 + 0.5 * np.sin(0.18 * frame_idx)), 0.0, 0.85))
    rain_rate = float(1.0 + 3.0 * (0.5 + 0.5 * np.sin(0.11 * frame_idx)))
    cloud_cover = float(np.clip(0.20 + 0.45 * (0.5 + 0.5 * np.cos(0.09 * frame_idx)), 0.0, 0.95))
    ambient_temp_c = float(22.0 - 0.35 * pos[2] + 2.2 * np.sin(0.12 * frame_idx))
    relative_humidity_pct = float(np.clip(48.0 + 6.0 * rain_rate + 18.0 * cloud_cover, 18.0, 98.0))
    illuminance_lux = float(np.clip(42_000.0 * (1.0 - 0.75 * cloud_cover) * (1.0 - 0.55 * obstruction), 80.0, 95_000.0))
    wind_ms = np.array([1.3 + 0.35 * np.sin(0.08 * frame_idx), 0.25 * np.cos(0.16 * frame_idx), 0.0], dtype=np.float64)
    water_current_ms = np.array(
        [0.30 + 0.07 * np.sin(0.18 * frame_idx), -0.10 + 0.04 * np.cos(0.13 * frame_idx), 0.0],
        dtype=np.float64,
    )
    current_layers = [
        {"depth_m": 1.5, "vel": water_current_ms + np.array([-0.03, 0.02, 0.0], dtype=np.float64)},
        {"depth_m": 4.0, "vel": water_current_ms + np.array([0.02, -0.01, 0.0], dtype=np.float64)},
        {"depth_m": 8.0, "vel": water_current_ms + np.array([0.08, -0.05, 0.0], dtype=np.float64)},
    ]
    gas_sources = [
        {
            "pos": np.array([pos[0] + 1.2, pos[1] + 0.3 * np.sin(phase), 0.0], dtype=np.float64),
            "peak_ppm": 1200.0,
            "sigma_m": 0.9,
        }
    ]
    uwb_anchors = [
        {"id": "dock_nw", "pos": np.array([0.0, 0.0, 0.0], dtype=np.float64), "los": obstruction < 0.85},
        {"id": "dock_ne", "pos": np.array([12.0, 0.0, 0.0], dtype=np.float64), "los": True},
        {"id": "dock_sw", "pos": np.array([0.0, 12.0, 0.0], dtype=np.float64), "los": True},
        {"id": "dock_top", "pos": np.array([0.0, 0.0, 6.0], dtype=np.float64), "los": True},
    ]
    radar_targets = [
        {
            "id": "lead_vehicle",
            "pos": pos + np.array([8.0, 1.2 * np.sin(0.35 * sim_time), -0.2], dtype=np.float64),
            "vel": np.array([-0.7, 0.15 * np.cos(0.35 * sim_time), 0.0], dtype=np.float64),
            "rcs_dbsm": 14.0,
        },
        {
            "id": "side_obstacle",
            "pos": pos + np.array([6.0, -2.5 + 0.7 * np.cos(0.22 * sim_time), 0.4], dtype=np.float64),
            "vel": np.array([0.0, 0.12, 0.0], dtype=np.float64),
            "rcs_dbsm": 9.0,
        },
    ]
    ultrasonic_ranges = {
        "front_left": max(0.08, 0.80 + 0.20 * np.sin(0.45 * sim_time)),
        "front_right": max(0.08, 1.05 + 0.18 * np.cos(0.36 * sim_time)),
        "left": max(0.08, 0.60 + 0.14 * np.cos(0.28 * sim_time + 0.2)),
        "right": max(0.08, 1.25 + 0.20 * np.sin(0.31 * sim_time + 0.4)),
    }
    sonar_targets = [
        {
            "id": "dock_pylon",
            "pos": pos + np.array([5.0, 0.4 * np.sin(0.4 * sim_time), -0.3], dtype=np.float64),
            "strength": 1.0,
            "extent_deg": 3.0,
        },
        {
            "id": "reef_port",
            "pos": pos + np.array([7.0, 2.8 + 0.3 * np.cos(0.3 * sim_time), -0.8], dtype=np.float64),
            "strength": 0.85,
            "extent_deg": 4.0,
        },
        {
            "id": "wreck_starboard",
            "pos": pos + np.array([6.5, -3.8 + 0.4 * np.sin(0.25 * sim_time), -1.0], dtype=np.float64),
            "strength": 0.9,
            "extent_deg": 4.5,
        },
    ]
    water_turbidity_ntu = float(np.clip(3.0 + 6.0 * rain_rate + 2.0 * obstruction, 0.5, 40.0))
    water_depth_m = float(2.4 + 0.6 * np.sin(0.18 * frame_idx))
    water_salinity_ppt = float(34.5 + 0.6 * np.sin(0.07 * frame_idx + 0.3))
    water_temperature_c = float(11.5 + 1.4 * np.cos(0.09 * frame_idx))
    water_ingress_ml = float(
        np.clip(
            0.05 + 0.12 * (1.0 + np.sin(0.14 * frame_idx)) + (0.9 if rain_rate >= 6.0 else 0.0),
            0.0,
            2.5,
        )
    )
    hull_breach = rain_rate >= 6.0 and frame_idx % 6 == 0
    acoustic_sources = [
        {
            "pos": pos + np.array([6.5, 1.2 * np.sin(0.22 * sim_time), -water_depth_m], dtype=np.float64),
            "frequency_hz": 18_000.0,
            "source_level_db": 158.0,
        },
        {
            "pos": pos + np.array([10.0, -2.5, -water_depth_m - 0.6], dtype=np.float64),
            "frequency_hz": 26_000.0,
            "source_level_db": 152.0,
        },
    ]
    tof_rows, tof_cols = 8, 8
    tof_x = np.linspace(-1.0, 1.0, tof_cols, dtype=np.float32)[None, :]
    tof_y = np.linspace(-1.0, 1.0, tof_rows, dtype=np.float32).reshape(-1, 1)
    tof_ranges_m = np.clip(
        0.35
        + 0.18 * np.sqrt(tof_x**2 + tof_y**2)
        + 0.05 * np.sin(phase + 2.0 * tof_x)
        - 0.03 * np.cos(phase + 3.0 * tof_y),
        0.08,
        2.8,
    ).astype(np.float32)
    load_force_n = float(45.0 + 8.0 * np.sin(0.33 * sim_time) + 4.0 * speed)
    extension_m = float(np.clip(0.9 + 0.3 * np.sin(0.26 * sim_time), 0.0, 2.0))

    return {
        "rgb": rgb_uint8,
        "rgb_right": rgb_right,
        "gray": gray,
        "depth": depth,
        "seg": seg,
        "range_image": range_image,
        "intensity_image": intensity_image,
        "temperature_map": {1: 21.0 + 2.0 * speed, 2: 54.0 + 6.0 * np.cos(phase), 3: 12.0},
        "obstruction": obstruction,
        "weather": {
            "rain_rate_mm_h": rain_rate,
            "cloud_cover": cloud_cover,
            "ambient_temp_c": ambient_temp_c,
            "relative_humidity_pct": relative_humidity_pct,
            "illuminance_lux": illuminance_lux,
            "wind_speed_ms": float(np.linalg.norm(wind_ms[:2])),
            "wind_direction_deg": float((np.degrees(np.arctan2(wind_ms[1], wind_ms[0])) + 360.0) % 360.0),
        },
        "ambient_temp_c": ambient_temp_c,
        "ambient_temperature_c": ambient_temp_c,
        "temperature_c": ambient_temp_c,
        "relative_humidity_pct": relative_humidity_pct,
        "illuminance_lux": illuminance_lux,
        "wind_ms": wind_ms,
        "water_current_ms": water_current_ms,
        "current_layers": current_layers,
        "gas_sources": gas_sources,
        "uwb_anchors": uwb_anchors,
        "radar_targets": radar_targets,
        "ultrasonic_ranges_m": ultrasonic_ranges,
        "sonar_targets": sonar_targets,
        "water_turbidity_ntu": water_turbidity_ntu,
        "depth_m": water_depth_m,
        "water_temperature_c": water_temperature_c,
        "water_salinity_ppt": water_salinity_ppt,
        "ambient_noise_db": 58.0 + 4.0 * obstruction,
        "acoustic_sources": acoustic_sources,
        "hull_breach": hull_breach,
        "water_ingress_ml": water_ingress_ml,
        "remote_pos": pos + np.array([22.0, -7.0, -water_depth_m], dtype=np.float64),
        "motor_current_a": float(np.clip(10.0 + 3.0 * speed, 0.0, 40.0)),
        "motor_speed_rads": float(240.0 + 40.0 * np.sin(0.22 * sim_time)),
        "tof_ranges_m": tof_ranges_m,
        "load_force_n": load_force_n,
        "extension_m": extension_m,
        "range_m": max(0.05, float(pos[2])),
    }


def _build_multimodal_suite(seed_for: Callable[[int], int | None]) -> tuple[SensorSuite, RadioLinkModel]:
    """Create a compact perception-heavy sensor suite using upstream presets."""
    rgb_cfg = RASPBERRY_PI_V2.model_copy(update={"name": "front_rgb", "resolution": (96, 72), "seed": seed_for(0)})
    stereo_cfg = ZED2_STEREO.model_copy(update={"name": "front_stereo", "resolution": (96, 72), "seed": seed_for(1)})
    thermal_cfg = FLIR_BOSON_320.model_copy(
        update={"name": "thermal_view", "resolution": (96, 72), "seed": seed_for(2)}
    )
    lidar_cfg = VELODYNE_VLP16.model_copy(
        update={"name": "front_lidar", "n_channels": 8, "h_resolution": 64, "max_range_m": 12.0, "seed": seed_for(3)}
    )
    radio = RadioLinkModel(name="telemetry_radio", update_rate_hz=60.0, seed=seed_for(4))

    suite = SensorSuite(
        rgb_camera=CameraModel.from_config(rgb_cfg),
        event_camera=EventCameraModel.from_config(DAVIS_346.model_copy(update={"seed": seed_for(5)})),
        thermal_camera=ThermalCameraModel.from_config(thermal_cfg),
        lidar=LidarModel.from_config(lidar_cfg),
        stereo_camera=StereoCameraModel.from_config(stereo_cfg),
        radio=radio,
        uwb=UWBRangingModel.from_config(QORVO_DWM3001C.model_copy(update={"seed": seed_for(6)})),
        radar=RadarModel.from_config(TI_IWR6843AOP.model_copy(update={"seed": seed_for(7)})),
        imu=IMUModel(update_rate_hz=200.0, seed=seed_for(8)),
        gnss=GNSSModel(update_rate_hz=5.0, noise_m=0.25, vel_noise_ms=0.02, seed=seed_for(9)),
        barometer=BarometerModel(update_rate_hz=50.0, seed=seed_for(10)),
        magnetometer=MagnetometerModel(update_rate_hz=100.0, seed=seed_for(11)),
        thermometer=ThermometerModel.from_config(DS18B20_PROBE.model_copy(update={"seed": seed_for(12)})),
        hygrometer=HygrometerModel.from_config(SHT31_HUMIDITY.model_copy(update={"seed": seed_for(13)})),
        light_sensor=LightSensorModel.from_config(TSL2591_LIGHT.model_copy(update={"seed": seed_for(14)})),
        gas_sensor=GasSensorModel.from_config(SGP30_AIR_QUALITY.model_copy(update={"seed": seed_for(15)})),
        anemometer=AnemometerModel.from_config(DAVIS_6410_ANEMOMETER.model_copy(update={"seed": seed_for(16)})),
        airspeed=AirspeedModel(update_rate_hz=50.0, seed=seed_for(17)),
        rangefinder=RangefinderModel(update_rate_hz=20.0, seed=seed_for(18)),
        ultrasonic=UltrasonicArrayModel.from_config(HC_SR04_ARRAY4.model_copy(update={"seed": seed_for(19)})),
        imaging_sonar=ImagingSonarModel.from_config(BLUEVIEW_P900_130.model_copy(update={"seed": seed_for(20)})),
        side_scan=SideScanSonarModel.from_config(EDGETECH_4125.model_copy(update={"seed": seed_for(21)})),
        dvl=DVLModel.from_config(NORTEK_DVL1000.model_copy(update={"seed": seed_for(22)})),
        current_profiler=AcousticCurrentProfilerModel.from_config(
            TELEDYNE_WORKHORSE_600.model_copy(update={"seed": seed_for(23)})
        ),
        water_pressure=WaterPressureModel(update_rate_hz=10.0, seed=seed_for(24)),
        hydrophone=HydrophoneModel(update_rate_hz=4.0, seed=seed_for(25)),
        leak_detector=LeakDetectorModel(update_rate_hz=2.0, seed=seed_for(26)),
        underwater_modem=UnderwaterModemModel(update_rate_hz=2.0, seed=seed_for(27)),
        optical_flow=OpticalFlowModel(update_rate_hz=100.0, seed=seed_for(28)),
        battery=BatteryModel(n_cells=4, capacity_mah=5000.0, seed=seed_for(29)),
        wheel_odometry=WheelOdometryModel.from_config(
            DIFF_DRIVE_ENCODER_50HZ.model_copy(update={"seed": seed_for(30)})
        ),
        inclinometer=InclinometerModel(update_rate_hz=50.0, seed=seed_for(31)),
        proximity_tof=ProximityToFArrayModel(update_rate_hz=15.0, seed=seed_for(32)),
        load_cell=LoadCellModel(update_rate_hz=25.0, seed=seed_for(33)),
        wire_encoder=WireEncoderModel(update_rate_hz=50.0, seed=seed_for(34)),
        motor_temperature=MotorTemperatureModel(update_rate_hz=10.0, seed=seed_for(35)),
    )
    return suite, radio


def _queue_demo_packet(radio: RadioLinkModel, *, sim_time: float, pos: np.ndarray, frame_idx: int) -> None:
    """Push a small telemetry packet periodically so radio observations stay interesting."""
    if frame_idx % 20 != 0:
        return
    dst = np.array([0.0, 0.0, max(0.5, float(pos[2]))], dtype=np.float64)
    radio.transmit(
        packet={"kind": "telemetry", "frame": frame_idx},
        src_pos=np.asarray(pos, dtype=np.float64),
        dst_pos=dst,
        sim_time=sim_time,
        has_los=frame_idx % 60 != 40,
    )


def make_drone_navigation_rig(entity: Any, *, dt: float = 0.01, seed: int | None = 0) -> SensorRig:
    """Create a navigation-oriented sensor bundle for a Genesis drone entity."""
    cache = VelocityCache()
    seed_for = _seed_getter(seed)

    suite = SensorSuite(
        imu=IMUModel(update_rate_hz=200.0, seed=seed_for(0)),
        gnss=GNSSModel(update_rate_hz=5.0, noise_m=0.25, vel_noise_ms=0.02, seed=seed_for(1)),
        barometer=BarometerModel(update_rate_hz=50.0, seed=seed_for(2)),
        magnetometer=MagnetometerModel(update_rate_hz=100.0, seed=seed_for(3)),
        uwb=UWBRangingModel.from_config(QORVO_DWM3001C.model_copy(update={"seed": seed_for(4)})),
        radar=RadarModel.from_config(TI_IWR6843AOP.model_copy(update={"seed": seed_for(5)})),
        thermometer=ThermometerModel.from_config(DS18B20_PROBE.model_copy(update={"seed": seed_for(6)})),
        hygrometer=HygrometerModel.from_config(SHT31_HUMIDITY.model_copy(update={"seed": seed_for(7)})),
        light_sensor=LightSensorModel.from_config(TSL2591_LIGHT.model_copy(update={"seed": seed_for(8)})),
        anemometer=AnemometerModel.from_config(DAVIS_6410_ANEMOMETER.model_copy(update={"seed": seed_for(9)})),
        airspeed=AirspeedModel(update_rate_hz=50.0, seed=seed_for(10)),
        rangefinder=RangefinderModel(update_rate_hz=20.0, seed=seed_for(11)),
        ultrasonic=UltrasonicArrayModel.from_config(HC_SR04_ARRAY4.model_copy(update={"seed": seed_for(12)})),
        optical_flow=OpticalFlowModel(update_rate_hz=100.0, seed=seed_for(13)),
        battery=BatteryModel(n_cells=4, capacity_mah=5000.0, seed=seed_for(14)),
    )

    def _state_fn() -> dict[str, Any]:
        sim_time = cache.frame_idx * dt
        state: dict[str, Any] = dict(
            extract_rigid_body_state(
                entity,
                prev_vel=cache.prev_world_vel,
                dt=dt,
                current_a=12.0,
                wind_ms=np.array([0.8, 0.0, 0.0], dtype=np.float64),
            )
        )
        state.update(
            _make_perception_state(
                sim_time=sim_time,
                frame_idx=cache.frame_idx,
                pos=np.asarray(state["pos"], dtype=np.float64),
                vel=np.asarray(state["vel"], dtype=np.float64),
                resolution=(64, 48),
                lidar_shape=(6, 48),
            )
        )
        cache.prev_world_vel = np.asarray(state["vel"], dtype=np.float64).copy()
        cache.frame_idx += 1
        return state

    return SensorRig(
        name="drone_navigation",
        suite=suite,
        state_fn=_state_fn,
        metadata={"dt": dt, "profile": "navigation"},
        reset_fn=lambda: _reset_motion_cache(cache),
    )


def make_franka_wrist_rig(
    entity: Any,
    *,
    hand_link: str = "hand",
    arm_dofs: list[int] | None = None,
    dt: float = 0.01,
    seed: int | None = 0,
) -> SensorRig:
    """Create a wrist/proprioception-focused rig for Franka-like manipulators."""
    cache = VelocityCache()
    seed_for = _seed_getter(seed)
    arm_dofs = arm_dofs or list(range(7))
    depth_cfg = INTEL_D435.model_copy(update={"name": "wrist_depth", "resolution": (64, 48)})

    suite = SensorSuite(
        joint_state=JointStateSensor.from_config(FRANKA_JOINT_ENCODER.model_copy(update={"seed": seed_for(0)})),
        force_torque=ForceTorqueSensorModel.from_config(ATI_MINI45.model_copy(update={"seed": seed_for(1)})),
        contact=ContactSensor.from_config(FINGERTIP_TACTILE_200HZ.model_copy(update={"seed": seed_for(2)})),
        depth_camera=DepthCameraModel.from_config(depth_cfg.model_copy(update={"seed": seed_for(3)})),
        tactile_array=TactileArraySensor.from_config(FINGERTIP_TACTILE_4X4.model_copy(update={"seed": seed_for(4)})),
        current=CurrentSensor.from_config(INA226_10A.model_copy(update={"seed": seed_for(5)})),
        rpm=RPMSensor.from_config(OPTICAL_ENC_1024.model_copy(update={"seed": seed_for(6)})),
        imu=IMUModel(update_rate_hz=200.0, seed=seed_for(7)),
    )

    def _state_fn() -> dict[str, Any]:
        joint_state: dict[str, Any] = dict(extract_joint_state(entity, dofs_idx_local=arm_dofs))
        hand_state: dict[str, Any] = dict(
            extract_link_imu_state(entity, hand_link, prev_vel_world=cache.prev_world_vel, dt=dt)
        )
        cache.prev_world_vel = np.asarray(hand_state["vel"], dtype=np.float64).copy()

        contact_force_n = extract_link_contact_force_n(entity, hand_link)
        current_a = float(np.clip(np.abs(joint_state["joint_torque"]).mean() * 0.35, 0.0, INA226_10A.range_a))
        rpm = float(np.abs(joint_state["joint_vel"]).mean() * 60.0 / (2.0 * np.pi))
        depth = np.full(
            (depth_cfg.resolution[1], depth_cfg.resolution[0]),
            max(0.15, float(hand_state["pos"][2])),
            dtype=np.float32,
        )

        state = {
            **joint_state,
            **dict(extract_link_ft_state(entity, hand_link)),
            "contact_force_n": contact_force_n,
            "pressure_map": _pressure_patch(contact_force_n, FINGERTIP_TACTILE_4X4.resolution),
            "depth": depth,
            "current_a": current_a,
            "voltage_v": INA226_10A.voltage_nominal_v,
            "rpm": rpm,
        }
        return state

    return SensorRig(
        name="franka_wrist",
        suite=suite,
        state_fn=_state_fn,
        metadata={"hand_link": hand_link, "profile": "manipulation"},
        reset_fn=lambda: _reset_motion_cache(cache),
    )


def make_synthetic_multimodal_rig(*, dt: float = 0.05, seed: int | None = 0) -> SensorRig:
    """Create a headless full-stack rig mirroring the richer upstream sensor examples."""
    cache = VelocityCache()
    seed_for = _seed_getter(seed, n_children=48)
    suite, radio = _build_multimodal_suite(seed_for)

    def _state_fn() -> dict[str, Any]:
        sim_time = cache.frame_idx * dt
        pos = np.array(
            [0.8 * np.cos(0.35 * sim_time), 0.6 * np.sin(0.45 * sim_time), 1.1 + 0.15 * np.cos(0.25 * sim_time)],
            dtype=np.float64,
        )
        vel = np.array(
            [-0.28 * np.sin(0.35 * sim_time), 0.27 * np.cos(0.45 * sim_time), -0.04 * np.sin(0.25 * sim_time)],
            dtype=np.float64,
        )
        ang_vel = np.array([0.05 * np.sin(sim_time), 0.08 * np.cos(0.5 * sim_time), 0.22], dtype=np.float64)
        state = {
            "pos": pos,
            "vel": vel,
            "lin_acc": (vel - cache.prev_world_vel) / max(dt, 1e-6),
            "ang_vel": ang_vel,
            "gravity_body": np.array([0.0, 0.0, 9.80665], dtype=np.float64),
            "current_a": float(np.clip(8.0 + 2.5 * np.linalg.norm(vel), 0.0, 30.0)),
            "voltage_v": 14.8,
            "wind_ms": np.array([1.2, 0.2 * np.cos(sim_time), 0.0], dtype=np.float64),
        }
        state.update(
            _make_perception_state(
                sim_time=sim_time,
                frame_idx=cache.frame_idx,
                pos=pos,
                vel=vel,
            )
        )
        _queue_demo_packet(radio, sim_time=sim_time, pos=pos, frame_idx=cache.frame_idx)
        cache.prev_world_vel = vel.copy()
        cache.frame_idx += 1
        return state

    return SensorRig(
        name="synthetic_multimodal",
        suite=suite,
        state_fn=_state_fn,
        metadata={"dt": dt, "profile": "synthetic_multimodal"},
        reset_fn=lambda: _reset_motion_cache(cache),
    )


def make_drone_perception_rig(entity: Any, *, dt: float = 0.01, seed: int | None = 0) -> SensorRig:
    """Create a richer drone rig with RGB, stereo, event, thermal, LiDAR, and radio sensors."""
    cache = VelocityCache()
    seed_for = _seed_getter(seed, n_children=48)
    suite, radio = _build_multimodal_suite(seed_for)

    def _state_fn() -> dict[str, Any]:
        sim_time = cache.frame_idx * dt
        state: dict[str, Any] = dict(
            extract_rigid_body_state(
                entity,
                prev_vel=cache.prev_world_vel,
                dt=dt,
                current_a=12.0,
                wind_ms=np.array([0.8, 0.1 * np.cos(sim_time), 0.0], dtype=np.float64),
            )
        )
        pos = np.asarray(state["pos"], dtype=np.float64)
        vel = np.asarray(state["vel"], dtype=np.float64)
        state.update(_make_perception_state(sim_time=sim_time, frame_idx=cache.frame_idx, pos=pos, vel=vel))
        _queue_demo_packet(radio, sim_time=sim_time, pos=pos, frame_idx=cache.frame_idx)
        cache.prev_world_vel = vel.copy()
        cache.frame_idx += 1
        return state

    return SensorRig(
        name="drone_perception",
        suite=suite,
        state_fn=_state_fn,
        metadata={"dt": dt, "profile": "perception"},
        reset_fn=lambda: _reset_motion_cache(cache),
    )


def make_go2_rig(
    entity: Any,
    *,
    dt: float = 0.01,
    foot_links: tuple[str, ...] = _DEFAULT_GO2_FEET,
    seed: int | None = 0,
) -> SensorRig:
    """Create a proprioception + per-foot contact rig for Go2-like quadrupeds."""
    cache = VelocityCache()
    seed_for = _seed_getter(seed, n_children=48)

    extra_sensors: list[tuple[str, Any]] = []
    for idx, foot_link in enumerate(foot_links):
        contact_cfg = BUMPER_50HZ.model_copy(update={"name": f"{foot_link}_contact", "seed": seed_for(8 + idx)})
        extra_sensors.append(
            (f"{foot_link}_contact", NamedContactSensor(link_name=foot_link, **contact_cfg.model_dump()))
        )

    suite = SensorSuite(
        imu=IMUModel(update_rate_hz=200.0, seed=seed_for(0)),
        joint_state=JointStateSensor.from_config(GENERIC_SERVO_ENCODER.model_copy(update={"seed": seed_for(1)})),
        current=CurrentSensor.from_config(INA226_10A.model_copy(update={"seed": seed_for(2)})),
        rpm=RPMSensor.from_config(T_MOTOR_HALL_6P.model_copy(update={"seed": seed_for(3)})),
        extra_sensors=extra_sensors,
    )

    def _state_fn() -> dict[str, Any]:
        state: dict[str, Any] = dict(
            extract_rigid_body_state(
                entity,
                prev_vel=cache.prev_world_vel,
                dt=dt,
                current_a=0.0,
                wind_ms=np.array([0.2, 0.0, 0.0], dtype=np.float64),
            )
        )
        cache.prev_world_vel = np.asarray(state["vel"], dtype=np.float64).copy()

        joint_state: dict[str, Any] = dict(extract_joint_state(entity))
        current_a = float(np.clip(np.abs(joint_state["joint_torque"]).mean() * 0.2, 0.0, INA226_10A.range_a))
        rpm = float(np.abs(joint_state["joint_vel"]).mean() * 60.0 / (2.0 * np.pi))
        state.update(joint_state)
        state["current_a"] = current_a
        state["voltage_v"] = 28.8
        state["rpm"] = rpm
        state["contact_forces"] = {foot: extract_link_contact_force_n(entity, foot) for foot in foot_links}
        return state

    return SensorRig(
        name="go2_proprioception",
        suite=suite,
        state_fn=_state_fn,
        metadata={"foot_links": foot_links, "profile": "quadruped"},
        reset_fn=lambda: _reset_motion_cache(cache),
    )


def _wheel_odometry_state(
    entity: Any,
    wheel_joint_indices: list[int],
    wheel_radius: float,
    prev_wheel_angles: np.ndarray | None,
    dt: float,
) -> dict[str, Any]:
    """Extract per-wheel odometry state from Genesis entity joint velocities.

    Returns a dict with per-wheel angular velocity and estimated ground velocity,
    suitable for feeding WheelOdometryModel instances.
    """
    import numpy as np

    try:
        dof_vels = entity.get_dofs_velocity()
        if hasattr(dof_vels, "cpu"):
            dof_vels = dof_vels.cpu().numpy().ravel()
        else:
            dof_vels = np.asarray(dof_vels).ravel()
    except (AttributeError, RuntimeError):
        dof_vels = np.zeros(max(wheel_joint_indices) + 1 if wheel_joint_indices else 4)

    wheel_ang_vels: dict[str, float] = {}
    wheel_names = ["FL", "FR", "RL", "RR"]
    for i, idx in enumerate(wheel_joint_indices):
        name = wheel_names[i] if i < len(wheel_names) else f"wheel_{i}"
        omega = float(dof_vels[idx]) if idx < len(dof_vels) else 0.0
        wheel_ang_vels[name] = omega

    result: dict[str, Any] = {
        "wheel_angular_velocity_rads": wheel_ang_vels,
        "wheel_radius_m": wheel_radius,
    }
    return result


def _build_ugv_sensor_suite(
    seed_for: Callable[[int], int | None],
) -> tuple[SensorSuite, dict[str, WheelOdometryModel]]:
    """Create a UGV-focused sensor suite with 4 independent wheel odometry sensors."""
    lidar_cfg = VELODYNE_VLP16.model_copy(
        update={"name": "front_lidar", "n_channels": 16, "h_resolution": 360, "max_range_m": 100.0, "seed": seed_for(0)}
    )
    rgb_cfg = RASPBERRY_PI_V2.model_copy(
        update={"name": "front_rgb", "resolution": (640, 480), "seed": seed_for(1)}
    )

    wheel_odom: dict[str, WheelOdometryModel] = {}
    wheel_names = ["FL", "FR", "RL", "RR"]
    for i, name in enumerate(wheel_names):
        w_cfg = DIFF_DRIVE_ENCODER_50HZ.model_copy(
            update={"name": f"wheel_odometry_{name}", "seed": seed_for(10 + i)}
        )
        wheel_odom[name] = WheelOdometryModel.from_config(w_cfg)

    extra_sensors: list[tuple[str, Any]] = []
    for name, w_odom in wheel_odom.items():
        extra_sensors.append((f"wheel_odometry_{name}", w_odom))

    suite = SensorSuite(
        rgb_camera=CameraModel.from_config(rgb_cfg),
        lidar=LidarModel.from_config(lidar_cfg),
        imu=IMUModel(update_rate_hz=100.0, seed=seed_for(2)),
        gnss=GNSSModel(update_rate_hz=5.0, noise_m=0.25, vel_noise_ms=0.02, seed=seed_for(3)),
        barometer=BarometerModel(update_rate_hz=50.0, seed=seed_for(4)),
        inclinometer=InclinometerModel(update_rate_hz=50.0, seed=seed_for(5)),
        magnetometer=MagnetometerModel(update_rate_hz=50.0, seed=seed_for(6)),
        battery=BatteryModel(n_cells=4, capacity_mah=10000.0, seed=seed_for(7)),
        extra_sensors=extra_sensors,
    )
    return suite, wheel_odom


def _ugv_synthetic_state(
    *,
    sim_time: float,
    frame_idx: int,
    prev_pos: np.ndarray,
    prev_vel: np.ndarray,
    dt: float,
    seed: int = 0,
) -> dict[str, Any]:
    """Generate synthetic UGV motion state for headless/synthetic rig mode."""
    rng = np.random.default_rng(seed + frame_idx)
    speed = 1.5 + 0.3 * np.sin(0.2 * sim_time)
    heading = 0.3 * np.sin(0.15 * sim_time)
    vx = speed * np.cos(heading)
    vy = speed * np.sin(heading)
    vel = np.array([vx, vy, 0.0], dtype=np.float64)
    pos = prev_pos.astype(np.float64) + vel * dt
    ang_vel = np.array([0.0, 0.0, 0.15 * np.cos(0.15 * sim_time)], dtype=np.float64)
    lin_acc = (vel - prev_vel) / max(dt, 1e-6)

    wheel_base_omega = speed / 0.35
    wheel_angular = {
        "FL": float(wheel_base_omega + ang_vel[2] * 0.8 + rng.normal(0, 0.1)),
        "FR": float(wheel_base_omega - ang_vel[2] * 0.8 + rng.normal(0, 0.1)),
        "RL": float(wheel_base_omega + ang_vel[2] * 0.8 + rng.normal(0, 0.1)),
        "RR": float(wheel_base_omega - ang_vel[2] * 0.8 + rng.normal(0, 0.1)),
    }

    return {
        "pos": pos,
        "vel": vel,
        "lin_acc": lin_acc,
        "ang_vel": ang_vel,
        "gravity_body": np.array([0.0, 0.0, 9.80665], dtype=np.float64),
        "current_a": float(np.clip(8.0 + 2.0 * speed, 0.0, 30.0)),
        "voltage_v": 24.0,
        "wheel_angular_velocity_rads": wheel_angular,
        "wheel_radius_m": 0.35,
    }


def make_ugv_rig(
    genesis_entity: Any = None,
    *,
    wheel_joint_indices: list[int] | None = None,
    wheel_radius: float = 0.35,
    dt: float = 0.01,
    seed: int | None = 0,
) -> SensorRig:
    """Create a multi-modal sensor rig for wheeled UGV (Unmanned Ground Vehicle).

    Includes 4 independent wheel odometry sensors (FL, FR, RL, RR) each with
    configurable slip and noise, plus IMU, GNSS, LiDAR, RGB camera, barometer,
    inclinometer, magnetometer, and battery monitor.

    When *genesis_entity* is None the rig operates in synthetic/headless mode
    using a kinematic model. When an entity is provided, wheel velocities are
    extracted from Genesis joint states.

    Args:
        genesis_entity: Live Genesis entity, or None for synthetic mode.
        wheel_joint_indices: DOF indices for [FL, FR, RL, RR] wheel joints.
            Defaults to [0, 1, 2, 3].
        wheel_radius: Wheel radius in metres (default 0.35).
        dt: Simulation timestep in seconds.
        seed: RNG seed for reproducible sensor noise.

    Returns:
        SensorRig with 4-wheel odometry + full UGV sensor suite.
    """
    cache = VelocityCache()
    seed_for = _seed_getter(seed, n_children=48)
    suite, wheel_odom = _build_ugv_sensor_suite(seed_for)
    wheel_indices = wheel_joint_indices or [0, 1, 2, 3]

    if genesis_entity is not None:
        def _state_fn() -> dict[str, Any]:
            sim_time = cache.frame_idx * dt
            state: dict[str, Any] = dict(
                extract_rigid_body_state(
                    genesis_entity,
                    prev_vel=cache.prev_world_vel,
                    dt=dt,
                    current_a=12.0,
                    wind_ms=np.array([0.5, 0.0, 0.0], dtype=np.float64),
                )
            )
            wheel_state = _wheel_odometry_state(
                genesis_entity, wheel_indices, wheel_radius,
                None, dt,
            )
            state.update(wheel_state)
            cache.prev_world_vel = np.asarray(state["vel"], dtype=np.float64).copy()
            cache.frame_idx += 1
            return state

        profile = "ugv_attached"
    else:
        prev_pos = np.zeros(3, dtype=np.float64)

        def _synthetic_closure() -> dict[str, Any]:
            nonlocal prev_pos
            sim_time = cache.frame_idx * dt
            state = _ugv_synthetic_state(
                sim_time=sim_time, frame_idx=cache.frame_idx,
                prev_pos=prev_pos, prev_vel=cache.prev_world_vel,
                dt=dt, seed=seed or 0,
            )
            prev_pos = np.asarray(state["pos"], dtype=np.float64).copy()
            cache.prev_world_vel = np.asarray(state["vel"], dtype=np.float64).copy()
            cache.frame_idx += 1
            return state

        _state_fn = _synthetic_closure
        profile = "ugv_synthetic"

    return SensorRig(
        name="ugv_perception",
        suite=suite,
        state_fn=_state_fn,
        metadata={
            "dt": dt,
            "profile": profile,
            "wheel_count": 4,
            "wheel_radius": wheel_radius,
            "wheel_joint_indices": wheel_indices,
        },
        reset_fn=lambda: _reset_motion_cache(cache),
    )
