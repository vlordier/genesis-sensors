"""Reusable, self-contained sensor rig factories for Genesis scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

from ._compat import (
    ATI_MINI45,
    BUMPER_50HZ,
    DAVIS_346,
    DIFF_DRIVE_ENCODER_50HZ,
    FLIR_BOSON_320,
    FRANKA_JOINT_ENCODER,
    FINGERTIP_TACTILE_4X4,
    FINGERTIP_TACTILE_200HZ,
    GENERIC_SERVO_ENCODER,
    INA226_10A,
    INTEL_D435,
    OPTICAL_ENC_1024,
    RASPBERRY_PI_V2,
    T_MOTOR_HALL_6P,
    VELODYNE_VLP16,
    ZED2_STEREO,
    AirspeedModel,
    BarometerModel,
    BatteryModel,
    CameraModel,
    ContactSensor,
    CurrentSensor,
    DepthCameraModel,
    EventCameraModel,
    ForceTorqueSensorModel,
    GNSSModel,
    IMUModel,
    JointStateSensor,
    LidarModel,
    MagnetometerModel,
    OpticalFlowModel,
    RPMSensor,
    RadioLinkModel,
    RangefinderModel,
    SensorSuite,
    StereoCameraModel,
    TactileArraySensor,
    ThermalCameraModel,
    WheelOdometryModel,
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


def _seed_getter(base_seed: int | None, n_children: int = 16) -> Callable[[int], int | None]:
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

    return {
        "rgb": rgb_uint8,
        "rgb_right": rgb_right,
        "gray": gray,
        "depth": depth,
        "seg": seg,
        "range_image": range_image,
        "intensity_image": intensity_image,
        "temperature_map": {1: 21.0 + 2.0 * speed, 2: 54.0 + 6.0 * np.cos(phase), 3: 12.0},
        "obstruction": float(np.clip(0.15 + 0.3 * (0.5 + 0.5 * np.sin(0.18 * frame_idx)), 0.0, 0.85)),
        "weather": {"rain_rate_mm_h": float(1.0 + 3.0 * (0.5 + 0.5 * np.sin(0.11 * frame_idx)))},
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
        imu=IMUModel(update_rate_hz=200.0, seed=seed_for(6)),
        gnss=GNSSModel(update_rate_hz=5.0, noise_m=0.25, vel_noise_ms=0.02, seed=seed_for(7)),
        barometer=BarometerModel(update_rate_hz=50.0, seed=seed_for(8)),
        magnetometer=MagnetometerModel(update_rate_hz=100.0, seed=seed_for(9)),
        airspeed=AirspeedModel(update_rate_hz=50.0, seed=seed_for(10)),
        rangefinder=RangefinderModel(update_rate_hz=20.0, seed=seed_for(11)),
        optical_flow=OpticalFlowModel(update_rate_hz=100.0, seed=seed_for(12)),
        battery=BatteryModel(n_cells=4, capacity_mah=5000.0, seed=seed_for(13)),
        wheel_odometry=WheelOdometryModel.from_config(
            DIFF_DRIVE_ENCODER_50HZ.model_copy(update={"seed": seed_for(14)})
        ),
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
        airspeed=AirspeedModel(update_rate_hz=50.0, seed=seed_for(4)),
        rangefinder=RangefinderModel(update_rate_hz=20.0, seed=seed_for(5)),
        optical_flow=OpticalFlowModel(update_rate_hz=100.0, seed=seed_for(6)),
        battery=BatteryModel(n_cells=4, capacity_mah=5000.0, seed=seed_for(7)),
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
    seed_for = _seed_getter(seed, n_children=32)
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
    seed_for = _seed_getter(seed, n_children=32)
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
    seed_for = _seed_getter(seed, n_children=32)

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
