"""Reusable, self-contained sensor rig factories for Genesis scenes."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, TypeAlias

import numpy as np

from genesis.sensors import (
    ATI_MINI45,
    BUMPER_50HZ,
    AirspeedModel,
    BarometerModel,
    BatteryModel,
    ContactSensor,
    CurrentSensor,
    DepthCameraModel,
    FINGERTIP_TACTILE_4X4,
    FINGERTIP_TACTILE_200HZ,
    FRANKA_JOINT_ENCODER,
    GNSSModel,
    GENERIC_SERVO_ENCODER,
    IMUModel,
    INA226_10A,
    INTEL_D435,
    JointStateSensor,
    MagnetometerModel,
    OPTICAL_ENC_1024,
    OpticalFlowModel,
    RPMSensor,
    RangefinderModel,
    SensorSuite,
    T_MOTOR_HALL_6P,
    TactileArraySensor,
    ForceTorqueSensorModel,
)
from genesis.sensors.genesis_bridge import (
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
    """Track the previous world-frame velocity for finite-difference IMU acceleration."""

    prev_world_vel: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))


@dataclass
class SensorRig:
    """Small wrapper around a `SensorSuite` plus a state-building callback."""

    name: str
    suite: SensorSuite
    state_fn: StateBuilder
    metadata: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.suite.reset()

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
        state = extract_rigid_body_state(
            entity,
            prev_vel=cache.prev_world_vel,
            dt=dt,
            current_a=12.0,
            wind_ms=np.array([0.8, 0.0, 0.0], dtype=np.float64),
        )
        cache.prev_world_vel = np.asarray(state["vel"], dtype=np.float64).copy()
        return state

    return SensorRig(name="drone_navigation", suite=suite, state_fn=_state_fn, metadata={"dt": dt})


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
    )

    def _state_fn() -> dict[str, Any]:
        joint_state = extract_joint_state(entity, dofs_idx_local=arm_dofs)
        hand_state = extract_link_imu_state(entity, hand_link, prev_vel_world=cache.prev_world_vel, dt=dt)
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
            **extract_link_ft_state(entity, hand_link),
            "contact_force_n": contact_force_n,
            "pressure_map": _pressure_patch(contact_force_n, FINGERTIP_TACTILE_4X4.resolution),
            "depth": depth,
            "current_a": current_a,
            "voltage_v": INA226_10A.voltage_nominal_v,
            "rpm": rpm,
        }
        return state

    return SensorRig(name="franka_wrist", suite=suite, state_fn=_state_fn, metadata={"hand_link": hand_link})


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

    extra_sensors = []
    for idx, foot_link in enumerate(foot_links):
        contact_cfg = BUMPER_50HZ.model_copy(update={"name": f"{foot_link}_contact", "seed": seed_for(8 + idx)})
        extra_sensors.append((f"{foot_link}_contact", NamedContactSensor(link_name=foot_link, **contact_cfg.model_dump())))

    suite = SensorSuite(
        imu=IMUModel(update_rate_hz=200.0, seed=seed_for(0)),
        joint_state=JointStateSensor.from_config(GENERIC_SERVO_ENCODER.model_copy(update={"seed": seed_for(1)})),
        current=CurrentSensor.from_config(INA226_10A.model_copy(update={"seed": seed_for(2)})),
        rpm=RPMSensor.from_config(T_MOTOR_HALL_6P.model_copy(update={"seed": seed_for(3)})),
        extra_sensors=extra_sensors,
    )

    def _state_fn() -> dict[str, Any]:
        state = extract_rigid_body_state(
            entity,
            prev_vel=cache.prev_world_vel,
            dt=dt,
            current_a=0.0,
            wind_ms=np.array([0.2, 0.0, 0.0], dtype=np.float64),
        )
        cache.prev_world_vel = np.asarray(state["vel"], dtype=np.float64).copy()

        joint_state = extract_joint_state(entity)
        current_a = float(np.clip(np.abs(joint_state["joint_torque"]).mean() * 0.2, 0.0, INA226_10A.range_a))
        rpm = float(np.abs(joint_state["joint_vel"]).mean() * 60.0 / (2.0 * np.pi))
        state.update(joint_state)
        state["current_a"] = current_a
        state["voltage_v"] = 28.8
        state["rpm"] = rpm
        state["contact_forces"] = {foot: extract_link_contact_force_n(entity, foot) for foot in foot_links}
        return state

    return SensorRig(name="go2_proprioception", suite=suite, state_fn=_state_fn, metadata={"foot_links": foot_links})
