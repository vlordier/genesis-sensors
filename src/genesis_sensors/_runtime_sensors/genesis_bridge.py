"""
Genesis Physics → External Sensor Layer Bridge.

This module provides utility functions that read simulation state from
Genesis entities after ``scene.step()`` and convert it into the ``state``
dict expected by :class:`~genesis.sensors.SensorSuite` and individual
sensor ``step()`` methods.

All heavy imports (``genesis.utils.misc``, ``numpy``) are resolved lazily
inside each function so that importing this module does *not* trigger
Genesis initialisation.

Quick reference
---------------
:func:`extract_rigid_body_state`
    Navigation sensors for drones and mobile bases — IMU, GNSS, barometer,
    magnetometer, airspeed, rangefinder, optical-flow, battery, stereo
    camera, depth camera.

:func:`extract_joint_state`
    Proprioceptive sensors for arms and legged robots — JointStateSensor.

:func:`extract_link_contact_force_n`
    Scalar contact-force magnitude on one link — ContactSensor.

:func:`extract_link_ft_state`
    Wrench (force + torque) on a link — ForceTorqueSensorModel.

:func:`extract_link_imu_state`
    IMU-compatible kinematics from any rigid-body link — IMUModel.

:func:`quat_wxyz_to_rotation_matrix`
    Utility: ``[w, x, y, z]`` quaternion → 3×3 rotation matrix.

Examples
--------
**Drone navigation suite** ::

    from genesis.sensors.genesis_bridge import extract_rigid_body_state
    from genesis.sensors import SensorSuite

    suite = SensorSuite.default(imu_rate_hz=200.0, gnss_rate_hz=5.0)
    suite.reset()
    prev_vel = np.zeros(3)

    for step in range(n_steps):
        scene.step()
        state = extract_rigid_body_state(drone, prev_vel=prev_vel, dt=dt)
        obs   = suite.step(sim_time, state)
        prev_vel = state["vel"].copy()

**Franka arm proprioception** ::

    from genesis.sensors.genesis_bridge import (
        extract_joint_state, extract_link_contact_force_n, extract_link_ft_state)

    scene.step()
    state = {}
    state.update(extract_joint_state(franka, dofs_idx_local=list(range(7))))
    state.update(extract_link_ft_state(franka, "hand"))
    state["contact_force_n"] = extract_link_contact_force_n(franka, "hand")
    obs = arm_suite.step(sim_time, state)
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeAlias

_G_WORLD_Z = 9.80665  # m/s²

if TYPE_CHECKING:
    import numpy as np
    from typing import Protocol

    from .types import SensorState

    class LinkLike(Protocol):
        idx_local: int

    LinkRef: TypeAlias = str | LinkLike
else:
    LinkRef: TypeAlias = Any


def _tensor_to_array(value: Any, *, dtype: "type[np.generic] | None" = None) -> "np.ndarray":
    """Lazily convert a Genesis tensor-like object to a NumPy array."""
    from genesis.utils.misc import tensor_to_array

    return tensor_to_array(value, dtype=dtype)


def _resolve_link(entity: Any, link: LinkRef | None) -> Any | None:
    """Resolve a link name to a Genesis link object, keeping objects unchanged."""
    if link is None or not isinstance(link, str):
        return link
    return entity.get_link(link)


def _extract_pose_velocity(entity: Any, link: LinkRef | None = None) -> tuple["np.ndarray", "np.ndarray"]:
    """Read world-frame position and linear velocity from an entity or one of its links."""
    import numpy as np

    resolved_link = _resolve_link(entity, link)
    if resolved_link is None:
        pos = _tensor_to_array(entity.get_pos(), dtype=np.float64).reshape(-1)
        vel = _tensor_to_array(entity.get_vel(), dtype=np.float64).reshape(-1)
        return pos, vel

    idx_local = int(resolved_link.idx_local)
    links_pos = _tensor_to_array(entity.get_links_pos(), dtype=np.float64).reshape(-1, 3)
    links_vel = _tensor_to_array(entity.get_links_vel(), dtype=np.float64).reshape(-1, 3)
    if 0 <= idx_local < min(len(links_pos), len(links_vel)):
        return links_pos[idx_local].copy(), links_vel[idx_local].copy()
    return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)


def _extract_link_force_vector(entity: Any, link: LinkRef) -> "np.ndarray":
    """Return the net contact force on a link as a float32 3-vector."""
    import numpy as np

    resolved_link = _resolve_link(entity, link)
    if resolved_link is None:
        return np.zeros(3, dtype=np.float32)

    idx_local = int(resolved_link.idx_local)
    forces = _tensor_to_array(entity.get_links_net_contact_force(), dtype=np.float32).reshape(-1, 3)
    if 0 <= idx_local < len(forces):
        return forces[idx_local].astype(np.float32, copy=True)
    return np.zeros(3, dtype=np.float32)


def _build_motion_state(
    *,
    pos: "np.ndarray",
    vel: "np.ndarray",
    ang_vel_world: "np.ndarray",
    quat: "np.ndarray",
    prev_vel_world: "np.ndarray | None",
    dt: float,
) -> "SensorState":
    """Construct the shared IMU/navigation subset of the sensor-layer state."""
    import numpy as np

    R_bw = quat_wxyz_to_rotation_matrix(quat)
    R_wb = R_bw.T

    if prev_vel_world is not None and dt > 0.0:
        previous = np.asarray(prev_vel_world, dtype=np.float64).reshape(3)
        a_world = (vel - previous) / dt
    else:
        a_world = np.zeros(3, dtype=np.float64)

    gravity_world = np.array([0.0, 0.0, _G_WORLD_Z], dtype=np.float64)
    state: SensorState = {
        "pos": pos.astype(np.float64, copy=False),
        "vel": vel.astype(np.float64, copy=False),
        "lin_acc": (R_wb @ a_world).astype(np.float64, copy=False),
        "ang_vel": (R_wb @ ang_vel_world).astype(np.float32),
        "gravity_body": (R_wb @ gravity_world).astype(np.float64, copy=False),
        "attitude_mat": R_bw.astype(np.float32),
        "range_m": max(0.05, float(pos[2])),
    }
    return state


def quat_wxyz_to_rotation_matrix(q: "np.ndarray") -> "np.ndarray":
    """Convert a ``[w, x, y, z]`` quaternion to a 3×3 body-to-world rotation matrix."""
    import numpy as np

    q = np.asarray(q, dtype=np.float64).reshape(-1)
    if q.size != 4:
        raise ValueError(f"Expected quaternion with 4 elements in [w, x, y, z] order, got shape {q.shape}")

    norm = float(np.linalg.norm(q))
    if norm == 0.0:
        return np.eye(3, dtype=np.float64)
    w, x, y, z = q / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def extract_rigid_body_state(
    entity: Any,
    *,
    prev_vel: "np.ndarray | None" = None,
    dt: float = 0.01,
    current_a: float = 0.0,
    wind_ms: "np.ndarray | None" = None,
    rgb: "np.ndarray | None" = None,
    depth: "np.ndarray | None" = None,
) -> "SensorState":
    """Build a navigation-sensor state dict from a Genesis rigid-body entity."""
    import numpy as np

    pos, vel = _extract_pose_velocity(entity)
    ang_vel_world = _tensor_to_array(entity.get_ang(), dtype=np.float64).reshape(-1)
    quat = _tensor_to_array(entity.get_quat(), dtype=np.float64).reshape(-1)
    state = _build_motion_state(
        pos=pos,
        vel=vel,
        ang_vel_world=ang_vel_world,
        quat=quat,
        prev_vel_world=prev_vel,
        dt=dt,
    )
    state["wind"] = np.asarray(wind_ms if wind_ms is not None else np.zeros(3), dtype=np.float64)
    state["current_a"] = float(current_a)
    if rgb is not None:
        state["rgb"] = rgb
    if depth is not None:
        state["depth"] = depth
    return state


def extract_joint_state(entity: Any, dofs_idx_local: list[int] | None = None) -> "SensorState":
    """Extract joint position, velocity, and actuation torque from a Genesis entity."""
    import numpy as np

    return {
        "joint_pos": _tensor_to_array(entity.get_dofs_position(dofs_idx_local=dofs_idx_local), dtype=np.float32)
        .reshape(-1)
        .astype(np.float32, copy=False),
        "joint_vel": _tensor_to_array(entity.get_dofs_velocity(dofs_idx_local=dofs_idx_local), dtype=np.float32)
        .reshape(-1)
        .astype(np.float32, copy=False),
        "joint_torque": _tensor_to_array(entity.get_dofs_force(dofs_idx_local=dofs_idx_local), dtype=np.float32)
        .reshape(-1)
        .astype(np.float32, copy=False),
    }


def extract_link_contact_force_n(entity: Any, link: LinkRef) -> float:
    """Return the scalar contact-force magnitude (N) on a specific link."""
    import numpy as np

    return float(np.linalg.norm(_extract_link_force_vector(entity, link)))


def extract_link_ft_state(entity: Any, link: LinkRef) -> "SensorState":
    """Extract the net contact wrench on a link using the shared sensor-state layout."""
    import numpy as np

    return {
        "force": _extract_link_force_vector(entity, link),
        "torque": np.zeros(3, dtype=np.float32),
    }


def extract_link_imu_state(
    entity: Any,
    link: LinkRef | None = None,
    *,
    prev_vel_world: "np.ndarray | None" = None,
    dt: float = 0.01,
) -> "SensorState":
    """Extract IMU-compatible kinematics from a Genesis rigid-body entity or link."""
    import numpy as np

    pos, vel = _extract_pose_velocity(entity, link)
    ang_vel_world = _tensor_to_array(entity.get_ang(), dtype=np.float64).reshape(-1)
    quat = _tensor_to_array(entity.get_quat(), dtype=np.float64).reshape(-1)
    return _build_motion_state(
        pos=pos,
        vel=vel,
        ang_vel_world=ang_vel_world,
        quat=quat,
        prev_vel_world=prev_vel_world,
        dt=dt,
    )


__all__ = [
    "extract_joint_state",
    "extract_link_contact_force_n",
    "extract_link_ft_state",
    "extract_link_imu_state",
    "extract_rigid_body_state",
    "quat_wxyz_to_rotation_matrix",
]
