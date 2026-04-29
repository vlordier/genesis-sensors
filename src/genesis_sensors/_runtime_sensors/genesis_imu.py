"""
Genesis-native IMU sensor using Genesis's built-in IMU simulation.

This module wraps the native Genesis IMU sensor (added via scene.add_sensor)
to provide Isaac Sim-quality inertial measurement. The native Genesis IMU
provides physically-based simulation with configurable noise, bias drift,
and sensor characteristics.

Usage
-----
::

    import genesis as gs
    from genesis_sensors import GenesisIMU

    gs.init(backend=gs.gpu)
    scene = gs.Scene(...)

    # Create entity with a link
    robot = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    end_effector = robot.get_link("hand")

    # Create Genesis-native IMU
    imu = scene.add_sensor(gs.sensors.IMUOptions(
        entity_idx=robot.idx,
        link_idx_local=end_effector.idx_local,
        acc_noise_std=(0.01, 0.01, 0.01),
        gyro_noise_std=(0.01, 0.01, 0.01),
        acc_bias_drift_std=(0.001, 0.001, 0.001),
        gyro_bias_drift_std=(0.001, 0.001, 0.001),
        delay=0.01,
    ))

    # Wrap with GenesisIMU for unified API
    genesis_imu = GenesisIMU(imu_sensor=imu, name="imu")

    for event in node:
        scene.step()
        obs = genesis_imu.step(sim_time, {})
        # obs["lin_acc"]  # accelerometer reading
        # obs["ang_vel"]  # gyroscope reading
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from .base import BaseSensor
from .types import ImuObservation

if TYPE_CHECKING:
    from .config import IMUConfig


class GenesisIMU(BaseSensor):
    """
    Wrapper for native Genesis IMU sensor.

    Wraps a Genesis IMU sensor (from ``scene.add_sensor(gs.sensors.IMUOptions(...))``)
    to provide a unified sensor API compatible with genesis-sensors.

    The underlying Genesis IMU provides:
    - Physically-based accelerometer and gyroscope simulation
    - Configurable noise, bias drift, axis skew
    - Realistic delay and interpolation
    - Ground truth + measured outputs

    Parameters
    ----------
    imu_sensor:
        The native Genesis IMU sensor object returned by scene.add_sensor.
    name:
        Human-readable identifier.
    add_gravity:
        When True, adds gravity vector to accelerometer reading (specific force).
    """

    def __init__(
        self,
        imu_sensor: Any,
        name: str = "genesis_imu",
        add_gravity: bool = True,
    ) -> None:
        super().__init__(name=name, update_rate_hz=100.0)
        self._imu_sensor = imu_sensor
        self._add_gravity = bool(add_gravity)
        self._last_obs: dict[str, Any] = {}
        self._last_update_time = -1.0

    @classmethod
    def from_config(cls, config: "IMUConfig") -> "GenesisIMU":
        """Not supported for GenesisIMU - requires live Genesis sensor."""
        raise NotImplementedError(
            "GenesisIMU cannot be created from config - requires a live "
            "Genesis IMU sensor from scene.add_sensor(). Use direct construction."
        )

    def get_config(self) -> "IMUConfig":
        """Not supported for GenesisIMU."""
        raise NotImplementedError(
            "GenesisIMU does not support get_config() - the underlying "
            "Genesis sensor is configured via gs.sensors.IMUOptions at creation."
        )

    def reset(self, env_id: int = 0) -> None:
        """Reset the IMU sensor."""
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> ImuObservation | dict[str, Any]:
        """
        Read the Genesis IMU sensor.

        Parameters
        ----------
        sim_time:
            Current simulation time (s).
        state:
            Optional state dict. If contains "gravity_body", uses that for
            gravity compensation instead of default.

        Returns
        -------
        ImuObservation
            Dictionary with "lin_acc" and "ang_vel" readings.
        """
        try:
            measured = self._imu_sensor.read()
        except Exception:
            measured = None

        try:
            ground_truth = self._imu_sensor.read_ground_truth()
        except Exception:
            ground_truth = None

        if measured is None:
            self._last_obs = {
                "lin_acc": np.zeros(3, dtype=np.float64),
                "ang_vel": np.zeros(3, dtype=np.float64),
            }
            self._mark_updated(sim_time)
            return self._last_obs

        lin_acc = np.array(measured[:3], dtype=np.float64)
        ang_vel = np.array(measured[3:6], dtype=np.float64)

        if self._add_gravity:
            gravity_body = state.get("gravity_body")
            if gravity_body is not None:
                g_vec = np.asarray(gravity_body, dtype=np.float64)
            else:
                g_vec = np.array([0.0, 0.0, 9.80665], dtype=np.float64)
            lin_acc = lin_acc + g_vec

        obs: ImuObservation = {
            "lin_acc": lin_acc,
            "ang_vel": ang_vel,
        }

        if ground_truth is not None:
            obs["lin_acc_gt"] = np.array(ground_truth[:3], dtype=np.float64)
            obs["ang_vel_gt"] = np.array(ground_truth[3:6], dtype=np.float64)

        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        """Return the most recent observation."""
        return self._last_obs

    def read_ground_truth(self) -> dict[str, np.ndarray] | None:
        """Read ground truth from the underlying Genesis IMU."""
        try:
            gt = self._imu_sensor.read_ground_truth()
            return {
                "lin_acc": np.array(gt[:3], dtype=np.float64),
                "ang_vel": np.array(gt[3:6], dtype=np.float64),
            }
        except Exception:
            return None

    def read_measured(self) -> dict[str, np.ndarray] | None:
        """Read measured (noisy) values from the underlying Genesis IMU."""
        try:
            measured = self._imu_sensor.read()
            return {
                "lin_acc": np.array(measured[:3], dtype=np.float64),
                "ang_vel": np.array(measured[3:6], dtype=np.float64),
            }
        except Exception:
            return None


__all__ = ["GenesisIMU"]
