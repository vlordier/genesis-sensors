from __future__ import annotations

import numpy as np

from genesis_sensors._runtime_sensors.config import ProximityToFArrayConfig
from genesis_sensors._runtime_sensors.genesis_bridge import _build_motion_state
from genesis_sensors._runtime_sensors.motor_temperature import MotorTemperatureModel
from genesis_sensors._runtime_sensors.proximity_tof import ProximityToFArrayModel
from genesis_sensors._runtime_sensors.rangefinder import RangefinderModel


def test_rangefinder_clamps_dropout_probability() -> None:
    sensor = RangefinderModel(dropout_prob=2.0, noise_floor_m=0.0, noise_slope=0.0, resolution_m=0.0, seed=0)
    assert sensor.dropout_prob == 1.0

    obs = sensor.step(0.0, {"range_m": 1.0})
    assert obs["in_range"] is False

    sensor.dropout_prob = -1.0
    obs = sensor.step(0.1, {"range_m": 1.0})
    assert obs["in_range"] is True


def test_proximity_tof_config_roundtrip_preserves_min_range_and_fov() -> None:
    cfg = ProximityToFArrayConfig(min_range_m=0.15, max_range_m=5.5, fov_deg=95.0)
    sensor = ProximityToFArrayModel.from_config(cfg)
    recovered = sensor.get_config()
    assert recovered.model_dump() == cfg.model_dump()


def test_proximity_tof_marks_no_hit_zones_invalid() -> None:
    sensor = ProximityToFArrayModel(
        rows=1,
        cols=2,
        min_range_m=0.1,
        max_range_m=4.0,
        noise_sigma_base_m=0.0,
        noise_sigma_scale=0.0,
        crosstalk_fraction=0.0,
        seed=0,
    )
    obs = sensor.step(sim_time=0.0, state={"tof_ranges_m": [[0.5, 4.0]]})
    np.testing.assert_array_equal(np.asarray(obs["valid_mask"]), np.array([[True, False]]))
    assert obs["n_valid_zones"] == 1
    assert float(obs["min_range_m"]) == 0.5

    missing_obs = sensor.step(sim_time=0.1, state={})
    np.testing.assert_array_equal(np.asarray(missing_obs["valid_mask"]), np.array([[False, False]]))
    assert missing_obs["n_valid_zones"] == 0
    assert float(missing_obs["min_range_m"]) == sensor.max_range_m


def test_motor_temperature_step_uses_mark_updated_hook() -> None:
    sensor = MotorTemperatureModel(seed=0)
    marker: dict[str, float] = {}

    def _capture_mark_updated(sim_time: float) -> None:
        marker["sim_time"] = sim_time
        sensor._last_update_time = sim_time

    sensor._mark_updated = _capture_mark_updated  # type: ignore[method-assign]
    sensor.step(sim_time=1.25, state={})
    assert marker["sim_time"] == 1.25


def test_bridge_uses_negative_world_z_gravity() -> None:
    state = _build_motion_state(
        pos=np.zeros(3, dtype=np.float64),
        vel=np.zeros(3, dtype=np.float64),
        ang_vel_world=np.zeros(3, dtype=np.float64),
        quat=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        prev_vel_world=None,
        dt=0.01,
    )
    np.testing.assert_allclose(np.asarray(state["gravity_body"], dtype=np.float64), np.array([0.0, 0.0, -9.80665]))
