from __future__ import annotations

import numpy as np
import pytest

import genesis_sensors
from genesis_sensors._runtime_sensors import IMUModel, RangefinderModel


_DEF_IMU_STATE = {
    "lin_acc": np.array([0.1, 0.2, 0.3], dtype=np.float64),
    "ang_vel": np.array([0.01, 0.02, 0.03], dtype=np.float64),
    "gravity_body": np.array([0.0, 0.0, 9.80665], dtype=np.float64),
}


def test_robust_wrapper_applies_latency_and_metadata() -> None:
    sensor = RangefinderModel(
        update_rate_hz=20.0,
        noise_floor_m=0.0,
        noise_slope=0.0,
        dropout_prob=0.0,
        resolution_m=0.0,
        seed=0,
    )
    wrapped = genesis_sensors.RobustSensorWrapper(sensor, latency_s=0.05, seed=0)

    obs0 = wrapped.step(0.0, {"range_m": 1.0})
    assert obs0["_meta"]["status"] == "warming_up"

    obs1 = wrapped.step(0.05, {"range_m": 2.0})
    assert float(obs1["range_m"]) == 1.0
    assert obs1["_meta"]["status"] == "delayed"
    assert obs1["_meta"]["source_time"] == 0.0


def test_robust_wrapper_can_hold_last_value_on_dropout() -> None:
    sensor = RangefinderModel(
        update_rate_hz=20.0,
        noise_floor_m=0.0,
        noise_slope=0.0,
        dropout_prob=0.0,
        resolution_m=0.0,
        seed=0,
    )
    wrapped = genesis_sensors.RobustSensorWrapper(sensor, dropout_prob=0.0, seed=0)

    first = wrapped.step(0.0, {"range_m": 1.25})
    wrapped.dropout_prob = 1.0
    second = wrapped.step(0.05, {"range_m": 2.5})

    assert float(first["range_m"]) == 1.25
    assert float(second["range_m"]) == 1.25
    assert second["_meta"]["status"] == "dropout_hold"
    assert second["_meta"]["dropped"] is True


def test_wrap_suite_with_faults_preserves_sensor_names_and_meta() -> None:
    suite = genesis_sensors.SensorSuite(
        imu=IMUModel(update_rate_hz=100.0, seed=0),
        rangefinder=RangefinderModel(
            update_rate_hz=20.0,
            noise_floor_m=0.0,
            noise_slope=0.0,
            dropout_prob=0.0,
            resolution_m=0.0,
            seed=1,
        ),
    )
    wrapped = genesis_sensors.wrap_suite_with_faults(
        suite,
        latency_s={"rangefinder": 0.05},
        dropout_prob={"imu": 0.0},
        seed=42,
    )

    assert set(wrapped.sensor_names()) == {"imu", "rangefinder"}
    obs = wrapped.step(0.0, {**_DEF_IMU_STATE, "range_m": 1.5})
    assert "_meta" in obs["imu"]
    assert obs["rangefinder"]["_meta"]["sensor_name"] == "rangefinder"


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [({"latency_s": -0.01}, "latency_s"), ({"dropout_prob": 1.1}, "dropout_prob")],
)
def test_fault_config_validation_rejects_invalid_values(kwargs: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        genesis_sensors.SensorFaultConfig(**kwargs)


def test_robust_wrapper_can_disable_metadata() -> None:
    sensor = RangefinderModel(
        update_rate_hz=20.0,
        noise_floor_m=0.0,
        noise_slope=0.0,
        dropout_prob=0.0,
        resolution_m=0.0,
        seed=0,
    )
    wrapped = genesis_sensors.RobustSensorWrapper(sensor, include_metadata=False, seed=0)

    obs = wrapped.step(0.0, {"range_m": 1.5})
    assert "_meta" not in obs
    assert float(obs["range_m"]) == 1.5


def test_robust_wrapper_reset_replays_dropout_sequence_with_same_seed() -> None:
    sensor = RangefinderModel(
        update_rate_hz=20.0,
        noise_floor_m=0.0,
        noise_slope=0.0,
        dropout_prob=0.0,
        resolution_m=0.0,
        seed=0,
    )
    wrapped = genesis_sensors.RobustSensorWrapper(sensor, dropout_prob=0.35, seed=123)
    times = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25]

    first_run = [wrapped.step(t, {"range_m": 2.0})["_meta"]["dropped"] for t in times]
    wrapped.reset()
    second_run = [wrapped.step(t, {"range_m": 2.0})["_meta"]["dropped"] for t in times]

    assert first_run == second_run
