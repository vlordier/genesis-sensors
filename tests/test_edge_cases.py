"""Edge-case, boundary-condition, and error-handling tests.

Covers scenarios that shouldn't happen often but must not crash:
  - NaN/Inf in state inputs
  - get_observation() before first step()
  - SensorScheduler error wrapping
  - set_seed determinism
  - Extreme update rates
  - BaseSensor validation
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors._runtime_sensors import (
    BarometerModel,
    CameraModel,
    GNSSModel,
    IMUModel,
    LidarModel,
    MagnetometerModel,
    RangefinderModel,
    SensorScheduler,
    SensorSuite,
)
from genesis_sensors import make_synthetic_sensor_state


# ────────────────────────────────────────────────────────────────────
# BaseSensor validation
# ────────────────────────────────────────────────────────────────────


class TestBaseSensorValidation:
    """Verify that invalid construction parameters are rejected."""

    def test_negative_update_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            IMUModel(update_rate_hz=-1.0)

    def test_zero_update_rate_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            BarometerModel(update_rate_hz=0.0)

    def test_non_string_name_raises(self) -> None:
        with pytest.raises(TypeError, match="str"):
            IMUModel(name=42)  # type: ignore[arg-type]


# ────────────────────────────────────────────────────────────────────
# get_observation() before step()
# ────────────────────────────────────────────────────────────────────


class TestGetObservationBeforeStep:
    """Calling get_observation() before any step() should not crash."""

    @pytest.mark.parametrize(
        "sensor_cls,kwargs",
        [
            (IMUModel, {"seed": 0}),
            (BarometerModel, {"seed": 0}),
            (GNSSModel, {"seed": 0}),
            (MagnetometerModel, {"seed": 0}),
            (CameraModel, {"seed": 0}),
            (RangefinderModel, {"seed": 0}),
            (LidarModel, {"seed": 0}),
        ],
        ids=["imu", "baro", "gnss", "mag", "cam", "range", "lidar"],
    )
    def test_returns_dict_without_crash(self, sensor_cls, kwargs) -> None:
        sensor = sensor_cls(**kwargs)
        obs = sensor.get_observation()
        assert isinstance(obs, dict)


# ────────────────────────────────────────────────────────────────────
# NaN / Inf inputs
# ────────────────────────────────────────────────────────────────────


class TestNaNInfInputs:
    """Sensors should not crash on NaN/Inf state values — output may be NaN."""

    def test_imu_nan_input_produces_finite_or_nan(self) -> None:
        imu = IMUModel(seed=0)
        imu.reset()
        state = {
            "lin_acc": np.array([np.nan, 0.0, 0.0]),
            "ang_vel": np.array([0.0, 0.0, 0.0]),
        }
        obs = imu.step(0.0, state)
        # Should not raise; output may contain NaN but must be a valid dict
        assert "lin_acc" in obs
        assert "ang_vel" in obs

    def test_barometer_inf_altitude(self) -> None:
        baro = BarometerModel(seed=0)
        baro.reset()
        state = {"pos": np.array([0.0, 0.0, np.inf])}
        obs = baro.step(0.0, state)
        assert "altitude_m" in obs

    def test_gnss_nan_position(self) -> None:
        gnss = GNSSModel(seed=0)
        gnss.reset()
        state = {"pos": np.array([np.nan, 0.0, 0.0])}
        obs = gnss.step(0.0, state)
        assert "pos_llh" in obs


# ────────────────────────────────────────────────────────────────────
# SensorScheduler error wrapping
# ────────────────────────────────────────────────────────────────────


class _FailingSensor(IMUModel):
    """IMU subclass that always raises in step()."""

    def step(self, sim_time, state):
        raise RuntimeError("sensor hardware fault")


class TestSchedulerErrorWrapping:
    """Scheduler should wrap sensor exceptions with RuntimeError."""

    def test_scheduler_wraps_exception_with_sensor_name(self) -> None:
        scheduler = SensorScheduler()
        scheduler.add(_FailingSensor(seed=0), name="bad_imu")
        scheduler.reset()

        with pytest.raises(RuntimeError, match="bad_imu"):
            scheduler.update(0.0, make_synthetic_sensor_state(0))

    def test_original_exception_is_chained(self) -> None:
        scheduler = SensorScheduler()
        scheduler.add(_FailingSensor(seed=0), name="bad_imu")
        scheduler.reset()

        with pytest.raises(RuntimeError) as exc_info:
            scheduler.update(0.0, make_synthetic_sensor_state(0))
        assert exc_info.value.__cause__ is not None
        assert "hardware fault" in str(exc_info.value.__cause__)


# ────────────────────────────────────────────────────────────────────
# SensorSuite.set_seed determinism
# ────────────────────────────────────────────────────────────────────


class TestSetSeedDeterminism:
    """set_seed should produce identical outputs on replay."""

    def test_set_seed_produces_identical_observations(self) -> None:
        suite = SensorSuite.default(
            imu_rate_hz=200.0,
            baro_rate_hz=50.0,
            gnss_rate_hz=0.0,
            rgb_rate_hz=0.0,
            event_rate_hz=0.0,
            thermal_rate_hz=0.0,
            lidar_rate_hz=0.0,
            radio_rate_hz=0.0,
            mag_rate_hz=0.0,
            seed=0,
        )

        state = make_synthetic_sensor_state(0)

        # Run 1
        suite.set_seed(42)
        suite.reset()
        obs1 = suite.step(0.0, state)

        # Run 2 with same seed
        suite.set_seed(42)
        suite.reset()
        obs2 = suite.step(0.0, state)

        np.testing.assert_array_equal(obs1["imu"]["lin_acc"], obs2["imu"]["lin_acc"])
        np.testing.assert_array_equal(
            obs1["barometer"]["altitude_m"], obs2["barometer"]["altitude_m"]
        )


# ────────────────────────────────────────────────────────────────────
# Extreme update rates
# ────────────────────────────────────────────────────────────────────


class TestExtremeRates:
    """Sensors should handle very high or very low rates gracefully."""

    def test_very_high_rate_does_not_crash(self) -> None:
        imu = IMUModel(update_rate_hz=10_000.0, seed=0)
        imu.reset()
        state = {
            "lin_acc": np.zeros(3),
            "ang_vel": np.zeros(3),
        }
        obs = imu.step(0.0, state)
        assert np.all(np.isfinite(obs["lin_acc"]))

    def test_very_low_rate_does_not_crash(self) -> None:
        baro = BarometerModel(update_rate_hz=0.01, seed=0)
        baro.reset()
        obs = baro.step(0.0, {"pos": np.zeros(3)})
        assert np.isfinite(obs["altitude_m"])


# ────────────────────────────────────────────────────────────────────
# Scheduler duplicate name detection
# ────────────────────────────────────────────────────────────────────


class TestSchedulerDuplicateNames:
    """Adding two sensors with the same name should raise."""

    def test_duplicate_name_raises(self) -> None:
        scheduler = SensorScheduler()
        scheduler.add(IMUModel(seed=0), name="my_imu")
        with pytest.raises((ValueError, KeyError)):
            scheduler.add(IMUModel(seed=1), name="my_imu")
