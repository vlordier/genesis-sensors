"""Edge-case, boundary-condition, and error-handling tests.

Covers scenarios that shouldn't happen often but must not crash:
  - NaN / Inf in state inputs
  - get_observation() before first step()
  - SensorScheduler error wrapping
  - set_seed determinism
  - Extreme update rates
  - BaseSensor validation
  - Config boundary values (min == max, zero noise, max noise)
  - Saturated / clipped output
  - Reset idempotency
  - Empty / missing state keys
"""

from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors._runtime_sensors import (
    AirspeedModel,
    BarometerModel,
    BatteryModel,
    CameraModel,
    ContactSensor,
    CurrentSensor,
    DepthCameraModel,
    ForceTorqueSensorModel,
    GNSSModel,
    IMUModel,
    JointStateSensor,
    LidarModel,
    MagnetometerModel,
    OpticalFlowModel,
    RangefinderModel,
    RPMSensor,
    SensorScheduler,
    SensorSuite,
    TactileArraySensor,
    WheelOdometryModel,
)
from genesis_sensors._runtime_sensors.config import (
    AirspeedConfig,
    CameraConfig,
    LidarConfig,
    MagnetometerConfig,
    RadarConfig,
    RangefinderConfig,
    ThermalCameraConfig,
    UltrasonicArrayConfig,
)
from genesis_sensors._runtime_sensors.presets import get_preset, list_presets
from genesis_sensors import make_synthetic_sensor_state


# ════════════════════════════════════════════════════════════════════
# BaseSensor validation
# ════════════════════════════════════════════════════════════════════


class TestBaseSensorValidation:
    """Verify that invalid construction parameters are rejected."""

    @pytest.mark.parametrize(
        "sensor_cls",
        [IMUModel, BarometerModel, GNSSModel, MagnetometerModel, CameraModel, LidarModel, RangefinderModel],
        ids=lambda c: c.__name__,
    )
    def test_negative_update_rate_raises(self, sensor_cls) -> None:
        with pytest.raises(ValueError, match="positive"):
            sensor_cls(update_rate_hz=-1.0)

    @pytest.mark.parametrize(
        "sensor_cls",
        [IMUModel, BarometerModel, GNSSModel, MagnetometerModel, CameraModel, LidarModel, RangefinderModel],
        ids=lambda c: c.__name__,
    )
    def test_zero_update_rate_raises(self, sensor_cls) -> None:
        with pytest.raises(ValueError, match="positive"):
            sensor_cls(update_rate_hz=0.0)

    def test_non_string_name_raises(self) -> None:
        with pytest.raises(TypeError, match="str"):
            IMUModel(name=42)  # type: ignore[arg-type]


# ════════════════════════════════════════════════════════════════════
# get_observation() before step()
# ════════════════════════════════════════════════════════════════════


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
            (AirspeedModel, {"seed": 0}),
            (OpticalFlowModel, {"seed": 0}),
            (WheelOdometryModel, {"seed": 0}),
            (BatteryModel, {"seed": 0}),
            (ForceTorqueSensorModel, {"seed": 0}),
            (JointStateSensor, {"seed": 0}),
            (ContactSensor, {"seed": 0}),
            (TactileArraySensor, {"seed": 0}),
            (CurrentSensor, {"seed": 0}),
            (RPMSensor, {"seed": 0}),
            (DepthCameraModel, {"seed": 0}),
        ],
        ids=lambda x: x.__name__ if isinstance(x, type) else "",
    )
    def test_returns_dict_without_crash(self, sensor_cls, kwargs) -> None:
        sensor = sensor_cls(**kwargs)
        obs = sensor.get_observation()
        assert isinstance(obs, dict)


# ════════════════════════════════════════════════════════════════════
# NaN / Inf inputs
# ════════════════════════════════════════════════════════════════════


class TestNaNInfInputs:
    """Sensors should not crash on NaN/Inf state values — output may be NaN."""

    @pytest.mark.parametrize(
        "corrupted_key,corrupted_value",
        [
            ("lin_acc", np.array([np.nan, 0.0, 0.0])),
            ("lin_acc", np.array([np.inf, 0.0, 0.0])),
            ("ang_vel", np.array([0.0, np.nan, 0.0])),
            ("ang_vel", np.array([0.0, 0.0, -np.inf])),
        ],
        ids=["nan_acc", "inf_acc", "nan_gyr", "neginf_gyr"],
    )
    def test_imu_handles_corrupt_input(self, corrupted_key, corrupted_value) -> None:
        imu = IMUModel(seed=0)
        imu.reset()
        state = {
            "lin_acc": np.zeros(3),
            "ang_vel": np.zeros(3),
            corrupted_key: corrupted_value,
        }
        obs = imu.step(0.0, state)
        assert "lin_acc" in obs and "ang_vel" in obs

    def test_barometer_inf_altitude(self) -> None:
        baro = BarometerModel(seed=0)
        baro.reset()
        obs = baro.step(0.0, {"pos": np.array([0.0, 0.0, np.inf])})
        assert "altitude_m" in obs

    def test_gnss_nan_position(self) -> None:
        gnss = GNSSModel(seed=0)
        gnss.reset()
        obs = gnss.step(0.0, {"pos": np.array([np.nan, 0.0, 0.0])})
        assert "pos_llh" in obs

    def test_magnetometer_nan_attitude(self) -> None:
        mag = MagnetometerModel(seed=0)
        mag.reset()
        obs = mag.step(0.0, {"attitude_mat": np.full((3, 3), np.nan)})
        assert "mag_field_ut" in obs

    def test_rangefinder_nan_range(self) -> None:
        rf = RangefinderModel(seed=0)
        rf.reset()
        obs = rf.step(0.0, {"range_m": np.nan})
        assert "range_m" in obs

    def test_airspeed_inf(self) -> None:
        asp = AirspeedModel(seed=0)
        asp.reset()
        obs = asp.step(0.0, {"airspeed_ms": np.inf, "pos": np.zeros(3)})
        assert "airspeed_ms" in obs


# ════════════════════════════════════════════════════════════════════
# SensorScheduler error wrapping
# ════════════════════════════════════════════════════════════════════


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


# ════════════════════════════════════════════════════════════════════
# SensorSuite.set_seed determinism
# ════════════════════════════════════════════════════════════════════


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
        np.testing.assert_array_equal(obs1["barometer"]["altitude_m"], obs2["barometer"]["altitude_m"])

    def test_different_seeds_produce_different_observations(self) -> None:
        """Two different seeds should diverge."""
        suite = SensorSuite.default(
            imu_rate_hz=200.0,
            baro_rate_hz=0.0,
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

        suite.set_seed(42)
        suite.reset()
        obs1 = suite.step(0.0, state)

        suite.set_seed(99)
        suite.reset()
        obs2 = suite.step(0.0, state)

        # Very unlikely to be exactly equal with different seeds
        assert not np.array_equal(obs1["imu"]["lin_acc"], obs2["imu"]["lin_acc"])


# ════════════════════════════════════════════════════════════════════
# Extreme update rates
# ════════════════════════════════════════════════════════════════════


class TestExtremeRates:
    """Sensors should handle very high or very low rates gracefully."""

    @pytest.mark.parametrize("rate", [10_000.0, 50_000.0])
    def test_very_high_rate_does_not_crash(self, rate) -> None:
        imu = IMUModel(update_rate_hz=rate, seed=0)
        imu.reset()
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3)})
        assert np.all(np.isfinite(obs["lin_acc"]))

    @pytest.mark.parametrize("rate", [0.01, 0.001])
    def test_very_low_rate_does_not_crash(self, rate) -> None:
        baro = BarometerModel(update_rate_hz=rate, seed=0)
        baro.reset()
        obs = baro.step(0.0, {"pos": np.zeros(3)})
        assert np.isfinite(obs["altitude_m"])


# ════════════════════════════════════════════════════════════════════
# Scheduler duplicate name detection
# ════════════════════════════════════════════════════════════════════


class TestSchedulerDuplicateNames:
    """Adding two sensors with the same name should raise."""

    def test_duplicate_name_raises(self) -> None:
        scheduler = SensorScheduler()
        scheduler.add(IMUModel(seed=0), name="my_imu")
        with pytest.raises((ValueError, KeyError)):
            scheduler.add(IMUModel(seed=1), name="my_imu")


# ════════════════════════════════════════════════════════════════════
# Config boundary values
# ════════════════════════════════════════════════════════════════════


class TestConfigBoundaryValues:
    """Configs should reject invalid ranges and accept valid edge cases."""

    def test_camera_zero_resolution_rejected(self) -> None:
        with pytest.raises(ValueError):
            CameraConfig(resolution=(0, 480))

    def test_camera_negative_resolution_rejected(self) -> None:
        with pytest.raises(ValueError):
            CameraConfig(resolution=(640, -1))

    def test_lidar_v_fov_inverted_rejected(self) -> None:
        with pytest.raises(ValueError):
            LidarConfig(v_fov_deg=(15.0, -15.0))

    def test_rangefinder_min_ge_max_rejected(self) -> None:
        with pytest.raises(ValueError):
            RangefinderConfig(min_range_m=12.0, max_range_m=5.0)

    def test_radar_min_ge_max_rejected(self) -> None:
        with pytest.raises(ValueError):
            RadarConfig(min_range_m=200.0, max_range_m=100.0)

    def test_airspeed_min_ge_max_rejected(self) -> None:
        with pytest.raises(ValueError):
            AirspeedConfig(min_detectable_ms=100.0, max_speed_ms=50.0)

    def test_thermal_temp_range_inverted_rejected(self) -> None:
        with pytest.raises(ValueError):
            ThermalCameraConfig(temp_range_c=(140.0, -20.0))

    def test_lidar_channel_offsets_wrong_length_rejected(self) -> None:
        with pytest.raises(ValueError):
            LidarConfig(n_channels=16, channel_offsets_m=[0.0] * 8)

    def test_ultrasonic_beam_angles_wrong_length_rejected(self) -> None:
        with pytest.raises(ValueError):
            UltrasonicArrayConfig(n_beams=4, beam_angles_deg=[0.0, 45.0])

    def test_magnetometer_soft_iron_must_be_positive(self) -> None:
        with pytest.raises(ValueError):
            MagnetometerConfig(soft_iron_scale=[1.0, 0.0, 1.0])

    def test_magnetometer_wrong_element_count(self) -> None:
        with pytest.raises(ValueError):
            MagnetometerConfig(hard_iron_ut=[1.0, 2.0])


# ════════════════════════════════════════════════════════════════════
# Zero-noise sensors produce near-ideal output
# ════════════════════════════════════════════════════════════════════


class TestZeroNoiseSensors:
    """When all noise parameters are zero, output should match input closely."""

    def test_imu_zero_noise(self) -> None:
        imu = IMUModel(
            noise_density_acc=1e-15,
            noise_density_gyr=1e-15,
            bias_sigma_acc=0.0,
            bias_sigma_gyr=0.0,
            scale_factor_acc=0.0,
            scale_factor_gyr=0.0,
            cross_axis_sensitivity_acc=0.0,
            cross_axis_sensitivity_gyr=0.0,
            add_gravity=False,
            seed=0,
        )
        imu.reset()
        true_acc = np.array([1.0, 2.0, 3.0])
        true_gyr = np.array([0.1, 0.2, 0.3])
        obs = imu.step(0.0, {"lin_acc": true_acc, "ang_vel": true_gyr})
        np.testing.assert_allclose(obs["lin_acc"], true_acc, atol=1e-4)
        np.testing.assert_allclose(obs["ang_vel"], true_gyr, atol=1e-4)

    def test_barometer_zero_noise(self) -> None:
        baro = BarometerModel(
            noise_sigma_m=0.0,
            bias_sigma_m=0.0,
            seed=0,
        )
        baro.reset()
        alt = 100.0
        obs = baro.step(0.0, {"pos": np.array([0.0, 0.0, alt])})
        assert abs(obs["altitude_m"] - alt) < 1.0  # ISA conversion introduces offset

    def test_rangefinder_zero_noise(self) -> None:
        rf = RangefinderModel(
            noise_floor_m=0.0,
            noise_slope=0.0,
            dropout_prob=0.0,
            resolution_m=0.0,
            seed=0,
        )
        rf.reset()
        true_range = 5.0
        obs = rf.step(0.0, {"range_m": true_range})
        assert abs(obs["range_m"] - true_range) < 1e-6


# ════════════════════════════════════════════════════════════════════
# Saturation / clipping
# ════════════════════════════════════════════════════════════════════


class TestSaturation:
    """Sensors should clip to their configured ranges."""

    def test_rangefinder_above_max_returns_no_hit(self) -> None:
        rf = RangefinderModel(max_range_m=10.0, seed=0)
        rf.reset()
        obs = rf.step(0.0, {"range_m": 999.0})
        assert obs["range_m"] == rf.no_hit_value or obs["in_range"] is False

    def test_rangefinder_below_min_returns_no_hit(self) -> None:
        rf = RangefinderModel(min_range_m=0.5, seed=0)
        rf.reset()
        obs = rf.step(0.0, {"range_m": 0.01})
        assert obs["in_range"] is False

    def test_contact_below_threshold(self) -> None:
        cs = ContactSensor(force_threshold_n=5.0, seed=0)
        cs.reset()
        obs = cs.step(0.0, {"contact_force_n": 0.1})
        assert obs["contact_detected"] is False

    def test_contact_above_threshold(self) -> None:
        cs = ContactSensor(force_threshold_n=5.0, noise_sigma_n=0.0, seed=0)
        cs.reset()
        obs = cs.step(0.0, {"contact_force_n": 10.0})
        assert obs["contact_detected"] is True

    def test_current_sensor_clips_to_range(self) -> None:
        cs = CurrentSensor(range_a=10.0, noise_sigma_a=0.0, offset_a=0.0, seed=0)
        cs.reset()
        obs = cs.step(0.0, {"current_a": 999.0})
        assert obs["current_a"] <= 10.0

    def test_rpm_sensor_clips_to_range(self) -> None:
        rpm = RPMSensor(rpm_range=5000.0, noise_sigma_rpm=0.0, cpr=0, seed=0)
        rpm.reset()
        obs = rpm.step(0.0, {"rpm": 99999.0})
        assert obs["rpm"] <= 5000.0


# ════════════════════════════════════════════════════════════════════
# Reset idempotency
# ════════════════════════════════════════════════════════════════════


class TestResetIdempotency:
    """Multiple resets should not corrupt state."""

    @pytest.mark.parametrize(
        "sensor_cls",
        [IMUModel, BarometerModel, GNSSModel, MagnetometerModel, RangefinderModel, BatteryModel],
        ids=lambda c: c.__name__,
    )
    def test_double_reset_does_not_crash(self, sensor_cls) -> None:
        sensor = sensor_cls(seed=0)
        sensor.reset()
        sensor.reset()
        obs = sensor.get_observation()
        assert isinstance(obs, dict)


# ════════════════════════════════════════════════════════════════════
# Empty / missing state keys
# ════════════════════════════════════════════════════════════════════


class TestMissingStateKeys:
    """Sensors should use defaults when optional state keys are absent."""

    def test_battery_no_current_uses_zero(self) -> None:
        bat = BatteryModel(seed=0)
        bat.reset()
        obs = bat.step(0.0, {})
        assert "voltage_v" in obs

    def test_current_sensor_no_current(self) -> None:
        cs = CurrentSensor(seed=0)
        cs.reset()
        obs = cs.step(0.0, {})
        assert "current_a" in obs

    def test_rpm_sensor_no_rpm(self) -> None:
        rpm = RPMSensor(seed=0)
        rpm.reset()
        obs = rpm.step(0.0, {})
        assert "rpm" in obs

    def test_wheel_odometry_no_vel(self) -> None:
        wo = WheelOdometryModel(seed=0)
        wo.reset()
        obs = wo.step(0.0, {})
        assert "delta_pos_m" in obs

    def test_contact_sensor_no_force(self) -> None:
        cs = ContactSensor(seed=0)
        cs.reset()
        obs = cs.step(0.0, {})
        assert "contact_detected" in obs

    def test_rangefinder_no_range(self) -> None:
        rf = RangefinderModel(seed=0)
        rf.reset()
        obs = rf.step(0.0, {})
        # Empty state → empty observation (no range input available)
        assert isinstance(obs, dict)


# ════════════════════════════════════════════════════════════════════
# Preset override API
# ════════════════════════════════════════════════════════════════════


class TestPresetOverrides:
    """get_preset(**overrides) and Model.from_preset() should work."""

    def test_get_preset_with_overrides(self) -> None:
        cfg = get_preset("PIXHAWK_ICM20689", update_rate_hz=500.0)
        assert cfg.update_rate_hz == 500.0
        assert cfg.name == "PIXHAWK_ICM20689"  # original name preserved

    def test_get_preset_no_override_is_identity(self) -> None:
        cfg = get_preset("BMP388")
        assert cfg.name == "BMP388"

    def test_get_preset_unknown_raises(self) -> None:
        with pytest.raises(KeyError, match="DOES_NOT_EXIST"):
            get_preset("DOES_NOT_EXIST")

    def test_from_preset_on_model(self) -> None:
        imu = IMUModel.from_preset("PIXHAWK_ICM20689")
        assert imu.name == "PIXHAWK_ICM20689"
        imu.reset()
        obs = imu.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3)})
        assert "lin_acc" in obs

    def test_from_preset_with_overrides_on_model(self) -> None:
        imu = IMUModel.from_preset("PIXHAWK_ICM20689", update_rate_hz=1000.0)
        assert imu.update_rate_hz == 1000.0

    def test_preset_does_not_mutate_original(self) -> None:
        original = get_preset("BMP388")
        original_rate = original.update_rate_hz
        _ = get_preset("BMP388", update_rate_hz=9999.0)
        assert get_preset("BMP388").update_rate_hz == original_rate

    @pytest.mark.parametrize("preset_name", list_presets()[:10])
    def test_presets_are_valid_configs(self, preset_name) -> None:
        cfg = get_preset(preset_name)
        # Should be a pydantic model with model_dump
        d = cfg.model_dump()
        assert isinstance(d, dict)
        assert "name" in d


# ════════════════════════════════════════════════════════════════════
# Persistent tube blockage
# ════════════════════════════════════════════════════════════════════


class TestPersistentTubeBlockage:
    """Once blocked, airspeed stays at 0 until reset."""

    def test_blockage_persists_across_steps(self) -> None:
        # Use high prob to guarantee blockage quickly
        asp = AirspeedModel(tube_blockage_prob=1.0, seed=0)
        asp.reset()
        state = {"airspeed_ms": 50.0, "pos": np.zeros(3)}
        obs1 = asp.step(0.0, state)
        assert obs1["airspeed_ms"] == 0.0
        # Second step — still blocked
        obs2 = asp.step(0.02, state)
        assert obs2["airspeed_ms"] == 0.0

    def test_reset_clears_blockage(self) -> None:
        asp = AirspeedModel(tube_blockage_prob=1.0, noise_sigma_ms=0.0, bias_sigma_ms=0.0, seed=0)
        asp.reset()
        obs1 = asp.step(0.0, {"airspeed_ms": 50.0, "pos": np.zeros(3)})
        assert obs1["airspeed_ms"] == 0.0
        # Reset clears blockage
        asp.reset()
        # Now use zero blockage prob to avoid re-trigger
        asp.tube_blockage_prob = 0.0
        obs2 = asp.step(0.0, {"airspeed_ms": 50.0, "pos": np.zeros(3)})
        assert obs2["airspeed_ms"] > 0.0
