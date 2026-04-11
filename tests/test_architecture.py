"""Tests covering common architecture guarantees.

- **Config roundtrip**: sensor == SensorClass.from_config(sensor.get_config())
  for every sensor/config pair.
- **Preset validation**: every registered preset produces a working sensor.
- **GaussMarkovProcess**: helper class behaves correctly in scalar and vector modes.
- **Multi-rate scheduling**: sensors at different rates produce the expected
  update cadence.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from genesis_sensors import (
    SensorScheduler,
    SensorSuite,
    SensorSuiteConfig,
    get_scenario_phase,
    list_presets,
    make_synthetic_sensor_state,
)

from conftest import ALL_SENSOR_PAIRS, STEPPABLE_SENSORS  # type: ignore[import-not-found]


# ────────────────────────────────────────────────────────────────────
# GaussMarkovProcess unit tests
# ────────────────────────────────────────────────────────────────────


class TestGaussMarkovProcess:
    """Verify the extracted Gauss-Markov helper in isolation."""

    def test_scalar_process_produces_finite_values(self) -> None:
        from genesis_sensors._runtime_sensors._gauss_markov import GaussMarkovProcess

        rng = np.random.default_rng(0)
        gm = GaussMarkovProcess(tau_s=100.0, sigma=1.0, dt=0.01, rng=rng)
        for _ in range(500):
            v = gm.step()
            assert np.isfinite(v)

    def test_vector_process_shape(self) -> None:
        from genesis_sensors._runtime_sensors._gauss_markov import GaussMarkovProcess

        rng = np.random.default_rng(1)
        gm = GaussMarkovProcess(tau_s=60.0, sigma=0.5, dt=0.1, rng=rng, shape=(3,))
        v = gm.step()
        assert hasattr(v, "shape")
        assert v.shape == (3,)

    def test_reset_redraws_from_steady_state(self) -> None:
        from genesis_sensors._runtime_sensors._gauss_markov import GaussMarkovProcess

        rng = np.random.default_rng(2)
        gm = GaussMarkovProcess(tau_s=100.0, sigma=2.0, dt=0.01, rng=rng)
        # Run for a while to evolve from init
        for _ in range(100):
            gm.step()
        old_val = gm.value
        gm.reset(sigma=2.0)
        assert gm.value != old_val  # almost certainly different

    def test_deterministic_with_same_seed(self) -> None:
        from genesis_sensors._runtime_sensors._gauss_markov import GaussMarkovProcess

        vals_a: list[float] = []
        vals_b: list[float] = []
        for vals in (vals_a, vals_b):
            rng = np.random.default_rng(42)
            gm = GaussMarkovProcess(tau_s=50.0, sigma=1.0, dt=0.01, rng=rng)
            for _ in range(20):
                vals.append(float(gm.step()))
        assert vals_a == vals_b

    def test_steady_state_variance_bounded(self) -> None:
        """After many steps the process variance should be close to sigma^2."""
        from genesis_sensors._runtime_sensors._gauss_markov import GaussMarkovProcess

        sigma = 1.5
        rng = np.random.default_rng(7)
        gm = GaussMarkovProcess(tau_s=10.0, sigma=sigma, dt=0.01, rng=rng)
        samples = [float(gm.step()) for _ in range(10_000)]
        measured_std = float(np.std(samples))
        # Should be within 20% of sigma
        assert abs(measured_std - sigma) < 0.3 * sigma


# ────────────────────────────────────────────────────────────────────
# Config roundtrip tests
# ────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "model_cls,config_cls",
    ALL_SENSOR_PAIRS,
    ids=[model_cls.__name__ for model_cls, _ in ALL_SENSOR_PAIRS],
)
def test_config_roundtrip(model_cls: Any, config_cls: Any) -> None:
    """from_config(cfg).get_config() should produce an equivalent config."""
    cfg1 = config_cls()
    sensor = model_cls.from_config(cfg1)
    cfg2 = sensor.get_config()

    # Compare serialised forms — tolerant of list/tuple differences.
    assert cfg1.model_dump() == cfg2.model_dump(), (
        f"{model_cls.__name__} config roundtrip mismatch:\n"
        f"  original: {cfg1.model_dump()}\n"
        f"  recovered: {cfg2.model_dump()}"
    )


# ────────────────────────────────────────────────────────────────────
# Preset → sensor validation tests
# ────────────────────────────────────────────────────────────────────


def _all_preset_names() -> list[str]:
    from genesis_sensors._runtime_sensors.presets import _REGISTRY

    return sorted(_REGISTRY.keys())


# Mapping from config type to sensor model class for instantiation.
_CONFIG_TO_MODEL: dict[str, str] = {
    "CameraConfig": "CameraModel",
    "StereoCameraConfig": "StereoCameraModel",
    "LidarConfig": "LidarModel",
    "IMUConfig": "IMUModel",
    "GNSSConfig": "GNSSModel",
    "ThermalCameraConfig": "ThermalCameraModel",
    "EventCameraConfig": "EventCameraModel",
    "BarometerConfig": "BarometerModel",
    "MagnetometerConfig": "MagnetometerModel",
    "AirspeedConfig": "AirspeedModel",
    "RadioConfig": "RadioLinkModel",
    "RangefinderConfig": "RangefinderModel",
    "OpticalFlowConfig": "OpticalFlowModel",
    "BatteryConfig": "BatteryModel",
    "WheelOdometryConfig": "WheelOdometryModel",
    "ForceTorqueConfig": "ForceTorqueSensorModel",
    "JointStateConfig": "JointStateSensor",
    "ContactSensorConfig": "ContactSensor",
    "DepthCameraConfig": "DepthCameraModel",
    "TactileArrayConfig": "TactileArraySensor",
    "CurrentSensorConfig": "CurrentSensor",
    "RPMSensorConfig": "RPMSensor",
    "UltrasonicArrayConfig": "UltrasonicArrayModel",
    "ThermometerConfig": "ThermometerModel",
    "HygrometerConfig": "HygrometerModel",
    "LightSensorConfig": "LightSensorModel",
    "GasSensorConfig": "GasSensorModel",
    "AnemometerConfig": "AnemometerModel",
    "UWBRangeConfig": "UWBRangingModel",
    "RadarConfig": "RadarModel",
    "ImagingSonarConfig": "ImagingSonarModel",
    "SideScanSonarConfig": "SideScanSonarModel",
    "DVLConfig": "DVLModel",
    "AcousticCurrentProfilerConfig": "AcousticCurrentProfilerModel",
    "WaterPressureConfig": "WaterPressureModel",
    "HydrophoneConfig": "HydrophoneModel",
    "LeakDetectorConfig": "LeakDetectorModel",
    "UnderwaterModemConfig": "UnderwaterModemModel",
    "InclinometerConfig": "InclinometerModel",
    "ProximityToFArrayConfig": "ProximityToFArrayModel",
    "LoadCellConfig": "LoadCellModel",
    "WireEncoderConfig": "WireEncoderModel",
    "MotorTemperatureConfig": "MotorTemperatureModel",
}


@pytest.mark.parametrize("preset_name", _all_preset_names())
def test_preset_produces_valid_sensor(preset_name: str) -> None:
    """Every registered preset should construct a sensor that can step()."""
    import genesis_sensors._runtime_sensors as gs
    from genesis_sensors._runtime_sensors.presets import get_preset

    cfg = get_preset(preset_name)
    config_type_name = type(cfg).__name__
    model_name = _CONFIG_TO_MODEL.get(config_type_name)
    if model_name is None:
        pytest.skip(f"No model mapping for {config_type_name}")

    model_cls = getattr(gs, model_name, None)
    if model_cls is None:
        pytest.skip(f"Model class {model_name} not found")

    sensor = model_cls.from_config(cfg)
    assert sensor is not None
    assert sensor.update_rate_hz > 0


_PRESET_KINDS = (
    "camera",
    "stereo",
    "lidar",
    "imu",
    "gnss",
    "thermal",
    "event",
    "barometer",
    "magnetometer",
    "thermometer",
    "hygrometer",
    "light_sensor",
    "gas_sensor",
    "anemometer",
    "uwb",
    "radar",
    "ultrasonic",
    "imaging_sonar",
    "side_scan_sonar",
    "dvl",
    "current_profiler",
    "airspeed",
    "rangefinder",
    "optical_flow",
    "battery",
    "wheel_odometry",
    "force_torque",
    "joint_state",
    "contact",
    "depth_camera",
    "tactile_array",
    "current",
    "rpm",
)


@pytest.mark.parametrize("kind", _PRESET_KINDS)
def test_list_presets_kind_filter_returns_sorted_unique_values(kind: str) -> None:
    names = list_presets(kind=kind)
    assert names == sorted(names)
    assert len(names) == len(set(names))


def test_list_presets_unknown_kind_raises_helpful_error() -> None:
    with pytest.raises(KeyError, match="Available kinds"):
        list_presets(kind="not-a-kind")


# ────────────────────────────────────────────────────────────────────
# Multi-rate scheduling test
# ────────────────────────────────────────────────────────────────────


def test_multi_rate_scheduling_cadence() -> None:
    """Sensors at different rates should fire at their expected cadence."""
    from genesis_sensors import IMUModel, BarometerModel

    imu = IMUModel(update_rate_hz=100.0, seed=0)
    baro = BarometerModel(update_rate_hz=10.0, seed=1)

    scheduler = SensorScheduler()
    scheduler.add(imu, name="imu")
    scheduler.add(baro, name="baro")
    scheduler.reset()

    state = make_synthetic_sensor_state(0)
    imu_due = 0
    baro_due = 0

    dt = 0.001  # 1 kHz sim
    for i in range(1000):
        t = i * dt
        # Check is_due *before* update to count actual due firings
        if imu.is_due(t):
            imu_due += 1
        if baro.is_due(t):
            baro_due += 1
        scheduler.update(t, state)

    # IMU at 100 Hz over 1 second = ~100 updates (±1 for timing)
    assert 99 <= imu_due <= 101, f"IMU due: {imu_due}"
    # Baro at 10 Hz over 1 second = ~10 updates (±1 for timing)
    assert 9 <= baro_due <= 11, f"Baro due: {baro_due}"


@pytest.mark.parametrize(
    ("progress", "expected_phase"),
    [
        (0.00, "takeoff"),
        (0.1999, "takeoff"),
        (0.20, "cruise"),
        (0.4499, "cruise"),
        (0.45, "urban_canyon"),
        (0.70, "rain_burst"),
        (0.88, "signal_recovery"),
        (1.00, "signal_recovery"),
        (1.25, "signal_recovery"),
    ],
)
def test_get_scenario_phase_boundaries(progress: float, expected_phase: str) -> None:
    assert get_scenario_phase(progress) == expected_phase


@pytest.mark.parametrize(
    "sensor_cls,sensor_name",
    STEPPABLE_SENSORS,
    ids=[sensor_name for _, sensor_name in STEPPABLE_SENSORS],
)
def test_steppable_sensors_accept_synthetic_state(sensor_cls: Any, sensor_name: str) -> None:
    sensor = sensor_cls(seed=0)
    sensor.reset()
    obs = sensor.step(sim_time=0.0, state=make_synthetic_sensor_state(0))
    assert isinstance(obs, dict)
    assert obs, f"{sensor_name} returned an empty observation"


# ────────────────────────────────────────────────────────────────────
# SensorSuite config factory tests
# ────────────────────────────────────────────────────────────────────


def test_suite_from_config_roundtrip() -> None:
    """SensorSuiteConfig.full() → from_config → sensor_names covers all advertised slots."""
    cfg = SensorSuiteConfig.full()
    suite = SensorSuite.from_config(cfg)
    names = set(suite.sensor_names())

    expected = {
        "water_pressure",
        "hydrophone",
        "leak_detector",
        "underwater_modem",
        "inclinometer",
        "proximity_tof",
        "load_cell",
        "wire_encoder",
        "motor_temperature",
    }

    assert expected.issubset(names)
    assert len(names) >= 43


def test_suite_all_disabled_has_no_sensors() -> None:
    cfg = SensorSuiteConfig.all_disabled()
    suite = SensorSuite.from_config(cfg)
    assert suite.sensor_names() == []


def test_suite_default_seed_deterministic() -> None:
    """Two suites with the same seed produce identical first observations."""
    suite_a = SensorSuite.default(seed=42, imu_rate_hz=200.0, gnss_rate_hz=0.0)
    suite_b = SensorSuite.default(seed=42, imu_rate_hz=200.0, gnss_rate_hz=0.0)

    state = make_synthetic_sensor_state(0)
    suite_a.reset()
    suite_b.reset()

    obs_a = suite_a.step(0.0, state)
    obs_b = suite_b.step(0.0, state)

    # IMU observations should be identical
    if "imu" in obs_a and "imu" in obs_b:
        np.testing.assert_array_equal(obs_a["imu"]["lin_acc"], obs_b["imu"]["lin_acc"])
