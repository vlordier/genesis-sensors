from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from genesis_sensors._runtime_sensors.config import CurrentSensorConfig, IMUConfig, SensorSuiteConfig
from genesis_sensors import (
    CurrentSensor,
    IMUModel,
    SensorSuite,
    ThermalCameraModel,
    get_preset,
    list_presets,
    make_synthetic_sensor_state,
)
from genesis_sensors.rigs import NamedContactSensor, RigProfile, SensorRig, make_synthetic_multimodal_rig


def test_named_contact_sensor_reads_only_its_link_force() -> None:
    sensor = NamedContactSensor(
        link_name="FL_calf",
        force_threshold_n=1.0,
        noise_sigma_n=0.0,
        debounce_steps=0,
        seed=0,
    )
    obs = sensor.step(0.0, {"contact_forces": {"FL_calf": 2.0, "FR_calf": 0.0}})
    assert obs["contact_detected"] is True
    assert float(obs["force_n"]) == pytest.approx(2.0)


def test_sensor_rig_merges_extra_state_before_stepping() -> None:
    suite = SensorSuite(current=CurrentSensor(noise_sigma_a=0.0, offset_a=0.0, voltage_nominal_v=12.0, seed=0))
    rig = SensorRig(name="test", suite=suite, state_fn=lambda: {"current_a": 1.0})

    obs = rig.step(0.0, extra_state={"voltage_v": 24.0})

    assert float(obs["current"]["current_a"]) == pytest.approx(1.0)
    assert float(obs["current"]["power_w"]) == pytest.approx(24.0)


def test_sensor_rig_exposes_sensor_names() -> None:
    suite = SensorSuite(current=CurrentSensor(noise_sigma_a=0.0, seed=0))
    rig = SensorRig(name="test", suite=suite, state_fn=lambda: {"current_a": 0.0})
    assert rig.sensor_names() == ["current"]


def test_synthetic_multimodal_rig_surfaces_more_upstream_sensors() -> None:
    rig = make_synthetic_multimodal_rig(dt=0.05, seed=0)
    rig.reset()

    obs0 = rig.step(0.0)
    obs1 = rig.step(0.05)

    assert {
        "rgb",
        "events",
        "thermal",
        "lidar",
        "radio",
        "stereo",
        "wheel_odometry",
        "thermometer",
        "hygrometer",
        "light_sensor",
        "gas_sensor",
        "anemometer",
        "uwb",
        "radar",
        "ultrasonic",
        "imaging_sonar",
        "side_scan",
        "dvl",
        "current_profiler",
    }.issubset(set(rig.sensor_names()))
    assert np.asarray(obs0["rgb"]["rgb"]).shape[-1] == 3
    assert np.asarray(obs0["thermal"]["temperature_c"]).ndim == 2
    assert "points" in obs0["lidar"]
    assert "queue_depth" in obs1["radio"]
    assert "linear_vel_ms" in obs1["wheel_odometry"]
    assert "temperature_c" in obs0["thermometer"]
    assert "relative_humidity_pct" in obs0["hygrometer"]
    assert "illuminance_lux" in obs0["light_sensor"]
    assert "concentration_ppm" in obs0["gas_sensor"]
    assert "wind_speed_ms" in obs0["anemometer"]
    assert "position_estimate" in obs0["uwb"]
    assert "n_detections" in obs0["radar"]
    assert "nearest_range_m" in obs0["ultrasonic"]
    assert "intensity_image" in obs0["imaging_sonar"]
    assert "port_intensity" in obs0["side_scan"]
    assert "bottom_lock" in obs0["dvl"]
    assert "current_profile_ms" in obs0["current_profiler"]


def test_sensor_rig_describe_reports_typed_profile_and_sensor_count() -> None:
    rig = make_synthetic_multimodal_rig(dt=0.05, seed=0)

    summary = rig.describe()
    assert summary.profile == RigProfile.SYNTHETIC_MULTIMODAL
    assert summary.sensor_count == len(summary.sensor_names)
    assert summary.has_sensor("imu") is True
    assert "imu" in summary.preview_sensor_names(limit=10)
    assert summary.as_dict()["profile"] == "synthetic_multimodal"
    assert summary.as_dict()["sensor_count"] >= 10


def test_synthetic_state_and_preset_helpers_expose_upstream_surface() -> None:
    state = make_synthetic_sensor_state(2)

    assert {
        "rgb",
        "rgb_right",
        "depth",
        "seg",
        "range_image",
        "temperature_map",
        "ambient_temp_c",
        "relative_humidity_pct",
        "illuminance_lux",
        "gas_sources",
        "uwb_anchors",
        "radar_targets",
        "ultrasonic_ranges_m",
        "sonar_targets",
        "water_turbidity_ntu",
        "water_current_ms",
        "current_layers",
    }.issubset(state)
    assert "ZED2_STEREO" in list_presets(kind="stereo")
    assert "DS18B20_PROBE" in list_presets(kind="thermometer")
    assert "DAVIS_6410_ANEMOMETER" in list_presets(kind="anemometer")
    assert "QORVO_DWM3001C" in list_presets(kind="uwb")
    assert "TI_IWR6843AOP" in list_presets(kind="radar")
    assert "HC_SR04_ARRAY4" in list_presets(kind="ultrasonic")
    assert "BLUEVIEW_P900_130" in list_presets(kind="imaging_sonar")
    assert "EDGETECH_4125" in list_presets(kind="side_scan_sonar")
    assert "NORTEK_DVL1000" in list_presets(kind="dvl")
    assert "TELEDYNE_WORKHORSE_600" in list_presets(kind="current_profiler")
    assert get_preset("FLIR_BOSON_320").name == "FLIR_BOSON_320"
    assert get_preset("TSL2591_LIGHT").name == "TSL2591_LIGHT"


def _load_doc_assets_module():
    module_name = "_test_generate_sensor_doc_assets"
    module_path = Path(__file__).resolve().parents[1] / "examples" / "generate_sensor_doc_assets.py"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:
        if "torch" in str(exc).lower() or "genesis" in str(exc).lower():
            pytest.skip("Genesis/Torch runtime is not available in the current environment")
        raise
    return module


def test_doc_asset_specs_are_all_backed_by_genesis_demos() -> None:
    doc_assets = _load_doc_assets_module()

    assert set(doc_assets._build_specs()) <= set(doc_assets.DEMO_SENSOR_KEYS)


def test_generate_assets_requires_genesis_capture(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    doc_assets = _load_doc_assets_module()

    monkeypatch.setattr(
        doc_assets,
        "_collect_genesis_captures",
        lambda specs, *, frames, dt, seed, only: {},
    )

    with pytest.raises(RuntimeError, match="Genesis-backed capture"):
        doc_assets.generate_assets(tmp_path, frames=2, dt=0.05, seed=0, only={"imu"})


def test_common_noise_model_can_disable_runtime_noise() -> None:
    current = CurrentSensor(noise_sigma_a=1.5, offset_a=0.0, voltage_nominal_v=12.0, seed=0)
    current.configure_noise_model("none")

    current_obs = current.step(0.0, {"current_a": 3.0, "voltage_v": 12.0})
    assert float(current_obs["current_a"]) == pytest.approx(3.0)
    assert float(current_obs["power_w"]) == pytest.approx(36.0)

    imu = IMUModel.from_preset("PIXHAWK_ICM20689", noise_model="none")
    assert imu.noise_model == "none"

    thermal = ThermalCameraModel(noise_sigma=0.8, nuc_sigma=0.0, resolution=(6, 4), seed=0)
    thermal.configure_noise_model("none")
    thermal_obs = thermal.step(
        0.0,
        {
            "seg": np.zeros((4, 6), dtype=np.int32),
            "temperature_map": {0: 25.0},
        },
    )
    assert np.allclose(np.asarray(thermal_obs["temperature_c"], dtype=float), 25.0)


def test_sensor_suite_set_seed_preserves_noise_model_configuration() -> None:
    imu = IMUModel(noise_density_acc=0.2, noise_density_gyr=0.2, bias_sigma_acc=0.0, bias_sigma_gyr=0.0, seed=0)
    imu.configure_noise_model("uniform")

    suite = SensorSuite(imu=imu)
    suite.set_seed(123)
    suite.reset()
    obs = suite.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": np.zeros(3)})

    assert suite.get_sensor("imu").noise_model == "uniform"
    assert np.max(np.abs(np.asarray(obs["imu"]["lin_acc"], dtype=float))) <= np.sqrt(3.0) * imu._sigma_acc + 1e-6


def test_from_config_applies_noise_model_controls() -> None:
    cfg = CurrentSensorConfig(
        noise_sigma_a=0.5,
        offset_a=0.0,
        voltage_nominal_v=12.0,
        noise_model="uniform",
        noise_outlier_prob=0.2,
        noise_outlier_scale=3.0,
        seed=0,
    )
    sensor = CurrentSensor.from_config(cfg)

    assert sensor.noise_model == "uniform"
    assert sensor.noise_outlier_prob == pytest.approx(0.2)
    assert sensor.noise_outlier_scale == pytest.approx(3.0)


def test_sensor_suite_from_config_applies_noise_models() -> None:
    suite = SensorSuite.from_config(
        SensorSuiteConfig(
            imu=IMUConfig(
                noise_density_acc=0.2,
                noise_density_gyr=0.2,
                bias_sigma_acc=0.0,
                bias_sigma_gyr=0.0,
                noise_model="none",
                seed=0,
            )
        )
    )

    suite.reset()
    obs = suite.step(0.0, {"lin_acc": np.zeros(3), "ang_vel": np.zeros(3), "gravity_body": np.zeros(3)})
    assert suite.get_sensor("imu").noise_model == "none"
    assert np.allclose(np.asarray(obs["imu"]["lin_acc"], dtype=float), 0.0)
