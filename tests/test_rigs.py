from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors import CurrentSensor, SensorSuite, get_preset, list_presets, make_synthetic_sensor_state
from genesis_sensors.rigs import NamedContactSensor, SensorRig, make_synthetic_multimodal_rig


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
