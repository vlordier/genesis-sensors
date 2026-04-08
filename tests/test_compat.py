from __future__ import annotations

import numpy as np

import genesis_sensors
from genesis_sensors._runtime_sensors import (
    CameraModel as BundledCameraModel,
    ContactSensor as BundledContactSensor,
    DepthCameraModel as BundledDepthCameraModel,
    JointStateSensor as BundledJointStateSensor,
    OpticalFlowModel as BundledOpticalFlowModel,
)


def test_headless_helpers_import_without_upstream_sensor_layer() -> None:
    state = genesis_sensors.make_synthetic_sensor_state(0)

    assert genesis_sensors.SENSOR_BACKEND in {"upstream", "bundled"}
    assert isinstance(genesis_sensors.has_upstream_sensors(), bool)
    assert {"rgb", "depth", "range_image", "temperature_map"}.issubset(state)
    assert genesis_sensors.get_scenario_phase(0.5) == "urban_canyon"


def test_bundled_sensor_backend_exposes_presets_and_models() -> None:
    assert "ZED2_STEREO" in genesis_sensors.list_presets(kind="stereo")

    scheduler = genesis_sensors.SensorScheduler()
    scheduler.add(genesis_sensors.CameraModel(resolution=(16, 12), seed=0), name="rgb")
    scheduler.reset()

    obs = scheduler.update(0.0, genesis_sensors.make_synthetic_sensor_state(0, resolution=(16, 12)))
    assert obs["rgb"]["rgb"].shape == (12, 16, 3)


def test_bundled_optical_flow_matches_documented_axis_convention() -> None:
    flow = BundledOpticalFlowModel(noise_floor_rad_s=0.0, noise_slope=0.0, seed=0)
    obs = flow.step(0.0, {"vel": np.array([1.0, 2.0, 0.0]), "pos": np.array([0.0, 0.0, 2.0])})

    assert float(obs["flow_rate_x_rad_s"]) == 1.0
    assert float(obs["flow_rate_y_rad_s"]) == -0.5


def test_bundled_camera_config_preserves_focal_length_override() -> None:
    camera = BundledCameraModel(resolution=(16, 12), focal_length_px=24.0, seed=0)
    cfg = camera.get_config()

    assert cfg.focal_length_px == 24.0


def test_reset_restores_scheduler_state_for_bundled_sensors() -> None:
    for sensor in (
        BundledContactSensor(),
        BundledDepthCameraModel(),
        BundledJointStateSensor(),
    ):
        sensor._last_update_time = 123.0
        sensor.reset()
        assert sensor._last_update_time == -1.0
