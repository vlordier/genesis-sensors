from __future__ import annotations

import pytest

from genesis_sensors import (
    DemoCatalogSummary,
    DemoSceneSpec,
    HeadlessScene,
    ObservationStatus,
    RigProfile,
    ScenarioPhase,
    SceneRuntimeMode,
    SensorRigSummary,
    SyntheticRolloutSummary,
    SyntheticScenarioConfig,
    build_synthetic_demo,
    describe_demo_catalog,
    filter_demo_scenes,
    get_demo_scene_spec,
    list_demo_scene_names,
    list_demo_scenes,
    list_scenario_windows,
    make_synthetic_rollout,
    summarize_synthetic_rollout,
)
from genesis_sensors import config as config_module
from genesis_sensors import genesis_bridge as bridge_module
from genesis_sensors import presets as presets_module
from genesis_sensors import synthetic as synthetic_module


def test_config_module_reexports_extended_sensor_configs() -> None:
    expected = {
        "WaterPressureConfig",
        "HydrophoneConfig",
        "LeakDetectorConfig",
        "UnderwaterModemConfig",
        "InclinometerConfig",
        "ProximityToFArrayConfig",
        "LoadCellConfig",
        "WireEncoderConfig",
        "MotorTemperatureConfig",
    }

    assert expected.issubset(set(dir(config_module)))
    for name in expected:
        value = getattr(config_module, name)
        assert value.__name__ == name


def test_presets_module_exposes_helpers_and_constants() -> None:
    assert "get_preset" in dir(presets_module)
    assert "list_presets" in dir(presets_module)
    assert "PIXHAWK_ICM20689" in dir(presets_module)

    names = presets_module.list_presets(kind="imu")
    assert "PIXHAWK_ICM20689" in names

    cfg = presets_module.get_preset("PIXHAWK_ICM20689")
    assert type(cfg).__name__ == "IMUConfig"


def test_genesis_bridge_module_reexports_state_helpers() -> None:
    expected = {
        "extract_joint_state",
        "extract_link_contact_force_n",
        "extract_link_ft_state",
        "extract_link_imu_state",
        "extract_rigid_body_state",
    }

    assert expected.issubset(set(dir(bridge_module)))
    for name in expected:
        assert callable(getattr(bridge_module, name))


@pytest.mark.parametrize("module", [config_module, presets_module, bridge_module])
def test_public_api_modules_raise_attribute_error_for_unknown_names(module: object) -> None:
    with pytest.raises(AttributeError):
        getattr(module, "definitely_missing_symbol")


def test_synthetic_module_exports_defaults_and_helpers() -> None:
    expected = {
        "DEFAULT_DT",
        "DEFAULT_LIDAR_SHAPE",
        "DEFAULT_RESOLUTION",
        "DEFAULT_TOTAL_FRAMES",
        "GNSS_ORIGIN_LLH",
        "ScenarioPhase",
        "SyntheticRolloutSummary",
        "SyntheticScenarioConfig",
        "get_scenario_phase",
        "list_scenario_windows",
        "make_synthetic_rollout",
        "make_synthetic_sensor_state",
        "summarize_synthetic_rollout",
    }

    assert expected.issubset(set(synthetic_module.__all__))
    for name in expected:
        assert getattr(synthetic_module, name) is not None


def test_public_enums_expose_expected_values() -> None:
    assert ObservationStatus.OK.value == "ok"
    assert ObservationStatus.DROPOUT_HOLD.value == "dropout_hold"
    assert ScenarioPhase.TAKEOFF.value == "takeoff"
    assert ScenarioPhase.SIGNAL_RECOVERY.value == "signal_recovery"


def test_synthetic_rollout_helpers_are_public_and_typed() -> None:
    config = SyntheticScenarioConfig(
        dt=0.02, total_frames=10, resolution=(24, 16), lidar_shape=(4, 8), tof_shape=(4, 4)
    )
    windows = list_scenario_windows()
    rollout = make_synthetic_rollout(frame_count=4, config=config)
    summary = summarize_synthetic_rollout(frame_count=4, config=config)

    assert windows[0].phase is ScenarioPhase.TAKEOFF
    assert rollout[0]["rgb"].shape == (16, 24, 3)
    assert summary.__class__.__name__ == SyntheticRolloutSummary.__name__
    assert summary.has_sensor_key("tof_ranges_m")


def test_build_synthetic_demo_is_headless_and_public() -> None:
    demo = build_synthetic_demo(dt=0.02, seed=0)

    assert demo.scene.__class__.__name__ == HeadlessScene.__name__
    assert demo.name == "synthetic"
    assert demo.rig.name == "synthetic_multimodal"

    summary = demo.rig.describe()
    assert summary.__class__.__name__ == SensorRigSummary.__name__
    assert summary.profile == RigProfile.SYNTHETIC_MULTIMODAL


def test_list_demo_scenes_exposes_catalog_metadata() -> None:
    specs = list_demo_scenes()

    assert specs
    assert specs[0].__class__.__name__ == DemoSceneSpec.__name__
    assert {spec.name for spec in specs} >= {"drone", "perception", "franka", "go2", "synthetic"}
    synthetic = next(spec for spec in specs if spec.name == "synthetic")
    assert synthetic.requires_runtime is False
    assert synthetic.profile == RigProfile.SYNTHETIC_MULTIMODAL
    assert synthetic.runtime_mode == SceneRuntimeMode.HEADLESS
    assert synthetic.is_headless is True
    assert synthetic.as_dict()["runtime_mode"] == SceneRuntimeMode.HEADLESS.value

    assert get_demo_scene_spec("synthetic").name == "synthetic"
    assert filter_demo_scenes(profile=RigProfile.SYNTHETIC_MULTIMODAL, requires_runtime=False) == (synthetic,)


def test_demo_catalog_summary_helpers_are_public() -> None:
    summary = describe_demo_catalog()

    assert summary.__class__.__name__ == DemoCatalogSummary.__name__
    assert summary.scene_count == len(list_demo_scenes())
    assert summary.profile_counts[RigProfile.SYNTHETIC_MULTIMODAL.value] == 1
    assert "synthetic" in summary.headless_scene_names()
    assert "drone" in summary.runtime_scene_names()
    assert list_demo_scene_names(query="syn") == ("synthetic",)
