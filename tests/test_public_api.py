from __future__ import annotations

import pytest

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
        "get_scenario_phase",
        "make_synthetic_sensor_state",
    }

    assert expected.issubset(set(synthetic_module.__all__))
    for name in expected:
        assert getattr(synthetic_module, name) is not None
