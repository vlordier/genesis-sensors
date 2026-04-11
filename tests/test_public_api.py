from __future__ import annotations

from genesis_sensors import config as config_module
from genesis_sensors import genesis_bridge as bridge_module
from genesis_sensors import presets as presets_module


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
