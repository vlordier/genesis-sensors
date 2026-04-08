from __future__ import annotations

import pytest

from genesis.sensors import CurrentSensor, SensorSuite
from genesis_sensors.rigs import NamedContactSensor, SensorRig


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
