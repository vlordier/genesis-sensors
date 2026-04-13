from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors import AnemometerModel, GasSensorModel, HygrometerModel, LightSensorModel, ThermometerModel


def test_thermometer_reports_celsius_and_fahrenheit() -> None:
    sensor = ThermometerModel(noise_sigma_c=0.0, bias_sigma_c=0.0, response_tau_s=0.01, seed=0)

    obs = sensor.step(0.0, {"ambient_temp_c": 25.0, "pos": np.zeros(3, dtype=np.float64)})

    assert float(obs["temperature_c"]) == pytest.approx(25.0)
    assert float(obs["temperature_f"]) == pytest.approx(77.0)


def test_hygrometer_reports_humidity_and_dew_point() -> None:
    sensor = HygrometerModel(noise_sigma_pct=0.0, bias_sigma_pct=0.0, response_tau_s=0.01, seed=0)

    obs = sensor.step(0.0, {"relative_humidity_pct": 65.0, "ambient_temp_c": 24.0})

    assert float(obs["relative_humidity_pct"]) == pytest.approx(65.0)
    assert 0.0 < float(obs["dew_point_c"]) < 24.0


def test_light_sensor_saturates_at_configured_maximum() -> None:
    sensor = LightSensorModel(noise_sigma_ratio=0.0, max_lux=1000.0, response_tau_s=0.01, seed=0)

    obs = sensor.step(0.0, {"illuminance_lux": 5000.0})

    assert float(obs["illuminance_lux"]) == pytest.approx(1000.0)
    assert obs["is_saturated"] is True


def test_gas_sensor_alarm_trips_inside_plume() -> None:
    sensor = GasSensorModel(
        noise_sigma_ppm=0.0,
        response_tau_s=0.01,
        background_ppm=400.0,
        alarm_threshold_ppm=800.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "pos": np.zeros(3, dtype=np.float64),
            "wind_ms": np.array([1.0, 0.0, 0.0], dtype=np.float64),
            "gas_sources": [{"pos": [0.0, 0.0, 0.0], "peak_ppm": 1500.0, "sigma_m": 1.0}],
        },
    )

    assert float(obs["concentration_ppm"]) > 800.0
    assert obs["alarm"] is True


def test_anemometer_reports_speed_and_direction() -> None:
    sensor = AnemometerModel(noise_sigma_ms=0.0, direction_noise_deg=0.0, seed=0)

    obs = sensor.step(0.0, {"wind_ms": np.array([3.0, 4.0, 0.0], dtype=np.float64)})

    assert float(obs["wind_speed_ms"]) == pytest.approx(5.0)
    assert float(obs["wind_direction_deg"]) == pytest.approx(53.130102, rel=1e-5)
    np.testing.assert_allclose(np.asarray(obs["wind_vector_ms"]), np.array([3.0, 4.0, 0.0], dtype=np.float64))
