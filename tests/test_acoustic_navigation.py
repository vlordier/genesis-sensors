from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors import AcousticCurrentProfilerModel, DVLModel


def test_dvl_reports_bottom_lock_velocity_and_beam_ranges() -> None:
    sensor = DVLModel(
        beam_angle_deg=30.0,
        velocity_noise_sigma_ms=0.0,
        range_noise_sigma_m=0.0,
        dropout_prob=0.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "vel": np.array([1.2, -0.3, 0.1], dtype=np.float64),
            "range_m": 3.5,
            "water_current_ms": np.array([0.2, 0.1, 0.0], dtype=np.float64),
        },
    )

    velocity = np.asarray(obs["velocity_body_ms"])
    beam_ranges = np.asarray(obs["beam_ranges_m"])
    assert obs["bottom_lock"] is True
    assert velocity.shape == (3,)
    assert beam_ranges.shape == (4,)
    assert float(obs["altitude_m"]) == pytest.approx(3.5)
    assert float(obs["speed_ms"]) == pytest.approx(np.linalg.norm([1.2, -0.3, 0.1]), abs=1e-6)
    assert np.all(beam_ranges > 3.5)


def test_current_profiler_reports_layered_water_current_profile() -> None:
    sensor = AcousticCurrentProfilerModel(
        n_cells=4,
        max_depth_m=12.0,
        velocity_noise_sigma_ms=0.0,
        false_bin_rate=0.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "water_current_ms": np.array([0.4, -0.1, 0.0], dtype=np.float64),
            "current_layers": [
                {"depth_m": 1.5, "vel": [0.3, 0.0, 0.0]},
                {"depth_m": 4.0, "vel": [0.45, -0.1, 0.0]},
                {"depth_m": 8.0, "vel": [0.6, -0.2, 0.0]},
            ],
        },
    )

    profile = np.asarray(obs["current_profile_ms"])
    mean_current = np.asarray(obs["mean_current_ms"])
    assert profile.shape == (4, 3)
    assert np.asarray(obs["depth_bins_m"]).shape == (4,)
    assert np.asarray(obs["speed_profile_ms"]).shape == (4,)
    assert int(obs["n_valid_bins"]) == 4
    assert float(mean_current[0]) > 0.3
    assert float(mean_current[1]) < 0.0
