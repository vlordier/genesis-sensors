from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors import ImagingSonarModel, SideScanSonarModel


def test_imaging_sonar_highlights_targets_in_view() -> None:
    sensor = ImagingSonarModel(
        azimuth_bins=32,
        range_bins=48,
        max_range_m=20.0,
        azimuth_fov_deg=120.0,
        speckle_sigma=0.0,
        false_alarm_rate=0.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "pos": np.zeros(3, dtype=np.float64),
            "sonar_targets": [
                {"pos": [6.0, 0.0, 0.0], "strength": 1.0},
                {"pos": [8.0, 2.0, 0.0], "strength": 0.8},
                {"pos": [-4.0, 0.0, 0.0], "strength": 1.0},
            ],
        },
    )

    image = np.asarray(obs["intensity_image"])
    assert image.shape == (48, 32)
    assert int(obs["n_returns"]) == 2
    assert float(obs["strongest_return_m"]) == pytest.approx(6.0, abs=0.75)
    assert float(np.max(image)) > 0.2


def test_side_scan_sonar_separates_port_and_starboard() -> None:
    sensor = SideScanSonarModel(
        range_bins=64,
        max_range_m=30.0,
        speckle_sigma=0.0,
        false_alarm_rate=0.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "pos": np.zeros(3, dtype=np.float64),
            "sonar_targets": [
                {"pos": [4.0, 6.0, 0.0], "strength": 1.0},
                {"pos": [5.0, -8.0, 0.0], "strength": 0.9},
            ],
        },
    )

    port = np.asarray(obs["port_intensity"])
    starboard = np.asarray(obs["starboard_intensity"])
    assert port.shape == (64,)
    assert starboard.shape == (64,)
    assert int(obs["port_hits"]) == 1
    assert int(obs["starboard_hits"]) == 1
    assert float(np.max(port)) > 0.2
    assert float(np.max(starboard)) > 0.2
