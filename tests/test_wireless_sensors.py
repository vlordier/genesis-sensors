from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors import RadarModel, UWBRangingModel


def test_uwb_ranging_estimates_position_from_anchors() -> None:
    sensor = UWBRangingModel(range_noise_sigma_m=0.0, dropout_prob=0.0, estimate_position=True, seed=0)
    pos = np.array([0.5, 0.25, 1.0], dtype=np.float64)
    anchors = [
        {"id": "a0", "pos": [0.0, 0.0, 0.0]},
        {"id": "a1", "pos": [2.0, 0.0, 0.0]},
        {"id": "a2", "pos": [0.0, 2.0, 0.0]},
        {"id": "a3", "pos": [0.0, 0.0, 3.0]},
    ]

    obs = sensor.step(0.0, {"pos": pos, "uwb_anchors": anchors})

    expected = np.array([np.linalg.norm(pos - np.asarray(anchor["pos"], dtype=np.float64)) for anchor in anchors])
    assert obs["anchor_ids"] == ["a0", "a1", "a2", "a3"]
    np.testing.assert_allclose(np.asarray(obs["ranges_m"]), expected, atol=1e-6)
    np.testing.assert_allclose(np.asarray(obs["position_estimate"]), pos, atol=1e-6)
    assert np.all(np.asarray(obs["valid_mask"]))


def test_radar_model_reports_front_facing_targets() -> None:
    sensor = RadarModel(
        range_noise_sigma_m=0.0,
        velocity_noise_sigma_ms=0.0,
        azimuth_noise_deg=0.0,
        elevation_noise_deg=0.0,
        detection_prob=1.0,
        false_alarm_rate=0.0,
        azimuth_fov_deg=120.0,
        elevation_fov_deg=40.0,
        max_range_m=100.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "pos": np.zeros(3, dtype=np.float64),
            "vel": np.zeros(3, dtype=np.float64),
            "radar_targets": [
                {"pos": [10.0, 0.0, 0.0], "vel": [-2.0, 0.0, 0.0], "rcs_dbsm": 12.0},
                {"pos": [8.0, 8.0, 0.0], "vel": [0.0, -1.0, 0.0], "rcs_dbsm": 8.0},
                {"pos": [-6.0, 0.0, 0.0], "vel": [1.0, 0.0, 0.0], "rcs_dbsm": 10.0},
            ],
        },
    )

    detections = np.asarray(obs["detections"])
    assert int(obs["n_detections"]) == 2
    assert detections.shape == (2, 5)
    assert float(detections[0, 0]) == pytest.approx(10.0)
    assert float(detections[0, 3]) == pytest.approx(-2.0)
    assert np.asarray(obs["points_xyz"]).shape == (2, 3)
