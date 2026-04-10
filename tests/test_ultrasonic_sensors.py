from __future__ import annotations

import numpy as np
import pytest

from genesis_sensors import UltrasonicArrayModel


def test_ultrasonic_array_reports_per_beam_ranges() -> None:
    sensor = UltrasonicArrayModel(
        n_beams=4,
        beam_span_deg=90.0,
        noise_floor_m=0.0,
        noise_slope=0.0,
        dropout_prob=0.0,
        cross_talk_prob=0.0,
        seed=0,
    )

    obs = sensor.step(
        0.0,
        {
            "ambient_temp_c": 20.0,
            "ultrasonic_ranges_m": {
                "front_left": 0.8,
                "front_right": 1.2,
                "rear_left": 2.0,
                "rear_right": 3.5,
            },
        },
    )

    assert obs["beam_ids"] == ["front_left", "front_right", "rear_left", "rear_right"]
    np.testing.assert_allclose(np.asarray(obs["ranges_m"]), np.array([0.8, 1.2, 2.0, 3.5]), atol=1e-6)
    assert np.all(np.asarray(obs["valid_mask"]))
    assert float(obs["nearest_range_m"]) == pytest.approx(0.8)


def test_ultrasonic_array_marks_out_of_range_beams() -> None:
    sensor = UltrasonicArrayModel(
        n_beams=4,
        min_range_m=0.1,
        max_range_m=4.0,
        noise_floor_m=0.0,
        noise_slope=0.0,
        dropout_prob=0.0,
        cross_talk_prob=0.0,
        no_hit_value=0.0,
        seed=0,
    )

    obs = sensor.step(0.0, {"ultrasonic_ranges_m": [0.02, 1.0, 6.0, float("nan")]})

    np.testing.assert_array_equal(np.asarray(obs["valid_mask"]), np.array([False, True, False, False]))
    np.testing.assert_allclose(np.asarray(obs["ranges_m"]), np.array([0.0, 1.0, 0.0, 0.0]), atol=1e-6)
    assert float(obs["nearest_range_m"]) == pytest.approx(1.0)
