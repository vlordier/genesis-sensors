"""Ultrasonic proximity / sonar array sensor model.

This module simulates low-cost ultrasound transducer arrays used for bumper
avoidance, docking, and short-range altitude / wall sensing. Compared with the
existing single-beam `RangefinderModel`, this model adds:

* **multiple beams** spanning a configurable arc or ring,
* **temperature-dependent speed-of-sound effects**, and
* **cross-talk / echo bleed** between neighbouring transducers.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation
from .types import UltrasonicObservation

if TYPE_CHECKING:
    from .config import UltrasonicArrayConfig

_REF_SOUND_SPEED_MS = 343.0


def _default_angles(n_beams: int, beam_span_deg: float) -> np.ndarray:
    if n_beams <= 1:
        return np.zeros(1, dtype=np.float32)
    return np.linspace(-beam_span_deg / 2.0, beam_span_deg / 2.0, n_beams, dtype=np.float32)


def _coerce_ranges(
    raw: Any,
    *,
    n_beams: int,
    default_range_m: float | None,
) -> tuple[list[str], list[float]]:
    beam_ids: list[str] = []
    values: list[float] = []

    if isinstance(raw, Mapping):
        for key, value in cast(Mapping[Any, Any], raw).items():
            beam_ids.append(str(key))
            values.append(float(value) if value is not None else float("nan"))
    elif isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for idx, value in enumerate(cast(Sequence[Any], raw)):
            beam_ids.append(f"beam_{idx}")
            values.append(float(value) if value is not None else float("nan"))

    if not values and default_range_m is not None:
        beam_ids = [f"beam_{idx}" for idx in range(n_beams)]
        values = [float(default_range_m)] * n_beams

    if not values:
        beam_ids = [f"beam_{idx}" for idx in range(n_beams)]
        values = [float("nan")] * n_beams

    if len(values) < n_beams:
        for idx in range(len(values), n_beams):
            beam_ids.append(f"beam_{idx}")
            values.append(float(default_range_m) if default_range_m is not None else float("nan"))
    elif len(values) > n_beams:
        beam_ids = beam_ids[:n_beams]
        values = values[:n_beams]

    return beam_ids, values


class UltrasonicArrayModel(BaseSensor[UltrasonicObservation]):
    """Short-range ultrasonic / sonar array for proximity sensing."""

    def __init__(
        self,
        name: str = "ultrasonic",
        update_rate_hz: float = 15.0,
        n_beams: int = 4,
        beam_span_deg: float = 120.0,
        beam_angles_deg: list[float] | None = None,
        min_range_m: float = 0.02,
        max_range_m: float = 4.5,
        noise_floor_m: float = 0.005,
        noise_slope: float = 0.01,
        dropout_prob: float = 0.02,
        cross_talk_prob: float = 0.03,
        beam_width_deg: float = 25.0,
        temperature_compensation: bool = True,
        no_hit_value: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if n_beams <= 0:
            raise ValueError(f"n_beams must be positive, got {n_beams}")
        if min_range_m >= max_range_m:
            raise ValueError(f"min_range_m ({min_range_m}) must be less than max_range_m ({max_range_m})")

        self.n_beams = int(n_beams)
        self.beam_span_deg = float(max(0.0, beam_span_deg))
        self._custom_beam_angles = beam_angles_deg is not None
        if beam_angles_deg is None:
            self.beam_angles_deg = _default_angles(self.n_beams, self.beam_span_deg)
        else:
            if len(beam_angles_deg) != self.n_beams:
                raise ValueError(
                    f"beam_angles_deg must have exactly n_beams ({self.n_beams}) values, got {len(beam_angles_deg)}"
                )
            self.beam_angles_deg = np.asarray(beam_angles_deg, dtype=np.float32)

        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.noise_floor_m = float(max(0.0, noise_floor_m))
        self.noise_slope = float(max(0.0, noise_slope))
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 1.0))
        self.cross_talk_prob = float(np.clip(cross_talk_prob, 0.0, 1.0))
        self.beam_width_deg = float(max(0.0, beam_width_deg))
        self.temperature_compensation = bool(temperature_compensation)
        self.no_hit_value = float(no_hit_value)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: UltrasonicObservation | SensorObservation = {
            "beam_ids": [f"beam_{idx}" for idx in range(self.n_beams)],
            "beam_angles_deg": self.beam_angles_deg.copy(),
            "ranges_m": np.full(self.n_beams, self.no_hit_value, dtype=np.float32),
            "valid_mask": np.zeros(self.n_beams, dtype=bool),
            "echo_strength": np.zeros(self.n_beams, dtype=np.float32),
            "nearest_range_m": self.no_hit_value,
        }

    @classmethod
    def from_config(cls, config: "UltrasonicArrayConfig") -> "UltrasonicArrayModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "UltrasonicArrayConfig":
        from .config import UltrasonicArrayConfig

        return UltrasonicArrayConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            n_beams=self.n_beams,
            beam_span_deg=self.beam_span_deg,
            beam_angles_deg=self.beam_angles_deg.tolist() if self._custom_beam_angles else None,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            noise_floor_m=self.noise_floor_m,
            noise_slope=self.noise_slope,
            dropout_prob=self.dropout_prob,
            cross_talk_prob=self.cross_talk_prob,
            beam_width_deg=self.beam_width_deg,
            temperature_compensation=self.temperature_compensation,
            no_hit_value=self.no_hit_value,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {
            "beam_ids": [f"beam_{idx}" for idx in range(self.n_beams)],
            "beam_angles_deg": self.beam_angles_deg.copy(),
            "ranges_m": np.full(self.n_beams, self.no_hit_value, dtype=np.float32),
            "valid_mask": np.zeros(self.n_beams, dtype=bool),
            "echo_strength": np.zeros(self.n_beams, dtype=np.float32),
            "nearest_range_m": self.no_hit_value,
        }
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> UltrasonicObservation:
        default_range_m = state.get("range_m")
        default_range = float(default_range_m) if default_range_m is not None else None
        beam_ids, raw_ranges = _coerce_ranges(
            state.get("ultrasonic_ranges_m"),
            n_beams=self.n_beams,
            default_range_m=default_range,
        )

        beam_angles = self.beam_angles_deg
        if beam_angles.shape[0] != len(raw_ranges):
            beam_angles = _default_angles(len(raw_ranges), self.beam_span_deg)

        ambient_temp_c = float(state.get("ambient_temp_c", 20.0))
        humidity_pct = float(state.get("relative_humidity_pct", 50.0))
        sound_speed_ms = 331.3 + 0.606 * ambient_temp_c
        temp_scale = 1.0 if self.temperature_compensation else sound_speed_ms / _REF_SOUND_SPEED_MS

        measured_ranges: list[float] = []
        valid_mask: list[bool] = []
        echo_strength: list[float] = []
        prev_true_range: float | None = None
        valid_measurements: list[float] = []

        for idx, raw_range in enumerate(raw_ranges):
            if not math.isfinite(raw_range):
                measured_ranges.append(self.no_hit_value)
                valid_mask.append(False)
                echo_strength.append(0.0)
                continue

            true_range = float(raw_range)
            if prev_true_range is not None and abs(prev_true_range - true_range) < 0.12:
                if self.cross_talk_prob > 0.0 and float(self._rng.random()) < self.cross_talk_prob:
                    true_range = 0.5 * (true_range + prev_true_range)

            if true_range < self.min_range_m or true_range > self.max_range_m:
                measured_ranges.append(self.no_hit_value)
                valid_mask.append(False)
                echo_strength.append(0.0)
                prev_true_range = true_range
                continue

            if self.dropout_prob > 0.0 and float(self._rng.random()) < self.dropout_prob:
                measured_ranges.append(self.no_hit_value)
                valid_mask.append(False)
                echo_strength.append(0.0)
                prev_true_range = true_range
                continue

            sigma = self.noise_floor_m + self.noise_slope * true_range
            measured = true_range * temp_scale + float(self._rng.normal(0.0, sigma))
            measured = float(np.clip(measured, self.min_range_m, self.max_range_m))

            if self.beam_width_deg > 0.0 and self.beam_span_deg > 0.0:
                angular_scale = (
                    1.0 - min(abs(float(beam_angles[idx])) / max(self.beam_span_deg / 2.0, 1e-6), 1.0) * 0.35
                )
            else:
                angular_scale = 1.0
            humidity_scale = float(np.clip(1.0 - 0.0015 * humidity_pct, 0.65, 1.0))
            strength = float(
                np.clip(
                    math.exp(-2.2 * measured / max(self.max_range_m, 1e-6)) * angular_scale * humidity_scale, 0.0, 1.0
                )
            )

            measured_ranges.append(measured)
            valid_mask.append(True)
            echo_strength.append(strength)
            valid_measurements.append(measured)
            prev_true_range = true_range

        obs: UltrasonicObservation = {
            "beam_ids": beam_ids,
            "beam_angles_deg": np.asarray(beam_angles, dtype=np.float32),
            "ranges_m": np.asarray(measured_ranges, dtype=np.float32),
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "echo_strength": np.asarray(echo_strength, dtype=np.float32),
            "nearest_range_m": float(min(valid_measurements)) if valid_measurements else self.no_hit_value,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> UltrasonicObservation | SensorObservation:
        return self._last_obs


__all__ = ["UltrasonicArrayModel"]
