"""Underwater acoustic navigation sensor models.

This module adds:

* ``DVLModel`` — Doppler velocity log with bottom-lock and water-track modes.
* ``AcousticCurrentProfilerModel`` — depth-binned water-current estimator.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation
from .types import AcousticCurrentProfilerObservation, DVLObservation, Float64Array

if TYPE_CHECKING:
    from .config import AcousticCurrentProfilerConfig, DVLConfig


def _as_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Float64Array:
    arr = np.asarray(default if value is None else value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.asarray(default, dtype=np.float64)
    if arr.size == 1:
        return np.array([float(arr[0]), 0.0, 0.0], dtype=np.float64)
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float64)


def _iter_current_layers(raw: Any) -> list[tuple[float, Float64Array]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []

    layers: list[tuple[float, Float64Array]] = []
    for item in cast(Sequence[Any], raw):
        if not isinstance(item, Mapping):
            continue
        item_map = cast(Mapping[str, Any], item)
        depth = float(item_map.get("depth_m", item_map.get("depth", 0.0)))
        vel = _as_vec3(item_map.get("vel", item_map.get("velocity", item_map.get("current_ms"))))
        layers.append((depth, vel))
    layers.sort(key=lambda pair: pair[0])
    return layers


class DVLModel(BaseSensor[DVLObservation]):
    """Doppler velocity log for underwater navigation and bottom tracking."""

    def __init__(
        self,
        name: str = "dvl",
        update_rate_hz: float = 5.0,
        n_beams: int = 4,
        beam_angle_deg: float = 30.0,
        min_altitude_m: float = 0.2,
        max_altitude_m: float = 80.0,
        velocity_noise_sigma_ms: float = 0.01,
        range_noise_sigma_m: float = 0.02,
        dropout_prob: float = 0.01,
        water_track_blend: float = 0.20,
        nominal_sos_ms: float = 1500.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if n_beams <= 0:
            raise ValueError("n_beams must be positive")
        if min_altitude_m >= max_altitude_m:
            raise ValueError(f"min_altitude_m ({min_altitude_m}) must be less than max_altitude_m ({max_altitude_m})")

        self.n_beams = int(n_beams)
        self.beam_angle_deg = float(beam_angle_deg)
        self.min_altitude_m = float(min_altitude_m)
        self.max_altitude_m = float(max_altitude_m)
        self.velocity_noise_sigma_ms = float(max(0.0, velocity_noise_sigma_ms))
        self.range_noise_sigma_m = float(max(0.0, range_noise_sigma_m))
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 1.0))
        self.water_track_blend = float(np.clip(water_track_blend, 0.0, 1.0))
        self.nominal_sos_ms = float(max(1.0, nominal_sos_ms))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed

        # Pre-compute Janus beam directions (equally spaced around Z axis)
        _beam_angs = np.linspace(0.0, 2.0 * np.pi, self.n_beams, endpoint=False)
        theta = math.radians(self.beam_angle_deg)
        sin_t = math.sin(theta)
        cos_t = math.cos(theta)
        # Each column is a unit beam direction in body frame
        self._beam_dirs = np.column_stack(
            [sin_t * np.cos(_beam_angs), sin_t * np.sin(_beam_angs), -cos_t * np.ones(self.n_beams)]
        ).T  # shape (3, n_beams)

        self._last_obs: DVLObservation | SensorObservation = {
            "velocity_body_ms": np.zeros(3, dtype=np.float32),
            "water_track_velocity_ms": np.zeros(3, dtype=np.float32),
            "beam_ranges_m": np.zeros(self.n_beams, dtype=np.float32),
            "beam_velocities_ms": np.zeros(self.n_beams, dtype=np.float32),
            "altitude_m": 0.0,
            "speed_ms": 0.0,
            "bottom_lock": False,
            "quality": 0,
        }

    @classmethod
    def from_config(cls, config: "DVLConfig") -> "DVLModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "DVLConfig":
        from .config import DVLConfig

        return DVLConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            n_beams=self.n_beams,
            beam_angle_deg=self.beam_angle_deg,
            min_altitude_m=self.min_altitude_m,
            max_altitude_m=self.max_altitude_m,
            velocity_noise_sigma_ms=self.velocity_noise_sigma_ms,
            range_noise_sigma_m=self.range_noise_sigma_m,
            dropout_prob=self.dropout_prob,
            water_track_blend=self.water_track_blend,
            nominal_sos_ms=self.nominal_sos_ms,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_update_time = -1.0
        self._last_obs = {
            "velocity_body_ms": np.zeros(3, dtype=np.float32),
            "water_track_velocity_ms": np.zeros(3, dtype=np.float32),
            "beam_ranges_m": np.zeros(self.n_beams, dtype=np.float32),
            "beam_velocities_ms": np.zeros(self.n_beams, dtype=np.float32),
            "altitude_m": 0.0,
            "speed_ms": 0.0,
            "bottom_lock": False,
            "quality": 0,
        }

    def step(self, sim_time: float, state: SensorInput) -> DVLObservation:
        vel = _as_vec3(state.get("vel"))
        water_current = _as_vec3(state.get("water_current_ms", state.get("water_current")))
        altitude = float(state.get("range_m", max(0.0, _as_vec3(state.get("pos"))[2])))

        # Speed-of-sound correction factor
        actual_sos = float(state.get("speed_of_sound_ms", self.nominal_sos_ms))
        sos_scale = actual_sos / self.nominal_sos_ms

        bottom_lock = self.min_altitude_m <= altitude <= self.max_altitude_m and self._rng.random() >= self.dropout_prob
        water_track_velocity = vel - water_current + self._rng.normal(0.0, self.velocity_noise_sigma_ms, size=3)

        if bottom_lock:
            measured_velocity = vel + self._rng.normal(0.0, self.velocity_noise_sigma_ms, size=3)
            # Apply SoS scaling error (DVL measures Doppler shift proportional to SoS)
            measured_velocity = measured_velocity * sos_scale
            beam_base = altitude / max(math.cos(math.radians(self.beam_angle_deg)), 1e-6)
            beam_ranges = beam_base + self._rng.normal(0.0, self.range_noise_sigma_m, size=self.n_beams)
            beam_ranges = beam_ranges * sos_scale
            quality = int(np.clip(220 + 35 * (1.0 - altitude / max(self.max_altitude_m, 1e-6)), 0, 255))
        else:
            measured_velocity = (1.0 - self.water_track_blend) * water_track_velocity + self.water_track_blend * vel
            beam_ranges = np.zeros(self.n_beams, dtype=np.float64)
            quality = 40 if altitude > 0.0 else 0

        # Per-beam along-beam velocity: projection of body velocity onto each beam direction
        beam_velocities = self._beam_dirs.T @ measured_velocity  # shape (n_beams,)
        beam_velocities += self._rng.normal(0.0, self.velocity_noise_sigma_ms * 1.5, size=self.n_beams)

        obs: DVLObservation = {
            "velocity_body_ms": np.asarray(measured_velocity, dtype=np.float32),
            "water_track_velocity_ms": np.asarray(water_track_velocity, dtype=np.float32),
            "beam_ranges_m": np.asarray(np.clip(beam_ranges, 0.0, None), dtype=np.float32),
            "beam_velocities_ms": np.asarray(beam_velocities, dtype=np.float32),
            "altitude_m": float(max(0.0, altitude)),
            "speed_ms": float(np.linalg.norm(measured_velocity)),
            "bottom_lock": bool(bottom_lock),
            "quality": int(quality),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> DVLObservation | SensorObservation:
        return self._last_obs


class AcousticCurrentProfilerModel(BaseSensor[AcousticCurrentProfilerObservation]):
    """Depth-binned acoustic current profiler for water-column flow estimation."""

    def __init__(
        self,
        name: str = "current_profiler",
        update_rate_hz: float = 2.0,
        n_cells: int = 8,
        min_depth_m: float = 1.0,
        max_depth_m: float = 20.0,
        velocity_noise_sigma_ms: float = 0.02,
        attenuation_per_m: float = 0.03,
        false_bin_rate: float = 0.01,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if n_cells <= 0:
            raise ValueError("n_cells must be positive")
        if min_depth_m >= max_depth_m:
            raise ValueError(f"min_depth_m ({min_depth_m}) must be less than max_depth_m ({max_depth_m})")

        self.n_cells = int(n_cells)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self.velocity_noise_sigma_ms = float(max(0.0, velocity_noise_sigma_ms))
        self.attenuation_per_m = float(max(0.0, attenuation_per_m))
        self.false_bin_rate = float(np.clip(false_bin_rate, 0.0, 1.0))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: AcousticCurrentProfilerObservation | SensorObservation = {
            "depth_bins_m": np.linspace(self.min_depth_m, self.max_depth_m, self.n_cells, dtype=np.float32),
            "current_profile_ms": np.zeros((self.n_cells, 3), dtype=np.float32),
            "speed_profile_ms": np.zeros(self.n_cells, dtype=np.float32),
            "mean_current_ms": np.zeros(3, dtype=np.float32),
            "valid_mask": np.zeros(self.n_cells, dtype=bool),
            "n_valid_bins": 0,
        }

    @classmethod
    def from_config(cls, config: "AcousticCurrentProfilerConfig") -> "AcousticCurrentProfilerModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "AcousticCurrentProfilerConfig":
        from .config import AcousticCurrentProfilerConfig

        return AcousticCurrentProfilerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            n_cells=self.n_cells,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            velocity_noise_sigma_ms=self.velocity_noise_sigma_ms,
            attenuation_per_m=self.attenuation_per_m,
            false_bin_rate=self.false_bin_rate,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_update_time = -1.0
        self._last_obs = {
            "depth_bins_m": np.linspace(self.min_depth_m, self.max_depth_m, self.n_cells, dtype=np.float32),
            "current_profile_ms": np.zeros((self.n_cells, 3), dtype=np.float32),
            "speed_profile_ms": np.zeros(self.n_cells, dtype=np.float32),
            "mean_current_ms": np.zeros(3, dtype=np.float32),
            "valid_mask": np.zeros(self.n_cells, dtype=bool),
            "n_valid_bins": 0,
        }

    def step(self, sim_time: float, state: SensorInput) -> AcousticCurrentProfilerObservation:
        depth_bins = np.linspace(self.min_depth_m, self.max_depth_m, self.n_cells, dtype=np.float32)
        default_current = _as_vec3(state.get("water_current_ms", state.get("water_current")))
        layers = _iter_current_layers(state.get("current_layers", ()))

        profile = np.tile(default_current, (self.n_cells, 1)).astype(np.float64)
        if layers:
            sample_depths = np.asarray([depth for depth, _ in layers], dtype=np.float64)
            layer_vectors = np.asarray([vec for _, vec in layers], dtype=np.float64)
            for axis in range(3):
                profile[:, axis] = np.interp(
                    depth_bins.astype(np.float64),
                    sample_depths,
                    layer_vectors[:, axis],
                    left=layer_vectors[0, axis],
                    right=layer_vectors[-1, axis],
                )

        attenuation = np.exp(-self.attenuation_per_m * depth_bins.astype(np.float64))[:, None]
        profile *= attenuation
        if self.velocity_noise_sigma_ms > 0.0:
            profile += self._rng.normal(0.0, self.velocity_noise_sigma_ms, size=profile.shape)

        valid_mask = self._rng.random(self.n_cells) >= self.false_bin_rate
        profile[~valid_mask] = 0.0
        speed_profile = np.linalg.norm(profile, axis=1)
        mean_current = profile[valid_mask].mean(axis=0) if np.any(valid_mask) else np.zeros(3, dtype=np.float64)

        obs: AcousticCurrentProfilerObservation = {
            "depth_bins_m": depth_bins,
            "current_profile_ms": np.asarray(profile, dtype=np.float32),
            "speed_profile_ms": np.asarray(speed_profile, dtype=np.float32),
            "mean_current_ms": np.asarray(mean_current, dtype=np.float32),
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "n_valid_bins": int(np.count_nonzero(valid_mask)),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> AcousticCurrentProfilerObservation | SensorObservation:
        return self._last_obs


__all__ = ["AcousticCurrentProfilerModel", "DVLModel"]
