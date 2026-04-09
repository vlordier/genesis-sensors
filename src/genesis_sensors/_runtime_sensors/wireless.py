"""Wireless ranging and radar sensor models.

This module adds radio-centric situational-awareness sensors beyond the packet
telemetry link itself:

* ``UWBRangingModel`` for anchor-based UWB distance measurements and optional
  trilateration.
* ``RadarModel`` for coarse range / azimuth / radial-velocity detections.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation
from .types import Float64Array, RadarObservation, UWBObservation

if TYPE_CHECKING:
    from .config import RadarConfig, UWBRangeConfig

_MIN_DISTANCE_M: Final[float] = 0.05
_UWB_FREQ_MHZ: Final[float] = 6489.6


def _as_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Float64Array:
    arr = np.asarray(default if value is None else value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.asarray(default, dtype=np.float64)
    if arr.size == 1:
        return np.array([float(arr[0]), 0.0, 0.0], dtype=np.float64)
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float64)


def _weather_value(state: SensorInput, key: str, default: float) -> float:
    if key in state:
        return float(state.get(key, default))
    weather = state.get("weather", {})
    if isinstance(weather, Mapping):
        return float(cast(Mapping[str, Any], weather).get(key, default))
    return float(default)


def _iter_anchor_specs(raw: Any) -> list[tuple[str, Float64Array, bool]]:
    anchors: list[tuple[str, Float64Array, bool]] = []
    if raw is None:
        return anchors

    if isinstance(raw, Mapping):
        for anchor_id, spec in cast(Mapping[Any, Any], raw).items():
            if isinstance(spec, Mapping):
                spec_map = cast(Mapping[str, Any], spec)
                pos = _as_vec3(spec_map.get("pos", spec_map.get("position")))
                los = bool(spec_map.get("los", True))
            else:
                pos = _as_vec3(spec)
                los = True
            anchors.append((str(anchor_id), pos, los))
        return anchors

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for idx, item in enumerate(cast(Sequence[Any], raw)):
            if isinstance(item, Mapping):
                item_map = cast(Mapping[str, Any], item)
                anchor_id = str(item_map.get("id", f"anchor_{idx}"))
                pos = _as_vec3(item_map.get("pos", item_map.get("position")))
                los = bool(item_map.get("los", True))
            else:
                anchor_id = f"anchor_{idx}"
                pos = _as_vec3(item)
                los = True
            anchors.append((anchor_id, pos, los))
    return anchors


def _iter_radar_targets(raw: Any) -> list[dict[str, Any]]:
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        return []

    targets: list[dict[str, Any]] = []
    for idx, item in enumerate(cast(Sequence[Any], raw)):
        if isinstance(item, Mapping):
            item_map = cast(Mapping[str, Any], item)
            targets.append(
                {
                    "id": str(item_map.get("id", f"target_{idx}")),
                    "pos": _as_vec3(item_map.get("pos", item_map.get("position"))),
                    "vel": _as_vec3(item_map.get("vel", item_map.get("velocity"))),
                    "rcs_dbsm": float(item_map.get("rcs_dbsm", 10.0)),
                }
            )
        else:
            targets.append({"id": f"target_{idx}", "pos": _as_vec3(item), "vel": np.zeros(3), "rcs_dbsm": 10.0})
    return targets


def _estimate_position(anchor_positions: Float64Array, ranges_m: Float64Array) -> Float64Array | None:
    if anchor_positions.shape[0] < 4:
        return None
    a0 = anchor_positions[0]
    r0 = float(ranges_m[0])
    a = 2.0 * (anchor_positions[1:] - a0)
    b = r0**2 - ranges_m[1:] ** 2 - np.sum(a0**2) + np.sum(anchor_positions[1:] ** 2, axis=1)
    solution, *_ = np.linalg.lstsq(a, b, rcond=None)
    return solution.astype(np.float64)


class UWBRangingModel(BaseSensor[UWBObservation]):
    """Anchor-based ultra-wideband ranging and optional trilateration."""

    def __init__(
        self,
        name: str = "uwb",
        update_rate_hz: float = 20.0,
        range_noise_sigma_m: float = 0.05,
        dropout_prob: float = 0.02,
        max_range_m: float = 60.0,
        nlos_bias_m: float = 0.3,
        tx_power_dbm: float = 0.0,
        estimate_position: bool = True,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.range_noise_sigma_m = float(max(0.0, range_noise_sigma_m))
        self.dropout_prob = float(np.clip(dropout_prob, 0.0, 1.0))
        self.max_range_m = float(max(max_range_m, 0.1))
        self.nlos_bias_m = float(max(0.0, nlos_bias_m))
        self.tx_power_dbm = float(tx_power_dbm)
        self.estimate_position = bool(estimate_position)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: UWBObservation | SensorObservation = {
            "anchor_ids": [],
            "ranges_m": np.empty(0, dtype=np.float32),
            "rssi_dbm": np.empty(0, dtype=np.float32),
            "los": np.empty(0, dtype=bool),
            "valid_mask": np.empty(0, dtype=bool),
            "position_estimate": np.full(3, np.nan, dtype=np.float64),
        }

    @classmethod
    def from_config(cls, config: "UWBRangeConfig") -> "UWBRangingModel":
        return cls(**config.model_dump())

    def get_config(self) -> "UWBRangeConfig":
        from .config import UWBRangeConfig

        return UWBRangeConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            range_noise_sigma_m=self.range_noise_sigma_m,
            dropout_prob=self.dropout_prob,
            max_range_m=self.max_range_m,
            nlos_bias_m=self.nlos_bias_m,
            tx_power_dbm=self.tx_power_dbm,
            estimate_position=self.estimate_position,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {
            "anchor_ids": [],
            "ranges_m": np.empty(0, dtype=np.float32),
            "rssi_dbm": np.empty(0, dtype=np.float32),
            "los": np.empty(0, dtype=bool),
            "valid_mask": np.empty(0, dtype=bool),
            "position_estimate": np.full(3, np.nan, dtype=np.float64),
        }
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> UWBObservation:
        pos = _as_vec3(state.get("pos"))
        anchors = _iter_anchor_specs(state.get("uwb_anchors"))

        if not anchors:
            obs: UWBObservation = {
                "anchor_ids": [],
                "ranges_m": np.empty(0, dtype=np.float32),
                "rssi_dbm": np.empty(0, dtype=np.float32),
                "los": np.empty(0, dtype=bool),
                "valid_mask": np.empty(0, dtype=bool),
                "position_estimate": np.full(3, np.nan, dtype=np.float64),
            }
            self._last_obs = obs
            self._mark_updated(sim_time)
            return obs

        anchor_ids: list[str] = []
        ranges: list[float] = []
        rssi_values: list[float] = []
        los_flags: list[bool] = []
        valid_mask: list[bool] = []
        valid_positions: list[Float64Array] = []
        valid_ranges: list[float] = []

        for anchor_id, anchor_pos, has_los in anchors:
            anchor_ids.append(anchor_id)
            los_flags.append(has_los)
            dist_true = max(float(np.linalg.norm(pos - anchor_pos)), _MIN_DISTANCE_M)
            fspl_db = 32.44 + 20.0 * math.log10(_UWB_FREQ_MHZ) + 20.0 * math.log10(dist_true / 1000.0)
            nlos_loss = 0.0 if has_los else 6.0
            rssi_values.append(self.tx_power_dbm - fspl_db - nlos_loss)

            is_valid = dist_true <= self.max_range_m and self._rng.random() >= self.dropout_prob
            valid_mask.append(is_valid)
            if not is_valid:
                ranges.append(float("nan"))
                continue

            sigma = self.range_noise_sigma_m * (2.0 if not has_los else 1.0)
            measurement = dist_true + float(self._rng.normal(0.0, sigma)) + (0.0 if has_los else self.nlos_bias_m)
            measurement = float(np.clip(measurement, 0.0, self.max_range_m))
            ranges.append(measurement)
            valid_positions.append(anchor_pos)
            valid_ranges.append(measurement)

        if self.estimate_position and len(valid_positions) >= 4:
            estimate = _estimate_position(np.vstack(valid_positions), np.asarray(valid_ranges, dtype=np.float64))
            position_estimate = estimate if estimate is not None else np.full(3, np.nan, dtype=np.float64)
        else:
            position_estimate = np.full(3, np.nan, dtype=np.float64)

        obs = {
            "anchor_ids": anchor_ids,
            "ranges_m": np.asarray(ranges, dtype=np.float32),
            "rssi_dbm": np.asarray(rssi_values, dtype=np.float32),
            "los": np.asarray(los_flags, dtype=bool),
            "valid_mask": np.asarray(valid_mask, dtype=bool),
            "position_estimate": np.asarray(position_estimate, dtype=np.float64),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> UWBObservation | SensorObservation:
        return self._last_obs


class RadarModel(BaseSensor[RadarObservation]):
    """Simple scanning radar model with range, angle, and radial-velocity detections."""

    def __init__(
        self,
        name: str = "radar",
        update_rate_hz: float = 15.0,
        max_range_m: float = 120.0,
        min_range_m: float = 0.5,
        azimuth_fov_deg: float = 120.0,
        elevation_fov_deg: float = 40.0,
        range_noise_sigma_m: float = 0.15,
        velocity_noise_sigma_ms: float = 0.08,
        azimuth_noise_deg: float = 0.4,
        elevation_noise_deg: float = 0.25,
        detection_prob: float = 0.97,
        false_alarm_rate: float = 0.05,
        rain_attenuation_db_per_mm_h: float = 0.12,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.max_range_m = float(max(max_range_m, 1.0))
        self.min_range_m = float(np.clip(min_range_m, 0.0, self.max_range_m - 1e-6))
        self.azimuth_fov_deg = float(max(azimuth_fov_deg, 1.0))
        self.elevation_fov_deg = float(max(elevation_fov_deg, 1.0))
        self.range_noise_sigma_m = float(max(0.0, range_noise_sigma_m))
        self.velocity_noise_sigma_ms = float(max(0.0, velocity_noise_sigma_ms))
        self.azimuth_noise_deg = float(max(0.0, azimuth_noise_deg))
        self.elevation_noise_deg = float(max(0.0, elevation_noise_deg))
        self.detection_prob = float(np.clip(detection_prob, 0.0, 1.0))
        self.false_alarm_rate = float(max(0.0, false_alarm_rate))
        self.rain_attenuation_db_per_mm_h = float(max(0.0, rain_attenuation_db_per_mm_h))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: RadarObservation | SensorObservation = {
            "detections": np.empty((0, 5), dtype=np.float32),
            "points_xyz": np.empty((0, 3), dtype=np.float32),
            "n_detections": 0,
        }

    @classmethod
    def from_config(cls, config: "RadarConfig") -> "RadarModel":
        return cls(**config.model_dump())

    def get_config(self) -> "RadarConfig":
        from .config import RadarConfig

        return RadarConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            max_range_m=self.max_range_m,
            min_range_m=self.min_range_m,
            azimuth_fov_deg=self.azimuth_fov_deg,
            elevation_fov_deg=self.elevation_fov_deg,
            range_noise_sigma_m=self.range_noise_sigma_m,
            velocity_noise_sigma_ms=self.velocity_noise_sigma_ms,
            azimuth_noise_deg=self.azimuth_noise_deg,
            elevation_noise_deg=self.elevation_noise_deg,
            detection_prob=self.detection_prob,
            false_alarm_rate=self.false_alarm_rate,
            rain_attenuation_db_per_mm_h=self.rain_attenuation_db_per_mm_h,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {
            "detections": np.empty((0, 5), dtype=np.float32),
            "points_xyz": np.empty((0, 3), dtype=np.float32),
            "n_detections": 0,
        }
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> RadarObservation:
        pos = _as_vec3(state.get("pos"))
        sensor_vel = _as_vec3(state.get("vel"))
        rain_rate = _weather_value(state, "rain_rate_mm_h", 0.0)
        raw_targets = _iter_radar_targets(state.get("radar_targets", ()))

        detections: list[list[float]] = []
        points_xyz: list[list[float]] = []

        for target in raw_targets:
            rel = cast(Float64Array, target["pos"]) - pos
            range_true = float(np.linalg.norm(rel))
            if not (self.min_range_m <= range_true <= self.max_range_m):
                continue

            xy_norm = float(np.linalg.norm(rel[:2]))
            azimuth_deg = float(math.degrees(math.atan2(rel[1], rel[0])))
            elevation_deg = float(math.degrees(math.atan2(rel[2], max(xy_norm, 1e-9))))
            if abs(azimuth_deg) > self.azimuth_fov_deg / 2.0 or abs(elevation_deg) > self.elevation_fov_deg / 2.0:
                continue

            rel_vel = cast(Float64Array, target["vel"]) - sensor_vel
            radial_velocity = float(np.dot(rel_vel, rel / max(range_true, 1e-9)))
            snr_db = 26.0 + float(target["rcs_dbsm"]) - 20.0 * math.log10(max(range_true, 1.0))
            snr_db -= rain_rate * self.rain_attenuation_db_per_mm_h

            # Treat `detection_prob` as the nominal in-FOV hit rate under
            # reasonable SNR, and only attenuate it for marginal returns.
            snr_scale = float(np.clip((snr_db + 5.0) / 25.0, 0.0, 1.0))
            if snr_db >= 8.0:
                snr_scale = 1.0
            p_detect = self.detection_prob * snr_scale
            if self._rng.random() > p_detect:
                continue

            det_range = float(
                np.clip(range_true + self._rng.normal(0.0, self.range_noise_sigma_m), 0.0, self.max_range_m)
            )
            det_az = azimuth_deg + float(self._rng.normal(0.0, self.azimuth_noise_deg))
            det_el = elevation_deg + float(self._rng.normal(0.0, self.elevation_noise_deg))
            det_vel = radial_velocity + float(self._rng.normal(0.0, self.velocity_noise_sigma_ms))
            detections.append([det_range, det_az, det_el, det_vel, snr_db])

            az_rad = math.radians(det_az)
            el_rad = math.radians(det_el)
            cos_el = math.cos(el_rad)
            points_xyz.append(
                [
                    det_range * cos_el * math.cos(az_rad),
                    det_range * cos_el * math.sin(az_rad),
                    det_range * math.sin(el_rad),
                ]
            )

        n_false = int(self._rng.poisson(self.false_alarm_rate)) if self.false_alarm_rate > 0.0 else 0
        for _ in range(n_false):
            det_range = float(self._rng.uniform(self.min_range_m, self.max_range_m))
            det_az = float(self._rng.uniform(-self.azimuth_fov_deg / 2.0, self.azimuth_fov_deg / 2.0))
            det_el = float(self._rng.uniform(-self.elevation_fov_deg / 2.0, self.elevation_fov_deg / 2.0))
            det_vel = float(self._rng.normal(0.0, max(self.velocity_noise_sigma_ms, 0.2)))
            snr_db = float(self._rng.uniform(2.0, 8.0))
            detections.append([det_range, det_az, det_el, det_vel, snr_db])

            az_rad = math.radians(det_az)
            el_rad = math.radians(det_el)
            cos_el = math.cos(el_rad)
            points_xyz.append(
                [
                    det_range * cos_el * math.cos(az_rad),
                    det_range * cos_el * math.sin(az_rad),
                    det_range * math.sin(el_rad),
                ]
            )

        if detections:
            det_arr = np.asarray(detections, dtype=np.float32)
            order = np.argsort(det_arr[:, 0])
            det_arr = det_arr[order]
            points_arr = np.asarray(points_xyz, dtype=np.float32)[order]
        else:
            det_arr = np.empty((0, 5), dtype=np.float32)
            points_arr = np.empty((0, 3), dtype=np.float32)

        obs: RadarObservation = {
            "detections": det_arr,
            "points_xyz": points_arr,
            "n_detections": int(det_arr.shape[0]),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> RadarObservation | SensorObservation:
        return self._last_obs


__all__ = ["RadarModel", "UWBRangingModel"]
