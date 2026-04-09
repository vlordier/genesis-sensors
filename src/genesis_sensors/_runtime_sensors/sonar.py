"""Imaging and side-scan sonar sensor models.

These models provide underwater / acoustic perception beyond simple ultrasonic
proximity sensing:

* ``ImagingSonarModel`` renders a forward-looking acoustic intensity image.
* ``SideScanSonarModel`` produces port/starboard waterfall-style returns.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation
from .types import Float64Array, ImagingSonarObservation, SideScanSonarObservation

if TYPE_CHECKING:
    from .config import ImagingSonarConfig, SideScanSonarConfig


def _as_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> Float64Array:
    arr = np.asarray(default if value is None else value, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return np.asarray(default, dtype=np.float64)
    if arr.size == 1:
        return np.array([float(arr[0]), 0.0, 0.0], dtype=np.float64)
    if arr.size == 2:
        return np.array([float(arr[0]), float(arr[1]), 0.0], dtype=np.float64)
    return np.array([float(arr[0]), float(arr[1]), float(arr[2])], dtype=np.float64)


def _iter_sonar_targets(raw: Any) -> list[dict[str, Any]]:
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
                    "strength": float(item_map.get("strength", item_map.get("reflectivity", 1.0))),
                    "extent_deg": float(item_map.get("extent_deg", 3.0)),
                }
            )
        else:
            targets.append({"id": f"target_{idx}", "pos": _as_vec3(item), "strength": 1.0, "extent_deg": 3.0})
    return targets


def _deposit_blob(
    image: np.ndarray, row: float, col: float, amplitude: float, row_sigma: float, col_sigma: float
) -> None:
    row_idx = np.arange(image.shape[0], dtype=np.float32)[:, None]
    col_idx = np.arange(image.shape[1], dtype=np.float32)[None, :]
    blob = np.exp(
        -0.5 * (((row_idx - row) / max(row_sigma, 1e-3)) ** 2 + ((col_idx - col) / max(col_sigma, 1e-3)) ** 2)
    )
    image += amplitude * blob.astype(np.float32)


def _deposit_line(signal: np.ndarray, idx: float, amplitude: float, sigma: float) -> None:
    bins = np.arange(signal.shape[0], dtype=np.float32)
    blob = np.exp(-0.5 * (((bins - idx) / max(sigma, 1e-3)) ** 2))
    signal += amplitude * blob.astype(np.float32)


class ImagingSonarModel(BaseSensor[ImagingSonarObservation]):
    """Forward-looking imaging sonar that renders a polar intensity image."""

    def __init__(
        self,
        name: str = "imaging_sonar",
        update_rate_hz: float = 8.0,
        azimuth_bins: int = 96,
        range_bins: int = 128,
        azimuth_fov_deg: float = 120.0,
        min_range_m: float = 0.5,
        max_range_m: float = 30.0,
        range_noise_sigma_m: float = 0.05,
        azimuth_noise_deg: float = 1.0,
        speckle_sigma: float = 0.04,
        attenuation_db_per_m: float = 0.12,
        false_alarm_rate: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if azimuth_bins <= 0 or range_bins <= 0:
            raise ValueError("azimuth_bins and range_bins must be positive")
        if min_range_m >= max_range_m:
            raise ValueError(f"min_range_m ({min_range_m}) must be less than max_range_m ({max_range_m})")

        self.azimuth_bins = int(azimuth_bins)
        self.range_bins = int(range_bins)
        self.azimuth_fov_deg = float(azimuth_fov_deg)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.range_noise_sigma_m = float(max(0.0, range_noise_sigma_m))
        self.azimuth_noise_deg = float(max(0.0, azimuth_noise_deg))
        self.speckle_sigma = float(max(0.0, speckle_sigma))
        self.attenuation_db_per_m = float(max(0.0, attenuation_db_per_m))
        self.false_alarm_rate = float(max(0.0, false_alarm_rate))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: ImagingSonarObservation | SensorObservation = {
            "intensity_image": np.zeros((self.range_bins, self.azimuth_bins), dtype=np.float32),
            "range_axis_m": np.linspace(self.min_range_m, self.max_range_m, self.range_bins, dtype=np.float32),
            "azimuth_axis_deg": np.linspace(
                -self.azimuth_fov_deg / 2.0,
                self.azimuth_fov_deg / 2.0,
                self.azimuth_bins,
                dtype=np.float32,
            ),
            "n_returns": 0,
            "strongest_return_m": 0.0,
        }

    @classmethod
    def from_config(cls, config: "ImagingSonarConfig") -> "ImagingSonarModel":
        return cls(**config.model_dump())

    def get_config(self) -> "ImagingSonarConfig":
        from .config import ImagingSonarConfig

        return ImagingSonarConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            azimuth_bins=self.azimuth_bins,
            range_bins=self.range_bins,
            azimuth_fov_deg=self.azimuth_fov_deg,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            range_noise_sigma_m=self.range_noise_sigma_m,
            azimuth_noise_deg=self.azimuth_noise_deg,
            speckle_sigma=self.speckle_sigma,
            attenuation_db_per_m=self.attenuation_db_per_m,
            false_alarm_rate=self.false_alarm_rate,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_update_time = -1.0
        self._last_obs = {
            "intensity_image": np.zeros((self.range_bins, self.azimuth_bins), dtype=np.float32),
            "range_axis_m": np.linspace(self.min_range_m, self.max_range_m, self.range_bins, dtype=np.float32),
            "azimuth_axis_deg": np.linspace(
                -self.azimuth_fov_deg / 2.0,
                self.azimuth_fov_deg / 2.0,
                self.azimuth_bins,
                dtype=np.float32,
            ),
            "n_returns": 0,
            "strongest_return_m": 0.0,
        }

    def step(self, sim_time: float, state: SensorInput) -> ImagingSonarObservation:
        pos = _as_vec3(state.get("pos"))
        targets = _iter_sonar_targets(state.get("sonar_targets", ()))
        turbidity = float(state.get("water_turbidity_ntu", 0.0))

        image = np.zeros((self.range_bins, self.azimuth_bins), dtype=np.float32)
        strongest_amp = 0.0
        strongest_range = 0.0
        n_returns = 0

        for target in targets:
            rel = cast(Float64Array, target["pos"]) - pos
            range_true = float(np.linalg.norm(rel))
            if not (self.min_range_m <= range_true <= self.max_range_m):
                continue

            azimuth_deg = float(math.degrees(math.atan2(rel[1], rel[0])))
            if abs(azimuth_deg) > self.azimuth_fov_deg / 2.0:
                continue

            n_returns += 1
            det_range = range_true + float(self._rng.normal(0.0, self.range_noise_sigma_m))
            det_az = azimuth_deg + float(self._rng.normal(0.0, self.azimuth_noise_deg))
            det_range = float(np.clip(det_range, self.min_range_m, self.max_range_m))
            det_az = float(np.clip(det_az, -self.azimuth_fov_deg / 2.0, self.azimuth_fov_deg / 2.0))

            row = (
                (det_range - self.min_range_m) / max(self.max_range_m - self.min_range_m, 1e-6) * (self.range_bins - 1)
            )
            col = (det_az + self.azimuth_fov_deg / 2.0) / max(self.azimuth_fov_deg, 1e-6) * (self.azimuth_bins - 1)

            attenuation = math.exp(-(0.01 * self.attenuation_db_per_m + 0.002 * turbidity) * det_range)
            amplitude = float(np.clip(float(target["strength"]) * attenuation, 0.0, 1.0))
            row_sigma = 1.0 + 0.02 * range_true
            col_sigma = max(1.0, float(target["extent_deg"]) / max(self.azimuth_fov_deg, 1e-6) * self.azimuth_bins)
            _deposit_blob(image, row=row, col=col, amplitude=amplitude, row_sigma=row_sigma, col_sigma=col_sigma)

            if amplitude > strongest_amp:
                strongest_amp = amplitude
                strongest_range = det_range

        if self.false_alarm_rate > 0.0:
            for _ in range(int(self._rng.poisson(self.false_alarm_rate))):
                row = float(self._rng.uniform(0, self.range_bins - 1))
                col = float(self._rng.uniform(0, self.azimuth_bins - 1))
                _deposit_blob(
                    image,
                    row=row,
                    col=col,
                    amplitude=float(self._rng.uniform(0.02, 0.12)),
                    row_sigma=1.0,
                    col_sigma=1.0,
                )

        if self.speckle_sigma > 0.0:
            image += np.abs(self._rng.normal(0.0, self.speckle_sigma, size=image.shape)).astype(np.float32)
        image = np.clip(image, 0.0, 1.0)

        obs: ImagingSonarObservation = {
            "intensity_image": image,
            "range_axis_m": np.linspace(self.min_range_m, self.max_range_m, self.range_bins, dtype=np.float32),
            "azimuth_axis_deg": np.linspace(
                -self.azimuth_fov_deg / 2.0,
                self.azimuth_fov_deg / 2.0,
                self.azimuth_bins,
                dtype=np.float32,
            ),
            "n_returns": int(n_returns),
            "strongest_return_m": float(strongest_range),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> ImagingSonarObservation | SensorObservation:
        return self._last_obs


class SideScanSonarModel(BaseSensor[SideScanSonarObservation]):
    """Port/starboard side-scan sonar for seafloor and obstacle strip imaging."""

    def __init__(
        self,
        name: str = "side_scan",
        update_rate_hz: float = 4.0,
        range_bins: int = 128,
        min_range_m: float = 0.5,
        max_range_m: float = 40.0,
        range_noise_sigma_m: float = 0.06,
        speckle_sigma: float = 0.04,
        attenuation_db_per_m: float = 0.10,
        false_alarm_rate: float = 0.02,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if range_bins <= 0:
            raise ValueError("range_bins must be positive")
        if min_range_m >= max_range_m:
            raise ValueError(f"min_range_m ({min_range_m}) must be less than max_range_m ({max_range_m})")

        self.range_bins = int(range_bins)
        self.min_range_m = float(min_range_m)
        self.max_range_m = float(max_range_m)
        self.range_noise_sigma_m = float(max(0.0, range_noise_sigma_m))
        self.speckle_sigma = float(max(0.0, speckle_sigma))
        self.attenuation_db_per_m = float(max(0.0, attenuation_db_per_m))
        self.false_alarm_rate = float(max(0.0, false_alarm_rate))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: SideScanSonarObservation | SensorObservation = {
            "port_intensity": np.zeros(self.range_bins, dtype=np.float32),
            "starboard_intensity": np.zeros(self.range_bins, dtype=np.float32),
            "slant_range_axis_m": np.linspace(self.min_range_m, self.max_range_m, self.range_bins, dtype=np.float32),
            "port_hits": 0,
            "starboard_hits": 0,
        }

    @classmethod
    def from_config(cls, config: "SideScanSonarConfig") -> "SideScanSonarModel":
        return cls(**config.model_dump())

    def get_config(self) -> "SideScanSonarConfig":
        from .config import SideScanSonarConfig

        return SideScanSonarConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            range_bins=self.range_bins,
            min_range_m=self.min_range_m,
            max_range_m=self.max_range_m,
            range_noise_sigma_m=self.range_noise_sigma_m,
            speckle_sigma=self.speckle_sigma,
            attenuation_db_per_m=self.attenuation_db_per_m,
            false_alarm_rate=self.false_alarm_rate,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_update_time = -1.0
        self._last_obs = {
            "port_intensity": np.zeros(self.range_bins, dtype=np.float32),
            "starboard_intensity": np.zeros(self.range_bins, dtype=np.float32),
            "slant_range_axis_m": np.linspace(self.min_range_m, self.max_range_m, self.range_bins, dtype=np.float32),
            "port_hits": 0,
            "starboard_hits": 0,
        }

    def step(self, sim_time: float, state: SensorInput) -> SideScanSonarObservation:
        pos = _as_vec3(state.get("pos"))
        targets = _iter_sonar_targets(state.get("sonar_targets", ()))
        turbidity = float(state.get("water_turbidity_ntu", 0.0))

        port = np.zeros(self.range_bins, dtype=np.float32)
        starboard = np.zeros(self.range_bins, dtype=np.float32)
        port_hits = 0
        starboard_hits = 0

        for target in targets:
            rel = cast(Float64Array, target["pos"]) - pos
            lateral_range = float(np.linalg.norm(rel[1:]))
            if not (self.min_range_m <= lateral_range <= self.max_range_m):
                continue

            det_range = lateral_range + float(self._rng.normal(0.0, self.range_noise_sigma_m))
            det_range = float(np.clip(det_range, self.min_range_m, self.max_range_m))
            bin_idx = (
                (det_range - self.min_range_m) / max(self.max_range_m - self.min_range_m, 1e-6) * (self.range_bins - 1)
            )
            amplitude = float(
                np.clip(
                    float(target["strength"])
                    * math.exp(-(0.01 * self.attenuation_db_per_m + 0.002 * turbidity) * det_range),
                    0.0,
                    1.0,
                )
            )

            if rel[1] >= 0.0:
                _deposit_line(port, idx=bin_idx, amplitude=amplitude, sigma=1.5)
                port_hits += 1
            else:
                _deposit_line(starboard, idx=bin_idx, amplitude=amplitude, sigma=1.5)
                starboard_hits += 1

        if self.false_alarm_rate > 0.0:
            for _ in range(int(self._rng.poisson(self.false_alarm_rate))):
                signal = port if self._rng.random() < 0.5 else starboard
                _deposit_line(
                    signal,
                    idx=float(self._rng.uniform(0, self.range_bins - 1)),
                    amplitude=float(self._rng.uniform(0.01, 0.08)),
                    sigma=1.0,
                )

        if self.speckle_sigma > 0.0:
            port += np.abs(self._rng.normal(0.0, self.speckle_sigma, size=port.shape)).astype(np.float32)
            starboard += np.abs(self._rng.normal(0.0, self.speckle_sigma, size=starboard.shape)).astype(np.float32)
        port = np.clip(port, 0.0, 1.0)
        starboard = np.clip(starboard, 0.0, 1.0)

        obs: SideScanSonarObservation = {
            "port_intensity": port,
            "starboard_intensity": starboard,
            "slant_range_axis_m": np.linspace(self.min_range_m, self.max_range_m, self.range_bins, dtype=np.float32),
            "port_hits": int(port_hits),
            "starboard_hits": int(starboard_hits),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> SideScanSonarObservation | SensorObservation:
        return self._last_obs


__all__ = ["ImagingSonarModel", "SideScanSonarModel"]
