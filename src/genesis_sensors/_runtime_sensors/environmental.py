"""Environmental and situational awareness sensor models.

These sensors complement the existing navigation/perception stack with more
field-robotics style telemetry: wind, ambient temperature, humidity,
illuminance, and gas concentration.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, Final, cast

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation
from .types import (
    AnemometerObservation,
    GasObservation,
    HygrometerObservation,
    LightSensorObservation,
    ThermometerObservation,
)

if TYPE_CHECKING:
    from .config import AnemometerConfig, GasSensorConfig, HygrometerConfig, LightSensorConfig, ThermometerConfig

_MIN_TAU_S: Final[float] = 1e-3


def _as_vec3(value: Any, default: tuple[float, float, float] = (0.0, 0.0, 0.0)) -> np.ndarray:
    arr = np.asarray(value if value is not None else default, dtype=np.float64).reshape(-1)
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
        weather_map = cast(Mapping[str, Any], weather)
        return float(weather_map.get(key, default))
    return float(default)


def _dew_point_c(temp_c: float, rh_pct: float) -> float:
    """Approximate dew point using the Magnus formula."""
    rh = float(np.clip(rh_pct, 1e-3, 100.0)) / 100.0
    a = 17.62
    b = 243.12
    gamma = math.log(rh) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


class ThermometerModel(BaseSensor[ThermometerObservation]):
    """Ambient temperature sensor model with bias, noise, and thermal lag."""

    def __init__(
        self,
        name: str = "thermometer",
        update_rate_hz: float = 4.0,
        noise_sigma_c: float = 0.08,
        bias_tau_s: float = 300.0,
        bias_sigma_c: float = 0.2,
        response_tau_s: float = 2.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_c = float(max(0.0, noise_sigma_c))
        self.bias_tau_s = float(max(bias_tau_s, _MIN_TAU_S))
        self.bias_sigma_c = float(max(0.0, bias_sigma_c))
        self.response_tau_s = float(max(response_tau_s, _MIN_TAU_S))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._dt = 1.0 / self.update_rate_hz
        self._alpha_bias = math.exp(-self._dt / self.bias_tau_s)
        self._drive_sigma = self.bias_sigma_c * math.sqrt(1.0 - self._alpha_bias**2)
        self._response_alpha = min(1.0, self._dt / self.response_tau_s)
        self._bias_c = float(self._rng.normal(0.0, self.bias_sigma_c))
        self._temp_state_c: float | None = None
        self._last_obs: ThermometerObservation | SensorObservation = {}

    @classmethod
    def from_config(cls, config: "ThermometerConfig") -> "ThermometerModel":
        return cls(**config.model_dump())

    def get_config(self) -> "ThermometerConfig":
        from .config import ThermometerConfig

        return ThermometerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_c=self.noise_sigma_c,
            bias_tau_s=self.bias_tau_s,
            bias_sigma_c=self.bias_sigma_c,
            response_tau_s=self.response_tau_s,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._bias_c = float(self._rng.normal(0.0, self.bias_sigma_c))
        self._temp_state_c = None
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> ThermometerObservation:
        altitude_m = float(max(0.0, _as_vec3(state.get("pos"))[2]))
        baseline_c = _weather_value(state, "ambient_temp_c", 21.0)
        sun_load_c = _weather_value(state, "sun_load_c", 0.0)
        true_temp_c = baseline_c - 0.0065 * altitude_m + sun_load_c

        if self._temp_state_c is None:
            self._temp_state_c = true_temp_c
        else:
            self._temp_state_c += self._response_alpha * (true_temp_c - self._temp_state_c)

        self._bias_c = self._alpha_bias * self._bias_c + float(self._rng.normal(0.0, self._drive_sigma))
        measured_c = self._temp_state_c + self._bias_c + float(self._rng.normal(0.0, self.noise_sigma_c))
        obs: ThermometerObservation = {
            "temperature_c": float(measured_c),
            "temperature_f": float(measured_c * 9.0 / 5.0 + 32.0),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> ThermometerObservation | SensorObservation:
        return self._last_obs


class HygrometerModel(BaseSensor[HygrometerObservation]):
    """Relative humidity sensor with lag and dew-point estimation."""

    def __init__(
        self,
        name: str = "hygrometer",
        update_rate_hz: float = 2.0,
        noise_sigma_pct: float = 1.5,
        bias_tau_s: float = 400.0,
        bias_sigma_pct: float = 3.0,
        response_tau_s: float = 4.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_pct = float(max(0.0, noise_sigma_pct))
        self.bias_tau_s = float(max(bias_tau_s, _MIN_TAU_S))
        self.bias_sigma_pct = float(max(0.0, bias_sigma_pct))
        self.response_tau_s = float(max(response_tau_s, _MIN_TAU_S))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._dt = 1.0 / self.update_rate_hz
        self._alpha_bias = math.exp(-self._dt / self.bias_tau_s)
        self._drive_sigma = self.bias_sigma_pct * math.sqrt(1.0 - self._alpha_bias**2)
        self._response_alpha = min(1.0, self._dt / self.response_tau_s)
        self._bias_pct = float(self._rng.normal(0.0, self.bias_sigma_pct))
        self._rh_state_pct: float | None = None
        self._last_obs: HygrometerObservation | SensorObservation = {}

    @classmethod
    def from_config(cls, config: "HygrometerConfig") -> "HygrometerModel":
        return cls(**config.model_dump())

    def get_config(self) -> "HygrometerConfig":
        from .config import HygrometerConfig

        return HygrometerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_pct=self.noise_sigma_pct,
            bias_tau_s=self.bias_tau_s,
            bias_sigma_pct=self.bias_sigma_pct,
            response_tau_s=self.response_tau_s,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._bias_pct = float(self._rng.normal(0.0, self.bias_sigma_pct))
        self._rh_state_pct = None
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> HygrometerObservation:
        rain_rate = _weather_value(state, "rain_rate_mm_h", 0.0)
        cloud_cover = _weather_value(state, "cloud_cover", 0.2)
        obstruction = float(np.clip(state.get("obstruction", 0.0), 0.0, 1.0))
        baseline_rh = _weather_value(
            state,
            "relative_humidity_pct",
            42.0 + 25.0 * math.tanh(rain_rate / 4.0) + 10.0 * cloud_cover + 8.0 * obstruction,
        )
        true_rh = float(np.clip(baseline_rh, 0.0, 100.0))

        if self._rh_state_pct is None:
            self._rh_state_pct = true_rh
        else:
            self._rh_state_pct += self._response_alpha * (true_rh - self._rh_state_pct)

        self._bias_pct = self._alpha_bias * self._bias_pct + float(self._rng.normal(0.0, self._drive_sigma))
        measured_rh = float(
            np.clip(self._rh_state_pct + self._bias_pct + self._rng.normal(0.0, self.noise_sigma_pct), 0.0, 100.0)
        )
        temp_c = _weather_value(state, "ambient_temp_c", 21.0)
        obs: HygrometerObservation = {
            "relative_humidity_pct": measured_rh,
            "dew_point_c": float(_dew_point_c(temp_c, measured_rh)),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> HygrometerObservation | SensorObservation:
        return self._last_obs


class LightSensorModel(BaseSensor[LightSensorObservation]):
    """Ambient illuminance sensor model (lux) with saturation behavior."""

    def __init__(
        self,
        name: str = "light_sensor",
        update_rate_hz: float = 10.0,
        noise_sigma_ratio: float = 0.04,
        min_lux: float = 0.0,
        max_lux: float = 120_000.0,
        response_tau_s: float = 0.4,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_ratio = float(max(0.0, noise_sigma_ratio))
        self.min_lux = float(max(0.0, min_lux))
        self.max_lux = float(max(max_lux, self.min_lux + 1.0))
        self.response_tau_s = float(max(response_tau_s, _MIN_TAU_S))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._dt = 1.0 / self.update_rate_hz
        self._response_alpha = min(1.0, self._dt / self.response_tau_s)
        self._lux_state: float | None = None
        self._last_obs: LightSensorObservation | SensorObservation = {}

    @classmethod
    def from_config(cls, config: "LightSensorConfig") -> "LightSensorModel":
        return cls(**config.model_dump())

    def get_config(self) -> "LightSensorConfig":
        from .config import LightSensorConfig

        return LightSensorConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_ratio=self.noise_sigma_ratio,
            min_lux=self.min_lux,
            max_lux=self.max_lux,
            response_tau_s=self.response_tau_s,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._lux_state = None
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> LightSensorObservation:
        cloud_cover = _weather_value(state, "cloud_cover", 0.25)
        obstruction = float(np.clip(state.get("obstruction", 0.0), 0.0, 1.0))
        baseline_lux = _weather_value(
            state,
            "illuminance_lux",
            35_000.0 * (1.0 - 0.75 * cloud_cover) * (1.0 - 0.6 * obstruction),
        )
        target_lux = float(np.clip(baseline_lux, self.min_lux, self.max_lux))

        if self._lux_state is None:
            self._lux_state = target_lux
        else:
            self._lux_state += self._response_alpha * (target_lux - self._lux_state)

        sigma = 0.0 if self.noise_sigma_ratio <= 0.0 else max(1.0, abs(self._lux_state) * self.noise_sigma_ratio)
        measured = float(np.clip(self._lux_state + self._rng.normal(0.0, sigma), self.min_lux, self.max_lux))
        obs: LightSensorObservation = {
            "illuminance_lux": measured,
            "is_saturated": bool(measured >= self.max_lux - 1e-9),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> LightSensorObservation | SensorObservation:
        return self._last_obs


class GasSensorModel(BaseSensor[GasObservation]):
    """Generic gas concentration sensor with a simple plume approximation."""

    def __init__(
        self,
        name: str = "gas_sensor",
        update_rate_hz: float = 5.0,
        background_ppm: float = 420.0,
        noise_sigma_ppm: float = 8.0,
        response_tau_s: float = 3.0,
        alarm_threshold_ppm: float = 900.0,
        max_ppm: float = 10_000.0,
        plume_sigma_m: float = 0.8,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.background_ppm = float(max(0.0, background_ppm))
        self.noise_sigma_ppm = float(max(0.0, noise_sigma_ppm))
        self.response_tau_s = float(max(response_tau_s, _MIN_TAU_S))
        self.alarm_threshold_ppm = float(max(0.0, alarm_threshold_ppm))
        self.max_ppm = float(max(max_ppm, self.background_ppm + 1.0))
        self.plume_sigma_m = float(max(plume_sigma_m, 0.1))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._dt = 1.0 / self.update_rate_hz
        self._response_alpha = min(1.0, self._dt / self.response_tau_s)
        self._ppm_state: float | None = None
        self._last_obs: GasObservation | SensorObservation = {}

    @classmethod
    def from_config(cls, config: "GasSensorConfig") -> "GasSensorModel":
        return cls(**config.model_dump())

    def get_config(self) -> "GasSensorConfig":
        from .config import GasSensorConfig

        return GasSensorConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            background_ppm=self.background_ppm,
            noise_sigma_ppm=self.noise_sigma_ppm,
            response_tau_s=self.response_tau_s,
            alarm_threshold_ppm=self.alarm_threshold_ppm,
            max_ppm=self.max_ppm,
            plume_sigma_m=self.plume_sigma_m,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._ppm_state = None
        self._last_obs = {}
        self._last_update_time = -1.0

    def _compute_true_ppm(self, state: SensorInput) -> float:
        if "gas_ppm" in state:
            return float(np.clip(state["gas_ppm"], 0.0, self.max_ppm))

        concentration = float(np.clip(state.get("gas_background_ppm", self.background_ppm), 0.0, self.max_ppm))
        pos = _as_vec3(state.get("pos"))
        wind = _as_vec3(state.get("wind_ms", state.get("wind", (1.0, 0.0, 0.0))))
        wind_xy = wind[:2]
        wind_norm = float(np.linalg.norm(wind_xy))
        if wind_norm < 1e-6:
            wind_dir = np.array([1.0, 0.0], dtype=np.float64)
        else:
            wind_dir = wind_xy / wind_norm

        raw_sources = state.get("gas_sources")
        if raw_sources is None and "gas_source_pos" in state:
            raw_sources = [
                {
                    "pos": state["gas_source_pos"],
                    "peak_ppm": state.get("gas_source_strength_ppm", 1200.0),
                    "sigma_m": state.get("gas_source_sigma_m", self.plume_sigma_m),
                }
            ]
        if not raw_sources:
            return concentration

        for source in raw_sources:
            if not isinstance(source, Mapping):
                continue
            source_map = cast(Mapping[str, Any], source)
            src_pos = _as_vec3(source_map.get("pos"))
            peak_ppm = float(source_map.get("peak_ppm", source_map.get("strength_ppm", 0.0)) or 0.0)
            peak_ppm = max(0.0, peak_ppm)
            sigma_m = float(source_map.get("sigma_m", self.plume_sigma_m) or self.plume_sigma_m)
            sigma_m = max(0.1, sigma_m)
            delta = pos - src_pos
            along = float(np.dot(delta[:2], wind_dir))
            cross = float(np.linalg.norm(delta[:2] - along * wind_dir))
            vertical = abs(float(delta[2]))
            if along >= 0.0:
                plume = peak_ppm * math.exp(-0.5 * (cross / sigma_m) ** 2 - 0.5 * (vertical / sigma_m) ** 2)
                plume /= 1.0 + along / max(sigma_m, 1e-6)
            else:
                plume = 0.15 * peak_ppm * math.exp(-0.5 * (np.linalg.norm(delta) / (1.5 * sigma_m)) ** 2)
            concentration += plume
        return float(np.clip(concentration, 0.0, self.max_ppm))

    def step(self, sim_time: float, state: SensorInput) -> GasObservation:
        target_ppm = self._compute_true_ppm(state)
        if self._ppm_state is None:
            self._ppm_state = target_ppm
        else:
            self._ppm_state += self._response_alpha * (target_ppm - self._ppm_state)

        measured_ppm = float(np.clip(self._ppm_state + self._rng.normal(0.0, self.noise_sigma_ppm), 0.0, self.max_ppm))
        obs: GasObservation = {
            "concentration_ppm": measured_ppm,
            "alarm": bool(measured_ppm >= self.alarm_threshold_ppm),
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> GasObservation | SensorObservation:
        return self._last_obs


class AnemometerModel(BaseSensor[AnemometerObservation]):
    """Wind-speed and direction sensor model."""

    def __init__(
        self,
        name: str = "anemometer",
        update_rate_hz: float = 10.0,
        noise_sigma_ms: float = 0.15,
        direction_noise_deg: float = 2.0,
        measure_relative_wind: bool = False,
        max_speed_ms: float = 80.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_sigma_ms = float(max(0.0, noise_sigma_ms))
        self.direction_noise_deg = float(max(0.0, direction_noise_deg))
        self.measure_relative_wind = bool(measure_relative_wind)
        self.max_speed_ms = float(max(max_speed_ms, 0.1))
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: AnemometerObservation | SensorObservation = {}

    @classmethod
    def from_config(cls, config: "AnemometerConfig") -> "AnemometerModel":
        return cls(**config.model_dump())

    def get_config(self) -> "AnemometerConfig":
        from .config import AnemometerConfig

        return AnemometerConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_sigma_ms=self.noise_sigma_ms,
            direction_noise_deg=self.direction_noise_deg,
            measure_relative_wind=self.measure_relative_wind,
            max_speed_ms=self.max_speed_ms,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: SensorInput) -> AnemometerObservation:
        wind = state.get("wind_ms", state.get("wind"))
        if wind is None:
            speed = _weather_value(state, "wind_speed_ms", 0.0)
            direction_deg = _weather_value(state, "wind_direction_deg", 0.0)
            theta = math.radians(direction_deg)
            wind_vec = np.array([speed * math.cos(theta), speed * math.sin(theta), 0.0], dtype=np.float64)
        else:
            wind_vec = _as_vec3(wind)

        if self.measure_relative_wind and "vel" in state:
            wind_vec = wind_vec - _as_vec3(state.get("vel"))

        noisy_vec = wind_vec + self._rng.normal(0.0, self.noise_sigma_ms, 3)
        speed_ms = float(np.clip(np.linalg.norm(noisy_vec), 0.0, self.max_speed_ms))
        if speed_ms < 1e-9:
            direction_deg = 0.0
            noisy_vec = np.zeros(3, dtype=np.float64)
        else:
            direction_deg = float((math.degrees(math.atan2(noisy_vec[1], noisy_vec[0])) + 360.0) % 360.0)
            direction_deg = float((direction_deg + self._rng.normal(0.0, self.direction_noise_deg)) % 360.0)
            noisy_norm = float(np.linalg.norm(noisy_vec))
            scale = speed_ms / max(noisy_norm, 1e-9)
            noisy_vec = noisy_vec * scale

        obs: AnemometerObservation = {
            "wind_vector_ms": noisy_vec.astype(np.float64),
            "wind_speed_ms": speed_ms,
            "wind_direction_deg": direction_deg,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> AnemometerObservation | SensorObservation:
        return self._last_obs


__all__ = [
    "AnemometerModel",
    "GasSensorModel",
    "HygrometerModel",
    "LightSensorModel",
    "ThermometerModel",
]
