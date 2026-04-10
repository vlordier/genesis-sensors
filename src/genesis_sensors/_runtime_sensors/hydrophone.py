"""Hydrophone / passive sonar sensor model.

Simulates a single-element or array hydrophone for passive acoustic
detection of sound sources in water.  Commonly used on AUV, ROV, and USV
for target detection, localization, and bio-acoustic monitoring.

State keys consumed
-------------------
``"acoustic_sources"``
    List of dicts, each with ``"pos"`` (3-vec, world m), ``"frequency_hz"``,
    ``"source_level_db"`` (re 1 µPa @ 1 m).
``"pos"``
    Sensor world position (3-vec, m).
``"water_temperature_c"``
    Water temperature (°C).  Used for speed-of-sound estimation.
``"water_salinity_ppt"``
    Water salinity (‰).  Used for speed-of-sound and absorption.
``"depth_m"``
    Current depth (m).  Used for speed-of-sound.
``"ambient_noise_db"``
    Ambient noise spectral level (dB re 1 µPa).  Default: 60 dB (sea state 3).
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import HydrophoneConfig


def _speed_of_sound(temp_c: float, salinity_ppt: float, depth_m: float) -> float:
    """Mackenzie (1981) equation for speed of sound in seawater (m/s)."""
    t = temp_c
    s = salinity_ppt
    d = depth_m
    c = (
        1448.96
        + 4.591 * t
        - 5.304e-2 * t**2
        + 2.374e-4 * t**3
        + 1.340 * (s - 35.0)
        + 1.630e-2 * d
        + 1.675e-7 * d**2
        - 1.025e-2 * t * (s - 35.0)
        - 7.139e-13 * t * d**3
    )
    return float(c)


def _absorption_db_per_m(frequency_hz: float, temp_c: float, salinity_ppt: float, depth_m: float) -> float:
    """Francois-Garrison absorption model (simplified, dB/m)."""
    f_khz = frequency_hz / 1000.0
    # Very simplified — dominant relaxation terms
    alpha = 0.106 * (f_khz**2) / (1.0 + f_khz**2) * math.exp((temp_c - 18.0) / 26.0)
    alpha += 0.003 * f_khz**2  # high-frequency viscous absorption
    return float(alpha / 1000.0)  # convert dB/km to dB/m


class HydrophoneModel(BaseSensor):
    """Passive acoustic hydrophone sensor.

    Detects acoustic sources in water by computing received level (RL)
    from the passive sonar equation:  ``RL = SL - TL``, where TL includes
    geometric spreading and frequency-dependent absorption.  Detection
    occurs when ``RL - NL > detection_threshold_db``.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    sensitivity_db:
        Hydrophone sensitivity (dB re 1 V/µPa).  Typical: −170 to −200.
    frequency_range_hz:
        Tuple (min_hz, max_hz) passband.
    detection_threshold_db:
        Minimum SNR for detection (dB).
    noise_floor_db:
        Self-noise floor (dB re 1 µPa).
    directivity_index_db:
        Array/element directivity index (dB).  0 = omnidirectional.
    max_range_m:
        Maximum detection range (m).  Sources beyond are ignored.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "hydrophone",
        update_rate_hz: float = 10.0,
        sensitivity_db: float = -180.0,
        frequency_range_hz: tuple[float, float] = (100.0, 100_000.0),
        detection_threshold_db: float = 10.0,
        noise_floor_db: float = 30.0,
        directivity_index_db: float = 0.0,
        max_range_m: float = 5000.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.sensitivity_db = float(sensitivity_db)
        self.frequency_range_hz = (float(frequency_range_hz[0]), float(frequency_range_hz[1]))
        self.detection_threshold_db = float(detection_threshold_db)
        self.noise_floor_db = float(noise_floor_db)
        self.directivity_index_db = float(directivity_index_db)
        self.max_range_m = float(max_range_m)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "HydrophoneConfig") -> "HydrophoneModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "HydrophoneConfig":
        from .config import HydrophoneConfig

        return HydrophoneConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            sensitivity_db=self.sensitivity_db,
            frequency_range_hz=self.frequency_range_hz,
            detection_threshold_db=self.detection_threshold_db,
            noise_floor_db=self.noise_floor_db,
            directivity_index_db=self.directivity_index_db,
            max_range_m=self.max_range_m,
            seed=self._seed,
        )

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        sensor_pos = np.asarray(state.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)[:3]
        temp_c = float(state.get("water_temperature_c", state.get("temperature_c", 15.0)))
        salinity = float(state.get("water_salinity_ppt", 35.0))
        depth = float(state.get("depth_m", max(0.0, -float(sensor_pos[2]))))
        ambient_noise_db = float(state.get("ambient_noise_db", 60.0))

        # Effective noise level
        nl = max(ambient_noise_db, self.noise_floor_db) - self.directivity_index_db

        sources = state.get("acoustic_sources", [])
        if not isinstance(sources, Sequence) or isinstance(sources, (str, bytes)):
            sources = []

        detections: list[dict[str, Any]] = []

        for src in cast(Sequence[Any], sources):
            if not hasattr(src, "get"):
                continue
            src_pos = np.asarray(src.get("pos", [0, 0, 0]), dtype=np.float64)[:3]
            src_freq = float(src.get("frequency_hz", 1000.0))
            src_level_db = float(src.get("source_level_db", 120.0))

            # Check passband
            if src_freq < self.frequency_range_hz[0] or src_freq > self.frequency_range_hz[1]:
                continue

            # Range and transmission loss
            diff = src_pos - sensor_pos
            rng = float(np.linalg.norm(diff))
            if rng < 0.1 or rng > self.max_range_m:
                continue

            # Spherical spreading + absorption
            alpha = _absorption_db_per_m(src_freq, temp_c, salinity, depth)
            tl = 20.0 * math.log10(rng) + alpha * rng

            rl = src_level_db - tl

            # Add noise to RL estimate
            rl_noisy = rl + float(self._rng.normal(0.0, 1.5))
            snr_noisy = rl_noisy - nl

            if snr_noisy >= self.detection_threshold_db:
                # Bearing estimation (azimuth, elevation)
                bearing_az = math.degrees(math.atan2(diff[1], diff[0]))
                bearing_el = math.degrees(math.atan2(-diff[2], math.sqrt(diff[0] ** 2 + diff[1] ** 2)))

                detections.append(
                    {
                        "bearing_azimuth_deg": bearing_az + float(self._rng.normal(0.0, 3.0)),
                        "bearing_elevation_deg": bearing_el + float(self._rng.normal(0.0, 5.0)),
                        "received_level_db": rl_noisy,
                        "snr_db": snr_noisy,
                        "range_estimate_m": rng + float(self._rng.normal(0.0, rng * 0.1)),
                        "frequency_hz": src_freq,
                    }
                )

        self._last_obs = {
            "detections": detections,
            "n_detections": len(detections),
            "ambient_noise_db": ambient_noise_db,
            "speed_of_sound_ms": _speed_of_sound(temp_c, salinity, depth),
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0
