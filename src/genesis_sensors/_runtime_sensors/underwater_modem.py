"""Underwater acoustic modem sensor model.

Models point-to-point underwater acoustic communication with range
and depth-dependent attenuation, multi-path fading, and Doppler shift.
Analogous to RadioLinkModel but for the underwater acoustic channel.

State keys consumed
-------------------
``"pos"``
    Sender world position (3-vec, m).
``"remote_pos"``
    Receiver/remote node position (3-vec, m).
``"depth_m"``
    Current depth (m, positive downward).
``"water_temperature_c"``
    Water temperature (°C).
``"water_salinity_ppt"``
    Water salinity (‰).
``"vel"``
    Platform velocity (3-vec, m/s).  For Doppler shift estimation.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import UnderwaterModemConfig


def _speed_of_sound(temp_c: float, salinity_ppt: float, depth_m: float) -> float:
    """Mackenzie (1981) equation for speed of sound in seawater (m/s)."""
    t = temp_c
    s = salinity_ppt
    d = depth_m
    return float(
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


class UnderwaterModemModel(BaseSensor):
    """Acoustic underwater communication modem.

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Modem poll rate (Hz).
    frequency_hz:
        Carrier frequency (Hz).  Typical: 10–30 kHz.
    bandwidth_hz:
        Communication bandwidth (Hz).
    source_level_db:
        Transmit source level (dB re 1 µPa @ 1 m).
    max_range_m:
        Maximum communication range (m).
    noise_spectral_db:
        Ambient noise spectral density (dB re 1 µPa/Hz).
    spreading_factor:
        Geometric spreading factor.  1.5 = practical, 2.0 = spherical.
    data_rate_bps:
        Nominal data rate (bits per second).
    packet_size_bits:
        Default packet size (bits).
    ber_threshold:
        Bit-error rate above which packets are dropped.
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "underwater_modem",
        update_rate_hz: float = 1.0,
        frequency_hz: float = 25_000.0,
        bandwidth_hz: float = 5000.0,
        source_level_db: float = 185.0,
        max_range_m: float = 3000.0,
        noise_spectral_db: float = 50.0,
        spreading_factor: float = 1.5,
        data_rate_bps: float = 9600.0,
        packet_size_bits: int = 256,
        ber_threshold: float = 1e-3,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.frequency_hz = float(max(100.0, frequency_hz))
        self.bandwidth_hz = float(max(1.0, bandwidth_hz))
        self.source_level_db = float(source_level_db)
        self.max_range_m = float(max(1.0, max_range_m))
        self.noise_spectral_db = float(noise_spectral_db)
        self.spreading_factor = float(max(1.0, spreading_factor))
        self.data_rate_bps = float(max(1.0, data_rate_bps))
        self.packet_size_bits = int(max(1, packet_size_bits))
        self.ber_threshold = float(ber_threshold)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "UnderwaterModemConfig") -> "UnderwaterModemModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "UnderwaterModemConfig":
        from .config import UnderwaterModemConfig

        return UnderwaterModemConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            frequency_hz=self.frequency_hz,
            bandwidth_hz=self.bandwidth_hz,
            source_level_db=self.source_level_db,
            max_range_m=self.max_range_m,
            noise_spectral_db=self.noise_spectral_db,
            spreading_factor=self.spreading_factor,
            data_rate_bps=self.data_rate_bps,
            packet_size_bits=self.packet_size_bits,
            ber_threshold=self.ber_threshold,
            seed=self._seed,
        )

    def _absorption_db_per_m(self, temp_c: float, salinity_ppt: float, depth_m: float) -> float:
        """Francois-Garrison simplified absorption (dB/m)."""
        f_khz = self.frequency_hz / 1000.0
        alpha = 0.106 * (f_khz**2) / (1.0 + f_khz**2) * math.exp((temp_c - 18.0) / 26.0)
        alpha += 0.003 * f_khz**2
        return float(alpha / 1000.0)  # dB/km → dB/m

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        pos = np.asarray(state.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)[:3]
        remote_pos = state.get("remote_pos")
        if remote_pos is None:
            self._last_obs = {
                "connected": False,
                "snr_db": -999.0,
                "range_m": 0.0,
                "packet_delivered": False,
                "latency_s": 0.0,
                "ber": 1.0,
                "doppler_shift_hz": 0.0,
            }
            return self._last_obs

        remote = np.asarray(remote_pos, dtype=np.float64)[:3]
        diff = remote - pos
        rng = float(np.linalg.norm(diff))

        temp_c = float(state.get("water_temperature_c", state.get("temperature_c", 15.0)))
        salinity = float(state.get("water_salinity_ppt", 35.0))
        depth = float(state.get("depth_m", max(0.0, -float(pos[2]))))

        # Transmission loss: spreading + absorption
        if rng < 0.1:
            rng = 0.1
        tl = self.spreading_factor * 10.0 * math.log10(rng)
        alpha = self._absorption_db_per_m(temp_c, salinity, depth)
        tl += alpha * rng

        # Received level and SNR
        rl = self.source_level_db - tl
        noise_level = self.noise_spectral_db + 10.0 * math.log10(self.bandwidth_hz)
        snr = rl - noise_level

        # Add fading variation
        snr += float(self._rng.normal(0.0, 3.0))

        # BER estimation (QPSK-like)
        if snr > 0:
            ber = 0.5 * math.erfc(math.sqrt(10.0 ** (snr / 10.0)))
        else:
            ber = 0.5

        # Packet delivery
        per = 1.0 - (1.0 - ber) ** self.packet_size_bits
        delivered = float(self._rng.random()) > per and rng <= self.max_range_m

        # Propagation latency
        sos = _speed_of_sound(temp_c, salinity, depth)
        latency = rng / sos

        # Doppler shift from relative velocity
        vel = np.asarray(state.get("vel", [0.0, 0.0, 0.0]), dtype=np.float64)[:3]
        if rng > 0.1:
            radial_vel = float(np.dot(vel, diff / rng))
            doppler = -self.frequency_hz * radial_vel / sos
        else:
            doppler = 0.0

        self._last_obs = {
            "connected": delivered,
            "snr_db": snr,
            "range_m": rng,
            "packet_delivered": delivered,
            "latency_s": latency,
            "ber": ber,
            "doppler_shift_hz": doppler,
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0
