"""Motor temperature sensor model.

Simulates winding/stator temperature measurement for electric motors
using a simple thermal model.  Critical for torque-limited operation
and thermal management on industrial arms, legged robots, and drones.

State keys consumed
-------------------
``"motor_current_a"``
    Instantaneous motor current (A).  Generates I²R heating.
``"motor_speed_rads"``
    Motor shaft speed (rad/s).  Contributes friction heating.
``"ambient_temperature_c"``
    Ambient temperature (°C).  Default 25.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor, SensorInput, SensorObservation

if TYPE_CHECKING:
    from .config import MotorTemperatureConfig


class MotorTemperatureModel(BaseSensor):
    """Motor winding temperature sensor with first-order thermal model.

    Uses a lumped-parameter thermal model:
    ``dT/dt = (P_heat - (T - T_amb) / R_th) / C_th``

    Parameters
    ----------
    name:
        Sensor instance name.
    update_rate_hz:
        Output rate (Hz).
    winding_resistance_ohm:
        Motor winding DC resistance (Ω).  Determines I²R heating.
    thermal_resistance_c_per_w:
        Thermal resistance winding→ambient (°C/W).
    thermal_capacitance_j_per_c:
        Thermal capacitance of winding mass (J/°C).
    friction_coeff_w_per_rads:
        Speed-dependent friction heating coefficient (W per rad/s).
    noise_sigma_c:
        1-σ temperature measurement noise (°C).
    overtemp_threshold_c:
        Alarm threshold (°C).
    initial_temperature_c:
        Starting winding temperature (°C).
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "motor_temperature",
        update_rate_hz: float = 10.0,
        winding_resistance_ohm: float = 0.5,
        thermal_resistance_c_per_w: float = 2.0,
        thermal_capacitance_j_per_c: float = 50.0,
        friction_coeff_w_per_rads: float = 0.001,
        noise_sigma_c: float = 0.5,
        overtemp_threshold_c: float = 120.0,
        initial_temperature_c: float = 25.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.winding_resistance_ohm = float(max(0.0, winding_resistance_ohm))
        self.thermal_resistance_c_per_w = float(max(0.01, thermal_resistance_c_per_w))
        self.thermal_capacitance_j_per_c = float(max(0.01, thermal_capacitance_j_per_c))
        self.friction_coeff_w_per_rads = float(max(0.0, friction_coeff_w_per_rads))
        self.noise_sigma_c = float(max(0.0, noise_sigma_c))
        self.overtemp_threshold_c = float(overtemp_threshold_c)
        self.initial_temperature_c = float(initial_temperature_c)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed
        self._winding_temp_c = float(initial_temperature_c)
        self._prev_time: float | None = None
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "MotorTemperatureConfig") -> "MotorTemperatureModel":
        return cls._from_config_with_noise(config)

    def get_config(self) -> "MotorTemperatureConfig":
        from .config import MotorTemperatureConfig

        return MotorTemperatureConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            winding_resistance_ohm=self.winding_resistance_ohm,
            thermal_resistance_c_per_w=self.thermal_resistance_c_per_w,
            thermal_capacitance_j_per_c=self.thermal_capacitance_j_per_c,
            friction_coeff_w_per_rads=self.friction_coeff_w_per_rads,
            noise_sigma_c=self.noise_sigma_c,
            overtemp_threshold_c=self.overtemp_threshold_c,
            initial_temperature_c=self.initial_temperature_c,
            seed=self._seed,
        )

    def step(self, *, sim_time: float, state: SensorInput) -> SensorObservation:
        self._last_update_time = sim_time

        current_a = float(state.get("motor_current_a", 0.0))
        speed_rads = float(state.get("motor_speed_rads", 0.0))
        t_amb = float(state.get("ambient_temperature_c", state.get("temperature_c", 25.0)))

        # Thermal model integration
        if self._prev_time is not None:
            dt = sim_time - self._prev_time
            if dt > 0.0:
                p_joule = current_a**2 * self.winding_resistance_ohm
                p_friction = self.friction_coeff_w_per_rads * abs(speed_rads)
                p_total = p_joule + p_friction
                p_dissipated = (self._winding_temp_c - t_amb) / self.thermal_resistance_c_per_w
                d_temp = (p_total - p_dissipated) / self.thermal_capacitance_j_per_c * dt
                self._winding_temp_c += d_temp
        self._prev_time = sim_time

        # Measurement noise
        measured = self._winding_temp_c
        if self.noise_sigma_c > 0.0:
            measured += float(self._rng.normal(0.0, self.noise_sigma_c))

        overtemp = measured >= self.overtemp_threshold_c

        self._last_obs = {
            "temperature_c": measured,
            "overtemp_alarm": overtemp,
            "power_dissipated_w": (self._winding_temp_c - t_amb) / self.thermal_resistance_c_per_w,
        }
        return self._last_obs

    def get_observation(self) -> SensorObservation:
        return self._last_obs

    def reset(self, env_id: int = 0) -> None:
        self._winding_temp_c = self.initial_temperature_c
        self._prev_time = None
        self._last_obs = {}
        self._last_update_time = -1.0
