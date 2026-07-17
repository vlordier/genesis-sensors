"""
IMU (Inertial Measurement Unit) sensor model.

Simulates a 6-DOF MEMS IMU (accelerometer + gyroscope) from ideal body-frame
kinematics produced by Genesis.  The error model is based on the standard
IEEE-1554 / Allan-deviation characterisation used by Isaac Lab and most
flight-sim frameworks:

* **White noise** (angle/velocity random walk) — ``noise_density_*`` in
  ``units/√Hz``, converted to per-sample sigma as ``σ = nd / √dt``.
* **Bias instability** (Gauss-Markov first-order drift) — ``bias_tau_*`` and
  ``bias_sigma_*``.
* **Scale-factor error** — a relative gain applied before noise, simulating
  manufacturing mismatch.

Gravity
-------
Genesis reports *pure* linear acceleration (no gravity).  Real IMUs measure
*specific force* = linear acceleration − gravity vector.  When
``add_gravity=True`` the model expects a ``"gravity_body"`` key in *state*
(gravity vector in body frame, m/s²) or uses the default ENU gravity
``[0, 0, 9.81]`` rotated to body frame (caller's responsibility to provide
the correct vector).

Usage
-----
::

    imu = IMUModel(update_rate_hz=200.0, noise_density_acc=2.0e-3)
    obs = imu.step(sim_time, {
        "lin_acc":     np.array([ax, ay, az]),   # body-frame m/s²
        "ang_vel":     np.array([wx, wy, wz]),   # body-frame rad/s
        "gravity_body": np.array([0.0, 0.0, 9.81]),  # body-frame gravity
    })
    print(obs["lin_acc"])   # noisy specific force (m/s²)
    print(obs["ang_vel"])   # noisy angular velocity (rad/s)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from ._gauss_markov import GaussMarkovProcess
from .base import BaseSensor
from .types import Float64Array, ImuObservation

if TYPE_CHECKING:
    from .config import IMUConfig

# Standard gravity constant (m/s²)
_G: Final[float] = 9.80665
# Default specific-force addend in body frame, assuming level flight in ENU.
# Real IMUs measure *specific force* = a_body − g_body.  For a level vehicle
# in ENU the gravity vector is [0, 0, −9.81]; subtracting it is equivalent to
# adding [0, 0, +9.81].  This constant is that addend (+Z, not −Z).
_DEFAULT_GRAVITY_BODY: Final[Float64Array] = np.array([0.0, 0.0, _G], dtype=np.float64)

# Minimum allowable time-constant for Gauss-Markov processes (seconds).
_MIN_BIAS_TAU_S: Final[float] = 1e-3


class IMUModel(BaseSensor):
    """
    Realistic 6-DOF MEMS IMU (accelerometer + gyroscope) sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        IMU output rate in Hz (100–1000 Hz for typical MEMS devices).
    noise_density_acc:
        Accelerometer white-noise density in m/s²/√Hz (velocity random walk).
        Converted internally to per-sample sigma = ``noise_density_acc /
        sqrt(1 / update_rate_hz)``.  Typical MEMS value: 2–5 × 10⁻³ m/s²/√Hz.
    noise_density_gyr:
        Gyroscope white-noise density in rad/s/√Hz (angle random walk).
        Typical MEMS value: 1–5 × 10⁻⁴ rad/s/√Hz.
    bias_tau_acc_s:
        Accelerometer bias correlation time (Gauss-Markov first-order process),
        seconds.  Typical value: 300–3600 s.
    bias_sigma_acc:
        Steady-state accelerometer bias standard deviation (m/s²).
    bias_tau_gyr_s:
        Gyroscope bias correlation time (s).
    bias_sigma_gyr:
        Steady-state gyroscope bias standard deviation (rad/s).
    scale_factor_acc:
        Relative accelerometer scale-factor error (dimensionless, ≥ −1).
        Measurement = (1 + scale_factor_acc) × true_value + bias + noise.
        Default 0 = no scale error.
    scale_factor_gyr:
        Relative gyroscope scale-factor error (dimensionless, ≥ −1).
    cross_axis_sensitivity_acc:
        Off-diagonal coupling coefficient for the accelerometer (dimensionless).
        A value of 0.01 means each axis picks up 1 % of the orthogonal axes'
        true acceleration.  Models imperfect physical alignment of MEMS sense
        elements; typical MEMS value: 0.005–0.03.  Default 0 = no coupling.
    cross_axis_sensitivity_gyr:
        Off-diagonal coupling coefficient for the gyroscope (dimensionless).
        Same interpretation as ``cross_axis_sensitivity_acc``.
    add_gravity:
        When ``True`` the model adds the gravity vector (from
        ``state["gravity_body"]`` or the default ENU level-flight value) to
        the ideal acceleration before applying noise.  This replicates how a
        real accelerometer measures specific force.
    max_acc_mps2:
        Accelerometer saturation limit per axis (m/s²).  Readings are
        clipped to ``[−max_acc, +max_acc]``.  ``0`` = no saturation.
        Typical MEMS devices: 160 m/s² (±16 g).
    max_gyr_rads:
        Gyroscope saturation limit per axis (rad/s).  Readings are clipped
        to ``[−max_gyr, +max_gyr]``.  ``0`` = no saturation.
        Typical MEMS devices: 34.9 rad/s (±2000 °/s).
    g_sensitivity_mps3:
        Gyroscope g-sensitivity in (rad/s)/(m/s²).  Models the rectification
        error where linear acceleration couples into the gyroscope output.
        Typical MEMS value: 0.005–0.015 (rad/s)/(m/s²).  ``0`` = disabled.
    temp_coeff_bias_acc:
        Accelerometer bias temperature coefficient (m/s²/°C).  Shifts the
        bias linearly when ``state["temperature_c"]`` differs from 25 °C.
        ``0`` = disabled.
    temp_coeff_bias_gyr:
        Gyroscope bias temperature coefficient (rad/s/°C).  ``0`` = disabled.
    adc_resolution_acc:
        Accelerometer ADC resolution (bits).  Output is quantised to the
        LSB = 2 × full_scale / 2^bits.  ``0`` = no quantisation.
    adc_resolution_gyr:
        Gyroscope ADC resolution (bits).  ``0`` = no quantisation.
    bandwidth_hz:
        First-order low-pass filter bandwidth (Hz) on the output.
        Models the anti-alias / digital filter inside real IMUs.
        ``0`` = no filtering (infinite bandwidth).
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "imu",
        update_rate_hz: float = 200.0,
        noise_density_acc: float = 2.0e-3,
        noise_density_gyr: float = 1.7e-4,
        bias_tau_acc_s: float = 300.0,
        bias_sigma_acc: float = 5.0e-3,
        bias_tau_gyr_s: float = 300.0,
        bias_sigma_gyr: float = 1.0e-4,
        scale_factor_acc: float = 0.0,
        scale_factor_gyr: float = 0.0,
        cross_axis_sensitivity_acc: float = 0.0,
        cross_axis_sensitivity_gyr: float = 0.0,
        add_gravity: bool = True,
        max_acc_mps2: float = 0.0,
        max_gyr_rads: float = 0.0,
        g_sensitivity_mps3: float = 0.0,
        temp_coeff_bias_acc: float = 0.0,
        temp_coeff_bias_gyr: float = 0.0,
        adc_resolution_acc: int = 0,
        adc_resolution_gyr: int = 0,
        bandwidth_hz: float = 0.0,
        seed: int | None = None,
        # Vibration injection (engine + terrain broadband)
        vib_engine_hz: float = 35.0,      # Base engine vibration frequency
        vib_engine_harmonics: int = 3,     # Number of harmonics (1=fundamental only)
        vib_engine_amplitude_ms2: float = 0.3,  # Amplitude at full throttle
        vib_terrain_scale: float = 0.1,    # Terrain roughness → vibration scale
        # Sensor mount misalignment
        misalignment_deg: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.noise_density_acc = float(noise_density_acc)
        self.noise_density_gyr = float(noise_density_gyr)
        self.bias_tau_acc_s = float(max(bias_tau_acc_s, _MIN_BIAS_TAU_S))
        self.bias_sigma_acc = float(bias_sigma_acc)
        self.bias_tau_gyr_s = float(max(bias_tau_gyr_s, _MIN_BIAS_TAU_S))
        self.bias_sigma_gyr = float(bias_sigma_gyr)
        self.scale_factor_acc = float(scale_factor_acc)
        self.scale_factor_gyr = float(scale_factor_gyr)
        self.cross_axis_sensitivity_acc = float(cross_axis_sensitivity_acc)
        self.cross_axis_sensitivity_gyr = float(cross_axis_sensitivity_gyr)
        self.add_gravity = bool(add_gravity)
        self.max_acc_mps2 = float(max(max_acc_mps2, 0.0))
        self.max_gyr_rads = float(max(max_gyr_rads, 0.0))
        self.g_sensitivity_mps3 = float(g_sensitivity_mps3)
        self.temp_coeff_bias_acc = float(temp_coeff_bias_acc)
        self.temp_coeff_bias_gyr = float(temp_coeff_bias_gyr)
        self.adc_resolution_acc = int(max(0, adc_resolution_acc))
        self.adc_resolution_gyr = int(max(0, adc_resolution_gyr))
        self.bandwidth_hz = float(max(0.0, bandwidth_hz))
        self.vib_engine_hz = float(vib_engine_hz)
        self.vib_engine_harmonics = int(vib_engine_harmonics)
        self.vib_engine_amplitude_ms2 = float(vib_engine_amplitude_ms2)
        self.vib_terrain_scale = float(vib_terrain_scale)
        self.misalignment_deg = tuple(float(v) for v in misalignment_deg)
        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed

        # Pre-compute misalignment rotation matrix
        rx, ry, rz = np.radians(self.misalignment_deg)
        cx, sx = np.cos(rx), np.sin(rx)
        cy, sy = np.cos(ry), np.sin(ry)
        cz, sz = np.cos(rz), np.sin(rz)
        self._M_misalign = np.array([
            [cy*cz, cy*sz, -sy],
            [sx*sy*cz - cx*sz, sx*sy*sz + cx*cz, sx*cy],
            [cx*sy*cz + sx*sz, cx*sy*sz - sx*cz, cx*cy],
        ], dtype=np.float64)

        # ------------------------------------------------------------------
        # Pre-computed noise and Gauss-Markov coefficients
        # ------------------------------------------------------------------
        dt = 1.0 / self.update_rate_hz

        # White-noise sigma per sample: σ = noise_density / √dt
        self._sigma_acc: float = self.noise_density_acc / math.sqrt(dt)
        self._sigma_gyr: float = self.noise_density_gyr / math.sqrt(dt)

        # Gauss-Markov bias processes for accelerometer and gyroscope
        self._bias_process_acc = GaussMarkovProcess(
            tau_s=self.bias_tau_acc_s,
            sigma=self.bias_sigma_acc,
            dt=dt,
            rng=self._rng,
            shape=(3,),
        )
        self._bias_process_gyr = GaussMarkovProcess(
            tau_s=self.bias_tau_gyr_s,
            sigma=self.bias_sigma_gyr,
            dt=dt,
            rng=self._rng,
            shape=(3,),
        )

        # Pre-compute (1 + scale_factor) for fast scalar multiply in step()
        self._gain_acc: float = 1.0 + self.scale_factor_acc
        self._gain_gyr: float = 1.0 + self.scale_factor_gyr

        # Pre-compute cross-axis coupling matrices.
        # M_acc[i, j] = gain_acc if i == j else cross_axis_sensitivity_acc
        # Applied in step() as: coupled_acc = M_acc @ true_acc
        self._M_acc: Float64Array = np.full((3, 3), self.cross_axis_sensitivity_acc, dtype=np.float64) + np.diag(
            np.full(3, self._gain_acc - self.cross_axis_sensitivity_acc)
        )
        self._M_gyr: Float64Array = np.full((3, 3), self.cross_axis_sensitivity_gyr, dtype=np.float64) + np.diag(
            np.full(3, self._gain_gyr - self.cross_axis_sensitivity_gyr)
        )

        # ADC quantisation steps (LSB = 2*full_scale / 2^bits)
        self._lsb_acc: float = 0.0
        if self.adc_resolution_acc > 0 and self.max_acc_mps2 > 0.0:
            self._lsb_acc = 2.0 * self.max_acc_mps2 / (2**self.adc_resolution_acc)
        self._lsb_gyr: float = 0.0
        if self.adc_resolution_gyr > 0 and self.max_gyr_rads > 0.0:
            self._lsb_gyr = 2.0 * self.max_gyr_rads / (2**self.adc_resolution_gyr)

        # Bandwidth LPF state
        self._lpf_acc: Float64Array = np.zeros(3, dtype=np.float64)
        self._lpf_gyr: Float64Array = np.zeros(3, dtype=np.float64)
        self._lpf_initialized: bool = False

        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "IMUConfig") -> "IMUModel":
        """Construct an :class:`IMUModel` from an :class:`~genesis.sensors.config.IMUConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "IMUConfig":
        """Return the current parameters as an :class:`~genesis.sensors.config.IMUConfig`."""
        from .config import IMUConfig

        return IMUConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            noise_density_acc=self.noise_density_acc,
            noise_density_gyr=self.noise_density_gyr,
            bias_tau_acc_s=self.bias_tau_acc_s,
            bias_sigma_acc=self.bias_sigma_acc,
            bias_tau_gyr_s=self.bias_tau_gyr_s,
            bias_sigma_gyr=self.bias_sigma_gyr,
            scale_factor_acc=self.scale_factor_acc,
            scale_factor_gyr=self.scale_factor_gyr,
            cross_axis_sensitivity_acc=self.cross_axis_sensitivity_acc,
            cross_axis_sensitivity_gyr=self.cross_axis_sensitivity_gyr,
            add_gravity=self.add_gravity,
            max_acc_mps2=self.max_acc_mps2,
            max_gyr_rads=self.max_gyr_rads,
            g_sensitivity_mps3=self.g_sensitivity_mps3,
            temp_coeff_bias_acc=self.temp_coeff_bias_acc,
            temp_coeff_bias_gyr=self.temp_coeff_bias_gyr,
            adc_resolution_acc=self.adc_resolution_acc,
            adc_resolution_gyr=self.adc_resolution_gyr,
            bandwidth_hz=self.bandwidth_hz,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Public properties (inspection / debugging)
    # ------------------------------------------------------------------

    @property
    def bias_acc(self) -> Float64Array:
        """Current accelerometer bias vector (m/s²).  Copy to prevent mutation."""
        return np.asarray(self._bias_process_acc.value, dtype=np.float64).copy()

    @property
    def bias_gyr(self) -> Float64Array:
        """Current gyroscope bias vector (rad/s).  Copy to prevent mutation."""
        return np.asarray(self._bias_process_gyr.value, dtype=np.float64).copy()

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        # Draw fresh steady-state biases each episode so that each rollout
        # experiences independent, realistic initial conditions.
        self._bias_process_acc.reset(self.bias_sigma_acc)
        self._bias_process_gyr.reset(self.bias_sigma_gyr)
        self._lpf_acc = np.zeros(3, dtype=np.float64)
        self._lpf_gyr = np.zeros(3, dtype=np.float64)
        self._lpf_initialized = False
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> ImuObservation | dict[str, Any]:
        """
        Produce a realistic IMU measurement.

        Expected keys in *state*:
        - ``"lin_acc"``  -- ``np.ndarray[3]`` ideal linear acceleration in
          **body frame** (m/s²), NOT including gravity.  Genesis provides this
          as the rigid-body acceleration minus external forces / gravity.
        - ``"ang_vel"``  -- ``np.ndarray[3]`` ideal angular velocity in
          **body frame** (rad/s).
        - ``"gravity_body"`` *(optional)* -- gravity vector rotated to body
          frame (m/s²).  Only used when ``add_gravity=True``.  If absent and
          ``add_gravity=True``, the default ENU level-flight gravity
          ``[0, 0, 9.81]`` is used (valid when the vehicle is near-level).
        """
        true_acc: Float64Array = np.asarray(state.get("lin_acc", [0.0, 0.0, 0.0]), dtype=np.float64)
        true_gyr: Float64Array = np.asarray(state.get("ang_vel", [0.0, 0.0, 0.0]), dtype=np.float64)
        if true_acc.shape != (3,):
            raise ValueError(f"state['lin_acc'] must be a 3-element array, got shape {true_acc.shape}")
        if true_gyr.shape != (3,):
            raise ValueError(f"state['ang_vel'] must be a 3-element array, got shape {true_gyr.shape}")

        # ------------------------------------------------------------------
        # Gravity injection
        # ------------------------------------------------------------------
        if self.add_gravity:
            gravity_body = state.get("gravity_body")
            if gravity_body is not None:
                g_vec: Float64Array = np.asarray(gravity_body, dtype=np.float64)
                if g_vec.shape != (3,):
                    raise ValueError(f"state['gravity_body'] must be a 3-element array, got shape {g_vec.shape}")
            else:
                g_vec = _DEFAULT_GRAVITY_BODY
            true_acc = true_acc + g_vec

        # ------------------------------------------------------------------
        # Vibration injection (engine + terrain broadband)
        # Applied BEFORE scale-factor/cross-axis — vibration is real acceleration
        # ------------------------------------------------------------------
        throttle = float(state.get("throttle", 0.0))
        terrain_roughness = float(state.get("terrain_roughness", 0.0))
        if abs(throttle) > 0.01 or terrain_roughness > 0.01:
            vib = np.zeros(3, dtype=np.float64)
            # Engine vibration: harmonic series at base frequency
            if abs(throttle) > 0.01:
                t = sim_time
                amp = abs(throttle) * self.vib_engine_amplitude_ms2
                for h in range(1, self.vib_engine_harmonics + 1):
                    vib[0] += amp * np.sin(2 * np.pi * self.vib_engine_hz * h * t + h * 0.7) / h
                    vib[1] += amp * np.sin(2 * np.pi * self.vib_engine_hz * h * t + h * 1.3) / h
                    vib[2] += amp * 0.5 * np.sin(2 * np.pi * self.vib_engine_hz * h * t + h) / h
            # Terrain vibration: broadband Gaussian proportional to roughness
            if terrain_roughness > 0.01:
                vib += self._rng.normal(0, terrain_roughness * self.vib_terrain_scale, 3)
            true_acc = true_acc + vib

        # ------------------------------------------------------------------
        # Sensor mount misalignment (rotation of measurement frame)
        # ------------------------------------------------------------------
        if np.any(self.misalignment_deg):
            true_acc = self._M_misalign @ true_acc
            true_gyr = self._M_misalign @ true_gyr

        # ------------------------------------------------------------------
        # Scale-factor error + cross-axis coupling (applied before bias/noise)
        # ------------------------------------------------------------------
        true_acc = self._M_acc @ true_acc
        true_gyr = self._M_gyr @ true_gyr

        # ------------------------------------------------------------------
        # Gauss-Markov bias drift
        # ------------------------------------------------------------------
        bias_acc = self._bias_process_acc.step()
        bias_gyr = self._bias_process_gyr.step()

        # ------------------------------------------------------------------
        # Temperature-dependent bias shift
        # ------------------------------------------------------------------
        if self.temp_coeff_bias_acc != 0.0 or self.temp_coeff_bias_gyr != 0.0:
            temp_c = float(state.get("temperature_c", 25.0))
            delta_t = temp_c - 25.0  # reference temperature = 25 °C
            if self.temp_coeff_bias_acc != 0.0:
                bias_acc = bias_acc + self.temp_coeff_bias_acc * delta_t
            if self.temp_coeff_bias_gyr != 0.0:
                bias_gyr = bias_gyr + self.temp_coeff_bias_gyr * delta_t

        # ------------------------------------------------------------------
        # G-sensitivity: linear acceleration coupling into gyroscope
        # ------------------------------------------------------------------
        if self.g_sensitivity_mps3 != 0.0:
            bias_gyr = bias_gyr + self.g_sensitivity_mps3 * true_acc

        # ------------------------------------------------------------------
        # White noise (angle/velocity random walk)
        # ------------------------------------------------------------------
        noisy_acc: Float64Array = true_acc + bias_acc + self._rng.normal(0.0, self._sigma_acc, 3)
        noisy_gyr: Float64Array = true_gyr + bias_gyr + self._rng.normal(0.0, self._sigma_gyr, 3)

        # ------------------------------------------------------------------
        # Saturation / clipping
        # ------------------------------------------------------------------
        if self.max_acc_mps2 > 0.0:
            noisy_acc = np.clip(noisy_acc, -self.max_acc_mps2, self.max_acc_mps2)
        if self.max_gyr_rads > 0.0:
            noisy_gyr = np.clip(noisy_gyr, -self.max_gyr_rads, self.max_gyr_rads)

        # ------------------------------------------------------------------
        # ADC quantisation
        # ------------------------------------------------------------------
        if self._lsb_acc > 0.0:
            noisy_acc = np.round(noisy_acc / self._lsb_acc) * self._lsb_acc
        if self._lsb_gyr > 0.0:
            noisy_gyr = np.round(noisy_gyr / self._lsb_gyr) * self._lsb_gyr

        # ------------------------------------------------------------------
        # Output bandwidth LPF (first-order IIR)
        # ------------------------------------------------------------------
        if self.bandwidth_hz > 0.0:
            dt = 1.0 / self.update_rate_hz
            alpha = min(1.0, dt * 2.0 * math.pi * self.bandwidth_hz / (1.0 + dt * 2.0 * math.pi * self.bandwidth_hz))
            if not self._lpf_initialized:
                self._lpf_acc = noisy_acc.copy()
                self._lpf_gyr = noisy_gyr.copy()
                self._lpf_initialized = True
            else:
                self._lpf_acc += alpha * (noisy_acc - self._lpf_acc)
                self._lpf_gyr += alpha * (noisy_gyr - self._lpf_gyr)
            noisy_acc = self._lpf_acc.copy()
            noisy_gyr = self._lpf_gyr.copy()

        obs: ImuObservation = {
            "lin_acc": noisy_acc,
            "ang_vel": noisy_gyr,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["IMUModel"]
