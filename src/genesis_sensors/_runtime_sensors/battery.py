"""
LiPo / LiIon battery monitor sensor model.

Models the voltage, current, and state-of-charge (SoC) readout typically
found on commercial drone power modules (e.g. Holybro PM02, Matek MR-BAT or
a Pixhawk-compatible BEC with analog current/voltage port).

Physical model
--------------
A first-order Thevenin equivalent circuit::

    V_terminal = V_oc(SoC) · n_cells  –  I · R_internal

where

* ``V_oc(SoC)`` is the open-circuit voltage per cell from a piecewise-linear
  lookup table derived from typical LiPo/LiHV discharge curves.
* ``I`` is the instantaneous current from ``state["current_a"]``.
* ``R_internal`` is the DC internal resistance of the entire pack.

SoC tracking uses Coulomb counting::

    SoC(t) = SoC(0) – ∫ I dt / C_total    (C_total in A·s)

Sensor noise
------------
Both voltage and current readings are corrupted by independent Gaussian noise
with configurable standard deviations.

Usage
-----
::

    batt = BatteryModel(n_cells=4, capacity_mah=5000.0)
    obs = batt.step(sim_time, {"current_a": 12.5})
    print(obs["voltage_v"])          # terminal voltage (V)
    print(obs["soc"])                # state of charge 0–1
    print(obs["is_low"])             # True when below warning threshold

SoC-to-OCV table
----------------
Built-in table for a standard LiPo cell (3.50–4.20 V).  A second table for
LiHV (3.50–4.35 V) can be selected at construction via ``cell_chemistry``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Final, Literal

import numpy as np

from .base import BaseSensor
from .types import BatteryObservation

if TYPE_CHECKING:
    from .config import BatteryConfig


# ---------------------------------------------------------------------------
# SoC → OCV piecewise lookup tables (per cell, V)
# ---------------------------------------------------------------------------

# Standard LiPo 4.20 V / cell
_LIPO_SOC: Final[list[float]] = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
_LIPO_OCV: Final[list[float]] = [3.27, 3.50, 3.60, 3.68, 3.73, 3.78, 3.82, 3.87, 3.94, 4.02, 4.11, 4.20]

# LiHV (High Voltage LiPo) 4.35 V / cell
_LIHV_SOC: Final[list[float]] = [0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]
_LIHV_OCV: Final[list[float]] = [3.27, 3.50, 3.62, 3.72, 3.78, 3.84, 3.90, 3.97, 4.06, 4.15, 4.26, 4.35]


def _interp_ocv(soc: float, soc_table: list[float], ocv_table: list[float]) -> float:
    """Piecewise linear interpolation of OCV from SoC (clamped to table range)."""
    soc = max(soc_table[0], min(soc_table[-1], soc))
    # Binary search for the segment
    lo, hi = 0, len(soc_table) - 2
    while lo < hi:
        mid = (lo + hi) // 2
        if soc_table[mid + 1] <= soc:
            lo = mid + 1
        else:
            hi = mid
    t = (soc - soc_table[lo]) / (soc_table[lo + 1] - soc_table[lo])
    return ocv_table[lo] + t * (ocv_table[lo + 1] - ocv_table[lo])


class BatteryModel(BaseSensor):
    """
    LiPo / LiHV battery monitor sensor model.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Rate at which observations are emitted (Hz).  SoC is updated every
        ``step()`` call regardless of this rate.
    n_cells:
        Number of cells in series (e.g. ``3`` for a 3S pack, ``4`` for 4S).
    capacity_mah:
        Rated capacity in milli-ampere-hours.
    cell_chemistry:
        ``"lipo"`` (default, 4.20 V full) or ``"lihv"`` (4.35 V full).
    internal_resistance_ohm:
        Pack-level DC internal resistance (Ω).  Typical single-cell LiPo:
        3–15 mΩ/cell × n_cells.
    v_warn_per_cell_v:
        Per-cell voltage below which ``is_low`` is set to ``True``.
        Typical flight controllers alarm at ≈3.7 V/cell.
    current_noise_a:
        Standard deviation of Gaussian current measurement noise (A).
        Typical shunt/Hall sensor: 0.02–0.2 A.
    voltage_noise_v:
        Standard deviation of Gaussian voltage measurement noise (V).
        Typical ADC: 0.005–0.02 V.
    initial_soc:
        Starting state of charge (0 = empty, 1 = full).
    temp_coeff_resistance:
        Temperature coefficient for internal resistance.  The effective
        resistance is scaled by ``1 + coeff × (T - 25)``.  LiPo cells
        have higher internal resistance at low temperatures.  Typical
        value: 0.005–0.02 /°C.  ``0`` = temperature-independent.
    temp_coeff_capacity:
        Temperature coefficient for effective capacity.  At low temps,
        available capacity drops.  Scaled as ``C_eff = C × max(0.1,
        1 + coeff × (T - 25))``.  Typical value: −0.005 /°C (capacity
        decreases ~0.5 % per °C below 25 °C).  ``0`` = disabled.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "battery",
        update_rate_hz: float = 10.0,
        n_cells: int = 4,
        capacity_mah: float = 5000.0,
        cell_chemistry: Literal["lipo", "lihv"] = "lipo",
        internal_resistance_ohm: float = 0.012,
        v_warn_per_cell_v: float = 3.65,
        current_noise_a: float = 0.05,
        voltage_noise_v: float = 0.01,
        initial_soc: float = 1.0,
        temp_coeff_resistance: float = 0.0,
        temp_coeff_capacity: float = 0.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        if n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {n_cells}")
        if capacity_mah <= 0.0:
            raise ValueError(f"capacity_mah must be > 0, got {capacity_mah}")
        if not 0.0 <= initial_soc <= 1.0:
            raise ValueError(f"initial_soc must be in [0, 1], got {initial_soc}")

        self.n_cells = int(n_cells)
        self.capacity_mah = float(capacity_mah)
        self.cell_chemistry: Literal["lipo", "lihv"] = cell_chemistry
        self.internal_resistance_ohm = float(internal_resistance_ohm)
        self.v_warn_per_cell_v = float(v_warn_per_cell_v)
        self.current_noise_a = float(current_noise_a)
        self.voltage_noise_v = float(voltage_noise_v)
        self.initial_soc = float(initial_soc)
        self.temp_coeff_resistance = float(temp_coeff_resistance)
        self.temp_coeff_capacity = float(temp_coeff_capacity)

        # SoC lookup tables
        if cell_chemistry == "lihv":
            self._soc_table = _LIHV_SOC
            self._ocv_table = _LIHV_OCV
        else:
            self._soc_table = _LIPO_SOC
            self._ocv_table = _LIPO_OCV

        # Mutable state
        self._soc: float = self.initial_soc
        self._capacity_used_mah: float = 0.0
        self._prev_time: float | None = None
        self._last_obs: dict[str, Any] = {}

        self._rng = np.random.default_rng(seed=seed)
        self._seed = seed

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "BatteryConfig") -> "BatteryModel":
        """Construct from a :class:`~genesis.sensors.config.BatteryConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "BatteryConfig":
        """Return current parameters as a :class:`~genesis.sensors.config.BatteryConfig`."""
        from .config import BatteryConfig

        return BatteryConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            n_cells=self.n_cells,
            capacity_mah=self.capacity_mah,
            cell_chemistry=self.cell_chemistry,
            internal_resistance_ohm=self.internal_resistance_ohm,
            v_warn_per_cell_v=self.v_warn_per_cell_v,
            current_noise_a=self.current_noise_a,
            voltage_noise_v=self.voltage_noise_v,
            initial_soc=self.initial_soc,
            temp_coeff_resistance=self.temp_coeff_resistance,
            temp_coeff_capacity=self.temp_coeff_capacity,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._soc = self.initial_soc
        self._capacity_used_mah = 0.0
        self._prev_time = None
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> BatteryObservation | dict[str, Any]:
        """
        Advance the battery model by one simulation step.

        Expected keys in *state*:

        - ``"current_a"`` *(optional)* -- instantaneous current draw (A,
          positive = discharge).  Defaults to ``0.0`` when absent.

        The SoC is integrated from the current draw on every call.
        Each call computes and returns a fresh observation, and ``_last_obs``
        is updated to that newly generated measurement. Any sensor-rate
        gating is expected to be handled by the surrounding sensor scheduler.

        Returns
        -------
        BatteryObservation
            Measurement dict (see :class:`~genesis.sensors.types.BatteryObservation`).
        """
        current_a = float(state.get("current_a", 0.0))
        temp_c = float(state.get("temperature_c", 25.0))

        # ------------------------------------------------------------------
        # Temperature-dependent resistance and capacity
        # ------------------------------------------------------------------
        delta_t = temp_c - 25.0
        if self.temp_coeff_resistance != 0.0:
            effective_resistance = self.internal_resistance_ohm * max(0.1, 1.0 + self.temp_coeff_resistance * delta_t)
        else:
            effective_resistance = self.internal_resistance_ohm

        if self.temp_coeff_capacity != 0.0:
            effective_capacity_mah = self.capacity_mah * max(0.1, 1.0 + self.temp_coeff_capacity * delta_t)
        else:
            effective_capacity_mah = self.capacity_mah

        # ------------------------------------------------------------------
        # Coulomb counting — update SoC every simulation step
        # ------------------------------------------------------------------
        if self._prev_time is not None:
            dt = sim_time - self._prev_time
            if dt > 0.0:
                capacity_as = effective_capacity_mah * 1e-3 * 3600.0  # A·s
                self._soc -= current_a * dt / capacity_as
                self._soc = max(0.0, min(1.0, self._soc))
                self._capacity_used_mah += current_a * dt / 3.6  # A·s → mAh
        self._prev_time = sim_time

        # ------------------------------------------------------------------
        # OCV and terminal voltage (Thevenin model)
        # ------------------------------------------------------------------
        v_oc_per_cell = _interp_ocv(self._soc, self._soc_table, self._ocv_table)
        v_terminal = v_oc_per_cell * self.n_cells - current_a * effective_resistance
        v_terminal = max(0.0, v_terminal)  # physical lower bound

        # ------------------------------------------------------------------
        # Sensor noise
        # ------------------------------------------------------------------
        measured_voltage_v = v_terminal + float(self._rng.normal(0.0, self.voltage_noise_v))
        measured_current_a = current_a + float(self._rng.normal(0.0, self.current_noise_a))
        # Clamp to non-negative for presentation consistency
        measured_voltage_v = max(0.0, measured_voltage_v)
        measured_current_a = max(0.0, measured_current_a)

        measured_v_per_cell = measured_voltage_v / self.n_cells
        power_w = measured_voltage_v * measured_current_a
        is_low = measured_v_per_cell < self.v_warn_per_cell_v

        obs: BatteryObservation = {
            "voltage_v": measured_voltage_v,
            "voltage_per_cell_v": measured_v_per_cell,
            "current_a": measured_current_a,
            "power_w": power_w,
            "soc": self._soc,
            "capacity_used_mah": self._capacity_used_mah,
            "is_low": is_low,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def soc(self) -> float:
        """Current state of charge (0–1)."""
        return self._soc

    @property
    def capacity_used_mah(self) -> float:
        """Cumulative capacity consumed since last reset (mAh)."""
        return self._capacity_used_mah

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["BatteryModel"]
