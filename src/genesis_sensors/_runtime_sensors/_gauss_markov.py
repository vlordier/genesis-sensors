"""Reusable first-order Gauss-Markov random process for sensor bias drift.

Many sensor models (IMU, GNSS, barometer, airspeed, thermometer, hygrometer,
…) share the same exponentially-correlated bias drift. This module extracts
that pattern into a lightweight helper so each sensor doesn't re-derive the
math.

The discrete-time first-order Gauss-Markov process is::

    x[k+1] = α · x[k] + σ_drive · w[k],   w ~ N(0, 1)

where::

    α          = exp(−dt / τ)
    σ_drive    = σ_ss · √(1 − α²)

* ``τ``     — correlation time (seconds).
* ``σ_ss``  — steady-state standard deviation.
* ``dt``    — time step (seconds).
"""

from __future__ import annotations

import math
from typing import overload

import numpy as np

from .types import Float64Array

# Floor for tau to avoid division by ≤ 0.
_MIN_TAU_S: float = 1e-3


class GaussMarkovProcess:
    """Scalar or vector first-order Gauss-Markov bias process.

    Parameters
    ----------
    tau_s:
        Correlation time (seconds).  Clamped to a minimum of 1 ms.
    sigma:
        Steady-state standard deviation (units match the bias).
    dt:
        Fixed time step between updates (seconds).
    rng:
        NumPy random generator instance (shared with the parent sensor).
    shape:
        Shape of the state.  ``()`` for scalar, ``(3,)`` for a 3-D vector, etc.
        Default: scalar.
    """

    __slots__ = ("_alpha", "_drive_sigma", "_rng", "_shape", "_value")

    def __init__(
        self,
        tau_s: float,
        sigma: float,
        dt: float,
        rng: np.random.Generator,
        shape: tuple[int, ...] = (),
    ) -> None:
        tau_s = max(float(tau_s), _MIN_TAU_S)
        sigma = float(sigma)
        dt = float(dt)
        self._alpha: float = math.exp(-dt / tau_s)
        self._drive_sigma: float = sigma * math.sqrt(1.0 - self._alpha**2)
        self._rng = rng
        self._shape = shape
        # Initialise from steady-state distribution so the bias is
        # physically realistic from t = 0.
        if shape == ():
            self._value: float | Float64Array = float(rng.normal(0.0, sigma))
        else:
            self._value = rng.normal(0.0, sigma, shape).astype(np.float64)

    # ------------------------------------------------------------------
    # Advance
    # ------------------------------------------------------------------

    @overload
    def step(self) -> float: ...
    @overload
    def step(self) -> Float64Array: ...

    def step(self) -> float | Float64Array:
        """Advance one time step and return the new bias value."""
        if self._shape == ():
            self._value = self._alpha * self._value + float(self._rng.normal(0.0, self._drive_sigma))
        else:
            self._value = self._alpha * self._value + self._rng.normal(0.0, self._drive_sigma, self._shape)
        return self._value

    # ------------------------------------------------------------------
    # Inspection / reset
    # ------------------------------------------------------------------

    @property
    def value(self) -> float | Float64Array:
        """Current bias value (scalar or array).  Returns a copy for arrays."""
        if isinstance(self._value, np.ndarray):
            return self._value.copy()
        return self._value

    def reset(self, sigma: float) -> None:
        """Re-draw from steady-state distribution (used on episode reset)."""
        if self._shape == ():
            self._value = float(self._rng.normal(0.0, sigma))
        else:
            self._value = self._rng.normal(0.0, sigma, self._shape).astype(np.float64)

    def __repr__(self) -> str:
        return f"GaussMarkovProcess(α={self._alpha:.4f}, σ_drive={self._drive_sigma:.4g}, shape={self._shape})"


__all__ = ["GaussMarkovProcess"]
