"""
Base sensor interface for the Genesis external sensor realism layer.

All sensor models in this package implement this interface so that the
:class:`SensorScheduler` and :class:`SensorSuite` can drive them in a
uniform, rate-aware fashion.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Final, Generic, Literal, Self, TypeVar

import numpy as np

# Half a simulation sub-step at 10 kHz — used to guard against float drift in
# the scheduling comparison.
_SCHEDULE_TOLERANCE_S: Final[float] = 5e-5

# Shared type aliases used throughout the external sensor stack.
SensorInput = Mapping[str, Any]
SensorObservation = dict[str, Any]
ObservationT = TypeVar("ObservationT", bound=Mapping[str, Any])
NoiseModelName = Literal["gaussian", "laplace", "uniform", "none"]


class _SensorRNGProxy:
    """Proxy around ``numpy.random.Generator`` with configurable white-noise behavior."""

    _VALID_MODELS: Final[set[str]] = {"gaussian", "laplace", "uniform", "none"}

    def __init__(
        self,
        base_rng: np.random.Generator,
        *,
        noise_model: NoiseModelName = "gaussian",
        outlier_prob: float = 0.0,
        outlier_scale: float = 6.0,
    ) -> None:
        self._base_rng = base_rng
        self.configure(noise_model=noise_model, outlier_prob=outlier_prob, outlier_scale=outlier_scale)

    def configure(
        self,
        noise_model: str = "gaussian",
        *,
        outlier_prob: float = 0.0,
        outlier_scale: float = 6.0,
    ) -> None:
        model = str(noise_model).strip().lower()
        if model not in self._VALID_MODELS:
            raise ValueError(f"noise_model must be one of {sorted(self._VALID_MODELS)}, got {noise_model!r}")
        self.noise_model = model
        self.noise_outlier_prob = float(min(1.0, max(0.0, outlier_prob)))
        self.noise_outlier_scale = float(max(1.0, outlier_scale))

    def normal(self, loc: Any = 0.0, scale: Any = 1.0, size: Any = None):
        loc_arr = np.asarray(loc, dtype=float)
        sigma_arr = np.clip(np.asarray(scale, dtype=float), 0.0, None)

        if size is None:
            shape = np.broadcast(loc_arr, sigma_arr).shape
        elif isinstance(size, tuple):
            shape = size
        else:
            shape = (int(size),)

        noise = self._sample_zero_mean_noise(sigma_arr, shape)
        if shape == ():
            return float(np.asarray(loc_arr, dtype=float) + float(noise))
        return np.broadcast_to(loc_arr, shape).astype(float, copy=False) + np.asarray(noise, dtype=float)

    def _sample_zero_mean_noise(self, sigma: np.ndarray, shape: tuple[int, ...]):
        if self.noise_model == "none" or not np.any(sigma > 0.0):
            return 0.0 if shape == () else np.zeros(shape, dtype=float)

        sigma_b = float(sigma) if shape == () else np.broadcast_to(sigma, shape).astype(float, copy=False)
        if self.noise_model == "gaussian":
            samples = self._base_rng.normal(0.0, sigma_b, size=None if shape == () else shape)
        elif self.noise_model == "laplace":
            samples = self._base_rng.laplace(0.0, np.asarray(sigma_b) / math.sqrt(2.0), size=None if shape == () else shape)
        else:
            amp = np.asarray(sigma_b) * math.sqrt(3.0)
            samples = self._base_rng.uniform(-amp, amp, size=None if shape == () else shape)

        if self.noise_outlier_prob > 0.0:
            mask = self._base_rng.random(size=None if shape == () else shape) < self.noise_outlier_prob
            if bool(mask) if shape == () else np.any(mask):
                outlier_sigma = np.asarray(sigma_b) * self.noise_outlier_scale
                spikes = self._base_rng.normal(0.0, outlier_sigma, size=None if shape == () else shape)
                samples = spikes if shape == () and bool(mask) else np.where(mask, spikes, samples)

        return samples

    def __getattr__(self, name: str):
        return getattr(self._base_rng, name)


class BaseSensor(ABC, Generic[ObservationT]):
    """
    Abstract base class for every external sensor model.

    Subclasses receive an *ideal* state dict produced by Genesis at every
    scheduled step and return device-like (noisy, delayed, corrupted)
    observations.

    Parameters
    ----------
    name:
        Human-readable identifier used in logs and the :class:`SensorSuite`.
    update_rate_hz:
        Desired sensor update rate in Hz.  The :class:`SensorScheduler` uses
        this to determine when ``step()`` should be called.
    """

    noise_model: NoiseModelName
    noise_outlier_prob: float
    noise_outlier_scale: float

    def __init__(self, name: str = "", update_rate_hz: float = 1.0) -> None:
        if not isinstance(name, str):
            raise TypeError(f"name must be a str, got {type(name).__name__!r}")
        if update_rate_hz <= 0:
            raise ValueError(f"update_rate_hz must be positive, got {update_rate_hz}")
        self.name = name or type(self).__name__
        self.update_rate_hz = float(update_rate_hz)
        self.noise_model: NoiseModelName = "gaussian"
        self.noise_outlier_prob: float = 0.0
        self.noise_outlier_scale: float = 6.0
        self._last_update_time: float = -1.0

    def _make_rng(self, seed: int | None = None, *, rng: np.random.Generator | None = None) -> _SensorRNGProxy:
        base_rng = rng if rng is not None else np.random.default_rng(seed=seed)
        return _SensorRNGProxy(
            base_rng,
            noise_model=self.noise_model,
            outlier_prob=self.noise_outlier_prob,
            outlier_scale=self.noise_outlier_scale,
        )

    def configure_noise_model(
        self,
        noise_model: NoiseModelName | str = "gaussian",
        *,
        outlier_prob: float = 0.0,
        outlier_scale: float = 6.0,
    ) -> Self:
        """Configure the common white-noise sampler used by this sensor.

        Parameters
        ----------
        noise_model:
            One of ``"gaussian"`` (default), ``"laplace"`` (heavier tails),
            ``"uniform"`` (bounded noise), or ``"none"`` (disable white noise).
        outlier_prob:
            Optional per-sample probability of replacing the nominal noise draw
            with a larger outlier.
        outlier_scale:
            Multiplicative scale factor for those rare outlier events.
        """
        model = str(noise_model).strip().lower()
        if model not in _SensorRNGProxy._VALID_MODELS:
            raise ValueError(f"noise_model must be one of {sorted(_SensorRNGProxy._VALID_MODELS)}, got {noise_model!r}")
        self.noise_model = model  # type: ignore[assignment]
        self.noise_outlier_prob = float(min(1.0, max(0.0, outlier_prob)))
        self.noise_outlier_scale = float(max(1.0, outlier_scale))

        rng = getattr(self, "_rng", None)
        if rng is None:
            return self
        if isinstance(rng, _SensorRNGProxy):
            rng.configure(noise_model=noise_model, outlier_prob=outlier_prob, outlier_scale=outlier_scale)
        elif isinstance(rng, np.random.Generator):
            self._rng = self._make_rng(rng=rng)

        gauss_markov_type = None
        try:
            from ._gauss_markov import GaussMarkovProcess as _GaussMarkovProcess

            gauss_markov_type = _GaussMarkovProcess
        except Exception:  # pragma: no cover - defensive import guard
            pass

        wrapped_rng = getattr(self, "_rng", None)
        if gauss_markov_type is not None:
            for attr_name in dir(self):
                attr = getattr(self, attr_name, None)
                if isinstance(attr, gauss_markov_type):
                    attr._rng = wrapped_rng  # type: ignore[assignment]
        return self

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def reset(self, env_id: int = 0) -> None:
        """
        Reset all internal state for the given environment index.

        Called at the start of each episode / scene reset.
        """

    @classmethod
    def _from_config_with_noise(cls, config: Any):
        """Construct from a config object while honoring the shared noise controls."""
        cfg = config.model_dump() if hasattr(config, "model_dump") else dict(config)
        noise_model = cfg.pop("noise_model", "gaussian")
        noise_outlier_prob = cfg.pop("noise_outlier_prob", 0.0)
        noise_outlier_scale = cfg.pop("noise_outlier_scale", 6.0)
        sensor = cls(**cfg)
        sensor.configure_noise_model(
            noise_model,
            outlier_prob=noise_outlier_prob,
            outlier_scale=noise_outlier_scale,
        )
        return sensor

    @classmethod
    def from_config(cls, config: Any):
        """Construct a sensor from its validated config object."""
        raise NotImplementedError(f"{cls.__name__}.from_config() must be implemented by subclasses")

    @abstractmethod
    def step(self, sim_time: float, state: SensorInput) -> ObservationT:
        """
        Advance the sensor model by one update tick.

        Parameters
        ----------
        sim_time:
            Current simulation time in seconds.
        state:
            Ideal measurements from Genesis.  Which keys are present
            depends on the sensor type; each subclass documents what it
            expects.

        Returns
        -------
        dict
            Sensor observation after applying the device model (noise,
            distortion, timing artefacts, …).
        """

    def get_observation(self) -> ObservationT | SensorObservation:
        """Return the most recent sensor observation without re-computing it.

        Returns an empty dict if ``step()`` has not yet been called.
        Subclasses may override this to return a richer default.
        """
        return getattr(self, "_last_obs", {})

    # ------------------------------------------------------------------
    # Scheduling helpers
    # ------------------------------------------------------------------

    def is_due(self, sim_time: float) -> bool:
        """
        Return ``True`` when the sensor is ready for the next update.

        Comparison is done with a small tolerance (half a simulation
        sub-step at 10 kHz) to avoid floating-point drift.
        """
        if self._last_update_time < 0:
            return True
        period = 1.0 / self.update_rate_hz
        return (sim_time - self._last_update_time) >= period - _SCHEDULE_TOLERANCE_S

    def _mark_updated(self, sim_time: float) -> None:
        self._last_update_time = sim_time

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @classmethod
    def from_preset(cls, name: str, **overrides):
        """
        Construct a sensor from a named preset, with optional overrides.

        Parameters
        ----------
        name:
            Preset name (case-insensitive), e.g. ``"PIXHAWK_ICM20689"``.
        **overrides:
            Keyword arguments forwarded to the config's ``model_copy(update=...)``.

        Returns
        -------
        BaseSensor
            A new sensor instance.

        Examples
        --------
        ::

            imu = IMUModel.from_preset("PIXHAWK_ICM20689")
            imu = IMUModel.from_preset("PIXHAWK_ICM20689", update_rate_hz=500.0)
        """
        from .presets import get_preset

        noise_override_keys = ("noise_model", "noise_outlier_prob", "noise_outlier_scale")
        noise_overrides = {key: overrides.pop(key) for key in noise_override_keys if key in overrides}

        cfg = get_preset(name, **overrides)
        sensor = cls.from_config(cfg)
        if noise_overrides:
            sensor.configure_noise_model(
                noise_overrides.get("noise_model", sensor.noise_model),
                outlier_prob=noise_overrides.get("noise_outlier_prob", sensor.noise_outlier_prob),
                outlier_scale=noise_overrides.get("noise_outlier_scale", sensor.noise_outlier_scale),
            )
        return sensor

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, rate={self.update_rate_hz} Hz)"


__all__ = ["BaseSensor", "SensorInput", "SensorObservation"]
