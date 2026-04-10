"""
Depth camera (RGBD) sensor model.

Adds realistic depth-measurement noise to an ideal depth image from Genesis.
The noise model follows the structured-light / time-of-flight literature:

* **Base noise** — Gaussian with constant sigma ``depth_noise_sigma_m``.
* **Range-dependent noise** — Additional sigma proportional to ``z²`` via
  ``depth_noise_scale_z`` to capture the 1/z² photon-count fall-off.
* **Edge erosion** — A thin border of ``missing_edge_px`` pixels is zeroed
  to simulate depth shadows / occlusion artefacts near depth discontinuities.
* **Range clipping** — Values outside ``[min_depth_m, max_depth_m]`` are set
  to 0 (no return).

State keys consumed
-------------------
``"depth"``
    Ideal depth image in metres, shape ``(H, W)`` as float32/float64.
    Defaults to a zero array when absent.

Observation keys
----------------
``"depth_m"``
    ``float32`` shape ``(H, W)`` — noisy depth in metres; 0 where invalid.
``"valid_mask"``
    ``bool`` shape ``(H, W)`` — ``True`` at pixels with a valid depth return.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from .base import BaseSensor

if TYPE_CHECKING:
    from .config import DepthCameraConfig


class DepthCameraModel(BaseSensor):
    """
    Depth camera / RGBD sensor model.

    Parameters
    ----------
    name:
        Sensor identifier.
    update_rate_hz:
        Output frame rate (Hz).
    resolution:
        ``(width, height)`` in pixels.  When the incoming depth image differs
        in size it is used as-is; this resolution is only used as a fall-back
        when no depth image is provided in *state*.
    depth_noise_sigma_m:
        Constant Gaussian depth noise 1-σ (m).
    depth_noise_scale_z:
        Range-dependent noise coefficient; additional sigma = scale × z² (m).
    missing_edge_px:
        Width of the invalid-depth border (pixels).  Set 0 to disable.
    min_depth_m:
        Minimum measurable range (m); shallower returns are marked invalid.
    max_depth_m:
        Maximum measurable range (m); deeper returns are marked invalid.
    seed:
        Optional RNG seed for reproducibility.
    """

    def __init__(
        self,
        name: str = "depth_camera",
        update_rate_hz: float = 30.0,
        resolution: tuple[int, int] = (640, 480),
        depth_noise_sigma_m: float = 0.002,
        depth_noise_scale_z: float = 0.0005,
        missing_edge_px: int = 2,
        min_depth_m: float = 0.2,
        max_depth_m: float = 10.0,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.resolution = tuple(resolution)
        self.depth_noise_sigma_m = float(depth_noise_sigma_m)
        self.depth_noise_scale_z = float(depth_noise_scale_z)
        self.missing_edge_px = int(missing_edge_px)
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self._rng = np.random.default_rng(seed)
        self._seed = seed
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Factory helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "DepthCameraConfig") -> "DepthCameraModel":
        """Construct from a :class:`~genesis.sensors.config.DepthCameraConfig`."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "DepthCameraConfig":
        """Serialise parameters back to a :class:`~genesis.sensors.config.DepthCameraConfig`."""
        from .config import DepthCameraConfig

        return DepthCameraConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            resolution=self.resolution,
            depth_noise_sigma_m=self.depth_noise_sigma_m,
            depth_noise_scale_z=self.depth_noise_scale_z,
            missing_edge_px=self.missing_edge_px,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # Sensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        """Clear the cached observation and reset scheduler state."""
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Compute a noisy depth observation.

        Parameters
        ----------
        sim_time:
            Current simulation time (s).
        state:
            Dict with optional key ``"depth"`` (float array, metres,
            shape ``(H, W)``).

        Returns
        -------
        dict
            ``"depth_m"`` (float32, (H, W)) and ``"valid_mask"`` (bool, (H, W)).
        """
        if "depth" in state:
            ideal = np.asarray(state["depth"], dtype=np.float32)
        else:
            h, w = int(self.resolution[1]), int(self.resolution[0])
            ideal = np.zeros((h, w), dtype=np.float32)

        # --- Range-dependent Gaussian noise ---
        # sigma(z) = sigma_const + scale_z * z^2
        sigma = self.depth_noise_sigma_m + self.depth_noise_scale_z * (ideal**2)
        noise = self._rng.normal(0.0, 1.0, size=ideal.shape).astype(np.float32) * sigma
        noisy = ideal + noise

        # --- Validity mask: range gate ---
        valid = (ideal > 0) & (noisy >= self.min_depth_m) & (noisy <= self.max_depth_m)

        # --- Edge erosion ---
        ep = self.missing_edge_px
        if ep > 0 and valid.ndim == 2 and valid.shape[0] > 2 * ep and valid.shape[1] > 2 * ep:
            valid[:ep, :] = False
            valid[-ep:, :] = False
            valid[:, :ep] = False
            valid[:, -ep:] = False

        # Zero-out invalid returns
        depth_out = np.where(valid, noisy, np.float32(0.0))

        obs: dict[str, Any] = {
            "depth_m": depth_out.astype(np.float32),
            "valid_mask": valid,
        }
        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        """Return the most recent observation without triggering a new step."""
        return self._last_obs

    def __repr__(self) -> str:
        w, h = self.resolution
        return (
            f"DepthCameraModel(name={self.name!r}, rate={self.update_rate_hz} Hz, "
            f"res={w}×{h}, noise={self.depth_noise_sigma_m} m)"
        )
