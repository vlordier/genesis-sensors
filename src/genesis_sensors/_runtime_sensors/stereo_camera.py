"""
Stereo camera pair sensor model.

Simulates a rigidly mounted left/right camera pair with a known baseline.
Each eye runs the same :class:`~genesis.sensors.CameraModel` corruption
pipeline (shot/read noise, vignetting, chromatic aberration, rolling shutter,
…), producing correlated but independently noisy images.

Depth reconstruction
--------------------
When ``state["depth"]`` is provided (the ideal per-pixel depth map rendered by
Genesis), the model:

1. Converts depth → disparity using the standard pinhole formula
   ``d = f · B / Z`` where *f* is the horizontal focal length derived from
   the resolution (``f = W / (2 · tan(hfov/2))``) and *B* is the baseline.
2. Adds sub-pixel disparity noise to match the performance spec of the
   selected hardware preset.
3. Reconstructs a noisy depth map from the noisy disparity.
4. Reprojects to a (H·W, 3) point cloud in the left-camera frame.

The ideal scenario: Genesis renders both views directly via two sensor cameras.
The simplification: Genesis renders only the left view + depth, and this model
synthesises the right view by applying a horizontal pixel shift proportional to
disparity (a first-order approximation valid for planar or far-field scenes).
Both approximations are clearly flagged in the observation dict.

State keys consumed
-------------------
- ``"rgb"``       — left-camera ideal image ``(H, W, 3)`` uint8 or float32.
- ``"depth"``     — ideal per-pixel depth map ``(H, W)`` float32 metres.
                    When absent, depth / disparity outputs are skipped.
- ``"rgb_right"`` — right-camera ideal image ``(H, W, 3)`` *(optional)*.
                    When absent, synthesised from ``"rgb"`` + ``"depth"``.

Observation keys emitted
------------------------
- ``"rgb_left"``      — left-camera corrupted image ``(H, W, 3)`` uint8.
- ``"rgb_right"``     — right-camera corrupted image ``(H, W, 3)`` uint8.
- ``"disparity"``     — noisy disparity map ``(H, W)`` float32 pixels (≥ 0).
- ``"depth"``         — noisy depth map ``(H, W)`` float32 metres.
- ``"point_cloud"``   — reprojected points ``(N, 3)`` float32 metres in the
                        left-camera frame (+X right, +Y down, +Z forward).
- ``"valid_mask"``    — boolean mask ``(H, W)`` True where depth is valid.
- ``"baseline_m"``    — the configured baseline (informational, float).
- ``"focal_px"``      — computed focal length in pixels (informational, float).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .camera_model import CameraModel
from .types import FloatArray, UInt8Array

if TYPE_CHECKING:
    from .config import StereoCameraConfig

# Minimum valid depth to avoid division-by-zero in disparity.
_MIN_DEPTH_M: Final[float] = 0.05
# cap on maximum disparity (pixels) to avoid absurd values for very near pixels.
_MAX_DISPARITY_PX: Final[float] = 512.0
# Default horizontal field-of-view (radians) used when not specified.
_DEFAULT_HFOV_RAD: Final[float] = math.radians(90.0)


def _depth_to_disparity(depth: FloatArray, focal_px: float, baseline_m: float) -> FloatArray:
    """Convert a depth map (m) to disparity (px) using the pinhole formula.

    ``d = f · B / Z``

    Invalid (non-positive) depth pixels produce zero disparity.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        disparity = np.where(depth > _MIN_DEPTH_M, focal_px * baseline_m / np.maximum(depth, _MIN_DEPTH_M), 0.0)
    return np.clip(disparity, 0.0, _MAX_DISPARITY_PX).astype(np.float32)


def _disparity_to_depth(disparity: FloatArray, focal_px: float, baseline_m: float) -> FloatArray:
    """Invert disparity to depth; zero-disparity pixels → infinity (set to 0 for invalid)."""
    with np.errstate(divide="ignore", invalid="ignore"):
        depth = np.where(disparity > 0.0, focal_px * baseline_m / np.maximum(disparity, 1e-6), 0.0)
    return depth.astype(np.float32)


def _reproject_to_pointcloud(depth: FloatArray, focal_px: float) -> FloatArray:
    """Reproject a depth map to a (N, 3) point cloud in the left-camera frame.

    Coordinate convention: +X right, +Y down, +Z into the scene (OpenCV).
    """
    h, w = depth.shape
    u = np.arange(w, dtype=np.float32) - (w - 1) * 0.5
    v = np.arange(h, dtype=np.float32) - (h - 1) * 0.5
    ug, vg = np.meshgrid(u, v)  # (H, W)
    valid = depth > _MIN_DEPTH_M
    Z = depth[valid]
    X = ug[valid] * Z / focal_px
    Y = vg[valid] * Z / focal_px
    return np.stack([X, Y, Z], axis=1).astype(np.float32)


def _synthesise_right_view(rgb: UInt8Array, disparity: FloatArray) -> UInt8Array:
    """Synthesise a right-camera view by shifting the left image leftward.

    This is a first-order approximation: ``I_R(u, v) ≈ I_L(u − d(u,v), v)``.
    Pixels shifted out of the frame are filled with the border colour.
    Uses nearest-neighbour sampling (fast, sufficient for the noise floor
    of the stereo algorithm).
    """
    h, w = rgb.shape[:2]
    # Integer pixel shift per row (nearest-neighbour)
    shift = np.round(disparity).astype(np.int32)  # (H, W)
    u_src = np.clip(np.arange(w, dtype=np.int32)[None, :] - shift, 0, w - 1)  # (H, W)
    v_idx = np.arange(h, dtype=np.int32)[:, None]  # (H, 1) broadcast
    return rgb[v_idx, u_src]  # (H, W, 3) integer index broadcast


class StereoCameraModel(BaseSensor):
    """
    Stereo camera pair with depth reconstruction.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        Frame rate in Hz (both cameras share the same rate).
    resolution:
        ``(width, height)`` in pixels. Applies to both eyes.
    baseline_m:
        Physical separation between the left and right optical centres (m).
        Typical values: 0.06 m (ZED 2), 0.12 m (MYNT EYE D), 0.05 m (D435).
    hfov_deg:
        Horizontal field of view in degrees. Used to compute the focal
        length in pixels: ``f = W / (2 · tan(hfov/2))``.
    disparity_noise_sigma_px:
        Standard deviation of Gaussian noise added to the ideal disparity
        (pixels). Typical SGBM performance: 0.3–1.0 px at mid-range.
    disparity_noise_scale_z:
        Range-dependent disparity noise coefficient.  When > 0, additional
        noise sigma = ``scale_z × Z²`` is added where Z is the depth in
        metres.  This models the degradation of stereo matching at greater
        distances where disparity shrinks and sub-pixel accuracy drops.
        Typical value: 0.001–0.01.  ``0`` = constant noise only.
    min_depth_m:
        Minimum valid depth. Pixels closer than this receive zero disparity.
    max_depth_m:
        Maximum measurable depth. Beyond this the disparity is too small to
        resolve and the pixel is marked invalid.
    left_config:
        If provided, used to construct the left :class:`CameraModel`.
        Otherwise a default :class:`CameraModel` is built from the other
        construction parameters.
    right_config:
        Same as ``left_config`` but for the right eye. When ``None``, the
        right eye is a clone of the left (same noise parameters, different
        RNG seed for statistical independence).
    seed:
        Optional RNG seed.
    """

    def __init__(
        self,
        name: str = "stereo_camera",
        update_rate_hz: float = 30.0,
        resolution: tuple[int, int] = (640, 480),
        baseline_m: float = 0.06,
        hfov_deg: float = 90.0,
        disparity_noise_sigma_px: float = 0.5,
        disparity_noise_scale_z: float = 0.0,
        min_depth_m: float = _MIN_DEPTH_M,
        max_depth_m: float = 20.0,
        left_config: "StereoCameraConfig | None" = None,
        right_config: "StereoCameraConfig | None" = None,
        seed: int | None = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)

        self.resolution = tuple(resolution)
        self.baseline_m = float(baseline_m)
        self.hfov_deg = float(hfov_deg)
        self.disparity_noise_sigma_px = float(max(0.0, disparity_noise_sigma_px))
        self.disparity_noise_scale_z = float(max(0.0, disparity_noise_scale_z))
        self.min_depth_m = float(min_depth_m)
        self.max_depth_m = float(max_depth_m)
        self._seed = seed

        # Focal length derived from resolution + HFOV
        w = int(resolution[0])
        hfov_rad = math.radians(self.hfov_deg)
        self.focal_px: float = w / (2.0 * math.tan(hfov_rad * 0.5))

        # Build left and right CameraModel instances.
        # Use SeedSequence to give each eye a statistically independent seed.
        if seed is not None:
            child_seeds = np.random.SeedSequence(seed).spawn(3)
            seed_left = int(child_seeds[0].generate_state(1)[0])
            seed_right = int(child_seeds[1].generate_state(1)[0])
            seed_rng = int(child_seeds[2].generate_state(1)[0])
        else:
            seed_left = seed_right = seed_rng = None

        if left_config is not None:
            cam_kw = left_config.to_camera_kwargs()
            cam_kw.setdefault("seed", seed_left)
            self._left = CameraModel(**cam_kw)
        else:
            self._left = CameraModel(
                name="stereo_left",
                update_rate_hz=update_rate_hz,
                resolution=resolution,
                seed=seed_left,
            )

        if right_config is not None:
            cam_kw = right_config.to_camera_kwargs()
            cam_kw.setdefault("seed", seed_right)
            self._right = CameraModel(**cam_kw)
        else:
            self._right = CameraModel(
                name="stereo_right",
                update_rate_hz=update_rate_hz,
                resolution=resolution,
                seed=seed_right,
            )

        self._rng = np.random.default_rng(seed=seed_rng)
        self._last_obs: dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Config factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: "StereoCameraConfig") -> "StereoCameraModel":
        """Construct a :class:`StereoCameraModel` from a :class:`~genesis.sensors.config.StereoCameraConfig`."""
        return cls(
            name=config.name,
            update_rate_hz=config.update_rate_hz,
            resolution=config.resolution,
            baseline_m=config.baseline_m,
            hfov_deg=config.hfov_deg,
            disparity_noise_sigma_px=config.disparity_noise_sigma_px,
            disparity_noise_scale_z=getattr(config, "disparity_noise_scale_z", 0.0),
            min_depth_m=config.min_depth_m,
            max_depth_m=config.max_depth_m,
            seed=config.seed,
        )

    def get_config(self) -> "StereoCameraConfig":
        """Return current parameters as a :class:`~genesis.sensors.config.StereoCameraConfig`."""
        from .config import StereoCameraConfig

        return StereoCameraConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            resolution=self.resolution,
            baseline_m=self.baseline_m,
            hfov_deg=self.hfov_deg,
            disparity_noise_sigma_px=self.disparity_noise_sigma_px,
            disparity_noise_scale_z=self.disparity_noise_scale_z,
            min_depth_m=self.min_depth_m,
            max_depth_m=self.max_depth_m,
            seed=self._seed,
        )

    # ------------------------------------------------------------------
    # BaseSensor interface
    # ------------------------------------------------------------------

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._left.reset(env_id)
        self._right.reset(env_id)
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Produce a stereo observation from the ideal state.

        Expected keys in *state*:
        - ``"rgb"``       — left-camera ideal image ``(H, W, 3)`` uint8.
        - ``"depth"``     — ideal depth map ``(H, W)`` float32 metres *(optional)*.
        - ``"rgb_right"`` — right-camera ideal image *(optional)*.
        """
        rgb_left_ideal = state.get("rgb")
        if rgb_left_ideal is None:
            self._last_obs = {}
            return self._last_obs

        rgb_left_ideal_arr = np.asarray(rgb_left_ideal, dtype=np.uint8)

        # ---------------------------------------------------------------
        # 1. Depth-based stereo reconstruction
        # ---------------------------------------------------------------
        depth_raw = state.get("depth")
        disparity: FloatArray | None = None
        noisy_depth: FloatArray | None = None
        point_cloud: FloatArray | None = None
        valid_mask: np.ndarray | None = None

        if depth_raw is not None:
            depth_ideal = np.asarray(depth_raw, dtype=np.float32)
            # Compute ideal disparity from depth
            d_ideal = _depth_to_disparity(depth_ideal, self.focal_px, self.baseline_m)
            # Add sub-pixel disparity noise (optionally range-dependent)
            if self.disparity_noise_sigma_px > 0.0 or self.disparity_noise_scale_z > 0.0:
                sigma_disp = self.disparity_noise_sigma_px
                if self.disparity_noise_scale_z > 0.0:
                    # Range-dependent: σ_total(Z) = σ_base + scale_z × Z²
                    depth_clamped = np.clip(depth_ideal, _MIN_DEPTH_M, None)
                    sigma_map = sigma_disp + self.disparity_noise_scale_z * depth_clamped**2
                    noise = (self._rng.normal(0.0, 1.0, d_ideal.shape) * sigma_map).astype(np.float32)
                else:
                    noise = self._rng.normal(0.0, sigma_disp, d_ideal.shape).astype(np.float32)
                d_noisy = np.clip(d_ideal + noise, 0.0, _MAX_DISPARITY_PX).astype(np.float32)
            else:
                d_noisy = d_ideal.copy()
            # Invalidate pixels beyond max_depth / below min_depth
            d_noisy[depth_ideal < self.min_depth_m] = 0.0
            d_noisy[depth_ideal > self.max_depth_m] = 0.0
            disparity = d_noisy
            noisy_depth = _disparity_to_depth(d_noisy, self.focal_px, self.baseline_m)
            valid_mask = (disparity > 0.0) & np.isfinite(noisy_depth) & (noisy_depth > self.min_depth_m)
            point_cloud = _reproject_to_pointcloud(np.where(valid_mask, noisy_depth, 0.0), self.focal_px)

        # ---------------------------------------------------------------
        # 2. Synthesise right view if not provided
        # ---------------------------------------------------------------
        rgb_right_ideal_arr: UInt8Array
        if "rgb_right" in state:
            rgb_right_ideal_arr = np.asarray(state["rgb_right"], dtype=np.uint8)
        elif disparity is not None:
            rgb_right_ideal_arr = _synthesise_right_view(rgb_left_ideal_arr, disparity)
        else:
            # No depth available — use the left image as a degenerate fallback
            rgb_right_ideal_arr = rgb_left_ideal_arr.copy()

        # ---------------------------------------------------------------
        # 3. Apply camera corruption pipeline to both eyes
        # ---------------------------------------------------------------
        # Force the sub-cameras to step even if their internal rate hasn't
        # fired (the stereo model controls the overall gate rate).
        self._left._last_update_time = -1.0
        self._right._last_update_time = -1.0
        left_obs = self._left.step(sim_time, {**state, "rgb": rgb_left_ideal_arr})
        right_obs = self._right.step(sim_time, {**state, "rgb": rgb_right_ideal_arr})

        obs: dict[str, Any] = {
            "rgb_left": left_obs["rgb"],
            "rgb_right": right_obs["rgb"],
            "baseline_m": self.baseline_m,
            "focal_px": self.focal_px,
        }
        if disparity is not None:
            obs["disparity"] = disparity
            obs["depth"] = noisy_depth
            obs["valid_mask"] = valid_mask
            obs["point_cloud"] = point_cloud

        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs


__all__ = ["StereoCameraModel"]
