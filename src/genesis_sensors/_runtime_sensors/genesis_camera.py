"""
Genesis-native camera using Genesis's built-in rendering.

This module wraps Genesis Camera sensors to provide Isaac Sim-quality
RGB, depth, segmentation, and normal rendering via Genesis's
BatchRenderer, Raytracer, or Rasterizer.

Usage
-----
::

    import genesis as gs
    from genesis_sensors import GenesisCamera

    gs.init(backend=gs.gpu)
    scene = gs.Scene(...)

    # Add camera to scene
    cam = scene.add_camera(
        res=(640, 480),
        pos=(3.5, 0.0, 2.5),
        lookat=(0.0, 0.0, 0.5),
        up=(0.0, 0.0, 1.0),
        model=gs.Camera.Model.PINHOLE,
        fov=40,
        aperture=0.0,
        focus_dist=5.0,
        GUI=False,
        spp=256,
        denoise=True,
    )

    # Build scene
    scene.build()

    # Wrap with GenesisCamera for unified API
    genesis_cam = GenesisCamera(camera=cam, name="rgb_camera")

    for event in node:
        scene.step()
        obs = genesis_cam.step(sim_time, {})
        # obs["rgb"]       # RGB image
        # obs["depth"]     # Depth image (meters)
        # obs["segmentation"]  # Segmentation mask
        # obs["normal"]    # Surface normals
"""

from __future__ import annotations

from typing import Any, TYPE_CHECKING

import numpy as np

from .base import BaseSensor

if TYPE_CHECKING:
    from .config import CameraConfig


class GenesisCamera(BaseSensor):
    """
    Wrapper for native Genesis Camera sensor.

    Wraps a Genesis Camera (from ``scene.add_camera()``) to provide
    unified sensor API with realistic camera models (pinhole, thinlens).
    Renders RGB, depth, segmentation, and normals via Genesis's rendering
    pipeline (BatchRenderer, Raytracer, or Rasterizer).

    Parameters
    ----------
    camera:
        The Genesis Camera object returned by scene.add_camera().
    name:
        Human-readable identifier.
    update_rate_hz:
        Camera output rate in Hz.
    enable_rgb:
        Enable RGB rendering.
    enable_depth:
        Enable depth rendering.
    enable_segmentation:
        Enable segmentation rendering.
    enable_normal:
        Enable surface normal rendering.
    depth_noise_sigma_m:
        Depth noise sigma for ToF/structured-light noise model.
    """

    def __init__(
        self,
        camera: Any,
        name: str = "genesis_camera",
        update_rate_hz: float = 30.0,
        enable_rgb: bool = True,
        enable_depth: bool = True,
        enable_segmentation: bool = False,
        enable_normal: bool = False,
        depth_noise_sigma_m: float = 0.0,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self._camera = camera
        self._enable_rgb = bool(enable_rgb)
        self._enable_depth = bool(enable_depth)
        self._enable_segmentation = bool(enable_segmentation)
        self._enable_normal = bool(enable_normal)
        self._depth_noise_sigma = float(depth_noise_sigma_m)
        self._last_obs: dict[str, Any] = {}
        self._last_update_time = -1.0

    @classmethod
    def from_config(cls, config: "CameraConfig") -> "GenesisCamera":
        """Not supported for GenesisCamera - requires live Genesis camera."""
        raise NotImplementedError(
            "GenesisCamera cannot be created from config - requires a live "
            "Genesis camera from scene.add_camera(). Use direct construction."
        )

    def get_config(self) -> "CameraConfig":
        """Not supported for GenesisCamera."""
        raise NotImplementedError(
            "GenesisCamera does not support get_config() - the underlying "
            "Genesis camera is configured via scene.add_camera() at creation."
        )

    def reset(self, env_id: int = 0) -> None:
        """Reset the camera sensor."""
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> dict[str, Any]:
        """
        Render from the Genesis camera.

        Parameters
        ----------
        sim_time:
            Current simulation time (s).
        state:
            Optional state dict (unused for camera).

        Returns
        -------
        dict
            Dictionary containing requested outputs (rgb, depth, segmentation, normal).
        """
        obs: dict[str, Any] = {}

        try:
            rgb_arr, depth_arr, seg_arr, normal_arr = self._camera.render(
                rgb=self._enable_rgb,
                depth=self._enable_depth,
                segmentation=self._enable_segmentation,
                normal=self._enable_normal,
            )

            if self._enable_rgb and rgb_arr is not None:
                if hasattr(rgb_arr, 'cpu'):
                    rgb_arr = rgb_arr.cpu().numpy()
                obs["rgb"] = rgb_arr

            if self._enable_depth and depth_arr is not None:
                if hasattr(depth_arr, 'cpu'):
                    depth_arr = depth_arr.cpu().numpy()

                depth_m = depth_arr
                if self._depth_noise_sigma > 0:
                    noise = np.random.normal(0, self._depth_noise_sigma, depth_m.shape)
                    depth_m = depth_m + noise.astype(np.float32)

                obs["depth"] = depth_m

            if self._enable_segmentation and seg_arr is not None:
                if hasattr(seg_arr, 'cpu'):
                    seg_arr = seg_arr.cpu().numpy()
                obs["segmentation"] = seg_arr

            if self._enable_normal and normal_arr is not None:
                if hasattr(normal_arr, 'cpu'):
                    normal_arr = normal_arr.cpu().numpy()
                obs["normal"] = normal_arr

        except Exception:
            pass

        self._last_obs = obs
        self._mark_updated(sim_time)
        return obs

    def get_observation(self) -> dict[str, Any]:
        """Return the most recent observation."""
        return self._last_obs

    def render_pointcloud(self, world_frame: bool = True) -> tuple[np.ndarray, np.ndarray] | None:
        """
        Render point cloud from camera's depth.

        Parameters
        ----------
        world_frame:
            If True, returns points in world frame; otherwise camera frame.

        Returns
        -------
        tuple of (point_cloud, valid_mask) or None if rendering fails.
        """
        try:
            return self._camera.render_pointcloud(world_frame=world_frame)
        except Exception:
            return None

    @property
    def intrinsics(self) -> np.ndarray:
        """Get camera intrinsic matrix."""
        try:
            return self._camera.intrinsics
        except Exception:
            return np.eye(3)

    @property
    def extrinsics(self) -> np.ndarray:
        """Get camera extrinsic matrix."""
        try:
            return self._camera.extrinsics
        except Exception:
            return np.eye(4)

    @property
    def resolution(self) -> tuple[int, int]:
        """Get camera resolution (width, height)."""
        try:
            return self._camera._res
        except Exception:
            return (640, 480)


__all__ = ["GenesisCamera"]
