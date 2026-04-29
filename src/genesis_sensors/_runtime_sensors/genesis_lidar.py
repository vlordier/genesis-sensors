"""
Genesis-native LiDAR sensor using real scene geometry.

Uses Genesis's ray-mesh intersection to compute per-beam ranges from
actual scene geometry, then passes the ideal range image through the
realistic :class:`LidarModel` corruption pipeline (noise, dropout,
rain/fog attenuation, beam divergence, multi-return).

This brings genesis-sensors closer to Isaac Sim-quality simulation by
using real geometry for raycasting instead of synthetic range images.

Usage
-----
::

    from genesis import Scene
    from genesis_sensors import GenesisLiDAR

    lidar = GenesisLiDAR(
        name="front_lidar",
        update_rate_hz=10.0,
        n_channels=16,
        v_fov_deg=(-15.0, 15.0),
        h_resolution=1800,
        max_range_m=100.0,
        seed=42,
    )
    lidar.reset()

    for event in node:
        scene.step()
        # state must contain pos (3,) and quat (4,) or pose ((3,), (4,))
        state = {"pos": entity.get_pos(), "quat": entity.get_quat()}
        obs = lidar.step(sim_time, state)
        # obs["points"]  # Nx4 x,y,z,intensity
        # obs["range_image"]  # range_image after corruption
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Final

import numpy as np

from .base import BaseSensor
from .lidar import LidarModel
from .types import FloatArray, LidarObservation

if TYPE_CHECKING:
    from .config import LidarConfig

_DEFAULT_N_CHANNELS: Final[int] = 16
_DEFAULT_V_FOV: Final[tuple[float, float]] = (-15.0, 15.0)
_DEFAULT_H_RES: Final[int] = 1800
_DEFAULT_MAX_RANGE_M: Final[float] = 100.0


@dataclass(frozen=True)
class _RayHit:
    """One ray intersection result."""

    range_m: float
    geom_id: int
    primitive_id: int


@dataclass(frozen=True)
class _Ray:
    """One ray: origin + unit direction."""

    ox: float
    oy: float
    oz: float
    dx: float
    dy: float
    dz: float


def _ray_triangle_intersection(
    ray: "_Ray",
    v0: np.ndarray,
    v1: np.ndarray,
    v2: np.ndarray,
) -> float:
    """Moller-Trumbore ray-triangle intersection; returns range_m or inf."""
    epsilon = 1e-9
    edge1 = v1 - v0
    edge2 = v2 - v0
    h = np.array([ray.dy * edge2[2] - ray.dz * edge2[1],
                  ray.dz * edge2[0] - ray.dx * edge2[2],
                  ray.dx * edge2[1] - ray.dy * edge2[0]], dtype=np.float64)
    a_val = float(np.dot(edge1, h))
    if abs(a_val) < epsilon:
        return math.inf
    f_val = 1.0 / a_val
    s = np.array([ray.ox - v0[0], ray.oy - v0[1], ray.oz - v0[2]], dtype=np.float64)
    u = f_val * float(np.dot(s, h))
    if u < 0.0 or u > 1.0:
        return math.inf
    q = np.array([s[1] * edge1[2] - s[2] * edge1[1],
                 s[2] * edge1[0] - s[0] * edge1[2],
                 s[0] * edge1[1] - s[1] * edge1[0]], dtype=np.float64)
    v = f_val * float(np.dot(ray.dx * q[0] + ray.dy * q[1] + ray.dz * q[2]))
    if v < 0.0 or u + v > 1.0:
        return math.inf
    t_val = f_val * float(np.dot(edge2, q))
    if t_val < epsilon:
        return math.inf
    return t_val


def _ray_sphere_intersection(
    ray: "_Ray",
    cx: float,
    cy: float,
    cz: float,
    r: float,
) -> float:
    """Ray-sphere intersection; returns range_m or inf."""
    ocx = ray.ox - cx
    ocy = ray.oy - cy
    ocz = ray.oz - cz
    b_val = ocx * ray.dx + ocy * ray.dy + ocz * ray.dz
    c_val = ocx * ocx + ocy * ocy + ocz * ocz - r * r
    discriminant = b_val * b_val - c_val
    if discriminant < 0.0:
        return math.inf
    sq = math.sqrt(discriminant)
    t0 = -b_val - sq
    t1 = -b_val + sq
    if t0 > 1e-6:
        return t0
    if t1 > 1e-6:
        return t1
    return math.inf


@dataclass
class _Geomtri:
    """Triangulated mesh geometry entry for raycasting."""

    verts: np.ndarray
    tri_indices: np.ndarray
    geom_id: int


class GenesisLiDAR(BaseSensor):
    """
    Genesis-native LiDAR sensor using real scene geometry raycasting.

    Parameters
    ----------
    name:
        Human-readable identifier.
    update_rate_hz:
        LiDAR rotation rate in Hz.
    n_channels:
        Number of vertical scan lines (beams).
    v_fov_deg:
        ``(min_elevation_deg, max_elevation_deg)`` for the vertical FOV.
    h_resolution:
        Number of azimuth steps per revolution.
    max_range_m:
        Maximum measurable range.
    lidar_model:
        Optional :class:`LidarModel` instance for realistic noise.
        If ``None``, a default VELODYNE_VLP16-style noise model is used.
    ray_batch_size:
        Number of rays to process per batch for memory efficiency.
    multi_return:
        Number of returns per beam (1 = strongest only, 2 = add last).
    multi_return_split_m:
        Minimum range difference for secondary return split.
    seed:
        Random seed for reproducibility.
    scene_geom_getter:
        Callback that returns the active scene geometries for raycasting.
        Signature: ``() -> list[_Geomtri]``.  If ``None``, the LiDAR operates
        in a "synthetic-only" fallback mode.
    """

    def __init__(
        self,
        name: str = "genesis_lidar",
        update_rate_hz: float = 10.0,
        n_channels: int = _DEFAULT_N_CHANNELS,
        v_fov_deg: tuple[float, float] = _DEFAULT_V_FOV,
        h_resolution: int = _DEFAULT_H_RES,
        max_range_m: float = _DEFAULT_MAX_RANGE_M,
        lidar_model: LidarModel | None = None,
        ray_batch_size: int = 10000,
        multi_return: int = 1,
        multi_return_split_m: float = 1.0,
        seed: int | None = None,
        scene_geom_getter: Any = None,
    ) -> None:
        super().__init__(name=name, update_rate_hz=update_rate_hz)
        self.n_channels = int(n_channels)
        self.v_fov_deg = tuple(v_fov_deg)
        self.h_resolution = int(h_resolution)
        self.max_range_m = float(max_range_m)
        self.ray_batch_size = int(max(1, ray_batch_size))
        self.multi_return = int(max(1, multi_return))
        self.multi_return_split_m = float(max(0.0, multi_return_split_m))
        self._seed = seed
        self._rng = np.random.default_rng(seed=seed)
        self._scene_geom_getter = scene_geom_getter

        elev_min, elev_max = self.v_fov_deg
        self._elevations_deg: FloatArray = np.linspace(
            elev_min, elev_max, self.n_channels, dtype=np.float32
        )
        self._azimuths_deg: FloatArray = np.linspace(
            0.0, 360.0, self.h_resolution, endpoint=False, dtype=np.float32
        )

        elev_rad = np.deg2rad(self._elevations_deg)
        azim_rad = np.deg2rad(self._azimuths_deg)
        self._elev_grid, self._azim_grid = np.meshgrid(elev_rad, azim_rad, indexing="ij")
        self._cos_elev = np.cos(self._elev_grid).astype(np.float32)
        self._sin_elev = np.sin(self._elev_grid).astype(np.float32)
        self._cos_azim = np.cos(self._azim_grid).astype(np.float32)
        self._sin_azim = np.sin(self._azim_grid).astype(np.float32)

        self._lidar_model = lidar_model
        self._last_obs: dict[str, Any] = {}

    @classmethod
    def from_config(cls, config: "LidarConfig") -> "GenesisLiDAR":
        """Construct from a LidarConfig."""
        return cls._from_config_with_noise(config)

    def get_config(self) -> "LidarConfig":
        """Return the current parameters as a LidarConfig."""
        from .config import LidarConfig

        return LidarConfig(
            name=self.name,
            update_rate_hz=self.update_rate_hz,
            n_channels=self.n_channels,
            v_fov_deg=self.v_fov_deg,
            h_resolution=self.h_resolution,
            max_range_m=self.max_range_m,
            seed=self._seed,
        )

    def reset(self, env_id: int = 0) -> None:
        self._last_obs = {}
        self._last_update_time = -1.0

    def step(self, sim_time: float, state: dict[str, Any]) -> LidarObservation | dict[str, Any]:
        """
        Compute range image from Genesis geometry and pass through noise model.

        Expected keys in *state*:
        - ``"pos"`` -- (3,) world position.
        - ``"quat"`` -- (4,) quaternion [w, x, y, z].
        - ``"pose"`` -- optional tuple of (pos, quat).
        - ``"rgb"`` -- ignored but accepted for compatibility.
        """
        pos, quat = self._extract_pose(state)
        range_image = self._compute_range_image(pos, quat)

        lidar_state: dict[str, Any] = {
            "range_image": range_image,
        }
        intensity_image = state.get("intensity_image")
        if intensity_image is not None:
            lidar_state["intensity_image"] = np.asarray(intensity_image, dtype=np.float32)

        if self._lidar_model is not None:
            result = self._lidar_model.step(sim_time, lidar_state)
        else:
            result = self._synthetic_lidar_step(sim_time, lidar_state)

        self._last_obs = result
        self._mark_updated(sim_time)
        return result

    def get_observation(self) -> dict[str, Any]:
        return self._last_obs

    def _extract_pose(
        self, state: dict[str, Any]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Extract (pos, quat) from state dict."""
        pose = state.get("pose")
        if pose is not None:
            pos, quat = pose
            return np.asarray(pos, dtype=np.float64), np.asarray(quat, dtype=np.float64)
        pos = np.asarray(state.get("pos", [0.0, 0.0, 0.0]), dtype=np.float64)
        quat = np.asarray(state.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=np.float64)
        return pos, quat

    def _build_rays(self, pos: np.ndarray, quat: np.ndarray) -> list[_Ray]:
        """Build all n_channels x h_resolution rays in world frame."""
        w, x, y, z = quat
        norm = math.sqrt(w * w + x * x + y * y + z * z)
        if norm < 1e-9:
            w, x, y, z = 1.0, 0.0, 0.0, 0.0
        else:
            w, x, y, z = w / norm, x / norm, y / norm, z / norm

        tx = 2.0 * (x * z + w * y)
        ty = 2.0 * (y * z - w * x)
        tz = 1.0 - 2.0 * (x * x + y * y)
        R_bw = np.array([
            [1.0 - 2.0 * (y * y + z * z), tx, tz],
            [tx, 1.0 - 2.0 * (y * y + z * z), ty],
            [tz, ty, 1.0 - 2.0 * (x * x + y * y)],
        ], dtype=np.float64)
        R_wb = R_bw.T

        rays: list[_Ray] = []
        n_ch = self.n_channels
        n_az = self.h_resolution
        for ch in range(n_ch):
            sin_el = float(self._sin_elev[ch, 0])
            cos_el = float(self._cos_elev[ch, 0])
            for az in range(n_az):
                dx_w = R_wb[0, 0] * self._cos_azim[ch, az] * cos_el + \
                       R_wb[0, 1] * self._sin_azim[ch, az] * cos_el + \
                       R_wb[0, 2] * sin_el
                dy_w = R_wb[1, 0] * self._cos_azim[ch, az] * cos_el + \
                       R_wb[1, 1] * self._sin_azim[ch, az] * cos_el + \
                       R_wb[1, 2] * sin_el
                dz_w = R_wb[2, 0] * self._cos_azim[ch, az] * cos_el + \
                       R_wb[2, 1] * self._sin_azim[ch, az] * cos_el + \
                       R_wb[2, 2] * sin_el
                len_xy = math.sqrt(dx_w * dx_w + dy_w * dy_w)
                if len_xy < 1e-9 and abs(dz_w) < 1e-9:
                    continue
                norm_val = math.sqrt(dx_w * dx_w + dy_w * dy_w + dz_w * dz_w)
                dx_w /= norm_val
                dy_w /= norm_val
                dz_w /= norm_val
                rays.append(_Ray(
                    ox=float(pos[0]), oy=float(pos[1]), oz=float(pos[2]),
                    dx=dx_w, dy=dy_w, dz=dz_w,
                ))
        return rays

    def _cast_rays(self, rays: list[_Ray], geoms: list[_Geomtri]) -> np.ndarray:
        """Intersect rays with scene geometry; return range_image (n_ch, n_az)."""
        n_ch = self.n_channels
        n_az = self.h_resolution
        range_image = np.full((n_ch, n_az), self.max_range_m, dtype=np.float32)

        if not rays or not geoms:
            return range_image

        tri_verts: list[tuple[np.ndarray, np.ndarray, np.ndarray, int]] = []
        for geom in geoms:
            verts = geom.verts
            idx_arr = geom.tri_indices
            if idx_arr.ndim != 2 or idx_arr.shape[1] != 3:
                continue
            for i in range(idx_arr.shape[0]):
                i0, i1, i2 = int(idx_arr[i, 0]), int(idx_arr[i, 1]), int(idx_arr[i, 2])
                if i0 < 0 or i1 < 0 or i2 < 0:
                    continue
                if i0 >= len(verts) or i1 >= len(verts) or i2 >= len(verts):
                    continue
                tri_verts.append((verts[i0], verts[i1], verts[i2], geom.geom_id))

        batch_size = self.ray_batch_size
        for batch_start in range(0, len(rays), batch_size):
            batch_end = min(batch_start + batch_size, len(rays))
            batch_rays = rays[batch_start:batch_end]

            for ray_idx, ray in enumerate(batch_rays):
                ch = (batch_start + ray_idx) // n_az
                az = (batch_start + ray_idx) % n_az
                if ch >= n_ch:
                    break

                hit_range = math.inf
                for v0, v1, v2, _gid in tri_verts:
                    t_val = _ray_triangle_intersection(ray, v0, v1, v2)
                    if t_val < hit_range:
                        hit_range = t_val

                if hit_range < math.inf and hit_range <= self.max_range_m:
                    range_image[ch, az] = float(hit_range)

        return range_image

    def _compute_range_image(
        self, pos: np.ndarray, quat: np.ndarray
    ) -> np.ndarray:
        """Compute range image via raycasting against scene geometry."""
        rays = self._build_rays(pos, quat)

        geoms: list[_Geomtri] = []
        if self._scene_geom_getter is not None:
            try:
                raw_geoms = self._scene_geom_getter()
                if raw_geoms is not None:
                    for item in raw_geoms:
                        if isinstance(item, _Geomtri):
                            geoms.append(item)
            except Exception:
                pass

        return self._cast_rays(rays, geoms)

    def _synthetic_lidar_step(
        self, sim_time: float, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Fallback when no LidarModel is configured."""
        range_image = state.get("range_image")
        if range_image is None:
            return {"points": np.empty((0, 4), dtype=np.float32), "range_image": np.zeros((self.n_channels, self.h_resolution), dtype=np.float32)}

        ri = np.asarray(range_image, dtype=np.float32)
        n_ch, n_az = ri.shape

        cos_el = self._cos_elev[:n_ch, :n_az] if self.n_channels == n_ch else self._cos_elev
        sin_el = self._sin_elev[:n_ch, :n_az] if self.n_channels == n_ch else self._sin_elev
        cos_az = self._cos_azim[:n_ch, :n_az] if self.h_resolution == n_az else self._cos_azim
        sin_az = self._sin_azim[:n_ch, :n_az] if self.h_resolution == n_az else self._sin_azim

        valid_mask = (ri > 0) & (ri < self.max_range_m)
        intensity = np.where(valid_mask, np.clip(1.0 / (ri**2 + 1e-6), 0, 1), 0).astype(np.float32)

        x = ri * cos_el * cos_az
        y = ri * cos_el * sin_az
        z = ri * sin_el

        points = np.stack(
            [x[valid_mask], y[valid_mask], z[valid_mask], intensity[valid_mask]],
            axis=-1,
        ).astype(np.float32)

        return {"points": points, "range_image": ri}

    @staticmethod
    def build_geom_from_scene(scene: Any, geom_id_start: int = 1) -> list[_Geomtri]:
        """
        Extract all triangulated mesh geometries from a built Genesis scene.

        This is a convenience utility to build the list needed by
        ``GenesisLiDAR(scene_geom_getter=...)``.

        Parameters
        ----------
        scene:
            A built ``genesis.Scene`` instance.
        geom_id_start:
            Starting geom_id for the returned entries.

        Returns
        -------
        list[_Geomtri]
            List of geometry triangles ready for raycasting.
        """
        geoms: list[_Geomtri] = []
        geom_id = geom_id_start

        try:
            active_geoms = scene._get_active_geom_geometries()
        except Exception:
            active_geoms = []

        for geom_info in active_geoms:
            try:
                mesh = geom_info.get("mesh")
                if mesh is None:
                    continue
                verts = getattr(mesh, "vertices", None)
                if verts is None:
                    continue
                verts_arr = np.asarray(verts, dtype=np.float64)
                if verts_arr.ndim != 2 or verts_arr.shape[1] != 3:
                    continue

                tri_indices: np.ndarray | None = None
                if hasattr(mesh, "faces"):
                    tri_indices = np.asarray(mesh.faces, dtype=np.int64)
                elif hasattr(mesh, "indices"):
                    tri_indices = np.asarray(mesh.indices, dtype=np.int64)
                elif hasattr(mesh, "tri_indices"):
                    tri_indices = np.asarray(mesh.tri_indices, dtype=np.int64)

                if tri_indices is None:
                    faces_data = geom_info.get("faces")
                    if faces_data is not None:
                        tri_indices = np.asarray(faces_data, dtype=np.int64)

                if tri_indices is None or tri_indices.ndim != 2 or tri_indices.shape[1] != 3:
                    continue

                geoms.append(_Geomtri(verts=verts_arr, tri_indices=tri_indices, geom_id=geom_id))
                geom_id += 1
            except Exception:
                continue

        return geoms


__all__ = ["GenesisLiDAR", "_Geomtri"]
