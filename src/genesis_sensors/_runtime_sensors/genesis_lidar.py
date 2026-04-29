"""
Genesis-native LiDAR sensor using real scene geometry.

Uses Genesis geometry (vertices + triangles) to compute per-beam ranges
via ray-mesh intersection, then passes the ideal range image through the
realistic :class:`LidarModel` corruption pipeline (noise, dropout,
rain/fog attenuation, beam divergence, multi-return).

Material-aware intensity: uses surface roughness/metallic to compute
Lambertian reflectance for realistic LiDAR intensity simulation.
This brings genesis-sensors closer to or beyond Isaac Sim-quality simulation.

Usage
-----
::

    from genesis import Scene
    from genesis_sensors import GenesisLiDAR

    def geom_getter():
        return GenesisLiDAR.build_geom_from_scene(scene)

    lidar = GenesisLiDAR(
        name="front_lidar",
        update_rate_hz=10.0,
        n_channels=16,
        v_fov_deg=(-15.0, 15.0),
        h_resolution=1800,
        max_range_m=100.0,
        scene_geom_getter=geom_getter,
        seed=42,
    )
    lidar.reset()

    for event in node:
        scene.step()
        state = {"pos": entity.get_pos(), "quat": entity.get_quat()}
        obs = lidar.step(sim_time, state)
        # obs["points"]  # Nx4 x,y,z,intensity
        # obs["range_image"]  # range_image after corruption
"""

from __future__ import annotations

import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
class _Ray:
    """One ray: origin + unit direction."""

    ox: float
    oy: float
    oz: float
    dx: float
    dy: float
    dz: float


@dataclass
class _Geomtri:
    """Triangulated mesh geometry entry for raycasting."""

    verts: np.ndarray
    tri_indices: np.ndarray
    geom_id: int
    roughness: float = 0.5
    metallic: float = 0.0
    base_color: np.ndarray = field(default_factory=lambda: np.array([0.7, 0.7, 0.7], dtype=np.float32))


@dataclass
class _Triangle:
    """One triangle with precomputed data for fast intersection."""

    v0: np.ndarray
    v1: np.ndarray
    v2: np.ndarray
    edge1: np.ndarray
    edge2: np.ndarray
    face_normal: np.ndarray
    area: float
    geom_id: int
    roughness: float
    metallic: float
    base_color: np.ndarray


class GenesisLiDAR(BaseSensor):
    """
    Genesis-native LiDAR sensor using real scene geometry raycasting.

    Material-aware intensity using Lambertian + specular reflectance model:
    ``intensity = base_color * (1 - roughness) * (1 - metallic) * cos(theta)``

    Uses trimesh BVH acceleration when available for Isaac Sim-quality
    ray tracing performance, with fallback to vectorized numpy intersection.

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
    use_trimesh_bvh:
        Use trimesh BVH acceleration for ray intersection (requires trimesh).
        This is the preferred acceleration structure - much faster than cKDTree.
    n_threads:
        Number of threads for parallel ray casting. ``0`` = auto (use CPU count).
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
        use_trimesh_bvh: bool = True,
        n_threads: int = 0,
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
        self._use_trimesh = use_trimesh_bvh and self._trimesh_available()
        self._n_threads = n_threads
        if n_threads <= 0:
            import os
            self._n_threads = os.cpu_count() or 4

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
        self._tri_cache: list[_Triangle] = []
        self._centroids: np.ndarray | None = None
        self._trimesh_meshes: list[Any] = []
        self._geom_tri_info: dict[int, dict[str, Any]] = {}

    @staticmethod
    def _trimesh_available() -> bool:
        try:
            import importlib.util
            return importlib.util.find_spec("trimesh") is not None
        except Exception:
            return False

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
        self._tri_cache.clear()
        self._centroids = None
        self._trimesh_meshes.clear()
        self._geom_tri_info.clear()

    def step(self, sim_time: float, state: dict[str, Any]) -> LidarObservation | dict[str, Any]:
        """
        Compute range image from Genesis geometry and pass through noise model.

        Expected keys in *state*:
        - ``"pos"`` -- (3,) world position.
        - ``"quat"`` -- (4,) quaternion [w, x, y, z].
        - ``"pose"`` -- optional tuple of (pos, quat).
        """
        pos, quat = self._extract_pose(state)
        range_image, intensity_image = self._compute_range_image(pos, quat)

        lidar_state: dict[str, Any] = {
            "range_image": range_image,
            "intensity_image": intensity_image,
        }

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

    @staticmethod
    def _ray_triangle_intersection(
        ray: "_Ray",
        tri: "_Triangle",
    ) -> tuple[float, float]:
        """Moller-Trumbore intersection; returns (range_m, cos_incident_angle)."""
        epsilon = 1e-9

        hx = ray.dy * tri.edge2[2] - ray.dz * tri.edge2[1]
        hy = ray.dz * tri.edge2[0] - ray.dx * tri.edge2[2]
        hz = ray.dx * tri.edge2[1] - ray.dy * tri.edge2[0]
        a_val = float(np.dot(tri.edge1, np.array([hx, hy, hz])))
        if abs(a_val) < epsilon:
            return math.inf, 0.0

        f_val = 1.0 / a_val
        v0 = tri.v0
        sx = ray.ox - v0[0]
        sy = ray.oy - v0[1]
        sz = ray.oz - v0[2]
        u = f_val * float(np.dot(np.array([sx, sy, sz]), np.array([hx, hy, hz])))
        if u < 0.0 or u > 1.0:
            return math.inf, 0.0

        qx = sy * tri.edge1[2] - sz * tri.edge1[1]
        qy = sz * tri.edge1[0] - sx * tri.edge1[2]
        qz = sx * tri.edge1[1] - sy * tri.edge1[0]
        v = f_val * float(np.dot(ray.dx * np.array([qx, qy, qz])))
        if v < 0.0 or u + v > 1.0:
            return math.inf, 0.0

        t_val = f_val * float(np.dot(tri.edge2, np.array([qx, qy, qz])))
        if t_val < epsilon:
            return math.inf, 0.0

        cos_incident = float(np.dot(tri.face_normal, np.array([-ray.dx, -ray.dy, -ray.dz])))
        cos_incident = max(0.0, cos_incident)
        return t_val, cos_incident

    def _build_triangle_cache(self, geoms: list[_Geomtri]) -> None:
        """Precompute triangle data for fast intersection."""
        self._tri_cache.clear()
        self._geom_tri_info.clear()

        for geom in geoms:
            verts = geom.verts
            idx_arr = geom.tri_indices
            if idx_arr.ndim != 2 or idx_arr.shape[1] != 3:
                continue

            roughness = geom.roughness
            metallic = geom.metallic
            base_color = geom.base_color

            tri_start = len(self._tri_cache)
            for i in range(idx_arr.shape[0]):
                i0, i1, i2 = int(idx_arr[i, 0]), int(idx_arr[i, 1]), int(idx_arr[i, 2])
                if i0 < 0 or i1 < 0 or i2 < 0:
                    continue
                if i0 >= len(verts) or i1 >= len(verts) or i2 >= len(verts):
                    continue

                v0 = verts[i0]
                v1 = verts[i1]
                v2 = verts[i2]
                edge1 = v1 - v0
                edge2 = v2 - v0

                nx, ny, nz = np.cross(edge1, edge2)
                norm_val = math.sqrt(nx * nx + ny * ny + nz * nz)
                if norm_val < 1e-12:
                    continue
                face_normal = np.array([nx / norm_val, ny / norm_val, nz / norm_val], dtype=np.float64)

                area = 0.5 * norm_val

                self._tri_cache.append(_Triangle(
                    v0=v0, v1=v1, v2=v2,
                    edge1=edge1, edge2=edge2,
                    face_normal=face_normal,
                    area=area,
                    geom_id=geom.geom_id,
                    roughness=roughness,
                    metallic=metallic,
                    base_color=base_color,
                ))

            tri_end = len(self._tri_cache)
            if tri_end > tri_start:
                self._geom_tri_info[geom.geom_id] = {
                    "start": tri_start,
                    "end": tri_end,
                    "roughness": roughness,
                    "metallic": metallic,
                    "base_color": base_color,
                }

        if self._tri_cache:
            self._centroids = np.array([
                (t.v0 + t.v1 + t.v2) / 3.0 for t in self._tri_cache
            ], dtype=np.float64)

    def _build_trimesh_meshes(self, geoms: list[_Geomtri]) -> None:
        """Build trimesh objects from geometry for BVH raycasting."""
        import trimesh

        self._trimesh_meshes.clear()
        self._geom_tri_info.clear()

        for geom in geoms:
            verts = geom.verts
            idx_arr = geom.tri_indices
            if idx_arr.ndim != 2 or idx_arr.shape[1] != 3:
                continue

            try:
                tri_mesh = trimesh.Trimesh(
                    vertices=verts,
                    faces=idx_arr,
                    process=False,
                )
                tri_mesh.primitive.data["geom_id"] = geom.geom_id
                self._trimesh_meshes.append(tri_mesh)

                self._geom_tri_info[geom.geom_id] = {
                    "roughness": geom.roughness,
                    "metallic": geom.metallic,
                    "base_color": geom.base_color,
                }
            except Exception:
                continue

    def _cast_rays_trimesh_bvh(
        self, rays: list[_Ray], initial_range: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """trimesh BVH-accelerated ray intersection - fast and accurate."""
        n_ch = self.n_channels
        n_az = self.h_resolution
        range_image = initial_range.copy()
        intensity_image = np.zeros((n_ch, n_az), dtype=np.float32)

        if not rays or not self._trimesh_meshes:
            return range_image, intensity_image

        origins = np.array([[r.ox, r.oy, r.oz] for r in rays], dtype=np.float64)
        dirs = np.array([[r.dx, r.dy, r.dz] for r in rays], dtype=np.float64)

        locations_all: list[tuple] = []
        ray_idxs_all: list[int] = []

        for mesh in self._trimesh_meshes:
            geom_id = mesh.primitive.data.get("geom_id", 0)
            tri_info = self._geom_tri_info.get(geom_id, {})

            try:
                locs, idxs = mesh.ray.intersects_location(
                    origins, dirs, return_ray_id=True
                )
                for loc, ray_idx in zip(locs, idxs):
                    locations_all.append((loc, ray_idx, geom_id))
                    ray_idxs_all.append(ray_idx)
            except Exception:
                continue

        for loc, ray_idx, geom_id in locations_all:
            ch = ray_idx // n_az
            az = ray_idx % n_az
            if ch >= n_ch:
                continue

            dist = math.sqrt(
                (loc[0] - rays[ray_idx].ox)**2 +
                (loc[1] - rays[ray_idx].oy)**2 +
                (loc[2] - rays[ray_idx].oz)**2
            )

            if dist < range_image[ch, az] and dist <= self.max_range_m:
                range_image[ch, az] = dist

                tri_info = self._geom_tri_info.get(geom_id, {})
                roughness = tri_info.get("roughness", 0.5)
                metallic = tri_info.get("metallic", 0.0)
                base_color = tri_info.get("base_color", np.array([0.7, 0.7, 0.7]))

                ray_dir = np.array([rays[ray_idx].dx, rays[ray_idx].dy, rays[ray_idx].dz])
                hit_normal = loc[3:6] if len(loc) > 5 else np.array([0.0, 0.0, 1.0])
                cos_incident = max(0.0, float(np.dot(hit_normal, -ray_dir)))

                reflectance = self._compute_reflectance_scalar(
                    roughness, metallic, base_color, cos_incident
                )
                intensity_image[ch, az] = reflectance

        return range_image, intensity_image

    def _cast_rays_vectorized(
        self, rays: list[_Ray], initial_range: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorized ray intersection using numpy batch operations."""
        n_ch = self.n_channels
        n_az = self.h_resolution
        range_image = initial_range.copy()
        intensity_image = np.zeros((n_ch, n_az), dtype=np.float32)

        if not rays or not self._tri_cache:
            return range_image, intensity_image

        origins = np.array([[r.ox, r.oy, r.oz] for r in rays], dtype=np.float64)
        dirs = np.array([[r.dx, r.dy, r.dz] for r in rays], dtype=np.float64)

        n_rays = len(rays)

        tri_v0 = np.array([t.v0 for t in self._tri_cache], dtype=np.float64)
        tri_edge1 = np.array([t.edge1 for t in self._tri_cache], dtype=np.float64)
        tri_edge2 = np.array([t.edge2 for t in self._tri_cache], dtype=np.float64)
        tri_normals = np.array([t.face_normal for t in self._tri_cache], dtype=np.float64)
        tri_roughness = np.array([t.roughness for t in self._tri_cache], dtype=np.float64)
        tri_metallic = np.array([t.metallic for t in self._tri_cache], dtype=np.float64)
        tri_base_color = np.array([t.base_color for t in self._tri_cache], dtype=np.float64)

        batch_size = min(self.ray_batch_size, n_rays)

        for batch_start in range(0, n_rays, batch_size):
            batch_end = min(batch_start + batch_size, n_rays)
            batch_origins = origins[batch_start:batch_end]
            batch_dirs = dirs[batch_start:batch_end]

            h_vecs = np.cross(batch_dirs[:, np.newaxis, :], tri_edge2[np.newaxis, :, :])
            a_vals = np.sum(tri_edge1[np.newaxis, :, :] * h_vecs, axis=2)

            valid = np.abs(a_vals) > 1e-9
            a_inv = np.where(valid, 1.0 / a_vals, 0.0)

            s_vecs = batch_origins[:, np.newaxis, :] - tri_v0[np.newaxis, :, :]
            u_vals = np.sum(s_vecs * h_vecs, axis=2) * a_inv

            q_vecs = np.cross(s_vecs, tri_edge1[np.newaxis, :, :])
            v_vals = np.sum(batch_dirs[:, np.newaxis, :] * q_vecs, axis=2) * a_inv

            t_vals = np.sum(tri_edge2[np.newaxis, :, :] * q_vecs, axis=2) * a_inv

            hit_mask = (u_vals >= 0.0) & (v_vals >= 0.0) & (u_vals + v_vals <= 1.0) & (t_vals > 1e-9)

            for ray_local_idx in range(batch_end - batch_start):
                ray_global_idx = batch_start + ray_local_idx
                ch = ray_global_idx // n_az
                az = ray_global_idx % n_az
                if ch >= n_ch:
                    break

                local_hits = hit_mask[ray_local_idx]
                if not np.any(local_hits):
                    continue

                hit_t = t_vals[ray_local_idx][local_hits]
                hit_tri_idx = np.where(local_hits)[0][np.argmin(hit_t)]

                hit_range = t_vals[ray_local_idx, hit_tri_idx]
                if hit_range < range_image[ch, az] and hit_range <= self.max_range_m:
                    range_image[ch, az] = hit_range

                    cos_incident = max(0.0, float(np.dot(tri_normals[hit_tri_idx], -dirs[ray_global_idx])))

                    reflectance = self._compute_reflectance_scalar(
                        float(tri_roughness[hit_tri_idx]),
                        float(tri_metallic[hit_tri_idx]),
                        tri_base_color[hit_tri_idx],
                        cos_incident,
                    )
                    intensity_image[ch, az] = reflectance

        return range_image, intensity_image

    def _cast_rays_parallel(
        self, rays: list[_Ray], initial_range: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Multi-threaded naive ray intersection."""
        n_ch = self.n_channels
        n_az = self.h_resolution
        range_image = initial_range.copy()
        intensity_image = np.zeros((n_ch, n_az), dtype=np.float32)

        if not rays or not self._tri_cache:
            return range_image, intensity_image

        n_rays = len(rays)
        tri_cache = self._tri_cache
        batch_size = self.ray_batch_size

        def process_batch(batch_info: tuple) -> dict[int, tuple[float, float, _Triangle | None]]:
            start, end = batch_info
            results: dict[int, tuple[float, float, _Triangle | None]] = {}
            for ray_idx in range(start, end):
                ray = rays[ray_idx]
                hit_range = math.inf
                hit_cos = 0.0
                hit_tri: _Triangle | None = None

                for tri in tri_cache:
                    t_val, cos_val = self._ray_triangle_intersection(ray, tri)
                    if t_val < hit_range:
                        hit_range = t_val
                        hit_cos = cos_val
                        hit_tri = tri

                results[ray_idx] = (hit_range, hit_cos, hit_tri)
            return results

        batches = [(i, min(i + batch_size, n_rays)) for i in range(0, n_rays, batch_size)]

        with ThreadPoolExecutor(max_workers=self._n_threads) as executor:
            futures = list(executor.map(process_batch, batches))
            for batch_results in futures:
                for ray_idx, (hit_range, hit_cos, hit_tri) in batch_results.items():
                    ch = ray_idx // n_az
                    az = ray_idx % n_az
                    if ch >= n_ch:
                        break

                    if hit_range < math.inf and hit_range <= self.max_range_m:
                        range_image[ch, az] = float(hit_range)
                        if hit_tri is not None:
                            intensity_image[ch, az] = self._compute_reflectance(hit_tri, hit_cos)

        return range_image, intensity_image

    def _compute_reflectance(self, tri: "_Triangle", cos_incident: float) -> float:
        """Compute Lambertian + specular reflectance for a hit triangle."""
        return self._compute_reflectance_scalar(tri.roughness, tri.metallic, tri.base_color, cos_incident)

    def _compute_reflectance_scalar(
        self, roughness: float, metallic: float, base_color: np.ndarray, cos_incident: float
    ) -> float:
        """Isaac Sim-style intensity = base_color * (1 - roughness) * (1 - metallic) * cos(theta)"""
        diffuse = 1.0 - roughness
        metalness = metallic
        spec = metallic * (1.0 - roughness)

        reflectance = diffuse * (1.0 - metalness) * cos_incident
        reflectance += spec * cos_incident * cos_incident

        base_lum = float(np.dot(base_color, np.array([0.299, 0.587, 0.114])))
        reflectance *= max(0.1, base_lum)

        return float(np.clip(reflectance, 0.0, 1.0))

    def _compute_range_image(
        self, pos: np.ndarray, quat: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
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

        initial_range = np.full((self.n_channels, self.h_resolution), self.max_range_m, dtype=np.float32)

        if self._use_trimesh:
            self._build_trimesh_meshes(geoms)
            if self._trimesh_meshes:
                return self._cast_rays_trimesh_bvh(rays, initial_range)

        self._build_triangle_cache(geoms)
        if self._tri_cache:
            return self._cast_rays_vectorized(rays, initial_range)
        return self._cast_rays_parallel(rays, initial_range)

    def _synthetic_lidar_step(
        self, sim_time: float, state: dict[str, Any]
    ) -> dict[str, Any]:
        """Fallback when no LidarModel is configured."""
        range_image = state.get("range_image")
        if range_image is None:
            return {
                "points": np.empty((0, 4), dtype=np.float32),
                "range_image": np.zeros((self.n_channels, self.h_resolution), dtype=np.float32),
            }

        ri = np.asarray(range_image, dtype=np.float32)
        intensity_image = state.get("intensity_image")
        if intensity_image is None:
            valid_mask = (ri > 0) & (ri < self.max_range_m)
            intensity_image = np.where(valid_mask, np.clip(1.0 / (ri**2 + 1e-6), 0, 1), 0).astype(np.float32)
        else:
            intensity_image = np.asarray(intensity_image, dtype=np.float32)

        n_ch, n_az = ri.shape
        valid_mask = (ri > 0) & (ri < self.max_range_m)

        cos_el = self._cos_elev[:n_ch, :n_az] if self.n_channels == n_ch else self._cos_elev
        sin_el = self._sin_elev[:n_ch, :n_az] if self.n_channels == n_ch else self._sin_elev
        cos_az = self._cos_azim[:n_ch, :n_az] if self.h_resolution == n_az else self._cos_azim
        sin_az = self._sin_azim[:n_ch, :n_az] if self.h_resolution == n_az else self._sin_azim

        x = ri * cos_el * cos_az
        y = ri * cos_el * sin_az
        z = ri * sin_el

        points = np.stack(
            [x[valid_mask], y[valid_mask], z[valid_mask], intensity_image[valid_mask]],
            axis=-1,
        ).astype(np.float32)

        return {"points": points, "range_image": ri}

    @staticmethod
    def build_geom_from_scene(
        scene: Any,
        geom_id_start: int = 1,
        include_visual: bool = True,
        include_collision: bool = True,
        use_trimesh: bool = True,
    ) -> list[_Geomtri]:
        """
        Extract all triangulated mesh geometries from a built Genesis scene.

        Properly extracts vertices and triangles from Genesis scene entities
        using the official API methods, with material properties for
        physically-based intensity computation. Also builds trimesh objects
        for BVH-accelerated raycasting when use_trimesh=True.

        Parameters
        ----------
        scene:
            A built ``genesis.Scene`` instance.
        geom_id_start:
            Starting geom_id for the returned entries.
        include_visual:
            Include visual geometries.
        include_collision:
            Include collision geometries.
        use_trimesh:
            Also build trimesh.Trimesh objects for fast BVH ray intersection.

        Returns
        -------
        list[_Geomtri]
            List of geometry triangles ready for raycasting.
        """
        geoms: list[_Geomtri] = []
        geom_id = geom_id_start

        for entity in scene.entities:
            try:
                if not hasattr(entity, "get_links_pos"):
                    continue

                entity_geoms: list[tuple[np.ndarray, np.ndarray, Any]] = []

                if hasattr(entity, "_geoms") and include_collision:
                    for geom in entity._geoms:
                        try:
                            verts_tensor = geom.get_verts()
                            if verts_tensor is None:
                                continue
                            verts_arr = verts_tensor.cpu().numpy()
                            if verts_arr.ndim != 2 or verts_arr.shape[1] != 3:
                                continue

                            mesh_obj = getattr(geom, "mesh", None)
                            if mesh_obj is not None and hasattr(mesh_obj, "faces"):
                                faces_arr = np.asarray(mesh_obj.faces, dtype=np.int64)
                            elif mesh_obj is not None and hasattr(mesh_obj, "indices"):
                                faces_arr = np.asarray(mesh_obj.indices, dtype=np.int64)
                            else:
                                init_faces = getattr(geom, "init_faces", None)
                                if init_faces is not None:
                                    faces_arr = np.asarray(init_faces, dtype=np.int64)
                                else:
                                    continue

                            surface = getattr(geom, "surface", None)
                            entity_geoms.append((verts_arr, faces_arr, surface))
                        except Exception:
                            continue

                if hasattr(entity, "_vgeoms") and include_visual:
                    for vgeom in entity._vgeoms:
                        try:
                            mesh_obj = getattr(vgeom, "mesh", None)
                            if mesh_obj is None:
                                continue
                            verts = getattr(mesh_obj, "vertices", None) or getattr(mesh_obj, "verts", None)
                            if verts is None:
                                continue
                            verts_arr = np.asarray(verts, dtype=np.float64)
                            if verts_arr.ndim != 2 or verts_arr.shape[1] != 3:
                                continue

                            faces_arr = None
                            for attr in ("faces", "indices", "tri_indices"):
                                faces = getattr(mesh_obj, attr, None)
                                if faces is not None:
                                    faces_arr = np.asarray(faces, dtype=np.int64)
                                    break

                            if faces_arr is None:
                                init_vfaces = getattr(vgeom, "init_vfaces", None)
                                if init_vfaces is not None:
                                    faces_arr = np.asarray(init_vfaces, dtype=np.int64)

                            if faces_arr is None:
                                continue

                            surface = getattr(mesh_obj, "surface", None) or getattr(vgeom, "surface", None)
                            entity_geoms.append((verts_arr, faces_arr, surface))
                        except Exception:
                            continue

                for verts_arr, faces_arr, surface in entity_geoms:
                    if faces_arr.ndim != 2 or faces_arr.shape[1] != 3:
                        continue

                    roughness = 0.5
                    metallic = 0.0
                    base_color = np.array([0.7, 0.7, 0.7], dtype=np.float32)

                    if surface is not None:
                        roughness = float(getattr(surface, "roughness", 0.5) or 0.5)
                        metallic = float(getattr(surface, "metallic", 0.0) or 0.0)
                        color = getattr(surface, "color", None)
                        if color is not None:
                            base_color = np.array(color, dtype=np.float32)
                        elif hasattr(surface, "get_texture") and surface.get_texture() is not None:
                            tex = surface.get_texture()
                            if hasattr(tex, "color"):
                                base_color = np.array(tex.color, dtype=np.float32)

                    geoms.append(_Geomtri(
                        verts=verts_arr,
                        tri_indices=faces_arr,
                        geom_id=geom_id,
                        roughness=roughness,
                        metallic=metallic,
                        base_color=base_color,
                    ))
                    geom_id += 1

            except Exception:
                continue

        return geoms


__all__ = ["GenesisLiDAR", "_Geomtri"]
