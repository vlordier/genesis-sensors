"""Headless synthetic state builders for exercising the upstream sensor stack."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ._compat import ThermalCameraModel

_SKY_ENTITY_ID = int(getattr(ThermalCameraModel, "SKY_ENTITY_ID", -1))

DEFAULT_DT = 0.05
DEFAULT_TOTAL_FRAMES = 240
DEFAULT_RESOLUTION = (96, 72)
DEFAULT_LIDAR_SHAPE = (8, 64)
DEFAULT_TOF_SHAPE = (8, 8)
GNSS_ORIGIN_LLH = (37.4275, -122.1697, 30.0)


class ScenarioPhase(str, Enum):
    """Named rollout phases used by the synthetic-state generator."""

    TAKEOFF = "takeoff"
    CRUISE = "cruise"
    URBAN_CANYON = "urban_canyon"
    RAIN_BURST = "rain_burst"
    SIGNAL_RECOVERY = "signal_recovery"


@dataclass(frozen=True, slots=True)
class ScenarioWindow:
    """A single progress interval mapped to a named scenario phase."""

    phase: ScenarioPhase
    start: float
    end: float

    def contains(self, progress: float) -> bool:
        return self.start <= progress < self.end

    @property
    def duration(self) -> float:
        """Return the width of the progress interval."""
        return self.end - self.start


@dataclass(frozen=True, slots=True)
class SyntheticScenarioConfig:
    """Validated configuration for the synthetic-state and rollout helpers."""

    dt: float = DEFAULT_DT
    total_frames: int = DEFAULT_TOTAL_FRAMES
    resolution: tuple[int, int] = DEFAULT_RESOLUTION
    lidar_shape: tuple[int, int] = DEFAULT_LIDAR_SHAPE
    tof_shape: tuple[int, int] = DEFAULT_TOF_SHAPE

    def validated(self) -> "SyntheticScenarioConfig":
        """Return a normalized config with validated positive shapes and sizes."""
        if self.dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.total_frames <= 0:
            raise ValueError(f"total_frames must be > 0, got {self.total_frames}")
        return SyntheticScenarioConfig(
            dt=float(self.dt),
            total_frames=int(self.total_frames),
            resolution=_validate_positive_shape("resolution", self.resolution),
            lidar_shape=_validate_positive_shape("lidar_shape", self.lidar_shape),
            tof_shape=_validate_positive_shape("tof_shape", self.tof_shape),
        )

    def progress_for_frame(self, frame_idx: int) -> float:
        """Compute rollout progress for a frame index using the configured horizon."""
        return float(np.clip(int(frame_idx) / max(self.total_frames - 1, 1), 0.0, 1.0))

    def phase_for_frame(self, frame_idx: int) -> ScenarioPhase:
        """Return the named scenario phase for a frame index."""
        return _phase_from_progress(self.progress_for_frame(frame_idx))


@dataclass(frozen=True, slots=True)
class SyntheticRolloutSummary:
    """Compact metadata describing a generated synthetic rollout."""

    frame_count: int
    dt: float
    duration_s: float
    resolution: tuple[int, int]
    lidar_shape: tuple[int, int]
    tof_shape: tuple[int, int]
    phase_counts: dict[str, int]
    sensor_keys: tuple[str, ...]

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly rollout summary."""
        return {
            "frame_count": self.frame_count,
            "dt": self.dt,
            "duration_s": round(self.duration_s, 6),
            "resolution": list(self.resolution),
            "lidar_shape": list(self.lidar_shape),
            "tof_shape": list(self.tof_shape),
            "phase_counts": dict(self.phase_counts),
            "sensor_keys": list(self.sensor_keys),
        }

    def has_sensor_key(self, key: str) -> bool:
        """Check whether the rollout observations expose a named state key."""
        return key in self.sensor_keys


_SCENARIO_PHASES = (
    ScenarioWindow(ScenarioPhase.TAKEOFF, 0.00, 0.20),
    ScenarioWindow(ScenarioPhase.CRUISE, 0.20, 0.45),
    ScenarioWindow(ScenarioPhase.URBAN_CANYON, 0.45, 0.70),
    ScenarioWindow(ScenarioPhase.RAIN_BURST, 0.70, 0.88),
    ScenarioWindow(ScenarioPhase.SIGNAL_RECOVERY, 0.88, 1.01),
)


def _validate_positive_shape(name: str, shape: tuple[int, int]) -> tuple[int, int]:
    """Validate two-dimensional integer shapes used by the synthetic helpers."""
    if len(shape) != 2:
        raise ValueError(f"{name} must contain exactly two integers, got {shape!r}")
    dim0, dim1 = (int(shape[0]), int(shape[1]))
    if dim0 <= 0 or dim1 <= 0:
        raise ValueError(f"{name} dimensions must be > 0, got {shape!r}")
    return dim0, dim1


def list_scenario_windows() -> tuple[ScenarioWindow, ...]:
    """Return the ordered synthetic scenario windows used by the rollout helpers."""
    return _SCENARIO_PHASES


def _phase_from_progress(progress: float) -> ScenarioPhase:
    """Resolve a bounded rollout progress value to a typed scenario phase."""
    bounded_progress = max(0.0, float(progress))
    for window in _SCENARIO_PHASES:
        if window.contains(bounded_progress):
            return window.phase
    return _SCENARIO_PHASES[-1].phase


def get_scenario_phase(progress: float) -> str:
    """Map rollout progress to a human-readable scenario phase.

    Negative progress values are clamped to the first phase, while values above
    the final interval resolve to ``"signal_recovery"``.
    """
    return _phase_from_progress(progress).value


def make_synthetic_sensor_state(
    frame_idx: int,
    *,
    dt: float = DEFAULT_DT,
    total_frames: int = DEFAULT_TOTAL_FRAMES,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    lidar_shape: tuple[int, int] = DEFAULT_LIDAR_SHAPE,
    tof_shape: tuple[int, int] = DEFAULT_TOF_SHAPE,
) -> dict[str, Any]:
    """Create one rich synthetic state compatible with the upstream `genesis.sensors` models."""
    config = SyntheticScenarioConfig(
        dt=dt,
        total_frames=total_frames,
        resolution=resolution,
        lidar_shape=lidar_shape,
        tof_shape=tof_shape,
    ).validated()

    frame_idx = int(frame_idx)
    width, height = config.resolution
    lidar_channels, lidar_res = config.lidar_shape

    sim_time = frame_idx * config.dt
    progress = config.progress_for_frame(frame_idx)
    scenario_phase = config.phase_for_frame(frame_idx)
    phase_name = scenario_phase.value

    x = np.linspace(0.0, 1.0, width, dtype=np.float32)
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    pos = np.array(
        [
            2.0 + 0.10 * frame_idx,
            0.9 * np.sin(0.45 * sim_time),
            1.8 + 0.25 * np.cos(0.25 * sim_time),
        ],
        dtype=np.float64,
    )
    vel = np.array(
        [
            0.10,
            0.4 * np.cos(0.45 * sim_time),
            -0.06 * np.sin(0.25 * sim_time),
        ],
        dtype=np.float64,
    )
    lin_acc = np.array(
        [0.02 * np.sin(0.9 * sim_time), -0.08 * np.cos(0.6 * sim_time), 0.12 + 0.02 * np.sin(sim_time)],
        dtype=np.float64,
    )
    ang_vel = np.array(
        [0.08 * np.sin(0.3 * sim_time), 0.12 + 0.03 * np.cos(0.7 * sim_time), 0.28],
        dtype=np.float64,
    )
    speed = float(np.linalg.norm(vel))
    wave_phase = float(0.45 * sim_time + 0.17 * frame_idx)

    glow = np.clip(1.0 - 2.2 * np.sqrt((xx - 0.52) ** 2 + (yy - 0.48) ** 2), 0.0, 1.0)
    stripes = 0.18 * np.sin(6.0 * np.pi * xx + wave_phase) + 0.12 * np.cos(5.0 * np.pi * yy - 0.8 * wave_phase)
    gray = np.clip(0.12 + 0.32 * xx + 0.24 * yy + 0.38 * glow + stripes, 0.0, 1.0).astype(np.float32)

    box_w = max(6, width // 7)
    box_h = max(6, height // 6)
    box_x = int((0.18 * width + 0.28 * width * np.sin(0.31 * wave_phase)) % max(width - box_w, 1))
    box_y = int(np.clip(0.22 * height + 0.16 * height * np.cos(0.23 * wave_phase), 0, max(height - box_h, 0)))
    gray[box_y : box_y + box_h, box_x : box_x + box_w] = 1.0

    rgb = np.stack(
        [
            np.clip(gray + 0.16 * np.sin(2.0 * np.pi * yy + 0.35 * wave_phase), 0.0, 1.0),
            np.clip(0.25 + 0.7 * np.flipud(gray), 0.0, 1.0),
            np.clip(1.0 - 0.55 * gray + 0.18 * np.cos(2.0 * np.pi * xx - 0.2 * wave_phase), 0.0, 1.0),
        ],
        axis=-1,
    )
    rgb_uint8 = np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)

    disparity_shift = max(1, min(4, int(round(2.0 + speed))))
    rgb_right = np.roll(rgb_uint8, -disparity_shift, axis=1)
    rgb_right[:, -disparity_shift:] = rgb_uint8[:, -1:]

    depth = np.clip(float(pos[2]) + 1.4 * xx + 0.35 * yy - 0.28 * glow, 0.15, 8.0).astype(np.float32)

    seg = np.full((height, width), _SKY_ENTITY_ID, dtype=np.int32)
    seg[2:-2, 2:-2] = 1
    seg[box_y : box_y + box_h, box_x : box_x + box_w] = 2
    hot_mask = ((xx - (0.25 + 0.08 * np.sin(wave_phase))) ** 2 + (yy - 0.72) ** 2) < 0.01
    seg[hot_mask] = 3

    az = np.linspace(-1.0, 1.0, lidar_res, dtype=np.float32)[None, :]
    channel = np.linspace(-1.0, 1.0, lidar_channels, dtype=np.float32).reshape(-1, 1)
    moving_obstacle = np.exp(-((az - 0.55 * np.sin(0.7 * sim_time + 0.2)) ** 2) / 0.03)
    range_image = 5.5 + 1.2 * az + 0.7 * channel - 1.3 * moving_obstacle
    range_image += 0.15 * np.sin(wave_phase + np.pi * channel)
    range_image[:, :3] = 0.0
    range_image = np.clip(range_image, 0.0, 12.0).astype(np.float32)
    intensity_image = np.clip(1.1 - range_image / 12.0 + 0.05 * moving_obstacle, 0.0, 1.0).astype(np.float32)

    obstruction = float(np.clip(0.15 + 0.3 * (0.5 + 0.5 * np.sin(0.18 * frame_idx)), 0.0, 0.85))
    rain_rate = float(1.0 + 3.0 * (0.5 + 0.5 * np.sin(0.11 * frame_idx)))
    cloud_cover = float(np.clip(0.20 + 0.45 * (0.5 + 0.5 * np.cos(0.09 * frame_idx)), 0.0, 0.95))
    ambient_temp_c = float(21.5 - 0.35 * pos[2] + 2.2 * np.sin(0.12 * frame_idx))
    relative_humidity_pct = float(np.clip(48.0 + 6.0 * rain_rate + 18.0 * cloud_cover, 18.0, 98.0))
    illuminance_lux = float(np.clip(42_000.0 * (1.0 - 0.75 * cloud_cover) * (1.0 - 0.55 * obstruction), 80.0, 95_000.0))
    wind_ms = np.array([1.2, 0.2 * np.cos(sim_time), 0.0], dtype=np.float64)
    water_current_ms = np.array(
        [0.35 + 0.08 * np.sin(0.21 * frame_idx), -0.12 + 0.05 * np.cos(0.17 * frame_idx), 0.0],
        dtype=np.float64,
    )
    current_layers = [
        {"depth_m": 1.5, "vel": water_current_ms + np.array([-0.04, 0.02, 0.0], dtype=np.float64)},
        {"depth_m": 4.0, "vel": water_current_ms + np.array([0.03, -0.01, 0.0], dtype=np.float64)},
        {"depth_m": 8.0, "vel": water_current_ms + np.array([0.10, -0.04, 0.0], dtype=np.float64)},
    ]
    gas_sources = [
        {
            "pos": np.array([pos[0] + 1.2, pos[1] + 0.25 * np.sin(wave_phase), 0.0], dtype=np.float64),
            "peak_ppm": 1200.0,
            "sigma_m": 0.9,
        }
    ]
    uwb_anchors = [
        {"id": "dock_nw", "pos": np.array([0.0, 0.0, 0.0], dtype=np.float64), "los": obstruction < 0.85},
        {"id": "dock_ne", "pos": np.array([12.0, 0.0, 0.0], dtype=np.float64), "los": True},
        {"id": "dock_sw", "pos": np.array([0.0, 12.0, 0.0], dtype=np.float64), "los": True},
        {"id": "dock_top", "pos": np.array([0.0, 0.0, 6.0], dtype=np.float64), "los": True},
    ]
    radar_targets = [
        {
            "id": "lead_vehicle",
            "pos": pos + np.array([10.0, 1.5 * np.sin(0.35 * sim_time), -0.2], dtype=np.float64),
            "vel": np.array([-0.8, 0.2 * np.cos(0.35 * sim_time), 0.0], dtype=np.float64),
            "rcs_dbsm": 14.0,
        },
        {
            "id": "side_obstacle",
            "pos": pos + np.array([7.0, -3.0 + 0.8 * np.cos(0.22 * sim_time), 0.4], dtype=np.float64),
            "vel": np.array([0.0, 0.15, 0.0], dtype=np.float64),
            "rcs_dbsm": 8.0,
        },
    ]
    ultrasonic_ranges = {
        "front_left": max(0.08, 0.85 + 0.20 * np.sin(0.50 * sim_time)),
        "front_right": max(0.08, 1.10 + 0.18 * np.cos(0.40 * sim_time)),
        "left": max(0.08, 0.65 + 0.15 * np.cos(0.32 * sim_time + 0.2)),
        "right": max(0.08, 1.35 + 0.25 * np.sin(0.28 * sim_time + 0.6)),
    }
    sonar_targets = [
        {
            "id": "dock_pylon",
            "pos": pos + np.array([6.0, 0.5 * np.sin(0.4 * sim_time), -0.4], dtype=np.float64),
            "strength": 1.0,
            "extent_deg": 3.0,
        },
        {
            "id": "reef_port",
            "pos": pos + np.array([8.0, 3.0 + 0.3 * np.cos(0.3 * sim_time), -0.8], dtype=np.float64),
            "strength": 0.85,
            "extent_deg": 4.5,
        },
        {
            "id": "wreck_starboard",
            "pos": pos + np.array([7.5, -4.5 + 0.4 * np.sin(0.25 * sim_time), -1.0], dtype=np.float64),
            "strength": 0.9,
            "extent_deg": 5.0,
        },
    ]
    water_turbidity_ntu = float(np.clip(3.0 + 6.0 * rain_rate + 2.0 * obstruction, 0.5, 40.0))
    if scenario_phase is ScenarioPhase.URBAN_CANYON:
        obstruction = min(0.82, obstruction + 0.20)
        illuminance_lux *= 0.55
        ultrasonic_ranges["left"] = max(0.08, ultrasonic_ranges["left"] - 0.22)
    if scenario_phase is ScenarioPhase.SIGNAL_RECOVERY:
        ultrasonic_ranges["front_left"] = max(0.08, ultrasonic_ranges["front_left"] - 0.18)
    if scenario_phase is ScenarioPhase.RAIN_BURST:
        rain_rate = max(rain_rate, 6.0)
        relative_humidity_pct = min(98.0, relative_humidity_pct + 10.0)

    depth_m = float(2.6 + 0.5 * np.sin(0.18 * frame_idx))
    water_salinity_ppt = float(34.8 + 0.4 * np.sin(0.05 * frame_idx + 0.4))
    water_temperature_c = float(11.0 + 1.5 * np.cos(0.08 * frame_idx))
    acoustic_sources = [
        {
            "pos": pos + np.array([6.0, 1.0 * np.sin(0.2 * sim_time), -depth_m], dtype=np.float64),
            "frequency_hz": 18_000.0,
            "source_level_db": 158.0,
        },
        {
            "pos": pos + np.array([10.0, -2.0, -depth_m - 0.8], dtype=np.float64),
            "frequency_hz": 24_000.0,
            "source_level_db": 151.0,
        },
    ]
    water_ingress_ml = float(
        np.clip(
            0.05
            + 0.12 * (1.0 + np.sin(0.14 * frame_idx))
            + (0.9 if scenario_phase is ScenarioPhase.RAIN_BURST else 0.0),
            0.0,
            2.5,
        )
    )
    hull_breach = scenario_phase is ScenarioPhase.RAIN_BURST and frame_idx % 6 == 0
    tof_rows, tof_cols = config.tof_shape
    tof_x = np.linspace(-1.0, 1.0, tof_cols, dtype=np.float32)[None, :]
    tof_y = np.linspace(-1.0, 1.0, tof_rows, dtype=np.float32).reshape(-1, 1)
    tof_ranges_m = np.clip(
        0.32
        + 0.20 * np.sqrt(tof_x**2 + tof_y**2)
        + 0.04 * np.sin(wave_phase + 2.0 * tof_x)
        - 0.03 * np.cos(wave_phase + 3.0 * tof_y),
        0.08,
        2.8,
    ).astype(np.float32)

    return {
        "rgb": rgb_uint8,
        "rgb_right": rgb_right,
        "gray": gray,
        "depth": depth,
        "seg": seg,
        "range_image": range_image,
        "intensity_image": intensity_image,
        "temperature_map": {1: 21.0 + 2.0 * speed, 2: 54.0 + 6.0 * np.cos(wave_phase), 3: 12.0},
        "pos": pos,
        "vel": vel,
        "lin_acc": lin_acc,
        "ang_vel": ang_vel,
        "gravity_body": np.array([0.0, 0.0, 9.80665], dtype=np.float64),
        "obstruction": obstruction,
        "weather": {
            "rain_rate_mm_h": rain_rate,
            "cloud_cover": cloud_cover,
            "ambient_temp_c": ambient_temp_c,
            "relative_humidity_pct": relative_humidity_pct,
            "illuminance_lux": illuminance_lux,
            "wind_speed_ms": float(np.linalg.norm(wind_ms[:2])),
            "wind_direction_deg": float((np.degrees(np.arctan2(wind_ms[1], wind_ms[0])) + 360.0) % 360.0),
        },
        "ambient_temp_c": ambient_temp_c,
        "ambient_temperature_c": ambient_temp_c,
        "temperature_c": ambient_temp_c,
        "relative_humidity_pct": relative_humidity_pct,
        "illuminance_lux": illuminance_lux,
        "gas_sources": gas_sources,
        "uwb_anchors": uwb_anchors,
        "radar_targets": radar_targets,
        "ultrasonic_ranges_m": ultrasonic_ranges,
        "sonar_targets": sonar_targets,
        "water_turbidity_ntu": water_turbidity_ntu,
        "depth_m": depth_m,
        "water_temperature_c": water_temperature_c,
        "water_salinity_ppt": water_salinity_ppt,
        "ambient_noise_db": 58.0 + 4.0 * obstruction,
        "acoustic_sources": acoustic_sources,
        "water_ingress_ml": water_ingress_ml,
        "hull_breach": hull_breach,
        "remote_pos": pos + np.array([20.0, -6.0, -depth_m], dtype=np.float64),
        "motor_current_a": float(np.clip(10.0 + 3.0 * speed, 0.0, 40.0)),
        "motor_speed_rads": float(235.0 + 35.0 * np.sin(0.24 * sim_time)),
        "tof_ranges_m": tof_ranges_m,
        "load_force_n": float(42.0 + 7.0 * np.sin(0.31 * sim_time) + 3.0 * speed),
        "extension_m": float(np.clip(0.95 + 0.25 * np.sin(0.27 * sim_time), 0.0, 2.0)),
        "range_m": max(0.05, float(pos[2])),
        "current_a": float(np.clip(8.0 + 2.5 * speed, 0.0, 30.0)),
        "voltage_v": 14.8,
        "wind": wind_ms,
        "wind_ms": wind_ms,
        "water_current_ms": water_current_ms,
        "current_layers": current_layers,
        "frame_idx": frame_idx,
        "sim_time": float(sim_time),
        "scenario_progress": progress,
        "phase": phase_name,
        "fault_flags": (
            [phase_name.replace("_", " ")]
            if phase_name not in {ScenarioPhase.TAKEOFF.value, ScenarioPhase.CRUISE.value}
            else []
        ),
    }


def make_synthetic_rollout(
    frame_count: int = 16,
    *,
    start_frame: int = 0,
    config: SyntheticScenarioConfig | None = None,
    dt: float = DEFAULT_DT,
    total_frames: int = DEFAULT_TOTAL_FRAMES,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    lidar_shape: tuple[int, int] = DEFAULT_LIDAR_SHAPE,
    tof_shape: tuple[int, int] = DEFAULT_TOF_SHAPE,
) -> list[dict[str, Any]]:
    """Create a sequence of synthetic states for dataset and pipeline smoke tests."""
    if frame_count <= 0:
        raise ValueError(f"frame_count must be > 0, got {frame_count}")
    if start_frame < 0:
        raise ValueError(f"start_frame must be >= 0, got {start_frame}")

    scenario = (
        config.validated()
        if config is not None
        else SyntheticScenarioConfig(
            dt=dt,
            total_frames=total_frames,
            resolution=resolution,
            lidar_shape=lidar_shape,
            tof_shape=tof_shape,
        ).validated()
    )

    return [
        make_synthetic_sensor_state(
            start_frame + offset,
            dt=scenario.dt,
            total_frames=scenario.total_frames,
            resolution=scenario.resolution,
            lidar_shape=scenario.lidar_shape,
            tof_shape=scenario.tof_shape,
        )
        for offset in range(frame_count)
    ]


def summarize_synthetic_rollout(
    frame_count: int = 16,
    *,
    start_frame: int = 0,
    config: SyntheticScenarioConfig | None = None,
    dt: float = DEFAULT_DT,
    total_frames: int = DEFAULT_TOTAL_FRAMES,
    resolution: tuple[int, int] = DEFAULT_RESOLUTION,
    lidar_shape: tuple[int, int] = DEFAULT_LIDAR_SHAPE,
    tof_shape: tuple[int, int] = DEFAULT_TOF_SHAPE,
) -> SyntheticRolloutSummary:
    """Summarize a generated synthetic rollout for docs, tests, or dataset planning."""
    scenario = (
        config.validated()
        if config is not None
        else SyntheticScenarioConfig(
            dt=dt,
            total_frames=total_frames,
            resolution=resolution,
            lidar_shape=lidar_shape,
            tof_shape=tof_shape,
        ).validated()
    )
    rollout = make_synthetic_rollout(frame_count=frame_count, start_frame=start_frame, config=scenario)

    phase_counts: dict[str, int] = {}
    for state in rollout:
        phase_name = str(state["phase"])
        phase_counts[phase_name] = phase_counts.get(phase_name, 0) + 1

    sensor_keys = tuple(sorted(str(key) for key in rollout[-1]))
    return SyntheticRolloutSummary(
        frame_count=frame_count,
        dt=scenario.dt,
        duration_s=max(frame_count - 1, 0) * scenario.dt,
        resolution=scenario.resolution,
        lidar_shape=scenario.lidar_shape,
        tof_shape=scenario.tof_shape,
        phase_counts=phase_counts,
        sensor_keys=sensor_keys,
    )


__all__ = [
    "DEFAULT_DT",
    "DEFAULT_LIDAR_SHAPE",
    "DEFAULT_RESOLUTION",
    "DEFAULT_TOF_SHAPE",
    "DEFAULT_TOTAL_FRAMES",
    "GNSS_ORIGIN_LLH",
    "ScenarioPhase",
    "ScenarioWindow",
    "SyntheticRolloutSummary",
    "SyntheticScenarioConfig",
    "get_scenario_phase",
    "list_scenario_windows",
    "make_synthetic_rollout",
    "make_synthetic_sensor_state",
    "summarize_synthetic_rollout",
]
