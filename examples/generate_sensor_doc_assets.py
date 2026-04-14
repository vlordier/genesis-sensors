from __future__ import annotations

import argparse
import html
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from genesis_sensors import (
    AcousticCurrentProfilerModel,
    AirspeedModel,
    AnemometerModel,
    BarometerModel,
    BatteryModel,
    CameraModel,
    ContactSensor,
    CurrentSensor,
    DVLModel,
    DepthCameraModel,
    EventCameraModel,
    ForceTorqueSensorModel,
    GasSensorModel,
    GNSSModel,
    HydrophoneModel,
    HygrometerModel,
    IMUModel,
    ImagingSonarModel,
    InclinometerModel,
    JointStateSensor,
    LeakDetectorModel,
    LidarModel,
    LightSensorModel,
    LoadCellModel,
    MagnetometerModel,
    MotorTemperatureModel,
    OpticalFlowModel,
    ProximityToFArrayModel,
    RPMSensor,
    RadarModel,
    RadioLinkModel,
    RangefinderModel,
    SideScanSonarModel,
    StereoCameraModel,
    TactileArraySensor,
    ThermalCameraModel,
    ThermometerModel,
    UltrasonicArrayModel,
    UnderwaterModemModel,
    UWBRangingModel,
    WaterPressureModel,
    WheelOdometryModel,
    WireEncoderModel,
)
from genesis_sensors.synthetic import GNSS_ORIGIN_LLH

SVG_COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706")


@dataclass(frozen=True)
class ExampleSpec:
    title: str
    factory: Callable[[int], Any]
    snapshot_kind: str | None = None


@dataclass(frozen=True)
class DemoCapture:
    label: str
    history_by_sensor: dict[str, dict[str, list[float]]]
    observations: dict[str, dict[str, Any]]


def _get_demo_builders() -> dict[str, tuple[str, Callable[..., Any]]]:
    """Import scene builders lazily so metadata inspection works without Genesis/Torch."""
    from genesis_sensors.scenes import build_franka_demo, build_perception_demo

    return {
        "perception": ("Genesis perception demo", build_perception_demo),
        "franka": ("Genesis Franka demo", build_franka_demo),
    }


DEMO_SENSOR_KEYS: dict[str, tuple[str, str]] = {
    "imu": ("perception", "imu"),
    "gnss": ("perception", "gnss"),
    "barometer": ("perception", "barometer"),
    "magnetometer": ("perception", "magnetometer"),
    "airspeed": ("perception", "airspeed"),
    "optical_flow": ("perception", "optical_flow"),
    "wheel_odometry": ("perception", "wheel_odometry"),
    "camera_model": ("perception", "rgb"),
    "stereo_camera": ("perception", "stereo"),
    "depth_camera": ("franka", "depth_camera"),
    "thermal_camera": ("perception", "thermal"),
    "event_camera": ("perception", "events"),
    "lidar": ("perception", "lidar"),
    "rangefinder": ("perception", "rangefinder"),
    "ultrasonic": ("perception", "ultrasonic"),
    "imaging_sonar": ("perception", "imaging_sonar"),
    "side_scan_sonar": ("perception", "side_scan"),
    "dvl": ("perception", "dvl"),
    "acoustic_current_profiler": ("perception", "current_profiler"),
    "water_pressure": ("perception", "water_pressure"),
    "hydrophone": ("perception", "hydrophone"),
    "leak_detector": ("perception", "leak_detector"),
    "underwater_modem": ("perception", "underwater_modem"),
    "inclinometer": ("perception", "inclinometer"),
    "proximity_tof": ("perception", "proximity_tof"),
    "load_cell": ("perception", "load_cell"),
    "wire_encoder": ("perception", "wire_encoder"),
    "motor_temperature": ("perception", "motor_temperature"),
    "thermometer": ("perception", "thermometer"),
    "hygrometer": ("perception", "hygrometer"),
    "light_sensor": ("perception", "light_sensor"),
    "gas_sensor": ("perception", "gas_sensor"),
    "anemometer": ("perception", "anemometer"),
    "battery": ("perception", "battery"),
    "radio": ("perception", "radio"),
    "uwb_ranging": ("perception", "uwb"),
    "radar": ("perception", "radar"),
    "force_torque": ("franka", "force_torque"),
    "joint_state": ("franka", "joint_state"),
    "contact_sensor": ("franka", "contact"),
    "tactile_array": ("franka", "tactile_array"),
    "current_sensor": ("franka", "current"),
    "rpm_sensor": ("franka", "rpm"),
}


def _build_specs() -> dict[str, ExampleSpec]:
    return {
        "imu": ExampleSpec("IMU", lambda seed: IMUModel.from_preset("PIXHAWK_ICM20689", seed=seed)),
        "gnss": ExampleSpec(
            "GNSS", lambda seed: GNSSModel.from_preset("UBLOX_F9P_RTK", origin_llh=GNSS_ORIGIN_LLH, seed=seed)
        ),
        "barometer": ExampleSpec("Barometer", lambda seed: BarometerModel.from_preset("MS5611", seed=seed)),
        "magnetometer": ExampleSpec("Magnetometer", lambda seed: MagnetometerModel.from_preset("RM3100", seed=seed)),
        "airspeed": ExampleSpec("Airspeed", lambda seed: AirspeedModel.from_preset("MS4525DO", seed=seed)),
        "optical_flow": ExampleSpec("Optical Flow", lambda seed: OpticalFlowModel.from_preset("PX4FLOW", seed=seed)),
        "wheel_odometry": ExampleSpec(
            "Wheel Odometry", lambda seed: WheelOdometryModel.from_preset("DIFF_DRIVE_ENCODER_50HZ", seed=seed)
        ),
        "camera_model": ExampleSpec(
            "RGB Camera",
            lambda seed: CameraModel.from_preset("RASPBERRY_PI_V2", resolution=(96, 72), seed=seed),
            snapshot_kind="camera",
        ),
        "stereo_camera": ExampleSpec(
            "Stereo Camera",
            lambda seed: StereoCameraModel.from_preset("ZED2_STEREO", resolution=(96, 72), seed=seed),
            snapshot_kind="stereo",
        ),
        "depth_camera": ExampleSpec(
            "Depth Camera",
            lambda seed: DepthCameraModel.from_preset("INTEL_D435", resolution=(96, 72), seed=seed),
            snapshot_kind="depth",
        ),
        "thermal_camera": ExampleSpec(
            "Thermal Camera",
            lambda seed: ThermalCameraModel.from_preset("FLIR_BOSON_320", resolution=(96, 72), seed=seed),
            snapshot_kind="thermal",
        ),
        "event_camera": ExampleSpec(
            "Event Camera",
            lambda seed: EventCameraModel.from_preset("DAVIS_346", seed=seed),
            snapshot_kind="event",
        ),
        "lidar": ExampleSpec(
            "LiDAR",
            lambda seed: LidarModel.from_preset(
                "VELODYNE_VLP16", n_channels=8, h_resolution=64, max_range_m=12.0, seed=seed
            ),
            snapshot_kind="lidar",
        ),
        "rangefinder": ExampleSpec("Rangefinder", lambda seed: RangefinderModel.from_preset("TFMINI_PLUS", seed=seed)),
        "ultrasonic": ExampleSpec(
            "Ultrasonic Array", lambda seed: UltrasonicArrayModel.from_preset("HC_SR04_ARRAY4", seed=seed)
        ),
        "imaging_sonar": ExampleSpec(
            "Imaging Sonar",
            lambda seed: ImagingSonarModel.from_preset("TRITECH_GEMINI_720IK", seed=seed),
            snapshot_kind="imaging_sonar",
        ),
        "side_scan_sonar": ExampleSpec(
            "Side-Scan Sonar",
            lambda seed: SideScanSonarModel.from_preset("EDGETECH_4125", seed=seed),
            snapshot_kind="side_scan",
        ),
        "dvl": ExampleSpec("DVL", lambda seed: DVLModel.from_preset("NORTEK_DVL1000", seed=seed)),
        "acoustic_current_profiler": ExampleSpec(
            "Current Profiler",
            lambda seed: AcousticCurrentProfilerModel.from_preset("TELEDYNE_WORKHORSE_600", seed=seed),
        ),
        "water_pressure": ExampleSpec("Water Pressure", lambda seed: WaterPressureModel(seed=seed)),
        "hydrophone": ExampleSpec("Hydrophone", lambda seed: HydrophoneModel(seed=seed)),
        "leak_detector": ExampleSpec("Leak Detector", lambda seed: LeakDetectorModel(seed=seed)),
        "underwater_modem": ExampleSpec("Underwater Modem", lambda seed: UnderwaterModemModel(seed=seed)),
        "inclinometer": ExampleSpec("Inclinometer", lambda seed: InclinometerModel(seed=seed)),
        "proximity_tof": ExampleSpec("Proximity ToF", lambda seed: ProximityToFArrayModel(seed=seed)),
        "load_cell": ExampleSpec("Load Cell", lambda seed: LoadCellModel(seed=seed)),
        "wire_encoder": ExampleSpec("Wire Encoder", lambda seed: WireEncoderModel(seed=seed)),
        "motor_temperature": ExampleSpec("Motor Temperature", lambda seed: MotorTemperatureModel(seed=seed)),
        "thermometer": ExampleSpec(
            "Thermometer", lambda seed: ThermometerModel.from_preset("DS18B20_PROBE", seed=seed)
        ),
        "hygrometer": ExampleSpec("Hygrometer", lambda seed: HygrometerModel.from_preset("SHT31_HUMIDITY", seed=seed)),
        "light_sensor": ExampleSpec(
            "Light Sensor", lambda seed: LightSensorModel.from_preset("TSL2591_LIGHT", seed=seed)
        ),
        "gas_sensor": ExampleSpec(
            "Gas Sensor", lambda seed: GasSensorModel.from_preset("SGP30_AIR_QUALITY", seed=seed)
        ),
        "anemometer": ExampleSpec(
            "Anemometer", lambda seed: AnemometerModel.from_preset("DAVIS_6410_ANEMOMETER", seed=seed)
        ),
        "battery": ExampleSpec("Battery", lambda seed: BatteryModel.from_preset("LIPO_4S_5000MAH", seed=seed)),
        "radio": ExampleSpec("Radio Link", lambda seed: RadioLinkModel.from_preset("RADIO_URBAN", seed=seed)),
        "uwb_ranging": ExampleSpec(
            "UWB Ranging", lambda seed: UWBRangingModel.from_preset("QORVO_DWM3001C", seed=seed)
        ),
        "radar": ExampleSpec(
            "Radar",
            lambda seed: RadarModel.from_preset("TI_IWR6843AOP", seed=seed),
            snapshot_kind="radar",
        ),
        "force_torque": ExampleSpec(
            "Force / Torque", lambda seed: ForceTorqueSensorModel.from_preset("ATI_MINI45", seed=seed)
        ),
        "joint_state": ExampleSpec(
            "Joint State", lambda seed: JointStateSensor.from_preset("FRANKA_JOINT_ENCODER", seed=seed)
        ),
        "contact_sensor": ExampleSpec(
            "Contact Sensor", lambda seed: ContactSensor.from_preset("BUMPER_50HZ", seed=seed)
        ),
        "tactile_array": ExampleSpec(
            "Tactile Array", lambda seed: TactileArraySensor.from_preset("FINGERTIP_TACTILE_4X4", seed=seed)
        ),
        "current_sensor": ExampleSpec(
            "Current Sensor", lambda seed: CurrentSensor.from_preset("INA226_10A", seed=seed)
        ),
        "rpm_sensor": ExampleSpec("RPM Sensor", lambda seed: RPMSensor.from_preset("AS5048A_MAG_ENC", seed=seed)),
    }


def _extract_features(value: Any, prefix: str = "") -> dict[str, float]:
    features: dict[str, float] = {}
    if value is None:
        return features

    if isinstance(value, dict):
        for key, item in value.items():
            name = f"{prefix}.{key}" if prefix else str(key)
            features.update(_extract_features(item, name))
        return features

    if isinstance(value, (bool, np.bool_)):
        features[prefix] = float(value)
        return features

    if isinstance(value, (int, float, np.integer, np.floating)):
        numeric = float(value)
        if np.isfinite(numeric):
            features[prefix] = numeric
        return features

    if isinstance(value, np.ndarray):
        return _extract_array_features(value, prefix)

    if isinstance(value, (list, tuple)):
        if not value:
            features[f"{prefix}_count"] = 0.0
            return features
        if isinstance(value[0], dict):
            features[f"{prefix}_count"] = float(len(value))
            return features
        try:
            arr = np.asarray(value, dtype=float)
        except (TypeError, ValueError):
            features[f"{prefix}_count"] = float(len(value))
            return features
        return _extract_array_features(arr, prefix)

    return features


def _extract_array_features(arr: np.ndarray, prefix: str) -> dict[str, float]:
    features: dict[str, float] = {}
    array = np.asarray(arr)
    if array.size == 0:
        features[f"{prefix}_count"] = 0.0
        return features

    if np.issubdtype(array.dtype, np.bool_):
        array = array.astype(float)
    elif not np.issubdtype(array.dtype, np.number):
        features[f"{prefix}_count"] = float(array.size)
        return features

    flat = np.asarray(array, dtype=float).reshape(-1)
    finite = flat[np.isfinite(flat)]
    if finite.size == 0:
        features[f"{prefix}_count"] = float(array.size)
        return features

    if array.ndim == 1 and array.size <= 3:
        for idx, val in enumerate(flat):
            if np.isfinite(val):
                features[f"{prefix}[{idx}]"] = float(val)

    features[f"{prefix}_mean"] = float(np.mean(finite))
    if finite.size > 1:
        features[f"{prefix}_std"] = float(np.std(finite))
    features[f"{prefix}_max"] = float(np.max(finite))
    return features


def _collect_demo_capture(
    build_demo: Callable[..., Any], *, label: str, frames: int, dt: float, seed: int
) -> DemoCapture:
    import genesis as gs

    demo = None
    history_by_sensor: dict[str, dict[str, list[float]]] = {}
    last_obs: dict[str, dict[str, Any]] = {}
    best_obs: dict[str, dict[str, Any]] = {}

    try:
        gs.destroy()
        demo = build_demo(dt=dt, show_viewer=False, use_gpu=False, seed=seed)
        demo.rig.reset()

        for frame_idx in range(frames):
            sim_time = frame_idx * dt
            if demo.controller is not None:
                demo.controller(frame_idx)
            demo.scene.step()
            obs_map = demo.rig.step(sim_time)

            for sensor_name, obs in obs_map.items():
                if isinstance(obs, dict):
                    last_obs[sensor_name] = obs
                    if any(_observation_has_signal(value) for value in obs.values()):
                        best_obs[sensor_name] = obs
                features = _extract_features(obs)
                history = history_by_sensor.setdefault(sensor_name, {})
                all_keys = set(history) | set(features)
                for key in all_keys:
                    history.setdefault(key, [])
                    history[key].append(float(features.get(key, np.nan)))

        observations = {
            name: dict(best_obs.get(name) or last_obs.get(name, {})) for name in set(history_by_sensor) | set(last_obs)
        }
        return DemoCapture(label=label, history_by_sensor=history_by_sensor, observations=observations)
    finally:
        if demo is not None:
            try:
                demo.scene.destroy()
            except Exception:
                pass
        try:
            gs.destroy()
        except Exception:
            pass


def _collect_genesis_captures(
    specs: dict[str, ExampleSpec], *, frames: int, dt: float, seed: int, only: set[str] | None
) -> dict[str, DemoCapture]:
    requested_demos = {
        demo_name
        for slug, (demo_name, _sensor_name) in DEMO_SENSOR_KEYS.items()
        if slug in specs and (not only or slug in only)
    }
    captures: dict[str, DemoCapture] = {}
    demo_builders = _get_demo_builders()

    for demo_idx, demo_name in enumerate(sorted(requested_demos)):
        label, builder = demo_builders[demo_name]
        captures[demo_name] = _collect_demo_capture(
            builder,
            label=label,
            frames=frames,
            dt=dt,
            seed=seed + 10_000 * (demo_idx + 1),
        )
        print(f"captured {label.lower()} observations")

    return captures


def _observation_has_signal(value: Any) -> bool:
    if isinstance(value, np.ndarray):
        return value.size > 0
    if isinstance(value, (list, tuple, dict, str, bytes)):
        return len(value) > 0
    return value is not None


def _pick_series(history: dict[str, list[float]], limit: int = 4) -> list[tuple[str, np.ndarray]]:
    scored: list[tuple[tuple[float, float], str, np.ndarray]] = []
    preferred_tokens = ("range", "speed", "quality", "temperature", "pressure", "count", "soc", "voltage")

    for name, values in history.items():
        arr = np.asarray(values, dtype=float)
        finite = arr[np.isfinite(arr)]
        if finite.size == 0:
            continue
        spread = float(np.nanstd(finite))
        token_bonus = 1.0 if any(token in name for token in preferred_tokens) else 0.0
        score = (token_bonus + spread, float(finite.size))
        scored.append((score, name, arr))

    scored.sort(reverse=True)
    selected = [(name, arr) for _, name, arr in scored[:limit]]
    if selected:
        return selected

    fallback = [
        (name, np.asarray(values, dtype=float)) for name, values in list(history.items())[:limit] if len(values) > 0
    ]
    return fallback


def _polyline_points(
    values: np.ndarray, x0: float, y0: float, width: float, height: float, ymin: float, ymax: float
) -> str:
    points: list[str] = []
    n = max(len(values), 1)
    scale = ymax - ymin if ymax > ymin else 1.0
    for idx, val in enumerate(values):
        x = x0 + (width * idx / max(n - 1, 1))
        y = y0 + height - ((float(val) - ymin) / scale) * height if np.isfinite(val) else y0 + height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _downsample_rgb(rgb: np.ndarray, max_side: int = 96) -> np.ndarray:
    arr = np.asarray(rgb, dtype=np.uint8)
    h, w = arr.shape[:2]
    step = max(1, int(np.ceil(max(h, w) / max_side)))
    return arr[::step, ::step].copy()


def _palette_map(values: np.ndarray, palette: str) -> np.ndarray:
    palettes: dict[str, tuple[tuple[float, tuple[int, int, int]], ...]] = {
        "gray": ((0.0, (15, 23, 42)), (1.0, (241, 245, 249))),
        "depth": ((0.0, (68, 1, 84)), (0.33, (59, 82, 139)), (0.66, (33, 145, 140)), (1.0, (253, 231, 37))),
        "thermal": (
            (0.0, (0, 0, 0)),
            (0.25, (68, 1, 84)),
            (0.5, (187, 55, 84)),
            (0.75, (249, 142, 8)),
            (1.0, (252, 255, 164)),
        ),
        "event": ((0.0, (59, 130, 246)), (0.5, (15, 23, 42)), (1.0, (239, 68, 68))),
    }
    stops = palettes[palette]
    x = np.clip(values.astype(np.float32), 0.0, 1.0)
    rgb = np.zeros(x.shape + (3,), dtype=np.float32)
    for (start, c0), (end, c1) in zip(stops[:-1], stops[1:]):
        mask = (x >= start) & (x <= end)
        if not np.any(mask):
            continue
        t = (x[mask] - start) / max(end - start, 1e-6)
        c0_arr = np.asarray(c0, dtype=np.float32)
        c1_arr = np.asarray(c1, dtype=np.float32)
        rgb[mask] = c0_arr + (c1_arr - c0_arr) * t[:, None]
    rgb[x <= stops[0][0]] = np.asarray(stops[0][1], dtype=np.float32)
    rgb[x >= stops[-1][0]] = np.asarray(stops[-1][1], dtype=np.float32)
    return np.clip(rgb, 0, 255).astype(np.uint8)


def _scalar_to_rgb(values: np.ndarray, palette: str = "depth", invalid_mask: np.ndarray | None = None) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = np.repeat(arr[None, :], 32, axis=0)
        if invalid_mask is not None:
            invalid_mask = np.repeat(np.asarray(invalid_mask)[None, :], arr.shape[0], axis=0)

    finite = np.isfinite(arr)
    if invalid_mask is not None:
        finite &= ~np.asarray(invalid_mask, dtype=bool)
    if not np.any(finite):
        return np.full(arr.shape + (3,), (15, 23, 42), dtype=np.uint8)

    values_finite = arr[finite]
    lo = float(np.nanpercentile(values_finite, 2.0))
    hi = float(np.nanpercentile(values_finite, 98.0))
    if np.isclose(lo, hi):
        pad = 1.0 if np.isclose(lo, 0.0) else abs(lo) * 0.1
        lo -= pad
        hi += pad
    normalised = (np.clip(arr, lo, hi) - lo) / max(hi - lo, 1e-6)
    rgb = _palette_map(normalised, palette)
    rgb[~finite] = np.asarray((15, 23, 42), dtype=np.uint8)
    return _downsample_rgb(rgb)


def _signed_to_rgb(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    limit = float(np.max(np.abs(arr[np.isfinite(arr)]))) if np.isfinite(arr).any() else 1.0
    limit = max(limit, 1e-6)
    normalised = (np.clip(arr, -limit, limit) + limit) / (2.0 * limit)
    return _downsample_rgb(_palette_map(normalised, "event"))


def _ensure_rgb(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values)
    if arr.ndim == 2:
        return _scalar_to_rgb(arr, palette="gray")

    rgb = np.asarray(arr[..., :3], dtype=np.float32)
    if rgb.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    if float(np.nanmax(rgb)) <= 1.0:
        rgb = rgb * 255.0
    return _downsample_rgb(np.clip(rgb, 0.0, 255.0).astype(np.uint8))


def _point_cloud_to_rgb(points: np.ndarray, values: np.ndarray | None = None) -> np.ndarray:
    canvas = np.full((96, 96, 3), (10, 15, 30), dtype=np.uint8)
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[0] == 0 or pts.shape[1] < 2:
        return canvas

    xy = pts[:, :2]
    finite = np.isfinite(xy).all(axis=1)
    if not np.any(finite):
        return canvas
    xy = xy[finite]
    vals = np.asarray(values, dtype=np.float32)[finite] if values is not None and len(values) == len(pts) else None

    mins = np.min(xy, axis=0)
    maxs = np.max(xy, axis=0)
    span = np.maximum(maxs - mins, 1e-6)
    norm = (xy - mins) / span
    xs = np.clip((norm[:, 0] * 83 + 6).astype(int), 0, 95)
    ys = np.clip(((1.0 - norm[:, 1]) * 83 + 6).astype(int), 0, 95)

    if vals is None or not np.isfinite(vals).any():
        colors = np.tile(np.asarray([[56, 189, 248]], dtype=np.uint8), (len(xs), 1))
    else:
        lo = float(np.nanmin(vals))
        hi = float(np.nanmax(vals))
        if np.isclose(lo, hi):
            hi = lo + 1.0
        scaled = (np.clip(vals, lo, hi) - lo) / (hi - lo)
        colors = _palette_map(scaled, "thermal").reshape(-1, 3)

    for x, y, color in zip(xs, ys, colors, strict=False):
        canvas[max(y - 1, 0) : min(y + 2, 96), max(x - 1, 0) : min(x + 2, 96)] = color

    canvas[48, :, :] = np.maximum(canvas[48, :, :], np.asarray((51, 65, 85), dtype=np.uint8))
    canvas[:, 48, :] = np.maximum(canvas[:, 48, :], np.asarray((51, 65, 85), dtype=np.uint8))
    return canvas


def _event_image(events_array: np.ndarray) -> np.ndarray:
    events = np.asarray(events_array, dtype=np.float32)
    if events.ndim != 2 or events.shape[0] == 0 or events.shape[1] < 3:
        return np.full((72, 96, 3), (15, 23, 42), dtype=np.uint8)

    xs = events[:, 0].astype(int)
    ys = events[:, 1].astype(int)
    polarities = events[:, 2]
    width = int(xs.max()) + 1
    height = int(ys.max()) + 1
    acc = np.zeros((max(height, 1), max(width, 1)), dtype=np.float32)
    for x, y, polarity in zip(xs, ys, polarities, strict=False):
        if 0 <= x < width and 0 <= y < height:
            acc[y, x] += 1.0 if polarity > 0 else -1.0
    return _signed_to_rgb(acc)


def _snapshot_panels(kind: str, obs: dict[str, Any]) -> list[tuple[str, np.ndarray]]:
    if kind == "camera" and "rgb" in obs:
        return [("RGB frame", _ensure_rgb(obs["rgb"]))]
    if kind == "stereo" and "rgb_left" in obs and "rgb_right" in obs:
        panels = [("Left eye", _ensure_rgb(obs["rgb_left"])), ("Right eye", _ensure_rgb(obs["rgb_right"]))]
        if "valid_mask" in obs:
            panels.append(
                ("Valid mask", _scalar_to_rgb(np.asarray(obs["valid_mask"], dtype=np.float32), palette="gray"))
            )
        elif "depth" in obs:
            panels.append(
                ("Recovered depth", _scalar_to_rgb(np.asarray(obs["depth"], dtype=np.float32), palette="depth"))
            )
        return panels
    if kind == "depth" and "depth_m" in obs:
        invalid = ~np.asarray(obs.get("valid_mask", np.ones_like(obs["depth_m"], dtype=bool)), dtype=bool)
        return [
            (
                "Depth map",
                _scalar_to_rgb(np.asarray(obs["depth_m"], dtype=np.float32), palette="depth", invalid_mask=invalid),
            )
        ]
    if kind == "thermal" and "temperature_c" in obs:
        return [
            ("Temperature field", _scalar_to_rgb(np.asarray(obs["temperature_c"], dtype=np.float32), palette="thermal"))
        ]
    if kind == "event" and "events_array" in obs:
        return [("Event accumulation", _event_image(np.asarray(obs["events_array"], dtype=np.float32)))]
    if kind == "lidar" and "range_image" in obs:
        panels = [("Range image", _scalar_to_rgb(np.asarray(obs["range_image"], dtype=np.float32), palette="depth"))]
        if "points" in obs:
            pts = np.asarray(obs["points"], dtype=np.float32)
            if pts.ndim == 2 and pts.shape[1] >= 3:
                panels.append(
                    ("XY point cloud", _point_cloud_to_rgb(pts[:, :3], pts[:, 3] if pts.shape[1] > 3 else None))
                )
        return panels
    if kind == "imaging_sonar" and "intensity_image" in obs:
        return [
            ("Intensity image", _scalar_to_rgb(np.asarray(obs["intensity_image"], dtype=np.float32), palette="thermal"))
        ]
    if kind == "side_scan" and "port_intensity" in obs and "starboard_intensity" in obs:
        stacked = np.vstack(
            [
                np.asarray(obs["port_intensity"], dtype=np.float32),
                np.asarray(obs["starboard_intensity"], dtype=np.float32),
            ]
        )
        waterfall = np.repeat(stacked, 24, axis=0)
        return [("Port / starboard returns", _scalar_to_rgb(waterfall, palette="thermal"))]
    if kind == "radar" and "points_xyz" in obs:
        pts = np.asarray(obs["points_xyz"], dtype=np.float32)
        return [("Detected points", _point_cloud_to_rgb(pts[:, :3] if pts.ndim == 2 else pts))]
    return []


def _image_rects(rgb: np.ndarray, x0: float, y0: float, scale: float) -> str:
    arr = np.asarray(rgb, dtype=np.uint8)
    rects: list[str] = []
    for row in range(arr.shape[0]):
        for col in range(arr.shape[1]):
            r, g, b = arr[row, col]
            rects.append(
                f'<rect x="{x0 + col * scale:.1f}" y="{y0 + row * scale:.1f}" width="{scale + 0.15:.2f}" height="{scale + 0.15:.2f}" fill="#{r:02x}{g:02x}{b:02x}" />'
            )
    return "".join(rects)


def _write_snapshot_svg(path: Path, title: str, panels: list[tuple[str, np.ndarray]], provenance: str) -> None:
    if not panels:
        _write_placeholder(path, title, "No image-like observation payload was available for this sensor.")
        return

    scale = 2.2
    panel_gap = 18.0
    left = 24.0
    top = 66.0
    max_height = max(panel.shape[0] for _, panel in panels)
    content_width = sum(panel.shape[1] * scale for _, panel in panels) + panel_gap * max(len(panels) - 1, 0)
    width = int(max(360.0, left * 2 + content_width))
    height = int(top + max_height * scale + 46.0)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} example snapshot">',
        '  <rect width="100%" height="100%" fill="#ffffff" />',
        f'  <rect x="12" y="12" width="{width - 24}" height="{height - 24}" rx="12" fill="#f8fafc" stroke="#cbd5e1" />',
        f'  <text x="24" y="38" font-size="20" font-weight="700" fill="#0f172a">{html.escape(title)}</text>',
        '  <text x="24" y="56" font-size="12" fill="#475569">Snapshot rendered from a Genesis-driven sensor observation array</text>',
    ]

    cursor_x = left
    for label, panel in panels:
        panel_h, panel_w = panel.shape[:2]
        parts.append(
            f'  <text x="{cursor_x:.1f}" y="{top - 8:.1f}" font-size="12" fill="#334155">{html.escape(label)}</text>'
        )
        parts.append(
            f'  <rect x="{cursor_x - 1:.1f}" y="{top - 1:.1f}" width="{panel_w * scale + 2:.1f}" height="{panel_h * scale + 2:.1f}" fill="#0f172a" stroke="#94a3b8" />'
        )
        parts.append(_image_rects(panel, cursor_x, top, scale))
        cursor_x += panel_w * scale + panel_gap

    parts.append(f'  <text x="24" y="{height - 18}" font-size="12" fill="#475569">{html.escape(provenance)}</text>')
    parts.append("</svg>\n")
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_svg_plot(path: Path, title: str, history: dict[str, list[float]], provenance: str) -> None:
    width = 760
    height = 280
    plot_x = 56
    plot_y = 56
    plot_w = 460
    plot_h = 170
    series = _pick_series(history)

    if not series:
        _write_placeholder(path, title, "No numeric observation series were available for this sensor.")
        return

    stacked = np.concatenate([arr[np.isfinite(arr)] for _, arr in series if np.isfinite(arr).any()])
    ymin = float(np.min(stacked))
    ymax = float(np.max(stacked))
    if np.isclose(ymin, ymax):
        pad = 1.0 if np.isclose(ymin, 0.0) else abs(ymin) * 0.1
        ymin -= pad
        ymax += pad

    grid_lines = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = plot_y + plot_h * frac
        grid_lines.append(f'<line x1="{plot_x}" y1="{y:.1f}" x2="{plot_x + plot_w}" y2="{y:.1f}" stroke="#e5e7eb" />')

    lines = []
    legend = []
    for idx, (name, arr) in enumerate(series):
        color = SVG_COLORS[idx % len(SVG_COLORS)]
        points = _polyline_points(arr, plot_x, plot_y, plot_w, plot_h, ymin, ymax)
        lines.append(f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />')
        last_finite = arr[np.isfinite(arr)]
        last_value = float(last_finite[-1]) if last_finite.size else float("nan")
        legend_y = 64 + idx * 24
        label = html.escape(name.replace("_", " "))
        legend.append(
            f'<rect x="548" y="{legend_y - 10}" width="12" height="12" rx="2" fill="{color}" />'
            f'<text x="568" y="{legend_y}" font-size="12" fill="#111827">{label}: {last_value:.3g}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)} example plot">
  <rect width="100%" height="100%" fill="#ffffff" />
  <rect x="14" y="14" width="732" height="252" rx="12" fill="#f8fafc" stroke="#cbd5e1" />
  <text x="28" y="38" font-size="20" font-weight="700" fill="#0f172a">{html.escape(title)}</text>
  <text x="28" y="258" font-size="12" fill="#475569">{html.escape(provenance)}</text>
  <text x="28" y="54" font-size="12" fill="#475569">Representative scalar summaries extracted from the sensor observation dict</text>
  {"".join(grid_lines)}
  <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#94a3b8" stroke-width="1.5" />
  <line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#94a3b8" stroke-width="1.5" />
  <text x="18" y="{plot_y + 8}" font-size="11" fill="#64748b">{ymax:.3g}</text>
  <text x="18" y="{plot_y + plot_h}" font-size="11" fill="#64748b">{ymin:.3g}</text>
  {"".join(lines)}
  {"".join(legend)}
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def _write_placeholder(path: Path, title: str, message: str) -> None:
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="760" height="180" viewBox="0 0 760 180" role="img" aria-label="{html.escape(title)} placeholder">
  <rect width="100%" height="100%" fill="#ffffff" />
  <rect x="14" y="14" width="732" height="152" rx="12" fill="#f8fafc" stroke="#cbd5e1" />
  <text x="28" y="46" font-size="20" font-weight="700" fill="#0f172a">{html.escape(title)}</text>
  <text x="28" y="88" font-size="13" fill="#334155">{html.escape(message)}</text>
  <text x="28" y="118" font-size="12" fill="#64748b">Generated by examples/generate_sensor_doc_assets.py</text>
</svg>
'''
    path.write_text(svg, encoding="utf-8")


def generate_assets(output_dir: Path, frames: int, dt: float, seed: int, only: set[str] | None = None) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    specs = _build_specs()
    requested_slugs = [slug for slug in specs if not only or slug in only]
    missing_bindings = sorted(set(requested_slugs) - set(DEMO_SENSOR_KEYS))
    if missing_bindings:
        missing = ", ".join(missing_bindings)
        raise RuntimeError(f"Genesis demo bindings are missing for: {missing}")

    captures = _collect_genesis_captures(specs, frames=frames, dt=dt, seed=seed, only=only)
    written: list[Path] = []

    for slug, spec in specs.items():
        if only and slug not in only:
            continue

        demo_name, sensor_name = DEMO_SENSOR_KEYS[slug]
        capture = captures.get(demo_name)
        if capture is None:
            raise RuntimeError(f"Genesis-backed capture missing for '{slug}' (demo='{demo_name}')")

        history = capture.history_by_sensor.get(sensor_name, {})
        obs = dict(capture.observations.get(sensor_name, {}))
        if not history and not obs:
            raise RuntimeError(
                f"Genesis-backed capture for '{slug}' did not produce any observation payloads (sensor='{sensor_name}')"
            )

        provenance = f"Generated from {frames} {capture.label} timesteps (dt={dt:.2f}s) via examples/generate_sensor_doc_assets.py"
        out_path = output_dir / f"{slug}.svg"

        if spec.snapshot_kind:
            panels = _snapshot_panels(spec.snapshot_kind, obs)
            if panels:
                _write_snapshot_svg(out_path, spec.title, panels, provenance)
            else:
                _write_svg_plot(out_path, spec.title, history, provenance)
        else:
            _write_svg_plot(out_path, spec.title, history, provenance)

        written.append(out_path)
        print(f"wrote {out_path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs-ready SVG plots from Genesis-driven sensor examples")
    parser.add_argument("--output-dir", type=Path, default=Path("docs/assets/sensors"))
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of sensor slugs to render")
    args = parser.parse_args()

    generate_assets(
        output_dir=args.output_dir,
        frames=args.frames,
        dt=args.dt,
        seed=args.seed,
        only=set(args.only) if args.only else None,
    )


if __name__ == "__main__":
    main()
