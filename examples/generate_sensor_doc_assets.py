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
    HygrometerModel,
    IMUModel,
    ImagingSonarModel,
    JointStateSensor,
    LidarModel,
    LightSensorModel,
    MagnetometerModel,
    OpticalFlowModel,
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
    UWBRangingModel,
    WheelOdometryModel,
)
from genesis_sensors.synthetic import GNSS_ORIGIN_LLH, make_synthetic_sensor_state

SVG_COLORS = ("#2563eb", "#dc2626", "#059669", "#d97706")


@dataclass(frozen=True)
class ExampleSpec:
    title: str
    factory: Callable[[int], Any]


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
            "RGB Camera", lambda seed: CameraModel.from_preset("RASPBERRY_PI_V2", resolution=(96, 72), seed=seed)
        ),
        "stereo_camera": ExampleSpec(
            "Stereo Camera", lambda seed: StereoCameraModel.from_preset("ZED2_STEREO", resolution=(96, 72), seed=seed)
        ),
        "depth_camera": ExampleSpec(
            "Depth Camera", lambda seed: DepthCameraModel.from_preset("INTEL_D435", resolution=(96, 72), seed=seed)
        ),
        "thermal_camera": ExampleSpec(
            "Thermal Camera",
            lambda seed: ThermalCameraModel.from_preset("FLIR_BOSON_320", resolution=(96, 72), seed=seed),
        ),
        "event_camera": ExampleSpec("Event Camera", lambda seed: EventCameraModel.from_preset("DAVIS_346", seed=seed)),
        "lidar": ExampleSpec(
            "LiDAR",
            lambda seed: LidarModel.from_preset(
                "VELODYNE_VLP16", n_channels=8, h_resolution=64, max_range_m=12.0, seed=seed
            ),
        ),
        "rangefinder": ExampleSpec(
            "Rangefinder", lambda seed: RangefinderModel.from_preset("TFMINI_PLUS", seed=seed)
        ),
        "ultrasonic": ExampleSpec(
            "Ultrasonic Array", lambda seed: UltrasonicArrayModel.from_preset("HC_SR04_ARRAY4", seed=seed)
        ),
        "imaging_sonar": ExampleSpec(
            "Imaging Sonar", lambda seed: ImagingSonarModel.from_preset("TRITECH_GEMINI_720IK", seed=seed)
        ),
        "side_scan_sonar": ExampleSpec(
            "Side-Scan Sonar", lambda seed: SideScanSonarModel.from_preset("EDGETECH_4125", seed=seed)
        ),
        "dvl": ExampleSpec("DVL", lambda seed: DVLModel.from_preset("NORTEK_DVL1000", seed=seed)),
        "acoustic_current_profiler": ExampleSpec(
            "Current Profiler",
            lambda seed: AcousticCurrentProfilerModel.from_preset("TELEDYNE_WORKHORSE_600", seed=seed),
        ),
        "thermometer": ExampleSpec("Thermometer", lambda seed: ThermometerModel.from_preset("DS18B20_PROBE", seed=seed)),
        "hygrometer": ExampleSpec("Hygrometer", lambda seed: HygrometerModel.from_preset("SHT31_HUMIDITY", seed=seed)),
        "light_sensor": ExampleSpec("Light Sensor", lambda seed: LightSensorModel.from_preset("TSL2591_LIGHT", seed=seed)),
        "gas_sensor": ExampleSpec("Gas Sensor", lambda seed: GasSensorModel.from_preset("SGP30_AIR_QUALITY", seed=seed)),
        "anemometer": ExampleSpec(
            "Anemometer", lambda seed: AnemometerModel.from_preset("DAVIS_6410_ANEMOMETER", seed=seed)
        ),
        "battery": ExampleSpec("Battery", lambda seed: BatteryModel.from_preset("LIPO_4S_5000MAH", seed=seed)),
        "radio": ExampleSpec("Radio Link", lambda seed: RadioLinkModel.from_preset("RADIO_URBAN", seed=seed)),
        "uwb_ranging": ExampleSpec("UWB Ranging", lambda seed: UWBRangingModel.from_preset("QORVO_DWM3001C", seed=seed)),
        "radar": ExampleSpec("Radar", lambda seed: RadarModel.from_preset("TI_IWR6843AOP", seed=seed)),
        "force_torque": ExampleSpec(
            "Force / Torque", lambda seed: ForceTorqueSensorModel.from_preset("ATI_MINI45", seed=seed)
        ),
        "joint_state": ExampleSpec(
            "Joint State", lambda seed: JointStateSensor.from_preset("FRANKA_JOINT_ENCODER", seed=seed)
        ),
        "contact_sensor": ExampleSpec("Contact Sensor", lambda seed: ContactSensor.from_preset("BUMPER_50HZ", seed=seed)),
        "tactile_array": ExampleSpec(
            "Tactile Array", lambda seed: TactileArraySensor.from_preset("FINGERTIP_TACTILE_4X4", seed=seed)
        ),
        "current_sensor": ExampleSpec("Current Sensor", lambda seed: CurrentSensor.from_preset("INA226_10A", seed=seed)),
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


def _collect_history(sensor: Any, frames: int, dt: float) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    sensor.reset()

    for frame_idx in range(frames):
        sim_time = frame_idx * dt
        state = make_synthetic_sensor_state(frame_idx, dt=dt, total_frames=max(frames, 2))
        if isinstance(sensor, RadioLinkModel):
            sensor.transmit(
                packet={"frame": frame_idx},
                src_pos=np.asarray(state["pos"], dtype=np.float64),
                dst_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                sim_time=sim_time,
            )
        obs = sensor.step(sim_time, state)
        features = _extract_features(obs)

        all_keys = set(history) | set(features)
        for key in all_keys:
            history.setdefault(key, [])
            history[key].append(float(features.get(key, np.nan)))

    return history


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
        (name, np.asarray(values, dtype=float))
        for name, values in list(history.items())[:limit]
        if len(values) > 0
    ]
    return fallback


def _polyline_points(values: np.ndarray, x0: float, y0: float, width: float, height: float, ymin: float, ymax: float) -> str:
    points: list[str] = []
    n = max(len(values), 1)
    scale = ymax - ymin if ymax > ymin else 1.0
    for idx, val in enumerate(values):
        x = x0 + (width * idx / max(n - 1, 1))
        y = y0 + height - ((float(val) - ymin) / scale) * height if np.isfinite(val) else y0 + height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _write_svg_plot(path: Path, title: str, history: dict[str, list[float]], frames: int, dt: float) -> None:
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
        lines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="2.5" points="{points}" />'
        )
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
  <text x="28" y="258" font-size="12" fill="#475569">Generated from {frames} synthetic timesteps (dt={dt:.2f}s) via examples/generate_sensor_doc_assets.py</text>
  <text x="28" y="54" font-size="12" fill="#475569">Representative scalar summaries extracted from the sensor observation dict</text>
  {''.join(grid_lines)}
  <line x1="{plot_x}" y1="{plot_y}" x2="{plot_x}" y2="{plot_y + plot_h}" stroke="#94a3b8" stroke-width="1.5" />
  <line x1="{plot_x}" y1="{plot_y + plot_h}" x2="{plot_x + plot_w}" y2="{plot_y + plot_h}" stroke="#94a3b8" stroke-width="1.5" />
  <text x="18" y="{plot_y + 8}" font-size="11" fill="#64748b">{ymax:.3g}</text>
  <text x="18" y="{plot_y + plot_h}" font-size="11" fill="#64748b">{ymin:.3g}</text>
  {''.join(lines)}
  {''.join(legend)}
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
    written: list[Path] = []

    for idx, (slug, spec) in enumerate(specs.items()):
        if only and slug not in only:
            continue
        out_path = output_dir / f"{slug}.svg"
        try:
            sensor = spec.factory(seed + idx)
            history = _collect_history(sensor, frames=frames, dt=dt)
            _write_svg_plot(out_path, spec.title, history, frames, dt)
        except Exception as exc:  # pragma: no cover - best-effort docs asset generation
            _write_placeholder(out_path, spec.title, f"Example generation failed: {type(exc).__name__}: {exc}")
        written.append(out_path)
        print(f"wrote {out_path}")

    return written


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate docs-ready SVG plots for the Genesis sensor pages")
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
