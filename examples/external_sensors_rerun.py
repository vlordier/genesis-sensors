from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    import rerun as rr
except ImportError:  # pragma: no cover - optional dependency walkthrough
    rr = None  # type: ignore[assignment]

import numpy as np

from genesis_sensors import make_synthetic_multimodal_rig


def _normalize_image(image: np.ndarray) -> np.ndarray:
    """Convert numeric sensor images to uint8 for easy Rerun display."""
    arr = np.nan_to_num(np.asarray(image), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.dtype == np.uint8:
        return arr
    arr_f = arr.astype(np.float32, copy=False)
    lo = float(np.min(arr_f))
    hi = float(np.max(arr_f))
    if hi - lo < 1e-6:
        return np.zeros(arr_f.shape, dtype=np.uint8)
    scaled = (arr_f - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def _log_scalar(path: str, value: float | int) -> None:
    assert rr is not None
    rr.log(path, rr.Scalars(float(value)))


def _log_vector(path: str, values: np.ndarray, names: tuple[str, str, str] = ("x", "y", "z")) -> None:
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    for axis_name, value in zip(names, values[: len(names)], strict=False):
        _log_scalar(f"{path}/{axis_name}", float(value))


def _log_observation(frame: int, dt: float, obs: dict[str, Any]) -> None:
    assert rr is not None

    rr.set_time("frame", sequence=frame)
    rr.log("sensors/camera/rgb", rr.Image(np.asarray(obs["rgb"]["rgb"])))
    rr.log("sensors/thermal/temperature", rr.Image(_normalize_image(np.asarray(obs["thermal"]["temperature_c"]))))
    rr.log("sensors/lidar/range_image", rr.Image(_normalize_image(np.asarray(obs["lidar"]["range_image"]))))
    if "imaging_sonar" in obs:
        rr.log("sensors/sonar/imaging", rr.Image(_normalize_image(np.asarray(obs["imaging_sonar"]["intensity_image"]))))

    lidar_points = np.asarray(obs["lidar"]["points"], dtype=np.float32)
    if lidar_points.size:
        xyz = lidar_points[:, :3]
        if lidar_points.shape[1] >= 4:
            intensity = np.clip(lidar_points[:, 3], 0.0, 1.0)
            colors = np.stack([intensity * 255.0, intensity * 180.0, 255.0 - intensity * 255.0], axis=1)
            rr.log("sensors/lidar/points", rr.Points3D(xyz, colors=colors.astype(np.uint8)))
        else:
            rr.log("sensors/lidar/points", rr.Points3D(xyz))

    imu = obs["imu"]
    gnss = obs["gnss"]
    battery = obs["battery"]
    radio = obs["radio"]
    ultrasonic = obs.get("ultrasonic", {})
    imaging_sonar = obs.get("imaging_sonar", {})
    side_scan = obs.get("side_scan", {})
    uwb = obs.get("uwb", {})
    radar = obs.get("radar", {})
    flow = obs["optical_flow"]
    rangefinder = obs["rangefinder"]
    airspeed = obs["airspeed"]
    barometer = obs["barometer"]
    thermometer = obs.get("thermometer", {})
    hygrometer = obs.get("hygrometer", {})
    light_sensor = obs.get("light_sensor", {})
    gas_sensor = obs.get("gas_sensor", {})
    anemometer = obs.get("anemometer", {})
    event_count = len(obs["events"]["events"])

    _log_vector("traces/imu/lin_acc_mps2", np.asarray(imu["lin_acc"]))
    _log_vector("traces/imu/ang_vel_rads", np.asarray(imu["ang_vel"]))
    _log_scalar("traces/events/count", event_count)
    _log_scalar("traces/gnss/fix_quality", int(gnss.get("fix_quality", 0)))
    _log_scalar("traces/gnss/alt_m", float(np.asarray(gnss.get("pos_llh", [0.0, 0.0, 0.0]))[2]))
    _log_scalar("traces/barometer/altitude_m", float(barometer.get("altitude_m", 0.0)))
    _log_scalar("traces/airspeed/airspeed_ms", float(airspeed.get("airspeed_ms", 0.0)))
    _log_scalar("traces/rangefinder/range_m", float(rangefinder.get("range_m", 0.0)))
    _log_scalar("traces/optical_flow/quality", int(flow.get("quality", 0)))
    _log_scalar("traces/battery/voltage_v", float(battery.get("voltage_v", 0.0)))
    _log_scalar("traces/battery/current_a", float(battery.get("current_a", 0.0)))
    _log_scalar("traces/radio/queue_depth", int(radio.get("queue_depth", 0)))
    _log_scalar("traces/ultrasonic/nearest_range_m", float(ultrasonic.get("nearest_range_m", 0.0)))
    _log_scalar("traces/ultrasonic/valid_beams", int(np.sum(np.asarray(ultrasonic.get("valid_mask", []), dtype=bool))))
    _log_scalar("traces/sonar/imaging_returns", int(imaging_sonar.get("n_returns", 0)))
    _log_scalar("traces/sonar/port_hits", int(side_scan.get("port_hits", 0)))
    _log_scalar("traces/sonar/starboard_hits", int(side_scan.get("starboard_hits", 0)))
    _log_scalar("traces/uwb/anchor_count", len(uwb.get("ranges_m", {})))
    if "position_estimate" in uwb:
        _log_vector("traces/uwb/position_estimate_m", np.asarray(uwb["position_estimate"]))
    _log_scalar("traces/radar/detections", int(radar.get("n_detections", 0)))
    _log_scalar("traces/weather/temperature_c", float(thermometer.get("temperature_c", 0.0)))
    _log_scalar("traces/weather/humidity_pct", float(hygrometer.get("relative_humidity_pct", 0.0)))
    _log_scalar("traces/weather/illuminance_lux", float(light_sensor.get("illuminance_lux", 0.0)))
    _log_scalar("traces/weather/gas_ppm", float(gas_sensor.get("concentration_ppm", 0.0)))
    _log_scalar("traces/weather/wind_speed_ms", float(anemometer.get("wind_speed_ms", 0.0)))
    if "wind_vector_ms" in anemometer:
        _log_vector("traces/weather/wind_vector_ms", np.asarray(anemometer["wind_vector_ms"]))

    radar_points = np.asarray(radar.get("points_xyz", np.empty((0, 3))), dtype=np.float32)
    if radar_points.size:
        rr.log("sensors/radar/detections", rr.Points3D(radar_points, radii=0.12))

    if side_scan:
        port = np.asarray(side_scan.get("port_intensity", []), dtype=np.float32)
        starboard = np.asarray(side_scan.get("starboard_intensity", []), dtype=np.float32)
        if port.size and starboard.size:
            rr.log("sensors/sonar/side_scan", rr.Image(_normalize_image(np.stack([port, starboard], axis=0))))

    rr.log(
        "status/summary",
        rr.TextDocument(
            f"frame={frame:03d} t={frame * dt:.2f}s\n"
            f"events={event_count} range={float(rangefinder.get('range_m', 0.0)):.2f}m temp={float(thermometer.get('temperature_c', 0.0)):.1f}C\n"
            f"battery={float(battery.get('voltage_v', 0.0)):.2f}V fix={int(gnss.get('fix_quality', 0))} hum={float(hygrometer.get('relative_humidity_pct', 0.0)):.0f}%\n"
            f"ultra={float(ultrasonic.get('nearest_range_m', 0.0)):.2f}m sonar={int(imaging_sonar.get('n_returns', 0))} returns "
            f"uwb={len(uwb.get('ranges_m', {}))} anchors radar={int(radar.get('n_detections', 0))} detections"
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Stream Genesis sensor images and time-series traces to Rerun")
    parser.add_argument("--frames", type=int, default=60, help="Number of synthetic frames to log")
    parser.add_argument("--dt", type=float, default=0.05, help="Synthetic timestep between frames")
    parser.add_argument("--seed", type=int, default=0, help="Seed for the multimodal rig")
    parser.add_argument("--spawn", action="store_true", help="Spawn the Rerun viewer if installed")
    parser.add_argument("--save", type=Path, default=None, help="Optional `.rrd` output path to record the session")
    args = parser.parse_args()

    if rr is None:
        print(
            "`rerun` is not installed. Run `pip install rerun-sdk` or `pip install -e .[rerun]` to enable this example."
        )
        return

    rr.init("genesis-sensors-external")
    if args.save is not None:
        rr.save(str(args.save))
    if args.spawn:
        rr.spawn()

    rig = make_synthetic_multimodal_rig(dt=args.dt, seed=args.seed)
    rig.reset()

    for frame in range(args.frames):
        obs = rig.step(frame * args.dt)
        _log_observation(frame, args.dt, obs)
        print(
            f"logged frame {frame:03d} events={len(obs['events']['events']):4d} "
            f"range={float(obs['rangefinder'].get('range_m', 0.0)):.2f}m "
            f"battery={float(obs['battery'].get('voltage_v', 0.0)):.2f}V"
        )

    if args.save is not None:
        print(f"saved Rerun recording to {args.save}")


if __name__ == "__main__":
    main()
