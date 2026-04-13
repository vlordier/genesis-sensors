from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genesis_sensors import (
    CameraModel,
    EventCameraModel,
    SensorScheduler,
    StereoCameraModel,
    ThermalCameraModel,
    get_preset,
)
from genesis_sensors.synthetic import make_synthetic_sensor_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone camera-as-sensor walkthrough")
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--save-every", type=int, default=0, help="Save `.npz` snapshots every N steps")
    parser.add_argument("--out-dir", default="outputs/camera_as_sensor")
    args = parser.parse_args()

    scheduler = SensorScheduler()
    scheduler.add(
        CameraModel.from_config(get_preset("RASPBERRY_PI_V2").model_copy(update={"resolution": (96, 72), "seed": 0})),
        name="rgb",
    )
    scheduler.add(EventCameraModel.from_config(get_preset("DAVIS_346").model_copy(update={"seed": 1})), name="events")
    scheduler.add(
        ThermalCameraModel.from_config(
            get_preset("FLIR_BOSON_320").model_copy(update={"resolution": (96, 72), "seed": 2})
        ),
        name="thermal",
    )
    scheduler.add(
        StereoCameraModel.from_config(get_preset("ZED2_STEREO").model_copy(update={"resolution": (96, 72), "seed": 3})),
        name="stereo",
    )
    scheduler.reset()

    out_dir = Path(args.out_dir)
    if args.save_every > 0:
        out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        obs = scheduler.update(step * 0.05, make_synthetic_sensor_state(step, total_frames=args.steps))
        rgb_mean = float(np.mean(obs["rgb"]["rgb"]))
        n_events = len(obs["events"]["events"])
        thermal_peak = float(np.max(obs["thermal"]["temperature_c"]))
        stereo_valid = float(np.mean(obs["stereo"]["valid_mask"]))
        print(
            f"step={step:03d} rgb_mean={rgb_mean:6.1f} events={n_events:4d} "
            f"thermal_peak={thermal_peak:5.1f}C stereo_valid={stereo_valid:.1%}"
        )

        if args.save_every > 0 and step % args.save_every == 0:
            np.savez_compressed(
                out_dir / f"camera_step_{step:03d}.npz",
                rgb=np.asarray(obs["rgb"]["rgb"]),
                disparity=np.asarray(obs["stereo"].get("disparity", [])),
                thermal=np.asarray(obs["thermal"]["temperature_c"]),
            )


if __name__ == "__main__":
    main()
