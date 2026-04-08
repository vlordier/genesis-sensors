from __future__ import annotations

# ruff: noqa: E402

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genesis_sensors import make_synthetic_multimodal_rig, wrap_rig_with_faults


def main() -> None:
    parser = argparse.ArgumentParser(description="Demonstrate latency/dropout injection and health metadata")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gnss-latency", type=float, default=0.25, help="Injected GNSS latency in seconds")
    parser.add_argument("--camera-latency", type=float, default=0.10, help="Injected RGB/stereo latency in seconds")
    parser.add_argument("--dropout", type=float, default=0.15, help="Injected dropout probability for LiDAR and radio")
    args = parser.parse_args()

    rig = make_synthetic_multimodal_rig(dt=args.dt, seed=args.seed)
    robust_rig = wrap_rig_with_faults(
        rig,
        latency_s={
            "gnss": args.gnss_latency,
            "rgb": args.camera_latency,
            "stereo": args.camera_latency,
            "thermal": args.camera_latency,
        },
        dropout_prob={
            "lidar": args.dropout,
            "radio": max(0.05, args.dropout * 0.5),
        },
        seed=args.seed + 123,
    )
    robust_rig.reset()

    print(f"robust sensors: {', '.join(robust_rig.sensor_names())}")
    print("showing trace/status snapshots from the fault-injected suite")

    interesting = {0, 1, 2, args.frames // 2, args.frames - 1}
    for frame_idx in range(args.frames):
        obs = robust_rig.step(frame_idx * args.dt)
        if frame_idx not in interesting:
            continue

        gnss_meta = obs["gnss"].get("_meta", {})
        lidar_meta = obs["lidar"].get("_meta", {})
        radio_meta = obs["radio"].get("_meta", {})
        battery = obs["battery"]
        rangefinder = obs["rangefinder"]

        print(
            f"frame={frame_idx:03d} "
            f"gnss={gnss_meta.get('status', 'n/a'):>12} age={gnss_meta.get('age_s', 0.0)!s:>6} "
            f"lidar={lidar_meta.get('status', 'n/a'):>12} radio={radio_meta.get('status', 'n/a'):>12} "
            f"range={float(rangefinder.get('range_m', 0.0)):.2f}m "
            f"battery={float(battery.get('voltage_v', 0.0)):.2f}V"
        )


if __name__ == "__main__":
    main()
