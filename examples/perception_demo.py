from __future__ import annotations

import argparse
import os

import numpy as np

from genesis_sensors.scenes import build_perception_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Genesis Sensors multimodal perception demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 12) if "PYTEST_VERSION" in os.environ else args.steps
    demo = build_perception_demo(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    demo.rig.reset()

    print(f"perception rig: {', '.join(demo.rig.sensor_names())}")
    for step in range(steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        obs = demo.rig.step(step * args.dt)
        if step % max(1, steps // 5) == 0:
            rgb_mean = float(np.mean(obs["rgb"]["rgb"]))
            lidar_points = len(obs["lidar"]["points"])
            thermal_peak = float(np.max(obs["thermal"]["temperature_c"]))
            delivered = len(obs["radio"]["delivered"])
            print(
                f"step={step:03d} rgb_mean={rgb_mean:6.1f} "
                f"lidar_points={lidar_points:4d} thermal_peak={thermal_peak:5.1f}C radio_delivered={delivered}"
            )


if __name__ == "__main__":
    main()
