from __future__ import annotations

import argparse
import os

import numpy as np

from genesis_sensors.scenes import build_drone_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Genesis Sensors drone demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 12) if "PYTEST_VERSION" in os.environ else args.steps
    demo = build_drone_demo(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    demo.rig.reset()

    print(f"drone rig: {', '.join(demo.rig.sensor_names())}")
    for step in range(steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        obs = demo.rig.step(step * args.dt)
        if step % max(1, steps // 5) == 0:
            acc = np.asarray(obs["imu"]["lin_acc"], dtype=float)
            alt = float(obs["barometer"]["altitude_m"])
            print(f"step={step:03d} alt={alt:6.3f} imu={acc.round(3).tolist()}")


if __name__ == "__main__":
    main()
