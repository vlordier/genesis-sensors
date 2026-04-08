from __future__ import annotations

# ruff: noqa: E402

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from genesis_sensors.scenes import build_franka_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Franka wrist IMU walkthrough")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 12) if "PYTEST_VERSION" in os.environ else args.steps
    demo = build_franka_demo(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    demo.rig.reset()

    print("imu demo sensors:", ", ".join(demo.rig.sensor_names()))
    for step in range(steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        obs = demo.rig.step(step * args.dt)
        if step % max(1, steps // 4) == 0:
            lin_acc = np.asarray(obs["imu"]["lin_acc"], dtype=float).round(3).tolist()
            ang_vel = np.asarray(obs["imu"]["ang_vel"], dtype=float).round(3).tolist()
            print(f"step={step:03d} lin_acc={lin_acc} ang_vel={ang_vel}")


if __name__ == "__main__":
    main()
