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

from genesis_sensors.scenes import build_go2_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Go2 proprioception and contact demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 12) if "PYTEST_VERSION" in os.environ else args.steps
    demo = build_go2_demo(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    demo.rig.reset()

    print(f"go2 rig: {', '.join(demo.rig.sensor_names())}")
    for step in range(steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        obs = demo.rig.step(step * args.dt)
        if step % max(1, steps // 4) == 0:
            imu = np.asarray(obs["imu"]["lin_acc"], dtype=float).round(3).tolist()
            current_a = float(obs["current"]["current_a"])
            rpm = float(obs["rpm"]["rpm"])
            contacts = {
                name: float(data.get("force_n", 0.0)) for name, data in obs.items() if name.endswith("_contact")
            }
            print(f"step={step:03d} imu={imu} current={current_a:5.2f}A rpm={rpm:6.1f} contacts={contacts}")


if __name__ == "__main__":
    main()
