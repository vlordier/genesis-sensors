from __future__ import annotations

import argparse
import os

import numpy as np

from genesis_sensors.scenes import build_go2_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone Genesis Sensors Go2 demo")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=0.01)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 15) if "PYTEST_VERSION" in os.environ else args.steps
    demo = build_go2_demo(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    demo.rig.reset()

    print(f"go2 rig: {', '.join(demo.rig.sensor_names())}")
    for step in range(steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        obs = demo.rig.step(step * args.dt)
        if step % max(1, steps // 5) == 0:
            imu = np.asarray(obs["imu"]["lin_acc"], dtype=float)
            contacts = {name: int(data["contact_detected"]) for name, data in obs.items() if name.endswith("_contact")}
            print(f"step={step:03d} acc_z={imu[2]:7.3f} contacts={contacts}")


if __name__ == "__main__":
    main()
