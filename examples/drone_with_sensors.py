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

from genesis_sensors.scenes import build_drone_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone drone navigation sensors demo")
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
        if step % max(1, steps // 4) == 0:
            acc = np.asarray(obs["imu"]["lin_acc"], dtype=float).round(3).tolist()
            llh = np.asarray(obs["gnss"]["pos_llh"], dtype=float).round(5).tolist()
            alt = float(obs["barometer"]["altitude_m"])
            rng = float(obs["rangefinder"]["range_m"])
            soc = float(obs["battery"]["soc"]) * 100.0
            print(f"step={step:03d} alt={alt:6.3f} range={rng:5.2f}m soc={soc:5.1f}% imu={acc} gnss={llh}")


if __name__ == "__main__":
    main()
