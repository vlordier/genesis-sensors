"""Tiny CLI for running the bundled Genesis Sensors demos."""

from __future__ import annotations

import argparse
from typing import Callable

from .scenes import build_drone_demo, build_franka_demo, build_go2_demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone Genesis Sensors demos")
    parser.add_argument("demo", choices=("drone", "franka", "go2"), help="Demo scene to run")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis viewer")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU backend when available")
    args = parser.parse_args()

    builders: dict[str, Callable[..., object]] = {
        "drone": build_drone_demo,
        "franka": build_franka_demo,
        "go2": build_go2_demo,
    }
    demo = builders[args.demo](dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    demo.rig.reset()

    for step in range(args.steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        demo.rig.step(step * args.dt)


if __name__ == "__main__":
    main()
