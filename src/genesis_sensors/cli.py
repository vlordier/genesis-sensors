"""CLI for running Genesis Sensors scenes and utilities."""

from __future__ import annotations

import argparse
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .scenes import DemoScene


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Genesis Sensors example scenes")
    parser.add_argument("scene", choices=("drone", "perception", "franka", "go2"), help="Scene preset to run")
    parser.add_argument("--steps", type=int, default=200, help="Number of simulation steps")
    parser.add_argument("--dt", type=float, default=0.01, help="Simulation timestep")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis viewer")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU backend when available")
    args = parser.parse_args()

    try:
        from .scenes import build_drone_demo, build_franka_demo, build_go2_demo, build_perception_demo
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(
            "Running Genesis sensor scenes requires a working Genesis + PyTorch runtime. "
            "Install torch in the target environment first."
        ) from exc

    builders: dict[str, Callable[..., DemoScene]] = {
        "drone": build_drone_demo,
        "perception": build_perception_demo,
        "franka": build_franka_demo,
        "go2": build_go2_demo,
    }
    demo_builder = builders[args.scene]
    try:
        demo: DemoScene = demo_builder(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(
            "Running Genesis sensor scenes requires a working Genesis + PyTorch runtime. "
            "Install torch in the target environment first."
        ) from exc
    demo.rig.reset()

    for step in range(args.steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        demo.rig.step(step * args.dt)


if __name__ == "__main__":
    main()
