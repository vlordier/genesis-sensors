"""CLI for running Genesis Sensors scenes and utilities."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from enum import Enum
from typing import TYPE_CHECKING

from . import __version__

if TYPE_CHECKING:
    from .scenes import DemoScene


_RUNTIME_ERROR = (
    "Running Genesis sensor scenes requires a working Genesis + PyTorch runtime. "
    "Install torch in the target environment first."
)

DEFAULT_STEPS = 200
DEFAULT_DT_S = 0.01
_EXAMPLES = "Examples:\n  genesis-sensors drone --steps 200\n  genesis-sensors perception --gpu --vis"


class ScenePreset(str, Enum):
    """Supported built-in demo scene presets exposed by the CLI."""

    DRONE = "drone"
    PERCEPTION = "perception"
    FRANKA = "franka"
    GO2 = "go2"


_SCENE_CHOICES = tuple(preset.value for preset in ScenePreset)


def _positive_int(value: str) -> int:
    """Argparse validator for strictly positive integer arguments."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    """Argparse validator for strictly positive floating-point arguments."""
    parsed = float(value)
    if parsed <= 0.0:
        raise argparse.ArgumentTypeError(f"expected a positive float, got {value!r}")
    return parsed


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser so tests and other entry points can reuse it."""
    parser = argparse.ArgumentParser(
        prog="genesis-sensors",
        description="Run Genesis Sensors example scenes",
        epilog=_EXAMPLES,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", choices=_SCENE_CHOICES, help="Scene preset to run")
    parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS, help="Number of simulation steps")
    parser.add_argument("--dt", type=_positive_float, default=DEFAULT_DT_S, help="Simulation timestep")
    parser.add_argument("--vis", action="store_true", help="Open the Genesis viewer")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU backend when available")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _get_scene_builders() -> dict[ScenePreset, Callable[..., DemoScene]]:
    """Load scene builders lazily so ``--help`` and ``--version`` stay lightweight."""
    from .scenes import build_drone_demo, build_franka_demo, build_go2_demo, build_perception_demo

    return {
        ScenePreset.DRONE: build_drone_demo,
        ScenePreset.PERCEPTION: build_perception_demo,
        ScenePreset.FRANKA: build_franka_demo,
        ScenePreset.GO2: build_go2_demo,
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    try:
        builders = _get_scene_builders()
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(_RUNTIME_ERROR) from exc

    demo_builder = builders[ScenePreset(args.scene)]
    try:
        demo: DemoScene = demo_builder(dt=args.dt, show_viewer=args.vis, use_gpu=args.gpu)
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(_RUNTIME_ERROR) from exc

    demo.rig.reset()
    for step in range(args.steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        demo.rig.step(step * args.dt)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_DT_S", "DEFAULT_STEPS", "ScenePreset", "main"]
