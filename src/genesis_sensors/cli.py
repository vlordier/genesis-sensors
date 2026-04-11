"""CLI for running Genesis Sensors scenes and utilities."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
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
DEFAULT_SUMMARY_EVERY = 0
_SUMMARY_KEY_LIMIT = 6
_SUMMARY_PRIORITY_KEYS = ("imu", "gnss", "lidar", "rgb", "thermal", "radio")
_EXAMPLES = (
    "Examples:\n"
    "  genesis-sensors drone --steps 200\n"
    "  genesis-sensors perception --gpu --vis\n"
    "  genesis-sensors synthetic --steps 24 --summary-every 6"
)


class ScenePreset(str, Enum):
    """Supported built-in demo scene presets exposed by the CLI."""

    DRONE = "drone"
    PERCEPTION = "perception"
    FRANKA = "franka"
    GO2 = "go2"
    SYNTHETIC = "synthetic"


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


def _non_negative_int(value: str) -> int:
    """Argparse validator for non-negative integer arguments."""
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError(f"expected a non-negative integer, got {value!r}")
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
    parser.add_argument(
        "--summary-every",
        type=_non_negative_int,
        default=DEFAULT_SUMMARY_EVERY,
        help="Print observed sensor keys every N steps (0 disables)",
    )
    parser.add_argument("--vis", action="store_true", help="Open the Genesis viewer")
    parser.add_argument("--gpu", action="store_true", help="Use the GPU backend when available")
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    return parser


def _get_scene_builders() -> dict[ScenePreset, Callable[..., DemoScene]]:
    """Load scene builders lazily so ``--help`` and ``--version`` stay lightweight."""
    from .scenes import build_drone_demo, build_franka_demo, build_go2_demo, build_perception_demo, build_synthetic_demo

    return {
        ScenePreset.DRONE: build_drone_demo,
        ScenePreset.PERCEPTION: build_perception_demo,
        ScenePreset.FRANKA: build_franka_demo,
        ScenePreset.GO2: build_go2_demo,
        ScenePreset.SYNTHETIC: build_synthetic_demo,
    }


def _make_summary_callback(
    scene_name: str, *, summary_every: int
) -> Callable[[int, Mapping[str, object]], None] | None:
    """Return a step callback that prints a short sensor summary at a fixed cadence."""
    if summary_every <= 0:
        return None

    def _callback(step: int, observation: Mapping[str, object]) -> None:
        if step % summary_every != 0:
            return
        available_keys = {str(key) for key in observation}
        prioritized_keys = [key for key in _SUMMARY_PRIORITY_KEYS if key in available_keys]
        remaining_keys = sorted(key for key in available_keys if key not in _SUMMARY_PRIORITY_KEYS)
        keys = prioritized_keys + remaining_keys
        preview = ", ".join(keys[:_SUMMARY_KEY_LIMIT])
        if len(keys) > _SUMMARY_KEY_LIMIT:
            preview = f"{preview}, ..."
        print(f"[{scene_name}] step={step:03d} sensors={preview}")

    return _callback


def _run_demo(demo: DemoScene, *, steps: int, dt: float, summary_every: int) -> None:
    """Run a demo scene with optional periodic observation summaries."""
    on_step = _make_summary_callback(demo.name, summary_every=summary_every)
    run_method = getattr(demo, "run", None)
    if callable(run_method):
        run_method(steps=steps, dt=dt, on_step=on_step)
        return

    demo.rig.reset()
    for step in range(steps):
        if demo.controller is not None:
            demo.controller(step)
        demo.scene.step()
        observation = demo.rig.step(step * dt)
        if on_step is not None:
            on_step(step, observation)


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

    _run_demo(demo, steps=args.steps, dt=args.dt, summary_every=args.summary_every)


if __name__ == "__main__":
    main()


__all__ = ["DEFAULT_DT_S", "DEFAULT_STEPS", "DEFAULT_SUMMARY_EVERY", "ScenePreset", "main"]
