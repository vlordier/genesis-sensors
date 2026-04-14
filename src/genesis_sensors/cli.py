"""CLI for running Genesis Sensors scenes and utilities."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING

from . import __version__
from .rigs import RigProfile
from .scenes import SceneRuntimeMode

if TYPE_CHECKING:
    from .scenes import DemoScene


_RUNTIME_ERROR = (
    "Running Genesis sensor scenes requires a working Genesis + PyTorch runtime. "
    "Install torch in the target environment first."
)

DEFAULT_STEPS = 200
DEFAULT_DT_S = 0.01
DEFAULT_SUMMARY_EVERY = 0
_SCENE_DEFAULT_DT_MARKER = "scene-default"
_SCENE_NAME_WIDTH = 10
_RUNTIME_MODE_WIDTH = 8
_SUMMARY_KEY_LIMIT = 6
_SUMMARY_PRIORITY_KEYS = ("imu", "gnss", "lidar", "rgb", "thermal", "radio")
_EXAMPLES = (
    "Examples:\n"
    "  genesis-sensors --list-scenes\n"
    "  genesis-sensors --list-scenes --runtime-mode headless --show-commands --summary-format json\n"
    "  genesis-sensors --list-scenes --profile perception --summary-format json\n"
    "  genesis-sensors --catalog-summary --summary-format json\n"
    "  genesis-sensors --list-profiles --summary-format json\n"
    "  genesis-sensors --describe-scene synthetic --summary-format json\n"
    "  genesis-sensors --list-phases --summary-format json\n"
    "  genesis-sensors drone --steps 200\n"
    "  genesis-sensors perception --gpu --vis\n"
    "  genesis-sensors synthetic --steps 24 --summary-every 6\n"
    "  genesis-sensors synthetic --dry-run --summary-format json --write-summary /tmp/synthetic.json"
)


class ScenePreset(str, Enum):
    """Supported built-in demo scene presets exposed by the CLI."""

    DRONE = "drone"
    PERCEPTION = "perception"
    FRANKA = "franka"
    GO2 = "go2"
    SYNTHETIC = "synthetic"


class SummaryFormat(str, Enum):
    """Output format for structured CLI summaries."""

    TEXT = "text"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class CLIRunConfig:
    """Validated runtime configuration derived from CLI arguments."""

    scene: ScenePreset | None
    steps: int
    dt: float
    summary_every: int
    summary_format: SummaryFormat
    profile_filter: RigProfile | None
    runtime_mode_filter: SceneRuntimeMode | None
    search: str | None
    headless_only: bool
    show_viewer: bool
    use_gpu: bool
    dry_run: bool
    list_scenes: bool
    list_phases: bool
    list_profiles: bool
    catalog_summary: bool
    describe_scene: ScenePreset | None
    show_commands: bool
    write_summary: str | None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "CLIRunConfig":
        scene = ScenePreset(args.scene) if args.scene is not None else None
        dt = _default_dt_for_scene(scene) if args.dt == _SCENE_DEFAULT_DT_MARKER else float(args.dt)
        return cls(
            scene=scene,
            steps=args.steps,
            dt=dt,
            summary_every=args.summary_every,
            summary_format=SummaryFormat(args.summary_format),
            profile_filter=RigProfile(args.profile) if args.profile is not None else None,
            runtime_mode_filter=SceneRuntimeMode(args.runtime_mode) if args.runtime_mode is not None else None,
            search=args.search.strip() or None if args.search is not None else None,
            headless_only=args.headless_only,
            show_viewer=args.vis,
            use_gpu=args.gpu,
            dry_run=args.dry_run,
            list_scenes=args.list_scenes,
            list_phases=args.list_phases,
            list_profiles=args.list_profiles,
            catalog_summary=args.catalog_summary,
            describe_scene=ScenePreset(args.describe_scene) if args.describe_scene is not None else None,
            show_commands=args.show_commands,
            write_summary=args.write_summary,
        )


_SCENE_CHOICES = tuple(preset.value for preset in ScenePreset)
_PROFILE_CHOICES = RigProfile.values()
_RUNTIME_MODE_CHOICES = SceneRuntimeMode.values()
_SUMMARY_FORMAT_CHOICES = tuple(fmt.value for fmt in SummaryFormat)


def _positive_int(value: str) -> int:
    """Argparse validator for strictly positive integer arguments."""
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value!r}")
    return parsed


def _positive_float(value: str) -> float | str:
    """Argparse validator for strictly positive floating-point arguments."""
    if value == _SCENE_DEFAULT_DT_MARKER:
        return value
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


def _default_dt_for_scene(scene: ScenePreset | None) -> float:
    """Return the recommended timestep for the selected scene preset."""
    if scene is None:
        return DEFAULT_DT_S

    from .scenes import get_demo_scene_spec

    try:
        return get_demo_scene_spec(scene.value).default_dt
    except KeyError:
        return DEFAULT_DT_S


def _build_parser() -> argparse.ArgumentParser:
    """Create the CLI parser so tests and other entry points can reuse it."""
    parser = argparse.ArgumentParser(
        prog="genesis-sensors",
        description="Run Genesis Sensors example scenes",
        epilog=_EXAMPLES,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("scene", nargs="?", choices=_SCENE_CHOICES, help="Scene preset to run")
    parser.add_argument("--steps", type=_positive_int, default=DEFAULT_STEPS, help="Number of simulation steps")
    parser.add_argument(
        "--dt",
        type=_positive_float,
        default=_SCENE_DEFAULT_DT_MARKER,
        help="Simulation timestep (uses the scene default when omitted)",
    )
    parser.add_argument(
        "--summary-every",
        type=_non_negative_int,
        default=DEFAULT_SUMMARY_EVERY,
        help="Print observed sensor keys every N steps (0 disables)",
    )
    parser.add_argument(
        "--summary-format",
        choices=_SUMMARY_FORMAT_CHOICES,
        default=SummaryFormat.TEXT.value,
        help="Render dry-run and scene-list summaries as text or JSON",
    )
    parser.add_argument("--profile", choices=_PROFILE_CHOICES, help="Filter listed scenes by rig profile")
    parser.add_argument("--runtime-mode", choices=_RUNTIME_MODE_CHOICES, help="Filter listed scenes by runtime mode")
    parser.add_argument("--search", help="Filter listed scenes by name, description, or profile text")
    parser.add_argument("--headless-only", action="store_true", help="Only list scenes that do not require Genesis")
    parser.add_argument("--write-summary", help="Write the dry-run or scene-list summary to a file")
    parser.add_argument(
        "--dry-run", action="store_true", help="Build the demo and print its summary without stepping it"
    )
    parser.add_argument("--list-scenes", action="store_true", help="List the built-in scene presets and exit")
    parser.add_argument("--list-phases", action="store_true", help="List the synthetic scenario phases and exit")
    parser.add_argument("--list-profiles", action="store_true", help="List the available rig profiles and exit")
    parser.add_argument(
        "--catalog-summary", action="store_true", help="Print a summary of the built-in scene catalog and exit"
    )
    parser.add_argument(
        "--describe-scene", choices=_SCENE_CHOICES, help="Print metadata for one built-in scene and exit"
    )
    parser.add_argument(
        "--show-commands", action="store_true", help="Include recommended CLI commands in catalog text output"
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


def _emit_output(text: str, *, write_summary: str | None = None) -> None:
    """Print text to stdout and optionally persist it to a summary file."""
    print(text)
    if write_summary is not None:
        output_path = Path(write_summary)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(f"{text}\n", encoding="utf-8")


def _print_scene_catalog(
    *,
    summary_format: SummaryFormat,
    profile_filter: RigProfile | None,
    runtime_mode_filter: SceneRuntimeMode | None,
    search: str | None,
    headless_only: bool,
    show_commands: bool,
    write_summary: str | None,
) -> None:
    """Print the built-in scene catalog without requiring the Genesis runtime."""
    from .scenes import filter_demo_scenes

    specs = filter_demo_scenes(
        profile=profile_filter,
        requires_runtime=False if headless_only else None,
        runtime_mode=runtime_mode_filter,
        query=search,
    )
    if summary_format is SummaryFormat.JSON:
        payload = [spec.as_dict() for spec in specs]
        _emit_output(json.dumps(payload, indent=2, sort_keys=True), write_summary=write_summary)
        return

    lines = []
    for spec in specs:
        line = (
            f"{spec.name:<{_SCENE_NAME_WIDTH}} "
            f"{spec.runtime_mode.value:<{_RUNTIME_MODE_WIDTH}} "
            f"dt={spec.default_dt:.2f} "
            f"profile={spec.profile.value}  {spec.description}"
        )
        if show_commands:
            line = f"{line}\n  command: {spec.recommended_command()}"
        lines.append(line)
    _emit_output("\n".join(lines) if lines else "No scenes matched the current filters.", write_summary=write_summary)


def _print_catalog_summary(
    *,
    summary_format: SummaryFormat,
    profile_filter: RigProfile | None,
    runtime_mode_filter: SceneRuntimeMode | None,
    search: str | None,
    headless_only: bool,
    write_summary: str | None,
) -> None:
    """Print a compact summary of the built-in demo catalog."""
    from .scenes import describe_demo_catalog

    summary = describe_demo_catalog(
        profile=profile_filter,
        requires_runtime=False if headless_only else None,
        runtime_mode=runtime_mode_filter,
        query=search,
    )
    if summary_format is SummaryFormat.JSON:
        _emit_output(json.dumps(summary.as_dict(), indent=2, sort_keys=True), write_summary=write_summary)
        return

    _emit_output(
        "\n".join(
            [
                f"scene_count={summary.scene_count}",
                f"profiles={summary.profile_counts}",
                f"runtime_modes={summary.runtime_counts}",
                *[f"command[{index}]={command}" for index, command in enumerate(summary.preview_commands(), start=1)],
            ]
        ),
        write_summary=write_summary,
    )


def _print_profile_catalog(
    *,
    summary_format: SummaryFormat,
    search: str | None,
    write_summary: str | None,
) -> None:
    """Print the available demo rig profiles and their scene counts."""
    from .scenes import describe_demo_catalog, filter_demo_scenes

    summary = describe_demo_catalog(query=search)
    profiles: list[tuple[RigProfile, tuple[str, ...]]] = []
    for profile in RigProfile:
        scene_names = tuple(spec.name for spec in filter_demo_scenes(profile=profile, query=search))
        if not scene_names:
            continue
        profiles.append((profile, scene_names))

    if summary_format is SummaryFormat.JSON:
        payload = [
            {
                "profile": profile.value,
                "label": profile.label,
                "scene_count": len(scene_names),
                "scene_names": list(scene_names),
            }
            for profile, scene_names in profiles
        ]
        _emit_output(json.dumps(payload, indent=2, sort_keys=True), write_summary=write_summary)
        return

    lines = [f"matched_scenes={summary.scene_count}"]
    for profile, scene_names in profiles:
        lines.append(f"{profile.value:<22} scenes={len(scene_names)} names={', '.join(scene_names)}")
    _emit_output("\n".join(lines), write_summary=write_summary)


def _print_scene_description(
    scene_name: ScenePreset,
    *,
    summary_format: SummaryFormat,
    write_summary: str | None,
) -> None:
    """Print metadata for a single built-in scene without constructing the runtime demo."""
    from .scenes import get_demo_scene_spec

    spec = get_demo_scene_spec(scene_name.value)
    payload = spec.as_dict()
    if summary_format is SummaryFormat.JSON:
        _emit_output(json.dumps(payload, indent=2, sort_keys=True), write_summary=write_summary)
        return

    _emit_output(
        "\n".join(
            [
                f"scene={payload['name']} runtime={payload['runtime_mode']} profile={payload['profile']}",
                f"default_dt={payload['default_dt']:.2f} headless={payload['is_headless']}",
                payload["description"],
            ]
        ),
        write_summary=write_summary,
    )


def _print_phase_catalog(*, summary_format: SummaryFormat, write_summary: str | None) -> None:
    """Print the named synthetic scenario phases without requiring the Genesis runtime."""
    from .synthetic import list_scenario_windows

    windows = list_scenario_windows()
    if summary_format is SummaryFormat.JSON:
        payload = [
            {
                "phase": window.phase.value,
                "start": round(window.start, 4),
                "end": round(window.end, 4),
                "duration": round(window.duration, 4),
            }
            for window in windows
        ]
        _emit_output(json.dumps(payload, indent=2, sort_keys=True), write_summary=write_summary)
        return

    lines = [f"{window.phase.value:<18} start={window.start:.2f} end={window.end:.2f}" for window in windows]
    _emit_output("\n".join(lines), write_summary=write_summary)


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


def _print_demo_summary(
    demo: DemoScene,
    *,
    summary_format: SummaryFormat,
    write_summary: str | None = None,
) -> None:
    """Print a structured description of the selected demo without running it."""
    describe_demo = getattr(demo, "describe", None)
    if callable(describe_demo):
        summary = dict(describe_demo())
    else:
        summary = {
            "scene": demo.name,
            "entity_present": demo.entity is not None,
            "has_controller": demo.controller is not None,
            "rig": demo.rig.describe().as_dict(),
        }

    if summary_format is SummaryFormat.JSON:
        _emit_output(json.dumps(summary, indent=2, sort_keys=True), write_summary=write_summary)
        return

    rig = dict(summary["rig"])
    sensor_names = ", ".join(rig["sensor_names"])
    _emit_output(
        "\n".join(
            [
                f"scene={summary['scene']} profile={rig['profile']} sensors={rig['sensor_count']} "
                f"controller={summary['has_controller']}",
                f"sensor_names={sensor_names}",
            ]
        ),
        write_summary=write_summary,
    )


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
    config = CLIRunConfig.from_namespace(args)

    if config.headless_only and config.runtime_mode_filter == SceneRuntimeMode.RUNTIME:
        parser.error("conflicting runtime filters: --headless-only cannot be combined with --runtime-mode runtime")

    if config.list_scenes:
        _print_scene_catalog(
            summary_format=config.summary_format,
            profile_filter=config.profile_filter,
            runtime_mode_filter=config.runtime_mode_filter,
            search=config.search,
            headless_only=config.headless_only,
            show_commands=config.show_commands,
            write_summary=config.write_summary,
        )
        return
    if config.list_phases:
        _print_phase_catalog(summary_format=config.summary_format, write_summary=config.write_summary)
        return
    if config.list_profiles:
        _print_profile_catalog(
            summary_format=config.summary_format,
            search=config.search,
            write_summary=config.write_summary,
        )
        return
    if config.catalog_summary:
        _print_catalog_summary(
            summary_format=config.summary_format,
            profile_filter=config.profile_filter,
            runtime_mode_filter=config.runtime_mode_filter,
            search=config.search,
            headless_only=config.headless_only,
            write_summary=config.write_summary,
        )
        return
    if config.describe_scene is not None:
        _print_scene_description(
            config.describe_scene,
            summary_format=config.summary_format,
            write_summary=config.write_summary,
        )
        return
    if config.scene is None:
        parser.error("scene is required unless a catalog option like --list-scenes or --catalog-summary is used")

    try:
        builders = _get_scene_builders()
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(_RUNTIME_ERROR) from exc

    demo_builder = builders[config.scene]
    try:
        demo: DemoScene = demo_builder(dt=config.dt, show_viewer=config.show_viewer, use_gpu=config.use_gpu)
    except ImportError as exc:  # pragma: no cover - depends on optional runtime deps
        raise SystemExit(_RUNTIME_ERROR) from exc

    if config.dry_run:
        _print_demo_summary(demo, summary_format=config.summary_format, write_summary=config.write_summary)
        return

    _run_demo(demo, steps=config.steps, dt=config.dt, summary_every=config.summary_every)


if __name__ == "__main__":
    main()


__all__ = [
    "CLIRunConfig",
    "DEFAULT_DT_S",
    "DEFAULT_STEPS",
    "DEFAULT_SUMMARY_EVERY",
    "ScenePreset",
    "SummaryFormat",
    "main",
]
