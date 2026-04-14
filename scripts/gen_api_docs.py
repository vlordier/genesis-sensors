"""Generate API reference Markdown pages from docstrings.

Run with:
    python scripts/gen_api_docs.py

This creates one .md page per public module under ``docs/api/`` and
writes a nav-ready ``docs/api/index.md`` index.

The generated pages use mkdocstrings ::: directives so they render
automatically when ``mkdocs build`` or ``mkdocs serve`` is run.
"""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PACKAGE = "genesis_sensors._runtime_sensors"
API_DIR = Path(__file__).resolve().parent.parent / "docs" / "api"
ASSET_DIR = Path(__file__).resolve().parent.parent / "docs" / "assets" / "sensors"


@dataclass(frozen=True, slots=True)
class ModuleDocSpec:
    """A single API documentation page definition."""

    module_name: str
    page_title: str


# Modules to document, in nav order.
MODULES: tuple[ModuleDocSpec, ...] = (
    ModuleDocSpec("base", "Base Sensor"),
    ModuleDocSpec("scheduler", "Scheduler"),
    ModuleDocSpec("suite", "Sensor Suite"),
    ModuleDocSpec("config", "Configuration"),
    ModuleDocSpec("types", "Observation Types"),
    ModuleDocSpec("presets", "Presets"),
    ModuleDocSpec("_gauss_markov", "Gauss-Markov Process"),
    ModuleDocSpec("imu", "IMU"),
    ModuleDocSpec("gnss", "GNSS / GPS"),
    ModuleDocSpec("barometer", "Barometer"),
    ModuleDocSpec("magnetometer", "Magnetometer"),
    ModuleDocSpec("airspeed", "Airspeed"),
    ModuleDocSpec("camera_model", "RGB Camera"),
    ModuleDocSpec("stereo_camera", "Stereo Camera"),
    ModuleDocSpec("depth_camera", "Depth Camera"),
    ModuleDocSpec("thermal_camera", "Thermal Camera"),
    ModuleDocSpec("event_camera", "Event Camera"),
    ModuleDocSpec("lidar", "LiDAR"),
    ModuleDocSpec("rangefinder", "Rangefinder"),
    ModuleDocSpec("optical_flow", "Optical Flow"),
    ModuleDocSpec("ultrasonic", "Ultrasonic Array"),
    ModuleDocSpec("sonar", "Sonar (Imaging & Side-Scan)"),
    ModuleDocSpec("acoustic_navigation", "DVL & Current Profiler"),
    ModuleDocSpec("water_pressure", "Water Pressure"),
    ModuleDocSpec("hydrophone", "Hydrophone"),
    ModuleDocSpec("underwater_modem", "Underwater Modem"),
    ModuleDocSpec("environmental", "Environmental Sensors"),
    ModuleDocSpec("leak_detector", "Leak Detector"),
    ModuleDocSpec("wireless", "UWB & Radar"),
    ModuleDocSpec("radio", "Radio Link"),
    ModuleDocSpec("battery", "Battery"),
    ModuleDocSpec("wheel_odometry", "Wheel Odometry"),
    ModuleDocSpec("inclinometer", "Inclinometer"),
    ModuleDocSpec("force_torque", "Force / Torque"),
    ModuleDocSpec("joint_state", "Joint State"),
    ModuleDocSpec("contact_sensor", "Contact Sensor"),
    ModuleDocSpec("proximity_tof", "Proximity ToF Array"),
    ModuleDocSpec("tactile_array", "Tactile Array"),
    ModuleDocSpec("load_cell", "Load Cell"),
    ModuleDocSpec("current_sensor", "Current Sensor"),
    ModuleDocSpec("rpm_sensor", "RPM Sensor"),
    ModuleDocSpec("wire_encoder", "Wire Encoder"),
    ModuleDocSpec("motor_temperature", "Motor Temperature"),
)

EXAMPLE_ASSETS: dict[str, list[tuple[str, str]]] = {
    "imu": [("IMU generated example", "imu.svg")],
    "gnss": [("GNSS generated example", "gnss.svg")],
    "barometer": [("Barometer generated example", "barometer.svg")],
    "magnetometer": [("Magnetometer generated example", "magnetometer.svg")],
    "airspeed": [("Airspeed generated example", "airspeed.svg")],
    "camera_model": [("RGB camera snapshot", "camera_model.svg")],
    "stereo_camera": [("Stereo camera snapshot", "stereo_camera.svg")],
    "depth_camera": [("Depth camera snapshot", "depth_camera.svg")],
    "thermal_camera": [("Thermal camera snapshot", "thermal_camera.svg")],
    "event_camera": [("Event camera snapshot", "event_camera.svg")],
    "lidar": [("LiDAR snapshot", "lidar.svg")],
    "rangefinder": [("Rangefinder generated example", "rangefinder.svg")],
    "optical_flow": [("Optical flow generated example", "optical_flow.svg")],
    "ultrasonic": [("Ultrasonic array generated example", "ultrasonic.svg")],
    "sonar": [
        ("Imaging sonar snapshot", "imaging_sonar.svg"),
        ("Side-scan sonar snapshot", "side_scan_sonar.svg"),
    ],
    "acoustic_navigation": [
        ("DVL generated example", "dvl.svg"),
        ("Current profiler generated example", "acoustic_current_profiler.svg"),
    ],
    "water_pressure": [("Water-pressure generated example", "water_pressure.svg")],
    "hydrophone": [("Hydrophone generated example", "hydrophone.svg")],
    "underwater_modem": [("Underwater-modem generated example", "underwater_modem.svg")],
    "environmental": [
        ("Thermometer generated example", "thermometer.svg"),
        ("Hygrometer generated example", "hygrometer.svg"),
        ("Light sensor generated example", "light_sensor.svg"),
        ("Gas sensor generated example", "gas_sensor.svg"),
        ("Anemometer generated example", "anemometer.svg"),
    ],
    "wireless": [("UWB generated example", "uwb_ranging.svg"), ("Radar snapshot", "radar.svg")],
    "radio": [("Radio link generated example", "radio.svg")],
    "battery": [("Battery generated example", "battery.svg")],
    "wheel_odometry": [("Wheel odometry generated example", "wheel_odometry.svg")],
    "inclinometer": [("Inclinometer generated example", "inclinometer.svg")],
    "force_torque": [("Force / torque generated example", "force_torque.svg")],
    "joint_state": [("Joint-state generated example", "joint_state.svg")],
    "contact_sensor": [("Contact-sensor generated example", "contact_sensor.svg")],
    "proximity_tof": [("Proximity-ToF generated example", "proximity_tof.svg")],
    "tactile_array": [("Tactile-array generated example", "tactile_array.svg")],
    "load_cell": [("Load-cell generated example", "load_cell.svg")],
    "current_sensor": [("Current-sensor generated example", "current_sensor.svg")],
    "rpm_sensor": [("RPM-sensor generated example", "rpm_sensor.svg")],
    "wire_encoder": [("Wire-encoder generated example", "wire_encoder.svg")],
    "motor_temperature": [("Motor-temperature generated example", "motor_temperature.svg")],
    "leak_detector": [("Leak-detector generated example", "leak_detector.svg")],
}


def _module_path(mod_name: str) -> str:
    """Return fully qualified module path."""
    return f"{PACKAGE}.{mod_name}"


def _collect_public_names(mod_name: str) -> list[str]:
    """Import a module and return its __all__ or public names."""
    fqn = _module_path(mod_name)
    try:
        mod = importlib.import_module(fqn)
    except Exception as exc:
        print(f"  WARNING: cannot import {fqn}: {exc}")
        return []
    if hasattr(mod, "__all__"):
        return list(mod.__all__)
    return [name for name, obj in inspect.getmembers(mod) if not name.startswith("_") and inspect.getmodule(obj) is mod]


def _example_block(mod_name: str) -> list[str]:
    """Return optional example plot markdown for a module page."""
    assets = EXAMPLE_ASSETS.get(mod_name, [])
    if not assets:
        return []

    lines = [
        "## Generated example",
        "",
        "> Generated from `examples/generate_sensor_doc_assets.py` using the real sensor models driven by headless Genesis demo scenes.",
        "",
    ]
    added = False
    for caption, filename in assets:
        if not (ASSET_DIR / filename).exists():
            continue
        lines.extend([f"### {caption}", "", f"![{caption}](../assets/sensors/{filename})", ""])
        added = True
    return lines if added else []


def _write_module_page(mod_name: str, title: str) -> str:
    """Write a single API reference page and return its relative path."""
    fqn = _module_path(mod_name)
    public_names = _collect_public_names(mod_name)

    lines = [f"# {title}", "", *_example_block(mod_name)]
    if public_names:
        lines.extend(
            [
                "## Public symbols",
                "",
                ", ".join(f"`{name}`" for name in public_names),
                "",
            ]
        )
    lines.extend(
        [
            f"::: {fqn}",
            "    options:",
            "      show_root_heading: true",
            "      show_source: false",
            "      members_order: source",
            "      show_category_heading: true",
            "      merge_init_into_class: true",
        ]
    )

    filename = f"{mod_name.lstrip('_')}.md"
    filepath = API_DIR / filename
    filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"  wrote {filepath.relative_to(API_DIR.parent.parent)}")
    return f"api/{filename}"


def _write_index(page_paths: list[tuple[str, str]]) -> None:
    """Write the API reference index page."""
    lines = [
        "# API Reference",
        "",
        "Auto-generated from source docstrings.",
        "",
        "## Regenerate locally",
        "",
        "```bash",
        "PYTHONPATH=src python scripts/gen_api_docs.py",
        "```",
        "",
        "## Top-level helper modules",
        "",
        "- `genesis_sensors.config` — backend-aware validated config shims",
        "- `genesis_sensors.presets` — preset constants plus `get_preset()` / `list_presets()`",
        "- `genesis_sensors.genesis_bridge` — helper functions for extracting Genesis state",
        "- `genesis_sensors.synthetic` — headless synthetic-state builders for tests and demos",
        "- `genesis_sensors.robustness` — latency/dropout wrappers and health metadata",
        "",
        "## Runtime sensor modules",
        "",
    ]
    for title, rel_path in page_paths:
        basename = Path(rel_path).name.replace(".md", "")
        lines.append(f"- [{title}]({basename}.md)")
    lines.append("")

    index_path = API_DIR / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {index_path.relative_to(API_DIR.parent.parent)}")


def main() -> None:
    API_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating API docs in {API_DIR} ...")

    page_paths: list[tuple[str, str]] = []
    for spec in MODULES:
        rel = _write_module_page(spec.module_name, spec.page_title)
        page_paths.append((spec.page_title, rel))

    _write_index(page_paths)
    print(f"Done. {len(page_paths)} pages generated.")


if __name__ == "__main__":
    main()
