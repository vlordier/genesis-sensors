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
import os
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

PACKAGE = "genesis_sensors._runtime_sensors"
API_DIR = Path(__file__).resolve().parent.parent / "docs" / "api"
# Modules to document, in nav order.  Each tuple is (module_name, page_title).
MODULES: list[tuple[str, str]] = [
    ("base", "Base Sensor"),
    ("scheduler", "Scheduler"),
    ("suite", "Sensor Suite"),
    ("config", "Configuration"),
    ("types", "Observation Types"),
    ("presets", "Presets"),
    ("_gauss_markov", "Gauss-Markov Process"),
    ("imu", "IMU"),
    ("gnss", "GNSS / GPS"),
    ("barometer", "Barometer"),
    ("magnetometer", "Magnetometer"),
    ("airspeed", "Airspeed"),
    ("camera_model", "RGB Camera"),
    ("stereo_camera", "Stereo Camera"),
    ("depth_camera", "Depth Camera"),
    ("thermal_camera", "Thermal Camera"),
    ("event_camera", "Event Camera"),
    ("lidar", "LiDAR"),
    ("rangefinder", "Rangefinder"),
    ("optical_flow", "Optical Flow"),
    ("ultrasonic", "Ultrasonic Array"),
    ("sonar", "Sonar (Imaging & Side-Scan)"),
    ("acoustic_navigation", "DVL & Current Profiler"),
    ("environmental", "Environmental Sensors"),
    ("wireless", "UWB & Radar"),
    ("radio", "Radio Link"),
    ("battery", "Battery"),
    ("wheel_odometry", "Wheel Odometry"),
    ("force_torque", "Force / Torque"),
    ("joint_state", "Joint State"),
    ("contact_sensor", "Contact Sensor"),
    ("tactile_array", "Tactile Array"),
    ("current_sensor", "Current Sensor"),
    ("rpm_sensor", "RPM Sensor"),
]


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
    return [
        name
        for name, obj in inspect.getmembers(mod)
        if not name.startswith("_") and inspect.getmodule(obj) is mod
    ]


def _write_module_page(mod_name: str, title: str) -> str:
    """Write a single API reference page and return its relative path."""
    fqn = _module_path(mod_name)
    names = _collect_public_names(mod_name)

    lines = [
        f"# {title}",
        "",
        f"::: {fqn}",
        "    options:",
        "      show_root_heading: true",
        "      show_source: false",
        "      members_order: source",
        "      show_category_heading: true",
        "      merge_init_into_class: true",
    ]

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
        "## Modules",
        "",
    ]
    for title, rel_path in page_paths:
        basename = os.path.basename(rel_path).replace(".md", "")
        lines.append(f"- [{title}]({basename}.md)")
    lines.append("")

    index_path = API_DIR / "index.md"
    index_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"  wrote {index_path.relative_to(API_DIR.parent.parent)}")


def main() -> None:
    API_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Generating API docs in {API_DIR} ...")

    page_paths: list[tuple[str, str]] = []
    for mod_name, title in MODULES:
        rel = _write_module_page(mod_name, title)
        page_paths.append((title, rel))

    _write_index(page_paths)
    print(f"Done. {len(page_paths)} pages generated.")


if __name__ == "__main__":
    main()
