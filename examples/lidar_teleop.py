from __future__ import annotations

import argparse
import os
from typing import Any

import numpy as np

import genesis as gs
from genesis.utils.geom import euler_to_quat
from genesis.utils.misc import tensor_to_array


def _describe_reading(reading: Any) -> str:
    if hasattr(reading, "points"):
        pts = np.asarray(tensor_to_array(getattr(reading, "points")))
        return f"points={len(pts)}"
    try:
        arr = np.asarray(tensor_to_array(reading))
        return f"shape={arr.shape}"
    except Exception:
        return type(reading).__name__


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone LiDAR/depth teleop-inspired demo")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--pattern", choices=("spherical", "depth", "grid"), default="spherical")
    args = parser.parse_args()

    steps = min(args.steps, 8) if "PYTEST_VERSION" in os.environ else args.steps
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(camera_pos=(-5.0, 0.0, 3.5), camera_lookat=(0.0, 0.0, 0.5)),
        show_viewer=args.vis,
    )
    scene.add_entity(gs.morphs.Plane())
    for angle in np.linspace(0.0, 2.0 * np.pi, 8, endpoint=False):
        scene.add_entity(
            gs.morphs.Cylinder(radius=0.25, height=1.2, pos=(3.0 * np.cos(angle), 3.0 * np.sin(angle), 0.6), fixed=True)
        )

    robot = scene.add_entity(gs.morphs.Box(size=(0.2, 0.2, 0.2), pos=(0.0, 0.0, 0.35), fixed=True))
    if args.pattern == "depth":
        sensor = scene.add_sensor(
            gs.sensors.DepthCamera(
                entity_idx=robot.idx,
                pos_offset=(0.25, 0.0, 0.1),
                return_world_frame=True,
                draw_debug=args.vis,
                pattern=gs.sensors.DepthCameraPattern(),
            )
        )
    else:
        pattern_cfg = gs.sensors.GridPattern() if args.pattern == "grid" else gs.sensors.SphericalPattern()
        sensor = scene.add_sensor(
            gs.sensors.Lidar(
                entity_idx=robot.idx,
                pos_offset=(0.25, 0.0, 0.1),
                return_world_frame=True,
                draw_debug=args.vis,
                pattern=pattern_cfg,
            )
        )

    scene.build()

    pos = np.array([0.0, 0.0, 0.35], dtype=np.float32)
    euler = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    for step in range(steps):
        pos[:] = np.array([0.8 * np.cos(0.15 * step), 0.6 * np.sin(0.18 * step), 0.35], dtype=np.float32)
        euler[2] = 0.2 * step
        robot.set_pos(pos)
        robot.set_quat(euler_to_quat(euler))
        scene.step()
        reading = sensor.read_image() if args.pattern == "depth" else sensor.read()
        if step % max(1, steps // 4) == 0:
            print(f"step={step:03d} pattern={args.pattern} {_describe_reading(reading)}")


if __name__ == "__main__":
    main()
