from __future__ import annotations

import argparse
import os

import numpy as np

import genesis as gs
from genesis.utils.misc import tensor_to_array


PROBE_RADIUS = 0.05
PLATFORM_SIZE = 1.5
PLATFORM_HEIGHT = 0.3


def _build_probe_grid(grid_n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spacing = PLATFORM_SIZE / (grid_n + 1)
    centre = (grid_n - 1) / 2.0
    i = np.repeat(np.arange(grid_n), grid_n)
    j = np.tile(np.arange(grid_n), grid_n)
    x = (i - centre) * spacing
    y = (j - centre) * spacing
    z = np.full_like(x, PLATFORM_HEIGHT / 2)
    positions = np.stack([x, y, z], axis=-1)
    normals = np.tile([0.0, 0.0, 1.0], (grid_n * grid_n, 1))
    radii = np.full(grid_n * grid_n, PROBE_RADIUS)
    return positions, normals, radii


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone KinematicContactProbe walkthrough")
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 8) if "PYTEST_VERSION" in os.environ else args.steps
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(show_viewer=args.vis)
    scene.add_entity(gs.morphs.Plane())
    platform = scene.add_entity(
        gs.morphs.Box(
            size=(PLATFORM_SIZE, PLATFORM_SIZE, PLATFORM_HEIGHT), pos=(0.0, 0.0, PLATFORM_HEIGHT / 2), fixed=True
        )
    )
    probe_positions, probe_normals, probe_radii = _build_probe_grid(args.grid_size)
    probe = scene.add_sensor(
        gs.sensors.KinematicContactProbe(
            entity_idx=platform.idx,
            link_idx_local=0,
            probe_local_pos=probe_positions,
            probe_local_normal=probe_normals,
            probe_radius=probe_radii,
            stiffness=5000.0,
            draw_debug=args.vis,
        )
    )
    pusher = scene.add_entity(gs.morphs.Cylinder(radius=0.10, height=0.12, pos=(0.0, 0.0, PLATFORM_HEIGHT + 0.05)))
    scene.build()

    for step in range(steps):
        target = np.array(
            [0.35 * np.cos(0.25 * step), 0.35 * np.sin(0.2 * step), PLATFORM_HEIGHT + 0.05], dtype=np.float32
        )
        pusher.set_pos(target)
        scene.step()
        data = probe.read()
        penetration = np.asarray(tensor_to_array(data.penetration), dtype=float).reshape(-1)
        active = penetration[penetration > 0.0]
        if step % max(1, steps // 4) == 0:
            max_pen = float(active.max()) if active.size else 0.0
            print(f"step={step:03d} active_probes={int(active.size)} max_penetration={max_pen:.5f}m")


if __name__ == "__main__":
    main()
