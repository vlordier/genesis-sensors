from __future__ import annotations

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array


def _disp_norm_max(sensor: object) -> float:
    data = sensor.read()
    displacement = getattr(data, "displacement", None)
    if displacement is None:
        return 0.0
    arr = np.asarray(tensor_to_array(displacement), dtype=float)
    if arr.size == 0:
        return 0.0
    return float(np.linalg.norm(arr, axis=-1).max())


def main() -> None:
    parser = argparse.ArgumentParser(description="Standalone tactile elastomer sandbox walkthrough")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--grid", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 8) if "PYTEST_VERSION" in os.environ else args.steps
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, precision="32", logging_level="warning")

    scene = gs.Scene(show_viewer=args.vis)
    scene.add_entity(gs.morphs.Plane())
    for i in range(4):
        scene.add_entity(
            gs.morphs.Box(
                size=(1.3, 0.08, 0.25),
                pos=(
                    0.0 if i < 2 else (0.65 if i == 2 else -0.65),
                    0.65 if i == 0 else (-0.65 if i == 1 else 0.0),
                    0.125,
                ),
                euler=(0, 0, 90 if i >= 2 else 0),
                fixed=True,
            )
        )

    if args.grid:
        pusher = scene.add_entity(gs.morphs.Box(size=(0.10, 0.10, 0.06), pos=(0.2, -0.3, 0.08)))
        probe_local_pos = gu.generate_grid_points_on_plane(
            lo=[-0.05, -0.05, -0.03], hi=[0.05, 0.05, -0.03], normal=(0.0, 0.0, -1.0), nx=6, ny=8
        )
    else:
        pusher = scene.add_entity(gs.morphs.Sphere(radius=0.10, pos=(0.2, -0.3, 0.11)))
        theta = np.linspace(np.pi / 2, np.pi, 4, endpoint=False)
        phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        theta, phi = map(np.ravel, np.meshgrid(theta, phi, indexing="ij"))
        probe_local_pos = 0.10 * np.stack(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)], axis=-1
        )

    tactile = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(
            entity_idx=pusher.idx,
            link_idx_local=0,
            probe_local_pos=probe_local_pos,
            probe_local_normal=(0.0, 0.0, -1.0),
            probe_radius=0.01,
            draw_debug=args.vis,
            dilate_coefficient=1e1,
            shear_coefficient=1e-2,
            twist_coefficient=1e-2,
        )
    )
    scene.add_entity(gs.morphs.Box(size=(0.08, 0.12, 0.08), pos=(0.15, 0.0, 0.04)))
    scene.add_entity(gs.morphs.Sphere(radius=0.04, pos=(-0.15, -0.1, 0.04)))
    scene.build()

    for step in range(steps):
        target = np.array(
            [0.20 * np.cos(0.2 * step), 0.18 * np.sin(0.25 * step), 0.12 if not args.grid else 0.08], dtype=np.float32
        )
        pusher.set_pos(target)
        scene.step()
        if step % max(1, steps // 4) == 0:
            print(f"step={step:03d} tactile_max_disp={_disp_norm_max(tactile):.6f}")


if __name__ == "__main__":
    main()
