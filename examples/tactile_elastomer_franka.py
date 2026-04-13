from __future__ import annotations

import argparse
import os

import numpy as np

import genesis as gs
import genesis.utils.geom as gu
from genesis.utils.misc import tensor_to_array


OBJ_SIZE = 0.04
ROBOT_INIT_HEIGHT = 0.18


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
    parser = argparse.ArgumentParser(description="Standalone Franka tactile elastomer walkthrough")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    steps = min(args.steps, 8) if "PYTEST_VERSION" in os.environ else args.steps
    gs.init(backend=gs.cpu if args.cpu else gs.gpu, logging_level="warning")

    scene = gs.Scene(show_viewer=args.vis)
    scene.add_entity(gs.morphs.Plane())
    franka = scene.add_entity(gs.morphs.MJCF(file="xml/franka_emika_panda/panda.xml"))
    scene.add_entity(gs.morphs.Sphere(radius=OBJ_SIZE / 2, pos=(0.45, 0.0, OBJ_SIZE / 2)))

    probe_normal = (0.0, -1.0, 0.0)
    probe_local_pos = gu.generate_grid_points_on_plane(
        lo=(-0.006, 0.0, 0.04), hi=(0.008, 0.0, 0.05), normal=probe_normal, nx=6, ny=6
    )
    tactile_kwargs = dict(
        entity_idx=franka.idx,
        probe_local_pos=probe_local_pos,
        probe_local_normal=probe_normal,
        probe_radius=0.002,
        draw_debug=args.vis,
        dilate_coefficient=1e1,
        shear_coefficient=1e-2,
        twist_coefficient=1e-2,
    )
    left = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(link_idx_local=franka.get_link("left_finger").idx_local, **tactile_kwargs)
    )
    right = scene.add_sensor(
        gs.sensors.ElastomerDisplacement(link_idx_local=franka.get_link("right_finger").idx_local, **tactile_kwargs)
    )
    scene.build()

    motor_dofs = np.arange(franka.n_dofs - 2)
    ee_link = franka.get_link("hand")
    for step in range(steps):
        target_pos = np.array(
            [0.45 + 0.04 * np.cos(0.25 * step), 0.04 * np.sin(0.25 * step), ROBOT_INIT_HEIGHT], dtype=np.float32
        )
        target_quat = gu.xyz_to_quat(np.array([0.0, 180.0, 0.0], dtype=np.float32), degrees=True)
        qpos = franka.inverse_kinematics(link=ee_link, pos=target_pos, quat=target_quat, dofs_idx_local=motor_dofs)
        franka.control_dofs_position(qpos[motor_dofs], motor_dofs)
        scene.step()
        if step % max(1, steps // 4) == 0:
            print(
                f"step={step:03d} left_max_disp={_disp_norm_max(left):.6f} right_max_disp={_disp_norm_max(right):.6f}"
            )


if __name__ == "__main__":
    main()
