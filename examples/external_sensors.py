from __future__ import annotations

import argparse

import numpy as np

from genesis_sensors.rigs import make_synthetic_multimodal_rig
from genesis_sensors.synthetic import get_scenario_phase


def main() -> None:
    parser = argparse.ArgumentParser(description="Headless walkthrough of the standalone Genesis sensor stack")
    parser.add_argument("--frames", type=int, default=120)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rig = make_synthetic_multimodal_rig(dt=args.dt, seed=args.seed)
    rig.reset()

    print(f"sensors: {', '.join(rig.sensor_names())}")
    phase_starts = {
        0,
        max(0, int(0.20 * (args.frames - 1))),
        max(0, int(0.45 * (args.frames - 1))),
        max(0, int(0.70 * (args.frames - 1))),
        max(0, int(0.88 * (args.frames - 1))),
    }

    for frame_idx in range(args.frames):
        obs = rig.step(frame_idx * args.dt)
        if frame_idx in phase_starts or frame_idx == args.frames - 1:
            phase = get_scenario_phase(frame_idx / max(args.frames - 1, 1))
            rgb_mean = float(np.mean(obs["rgb"]["rgb"]))
            events = len(obs["events"]["events"])
            lidar_points = len(obs["lidar"]["points"])
            temp_max = float(np.max(obs["thermal"]["temperature_c"]))
            radio_delivered = len(obs["radio"]["delivered"])
            print(
                f"frame={frame_idx:03d} phase={phase:>15} rgb_mean={rgb_mean:6.1f} "
                f"events={events:4d} lidar_points={lidar_points:4d} thermal_peak={temp_max:5.1f}C "
                f"radio_delivered={radio_delivered}"
            )


if __name__ == "__main__":
    main()
