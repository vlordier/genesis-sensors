from __future__ import annotations

import argparse

from genesis_sensors import AnemometerModel, GasSensorModel, HygrometerModel, LightSensorModel, ThermometerModel
from genesis_sensors.synthetic import make_synthetic_sensor_state


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo the environmental sensor pack on top of the synthetic Genesis state")
    parser.add_argument("--frames", type=int, default=8)
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    thermometer = ThermometerModel(seed=args.seed)
    hygrometer = HygrometerModel(seed=args.seed + 1)
    light_sensor = LightSensorModel(seed=args.seed + 2)
    gas_sensor = GasSensorModel(seed=args.seed + 3)
    anemometer = AnemometerModel(seed=args.seed + 4)

    print("frame | temp_c | humidity | dew_point | lux | gas_ppm | wind_m/s | wind_deg")
    print("----- | ------ | -------- | --------- | --- | ------- | -------- | --------")
    for frame_idx in range(args.frames):
        sim_time = frame_idx * args.dt
        state = make_synthetic_sensor_state(frame_idx, dt=args.dt, total_frames=max(args.frames, 2))
        temp = thermometer.step(sim_time, state)
        humidity = hygrometer.step(sim_time, state)
        light = light_sensor.step(sim_time, state)
        gas = gas_sensor.step(sim_time, state)
        wind = anemometer.step(sim_time, state)
        print(
            f"{frame_idx:5d} | "
            f"{float(temp['temperature_c']):6.2f} | "
            f"{float(humidity['relative_humidity_pct']):8.2f} | "
            f"{float(humidity['dew_point_c']):9.2f} | "
            f"{float(light['illuminance_lux']):7.0f} | "
            f"{float(gas['concentration_ppm']):7.1f} | "
            f"{float(wind['wind_speed_ms']):8.2f} | "
            f"{float(wind['wind_direction_deg']):8.1f}"
        )


if __name__ == "__main__":
    main()
