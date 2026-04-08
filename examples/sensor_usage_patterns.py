from __future__ import annotations

import argparse
from typing import cast

import numpy as np

from genesis_sensors import (
    AirspeedModel,
    BarometerModel,
    BatteryModel,
    CameraModel,
    EventCameraModel,
    GNSSModel,
    IMUModel,
    LidarModel,
    MagnetometerModel,
    OpticalFlowModel,
    RadioLinkModel,
    RangefinderModel,
    SensorScheduler,
    SensorSuite,
    StereoCameraModel,
    ThermalCameraModel,
    WheelOdometryModel,
    get_preset,
    list_presets,
)
from genesis_sensors.config import (
    CameraConfig,
    EventCameraConfig,
    GNSSConfig,
    LidarConfig,
    StereoCameraConfig,
    ThermalCameraConfig,
    WheelOdometryConfig,
)
from genesis_sensors.synthetic import GNSS_ORIGIN_LLH, make_synthetic_sensor_state


def _fmt_vec(values: np.ndarray, precision: int = 3) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in np.asarray(values).tolist()) + "]"


def demo_direct_usage(seed: int) -> None:
    state0 = make_synthetic_sensor_state(0)
    state1 = make_synthetic_sensor_state(1)

    cam = CameraModel(resolution=(96, 72), seed=seed)
    event_cam = EventCameraModel(update_rate_hz=200.0, seed=seed)
    thermal = ThermalCameraModel(resolution=(96, 72), seed=seed)
    lidar = LidarModel(n_channels=8, h_resolution=64, max_range_m=12.0, seed=seed)
    gnss = GNSSModel(origin_llh=GNSS_ORIGIN_LLH, noise_m=0.25, vel_noise_ms=0.02, seed=seed)
    imu = IMUModel(update_rate_hz=200.0, seed=seed)
    baro = BarometerModel(update_rate_hz=50.0, seed=seed)
    mag = MagnetometerModel(update_rate_hz=100.0, seed=seed)
    airspeed = AirspeedModel(noise_sigma_ms=0.3, min_detectable_ms=2.0, seed=seed)
    rangefinder = RangefinderModel(update_rate_hz=20.0, seed=seed)
    flow = OpticalFlowModel(update_rate_hz=100.0, seed=seed)
    battery = BatteryModel(n_cells=4, capacity_mah=5000.0, seed=seed)

    cam_obs = cam.step(0.0, state1)
    event_cam.step(0.0, state0)
    event_obs = event_cam.step(0.05, state1)
    thermal_obs = thermal.step(0.0, state1)
    lidar_obs = lidar.step(0.0, state1)
    gnss_obs = gnss.step(0.0, state1)
    imu_obs = imu.step(0.0, state1)
    baro_obs = baro.step(0.0, state1)
    mag_obs = mag.step(0.0, state1)
    airspeed_obs = airspeed.step(0.0, state1)
    range_obs = rangefinder.step(0.0, state1)
    flow_obs = flow.step(0.0, state1)
    bat_obs = battery.step(0.0, state1)

    print("\n=== 1) Direct model stepping ===")
    print(
        f"rgb_mean={float(np.mean(cam_obs['rgb'])):.1f} events={len(event_obs['events'])} "
        f"thermal_peak={float(np.max(thermal_obs['temperature_c'])):.1f}C lidar_points={len(lidar_obs['points'])}"
    )
    print(
        f"gnss_hdop={float(gnss_obs['hdop']):.2f} imu_acc={_fmt_vec(np.asarray(imu_obs['lin_acc']))} "
        f"baro_alt={float(baro_obs['altitude_m']):.1f}m mag_norm={float(mag_obs['field_norm_ut']):.1f}uT"
    )
    print(
        f"airspeed={float(airspeed_obs['airspeed_ms']):.2f}m/s range={float(range_obs['range_m']):.2f}m "
        f"flow_quality={int(flow_obs['quality'])} bat_soc={float(bat_obs['soc']) * 100:.1f}%"
    )


def demo_preset_usage(seed: int) -> None:
    state = make_synthetic_sensor_state(3)

    stereo_cfg = cast(
        StereoCameraConfig,
        get_preset("ZED2_STEREO").model_copy(update={"resolution": (96, 72), "seed": seed}),
    )
    lidar_cfg = cast(
        LidarConfig,
        get_preset("VELODYNE_VLP16").model_copy(
            update={"n_channels": 8, "h_resolution": 64, "max_range_m": 12.0, "seed": seed}
        ),
    )
    thermal_cfg = cast(
        ThermalCameraConfig,
        get_preset("FLIR_BOSON_320").model_copy(update={"resolution": (96, 72), "seed": seed}),
    )
    gnss_cfg = cast(
        GNSSConfig,
        get_preset("UBLOX_F9P_RTK").model_copy(update={"origin_llh": GNSS_ORIGIN_LLH, "seed": seed}),
    )

    stereo = StereoCameraModel.from_config(stereo_cfg)
    lidar = LidarModel.from_config(lidar_cfg)
    thermal = ThermalCameraModel.from_config(thermal_cfg)
    gnss = GNSSModel.from_config(gnss_cfg)

    stereo_obs = stereo.step(0.0, state)
    lidar_obs = lidar.step(0.0, state)
    thermal_obs = thermal.step(0.0, state)
    gnss_obs = gnss.step(0.0, state)

    print("\n=== 2) Preset-driven usage ===")
    print(f"camera presets: {', '.join(list_presets(kind='camera')[:5])}")
    print(f"stereo presets: {', '.join(list_presets(kind='stereo'))}")
    print(f"thermal presets: {', '.join(list_presets(kind='thermal'))}")
    print(
        f"ZED2_STEREO valid_frac={float(np.mean(stereo_obs['valid_mask'])):.1%} "
        f"VELODYNE_VLP16 points={len(lidar_obs['points'])} FLIR_BOSON_320 peak={float(np.max(thermal_obs['temperature_c'])):.1f}C"
    )
    print(
        f"UBLOX_F9P_RTK fix_quality={int(gnss_obs['fix_quality'])} pos_llh={_fmt_vec(np.asarray(gnss_obs['pos_llh']), 5)}"
    )


def demo_suite_usage(frames: int, dt: float, seed: int) -> None:
    rgb_cfg = cast(
        CameraConfig,
        get_preset("RASPBERRY_PI_V2").model_copy(update={"resolution": (96, 72), "seed": seed}),
    )
    event_cfg = cast(EventCameraConfig, get_preset("DAVIS_346").model_copy(update={"seed": seed + 1}))
    thermal_cfg = cast(
        ThermalCameraConfig,
        get_preset("FLIR_BOSON_320").model_copy(update={"resolution": (96, 72), "seed": seed + 2}),
    )
    lidar_cfg = cast(
        LidarConfig,
        get_preset("VELODYNE_VLP16").model_copy(
            update={"n_channels": 8, "h_resolution": 64, "max_range_m": 12.0, "seed": seed + 3}
        ),
    )
    stereo_cfg = cast(
        StereoCameraConfig,
        get_preset("ZED2_STEREO").model_copy(update={"resolution": (96, 72), "seed": seed + 4}),
    )
    wheel_cfg = cast(
        WheelOdometryConfig,
        get_preset("DIFF_DRIVE_ENCODER_50HZ").model_copy(update={"seed": seed + 14}),
    )

    suite = SensorSuite(
        rgb_camera=CameraModel.from_config(rgb_cfg),
        event_camera=EventCameraModel.from_config(event_cfg),
        thermal_camera=ThermalCameraModel.from_config(thermal_cfg),
        lidar=LidarModel.from_config(lidar_cfg),
        stereo_camera=StereoCameraModel.from_config(stereo_cfg),
        gnss=GNSSModel(origin_llh=GNSS_ORIGIN_LLH, seed=seed + 5),
        imu=IMUModel(update_rate_hz=200.0, seed=seed + 6),
        barometer=BarometerModel(update_rate_hz=50.0, seed=seed + 7),
        magnetometer=MagnetometerModel(update_rate_hz=100.0, seed=seed + 8),
        airspeed=AirspeedModel(update_rate_hz=50.0, seed=seed + 9),
        rangefinder=RangefinderModel(update_rate_hz=20.0, seed=seed + 10),
        optical_flow=OpticalFlowModel(update_rate_hz=100.0, seed=seed + 11),
        battery=BatteryModel(n_cells=4, capacity_mah=5000.0, seed=seed + 12),
        radio=RadioLinkModel(update_rate_hz=60.0, seed=seed + 13),
        wheel_odometry=WheelOdometryModel.from_config(wheel_cfg),
    )
    suite.reset()

    print("\n=== 3) SensorSuite usage ===")
    for frame_idx in range(frames):
        state = make_synthetic_sensor_state(frame_idx, dt=dt, total_frames=frames)
        if frame_idx % 15 == 0:
            radio_sensor = suite.get_sensor("radio")
            assert isinstance(radio_sensor, RadioLinkModel)
            radio_sensor.transmit(
                packet={"frame": frame_idx},
                src_pos=np.asarray(state["pos"], dtype=np.float64),
                dst_pos=np.array([0.0, 0.0, 1.0], dtype=np.float64),
                sim_time=frame_idx * dt,
            )
        obs = suite.step(frame_idx * dt, state)
        if frame_idx in {0, frames // 2, frames - 1}:
            print(
                f"frame={frame_idx:03d} rgb={np.asarray(obs['rgb']['rgb']).shape} "
                f"stereo_valid={float(np.mean(obs['stereo']['valid_mask'])):.1%} radio_queue={int(obs['radio']['queue_depth'])}"
            )


def demo_scheduler_usage(frames: int, dt: float, seed: int) -> None:
    scheduler = SensorScheduler()
    scheduler.add(IMUModel(update_rate_hz=200.0, seed=seed), name="imu")
    scheduler.add(GNSSModel(update_rate_hz=5.0, origin_llh=GNSS_ORIGIN_LLH, seed=seed + 1), name="gnss")
    scheduler.add(BarometerModel(update_rate_hz=50.0, seed=seed + 2), name="barometer")
    scheduler.add(RangefinderModel(update_rate_hz=20.0, seed=seed + 3), name="rangefinder")
    scheduler.reset()

    print("\n=== 4) SensorScheduler usage ===")
    for frame_idx in range(frames):
        obs = scheduler.update(frame_idx * dt, make_synthetic_sensor_state(frame_idx, dt=dt, total_frames=frames))
        if frame_idx in {0, 1, 2, frames - 1}:
            print(
                f"frame={frame_idx:03d} imu_due={bool(obs['imu'])} gnss_fix={int(obs['gnss'].get('fix_quality', 0))} "
                f"range={float(obs['rangefinder'].get('range_m', 0.0)):.2f}m"
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Standalone sensor usage patterns inspired by the upstream Genesis examples"
    )
    parser.add_argument("--mode", choices=("all", "direct", "presets", "suite", "scheduler"), default="all")
    parser.add_argument("--frames", type=int, default=24)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.mode in {"all", "direct"}:
        demo_direct_usage(args.seed)
    if args.mode in {"all", "presets"}:
        demo_preset_usage(args.seed)
    if args.mode in {"all", "suite"}:
        demo_suite_usage(args.frames, args.dt, args.seed)
    if args.mode in {"all", "scheduler"}:
        demo_scheduler_usage(args.frames, args.dt, args.seed)


if __name__ == "__main__":
    main()
