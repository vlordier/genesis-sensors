# Overview

`genesis-sensors` is a lightweight companion package built around the Genesis external-sensor API surface, preferring upstream `genesis.sensors` when available and otherwise using its bundled compatible runtime.

## Mental model

```text
Genesis `gs.Scene`
    ↓
state extraction (`state_fn`)
    ↓
`SensorSuite` / `SensorRig`
    ↓
device-like observations for controllers, policies, and loggers
```

1. **Genesis owns the world state** — geometry, poses, velocities, contacts, and rendered arrays.
2. **A rig packages the sensors** — each `SensorRig` combines a `SensorSuite` with a callback that
   extracts the needed state from the scene.
3. **The runtime models add realism** — noise, bias, clipping, dropout, and modality-specific effects
   are applied before the observation is returned.

## Choose the right entry point

| Goal | Start with |
| --- | --- |
| ready-made drone perception scene | `build_perception_demo()` |
| ready-made Franka manipulation scene | `build_franka_demo()` |
| ready-made drone navigation scene | `build_drone_demo()` |
| attach sensors to an existing Genesis scene | `make_drone_perception_rig()`, `make_franka_wrist_rig()`, `make_go2_rig()` |
| headless smoke tests / scheduler debugging | `make_synthetic_multimodal_rig()` and `make_synthetic_sensor_state()` |

## Design goals

1. **Self-contained** — easy to clone and run like a normal research repo.
2. **Typed and simple** — small dataclass-based APIs for rigs and scenes.
3. **Composable** — each rig is just a `SensorSuite` plus a state-building callback.
4. **Upstream-friendly** — if native `genesis.sensors` ships in `Genesis`, this repo uses it automatically;
   otherwise it relies on its bundled compatible runtime.

## Main abstractions

- `SensorRig`: reusable bundle of a `SensorSuite` and a `state_fn`
- `DemoScene`: a scene, main entity, sensor rig, and optional controller callback
- `NamedContactSensor`: convenience wrapper for per-link contact observation in multi-sensor rigs
- `make_synthetic_multimodal_rig()`: headless way to exercise the broader upstream perception surface
- `make_synthetic_sensor_state()`: reusable synthetic state generation for preset and scheduler examples
- `docs/examples.md`: runnable `genesis-world` + `genesis_sensors` walkthroughs with generated outputs
  embedded directly in the docs

## Current coverage

The companion repo now mirrors four practical slices of the upstream `genesis.sensors` work:

- **navigation** — IMU, GNSS, barometer, magnetometer, airspeed, rangefinder, optical flow, battery
- **multimodal perception** — RGB camera, stereo, event camera, thermal, LiDAR, radio,
  wheel-odometry-style dead reckoning
- **manipulation** — joint state, force/torque, contact, depth, tactile array, current, RPM
- **legged proprioception** — IMU, joint state, current, RPM, per-foot contact
- **scene-sensor walkthroughs** — camera-as-sensor, LiDAR/depth teleop, kinematic contact probe,
  Franka and sandbox elastomer tactile examples
