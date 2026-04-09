# Overview

`genesis-sensors` is a lightweight companion package built around the upstream `genesis.sensors` module.

## Design goals

1. **Self-contained** — easy to clone and run like a normal research repo.
2. **Typed and simple** — small dataclass-based APIs for rigs and scenes.
3. **Composable** — each rig is just a `SensorSuite` plus a state-building callback.
4. **Upstream-friendly** — if native `genesis.sensors` ships in `Genesis`, this repo uses it automatically; otherwise it relies on its bundled compatible runtime.

## Main abstractions

- `SensorRig`: a reusable bundle of a `SensorSuite` and a `state_fn`
- `DemoScene`: a scene, main entity, sensor rig, and optional controller callback
- `NamedContactSensor`: convenience wrapper for per-link contact observation in multi-sensor rigs
- `make_synthetic_multimodal_rig()`: a headless way to exercise the broader upstream perception stack
- `make_synthetic_sensor_state()`: reusable synthetic state generation for preset and scheduler examples
- `docs/examples.md`: runnable `genesis-world` + `genesis_sensors` walkthroughs with the corresponding generated outputs embedded in the docs

## Current coverage

The companion repo now mirrors four practical slices of the upstream `genesis.sensors` work:

- **navigation** — IMU, GNSS, barometer, magnetometer, airspeed, rangefinder, optical flow, battery
- **multimodal perception** — RGB camera, stereo, event camera, thermal, LiDAR, radio, wheel-odometry-style dead reckoning
- **manipulation** — joint state, force/torque, contact, depth, tactile array, current, RPM
- **legged proprioception** — IMU, joint state, current, RPM, per-foot contact
- **scene-sensor walkthroughs** — camera-as-sensor, LiDAR/depth teleop, kinematic contact probe, Franka and sandbox elastomer tactile examples
