# Overview

`genesis-sensors` is a lightweight companion package built around the upstream `genesis.sensors` module.

## Design goals

1. **Self-contained** — easy to clone and run like a normal research repo.
2. **Typed and simple** — small dataclass-based APIs for rigs and scenes.
3. **Composable** — each rig is just a `SensorSuite` plus a state-building callback.
4. **Upstream-friendly** — engine and sensor-model logic stays in `Genesis`; this repo focuses on packaging and usage patterns.

## Main abstractions

- `SensorRig`: a reusable bundle of a `SensorSuite` and a `state_fn`
- `DemoScene`: a scene, main entity, sensor rig, and optional controller callback
- `NamedContactSensor`: convenience wrapper for per-link contact observation in multi-sensor rigs
