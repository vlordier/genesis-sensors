# Genesis Sensors

`genesis-sensors` is a lightweight companion package for building reusable sensor rigs and demos on top of Genesis. It works with plain `genesis-world` today and automatically uses native upstream `genesis.sensors` when available.

## Highlights

- reusable rigs for drones, Franka, Go2, and a richer multimodal perception stack
- a headless synthetic rig and state builder for quickly exercising more upstream sensor families
- preset helpers via `get_preset()` and `list_presets()`
- self-contained demo scenes and CLI entry point
- PyPI-ready packaging
- protected release flow with `develop` → `main`
- GitHub Pages documentation built with MkDocs Material

## Quick install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install genesis-world
pip install genesis-sensors
```

## Quick demo

```bash
genesis-sensors-demo drone --steps 200
genesis-sensors-demo perception --steps 200
python examples/drone_with_sensors.py --steps 200
python examples/camera_as_sensor.py --steps 24
python examples/external_sensors.py --frames 120
python examples/sensor_usage_patterns.py --mode all --frames 24
python examples/contact_force_go2.py --steps 120
python examples/imu_franka.py --steps 120
```
