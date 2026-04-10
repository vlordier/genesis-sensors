# Genesis Sensors

`genesis-sensors` is a lightweight companion package for building reusable sensor rigs and demos on top of
Genesis. It works with plain `genesis-world` today and automatically uses native upstream `genesis.sensors`
when available.

> **Documentation provenance:** the generated plots and snapshots in this site are produced from live
> Genesis demo scenes. Genesis provides the scene state and ideal arrays; `genesis_sensors` then applies
> the sensor model to produce the final observation dict shown here.

## Start here

| If you want to... | Open this page |
| --- | --- |
| run a ready-made demo scene | `Examples` |
| attach sensors to your own `gs.Scene` | `Examples` |
| understand the modeling and noise assumptions | `Sensor Models` |
| understand the repo structure and abstractions | `Overview` |
| browse every class and config field | `API Reference` |

## Highlights

- reusable rigs for drones, Franka, Go2, and a richer multimodal perception stack
- Genesis-backed documentation outputs, with the emitted observation arrays shown inline
- preset helpers via `get_preset()` and `list_presets()`
- self-contained demo scenes and a small CLI entry point
- GitHub Pages documentation built with MkDocs Material

## Quick install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install genesis-world
pip install genesis-sensors
```

If you only need the headless helpers first, verify the selected backend with:

```bash
python -c "import genesis_sensors as gs; print(gs.SENSOR_BACKEND)"
```

## 60-second quick start

```python
import genesis as gs
from genesis_sensors import make_drone_perception_rig

gs.init(backend=gs.cpu, logging_level="warning")
scene = gs.Scene(show_viewer=False)
scene.add_entity(gs.morphs.Plane())
drone = scene.add_entity(gs.morphs.Drone(file="urdf/drones/cf2x.urdf", pos=(0.0, 0.0, 0.7)))
scene.build()

rig = make_drone_perception_rig(drone, dt=0.02, seed=0)
rig.reset()

for step in range(12):
    scene.step()
    obs = rig.step(step * 0.02)
    print(obs["rgb"]["rgb"].shape, len(obs["lidar"]["points"]))
```

## Quick demos

```bash
genesis-sensors perception --steps 200
genesis-sensors franka --steps 200
PYTHONPATH=src python examples/perception_demo.py --steps 12 --dt 0.02
PYTHONPATH=src python examples/franka_demo.py --steps 12 --dt 0.02
```

For a fuller walkthrough with embedded outputs, go to `Examples`.
