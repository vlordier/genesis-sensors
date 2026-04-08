# 🎛️ Genesis Sensors

[![CI](https://github.com/vlordier/genesis-sensors/actions/workflows/ci.yml/badge.svg)](https://github.com/vlordier/genesis-sensors/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A **self-contained companion repository** for the external sensor stack in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).

Inspired by the standalone feel of `GenesisDroneEnv`, this package wraps the Genesis external-sensor stack into reusable **sensor rigs**, **scene builders**, and **demo scripts** for common robots.

> `genesis-sensors` now works directly with plain `genesis-world`: it uses the upstream `genesis.sensors` module when available and otherwise falls back to a bundled compatible runtime included in this package.

---

## ✨ Features

- **Reusable sensor rigs** for:
  - drones / navigation stacks
  - multimodal drone perception (`RGB`, stereo, event, thermal, LiDAR, radio)
  - Franka wrist + proprioception
  - Go2 proprioception + per-leg contact sensing
- **Bundled sensor runtime** so the package works with released `genesis-world` as-is
- **Headless synthetic rigs and state builders** for quickly exercising the full sensor surface
- **Robustness wrappers** for latency injection, packet dropouts, and per-observation health metadata
- **Preset-friendly helpers** via `get_preset()` and `list_presets()` re-exported from the companion package
- **Self-contained demos** under `examples/`, including upstream-style ports for camera, IMU, contact-force, LiDAR teleop, tactile walkthroughs, and optional **Rerun sensor traces**
- **Small packaging footprint**: depends on `genesis-world`, not on the full source tree layout
- **Typed bridge helpers** and simple dataclass-based APIs
- **GitHub-ready packaging** with CI, tests, and a small CLI entry point

---

## 📦 Repository layout

```text
genesis-sensors/
├── README.md
├── pyproject.toml
├── examples/
│   ├── drone_demo.py
│   ├── drone_with_sensors.py
│   ├── camera_as_sensor.py
│   ├── external_sensors.py
│   ├── external_sensors_rerun.py
│   ├── perception_demo.py
│   ├── sensor_usage_patterns.py
│   ├── fault_tolerant_sensors.py
│   ├── franka_demo.py
│   ├── franka_arm_sensors.py
│   ├── imu_franka.py
│   ├── go2_demo.py
│   ├── go2_sensors.py
│   ├── contact_force_go2.py
│   ├── lidar_teleop.py
│   ├── kinematic_contact_probe.py
│   ├── tactile_elastomer_franka.py
│   └── tactile_elastomer_sandbox.py
├── src/genesis_sensors/
│   ├── __init__.py
│   ├── rigs.py
│   └── scenes.py
└── tests/
    └── test_rigs.py
```

---

## 💻 Installation

```bash
# 1. Create an environment
python -m venv .venv
source .venv/bin/activate

# 2. Install PyTorch for your platform (CPU example)
pip install torch --index-url https://download.pytorch.org/whl/cpu

# 3. Install Genesis from PyPI
pip install genesis-world

# 4. Install this companion repo locally
cd genesis-sensors
pip install -e .[dev]

# 4b. Optional: install Rerun support for trace visualization
pip install -e .[rerun]
# or: pip install rerun-sdk

# 5. Or install the published package directly
pip install genesis-sensors

# If a future Genesis build ships native `genesis.sensors`,
# `genesis-sensors` will automatically use it instead of the bundled fallback.
```

---

## 🚀 Quick start

```python
from genesis_sensors import build_franka_demo

bundle = build_franka_demo(show_viewer=False)
bundle.rig.reset()

for step in range(200):
    bundle.controller(step)
    bundle.scene.step()
    obs = bundle.rig.step(float(bundle.scene.cur_t))
    print(obs["joint_state"]["joint_pos_rad"][:3])

# Or exercise a headless multimodal stack directly
from genesis_sensors import make_synthetic_multimodal_rig, list_presets

rig = make_synthetic_multimodal_rig(seed=0)
rig.reset()
print(rig.step(0.0).keys())
print(list_presets(kind="stereo"))
```

---

## 🎬 Demos

```bash
python examples/drone_demo.py --steps 200
python examples/drone_with_sensors.py --steps 200
python examples/perception_demo.py --steps 200
python examples/camera_as_sensor.py --steps 24
python examples/external_sensors.py --frames 120
python examples/external_sensors_rerun.py --frames 60 --spawn
python examples/external_sensors_rerun.py --frames 60 --save /tmp/genesis-sensors.rrd
python examples/fault_tolerant_sensors.py --frames 24 --dropout 0.2
python examples/sensor_usage_patterns.py --mode all --frames 24
python examples/franka_demo.py --steps 200
python examples/franka_arm_sensors.py --steps 200
python examples/imu_franka.py --steps 120
python examples/go2_demo.py --steps 200
python examples/go2_sensors.py --steps 200
python examples/contact_force_go2.py --steps 120
python examples/lidar_teleop.py --steps 60 --cpu --pattern grid
python examples/kinematic_contact_probe.py --steps 60 --cpu
python examples/tactile_elastomer_franka.py --steps 120 --cpu
python examples/tactile_elastomer_sandbox.py --steps 80 --cpu

# Or via the installed CLI
# preferred names
genesis-sensors drone --steps 200
genesis-sensors perception --steps 200
genesis-sensors franka --steps 200
genesis-sensors go2 --steps 200

# short alias
gs-sensors drone --steps 200

# legacy compatibility alias still works
genesis-sensors-demo drone --steps 200
```

---

## 📤 PyPI release

```bash
# local packaging checks
python -m pip install -e .[dev]
python -m build
python -m twine check dist/*

# or use the pyproject-managed task shortcuts
uv run --extra dev poe test
uv run --extra dev poe fault_demo
uv run --extra dev poe rerun_save
uv run --extra dev poe check

# publish from GitHub Actions after configuring PyPI trusted publishing
git tag v0.1.0
git push origin v0.1.0
```

The repo includes a release workflow at `.github/workflows/publish.yml` for tagged releases.

---

## 🧱 Repository standards

- default development branch: `develop`
- protected release branch: `main`
- required branch prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `chore/`, `test/`, `ci/`, `build/`, `release/`
- release policy: Semantic Versioning (`major.minor.patch`)
- automated releases: Conventional Commit PR titles on `main` drive semantic-release and changelog generation

See `CONTRIBUTING.md`, `RELEASE.md`, and `CHANGELOG.md` for the full workflow.
The docs site is built with **MkDocs Material** and published via **GitHub Pages**.

---

## 🧭 Notes

- The next implementation focus after core sensor coverage is **robustness tooling**: latency injection, packet dropouts, and health metadata for downstream pipelines. Those helpers are now available through `RobustSensorWrapper`, `wrap_suite_with_faults()`, and `wrap_rig_with_faults()`.
- This package automatically uses native upstream `genesis.sensors` when available and falls back to the bundled backend otherwise.
- The API centers on `SensorRig` and `DemoScene` so it stays easy to extend.
- See `docs/overview.md` for the design intent and extension points.
- The published documentation site is served from GitHub Pages.
