# 🎛️ Genesis Sensors

[![CI](https://github.com/vlordier/genesis-sensors/actions/workflows/ci.yml/badge.svg)](https://github.com/vlordier/genesis-sensors/actions/workflows/ci.yml)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)

A **self-contained companion repository** for the external sensor stack in [Genesis](https://github.com/Genesis-Embodied-AI/Genesis).

Inspired by the standalone feel of `GenesisDroneEnv`, this package wraps the upstream `genesis.sensors` work into reusable **sensor rigs**, **scene builders**, and **demo scripts** for common robots.

---

## ✨ Features

- **Reusable sensor rigs** for:
  - drones / navigation stacks
  - Franka wrist + proprioception
  - Go2 proprioception + per-leg contact sensing
- **Self-contained demos** under `examples/`
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
│   ├── franka_demo.py
│   └── go2_demo.py
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

# 3. Install a Genesis build that includes `genesis.sensors`
pip install "genesis-world @ git+https://github.com/vlordier/Genesis.git@copilot/genesis-sensors-integration"

# 4. Install this companion repo locally
cd genesis-sensors
pip install -e .[dev]

# 5. Once the upstream sensor layer is released in `genesis-world`,
#    plain PyPI installation becomes enough
pip install genesis-sensors
```

---

## 🚀 Quick start

```python
from genesis_sensors.scenes import build_franka_demo

bundle = build_franka_demo(show_viewer=False)
bundle.rig.reset()

for step in range(200):
    bundle.controller(step)
    bundle.scene.step()
    obs = bundle.rig.step(float(bundle.scene.cur_t))
    print(obs["joint_state"]["joint_pos_rad"][:3])
```

---

## 🎬 Demos

```bash
python examples/drone_demo.py --steps 200
python examples/franka_demo.py --steps 200
python examples/go2_demo.py --steps 200

# Or via the installed CLI
genesis-sensors-demo drone --steps 200
genesis-sensors-demo franka --steps 200
genesis-sensors-demo go2 --steps 200
```

---

## 📤 PyPI release

```bash
# local packaging checks
python -m pip install -e .[dev]
python -m build
python -m twine check dist/*

# publish from GitHub Actions after configuring PyPI trusted publishing
git tag v0.1.0
git push origin v0.1.0
```

The repo includes a release workflow at `.github/workflows/publish.yml` for tagged releases.

---

## � Repository standards

- default development branch: `develop`
- protected release branch: `main`
- required branch prefixes: `feat/`, `fix/`, `docs/`, `refactor/`, `chore/`, `test/`, `ci/`, `build/`, `release/`
- release policy: Semantic Versioning (`major.minor.patch`)
- automated releases: Conventional Commit PR titles on `main` drive semantic-release and changelog generation

See `CONTRIBUTING.md`, `RELEASE.md`, and `CHANGELOG.md` for the full workflow.
The docs site is built with **MkDocs Material** and published via **GitHub Pages**.

---

## �🧭 Notes

- This package assumes the upstream `genesis.sensors` package is available.
- The API centers on `SensorRig` and `DemoScene` so it stays easy to extend.
- See `docs/overview.md` for the design intent and extension points.
- The published documentation site is served from GitHub Pages.
