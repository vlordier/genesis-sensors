# Genesis Sensors

`genesis-sensors` is a lightweight companion package for building reusable sensor rigs and demos on top of the upstream `genesis.sensors` module.

## Highlights

- reusable rigs for drones, Franka, and Go2
- self-contained demo scenes and CLI entry point
- PyPI-ready packaging
- protected release flow with `develop` → `main`
- GitHub Pages documentation built with MkDocs Material

## Quick install

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "genesis-world @ git+https://github.com/vlordier/Genesis.git@copilot/genesis-sensors-integration"
pip install genesis-sensors
```

## Quick demo

```bash
genesis-sensors-demo drone --steps 200
```
