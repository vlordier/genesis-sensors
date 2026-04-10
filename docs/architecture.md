# Architecture

## Package Structure

```
genesis-sensors/
├── src/genesis_sensors/
│   ├── __init__.py          # Public API, lazy imports
│   ├── _compat.py           # Runtime fallback: bundled vs upstream
│   ├── synthetic.py         # Synthetic state generator for testing
│   ├── cli.py               # CLI entry points
│   ├── rigs.py              # Pre-built sensor rigs (drone, Franka, Go2)
│   └── _runtime_sensors/    # Bundled sensor backend
│       ├── base.py          # BaseSensor ABC
│       ├── scheduler.py     # Multi-rate SensorScheduler
│       ├── suite.py         # SensorSuite (43 sensor types)
│       ├── config.py        # Pydantic configuration models
│       ├── types.py         # TypedDict observation schemas
│       ├── presets.py        # 90+ real-world sensor presets
│       ├── _gauss_markov.py # Shared bias-drift utility
│       └── *.py             # Individual sensor models
├── tests/
├── docs/
├── examples/
└── scripts/
    └── gen_api_docs.py      # Auto-generate API reference pages
```

## Design Principles

### 1. Upstream-First Fallback

`genesis-sensors` bundles a complete sensor runtime under `_runtime_sensors/`.
The `_compat.py` bridge checks whether Genesis ships a native `genesis.sensors`
module. If so, it uses the upstream version; otherwise it falls back to the
bundled runtime. This makes the package work both standalone and as a Genesis
companion.

### 2. Registry-Driven Sensor Suite

The `SensorSuite` uses a data-driven `_SENSOR_SLOTS` tuple to manage all 43
sensor types. Adding a new sensor requires:

1. Implement the model class (subclass `BaseSensor`)
2. Add a config class to `config.py`
3. Add one tuple entry to `_SENSOR_SLOTS` in `suite.py`
4. Import in `__init__.py`

No if-statements, no manual wiring.

### 3. Multi-Rate Scheduling

Each sensor runs at its own update rate. The `SensorScheduler` tracks
`_last_update_time` per sensor and only calls `step()` when `is_due()`
returns `True`. Between updates, cached observations are returned via
`get_observation()`.

### 4. Gauss-Markov Bias Drift

Six sensor types (IMU×2, GNSS, barometer, airspeed, thermometer, hygrometer)
use first-order Gauss-Markov processes for realistic bias drift. The shared
`GaussMarkovProcess` class eliminates code duplication:

$$x_{k+1} = \alpha \cdot x_k + \sigma_{\text{drive}} \cdot w_k, \quad w \sim \mathcal{N}(0, 1)$$

where $\alpha = e^{-\Delta t / \tau}$ and $\sigma_{\text{drive}} = \sigma_{\text{ss}} \sqrt{1 - \alpha^2}$.

### 5. Deterministic Seeding

`SensorSuite.default(seed=N)` derives per-sensor seeds via `SeedSequence` so
that sensor RNG streams are statistically uncorrelated. The `set_seed()`
method allows re-seeding all sensors for replay.

## Sensor Catalogue

| Category | Sensors | Count |
|----------|---------|-------|
| **Navigation** | IMU, GNSS, Barometer, Magnetometer, Airspeed, Optical Flow, Wheel Odometry, Inclinometer | 8 |
| **Vision** | RGB Camera, Stereo Camera, Depth Camera, Thermal Camera, Event Camera, LiDAR | 6 |
| **Range Sensing** | Rangefinder, Ultrasonic Array, Imaging Sonar, Side-Scan Sonar, DVL, Current Profiler, Proximity ToF | 7 |
| **Environmental** | Thermometer, Hygrometer, Light Sensor, Gas Sensor, Anemometer, Battery | 6 |
| **Communication** | Radio Link, UWB Ranging, Radar, Underwater Modem | 4 |
| **Manipulation** | Force/Torque, Joint State, Contact, Tactile Array, Current Sensor, RPM Sensor, Load Cell, Wire Encoder, Motor Temperature | 9 |
| **Marine & Subsea** | Water Pressure, Hydrophone, Leak Detector | 3 |
| **Total** | | **43** |

## Data Flow

```
Genesis Simulation
        │
        ▼
   State Dict ─────────────────────────────────┐
   {pos, vel, lin_acc, ang_vel, rgb, depth, …}  │
        │                                       │
        ▼                                       │
  SensorScheduler.update(t, state)              │
        │                                       │
        ├── is_due("imu") → True ──→ IMU.step() │
        ├── is_due("rgb") → False → cached obs  │
        ├── is_due("gnss") → True → GNSS.step() │
        └── …                                   │
        │                                       │
        ▼                                       │
   Observation Dict                             │
   {imu: {lin_acc, ang_vel},                    │
    rgb: {rgb},                                 │
    gnss: {pos_llh, vel_ned, …}, …}             │
```

## Configuration System

Each sensor has a matching Pydantic `*Config` model:

```python
from genesis_sensors import IMUConfig, IMUModel

# Default config
cfg = IMUConfig()

# From preset
from genesis_sensors import get_preset
cfg = get_preset("PIXHAWK_ICM20689")

# Build sensor
sensor = IMUModel.from_config(cfg)

# Roundtrip
cfg2 = sensor.get_config()
assert cfg.model_dump() == cfg2.model_dump()
```

## Testing Strategy

| Test File | Purpose | Count |
|-----------|---------|-------|
| `test_architecture.py` | Config roundtrips, preset validation, GaussMarkov, suite factories | ~130 |
| `test_edge_cases.py` | NaN/Inf inputs, error wrapping, extreme rates, set_seed, config validation, saturation | ~100 |
| `test_rigs.py` | Pre-built rig integration | ~5 |
| `test_robustness.py` | Fault injection wrapper | ~3 |
| `test_*.py` (domain) | Per-domain sensor tests | ~30 |
