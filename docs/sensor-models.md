# Sensor Models

How every sensor category is modelled, what noise/error effects are applied,
and how configuration parameters map to physical behaviour.

> Visual sensors use true snapshots from the real observation arrays, while the
> rest use diagnostic plots, all generated with
> `PYTHONPATH=src python examples/generate_sensor_doc_assets.py --output-dir docs/assets/sensors`.
> When those assets are present, the per-sensor API docs embed them automatically.

---

## General Pipeline

Every sensor model follows the same processing pipeline:

```
True State → Error Injection → Quantisation/Clipping → Observation Dict
```

1. **True state** arrives via the `state` dict (e.g. `state["lin_acc"]`).
2. **Error injection** applies noise, bias, distortion, and environmental
   effects in a physically motivated order.
3. **Quantisation/clipping** enforces ADC resolution, range limits, and
   dropout logic.
4. The result is returned as a typed observation dict.

### Noise Types

| Type | Description | Applied By |
|------|-------------|------------|
| **White noise** | Zero-mean Gaussian, i.i.d. per step | All sensors |
| **Gauss-Markov bias** | First-order autocorrelated drift | IMU, GNSS, barometer, airspeed, thermometer, hygrometer |
| **Scale-factor error** | Multiplicative gain error | IMU |
| **Cross-axis sensitivity** | Off-diagonal coupling | IMU |
| **Multipath** | Slowly varying random offset | GNSS |
| **Dropout** | Random missed readings | LiDAR, rangefinder, DVL |
| **Speckle** | Multiplicative noise on sonar imagery | Imaging sonar, side-scan sonar |

### Gauss-Markov Process

Six sensor categories use the shared `GaussMarkovProcess` for bias drift.
The discrete-time update is:

$$x_{k+1} = \alpha \, x_k + \sigma_d \, w_k, \quad w_k \sim \mathcal{N}(0,1)$$

where $\alpha = e^{-\Delta t / \tau}$, $\sigma_d = \sigma_\text{ss}\sqrt{1 - \alpha^2}$,
$\tau$ is the correlation time (seconds), and $\sigma_\text{ss}$ is the
steady-state standard deviation.

**Configuration parameters:** `bias_tau_s` (correlation time), `bias_sigma_*`
(steady-state σ).

---

## Navigation Sensors

### IMU (Inertial Measurement Unit)

**Model class:** `IMUModel` · **Config:** `IMUConfig`

Accelerometer and gyroscope on a common time base.  Error model:

$$a_\text{meas} = (1 + s_a) \, R_\text{cross} \, a_\text{true} + b_a + n_a + g$$

$$\omega_\text{meas} = (1 + s_\omega) \, R_\text{cross} \, \omega_\text{true} + b_\omega + n_\omega$$

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| White noise density | `noise_density_acc`, `noise_density_gyr` | Allan deviation at 1 s (m/s²/√Hz, rad/s/√Hz) |
| Bias drift | `bias_sigma_acc`, `bias_tau_acc_s` | Gauss-Markov steady-state σ and correlation time |
| Scale factor | `scale_factor_acc`, `scale_factor_gyr` | Relative gain error (dimensionless) |
| Cross-axis | `cross_axis_sensitivity_acc`, `cross_axis_sensitivity_gyr` | Off-diagonal coupling factor |
| Gravity injection | `add_gravity` | Whether to add [0, 0, 9.81] to accelerometer |

**Application order:** scale factor → cross-axis → bias → white noise → gravity.

### GNSS (Global Navigation Satellite System)

**Model class:** `GNSSModel` · **Config:** `GNSSConfig`

Converts ENU position/velocity to geodetic coordinates with realistic errors.

| Effect | Config Field | Description |
|--------|-------------|-------------|
| Position noise | `noise_m` | Per-axis Gaussian σ (metres) |
| Velocity noise | `vel_noise_ms` | Per-axis Gaussian σ (m/s) |
| Bias drift | `bias_sigma_m`, `bias_tau_s` | Gauss-Markov slow wander |
| Multipath | `multipath_sigma_m` | Additional slowly varying offset |
| Jammer zones | `jammer_zones` | Spatial regions that degrade/deny fix |
| Min fix altitude | `min_fix_altitude_m` | Below this altitude → no fix |

**Coordinate conversion:** flat-Earth ENU → WGS-84 LLH using a configurable
`origin_llh` reference point.

### Barometer

**Model class:** `BarometerModel` · **Config:** `BarometerConfig`

Converts geometric altitude to pressure-altitude via the ISA atmosphere model.

$$P = P_0 \left(\frac{T_0 - Lh}{T_0}\right)^{g M / (R L)}$$

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| White noise | `noise_sigma_m` | Gaussian σ on altitude (metres) |
| Bias drift | `bias_sigma_m`, `bias_tau_s` | Gauss-Markov slow offset |
| Ground altitude | `ground_alt_m` | Reference ground level for AGL |
| ADC resolution | `resolution_m` | Quantisation step (metres) |

### Magnetometer

**Model class:** `MagnetometerModel` · **Config:** `MagnetometerConfig`

Simulates a 3-axis magnetometer measuring Earth's magnetic field.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| White noise | `noise_sigma_ut` | Per-axis Gaussian σ (µT) |
| Field amplitude | `field_amplitude_ut` | Local field magnitude (µT) |
| Declination/inclination | `declination_deg`, `inclination_deg` | Local magnetic dip angles |
| Hard-iron offset | `hard_iron_ut` | 3-element additive bias vector (µT) |
| Soft-iron distortion | `soft_iron_scale` | 3-element diagonal scale factors |

### Airspeed (Pitot Tube)

**Model class:** `AirspeedModel` · **Config:** `AirspeedConfig`

Simulates a Prandtl pitot tube.  Converts TAS → dynamic pressure → IAS.

$$q = \tfrac{1}{2} \rho(h) \, V_\text{TAS}^2, \quad V_\text{IAS} = \sqrt{\frac{2q}{\rho_0}}$$

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| White noise | `noise_sigma_ms` | Gaussian σ on airspeed (m/s) |
| Bias drift | `bias_sigma_ms`, `bias_tau_s` | Gauss-Markov zero-offset drift |
| Dead band | `min_detectable_ms` | Speeds below this → 0 |
| Saturation | `max_speed_ms` | Upper clamp |
| Tube blockage | `tube_blockage_prob` | Per-step probability of permanent blockage (persists until `reset()`) |

### Optical Flow

**Model class:** `OpticalFlowModel` · **Config:** `OpticalFlowConfig`

Simulates a downward-looking optical flow sensor (e.g. PX4FLOW).

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Noise floor | `noise_floor_rad_s` | Minimum noise at any height |
| Noise slope | `noise_slope` | Noise grows linearly with angular rate |
| Max detection | `max_detection_rad_s` | Saturates above this rate |
| Quality model | `nominal_quality_height_m`, `base_quality` | Quality degrades with height |

### Wheel Odometry

**Model class:** `WheelOdometryModel` · **Config:** `WheelOdometryConfig`

Incremental dead-reckoning from wheel encoders.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Position noise | `pos_noise_sigma_m` | Per-step position noise (m) |
| Heading noise | `heading_noise_sigma_rad` | Per-step heading noise (rad) |
| Slip | `slip_sigma` | Multiplicative wheel-slip noise |

---

## Vision Sensors

### RGB Camera

**Model class:** `CameraModel` · **Config:** `CameraConfig`

Full image-corruption pipeline applied to ideal RGB frames.

**Pipeline order:**
1. Exposure / ISO gain
2. Photon shot noise (Poisson → Gaussian approximation)
3. Read noise (Gaussian)
4. Dead/hot pixels
5. Vignetting (radial cos⁴ falloff)
6. Chromatic aberration (per-channel radial shift)
7. Rolling shutter (per-row temporal offset)
8. Motion blur (box kernel)
9. JPEG compression artefacts

### Thermal Camera

**Model class:** `ThermalCameraModel` · **Config:** `ThermalCameraConfig`

Uncooled microbolometer thermal imager.

| Effect | Config Field | Description |
|--------|-------------|-------------|
| Spatial blur | `psf_sigma` | Point-spread function σ in pixels |
| NUC residual | `nuc_sigma` | Non-uniformity correction residual (°C) |
| Temporal noise | `noise_sigma` | NEDT equivalent (°C) |
| Temperature range | `temp_range_c` | (min, max) scene temperature |
| Fog | `fog_density` | Atmospheric fog attenuation coefficient |

### Event Camera

**Model class:** `EventCameraModel` · **Config:** `EventCameraConfig`

Dynamic-vision sensor producing asynchronous brightness-change events.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| ON/OFF thresholds | `threshold_pos`, `threshold_neg` | Log-intensity change thresholds |
| Refractory period | `refractory_period_s` | Minimum inter-event interval per pixel |
| Threshold variation | `threshold_variation` | Pixel-to-pixel mismatch (fraction) |
| Background activity | `background_activity_rate_hz` | Spurious event rate |

### Depth Camera

**Model class:** `DepthCameraModel` · **Config:** `DepthCameraConfig`

Active IR / structured-light depth sensor.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Depth noise | `depth_noise_sigma_m` | Base Gaussian σ (m) |
| Z-scaling noise | `depth_noise_scale_z` | Noise grows quadratically with depth |
| Edge dropout | `missing_edge_px` | Pixels near depth discontinuities are invalid |
| Range limits | `min_depth_m`, `max_depth_m` | Clip and invalidate outside range |

### Stereo Camera

**Model class:** `StereoCameraModel` · **Config:** `StereoCameraConfig`

Passive stereo vision with disparity-based depth estimation.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Baseline | `baseline_m` | Inter-camera distance (m) |
| Disparity noise | `disparity_noise_sigma_px` | Gaussian noise on pixel disparity |
| Depth range | `min_depth_m`, `max_depth_m` | Valid depth range |

### LiDAR

**Model class:** `LidarModel` · **Config:** `LidarConfig`

Spinning or solid-state time-of-flight point cloud sensor.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Range noise | `range_noise_sigma_m` | Per-point Gaussian (m) |
| Intensity noise | `intensity_noise_sigma` | Reflectance channel noise |
| Dropout | `dropout_prob` | Per-point miss probability |
| Weather | `rain_rate_mm_h`, `fog_density` | Attenuate/drop points |
| Beam divergence | `beam_divergence_mrad` | Angular beam width |
| Channel offsets | `channel_offsets_m` | Per-channel range bias |

---

## Range Sensing

### Rangefinder

**Model class:** `RangefinderModel` · **Config:** `RangefinderConfig`

Single-beam ToF or triangulation rangefinder.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Noise | `noise_floor_m`, `noise_slope` | $\sigma = \text{floor} + \text{slope} \times r$ |
| Dropout | `dropout_prob` | Per-step miss probability |
| Resolution | `resolution_m` | ADC quantisation step (m) |
| Range limits | `min_range_m`, `max_range_m` | Outside → no-hit value |
| Accuracy mode | `accuracy_mode` | `"additive"` or `"max"` noise model |

### Ultrasonic Array

**Model class:** `UltrasonicArrayModel` · **Config:** `UltrasonicArrayConfig`

Multi-beam ultrasonic range sensor array.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Beam geometry | `n_beams`, `beam_angles_deg`, `beam_width_deg` | Array layout |
| Cross-talk | `cross_talk_prob` | Probability of inter-beam echo contamination |
| Temperature compensation | `temperature_compensation` | Adjust speed of sound for temperature |

### Imaging Sonar

**Model class:** `ImagingSonarModel` · **Config:** `ImagingSonarConfig`

Forward-looking imaging sonar producing azimuth-range images.

| Effect | Config Field | Description |
|--------|-------------|-------------|
| Range noise | `range_noise_sigma_m` | Gaussian σ on range bins |
| Azimuth noise | `azimuth_noise_deg` | Gaussian σ on bearing |
| Speckle | `speckle_sigma` | Multiplicative Rayleigh speckle |
| Attenuation | `attenuation_db_per_m` | Acoustic absorption loss |
| False alarms | `false_alarm_rate` | Per-bin spurious detection rate |

### DVL (Doppler Velocity Log)

**Model class:** `DVLModel` · **Config:** `DVLConfig`

Bottom-tracking velocity sensor for underwater vehicles.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Velocity noise | `velocity_noise_sigma_ms` | Per-beam velocity Gaussian σ |
| Range noise | `range_noise_sigma_m` | Altitude measurement noise |
| Dropout | `dropout_prob` | Per-beam lock-loss probability |
| Altitude limits | `min_altitude_m`, `max_altitude_m` | Valid bottom-track range |

---

## Environmental Sensors

### Thermometer, Hygrometer, Light Sensor, Gas Sensor, Anemometer

All environment sensors follow the same pattern:

1. **White noise** — Gaussian noise on the measurement
2. **Bias drift** — Gauss-Markov (thermometer, hygrometer) or none
3. **Response lag** — First-order exponential filter with time constant `response_tau_s`
4. **Clipping** — Range limits (light sensor, gas sensor, anemometer)

### Battery

**Model class:** `BatteryModel` · **Config:** `BatteryConfig`

Simulates a LiPo/LiHV battery pack with discharge curve and internal resistance.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Cell count | `n_cells` | Series cell count |
| Capacity | `capacity_mah` | Rated capacity |
| Chemistry | `cell_chemistry` | `"lipo"` or `"lihv"` (affects voltage curve) |
| Internal resistance | `internal_resistance_ohm` | Voltage sag under load |
| SoC | `initial_soc` | Starting state of charge [0, 1] |

---

## Communication Sensors

### Radio Link

**Model class:** `RadioLinkModel` · **Config:** `RadioConfig`

Packet-level radio link with path-loss, shadowing, and packet error rate.

$$\text{PL}(d) = \text{PL}_0 + 10 n \log_{10}(d) + X_\sigma$$

where $n$ is the path-loss exponent and $X_\sigma \sim \mathcal{N}(0, \sigma_\text{shadow}^2)$.

SNR determines packet error rate via a sigmoid transition:

$$\text{PER} = \frac{1}{1 + e^{(SNR - SNR_\text{min}) / \beta}}$$

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| TX power | `tx_power_dbm` | Transmit power (dBm) |
| Path loss | `path_loss_exponent` | Environment-dependent (2=free space, 3.5=urban) |
| Shadowing | `shadowing_sigma_db` | Log-normal shadow fading σ |
| Latency/jitter | `base_latency_s`, `jitter_sigma_s` | Bidirectional jitter on packet delivery |
| NLOS | `nlos_excess_loss_db`, `los_required` | Extra loss when line-of-sight is blocked |

### UWB Ranging

**Model class:** `UWBRangingModel` · **Config:** `UWBRangeConfig`

Ultra-wideband time-of-flight ranging with NLOS bias modelling.

### Radar

**Model class:** `RadarModel` · **Config:** `RadarConfig`

Automotive/industrial radar with range/velocity/angle measurements.

---

## Manipulation Sensors

### Force/Torque

**Model class:** `ForceTorqueSensorModel` · **Config:** `ForceTorqueConfig`

6-axis force/torque sensor with noise, bias, and saturation.

| Parameter | Config Field | Description |
|-----------|-------------|-------------|
| Force noise | `force_noise_sigma_n` | Per-axis Gaussian σ (N) |
| Torque noise | `torque_noise_sigma_nm` | Per-axis Gaussian σ (N·m) |
| Bias | `force_bias_n`, `torque_bias_nm` | Static offset |
| Saturation | `force_range_n`, `torque_range_nm` | Absolute clamp |

### Joint State, Contact, Tactile Array, Current, RPM

All follow the standard pattern: white noise + optional quantisation + range clipping.

---

## Presets

Presets are pre-validated `*Config` instances with parameters from real datasheets.
Use them as-is or override specific fields:

```python
from genesis_sensors import get_preset, IMUModel

# Use directly
imu = IMUModel.from_preset("PIXHAWK_ICM20689")

# Override fields
imu = IMUModel.from_preset("PIXHAWK_ICM20689", update_rate_hz=500.0)

# Or via config
cfg = get_preset("PIXHAWK_ICM20689", noise_density_acc=0.05)
```

See `list_presets()` or `list_presets(kind="imu")` for available names.

The complete list of 90+ presets covers: cameras (5), stereo (3), LiDAR (7),
IMU (8), GNSS (8), thermal (3), event (2), barometer (7), environmental (5),
radio (3), UWB (1), radar (2), ultrasonic (2), sonar (3), DVL/ADCP (2),
magnetometer (3), airspeed (2), rangefinder (3), optical flow (2),
battery (3), wheel odometry (2), force/torque (3), joint state (3),
contact (2), depth camera (5), tactile (3), current (3), and RPM (3).
