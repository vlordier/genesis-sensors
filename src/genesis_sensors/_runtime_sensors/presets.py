"""
Real-world sensor presets for the Genesis sensor layer.

Each constant is a fully-validated ``*Config`` instance whose parameters are
taken from publicly available datasheets.  They are module-level singletons
and should be treated as read-only; pass them directly to the corresponding
sensor's ``from_config()`` factory or copy-with-changes via
``model.model_copy(update={...})`` before mutating.

Usage
-----
::

    from genesis.sensors.presets import RASPBERRY_PI_V2, get_preset, list_presets

    # Use directly
    cam = CameraModel.from_config(RASPBERRY_PI_V2)

    # Discover available presets
    print(list_presets())           # ["GOPRO_HERO11_4K30", "INTEL_D435_RGB", ...]
    print(list_presets(kind="lidar"))  # ["LIVOX_AVIA", "OUSTER_OS1_64", ...]

    # Retrieve by name (case-insensitive)
    cfg = get_preset("velodyne_vlp16")

Sources are cited inline as ``# Source: <URL or spec>``.
"""

from __future__ import annotations

from .config import (
    AirspeedConfig,
    AnemometerConfig,
    BarometerConfig,
    BatteryConfig,
    CameraConfig,
    ContactSensorConfig,
    CurrentSensorConfig,
    DepthCameraConfig,
    EventCameraConfig,
    ForceTorqueConfig,
    GasSensorConfig,
    GNSSConfig,
    HygrometerConfig,
    IMUConfig,
    JointStateConfig,
    LidarConfig,
    LightSensorConfig,
    MagnetometerConfig,
    OpticalFlowConfig,
    RadarConfig,
    RadioConfig,
    RangefinderConfig,
    RPMSensorConfig,
    StereoCameraConfig,
    TactileArrayConfig,
    ThermalCameraConfig,
    ThermometerConfig,
    UWBRangeConfig,
    WheelOdometryConfig,
)

# ---------------------------------------------------------------------------
# Type alias for all preset config types
# ---------------------------------------------------------------------------

PresetConfig = (
    CameraConfig
    | StereoCameraConfig
    | LidarConfig
    | IMUConfig
    | GNSSConfig
    | ThermalCameraConfig
    | EventCameraConfig
    | BarometerConfig
    | MagnetometerConfig
    | ThermometerConfig
    | HygrometerConfig
    | LightSensorConfig
    | GasSensorConfig
    | AnemometerConfig
    | AirspeedConfig
    | RangefinderConfig
    | RadioConfig
    | UWBRangeConfig
    | RadarConfig
    | OpticalFlowConfig
    | BatteryConfig
    | WheelOdometryConfig
    | ForceTorqueConfig
    | JointStateConfig
    | ContactSensorConfig
    | DepthCameraConfig
    | TactileArrayConfig
    | CurrentSensorConfig
    | RPMSensorConfig
)

# ---------------------------------------------------------------------------
# Camera presets
# ---------------------------------------------------------------------------

RASPBERRY_PI_V2 = CameraConfig(
    # Source: https://www.raspberrypi.com/products/camera-module-v2/
    # Sony IMX219, 8 MP, 30 fps @ 1080p, rolling-shutter CMOS.
    # Readout time ≈ 19 ms at 30 fps (33.3 ms frame period) → RS fraction ≈ 0.95.
    name="RASPBERRY_PI_V2",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.95,
    # IMX219 sensitivity: ~1.1 e⁻/ADU at ISO 100; read noise ≈ 2 e⁻.
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=2.0,
    full_well_electrons=4600.0,
    # Fixed-pattern noise: ~0.01 % dead pixels reported in production batches.
    dead_pixel_fraction=0.0001,
    hot_pixel_fraction=0.00005,
    # Typical CMOS lens shading effect — moderate vignetting.
    vignetting_strength=0.3,
    # IMX219 has small lateral CA at the edges — 0.5 px max shift.
    chromatic_aberration_px=0.5,
)

INTEL_D435_RGB = CameraConfig(
    # Source: https://www.intelrealsense.com/depth-camera-d435/
    # Intel RealSense D435 RGB imager: OV2740, 1920×1080 @ 30 fps, rolling shutter.
    # Readout ≈ 26.4 ms of 33.3 ms frame → RS fraction ≈ 0.80.
    name="INTEL_D435_RGB",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.80,
    base_iso=100.0,
    iso=200.0,
    # OV2740 read noise ≈ 3 e⁻ at low gain.
    read_noise_sigma=3.0,
    full_well_electrons=3200.0,
    dead_pixel_fraction=0.0002,
    hot_pixel_fraction=0.0001,
    # Moderate vignetting from narrow fixed-focus lens.
    vignetting_strength=0.35,
    chromatic_aberration_px=0.8,
)

GOPRO_HERO11_4K30 = CameraConfig(
    # Source: https://community.gopro.com/s/article/HERO11-Black-FAQ
    # GoPro HERO 11 Black: 4K30, CMOS rolling shutter.
    # Typical RS skew for action cameras ≈ 16–18 ms / 33 ms → fraction ≈ 0.85.
    name="GOPRO_HERO11_4K30",
    update_rate_hz=30.0,
    resolution=(3840, 2160),
    rolling_shutter_fraction=0.85,
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=2.5,
    full_well_electrons=5000.0,
    dead_pixel_fraction=0.00005,
    hot_pixel_fraction=0.00002,
    # Wide-angle lens shows stronger vignetting & CA than typical surveillance cam.
    vignetting_strength=0.45,
    chromatic_aberration_px=1.2,
)

ZED2_LEFT = CameraConfig(
    # Source: https://www.stereolabs.com/assets/datasheets/zed2-datasheet.pdf
    # ZED 2 stereo camera: 1080p @ 30 fps per eye, rolling-shutter Sony IMX326.
    # Readout ≈ 25 ms / 33.3 ms → RS fraction ≈ 0.75.
    name="ZED2_LEFT",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.75,
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=1.8,
    full_well_electrons=4000.0,
    dead_pixel_fraction=0.0001,
    hot_pixel_fraction=0.00004,
    vignetting_strength=0.25,
    chromatic_aberration_px=0.6,
)

ZED2_RIGHT = CameraConfig(
    # Mirror of ZED2_LEFT — same sensor, same noise parameters; different name
    # for bookkeeping.  Each eye gets an independent RNG seed in practice.
    name="ZED2_RIGHT",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    rolling_shutter_fraction=0.75,
    base_iso=100.0,
    iso=100.0,
    read_noise_sigma=1.8,
    full_well_electrons=4000.0,
    dead_pixel_fraction=0.0001,
    hot_pixel_fraction=0.00004,
    vignetting_strength=0.25,
    chromatic_aberration_px=0.6,
)

ZED2_STEREO = StereoCameraConfig(
    # Source: https://www.stereolabs.com/assets/datasheets/zed2-datasheet.pdf
    # ZED 2: 12 cm baseline, 1080p @ 30 fps per eye, 110° HFOV.
    # Depth range: 0.2–20 m (indoor); up to 40 m in ideal conditions.
    # Disparity noise modelled at ~0.7 px (SGBM performance at mid-range).
    name="ZED2_STEREO",
    update_rate_hz=30.0,
    resolution=(1920, 1080),
    baseline_m=0.120,  # 12 cm
    hfov_deg=110.0,
    disparity_noise_sigma_px=0.7,
    min_depth_m=0.20,
    max_depth_m=20.0,
    iso=100.0,
    read_noise_sigma=1.8,
    vignetting_strength=0.25,
    chromatic_aberration_px=0.6,
)

INTEL_D435_STEREO = StereoCameraConfig(
    # Source: https://www.intelrealsense.com/depth-camera-d435/
    # Intel RealSense D435: 5 cm IR stereo baseline, 848×480 @ 90 fps.
    # Depth range: 0.1–10 m (active IR, indoor).
    # Active IR illumination drastically reduces failure on textureless surfaces.
    # Disparity noise equivalent: ~0.5 px (SGBM with active IR pattern).
    name="INTEL_D435_STEREO",
    update_rate_hz=30.0,
    resolution=(848, 480),
    baseline_m=0.050,  # 5 cm
    hfov_deg=86.0,  # D435 depth HFOV
    disparity_noise_sigma_px=0.5,
    min_depth_m=0.10,
    max_depth_m=10.0,
    iso=100.0,
    read_noise_sigma=2.0,
    vignetting_strength=0.0,  # active IR — uniform illumination
    chromatic_aberration_px=0.0,
)

MYNT_EYE_D_120 = StereoCameraConfig(
    # Source: https://www.myntai.com/mynteye/d (MYNT EYE D Wide, 120°)
    # MYNT EYE D-Wide: 12 cm baseline, 752×480 @ 60 fps, 120° fisheye HFOV.
    # Depth range: 0.6–6 m (active IR illumination).
    # Wide HFOV makes it popular for indoor UAV collision avoidance.
    name="MYNT_EYE_D_120",
    update_rate_hz=60.0,
    resolution=(752, 480),
    baseline_m=0.120,  # 12 cm
    hfov_deg=120.0,
    disparity_noise_sigma_px=0.8,  # fisheye rectification adds slight noise
    min_depth_m=0.30,  # reliable from 30 cm
    max_depth_m=6.0,  # active IR range
    iso=100.0,
    read_noise_sigma=2.0,
    vignetting_strength=0.40,  # strong fisheye vignetting
    chromatic_aberration_px=1.5,  # fisheye lens CA
)

# ---------------------------------------------------------------------------
# LiDAR presets
# ---------------------------------------------------------------------------

VELODYNE_VLP16 = LidarConfig(
    # Source: https://velodynelidar.com/wp-content/uploads/2019/12/63-9243-Rev-E-VLP-16-User-Manual.pdf
    # VLP-16 Puck: 16 channels, ±15° VFOV, 360° @ 10 Hz, ~1800 azimuth steps.
    # Range: 100 m, range noise: ±3 cm (1σ).
    name="VELODYNE_VLP16",
    update_rate_hz=10.0,
    n_channels=16,
    v_fov_deg=(-15.0, 15.0),
    h_resolution=1800,
    max_range_m=100.0,
    range_noise_sigma_m=0.03,
    intensity_noise_sigma=0.02,
    dropout_prob=0.001,
    beam_divergence_mrad=1.5,
)

VELODYNE_HDL64E = LidarConfig(
    # Source: https://velodynelidar.com/wp-content/uploads/2019/12/97-0038-Rev-N-HDL-64E-S3-S3D-DS.pdf
    # HDL-64E: 64 channels, -24.8° to +2° VFOV, 360° @ 10 Hz, 4500 azimuth steps.
    # Range: 120 m, range noise: ±2 cm.
    name="VELODYNE_HDL64E",
    update_rate_hz=10.0,
    n_channels=64,
    v_fov_deg=(-24.8, 2.0),
    h_resolution=4500,
    max_range_m=120.0,
    range_noise_sigma_m=0.02,
    intensity_noise_sigma=0.01,
    dropout_prob=0.0005,
    beam_divergence_mrad=1.3,
)

OUSTER_OS1_64 = LidarConfig(
    # Source: https://ouster.com/downloads/OS1_Lidar_Product_Datasheet.pdf
    # Ouster OS1-64: 64 channels, ±22.5° VFOV, 1024 pts/scan, 10/20 Hz.
    # Range: 120 m, range accuracy: ±3 cm.
    name="OUSTER_OS1_64",
    update_rate_hz=20.0,
    n_channels=64,
    v_fov_deg=(-22.5, 22.5),
    h_resolution=1024,
    max_range_m=120.0,
    range_noise_sigma_m=0.03,
    intensity_noise_sigma=0.01,
    dropout_prob=0.001,
    beam_divergence_mrad=2.0,
)

LIVOX_AVIA = LidarConfig(
    # Source: https://www.livoxtech.com/avia/specs
    # Livox Avia: 70.4°×77.2° non-repetitive FOV, 240k pts/s, range 450 m.
    # 6 channels modelled as equispaced over a 70° VFOV; 10 Hz rotation equivalent.
    # Range noise: 2 cm (1σ); beam divergence: 0.28 mrad (tight solid-state beam).
    name="LIVOX_AVIA",
    update_rate_hz=10.0,
    n_channels=6,
    v_fov_deg=(-35.0, 35.0),
    h_resolution=900,
    max_range_m=450.0,
    range_noise_sigma_m=0.02,
    intensity_noise_sigma=0.005,
    dropout_prob=0.0005,
    beam_divergence_mrad=0.28,
)

HESAI_XT32 = LidarConfig(
    # Source: HESAI XT32 datasheet, specification sheet rev 1.8
    # 32 channels, ±16° VFOV, 360° @ 10/20 Hz, 1800 azimuth steps.
    # Range: 120 m (10 % reflectivity), 200 m (80 % reflectivity).
    # Range accuracy: ±2 cm (1σ).  Beam divergence: 1.0 mrad.
    # Widely used in automotive and drone delivery applications.
    name="HESAI_XT32",
    update_rate_hz=10.0,
    n_channels=32,
    v_fov_deg=(-16.0, 16.0),
    h_resolution=1800,
    max_range_m=120.0,
    range_noise_sigma_m=0.02,
    intensity_noise_sigma=0.01,
    dropout_prob=0.0005,
    beam_divergence_mrad=1.0,
)

LIVOX_MID360 = LidarConfig(
    # Source: Livox Mid-360 datasheet
    # Compact 360° solid-state LiDAR widely used on mobile robots and drones.
    name="LIVOX_MID360",
    update_rate_hz=10.0,
    n_channels=32,
    v_fov_deg=(-7.0, 52.0),
    h_resolution=1200,
    max_range_m=70.0,
    range_noise_sigma_m=0.02,
    intensity_noise_sigma=0.008,
    dropout_prob=0.0007,
    beam_divergence_mrad=0.35,
)

SICK_TIM571 = LidarConfig(
    # Source: SICK TiM571 2D LiDAR datasheet
    # Rugged planar scanner used in warehouse AMRs and outdoor AGV stacks.
    name="SICK_TIM571",
    update_rate_hz=15.0,
    n_channels=1,
    v_fov_deg=(-0.1, 0.1),
    h_resolution=811,
    max_range_m=25.0,
    range_noise_sigma_m=0.015,
    intensity_noise_sigma=0.01,
    dropout_prob=0.001,
    beam_divergence_mrad=2.2,
)

# ---------------------------------------------------------------------------
# IMU presets
# ---------------------------------------------------------------------------

PIXHAWK_ICM20689 = IMUConfig(
    # Source: TDK InvenSense ICM-20689 datasheet, rev 1.1
    # Pixhawk 4 primary IMU.
    # Accel noise density: 300 µg/√Hz ≈ 2.94e-3 m/s²/√Hz
    # Gyro noise density: 0.01 °/s/√Hz ≈ 1.75e-4 rad/s/√Hz
    # Bias instability (accel): 0.05 mg; (gyro): 3.8 °/hr.
    name="PIXHAWK_ICM20689",
    update_rate_hz=400.0,
    noise_density_acc=2.94e-3,
    noise_density_gyr=1.75e-4,
    bias_tau_acc_s=200.0,
    bias_sigma_acc=4.9e-4,  # 0.05 mg in m/s²
    bias_tau_gyr_s=200.0,
    bias_sigma_gyr=1.83e-5,  # 3.8 °/hr in rad/s
    scale_factor_acc=3e-3,
    scale_factor_gyr=3e-3,
    add_gravity=True,
)

VECTORNAV_VN100 = IMUConfig(
    # Source: VectorNav VN-100 datasheet (Rev 2)
    # Tactical-grade MEMS IMU.
    # Accel noise density: 0.14 mg/√Hz ≈ 1.37e-3 m/s²/√Hz
    # Gyro noise density: 0.0035 °/s/√Hz ≈ 6.11e-5 rad/s/√Hz
    # Bias instability (accel): 0.04 mg; (gyro): 10 °/hr.
    name="VECTORNAV_VN100",
    update_rate_hz=800.0,
    noise_density_acc=1.37e-3,
    noise_density_gyr=6.11e-5,
    bias_tau_acc_s=500.0,
    bias_sigma_acc=3.92e-4,  # 0.04 mg in m/s²
    bias_tau_gyr_s=500.0,
    bias_sigma_gyr=4.85e-5,  # 10 °/hr in rad/s
    scale_factor_acc=1e-3,
    scale_factor_gyr=1e-3,
    add_gravity=True,
)

XSENS_MTI_3 = IMUConfig(
    # Source: Xsens MTi-3 datasheet, Document MT0605P, Rev AE
    # Accel noise density: 120 µg/√Hz ≈ 1.18e-3 m/s²/√Hz
    # Gyro noise density: 0.007 °/s/√Hz ≈ 1.22e-4 rad/s/√Hz
    # Accel bias stability: 15 µg; gyro in-run bias: 2 °/hr.
    name="XSENS_MTI_3",
    update_rate_hz=400.0,
    noise_density_acc=1.18e-3,
    noise_density_gyr=1.22e-4,
    bias_tau_acc_s=300.0,
    bias_sigma_acc=1.47e-4,  # 15 µg in m/s²
    bias_tau_gyr_s=300.0,
    bias_sigma_gyr=9.70e-6,  # 2 °/hr in rad/s
    scale_factor_acc=5e-4,
    scale_factor_gyr=5e-4,
    add_gravity=True,
)

BOSCH_BMI088 = IMUConfig(
    # Source: Bosch BMI088 datasheet, revision 1.8
    # Primary IMU on Pixhawk 6X and many modern flight-controller designs.
    # Accel noise density: 175 µg/√Hz ≈ 1.72e-3 m/s²/√Hz (low-noise mode)
    # Gyro noise density: 0.014 °/s/√Hz ≈ 2.44e-4 rad/s/√Hz
    # Accel bias stability: 20 µg; gyro bias: 3 °/hr.
    # Cross-axis sensitivity: 1 % typical → 0.01.
    name="BOSCH_BMI088",
    update_rate_hz=400.0,
    noise_density_acc=1.72e-3,
    noise_density_gyr=2.44e-4,
    bias_tau_acc_s=200.0,
    bias_sigma_acc=1.96e-4,  # 20 µg in m/s²
    bias_tau_gyr_s=200.0,
    bias_sigma_gyr=1.45e-5,  # 3 °/hr in rad/s
    scale_factor_acc=2e-3,
    scale_factor_gyr=2e-3,
    cross_axis_sensitivity_acc=0.01,
    cross_axis_sensitivity_gyr=0.01,
    add_gravity=True,
)

INVENSENSE_MPU9250 = IMUConfig(
    # Source: InvenSense MPU-9250 Product Specification, revision 1.1
    # Popular low-cost 9-axis IMU; used in maker boards (Raspberry Pi shields,
    # Arduino Nano 33 BLE, many hobbyist EFCs).
    # Accel noise density: 300 µg/√Hz ≈ 2.94e-3 m/s²/√Hz
    # Gyro noise density: 0.01 °/s/√Hz ≈ 1.75e-4 rad/s/√Hz
    # Accel bias stability: 60 µg; gyro bias: 5 °/hr.
    # Large cross-axis sensitivity: ~2%.
    name="INVENSENSE_MPU9250",
    update_rate_hz=200.0,
    noise_density_acc=2.94e-3,
    noise_density_gyr=1.75e-4,
    bias_tau_acc_s=200.0,
    bias_sigma_acc=5.89e-4,  # 60 µg in m/s²
    bias_tau_gyr_s=200.0,
    bias_sigma_gyr=2.42e-5,  # 5 °/hr in rad/s
    scale_factor_acc=5e-3,
    scale_factor_gyr=5e-3,
    cross_axis_sensitivity_acc=0.02,
    cross_axis_sensitivity_gyr=0.02,
    add_gravity=True,
)

# ---------------------------------------------------------------------------
# GNSS presets
# ---------------------------------------------------------------------------

UBLOX_M8N = GNSSConfig(
    # Source: u-blox NEO-M8N datasheet, UBX-13003366
    # CEP: 2.5 m, velocity accuracy: 0.05 m/s (RMS), update rate: 10 Hz.
    # 1-sigma per-axis (2D isotropic Gaussian): σ = CEP / √(2 ln 2) ≈ 2.5 / 1.1774 ≈ 2.12 m.
    name="UBLOX_M8N",
    update_rate_hz=10.0,
    noise_m=2.12,
    vel_noise_ms=0.05,
    bias_tau_s=60.0,
    bias_sigma_m=0.5,
    multipath_sigma_m=1.5,
    min_fix_altitude_m=0.5,
)

UBLOX_F9P_RTK = GNSSConfig(
    # Source: u-blox ZED-F9P datasheet, UBX-17051259
    # RTK fix: 0.01 m CEP (horizontal), heading accuracy: 0.4 °.
    # 1-sigma per-axis (from CEP): σ = 0.01 m / 1.1774 ≈ 0.0085 m; using 0.012 m
    # to include residual tropospheric / multipath contributions.
    # Float solution degrades to ~0.1 m.
    name="UBLOX_F9P_RTK",
    update_rate_hz=20.0,
    noise_m=0.012,
    vel_noise_ms=0.005,
    bias_tau_s=120.0,
    bias_sigma_m=0.01,
    multipath_sigma_m=0.05,
    min_fix_altitude_m=0.1,
)

NOVATEL_OEM7 = GNSSConfig(
    # Source: NovAtel OEM7 Solutions datasheet, OM-20000129
    # Autonomous GNSS: 1.5 m RMS horizontal, 2.5 m vertical.
    # SBAS-corrected: 0.6 m horizontal.  Using autonomous here.
    # Velocity: 0.03 m/s RMS.
    name="NOVATEL_OEM7",
    update_rate_hz=20.0,
    noise_m=1.5,
    vel_noise_ms=0.03,
    bias_tau_s=90.0,
    bias_sigma_m=0.3,
    multipath_sigma_m=0.8,
    min_fix_altitude_m=0.3,
)

# ---------------------------------------------------------------------------
# Thermal camera presets
# ---------------------------------------------------------------------------

FLIR_LEPTON_35 = ThermalCameraConfig(
    # Source: FLIR Lepton 3.5 datasheet, version 2.1.0
    # 160×120 LWIR uncooled VOx microbolometer, 8.7 Hz frame rate.
    # Spectral range: 8–14 µm.  NEDT < 50 mK → noise_sigma ≈ 0.05°C.
    # NUC residual (after factory shutter NUC): ~0.5°C.
    # PSF: diffraction-limited at f/1.1; ~1.5-pixel sigma.
    # Operating temp range: -10°C to 140°C (scene side).
    name="FLIR_LEPTON_35",
    update_rate_hz=8.7,
    resolution=(160, 120),
    psf_sigma=1.5,
    nuc_sigma=0.5,
    noise_sigma=0.05,
    bit_depth=14,
    temp_range_c=(-10.0, 140.0),
)

FLIR_BOSON_320 = ThermalCameraConfig(
    # Source: FLIR Boson 320 datasheet (doc 102-PS245-35)
    # 320×256 LWIR uncooled VOx, 60 Hz frame rate.
    # Spectral range: 7.5–13.5 µm.  NEDT < 60 mK → noise_sigma ≈ 0.06°C.
    # NUC: ~0.3°C residual (Boson performs on-camera NUC automatically).
    # PSF: f/1.0 lens; ~1.2-pixel sigma.
    # Wide operating temperature range: -20°C to 550°C (with extended range).
    name="FLIR_BOSON_320",
    update_rate_hz=60.0,
    resolution=(320, 256),
    psf_sigma=1.2,
    nuc_sigma=0.3,
    noise_sigma=0.06,
    bit_depth=14,
    temp_range_c=(-20.0, 450.0),
)

SEEK_THERMAL_COMPACT_PRO = ThermalCameraConfig(
    # Source: Seek Thermal CompactPro datasheet (320×240, 15 fps).
    # Semiconductor-based 320×240 LWIR detector, 15 Hz.
    # NEDT: ~100 mK → noise_sigma ≈ 0.1°C.  NUC residual: ~0.8°C.
    # Consumer-grade optics; larger PSF than FLIR (f/1.5 → ~2.0 px sigma).
    name="SEEK_THERMAL_COMPACT_PRO",
    update_rate_hz=15.0,
    resolution=(320, 240),
    psf_sigma=2.0,
    nuc_sigma=0.8,
    noise_sigma=0.1,
    bit_depth=14,
    temp_range_c=(-40.0, 330.0),
)

# ---------------------------------------------------------------------------
# Event camera presets
# ---------------------------------------------------------------------------

DAVIS_346 = EventCameraConfig(
    # Source: iniVation DAVIS346 datasheet
    # 346×260 px DVS+APS sensor; APS rate 200 fps, events at kHz rates.
    # Default threshold: 0.25 log-intensity; threshold variation: ~15% σ/C.
    # Refractory period: ~300 µs → 0.3 ms.
    # Background activity (dark room): ~0.1 events/px/s.
    # update_rate_hz set to APS frame rate (200 fps) for frame-driven step().
    name="DAVIS_346",
    update_rate_hz=200.0,
    threshold_pos=0.25,
    threshold_neg=0.25,
    refractory_period_s=3e-4,
    threshold_variation=0.10,
    background_activity_rate_hz=0.1,
)

PROPHESEE_EVK4 = EventCameraConfig(
    # Source: Prophesee Metavision EVK4 datasheet (IMX636 sensor)
    # 1280×720 HD event sensor; temporal resolution ~100 µs.
    # Low threshold (< 0.1 log-intensity) enables very fine motion detection.
    # Excellent threshold uniformity: ~5% σ/C.
    # Background activity: ~0.05 events/px/s.
    name="PROPHESEE_EVK4",
    update_rate_hz=1000.0,
    threshold_pos=0.15,
    threshold_neg=0.15,
    refractory_period_s=1e-4,
    threshold_variation=0.05,
    background_activity_rate_hz=0.05,
)

# ---------------------------------------------------------------------------
# Barometer presets
# ---------------------------------------------------------------------------

BMP388 = BarometerConfig(
    # Source: Bosch BMP388 datasheet, version 1.5
    # Used on Pixhawk 6C/6X as the primary barometer.
    # Relative accuracy at constant temperature: ±0.06 hPa → ±0.5 m.
    # Absolute accuracy (full operating range): ±0.5 hPa → ±4.2 m.
    # Update rate: up to 200 Hz; typical flight controller usage: 50-100 Hz.
    # Noise (1-sigma from datasheet noise density): ~0.12 m.
    # Bias drift (temperature): bias_sigma ~1.5 m over 5-min flight.
    name="BMP388",
    update_rate_hz=100.0,
    noise_sigma_m=0.12,
    bias_tau_s=200.0,
    bias_sigma_m=1.5,
    resolution_m=0.001,  # 24-bit ADC over ~40 hPa span → ≈0.15 mPa/LSB ≈ 0.01 m; rounded to 1 mm
)

MS5611 = BarometerConfig(
    # Source: TE Connectivity MS5611-01BA03 datasheet, revision D
    # Legacy high-res barometer on older Pixhawk and Ardupilot boards.
    # Resolution: 0.012 mbar → ~0.10 m altitude resolution.
    # RMS noise at max resolution (OSR 4096): ~0.5 m.
    # Absolute accuracy: ±1.5 mbar → ±12.5 m over temp; bias_sigma ~3 m.
    name="MS5611",
    update_rate_hz=50.0,
    noise_sigma_m=0.5,
    bias_tau_s=300.0,
    bias_sigma_m=3.0,
    resolution_m=0.1,  # 0.012 mbar ≈ 0.10 m at sea level
)

SPL06_001 = BarometerConfig(
    # Source: Goertek SPL06-001 datasheet, revision 1.4
    # Used on Holybro Pixhawk 6C Mini and many mid-range flight controllers.
    # RMS noise at 128x oversampling: ~0.07 m.
    # Accuracy: ±0.5 hPa → bias_sigma ~3 m over temperature range.
    name="SPL06_001",
    update_rate_hz=100.0,
    noise_sigma_m=0.07,
    bias_tau_s=250.0,
    bias_sigma_m=2.5,
    resolution_m=0.0,
)

BMP280 = BarometerConfig(
    # Source: Bosch BMP280 datasheet, revision 1.14
    # Entry-level barometric pressure sensor; found on many cheap FC boards and
    # maker kits.  Relative accuracy ±0.12 hPa → altitude noise ≈ 1 m RMS.
    # In-run drift over temperature: ~1 hPa → bias_sigma ≈ 8 m.
    name="BMP280",
    update_rate_hz=25.0,
    noise_sigma_m=1.0,
    bias_tau_s=120.0,
    bias_sigma_m=8.0,
    resolution_m=0.1,
)

DPS310 = BarometerConfig(
    # Source: Infineon DPS310 datasheet, revision 1.1
    # Used on Raspberry Pi Sense HAT 2 and several drone ESC/FC boards.
    # Pressure precision: ±0.06 Pa → altitude noise ≈ 0.05 m RMS.
    # Absolute accuracy: ±0.5 hPa → bias_sigma ≈ 4 m.
    name="DPS310",
    update_rate_hz=128.0,
    noise_sigma_m=0.05,
    bias_tau_s=300.0,
    bias_sigma_m=4.0,
    resolution_m=0.0,
)

# ---------------------------------------------------------------------------
# Environmental sensor presets
# ---------------------------------------------------------------------------

DS18B20_PROBE = ThermometerConfig(
    # Source: Maxim DS18B20 datasheet
    # Digital 1-wire probe; accuracy ±0.5 °C over the typical robotics range.
    # Conversion time ≈ 750 ms at 12-bit resolution.
    name="DS18B20_PROBE",
    update_rate_hz=1.0,
    noise_sigma_c=0.10,
    bias_tau_s=1200.0,
    bias_sigma_c=0.20,
    response_tau_s=0.75,
)

SHT31_HUMIDITY = HygrometerConfig(
    # Source: Sensirion SHT31 datasheet
    # Relative humidity accuracy ±2 %RH with sub-second response.
    name="SHT31_HUMIDITY",
    update_rate_hz=2.0,
    noise_sigma_pct=0.8,
    bias_tau_s=900.0,
    bias_sigma_pct=1.5,
    response_tau_s=0.5,
)

TSL2591_LIGHT = LightSensorConfig(
    # Source: AMS TSL2591 datasheet
    # High-dynamic-range ambient light sensor, up to ~88 klux practical range.
    name="TSL2591_LIGHT",
    update_rate_hz=10.0,
    noise_sigma_ratio=0.02,
    min_lux=0.0,
    max_lux=88_000.0,
    response_tau_s=0.2,
)

SGP30_AIR_QUALITY = GasSensorConfig(
    # Source: Sensirion SGP30 datasheet
    # VOC / eCO2 air-quality sensor with ~12 s response and alarm-friendly ppm output.
    name="SGP30_AIR_QUALITY",
    update_rate_hz=1.0,
    background_ppm=420.0,
    noise_sigma_ppm=15.0,
    response_tau_s=12.0,
    alarm_threshold_ppm=1000.0,
    max_ppm=60_000.0,
    plume_sigma_m=1.2,
)

DAVIS_6410_ANEMOMETER = AnemometerConfig(
    # Source: Davis Instruments 6410 anemometer specification sheet
    # Cup anemometer / vane weather station head, widely used in outdoor robotics setups.
    name="DAVIS_6410_ANEMOMETER",
    update_rate_hz=4.0,
    noise_sigma_ms=0.10,
    direction_noise_deg=3.0,
    measure_relative_wind=False,
    max_speed_ms=89.0,
)

# ---------------------------------------------------------------------------
# Wireless ranging / radar presets
# ---------------------------------------------------------------------------

QORVO_DWM3001C = UWBRangeConfig(
    # Source: Qorvo DWM3001C specification summary
    # Modern UWB anchor/tag module for indoor localization and robot docking.
    name="QORVO_DWM3001C",
    update_rate_hz=20.0,
    range_noise_sigma_m=0.03,
    dropout_prob=0.01,
    max_range_m=60.0,
    nlos_bias_m=0.20,
    tx_power_dbm=0.0,
    estimate_position=True,
)

TI_IWR6843AOP = RadarConfig(
    # Source: TI IWR6843AOP mmWave radar datasheet
    # Short-range 60 GHz FMCW radar for occupancy, tracking, and robot awareness.
    name="TI_IWR6843AOP",
    update_rate_hz=20.0,
    max_range_m=60.0,
    min_range_m=0.2,
    azimuth_fov_deg=120.0,
    elevation_fov_deg=60.0,
    range_noise_sigma_m=0.08,
    velocity_noise_sigma_ms=0.05,
    azimuth_noise_deg=0.25,
    elevation_noise_deg=0.25,
    detection_prob=0.98,
    false_alarm_rate=0.03,
    rain_attenuation_db_per_mm_h=0.08,
)

NAVTECH_CTS350X = RadarConfig(
    # Source: Navtech CTS350-X perimeter radar overview
    # Long-range mechanically scanned radar used for outdoor surveillance and autonomous vehicles.
    name="NAVTECH_CTS350X",
    update_rate_hz=4.0,
    max_range_m=250.0,
    min_range_m=1.0,
    azimuth_fov_deg=360.0,
    elevation_fov_deg=20.0,
    range_noise_sigma_m=0.25,
    velocity_noise_sigma_ms=0.10,
    azimuth_noise_deg=0.15,
    elevation_noise_deg=0.40,
    detection_prob=0.99,
    false_alarm_rate=0.08,
    rain_attenuation_db_per_mm_h=0.05,
)

# ---------------------------------------------------------------------------
# Magnetometer presets
# ---------------------------------------------------------------------------

IST8310 = MagnetometerConfig(
    # Source: iSentek IST8310 datasheet, revision 1.3
    # Used on Pixhawk 4, Cube Orange, Holybro Kakute boards.
    # Resolution: 0.3 µT per LSB.  RMS noise: ~0.3 µT/axis.
    # Hard-iron offset depends on installation; using 0 for clean installation.
    # Soft-iron scale: ~1-2% asymmetry in typical drone body → 1% on each axis.
    name="IST8310",
    update_rate_hz=200.0,
    noise_sigma_ut=0.3,
    field_amplitude_ut=50.0,
    declination_deg=0.0,
    inclination_deg=60.0,
)

HMC5883L = MagnetometerConfig(
    # Source: Honeywell HMC5883L datasheet, revision D
    # Classic 3-axis magnetometer; widely used on older Pixhawk hardware.
    # Resolution: 0.73 mGauss (≈0.073 µT) at default gain.
    # RMS noise: ~2 mGauss ≈ 0.2 µT at 8x oversampling.
    name="HMC5883L",
    update_rate_hz=160.0,
    noise_sigma_ut=0.2,
    field_amplitude_ut=50.0,
    declination_deg=0.0,
    inclination_deg=60.0,
)

RM3100 = MagnetometerConfig(
    # Source: PNI Sensor RM3100 datasheet, revision B
    # High-accuracy magneto-inductive sensor; used on Pixhawk cube / premium boards.
    # Much lower noise than MEMS alternatives: ~0.05 µT/axis at standard settings.
    # Used in precision applications (magnetometer-aided INS).
    name="RM3100",
    update_rate_hz=300.0,
    noise_sigma_ut=0.05,
    field_amplitude_ut=50.0,
    declination_deg=0.0,
    inclination_deg=60.0,
)

# ---------------------------------------------------------------------------
# GNSS additional presets
# ---------------------------------------------------------------------------

TRIMBLE_BD992 = GNSSConfig(
    # Source: Trimble BD992 GNSS Receiver Reference Manual, revision B
    # Survey-grade dual-constellation dual-frequency RTK receiver; used in
    # precision agriculture, civil-engineering drones, and ground-truth rigs.
    # RTK accuracy: 8 mm horizontal + 1 ppm baseline.
    # Horizontal noise modelled as 8 mm; multipath < 5 mm in open sky.
    # Very low bias drift; long correlation time (10 min).
    name="TRIMBLE_BD992",
    update_rate_hz=20.0,
    noise_m=0.008,
    vel_noise_ms=0.003,
    multipath_sigma_m=0.005,
    bias_tau_s=600.0,
    bias_sigma_m=0.01,
)

EMLID_REACH_RS2 = GNSSConfig(
    # Source: Emlid Reach RS2 specification sheet
    # Affordable dual-band RTK receiver; popular in open-source survey workflows.
    # RTK horizontal: 7 mm + 1 ppm.  Modelled at 5 Hz with 10 mm position noise.
    name="EMLID_REACH_RS2",
    update_rate_hz=5.0,
    noise_m=0.010,
    vel_noise_ms=0.005,
    multipath_sigma_m=0.01,
    bias_tau_s=300.0,
    bias_sigma_m=0.02,
)

# ---------------------------------------------------------------------------
# Airspeed presets
# ---------------------------------------------------------------------------

SDP33 = AirspeedConfig(
    # Source: Sensirion SDP33 datasheet, revision 2.0
    # High-precision differential-pressure sensor; popular on TBS/Matek
    # Pitot-tube modules and Ardupilot SITL reference builds.
    # Full-scale ±500 Pa; accuracy \u00b13 Pa \u2192 Δv \u2248 0.07 m/s at 10 m/s TAS.
    # Noise spectral density ~0.1 Pa/√Hz \u2192 effective speed noise ~0.1 m/s at 100 Hz.
    # Min detectable: ~0.5 m/s (sensor limited by pressure resolution).
    name="SDP33",
    update_rate_hz=100.0,
    noise_sigma_ms=0.1,
    bias_tau_s=600.0,
    bias_sigma_ms=0.3,
    min_detectable_ms=0.5,
    max_speed_ms=100.0,
    tube_blockage_prob=0.0002,
)

MS4525DO = AirspeedConfig(
    # Source: Measurement Specialties MS4525DO datasheet, DocID 001504
    # Standard pitot sensor used on 3DR Pixhawk 1 / mRo pitot assemblies and
    # ArduPlane stock pitot builds.
    # Full-scale ±1 PSI differential; typical airspeed range 0\u201360 m/s.
    # Effective speed noise: ~0.3 m/s (12-bit ADC over 1 PSI FSR).
    # Min detectable: ~2 m/s; max: ~60 m/s for standard UAV operation.
    name="MS4525DO",
    update_rate_hz=50.0,
    noise_sigma_ms=0.3,
    bias_tau_s=300.0,
    bias_sigma_ms=0.8,
    min_detectable_ms=2.0,
    max_speed_ms=60.0,
    tube_blockage_prob=0.0005,
)

# ---------------------------------------------------------------------------
# Rangefinder presets
# ---------------------------------------------------------------------------

TFMINI_PLUS = RangefinderConfig(
    # Source: Benewake TFmini Plus datasheet, revision 1.0
    # Compact single-point ToF LiDAR; widely used for drone terrain-following
    # and UAV landing.  Range: 0.1\u201312 m.  Accuracy: ±6 cm or ±1 %  (whichever
    # greater).  Detection rate: 100 Hz.  FOV: 3.6° (narrow beam).
    # accuracy_mode="max" correctly models "±6 cm OR ±1% (whichever is greater)".
    name="TFMINI_PLUS",
    update_rate_hz=100.0,
    min_range_m=0.1,
    max_range_m=12.0,
    noise_floor_m=0.06,
    noise_slope=0.01,
    accuracy_mode="max",
    dropout_prob=0.001,
    resolution_m=0.001,
    no_hit_value=0.0,
)

TERARANGER_ONE = RangefinderConfig(
    # Source: TeraRanger One datasheet (Terabee), v2.0
    # IR time-of-flight sensor; range 0.2\u201314 m; accuracy \u00b12 cm.
    # 240 Hz max measurement rate (100 Hz used here for compatibility).
    # Higher noise than LiDAR due to multi-path IR effects.
    name="TERARANGER_ONE",
    update_rate_hz=100.0,
    min_range_m=0.2,
    max_range_m=14.0,
    noise_floor_m=0.04,
    noise_slope=0.005,
    dropout_prob=0.002,
    resolution_m=0.001,
    no_hit_value=0.0,
)

GARMIN_LIDAR_LITE_V3 = RangefinderConfig(
    # Source: Garmin LIDAR-Lite v3 Technical Manual, revision 2
    # Popular laser rangefinder for drones; range 0\u201340 m; accuracy \u00b12.5 cm.
    # Standard mode: ~100 Hz; high-accuracy mode: 40 Hz.  Widely used for
    # ceiling/altitude measurement on indoor platforms and landing detection.
    name="GARMIN_LIDAR_LITE_V3",
    update_rate_hz=40.0,
    min_range_m=0.0,
    max_range_m=40.0,
    noise_floor_m=0.025,
    noise_slope=0.003,
    dropout_prob=0.001,
    resolution_m=0.01,  # 1 cm from 8-bit ADC over 40 m FSR
    no_hit_value=0.0,
)

# ---------------------------------------------------------------------------
# Registry and helpers
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Optical flow presets
# ---------------------------------------------------------------------------

PX4FLOW = OpticalFlowConfig(
    # Source: PX4FLOW v1.3 datasheet / PX4 autopilot documentation
    # CMOS optical flow camera with sonar; 752×480 pixel MT9V034 sensor;
    # processes at 400 fps internally, outputs integrated flow at up to 400 Hz.
    # FOV 64° (H) × 48° (V).  Nominal quality height: ~2 m (feature tracking
    # becomes unreliable above ~20 m AGL with standard lens).
    name="PX4FLOW",
    update_rate_hz=100.0,
    noise_floor_rad_s=0.003,  # quantisation + read noise at typical speed
    noise_slope=0.04,  # motion blur coefficient
    max_detection_rad_s=4.0,  # ±4 rad/s documented saturation
    nominal_quality_height_m=2.0,
    max_quality_height_m=20.0,
    base_quality=220,
    low_quality_threshold=25,
)

MATEKSYS_3901_L0X = OpticalFlowConfig(
    # Source: MatekSys 3901-L0X product page / Betaflight documentation
    # PAA3905EX optical flow + VL53L1X ToF rangefinder combo board.
    # Flow output: up to 100 Hz over UART/I²C.  Typical indoor flight range
    # 0.08–2 m; outdoor quality drops sharply above ~5 m (no IR illumination).
    name="MATEKSYS_3901_L0X",
    update_rate_hz=100.0,
    noise_floor_rad_s=0.006,  # slightly noisier CMOS sensor than PX4FLOW
    noise_slope=0.06,
    max_detection_rad_s=3.5,
    nominal_quality_height_m=1.0,  # PAA3905 optimised for <2 m indoor flight
    max_quality_height_m=8.0,  # quality collapses quickly without IR illumination
    base_quality=200,
    low_quality_threshold=30,
)

# ---------------------------------------------------------------------------
# Battery monitor presets
# ---------------------------------------------------------------------------

LIPO_3S_2200MAH = BatteryConfig(
    # Typical 3S 2200 mAh LiPo pack used in small racing/proximity drones.
    # Internal resistance estimate from common Tattu / Turnigy specs: ~15 mΩ/cell.
    name="LIPO_3S_2200MAH",
    update_rate_hz=10.0,
    n_cells=3,
    capacity_mah=2200.0,
    cell_chemistry="lipo",
    internal_resistance_ohm=0.045,  # 3 × 15 mΩ/cell
    v_warn_per_cell_v=3.65,
    current_noise_a=0.08,  # small Hall-effect sensor
    voltage_noise_v=0.012,
    initial_soc=1.0,
)

LIPO_4S_5000MAH = BatteryConfig(
    # Common 4S 5000 mAh pack used in X-class / heavy-lift drones.
    # Lower Ri/cell due to high-discharge rating (≈8–12 mΩ/cell).
    name="LIPO_4S_5000MAH",
    update_rate_hz=10.0,
    n_cells=4,
    capacity_mah=5000.0,
    cell_chemistry="lipo",
    internal_resistance_ohm=0.040,  # 4 × 10 mΩ/cell
    v_warn_per_cell_v=3.65,
    current_noise_a=0.10,  # Holybro PM02-style shunt sensor
    voltage_noise_v=0.015,
    initial_soc=1.0,
)

LIPO_6S_10000MAH = BatteryConfig(
    # High-capacity 6S pack typical for long-endurance mapping UAVs.
    name="LIPO_6S_10000MAH",
    update_rate_hz=10.0,
    n_cells=6,
    capacity_mah=10000.0,
    cell_chemistry="lipo",
    internal_resistance_ohm=0.054,  # 6 × 9 mΩ/cell
    v_warn_per_cell_v=3.65,
    current_noise_a=0.15,
    voltage_noise_v=0.020,
    initial_soc=1.0,
)


# ---------------------------------------------------------------------------
# Wheel odometry presets
# ---------------------------------------------------------------------------

DIFF_DRIVE_ENCODER_50HZ = WheelOdometryConfig(
    # Generic 100 mm-wheel differential-drive robot encoder stack.
    # Typical low-cost quadrature encoder: 2000 CPR → ~0.16 mm/tick.
    # Position noise modelled as ~2 mm per step at 50 Hz; heading noise ~1 mrad.
    # Slip sigma ~0.5 % (smooth surface, rubber wheels).
    name="DIFF_DRIVE_ENCODER_50HZ",
    update_rate_hz=50.0,
    pos_noise_sigma_m=0.002,
    heading_noise_sigma_rad=0.001,
    slip_sigma=0.005,
)

MECANUM_DRIVE_ENCODER_100HZ = WheelOdometryConfig(
    # Mecanum-wheel platform (150 mm wheels) with 4× encoders, 100 Hz output.
    # Mecanum wheels have higher lateral slip → larger pos_noise and slip_sigma.
    # Typical application: indoor warehouse AMR.
    name="MECANUM_DRIVE_ENCODER_100HZ",
    update_rate_hz=100.0,
    pos_noise_sigma_m=0.004,
    heading_noise_sigma_rad=0.002,
    slip_sigma=0.015,  # lateral slip is significant with mecanum rollers
)


# ---------------------------------------------------------------------------
# Force/torque sensor presets
# ---------------------------------------------------------------------------

ATI_MINI45 = ForceTorqueConfig(
    # Source: ATI Industrial Automation Mini45 datasheet (SI-580-20 range)
    # Range: ±580 N (Fx, Fy), ±1160 N (Fz); ±20 Nm (Tx, Ty, Tz).
    # Resolution: ~1/8192 of range per axis.
    # Noise floor (RMS): ~0.05 N force, ~0.001 Nm torque.
    # Very popular for robot manipulation research.
    name="ATI_MINI45",
    update_rate_hz=7000.0,
    force_noise_sigma_n=0.05,
    torque_noise_sigma_nm=0.001,
    force_range_n=580.0,
    torque_range_nm=20.0,
    seed=None,
)

ROKUBI_FT300 = ForceTorqueConfig(
    # Source: Robotiq FT 300-S datasheet (2022)
    # Range: ±300 N (Fx, Fy), ±300 N (Fz); ±30 Nm (Tx, Ty, Tz).
    # Noise: < 0.1 N RMS (force), < 0.005 Nm RMS (torque).
    # Widely used in collaborative robot (cobot) end-effectors.
    name="ROKUBI_FT300",
    update_rate_hz=100.0,
    force_noise_sigma_n=0.10,
    torque_noise_sigma_nm=0.005,
    force_range_n=300.0,
    torque_range_nm=30.0,
    seed=None,
)

OPTOFORCE_OMD = ForceTorqueConfig(
    # Source: OptoForce OMD-10-SE-10N datasheet
    # Range: ±10 N (Fx, Fy, Fz); ±500 mNm (Tx, Ty, Tz).
    # Resolution: ~3 mN (force), ~0.1 mNm (torque).
    # Compact optical sensor for dexterous manipulation and tactile sensing.
    name="OPTOFORCE_OMD",
    update_rate_hz=1000.0,
    force_noise_sigma_n=0.003,
    torque_noise_sigma_nm=0.0001,
    force_range_n=10.0,
    torque_range_nm=0.5,
    seed=None,
)


# ---------------------------------------------------------------------------
# Joint state presets
# ---------------------------------------------------------------------------

FRANKA_JOINT_ENCODER = JointStateConfig(
    # Source: Franka Emika Panda datasheet / FCI documentation
    # Absolute joint encoders with 13-bit resolution per joint (~0.04 mrad).
    # Velocity output is differentiated at 1 kHz with minimal filtering.
    name="FRANKA_JOINT_ENCODER",
    update_rate_hz=1000.0,
    pos_noise_sigma_rad=0.00005,
    vel_noise_sigma_rads=0.0005,
    torque_noise_sigma_nm=0.05,
    velocity_filter_alpha=0.0,
    seed=None,
)

UR5_JOINT_ENCODER = JointStateConfig(
    # Source: Universal Robots UR5e datasheet
    # 18-bit absolute encoders per joint; joint speed control at 500 Hz.
    # Slightly higher torque noise than Franka due to series-elastic actuation.
    name="UR5_JOINT_ENCODER",
    update_rate_hz=500.0,
    pos_noise_sigma_rad=0.0001,
    vel_noise_sigma_rads=0.001,
    torque_noise_sigma_nm=0.10,
    velocity_filter_alpha=0.2,
    seed=None,
)

GENERIC_SERVO_ENCODER = JointStateConfig(
    # Representative hobby / research servo using 12-bit magnetic encoder.
    # Typical for quadrupeds and small manipulators.
    name="GENERIC_SERVO_ENCODER",
    update_rate_hz=200.0,
    pos_noise_sigma_rad=0.001,
    vel_noise_sigma_rads=0.005,
    torque_noise_sigma_nm=0.5,
    velocity_filter_alpha=0.3,
    seed=None,
)


# ---------------------------------------------------------------------------
# Contact sensor presets
# ---------------------------------------------------------------------------

FINGERTIP_TACTILE_200HZ = ContactSensorConfig(
    # Representative fingertip tactile pad for dexterous manipulation.
    # Threshold set for light touch detection; fast 200 Hz update rate.
    name="FINGERTIP_TACTILE_200HZ",
    update_rate_hz=200.0,
    force_threshold_n=0.1,
    noise_sigma_n=0.01,
    force_range_n=20.0,
    debounce_steps=0,
    seed=None,
)

BUMPER_50HZ = ContactSensorConfig(
    # Simple binary bumper sensor for mobile robots.
    # Higher threshold and debounce to suppress vibration false positives.
    name="BUMPER_50HZ",
    update_rate_hz=50.0,
    force_threshold_n=2.0,
    noise_sigma_n=0.1,
    force_range_n=100.0,
    debounce_steps=3,
    seed=None,
)


# ---------------------------------------------------------------------------
# Depth camera presets
# ---------------------------------------------------------------------------

INTEL_D415 = DepthCameraConfig(
    # Source: Intel RealSense D415 depth camera datasheet
    # Structured-light stereo depth.  Depth accuracy ≈ 1.5 % at 1 m.
    # Range: 0.3 m – 10 m.  Native resolution: 1280×720 @ 30 fps.
    name="INTEL_D415",
    update_rate_hz=30.0,
    resolution=(1280, 720),
    depth_noise_sigma_m=0.002,
    depth_noise_scale_z=0.00015,
    missing_edge_px=3,
    min_depth_m=0.3,
    max_depth_m=10.0,
    seed=None,
)

INTEL_D435 = DepthCameraConfig(
    # Source: Intel RealSense D435 depth camera datasheet
    # Wide-baseline stereo, 30 fps.  Slightly higher noise than D415 at
    # long range but better close-range performance.
    name="INTEL_D435",
    update_rate_hz=30.0,
    resolution=(848, 480),
    depth_noise_sigma_m=0.003,
    depth_noise_scale_z=0.0002,
    missing_edge_px=2,
    min_depth_m=0.2,
    max_depth_m=10.0,
    seed=None,
)

INTEL_L515 = DepthCameraConfig(
    # Source: Intel RealSense L515 LiDAR-based depth camera datasheet
    # Time-of-flight, much lower range-dependent noise than structured light.
    # Range: 0.25 m – 9 m; accuracy < 0.5 % at 1 m.
    name="INTEL_L515",
    update_rate_hz=30.0,
    resolution=(1024, 768),
    depth_noise_sigma_m=0.001,
    depth_noise_scale_z=0.00005,
    missing_edge_px=2,
    min_depth_m=0.25,
    max_depth_m=9.0,
    seed=None,
)


# ---------------------------------------------------------------------------
# Tactile array presets
# ---------------------------------------------------------------------------

FINGERTIP_TACTILE_4X4 = TactileArrayConfig(
    name="FINGERTIP_TACTILE_4X4",
    update_rate_hz=200.0,
    resolution=(4, 4),
    max_pressure_pa=600_000.0,
    noise_sigma_pa=1000.0,
    contact_threshold_pa=5000.0,
    taxel_area_mm2=1.0,
    dead_zone_fraction=0.0,
    seed=None,
)

WRIST_TACTILE_8X2 = TactileArrayConfig(
    name="WRIST_TACTILE_8X2",
    update_rate_hz=100.0,
    resolution=(8, 2),
    max_pressure_pa=300_000.0,
    noise_sigma_pa=2000.0,
    contact_threshold_pa=8000.0,
    taxel_area_mm2=9.0,
    dead_zone_fraction=0.02,
    seed=None,
)

PALM_TACTILE_8X8 = TactileArrayConfig(
    name="PALM_TACTILE_8X8",
    update_rate_hz=50.0,
    resolution=(8, 8),
    max_pressure_pa=200_000.0,
    noise_sigma_pa=3000.0,
    contact_threshold_pa=10_000.0,
    taxel_area_mm2=25.0,
    dead_zone_fraction=0.05,
    seed=None,
)


# ---------------------------------------------------------------------------
# Current sensor presets
# ---------------------------------------------------------------------------

INA226_10A = CurrentSensorConfig(
    name="INA226_10A",
    update_rate_hz=500.0,
    noise_sigma_a=0.01,
    offset_a=0.0,
    range_a=10.0,
    voltage_nominal_v=24.0,
    seed=None,
)

MAUCH_HS_200 = CurrentSensorConfig(
    name="MAUCH_HS_200",
    update_rate_hz=100.0,
    noise_sigma_a=1.0,
    offset_a=0.0,
    range_a=200.0,
    voltage_nominal_v=51.8,
    seed=None,
)

ACS712_5A = CurrentSensorConfig(
    name="ACS712_5A",
    update_rate_hz=200.0,
    noise_sigma_a=0.1,
    offset_a=0.02,
    range_a=5.0,
    voltage_nominal_v=12.0,
    seed=None,
)


# ---------------------------------------------------------------------------
# RPM sensor presets
# ---------------------------------------------------------------------------

AS5048A_MAG_ENC = RPMSensorConfig(
    name="AS5048A_MAG_ENC",
    update_rate_hz=1000.0,
    cpr=16384,
    noise_sigma_rpm=1.0,
    rpm_range=25_000.0,
    seed=None,
)

T_MOTOR_HALL_6P = RPMSensorConfig(
    name="T_MOTOR_HALL_6P",
    update_rate_hz=500.0,
    cpr=36,
    noise_sigma_rpm=10.0,
    rpm_range=10_000.0,
    seed=None,
)

OPTICAL_ENC_1024 = RPMSensorConfig(
    name="OPTICAL_ENC_1024",
    update_rate_hz=1000.0,
    cpr=1024,
    noise_sigma_rpm=5.0,
    rpm_range=20_000.0,
    seed=None,
)


_REGISTRY: dict[str, PresetConfig] = {
    # Cameras
    "RASPBERRY_PI_V2": RASPBERRY_PI_V2,
    "INTEL_D435_RGB": INTEL_D435_RGB,
    "GOPRO_HERO11_4K30": GOPRO_HERO11_4K30,
    "ZED2_LEFT": ZED2_LEFT,
    "ZED2_RIGHT": ZED2_RIGHT,
    # Stereo cameras
    "ZED2_STEREO": ZED2_STEREO,
    "INTEL_D435_STEREO": INTEL_D435_STEREO,
    "MYNT_EYE_D_120": MYNT_EYE_D_120,
    # LiDAR
    "VELODYNE_VLP16": VELODYNE_VLP16,
    "VELODYNE_HDL64E": VELODYNE_HDL64E,
    "OUSTER_OS1_64": OUSTER_OS1_64,
    "LIVOX_AVIA": LIVOX_AVIA,
    "LIVOX_MID360": LIVOX_MID360,
    "HESAI_XT32": HESAI_XT32,
    "SICK_TIM571": SICK_TIM571,
    # IMU
    "PIXHAWK_ICM20689": PIXHAWK_ICM20689,
    "VECTORNAV_VN100": VECTORNAV_VN100,
    "XSENS_MTI_3": XSENS_MTI_3,
    "BOSCH_BMI088": BOSCH_BMI088,
    "INVENSENSE_MPU9250": INVENSENSE_MPU9250,
    # GNSS
    "UBLOX_M8N": UBLOX_M8N,
    "UBLOX_F9P_RTK": UBLOX_F9P_RTK,
    "NOVATEL_OEM7": NOVATEL_OEM7,
    "TRIMBLE_BD992": TRIMBLE_BD992,
    "EMLID_REACH_RS2": EMLID_REACH_RS2,
    # Thermal cameras
    "FLIR_LEPTON_35": FLIR_LEPTON_35,
    "FLIR_BOSON_320": FLIR_BOSON_320,
    "SEEK_THERMAL_COMPACT_PRO": SEEK_THERMAL_COMPACT_PRO,
    # Event cameras
    "DAVIS_346": DAVIS_346,
    "PROPHESEE_EVK4": PROPHESEE_EVK4,
    # Barometers
    "BMP388": BMP388,
    "MS5611": MS5611,
    "SPL06_001": SPL06_001,
    "BMP280": BMP280,
    "DPS310": DPS310,
    # Magnetometers
    "IST8310": IST8310,
    "HMC5883L": HMC5883L,
    "RM3100": RM3100,
    # Environmental sensors
    "DS18B20_PROBE": DS18B20_PROBE,
    "SHT31_HUMIDITY": SHT31_HUMIDITY,
    "TSL2591_LIGHT": TSL2591_LIGHT,
    "SGP30_AIR_QUALITY": SGP30_AIR_QUALITY,
    "DAVIS_6410_ANEMOMETER": DAVIS_6410_ANEMOMETER,
    # Wireless sensing
    "QORVO_DWM3001C": QORVO_DWM3001C,
    "TI_IWR6843AOP": TI_IWR6843AOP,
    "NAVTECH_CTS350X": NAVTECH_CTS350X,
    # Airspeed
    "SDP33": SDP33,
    "MS4525DO": MS4525DO,
    # Rangefinder
    "TFMINI_PLUS": TFMINI_PLUS,
    "TERARANGER_ONE": TERARANGER_ONE,
    "GARMIN_LIDAR_LITE_V3": GARMIN_LIDAR_LITE_V3,
    # Optical flow
    "PX4FLOW": PX4FLOW,
    "MATEKSYS_3901_L0X": MATEKSYS_3901_L0X,
    # Battery
    "LIPO_3S_2200MAH": LIPO_3S_2200MAH,
    "LIPO_4S_5000MAH": LIPO_4S_5000MAH,
    "LIPO_6S_10000MAH": LIPO_6S_10000MAH,
    # Wheel odometry
    "DIFF_DRIVE_ENCODER_50HZ": DIFF_DRIVE_ENCODER_50HZ,
    "MECANUM_DRIVE_ENCODER_100HZ": MECANUM_DRIVE_ENCODER_100HZ,
    # Force/torque
    "ATI_MINI45": ATI_MINI45,
    "ROKUBI_FT300": ROKUBI_FT300,
    "OPTOFORCE_OMD": OPTOFORCE_OMD,
    # Joint state
    "FRANKA_JOINT_ENCODER": FRANKA_JOINT_ENCODER,
    "UR5_JOINT_ENCODER": UR5_JOINT_ENCODER,
    "GENERIC_SERVO_ENCODER": GENERIC_SERVO_ENCODER,
    # Contact sensor
    "FINGERTIP_TACTILE_200HZ": FINGERTIP_TACTILE_200HZ,
    "BUMPER_50HZ": BUMPER_50HZ,
    # Depth cameras
    "INTEL_D415": INTEL_D415,
    "INTEL_D435": INTEL_D435,
    "INTEL_L515": INTEL_L515,
    # Tactile arrays
    "FINGERTIP_TACTILE_4X4": FINGERTIP_TACTILE_4X4,
    "WRIST_TACTILE_8X2": WRIST_TACTILE_8X2,
    "PALM_TACTILE_8X8": PALM_TACTILE_8X8,
    # Current sensors
    "INA226_10A": INA226_10A,
    "MAUCH_HS_200": MAUCH_HS_200,
    "ACS712_5A": ACS712_5A,
    # RPM sensors
    "AS5048A_MAG_ENC": AS5048A_MAG_ENC,
    "T_MOTOR_HALL_6P": T_MOTOR_HALL_6P,
    "OPTICAL_ENC_1024": OPTICAL_ENC_1024,
}

# Category → list of preset names; built dynamically from _REGISTRY config types
# so that adding a new preset only requires an entry in _REGISTRY.
_CONFIG_TYPE_TO_KIND: dict[type, str] = {
    CameraConfig: "camera",
    StereoCameraConfig: "stereo",
    LidarConfig: "lidar",
    IMUConfig: "imu",
    GNSSConfig: "gnss",
    ThermalCameraConfig: "thermal",
    EventCameraConfig: "event",
    BarometerConfig: "barometer",
    MagnetometerConfig: "magnetometer",
    ThermometerConfig: "thermometer",
    HygrometerConfig: "hygrometer",
    LightSensorConfig: "light_sensor",
    GasSensorConfig: "gas_sensor",
    AnemometerConfig: "anemometer",
    UWBRangeConfig: "uwb",
    RadarConfig: "radar",
    AirspeedConfig: "airspeed",
    RangefinderConfig: "rangefinder",
    OpticalFlowConfig: "optical_flow",
    BatteryConfig: "battery",
    WheelOdometryConfig: "wheel_odometry",
    ForceTorqueConfig: "force_torque",
    JointStateConfig: "joint_state",
    ContactSensorConfig: "contact",
    DepthCameraConfig: "depth_camera",
    TactileArrayConfig: "tactile_array",
    CurrentSensorConfig: "current",
    RPMSensorConfig: "rpm",
}

_PRESET_CATEGORIES: dict[str, list[str]] = {kind: [] for kind in _CONFIG_TYPE_TO_KIND.values()}
for _name, _cfg in _REGISTRY.items():
    _kind = _CONFIG_TYPE_TO_KIND.get(type(_cfg))
    if _kind is not None:
        _PRESET_CATEGORIES[_kind].append(_name)
del _name, _cfg, _kind  # clean up loop variables from module namespace


def list_presets(kind: str | None = None) -> list[str]:
    """
    Return a sorted list of all available preset names.

    Parameters
    ----------
    kind:
        Optional sensor-type filter.  Accepted values: ``"camera"``,
        ``"lidar"``, ``"imu"``, ``"gnss"``, ``"thermal"``, ``"event"``,
        ``"barometer"``, ``"magnetometer"``.  When *None* (default) all
        presets are returned.

    Raises
    ------
    KeyError
        If *kind* is not a recognised sensor category.

    Examples
    --------
    ::

        list_presets()                    # all presets, sorted
        list_presets(kind="lidar")        # ["HESAI_XT32", "LIVOX_AVIA", ...]
        list_presets(kind="gnss")         # ["EMLID_REACH_RS2", "NOVATEL_OEM7", ...]
        list_presets(kind="thermal")      # ["FLIR_BOSON_320", "FLIR_LEPTON_35", ...]
        list_presets(kind="barometer")    # ["BMP280", "BMP388", "DPS310", ...]
        list_presets(kind="magnetometer") # ["HMC5883L", "IST8310", "RM3100"]
        list_presets(kind="airspeed")     # ["MS4525DO", "SDP33"]
        list_presets(kind="rangefinder")  # ["GARMIN_LIDAR_LITE_V3", "TERARANGER_ONE", "TFMINI_PLUS"]
    """
    if kind is None:
        return sorted(_REGISTRY)
    key = kind.lower()
    if key not in _PRESET_CATEGORIES:
        available = ", ".join(sorted(_PRESET_CATEGORIES))
        raise KeyError(f"Unknown sensor kind {kind!r}.  Available kinds: {available}")
    return sorted(_PRESET_CATEGORIES[key])


def get_preset(name: str) -> PresetConfig:
    """
    Return a preset config by name (case-insensitive).

    Parameters
    ----------
    name:
        Preset identifier, e.g. ``"VELODYNE_VLP16"`` or ``"velodyne_vlp16"``.

    Returns
    -------
    PresetConfig
        One of :class:`~genesis.sensors.CameraConfig`,
        :class:`~genesis.sensors.LidarConfig`,
        :class:`~genesis.sensors.IMUConfig`,
        :class:`~genesis.sensors.GNSSConfig`,
        :class:`~genesis.sensors.ThermalCameraConfig`,
        :class:`~genesis.sensors.EventCameraConfig`,
        :class:`~genesis.sensors.BarometerConfig`,
        :class:`~genesis.sensors.MagnetometerConfig`,
        :class:`~genesis.sensors.AirspeedConfig`, or
        :class:`~genesis.sensors.RangefinderConfig`.

    Raises
    ------
    KeyError
        If *name* does not match any known preset.

    Examples
    --------
    ::

        from genesis.sensors.presets import get_preset
        cfg = get_preset("velodyne_vlp16")
        lidar = LidarModel.from_config(cfg)
    """
    key = name.upper()
    if key not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY))
        raise KeyError(f"Unknown sensor preset {name!r}.  Available: {available}")
    return _REGISTRY[key]


__all__ = [
    # Type alias
    "PresetConfig",
    # Camera presets
    "RASPBERRY_PI_V2",
    "INTEL_D435_RGB",
    "GOPRO_HERO11_4K30",
    "ZED2_LEFT",
    "ZED2_RIGHT",
    # Stereo camera presets
    "ZED2_STEREO",
    "INTEL_D435_STEREO",
    "MYNT_EYE_D_120",
    # LiDAR presets
    "VELODYNE_VLP16",
    "VELODYNE_HDL64E",
    "OUSTER_OS1_64",
    "LIVOX_AVIA",
    "LIVOX_MID360",
    "HESAI_XT32",
    "SICK_TIM571",
    # IMU presets
    "PIXHAWK_ICM20689",
    "VECTORNAV_VN100",
    "XSENS_MTI_3",
    "BOSCH_BMI088",
    "INVENSENSE_MPU9250",
    # GNSS presets
    "UBLOX_M8N",
    "UBLOX_F9P_RTK",
    "NOVATEL_OEM7",
    "TRIMBLE_BD992",
    "EMLID_REACH_RS2",
    # Thermal presets
    "FLIR_LEPTON_35",
    "FLIR_BOSON_320",
    "SEEK_THERMAL_COMPACT_PRO",
    # Event camera presets
    "DAVIS_346",
    "PROPHESEE_EVK4",
    # Barometer presets
    "BMP388",
    "MS5611",
    "SPL06_001",
    "BMP280",
    "DPS310",
    # Environmental presets
    "DS18B20_PROBE",
    "SHT31_HUMIDITY",
    "TSL2591_LIGHT",
    "SGP30_AIR_QUALITY",
    "DAVIS_6410_ANEMOMETER",
    # Wireless presets
    "QORVO_DWM3001C",
    "TI_IWR6843AOP",
    "NAVTECH_CTS350X",
    # Magnetometer presets
    "IST8310",
    "HMC5883L",
    "RM3100",
    # Airspeed presets
    "SDP33",
    "MS4525DO",
    # Rangefinder presets
    "TFMINI_PLUS",
    "TERARANGER_ONE",
    "GARMIN_LIDAR_LITE_V3",
    # Optical flow presets
    "PX4FLOW",
    "MATEKSYS_3901_L0X",
    # Battery presets
    "LIPO_3S_2200MAH",
    "LIPO_4S_5000MAH",
    "LIPO_6S_10000MAH",
    # Wheel odometry presets
    "DIFF_DRIVE_ENCODER_50HZ",
    "MECANUM_DRIVE_ENCODER_100HZ",
    # Force/torque presets
    "ATI_MINI45",
    "ROKUBI_FT300",
    "OPTOFORCE_OMD",
    # Joint state presets
    "FRANKA_JOINT_ENCODER",
    "UR5_JOINT_ENCODER",
    "GENERIC_SERVO_ENCODER",
    # Contact sensor presets
    "FINGERTIP_TACTILE_200HZ",
    "BUMPER_50HZ",
    # Depth camera presets
    "INTEL_D415",
    "INTEL_D435",
    "INTEL_L515",
    # Tactile array presets
    "FINGERTIP_TACTILE_4X4",
    "WRIST_TACTILE_8X2",
    "PALM_TACTILE_8X8",
    # Current sensor presets
    "INA226_10A",
    "MAUCH_HS_200",
    "ACS712_5A",
    # RPM sensor presets
    "AS5048A_MAG_ENC",
    "T_MOTOR_HALL_6P",
    "OPTICAL_ENC_1024",
    # Helpers
    "get_preset",
    "list_presets",
]
