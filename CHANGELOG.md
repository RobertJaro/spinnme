# Changelog

## 0.1.0

Initial public release of SPINNME for spherical neural Milne–Eddington inversions.

- Install with `pip install spinnme`; optional extras provide archive downloads,
  synthetic data generation and Weights & Biases logging.
- Run HMI, Hinode/SP, PHI or custom calibrated Stokes FITS observations using
  packaged instrument definitions and YAML run configurations.
- Integrate a nominal single-order Airy transmission profile in PHI HRT/FDT
  presets, with configurable quadrature and measured-profile overrides.
- Crop FITS images and cubes from the CLI using world or pixel bounds, preserving
  native pixels, spatial WCS and aligned image extensions. HMI subframes retain
  transmission-calibration detector coordinates.
- Jointly fit multiple instruments and spectral lines, including both Hinode
  Fe I 6301 and 6302 lines, with shared magnetic and velocity fields.
- Download HMI Stokes data and transmission profiles through separate CLI calls;
  prepare Hinode rasters with explicit WCS and per-slit acquisition times.
  Observer-velocity keywords are optional; available components supply the Doppler correction.
- Train with flexible wavelength grids, optional polarization oversampling,
  instrument corrections and portable `.pme` inference snapshots.
- Export explicitly selected parameters, including `Brtp`, `Blos` and `Vdop`,
  as FITS with WCS, NPZ or JPG through the shared CLI and Python API. Native
  raster exports preserve per-pixel acquisition offsets.
- Prepare saved, FITS or SunPy observing grids through the snapshot loader;
  crop/resize masks, scan times and detector calibration together. Evaluate
  physical SkyCoord samples, obtain SunPy maps and export `Vrtp`/`Vlos` using
  the same parameter and geometry interface.
- Include circular-pattern examples and scientific regression tests.
- Provide reusable masked comparison metrics, wrapped angular residuals, map
  comparison panels, density scatter plots and publication figure export.
  Dataset/study adapters and reference conventions remain in the evaluation project.
- Provide Sphinx documentation for installation, end-to-end inversions,
  training customization, new instruments, corrections and Python exports.

This release assumes correctly calibrated observations and valid WCS metadata.
Azimuth disambiguation and Cartesian inversions are outside its scope.
