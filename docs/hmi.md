# Invert HMI data

Follow the [quickstart](quickstart.md) for the complete download → transmission → run → export workflow. The bundled `hmi/full_disk` YAML accepts native `hmi.S_720s` or `hmi.S_90s` Stokes observations.

## Directory layout and transmission profiles

The bundled run expects `stokes/` and `transmission/` under the directory passed as `--data_path`. Keep all **24 segments per acquisition**, named `.I0.fits` through `.V5.fits`. SPINNME groups them by acquisition prefix and checks their grids and quality flags.

For downloaded Stokes files:

```bash
spinnme download hmi-transmission --input ./data/hmi/stokes \
  --output ./data/hmi/transmission --email "$JSOC_EMAIL"
spinnme run --config hmi/full_disk --data_path ./data/hmi --run_path ./runs/hmi
```

The transmission command uses each acquisition's calibration identity and camera. It writes the required profiles and a manifest without reading a training config. It accepts multiple input directories and reuses existing profiles; use `--overwrite` to regenerate them.

## Observed wavelengths versus transmission sampling

HMI supplies I/Q/U/V at **six tuning positions**, giving 24 Stokes segments per acquisition. SPINNME uses Fe I 6173.3433 Å and these nominal offsets, in segment order `0` through `5`:

```text
[+0.1720, +0.1032, +0.0344, -0.0344, -0.1032, -0.1720] Å
```

Segment 0 is the redmost position and segment 5 the bluemost. The nominal separation is 0.0688 Å. Each value is an observation through a finite passband, rather than a monochromatic intensity at its nominal wavelength. The [HMI observables paper](https://arxiv.org/abs/1606.02368) describes the six-wavelength observations and filter system.

The forward model synthesizes finer wavelength samples, integrates them with each passband's weights, and compares the resulting **six** values with the observations. Increasing numerical integration sampling does not create extra observed wavelengths. Keep all six tuning positions; changing YAML `wavelength_grid` or deleting segments is not a supported way to change HMI spectral sampling. Dataset `resolution` changes the **spatial image grid** only.

### How transmission profiles are constructed

`download hmi-transmission` identifies the definitive inversion phase-map record and camera for each acquisition. It downloads the phase maps and reconstructs the six passbands using the narrow-band and wide-band Michelsons, the tunable Lyot element, the fixed elements and a blocking-filter model. Acquisitions sharing one calibration/camera reuse one profile file. Training samples the spatial response maps at each pixel's detector location.

The reconstructed profiles use mean element contrasts and a Gaussian blocking-filter model with FWHM 8.43 Å. They are an approximation, not a download of six complete measured transmission curves; individual fixed-element phase variations, front-window fine structure and intensity ripple are omitted. The optical filter design is described in the [HMI instrument paper](https://ntrs.nasa.gov/api/citations/20140007405/downloads/20140007405.pdf).

By default, the solver uses **81 Gauss–Legendre nodes per filter** over the common line interval **−0.65 to +0.65 Å** relative to the line center. These are nonuniform numerical nodes, not an 81-point uniformly spaced observed spectrum. Stored offsets are relative to each filter's nominal tuning position, so all six integrations cover the same absolute line interval.

Throughput outside that interval is integrated separately and represented by `continuum_weights`: it contributes unpolarized continuum to I, with zero Q/U/V. Line weights plus continuum weight sum to one for each detector position and filter. Widen the explicit line interval if significant Doppler-shifted or broadened line signal reaches its edges; adding more nodes within an interval does not widen it.

### Increase or decrease numerical sampling

The CLI uses the defaults. For custom numerical settings, use the preparation function independently of the run YAML:

```python
import os
from spinnme.data.hmi_transmission import prepare_hmi_transmission_directory

prepare_hmi_transmission_directory(
    inputs=["./data/hmi/stokes"],
    output_directory="./data/hmi/transmission-161",
    email=os.environ["JSOC_EMAIL"],
    samples=161,       # Default 81; try 41 for a cheaper convergence check.
    half_width=0.65,   # Å about the line center, not about each tuning position.
)
```

| Setting | Default | Increase accuracy/coverage | Reduce cost |
| --- | --- | --- | --- |
| `samples` | 81 | Try 161 integration nodes per filter | Try 41; minimum accepted is 9 |
| `half_width` | 0.65 Å | Widen the explicitly synthesized line interval; increase `samples` too | Narrow only if the line remains inside it |
| `continuum_panel_width` | 0.1 Å | Decrease panel width, e.g. 0.05, for the outer throughput integral | Increase width, e.g. 0.2, after checking convergence |
| `continuum_panel_nodes` | 8 | Increase nodes per outer panel, e.g. 12 | Decrease, e.g. 4; minimum accepted is 2 |
| `blocker_tail_sigma` | 7 | Increase the outer blocking-filter integration extent | Decrease only after checking omitted throughput; minimum accepted is 3 |

Outer-panel settings affect profile preparation. `samples` also affects synthesis cost and memory during training/inference, approximately in proportion to the node count at a fixed batch size. `blocker_fwhm` and element contrasts describe the optical model; they are not numerical accuracy controls.

Use a separate profile directory for each sampling experiment. Existing files are reused regardless of new numerical arguments unless `overwrite=True` is supplied. To train with the new files:

```bash
spinnme config copy hmi/full_disk --output hmi-161.yaml
# In BOTH training and validation entries, set:
# transmission_profile_directory: ./data/hmi/transmission-161
spinnme run --config hmi-161.yaml --data_path ./data/hmi \
  --run_path ./runs/hmi-161
```

Keep one numerical sampling setup within each instrument ID; training and validation must use compatible response shapes. Compare synthesized Stokes profiles at fixed atmosphere parameters while increasing node count and interval width, aiming for changes below your noise/error budget. Smaller counts are a speed experiment, not an accuracy guarantee. Use a fresh run/cache for a changed setup; `run --reload` rebuilds an existing preparation cache after in-place calibration changes. Snapshots bundle the profiles used during training, so regenerating files does not change an existing `.pme` export.

## Download a collection

```bash
spinnme download hmi --product S_720s \
  --start 2024-03-23T01:00:00 --end 2024-03-23T03:00:00 --time-scale tai \
  --email "$JSOC_EMAIL" --output ./data/hmi/stokes
```

Use `--product S_90s` for 90-second data and a separate data directory. Omit `--end` to request one record. Download transmission profiles for that collection before running the inversion.

## Invert an HMI subframe

Use the generic `spinnme crop` command after downloading the Stokes segments. Supply all 24 segments per acquisition, as required by the HMI loader. Cropping processes every matching file independently and writes to a **new or empty** output directory; it does not validate Stokes group completeness.

For helioprojective bounds, use **XMIN YMIN XMAX YMAX in arcseconds**:

```bash
spinnme crop --input ./data/hmi/stokes \
  --output ./data/hmi-subframe/stokes --bounds -250 -200 -100 -50
spinnme download hmi-transmission --input ./data/hmi-subframe/stokes \
  --output ./data/hmi-subframe/transmission --email "$JSOC_EMAIL"
spinnme run --config hmi/subframe --data_path ./data/hmi-subframe \
  --run_path ./runs/hmi-subframe
spinnme export ./runs/hmi-subframe/inversion.pme --all-observations \
  --parameters Brtp Blos Vdop --format fits --output ./products/hmi-subframe
```

The executable [hmi_subframe.sh](https://github.com/RobertJaro/spinnme/blob/main/examples/hmi_subframe.sh) includes the initial HMI download too. Use the same commands for native 90 s data downloaded with `--product S_90s`.

For detector-image pixel bounds, use **X0 Y0 X1 Y1**, zero-based with the upper bounds excluded:

```bash
spinnme crop --input './data/hmi/stokes/*.fits' \
  --output ./data/hmi-pixel-subframe/stokes --pixels 1792 1792 2304 2304
```

This retains a 512 × 512 pixel region. Pixel bounds refer to the **input image**, including when cropping an existing subframe. The command accumulates the full-detector offset correctly. Existing cutouts must already contain valid `CCD_X0` and `CCD_Y0`; native 4096 × 4096 inputs start at detector origin `[0, 0]`.

Sky bounds are converted using each file's own WCS, including its rotation. The output is an enclosing native-pixel rectangle, so edges can extend slightly beyond the requested sky rectangle; parts outside the input image are clipped. A rectangle entirely outside any selected image is rejected. This is a fixed helioprojective region, not a feature-tracking or solar-rotation tracking operation.

The output preserves wavelength/Stokes segment names, `T_REC`, camera, quality and observer metadata. It updates spatial WCS and the zero-based `CCD_X0`/`CCD_Y0` cards used to sample transmission profiles at the original detector coordinates. It does not spatially resample pixels. Cropping a downsampled image is not supported by this native-detector workflow; crop the original segments first.

`hmi/subframe` keeps the cropped spatial sampling; `hmi/full_disk` instead resamples to the starter's 256 × 256 grid. To change sampling or training, copy `hmi/subframe` to your own YAML and edit it. Existing acquisition-matched transmission profiles can also be reused by setting `transmission_profile_directory` in both dataset entries to their directory; cropping does not change the optical calibration identity.

Source files and HDU layout are preserved. The same command supports other instruments and FITS cubes; see [FITS cropping](cropping.md) for axis selection and extension handling. For a subregion of an already trained snapshot, use [Python exports](python_exports.md) instead of preparing another inversion.

## Use custom settings

Copy `hmi/full_disk` to your own YAML. Edit `resolution`, paths, validation selection and training settings there; see [training options](training_options.md). The bundled starter uses 256 × 256 sampling and 1,000 steps. HMI dataset `resolution` uses `[nx, ny]` (SunPy ordering), while Python export `shape` uses `(ny, nx)`. This interpolates Stokes images and WCS; it is not photon-conserving binning, an instrument PSF model or a noise recalibration. Remove `resolution` from both datasets for native spatial sampling. For another directory layout, set both datasets' `data_path` and `transmission_profile_directory` explicitly.

## Export results

```bash
spinnme export ./runs/hmi/inversion.pme --all-observations \
  --parameters Brtp Blos Vdop vmac damping --format fits --output ./products/hmi
```

Each parameter gets its own FITS file with the selected grid's WCS. See [exports](exports.md) for NPZ, JPG and other products.

## Observer velocity, timing and cutouts

HMI normally supplies `OBS_VR`, `OBS_VW` and `OBS_VN` in m/s. SPINNME reads the available components and projects them along each pixel's line of sight. Missing components contribute zero; if none are available, the inversion runs without observer-motion correction. Spatial WCS, observer position and observation time are still used normally. This is separate from the optional learned `instrument.velocity_correction`.

HMI `DATE-OBS` sets the model epoch and observing geometry; `T_REC` identifies the archive/calibration record. `--time-scale tai` converts archive query times correctly: 2024-03-23 01:00:00 TAI is 00:59:23 UTC.

Cutouts must retain updated WCS and full-detector origin cards `CCD_X0`/`CCD_Y0`, because transmission sampling uses detector position. A native 90-second versus 720-second comparison also changes temporal integration and noise; [evaluation](evaluation.md) covers comparison choices.


## Detector orientation and magnetic azimuth

HMI training keeps Stokes Q/U in their supplied detector reference frame.
The geometry uses the FITS PC/CD rotation matrix, or CROTA2 when no matrix is
present, to project the Carrington-frame magnetic field onto detector x/y
and the LOS. Detector +y must not be assumed to point toward solar north.
HMI defines magnetic azimuth counterclockwise from detector +y; see
[Hoeksema et al. (2014), Section 6.1 and Table 6](https://arxiv.org/pdf/1404.1881).

The reported angle is `atan2(-B_detector_x, B_detector_y)`, equivalent modulo
180 degrees for Stokes inversion. The HMI atmosphere applies its fixed
90-degree reference conversion once, by negating both double-angle products.
This is separate from the FITS detector roll. Do not rotate the input Q/U
again when using native HMI segments. Cropping and the supported spatial
resampling retain this orientation. Arbitrarily reprojected/derotated Stokes
images require consistent treatment of the polarization reference frame.

Regression tests cover PC, CD and CROTA2 headers at 0, 23, -37, 90 and 179.9
degrees, checking the detector components, azimuth and all four synthesized
Stokes profiles against an independent detector-component calculation.
