# Invert custom FITS and add instruments

The [custom FITS Colab notebook](colab.md) provides an online upload, configuration, inversion and export example.

## Use the generic preset

For calibrated **Fe I 6173** data shaped `(wavelength, Stokes, y, x)`, with complete solar WCS and `WAVELNnn`/`WAVELNTH` wavelength metadata:

```bash
spinnme run --config generic/stokes_cube --input './data/custom/*.fits' --output ./runs/custom --export fits
```

If your format needs explicit settings, copy `generic/stokes_cube` with `spinnme config copy generic/stokes_cube --output custom.yaml`, then edit both training and validation entries.

To crop images or cubes before inversion, use [FITS cropping](cropping.md). The crop command preserves spectral samples and updates spatial WCS.

## Configure axes and wavelengths

Save the following as `custom.yaml`, replacing the copied template. It describes six wavelength offsets around Fe I 6173.3433 Å. Replace the samples, axis order, intensity normalization and paths with values for your calibrated data. The YAML anchor keeps training and validation reader settings identical.

```yaml
path: "<<run_path>>"
work_path: "<<work_path>>"
data:
  train_configs:
    - &observation
      type: fits
      data_path: ./data/custom/*.fits
      instrument_id: CUSTOM
      line_id: fe-i-6173
      data_hdu: 0
      axis_order: [wavelength, stokes, y, x]
      stokes_order: IQUV
      wavelength_center: 6173.3433
      wavelength_grid: [-0.17, -0.10, -0.03, 0.03, 0.10, 0.17]
      wavelength_unit: Angstrom
      wavelength_grid_is_offsets: true
      stokes_normalization: 1.0
      time_reference: observation
  valid_config:
    <<: *observation
    sample_idx: 0
  batch_size: 4096
  dataset_batch_size: 512
  num_workers: 0
  num_dataset_workers: 1
  seconds_per_dt: 3600
  gauss_per_dB: 1000
instrument:
  - type: generic
    instrument_id: CUSTOM
    line_id: fe-i-6173
model:
  type: mlp
  dim: 128
  n_layers: 4
training:
  epochs: -1
  max_steps: 10000
  optimizer_config: {type: adamw}
stokes_loss:
  type: mse
  continuum_normalization: true
  continuum_quantile: 0.9
logging:
  mode: disabled
```

Run it with:

```bash
spinnme config validate custom.yaml
spinnme run --config custom.yaml --run_path ./runs/custom --export fits
```

`axis_order` describes the NumPy array, not FITS axis numbering. `stokes_order` describes the stored components. Optional `stokes_signs: [1,1,1,-1]` reverses V after reordering to I/Q/U/V; use the signs required by your calibration.

Choose one wavelength source:

- YAML `wavelength_grid`, in `wavelength_unit`; set `wavelength_grid_is_offsets: false` for absolute wavelengths.
- `wavelength_hdu`, identifying an array HDU, with the same unit/offset settings.
- FITS `WAVELN01`, `WAVELN02`, … for absolute wavelengths, plus `WAVELNTH` and `WAVEUNIT`. Leave YAML `wavelength_grid` and `wavelength_center` unset to read them.

Use `spatial_header_hdu` when the spatial WCS is in another HDU. `spatial_wcs_axes` selects its two FITS spatial axes in x/y order. Missing observer metadata can be supplied through a verified `observer` mapping; [data requirements](data.md) lists the required information. The helper in `examples/write_custom_fits.py` writes existing calibrated arrays and a supplied spatial header into the generic cube format.

## Add a spectral response

Instrument entries without `spectral_response` use monochromatic sampling. The bundled [PHI presets](phi.md#set-wavelengths-continuum-and-time) explicitly configure a nominal Airy response. A finite Gaussian response can be added to a generic instrument entry:

```yaml
instrument:
  - type: generic
    instrument_id: CUSTOM
    line_id: fe-i-6173
    spectral_response:
      type: gaussian
      fwhm: 0.08  # Example only: use your calibrated filter width.
      unit: Angstrom
      samples: 9
```

A scalar FWHM applies to every tuning wavelength; a list supplies one width per wavelength. Increase `samples` to check quadrature convergence.

For a measured response, use a global profile or one profile per filter:

```yaml
spectral_response:
  type: tabulated
  unit: Angstrom
  offsets: [-0.04, 0.0, 0.04]
  weights: [0.25, 0.5, 0.25]
```

These illustrative weights include quadrature spacing. Offsets are relative to each tuning wavelength; two-dimensional arrays have shape `[filter, quadrature]`. Weights must be nonnegative and are normalized per filter. Optional `continuum_weights` adds an unpolarized continuum contribution. HMI uses its acquisition/detector calibration instead of this generic response.

## Combine datasets and instruments

Start with `spinnme config copy generic/multiple_lines --output multiple-lines.yaml` for a two-line example. Each dataset needs a matching `instrument_id`; use separate IDs for different channel counts. Multiple acquisitions may use different calibrated wavelength positions while keeping a compatible number of channels under one ID.

The magnetic and velocity fields are shared on one modeled surface; joint fitting assumes that field can describe the configured lines/observers, without resolving their formation heights. Datasets with the same `line_id` share line-specific atmosphere parameters; different lines get separate atmosphere outputs. Configure `training.instrument_weights` when you need an explicit contribution from each instrument.

The bundled two-line preset exposes separate input keys so one file is not silently used for different lines:

```bash
spinnme run --config generic/multiple_lines \
  --line6173_path './data/6173/*.fits' --line6302_path './data/6302/*.fits' \
  --run_path ./runs/joint
```

Use a custom YAML for other layouts or line choices. `--input` replaces the first training entry, additional unset entries and the validation input; named or explicit paths for other datasets remain independent.

## Define another atomic line

Bundled line IDs are `fe-i-6173`, `fe-i-6301` and `fe-i-6302`. For another line, set its rest wavelength in the datasets and provide an instrument definition with a unique `line_id` and verified atomic parameters:

```yaml
instrument:
  - type: generic
    instrument_id: CUSTOM_LINE
    line_id: my-line
    atomic_parameters:
      j_up: 1.0
      j_low: 0.0
      g_up: 2.5
      g_low: 0.0
```

The values above illustrate the schema; replace them with the line's actual quantum numbers and Landé factors. Match `instrument_id` and `line_id` in the datasets. IDs must not contain dots.

Reusable definitions can be loaded with `--instrument instrument.yaml` or a bundled name (`hmi`, `hinode`, `phi_hrt`, `phi_fdt`). This applies the forward model and reader defaults; select the dataset adapter appropriate for your files.
