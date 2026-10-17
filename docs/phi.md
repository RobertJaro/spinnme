# Invert Solar Orbiter PHI data

Use the [Colab notebook](colab.md) for an online download-to-export HRT/FDT walkthrough.

Use calibrated **L2 full-Stokes** HRT or FDT products. Check the selected release's WCS, wavelength and polarization conventions in the [PHI data guide](https://www.mps.mpg.de/solar-physics/solar-orbiter-phi/data) and [release notes](https://www.mps.mpg.de/solar-physics/solar-orbiter-phi/data-releases).

## Use local Stokes files

```bash
spinnme run --config phi/hrt --input './data/phi-hrt/*stokes*.fits' --run_path ./runs/phi-hrt --export fits
```

For FDT, use `--config phi/fdt` and your FDT input/output paths. Each preset connects its dataset to the appropriate instrument ID.

The reader supports the standard NumPy order `(wavelength, Stokes, y, x)` and calibrated `WAVELNnn`, `WAVELNTH` and `WAVEUNIT` cards. For a different product layout, copy the preset and set the actual axes, HDUs and wavelengths in both training and validation entries. The [custom-data guide](custom.md) shows those options.

Older releases can contain missing wavelengths or spatial WCS cards assigned to the wrong axes. Reprocess these files with verified calibration and WCS before running; the reader rejects them rather than guessing repairs.

Check observer position and acquisition time as well as spatial WCS. Observer velocity is optional; available components provide the Doppler correction described under [input metadata](data.md). The [PHI WCS tutorial](https://solar-orbiter-docs.readthedocs.io/en/latest/auto_gallery/phi_wcs.html) helps inspect the map geometry.

## Download Stokes products

With the `download` extra installed:

```bash
spinnme download phi --start 2024-03-23T00:00:00 --end 2024-03-24T00:00:00 --telescope hrt --product stokes --output ./data/phi-hrt
spinnme download phi --start 2024-03-23T00:00:00 --end 2024-03-24T00:00:00 --telescope fdt --product stokes --output ./data/phi-fdt
```

For a runnable HRT sequence example, see [phi_hrt.sh](https://github.com/RobertJaro/spinnme/blob/main/examples/phi_hrt.sh), which selects March 23 at 22:00 UTC through March 24 at 03:00 UTC.

These commands query the archive for that interval. Choose dates with public products for your telescope.

## Set wavelengths, continuum and time

The bundled HRT/FDT presets and instrument definitions integrate a **nominal Airy spectral response** at every calibrated tuning wavelength, including the near-continuum channel:

```yaml
instrument:
  - type: phi
    instrument_id: PHI_HRT  # PHI_FDT for the FDT preset
    line_id: fe-i-6173
    spectral_response:
      type: airy
      unit: Angstrom
      fwhm: 0.106
      free_spectral_range: 3.01
      samples: 161
```

HRT and FDT share the filtergraph. The nominal values follow the [SO/PHI instrument paper](https://doi.org/10.1051/0004-6361/201935325): the measured etalon free spectral range is 0.301 nm, and Figure 37 fits ground-calibration scans with an Airy profile of 106 mÅ FWHM. This is an idealized, spatially invariant response, **not an acquisition-specific calibration**.

The response is `T(offset) = 1 / (1 + F sin²(π offset / FSR))`, with `F = 1 / sin²(π FWHM / (2 FSR))`. Integration covers one order, `[-FSR/2, +FSR/2]`, assuming adjacent orders are blocked. The quadrature retains the wings and normalizes the integral so a constant spectrum stays constant. It uses Gauss–Legendre nodes in a sinh coordinate to resolve the narrow peak and broad wings. All four synthesized Stokes components are integrated before normalization and loss evaluation; observed spectra and calibrated tuning wavelengths are unchanged.

This approximation omits residual prefilter shape, adjacent-order leakage, spatial cavity shifts, and acquisition-dependent passband changes. It does not reverse the L2 prefilter calibration or model a spatial PSF. If a verified effective response is available, replace `spectral_response` with a `tabulated` profile whose weights include quadrature spacing; see [spectral responses](custom.md#add-a-spectral-response). Avoid applying a prefilter correction twice.

Increase `samples` to 321 to check numerical convergence for your spectra, especially narrow lines, large Doppler shifts, or strong Zeeman splitting. Finite-response synthesis costs more memory and time; reduce `data.batch_size` (and keep it divisible by `data.dataset_batch_size`) if necessary. For a monochromatic comparison, explicitly set `spectral_response: {type: monochromatic}`. Existing snapshots and custom instrument entries without a response retain their saved/monochromatic behavior; the nominal response is enabled explicitly by the presets.

The continuum index comes from `CONTPOS` when present, using its **one-based** position. YAML `continuum_index` is **zero-based**. If metadata are incomplete, set the verified position explicitly. The [PHI noise tutorial](https://solar-orbiter-docs.readthedocs.io/en/latest/auto_gallery/get_phi_noise_levels.html) shows continuum-based noise estimates.

The default model time is `DATE-OBS`. Set `time_reference: solar` or `earth` in both dataset entries to use `DATE_SUN` or `DATE_EAR`. Spatial geometry retains the native observation time; choose a consistent time policy when combining observers.

## Export and inspect corrections

```bash
spinnme export ./runs/phi-hrt/inversion.pme --all-observations --parameters Brtp Blos Vdop vmac damping --format fits --output ./runs/phi-hrt/products
```

Artifact and limb corrections are off by default. Enable [artifact correction](artifacts.md) as a separate run when testing residual intensity leakage. Archive field/velocity products can be used as registered [evaluation references](evaluation.md), with their release's units and sign conventions.
