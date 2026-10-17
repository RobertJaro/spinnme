# Invert Hinode SP data

An [online Colab notebook](colab.md) covers a prepared raster, both spectral lines, fit checks and exports.

Use calibrated Stokes spectra rather than pipeline magnetic-parameter maps as training inputs. The `hinode/sp` preset jointly fits Fe I **6301.5012 Å and 6302.4931 Å** from the same raster. Both lines constrain the same magnetic field and velocity; each has its own Milne–Eddington thermodynamic and source-function parameters.

## Start with a prepared raster

For a raster with NumPy order `(Stokes, wavelength, y, x)`, correct solar WCS, calibrated wavelengths and a `TIME_OFFSETS` extension:

```bash
spinnme run --config hinode/sp --input ./data/hinode-raster.fits --run_path ./runs/hinode --export fits
```

`TIME_OFFSETS` must be a finite two-dimensional `(ny, nx)` image of seconds from the raster header's `DATE-OBS`. Broadcast per-slit timestamps across the slit's spatial pixels before writing this extension; the preparation helper below does this automatically. Keep `time_reference: observation` for these DATE-OBS-relative offsets. If choosing a different model-time reference, rebase the offsets to that epoch first; the reader adds them to the selected model time without interpreting a separate DATEREF. The forward model uses those acquisition times. Wavelengths can come from `WAVELNnn`/`WAVELNTH`, a wavelength HDU or YAML settings. For other cube layouts, edit `axis_order` in both training and validation entries; see [custom FITS](custom.md).

## Crop a prepared raster

```bash
spinnme crop --input ./data/hinode/raster.fits --output ./data/hinode/subframe \
  --pixels 20 30 120 150
```

For the prepared raster above, the default FITS spatial axes `1 2` correspond to x/y. All Stokes components and wavelengths are retained, and the aligned `TIME_OFFSETS` image is cropped with the science data. Run the Hinode preset on the resulting raster. For other axis layouts, see [FITS cropping](cropping.md).

## Assemble calibrated slit files

The preparation helper accepts slit arrays shaped `(Stokes, y, wavelength)` or `(Stokes, 1, y, wavelength)`. Supply a FITS image/header containing the **verified WCS for the assembled raster**, including observer metadata and `DATE-OBS`.

Retain the full calibrated spectrum containing both lines.

Use the Python helper to write a calibrated raster, then pass that file to the run command above. It orders slits by `SLITINDX` when available, preserves the signed spectral increment and writes per-pixel scan times.

```python
from pathlib import Path
from astropy.io import fits
from spinnme.data.prepare_hinode import prepare_hinode_raster

prepare_hinode_raster(
    sorted(Path("data/hinode-slits").glob("*.fits")),
    "data/hinode-raster.fits",
    fits.getheader("data/verified-raster-wcs.fits"),
    data_hdu=0,
    stokes_order="IQUV",
    wavelength_unit="Angstrom",
)
```

## Measured wavelength sampling

Hinode/SP is a spectrograph: the raster contains a measured I/Q/U/V spectrum at each spatial pixel. Its nominal spectral sampling is about **0.0215 Å (21.5 mÅ)**, but SPINNME uses the calibration in your files rather than assuming that spacing. See the [Hinode/SP instrument description](https://www2.hao.ucar.edu/csac/csac-instruments/sp).

For calibrated slit files, `prepare_hinode_raster` constructs wavelengths as `CRVAL1 + (pixel + 1 - CRPIX1) * CDELT1`, converts the supplied spectral unit, and writes the absolute wavelengths to `WAVELNnn`. Negative dispersion is preserved, together with the corresponding Stokes ordering. All assembled slits must have matching wavelength grids; resample differing grids explicitly before assembly.

For prepared rasters, the dataset can read `WAVELNnn`, a `wavelength_hdu`, or a YAML `wavelength_grid`. A configured grid must have exactly one value per input spectral channel. With `wavelength_grid_is_offsets: true`, values are offsets from that dataset's `wavelength_center`; with `false`, they are absolute wavelengths. `wavelength_unit` controls configured values, while `WAVEUNIT` controls FITS-header wavelengths. Neither a finer declared grid nor a denser exported spatial grid creates additional measured spectral information.

### Retain fewer or more observed samples

The default preparation keeps **every supplied spectral pixel**. To retain every second measured sample when assembling slits, add this argument to the preparation example above:

```python
wavelength_slice=(None, None, 2),  # Python start, stop, step; default keeps all samples.
```

The helper slices all four Stokes components and their calibrated wavelengths together; it preserves spatial WCS and scan timing. This doubles the spacing between retained samples and roughly halves the spectral channel count. It selects samples without averaging them, so it does not provide the noise reduction of spectral binning. Use `(None, None, 3)` for every third sample, or restore `wavelength_slice=None` to retain the original grid. Always regenerate from the original slit files into a new raster when changing this setting.

To use more observed samples, retain the original pixels or widen the fitted windows below. Interpolating a coarse raster to a finer wavelength grid adds correlated values, not independent constraints. If you perform actual spectral binning or resampling, update the wavelengths, uncertainty assumptions and response model consistently before ingestion.

## Select spectral windows or fit one line

The preset reads the raster twice with separate line identifiers, `HINODE_6301` and `HINODE_6302`. It splits the calibrated spectrum at 6301.99715 Å, halfway between the rest wavelengths. Samples below this boundary fit 6301; samples at or above it fit 6302. This works for either direction of spectral dispersion and unequal sample counts. Validation uses the 6302 window; its scalar loss and images do not assess the 6301 fit. Inspect synthesized profiles for both line IDs, or use a separate diagnostic run with validation set to the other matching line.

To change the windows, copy the run YAML and edit `wavelength_range: [lower, upper]` in its dataset entries. Bounds are **absolute wavelengths** in `wavelength_unit`, with the upper bound excluded; `null` leaves that side open. Keep each window within the intended line and surrounding continuum. The input must contain both lines; an empty window raises an error.

For a single-line inversion, retain only that line's training dataset and instrument entry, and make validation use the same line. The standalone `--instrument hinode` definition describes only the 6302 channel; use `--config hinode/sp` for the joint inversion. The [multiple-line template](custom.md#combine-datasets-and-instruments) also shows how to combine different instruments.

### Narrow or widen the two fitted windows

A narrower `wavelength_range` reduces the number of fitted spectral pixels without changing their native spacing. A wider range includes more measured wing/continuum samples. For example, to fit ±0.30 Å around each line, edit these fields in your copied run YAML:

| Dataset | `wavelength_center` | `wavelength_range` |
| --- | --- | --- |
| Training `HINODE_6301` | 6301.5012 | `[6301.2012, 6301.8012]` |
| Training `HINODE_6302` | 6302.4931 | `[6302.1931, 6302.7931]` |
| Validation `HINODE_6302` | 6302.4931 | `[6302.1931, 6302.7931]` |

The [MERLIN product description](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description) uses ±0.30 Å fitting windows; this is an example window choice, not SPINNME's default. Retain enough line wings for your velocities and field strengths. For continuum-normalized training, the intensity quantile is estimated from the retained samples, so an overly narrow window changes that proxy. Increasing a window beyond the observed spectrum adds no samples. Avoid including the other Fe I line in a single line's window: the two atmospheres are evaluated separately, rather than summed as blended-line opacity.

Changing the channel count under an instrument ID requires compatible training and validation selections. Use distinct IDs for datasets with different channel counts. Train a new run after changing the sampling or response; do not resume a checkpoint fitted with a different observing model.

## Instrumental spectral response

The bundled Hinode preset uses **monochromatic sampling**: it synthesizes directly at each retained calibrated wavelength. It does not download a transmission profile or automatically convolve with an SP line-spread function. Spectral pixel spacing and instrumental spectral width are different quantities.

To account for finite instrumental broadening, copy the run YAML and replace its `instrument` section with explicit responses for **both** lines:

```yaml
instrument:
  - type: generic
    instrument_id: HINODE_6301
    line_id: fe-i-6301
    spectral_response:
      type: gaussian
      fwhm: 0.0248
      unit: Angstrom
      samples: 9
  - type: generic
    instrument_id: HINODE_6302
    line_id: fe-i-6302
    spectral_response:
      type: gaussian
      fwhm: 0.0248
      unit: Angstrom
      samples: 9
```

The illustrative **24.8 mÅ FWHM** comes from the nearly Gaussian SP response described by [HAO](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description). Use the response appropriate to your calibration/product. `fwhm` describes instrumental blur and should not be changed merely to make synthesis faster. The fitted `vmac` is a separate atmosphere broadening parameter; avoid counting the same instrumental blur twice through both a fixed broadening term and an explicit response.

Numerical integration samples are not added to the observations: `samples` sets the **Gauss–Hermite integration nodes per observed wavelength**. At 9 nodes, the forward model evaluates nine nearby wavelengths, combines them with normalized Gaussian weights, and returns one synthetic value for each measured pixel. The observed channel count and grid remain unchanged.

| Goal | Change |
| --- | --- |
| Increase integration accuracy | Try `samples: 17` in both responses |
| Reduce synthesis cost | Try `samples: 5`; minimum accepted is 3 |
| Recover monochromatic sampling | Remove `spectral_response`, or use `spectral_response: {type: monochromatic}` |
| Use a measured non-Gaussian response | Set `type: tabulated` with calibrated `offsets` and quadrature `weights` |

For a tabulated response, one-dimensional arrays apply the same line-spread function at every observed wavelength. Arrays shaped `[observed wavelength, integration node]` provide a different response for each channel; rows must follow the retained wavelength order after window selection or subsampling. Offsets are relative to each observed wavelength, in `unit`. Weights must include integration spacing, be finite and nonnegative; SPINNME normalizes them. Increase/decrease the number of entries by resampling and reintegrating the calibrated profile together, not by editing only its offset array. See [tabulated response settings](custom.md#add-a-spectral-response).

Higher node counts increase synthesis cost and memory approximately in proportion to the count at fixed batch size. Check convergence on fixed atmosphere parameters using, for example, 5, 9 and 17 nodes. Changing node count tests numerical integration; changing `fwhm` changes the physical response. The configured response is shared by training, validation and snapshot synthesis. Original spectral input files are not needed for exports from saved grids.

## Optional observer velocity

Hinode rasters do not need observer-velocity keywords. If the supplied raster header contains finite `OBS_VR`, `OBS_VW` or `OBS_VN` values in m/s, SPINNME uses those components for the Doppler correction. Missing components contribute zero; with no velocity keywords, no observer-motion correction is applied. Keep the correct spatial WCS, observer position and slit acquisition times. The preparation helper retains optional velocity cards supplied in the raster header.

A fitted spectral shift or a pipeline velocity map is not an observer-velocity keyword. Without a known observer correction, the fitted/exported Doppler velocity retains that unknown shift relative to the supplied wavelength calibration.

## Export the inversion

```bash
spinnme export ./runs/hinode/inversion.pme --all-observations --parameters Brtp Blos B_transverse Vdop --format fits --output ./runs/hinode/products
```

When comparing a scan with an HMI image, match actual slit times rather than assigning one time to the full raster.
