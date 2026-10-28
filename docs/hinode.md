# Invert Hinode SP data

An [online Colab notebook](colab.md) downloads or uploads slit files, fits both spectral lines, and checks fits and exports.

Use calibrated Stokes spectra rather than pipeline magnetic-parameter maps as training inputs. The `hinode/sp` preset jointly fits Fe I **6301.5012 Å and 6302.4931 Å** from the same raster. Both lines constrain the same magnetic field and velocity; each has its own Milne–Eddington thermodynamic and source-function parameters.

The preset matches the inversion settings in `spinnme-evaluation/configs/hmi_hinode_comparison/hinode.yaml`: a 512-wide, 8-layer MLP with isotropic positional encoding, source-function ratios, batches of 8192 and a 36,000-second time scale. SOAP runs for 200,000 steps, with validation every 2000 batches and the learning rate decaying from 0.001 to 0.0001 over the full budget. ME solver compilation is enabled. Continuum-normalized Stokes profiles use linear loss scales of 1.0 for I and 0.1 for Q/U/V, with asinh scaling disabled.

Input/output paths, reference time and logging remain configurable for your observation. The Colab notebook uses the same full inversion settings, with editable input paths and training budget.

## Load a downloaded slit scan

Point the preset at the folder containing calibrated Level-1 slit FITS files:

```bash
spinnme run --config hinode/sp --input ./data/hinode-slits --run_path ./runs/hinode --export fits
```

The Hinode dataset reads and assembles the scan during inversion. It retains
both spectral lines, the native wavelength calibration, each slit’s pointing,
roll and UTC acquisition time. Missing slit indices remain gaps in the native
coordinates; no missing spectra are invented. No intermediate FITS raster or
per-slit copies are written. Only the normal training cache is created under
`work_path`. The YAML needs the input folder, without FITS axis or timing-HDU
settings. No MERLIN reference is required.

Native slit geometry is saved with the inference snapshot, so exports retain
its positions and timing after the input files are removed. The observer is
explicitly approximated by Earth at each slit’s acquisition time. Each slit
keeps its own roll and any supplied observer-velocity components. Missing
velocity components contribute no correction. The raster header gives a nominal display WCS; saved
per-column coordinates supply the actual sampling positions. The inversion
model is evaluated in physical Carrington Cartesian coordinates and time;
raster indices only identify observations. Native spectra are never
interpolated onto the nominal display WCS. Each slit’s original spatial header,
physical observer frame and sampling coordinates are retained in snapshots and
export provenance. Rebuild caches from the earlier native-slit loader once with
`--reload` to retain the per-slit physical frames.

Coordinate convention version 2 uses the SolarSoft
[`stksimages_sbsp.pro`](https://soho.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/stksimages_sbsp.pro)
geometry: the clockwise spatial angle is `CROTA2 + FGSPROT`, and the slit origin
uses `SPCCDIX0/1` and `CAMSSUM` relative to the full detector center (511.5).
These metadata are required; a cropped slit is not assumed to be centered.
The spatial WCS angle is the negative of this clockwise angle.

Calibrated Level-1 Q/U have already been rotated into the solar polarization
frame by operation 9 of
[`calib_sbsp.pro`](https://soho.nascom.nasa.gov/solarsoft/hinode/sot/idl/sp/util/calib_sbsp.pro):
positive Q is east–west. Use `instrument.type: hinode` for these data.
The public azimuth remains counterclockwise from image +y, modulo 180 degrees;
the Hinode solver converts it to `chi = azimuth - pAng + 90 degrees`.
This conversion belongs in synthesis, not in a post-hoc rotation of exported
maps. It retains the physical projection of the inferred vector.

For a MERLIN comparison, transform both products into the same image basis.
For the audited March 23, 2024 raster, MERLIN profiles agree with the solver
angle measured counterclockwise from solar west; use
`(MERLIN azimuth + 90 degrees + HMI pAng) mod 180` for the HMI image basis.
The [CSAC description](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description)
says clockwise, which conflicts with this profile check. Do not negate MERLIN
azimuth on the strength of that wording alone; verify other product versions
against their calibrated Q/U. MERLIN coordinate images also need not reproduce
the native SolarSoft slit geometry exactly.

Old snapshots remain readable with a legacy-convention warning and retain
their original synthesis. They are not corrected inversions. Rebuild caches
with `--reload` and retrain in a fresh run directory; checkpoints cannot resume
across convention versions. A 90-degree map edit cannot repair the spatially
projected magnetic field or its physical regularization.

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

## Download calibrated slits

Download calibrated Level-1 Stokes files by time:

```bash
spinnme download hinode --start 2024-03-23T00:59:23 \
  --end 2024-03-23T02:59:23 --output data/hinode-slits
```

Bounds select archive filename timestamps in `[start, end)`, which have
0.1-second precision. Times default to UTC; use `--time-scale tai` for TAI
labels. This downloads the original FITS files and a checksum manifest.
For an explicit file, use `spinnme download hinode --url URL --output DIRECTORY`.
Preparation for inversion is separate; no reference file is used for download.

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

## Atomic parameters

Both channels resolve their atomic constants from `spinnme.instrument.SPECTRAL_LINES`:

| Line (air wavelength, Å) | Lower level | Upper level | `g_low` | `g_up` | Effective Landé factor |
| --- | --- | --- | --- | --- | --- |
| Fe I 6301.5012 | z⁵P₂, J=2 | e⁵D₂, J=2 | 11/6 | 3/2 | 5/3 ≈ 1.667 |
| Fe I 6302.4931 | z⁵P₁, J=1 | e⁵D₀, J=0 | 2.49 | 0 | 2.49 |

The level assignments follow [Okunev (2004), Table 4.1](https://ediss.uni-goettingen.de/bitstream/11858/00-1735-0000-0006-B4F7-D/1/okunev.pdf).
For 6302, SPINNME retains the adopted 2.49 convention also tabulated in the
[SOLARNET instrument report, Table 2.3](https://science-media.org/userfiles/10/papers/10_paper_930.pdf);
pure LS coupling gives 2.50, a 0.4% difference in magnetic sensitivity.
The solver sums the full anomalous Zeeman pattern for 6301 and the three components
for 6302. It does not replace 6301 with an effective triplet.
The air rest wavelengths retain the existing calibration convention; published
laboratory values vary at the mÅ level, which matters for absolute Doppler velocities.
Oscillator strengths and excitation energies are not explicit inputs to this
Milne–Eddington model: line opacity, damping and Doppler width are fitted independently
for each line, so no fixed oscillator-strength ratio is imposed.

## Select spectral windows or fit one line

The preset selects two views of one cached full-spectrum observation with separate line identifiers, `HINODE_6301` and `HINODE_6302`. Each view retains measured samples within ±0.30 Å of its line center. The windows work for either direction of spectral dispersion and unequal sample counts. Spectra, pixel IDs and scan times are stored once; selecting a window does not interpolate the data. Validation checks both windows. **Stokes profiles - Comparison** shows separate 6301/6302 panels on their calibrated wavelength axes, including each line’s spatial mean and polarization extrema. Parameter and integrated-Stokes images are labeled by instrument. The legacy `valid.*` metrics refer to the first validation entry (6302); the second line is logged under `valid.HINODE_6301.*`.

To change the windows, copy the run YAML and edit `wavelength_range: [lower, upper]` in its dataset entries. Bounds are **absolute wavelengths** in `wavelength_unit`, with the upper bound excluded; `null` leaves that side open. Keep each window within the intended line and surrounding continuum. The input must contain both lines; an empty window raises an error.

For a single-line inversion, retain only that line's training dataset and instrument entry, and make validation use only the same line. `data.valid_config` accepts a dataset mapping for one line or a list of mappings for multiple lines. The standalone `--instrument hinode` definition describes only the 6302 channel; use `--config hinode/sp` for the joint inversion. The [multiple-line template](custom.md#combine-datasets-and-instruments) also shows how to combine different instruments.

### Narrow or widen the two fitted windows

A narrower `wavelength_range` reduces the number of fitted spectral pixels without changing their native spacing. A wider range includes more measured wing/continuum samples. The bundled preset uses these ±0.30 Å windows:

| Dataset | `wavelength_center` | `wavelength_range` |
| --- | --- | --- |
| Training `HINODE_6301` | 6301.5012 | `[6301.2012, 6301.8012]` |
| Training `HINODE_6302` | 6302.4931 | `[6302.1931, 6302.7931]` |
| Validation `HINODE_6302` | 6302.4931 | `[6302.1931, 6302.7931]` |
| Validation `HINODE_6301` | 6301.5012 | `[6301.2012, 6301.8012]` |

The [MERLIN product description](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description) uses the same ±0.30 Å fitting windows. Retain enough line wings for your velocities and field strengths. For continuum-normalized training, the intensity quantile is estimated from the retained samples, so an overly narrow window changes that proxy. Increasing a window beyond the observed spectrum adds no samples. Avoid including the other Fe I line in a single line's window: the two atmospheres are evaluated separately, rather than summed as blended-line opacity.

Changing the channel count under an instrument ID requires compatible training and validation selections. Use distinct IDs for datasets with different channel counts. Rebuild an existing cache once with `--reload` after changing fitted windows or adding validation lines. Adding validation lines alone can be applied when resuming with `--resume --reload`; the training and forward-model settings must still match the checkpoint. Train a new run after changing the sampling or response; do not resume a checkpoint fitted with a different observing model.

## Instrumental spectral response

The bundled Hinode preset, standalone instrument definition and notebook use a **Gaussian spectral response with FWHM 0.0248 Å (24.8 mÅ)**. The forward model convolves all four synthetic Stokes components before comparing them with the calibrated observations. This response is configured for **both** lines; no transmission-file download is needed. Spectral pixel spacing and instrumental spectral width are different quantities.

The preset includes:

```yaml
instrument:
  - type: hinode
    instrument_id: HINODE_6301
    line_id: fe-i-6301
    spectral_response:
      type: gaussian
      fwhm: 0.0248
      unit: Angstrom
      samples: 9
  - type: hinode
    instrument_id: HINODE_6302
    line_id: fe-i-6302
    spectral_response:
      type: gaussian
      fwhm: 0.0248
      unit: Angstrom
      samples: 9
```

The **24.8 mÅ FWHM** matches the nearly Gaussian SP response used by MERLIN and described by [HAO](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description). Use the response appropriate to your calibration/product. `fwhm` describes instrumental blur and should not be changed merely to make synthesis faster. SPINNME’s fitted `vmac` controls the intrinsic Doppler width in the absorption profile. MERLIN’s fixed 0.7083 km/s macroturbulence represents its instrumental convolution; do not assign that value to SPINNME’s `vmac` or add that instrumental contribution a second time.

Numerical integration samples are not added to the observations: `samples` sets the **Gauss–Hermite integration nodes per observed wavelength**. At 9 nodes, the forward model evaluates nine nearby wavelengths, combines them with normalized Gaussian weights, and returns one synthetic value for each measured pixel. The observed channel count and grid remain unchanged.

| Goal | Change |
| --- | --- |
| Increase integration accuracy | Try `samples: 17` in both responses |
| Reduce synthesis cost | Try `samples: 5`; minimum accepted is 3 |
| Recover monochromatic sampling | Remove `spectral_response`, or use `spectral_response: {type: monochromatic}` |
| Use a measured non-Gaussian response | Set `type: tabulated` with calibrated `offsets` and quadrature `weights` |

For a tabulated response, one-dimensional arrays apply the same line-spread function at every observed wavelength. Arrays shaped `[observed wavelength, integration node]` provide a different response for each channel; rows must follow the retained wavelength order after window selection or subsampling. Offsets are relative to each observed wavelength, in `unit`. Weights must include integration spacing, be finite and nonnegative; SPINNME normalizes them. Increase/decrease the number of entries by resampling and reintegrating the calibrated profile together, not by editing only its offset array. See [tabulated response settings](custom.md#add-a-spectral-response).

Higher node counts increase synthesis cost and memory approximately in proportion to the count at fixed batch size. Check convergence on fixed atmosphere parameters using, for example, 5, 9 and 17 nodes. Changing node count tests numerical integration; changing `fwhm` changes the physical response. The configured response is shared by training, validation and snapshot synthesis. Original spectral input files are not needed for exports from saved grids.

## Comparison with MERLIN

Matching the spectral response and windows does not make the inversions identical. MERLIN additionally fits an empirical scattered-light Stokes-I profile and magnetic filling factor. The current Hinode preset retains `limb_correction: false`: SPINNME’s optional limb model mixes a synthesized field-free atmosphere with the magnetic atmosphere and is not MERLIN’s empirical scattered-light model. Account for this distinction when comparing field strength or magnetic flux density; see the [MERLIN description](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description).

## Optional observer velocity

Hinode rasters do not need observer-velocity keywords. If the supplied raster header contains finite `OBS_VR`, `OBS_VW` or `OBS_VN` values in m/s, SPINNME uses those components for the Doppler correction. Missing components contribute zero; with no velocity keywords, no observer-motion correction is applied. Keep the correct spatial WCS, observer position and slit acquisition times. The preparation helper retains optional velocity cards supplied in the raster header.

A fitted spectral shift or a pipeline velocity map is not an observer-velocity keyword. Without a known observer correction, the fitted/exported Doppler velocity retains that unknown shift relative to the supplied wavelength calibration.

## Export the inversion

```bash
spinnme export ./runs/hinode/inversion.pme --all-observations --parameters Brtp Blos B_transverse Vdop --format fits --output ./runs/hinode/products
```

When comparing a scan with an HMI image, match actual slit times rather than assigning one time to the full raster.
