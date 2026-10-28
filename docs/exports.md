# Snapshots and exports

An inversion writes `inversion.pme`. This file includes the model, instrument settings, saved observing grids and response calibration needed for inference. Use the same loader for FITS, NPZ and JPG output; the original training files are not needed for saved-grid exports.

## Export from the command line

```bash
spinnme inspect runs/hmi/inversion.pme
spinnme export runs/hmi/inversion.pme --all-observations --output products/fits
```

The default is one FITS file per `Br`, `Btheta` and `Bphi` map, for each selected observation. To choose a grid and other products, replace `SAVED_ID` with an ID reported by `inspect`:

```bash
spinnme export runs/hmi/inversion.pme --observation SAVED_ID --parameters B Blos B_transverse --format npz --output products/fields.npz
spinnme export runs/hmi/inversion.pme --observation SAVED_ID --parameters Blos --format jpg --output products/preview
```

Use `--overwrite` to replace existing outputs. Inference automatically prefers CUDA and falls back to CPU. Optionally set `--device cpu` to force CPU, and reduce `--batch-size` if memory is limited.

## Choose parameters

| Selection | Products |
| --- | --- |
| `Brtp` (default) | `Br`, `Btheta`, `Bphi` |
| `Vrtp` | `vr`, `vtheta`, `vphi` in m/s, in the Carrington frame |
| `Vlos` | Alias for `v_los`: positive redshift in m/s, includes solar rotation and excludes observer motion |
| `IQUV` | Synthesized Stokes I/Q/U/V cubes |
| `Artifact` | `artifact_alpha_Q`, `artifact_alpha_U`, `artifact_alpha_V`, `artifact_gate`, when enabled |
| `Blos` | Line-of-sight magnetic field, positive toward the observer, in gauss |
| `Vdop` | Solar line-of-sight Doppler velocity, positive redshift, in m/s; includes solar rotation, excludes observer motion |
| `B`, `B_transverse` | Magnetic strength and transverse strength, in gauss |
| `inclination`, `azimuth` | Viewing-frame angles in degrees; see [axis definitions](method.md#export-conventions) |
| `vmac` | Positive Doppler-width speed inside the solver’s Voigt profile, in m/s; no separate macroturbulent convolution |
| `damping` | Dimensionless Voigt damping parameter |
| `b0`, `b1` | Dimensionless source-function intercept/slope coefficients in the configured normalization |
| `kl` | Positive dimensionless line-opacity strength in this solver’s profile normalization |
| `vr`, `vtheta`, `vphi` | Spherical velocity components, in m/s |
| `forward_vdop`, `forward_vmac`, `forward_b0`, etc. | Individually selected atmosphere values used for synthesis after instrument corrections |
| Individual names | For example, `B_transverse`, `B_image_x`, `B_image_y`, `vr`, `vtheta`, `vphi`, `mu` |

Continuum-normalized fits use a ratio parameterization: `b0` and `b1` sum to one, fixing the unconstrained absolute source scale. Absolute-intensity fits use independent coefficients. These parameters are effective ME values; do not equate similarly named pipeline parameters without converting their definitions. Vector velocities are model-dependent under the [viewing constraints](method.md#interpretation).

Specify exactly the keys you need, for example `--parameters Brtp Vdop Blos vmac damping`. `LOS`, `ME` and `ME_forward` are not accepted. `Brtp` expands to three spherical components; other scalar keys create one map each, named by their key. `B_los` and `v_los` remain accepted spellings for `Blos` and `Vdop` and retain those filenames.

`vdop` is the intrinsic solver's Doppler velocity **including the available observer-motion correction**; `Vdop` excludes that correction. When exporting both `Vdop` and `vdop`, use NPZ or separate FITS/JPG directories because the names differ only by case. If observer velocities were absent, no correction was applied and an unknown observer shift can remain in the fitted velocity. Export metadata includes `observer_velocity` (the internal `[-west, north, radial]` vector in m/s) and `observer_velocity_correction_applied`. `forward_vdop` additionally includes configured instrument corrections. Other corrected keys are `forward_b_field`, `forward_inc`, `forward_azi`, `forward_vmac`, `forward_damping`, `forward_b0`, `forward_b1` and `forward_kl`; corrected angles use radians. Adding `IQUV` does not change intrinsic parameter exports. No post-processing azimuth disambiguation is applied.

## Choose a format

| Format | Contents |
| --- | --- |
| FITS | One file per parameter, with spatial WCS, observer metadata, units, `VALID_MASK` and JSON metadata. Spectral cubes include wavelength WCS for the actual sampling. |
| NPZ | Named arrays, validity mask, WCS header, units and metadata in one file. Read with `spinnme.exports.read_npz()`. |
| JPG | Scalar-map previews with WCS axes, units and a metadata sidecar. Use FITS or NPZ for scientific analysis. |

Saved slit grids also include `X_COORDINATE` and `Y_COORDINATE` image extensions
in arcseconds. These retain exact native pointing even when the primary image's
WCS is only a nominal display approximation. Per-slit viewing geometry and roll
remain in the `PROVENANCE` JSON extension.

Native raster exports retain scan timing: FITS includes a spatial `TIME_OFFSETS` extension in seconds from its UTC `DATEREF`, and NPZ metadata retains the offsets and their axis. An explicit `--time` evaluates the whole grid simultaneously and omits native scan offsets.

Invalid and off-disk pixels remain NaN. Source masks are preserved on saved grids; `--mu-limit` can exclude more of the limb. Write spectra as FITS or NPZ and vector diagnostics as NPZ. The scalar `Artifact` group also supports JPG.

Synthesized spectra use the saved instrument response and corrections. Continuum-normalized spectra remain normalized; absolute intensities require the corresponding pixel continuum.

## Export from Python

```python
from spinnme import load

with load("runs/hmi/inversion.pme") as snapshot:
    result = snapshot.evaluate(snapshot.observations[0]["id"], parameters=["Brtp", "Blos", "Vdop"])
    result.write_fits("products/fits")
    result.write_npz("products/fields.npz")
    result.write_jpg("products/preview")
```

See [custom Python exports](python_exports.md) to change resolution, select a subregion or export a time sequence.

## Supply another observing grid

```bash
spinnme export runs/hmi/inversion.pme --observation data/reference-grid.fits --instrument HMI --parameters Brtp --output products/reference-grid
```

The reference FITS supplies spatial WCS, observer position and the geometry epoch; observer velocity keywords are optional. Select an instrument ID present in the snapshot. `--time` changes the model epoch while retaining that observing geometry. For a changing viewpoint, provide the correct reference geometry for each epoch. External grids need their own quality masks and must lie within the fitted region.

Use `.pme` files for inference and `last.ckpt` for [resuming training](training.md).

Snapshots and exports record coordinate convention version 2. Azimuth is
counterclockwise from image +y, modulo 180 degrees. Hinode synthesis separately
accounts for its solar-frame Q/U basis and slit roll; HMI retains its image-frame
polarization convention. HMI headers preserve exposure start and record slot,
normalize `T_OBS` to an explicit TAI label, and provide the same effective epoch
as UTC `DATE-AVG`. Original time cards remain in provenance. Saved `model_time`
and `geometry_time` are explicit UTC dates; inference uses those dates directly,
with the FITS reference date as a fallback, independently of a version flag.
FITS exports retain both in `SPINNME MODEL_TIME` and `SPINNME GEOMETRY_TIME`.
Changing model time leaves the observing geometry date intact. Legacy Hinode
polarization still needs fresh training to correct the inferred physical field.
