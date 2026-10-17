# How the fit works and what the maps mean

SPINNME estimates solar magnetic, velocity and atmosphere maps from calibrated polarized spectra. Each fresh run fits a new neural network to the observations you select. The network takes **solar position and time** as inputs and returns physical parameters; measured Stokes I/Q/U/V are targets compared with synthesized spectra through an instrument model. It is a coordinate-based representation of your selected scene and time interval. A `.pme` snapshot stores that fitted representation for later exports.

## What the fit represents

The Milne–Eddington (ME) approximation uses an effective atmosphere with magnetic field, velocity and line parameters constant along optical depth, and a linearly varying source function. It does not retrieve a height-stratified atmosphere, temperature profile or arbitrary depth-dependent gradients. Complex asymmetric or multi-component spectra can therefore retain systematic residuals. The [CSAC inversion description](https://www2.hao.ucar.edu/csac/csac-data/sp-data-description) gives background on the ME approximation.

SPINNME represents one magnetic and velocity field on the modeled solar surface. Distinct `line_id` values give separate width, damping, opacity and source-function outputs; observations sharing a `line_id` share those atmosphere outputs too. Joint fitting assumes the same surface field can describe the lines/instruments at a given position/time. It does not resolve their different formation heights or sum multiple lines as blended opacity. See [custom instruments](custom.md).

The bundled presets use differentiable spectral synthesis and a shared continuous network. They do not enforce a divergence-free field or a time-evolution equation. Default synthesis uses one magnetic ME component; the optional [limb correction](artifacts.md#other-optional-corrections) adds a magnetic/field-free mixture. Instrumental spectral response is integrated when configured; ordinary inversion does not automatically convolve spatial PSFs or integrate exposure windows.

## One acquisition, a raster or a time series

| Input | Information supplied |
| --- | --- |
| One acquisition | Profiles at the sampled positions and acquisition time; no measured time evolution |
| Scanning raster | Different positions sampled at their slit times; a raster is not a simultaneous movie |
| Time series | Repeated observations constrain evolution at their sampled times, with gaps and exposure-dependent limits |

Native raster exports preserve scan times. An explicit export `time` requests a simultaneous model slice. A denser output grid/cadence samples the fitted model more densely; it adds no observations or guaranteed resolving power. See [Python exports](python_exports.md).

## Export conventions

Arrays use NumPy `(y,x)` order; FITS pixel coordinates use `x=column`, `y=row`.

| Product | Convention |
| --- | --- |
| `Br` | Radial field, positive outward |
| `Btheta` | Positive toward increasing colatitude: southward |
| `Bphi` | Positive toward increasing Carrington longitude |
| `Blos` (`B_los`) | Positive toward the observer |
| `inclination` | 0° toward observer, 90° transverse, 180° away |
| `azimuth` | From +image-y toward −image-x, modulo 180°; image axes follow observing geometry and image roll |
| `Vdop` (`v_los`) | Positive redshift; solar rotation included, observer correction excluded |
| `vr`, `vtheta`, `vphi` | Velocity components in the Carrington frame |
| `vdop` | Observer-shifted Doppler velocity before learned instrument corrections |
| `forward_vdop` | Doppler velocity after configured instrument corrections |

Fields are in **gauss**, velocities in **m/s**, and exported `inclination` and `azimuth` in **degrees**. Low-level `inc` and `azi` use radians. Do not infer vector directions from a plotting program's screen orientation; retain WCS and transform pipeline conventions before comparing angles. [Exports](exports.md) defines the atmosphere parameters and available keys.

Geometry includes the observer's finite distance. Observer-velocity correction uses available `OBS_VR`/`OBS_VW`/`OBS_VN` values; missing components contribute zero. Without them, fitted Doppler velocities retain any observer shift in the supplied spectra. Rest-wavelength calibration still sets the velocity zero-point; an enabled instrument velocity correction can trade a common offset against the fitted solar velocity.

## Interpretation

No azimuth disambiguation is applied. Azimuth is reported modulo 180 degrees; signed transverse and spherical components retain the model's chosen branch. For comparison with another inversion, use field strength, LOS field, transverse magnitude or wrapped azimuth differences.

A single viewing direction constrains a Doppler projection, not three independent velocity components. Exported vector velocities are model-dependent estimates; additional views only help where their coverage and shared-field assumptions supply information. `Vdop` is the usual product for a LOS Doppler comparison.

Maps are point estimates, with no automatic posterior or calibrated per-pixel uncertainty. A low loss, a smooth map or a successful export does not establish uniqueness or accuracy. Stay within the fitted region/time coverage: valid solar geometry and an export mask alone do not certify observational support at every requested position/time. Follow the [Stokes-fit and reference checks](evaluation.md) before interpreting the maps.
