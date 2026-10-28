# Prepare your FITS data

Choose the guide that matches your inputs:

| Input | Start here |
| --- | --- |
| HMI `S_90s` / `S_720s`, all 24 Stokes segments plus matching transmission profiles | [HMI](hmi.md) or the [quickstart](quickstart.md) |
| Hinode SP calibrated Stokes raster/slits containing both 6301/6302 lines and scan times | [Hinode](hinode.md) |
| Solar Orbiter PHI calibrated L2 full-Stokes HRT/FDT cube with verified release metadata | [PHI](phi.md) |
| Another calibrated Stokes cube | [Custom FITS and instruments](custom.md) |

Magnetic/velocity pipeline maps are comparison references, rather than Stokes training inputs. HMI requires its separate response preparation. Hinode presets use a 24.8 mÅ FWHM Gaussian spectral response. PHI presets integrate a nominal single-order Airy response; see [PHI response assumptions](phi.md#set-wavelengths-continuum-and-time).

## What the data must contain

SPINNME expects calibrated I/Q/U/V, correct solar WCS and the metadata needed to place each observation on the Sun:

- Spatial WCS and observation time (`DATE-OBS`, with the appropriate `TIMESYS`).
- Observer distance `DSUN_OBS` in metres and position, commonly `HGLN_OBS`/`HGLT_OBS` or `CRLN_OBS`/`CRLT_OBS` in degrees.
- The atomic line, rest wavelength and calibrated wavelength samples with their units.
- The cube's NumPy axis order, Stokes order/signs and intensity normalization.

Observer velocity is optional. SPINNME reads finite numeric `OBS_VR` (radial, positive away from the Sun), `OBS_VW` (westward) and `OBS_VN` (northward), in m/s. Available components provide the observer-motion Doppler correction; absent or unavailable components contribute zero. If all are absent, HMI, Hinode and generic FITS inversions run without observer-velocity correction. No manually inserted zero-valued cards are needed.

WCS describes the spatial grid; wavelength sampling and polarization conventions must also match the data. The generic reader accepts another HDU for the spatial header, an explicit wavelength array and verified observer overrides. See the [custom configuration](custom.md).

Inspect a file's HDUs, array shapes and headers before selecting reader settings:

```bash
spinnme inspect ./data/observation.fits
```

## Run an inversion

For a custom Fe I 6173 cube with wavelength metadata in its header:

```bash
spinnme run --config generic/stokes_cube --input './data/custom/*.fits' --run_path ./runs/custom --export fits
```

`run` prepares and reuses data caches automatically. Optional `prep` only checks observations and builds those caches; it never downloads calibration or changes the run YAML. Device selection is automatic; `--device cpu` is an optional override.

Keep quoted globs so SPINNME receives the complete input selection. Relative paths are resolved from the directory where you run the commands. `--input` replaces the first training dataset and the validation input; additional datasets are configured in YAML.

An existing cache is trusted and reused without scanning source files or checking stored chunks, shapes or preparation settings. Reruns in the same Python process reuse an already loaded datamodule while it remains in memory. Use `--reload` or a fresh work directory after changing inputs or preparation settings, or to replace a damaged cache. Older caches are loaded directly without their recorded file checks.
New caches store each full-spectrum observation once. Training lines and validation select wavelengths, masks and normalization from that shared store. Zarr chunks target 1 MiB to bound decompression for small training batches. Native pixel IDs, slit coordinates and acquisition times are retained. Validation reads storage sequentially and restores native image order for maps.

Loader settings (`batch_size`, `dataset_batch_size`, `num_workers`, `prefetch_factor`, `persistent_workers`, `pin_memory`, `balance_train_configs`) apply from the current config on every reload. Changing them requires no preparation. Scientific selections and normalization still require `--reload`.

Preparation writes the arrays and reload metadata directly. Cached data are trusted; no integrity inventory, checksum pass or verification scan is performed.

In distributed runs, only rank 0 prepares or rebuilds the cache. Other ranks reopen the published metadata and lazy arrays concurrently, without acquiring the preparation writer lock. Zarr stores open when a worker reads a batch. Lightning’s `LOCAL_RANK` is honored even when Slurm rank variables are inherited from the parent. Followers skip logger initialization and hardware/input inventories. Large profile selectors are read-only memory maps, including in spawned loader workers. For compatible physical conventions, on the first reuse of an older selector layout, rank 0 moves selectors out of JSON before launching followers; this preserves the existing spectra and does not read FITS or rebuild observations. Other ranks wait up to an hour only if no cache has been published.

## Times and valid pixels

HMI uses the effective `T_OBS` epoch (TAI in JSOC) for both the model and coordinate geometry. `DATE-OBS` remains the UTC exposure start, and `T_REC` is only an archive slot label. `T_OBS` is required. Other datasets default to `DATE-OBS`; the time origin is the earliest selected observation. Hinode slit folders are assembled during inversion and retain their slit acquisition times. `time_reference: average`, `solar` or `earth` selects `DATE-AVG`, `DATE_SUN` or `DATE_EAR` for the model time while keeping the native spatial geometry.

For a generic/Hinode `time_hdu`, offsets are seconds relative to the **selected model time**, not an independently read DATEREF. Helper-generated Hinode offsets use `DATE-OBS`, so retain `time_reference: observation` or rebase the offset map when changing that reference.

Non-finite Stokes samples and pixels outside the visible solar surface are excluded. Exports keep invalid pixels as NaN and include a validity mask. `mu_limit` can further restrict the limb.

A spatial WCS that depends on external distortion/lookup HDUs (`-TAB`, CPDIS or D2IM) must first be converted to a self-contained, validated spatial WCS.

## Download data

Install archive support with `pip install 'spinnme[download]'`. The mission guides give download commands and preparation steps. HMI requires a registered JSOC email; PHI uses the Solar Orbiter Archive. For a known Hinode file, use `spinnme download hinode --url URL --output DIRECTORY`. For calibrated Stokes rasters, use `spinnme download hinode --start START --end END --output DIRECTORY`; see the [Hinode guide](hinode.md).
