# Command-line reference

Use `spinnme COMMAND --help` for all arguments.

| Command | Purpose |
| --- | --- |
| `crop` | Crop FITS images and cubes, preserving spatial WCS |
| `download hmi/hmi-transmission/phi/hinode` | Download archive products or explicit Hinode URLs |
| `prep` | Validate inputs and prepare reusable data caches |
| `run` | Prepare and train an inversion |
| `inspect` | List FITS metadata or snapshot instruments/grids |
| `export` | Write FITS, NPZ or JPG products |
| `evaluate` | Compute map summaries or reference residuals |
| `config list/show/copy/validate/instrument` | Inspect and customize YAML |
| `synthetic` | Generate the offline circular-pattern example |

## Prepare and train

`prep` and `run` accept `--config NAME_OR_PATH`, `--input`, `--run_path`, `--work_path`, `--instrument`. A YAML placeholder `<<KEY>>` accepts `--KEY VALUE`; see [configuration](configuration.md#overwrite-named-values). `--output` remains a shortcut for the run path and its default cache. `--input` replaces the first training input, any additional unset training inputs, and the validation input. This lets the Hinode preset fit both lines from one raster. `--instrument` accepts a YAML definition or bundled `hmi`, `hinode`, `phi_hrt` or `phi_fdt` definition.

| Option | Use |
| --- | --- |
| `download hmi-transmission --input DATA --output PROFILES --email EMAIL` | Download profiles for the supplied Stokes acquisitions |
| `--reload` | Rebuild prepared caches |
| `run --device cpu/cuda/auto` | Optional training override; defaults to automatic accelerator detection |
| `run --resume` | Continue from `last.ckpt` |
| `run --export fits/npz/jpg` | Export saved grids after training |
| `run --profile` | Record a bounded performance profile |

`run` prepares caches automatically. Optional `prep` only builds caches from the same YAML; it does not fetch calibration or rewrite the run config. See [training](training.md) and [training options](training_options.md).

## Export and evaluate

Both commands take the `.pme` snapshot as their first positional argument and require `--output`.

| Option | Use |
| --- | --- |
| `--observation ID_OR_FITS` | Select a saved grid or reference FITS geometry |
| `--instrument ID` | Select an instrument present in the snapshot |
| `--parameters NAME ...` | Select explicit keys, e.g. `Brtp Vdop Blos`; default `Brtp` |
| `--time UTC_TIME` | Change model time, retaining the selected geometry |
| `--device cpu/cuda/cuda:N/auto` | Optional inference override; defaults to CUDA when available, otherwise CPU |
| `--batch-size INTEGER` | Limit evaluation memory use |
| `--mu-limit FLOAT` | Exclude pixels below a viewing-angle threshold |

`export` supports `--format fits/npz/jpg`, `--all-observations` and `--overwrite`. `evaluate --reference PATH` compares with FITS or NPZ. See [exports](exports.md), [Python examples](python_exports.md) and [evaluation](evaluation.md).

## Download and errors

HMI uses `--start`, optional `--end`, `--product`, `--time-scale utc/tai` and `--email`. `--cadence` sets archive sampling; `--segments` limits the returned segments. PHI uses `--telescope hrt/fdt` and `--product stokes`. Hinode uses `--start`, `--end` and `--time-scale utc/tai` to download calibrated Level-1 Stokes files. Repeated `--url` arguments download explicit files. See [data preparation](data.md).

Add global `--debug` before a command to show the full traceback:

```bash
spinnme --debug run --config custom.yaml
```

Commands return a nonzero exit status on failure. Use `--overwrite` to replace existing output from `config` or `export`.

## Crop FITS inputs

```bash
spinnme crop --input './data/*.fits' --output ./data/cropped \
  --pixels 100 100 356 356
```

Choose `--pixels X0 Y0 X1 Y1` (zero-based, upper bounds excluded) or `--bounds XMIN YMIN XMAX YMAX` in the input WCS, with `--unit` (default `arcsec`). `--hdu INDEX_OR_NAME` selects the science image; `--spatial-axes X Y` selects one-based FITS axes, default `1 2`. Outputs require a new/empty directory. See [FITS cropping](cropping.md) for cubes and [HMI subframes](hmi.md#invert-an-hmi-subframe) for a complete inversion example.
