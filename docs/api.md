# Python API

Use `spinnme.run()` to train and `spinnme.load()` to export. The [Python export guide](python_exports.md) contains complete resolution, subregion and cadence examples.

## Run an inversion

```{py:function} spinnme.run(config, *, resume=False, profile=False, reload=False)
Train from a YAML path, bundled preset or configuration dictionary. Optional `resume`, `reload` and `profile` control execution. Returns `(module, data_module, trainer)` and writes `inversion.pme` to the configured output directory.
```

A complete Python dictionary can be passed directly, without saving an input YAML:

```python
from spinnme import run

observation = {
    "type": "fits", "data_path": "data/*.fits", "instrument_id": "USER",
    "axis_order": ["wavelength", "stokes", "y", "x"],
    "stokes_normalization": 1.0,
}
config = {
    "path": "runs/custom",
    "data": {"train_configs": [observation], "valid_config": dict(observation, sample_idx=0),
             "num_workers": 4, "num_dataset_workers": 0},
    "instrument": [{"type": "generic", "instrument_id": "USER", "line_id": "fe-i-6173"}],
    "model": {"type": "mlp", "dim": 64, "n_layers": 3, "vector_potential": False},
    "training": {"max_steps": 1000, "optimizer_config": {"type": "adamw"}},
    "logging": {"mode": "disabled"},
}
if __name__ == "__main__":
    module, data_module, trainer = run(config)
```

Use calibrated Fe I 6173 input with spatial/observer/time and wavelength metadata. The
[Colab notebooks](colab.md) contain full dictionaries for every supported workflow.
`run()` copies and validates the input dictionary, including all model, instrument,
normalization, objective and training settings. The caller's dictionary is unchanged.
The runner writes `resolved.yaml` alongside the results for provenance and resume.
YAML remains supported by the same interface: `run("inversion.yaml")` (or a
`pathlib.Path`), as well as `spinnme run --config inversion.yaml` from the CLI.

To start from a bundled preset and apply typed overrides from Python:

```python
from spinnme import run
from spinnme.configuration import resolve_config

config = resolve_config(
    "generic/stokes_cube",
    input_path="data/*.fits",
    output="runs/custom",
    overrides=["training.max_steps=10000"],
)
run(config)
```

For the prepared HMI directory from the quickstart, the same path bindings are:

```python
config = resolve_config(
    "hmi/full_disk", output="runs/hmi-python",
    bindings={"data_path": "data/hmi"},
)
if __name__ == "__main__":
    module, data_module, trainer = run(config)
```

Training automatically detects the accelerator and available devices unless configured otherwise. `run()` returns training objects. Load the saved `runs/hmi-python/inversion.pme` to obtain a snapshot for exports.

Choose a preset matching your [input data](data.md), with the correct axes, wavelengths and instrument setup.

## Load and evaluate

```{py:function} spinnme.load(path, *, device=None)
Load a `.pme` snapshot. Omit `device` or use `"auto"` to prefer CUDA when available, otherwise CPU. Use `with load(...) as snapshot:` to close temporary calibration files automatically.
```

```{py:class} spinnme.snapshots.Snapshot(path, *, device=None)
Provides `observations` (saved grid records), `instrument_ids` and `metadata`. Create it through `spinnme.load()`.
```

The complete saved model state is restored strictly, including all forward,
velocity, limb and artifact correction models, normalization, wavelengths,
physics configuration and embedded calibration assets. `snapshot.module`
exposes the restored module; `config` and `times` expose the saved configuration
and epochs. No original training files are reopened. `.pme` is the portable
inference state; use the training checkpoint to resume optimizer/scheduler state.

```{py:method} spinnme.snapshots.Snapshot.grid(observation=None, *, shape=None, bounds=None, instrument_id=None)
Prepare a reusable grid from a saved observation ID, FITS path, SunPy Map or observation dictionary. `shape` is `(ny,nx)` and `bounds` is `(xmin,ymin,xmax,ymax)` in helioprojective arcsec. Cropping/resizing updates WCS, support masks, raster times, detector coordinates and artifact pixel coordinates together. The source record is not mutated.
```

```{py:method} spinnme.snapshots.Snapshot.evaluate(observation=None, *, parameters=None, instrument_id=None, time=None, batch_size=8192, mu_limit=0.0, progress=False, shape=None, bounds=None, coordinates=None)
Return an `EvaluationResult` on a saved grid ID or reference FITS path. `parameters` accepts explicit parameter keys such as `Brtp`, `Blos`, `Vdop`, `vmac`, `damping`, or `IQUV`; the default is `Brtp`. Select an observation explicitly when several are saved. `instrument_id` selects the relevant instrument. `batch_size` controls inference memory; `mu_limit` excludes pixels near the limb.
```

`time` selects the model epoch while retaining the chosen observing geometry. Omitting it preserves saved per-pixel acquisition times, including after a crop or resize. A custom observation dictionary supplies `id`, `instrument_id`, `shape` in `(ny,nx)` order, spatial `header` and `time.model_time`. Use `grid()` or `evaluate(..., shape=..., bounds=...)` to update grids centrally.

```{py:method} spinnme.snapshots.Snapshot.geometry(observation=None, *, indices=None, time=None, instrument_id=None)
Return normalized model `coords` in xyzt order, projection matrices, `mu`, observer LOS velocity, normalized artifact pixels and available detector coordinates. `indices` selects a flat batch of pixel IDs. Evaluation uses the same geometry implementation.
```

```{py:method} spinnme.snapshots.Snapshot.sample(coordinates, *, observation=None, time=None, **kwargs)
Evaluate physical `SkyCoord` positions using the same products, synthesis, units and batching as maps. `observation` selects the observer/calibration. Pass a scalar `time` for a simultaneous epoch, or a time array broadcastable to the coordinate shape for exact scan times; times are sliced together with coordinates across batches. Helioprojective coordinates with per-point observation times retain their own observers and epochs for spatial and magnetic projections. Samples are independent of saved pixel support and raster offsets. The result retains the coordinates and supports NPZ; FITS images and SunPy maps require an image grid. Raw numerical coordinates with implicit units are not accepted.
```

```{py:method} spinnme.snapshots.Snapshot.cadence_times(cadence_seconds, *, instrument_id=None)
Return regular UTC epochs within the instrument's saved training span, including scan offsets at its endpoints. Evaluate a reusable grid at those times for a fixed-observer sequence.
```

## Read and write results

```{py:class} spinnme.exports.EvaluationResult(arrays, units, header, mask, metadata)
`arrays` maps parameter names to NumPy arrays. Maps have leading `(y,x)` axes, a spatial WCS `header` and a 2D `mask`; coordinate samples retain the coordinate shape and have `header=None`. `units` gives physical units and `metadata` records model time, geometry and provenance. `result.maps` returns scalar products as SunPy Maps, including units and masks.
```

```{py:method} spinnme.exports.EvaluationResult.write_fits(directory, *, overwrite=False)
Write one FITS file per parameter, including WCS, units and validity mask. Native slit grids include exact `X_COORDINATE`/`Y_COORDINATE` images in arcseconds; scanned grids include UTC `TIME_OFFSETS`. Returns output paths.
```

```{py:method} spinnme.exports.EvaluationResult.write_npz(path, *, overwrite=False)
Write arrays, mask, WCS, units and metadata in one NPZ file. Returns output paths.
```

```{py:method} spinnme.exports.EvaluationResult.write_jpg(directory, *, overwrite=False, dpi=180, cmap=None)
Write scalar-map previews and metadata sidecars. Returns output paths.
```

```{py:function} spinnme.exports.read_npz(path)
Read an exported NPZ as an `EvaluationResult` without Python object deserialization.
```

For configuration discovery, use `spinnme.configuration.list_configs()` and `read_config(name_or_path)`. `spinnme.instruments.load_instrument_definition(name)` returns the packaged `hmi`, `hinode`, `phi_hrt` or `phi_fdt` definition. See [configuration](configuration.md) and [new instruments](custom.md) for their settings.
