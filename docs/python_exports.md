# Python loading and exports

`spinnme.load()` is the entry point for a portable inversion. Load once, then
prepare grids, evaluate maps or coordinates, and write results through that object.
The loader restores all saved model weights, corrections, normalization,
wavelengths and embedded calibration without reopening the training data.

## Native maps and exports

```python
from spinnme import load
from spinnme.exports import read_npz

with load("runs/hmi/inversion.pme") as snapshot:
    print(snapshot.instrument_ids, snapshot.config, snapshot.times)
    observation = snapshot.observations[0]["id"]
    result = snapshot.evaluate(
        observation, parameters=["B", "inclination", "azimuth", "Brtp", "Vrtp", "Vlos"]
    )
    result.write_fits("products/native/fits")
    result.write_npz("products/native/parameters.npz")
    result.write_jpg("products/native/preview")
    br_map = result.maps["Br"]  # SunPy Map: WCS, units and validity mask.

restored = read_npz("products/native/parameters.npz")
```

Map arrays use `(ny,nx)` order. `mask` marks supported visible pixels; invalid
values are NaN. Spectra have trailing wavelength axes. Writers accept
`overwrite=True`; `batch_size=4096` bounds inference memory. See
[exports](exports.md) for the available products.

## Coordinates, subframes and resolution

```python
from astropy import units as u

with load("runs/hmi/inversion.pme") as snapshot:
    observation = snapshot.observations[0]["id"]
    result = snapshot.evaluate(
        observation, bounds=(-250, -200, 250, 200), shape=(256, 384),
        parameters=["Brtp", "Vrtp", "Vlos"],
    )
    result.write_fits("products/region")

    # A reusable grid: no preliminary model evaluation is needed.
    grid = snapshot.grid(observation, shape=(512, 768))
    result = snapshot.evaluate(grid, parameters=["Brtp", "IQUV"])

    # Public geometry for transformations of independent reference data.
    geometry = snapshot.geometry(grid, indices=[0, 1, 2])
    print(geometry["coords"], geometry["rtp_to_img_transform"])

    # Arbitrary physical coordinates, in any transformable solar frame.
    positions = result.maps["Br"].pixel_to_world([100, 120]*u.pix, [100, 120]*u.pix)
    samples = snapshot.sample(positions, observation=grid, parameters=["Brtp", "Vrtp"])
    samples.write_npz("products/coordinate_samples.npz")
```

`sample()` accepts
`SkyCoord`, preserves its leading shape and returns the same named products and
units as maps. Samples have no image WCS; use NPZ to export them, or evaluate an
image grid to obtain SunPy maps and FITS images. Samples are simultaneous at
`time` (or the selected grid epoch), independently of saved pixel support/scan timing.

`bounds` is `(xmin,ymin,xmax,ymax)` in helioprojective arcsec; omit it for the
whole field. `shape` is `(ny,nx)`; omit it to retain native crop sampling.
Changing sampling does not change observed resolving power or integrate flux.
The loader carries support masks, nearest native scan times, detector mappings,
artifact pixel coordinates and acquisition response through crop/resize together.
An explicit `time` overrides scan times with a simultaneous model slice.

The observation argument accepts a saved ID, a reference FITS path, a SunPy Map,
or a dictionary with `id`, `instrument_id`, `shape`, `header` and `time.model_time`.
Reference FITS grids are read from headers, without loading their intensity cube.
They require spatial WCS, observer metadata and an epoch; available `OBS_V*`
components supply the observer correction. HMI uses `DATE-OBS`, matching training.
Synthetic HMI spectra additionally need a matching embedded acquisition response
and a known detector footprint. Merely changing date cards does not update geometry.

## Time sequences

```python
from pathlib import Path

with load("runs/hmi-series/inversion.pme") as snapshot:
    grid = snapshot.grid(snapshot.observations[0]["id"], shape=(256, 384))
    for index, epoch in enumerate(snapshot.cadence_times(30, instrument_id="HMI")):
        result = snapshot.evaluate(grid, time=epoch, parameters=["Brtp", "Vlos"])
        result.write_fits(Path("products/cadence") / f"frame_{index:04d}")
```

Cadence epochs stay within the saved training span, including raster offsets at
its endpoints. A fixed grid retains a fixed observer; model time is recorded
separately from geometry time. For changing observers, pass a different reference
FITS or Map at each epoch. Temporal interpolation requires a time-series fit;
staying inside the span does not establish coverage across gaps.

## Command line

From the project root:

```bash
spinnme export runs/hmi/inversion.pme --observation SAVED_ID \
  --bounds -250 -200 250 200 --shape 256 384 \
  --parameters Brtp Vrtp Vlos --format fits --output products/region

python examples/export_python.py runs/hmi-series/inversion.pme \
  --output products/cadence --shape 256 384 --cadence 30
```

The concise example script calls the same loader methods. It writes FITS, NPZ
and JPG, with automatic CUDA/CPU selection and an optional `--device` override.
