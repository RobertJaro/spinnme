# Evaluate an inversion

## Check the Stokes fit before comparing maps

1. Check axes, wavelengths, normalization, response calibration and masks on a small selection before a long run.
2. Inspect the [W&B validation images](training.md#weights--biases), enabled by default when W&B logging is active. Inspect the observed/synthesized I/Q/U/V comparison and component metrics across weak/strong signals and the limb. The built-in image sums absolute transformed signals over wavelength; it is a spatial overview, so also inspect individual spectra as below.
3. Compare profiles and residuals by wavelength and line. Check structured residuals and their scale against noise estimated in the **same intensity normalization**. The Hinode preset validates 6302 only; inspect 6301 separately or change the validation line in a separate diagnostic run.
4. Compare a longer fit with the same data/model/loss settings, and test an excluded acquisition or time block when assessing interpolation. Change one choice at a time and retain the resolved settings. A fixed step count or a smooth map does not establish convergence.

`valid.I/Q/U/V` are mean absolute differences after the configured continuum and asinh/linear transformations. `valid.objective` uses transformed squared residuals with the configured component/wavelength reductions. These values are not automatically noise-weighted chi-squared, physical field errors or uncertainty intervals; do not compare their magnitudes across different normalizations/weights as though they were the same statistic.

### Plot one observed and fitted spectrum

After running the [offline example](circular.md), this standalone Python example compares its first acquisition on the native grid. It writes `demo/stokes-fit.png` and prints per-component RMS residuals over valid pixels and wavelengths:

```python
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from spinnme import load
from spinnme.data.fits import read_fits_observation

observation = read_fits_observation(
    "demo/circular/circular_000.fits",
    axis_order=["stokes", "wavelength", "y", "x"],
)
with load("demo/circular/run/inversion.pme") as snapshot:
    saved = next(row for row in snapshot.observations if row["role"] == "train")
    result = snapshot.evaluate(saved["id"], parameters=["IQUV"])
    measured = observation["stokes"] / snapshot.stokes_normalization_by_instrument[saved["instrument_id"]]
    if snapshot.continuum_normalization_mode == "per_profile":
        quantile = snapshot.stokes_loss_config.get("continuum_quantile", 0.9)
        continuum = np.quantile(measured[..., 0:1, :], quantile, axis=-1, keepdims=True)
        measured = measured / np.where(continuum > 0, continuum, np.nan)

predicted = np.stack([result.arrays[key] for key in "IQUV"], axis=-2)
assert measured.shape == predicted.shape
valid = result.mask & np.isfinite(measured).all(axis=(-2, -1))
residual = predicted - measured
print("I/Q/U/V RMS:", np.sqrt(np.mean(residual[valid] ** 2, axis=(0, 2))))
y, x = np.argwhere(valid)[len(np.argwhere(valid)) // 2]
wavelength = (np.asarray(result.metadata["wavelength_grid_m"])
              + result.metadata["wavelength_center_m"]) * 1e10
figure, axes = plt.subplots(2, 4, figsize=(12, 5), sharex=True)
for index, component in enumerate("IQUV"):
    axes[0, index].plot(wavelength, measured[y, x, index], "o", label="observed")
    axes[0, index].plot(wavelength, predicted[y, x, index], label="fitted")
    axes[0, index].set_title(component)
    axes[1, index].plot(wavelength, residual[y, x, index])
    axes[1, index].axhline(0, color="k", linewidth=0.5)
    axes[1, index].set_xlabel("Wavelength [Å]")
axes[0, 0].set_ylabel("Stokes intensity")
axes[1, 0].set_ylabel("Fitted − observed")
axes[0, 0].legend()
figure.tight_layout()
Path("demo").mkdir(exist_ok=True)
figure.savefig("demo/stokes-fit.png", dpi=150)
plt.close(figure)
```

This example uses the offline cube's known axes, grid, wavelength selection and unit normalization. Adapt all of these for real data; HMI's 24 segments must first be assembled on the matching grid, and Hinode line windows must match their instrument IDs. Exported `IQUV` already includes the configured response and instrument corrections. For continuum-normalized fits it is divided by its own continuum proxy, with no asinh transform applied to the export; normalize the observed profiles equivalently before comparing. For absolute-intensity fits the exported spectrum is in the dataset's scaled units. W&B uses the additional asinh/linear transform. Compare matching epochs, spatial grids and wavelengths; this snippet does not align mismatched data or estimate parameter uncertainty.

## Summarize exported parameters

Use `inspect` to find a saved observation ID, then request the maps to summarize:

```bash
spinnme inspect runs/hmi/inversion.pme
spinnme evaluate runs/hmi/inversion.pme --observation SAVED_ID --parameters B Blos B_transverse --output evaluation/summary.json
```

This writes `summary.json` and `summary.csv` with finite-pixel counts, mean, standard deviation and range for each parameter.

## Compare with a reference

Prepare the reference on the same spatial grid, observer geometry and model epoch. Include physical units and its validity mask. A scalar FITS reference needs `BUNIT` and one selected parameter:

```bash
spinnme evaluate runs/hmi/inversion.pme --observation data/reference-grid.fits --instrument HMI --parameters Blos --reference products/reference-B_los.fits --output evaluation/comparison.json
```

For several parameters, supply an [exported NPZ](exports.md) containing matching parameter names and units:

```bash
spinnme evaluate runs/hmi/inversion.pme --observation data/reference-grid.fits --instrument HMI --parameters B Blos B_transverse --reference products/reference.npz --output evaluation/comparison.json
```

JSON and CSV report bias, MAE, RMSE and correlation on the common finite mask. Residuals are **prediction minus reference**. Pipeline products are estimates with their own assumptions; these metrics measure agreement, without propagating reference uncertainty. FITS units are converted when compatible; NPZ units must match the selected products. Mismatched grids, observers and epochs are rejected. Align or reproject the reference before evaluation; the command does not register images.

Compare field strength, LOS field, transverse magnitude or azimuth modulo 180 degrees when disambiguation is unavailable. Azimuth bias/MAE/RMSE use residuals wrapped to [−90°, 90°); Pearson correlation is omitted for wrapped angles. Check sign conventions, spatial resolution and integration times when using a pipeline inversion as the reference.

Before comparing different instruments, select overlapping times/regions, reconcile spatial PSF/resolution and temporal integration, convert signs/units, then align to a common observing grid and mask. The CLI checks grid/geometry compatibility; it does not match PSFs or exposure windows. Inspect difference maps alongside aggregate metrics.

## Cadence studies and figures

Treat native HMI 90 s versus 720 s as a product comparison: their noise and temporal processing also differ.

Keep model settings, normalization, optimizer budget, calibration and masks fixed across runs. Hold out complete acquisitions or time blocks before preparation to test interpolation, and record their actual UTC epochs. Correlated pixels are not independent validation samples.

Export NPZ maps and save CSV metrics before plotting. Use the core [plotting API](plotting.md) for reusable comparison panels, density scatters and figure export. Dataset readers, reference conventions, study matrices and publication selections belong in the evaluation project; they configure the shared framework functions.
