# Troubleshooting

## Command not found or CUDA unavailable

Activate the environment used to install SPINNME (for example `source .venv/bin/activate`) and run `python -m pip show spinnme`. Try `python -m spinnme --version` to check that interpreter directly. Install with `python -m pip install spinnme` inside the intended environment.

Device selection is automatic. If CUDA is expected but unavailable, check `python -c "import torch; print(torch.cuda.is_available())"` and the installed PyTorch build. `--device cpu` optionally forces CPU execution; the small [offline example](circular.md) works without a GPU.

## No progress after W&B starts

W&B starts before preparation. Progress bars show training-data and validation-data loading, including cache reuse.

- Native full-disk HMI preparation can be slow and use substantial RAM. Its float32 Stokes cube alone is about 1.5 GiB per acquisition; geometry, filtering and shuffling need additional memory. Keep `data.num_dataset_workers: 1`, or start with a small crop.
- After preparation, source-file hashing reads the FITS again for the run manifest; shared-filesystem throughput can delay training startup.
- If the first training batch stalls, set both `data.num_workers` and `data.num_dataset_workers` to `0` to isolate worker startup. Set `training.compile_config.me_solver: false` to separate compilation time from training time.

Run `spinnme prep --config my-run.yaml` to prepare and cache data independently of W&B and training. The next `run` using the same work directory reuses that cache. Use `--reload` after changing inputs or preparation settings.

## FITS preparation fails

```bash
spinnme inspect data/observation.fits
spinnme prep --config my-run.yaml
```

Check the reported HDUs, NumPy axis order, I/Q/U/V order, wavelength count and units against the [dataset configuration](data.md). Correct the reader settings or prepare a corrected FITS file before training.

Observer geometry errors require verified distance, longitude/latitude and acquisition time. Observer velocity is optional: `OBS_VR`, `OBS_VW` and `OBS_VN` are used when present and finite; missing components contribute no velocity correction. See the [custom data guide](custom.md).

For incomplete HMI acquisitions, download the missing segments. HMI cutouts also need their full-detector origin for transmission calibration. Hinode rasters need a verified spatial WCS and the intended line window. PHI requires the correct release-specific axes, wavelength and time settings; see the instrument chapters.

## No usable pixels

Check the spatial WCS, finite Stokes values, source masks, positive continuum and `mu_limit`. Reduce an overly restrictive limb threshold or disable optional polarization selection while checking the input. Inspect a small crop before preparing the full dataset.

## Memory errors or poor convergence

Start with a small spatial/time subset. Check Stokes residuals, normalization, wavelength positions and calibration before increasing model size. Reduce training `data.batch_size` and `data.dataset_batch_size` together, preserving divisibility; reduce inference `--batch-size` independently.

For a non-finite loss or gradient, read the run's `numerical_failures/` report. Check invalid inputs and normalization first. Disable optional compilation to isolate compiler issues; use `--profile` to identify slow operations.

## Resume fails

`--resume` needs `last.ckpt` in the output directory and a compatible scientific configuration. A `.pme` snapshot is for inference. Use a new output directory after changing data, model or objective settings; see [training](training.md).

## Export or reference comparison fails

Use `spinnme inspect inversion.pme` to choose a saved grid. For another FITS grid, supply complete WCS and observer metadata plus an instrument ID present in the snapshot. `--time` changes model time while retaining that geometry; use a new reference geometry when the observer changes.

Reference comparisons require matching grids, epochs and units. Align references explicitly and retain quality masks. Use NPZ for several parameters and FITS/NPZ for spectra; JPG supports scalar previews. Existing outputs require `--overwrite`.

For an unexpected transverse-field difference, check the unresolved 180-degree azimuth branch before comparing signed components. For velocity differences between instruments, reconcile observer motion, rotation and wavelength zero points first.
