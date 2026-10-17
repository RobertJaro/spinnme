# HMI quickstart

For an online walkthrough, use the [HMI Colab notebook](colab.md).

Download HMI Stokes data, download its transmission profiles, run the bundled inversion config, and export maps. You need a registered JSOC email and internet access for the two downloads. Run the commands from one working directory; `./data`, `./runs` and `./products` are created there. For the smallest CPU-only installation check, use the [offline example](circular.md). New terminology is explained in the [FAQ](faq.md).

## 1. Download Stokes data

Install SPINNME with archive support (Python 3.11+):

```bash
pip install 'spinnme[download]'
export JSOC_EMAIL='you@example.org'
spinnme download hmi --product S_720s \
  --start 2024-03-23T01:00:00 --time-scale tai \
  --email "$JSOC_EMAIL" --output ./data/hmi/stokes
```

This selects one acquisition with all 24 I/Q/U/V segments. Use your registered address; see [JSOC registration](https://docs.sunpy.org/en/stable/tutorial/acquiring_data/jsoc.html#setup). Add `--end` to select a time series. Use `--product S_90s` for native 90-second data.

## 2. Download transmission profiles

```bash
spinnme download hmi-transmission --input ./data/hmi/stokes \
  --output ./data/hmi/transmission --email "$JSOC_EMAIL"
```

This reads the downloaded FITS metadata, retrieves the matching JSOC calibration records and writes transmission profiles plus their acquisition manifest. It does not load a run YAML, prepare training caches or change training settings.

## 3. Run the inversion

Training detects an available GPU accelerator automatically and falls back to CPU. No device flag is needed; larger fits can be substantially more expensive than the offline example.

```bash
spinnme run --config hmi/full_disk \
  --data_path ./data/hmi --run_path ./runs/hmi
```

`--data_path` is the parent directory containing `stokes/` and `transmission/`. The bundled YAML sets the model, **256 × 256** sampling, **1,000 training steps** and validation on the first observation. The run prepares its data cache automatically. Use `--device cpu` only when you want to force CPU execution.

The outputs are `./runs/hmi/inversion.pme`, `last.ckpt` and `resolved.yaml`. The example budget checks the workflow; it does not establish convergence. Validation on a fitted observation measures fit quality. Before using the maps scientifically, follow the [Stokes-fit checks](evaluation.md#check-the-stokes-fit-before-comparing-maps).

## 4. Export maps

```bash
spinnme export ./runs/hmi/inversion.pme \
  --all-observations --format fits --output ./products/hmi
```

Each observation directory contains `Br.fits`, `Btheta.fits` and `Bphi.fits`, with WCS, units and a validity mask. Off-disk pixels are NaN. The saved validation observation is also exported.

Add `--parameters Brtp Blos Vdop vmac damping` for more parameters, or choose `--format npz` or `--format jpg`. See [exports](exports.md) and [Python exports](python_exports.md) for resolution, cadence and subregion selection.

## Customize a run

```bash
spinnme config copy hmi/full_disk --output hmi.yaml
```

Edit that YAML for a different grid, training budget, model, validation choice or directory layout, then use `--config hmi.yaml`. For native spatial sampling, remove `resolution` from both training and validation datasets. See [training options](training_options.md) for the available choices.

To invert a smaller region, use the [HMI subframe CLI workflow](hmi.md#invert-an-hmi-subframe).

Next: [HMI details](hmi.md), [your own FITS data](custom.md), [Hinode](hinode.md), [PHI](phi.md), or the [offline example](circular.md).
