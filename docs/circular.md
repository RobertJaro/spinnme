# Offline example

The same application is available as a [Colab notebook](colab.md), including a known-truth comparison.

Generate a circular-pattern solar Stokes scene, train and export FITS (GPU selected automatically when available; CPU also works) without archive access:

```bash
pip install 'spinnme[synthetic]'
spinnme synthetic --output ./demo/circular --size 32 --frames 2
spinnme run --config ./demo/circular/config.yaml \
  --run_path ./demo/circular/run --work_path ./demo/circular/run/cache --export fits
```

The generator writes Stokes FITS, known truth and a runnable YAML. Training uses five optimizer steps to check the workflow, not to obtain a converged inversion.

For a high-frequency sequence with controlled exposure windows, noise and cadence variants, use `spinnme synthetic --config dataset.yaml --output ./data/synthetic`. The dataset YAML contains `generation` and `acquisition` settings, separate from inversion settings. This mode writes dense truth in `dataset.json` and one input stream per noise realization in the shell-readable `acquisitions.tsv`.

Results are under `./demo/circular/run/`:

| Output | Use |
| --- | --- |
| `inversion.pme` | Load the fitted model and export new products |
| `last.ckpt` | Resume training |
| `resolved.yaml` | Inspect the run settings |
| `exports/` | Br, Btheta and Bphi FITS for each saved grid |

```{figure} _static/circular-overview.png
:alt: Circular-pattern magnetic truth used to generate the offline Stokes data

Generated magnetic truth, not the five-step fitted output. A short workflow check need not reproduce this pattern.
```

View the **fitted** line-of-sight field as JPG previews:

```bash
spinnme export ./demo/circular/run/inversion.pme --all-observations \
  --parameters Blos --format jpg --output ./demo/preview
```

Use the [fit-quality checks](evaluation.md#check-the-stokes-fit-before-comparing-maps) before interpreting the result. JPGs are previews; use FITS or NPZ for analysis.

Inspect the saved grids and write an NPZ bundle:

```bash
spinnme inspect ./demo/circular/run/inversion.pme
spinnme export ./demo/circular/run/inversion.pme \
  --all-observations --parameters Brtp Blos Vdop --format npz --output ./demo/npz
```

Continue with [HMI observations](quickstart.md), [custom FITS](custom.md), or [Python exports](python_exports.md).
