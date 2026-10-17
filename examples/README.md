# Runnable examples

For an online walkthrough, use the [Colab notebooks](colab/README.md): circular pattern, HMI, Hinode SP, PHI and custom FITS, each with basic evaluation and exports.

[Run hmi.sh](hmi.sh) for the four-step HMI workflow. Set `JSOC_EMAIL` first. Use [hmi_subframe.sh](hmi_subframe.sh) to crop and invert an HMI region. Use [hinode.sh](hinode.sh) for a prepared two-line raster or [phi_hrt.sh](phi_hrt.sh) for PHI downloads, training and exports.

All run configurations are installed with the package; use `spinnme config list`
from any directory. See the linked public guides for calibration and FITS details.

| Data | Packaged template | Guide |
| --- | --- | --- |
| HMI cropped Stokes | `hmi/subframe` | [Subframe workflow](../docs/hmi.md#invert-an-hmi-subframe) |
| HMI full-disk Stokes | `hmi/full_disk` | [Quickstart](../docs/quickstart.md), [HMI](../docs/hmi.md) |
| Offline circular pattern | `synthetic/circular` | [Circular example](../docs/circular.md) |
| Prepared Hinode/SP raster (6301 + 6302 Å) | `hinode/sp` | [Hinode](../docs/hinode.md) |
| PHI-HRT or PHI-FDT | `phi/hrt`, `phi/fdt` | [PHI](../docs/phi.md) |
| Custom Stokes cube | `generic/stokes_cube` | [Custom data](../docs/custom.md) |
| Two lines with different sampling | `generic/multiple_lines` | [Instrument setup](../docs/custom.md) |

The small offline example downloads nothing:

```bash
spinnme synthetic --output ./circular --size 32 --frames 2
spinnme run --config ./circular/config.yaml --run_path ./runs/circular --export fits
```

For a joint custom inversion, supply two calibrated cubes with correct WCS and
`WAVELNnn` metadata. This example uses Fe I 6173 and 6302 with independently sized
grids, shared B/v and separate ME nuisance heads:

```bash
spinnme config copy generic/multiple_lines --output joint.yaml
# Set both datasets and the validation input in joint.yaml.
spinnme run --config joint.yaml --run_path ./runs/joint
```

Set each cube's NumPy `axis_order`, Stokes conventions and normalization through
the configuration. A joint fit assumes the selected lines constrain a common
physical field; assess formation-height and calibration differences explicitly.
Different spectral sample counts require distinct instrument/setup identifiers;
sampling values may vary between acquisitions of a fixed setup.

Generate custom FITS from Python with [write_custom_fits.py](write_custom_fits.py),
then run `generic/stokes_cube` with its declared axes and spectral metadata.

For Python exports, [export_python.py](export_python.py) evaluates physical maps
on resized/cropped WCS grids and optionally at a regular cadence:

```bash
python examples/export_python.py runs/hmi/inversion.pme --output products/region --bounds -250 -200 250 200 --shape 256 384
```

Add `--cadence 30` for a time-series snapshot. See the [Python export guide](../docs/python_exports.md)
for fixed versus changing observer geometry, source masks and all output formats.
