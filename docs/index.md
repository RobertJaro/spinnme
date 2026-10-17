# SPINNME documentation

```{raw} html
<div class="spinnme-hero">
<p class="spinnme-eyebrow">Solar spectropolarimetry</p>
<h2>Invert Stokes observations.<br>Export solar field maps.</h2>
<p>Run spherical Milne–Eddington inversions of HMI, Hinode, PHI or custom FITS data with Python or the <code>spinnme</code> command.</p>
</div>
```

| I want to… | Start here |
| --- | --- |
| Understand the inputs, outputs and terminology | [Beginner FAQ](faq.md) · [How the fit works](method.md) |
| Run an inversion online without local setup | [Colab notebooks](colab.md) |
| Download HMI data and run my first inversion | [HMI quickstart](quickstart.md) |
| Try the software without downloading observations | [Offline example](circular.md) |
| Invert my own FITS files | [Custom data and instruments](custom.md) |
| Use Hinode or PHI data | [Hinode](hinode.md) · [PHI](phi.md) |
| Change the model, training budget, sampling or loss | [Training options](training_options.md) |
| Export a different resolution, region or cadence | [Python exports](python_exports.md) |
| Decide whether a fit is usable | [Fit quality and comparisons](evaluation.md) |
| Fix a failed command or unexpected result | [Troubleshooting](troubleshooting.md) |

SPINNME fits a new model to your selected polarized spectra to estimate magnetic-field, velocity and atmosphere maps. You can use the presets without writing neural-network code.

Inputs must contain calibrated Stokes I/Q/U/V and correct solar WCS, observer metadata and wavelength sampling. HMI additionally requires transmission profiles, downloaded separately in the quickstart. See [input requirements](data.md) and [field conventions](method.md).

```{toctree}
:maxdepth: 1
:caption: Start

installation
quickstart
circular
faq
colab
method
```

```{toctree}
:maxdepth: 1
:caption: Invert your data

custom
data
cropping
hmi
hinode
phi
training
training_options
artifacts
```

```{toctree}
:maxdepth: 1
:caption: Export and reference

exports
python_exports
evaluation
plotting
configuration
cli
api
troubleshooting
```
