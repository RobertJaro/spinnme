# Colab notebooks

Open a notebook, choose **Runtime → Change runtime type → GPU** if available, then run the cells in order. SPINNME selects devices automatically and falls back to CPU. No repository checkout or W&B account is needed.

| Application | Open online | Download notebook | Input |
| --- | --- | --- | --- |
| First inversion and known-truth comparison | [Colab](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/circular.ipynb) | {download}`Circular pattern <../examples/colab/circular.ipynb>` | Generated in the notebook; no archive account |
| HMI subframe | [Colab](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/hmi.ipynb) | {download}`HMI <../examples/colab/hmi.ipynb>` | One HMI acquisition plus separate transmission download; registered JSOC email |
| Hinode SP, both 6301/6302 lines | [Colab](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/hinode_sp.ipynb) | {download}`Hinode SP <../examples/colab/hinode_sp.ipynb>` | Uploaded prepared Stokes raster, or its direct URL |
| PHI HRT or FDT | [Colab](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/phi.ipynb) | {download}`PHI <../examples/colab/phi.ipynb>` | Archive Stokes acquisition with verified release metadata |
| Custom calibrated FITS | [Colab](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/custom_fits.ipynb) | {download}`Custom FITS <../examples/colab/custom_fits.ipynb>` | Uploaded Fe I 6173 cube with explicit reader settings |

You can also download a notebook and use Colab's **File → Upload notebook**. For Hinode slit files, first [assemble a calibrated raster](hinode.md#assemble-calibrated-slit-files). Other lines/instruments need a [matching configuration](custom.md).

Each notebook installs SPINNME, prepares its input, saves **one YAML per run**, fits a small trial, plots magnetic/velocity maps, compares observed and synthesized I/Q/U/V, and writes FITS, NPZ, JPG and summary metrics. The Hinode notebook checks both lines; the circular example also compares field strength with known truth. Increase the budget after assessing residuals; the short trials do not establish scientific convergence or parameter uncertainty.

The final cell downloads a ZIP with `.pme`, checkpoint, YAML, metrics and exports. Runtime files under `/content` are temporary and [Colab resources are limited](https://research.google.com/colaboratory/faq.html#resource-limits). Keep observations/calibration separately if you need to resume later; they are excluded from the ZIP. HMI's initial acquisition download still contains all 24 full-disk segments before cropping.

For larger studies use the same saved run YAML with the [command-line workflow](training.md). See the [beginner FAQ](faq.md), [spectral-fit checks](evaluation.md) and [Python export recipes](python_exports.md) for the next step.
