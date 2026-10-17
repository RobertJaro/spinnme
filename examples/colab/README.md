# Colab notebooks

Run cells from top to bottom. Select a GPU runtime if available; device selection is automatic. Each notebook saves one run YAML, performs a small inversion, checks Stokes residuals and exports FITS/NPZ/JPG. The last cell downloads a ZIP of the fitted model and results.

| Notebook | Open in Colab | Input |
| --- | --- | --- |
| [Circular pattern](circular.ipynb) | [Open](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/circular.ipynb) | Generated; includes known-truth comparison |
| [HMI](hmi.ipynb) | [Open](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/hmi.ipynb) | Archive acquisition, native subframe and transmission profiles |
| [Hinode SP](hinode_sp.ipynb) | [Open](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/hinode_sp.ipynb) | Prepared two-line Stokes raster upload or direct URL |
| [PHI](phi.ipynb) | [Open](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/phi.ipynb) | HRT/FDT archive Stokes product |
| [Custom FITS](custom_fits.ipynb) | [Open](https://colab.research.google.com/github/RobertJaro/spinnme/blob/main/examples/colab/custom_fits.ipynb) | Calibrated Fe I 6173 cube with editable reader settings |

Alternatively upload a downloaded `.ipynb` through Colab's File menu. HMI needs your registered JSOC email. Hinode accepts calibrated spectra, not Level-2 magnetic maps; prepare slit files using the instrument guide first. Verify PHI release-specific WCS and wavelength metadata.

Files under `/content` are temporary. Download results before ending the runtime. Keep inputs/calibration separately for resume. Use a fresh workspace when changing the data/crop and a fresh run name when changing the scientific setup. The trial budgets do not establish convergence.

See the [notebook guide](https://spinnme.readthedocs.io/en/stable/colab.html) for requirements and downloads. Notebooks use the public `pip install spinnme` release and contain no saved outputs or credentials.
