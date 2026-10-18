# Contributing

Install a development environment with Python 3.11 or later:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,download,synthetic,docs,wandb]'
python -m pytest
python -m build
sphinx-build -W --keep-going -b html docs docs/_build/html
```

Install the appropriate CPU/CUDA PyTorch distribution for your machine before
installing SPINNME. Keep machine-specific CUDA choices outside library dependency
metadata. CI runs a focused configuration, CLI, synthesis, spectral-response, and
Stokes-loss subset on Python 3.11 and 3.12, plus lint, package, installed-wheel
training/export, and documentation checks. The test file list is maintained in
`.github/workflows/ci.yml`. Run `python -m pytest` locally for the full scientific
suite, including ingestion, training, and validation plots; the latter require the
optional `wandb` dependency included above. Accelerator studies need separate
validation.

Changes to coordinate frames, signs, normalization, synthesis, losses, response
models, or snapshots need scientific regression tests. Include numerical parity
or an explicit explanation of a deliberate method change. Test differently sized
wavelength grids and multiple instruments when changing dataset or training APIs.

Keep raw mission observations, credentials, caches, model snapshots, generated
figures and scheduler-specific workflows out of the distribution. Small synthetic
fixtures belong in the package tests. Paper studies and large validation workflows
belong in the separate `spinnme-evaluation` repository.

New instruments must declare atomic-line properties, wavelength units/order,
polarization reference, normalization, observer/time conventions, and response
metadata. Correct input WCS is required; repairs must be explicit preparation
operations. Instrument contributions must include ingestion, geometry, synthesis
and snapshot/export tests.

Before a release, build and inspect both wheel and source distribution, test the
installed wheel outside the checkout, run the documented CPU circular example,
and build the documentation with warnings treated as errors. Publishing to PyPI
and activating Read the Docs are maintainer actions requiring account access.
See [RELEASE.md](RELEASE.md) for the publishing setup and release procedure.
