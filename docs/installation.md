# Installation

Use **Python 3.11 or newer**. A CPU is sufficient for the offline example. Training automatically detects an available GPU accelerator and falls back to CPU; larger inversions benefit from a GPU.

## Install with pip

Use a separate environment to keep dependencies together. For a POSIX shell (Linux/macOS):

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Then install and check the command in that environment:

```bash
pip install spinnme
spinnme --version
```

If you need to download observations or calibration profiles:

```bash
pip install 'spinnme[download]'
```

The `download` extra installs the HMI and Solar Orbiter archive clients. The base installation is sufficient if you already have observations and calibrations.

For CUDA, install the PyTorch build matching your machine using the [PyTorch installation selector](https://pytorch.org/get-started/locally/) before installing SPINNME. Check GPU access with:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

## Optional features

| Feature | Install |
| --- | --- |
| Circular-pattern data generation | `pip install 'spinnme[synthetic]'` |
| Weights & Biases logging | `pip install 'spinnme[wandb]'` |
| Documentation and development tools | `pip install 'spinnme[docs,dev]'` |

## Install from source

For contributing or editing the package, create an environment and install an editable checkout:

```bash
git clone https://github.com/RobertJaro/spinnme.git
cd spinnme
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e '.[download]'
```

Continue with the [HMI quickstart](quickstart.md), the [offline example](circular.md), or [your own FITS data](custom.md).
