"""Spherical Milne–Eddington inversions and portable solar data products."""

__version__ = "0.1.0"


def load(path, *, device=None):
    """Load a versioned ``.pme`` inference snapshot; no source data is required."""
    from .snapshots import Snapshot
    return Snapshot(path, device=device)


def run(config, **kwargs):
    """Run a spherical inversion from a configuration mapping or YAML path."""
    from .configuration import resolve_config
    from .training.runner import run as run_inversion
    return run_inversion(resolve_config(config), **kwargs)


__all__ = ["__version__", "load", "run"]
