"""Spherical Milne–Eddington inversions and portable solar data products."""

__version__ = "0.1.0"


def load(path, *, device=None):
    """Load a versioned ``.pme`` inference snapshot; no source data is required."""
    from .snapshots import Snapshot
    return Snapshot(path, device=device)


def run(config, *, resume=False, profile=False, reload=False):
    """Train from a full configuration dictionary, YAML path or bundled preset.

    Dictionary inputs are copied and validated without writing an input YAML.
    All scientific and training settings belong in ``config``. The runner saves
    the resolved settings with the results for provenance and explicit resume.
    Returns ``(module, data_module, trainer)``. ``reload=True`` rebuilds prepared
    data; ``profile=True`` enables diagnostics.
    """
    from .configuration import resolve_config
    from .training.runner import run as run_inversion
    return run_inversion(resolve_config(config), resume=resume, profile=profile, reload=reload)


__all__ = ["__version__", "load", "run"]
