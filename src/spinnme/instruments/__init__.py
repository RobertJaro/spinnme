"""Bundled instrument definitions, available after wheel installation."""
from importlib.resources import files
import yaml


def load_instrument_definition(name):
    """Return a fresh portable definition for hmi, phi_hrt, phi_fdt or hinode."""
    if name not in {'hmi', 'phi_hrt', 'phi_fdt', 'hinode'}:
        raise ValueError(f'Unknown instrument definition {name!r}.')
    return yaml.safe_load((files(__package__) / 'definitions' / f'{name}.yaml').read_text())


__all__ = ['load_instrument_definition']
