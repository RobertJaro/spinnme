"""Instrument wavelength definitions shared by loaders and synthetic data."""

import numpy as np
from astropy import units as u

HMI_WAVELENGTH_CENTER = 6173.3433 * u.AA

# hmi.S_720s segments I0..I5 are stored from the red to the blue wing.
# The nominal tuning separation is 68.8 mA and the outer positions are
# approximately +/-172 mA from the Fe I reference wavelength.
HMI_WAVELENGTH_GRID = np.array([
    0.1720,
    0.1032,
    0.0344,
    -0.0344,
    -0.1032,
    -0.1720,
]) * u.AA


def hmi_wavelength_config():
    """Return independent quantities so callers cannot mutate the constants."""
    return {
        'wavelength_center': HMI_WAVELENGTH_CENTER.copy(),
        'wavelength_grid': HMI_WAVELENGTH_GRID.copy(),
    }


# Air rest wavelengths; sampling and spectral response remain observations.
# Hinode transitions are z5P2 -> e5D2 (6301) and z5P1 -> e5D0 (6302),
# with lower/upper levels as tabulated in Okunev (2004), Table 4.1:
# https://ediss.uni-goettingen.de/bitstream/11858/00-1735-0000-0006-B4F7-D/1/okunev.pdf
# 6301 uses LS-coupling g factors (11/6, 3/2). For 6302 retain the
# adopted g_eff=2.49 convention (rather than the pure-LS value 2.50), see
# https://science-media.org/userfiles/10/papers/10_paper_930.pdf, Table 2.3.
# Its J=0 upper level is unsplit, so g_eff belongs to the lower J=1 level.
SPECTRAL_LINES = {
    'fe-i-6173': {'wavelength_center': 6173.3433 * u.AA, 'j_up': 1., 'j_low': 0., 'g_up': 2.50, 'g_low': 0.},
    'fe-i-6301': {'wavelength_center': 6301.5012 * u.AA, 'j_up': 2., 'j_low': 2., 'g_up': 1.50, 'g_low': 1.8333333333333333},
    'fe-i-6302': {'wavelength_center': 6302.4931 * u.AA, 'j_up': 0., 'j_low': 1., 'g_up': 0., 'g_low': 2.49},
}


def resolve_instrument_line(instrument, wavelength_config):
    """Validate a line and return its identity and solver atomic constants.

    Custom lines require explicit atomic parameters. Each instrument identifier
    names one line/response; use separate identifiers to fit multiple lines.
    """
    instrument_type = instrument['type'].lower()
    default = 'fe-i-6173' if instrument_type in ('hmi', 'phi') else None
    line_id = instrument.get('line_id', instrument.get('line', default))
    supplied = instrument.get('atomic_parameters', instrument.get('line_config'))
    if supplied is None and any(key in instrument for key in ('j_up', 'j_low', 'g_up', 'g_low')):
        supplied = {key: instrument[key] for key in ('j_up', 'j_low', 'g_up', 'g_low') if key in instrument}
    center = wavelength_config['wavelength_center'].to_value(u.AA)
    if line_id is None:
        if supplied is None:
            raise ValueError('Generic instruments require line_id and atomic_parameters, or a bundled line.')
        line_id = 'custom-' + str(instrument['instrument_id'])
    preset = SPECTRAL_LINES.get(line_id)
    if preset is None and supplied is None:
        raise ValueError(f'Unknown spectral line {line_id!r}; supply atomic_parameters for a custom line.')
    atomic = {key: float(value) for key, value in (supplied or preset).items() if key != 'wavelength_center'}
    required = {'j_up', 'j_low', 'g_up', 'g_low'}
    if atomic.keys() != required or not all(np.isfinite(value) for value in atomic.values()):
        raise ValueError(f'atomic_parameters must contain finite values for {sorted(required)}.')
    if preset is not None and not np.isclose(center, preset['wavelength_center'].to_value(u.AA), atol=.05, rtol=0):
        raise ValueError(f'Wavelength center {center} Angstrom does not match {line_id}.')
    if any(atomic[key] < 0 or not np.isclose(2 * atomic[key], round(2 * atomic[key])) for key in ('j_up', 'j_low')):
        raise ValueError('Atomic angular momenta must be nonnegative integer or half-integer values.')
    if (instrument_type in ('hmi', 'phi') and supplied is not None and preset is not None
            and any(not np.isclose(atomic[key], float(preset[key])) for key in required)):
        raise ValueError(f'{instrument_type} atomic parameters must match the bundled Fe I 6173 model.')
    if instrument_type in ('hmi', 'phi') and line_id != 'fe-i-6173':
        raise ValueError(f'{instrument_type} requires the Fe I 6173 line.')
    return line_id, atomic
