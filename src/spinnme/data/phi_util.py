"""Calibrated PHI continuum metadata."""
import numpy as np


def phi_continuum_index(header, wavelengths):
    """Resolve a calibrated one-based CONTPOS index or an isolated endpoint."""
    wave = np.asarray(wavelengths)
    if len(wave) < 2:
        raise ValueError('PHI continuum identification needs multiple wavelength samples.')
    value = header.get('CONTPOS')
    if value is not None:
        try:
            index = float(value)
        except (TypeError, ValueError):
            index = np.nan
        if isinstance(value, (bool, np.bool_)) or not np.isfinite(index) or not index.is_integer() or not 1 <= index <= len(wave):
            raise ValueError('CONTPOS must be a calibrated one-based integer wavelength index.')
        return int(index) - 1
    # A continuum sample is the isolated endpoint, on either wing. Its array
    # location is preserved; no sorting changes the data/wavelength pairing.
    sorted_indices = np.argsort(wave)
    sorted_wave = wave[sorted_indices]
    gaps = np.diff(sorted_wave)
    threshold = 1.5 * np.median(gaps)
    endpoint = 0 if gaps[0] > gaps[-1] else -1
    if gaps[endpoint] <= threshold or gaps[endpoint] < gaps.max():
        raise ValueError('Ambiguous PHI continuum; provide a valid calibrated CONTPOS index.')
    if endpoint == 0:
        return int(sorted_indices[0])
    return int(sorted_indices[-1])
