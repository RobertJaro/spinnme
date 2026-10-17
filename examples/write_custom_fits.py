"""Write an existing calibrated Stokes cube with caller-supplied solar geometry.

This helper does not fabricate data or repair WCS. Supply physical arrays and a
validated FITS spatial header, then select its axis order in generic/stokes_cube.
"""
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits


def write_stokes_cube(path, stokes, spatial_header, wavelengths, line_center, *, overwrite=False):
    """Write canonical (y,x,4,wavelength) profiles as (wavelength,Stokes,y,x) FITS."""
    stokes = np.asarray(stokes, dtype=np.float32)
    if stokes.ndim != 4 or stokes.shape[-2] != 4:
        raise ValueError('Stokes must have shape (y,x,4,wavelength).')
    wavelengths = u.Quantity(wavelengths).to_value(u.AA)
    if wavelengths.shape != (stokes.shape[-1],):
        raise ValueError('Supply one physical wavelength per spectral sample.')
    header = fits.Header(spatial_header)
    header['WAVELNTH'] = u.Quantity(line_center).to_value(u.AA)
    header['WAVEUNIT'] = 'Angstrom'
    for index, wavelength in enumerate(wavelengths, 1):
        header[f'WAVELN{index:02d}'] = float(wavelength)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.writeto(path, stokes.transpose(3, 2, 0, 1), header, overwrite=overwrite)
    return path
