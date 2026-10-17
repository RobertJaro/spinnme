"""Portable deterministic circular-pattern spherical FITS fixtures."""
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import frames

from spinnme.instrument import hmi_wavelength_config

from .create_spherical_test_set import SphericalTestSetGenerator


def generate_circular(output_directory, *, shape=(32, 32), n_time_steps=1,
                      start="2025-01-01T00:00:00", cadence_seconds=90,
                      observer_longitude=0, observer_latitude=0, observer_distance_au=1,
                      wavelength_center=None, wavelength_grid=None,
                      pattern_resolution=(180, 180), seed=0):
    """Write canonical cubes, HMI-style segment fixtures, and true parameters.

    Runs serially and on CPU; no cluster paths, plotting dependencies, or
    Cartesian inversion modules are needed. Each observer can be generated
    into its own directory for scientific multi-observer studies. An existing
    generated collection is never overwritten or mixed with a new sequence.
    """
    if len(shape) != 2 or min(shape) < 4 or n_time_steps < 1:
        raise ValueError("shape must be at least (4,4) and n_time_steps positive.")
    # The analytic pattern is deterministic; seed is recorded for downstream
    # noise experiments rather than altering the noiseless truth.
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.glob('circular_*.fits*')) or any(output.glob('parameters_*.npz')):
        raise FileExistsError('Circular output already contains generated data. '
                              'Choose a new directory to avoid mixing scientific inputs.')
    wave = hmi_wavelength_config()
    if wavelength_center is not None:
        wave['wavelength_center'] = float(wavelength_center) * u.AA
    if wavelength_grid is not None:
        wave['wavelength_grid'] = np.asarray(wavelength_grid) * u.AA
    generator = SphericalTestSetGenerator(nx=int(shape[0]), ny=int(shape[1]),
                    wavelength_center=wave['wavelength_center'], wavelength_grid=wave['wavelength_grid'],
                    pattern_resolution=pattern_resolution, tile_rows=2, tile_cols=4)
    paths = []
    for index in range(n_time_steps):
        time = Time(start) + index * cadence_seconds * u.s
        observer = SkyCoord(lon=observer_longitude*u.deg, lat=observer_latitude*u.deg,
                            radius=observer_distance_au*u.AU, obstime=time,
                            frame=frames.HeliographicStonyhurst)
        profiles, parameters, reference = generator.load_spherical_time_step(index, observer)
        header = reference.fits_header
        for name in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
            header[name] = 0.0
        header['WAVELNTH'] = wave['wavelength_center'].to_value(u.AA)
        header['WAVEUNIT'] = 'Angstrom'
        header['SPNSEED'] = int(seed)
        header['INSTRUME'] = 'SPINNME-CIRCULAR'
        for wl, offset in enumerate(wave['wavelength_grid'].to_value(u.AA)):
            header[f'WAVELN{wl+1:02d}'] = float(header['WAVELNTH'] + offset)
            for stokes_index, component in enumerate('IQUV'):
                path = output / f'circular_{index:03d}.{component}{wl}.fits'
                fits.writeto(path, profiles[..., stokes_index, wl], header, overwrite=True)
        cube = output / f'circular_{index:03d}.fits'
        fits.writeto(cube, profiles.transpose(2, 3, 0, 1), header, overwrite=True)
        truth = {key: value.detach().cpu().numpy() if hasattr(value, 'detach') else value
                 for key, value in parameters.items()}
        np.savez_compressed(output / f'parameters_{index:03d}.npz', **truth)
        paths.append(str(cube))
    return paths
