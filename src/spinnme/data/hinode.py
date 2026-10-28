"""Read a calibrated Hinode/SP slit scan directly into an in-memory raster."""
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from sunpy.coordinates import frames
from sunpy.coordinates.ephemeris import get_earth

from .fits import observer_velocity_from_header, parse_time, wavelength_configuration


def slit_coordinates(header, length):
    """SolarSoft stksimages_sbsp slit geometry; angles are clockwise from north.

    SPCCDIX* use pre-launch detector coordinates. Level-1 reverses the slit
    direction; retain the full-CCD origin (511.5), including ROI and binning.
    These spatial angles are independent of the calibrated Q/U reference.
    """
    required = ('CROTA2', 'FGSPROT', 'SPCCDIX0', 'SPCCDIX1', 'CAMSSUM',
                'XCEN', 'YCEN', 'YSCALE')
    missing = [key for key in required if key not in header]
    if missing:
        raise ValueError(f'Native Hinode slit geometry requires {missing}; supply calibrated Level-1 metadata.')
    if not all(np.isfinite(float(header[key])) for key in required):
        raise ValueError('Hinode slit coordinate metadata must be finite.')
    binning = float(header['CAMSSUM'])
    first, last = float(header['SPCCDIX0']), float(header['SPCCDIX1'])
    if (binning < 1 or not binning.is_integer() or not first.is_integer() or not last.is_integer() or first < 0 or last > 1023 or
            first > last or last-first+1 != length*binning or float(header['YSCALE']) <= 0):
        raise ValueError('Hinode detector ROI/binning do not match the slit length.')
    angle = np.deg2rad(float(header['CROTA2']) + float(header['FGSPROT']))
    origin = (1023-last-511.5)/binning
    dx, dy = np.sin(angle)*float(header['YSCALE']), np.cos(angle)*float(header['YSCALE'])
    return [float(header['XCEN'])+dx*origin, float(header['YCEN'])+dy*origin, dx, dy], angle


def read_hinode_slits(files, *, line_id='fe-i-6302', wavelength_center=None,
                       wavelength_range=None):
    """Retain native spectra, slit positions and UTC times without writing FITS."""
    center = wavelength_center or {'fe-i-6301': 6301.5012, 'fe-i-6302': 6302.4931}[line_id]
    if wavelength_range is None:
        wavelength_range = [None, 6301.99715] if line_id == 'fe-i-6301' else [6301.99715, None]
    records = []
    for path in files:
        header = fits.getheader(path)
        records.append((int(header['SLITINDX']), Path(path), header))
    records.sort(key=lambda row: row[0])
    if len({row[0] for row in records}) != len(records):
        raise ValueError('Duplicate Hinode slit indices; supply one scan folder per dataset.')
    epoch = min(parse_time(h.get('DATE-OBS', h.get('DATE_OBS'))) for _, _, h in records)
    spectra, offsets, positions, dates, rolls = [], [], [], [], []
    reference_waves = None
    for _, path, header in records:
        raw = fits.getdata(path).astype(np.float32)
        if raw.ndim == 4 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 3 or raw.shape[0] != 4:
            raise ValueError(f'Expected a calibrated I/Q/U/V slit, got {raw.shape} in {path}.')
        waves = (header['CRVAL1'] + (np.arange(raw.shape[-1])+1-header['CRPIX1'])*header['CDELT1'])
        waves = (waves*u.Unit(header.get('CUNIT1', 'Angstrom'))).to_value(u.AA)
        lower, upper = wavelength_range
        selected = (waves >= (-np.inf if lower is None else lower)) & (waves < (np.inf if upper is None else upper))
        if not selected.any():
            raise ValueError('wavelength_range selected no Hinode spectral samples.')
        waves = waves[selected]
        if reference_waves is not None and (waves.shape != reference_waves.shape or
                                            not np.allclose(waves, reference_waves, rtol=0, atol=1e-6)):
            raise ValueError('Hinode slits must have matching wavelength calibration.')
        reference_waves = waves
        spectra.append(raw.transpose(1, 0, 2))
        observed = parse_time(header.get('DATE-OBS', header.get('DATE_OBS')))
        dates.append(observed)
        offsets.append((observed-epoch).total_seconds())
        position, roll = slit_coordinates(header, raw.shape[1])
        positions.append(position)
        rolls.append(roll)
    stokes = np.stack(spectra, axis=1)
    ny, nx = stokes.shape[:2]
    first = records[0][2]
    observers = get_earth(Time(dates))
    carrington = observers.transform_to(frames.HeliographicCarrington(observer=observers, obstime=Time(dates)))
    slit_geometry = {'obs_lon': carrington.lon.to_value(u.rad),
                     'obs_lat': carrington.lat.to_value(u.rad),
                     'pAng': np.asarray(rolls),
                     'observer_distance': observers.radius.to_value(u.m)/695700000.,
                     'observer_velocity': np.array([observer_velocity_from_header(h) for _, _, h in records])}
    earth = observers[int(np.argmin(offsets))]
    header = fits.Header({'CTYPE1': 'HPLN-TAN', 'CTYPE2': 'HPLT-TAN',
        'CUNIT1': 'arcsec', 'CUNIT2': 'arcsec', 'CRPIX1': (nx+1)/2, 'CRPIX2': (ny+1)/2,
        'CRVAL1': float(np.mean([p[0]+p[2]*(ny-1)/2 for p in positions])),
        'CRVAL2': float(np.mean([p[1]+p[3]*(ny-1)/2 for p in positions])),
        'CDELT1': float(first['XSCALE']), 'CDELT2': float(first['YSCALE']),
        'CROTA2': -np.rad2deg(rolls[0]), 'DATE-OBS': epoch.isoformat(), 'TIMESYS': 'UTC',
        'DSUN_OBS': earth.radius.to_value(u.m), 'HGLN_OBS': 0., 'HGLT_OBS': earth.lat.to_value(u.deg),
        'RSUN_REF': 695700000., 'TELESCOP': 'Hinode', 'INSTRUME': 'SOT/SP', 'WAVEUNIT': 'Angstrom'})
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        if key in first:
            header[key] = first[key]
    positions = np.asarray(positions).T
    return {'stokes': stokes[..., selected], 'source_stokes': stokes,
            'wavelength_indices': np.flatnonzero(selected), 'header': header, 'model_time': epoch,
            'wavelength_config': wavelength_configuration(header, len(reference_waves),
                wavelength_center=center, wavelength_grid=reference_waves, wavelength_grid_is_offsets=False),
            'pixel_time_offsets': np.broadcast_to(offsets, (ny, nx)), 'slit_pixel_world': positions, 'slit_geometry': slit_geometry,
            'metadata': {'source_paths': [str(p.resolve()) for _, p, _ in records],
                         'dataset_type': 'hinode', 'line_id': line_id,
                         'polarization_basis': 'solar west/north; +Q east-west (calib_sbsp operation 9)',
                         'spatial_coordinate_convention': 'SolarSoft stksimages_sbsp ROI/binning and CROTA2+FGSPROT',
                         'observer_approximation': 'Earth at each slit acquisition time',
                         'slit_indices': [i for i, _, _ in records],
                         'slit_pixel_world': positions.tolist(),
                         'slit_geometry': {key: value.tolist() for key, value in slit_geometry.items()},
                         'slit_coordinate_headers': [{key: h[key] for key in
                             ('XCEN', 'YCEN', 'XSCALE', 'YSCALE', 'CROTA2', 'FGSPROT', 'SPCCDIX0', 'SPCCDIX1', 'CAMSSUM', 'CRPIX2', 'SLITINDX',
                              'DATE-OBS', 'DATE_OBS', 'TIMESYS', 'OBS_VR', 'OBS_VW', 'OBS_VN') if key in h}
                             for _, _, h in records],
                         'time': {'model_time': epoch.isoformat(), 'time_reference': 'observation'}}}
