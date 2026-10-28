"""Combine calibrated Hinode/SP slit files with a validated raster WCS.

No disk-center, Earth observer, scan timing, or spatial WCS is guessed from
incomplete Level-1 metadata. Supply the validated raster header explicitly.
"""
from pathlib import Path


import numpy as np
from astropy import units as u
from astropy.io import fits

from .fits import parse_time, spatial_header


def prepare_hinode_raster(files, output, spatial_wcs_header, *, wavelength_center=None,
                          wavelength_unit="Angstrom", data_hdu=0, wavelength_slice=None,
                          stokes_order="IQUV", overwrite=False):
    """Write a spherical-ready cube and per-pixel scan-time offsets.

    Input slit order is sorted by SLITINDX when present, otherwise the supplied
    order is preserved. Input shapes are (Stokes,y,wavelength) or
    (Stokes,CCD-side,y,wavelength), where a single CCD side is required.
    Spectral FITS coordinates use the one-based CRPIX convention without an
    ad-hoc wavelength sign reversal. All slit wavelength grids must agree.
    Both Fe I lines are retained by default; the run config selects each line's
    spectral window. wavelength_center is only needed for a single-line raster.
    """
    order = list(stokes_order)
    if sorted(order) != sorted('IQUV'):
        raise ValueError('stokes_order must contain I, Q, U and V exactly once.')
    records = []
    for path in files:
        with fits.open(path, memmap=False) as hs:
            h, raw = hs[data_hdu].header.copy(), np.asarray(hs[data_hdu].data)
        if raw.ndim == 4 and raw.shape[1] == 1:
            raw = raw[:, 0]
        if raw.ndim != 3 or raw.shape[0] != 4:
            raise ValueError(f"Unsupported Hinode slit shape {raw.shape} in {path}.")
        # The prepared raster is a reusable canonical cube. Downstream loaders
        # must not need to recover the original detector polarization order.
        raw = raw[[order.index(component) for component in 'IQUV']]
        count = raw.shape[-1]
        waves = float(h['CRVAL1']) + (np.arange(count)+1-float(h['CRPIX1']))*float(h['CDELT1'])
        waves = (waves*u.Unit(h.get('CUNIT1', wavelength_unit))).to_value(u.Unit(wavelength_unit))
        selected = slice(*wavelength_slice) if wavelength_slice is not None else slice(None)
        raw, waves = raw[..., selected], waves[selected]
        if not len(waves):
            raise ValueError('wavelength_slice selected no Hinode samples.')
        observed = h.get('DATE-OBS', h.get('DATE_OBS'))
        if observed is None:
            raise ValueError(f"Missing Hinode slit acquisition time in {path}.")
        records.append((h.get('SLITINDX'), raw, waves,
                        parse_time(observed, str(h.get('TIMESYS', 'UTC')).lower()),
                        str(Path(path).resolve())))
    if not records:
        raise ValueError("No Hinode slit files were supplied.")
    if all(record[0] is not None for record in records):
        records.sort(key=lambda record: int(record[0]))
        if len({record[0] for record in records}) != len(records):
            raise ValueError("Duplicate Hinode SLITINDX values.")
        steps = np.diff([int(record[0]) for record in records])
        if len(steps) and not np.all(steps == steps[0]):
            raise ValueError('Irregular Hinode slit indices cannot be assigned a uniform raster WCS. '
                             'Restore missing slit samples or explicitly remap before preparation.')
    for record in records[1:]:
        if record[1].shape != records[0][1].shape or not np.allclose(record[2], records[0][2], rtol=0, atol=1e-6):
            raise ValueError("Hinode slits must have matching shape and wavelength calibration.")
    # Stack x,stokes,y,wavelength -> stokes,wavelength,y,x for FITS x,y axes.
    cube = np.stack([record[1] for record in records], axis=-1).transpose(0, 2, 1, 3)
    header = fits.Header(spatial_wcs_header)
    if 'DATE-OBS' not in header:
        raise ValueError("The validated raster WCS header must include its DATE-OBS epoch.")
    time_scale = str(header.get('TIMESYS', 'UTC')).lower()
    origin = parse_time(header['DATE-OBS'], time_scale)
    times = np.array([(record[3]-origin).total_seconds() for record in records])
    header = spatial_header(header, cube.shape[-2:])
    if wavelength_center is not None:
        header['WAVELNTH'] = wavelength_center
    else:
        header.remove('WAVELNTH', ignore_missing=True)
    header['WAVEUNIT'] = wavelength_unit
    for index, wave in enumerate(records[0][2], start=1):
        header[f'WAVELN{index:02d}'] = float(wave)
    header['INSTRUME'] = 'SOT/SP'
    header['TELESCOP'] = 'Hinode'
    header['SPNAXES'] = 'stokes,wavelength,y,x'
    header['SPNSTOK'] = 'IQUV'
    # All newly written dates are UTC, matching the scan offsets and preserving
    # the physical raster epoch even when its supplied header used TAI.
    for key in ('DATE-OBS', 'DATE-BEG', 'DATE-AVG', 'DATE-END', 'DATE_EAR', 'DATE_SUN'):
        if key in header:
            header[key] = parse_time(header[key], time_scale).isoformat()
    header['TIMESYS'] = 'UTC'
    header['DATE-END'] = max(record[3] for record in records).isoformat()
    offsets = np.broadcast_to(times, cube.shape[-2:]).astype(np.float64)
    time_hdu = fits.ImageHDU(offsets, name='TIME_OFFSETS')
    time_hdu.header['BUNIT'] = 's'
    provenance = fits.BinTableHDU.from_columns([
        fits.Column(name='SOURCE', format=f'{max(len(r[4]) for r in records)}A', array=[r[4] for r in records]),
        fits.Column(name='DATE_OBS', format='32A', array=[r[3].isoformat() for r in records]),
    ], name='SOURCES')
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([fits.PrimaryHDU(cube.astype(np.float32), header), time_hdu, provenance]).writeto(output, overwrite=overwrite)
    return output
