"""Spatial cropping of FITS images/cubes with their aligned image extensions."""
from __future__ import annotations

import glob
import re
import tempfile
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS

_IMAGES = (fits.PrimaryHDU, fits.ImageHDU, fits.CompImageHDU)


def _files(inputs):
    inputs = [inputs] if isinstance(inputs, (str, Path)) else inputs
    paths = set()
    for value in inputs:
        value = Path(value).expanduser()
        if value.is_dir():
            found = [path for path in value.iterdir() if path.is_file()
                     and re.search(r'\.(fits?|fts)(\.(gz|fz))?$', path.name, re.IGNORECASE)]
        else:
            found = [Path(path) for path in glob.glob(str(value))]
        paths.update(path.resolve() for path in found if path.is_file())
    if not paths:
        raise ValueError('No FITS files matched --input.')
    return sorted(paths)


def _shape(hdu):
    return tuple(hdu.shape) if isinstance(hdu, _IMAGES) else ()


def _view(shape, axes, pixels):
    xaxis, yaxis = axes
    x0, y0, x1, y1 = pixels
    slices = [slice(None)]*len(shape)
    slices[-xaxis], slices[-yaxis] = slice(x0, x1), slice(y0, y1)
    return tuple(slices)


def _pixels(header, shape, axes, pixels, bounds, unit):
    width, height = shape[-axes[0]], shape[-axes[1]]
    if pixels is not None:
        x0, y0, x1, y1 = pixels
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f'Pixel crop must lie inside the input spatial grid [0,{width}) × [0,{height}).')
    else:
        for axis in axes:
            if not header.get(f'CTYPE{axis}'):
                raise ValueError('--bounds requires spatial WCS; use --pixels for an uncalibrated image.')
        wcs = WCS(header, naxis=list(axes))
        xmin, ymin, xmax, ymax = bounds
        # Bounds are numeric coordinates in the file's own WCS, without
        # assuming an instrument, observer, celestial frame or ephemeris.
        wx = (np.array([xmin, xmax, xmax, xmin])*u.Unit(unit)).to_value(u.Unit(wcs.world_axis_units[0]))
        wy = (np.array([ymin, ymin, ymax, ymax])*u.Unit(unit)).to_value(u.Unit(wcs.world_axis_units[1]))
        x, y = wcs.world_to_pixel_values(wx, wy)
        if not np.isfinite(x).all() or not np.isfinite(y).all():
            raise ValueError('World bounds do not map to finite image pixels.')
        x0, y0 = int(np.floor(np.min(x))), int(np.floor(np.min(y)))
        x1, y1 = int(np.ceil(np.max(x)))+1, int(np.ceil(np.max(y)))+1
        x0, y0, x1, y1 = max(0, x0), max(0, y0), min(width, x1), min(height, y1)
        if x0 >= x1 or y0 >= y1:
            raise ValueError('The requested world rectangle does not intersect the image.')
    return x0, y0, x1, y1


def _shift_header(header, shape, axes, pixels):
    """A crop shifts pixel reference positions; matrices and units stay intact."""
    header = header.copy()
    if any(key.startswith(('CPDIS', 'D2IMDIS', 'DET2IM')) for key in header) or any(
            '-TAB' in str(header.get(f'CTYPE{axis}', '')) for axis in axes):
        raise ValueError('Auxiliary distortion/table spatial WCS needs explicit reprojection before cropping.')
    for axis, start in zip(axes, pixels[:2]):
        keys = [key for key in header if re.fullmatch(fr'CRPIX{axis}[A-Z]?', key)]
        # Account for FITS' implicit zero reference pixel when a WCS exists.
        if not keys and any(f'{prefix}{axis}' in header for prefix in ('CTYPE', 'CRVAL', 'CDELT')):
            keys = [f'CRPIX{axis}']
        for key in keys:
            header[key] = float(header.get(key, 0))-start
    # Preserve known detector-coordinate metadata without imposing it on other
    # FITS files. Native HMI detector dimensions establish a missing origin.
    is_hmi = 'HMI' in str(header.get('INSTRUME', '')).upper() or 'HMI' in str(header.get('TELESCOP', '')).upper()
    for key, axis, start in zip(('CCD_X0', 'CCD_Y0'), axes, pixels[:2]):
        if key in header:
            header[key] = float(header[key])+start
        elif is_hmi and (shape[-axes[1]], shape[-axes[0]]) == (4096, 4096):
            header[key] = start
    # PHI origins are one-based full-detector pixels, before binning.
    for index, start in enumerate(pixels[:2], 1):
        key = f'PXBEG{index}'
        if key in header:
            header[key] = header[key] + start * header.get(f'NBIN{index}', 1)
    for key in ('BSCALE', 'BZERO', 'BLANK', 'CHECKSUM', 'DATASUM'):
        header.remove(key, ignore_missing=True)
    return header


def crop_fits(inputs, output, *, pixels=None, bounds=None, spatial_axes=(1, 2), hdu=None, unit='arcsec'):
    """Crop FITS files into a new/empty output directory, preserving HDU layout.

    Spatial axes use one-based FITS numbering in x,y order. Pixels use zero-based
    x0,y0,x1,y1 with exclusive upper bounds. World bounds use xmin,ymin,xmax,ymax
    in ``unit`` in each input WCS; default arcsec suits solar images. All other
    cube dimensions are retained. Co-shaped image extensions (e.g. masks or scan
    times) are cropped together; tables and unrelated extensions are preserved.
    """
    if (pixels is None) == (bounds is None):
        raise ValueError('Select exactly one of pixels or bounds.')
    if (len(spatial_axes) != 2 or len(set(spatial_axes)) != 2
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) or v < 1 for v in spatial_axes)):
        raise ValueError('spatial_axes must contain two distinct one-based FITS axis numbers in x,y order.')
    if pixels is not None:
        if len(pixels) != 4 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)) for v in pixels):
            raise ValueError('pixels must contain four integers: x0,y0,x1,y1.')
    else:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2]):
            raise ValueError('bounds must be finite xmin,ymin,xmax,ymax with positive width and height.')
    output = Path(output).expanduser().resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f'Choose a new or empty crop output directory: {output}.')
    plan, names = [], set()
    for path in _files(inputs):
        if path.name.casefold() in names:
            raise ValueError('Crop inputs have duplicate output filenames; use separate collections.')
        names.add(path.name.casefold())
        with fits.open(path, memmap=False) as hdus:
            index = (next((i for i, image in enumerate(hdus) if len(_shape(image)) >= 2), None)
                     if hdu is None else hdus.index_of(hdu))
            if index is None or len(_shape(hdus[index])) < max(spatial_axes):
                raise ValueError(f'Select an image HDU with the requested spatial axes: {path}.')
            shape = _shape(hdus[index])
            header = hdus[0].header.copy()
            header.update(hdus[index].header)
            region = _pixels(header, shape, spatial_axes, pixels, bounds, unit)
            views = {}
            for i, image in enumerate(hdus):
                current = _shape(image)
                axes = spatial_axes
                if i != index:
                    if len(current) == 2 and current == (shape[-spatial_axes[1]], shape[-spatial_axes[0]]):
                        axes = (1, 2)
                    elif (len(current) != len(shape) or not current
                          or any(current[-axis] != shape[-axis] for axis in spatial_axes)):
                        continue
                own_header = header if i == index else image.header
                shifted = _shift_header(own_header, current, axes, region)
                views[i] = (_view(current, axes, region), shifted)
            plan.append((path, views))
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.spinnme-crop-', dir=output.parent) as temporary:
        staging = Path(temporary)
        for path, views in plan:
            with fits.open(path, memmap=False) as hdus:
                cropped = []
                for index, image in enumerate(hdus):
                    if index not in views:
                        cropped.append(image.copy())
                        continue
                    view, header = views[index]
                    data = np.array(image.section[view], copy=True)
                    finite = data[np.isfinite(data)]
                    for key, reduce in (('DATAMIN', np.min), ('DATAMAX', np.max)):
                        if key in header:
                            if len(finite):
                                header[key] = float(reduce(finite))
                            else:
                                header.remove(key)
                    constructor = fits.PrimaryHDU if index == 0 else fits.ImageHDU
                    cropped.append(constructor(data, header))
                fits.HDUList(cropped).writeto(staging/path.name, checksum=True)
        if output.exists():
            output.rmdir()
        staging.rename(output)
    return [output/path.name for path, _ in plan]
