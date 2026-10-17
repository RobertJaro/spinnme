"""Explicit, reusable FITS ingestion for spherical Stokes inversions.

The canonical array order is ``(y, x, Stokes, wavelength)``. A generic
reader cannot infer an instrument, atomic line, polarization convention, or
missing observer geometry from a filename. Those are configuration inputs.
"""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

import numpy as np
from astropy import units as u
from astropy.io import fits
from astropy.time import Time
from astropy.wcs import WCS


def parse_wavelength_unit(value):
    """Accept an Astropy unit or a SOLARNET power-of-ten metre WAVEUNIT."""
    if isinstance(value, (int, float, np.number)):
        return u.Unit(10. ** float(value) * u.m)
    return u.Unit(value)


def observer_velocity_from_header(header):
    """Return the optional observer vector [-west, north, radial] in m/s.

    OBS_VR/VW/VN follow the solar FITS convention. Only finite numeric
    components contribute; absent or unavailable values supply no correction.
    Never interpret a fitted solar Doppler velocity as spacecraft motion.
    """
    components = []
    for key, sign in (('OBS_VW', -1), ('OBS_VN', 1), ('OBS_VR', 1)):
        try:
            value = float(header.get(key, 0.))
        except (TypeError, ValueError):
            value = 0.
        components.append(sign * value if np.isfinite(value) else 0.)
    return np.asarray(components, dtype=np.float64)


def parse_time(value, scale="utc"):
    """Return a timezone-free UTC datetime, including JSOC TAI labels."""
    if isinstance(value, datetime):
        return Time(value, scale=scale).utc.to_datetime()
    text = str(value).strip()
    if text.endswith("_TAI"):
        text = text[:-4]
        text = re.sub(r"^(\d{4})\.(\d{2})\.(\d{2})_", r"\1-\2-\3T", text)
        text = re.sub(r"^(\d{4})(\d{2})(\d{2})_", r"\1-\2-\3T", text)
        text = re.sub(r"T(\d{2})(\d{2})(\d{2})(?=\.|$)", r"T\1:\2:\3", text)
        scale = "tai"
    return Time(text, scale=scale).utc.to_datetime()


def observation_times(header, time_reference="observation", time_scale=None):
    """Preserve native dates; choose a separate model epoch explicitly.

    ``solar`` and ``earth`` use the supplied DATE_SUN/DATE_EAR values. They
    do not alter the time used by the spatial WCS or observer geometry.
    """
    scale = (time_scale or header.get("TIMESYS", "UTC")).lower()
    keys = {"observation": "DATE-OBS", "average": "DATE-AVG",
            "solar": "DATE_SUN", "earth": "DATE_EAR"}
    if time_reference not in keys:
        raise ValueError(f"Unknown time_reference {time_reference!r}; choose {tuple(keys)}.")
    key = keys[time_reference]
    if key not in header:
        raise ValueError(f"time_reference={time_reference!r} requires {key}.")
    original = {k: str(header[k]) for k in
                ("DATE-OBS", "DATE-BEG", "DATE-AVG", "DATE-END", "DATE_EAR", "DATE_SUN")
                if k in header}
    model_time = parse_time(header[key], scale)
    return {"original": original, "time_reference": time_reference,
            "time_scale": scale, "model_time": model_time.isoformat()}, model_time


def observation_header(hdus, data_hdu=0, spatial_header_hdu=None):
    """Merge primary/science metadata; a separate spatial HDU supplies the WCS.

    Science dates and spectral calibration take precedence over unrelated
    cards in the spatial HDU. Missing auxiliary metadata may be inherited.
    """
    header = hdus[0].header.copy()
    header.update(hdus[data_hdu].header)
    if spatial_header_hdu is not None and spatial_header_hdu != data_hdu:
        spatial = hdus[spatial_header_hdu].header
        wcs_card = re.compile(
            r'^(WCSAXES|WCSNAME|LONPOLE|LATPOLE|RADESYS|EQUINOX|'
            r'(CTYPE|CUNIT|CDELT|CRPIX|CRVAL|CROTA)\d+|'
            r'(PC|CD|PV|PS)\d+_\d+)[A-Z]?$|'
            r'^(CPDIS|D2IMDIS|DET2IM|DP\d|D2IM\d|'
            r'(A|B|AP|BP)_(ORDER|\d+_\d+))')
        for key in list(header):
            if wcs_card.match(key):
                del header[key]
        for card in spatial.cards:
            if ((wcs_card.match(card.keyword) or card.keyword not in header)
                    and not card.keyword.startswith('NAXIS')):
                header[card.keyword] = (card.value, card.comment)
    return header


def spatial_header(header, shape, spatial_wcs_axes=None):
    """Extract a 2D WCS without guessing or fixing erroneous WCS metadata."""
    original = header.copy()
    auxiliary = [key for key in original if key.startswith(('CPDIS', 'D2IMDIS', 'DET2IM', 'DP1', 'DP2'))]
    if auxiliary or any('-TAB' in str(original.get(f'CTYPE{i}', '')) for i in range(1,5)):
        raise ValueError('This WCS requires auxiliary distortion/table HDUs. Prepare a self-contained '
                         'validated spatial WCS before inversion; silently dropping those HDUs is unsupported.')
    if spatial_wcs_axes is not None:
        axes = list(spatial_wcs_axes)
        if len(axes) != 2 or len(set(axes)) != 2:
            raise ValueError("spatial_wcs_axes must identify two FITS axes in x,y order.")
        wcs_header = WCS(original, relax=True).sub(axes).to_header(relax=True)
    else:
        # With explicit NumPy axis_order, the caller's spatial header must
        # describe x,y in FITS axes 1,2. Never reorder a malformed PHI
        # header automatically merely because its CTYPE cards suggest it.
        wcs_header = WCS(original, naxis=2, relax=True).to_header(relax=True)
    result = original.copy()
    for key in list(result):
        if re.match(r"^(NAXIS\d+|CTYPE\d+|CUNIT\d+|CDELT\d+|CRPIX\d+|CRVAL\d+|CROTA\d+|PC\d+_\d+|CD\d+_\d+)$", key):
            del result[key]
    result.update(wcs_header)
    result["NAXIS"] = 2
    result["NAXIS1"], result["NAXIS2"] = int(shape[1]), int(shape[0])
    result["WCSAXES"] = 2
    if "CTYPE1" not in result or "CTYPE2" not in result:
        raise ValueError("A valid solar spatial WCS is required.")
    return result


def wavelength_configuration(header, count, wavelength_center=None,
                             wavelength_grid=None, wavelength_unit="Angstrom",
                             wavelength_grid_is_offsets=True, wavelength_hdu=None,
                             hdus=None):
    """Read a configurable, nonuniform wavelength grid with explicit units."""
    configured_unit = parse_wavelength_unit(wavelength_unit)
    grid_unit = configured_unit
    center_unit = configured_unit
    if wavelength_grid is None and wavelength_hdu is not None:
        wavelength_grid = np.asarray(hdus[wavelength_hdu].data).squeeze()
    if wavelength_grid is None:
        keys = [f"WAVELN{i + 1:02d}" for i in range(count)]
        if not all(key in header for key in keys):
            raise ValueError("Supply wavelength_grid/wavelength_hdu or complete WAVELNnn metadata.")
        wavelength_grid = [float(header[key]) for key in keys]
        grid_unit = parse_wavelength_unit(header.get("WAVEUNIT", wavelength_unit))
        wavelength_grid_is_offsets = False
    if wavelength_center is None:
        wavelength_center = header.get("WAVELNTH")
        center_unit = parse_wavelength_unit(header.get("WAVEUNIT", wavelength_unit))
    if wavelength_center is None:
        raise ValueError("Supply the atomic line wavelength_center; it cannot be inferred from a cube.")
    center = (wavelength_center if isinstance(wavelength_center, u.Quantity)
              else float(wavelength_center) * center_unit).to(u.m)
    grid = (wavelength_grid if isinstance(wavelength_grid, u.Quantity)
            else np.asarray(wavelength_grid, dtype=float) * grid_unit).to(u.m)
    if grid.shape != (count,) or not np.isfinite(grid.value).all() or not np.isfinite(center.value):
        raise ValueError(f"Wavelength grid must contain exactly {count} finite values.")
    if not wavelength_grid_is_offsets:
        grid = grid - center
    return {"wavelength_center": center, "wavelength_grid": grid}


def read_fits_observation(path, *, data_hdu=0, spatial_header_hdu=None,
                          axis_order=None, stokes_order="IQUV", stokes_signs=None,
                          wavelength_center=None, wavelength_grid=None,
                          wavelength_unit="Angstrom", wavelength_grid_is_offsets=True,
                          wavelength_hdu=None, wavelength_range=None, spatial_wcs_axes=None,
                          observer=None, time_reference="observation", time_scale=None,
                          line_id="custom", continuum_index=None):
    """Read one cube; explicit axes are specified in NumPy order.

    ``wavelength_range`` selects [lower, upper) absolute wavelengths in
    ``wavelength_unit``; either bound can be None. Array order is preserved.
    ``continuum_index`` addresses the original spectrum and is remapped.

    ``observer`` may supply missing FITS cards such as DSUN_OBS, HGLN_OBS,
    HGLT_OBS and optional OBS_VR/VW/VN. It is recorded as an override.
    No Earth observer is inferred. Missing velocity components contribute
    no observer-motion correction.
    """
    if axis_order is None:
        raise ValueError("axis_order is required, e.g. [wavelength, stokes, y, x].")
    if isinstance(axis_order, str):
        axis_order = axis_order.replace(",", " ").split()
    axes = [str(axis).lower() for axis in axis_order]
    aliases = {"wave": "wavelength", "lambda": "wavelength", "pol": "stokes"}
    axes = [aliases.get(axis, axis) for axis in axes]
    if sorted(axes) != sorted(["y", "x", "stokes", "wavelength"]):
        raise ValueError("axis_order must contain x, y, stokes and wavelength exactly once.")
    with fits.open(path, memmap=False) as hdus:
        raw = np.asarray(hdus[data_hdu].data)
        if raw.ndim != 4:
            raise ValueError(f"Expected a 4D Stokes cube in HDU {data_hdu}, got {raw.shape}.")
        stokes = np.transpose(raw, [axes.index(axis) for axis in ("y", "x", "stokes", "wavelength")])
        order = list(stokes_order)
        if sorted(order) != sorted("IQUV") or stokes.shape[-2] != 4:
            raise ValueError("A four-component I/Q/U/V Stokes order is required.")
        stokes = np.array(stokes[..., [order.index(s) for s in "IQUV"], :], dtype=np.float32)
        if stokes_signs is not None:
            signs = np.asarray(stokes_signs, dtype=float)
            if signs.shape != (4,) or not np.isin(signs, [-1, 1]).all():
                raise ValueError("stokes_signs must contain four signs in canonical I,Q,U,V order.")
            stokes *= signs[:, None]
        extension_header = hdus[data_hdu if spatial_header_hdu is None else spatial_header_hdu].header.copy()
        original_header = observation_header(hdus, data_hdu, spatial_header_hdu)
        header = original_header.copy()
        header.update(observer or {})
        wl = wavelength_configuration(header, stokes.shape[-1], wavelength_center,
                                      wavelength_grid, wavelength_unit,
                                      wavelength_grid_is_offsets, wavelength_hdu, hdus)
    if wavelength_range is not None:
        # Bounds are absolute wavelengths in wavelength_unit, regardless of
        # whether the input grid contains absolute values or line offsets.
        if len(wavelength_range) != 2:
            raise ValueError("wavelength_range must be [lower, upper], with optional null bounds.")
        lower, upper = wavelength_range
        lower = -np.inf if lower is None else float(lower)
        upper = np.inf if upper is None else float(upper)
        if np.isnan(lower) or np.isnan(upper) or lower >= upper:
            raise ValueError("wavelength_range bounds must be increasing finite values or null.")
        absolute = (wl['wavelength_grid'] + wl['wavelength_center']).to_value(
            parse_wavelength_unit(wavelength_unit))
        selected = np.flatnonzero((absolute >= lower) & (absolute < upper))
        if not len(selected):
            raise ValueError("wavelength_range selected no spectral samples.")
        if continuum_index is not None:
            position = np.flatnonzero(selected == continuum_index)
            if not len(position):
                raise ValueError("continuum_index is outside wavelength_range.")
            continuum_index = int(position[0])
        stokes = stokes[..., selected]
        wl['wavelength_grid'] = wl['wavelength_grid'][selected]
    time_metadata, model_time = observation_times(header, time_reference, time_scale)
    if continuum_index is not None and not 0 <= int(continuum_index) < stokes.shape[-1]:
        raise ValueError("continuum_index must be a zero-based wavelength index.")
    return {"stokes": stokes, "header": spatial_header(header, stokes.shape[:2], spatial_wcs_axes),
            "original_header": original_header, "wavelength_config": wl,
            "model_time": model_time, "continuum_index": continuum_index,
            "metadata": {"source_paths": [str(Path(path).resolve())], "line_id": line_id,
                         "original_header": original_header.tostring(sep="\n", endcard=False, padding=False),
                         "original_extension_header": extension_header.tostring(sep="\n", endcard=False, padding=False),
                         "time": time_metadata, "observer_overrides": dict(observer or {}),
                         "reader": {"data_hdu": data_hdu, "spatial_header_hdu": spatial_header_hdu,
                                    "axis_order": axes, "stokes_order": order,
                                    "stokes_signs": stokes_signs, "wavelength_range": wavelength_range}}}
