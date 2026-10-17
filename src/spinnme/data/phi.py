"""Version-aware PHI Stokes ingestion through the generic FITS reader."""
from astropy import units as u
from astropy.io import fits

from .fits import read_fits_observation
from .phi_util import phi_continuum_index


def read_phi_observation(path, *, telescope="hrt", **kwargs):
    if telescope not in ("hrt", "fdt"):
        raise ValueError("PHI telescope must be hrt or fdt.")
    with fits.open(path, memmap=False) as hdus:
        header = hdus[0].header.copy()
        header.update(hdus[kwargs.get('data_hdu',0)].header)
    count = int(header.get("NAXIS4", 0))
    # Current L2 products have spatial FITS axes 1,2 and NumPy order
    # wavelength,Stokes,y,x. Other schemas require an explicit axis mapping
    # and a validated spatial header.
    if "axis_order" not in kwargs:
        if (header.get("NAXIS3") == 4 and count >= 2 and
                str(header.get("CTYPE1", "")).startswith("HPLN") and
                str(header.get("CTYPE2", "")).startswith("HPLT")):
            kwargs["axis_order"] = ["wavelength", "stokes", "y", "x"]
        else:
            raise ValueError("Unrecognized PHI axis/WCS schema. Supply explicit axis_order "
                             "and a validated spatial_header_hdu; automatic header repair is disabled.")
    axes = kwargs["axis_order"]
    if isinstance(axes, str):
        axes = axes.replace(',', ' ').split()
    axes = [{'wave':'wavelength','lambda':'wavelength','pol':'stokes'}.get(str(v).lower(),str(v).lower()) for v in axes]
    kwargs['axis_order'] = axes
    wave_axis = axes.index("wavelength")
    count = int(header[f"NAXIS{4-wave_axis}"])
    keys = [f"WAVELN{i+1:02d}" for i in range(count)]
    has_processed = all(k in header for k in keys) and "WAVELNTH" in header
    if not has_processed and "wavelength_grid" not in kwargs:
        raise ValueError("PHI calibrated WAVELNnn metadata are missing. Supply wavelength_grid "
                         "and wavelength_center for a verified spectral calibration.")
    if kwargs.get('continuum_index') is None and 'CONTPOS' in header:
        # Resolve the native one-based index before generic wavelength selection
        # remaps it; applying CONTPOS to a cropped spectrum would pick the wrong sample.
        kwargs['continuum_index'] = phi_continuum_index(header, list(range(count)))
    if (kwargs.get('wavelength_range') is not None and kwargs.get('continuum_index') is None):
        raise ValueError('PHI wavelength_range requires CONTPOS or an explicit native continuum_index.')
    kwargs.setdefault("line_id", "fe-i-6173")
    obs = read_fits_observation(path, **kwargs)
    if obs["continuum_index"] is None:
        waves = (obs["wavelength_config"]["wavelength_grid"] +
                 obs["wavelength_config"]["wavelength_center"]).to_value(u.AA)
        obs["continuum_index"] = phi_continuum_index(header, waves)
    obs["metadata"].update(telescope=telescope, instrument="PHI", release=str(header.get("VERSION", "")),
                           pipeline_version=str(header.get("VERS_SW", "")),
                           onboard_averages=header.get("ACCRAWIT"),
                           spectral_frame=header.get("SPECSYS"),
                           calibration_version=str(header.get("VERS_CAL", "")))
    return obs
