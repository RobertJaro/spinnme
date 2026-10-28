"""HMI effective measurement/coordinate epoch, distinct from exposure start."""
from astropy.time import Time
from sunpy.map import Map

from .fits import parse_time


def hmi_observation_time(header):
    """T_OBS is the effective measurement and WCS epoch; T_REC is a slot label.

    T_OBS uses TAI in JSOC, including when its suffix is omitted. DATE-OBS is
    the exposure start in UTC and must never substitute for the effective time.
    """
    if not header.get('T_OBS'):
        raise ValueError('HMI geometry and model time require T_OBS; DATE-OBS is exposure start.')
    value = str(header['T_OBS']).strip()
    if not value.endswith('_TAI'):
        value += '_TAI'
    return parse_time(value)


def hmi_time_metadata(header):
    epoch = hmi_observation_time(header)
    return {'original': {key: str(header[key]) for key in
                        ('DATE-OBS', 'DATE-AVG', 'T_OBS', 'T_REC', 'TIMESYS') if key in header},
            'time_reference': 'T_OBS', 'time_scale': 'tai', 'model_time': epoch.isoformat()}


def hmi_coordinate_header(header):
    """Keep exposure start, normalize the effective epoch for SunPy and FITS.

    Native T_OBS spelling is retained in observation provenance. Canonical JSOC
    spelling makes the TAI scale explicit even when the input suffix is absent.
    DATE-AVG supplies the same epoch to GenericMap consumers.
    """
    utc_epoch = hmi_observation_time(header)
    epoch = Time(utc_epoch, precision=6 if utc_epoch.microsecond % 1000 else 3)
    meta = header.copy()
    meta['T_OBS'] = epoch.tai.isot.replace('-', '.').replace('T', '_') + '_TAI'
    meta['DATE-AVG'] = epoch.utc.isot
    # FITS defaults to UTC; T_OBS has an explicit TAI suffix. Avoid a global
    # TIMESYS that would mislabel one of these two differently scaled dates.
    if 'TIMESYS' in meta:
        del meta['TIMESYS']
    return meta


def load_hmi_map(path):
    """Load HMI with its effective measurement epoch as the coordinate epoch."""
    source = Map(path)
    return Map(source.data, hmi_coordinate_header(source.meta))


# Compatibility import; the old name no longer forces geometry to exposure start.
load_hmi_date_obs_map = load_hmi_map
