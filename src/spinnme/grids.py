"""Shared observing grids, support masks, timing and coordinate geometry."""
from __future__ import annotations

import base64
from copy import deepcopy
from numbers import Integral

import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.io import fits
from astropy.wcs import WCS
from sunpy.map import Map


def spatial_header(value):
    if isinstance(value, fits.Header):
        return value.copy()
    if isinstance(value, str):
        return fits.Header.fromstring(value, sep='\n')
    header = fits.Header()
    for key, item in value.items():
        if item is not None and key.upper() not in ('COMMENT', 'HISTORY', ''):
            key = key.upper()
            header['HIERARCH '+key if len(key) > 8 and not key.startswith('HIERARCH ') else key] = item
    return header


def support_mask(row):
    shape = tuple(row['shape'])
    if not row.get('valid_mask_bits'):
        return np.ones(shape, dtype=bool)
    bits = np.unpackbits(np.frombuffer(base64.b64decode(row['valid_mask_bits']), dtype=np.uint8))
    if bits.size < np.prod(shape):
        raise ValueError('Saved support mask is smaller than its grid.')
    return bits[:np.prod(shape)].reshape(shape).astype(bool)


def time_offsets(row):
    offsets = np.asarray(row['time_offsets_seconds'], dtype=float)
    axis = row.get('time_offset_axis', 'yx')
    if axis == 'x':
        offsets = offsets[None, :]
    elif axis == 'y':
        offsets = offsets[:, None]
    elif axis != 'yx':
        raise ValueError('time_offset_axis must be x, y or yx.')
    return np.broadcast_to(offsets, row['shape'])


def prepare_grid(row, *, shape=None, bounds=None):
    """Crop and resize WCS, carrying all pixel-dependent metadata together."""
    row = deepcopy(row)
    if shape is None and bounds is None:
        return row
    original_shape = tuple(row['shape'])
    original_header = spatial_header(row['header'])
    grid = Map(support_mask(row).astype(np.uint8), original_header)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
        if bounds.shape != (4,) or not np.isfinite(bounds).all() or np.any(bounds[2:] <= bounds[:2]):
            raise ValueError('bounds must be finite xmin,ymin,xmax,ymax in arcsec, with positive extent.')
        if grid.coordinate_frame.name != 'helioprojective':
            raise ValueError('Arcsec bounds require a helioprojective grid.')
        grid = grid.submap(SkyCoord(*bounds[:2]*u.arcsec, frame=grid.coordinate_frame),
                           top_right=SkyCoord(*bounds[2:]*u.arcsec, frame=grid.coordinate_frame))
    if not grid.data.size:
        raise ValueError('The subregion does not intersect the grid.')
    if shape is not None:
        if len(shape) != 2 or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral) or v < 2 for v in shape):
            raise ValueError('shape must contain two integer dimensions >=2 in (ny,nx) order.')
        grid = grid.resample(np.array(shape[::-1])*u.pix, method='nearest')
    # Affine map from new pixel centers to the previous grid's pixel centers.
    wcs = WCS(original_header).celestial
    probe = np.array([0., 1., 0.]), np.array([0., 0., 1.])
    px, py = wcs.world_to_pixel_values(*grid.wcs.pixel_to_world_values(*probe))
    origin, step = np.array([px[0], py[0]]), np.array([px[1]-px[0], py[2]-py[0]])
    y, x = np.indices(grid.data.shape)
    native_x = np.clip(np.rint(origin[0]+step[0]*x).astype(int), 0, original_shape[1]-1)
    native_y = np.clip(np.rint(origin[1]+step[1]*y).astype(int), 0, original_shape[0]-1)
    mask = support_mask(row)[native_y, native_x]
    row['valid_mask_bits'] = base64.b64encode(np.packbits(mask).tobytes()).decode()
    if row.get('time_offsets_seconds') is not None:
        row['time_offsets_seconds'] = time_offsets(row)[native_y, native_x].tolist()
        row['time_offset_axis'] = 'yx'
    if row.get('detector_mapping') is not None:
        detector_origin, detector_step = np.asarray(row['detector_mapping'])
        row['detector_mapping'] = [(detector_origin+origin*detector_step).tolist(), (step*detector_step).tolist()]
    if row.get('slit_pixel_world') is not None:
        coefficients = np.asarray(row['slit_pixel_world'])[:, native_x[0]]
        coefficients[:2] += coefficients[2:]*origin[1]
        coefficients[2:] *= step[1]
        row['slit_pixel_world'] = coefficients.tolist()
    if row.get('slit_geometry') is not None:
        row['slit_geometry'] = {key: np.asarray(value)[native_x[0]].tolist()
                                for key, value in row['slit_geometry'].items()}
    if row.get('slit_coordinate_headers') is not None:
        row['slit_coordinate_headers'] = [row['slit_coordinate_headers'][i] for i in native_x[0]]
    center = (np.array(original_shape)-1)/2
    pixel_origin, pixel_step = row.get('pixel_mapping', [-center/max(*center, 1.), np.ones(2)/max(*center, 1.)])
    row['pixel_mapping'] = [(np.asarray(pixel_origin)+origin[::-1]*pixel_step).tolist(),
                            (np.asarray(pixel_step)*step[::-1]).tolist()]
    row.update(shape=list(grid.data.shape), header=dict(grid.fits_header))
    return row


class GridGeometry:
    """One prepared observing frame, reused across inference batches."""

    def __init__(self, row):
        from .data.fits import observer_velocity_from_header
        from .data.util import image_p_angle
        from .loader.spherical import require_observer_metadata
        self.row = row
        self.header = spatial_header(row['header'])
        self.map = Map(np.zeros((1, 1), dtype=np.uint8), self.header)
        require_observer_metadata(self.map)
        self.wcs = WCS(self.header).celestial
        geometry = row.get('geometry', {})
        self.lon = geometry.get('obs_lon', self.map.carrington_longitude.to_value(u.rad))
        self.lat = geometry.get('obs_lat', self.map.carrington_latitude.to_value(u.rad))
        self.angle = geometry.get('pAng', image_p_angle(self.map))
        self.distance = geometry.get('observer_distance', (self.map.dsun/self.map.rsun_meters).to_value(u.one))
        self.velocity = np.asarray(geometry.get('observer_velocity', observer_velocity_from_header(self.header)))

    def pixels(self, indices, *, Rs_per_ds, normalized_time, seconds_per_dt, scan_times=True):
        from .loader.spherical import spherical_geometry_from_pixels
        row = self.row
        g = spherical_geometry_from_pixels(indices, row['shape'], self.wcs, self.lon, self.lat,
                                           self.angle, self.distance, self.velocity, Rs_per_ds, normalized_time,
                                           slit_pixel_world=row.get('slit_pixel_world'),
                                           slit_geometry=row.get('slit_geometry'))
        y, x = np.divmod(indices, row['shape'][1])
        if scan_times and row.get('time_offsets_seconds') is not None:
            g['coords'][..., -1] += time_offsets(row)[y, x]/seconds_per_dt
        if row.get('pixel_mapping') is not None:
            origin, step = np.asarray(row['pixel_mapping'])
            g['pix'] = np.stack([y, x], -1)*step+origin
        if row.get('detector_mapping') is not None:
            origin, step = np.asarray(row['detector_mapping'])
            g['detector_coords'] = np.clip(np.stack([x, y], -1)*step+origin, 0, 1)
            if row.get('pix_convention') == 'detector':
                g['pix'] = 2*g['detector_coords'][..., ::-1]-1
        return g

    def points(self, coordinates, *, Rs_per_ds, normalized_time, include_pixels=True):
        """Physical SkyCoord positions in the selected observing frame.

        Intrinsic maps can omit detector pixels and their cross-epoch WCS
        reprojection. Spectra and corrections retain the full pixel geometry.
        """
        from sunpy.coordinates import frames

        from .data.util import (
            cartesian_to_spherical_matrix,
            image_to_spherical_matrix,
            observer_ray_and_mu,
            spherical_to_cartesian,
        )
        observer, obstime = self.map.observer_coordinate, self.map.reference_date
        lonc, latc, distance = self.lon, self.lat, self.distance
        scanned = isinstance(coordinates.frame, frames.Helioprojective) and not coordinates.obstime.isscalar
        coeval_observers = False
        if scanned:
            # A scanned image carries its own observer and epoch at every point.
            # Keep those frames through the spatial and magnetic projections.
            observer, obstime = coordinates.observer, coordinates.obstime
            coeval_observers = not observer.isscalar and np.all(observer.obstime == obstime)
            if not coeval_observers:
                observer_carrington = observer.transform_to(frames.HeliographicCarrington(
                    observer=observer, obstime=obstime))
                lonc = observer_carrington.lon.to_value(u.rad)
                latc = observer_carrington.lat.to_value(u.rad)
            distance = (observer.radius/self.map.rsun_meters).to_value(u.one)
        if coeval_observers:
            # All rows in a slit share an observer and epoch. Transform its
            # ephemeris once, then rotate the HPC Cartesian rays numerically.
            # This is the same observer basis used by native raster geometry.
            from astropy.coordinates import SkyCoord
            observer = SkyCoord(observer)
            signature = np.stack([obstime.jd1, obstime.jd2,
                                  observer.spherical.lon.to_value(u.rad),
                                  observer.spherical.lat.to_value(u.rad),
                                  observer.spherical.distance.to_value(u.m)], axis=-1)
            _, first, inverse = np.unique(signature, axis=0, return_index=True, return_inverse=True)
            unique = observer[first]
            central = unique.transform_to(frames.HeliographicCarrington(
                observer=unique, obstime=obstime[first]))
            lonc = central.lon.to_value(u.rad)[inverse]
            latc = central.lat.to_value(u.rad)[inverse]
            if not coordinates.distance.unit.is_equivalent(u.m):
                coordinates = SkyCoord(coordinates.make_3d())
            tx, ty = coordinates.Tx.to_value(u.rad), coordinates.Ty.to_value(u.rad)
            travel = coordinates.distance.to_value(u.Rsun)
            surface = np.stack([travel*np.cos(ty)*np.sin(tx), travel*np.sin(ty),
                                observer.radius.to_value(u.Rsun)-travel*np.cos(ty)*np.cos(tx)], axis=-1)
            sl, cl, sb, cb = np.sin(lonc), np.cos(lonc), np.sin(latc), np.cos(latc)
            basis = np.stack([np.stack([-sl, -sb*cl, cb*cl], axis=-1),
                              np.stack([cl, -sb*sl, cb*sl], axis=-1),
                              np.stack([np.zeros_like(sl), cb, sb], axis=-1)], axis=-2)
            xyz = np.einsum('...ij,...j->...i', basis, surface)
            lon = np.arctan2(xyz[..., 1], xyz[..., 0])
            lat = np.arctan2(xyz[..., 2], np.hypot(xyz[..., 0], xyz[..., 1]))
            radius = np.linalg.norm(xyz, axis=-1)
        else:
            carrington = coordinates.transform_to(frames.HeliographicCarrington(
                observer=observer, obstime=obstime))
            lon, lat = carrington.lon.to_value(u.rad), carrington.lat.to_value(u.rad)
            radius = carrington.radius.to_value(u.Rsun)
        spherical = np.stack([radius, np.pi/2-lat, lon], -1)
        c2s = cartesian_to_spherical_matrix(spherical)
        rtp2img = np.swapaxes(image_to_spherical_matrix(lon, lat, lonc, latc, self.angle,
                            observer_distance=distance, radius=radius), -1, -2)
        ray, mu = observer_ray_and_mu(lon, lat, lonc, latc, distance, radius=radius)
        g = {'coords': np.concatenate([spherical_to_cartesian(spherical)/Rs_per_ds,
                                       np.broadcast_to(normalized_time, lon.shape)[..., None]], -1),
             'cartesian_to_spherical_transform': c2s, 'rtp_to_img_transform': rtp2img,
             'image_p_angle': np.broadcast_to(self.angle, lon.shape)[..., None].copy(),
             'v_obs_los': (ray@self.velocity)[..., None], 'mu': mu[..., None]}
        if not include_pixels:
            return g
        image_coordinates = coordinates.transform_to(self.map.coordinate_frame)
        x, y = self.wcs.world_to_pixel(image_coordinates)
        center = (np.array(self.row['shape'])-1)/2
        origin, step = self.row.get('pixel_mapping', [-center/max(*center, 1.), np.ones(2)/max(*center, 1.)])
        g['pix'] = np.stack([y, x], -1)*step+origin
        if self.row.get('detector_mapping') is not None:
            origin, step = np.asarray(self.row['detector_mapping'])
            g['detector_coords'] = np.clip(np.stack([x, y], -1)*step+origin, 0, 1)
            if self.row.get('pix_convention') == 'detector':
                g['pix'] = 2*g['detector_coords'][..., ::-1]-1
        return g
