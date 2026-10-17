import os
import os.path

import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import Angle, SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header, make_heliographic_header
from sunpy.sun import constants

from spinnme.data.circular_pattern import CircularPatternGenerator
from spinnme.data.util import image_p_angle, image_to_spherical_matrix, solar_differential_rotation_velocity


def image_frame_inclination(b_img):
    """Return ME inclination quantities for an image-frame magnetic field.

    The image ``+z`` axis points toward the observer, so an observer-directed
    field has ``cos(inclination) > 0``.
    """
    b_field = np.linalg.norm(b_img, axis=-1)
    transverse_b2 = b_img[..., 0] ** 2 + b_img[..., 1] ** 2
    sin_inc2 = np.divide(
        transverse_b2,
        b_field ** 2,
        out=np.zeros_like(b_field),
        where=b_field > 0,
    )
    cos_inc = np.divide(
        b_img[..., 2],
        b_field,
        out=np.zeros_like(b_field),
        where=b_field > 0,
    )
    inc = np.arccos(np.clip(cos_inc, -1.0, 1.0))
    return b_field, sin_inc2, cos_inc, inc


def disk_center_doppler_to_radial_velocity(vdop):
    """Convert positive-redshift Doppler velocity to radial velocity.

    Positive radial velocity points outward, hence toward an Earth-like
    observer at disk center. Positive Doppler velocity points away from the
    observer, so the two conventions have opposite signs.
    """
    return -vdop


class SphericalTestSetGenerator(CircularPatternGenerator):
    """Circular Milne-Eddington pattern projected with finite-distance geometry."""
    __test__ = False

    def __init__(self, *args, pattern_resolution=(180, 180), tile_rows=2, tile_cols=4, **kwargs):
        super().__init__(*args, **kwargs)
        self.pattern_resolution = tuple(pattern_resolution)
        self.tile_rows, self.tile_cols = tile_rows, tile_cols

    def load_spherical_time_step(self, time_step, obs_coord):
        parameters = self._load_parameters(time_step, resolution=self.pattern_resolution)
        # convert to numpy arrays
        parameters = {k: create_collage(v.detach().cpu().numpy(), rows=self.tile_rows, cols=self.tile_cols) for k, v in parameters.items()}
        transformed_parameters, dummy_helioprojective_map = self._transform_parameters(parameters, obs_coord)
        # convert back to torch tensors
        transformed_parameters = {k: torch.tensor(v, dtype=torch.float32) for k, v in transformed_parameters.items()}
        input_parameters = {k: v for k, v in transformed_parameters.items() if k not in ['b_rtp', 'v_rtp', 'azi', 'inc']}
        stokes_profiles = self.convert_to_profiles(**input_parameters)
        return stokes_profiles, transformed_parameters, dummy_helioprojective_map

    def _transform_parameters(self, input_parameters, obs_coord):
        # setup b_rtp
        b_field = input_parameters['b_field']
        b_inc = input_parameters['inc']
        b_azi = input_parameters['azi']

        # Convert to spherical coordinates
        b_r = b_field * np.cos(b_inc)
        b_theta = b_field * np.sin(b_inc) * np.cos(b_azi)
        b_phi = b_field * np.sin(b_inc) * np.sin(b_azi)
        # add to parameter dict
        input_parameters['b_r'] = b_r
        input_parameters['b_theta'] = b_theta
        input_parameters['b_phi'] = b_phi

        # Convert the inherited positive-redshift LOS velocity into the radial
        # velocity that produces it at disk center. Away from disk center its
        # LOS projection naturally decreases with mu.
        vdop = input_parameters['vdop']
        v_r = disk_center_doppler_to_radial_velocity(vdop)
        v_theta = np.zeros_like(v_r)
        v_phi = np.zeros_like(v_r)
        # add differential rotation
        latitudes = np.linspace(-np.pi / 2, np.pi / 2, v_r.shape[0]) * u.rad
        v_diff = solar_differential_rotation_velocity(latitudes).to_value(u.m / u.s)
        v_phi += v_diff[:, None]
        # add to parameter dict
        input_parameters['v_r'] = v_r
        input_parameters['v_theta'] = v_theta
        input_parameters['v_phi'] = v_phi

        # Use the supplied observer, including non-Earth viewpoints.
        carrington_header = make_heliographic_header(obs_coord.obstime, obs_coord,
                                                     b_r.shape, frame='carrington')

        # create helioprojective map header
        solar_semidiameter_rad = np.arcsin(constants.radius / obs_coord.radius)
        angular_radius = Angle(solar_semidiameter_rad.to(u.arcsec))

        scale = angular_radius.to(u.arcsec) / (self.ny / 2 * u.pix), angular_radius.to(u.arcsec) / (
                self.nx / 2 * u.pix)
        dummy_data = np.zeros((self.nx, self.ny), dtype=np.float32)
        reference_coord = SkyCoord(0 * u.arcsec, 0 * u.arcsec, observer=obs_coord, frame=frames.Helioprojective)
        helioprojective_header = make_fitswcs_header(dummy_data, reference_coord, scale=u.Quantity(scale))

        # transform all parameters to helioprojective frame
        exclude_parameters = ['b_field', 'inc', 'azi', 'vdop']
        input_parameters = {k: v for k, v in input_parameters.items() if k not in exclude_parameters}
        transformed_parameters = {}

        for k, parameter in input_parameters.items():
            carrington_map = Map(parameter, carrington_header)
            helioprojective_map = carrington_map.reproject_to(helioprojective_header)
            transformed_parameters[k] = helioprojective_map.data

        # create dummy helioprojective map
        dummy_carrington = np.zeros_like(b_r)
        carrington_map = Map(dummy_carrington, carrington_header)
        helioprojective_map = carrington_map.reproject_to(helioprojective_header)
        # Geometry and mu come from the same finite-distance projection.
        map_coords = all_coordinates_from_map(helioprojective_map)

        # Transform vector quantities -- B, V -- to image coordinates
        # create transformation matrix
        carrington_coords = map_coords.transform_to(frames.HeliographicCarrington)
        lat, lon = carrington_coords.lat.to_value(u.rad), carrington_coords.lon.to_value(u.rad)
        latc, lonc = helioprojective_map.carrington_latitude.to_value(
            u.rad), helioprojective_map.carrington_longitude.to_value(u.rad)

        pAng = image_p_angle(helioprojective_map)
        a_matrix = image_to_spherical_matrix(
            lon, lat, lonc, latc, pAng=pAng,
            observer_distance=(helioprojective_map.dsun / helioprojective_map.rsun_meters).to_value(u.dimensionless_unscaled),
        )
        transformed_parameters['mu'] = a_matrix[..., 0, 2].astype(np.float32)
        rtp_to_img_transform = np.linalg.inv(a_matrix)

        # stack b vector
        b_r = transformed_parameters.pop('b_r')
        b_theta = transformed_parameters.pop('b_theta')
        b_phi = transformed_parameters.pop('b_phi')
        b_rtp = np.stack([b_r, b_theta, b_phi], -1)

        # transform b vector to image frame
        b_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, b_rtp)  # in image xyz

        b_field, sin_inc2, cos_inc, inc = image_frame_inclination(b_img)

        # sin(2*(x + pi/2)) = sin(2*x + pi) = -sin(2*x)
        # cos(2*(x + pi/2)) = cos(2*x + pi) = -cos(2*x)
        # 2 * sin(x) * cos(x) = sin(2*x)
        # sin(x)**2 - cos(x)**2 = -cos(2*x)
        field_denominator = b_field ** 2 + 1e-8
        sin_inc2_sin2azi = -2 * b_img[..., 0] * b_img[..., 1] / field_denominator
        sin_inc2_cos2azi = -(b_img[..., 0] ** 2 - b_img[..., 1] ** 2) / field_denominator

        azi = np.arctan2(-b_img[..., 0], b_img[..., 1])

        # stack v vector
        v_r = transformed_parameters.pop('v_r')
        v_theta = transformed_parameters.pop('v_theta')
        v_phi = transformed_parameters.pop('v_phi')
        v_rtp = np.stack([v_r, v_theta, v_phi], -1)

        # transform velocity vector to image frame
        v_img = np.einsum("...ij,...j->...i", rtp_to_img_transform, v_rtp)  # in image xyz
        # convert to ME parameters
        # Positive Doppler velocity is a redshift (motion away from the
        # observer); the image z axis points toward the observer.
        vdop = -v_img[..., 2]

        transformed_parameters['b_field'] = b_field

        transformed_parameters['sin_inc2_sin2azi'] = sin_inc2_sin2azi
        transformed_parameters['sin_inc2_cos2azi'] = sin_inc2_cos2azi
        transformed_parameters['azi'] = azi
        transformed_parameters['sin_inc2'] = sin_inc2
        transformed_parameters['cos_inc'] = cos_inc
        transformed_parameters['inc'] = inc

        transformed_parameters['vdop'] = vdop

        transformed_parameters['b_rtp'] = b_rtp
        transformed_parameters['v_rtp'] = v_rtp

        return transformed_parameters, helioprojective_map

    def create_spherical_time_step_file(self, t_step, base_path, obs_coords):
        profiles, parameters, dummy_map = self.load_spherical_time_step(t_step, obs_coords)
        header = dummy_map.meta
        header['OBS_VR'] = 0
        header['OBS_VW'] = 0
        header['OBS_VN'] = 0
        for wl_idx in range(profiles.shape[-1]):
            for s_ixd, s_id in enumerate(['I', 'Q', 'U', 'V']):
                p_map = Map(profiles[:, :, s_ixd, wl_idx], header)
                file_path = os.path.join(base_path, f'stokes_{t_step:03d}.{s_id}{wl_idx}.fits')
                p_map.save(file_path, overwrite=True)

        np.savez(os.path.join(base_path, f'parameters_{t_step:03d}.npz'), **parameters)


def create_collage(image_array, rows=12, cols=24):
    """
    Tile a two-dimensional circular motif over a heliographic grid.

    Parameters:
    image_array (numpy.ndarray): Input grayscale image (H, W)
    rows (int): Number of rows in the collage
    cols (int): Number of columns in the collage

    Returns:
    numpy.ndarray: Collaged image array (rows*H, cols*W)
    """
    # Create one row with `cols` duplicates
    row = np.concatenate([image_array] * cols, axis=1)

    # Stack `rows` copies of that row to form full collage
    collage = np.concatenate([row] * rows, axis=0)

    return collage
