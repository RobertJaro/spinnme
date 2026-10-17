import numpy as np
import torch
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames

from spinnme.data.create_spherical_test_set import (
    SphericalTestSetGenerator,
    disk_center_doppler_to_radial_velocity,
    image_frame_inclination,
)


def test_image_frame_inclination_points_toward_observer():
    b_img = np.asarray([
        [0.0, 0.0, 1000.0],
        [0.0, 0.0, -1000.0],
        [1000.0, 0.0, 0.0],
    ])

    _, sin_inc2, cos_inc, inc = image_frame_inclination(b_img)

    np.testing.assert_allclose(sin_inc2, [0.0, 0.0, 1.0], atol=1e-10)
    np.testing.assert_allclose(cos_inc, [1.0, -1.0, 0.0], atol=1e-10)
    np.testing.assert_allclose(inc, [0.0, np.pi, np.pi / 2], atol=1e-10)


def test_positive_disk_center_doppler_maps_to_inward_radial_velocity():
    vdop = np.asarray([-2000.0, 0.0, 2000.0])
    v_r = disk_center_doppler_to_radial_velocity(vdop)

    np.testing.assert_array_equal(v_r, [2000.0, 0.0, -2000.0])
    # At disk center image +z is radial and positive Doppler is -v_img,z.
    np.testing.assert_array_equal(-v_r, vdop)


def test_spherical_profiles_match_saved_truth_azimuth():
    generator = SphericalTestSetGenerator(
        wavelength_center=6173.3433*u.AA,
        wavelength_grid=np.array([-.15, -.05, .05, .15])*u.AA,
        nx=8, ny=10, pattern_resolution=(16,16))
    observer = SkyCoord(lon=0*u.deg, lat=7*u.deg, radius=1*u.AU,
                        obstime='2024-03-23T00:00:00',
                        frame=frames.HeliographicStonyhurst)
    profiles, truth, _ = generator.load_spherical_time_step(7, observer)
    # Reconstruct polarization from the stored inclination/azimuth, rather
    # than trusting the generator's redundant double-angle terms. A sign
    # mismatch here silently rotates synthetic linear polarization by 90 deg.
    expected = {key:value for key,value in truth.items()
                if key not in ('b_rtp','v_rtp','inc','azi')}
    expected['sin_inc2_cos2azi'] = truth['sin_inc2'] * torch.cos(2*truth['azi'])
    expected['sin_inc2_sin2azi'] = truth['sin_inc2'] * torch.sin(2*truth['azi'])
    reconstructed = generator.convert_to_profiles(**expected)
    finite = np.isfinite(profiles).all(axis=(-2,-1))
    assert finite.any()
    np.testing.assert_allclose(profiles[finite], reconstructed[finite], rtol=2e-5, atol=2e-7)
