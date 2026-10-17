import numpy as np
import pytest
from astropy import units as u
from astropy.coordinates import SkyCoord
from sunpy.coordinates import frames
from sunpy.map import Map, all_coordinates_from_map, make_fitswcs_header

from spinnme.data.util import image_to_spherical_matrix, spherical_to_cartesian_matrix
from spinnme.loader.spherical import load_map_data, load_v_observer_LOS


@pytest.mark.parametrize('distance', [3., 107., 215.])
@pytest.mark.parametrize('roll', [0., .4, -1.2])
def test_projection_and_mu_match_cartesian_surface_to_observer_ray(distance, roll):
    lon = np.array([.2, .4, -.3, 1.4])
    lat = np.array([.1, -.5, .8, 0.])
    lonc, latc = .2, .1
    spherical = np.stack([np.ones_like(lon), np.pi / 2 - lat, lon], axis=-1)
    local_to_xyz = spherical_to_cartesian_matrix(spherical)
    radial = local_to_xyz[..., :, 0]
    observer = distance * np.array([np.cos(latc)*np.cos(lonc), np.cos(latc)*np.sin(lonc), np.sin(latc)])
    ray = observer - radial
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    projection = image_to_spherical_matrix(lon, lat, lonc, latc, roll, observer_distance=distance)
    # Column z of image->local is the LOS, expressed in the local basis.
    reconstructed_ray = np.einsum('...ij,...j->...i', local_to_xyz, projection[..., :, 2])
    np.testing.assert_allclose(reconstructed_ray, ray, atol=1e-14)
    np.testing.assert_allclose(projection[..., 0, 2], (radial*ray).sum(-1), atol=1e-14)
    np.testing.assert_allclose(projection @ projection.swapaxes(-1, -2), np.broadcast_to(np.eye(3), projection.shape), atol=1e-14)
    np.testing.assert_allclose(np.linalg.det(projection), 1, atol=1e-14)
    assert projection[0, 0, 2] == pytest.approx(1.)


def test_parallel_ray_limit_and_finite_distance_limb():
    lon = np.array([0., .4, 1.])
    np.testing.assert_allclose(
        image_to_spherical_matrix(lon, lon*.1, 0., .1, .3, observer_distance=1e12),
        image_to_spherical_matrix(lon, lon*.1, 0., .1, .3), atol=2e-12,
    )
    # At the tangent point cos(longitude)=R/D, mu must be zero.
    limb = np.arccos(1/3)
    matrix = image_to_spherical_matrix(np.array([limb, limb+.01]), 0., 0., 0., 0., observer_distance=3.)
    assert matrix[0, 0, 2] == pytest.approx(0., abs=1e-14)
    assert matrix[1, 0, 2] < 0


def _map():
    observer = SkyCoord(0*u.deg, 7*u.deg, .7*u.AU, frame=frames.HeliographicStonyhurst, obstime='2024-03-23')
    center = SkyCoord(0*u.arcsec, 0*u.arcsec, frame=frames.Helioprojective(observer=observer, obstime=observer.obstime))
    data = np.zeros((7, 7))
    header = make_fitswcs_header(data, center, scale=[500, 500]*u.arcsec/u.pix, rotation_angle=23*u.deg)
    header.update({'obs_vr': 1300., 'obs_vw': 2000., 'obs_vn': -500.})
    return Map(data, header)


def test_map_mu_matches_sunpy_three_dimensional_rays_and_doppler_signs():
    solar_map = _map()
    result = load_map_data(solar_map)
    hpc = all_coordinates_from_map(solar_map)
    surface = hpc.make_3d().transform_to(frames.Heliocentric(observer=solar_map.observer_coordinate, obstime=solar_map.date))
    xyz = np.moveaxis(surface.cartesian.xyz.to_value(u.m), 0, -1)
    normal = xyz / np.linalg.norm(xyz, axis=-1, keepdims=True)
    ray = np.array([0, 0, solar_map.dsun.to_value(u.m)]) - xyz
    ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
    expected_mu = np.sum(normal*ray, axis=-1)
    np.testing.assert_allclose(result['mu'], expected_mu, atol=1e-7, equal_nan=True)
    assert np.isnan(result['mu']).any()
    assert result['mu'][3, 3] == pytest.approx(1.)
    expected_doppler = -2000*ray[..., 0] - 500*ray[..., 1] + 1300*ray[..., 2]
    np.testing.assert_allclose(load_v_observer_LOS(solar_map), expected_doppler, atol=1e-7, equal_nan=True)
    assert result['v_obs_los'][3, 3] == pytest.approx(1300.)


def test_dataset_mu_is_exactly_the_batch_projection(tmp_path):
    import torch
    from spinnme.instrument import hmi_wavelength_config
    from spinnme.loader.spherical import SphericalDataset

    solar_map = _map()
    map_data = load_map_data(solar_map)
    dataset = SphericalDataset(
        stokes=np.ones((7, 7, 4, 6), dtype=np.float32),
        map_data=map_data, wavelength_config=hmi_wavelength_config(),
        ds_id=0, instrument_id='HMI', seconds_per_dt=3600., Rs_per_ds=1.,
        ref_time=map_data['time'], stokes_normalization=1., work_directory=str(tmp_path),
        batch_size=64, shuffle=False,
    )
    batch = dataset[0]
    torch.testing.assert_close(batch['mu'], batch['rtp_to_img_transform'][..., 2, 0:1], rtol=0, atol=0)
    assert torch.isfinite(batch['mu']).all()
    assert (batch['mu'] > 0).all()
    # A unit radial field must project to mu along the LOS.
    radial = torch.zeros((batch['mu'].shape[0], 3))
    radial[:, 0] = 1
    projected = torch.einsum('bij,bj->bi', batch['rtp_to_img_transform'], radial)
    torch.testing.assert_close(projected[:, 2:3], batch['mu'], rtol=0, atol=0)


@pytest.mark.parametrize('filter_nans,shuffle', [(True, True), (True, False), (False, False)])
def test_minimal_pixel_cache_reconstructs_original_samples(tmp_path, filter_nans, shuffle):
    from spinnme.instrument import hmi_wavelength_config
    from spinnme.loader.spherical import SphericalDataset, hmi_detector_mapping, load_hmi_detector_coords
    from spinnme.data.util import spherical_to_cartesian

    solar_map = _map()
    solar_map.meta['CCD_X0'] = 100
    solar_map.meta['CCD_Y0'] = 200
    map_data = load_map_data(solar_map)
    map_data['detector_mapping'] = hmi_detector_mapping(solar_map)
    stokes = np.broadcast_to(np.arange(49, dtype=np.float32).reshape(7, 7, 1, 1) + 1, (7, 7, 4, 6)).copy()
    stokes[3, 3] = np.nan  # missing observation inside the visible disk
    dataset = SphericalDataset(
        stokes=stokes, map_data=map_data, wavelength_config=hmi_wavelength_config(),
        ds_id=0, instrument_id='HMI', seconds_per_dt=3600., Rs_per_ds=2.,
        ref_time=map_data['time'], stokes_normalization=1., work_directory=str(tmp_path),
        batch_size=8, shuffle=shuffle, filter_nans=filter_nans,
    )
    expected_keys = {'stokes', 'pixel_index'} if filter_nans or shuffle else {'stokes'}
    assert set(dataset.data) == expected_keys
    assert not hasattr(dataset, 'v_obs_los')
    batches = [dataset[i] for i in range(len(dataset))]
    stokes_out = np.concatenate([b['stokes'].numpy() for b in batches])
    valid_out = np.isfinite(stokes_out).all(axis=(-2, -1))
    indices = stokes_out[valid_out, 0, 0].astype(int) - 1
    expected_xyz = spherical_to_cartesian(map_data['spherical_coords']).reshape(-1, 3)[indices] / 2
    coords = np.concatenate([b['coords'].numpy() for b in batches])
    np.testing.assert_allclose(coords[valid_out, :3], expected_xyz, atol=1e-7)
    for key in ('mu', 'v_obs_los'):
        actual = np.concatenate([b[key].numpy() for b in batches])[valid_out, 0]
        np.testing.assert_allclose(actual, map_data[key].reshape(-1)[indices], rtol=2e-7, atol=1e-6)
    detector = np.concatenate([b['detector_coords'].numpy() for b in batches])
    np.testing.assert_allclose(detector[valid_out], load_hmi_detector_coords(solar_map).reshape(-1, 2)[indices], atol=1e-7)
    row, col = np.divmod(indices, 7)
    pix = np.concatenate([b['pix'].numpy() for b in batches])
    np.testing.assert_allclose(pix[valid_out],
                               np.stack([(row+200)/4095, (col+100)/4095], -1)*2-1, atol=1e-7)
    if not filter_nans:
        assert len(coords) == 49
        assert np.isnan(coords[~valid_out]).all()
        np.testing.assert_array_equal(indices, np.flatnonzero(valid_out))
    else:
        assert valid_out.all()


def test_resampled_pixel_geometry_matches_sunpy():
    from spinnme.loader.spherical import spherical_geometry_from_pixels
    from spinnme.data.util import spherical_to_cartesian
    solar_map = _map().resample([5, 6]*u.pix)
    info = load_map_data(solar_map)
    result = spherical_geometry_from_pixels(
        np.arange(solar_map.data.size), solar_map.data.shape, solar_map.wcs,
        info['obs_lon'].to_value(u.rad), info['obs_lat'].to_value(u.rad), info['pAng'].to_value(u.rad),
        info['observer_distance'], info['observer_velocity'], 1., .5,
    )
    expected = spherical_to_cartesian(info['spherical_coords']).reshape(-1, 3)
    np.testing.assert_allclose(result['coords'][:, :3], expected, atol=2e-10, equal_nan=True)
    np.testing.assert_allclose(result['mu'][:, 0], info['mu'].ravel(), atol=1e-7, equal_nan=True)


@pytest.mark.parametrize('finite_distance', [False, True])
def test_full_vector_transform_against_independent_sunpy_basis(finite_distance):
    """Check all three axes away from disk center using Cartesian dot products."""
    solar_map = _map().resample([5, 9]*u.pix)
    hpc = all_coordinates_from_map(solar_map)
    surface = hpc.make_3d()
    carrington = surface.transform_to(frames.HeliographicCarrington(
        observer=solar_map.observer_coordinate, obstime=solar_map.date))
    lon, lat = carrington.lon.to_value(u.rad), carrington.lat.to_value(u.rad)
    # Independent physical definitions: outward, southward, westward.
    radial = np.stack([np.cos(lat)*np.cos(lon), np.cos(lat)*np.sin(lon), np.sin(lat)], -1)
    south = np.stack([np.sin(lat)*np.cos(lon), np.sin(lat)*np.sin(lon), -np.cos(lat)], -1)
    west = np.stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)], -1)
    local_axes = np.stack([radial, south, west], -2)
    L = solar_map.carrington_longitude.to_value(u.rad)
    B = solar_map.carrington_latitude.to_value(u.rad)
    observer_axes = np.array([
        [-np.sin(L), np.cos(L), 0],
        [-np.sin(B)*np.cos(L), -np.sin(B)*np.sin(L), np.cos(B)],
        [np.cos(B)*np.cos(L), np.cos(B)*np.sin(L), np.sin(B)],
    ])
    if finite_distance:
        hcc = surface.transform_to(frames.Heliocentric(
            observer=solar_map.observer_coordinate, obstime=solar_map.date))
        xyz = np.moveaxis(hcc.cartesian.xyz.to_value(u.m), 0, -1)
        ray = np.array([0, 0, solar_map.dsun.to_value(u.m)]) - xyz
        ray /= np.linalg.norm(ray, axis=-1, keepdims=True)
        tx, ty = hpc.Tx.to_value(u.rad), hpc.Ty.to_value(u.rad)
        xaxis = np.stack([np.cos(tx), np.zeros_like(tx), np.sin(tx)], -1)
        yaxis = np.stack([-np.sin(ty)*np.sin(tx), np.cos(ty), np.sin(ty)*np.cos(tx)], -1)
    else:
        xaxis = np.broadcast_to([1., 0., 0.], (*lon.shape, 3))
        yaxis = np.broadcast_to([0., 1., 0.], (*lon.shape, 3))
        ray = np.broadcast_to([0., 0., 1.], (*lon.shape, 3))
    roll = np.deg2rad(23.)
    detector_axes = np.stack([np.cos(roll)*xaxis+np.sin(roll)*yaxis,
                              -np.sin(roll)*xaxis+np.cos(roll)*yaxis, ray], -2)
    detector_global = detector_axes @ observer_axes
    expected = local_axes @ detector_global.swapaxes(-1, -2)
    actual = image_to_spherical_matrix(lon, lat, L, B, -roll,
        observer_distance=(solar_map.dsun/solar_map.rsun_meters).value if finite_distance else None)
    valid = np.isfinite(lon)
    assert valid.sum() > 5
    np.testing.assert_allclose(actual[valid], expected[valid], atol=2e-10)


@pytest.mark.parametrize('metadata, expected', [
    ({}, [0., 0., 0.]),
    ({'OBS_VR': 1200.}, [0., 0., 1200.]),
    ({'OBS_VR': '1200', 'OBS_VW': 200., 'OBS_VN': -50.}, [-200., -50., 1200.]),
    ({'OBS_VR': None, 'OBS_VW': '', 'OBS_VN': np.nan}, [0., 0., 0.]),
    ({'OBS_VR': np.inf, 'OBS_VN': 20.}, [0., 20., 0.]),
    ({'VLOS': 3000., 'VELOS': 2000.}, [0., 0., 0.]),
])
def test_optional_observer_velocity_keywords(metadata, expected):
    from spinnme.data.fits import observer_velocity_from_header
    np.testing.assert_array_equal(observer_velocity_from_header(metadata), expected)


@pytest.mark.parametrize('radial', [None, 1300.])
def test_map_geometry_works_with_absent_or_radial_only_velocity(radial):
    solar_map = _map()
    for key in ('OBS_VR', 'OBS_VW', 'OBS_VN'):
        solar_map.meta.pop(key)
    if radial is not None:
        solar_map.meta['OBS_VR'] = radial
    result = load_map_data(solar_map)
    np.testing.assert_array_equal(result['observer_velocity'], [0., 0., radial or 0.])
    finite = np.isfinite(result['mu'])
    if radial is None:
        np.testing.assert_array_equal(result['v_obs_los'][finite], 0.)
    else:
        assert result['v_obs_los'][3, 3] == pytest.approx(radial)
    assert 'OBS_VW' not in solar_map.meta
    assert 'OBS_VN' not in solar_map.meta


@pytest.mark.parametrize('roll_deg', [0., 23., -37., 90., 179.9])
@pytest.mark.parametrize('representation', ['pc', 'cd', 'crota'])
@pytest.mark.parametrize('resample', [False, True])
def test_detector_roll_and_hmi_azimuth_follow_wcs(roll_deg, representation, resample):
    """Check detector handedness independently of the projection implementation."""
    import torch
    from spinnme.data.util import spherical_to_cartesian_matrix
    from spinnme.loader.spherical import spherical_geometry_from_pixels
    from spinnme.train.spherical_synthesis import transform_spherical_parameters
    from spinnme.train.me_atmosphere import HMIMEAtmosphere, MEAtmosphere
    from spinnme.instrument import HMI_WAVELENGTH_CENTER

    solar_map = _map()
    header = dict(solar_map.meta)
    theta = np.deg2rad(roll_deg)
    rotation = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    for key in list(header):
        if key.lower().startswith(('pc1_', 'pc2_', 'cd1_', 'cd2_', 'crota')):
            del header[key]
    if representation == 'crota':
        header['crota2'] = roll_deg
    else:
        for i in range(2):
            for j in range(2):
                header[f'{representation}{i+1}_{j+1}'] = rotation[i, j] * (
                    header[f'cdelt{i+1}'] if representation == 'cd' else 1.)
    solar_map = Map(solar_map.data, header)
    if resample:
        # Unequal scale changes must not change the detector orientation.
        solar_map = solar_map.resample([5, 9]*u.pix)
    info = load_map_data(solar_map)
    from spinnme.grids import GridGeometry
    grid = GridGeometry({'shape': solar_map.data.shape, 'header': dict(solar_map.meta)})
    assert grid.angle == pytest.approx(-theta, abs=1e-12)
    assert info['pAng'].to_value(u.rad) == pytest.approx(-theta, abs=1e-12)
    ny, nx = solar_map.data.shape
    cy, cx = ny//2, nx//2
    geometry = spherical_geometry_from_pixels(
        np.array([cy*nx+cx]), solar_map.data.shape, solar_map.wcs,
        info['obs_lon'].to_value(u.rad), info['obs_lat'].to_value(u.rad), info['pAng'].to_value(u.rad),
        info['observer_distance'], info['observer_velocity'], 1., 0.,
    )
    # The WCS's columns specify detector +x and +y in solar west/north.
    world0 = np.array(solar_map.wcs.pixel_to_world_values(cx, cy))
    worldx = np.array(solar_map.wcs.pixel_to_world_values(cx+.001, cy))
    wx = worldx - world0
    wx[0] = (wx[0] + 180.) % 360. - 180.
    wx /= np.linalg.norm(wx)
    np.testing.assert_allclose(wx, rotation[:, 0], atol=1e-8)
    # At disk center, local phi points west and -theta points north.
    local_field = np.array([.4, -.8, .3])
    spherical = info['spherical_coords'][cy, cx]
    xyz = spherical_to_cartesian_matrix(spherical) @ local_field
    coords = torch.tensor(geometry['coords'], dtype=torch.float64)
    output = {f'b_{axis}': torch.tensor([[value]], dtype=torch.float64) for axis, value in zip('xyz', xyz)}
    output.update({f'v_{axis}': torch.zeros(1, 1, dtype=torch.float64) for axis in 'xyz'})
    transformed = transform_spherical_parameters(
        output, coords, torch.tensor(geometry['cartesian_to_spherical_transform']),
        torch.tensor(geometry['rtp_to_img_transform']), 695700000., 36000.,
    )
    bx = .3*np.cos(theta) + .8*np.sin(theta)
    by = -.3*np.sin(theta) + .8*np.cos(theta)
    torch.testing.assert_close(transformed['b_img'], torch.tensor([[bx, by, .4]], dtype=torch.float64))
    psi = np.arctan2(-bx, by)
    assert transformed['azi'].item() == pytest.approx(psi, abs=1e-7)
    # Independent HMI synthesis reference: azimuth from detector +x. The public
    # angle is from +y, so HMI must convert its double-angle products once.
    b2 = bx*bx + by*by + .4**2
    parameters = dict(b_field=torch.tensor([[1200.]], dtype=torch.float64),
        sin_inc2=transformed['sin_inc2'], cos_inc=transformed['cos_inc'],
        vmac=torch.tensor([[2000.]], dtype=torch.float64), damping=torch.tensor([[.2]], dtype=torch.float64),
        b0=torch.tensor([[.8]], dtype=torch.float64), b1=torch.tensor([[.2]], dtype=torch.float64),
        mu=torch.ones(1, 1, dtype=torch.float64), vdop=torch.zeros(1, 1, dtype=torch.float64),
        kl=torch.tensor([[25.]], dtype=torch.float64),
        wavelength_grid=torch.linspace(-.2, .2, 31, dtype=torch.float64)[None]*1e-10)
    actual = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).double()(
        **parameters, sin_inc2_cos2azi=transformed['sin_inc2_cos2azi'],
        sin_inc2_sin2azi=transformed['sin_inc2_sin2azi'])
    expected = MEAtmosphere(HMI_WAVELENGTH_CENTER, 1, 0, 2.5, 0).double()(
        **parameters, sin_inc2_cos2azi=torch.tensor([[(bx*bx-by*by)/(b2+1e-8)]], dtype=torch.float64),
        sin_inc2_sin2azi=torch.tensor([[2*bx*by/(b2+1e-8)]], dtype=torch.float64))
    for measured, reference in zip(actual, expected):
        torch.testing.assert_close(measured, reference)
