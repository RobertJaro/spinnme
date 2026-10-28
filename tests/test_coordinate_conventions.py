"""Independent physical bases, calibrated SP geometry, and effective HMI time."""
import numpy as np
import pytest
import torch
from astropy import units as u

from spinnme.data.hinode import slit_coordinates
from spinnme.data.hmi_time import hmi_coordinate_header, hmi_observation_time
from spinnme.data.util import image_to_spherical_matrix
from spinnme.instrument import SPECTRAL_LINES
from spinnme.train.me_atmosphere import HinodeMEAtmosphere, MEAtmosphere


@pytest.mark.parametrize('line_id', ['fe-i-6301', 'fe-i-6302'])
@pytest.mark.parametrize('roll', [-2.8, -.3, 0., .7, np.pi/2])
def test_calibrated_hinode_profiles_invariant_under_image_roll(line_id, roll):
    # At disk center the independent physical components are radial, south, west.
    # Project the SAME vector through the actual public geometric transform.
    radial, north, west = 800., 450., -670.
    image = image_to_spherical_matrix(0., 0., 0., 0., roll).T @ [radial, -north, west]
    bx, by, bz = image
    b2 = radial**2+north**2+west**2
    t = lambda x: torch.tensor([[x]], dtype=torch.float64)
    inputs = dict(b_field=t(np.sqrt(b2)), sin_inc2=t((north**2+west**2)/b2),
                  cos_inc=t(radial/np.sqrt(b2)), vmac=t(1500.), damping=t(.2),
                  b0=t(.3), b1=t(.7), mu=t(1.), vdop=t(500.), kl=t(10.),
                  wavelength_grid=torch.linspace(-.25, .25, 31, dtype=torch.float64)[None]*1e-10)
    calibrated = MEAtmosphere(**SPECTRAL_LINES[line_id]).double()
    reference = calibrated(**inputs, sin_inc2_cos2azi=t((west**2-north**2)/b2),
                           sin_inc2_sin2azi=t(2*west*north/b2))
    public_c, public_s = t(-(bx**2-by**2)/b2), t(-2*bx*by/b2)
    public_c.requires_grad_(); public_s.requires_grad_()
    actual = HinodeMEAtmosphere(**SPECTRAL_LINES[line_id]).double()(
        **inputs, sin_inc2_cos2azi=public_c, sin_inc2_sin2azi=public_s, image_p_angle=t(roll))
    for predicted, expected in zip(actual, reference):
        torch.testing.assert_close(predicted, expected, atol=1e-12, rtol=1e-10)
    sum(value.square().sum() for value in actual).backward()
    assert torch.isfinite(public_c.grad).all() and torch.isfinite(public_s.grad).all()


def test_hinode_requires_explicit_roll():
    with pytest.raises(ValueError, match='image_p_angle'):
        HinodeMEAtmosphere(**SPECTRAL_LINES['fe-i-6302'])(sin_inc2_cos2azi=torch.ones(1),
                                                       sin_inc2_sin2azi=torch.zeros(1))


@pytest.mark.parametrize('first,last,binning', [(0, 1023, 1), (0, 1023, 2), (100, 499, 2), (700, 799, 1)])
def test_solarsoft_slit_roi_origin_and_clockwise_spatial_roll(first, last, binning):
    header = dict(SPCCDIX0=first, SPCCDIX1=last, CAMSSUM=binning, CROTA2=29., FGSPROT=1.,
                  XCEN=10., YCEN=-20., YSCALE=.32)
    length = (last-first+1)//binning
    coefficients, angle = slit_coordinates(header, length)
    # Literal full detector endpoints from SolarSoft stksimages_sbsp.pro.
    offsets = (1023-last + np.array([0, length-1])*binning - 511.5)*.32/binning
    expected = np.stack([10.+.5*offsets, -20.+np.sqrt(3)/2*offsets])
    actual = np.asarray(coefficients[:2])[:, None] + np.asarray(coefficients[2:])[:, None]*[0, length-1]
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert angle == pytest.approx(np.pi/6)


def test_slit_geometry_does_not_guess_missing_roi():
    with pytest.raises(ValueError, match='SPCCDIX'):
        slit_coordinates(dict(CROTA2=0., FGSPROT=0., XCEN=0., YCEN=0., YSCALE=.32), 10)


@pytest.mark.parametrize('suffix', ['', '_TAI'])
def test_hmi_effective_tai_time_is_not_exposure_start_or_record_slot(suffix):
    header = dict(T_OBS='2024.03.23_01:00:00.125'+suffix,
                  T_REC='2024.03.23_01:00:02_TAI', **{'DATE-OBS': '2024-03-23T00:58:38.125'})
    assert hmi_observation_time(header).isoformat() == '2024-03-23T00:59:23.125000'
    normalized = hmi_coordinate_header(header)
    assert normalized['DATE-OBS'] == header['DATE-OBS']
    assert normalized['DATE-AVG'] == '2024-03-23T00:59:23.125'
    assert normalized['T_REC'] == header['T_REC']
    assert header['T_OBS'].endswith(suffix)


@pytest.mark.parametrize('version,expected', [(1, MEAtmosphere), (2, HinodeMEAtmosphere)])
def test_snapshot_init_selects_saved_hinode_polarization_convention(version, expected):
    from spinnme.train.me_spherical_module import MESphericalModule
    module = MESphericalModule(image_shape=(2, 2), wavelength_config={'SP': {
        'wavelength_center': 6302.4931*u.AA, 'wavelength_grid': np.linspace(-.2, .2, 5)*u.AA}},
        gauss_per_dB=1000., Rs_per_ds=1., seconds_per_dt=3600.,
        model_config={'type': 'mlp', 'dim': 8, 'n_layers': 2, 'encoding_config': {'type': 'identity'}},
        instrument_config=[{'type': 'hinode', 'instrument_id': 'SP', 'line_id': 'fe-i-6302'}],
        coordinate_convention_version=version)
    assert type(module.forward_models['SP']) is expected
