"""Passband quadrature and reuse in spherical observations."""
import copy

import numpy as np
import pytest
import torch
from astropy import units as u
from scipy.integrate import trapezoid

from _training_helpers import physical_inputs
from spinnme.instrument import hmi_wavelength_config
from spinnme.train.me_atmosphere import PHIMEAtmosphere
from spinnme.train.me_spherical_module import MESphericalModule
from spinnme.train.spectral_response import InstrumentSpectralResponse


@pytest.mark.parametrize('config', [
    {'type': 'gaussian', 'fwhm': .08, 'unit': 'Angstrom', 'samples': 7},
    {'type': 'airy', 'fwhm': .106, 'free_spectral_range': 3.01, 'unit': 'Angstrom', 'samples': 161},
])
def test_integration_matches_explicit_monochromatic_sum_and_gradients(config):
    response = InstrumentSpectralResponse(config).double()
    samples = response(4, 6)
    parameters, grid = physical_inputs(4, 6, 'cpu', torch.float64)
    atmosphere = PHIMEAtmosphere(wavelength_center=6173.3433 * u.AA).double()
    actual = atmosphere(wavelength_grid=grid, **parameters, **samples)
    manual = [torch.zeros_like(grid) for _ in range(4)]
    for i in range(samples['spectral_offsets'].shape[-1]):
        components = atmosphere(wavelength_grid=grid + samples['spectral_offsets'][..., i], **parameters)
        for channel, value in enumerate(components):
            manual[channel] = manual[channel] + value * samples['spectral_weights'][..., i]
    for left, right in zip(actual, manual):
        torch.testing.assert_close(left, right, atol=1e-12, rtol=1e-10)
    actual_grad = torch.autograd.grad(sum(v.square().sum() for v in actual), tuple(parameters.values()), retain_graph=True)
    manual_grad = torch.autograd.grad(sum(v.square().sum() for v in manual), tuple(parameters.values()))
    for left, right in zip(actual_grad, manual_grad):
        torch.testing.assert_close(left, right, atol=1e-11, rtol=1e-9)
    # One global response supports different observation wavelength counts.
    assert response(3, 11)['spectral_offsets'].shape == (3, 11, config['samples'])


def test_nominal_phi_airy_matches_dense_single_order_integral():
    # Independent uniform-grid reference, including line absorption in the
    # passband wings (the +/-300 mA continuum channel also needs convolution).
    x = np.linspace(-1.505, 1.505, 100001)
    transmission = 1 / (1 + (np.sin(np.pi * x / 3.01) / np.sin(np.pi * .106 / (2 * 3.01))) ** 2)
    area = trapezoid(transmission, x)
    previous = None
    for samples in (161, 321):
        response = InstrumentSpectralResponse({'type': 'airy', 'fwhm': .106,
            'free_spectral_range': 3.01, 'unit': 'Angstrom', 'samples': samples})
        offsets = response.offsets.numpy()[0] / 1e-10
        weights = response.weights.numpy()[0]
        assert np.all(weights > 0)
        np.testing.assert_allclose(weights.sum(), 1, atol=1e-7)
        np.testing.assert_allclose(offsets, -offsets[::-1], atol=1e-7)
        assert np.max(np.abs(offsets)) < 1.505
        values = []
        for shift in (0, -.14, .14, -.3, .3, .7):
            reference = trapezoid(transmission * np.exp(-((x - shift) / .03) ** 2), x) / area
            actual = np.sum(weights * np.exp(-((offsets - shift) / .03) ** 2))
            np.testing.assert_allclose(actual, reference, atol=1e-6, rtol=1e-5)
            values.append(actual)
        if previous is not None:
            np.testing.assert_allclose(values, previous, atol=1e-6, rtol=1e-5)
        previous = values


def test_airy_units_and_continuum_contribution():
    angstrom = InstrumentSpectralResponse({'type': 'airy', 'fwhm': .106,
        'free_spectral_range': 3.01, 'unit': 'Angstrom'})
    nm = InstrumentSpectralResponse({'type': 'airy', 'fwhm': .0106,
        'free_spectral_range': .301, 'unit': 'nm', 'continuum_weights': .25})
    torch.testing.assert_close(angstrom.offsets, nm.offsets)
    torch.testing.assert_close(angstrom.weights * .8, nm.weights)
    torch.testing.assert_close(nm.continuum, torch.tensor([.2]))
    # Both paths preserve a flat unpolarized spectrum.
    for response in (angstrom, nm):
        torch.testing.assert_close(response.weights.sum(-1) + response.continuum, torch.ones(1))


def test_tabulated_per_filter_profiles_and_continuum_are_normalized():
    response = InstrumentSpectralResponse({'type': 'tabulated', 'unit': 'mAngstrom',
        'offsets': [-20, 0, 20], 'weights': [[1, 2, 1], [1, 1, 1]], 'continuum_weights': [0, 1]})
    values = response(5, 2)
    torch.testing.assert_close(values['spectral_weights'].sum(-1) + values['continuum_weights'], torch.ones(5, 2))
    torch.testing.assert_close(values['spectral_offsets'][0, 0], torch.tensor([-2e-12, 0., 2e-12]))
    with pytest.raises(ValueError, match='observation has'):
        response(5, 6)


@pytest.mark.parametrize('response_config', [
    {'type': 'gaussian', 'fwhm': .08, 'unit': 'Angstrom', 'samples': 5},
    {'type': 'airy', 'fwhm': .106, 'free_spectral_range': 3.01, 'unit': 'Angstrom', 'samples': 161},
])
def test_configured_response_is_used_for_training_and_snapshot_constructor(response_config):
    module = MESphericalModule(image_shape=(2, 2), wavelength_config={'PHI': hmi_wavelength_config()},
        gauss_per_dB=1000, Rs_per_ds=1, seconds_per_dt=3600,
        instrument_config=[{'type': 'phi', 'instrument_id': 'PHI', 'spectral_response': response_config}],
        model_config={'type': 'mlp', 'dim': 4, 'n_layers': 1, 'encoding_config': {'type': 'identity'}})
    batch = {'coords': torch.zeros(4, 4), 'wavelength_grid': torch.zeros(4, 6)}
    actual = module._get_spectral_response(batch, 'PHI')
    assert actual['spectral_offsets'].shape == (4, 6, response_config['samples'])
    restored = MESphericalModule(**copy.deepcopy(module.snapshot_init))
    restored.load_state_dict(module.state_dict())
    for key, value in actual.items():
        torch.testing.assert_close(value, restored.default_spectral_response('PHI', 4, 6)[key])
    # Observation-specific calibration takes precedence over the instrument default.
    explicit = {key: value + 1 for key, value in actual.items()}
    selected = module._get_spectral_response({**batch, **explicit}, 'PHI')
    assert all(selected[key] is explicit[key] for key in explicit)


@pytest.mark.parametrize('config', [
    {'type': 'gaussian', 'fwhm': .08},
    {'type': 'gaussian', 'fwhm': -1, 'unit': 'Angstrom'},
    {'type': 'gaussian', 'fwhm': .08, 'unit': 'Angstrom', 'samples': 1},
    {'type': 'tabulated', 'unit': 'Angstrom', 'offsets': [0], 'weights': [-1]},
    {'type': 'tabulated', 'unit': 'Angstrom', 'offsets': [0], 'weights': [0]},
    {'type': 'airy', 'unit': 'Angstrom', 'fwhm': -.106, 'free_spectral_range': 3.01},
    {'type': 'airy', 'unit': 'Angstrom', 'fwhm': .106, 'free_spectral_range': .1},
    {'type': 'airy', 'unit': 'Angstrom', 'fwhm': .106, 'free_spectral_range': float('inf')},
    {'type': 'airy', 'unit': 'Angstrom', 'fwhm': .106, 'free_spectral_range': 3.01, 'samples': True},
    {'type': 'airy', 'unit': 'Angstrom', 'fwhm': .106, 'free_spectral_range': 3.01, 'samples': 2},
])
def test_invalid_passbands_are_rejected(config):
    with pytest.raises(ValueError):
        InstrumentSpectralResponse(config)
