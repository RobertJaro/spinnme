import pytest
import numpy as np
import torch
from astropy import units as u

from spinnme.instrument import HMI_WAVELENGTH_CENTER
from spinnme.train.me_atmosphere import HMIMEAtmosphere, MEAtmosphere
from spinnme.train.profile_functions import FaradayVoigt


def _atmosphere_parameters(batch_size=1):
    return {
        'b_field': torch.full((batch_size, 1), 1200.0),
        'sin_inc2_cos2azi': torch.full((batch_size, 1), 0.75 * 0.3),
        'sin_inc2_sin2azi': torch.full((batch_size, 1), 0.75 * np.sqrt(1 - 0.3 ** 2)),
        'sin_inc2': torch.full((batch_size, 1), 0.75),
        'cos_inc': torch.full((batch_size, 1), 0.5),
        'vmac': torch.full((batch_size, 1), 2000.0),
        'damping': torch.full((batch_size, 1), 0.2),
        'b0': torch.full((batch_size, 1), 0.8),
        'b1': torch.full((batch_size, 1), 0.2),
        'mu': torch.ones(batch_size, 1),
        'vdop': torch.zeros(batch_size, 1),
        'kl': torch.full((batch_size, 1), 25.0),
    }


def test_stokes_v_contains_magneto_optical_cross_term():
    atmosphere = MEAtmosphere(HMI_WAVELENGTH_CENTER, 1, 0, 2.5, 0)
    values = {name: torch.tensor([[value]], dtype=torch.float64) for name, value in {
        'b1': 0.2, 'delta': 3.0, 'mu': 0.8, 'eta_I': 0.4,
        'rho_Q': 0.3, 'rho_U': -0.2, 'rho_V': 0.1,
        'eta_Q': -0.5, 'eta_U': 0.25, 'eta_V': 0.7,
    }.items()}

    actual = atmosphere.compute_V(**values)
    k = 1 + values['eta_I']
    dot = (values['eta_Q'] * values['rho_Q'] + values['eta_U'] * values['rho_U']
           + values['eta_V'] * values['rho_V'])
    expected = -values['mu'] * values['b1'] / values['delta'] * (
        k.square() * values['eta_V']
        + k * (values['eta_U'] * values['rho_Q'] - values['eta_Q'] * values['rho_U'])
        + values['rho_V'] * dot
    )
    torch.testing.assert_close(actual, expected)


def test_analytic_stokes_solution_matches_propagation_matrix_inverse():
    atmosphere = MEAtmosphere(HMI_WAVELENGTH_CENTER, 1, 0, 2.5, 0)
    values = {name: torch.tensor([[value]], dtype=torch.float64) for name, value in {
        'eta_I': 0.4, 'eta_Q': -0.5, 'eta_U': 0.25, 'eta_V': 0.7,
        'rho_Q': 0.3, 'rho_U': -0.2, 'rho_V': 0.1,
        'b0': 0.8, 'b1': 0.2, 'mu': 0.8,
    }.items()}
    values['delta'] = atmosphere.delta(**values)
    analytic = torch.cat([
        atmosphere.compute_I(**values),
        atmosphere.compute_Q(**values),
        atmosphere.compute_U(**values),
        atmosphere.compute_V(**values),
    ], dim=-1).flatten()

    eta = 1 + values['eta_I'].item()
    propagation_matrix = torch.tensor([
        [eta, -0.5, 0.25, 0.7],
        [-0.5, eta, 0.1, 0.2],
        [0.25, -0.1, eta, 0.3],
        [0.7, -0.2, -0.3, eta],
    ], dtype=torch.float64)
    source = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64)
    expected = torch.tensor([0.8, 0.0, 0.0, 0.0], dtype=torch.float64)
    expected += 0.8 * 0.2 * torch.linalg.solve(propagation_matrix, source)

    torch.testing.assert_close(analytic, expected, rtol=1e-12, atol=1e-12)


def test_positive_hmi_doppler_velocity_redshifts_line():
    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER)
    wavelengths = torch.linspace(-0.15, 0.15, 1201).reshape(1, -1) * 1e-10
    parameters = _atmosphere_parameters()
    parameters.update({
        'b_field': torch.zeros(1, 1),
        'vdop': torch.full((1, 1), 3000.0),
    })

    intensity, _, _, _ = atmosphere(wavelength_grid=wavelengths, **parameters)
    line_minimum = wavelengths[0, torch.argmin(intensity[0])]
    expected_shift = HMI_WAVELENGTH_CENTER.to_value(u.m) * 3000.0 / 299792458.0

    assert line_minimum.item() > 0
    assert abs(line_minimum.item() - expected_shift) < 3e-13


def test_tabulated_spectral_profile_matches_explicit_integration_and_has_gradients():
    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER)
    wavelengths = torch.tensor([[-0.10, 0.0, 0.10]], dtype=torch.float64) * 1e-10
    parameters = {key: value.to(torch.float64) for key, value in _atmosphere_parameters().items()}
    parameters['vdop'].requires_grad_()

    offsets = torch.tensor([-0.02, 0.0, 0.02], dtype=torch.float64) * 1e-10
    weights = torch.tensor([1.0, 2.0, 1.0], dtype=torch.float64)
    weights /= weights.sum()
    spectral_offsets = offsets.reshape(1, 1, 3).expand(1, 3, 3)
    spectral_weights = weights.reshape(1, 1, 3).expand(1, 3, 3)
    continuum_weights = torch.zeros(1, 3, dtype=torch.float64)
    actual = atmosphere.to(torch.float64)(
        wavelength_grid=wavelengths,
        spectral_offsets=spectral_offsets,
        spectral_weights=spectral_weights,
        continuum_weights=continuum_weights,
        **parameters,
    )
    expanded = (wavelengths[..., None] + offsets).reshape(1, -1)
    explicit = atmosphere(wavelength_grid=expanded, **parameters)
    expected = tuple(component.reshape(1, 3, 3).mul(weights).sum(-1) for component in explicit)

    for actual_component, expected_component in zip(actual, expected):
        torch.testing.assert_close(actual_component, expected_component, rtol=2e-8, atol=1e-11)

    sum(component.square().mean() for component in actual).backward()
    assert torch.isfinite(parameters['vdop'].grad).all()


def test_far_wing_throughput_is_added_as_unpolarized_continuum():
    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER)
    parameters = _atmosphere_parameters()
    wavelengths = torch.tensor([[-0.1, 0.0, 0.1]]) * 1e-10

    actual = atmosphere(
        wavelength_grid=wavelengths,
        spectral_offsets=torch.zeros(1, 3, 1),
        spectral_weights=torch.full((1, 3, 1), 0.6),
        continuum_weights=torch.full((1, 3), 0.4),
        **parameters,
    )
    line = atmosphere(wavelength_grid=wavelengths, **parameters)
    continuum = parameters['b0'] + parameters['mu'] * parameters['b1']

    torch.testing.assert_close(actual[0], 0.6 * line[0] + 0.4 * continuum)
    for actual_component, line_component in zip(actual[1:], line[1:]):
        torch.testing.assert_close(actual_component, 0.6 * line_component)


def test_source_ratio_matches_scaled_observations_across_mu_and_doppler_shifts():
    from spinnme.model import NormalizationModule
    from spinnme.train.stokes_loss import StokesLossModule

    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).double()
    parameters = {key: value.double() for key, value in _atmosphere_parameters(6).items()}
    parameters['mu'] = torch.tensor([1., .8, .6, .4, .2, .1], dtype=torch.float64)[:, None]
    parameters['vdop'] = torch.linspace(-3000, 3000, 6, dtype=torch.float64)[:, None]
    wavelengths = torch.linspace(-.17, .17, 6, dtype=torch.float64)[None, :].expand(6, -1) * 1e-10
    response = dict(
        wavelength_grid=wavelengths,
        spectral_offsets=torch.tensor([-.015, 0., .015], dtype=torch.float64)[None, None, :].expand(6, 6, -1) * 1e-10,
        spectral_weights=torch.tensor([.2, .4, .2], dtype=torch.float64)[None, None, :].expand(6, 6, -1),
        continuum_weights=torch.full((6, 6), .2, dtype=torch.float64),
    )
    reference = torch.stack(atmosphere(**parameters, **response), dim=-2)
    gain = torch.linspace(.3, 2., 6, dtype=torch.float64)[:, None]
    scaled_parameters = {**parameters, 'b0': parameters['b0'] * gain, 'b1': parameters['b1'] * gain}
    observed = torch.stack(atmosphere(**scaled_parameters, **response), dim=-2).detach()
    loss = StokesLossModule(continuum_normalization=True)
    normalization = NormalizationModule(stokes_scales=[1, .01, .01, .01]).double()
    torch.testing.assert_close(loss(reference, observed, normalization), torch.zeros_like(reference), atol=1e-12, rtol=0)

    # Removing the overall source amplitude must retain sensitivity to its ratio.
    ratio = torch.full((6, 1), .25, dtype=torch.float64, requires_grad=True)
    shifted = {**parameters, 'b0': 1 - ratio, 'b1': ratio}
    prediction = torch.stack(atmosphere(**shifted, **response), dim=-2)
    objective = loss(prediction, observed, normalization).sum()
    objective.backward()
    assert objective > 0
    assert torch.isfinite(ratio.grad).all()
    assert (ratio.grad.abs() > 0).all()


def test_shared_faddeeva_preserves_synthesized_stokes_and_gradients():
    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).double()
    reference = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).double()
    reference.load_state_dict(atmosphere.state_dict(), strict=True)
    # Evaluate absorption and dispersion independently as a reference.
    reference.voigt.forward_pair_unit = lambda x, gamma: (
        reference.voigt(x, torch.ones_like(x), gamma),
        FaradayVoigt().to(x)(x, torch.ones_like(x), gamma))
    params = {k: v.double().requires_grad_() for k, v in _atmosphere_parameters(3).items()}
    wavelengths = torch.linspace(-.15, .15, 31, dtype=torch.float64).reshape(1, -1) * 1e-10
    calls = []
    handle = atmosphere.voigt.faddeeva.register_forward_hook(lambda *args: calls.append(1))
    actual = atmosphere(wavelength_grid=wavelengths, **params)
    handle.remove()
    expected = reference(wavelength_grid=wavelengths, **params)
    assert len(calls) == 3  # HMI's three Zeeman components: previously six evaluations.
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-11, atol=1e-12)
    a_grad = torch.autograd.grad(sum(v.square().sum() for v in actual), tuple(params.values()))
    b_grad = torch.autograd.grad(sum(v.square().sum() for v in expected), tuple(params.values()))
    for a, b in zip(a_grad, b_grad):
        torch.testing.assert_close(a, b, rtol=1e-9, atol=1e-11)


def _reference_profiles(atmosphere, nu, nu_m, damping, lambda_dop, **kwargs):
    gamma, sigma = torch.ones_like(nu) * damping, torch.ones_like(nu)
    result = {f'{prefix}_{component}': torch.zeros_like(nu)
              for component in ('b', 'p', 'r') for prefix in ('phi', 'psi')}
    for i_up in range(int(2 * atmosphere.j_up + 1)):
        m_up = atmosphere.j_up - i_up
        for component, delta_m in zip(('b', 'p', 'r'), (-1, 0, 1)):
            m_low = m_up + delta_m
            if abs(m_low) <= abs(atmosphere.j_low):
                strength = atmosphere.zeeman_strength(atmosphere.j_up, atmosphere.j_low, m_up, m_low)
                splitting = atmosphere.g_up * m_up - atmosphere.g_low * m_low
                shift = torch.ones_like(nu) * (lambda_dop - splitting * nu_m)
                result[f'phi_{component}'] += strength * atmosphere.voigt(nu - shift, sigma, gamma)
                result[f'psi_{component}'] += strength * FaradayVoigt().to(nu)(nu - shift, sigma, gamma)
    return result


def _reference_solve(atmosphere, **state):
    for key in ('eta_I', 'eta_Q', 'eta_U', 'eta_V', 'rho_Q', 'rho_U', 'rho_V', 'delta'):
        state[key] = getattr(atmosphere, key)(**state)
    return tuple(getattr(atmosphere, f'compute_{key}')(**state) for key in 'IQUV')


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
@pytest.mark.parametrize('levels', [(1, 0), (1.5, .5), (1, 1)])
def test_optimized_synthesis_matches_reference_outputs_and_gradients(dtype, levels):
    atmosphere = MEAtmosphere(HMI_WAVELENGTH_CENTER, *levels, 2.5, 1.2).to(dtype)
    reference = MEAtmosphere(HMI_WAVELENGTH_CENTER, *levels, 2.5, 1.2).to(dtype)
    reference.load_state_dict(atmosphere.state_dict(), strict=True)
    reference.calculate_voigt_faraday_profiles = lambda **state: _reference_profiles(reference, **state)
    reference._solve_profiles = lambda **state: _reference_solve(reference, **state)
    params = {k: v.to(dtype).requires_grad_() for k, v in _atmosphere_parameters(5).items()}
    params['damping'] = torch.tensor([0., 1e-6, .2, 1., 2.], dtype=dtype)[:, None].requires_grad_()
    params['mu'] = torch.tensor([1., .8, .5, .2, .01], dtype=dtype)[:, None].requires_grad_()
    params['vdop'] = torch.linspace(-4000, 4000, 5, dtype=dtype)[:, None].requires_grad_()
    wavelengths = torch.linspace(-.2, .2, 37, dtype=dtype)[None] * 1e-10
    actual = atmosphere(wavelength_grid=wavelengths, **params)
    expected = reference(wavelength_grid=wavelengths, **params)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=3e-5 if dtype == torch.float32 else 1e-10,
                                   atol=1e-7 if dtype == torch.float32 else 1e-12)
    actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), tuple(params.values()))
    expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), tuple(params.values()))
    for a, b in zip(actual_grad, expected_grad):
        torch.testing.assert_close(a, b, rtol=2e-4 if dtype == torch.float32 else 1e-9,
                                   atol=2e-6 if dtype == torch.float32 else 1e-11)


def test_transition_cache_load_and_no_scalar_reads():
    atmosphere = MEAtmosphere(HMI_WAVELENGTH_CENTER, 1, 0, 2.5, 0)
    state = atmosphere.state_dict()
    assert not any('transition' in key for key in state)
    state['g_up'] = torch.tensor(1.7)
    atmosphere.load_state_dict(state, strict=True)
    torch.testing.assert_close(atmosphere._transition_splittings, torch.tensor([1.7, 0., -1.7]))
    params = _atmosphere_parameters()
    wavelengths = torch.linspace(-.15, .15, 6)[None] * 1e-10
    atmosphere(wavelength_grid=wavelengths, **params)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU]) as profile:
        atmosphere(wavelength_grid=wavelengths, **params)
    assert 'aten::_local_scalar_dense' not in {event.key for event in profile.key_averages()}
