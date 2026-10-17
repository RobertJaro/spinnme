"""Public spherical training contracts across instruments and spectral lines."""
import copy

import pytest
import torch
from astropy import units as u

from spinnme.instrument import hmi_wavelength_config
from spinnme.train.artifact_correction import ArtifactCorrectionModule
from spinnme.train.me_spherical_module import MESphericalModule
from _training_helpers import _module, _physical_batch


def multiline_module(**kwargs):
    return MESphericalModule(
        image_shape=(4, 4), gauss_per_dB=1000, Rs_per_ds=1, seconds_per_dt=36000,
        wavelength_config={'HMI': hmi_wavelength_config(), 'Hinode': {
            'wavelength_center': 6302.4931 * u.AA,
            'wavelength_grid': torch.linspace(-.2, .2, 11).numpy() * u.AA}},
        instrument_config=[{'type': 'hmi', 'instrument_id': 'HMI'},
                           {'type': 'generic', 'instrument_id': 'Hinode', 'line_id': 'fe-i-6302'}],
        model_config={'type': 'mlp', 'dim': 8, 'n_layers': 2,
                      'encoding_config': {'type': 'identity'}},
        stokes_loss_config={'continuum_normalization': True}, **kwargs)


def test_heterogeneous_wavelength_training_has_finite_gradients_and_shared_fields():
    module = multiline_module(instrument_weights={'HMI': 1, 'Hinode': 2})
    batch = _physical_batch()
    hinode = {key: value.clone() for key, value in batch['HMI'].items()
              if key not in ('spectral_offsets', 'spectral_weights', 'continuum_weights')}
    hinode['stokes'] = torch.zeros(16, 4, 11)
    hinode['stokes'][:, 0] = 1
    hinode['wavelength_grid'] = torch.linspace(-.2, .2, 11).expand(16, -1) * 1e-10
    batch['Hinode'] = hinode
    result = module.training_step(batch, 0)
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    for head in module.parameter_model.line_heads.values():
        assert head.weight.grad is not None and torch.isfinite(head.weight.grad).all()
    hmi = module.parameter_model(batch['HMI']['coords'], instrument_id='HMI')
    hinode_result = module.parameter_model(batch['HMI']['coords'], instrument_id='Hinode')
    for key in ('b_x', 'b_y', 'b_z', 'v_x', 'v_y', 'v_z'):
        torch.testing.assert_close(hmi[key], hinode_result[key])
    with torch.no_grad():
        module.parameter_model.line_heads['fe-i-6302'].bias[0] += 1
    assert not torch.allclose(hmi['vmac'], module.parameter_model(
        batch['HMI']['coords'], instrument_id='Hinode')['vmac'])


def test_default_objective_preserves_profile_weighted_single_line_behavior():
    module = _module()
    losses = {'HMI': torch.rand(3, 4, 6), 'PHI': torch.rand(7, 4, 6)}
    expected = module._reduce_stokes_objective(torch.cat(list(losses.values())))
    actual = module.reduce_instrument_losses(losses)
    for left, right in zip(actual, expected):
        torch.testing.assert_close(left, right)


def test_explicit_instrument_weights_and_wavelength_mean():
    module = multiline_module(instrument_weights={'HMI': 1, 'Hinode': 3}, wavelength_reduction='mean')
    components, objective = module.reduce_instrument_losses(
        {'HMI': torch.ones(2, 4, 6), 'Hinode': torch.full((9, 4, 11), 2.)})
    torch.testing.assert_close(components, torch.full((4,), 1.75))
    torch.testing.assert_close(objective, torch.tensor(7.))


def test_snapshot_constructor_recreates_line_heads_without_double_scaling():
    module = multiline_module()
    restored = MESphericalModule(**copy.deepcopy(module.snapshot_init))
    restored.load_state_dict(module.state_dict())
    coords = _physical_batch()['HMI']['coords']
    for instrument in ('HMI', 'Hinode'):
        expected = module.parameter_model(coords, instrument_id=instrument)
        actual = restored.parameter_model(coords, instrument_id=instrument)
        for key in expected:
            if expected[key] is not None:
                torch.testing.assert_close(actual[key], expected[key])


def test_replay_configuration_is_rejected():
    with pytest.raises(ValueError, match='Replay sampling was removed'):
        _module(replay_config={})


def test_artifact_initialization_is_identity_and_all_channels_are_active():
    module = ArtifactCorrectionModule()
    coords = torch.randn(5, 2)
    stokes = torch.randn(5, 4, 6)
    result = module(coords, stokes)
    torch.testing.assert_close(result['stokes_corr'], stokes, rtol=0, atol=0)
    torch.testing.assert_close(result['correction_gate'], torch.full((5, 1), .5))
    assert result['correction_params'].shape == (5, 4)
    assert 'filling_factor' not in result
    # At identity leakage channels receive gradients; after their first update
    # the gate also receives gradients. No unused neural output remains.
    with torch.no_grad():
        module.model.out_layer.bias[:3].fill_(1.)
    module(coords, stokes)['stokes_corr'].square().sum().backward()
    assert torch.isfinite(module.model.out_layer.bias.grad).all()
    assert (module.model.out_layer.bias.grad != 0).all()


def test_artifact_gate_has_correct_zero_and_full_limits():
    module = ArtifactCorrectionModule().double()
    stokes = torch.zeros(1, 4, 3, dtype=torch.float64)
    stokes[:, 0] = 2
    coords = torch.zeros(1, 2, dtype=torch.float64)
    with torch.no_grad():
        module.model.out_layer.bias.copy_(torch.tensor([2., 0, 0, -100.]))
    torch.testing.assert_close(module(coords, stokes)['stokes_corr'], stokes, atol=1e-15, rtol=0)
    with torch.no_grad():
        module.model.out_layer.bias[-1] = 100
    result = module(coords, stokes)
    torch.testing.assert_close(result['stokes_corr'][:, 1], torch.full((1, 3), .04, dtype=torch.float64))
    torch.testing.assert_close(result['stokes_corr'][:, 0], stokes[:, 0])


def test_dataset_weights_apply_within_a_shared_instrument():
    module = _module(wavelength_reduction='mean')
    losses = {'HMI': torch.stack((torch.ones(4, 6), torch.full((4, 6), 3.)))}
    components, objective = module.reduce_instrument_losses(losses, {'HMI': torch.tensor([[1.], [3.]])})
    torch.testing.assert_close(components, torch.full((4,), 2.5))
    torch.testing.assert_close(objective, torch.tensor(10.))


def test_hmi_atomic_overrides_cannot_silently_disagree_with_its_solver():
    from spinnme.instrument import resolve_instrument_line
    with pytest.raises(ValueError, match='atomic parameters'):
        resolve_instrument_line({'type': 'hmi', 'instrument_id': 'HMI',
            'atomic_parameters': {'j_up': 1., 'j_low': 0., 'g_up': 1., 'g_low': 0.}},
            hmi_wavelength_config())
