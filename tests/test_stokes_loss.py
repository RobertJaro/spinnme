import copy

import pytest
import torch

from spinnme.model import NormalizationModule
from spinnme.train.me_spherical_module import MESphericalModule
from spinnme.train.stokes_loss import StokesLossModule


def _profiles():
    target = torch.tensor([
        [
            [0.8, 0.7, 0.6, 0.5, 0.9, 1.0],
            [0.001, -0.002, 0.003, -0.004, 0.005, -0.006],
            [-0.002, 0.003, -0.004, 0.005, -0.006, 0.007],
            [0.004, -0.005, 0.006, -0.007, 0.008, -0.009],
        ]
    ], dtype=torch.float64)
    prediction = target + torch.tensor([0.01, 0.001, -0.001, 0.002], dtype=torch.float64).reshape(1, 4, 1)
    return prediction, target


def test_default_loss_is_transformed_mse():
    prediction, target = _profiles()
    normalization = NormalizationModule(asinh_alphas=.05).to(torch.float64)
    loss = StokesLossModule()

    actual = loss(prediction, target, normalization)
    expected = (normalization(prediction) - normalization(target)).square()

    torch.testing.assert_close(actual, expected)
    assert loss.configuration()['type'] == 'mse'


def test_loss_does_not_apply_continuum_normalization():
    prediction, target = _profiles()
    normalization = NormalizationModule().to(torch.float64)
    loss = StokesLossModule()

    actual = loss(prediction, target, normalization)
    torch.testing.assert_close(actual, (prediction - target).square())


def test_continuum_training_transform_matches_validation_and_preserves_gradients():
    prediction, target = _profiles()
    prediction = (prediction * 2).requires_grad_()
    target = target * 2
    loss = StokesLossModule(continuum_normalization=True)
    normalization = NormalizationModule(asinh_alphas=1e-3).to(torch.float64)
    prediction_plot, target_plot = loss.transform_profiles(prediction, target, normalization)
    continuum = torch.quantile(target[:, 0:1], .9, dim=-1, keepdim=True)
    predicted_continuum = torch.quantile(prediction[:, 0:1], .9, dim=-1, keepdim=True)
    torch.testing.assert_close(prediction_plot, normalization(prediction / predicted_continuum))
    torch.testing.assert_close(target_plot, normalization(target / continuum))
    torch.testing.assert_close(prediction_plot[:, 0], prediction[:, 0] / predicted_continuum[:, 0])
    actual = loss(prediction, target, normalization)
    torch.testing.assert_close(actual, (prediction_plot - target_plot).square())
    torch.testing.assert_close(actual, loss(prediction * 3, target * 7, normalization))
    actual.sum().backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad.abs().sum() > 0


def test_observed_continuum_variations_do_not_modulate_prediction():
    prediction, target = _profiles()
    prediction = prediction.expand(3, -1, -1).clone().requires_grad_()
    target = target * torch.tensor([.5, 2., 3.], dtype=target.dtype)[:, None, None]
    loss = StokesLossModule(continuum_normalization=True)
    prediction_plot, target_plot = loss.transform_profiles(prediction, target, NormalizationModule())
    torch.testing.assert_close(prediction_plot, prediction_plot[:1].expand_as(prediction_plot))
    torch.testing.assert_close(target_plot, target_plot[:1].expand_as(target_plot))
    actual = loss(prediction, target, NormalizationModule())
    actual.sum().backward()
    torch.testing.assert_close(prediction.grad, prediction.grad[:1].expand_as(prediction.grad))
    assert torch.isfinite(prediction.grad).all()


def test_continuum_configuration_round_trip():
    loss = StokesLossModule(continuum_normalization=True, continuum_quantile=.85)
    restored = StokesLossModule(**loss.configuration())
    assert restored.continuum_normalization
    assert restored.continuum_quantile == .85
    assert restored.configuration() != StokesLossModule().configuration()


def test_continuum_percentile_limits_single_bright_outlier():
    target = torch.ones(2, 4, 21)
    target[1, 0, -1] = 100
    prediction = torch.ones_like(target)
    loss = StokesLossModule(continuum_normalization=True)
    predicted, observed = loss.transform_profiles(prediction, target, NormalizationModule())
    torch.testing.assert_close(predicted, prediction)
    torch.testing.assert_close(observed[:, :, :-1], target[:, :, :-1])
    assert loss.valid_sample_mask(target).all()


def test_continuum_mask_uses_percentile_instead_of_positive_peak():
    target = torch.zeros(1, 4, 21)
    target[0, 0, -1] = 1
    loss = StokesLossModule(continuum_normalization=True)
    assert not loss.valid_sample_mask(target).any()
    assert torch.isnan(loss(target, target, NormalizationModule())).all()


@pytest.mark.parametrize('quantile', [0, -1, 1.1, float('nan')])
def test_invalid_continuum_quantile_is_rejected(quantile):
    with pytest.raises(ValueError, match='continuum_quantile'):
        StokesLossModule(continuum_quantile=quantile)


def test_continuum_mask_rejects_nonpositive_observed_intensity():
    _, target = _profiles()
    target[:, 0] = 0
    loss = StokesLossModule(continuum_normalization=True)
    assert not loss.valid_sample_mask(target).any()
    assert torch.isnan(loss(target, target, NormalizationModule())).all()


def test_asinh_is_applied_before_residual_subtraction():
    target = torch.ones(1, 4, 6, dtype=torch.float64)
    target[:, 1:] = .01
    prediction = target.clone()
    prediction[:, 1:] = .011
    normalization = NormalizationModule(asinh_alphas=.05).to(torch.float64)
    actual = normalization(prediction)[:, 1:] - normalization(target)[:, 1:]
    denominator = torch.asinh(torch.tensor(20.0, dtype=torch.float64))
    expected = (
        torch.asinh(torch.tensor(.011 / .05, dtype=torch.float64))
        - torch.asinh(torch.tensor(.01 / .05, dtype=torch.float64))
    ) / denominator
    incorrect = torch.asinh(torch.tensor(.001 / .05, dtype=torch.float64)) / denominator

    torch.testing.assert_close(actual, torch.full_like(actual, expected))
    assert not torch.isclose(actual.flatten()[0], incorrect)


def test_valid_sample_mask_depends_only_on_target():
    prediction, target = _profiles()
    prediction = torch.cat([prediction, prediction.clone()], dim=0)
    target = torch.cat([target, target.clone()], dim=0)
    prediction[1] = torch.nan
    loss = StokesLossModule()

    assert loss.valid_sample_mask(target).tolist() == [True, True]

    elementwise = loss(
        prediction,
        target,
        NormalizationModule(asinh_alphas=.05).to(torch.float64),
    )
    assert torch.isnan(elementwise[1]).all()


def test_mse_has_finite_prediction_gradients():
    prediction, target = _profiles()
    prediction.requires_grad_()
    normalization = NormalizationModule(asinh_alphas=.05).to(torch.float64)
    loss = StokesLossModule()

    loss(prediction, target, normalization).sum().backward()

    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()


def test_invalid_loss_configuration_is_rejected():
    with pytest.raises(ValueError, match='Unknown Stokes loss type'):
        StokesLossModule(type='unknown')


def _module(asinh_alpha=.05, weights_config=None, stokes_loss_config=None, stokes_scales=None):
    return MESphericalModule(
        image_shape=(1, 1),
        wavelength_config={},
        gauss_per_dB=1000,
        Rs_per_ds=1,
        seconds_per_dt=36000,
        instrument_config=[],
        model_config={
            'type': 'mlp',
            'dim': 4,
            'n_layers': 1,
            'encoding_config': {'type': 'identity'},
        },
        normalization_config={'asinh_alphas': asinh_alpha, 'stokes_scales': stokes_scales},
        stokes_loss_config=stokes_loss_config,
        weights_config=weights_config,
    )


def test_checkpoint_round_trip_and_required_metadata():
    module = _module()
    checkpoint = {'state_dict': copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(copy.deepcopy(checkpoint))
    for field in ('spinnme_checkpoint_version', 'spinnme_stokes_loss_config',
                  'spinnme_stokes_weights_config', 'spinnme_normalization_config'):
        incomplete = copy.deepcopy(checkpoint)
        incomplete.pop(field)
        with pytest.raises(RuntimeError, match='Unsupported SPINNME training checkpoint'):
            module.on_load_checkpoint(incomplete)


def test_checkpoint_rejects_silent_asinh_scale_switch():
    original = _module(asinh_alpha=.05)
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(original.state_dict())}
    original.on_save_checkpoint(checkpoint)

    changed = _module(asinh_alpha=.01)
    with pytest.raises(RuntimeError, match='fresh path'):
        changed.on_load_checkpoint(checkpoint)


def test_checkpoint_rejects_silent_continuum_switch():
    original = _module()
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(original.state_dict())}
    original.on_save_checkpoint(checkpoint)
    changed = _module(stokes_loss_config={'continuum_normalization': True})
    with pytest.raises(RuntimeError, match='fresh path'):
        changed.on_load_checkpoint(checkpoint)
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(changed.state_dict())}
    changed.on_save_checkpoint(checkpoint)
    changed.on_load_checkpoint(checkpoint)




def test_checkpoint_rejects_missing_continuum_mode():
    module = _module(stokes_loss_config={'continuum_normalization': True})
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    assert checkpoint.pop('spinnme_continuum_normalization_mode') == 'per_profile'
    with pytest.raises(RuntimeError, match='fresh path'):
        module.on_load_checkpoint(checkpoint)


def test_checkpoint_rejects_changed_quantile():
    module = _module(stokes_loss_config={'continuum_normalization': True})
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    checkpoint['spinnme_stokes_loss_config']['continuum_quantile'] = 1.0
    with pytest.raises(RuntimeError, match='fresh path'):
        module.on_load_checkpoint(checkpoint)
    peak_module = _module(stokes_loss_config={
        'continuum_normalization': True, 'continuum_quantile': 1.0,
    })
    peak_module.on_load_checkpoint(checkpoint)


def test_stokes_weight_schedules_are_stateless_and_replace_checkpoint_weights():
    schedule = {
        'type': 'smoothstep', 'start': 0.0, 'end': 1.0,
        'warmup_iterations': 20, 'iterations': 120,
    }
    module = _module(weights_config={'I': 1, 'Q': schedule, 'U': schedule, 'V': 1})

    module._set_stokes_weights_step(0)
    assert module.stokes_weights['Q'].item() == pytest.approx(0.0)
    module._set_stokes_weights_step(80)
    assert module.stokes_weights['Q'].item() == pytest.approx(0.5)
    module._set_stokes_weights_step(140)
    assert module.stokes_weights['Q'].item() == pytest.approx(1.0)
    assert module.stokes_weights['I'].item() == pytest.approx(1.0)

    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(module.state_dict()), 'global_step': 80}
    module.on_save_checkpoint(checkpoint)
    module.on_load_checkpoint(checkpoint)
    assert module.stokes_weights['Q'].item() == pytest.approx(0.5)

    replacement = {
        'type': 'exponential', 'start': 1.0e-6, 'end': 1.0,
        'iterations': 100000,
    }
    changed = _module(
        weights_config={'I': .5, 'Q': replacement, 'U': replacement, 'V': .25}
    )
    checkpoint['global_step'] = 50000
    with pytest.warns(UserWarning, match='will replace the checkpoint weights'):
        changed.on_load_checkpoint(checkpoint)
    assert changed.stokes_weights['I'].item() == pytest.approx(.5)
    assert changed.stokes_weights['Q'].item() == pytest.approx(1.0e-3)
    assert changed.stokes_weights['U'].item() == pytest.approx(1.0e-3)
    assert changed.stokes_weights['V'].item() == pytest.approx(.25)
    assert checkpoint['spinnme_stokes_weights_config'] == changed.stokes_weights_config


def test_fixed_scales_match_weighted_mse_and_gradients():
    prediction, target = _profiles()
    prediction.requires_grad_()
    normalization = NormalizationModule(stokes_scales={'I': 1, 'Q': .01, 'U': .01, 'V': .01}).double()
    loss = StokesLossModule()
    actual = loss(prediction, target, normalization)
    weights = prediction.new_tensor([1, 1e4, 1e4, 1e4]).reshape(1, 4, 1)
    expected = (prediction - target).square() * weights
    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    torch.testing.assert_close(prediction.grad, 2 * (prediction.detach() - target) * weights)


@pytest.mark.parametrize('scales', [[1, 0, 1, 1], [1, -1, 1, 1], [1, 1, 1], [1, float('nan'), 1, 1]])
def test_invalid_fixed_scales_are_rejected(scales):
    with pytest.raises(ValueError, match='finite positive'):
        NormalizationModule(stokes_scales=scales)


def test_fixed_scales_cannot_be_combined_with_asinh():
    with pytest.raises(ValueError, match='either'):
        NormalizationModule(asinh_alphas=.01, stokes_scales=[1, .01, .01, .01])


def test_fixed_scale_checkpoint_round_trip_and_objective_guard():
    module = _module(asinh_alpha=None, stokes_scales=[1, .01, .01, .01])
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    restored = _module(asinh_alpha=None, stokes_scales=[1, .01, .01, .01])
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'])
    for scales in [None, [1, .02, .01, .01]]:
        changed = _module(asinh_alpha=None, stokes_scales=scales)
        with pytest.raises(RuntimeError, match='fresh path'):
            changed.on_load_checkpoint(checkpoint)
    unscaled = _module(asinh_alpha=None)
    checkpoint = {'state_dict': copy.deepcopy(unscaled.state_dict())}
    unscaled.on_save_checkpoint(checkpoint)
    with pytest.raises(RuntimeError, match='fresh path'):
        module.on_load_checkpoint(checkpoint)


def test_checkpoint_rejects_target_only_normalization():
    module = _module(stokes_loss_config={'continuum_normalization': True})
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    checkpoint['spinnme_continuum_normalization_mode'] = 'target_only'
    with pytest.raises(RuntimeError, match='fresh path'):
        module.on_load_checkpoint(checkpoint)


def test_synthetic_continuum_gradient_matches_finite_differences():
    prediction, _ = _profiles()
    prediction.requires_grad_()
    loss = StokesLossModule(continuum_normalization=True)
    assert torch.autograd.gradcheck(loss.normalize_profile, (prediction,))


@pytest.mark.parametrize('model_type', ['mlp', 'siren'])
def test_normalized_spherical_source_uses_one_ratio(model_type):
    from spinnme.model import MESphericalModel, MESphericalSirenModel
    cls = MESphericalModel if model_type == 'mlp' else MESphericalSirenModel
    model = cls(dim=8, n_layers=2, source_function='ratio')
    coords = torch.randn(4, 4, requires_grad=True)
    output = model(coords)
    assert model.out_layer.out_features == 10
    torch.testing.assert_close(output['b0'] + output['b1'], torch.ones(4, 1))
    assert (output['b0'] > 0).all() and (output['b1'] > 0).all()
    (output['b1'].sum() + output['v_x'].sum() + output['kl'].sum()).backward()
    assert torch.isfinite(coords.grad).all()


def test_normalized_module_automatically_selects_source_ratio():
    module = _module(stokes_loss_config={'continuum_normalization': True})
    assert module.parameter_model.source_function == 'ratio'
    assert module.parameter_model.out_layer.out_features == 10
    checkpoint = {'coordinate_order': 'xyzt', 'state_dict': copy.deepcopy(module.state_dict())}
    module.on_save_checkpoint(checkpoint)
    restored = _module(stokes_loss_config={'continuum_normalization': True})
    restored.on_load_checkpoint(checkpoint)
    restored.load_state_dict(checkpoint['state_dict'])
