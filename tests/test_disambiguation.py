"""Continuous synthesis-only LOS rotation, independent of physical B losses."""
import copy
import math

import numpy as np
import pytest
import torch

from spinnme.model import MESphericalModel, MESphericalSirenModel
from spinnme.train.collocation import LongitudeInterval, SphericalCollocationDomain
from spinnme.train.spherical_synthesis import transform_spherical_parameters
from spinnme.train.me_spherical_module import MESphericalModule
from _training_helpers import _module, _physical_batch


def transform(field, angle=None):
    field = field.reshape(1, 3)
    output = {f'b_{axis}': field[:, i:i+1] for i, axis in enumerate('xyz')}
    output.update({f'v_{axis}': torch.zeros_like(field[:, :1]) for axis in 'xyz'})
    if angle is not None:
        output['disambiguation_angle'] = angle.reshape(1, 1)
    matrix = torch.eye(3, dtype=field.dtype)[None]
    return transform_spherical_parameters(
        output, field.new_tensor([[1., 0., 0., 0.]]), matrix, matrix, 695700000., 3600.)


@pytest.mark.parametrize('angle', [0., math.pi/2, math.pi, 1.234, 2*math.pi])
def test_rotation_changes_only_synthesis_transverse_field(angle):
    field = torch.tensor([3., 4., 12.], dtype=torch.float64, requires_grad=True)
    rotation = field.new_tensor(angle, requires_grad=True)
    actual = transform(field, rotation)
    base = transform(field)
    expected = field.new_tensor([[3*math.cos(angle)-4*math.sin(angle),
                                 3*math.sin(angle)+4*math.cos(angle), 12.]])
    torch.testing.assert_close(actual['b_synthesis_img'], expected)
    for key in ('b_img', 'b_xyz', 'b_rtp', 'b_field', 'cos_inc', 'sin_inc2', 'inc', 'vdop'):
        torch.testing.assert_close(actual[key], base[key])
    psi = base['azi'] + angle
    torch.testing.assert_close(actual['sin_inc2_sin2azi'], base['sin_inc2'] * torch.sin(2*psi))
    torch.testing.assert_close(actual['sin_inc2_cos2azi'], base['sin_inc2'] * torch.cos(2*psi))
    actual['sin_inc2_sin2azi'].sum().backward()
    assert rotation.grad is not None and torch.isfinite(rotation.grad)
    assert torch.isfinite(field.grad).all()


@pytest.mark.parametrize('model_type', [MESphericalModel, MESphericalSirenModel])
@pytest.mark.parametrize('source', ['ratio', 'independent'])
def test_optional_head_restores_and_starts_with_broad_angles(model_type, source):
    torch.manual_seed(13)
    model = model_type(dim=16, n_layers=4, disambiguation=True, source_function=source)
    coords = torch.randn(2048, 4)
    coords[:, :3] = torch.nn.functional.normalize(coords[:, :3], dim=-1)
    coords[:, 3] *= .05
    output = model(coords)
    angles = output['disambiguation_angle'].detach().flatten()
    assert angles.quantile(.95) - angles.quantile(.05) > .5
    assert model.out_layer.out_features == (11 if source == 'ratio' else 12)
    assert not hasattr(model, 'disambiguation_head')
    torch.testing.assert_close(output['disambiguation_angle'],
                               model.out_layer(model.features(coords))[:, -1:])
    with torch.no_grad():
        original_bias = model.out_layer.bias[-1].clone()
        model.out_layer.bias[-1].add_(.3)
        shifted = model(coords)
        for key, value in output.items():
            if value is not None:
                expected = value + .3 if key == 'disambiguation_angle' else value
                torch.testing.assert_close(shifted[key], expected)
        model.out_layer.bias[-1].copy_(original_bias)
    restored = model_type(dim=16, n_layers=4, disambiguation=True, source_function=source)
    restored.load_state_dict(model.state_dict(), strict=True)
    for key, value in output.items():
        if value is not None:
            torch.testing.assert_close(restored(coords)[key], value)
    disabled = model_type(dim=16, n_layers=4, source_function=source)
    assert 'disambiguation_angle' not in disabled(coords)
    assert not any('disambiguation' in key for key in disabled.state_dict())


def make_module():
    config = copy.deepcopy(_module().snapshot_init)
    config['model_config']['disambiguation'] = True
    config['physics_config'] = {
        'num_points': 32, 'normalization': 'characteristic', 'radius_range_Rs': [1., 1.1],
        'constraints': {'divergence': .1, 'force_free': .1}}
    config['physics_domain'] = SphericalCollocationDomain(
        (0., 1.), (-.2, .2), LongitudeInterval(0., .5), (1., 1.1))
    return MESphericalModule(**config)


@pytest.mark.parametrize('vector_potential', [False, True])
@pytest.mark.parametrize('compiled', [False, True])
def test_physics_ignores_rotation_but_stokes_trains_it(vector_potential, compiled, monkeypatch):
    torch.manual_seed(7)
    module = make_module()
    config = copy.deepcopy(module.snapshot_init)
    config['model_config']['vector_potential'] = vector_potential
    if compiled:
        import spinnme.train.me_atmosphere as atmosphere
        solver = torch.compile(atmosphere.MEAtmosphere._solve_profiles, backend='aot_eager', fullgraph=True)
        monkeypatch.setattr(atmosphere, 'compiled_profile_solver', lambda mode: solver)
        config['compile_config'] = {'me_solver': True}
    module = MESphericalModule(**config)
    head = module.parameter_model.out_layer
    reference = torch.tensor(0.)
    first = module.physics_regularization.evaluate(module.parameter_model, global_step=0, reference=reference)
    grads = torch.autograd.grad(first.total, (head.weight, head.bias), allow_unused=True)
    assert all(grad is None or torch.count_nonzero(grad[-1]) == 0 for grad in grads)
    with torch.no_grad():
        head.bias[-1].add_(.3)
    second = module.physics_regularization.evaluate(module.parameter_model, global_step=0, reference=reference)
    torch.testing.assert_close(second.total, first.total)
    batch = _physical_batch()
    # Nonzero polarization makes the Stokes objective sensitive to rotation.
    batch['HMI']['stokes'][:, 1] = .01
    batch['HMI']['stokes'][:, 2] = -.02
    result = module.training_step(batch, 0)
    result['loss'].backward()
    assert torch.isfinite(result['loss'])
    assert head.bias.grad is not None and head.bias.grad[-1].abs() > 0
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    restored = MESphericalModule(**copy.deepcopy(module.snapshot_init))
    restored.load_state_dict(module.state_dict(), strict=True)
    coords = _physical_batch()['HMI']['coords'].requires_grad_()
    torch.testing.assert_close(restored.parameter_model(coords)['disambiguation_angle'],
                               module.parameter_model(coords)['disambiguation_angle'])


def test_validation_plot_displays_rotation_only_when_present(monkeypatch):
    module = make_module()
    batch = _physical_batch()['HMI']
    batch.update(instrument_id='HMI', lin_idx=torch.tensor([0]), pix=torch.zeros(16, 2))
    result = module.validation_step(batch, 0)
    assert result['disambiguation_angle'].shape == (16, 1)
    parameters = {key: result[key].detach().numpy().reshape(4, 4) for key in
                  ('b_field', 'inc', 'azi', 'vmac', 'damping', 'b0', 'b1', 'vdop', 'kl', 'disambiguation_angle')}
    figures = []
    monkeypatch.setattr(module, '_log_validation_image', lambda fig, name: figures.append(fig))
    module.plot_parameter_overview(parameters)
    image = figures[0].axes[7].images[0]
    np.testing.assert_allclose(image.get_array(), parameters['disambiguation_angle'] % (2*np.pi))
    assert image.get_clim() == (0, 2*np.pi)
    assert 'rotation' in figures[0].axes[7].get_title()
    del parameters['disambiguation_angle']
    module.plot_parameter_overview(parameters)
    assert not figures[1].axes[7].images


def test_portable_snapshot_preserves_rotation_and_synthesized_profiles(tmp_path):
    from datetime import datetime
    from types import SimpleNamespace
    from spinnme.snapshots import save_snapshot, Snapshot

    module = make_module()
    batch = _physical_batch()['HMI']
    batch.update(instrument_id='HMI', lin_idx=torch.tensor([0]), pix=torch.zeros(16, 2))
    expected = module.validation_step(batch, 0)
    data = SimpleNamespace(
        image_shape=module.image_shape, wavelength_config=module.wavelength_config,
        ref_time=datetime(2024, 3, 23), times=[datetime(2024, 3, 23)])
    path = save_snapshot(module, data, {}, tmp_path/'rotation.pme')
    with Snapshot(path, device='cpu') as snapshot:
        actual = snapshot.synthesize_observation(
            batch['coords'], batch['cartesian_to_spherical_transform'], batch['rtp_to_img_transform'],
            batch['v_obs_los'], instrument_id='HMI', wavelength_grid=batch['wavelength_grid'],
            spectral_offsets=batch['spectral_offsets'], spectral_weights=batch['spectral_weights'],
            continuum_weights=batch['continuum_weights'], progress=False)
        np.testing.assert_allclose(actual['disambiguation_angle'],
                                   expected['disambiguation_angle'].detach().numpy(), atol=1e-6)
        # Synthesis returns continuum-normalized profiles, before the training
        # component scaling. Apply that same scaling before comparison.
        normalized = module.normalization(torch.tensor(actual['stokes']))
        torch.testing.assert_close(normalized, expected['stokes_pred'].detach(), atol=1e-6, rtol=1e-5)
        np.testing.assert_allclose(actual['b_img'][:, 2], actual['b_synthesis_img'][:, 2], atol=1e-6)


@pytest.mark.parametrize('angle', [.4, math.pi/2, math.pi, 2*math.pi])
def test_hmi_spectra_follow_transverse_rotation_with_unchanged_i_and_v(angle):
    from spinnme.instrument import HMI_WAVELENGTH_CENTER
    from spinnme.train.me_atmosphere import HMIMEAtmosphere

    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).double()
    field = torch.tensor([.3, .4, 1.2], dtype=torch.float64)
    parameters = {key: torch.tensor([[value]], dtype=torch.float64) for key, value in {
        'b_field': 1300., 'vmac': 2000., 'damping': .2, 'b0': .8, 'b1': .2,
        'mu': .9, 'vdop': 0., 'kl': 25.,
    }.items()}
    parameters['wavelength_grid'] = torch.linspace(-.2, .2, 31, dtype=torch.float64)[None]*1e-10

    def profiles(rotation):
        result = transform(field, field.new_tensor(rotation))
        return atmosphere(**parameters, **{key: result[key] for key in (
            'sin_inc2', 'cos_inc', 'sin_inc2_sin2azi', 'sin_inc2_cos2azi')})

    i, q, u, v = profiles(0.)
    actual_i, actual_q, actual_u, actual_v = profiles(angle)
    torch.testing.assert_close(actual_i, i)
    torch.testing.assert_close(actual_v, v)
    torch.testing.assert_close(actual_q, math.cos(2*angle)*q - math.sin(2*angle)*u)
    torch.testing.assert_close(actual_u, math.sin(2*angle)*q + math.cos(2*angle)*u)


@pytest.mark.parametrize('bias', [-100., 100.])
def test_single_linear_output_is_unbounded_and_unscaled(bias):
    model = MESphericalModel(dim=16, n_layers=2, disambiguation=True,
                             encoding_config={'type': 'identity'})
    coords = torch.randn(128, 4)
    with torch.no_grad():
        model.out_layer.weight[-1].zero_()
        model.out_layer.bias[-1] = bias
    angle = model(coords)['disambiguation_angle']
    torch.testing.assert_close(angle, torch.full((128, 1), bias))
    angle.sum().backward()
    torch.testing.assert_close(model.out_layer.bias.grad[-1], torch.tensor(128.))
    model.zero_grad()
    model(coords)['disambiguation_angle'].sin().sum().backward()
    assert model.out_layer.bias.grad[-1].abs() > 0
    assert torch.isfinite(model.out_layer.bias.grad).all()


def test_old_separate_rotation_head_checkpoint_is_rejected():
    model = MESphericalModel(dim=16, n_layers=2, disambiguation=True)
    previous_state = model.state_dict()
    previous_state['out_layer.weight'] = previous_state['out_layer.weight'][:-1]
    previous_state['out_layer.bias'] = previous_state['out_layer.bias'][:-1]
    with pytest.raises(RuntimeError, match='size mismatch'):
        model.load_state_dict(previous_state, strict=True)
