"""Configuration failures should precede data preparation or optimizer updates."""
import copy

import numpy as np
import pytest
from astropy import units as u

from spinnme.configuration import list_configs, read_config, resolve_config, training_budget, validate_config
from spinnme.instrument import SPECTRAL_LINES
from spinnme.model import NormalizationModule
from spinnme.train.me_spherical_module import MESphericalModule
from spinnme.train.stokes_loss import StokesLossModule
from spinnme.training.runner import run


@pytest.mark.parametrize('name', list_configs())
def test_packaged_templates_construct_matching_spherical_models(name):
    config = validate_config(read_config(name), require_input=False)
    wavelengths = {}
    for instrument in config['instrument']:
        center = SPECTRAL_LINES[instrument['line_id']]['wavelength_center']
        wavelengths[instrument['instrument_id']] = {
            'wavelength_center': center, 'wavelength_grid': np.linspace(-.2, .2, 6) * u.AA}
    training = copy.deepcopy(config['training'])
    for key in ('epochs', 'max_steps', 'check_val_every_n_epoch', 'val_check_interval'):
        training.pop(key, None)
    module = MESphericalModule(
        image_shape=(4, 4), wavelength_config=wavelengths,
        instrument_config=config['instrument'], model_config=config['model'],
        stokes_loss_config=config.get('stokes_loss'), normalization_config=config.get('normalization'),
        weights_config=config.get('weights'), gauss_per_dB=config['data']['gauss_per_dB'],
        Rs_per_ds=config['data'].get('Rs_per_ds', 1), seconds_per_dt=config['data']['seconds_per_dt'],
        **training)
    assert module.parameter_model.source_function == (
        'ratio' if module.stokes_loss.continuum_normalization else 'independent')
    assert set(module.forward_models) == set(wavelengths)


@pytest.mark.parametrize('budget', [{}, {'epochs': -1}, {'max_steps': True}, {'epochs': 1.5},
                                    {'max_steps': 1, 'epochs': 0}, {'max_steps': 1, 'epochs': -2}])
def test_invalid_budgets_are_rejected_by_yaml_and_python_before_writing(tmp_path, budget):
    config = read_config('synthetic/circular')
    config['training'] = budget
    with pytest.raises(ValueError):
        validate_config(config, require_input=False)
    with pytest.raises(ValueError):
        run({'path': str(tmp_path / 'untouched'), 'training': budget})
    assert not (tmp_path / 'untouched').exists()


def test_limits_are_explicit_and_default_unset_limit_is_unbounded():
    assert training_budget({'max_steps': 1}) == (-1, 1)
    assert training_budget({'epochs': 2}) == (2, -1)


@pytest.mark.parametrize(('field', 'value'), [('training', None), ('data', None),
    ('instrument', {}), ('instrument', [{}])])
def test_malformed_configuration_containers_raise_actionable_errors(field, value):
    config = read_config('synthetic/circular')
    config[field] = value
    with pytest.raises(ValueError, match=field):
        validate_config(config, require_input=False)


@pytest.mark.parametrize('alpha', [float('nan'), float('inf'), -1, 0])
def test_asinh_scales_cannot_poison_training(alpha):
    with pytest.raises(ValueError, match='finite and strictly positive'):
        NormalizationModule(asinh_alphas=alpha)


def test_continuum_normalization_requires_explicit_boolean():
    with pytest.raises(ValueError, match='boolean'):
        StokesLossModule(continuum_normalization='false')


def test_malformed_yaml_identifies_configuration_override_and_instrument(tmp_path):
    path = tmp_path / 'broken.yaml'
    path.write_text('data: [unfinished')
    with pytest.raises(ValueError, match='Invalid YAML in configuration'):
        read_config(path)
    with pytest.raises(ValueError, match='Invalid YAML in override'):
        resolve_config('synthetic/circular', overrides=['training.max_steps=[unfinished'])
    with pytest.raises(ValueError, match='Invalid YAML in instrument'):
        resolve_config('synthetic/circular', instrument=path)


@pytest.mark.parametrize('content', ['null', '[null]', 'model: []', 'instrument_id: X\ndata: []'])
def test_instrument_yaml_requires_mapping_settings(tmp_path, content):
    path = tmp_path / 'instrument.yaml'
    path.write_text(content)
    with pytest.raises(ValueError, match='mapping'):
        resolve_config('synthetic/circular', instrument=path)


def test_run_configuration_has_no_seed_and_uses_named_stokes_weights():
    config = resolve_config('synthetic/circular', require_input=False)
    assert 'seed' not in config
    assert config['weights'] == {'I': 1, 'Q': 1, 'U': 1, 'V': 1}
    assert config['logging']['project'] == 'spinnme'
    config['lambda'] = config.pop('weights')
    with pytest.raises(ValueError, match='lambda'):
        validate_config(config, require_input=False)


def test_wandb_project_is_spinnme():
    config = read_config('synthetic/circular')
    config['logging']['project'] = 'other'
    with pytest.raises(ValueError, match='logging.project must be spinnme'):
        validate_config(config, require_input=False)
