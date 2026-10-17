"""Supported optimizer selection, parameter coverage, schedules and resume guards."""
import pytest
import torch

from spinnme.train.me_spherical_module import MESphericalModule
from spinnme.train.soap import SOAP


def _module(model_type='mlp', optimizer_config=None):
    return MESphericalModule(
        image_shape=(1, 1), wavelength_config={}, instrument_config=[],
        gauss_per_dB=1000, Rs_per_ds=1, seconds_per_dt=36000,
        model_config={'type': model_type, 'dim': 8, 'n_layers': 3},
        lr_params={'start': .001, 'end': .0001, 'iterations': 10},
        optimizer_config=optimizer_config,
    )


@pytest.mark.parametrize('model_type', ['mlp', 'siren'])
@pytest.mark.parametrize('optimizer_type', ['soap', 'adamw'])
def test_optimizer_covers_trainable_parameters_once(model_type, optimizer_type):
    module = _module(model_type, {'type': optimizer_type})
    module.velocity_correction_models['test'] = torch.nn.Linear(2, 1)
    module.parameter_model.in_layer.requires_grad_(False)
    optimizer = module.configure_optimizers()['optimizer']
    actual = [id(p) for group in optimizer.param_groups for p in group['params']]
    assert len(actual) == len(set(actual))
    assert set(actual) == {id(p) for p in module.parameters() if p.requires_grad}


@pytest.mark.parametrize('optimizer_type', ['soap', 'adamw'])
def test_default_optimizer_and_schedule(optimizer_type):
    assert isinstance(_module().configure_optimizers()['optimizer'], SOAP)
    setup = _module(optimizer_config={'type': optimizer_type}).configure_optimizers()
    optimizer = setup['optimizer']
    scheduler = setup['lr_scheduler']['scheduler']
    assert all(group['weight_decay'] == 0 for group in optimizer.param_groups)
    for _ in range(15):
        optimizer.step()
        scheduler.step()
    assert [group['lr'] for group in optimizer.param_groups] == pytest.approx([.0001])


def test_checkpoint_rejects_switching_optimizer():
    soap = _module()
    checkpoint = {'state_dict': soap.state_dict(), 'optimizer_states': [{'state': {}}]}
    soap.on_save_checkpoint(checkpoint)
    with pytest.raises(ValueError, match='different optimizer'):
        _module(optimizer_config={'type': 'adamw'}).on_load_checkpoint(checkpoint)
    checkpoint.pop('spinnme_optimizer_type')
    with pytest.raises(RuntimeError, match='Unsupported SPINNME training checkpoint'):
        _module(optimizer_config={'type': 'adamw'}).on_load_checkpoint(checkpoint)


def test_unknown_optimizer_lists_supported_choices():
    with pytest.raises(ValueError, match='Use adamw or soap'):
        _module(optimizer_config={'type': 'unknown'}).configure_optimizers()
