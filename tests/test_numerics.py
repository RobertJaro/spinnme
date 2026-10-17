import json
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from _training_helpers import _module, _physical_batch


def attach_report_directory(module, directory):
    module._trainer = SimpleNamespace(default_root_dir=str(directory), checkpoint_callback=None,
                                      global_step=12, global_rank=2)


def test_bad_gradients_are_reported_without_clipping_or_updating(tmp_path):
    module = _module(None)
    attach_report_directory(module, tmp_path)
    optimizer = module.configure_optimizers()['optimizer']
    parameter = module.parameter_model.out_layer.weight
    original = parameter.detach().clone()
    parameter.grad = torch.ones_like(parameter)
    parameter.grad[0, 0] = float('nan')
    original_grad = parameter.grad.clone()
    with pytest.raises(RuntimeError, match='optimizer update aborted'):
        module.configure_gradient_clipping(optimizer, .5, 'norm')
    torch.testing.assert_close(parameter, original)
    torch.testing.assert_close(parameter.grad, original_grad, equal_nan=True)
    assert len(optimizer.state) == 0
    report = json.loads((tmp_path / 'numerical_failures/rank2-step12-backward.json').read_text())
    assert report['gradients']['parameter_model.out_layer.weight']['nan'] == 1
    assert report['parameters']['parameter_model.out_layer.weight']['nonfinite'] == 0


def test_finite_clipping_matches_previous_behavior(tmp_path):
    module = _module(None)
    attach_report_directory(module, tmp_path)
    optimizer = module.configure_optimizers()['optimizer']
    parameter = module.parameter_model.out_layer.weight
    parameter.grad = torch.randn_like(parameter)
    reference = torch.nn.Parameter(parameter.detach().clone())
    reference.grad = parameter.grad.clone()
    torch.nn.utils.clip_grad_norm_([reference], .5)
    module.configure_gradient_clipping(optimizer, .5, 'norm')
    torch.testing.assert_close(parameter.grad, reference.grad)
    assert not (tmp_path / 'numerical_failures').exists()


@pytest.mark.parametrize('invalid', ['nan_target', 'zero_continuum', 'nan_weights'])
def test_forward_report_distinguishes_bad_data_and_weights(tmp_path, invalid):
    module = _module(None)
    attach_report_directory(module, tmp_path)
    batch = _physical_batch()
    if invalid == 'nan_target':
        batch['HMI']['stokes'][0, 0, 0] = float('nan')
    elif invalid == 'zero_continuum':
        batch['HMI']['stokes'].zero_()
    else:
        with torch.no_grad():
            module.parameter_model.out_layer.bias.fill_(float('nan'))
    with pytest.raises(RuntimeError, match='non-finite total loss'):
        module.training_step(batch, 0)
    report = json.loads((tmp_path / 'numerical_failures/rank2-step12-forward.json').read_text())
    assert report['stage'] == 'forward'
    assert report['tensors']['total_loss']['nonfinite'] == 1
    if invalid == 'nan_target':
        assert report['tensors']['batch.HMI.stokes']['nan'] == 1
        assert report['tensors']['profiles_by_instrument.HMI.prediction']['nonfinite'] == 0
    elif invalid == 'zero_continuum':
        assert report['tensors']['profiles_by_instrument.HMI.observed_continuum']['max'] == 0
    else:
        assert report['parameters']['parameter_model.out_layer.bias']['nonfinite'] > 0
        assert report['tensors']['decoded.vmac']['nonfinite'] > 0


def test_lightning_soap_stops_bad_backward_before_optimizer_state_update(tmp_path):
    module = _module(None)
    module.optimizer_config = {'type': 'soap'}
    original = {key: value.detach().clone() for key, value in module.parameter_model.state_dict().items()}
    hook = module.parameter_model.out_layer.weight.register_hook(lambda gradient: gradient * float('nan'))
    trainer = Trainer(accelerator='cpu', devices=1, max_steps=1, logger=False,
                      default_root_dir=tmp_path, gradient_clip_val=.5,
                      enable_checkpointing=False, enable_model_summary=False,
                      enable_progress_bar=False, num_sanity_val_steps=0)
    with pytest.raises(RuntimeError, match='optimizer update aborted'):
        trainer.fit(module, train_dataloaders=DataLoader([_physical_batch()], batch_size=None))
    hook.remove()
    for key, value in module.parameter_model.state_dict().items():
        torch.testing.assert_close(value, original[key])
    assert len(trainer.optimizers[0].state) == 0
    assert (tmp_path / 'numerical_failures/rank0-step0-backward.json').exists()
