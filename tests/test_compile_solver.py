"""Compiler capture/autograd regressions; SPINNME_TEST_INDUCTOR=1 tests native kernels."""
import io
import os

import pytest
import torch
from _training_helpers import _module, _physical_batch, physical_inputs
from pytorch_lightning import Trainer
from torch.utils.data import DataLoader

from spinnme.instrument import HMI_WAVELENGTH_CENTER
from spinnme.train.me_atmosphere import HMIMEAtmosphere, MEAtmosphere


@pytest.fixture(scope='module')
def compiled_solver():
    # The default exercises full graph capture and compiled autograd without
    # requiring a C++ toolchain on every developer machine. CI/GPU benchmarks
    # can exercise actual Inductor kernels with the environment flag.
    backend = 'inductor' if os.environ.get('SPINNME_TEST_INDUCTOR') == '1' else 'aot_eager'
    return torch.compile(MEAtmosphere._solve_profiles, backend=backend, fullgraph=True)


@pytest.mark.parametrize('dtype', [torch.float32, torch.float64])
def test_compiled_synthesis_values_gradients_and_changing_batch_size(monkeypatch, compiled_solver, dtype):
    import spinnme.train.me_atmosphere as atmosphere_module
    monkeypatch.setattr(atmosphere_module, 'compiled_profile_solver', lambda mode: compiled_solver)
    atmosphere = HMIMEAtmosphere(wavelength_center=HMI_WAVELENGTH_CENTER).to(dtype)
    for size in (4, 7):
        parameters, grid = physical_inputs(size, 9, 'cpu', dtype)
        atmosphere._compile_solver_mode = None
        expected = atmosphere(wavelength_grid=grid, **parameters)
        expected_grad = torch.autograd.grad(sum(x.square().sum() for x in expected), tuple(parameters.values()))
        atmosphere._compile_solver_mode = 'default'
        actual = atmosphere(wavelength_grid=grid, **parameters)
        actual_grad = torch.autograd.grad(sum(x.square().sum() for x in actual), tuple(parameters.values()))
        for a, b in zip((*actual, *actual_grad), (*expected, *expected_grad)):
            torch.testing.assert_close(a, b, rtol=3e-4 if dtype == torch.float32 else 1e-9,
                                       atol=1e-7 if dtype == torch.float32 else 1e-11)


def test_compile_config_state_dict_export_and_eager_evaluation(monkeypatch):
    import spinnme.train.me_atmosphere as atmosphere_module
    module = _module(None, compile_config={'me_solver': True, 'mode': 'default'})
    eager = _module(None)
    eager.load_state_dict(module.state_dict(), strict=True)
    assert module.state_dict().keys() == eager.state_dict().keys()
    atmosphere = module.forward_models['HMI']
    assert atmosphere._compile_solver_mode == 'default'
    # Serialization includes plain settings, never a compiled callable/cache.
    stream = io.BytesIO()
    torch.save(module.forward_models, stream)
    stream.seek(0)
    restored = torch.load(stream, weights_only=False)['HMI'].eval()
    def forbidden(mode):
        raise AssertionError('Evaluation must not invoke the compiler')
    monkeypatch.setattr(atmosphere_module, 'compiled_profile_solver', forbidden)
    params, grid = physical_inputs(4, 6, 'cpu')
    with torch.no_grad():
        expected = atmosphere(wavelength_grid=grid, **params)
    actual = restored(wavelength_grid=grid, **params)  # Eval stays eager even with gradients enabled.
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b)


def test_compiled_training_and_checkpoint_resume(tmp_path, monkeypatch, compiled_solver):
    import spinnme.train.me_atmosphere as atmosphere_module
    monkeypatch.setattr(atmosphere_module, 'compiled_profile_solver', lambda mode: compiled_solver)
    options = {'accelerator': 'cpu', 'devices': 1, 'logger': False, 'enable_checkpointing': False,
                   'enable_model_summary': False, 'enable_progress_bar': False, 'num_sanity_val_steps': 0}
    module = _module(None, compile_config={'me_solver': True})
    loader = DataLoader([_physical_batch() for _ in range(3)], batch_size=None)
    trainer = Trainer(max_steps=2, **options)
    trainer.fit(module, train_dataloaders=loader)
    assert torch.isfinite(trainer.callback_metrics['train.loss'])
    checkpoint = tmp_path / 'compiled.ckpt'
    trainer.save_checkpoint(checkpoint)
    # Compilation is an execution setting; existing optimizer/model states can
    # also resume eagerly without renamed keys or objective changes.
    resumed = _module(None)
    resumed_trainer = Trainer(max_steps=3, **options)
    resumed_trainer.fit(resumed, train_dataloaders=loader, ckpt_path=checkpoint)
    assert resumed_trainer.global_step == 3
    assert torch.isfinite(resumed_trainer.callback_metrics['train.loss'])


@pytest.mark.parametrize('config', [{'me_solver': 'yes'}, {'mode': 'invalid'}, {'unknown': True}])
def test_invalid_compile_config(config):
    with pytest.raises(ValueError):
        _module(None, compile_config=config)
