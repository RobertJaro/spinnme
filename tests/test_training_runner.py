"""CPU orchestration smoke tests without data downloads or external logging."""
import copy
import json
from pathlib import Path

import pytest
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from _training_helpers import _module, _physical_batch
from spinnme.training import runner


class PreparedData(LightningDataModule):
    def __init__(self, work_directory):
        super().__init__()
        reference = _module()
        self.image_shape = (4, 4)
        self.wavelength_config = reference.wavelength_config
        self.Rs_per_ds = 1
        self.seconds_per_dt = 36000
        self.gauss_per_dB = 1000

    def train_dataloader(self):
        return DataLoader([_physical_batch()], batch_size=None)


def configuration(directory):
    module = _module()
    return {'path': str(directory), 'data': {}, 'model': module.snapshot_init['model_config'],
            'instrument': module.instrument_config,
            'normalization': {'stokes_scales': [1, .01, .01, .01]},
            'stokes_loss': {'continuum_normalization': True},
            'logging': {'mode': 'disabled'}, 'training': {
                'epochs': -1, 'max_steps': 1, 'accelerator': 'cpu', 'devices': 1,
                'limit_val_batches': 0, 'enable_progress_bar': False, 'enable_model_summary': False}}


def test_one_step_cpu_run_saves_final_snapshot_without_network(tmp_path, monkeypatch):
    import spinnme.snapshots as snapshots
    monkeypatch.setattr(runner, 'load_or_prepare_data_module', lambda config, work_directory, reload=False: PreparedData(work_directory))
    saved = []
    def save(module, data, config, path):
        saved.append(Path(path))
        Path(path).write_bytes(b'snapshot-test-hook')
    monkeypatch.setattr(snapshots, 'save_snapshot', save)
    config = configuration(tmp_path)
    unchanged = copy.deepcopy(config)
    module, data, trainer = runner.run(config)
    assert trainer.global_step == 1
    assert saved[-1] == tmp_path / 'inversion.pme'
    assert config == unchanged
    assert trainer.logger is None
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert manifest['status'] == 'complete' and manifest['steps'] == 1
    assert (tmp_path / 'resolved.yaml').is_file()
    with pytest.raises(FileExistsError):
        runner.run(config)


def test_unbounded_training_is_rejected_before_data_preparation(tmp_path):
    config = configuration(tmp_path)
    config['training'].pop('max_steps')
    with pytest.raises(ValueError, match='unbounded'):
        runner.run(config)


def test_resume_is_explicit_and_requires_checkpoint(tmp_path):
    with pytest.raises(FileNotFoundError, match='Cannot resume'):
        runner.run(configuration(tmp_path), resume=True)


def test_resume_comparison_preserves_old_config_on_scientific_change(tmp_path):
    from spinnme.configuration import write_config
    original = configuration(tmp_path)
    path = write_config(original, tmp_path / 'resolved.yaml')
    before = path.read_bytes()
    runtime = copy.deepcopy(original)
    runtime['training']['max_steps'] = 2
    runner._resume_configuration(runtime, path)
    changed = copy.deepcopy(original)
    changed['stokes_loss']['continuum_normalization'] = False
    with pytest.raises(ValueError, match='scientific'):
        runner._resume_configuration(changed, path)
    assert path.read_bytes() == before


def test_preparation_failure_records_status(tmp_path, monkeypatch):
    def fail(**kwargs):
        raise ValueError('invalid observation geometry')
    monkeypatch.setattr(runner, 'load_or_prepare_data_module', lambda *args, **kwargs: fail(**kwargs))
    with pytest.raises(ValueError, match='geometry'):
        runner.run(configuration(tmp_path))
    manifest = json.loads((tmp_path / 'run_manifest.json').read_text())
    assert manifest['status'] == 'failed'
    assert manifest['error']['type'] == 'ValueError'


def test_input_inventory_records_unique_paths_without_accessing_files(tmp_path):
    from types import SimpleNamespace
    file = tmp_path / 'input.fits'
    data = SimpleNamespace(observations=[{'source_paths': [str(file)]}, {'source_paths': [str(file)]}])
    inventory = runner._input_inventory(data)
    assert inventory == [{'path': str(file)}]


def test_final_checkpoint_matches_snapshot_between_validation_boundaries(tmp_path):
    import torch
    from spinnme.data.circular import create_circular_dataset
    from spinnme.configuration import resolve_config
    from spinnme.snapshots import Snapshot
    config_path = create_circular_dataset(tmp_path/'data', size=8, frames=1)
    config = resolve_config(config_path, output=str(tmp_path/'run'), overrides=[
        'model.dim=8', 'model.n_layers=2', 'data.batch_size=32', 'data.dataset_batch_size=16',
        'training.max_steps=3', 'training.check_val_every_n_epoch=null', 'training.val_check_interval=2',
        'training.accelerator=cpu', 'training.devices=1',
        'training.enable_progress_bar=false', 'training.enable_model_summary=false'])
    module, _, trainer = runner.run(config)
    checkpoint = torch.load(tmp_path/'run/last.ckpt', map_location='cpu', weights_only=False)
    assert trainer.global_step == checkpoint['global_step'] == 3
    with Snapshot(tmp_path/'run/inversion.pme', device='cpu') as snapshot:
        for key, value in module.state_dict().items():
            torch.testing.assert_close(checkpoint['state_dict'][key], value)
            torch.testing.assert_close(snapshot.module.state_dict()[key], value)
    config['training']['max_steps'] = 4
    _, _, resumed = runner.run(config, resume=True)
    assert resumed.global_step == 4
    assert torch.load(tmp_path/'run/last.ckpt', map_location='cpu', weights_only=False)['global_step'] == 4
