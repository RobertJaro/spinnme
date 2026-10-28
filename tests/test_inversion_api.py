"""The public inversion API accepts equivalent complete dictionaries and YAML."""
import copy
import json
from pathlib import Path

import pytest
import torch

from spinnme import load, run
from spinnme.configuration import resolve_config, write_config
from spinnme.data.synthetic import generate_circular


@pytest.mark.integration
@pytest.mark.parametrize('input_kind', ['dict', 'yaml_string', 'yaml_path'])
def test_public_inversion_inputs_train_export_and_resume(tmp_path, input_kind):
    data = tmp_path / 'data'
    generate_circular(data, shape=(4, 4), n_time_steps=1)
    config = resolve_config('synthetic/circular', input_path=str(data / 'circular_000.fits'),
                            output=str(tmp_path / 'run'))
    config['data'].update(batch_size=16, dataset_batch_size=16)
    config['model'].update(dim=8, n_layers=2)
    config['training'].update(max_steps=1, accelerator='cpu', devices=1,
                              plot_validation=False, enable_model_summary=False, enable_progress_bar=False)
    original = copy.deepcopy(config)
    source = config
    if input_kind != 'dict':
        source = write_config(config, tmp_path / 'input.yaml')
        if input_kind == 'yaml_string':
            source = str(source)
    module, _, trainer = run(source, reload=True)
    assert trainer.global_step == 1
    assert config == original
    output = Path(config['path'])
    assert json.loads((output / 'run_manifest.json').read_text())['status'] == 'complete'
    with load(output / 'inversion.pme', device='cpu') as snapshot:
        for key, value in module.state_dict().items():
            torch.testing.assert_close(snapshot.module.state_dict()[key], value)
        assert snapshot.config['normalization'] == config['normalization']
        assert snapshot.config['stokes_loss'] == config['stokes_loss']
        result = snapshot.evaluate(snapshot.observations[0]['id'], parameters=['B', 'Blos', 'Vdop'])
        assert result.mask.any()
        result.write_npz(output / 'maps.npz')
    # A dictionary can resume a run started with either YAML interface.
    config['training']['max_steps'] = 2
    before_resume = copy.deepcopy(config)
    _, _, resumed = run(config, resume=True)
    assert resumed.global_step == 2
    assert config == before_resume
    if input_kind == 'dict':
        assert not (tmp_path / 'input.yaml').exists()
