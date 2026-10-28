"""DDP re-entry must reload metadata without repeating parent startup work."""
import json
import os
import subprocess
import sys

import pytest

from spinnme.distributed import is_primary_process
from test_fits_observations import cube


@pytest.mark.parametrize('ranks, expected', [({}, True), ({'RANK': '1'}, False),
    ({'LOCAL_RANK': '2', 'SLURM_PROCID': '0'}, False),
    ({'RANK': '0', 'LOCAL_RANK': '3', 'SLURM_PROCID': '0'}, False),
    ({'LOCAL_RANK': '0', 'NODE_RANK': '1'}, False),
    ({'LOCAL_RANK': '0', 'SLURM_PROCID': '4'}, False)])
def test_rank_identity_before_lightning_initializes(monkeypatch, ranks, expected):
    for key in ('RANK', 'LOCAL_RANK', 'NODE_RANK', 'GROUP_RANK', 'SLURM_PROCID', 'SLURM_NODEID', 'JSM_NAMESPACE_RANK'):
        monkeypatch.delenv(key, raising=False)
    for key, value in ranks.items():
        monkeypatch.setenv(key, value)
    assert is_primary_process() is expected


@pytest.mark.integration
@pytest.mark.parametrize('legacy_cache', [False, True])
def test_four_rank_lightning_reentry_only_reloads_followers(tmp_path, legacy_cache):
    from spinnme.configuration import resolve_config
    path, _ = cube(tmp_path)
    config = resolve_config('generic/stokes_cube', input_path=str(path), output=str(tmp_path/'run'))
    for dataset in [*config['data']['train_configs'], config['data']['valid_config']]:
        dataset['axis_order'] = ['stokes', 'wavelength', 'y', 'x']
    config['data'].update(batch_size=8, dataset_batch_size=4, num_workers=0, num_dataset_workers=0)
    config['model'].update(dim=4, n_layers=1, encoding_config={'type': 'identity'})
    config['training'].update(max_steps=1, accelerator='cpu', devices=4, plot_validation=False,
        limit_val_batches=0, num_sanity_val_steps=0, enable_progress_bar=False, enable_model_summary=False,
        optimizer_config={'type': 'adamw'})
    source = tmp_path/'config.json'
    source.write_text(json.dumps(config))
    script = tmp_path/'launch.py'
    script.write_text('''
import json
import os
from pathlib import Path
import sys
from spinnme.data import cache
from spinnme.distributed import is_primary_process
from spinnme.training import runner
import numpy as np

if __name__ == '__main__':
    config = json.loads(Path(sys.argv[1]).read_text())
    legacy_cache = sys.argv[2] == 'legacy'
    if legacy_cache and is_primary_process():
        data = cache.load_or_prepare_data_module(config['data'], config.get('work_path', Path(config['path'])/'cache'))
        manifest = Path(data.cache_manifest)
        metadata = cache._expand_metadata(json.loads(manifest.read_text()))
        rows = np.resize(data.train_datasets[0].profile_rows, 10000)
        metadata['training'][0]['attributes']['profile_rows'] = cache._encode(rows)
        cache._intern_metadata(metadata)
        manifest.write_text(json.dumps(metadata))
        cache._LOADED_MODULES.clear()
    if not is_primary_process():
        def unexpected(*args, **kwargs):
            raise AssertionError('Follower repeated preparation, locking, logger, hardware or provenance work')
        cache.FileLock = unexpected
        cache.SphericalDataModule.__init__ = unexpected
        runner._logger = unexpected
        runner._environment = unexpected
        runner._input_inventory = unexpected
        original_asarray = np.asarray
        def no_large_json_arrays(value, *args, **kwargs):
            if isinstance(value, list) and len(value) >= 8192:
                raise AssertionError('Follower materialized JSON selectors')
            return original_asarray(value, *args, **kwargs)
        np.asarray = no_large_json_arrays
    module, data, trainer = runner.run(config, reload=not legacy_cache)
    rank = os.environ.get('LOCAL_RANK', '0')
    Path(sys.argv[1]).with_name('rank-'+rank+'.json').write_text(json.dumps({
        'step': trainer.global_step, 'reused': data.cache_reused, 'signature': data.cache_signature}))
''')
    environment = os.environ.copy()
    for key in ('RANK', 'LOCAL_RANK', 'NODE_RANK', 'GROUP_RANK', 'WORLD_SIZE', 'SLURM_NTASKS', 'SLURM_NODEID'):
        environment.pop(key, None)
    environment.update(SLURM_PROCID='0', MASTER_ADDR='127.0.0.1',
                       OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                       GLOO_SOCKET_IFNAME='lo0' if sys.platform == 'darwin' else 'lo')
    completed = subprocess.run([sys.executable, str(script), str(source), 'legacy' if legacy_cache else 'new'], env=environment,
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 0, completed.stdout+'\n'+completed.stderr
    ranks = [json.loads((tmp_path/f'rank-{rank}.json').read_text()) for rank in range(4)]
    assert all(rank['step'] == 1 for rank in ranks)
    assert [rank['reused'] for rank in ranks] == [legacy_cache, True, True, True]
    assert len({rank['signature'] for rank in ranks}) == 1
    assert len(list((tmp_path/'run'/'cache').glob('data-*'))) == 1
