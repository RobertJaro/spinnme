import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from astropy.io import fits
from test_fits_observations import cube

from spinnme.data.cache import load_or_prepare_data_module


def configuration(path):
    dataset={'type': 'fits','data_path': str(path),'instrument_id': 'A',
                 'axis_order': ['stokes','wavelength','y','x'],'line_id': 'fe-i-6173'}
    return {'train_configs': [dataset],'valid_config': {**dataset,'sample_idx':0},
                'batch_size': 8,'dataset_batch_size': 4,'num_workers': 0,'num_dataset_workers': 0}


def paths(module):
    return [str(path) for dataset in [*module.train_datasets,module.valid_dataset]
            for path in dataset.batches_file_paths.values()]


def test_prep_then_run_reuses_same_arrays_and_preserves_rng(tmp_path, capsys):
    path,_=cube(tmp_path)
    config=configuration(path)
    np.random.seed(31)
    random.seed(31)
    torch.manual_seed(31)
    np_state,py_state,torch_state=np.random.get_state(),random.getstate(),torch.get_rng_state()
    expected=(np.random.random(),random.random(),torch.rand(1))
    np.random.set_state(np_state)
    random.setstate(py_state)
    torch.set_rng_state(torch_state)
    first=load_or_prepare_data_module(config,tmp_path/'cache')
    assert not first.cache_reused
    actual=(np.random.random(),random.random(),torch.rand(1))
    assert actual[0:2] == expected[0:2]
    torch.testing.assert_close(actual[2],expected[2])
    stored=list((tmp_path/'cache').rglob('*.zarr'))
    second=load_or_prepare_data_module(config,tmp_path/'cache')
    assert second is first
    assert second.cache_reused
    assert paths(first) == paths(second)
    assert list((tmp_path/'cache').rglob('*.zarr')) == stored
    torch.testing.assert_close(first.train_datasets[0][0]['coords'],second.train_datasets[0][0]['coords'])
    torch.testing.assert_close(first.train_datasets[0][0]['stokes'],second.train_datasets[0][0]['stokes'])
    assert second.get_observations() == first.get_observations()
    output = capsys.readouterr().err
    assert 'Loading training data' in output
    assert 'Loading validation data' in output
    assert 'Writing stokes cache' not in output
    assert 'Indexing prepared observations' not in output


def test_rerun_skips_restore_and_explicit_reload_rebuilds(tmp_path, monkeypatch):
    from spinnme.data import cache

    path, _ = cube(tmp_path)
    config = configuration(path)
    first = load_or_prepare_data_module(config, tmp_path/'cache')

    def unexpected_restore(*args, **kwargs):
        raise AssertionError('Already loaded datamodule should be reused')

    monkeypatch.setattr(cache, '_restore', unexpected_restore)
    assert load_or_prepare_data_module(config, tmp_path/'cache') is first
    reloaded = load_or_prepare_data_module(config, tmp_path/'cache', reload=True)
    assert reloaded is not first
    assert not reloaded.cache_reused
    assert paths(reloaded) != paths(first)
    assert load_or_prepare_data_module(config, tmp_path/'cache') is reloaded


@pytest.mark.parametrize('rank', ['1', '2'])
def test_other_ranks_restore_without_rebuilding_even_with_reload(tmp_path, monkeypatch, rank):
    from spinnme.data import cache

    path, _ = cube(tmp_path)
    config = configuration(path)
    monkeypatch.setenv('RANK', '0')
    prepared = load_or_prepare_data_module(config, tmp_path/'cache', reload=True)
    cache._LOADED_MODULES.clear()
    monkeypatch.setenv('RANK', rank)

    def unexpected_prepare(*args, **kwargs):
        raise AssertionError('Only rank zero may build a datamodule')

    monkeypatch.setattr(cache.SphericalDataModule, '__init__', unexpected_prepare)
    restored = load_or_prepare_data_module(config, tmp_path/'cache', reload=True)
    assert restored is not prepared
    assert restored.cache_reused
    assert paths(restored) == paths(prepared)
    assert len(list((tmp_path/'cache').glob('data-*'))) == 1


def test_follower_waits_instead_of_preparing_missing_cache(tmp_path, monkeypatch):
    from spinnme.data import cache

    path, _ = cube(tmp_path)
    config = configuration(path)
    monkeypatch.setenv('RANK', '1')

    def rank_zero_prepares(_):
        cache._load_or_prepare_data_module(config, tmp_path/'cache')

    monkeypatch.setattr(cache.time, 'sleep', rank_zero_prepares)
    restored = load_or_prepare_data_module(config, tmp_path/'cache', reload=True)
    assert restored.cache_reused
    assert len(list((tmp_path/'cache').glob('data-*'))) == 1


@pytest.mark.parametrize('restore', [False, True])
def test_cache_reuse_ignores_source_and_configuration_changes(tmp_path, restore):
    from spinnme.data import cache

    path, _ = cube(tmp_path)
    config = configuration(path)
    first = load_or_prepare_data_module(config, tmp_path/'cache')
    expected = first.train_datasets[0][0]['stokes'].clone()
    path.unlink()
    config['train_configs'][0]['stokes_normalization'] = 2.
    config['preparation_seed'] = 28
    if restore:
        cache._LOADED_MODULES.clear()
    second = load_or_prepare_data_module(config, tmp_path/'cache')
    assert second.cache_reused
    assert second.cache_signature == first.cache_signature
    assert paths(second) == paths(first)
    torch.testing.assert_close(second.train_datasets[0][0]['stokes'], expected)


def test_restore_ignores_legacy_file_inventory_and_array_checks(tmp_path):
    from spinnme.data import cache

    path, _ = cube(tmp_path)
    config = configuration(path)
    first = load_or_prepare_data_module(config, tmp_path/'cache')
    manifest_path = Path(first.cache_manifest)
    manifest = json.loads(manifest_path.read_text())
    for dataset in [*manifest['training'], manifest['validation']]:
        for description in dataset['arrays'].values():
            description.update(shape=[-1], dtype='obsolete',
                               files=[{'path': 'missing-chunk', 'bytes': 123}])
    manifest['fingerprint'] = {'obsolete': True}
    manifest_path.write_text(json.dumps(manifest))
    cache._LOADED_MODULES.clear()
    restored = load_or_prepare_data_module(config, tmp_path/'cache')
    assert restored.cache_reused
    assert paths(restored) == paths(first)
    torch.testing.assert_close(restored.train_datasets[0][0]['stokes'], first.train_datasets[0][0]['stokes'])


def test_preparation_seed_changes_cache_and_is_deterministic(tmp_path):
    path,_=cube(tmp_path)
    config=configuration(path)
    config['preparation_seed']=27
    first=load_or_prepare_data_module(config,tmp_path/'one')
    second=load_or_prepare_data_module(config,tmp_path/'two')
    torch.testing.assert_close(first.train_datasets[0][0]['stokes'],second.train_datasets[0][0]['stokes'])
    config['preparation_seed']=28
    third=load_or_prepare_data_module(config,tmp_path/'one',reload=True)
    assert first.cache_signature != third.cache_signature


def test_generic_solar_projection_survives_safe_cache_restore(tmp_path):
    from spinnme.data import cache
    path,_=cube(tmp_path)
    with fits.open(path,mode='update') as hs:
        hs[0].header['CTYPE1']='HPLN-SIN'
        hs[0].header['CTYPE2']='HPLT-SIN'
    config=configuration(path)
    first=load_or_prepare_data_module(config,tmp_path/'cache')
    cache._LOADED_MODULES.clear()
    restored=load_or_prepare_data_module(config,tmp_path/'cache')
    assert restored.cache_reused
    torch.testing.assert_close(first.train_datasets[0][0]['coords'],restored.train_datasets[0][0]['coords'])
    torch.testing.assert_close(first.train_datasets[0][0]['mu'],restored.train_datasets[0][0]['mu'])


def test_spawn_workers_prepare_and_read_same_profiles_as_serial(tmp_path):
    import shutil
    path, _ = cube(tmp_path)
    shutil.copy2(path, tmp_path/'second.fits')
    config = configuration(str(tmp_path/'*.fits'))
    serial = load_or_prepare_data_module(config, tmp_path/'serial')
    config.update(num_dataset_workers=2, num_workers=1)
    parallel = load_or_prepare_data_module(config, tmp_path/'parallel')
    assert len(parallel.train_datasets) == 2
    for expected, actual in zip(serial.train_datasets, parallel.train_datasets):
        torch.testing.assert_close(expected[0]['stokes'], actual[0]['stokes'])
        torch.testing.assert_close(expected[0]['coords'], actual[0]['coords'])
    # Exercise actual spawned training/validation workers, including Zarr handles.
    loader = parallel.train_dataloader()
    assert next(iter(loader))
    validation = parallel.val_dataloader()
    torch.testing.assert_close(next(iter(validation))['stokes'], parallel.valid_dataset[0]['stokes'])
    del loader, validation
