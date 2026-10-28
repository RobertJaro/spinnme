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
    manifest['integrity'] = {'nonexistent-file': {'size': 42, 'sha256': 'obsolete'}}
    manifest['metadata_sha256'] = 'obsolete'
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
    batch = next(iter(validation))
    expected = torch.cat([parallel.valid_dataset[i]['stokes'] for i in range(len(parallel.valid_dataset))])
    torch.testing.assert_close(batch['stokes'], expected[batch['pixel_index']])
    del loader, validation


def test_worker_changes_apply_to_existing_cache_without_rebuilding(tmp_path):
    from spinnme.data import cache
    path, _ = cube(tmp_path)
    config = configuration(path)
    original = load_or_prepare_data_module(config, tmp_path/'cache')
    stored = paths(original)
    signature = original.cache_signature
    config['num_workers'] = 4
    updated = load_or_prepare_data_module(config, tmp_path/'cache')
    assert updated is original
    assert updated.num_workers == 4
    assert updated.train_dataloader().num_workers == 4
    assert updated.val_dataloader().num_workers == 4
    cache._LOADED_MODULES.clear()
    restored = load_or_prepare_data_module(config, tmp_path/'cache')
    assert restored.num_workers == 4
    assert restored.cache_signature == signature
    assert paths(restored) == stored


@pytest.mark.parametrize('local_rank', ['1', '2', '3'])
def test_lightning_followers_ignore_reload_and_never_take_writer_lock(tmp_path, monkeypatch, local_rank):
    from filelock import FileLock
    from spinnme.data import cache
    path, _ = cube(tmp_path)
    config = configuration(path)
    prepared = load_or_prepare_data_module(config, tmp_path/'cache')
    cache._LOADED_MODULES.clear()
    monkeypatch.delenv('RANK', raising=False)
    monkeypatch.setenv('LOCAL_RANK', local_rank)
    monkeypatch.setenv('SLURM_PROCID', '0')  # inherited from the launching Slurm task
    def unexpected(*args, **kwargs):
        pytest.fail('A follower must only reopen the prepared module, without a writer lock or preparation.')
    with FileLock(str(tmp_path/'cache'/'preparation.lock')):
        monkeypatch.setattr(cache, 'FileLock', unexpected)
        monkeypatch.setattr(cache.SphericalDataModule, '__init__', unexpected)
        monkeypatch.setattr(fits, 'open', unexpected)
        restored = load_or_prepare_data_module(config, tmp_path/'cache', reload=True)
    assert restored.cache_reused
    assert paths(restored) == paths(prepared)
    assert len(list((tmp_path/'cache').glob('data-*'))) == 1


def test_shared_metadata_expansion_reuses_large_numeric_lists():
    from spinnme.data.cache import _expand_metadata
    class NumericList(list):
        def __iter__(self):
            pytest.fail('Reload must not recursively walk numeric profile selectors.')
    values = NumericList(range(100000))
    array = {'__kind__': 'array', 'dtype': 'int64', 'value': {'__shared__': 'rows'}}
    metadata = {'signature': 'test', 'shared': {'rows': values, 'array': array},
                'module': {'rows': {'__shared__': 'array'}},
                'training': [{'rows': {'__shared__': 'array'}}],
                'validation': {'rows': {'__shared__': 'array'}}}
    expanded = _expand_metadata(metadata)
    assert expanded['module']['rows']['value'] is values
    assert expanded['training'][0]['rows'] is expanded['module']['rows']
    assert expanded['validation']['rows'] is expanded['module']['rows']


def test_large_profile_selectors_are_lazy_and_pickle_by_path(tmp_path):
    import pickle
    from spinnme.data.cache import _dataset_metadata, _restore_dataset
    from spinnme.loader.spherical import SphericalDataset
    from spinnme.train.data_loader import TensorsDataset
    tensors = TensorsDataset({'stokes': np.zeros((100000, 4, 2), dtype=np.float32)},
                             str(tmp_path), 512, shuffle=False)
    dataset = SphericalDataset.__new__(SphericalDataset)
    dataset.__dict__.update(tensors.__dict__)
    dataset.ds_id = 'large'
    dataset.profile_rows = np.arange(100000, dtype=np.int64)
    dataset.profile_valid = np.ones(100000, dtype=bool)
    metadata = _dataset_metadata(dataset, tmp_path)
    assert len(json.dumps(metadata)) < 4096
    restored = _restore_dataset(metadata, tmp_path)
    assert isinstance(restored.profile_rows, np.memmap)
    assert isinstance(restored.profile_valid, np.memmap)
    serialized = pickle.dumps(restored)
    assert len(serialized) < 16384  # no selector bodies copied into spawned workers
    worker = pickle.loads(serialized)
    assert isinstance(worker.profile_rows, np.memmap)
    np.testing.assert_array_equal(worker.profile_rows[:8], np.arange(8))
    assert worker.profile_valid[:8].all()


def test_legacy_shared_matrix_rows_expand_without_leaving_references():
    from spinnme.data.cache import _encode, _decode, _intern_metadata, _expand_metadata
    matrix = np.arange(6000, dtype=np.float64).reshape(2, 3000)
    metadata = {'signature': 'test', 'module': {'matrix': _encode(matrix)},
                'training': [], 'validation': {'attributes': {}}}
    _intern_metadata(metadata)
    restored = _decode(_expand_metadata(metadata)['module'])
    np.testing.assert_array_equal(restored['matrix'], matrix)


def test_legacy_selectors_are_converted_once_before_follower_reload(tmp_path, monkeypatch):
    from spinnme.data import cache
    path, _ = cube(tmp_path)
    config = configuration(path)
    prepared = load_or_prepare_data_module(config, tmp_path/'cache')
    manifest_path = Path(prepared.cache_manifest)
    metadata = cache._expand_metadata(json.loads(manifest_path.read_text()))
    metadata['coordinate_convention_version'] = 2  # Legacy selector storage, current physical geometry.
    for key in ('prefetch_factor', 'persistent_workers', 'pin_memory'):
        metadata['module'].pop(key, None)
    rows = np.arange(100000, dtype=np.int64)
    metadata['training'][0]['attributes']['profile_rows'] = cache._encode(rows)
    metadata['training'][0]['attributes']['profile_valid'] = cache._encode(np.ones(len(rows), dtype=bool))
    # Reproduce the existing shared JSON format, rather than preparing a new cache.
    cache._intern_metadata(metadata)
    manifest_path.write_text(json.dumps(metadata))
    old_size = manifest_path.stat().st_size
    stores = list((tmp_path/'cache').rglob('*.zarr'))
    cache._LOADED_MODULES.clear()

    def unexpected(*args, **kwargs):
        pytest.fail('Reload must not prepare data, open FITS, or open Zarr stores.')

    import zarr
    monkeypatch.setattr(cache.SphericalDataModule, '__init__', unexpected)
    monkeypatch.setattr(fits, 'open', unexpected)
    monkeypatch.setattr(zarr, 'open_array', unexpected)
    restored = load_or_prepare_data_module(config, tmp_path/'cache')
    assert restored.cache_signature == prepared.cache_signature
    assert isinstance(restored.train_datasets[0].profile_rows, np.memmap)
    assert manifest_path.stat().st_size < old_size / 5
    assert list((tmp_path/'cache').rglob('*.zarr')) == stores
    assert paths(restored) == paths(prepared)

    cache._LOADED_MODULES.clear()
    monkeypatch.setenv('LOCAL_RANK', '1')
    monkeypatch.setenv('SLURM_PROCID', '0')
    monkeypatch.setattr(cache, '_write_metadata', unexpected)
    asarray = np.asarray
    def no_large_json_arrays(value, *args, **kwargs):
        if isinstance(value, list) and len(value) >= 65536:
            pytest.fail('Follower must not materialize large JSON selectors.')
        return asarray(value, *args, **kwargs)
    monkeypatch.setattr(np, 'asarray', no_large_json_arrays)
    follower = load_or_prepare_data_module(config, tmp_path/'cache')
    assert isinstance(follower.train_datasets[0].profile_rows, np.memmap)
    np.testing.assert_array_equal(follower.train_datasets[0].profile_rows[:8], rows[:8])


def test_restored_stores_open_only_when_reading_a_batch(tmp_path, monkeypatch):
    import pickle
    import zarr
    from spinnme.data import cache
    path, _ = cube(tmp_path)
    config = configuration(path)
    prepared = load_or_prepare_data_module(config, tmp_path/'cache')
    expected = prepared.train_datasets[0][0]['stokes'].clone()
    cache._LOADED_MODULES.clear()
    opened = []
    original = zarr.open_array
    def track(*args, **kwargs):
        opened.append(args[0])
        return original(*args, **kwargs)
    monkeypatch.setattr(zarr, 'open_array', track)
    restored = load_or_prepare_data_module(config, tmp_path/'cache')
    assert not opened
    torch.testing.assert_close(restored.train_datasets[0][0]['stokes'], expected)
    assert opened
    worker = pickle.loads(pickle.dumps(restored.train_datasets[0]))
    assert not worker.data._opened
    torch.testing.assert_close(worker[0]['stokes'], expected)


def test_legacy_coordinate_cache_requires_rebuild(tmp_path):
    from spinnme.data.cache import _LOADED_MODULES
    path, _ = cube(tmp_path)
    config = configuration(path)
    module = load_or_prepare_data_module(config, tmp_path/'cache')
    manifest = Path(module.cache_manifest)
    metadata = json.loads(manifest.read_text())
    assert metadata.pop('coordinate_convention_version') == 2
    manifest.write_text(json.dumps(metadata))
    _LOADED_MODULES.clear()
    with pytest.raises(ValueError, match='--reload'):
        load_or_prepare_data_module(config, tmp_path/'cache')
