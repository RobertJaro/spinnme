"""Shared observation storage, bounded reads and lightweight cache reloads."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from spinnme.data import cache
from spinnme.train.data_loader import TensorsDataset
from test_preparation_cache import configuration, paths
from test_fits_observations import cube
from test_hinode_slits import scan, config


def test_wide_spectra_have_byte_bounded_chunks(tmp_path):
    spectra = np.ones((64, 4, 8192), dtype=np.float32)
    ds = TensorsDataset({'stokes': spectra}, str(tmp_path), 4, shuffle=False)
    stored = ds.data['stokes']
    assert np.prod(stored.chunks) * stored.dtype.itemsize <= 1024 ** 2
    np.testing.assert_array_equal(ds[0]['stokes'], spectra[:4])


def test_hinode_lines_and_validation_share_full_spectra_and_timing(tmp_path):
    folder = scan(tmp_path)
    c = config(folder, tmp_path)['data']
    c.update(num_workers=0)
    # Exercise independent line masks/normalization and global shuffled training.
    for ds in c['train_configs']:
        ds.pop('shuffle', None)
        ds.pop('filter_nans', None)
    dm = cache.load_or_prepare_data_module(c, tmp_path/'cache')
    one, two = dm.train_datasets
    valid = dm.valid_dataset
    assert one.batches_file_paths == two.batches_file_paths == valid.batches_file_paths
    assert one.data['stokes'].shape == (12, 4, 112)
    assert len(list((tmp_path/'cache').rglob('*stokes.zarr'))) == 1
    for ds in (one, two, valid):
        batch = ds[0]
        pixels = ds.data['pixel_index'][ds.profile_rows, 0]
        np.testing.assert_array_equal(batch['stokes'][:, 0, 0], 101 + pixels % 3)
        np.testing.assert_allclose(batch['coords'][:, 3], (pixels % 3)*4/ds.seconds_per_dt)
    # Validation retains native raster order even though storage is shuffled.
    np.testing.assert_array_equal(valid[0]['stokes'][:, 0, 0], np.tile([101, 102, 103], 4))
    validation = next(iter(dm.val_dataloader()))
    torch.testing.assert_close(validation['stokes'][torch.argsort(validation['pixel_index'])], valid[0]['stokes'])
    for view in dm.val_dataloader().dataset.datasets:
        np.testing.assert_array_equal(view.profile_rows, np.arange(12))
    expected = {ds.ds_id: ds[0] for ds in (one, two, valid)}
    for path in folder.glob('*.fits'):
        path.unlink()
    cache._LOADED_MODULES.clear()
    restored = cache.load_or_prepare_data_module(c, tmp_path/'cache')
    for ds in (*restored.train_datasets, restored.valid_dataset):
        torch.testing.assert_close(ds[0]['stokes'], expected[ds.ds_id]['stokes'])
        torch.testing.assert_close(ds[0]['coords'], expected[ds.ds_id]['coords'])


def test_runtime_options_apply_on_hot_and_cold_reload(tmp_path):
    path, _ = cube(tmp_path)
    c = configuration(path)
    dm = cache.load_or_prepare_data_module(c, tmp_path/'cache')
    signature, stored = dm.cache_signature, paths(dm)
    c.update(batch_size=12, dataset_batch_size=3, num_workers=4,
             prefetch_factor=3, persistent_workers=False, pin_memory=True)
    path.unlink()
    for cold in (False, True):
        if cold:
            cache._LOADED_MODULES.clear()
        updated = cache.load_or_prepare_data_module(c, tmp_path/'cache')
        assert updated.cache_signature == signature
        assert paths(updated) == stored
        loader = updated.train_dataloader()
        assert (updated.batch_size, updated.dataset_batch_size) == (12, 3)
        assert loader.prefetch_factor == 3
        assert loader.persistent_workers is False
        assert loader.pin_memory is True
        assert updated.valid_dataset.batch_size == 8  # updated when its loader is requested
        assert updated.val_dataloader().dataset.batch_size == 12
    bad = {**c, 'dataset_batch_size': 5}
    with pytest.raises(ValueError, match='divisible'):
        cache.load_or_prepare_data_module(bad, tmp_path/'cache')
    assert updated.dataset_batch_size == 3


def test_multiple_validation_views_reload_without_duplicate_spectra(tmp_path):
    folder = scan(tmp_path)
    c = config(folder, tmp_path)['data']
    c['num_workers'] = 0
    single = {**c, 'valid_config': c['valid_config'][0]}
    cache.load_or_prepare_data_module(single, tmp_path/'cache')
    with pytest.raises(ValueError, match='--reload'):
        cache.load_or_prepare_data_module(c, tmp_path/'cache')
    prepared = cache.load_or_prepare_data_module(c, tmp_path/'cache', reload=True)
    assert len(prepared.valid_datasets) == 2
    assert prepared.valid_dataset is prepared.valid_datasets[0]
    shared = prepared.train_datasets[0].batches_file_paths
    assert all(dataset.batches_file_paths == shared for dataset in prepared.valid_datasets)
    for path in folder.glob('*.fits'):
        path.unlink()
    cache._LOADED_MODULES.clear()
    restored = cache.load_or_prepare_data_module(c, tmp_path/'cache')
    assert restored.cache_reused
    assert len(restored.valid_datasets) == 2
    assert len([row for row in restored.observations if row['role'] == 'validation']) == 2
    for expected, actual in zip(prepared.valid_datasets, restored.valid_datasets):
        torch.testing.assert_close(actual[0]['stokes'], expected[0]['stokes'])
        torch.testing.assert_close(actual[0]['coords'], expected[0]['coords'])
    batches = list(restored.val_dataloader())
    assert {batch['instrument_id'] for batch in batches} == {'HINODE_6301', 'HINODE_6302'}
    assert {int(batch['validation_index'][0]) for batch in batches} == {0, 1}


def test_initial_preparation_and_reload_do_not_scan_or_hash_cache(tmp_path, monkeypatch, capsys):
    path, _ = cube(tmp_path)
    config = configuration(path)
    def unexpected(*args, **kwargs):
        pytest.fail('Ordinary preparation/reload must not scan or checksum cache files.')
    with monkeypatch.context() as patch:
        patch.setattr(Path, 'rglob', unexpected)
        module = cache.load_or_prepare_data_module(config, tmp_path/'cache')
        cache._LOADED_MODULES.clear()
        restored = cache.load_or_prepare_data_module(config, tmp_path/'cache')
        torch.testing.assert_close(module.train_datasets[0][0]['stokes'], restored.train_datasets[0][0]['stokes'])
    metadata = json.loads(Path(module.cache_manifest).read_text())
    assert 'integrity' not in metadata
    assert 'metadata_sha256' not in metadata
    assert 'Indexing cache integrity' not in capsys.readouterr().err
