"""Reusable offline preparation caches with JSON metadata and lazy Zarr arrays.

Only a fixed set of package dataset classes is restored. The cache contains no
pickled Python objects and is not a portable inference snapshot. A preparation
lock and atomic metadata replacement make consecutive prep/run calls share the
same arrays. Inputs are never downloaded from this module.
"""
from __future__ import annotations

import json
import hashlib
import random
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from weakref import WeakValueDictionary

import numpy as np
import torch
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from filelock import FileLock
from pytorch_lightning import LightningDataModule

from spinnme.loader.spherical import (
    FITSSphericalDataset,
    HinodeSphericalDataset,
    HMISphericalDataset,
    PHIFDTSphericalDataset,
    PHIHRTSphericalDataset,
    SphericalDataModule,
    SphericalDataset,
    TestSphericalDataset,
)
from spinnme.distributed import is_primary_process
from spinnme.progress import progress
from spinnme.conventions import COORDINATE_CONVENTION_VERSION
from spinnme.train.data_loader import LazyZarrArrays

_LOADED_MODULES = WeakValueDictionary()
DATASET_CLASSES = {cls.__name__:cls for cls in (SphericalDataset, FITSSphericalDataset,
    HMISphericalDataset, PHIHRTSphericalDataset, PHIFDTSphericalDataset,
    HinodeSphericalDataset, TestSphericalDataset)}
MODULE_FIELDS = ('spectral_response_files','spectral_response_by_acquisition',
    'batch_size','dataset_batch_size','balance_train_configs','num_workers','num_dataset_workers',
    'train_config_indices','wavelength_config','stokes_normalization_by_instrument',
    'ref_time','seconds_per_dt','Rs_per_ds','gauss_per_dB','times','image_shape',
    'value_range','data_range','preparation_seed','prefetch_factor','persistent_workers','pin_memory')


def _encode(value):
    if isinstance(value, WCS):
        return {'__kind__':'wcs','header':value.to_header(relax=True).tostring(sep='\n',endcard=False,padding=False)}
    if isinstance(value, u.Quantity):
        return {'__kind__':'quantity','value':np.asarray(value.value).tolist(),'unit':str(value.unit)}
    if isinstance(value, datetime):
        return {'__kind__':'datetime','value':value.isoformat()}
    if isinstance(value, torch.Tensor):
        return {'__kind__':'tensor','value':value.detach().cpu().numpy().tolist(),'dtype':str(value.dtype).split('.')[-1]}
    if isinstance(value, np.ndarray):
        return {'__kind__':'array','value':value.tolist(),'dtype':str(value.dtype)}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return {'__kind__':'tuple','value':[_encode(v) for v in value]}
    if isinstance(value, list):
        return [_encode(v) for v in value]
    if isinstance(value, dict):
        return {str(k):_encode(v) for k,v in value.items()}
    if value is None or isinstance(value,(str,int,float,bool)):
        return value
    raise TypeError(f'Unsupported cache metadata type: {type(value).__name__}')


def _decode(value, root=None, memo=None):
    memo = {} if memo is None else memo
    if not isinstance(value, (list, dict)):
        return value
    identity = id(value)
    if identity in memo:
        return memo[identity]
    def decode(child):
        return _decode(child, root, memo)
    if isinstance(value, list):
        result = [decode(v) for v in value]
    else:
        kind = value.get('__kind__')
        if kind == 'wcs':
            result = WCS(fits.Header.fromstring(value['header'], sep='\n'))
        elif kind == 'quantity':
            result = np.asarray(value['value'])*u.Unit(value['unit'])
        elif kind == 'datetime':
            result = datetime.fromisoformat(value['value'])
        elif kind == 'tuple':
            result = tuple(decode(v) for v in value['value'])
        elif kind == 'array':
            result = np.asarray(value['value'], dtype=value['dtype'])
        elif kind == 'npy':
            result = np.load(_inside(root/value['path'], root), mmap_mode='r', allow_pickle=False)
        elif kind == 'tensor':
            dtype = {'float32': torch.float32, 'float64': torch.float64, 'int32': torch.int32,
                     'int64': torch.int64, 'bool': torch.bool}.get(value['dtype'])
            if dtype is None:
                raise ValueError('Unsupported tensor dtype in prepared cache.')
            result = torch.tensor(value['value'], dtype=dtype)
        elif kind is not None:
            raise ValueError(f'Unknown cache metadata type {kind!r}.')
        else:
            result = {key: decode(v) for key, v in value.items()}
    memo[identity] = result
    return result


def _inside(path, root):
    path=Path(path).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Prepared cache references a file outside its cache directory.')
    return path


def _dataset_metadata(dataset, root):
    name=type(dataset).__name__
    if name not in DATASET_CLASSES:
        raise ValueError(f'Cannot cache unsupported dataset type {name!r}.')
    attrs={key:value for key,value in vars(dataset).items() if key != 'data'}
    # Profile selectors are arrays, not JSON metadata. Reopen them as read-only
    # memory maps without reading their contents on each GPU rank.
    directory = Path(next(iter(dataset.batches_file_paths.values()))).parent
    for key, value in list(attrs.items()):
        if isinstance(value, np.ndarray) and value.nbytes >= 65536:
            if isinstance(value, np.memmap):
                path = Path(value.filename)
            else:
                path = directory/f'{dataset.ds_id}_{key}.npy'
                np.save(path, value, allow_pickle=False)
                setattr(dataset, key, np.load(path, mmap_mode='r', allow_pickle=False))
            attrs[key] = {'__kind__': 'npy', 'path': str(_inside(path, root).relative_to(root))}
    arrays={}
    for key,path in dataset.batches_file_paths.items():
        directory=_inside(path,root)
        arrays[key]={'path':str(directory.relative_to(root))}
    return {'class':name,'attributes':_encode(attrs),'arrays':arrays}


def _restore_dataset(metadata, root, memo=None):
    dataset_class=DATASET_CLASSES.get(metadata['class'])
    if dataset_class is None:
        raise ValueError('Unsupported dataset class in cache.')
    attributes = _decode(metadata['attributes'], root, memo)
    if (dataset_class is HinodeSphericalDataset and 'slit_pixel_world' in attributes
            and 'slit_geometry' not in attributes):
        raise ValueError('This Hinode cache lacks per-slit physical frames; rebuild it once with --reload.')
    paths={}
    for key,description in metadata['arrays'].items():
        directory=_inside(root/description['path'],root)
        paths[key]=str(directory)
    dataset=dataset_class.__new__(dataset_class)
    dataset.__dict__.update(attributes)
    dataset.batches_file_paths=paths
    dataset.data=LazyZarrArrays(paths)
    return dataset


def _restore(metadata, root):
    module=SphericalDataModule.__new__(SphericalDataModule)
    LightningDataModule.__init__(module)
    memo = {}
    attrs=_decode(metadata['module'], root, memo)
    module.__dict__.update(attrs)
    module.train_datasets=[_restore_dataset(v,root,memo) for v in
                           progress(metadata['training'], desc='Loading training data (cache)')]
    validations = [metadata['validation'], *metadata.get('additional_validation', [])]
    module.valid_datasets = [_restore_dataset(value, root, memo) for value in
                             progress(validations, desc='Loading validation data (cache)')]
    module.valid_dataset = module.valid_datasets[0]
    module._observations = None
    return module


class _PreparationPending(RuntimeError):
    pass


def _apply_runtime_settings(module, config):
    defaults = {'num_workers': 4, 'prefetch_factor': 2,
                'persistent_workers': True, 'pin_memory': False,
                'batch_size': 65536, 'dataset_batch_size': 4096,
                'balance_train_configs': False}
    values = {key: config.get(key, default) for key, default in defaults.items()}
    if values['num_workers'] is None:
        values['num_workers'] = 4
    for key in ('num_workers', 'batch_size', 'dataset_batch_size', 'prefetch_factor'):
        value = values[key]
        if not isinstance(value, int) or isinstance(value, bool) or value < (0 if key == 'num_workers' else 1):
            raise ValueError(f'{key} must be a {"nonnegative" if key == "num_workers" else "positive"} integer.')
    for key in ('persistent_workers', 'pin_memory', 'balance_train_configs'):
        if not isinstance(values[key], bool):
            raise ValueError(f'{key} must be a boolean.')
    previous = {key: getattr(module, key, None) for key in values}
    module.__dict__.update(values)
    try:
        module._validate_batch_sizes()
    except ValueError:
        module.__dict__.update(previous)
        raise


def _intern_metadata(metadata):
    """Store identical geometry/timing and view metadata once in the manifest."""
    shared = {}
    def intern(value):
        if isinstance(value, list):
            result = [intern(v) for v in value]
        elif isinstance(value, dict):
            result = {k: intern(v) for k, v in value.items()}
        else:
            return value
        serialized = json.dumps(result, sort_keys=True, allow_nan=False)
        if len(serialized) < 1024:
            return result
        key = hashlib.sha256(serialized.encode()).hexdigest()
        shared.setdefault(key, result)
        return {'__shared__': key}
    metadata['module'] = {key: intern(value) for key, value in metadata['module'].items()}
    for dataset in [*metadata['training'], metadata['validation'], *metadata.get('additional_validation', [])]:
        dataset['attributes'] = {key: intern(value) for key, value in dataset['attributes'].items()}
    metadata['shared'] = shared


def _expand_metadata(metadata):
    shared, memo = metadata.get('shared', {}), {}
    def expand(value, numeric=False):
        if isinstance(value, list):
            # Encoded array bodies are complete numeric values, never refs.
            # Reuse them without walking/copying millions of scalar entries.
            if numeric and (not value or not isinstance(value[0], (list, dict))):
                return value
            return [expand(v, numeric=numeric) for v in value]
        if isinstance(value, dict):
            if set(value) == {'__shared__'}:
                key = value['__shared__']
                if key not in memo:
                    memo[key] = expand(shared[key], numeric=numeric)
                return memo[key]
            array = value.get('__kind__') in ('array', 'tensor', 'quantity')
            return {k: expand(v, numeric=array and k == 'value') for k, v in value.items()}
        return value
    return {key: expand(metadata[key]) for key in
            ('signature', 'module', 'training', 'validation', 'additional_validation') if key in metadata}


def _load_cached_module(data_config, root, manifest):
    module = _LOADED_MODULES.get(root)
    cold = module is None
    if module is None:
        metadata = json.loads(manifest.read_text())
        if metadata.get('coordinate_convention_version', 1) != COORDINATE_CONVENTION_VERSION:
            raise ValueError('Prepared data use legacy coordinate conventions. Rebuild once with --reload; retrain affected inversions.')
        module = _restore(_expand_metadata(metadata), root)
        module.cache_signature = metadata['signature']
        module.cache_manifest = str(manifest)
        del metadata
    validation = data_config.get('valid_config')
    requested = len(validation) if isinstance(validation, list) else 1
    if requested != len(getattr(module, 'valid_datasets', [module.valid_dataset])):
        raise ValueError('Validation dataset selection changed; rebuild prepared data once with --reload.')
    _apply_runtime_settings(module, data_config)
    # Older manifests embedded selectors as JSON lists. Convert them once
    # on the parent before Lightning relaunches its follower processes.
    if cold and is_primary_process() and any(
            isinstance(value, np.ndarray) and not isinstance(value, np.memmap) and value.nbytes >= 65536
            for dataset in [*module.train_datasets, *module.valid_datasets] for value in vars(dataset).values()):
        print('Moving cached selectors out of JSON for lightweight reload (once).', flush=True)
        _write_metadata(module, root, manifest, module.cache_signature)
    _LOADED_MODULES[root] = module
    module.cache_reused = True
    return module


def _write_metadata(module, root, manifest, signature):
    metadata={'signature':signature, 'coordinate_convention_version': COORDINATE_CONVENTION_VERSION,
              'module':_encode({key:getattr(module,key) for key in MODULE_FIELDS}),
              'training':[_dataset_metadata(ds,root) for ds in module.train_datasets],
              'validation':_dataset_metadata(module.valid_dataset,root),
              'additional_validation': [_dataset_metadata(ds, root) for ds in module.valid_datasets[1:]]}
    _intern_metadata(metadata)
    with tempfile.NamedTemporaryFile(mode='w',dir=root,suffix='.json.tmp',delete=False) as stream:
        json.dump(metadata,stream,allow_nan=False)
        temporary=Path(stream.name)
    temporary.replace(manifest)


def load_or_prepare_data_module(data_config, work_directory, reload=False):
    """Prepare on rank zero; other ranks only load its prepared cache.

    Followers wait up to an hour for rank zero when no cache exists.
    ``reload`` forces preparation only on rank zero, including when Lightning
    relaunches the script with the original command-line arguments.
    """
    primary = is_primary_process()
    deadline = time.monotonic() + 3600
    while True:
        try:
            return _load_or_prepare_data_module(data_config, work_directory,
                                                reload=reload and primary, prepare=primary)
        except _PreparationPending as error:
            if time.monotonic() >= deadline:
                raise RuntimeError('Timed out waiting for rank zero to prepare the data cache.') from error
            time.sleep(.1)


def _load_or_prepare_data_module(data_config, work_directory, reload=False, prepare=True):
    """Trust the existing cache; rebuild only when absent or explicitly requested.

    Source files, preparation settings and stored chunks are not rechecked.
    Use ``reload=True`` after changing inputs or preparation settings. Old
    generations are retained while readers may still use them.
    Preparation preserves the caller's Python, NumPy and CPU Torch RNG states.
    """
    root=Path(work_directory).expanduser().resolve()
    manifest=root/'prepared-data.json'
    # Manifests are replaced atomically. Readers never acquire the writer lock,
    # and followers never prepare, even when the command includes --reload.
    if manifest.is_file() and not reload:
        return _load_cached_module(data_config, root, manifest)
    if not prepare:
        raise _PreparationPending('Waiting for rank zero to prepare the data cache.')
    root.mkdir(parents=True,exist_ok=True)
    with FileLock(str(root/'preparation.lock')):
        if manifest.is_file() and not reload:
            return _load_cached_module(data_config, root, manifest)
        config=dict(data_config)
        preparation_seed=config.pop('preparation_seed',0)
        if not isinstance(preparation_seed,int) or isinstance(preparation_seed,bool) or preparation_seed < 0:
            raise ValueError('preparation_seed must be a nonnegative integer.')
        signature=uuid.uuid4().hex
        generation=root/f'data-{signature}'
        generation.mkdir()
        numpy_state,python_state,torch_state=np.random.get_state(),random.getstate(),torch.random.get_rng_state()
        try:
            np.random.seed(preparation_seed)
            random.seed(preparation_seed)
            torch.random.default_generator.manual_seed(preparation_seed)
            module=SphericalDataModule(**config,work_directory=str(generation),preparation_seed=preparation_seed)
        finally:
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            torch.random.set_rng_state(torch_state)
        _apply_runtime_settings(module, data_config)
        _write_metadata(module, root, manifest, signature)
        module.cache_signature=signature
        module.cache_manifest=str(manifest)
        module.cache_reused=False
        _LOADED_MODULES[root]=module
        return module
