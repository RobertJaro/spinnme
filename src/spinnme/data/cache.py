"""Reusable offline preparation caches with JSON metadata and lazy Zarr arrays.

Only a fixed set of package dataset classes is restored. The cache contains no
pickled Python objects and is not a portable inference snapshot. A preparation
lock and atomic metadata replacement make consecutive prep/run calls share the
same arrays. Inputs are never downloaded from this module.
"""
from __future__ import annotations

import json
import os
import random
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from weakref import WeakValueDictionary

import numpy as np
import torch
import zarr
from astropy import units as u
from astropy.io import fits
from astropy.wcs import WCS
from filelock import FileLock
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import rank_zero_only

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
from spinnme.progress import progress

_LOADED_MODULES = WeakValueDictionary()
DATASET_CLASSES = {cls.__name__:cls for cls in (SphericalDataset, FITSSphericalDataset,
    HMISphericalDataset, PHIHRTSphericalDataset, PHIFDTSphericalDataset,
    HinodeSphericalDataset, TestSphericalDataset)}
MODULE_FIELDS = ('spectral_response_files','spectral_response_by_acquisition',
    'batch_size','dataset_batch_size','balance_train_configs','num_workers','num_dataset_workers',
    'train_config_indices','wavelength_config','stokes_normalization_by_instrument',
    'ref_time','seconds_per_dt','Rs_per_ds','gauss_per_dB','times','image_shape',
    'value_range','data_range','preparation_seed')


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


def _decode(value):
    if isinstance(value,list):
        return [_decode(v) for v in value]
    if not isinstance(value,dict):
        return value
    kind=value.get('__kind__')
    if kind == 'wcs':
        return WCS(fits.Header.fromstring(value['header'],sep='\n'))
    if kind == 'quantity':
        return np.asarray(value['value'])*u.Unit(value['unit'])
    if kind == 'datetime':
        return datetime.fromisoformat(value['value'])
    if kind == 'tuple':
        return tuple(_decode(v) for v in value['value'])
    if kind == 'array':
        return np.asarray(value['value'],dtype=value['dtype'])
    if kind == 'tensor':
        dtype={'float32':torch.float32,'float64':torch.float64,'int32':torch.int32,
               'int64':torch.int64,'bool':torch.bool}.get(value['dtype'])
        if dtype is None:
            raise ValueError('Unsupported tensor dtype in prepared cache.')
        return torch.tensor(value['value'],dtype=dtype)
    if kind is not None:
        raise ValueError(f'Unknown cache metadata type {kind!r}.')
    return {key:_decode(v) for key,v in value.items()}


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
    arrays={}
    for key,path in dataset.batches_file_paths.items():
        directory=_inside(path,root)
        arrays[key]={'path':str(directory.relative_to(root))}
    return {'class':name,'attributes':_encode(attrs),'arrays':arrays}


def _restore_dataset(metadata, root):
    dataset_class=DATASET_CLASSES.get(metadata['class'])
    if dataset_class is None:
        raise ValueError('Unsupported dataset class in cache.')
    data,paths={},{}
    for key,description in metadata['arrays'].items():
        directory=_inside(root/description['path'],root)
        array=zarr.open_array(str(directory),mode='r')
        data[key],paths[key]=array,str(directory)
    dataset=dataset_class.__new__(dataset_class)
    dataset.__dict__.update(_decode(metadata['attributes']))
    dataset.batches_file_paths=paths
    dataset.data=data
    return dataset


def _restore(metadata, root):
    module=SphericalDataModule.__new__(SphericalDataModule)
    LightningDataModule.__init__(module)
    attrs=_decode(metadata['module'])
    module.__dict__.update(attrs)
    module.train_datasets=[_restore_dataset(v,root) for v in
                           progress(metadata['training'], desc='Loading training data (cache)')]
    for validation in progress([metadata['validation']], desc='Loading validation data (cache)'):
        module.valid_dataset=_restore_dataset(validation,root)
    module.observations=module.get_observations()
    return module


class _PreparationPending(RuntimeError):
    pass


def load_or_prepare_data_module(data_config, work_directory, reload=False):
    """Prepare on rank zero; other ranks only load its prepared cache.

    Followers wait up to an hour for rank zero when no cache exists.
    ``reload`` forces preparation only on rank zero, including when Lightning
    relaunches the script with the original command-line arguments.
    """
    rank = int(os.environ.get('RANK', os.environ.get('SLURM_PROCID', rank_zero_only.rank)))
    deadline = time.monotonic() + 3600
    while True:
        try:
            return _load_or_prepare_data_module(data_config, work_directory,
                                                reload=reload and rank == 0, prepare=rank == 0)
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
    root.mkdir(parents=True,exist_ok=True)
    manifest=root/'prepared-data.json'
    with FileLock(str(root/'preparation.lock')):
        if manifest.is_file() and not reload:
            module=_LOADED_MODULES.get(root)
            if module is None:
                metadata=json.loads(manifest.read_text())
                module=_restore(metadata,root)
                module.cache_signature=metadata['signature']
                module.cache_manifest=str(manifest)
                _LOADED_MODULES[root]=module
            module.cache_reused=True
            return module
        if not prepare:
            raise _PreparationPending('Waiting for rank zero to prepare the data cache.')
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
        metadata={'signature':signature,
                  'module':_encode({key:getattr(module,key) for key in MODULE_FIELDS}),
                  'training':[_dataset_metadata(ds,root) for ds in module.train_datasets],
                  'validation':_dataset_metadata(module.valid_dataset,root)}
        with tempfile.NamedTemporaryFile(mode='w',dir=root,suffix='.json.tmp',delete=False) as stream:
            json.dump(metadata,stream,allow_nan=False)
            temporary=Path(stream.name)
        temporary.replace(manifest)
        module.cache_signature=signature
        module.cache_manifest=str(manifest)
        module.cache_reused=False
        _LOADED_MODULES[root]=module
        return module
