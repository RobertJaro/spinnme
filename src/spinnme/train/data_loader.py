"""Zarr profile caches and time-stratified spherical batch sampling."""
import gc
import hashlib
from collections.abc import Mapping
from pathlib import Path

from filelock import FileLock
import math
import os
import shutil
import uuid

import numpy as np
import torch
import torch.distributed as dist
import zarr
from torch.utils.data import Dataset, Sampler


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value <= 0:
        raise ValueError(f'{name} must be a positive integer.')
    return int(value)


class CombinedDataset(Dataset):

    def __init__(self, datasets):
        super().__init__()
        self.datasets = datasets
        self.sample_indices_by_dataset = []
        for i, d in enumerate(datasets):
            indices = [(i, j) for j in range(len(d))] * d.oversample_factor
            self.sample_indices_by_dataset.append(indices)
        self.sample_indices = [index for indices in self.sample_indices_by_dataset for index in indices]

    def __len__(self):
        return len(self.sample_indices)

    def __getitem__(self, sample_indices):
        # The sampler supplies one complete optimizer batch as dataset/chunk pairs.
        if isinstance(sample_indices, tuple) and len(sample_indices) == 2 \
                and all(isinstance(value, (int, np.integer)) for value in sample_indices):
            sample_indices = [sample_indices]
        batch = {}
        for i, j in sample_indices:
            sample = self.datasets[i][j]
            instrument_id = sample['instrument_id']
            if instrument_id not in batch:
                batch[instrument_id] = {}
            for tensor_key, tensor_value in sample.items():
                if tensor_key == 'instrument_id':
                    continue
                if tensor_key not in batch[instrument_id]:
                    batch[instrument_id][tensor_key] = []
                batch[instrument_id][tensor_key].append(tensor_value)

        # concatenate tensors in the batch
        concatenated_batch = {}
        for ds_key, ds_value in batch.items():
            # use the first instrument_id per dataset
            value = {tensor_key: torch.cat(tensor_value)
                     for tensor_key, tensor_value in ds_value.items()}
            concatenated_batch[ds_key] = value
        return concatenated_batch


class TimeStratifiedBatchSampler(Sampler):
    """Time-balanced chunks, optionally equally split across config entries.

    The first pool defines epoch length; each additional pool cycles independently
    and contributes ``chunks_per_batch`` chunks to every global batch. A short
    final batch is retained. Balanced pools repeat primary chunks to preserve
    their requested mixture; DDP repeats whole batches to give every rank the
    same number of optimizer steps, including for a small field of view.
    """

    def __init__(self, sample_indices_by_dataset, chunks_per_batch, additional_dataset_groups=None):
        self.sample_indices_by_dataset = [list(indices) for indices in sample_indices_by_dataset]
        self.chunks_per_batch = _positive_integer(chunks_per_batch, 'chunks_per_batch')
        self.epoch = 0
        self.additional_dataset_groups = additional_dataset_groups or []
        if not any(self.sample_indices_by_dataset):
            raise ValueError('The primary sampling pool must contain at least one chunk.')
        if any(not any(group) for group in self.additional_dataset_groups):
            raise ValueError('Additional sampling pool must contain at least one chunk.')

    @staticmethod
    def _additional_indices(groups, generator):
        """Cycle each acquisition independently for equal time coverage."""
        groups = [list(group) for group in groups if group]
        queues = [[] for _ in groups]
        while True:
            for i in generator.permutation(len(groups)):
                if not queues[i]:
                    queues[i] = [groups[i][j] for j in generator.permutation(len(groups[i]))]
                yield queues[i].pop()

    @staticmethod
    def _distributed_context():
        if dist.is_available() and dist.is_initialized():
            return dist.get_rank(), dist.get_world_size()
        return 0, 1

    def _global_batch_count(self):
        return math.ceil(sum(map(len, self.sample_indices_by_dataset)) / self.chunks_per_batch)

    def __len__(self):
        _, world_size = self._distributed_context()
        return math.ceil(self._global_batch_count() / world_size)

    def __iter__(self):
        rank, world_size = self._distributed_context()
        generator = np.random.default_rng(self.epoch)
        self.epoch += 1

        queues = []
        for indices in self.sample_indices_by_dataset:
            permutation = generator.permutation(len(indices))
            queues.append([indices[index] for index in permutation])

        active = [index for index, queue in enumerate(queues) if queue]
        ordered = []
        while active:
            generator.shuffle(active)
            next_active = []
            for dataset_index in active:
                ordered.append(queues[dataset_index].pop())
                if queues[dataset_index]:
                    next_active.append(dataset_index)
            active = next_active

        global_batch_count = self._global_batch_count()
        if self.additional_dataset_groups:
            # A short primary batch must not increase the additional pools'
            # relative contribution. Cycle only the few missing chunks.
            required = global_batch_count * self.chunks_per_batch
            count = len(ordered)
            ordered.extend(ordered[index % count] for index in range(required - count))
        batches = [
            ordered[index * self.chunks_per_batch:(index + 1) * self.chunks_per_batch]
            for index in range(global_batch_count)
        ]
        # Build globally before rank sharding, preserving equal DDP step counts.
        for group in self.additional_dataset_groups:
            extra = self._additional_indices(group, generator)
            for batch in batches:
                batch.extend(next(extra) for _ in range(self.chunks_per_batch))
        # Retain all primary chunks while matching step counts across ranks.
        required = math.ceil(global_batch_count / world_size) * world_size
        batches.extend(list(batches[index % global_batch_count])
                       for index in range(required - global_batch_count))
        for batch_index in range(rank, required, world_size):
            yield batches[batch_index]


class LazyZarrArrays(Mapping):
    """Keep only store paths until a worker requests an array."""

    def __init__(self, paths):
        self.paths = dict(paths)
        self._opened = {}

    def __getitem__(self, key):
        if key not in self._opened:
            self._opened[key] = zarr.open_array(self.paths[key], mode='r')
        return self._opened[key]

    def __iter__(self):
        return iter(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getstate__(self):
        return {'paths': self.paths, '_opened': {}}


class BatchesDataset(Dataset):

    def __init__(self, batches_file_paths, batch_size=2 ** 13, **kwargs):
        """Data set for lazy loading a pre-batched numpy data array.

        :param batches_path: path to the numpy array.
        """
        self.batches_file_paths = batches_file_paths
        self.batch_size = _positive_integer(batch_size, 'batch_size')
        # Passing the path directly works with both Zarr 2 and Zarr 3. In Zarr 3,
        # filesystem paths are backed by LocalStore; NestedDirectoryStore was
        # removed from the public API.
        self.data = LazyZarrArrays(self.batches_file_paths)

    def __getstate__(self):
        state = self.__dict__.copy()
        mapped = {key: str(value.filename) for key, value in state.items() if isinstance(value, np.memmap)}
        for key in mapped:
            del state[key]
        state['_mapped_arrays'] = mapped
        return state

    def __setstate__(self, state):
        mapped = state.pop('_mapped_arrays', {})
        self.__dict__.update(state)
        for key, path in mapped.items():
            setattr(self, key, np.load(path, mmap_mode='r', allow_pickle=False))

    def __len__(self):
        count = len(self.profile_rows) if hasattr(self, 'profile_rows') else next(iter(self.data.values())).shape[0]
        n_batches = np.ceil(count / self.batch_size)
        return n_batches.astype(np.int32)

    def __getitem__(self, idx):
        if hasattr(self, 'profile_rows'):
            return ObservationView.read(self, idx)
        # lazy load data
        count = next(iter(self.data.values())).shape[0]
        if getattr(self, 'pad_last_batch', False) and (idx + 1) * self.batch_size > count:
            indices = np.arange(idx * self.batch_size, (idx + 1) * self.batch_size) % count
            data = {k: torch.from_numpy(np.asarray(v.oindex[indices])) for k, v in self.data.items()}
        else:
            data = {k: torch.from_numpy(np.asarray(v[idx * self.batch_size: (idx + 1) * self.batch_size])) for k, v in
                    self.data.items()}
        data['lin_idx'] = torch.tensor([idx], dtype=torch.long)
        return data

    def clear(self):
        if hasattr(self, 'profile_rows'):
            raise RuntimeError('Shared observation arrays cannot be removed through a dataset view.')
        for path in self.batches_file_paths.values():
            shutil.rmtree(path)

    def shuffle(self):
        if hasattr(self, 'profile_rows'):
            raise RuntimeError('Shared observation arrays cannot be shuffled in place.')
        ref_data = next(iter(self.data.values()))
        indices = np.random.permutation(ref_data.shape[0])
        for path in self.batches_file_paths.values():
            array = zarr.open_array(path, mode='r+')
            array[:] = array[:][indices]


class TensorsDataset(BatchesDataset):

    STORAGE_CHUNK_BYTES = 1024 ** 2

    @classmethod
    def storage_chunk_size(cls, array, batch_size):
        row_bytes = max(1, int(np.prod(array.shape[1:])) * array.dtype.itemsize)
        rows = max(1, cls.STORAGE_CHUNK_BYTES // row_bytes)
        if rows >= batch_size:
            rows = rows // batch_size * batch_size
        return min(max(1, array.shape[0]), rows)

    def __init__(self, tensors, work_directory, batch_size, filter_nans=True, shuffle=True, ds_name=None, row_order=None, **kwargs):
        batch_size = _positive_integer(batch_size, 'batch_size')
        # filter nan entries
        if filter_nans:
            nan_mask = np.any([np.any(np.isnan(t), axis=tuple(range(1, t.ndim))) for t in tensors.values()], axis=0)
            if nan_mask.any():
                tensors = {k: v[~nan_mask] for k, v in tensors.items()}

        # shuffle data
        if shuffle:
            r = np.random.permutation(next(iter(tensors.values())).shape[0])
            row_order = r if row_order is None else row_order[r]

        ds_name = uuid.uuid4() if ds_name is None else ds_name
        batches_paths = {}
        for k, v in tensors.items():
            coords_zarr_path = os.path.join(work_directory, f'{ds_name}_{k}.zarr')
            # write data to zarr - chunks
            # Bound decompression cost by bytes, including wide spectra.
            chunk_size = self.storage_chunk_size(v, batch_size)
            z = zarr.open_array(coords_zarr_path, mode='w', shape=v.shape,
                                chunks=(chunk_size, *v.shape[1:]),
                                dtype=v.dtype if np.issubdtype(v.dtype, np.integer) else 'float32')
            for i in range(0, v.shape[0], chunk_size):
                selection = slice(i, i+chunk_size) if row_order is None else row_order[i:i+chunk_size]
                z[i:i + chunk_size] = v[selection]
            batches_paths[k] = coords_zarr_path

        # cleanup memory to avoid out of memory errors when loading data
        del tensors
        gc.collect()

        super().__init__(batches_paths, batch_size=batch_size, **kwargs)


class ObservationView:
    """Persist one randomized observation; select pixels and wavelengths at read time.

    The shared permutation keeps training reads local while preserving uniform
    random profile ordering. Validation maps back to native raster order.
    """

    @staticmethod
    def prepare(dataset, spectra, valid, offsets, wavelengths, work_directory,
                batch_size, filter_nans, shuffle, seed=0):
        spectra = np.asarray(spectra, dtype=np.float32)
        flat = spectra.reshape((-1, *spectra.shape[-2:]))
        digest = hashlib.sha256(str((spectra.shape, spectra.dtype)).encode())
        for start in range(0, len(flat), 4096):
            digest.update(np.ascontiguousarray(flat[start:start+4096]).tobytes())
        if offsets is not None:
            digest.update(np.ascontiguousarray(offsets, dtype=np.float32).tobytes())
        directory = Path(work_directory) / ('observation-' + digest.hexdigest())
        with FileLock(str(directory) + '.lock'):
            marker = directory / 'complete'
            if not marker.exists():
                directory.mkdir(exist_ok=True)
                order = np.random.default_rng(seed ^ int(digest.hexdigest()[:16], 16)).permutation(len(flat)).astype(np.int64)
                tensors = {'stokes': flat, 'pixel_index': np.arange(len(flat), dtype=np.int64)[:, None]}
                if offsets is not None:
                    tensors['time_offset'] = np.asarray(offsets, dtype=np.float32).reshape(-1, 1)
                TensorsDataset(tensors, str(directory), batch_size, filter_nans=False,
                               shuffle=False, ds_name='observation', row_order=order)
                marker.touch()
        paths = {key: str(directory / f'observation_{key}.zarr') for key in
                 ('stokes', 'pixel_index', 'time_offset') if (directory / f'observation_{key}.zarr').exists()}
        order = zarr.open_array(paths['pixel_index'], mode='r')[:, 0]
        rows = np.arange(len(order), dtype=np.int64)
        if not shuffle:
            rows = np.argsort(order)
        if filter_nans:
            rows = rows[valid.ravel()[order[rows]]]
        dataset.profile_rows = rows
        dataset.profile_valid = valid.ravel()
        dataset.profile_wavelengths = np.asarray(wavelengths, dtype=np.int64)
        BatchesDataset.__init__(dataset, paths, batch_size=batch_size)

    @staticmethod
    def read(dataset, idx):
        rows = dataset.profile_rows[idx * dataset.batch_size:(idx+1) * dataset.batch_size]
        if getattr(dataset, 'pad_last_batch', False) and len(rows) < dataset.batch_size:
            rows = dataset.profile_rows[np.arange(idx * dataset.batch_size,
                                                  (idx+1) * dataset.batch_size) % len(dataset.profile_rows)]
        # Most training reads are contiguous (only masked pixels create gaps).
        selection = slice(int(rows[0]), int(rows[-1])+1) if len(rows) and np.all(np.diff(rows) == 1) else rows
        values = {key: np.asarray(array[selection] if isinstance(selection, slice) else array.oindex[selection])
                  for key, array in dataset.data.items()}
        pixels = values['pixel_index'].reshape(-1)
        values['stokes'] = values['stokes'][..., dataset.profile_wavelengths] / dataset.stokes_normalization
        values['stokes'][~dataset.profile_valid[pixels]] = np.nan
        if 'time_offset' in values:
            values['time_offset'] = values['time_offset'] / dataset.seconds_per_dt
        return {**{key: torch.from_numpy(value) for key, value in values.items()},
                'lin_idx': torch.tensor([idx], dtype=torch.long)}
