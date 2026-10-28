"""Import-safe training orchestration for the public CLI and Python API."""
from __future__ import annotations

import copy
import json
import os
import platform
import subprocess
import time
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import LambdaCallback, ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from pytorch_lightning.utilities import rank_zero_only

from spinnme.data.cache import load_or_prepare_data_module
from spinnme.distributed import is_primary_process
from spinnme.train.me_spherical_module import MESphericalModule
from spinnme.train.physics_regularization import build_spherical_collocation_domain, split_stokes_and_physics_config

TRAINER_OPTIONS = {
    'accelerator', 'devices', 'num_nodes', 'precision', 'limit_train_batches',
    'limit_val_batches', 'enable_progress_bar', 'enable_model_summary',
    'gradient_clip_val', 'deterministic', 'log_every_n_steps',
    'accumulate_grad_batches', 'check_val_every_n_epoch', 'val_check_interval',
    'num_sanity_val_steps', 'strategy',
}



def _logger(configuration, directory):
    configuration = copy.deepcopy(configuration or {})
    configuration.setdefault('project', 'spinnme')
    if configuration['project'] != 'spinnme':
        raise ValueError('logging.project must be spinnme.')
    configuration.setdefault('name', Path(directory).parent.name)
    mode = configuration.pop('mode', 'disabled')
    if mode == 'disabled':
        return False
    if mode not in ('offline', 'online'):
        raise ValueError('logging.mode must be disabled, offline, or online.')
    try:
        from pytorch_lightning.loggers import WandbLogger
    except ImportError as error:
        raise ImportError('Install spinnme[wandb] to enable Weights & Biases logging.') from error
    logger = WandbLogger(save_dir=str(directory), offline=mode == 'offline', **configuration)
    # Initialize before data preparation, whose diagnostics use an active run.
    _ = logger.experiment
    return logger



@rank_zero_only
def _write_manifest(path, manifest):
    if not is_primary_process():
        return
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(manifest, indent=2, default=str, allow_nan=False))
    temporary.replace(path)


def _environment():
    versions = {}
    for package in ('spinnme', 'torch', 'pytorch-lightning', 'numpy', 'astropy', 'sunpy', 'zarr'):
        try:
            versions[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            versions[package] = None
    source = Path(__file__).resolve().parents[3]
    revision, dirty = None, None
    try:
        revision = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=source, capture_output=True,
                                  text=True, check=True, timeout=3).stdout.strip()
        dirty = bool(subprocess.run(['git', 'status', '--porcelain'], cwd=source, capture_output=True,
                                    text=True, check=True, timeout=3).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {'python': platform.python_version(), 'platform': platform.platform(), 'versions': versions,
            'git_revision': revision, 'git_dirty': dirty,
            'hardware': {'cpu_count': os.cpu_count(), 'cuda_available': torch.cuda.is_available(),
                         'cuda_version': torch.version.cuda,
                         'gpus': [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]}}


def _resume_configuration(config, resolved_path):
    """Compare scientific settings before replacing a run's resolved YAML."""
    from spinnme.configuration import config_hash, read_config
    if not resolved_path.is_file():
        raise FileNotFoundError(f'Cannot validate resume settings: {resolved_path} does not exist.')
    def scientific(configuration):
        selected = {key: copy.deepcopy(configuration.get(key)) for key in
                    ('data', 'instrument', 'model', 'normalization', 'stokes_loss', 'weights', 'physics')}
        # Diagnostic selections do not alter the fitted model or training loss.
        if selected['data'] is not None:
            selected['data'].pop('valid_config', None)
        runtime = TRAINER_OPTIONS | {'epochs', 'max_steps', 'compile_config', 'plot_validation', 'metric_log_interval'}
        selected['training'] = {key: value for key, value in configuration.get('training', {}).items()
                                if key not in runtime}
        return selected
    if config_hash(scientific(read_config(resolved_path))) != config_hash(scientific(config)):
        raise ValueError('Resume changes scientific/data/objective configuration. Use a fresh output directory.')


def _input_inventory(data_module):
    observations = (data_module.get_observations() if hasattr(data_module, 'get_observations')
                    else getattr(data_module, 'observations', []))
    paths = set()
    for row in observations:
        paths.update(row.get('source_paths', row.get('metadata', {}).get('source_paths', [])))
    return [{'path': value} for value in sorted(paths)]

def run(config, resume=False, profile=False, reload=False):
    """Prepare data, train for a finite budget, and save ``inversion.pme``.

    No run resumes implicitly. ``resume=True`` requires ``last.ckpt`` in the
    output directory. Logging is disabled unless explicitly configured. Returns
    ``(module, data_module, trainer)`` for callers that need in-memory results.
    """
    from spinnme.configuration import config_hash, training_budget, write_config
    from spinnme.snapshots import save_snapshot

    primary = is_primary_process()
    config = copy.deepcopy(config)
    output = Path(config.get('path', config.get('output', 'results'))).expanduser()
    work = Path(config.get('work_path', output / 'cache')).expanduser()
    training = copy.deepcopy(config.get('training', {}))
    if 'replay_config' in training:
        raise ValueError('Replay sampling was removed; use opt-in polarization oversampling.')
    epochs, max_steps = training_budget(training)
    training.pop('epochs', None)
    training.pop('max_steps', None)
    if (output / 'inversion.pme').exists() and not resume:
        raise FileExistsError(f'{output}/inversion.pme exists; choose a fresh output or resume explicitly.')
    checkpoint_path = output / 'last.ckpt'
    if resume and not checkpoint_path.is_file():
        raise FileNotFoundError(f'Cannot resume: {checkpoint_path} does not exist.')
    if resume:
        _resume_configuration(config, output / 'resolved.yaml')
    output.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    manifest_path = output / 'run_manifest.json'
    manifest = {'schema_version': 1, 'status': 'preparing',
                'started_utc': datetime.now(UTC).isoformat(),
                'config_hash': config_hash(config),
                'budget': {'epochs': epochs, 'max_steps': max_steps}, 'resume': bool(resume),
                'environment': _environment() if primary else None, 'inputs': []}
    _write_manifest(manifest_path, manifest)
    if primary:
        write_config(config, output / 'resolved.yaml')
    try:
        logger = _logger(config.get('logging'), work) if primary else False
        if logger:
            logger.log_hyperparams(config)
        data_module = load_or_prepare_data_module(copy.deepcopy(config['data']), work_directory=str(work), reload=reload)
        manifest['preparation'] = {'cache_signature': getattr(data_module, 'cache_signature', None),
                                   'cache_reused': getattr(data_module, 'cache_reused', False),
                                   'cache_manifest': str(getattr(data_module, 'cache_manifest', ''))}
        if primary:
            manifest['inputs'] = _input_inventory(data_module)
        manifest['status'] = 'training'
        _write_manifest(manifest_path, manifest)
        trainer_options = {key: training.pop(key) for key in list(training) if key in TRAINER_OPTIONS}
        weights_config, physics_config = split_stokes_and_physics_config(
            copy.deepcopy(config.get('weights', {})), copy.deepcopy(config.get('physics')))
        physics_domain = None
        if physics_config is not None:
            physics_domain = build_spherical_collocation_domain(
                data_module.train_datasets, data_module.Rs_per_ds,
                physics_config.get('radius_range_Rs'))
        module = MESphericalModule(
            image_shape=data_module.image_shape, wavelength_config=data_module.wavelength_config,
            model_config=copy.deepcopy(config.get('model', {})),
            instrument_config=copy.deepcopy(config.get('instrument', [])),
            normalization_config=copy.deepcopy(config.get('normalization', {})),
            stokes_loss_config=copy.deepcopy(config.get('stokes_loss', {})),
            physics_config=physics_config, physics_domain=physics_domain,
            Rs_per_ds=data_module.Rs_per_ds, seconds_per_dt=data_module.seconds_per_dt,
            gauss_per_dB=data_module.gauss_per_dB, weights_config=weights_config, **training)

        @rank_zero_only
        def persist(*args, **kwargs):
            save_snapshot(module, data_module, config, output / 'inversion.pme')

        callbacks = [ModelCheckpoint(dirpath=str(output), save_last=True),
                     LambdaCallback(on_validation_epoch_end=persist)]
        profiler = None
        if profile:
            from spinnme.train.diagnostics import training_diagnostics
            profiler, monitor = training_diagnostics(str(output / 'profiling'))
            callbacks.append(monitor)
        options = {'accelerator': 'auto', 'devices': 'auto', 'num_sanity_val_steps': 0,
                       'gradient_clip_val': .5, 'use_distributed_sampler': False,
                       'check_val_every_n_epoch': 1}
        options.update(trainer_options)
        if options.get('check_val_every_n_epoch') is None and 'val_check_interval' not in options:
            raise ValueError('val_check_interval is required when check_val_every_n_epoch is null.')
        # Separate line/instrument branches need unused-parameter detection in DDP.
        devices = options.get('devices')
        device_count = (len(devices) if isinstance(devices, (list, tuple)) else
                        devices if isinstance(devices, int) else max(1, torch.cuda.device_count()))
        if 'strategy' not in options and (int(options.get('num_nodes', 1)) > 1 or device_count > 1):
            options['strategy'] = DDPStrategy(find_unused_parameters=True)
        trainer = Trainer(max_epochs=epochs, max_steps=max_steps, logger=logger,
                          callbacks=callbacks, profiler=profiler, **options)
        trainer.fit(module, data_module, ckpt_path=str(checkpoint_path) if resume else None)
        # Validation may not coincide with max_steps. All ranks participate in
        # Lightning's strategy-aware save; final resume and inference state agree.
        trainer.save_checkpoint(str(checkpoint_path))
        persist()
        manifest.update(status='complete', completed_utc=datetime.now(UTC).isoformat(),
                        elapsed_seconds=time.perf_counter() - started, steps=int(trainer.global_step))
        _write_manifest(manifest_path, manifest)
        return module, data_module, trainer
    except BaseException as error:
        manifest.update(status='failed', completed_utc=datetime.now(UTC).isoformat(),
                        elapsed_seconds=time.perf_counter() - started,
                        error={'type': type(error).__name__, 'message': str(error)})
        _write_manifest(manifest_path, manifest)
        raise
