"""Packaged YAML configuration discovery and typed, validated overrides."""
from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from importlib.resources import files
from pathlib import Path

import yaml

SCHEMA_VERSION = 1
TOP_LEVEL_KEYS = {
    "schema_version", "path", "work_path", "data", "instrument",
    "model", "training", "normalization", "stokes_loss", "weights", "physics", "logging",
}
TRAINING_KEYS = {'epochs', 'max_steps', 'check_val_every_n_epoch', 'val_check_interval', 'num_sanity_val_steps',
                 'lr_params', 'optimizer_config', 'time_shift_config', 'metric_log_interval', 'plot_validation',
                 'compile_config', 'instrument_weights', 'wavelength_reduction', 'accelerator', 'devices',
                 'num_nodes', 'precision', 'strategy', 'limit_train_batches', 'limit_val_batches', 'enable_progress_bar',
                 'enable_model_summary', 'gradient_clip_val', 'deterministic', 'log_every_n_steps',
                 'accumulate_grad_batches'}
DATA_KEYS = {'train_configs', 'valid_config', 'seconds_per_dt', 'Rs_per_ds', 'gauss_per_dB',
             'stokes_normalization', 'ref_time', 'batch_size', 'dataset_batch_size', 'num_workers',
             'num_dataset_workers', 'balance_train_configs', 'preparation_seed',
             'prefetch_factor', 'persistent_workers', 'pin_memory'}
DATASET_KEYS = {'type', 'data_path', 'instrument_id', 'line_id', 'loss_weight', 'stokes_normalization',
                'data_hdu', 'spatial_header_hdu', 'axis_order', 'stokes_order', 'stokes_signs',
                'wavelength_center', 'wavelength_grid', 'wavelength_unit', 'wavelength_grid_is_offsets',
                'wavelength_hdu', 'wavelength_range', 'spatial_wcs_axes', 'observer', 'time_reference', 'time_scale',
                'continuum_index', 'time_hdu', 'detector_shape', 'detector_origin', 'detector_step',
                'mu_limit', 'pixel_mask', 'polarization_filter', 'oversample_factor', 'noise',
                'sample_idx', 'n_samples', 'first_n_samples',
                'time_range', 'exclude_samples', 'exclude_times', 'transmission_profile_directory',
                'resolution',
                'filter_nans', 'shuffle', 'pad_last_batch'}


def list_configs():
    """Return bundled configuration names, independent of the working directory."""
    root = files("spinnme").joinpath("configs")
    names = []
    for directory in sorted(root.iterdir(), key=lambda p: p.name):
        if directory.is_dir():
            names.extend(f"{directory.name}/{p.name[:-5]}" for p in directory.iterdir() if p.name.endswith(".yaml"))
    return sorted(names)


def _parse_yaml(content, source):
    try:
        return yaml.safe_load(content)
    except yaml.YAMLError as error:
        raise ValueError(f'Invalid YAML in {source}: {error}') from error


def read_config(source):
    if isinstance(source, dict):
        return deepcopy(source)
    path = Path(source).expanduser()
    if path.is_file():
        content = path.read_text()
    else:
        name = str(source).removesuffix(".yaml")
        if name not in list_configs():
            raise ValueError(f"Unknown configuration {source!r}. Use 'spinnme config list'.")
        content = files("spinnme").joinpath("configs", name + ".yaml").read_text()
    config = _parse_yaml(content, f'configuration {source!r}')
    if not isinstance(config, dict):
        raise ValueError("Configuration must be a YAML mapping.")
    return config


PLACEHOLDER = re.compile(r'<<([A-Za-z_][A-Za-z0-9_]*)>>')


def placeholder_keys(value):
    """Find named overwrite keys anywhere in a YAML value."""
    if isinstance(value, dict):
        return set().union(*(placeholder_keys(item) for item in value.values())) if value else set()
    if isinstance(value, list):
        return set().union(*(placeholder_keys(item) for item in value)) if value else set()
    return set(PLACEHOLDER.findall(value)) if isinstance(value, str) else set()


def substitute_placeholders(value, bindings, *, strict=True):
    """Replace <<KEY>> values without executing code or shell expressions."""
    def replace(item):
        if isinstance(item, dict):
            return {key: replace(child) for key, child in item.items()}
        if isinstance(item, list):
            return [replace(child) for child in item]
        if not isinstance(item, str):
            return item
        match = PLACEHOLDER.fullmatch(item)
        if match and match[1] in bindings:
            return deepcopy(bindings[match[1]])
        return PLACEHOLDER.sub(lambda match: str(bindings[match[1]]) if match[1] in bindings else match[0], item)
    result = replace(value)
    missing = placeholder_keys(result)
    if strict and missing:
        flags = ', '.join('--'+key for key in sorted(missing))
        raise ValueError(f'Missing YAML overwrite values: {flags}. Supply --KEY VALUE.')
    return result


def _override(config, expression):
    if "=" not in expression:
        raise ValueError(f"Override must be KEY=VALUE: {expression!r}")
    name, raw = expression.split("=", 1)
    parts = name.split(".")
    node = config
    for part in parts[:-1]:
        if isinstance(node, list):
            try:
                node = node[int(part)]
            except (ValueError, IndexError) as error:
                raise ValueError(f"Invalid list index in override {name!r}.") from error
        elif isinstance(node, dict) and part in node:
            node = node[part]
        else:
            raise ValueError(f"Unknown configuration path {name!r}.")
    leaf = parts[-1]
    value = _parse_yaml(raw, f'override {name!r}')
    if isinstance(node, list):
        try:
            node[int(leaf)] = value
        except (ValueError, IndexError) as error:
            raise ValueError(f"Invalid list index in override {name!r}.") from error
    elif isinstance(node, dict):
        node[leaf] = value
    else:
        raise ValueError(f"Cannot override scalar configuration path {name!r}.")


def training_budget(training):
    """Return explicit finite integer limits shared by YAML and Python runs."""
    if not isinstance(training, dict):
        raise ValueError('training must be a YAML mapping.')
    limits = {}
    for key in ('epochs', 'max_steps'):
        value = training.get(key, -1)
        if isinstance(value, bool) or not isinstance(value, int):
            raise ValueError(f'training.{key} must be an integer.')
        if value != -1 and value < 1:
            raise ValueError(f'training.{key} must be -1 or a positive integer.')
        limits[key] = value
    if max(limits.values()) < 1:
        raise ValueError('Set a finite positive training.max_steps or training.epochs; an unbounded run is not allowed.')
    return limits['epochs'], limits['max_steps']


def validate_config(config, *, require_input=True):
    """Validate the public envelope; adapters and model constructors validate their options."""
    if not isinstance(config, dict):
        raise ValueError('Configuration must be a YAML mapping.')
    unknown = set(config) - TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(f"Unknown top-level configuration keys: {sorted(unknown)}")
    if config.get("schema_version", 1) != SCHEMA_VERSION:
        raise ValueError(f"Unsupported schema_version: {config.get('schema_version')}")
    def has_replay(node):
        if isinstance(node, dict):
            return any("replay" in str(key).lower() or has_replay(v) for key, v in node.items())
        return isinstance(node, list) and any(has_replay(v) for v in node)
    if has_replay(config):
        raise ValueError("Replay has been removed. Use optional polarization oversampling instead.")
    logging = config.get('logging', {})
    if not isinstance(logging, dict):
        raise ValueError('logging must be a YAML mapping.')
    if logging.get('project', 'spinnme') != 'spinnme':
        raise ValueError('logging.project must be spinnme.')
    training = config.get("training", {})
    training_budget(training)
    unknown_training = set(training) - TRAINING_KEYS
    if unknown_training:
        raise ValueError(f'Unknown training configuration keys: {sorted(unknown_training)}')
    data = config.get('data', {})
    if not isinstance(data, dict):
        raise ValueError('data must be a YAML mapping.')
    unknown_data = set(data) - DATA_KEYS
    if unknown_data:
        raise ValueError(f'Unknown data configuration keys: {sorted(unknown_data)}')
    datasets = data.get("train_configs", [])
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("data.train_configs must contain at least one spherical dataset.")
    supported = {"hmi", "phi-hrt", "phi-fdt", "phi_hrt", "phi_fdt", "fits", "hinode", "test"}
    definitions = config.get('instrument', [])
    if not isinstance(definitions, list) or not definitions or any(
            not isinstance(row, dict) or not isinstance(row.get('instrument_id'), str)
            or not row['instrument_id'] or '.' in row['instrument_id'] for row in definitions):
        raise ValueError('instrument must list mappings with nonempty instrument_id strings without dots.')
    instruments = {row['instrument_id'] for row in definitions}
    if len(instruments) != len(definitions):
        raise ValueError('Instrument identifiers must be unique.')
    validation = data.get('valid_config')
    validations = validation if isinstance(validation, list) else [validation]
    if not validations or any(not isinstance(value, dict) for value in validations):
        raise ValueError('data.valid_config must be a dataset mapping or a nonempty list of dataset mappings.')
    for dataset in [*datasets, *validations]:
        if not isinstance(dataset, dict):
            raise ValueError('Each dataset configuration must be a YAML mapping.')
        unknown_dataset = set(dataset) - DATASET_KEYS
        if unknown_dataset:
            raise ValueError(f'Unknown dataset configuration keys: {sorted(unknown_dataset)}')
        if dataset.get("type") not in supported:
            raise ValueError(f"Unsupported spherical dataset type {dataset.get('type')!r}.")
        if dataset.get("instrument_id") not in instruments:
            raise ValueError(f"Dataset instrument_id {dataset.get('instrument_id')!r} has no instrument definition.")
        if dataset.get('type') == 'hinode':
            definition = next(row for row in definitions if row['instrument_id'] == dataset['instrument_id'])
            if definition.get('type') != 'hinode':
                raise ValueError('Calibrated Hinode data require instrument type: hinode for solar-frame polarization.')
        if require_input and not dataset.get("data_path"):
            raise ValueError("Supply --input or data.train_configs[*].data_path.")
    return config


def resolve_config(source, *, overrides=(), input_path=None, output=None, work_path=None, instrument=None, bindings=None,
                   require_input=True):
    config = read_config(source)
    variables = dict(bindings or {})
    keys = placeholder_keys(config)
    if instrument:
        instrument_path = Path(instrument)
        if instrument_path.is_file():
            definitions = _parse_yaml(instrument_path.read_text(), f'instrument {str(instrument_path)!r}')
        else:
            from .instruments import load_instrument_definition
            definitions = load_instrument_definition(instrument)
        definitions = definitions if isinstance(definitions, list) else [definitions]
        if not definitions or any(not isinstance(definition, dict) for definition in definitions):
            raise ValueError('Instrument YAML must contain a mapping or a nonempty list of mappings.')
        forward_keys = {'type', 'instrument_id', 'line_id', 'line', 'atomic_parameters', 'line_config',
                        'j_up', 'j_low', 'g_up', 'g_low', 'velocity_correction', 'limb_correction', 'artifact_correction',
                        'spectral_response'}
        data_keys = {'axis_order', 'stokes_order', 'stokes_signs', 'wavelength_center', 'wavelength_grid',
                     'wavelength_unit', 'wavelength_grid_is_offsets', 'wavelength_hdu', 'wavelength_range', 'time_reference',
                     'time_hdu', 'continuum_index'}
        config['instrument'] = [definition.get('model', {k: v for k, v in definition.items() if k in forward_keys})
                                for definition in definitions]
        for forward in config['instrument']:
            if not isinstance(forward, dict):
                raise ValueError('Instrument model settings must be a YAML mapping.')
            # Some bundled definitions describe the required acquisition calibration
            # in prose; a runtime passband is always an explicit mapping.
            if isinstance(forward.get('spectral_response'), str):
                if forward['spectral_response'] == 'acquisition_calibration_required':
                    forward.pop('spectral_response')
                else:
                    forward['spectral_response'] = {'type': forward['spectral_response']}
        if len(definitions) == 1:
            definition = definitions[0]
            forward = config['instrument'][0]
            changes = definition.get('data', {k: v for k, v in definition.items() if k in data_keys})
            if not isinstance(changes, dict):
                raise ValueError('Instrument data settings must be a YAML mapping.')
            datasets = [config['data']['train_configs'][0]]
            if 'valid_config' in config['data']:
                validation = config['data']['valid_config']
                datasets.extend(validation if isinstance(validation, list) else [validation])
            for dataset in datasets:
                dataset.update(deepcopy(changes))
                dataset['instrument_id'] = forward['instrument_id']
                if 'line_id' in forward:
                    dataset['line_id'] = forward['line_id']
    if input_path is not None:
        for index, dataset in enumerate(config["data"]["train_configs"]):
            if index == 0 or dataset.get("data_path") is None:
                dataset["data_path"] = input_path
        if "valid_config" in config["data"]:
            validation = config["data"]["valid_config"]
            for dataset in validation if isinstance(validation, list) else [validation]:
                dataset["data_path"] = input_path
    if output is not None:
        variables['run_path'] = str(output)
        config['path'] = '<<run_path>>'
        if work_path is None:
            variables.setdefault('work_path', str(Path(output)/'cache'))
            config['work_path'] = '<<work_path>>'
    if work_path is not None:
        variables['work_path'] = str(work_path)
        config['work_path'] = '<<work_path>>'
    for expression in overrides:
        name, separator, raw = expression.partition('=')
        if separator and name in keys | {'run_path', 'work_path'}:
            variables[name] = _parse_yaml(raw, f'placeholder {name!r}')
        else:
            _override(config, expression)
    config = substitute_placeholders(config, variables, strict=require_input)
    config.setdefault("schema_version", SCHEMA_VERSION)
    config.setdefault("path", "./runs/spinnme")
    config.setdefault("work_path", str(Path(config["path"]) / "cache"))
    config.setdefault("logging", {"mode": "disabled"})
    return validate_config(config, require_input=require_input)


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()


def dump_config(config):
    """Serialize YAML with inline short scalar lists and explicit path headers."""
    if isinstance(config, dict):
        config = {**{key: config[key] for key in ('path', 'work_path') if key in config},
                  **{key: value for key, value in config.items() if key not in ('path', 'work_path')}}

    class Dumper(yaml.SafeDumper):
        pass

    def string(dumper, value):
        return dumper.represent_scalar('tag:yaml.org,2002:str', value,
                                       style='"' if PLACEHOLDER.search(value) else None)

    def sequence(dumper, values):
        scalar = all(isinstance(value, (str, int, float, bool, type(None))) for value in values)
        compact = scalar and len(values) <= 12 and sum(len(str(value)) for value in values) <= 120
        return dumper.represent_sequence('tag:yaml.org,2002:seq', values, flow_style=compact)

    Dumper.add_representer(str, string)
    Dumper.add_representer(list, sequence)
    Dumper.add_representer(tuple, sequence)
    return yaml.dump(config, Dumper=Dumper, sort_keys=False, width=1000)


def write_config(config, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_config(config))
    return path
