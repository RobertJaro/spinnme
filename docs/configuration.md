# Configuration reference

Keep one YAML file per inversion. Start from a bundled YAML and set its inputs, output paths and training options directly. The [training customization guide](training_options.md) explains the common choices with examples.

## Find and copy a preset

```bash
spinnme config list
spinnme config show hmi/full_disk
spinnme config copy phi/hrt --output phi.yaml
spinnme config validate phi.yaml
```

Available presets are `hmi/full_disk`, `hmi/subframe`, `hinode/sp`, `phi/hrt`, `phi/fdt`, `generic/stokes_cube`, `generic/multiple_lines` and `synthetic/circular`. Presets expose their required input paths as placeholders or unset values. Both HMI presets use `<<data_path>>` for the root containing `stokes/` and `transmission/`. `config validate` checks the configuration structure; `prep` additionally opens the data and checks observation metadata.

Bundled instrument definitions can also be inspected or saved:

```bash
spinnme config instrument phi_hrt
spinnme config instrument hinode --output hinode-instrument.yaml
```

`prep` and `run` accept `--instrument` with a YAML path or a bundled name: `hmi`, `hinode`, `phi_hrt`, or `phi_fdt`. This sets the forward model and compatible reader defaults. The standalone `hinode` definition is a single Fe I 6302 channel; use `hinode/sp` without `--instrument` to jointly fit both Hinode lines. Select the corresponding dataset `type` separately.

## Overwrite named values

Run YAMLs begin with:

```yaml
path: "<<run_path>>"
work_path: "<<work_path>>"
```

Each `<<KEY>>` declares an overwrite flag `--KEY VALUE`; repeated occurrences share the value. For example:

```bash
spinnme run --config hmi/full_disk --data_path ./data/hmi \
  --run_path ./runs/hmi90 \
  --work_path ./work/hmi90
```

Use the same syntax for custom placeholders, such as `data_path: "<<stokes_files>>"` with `--stokes_files '/path/*.fits'`. Missing values fail before data preparation. `config show` and `config copy` retain unfilled placeholders. Put model, training and instrument customizations in your own YAML.

## Customize settings in YAML

Use `spinnme config copy NAME --output inversion.yaml`, edit the required sections, and run `spinnme run --config inversion.yaml`. Keep the model, training, normalization and instrument choices together in that file. `config copy/show` also accept typed `--set KEY=VALUE` overrides when creating or inspecting a config.

| Shortcut | Effect |
| --- | --- |
| `--input PATH` | Set the first training dataset and validation input paths |
| `--run_path DIRECTORY` | Fill `<<run_path>>`; `--output` is also accepted |
| `--work_path DIRECTORY` | Fill `<<work_path>>` and choose the data cache/working directory |
| `--device cpu/cuda/auto` | Optional single-device training override; omission retains automatic detection and YAML settings |
| `--reload` | Rebuild the prepared-data cache |

## Main sections

| Key | Purpose |
| --- | --- |
| `path`, `work_path` | Run outputs and working files |
| `data.train_configs` | List of training datasets and their reader settings |
| `data.valid_config` | One validation dataset mapping, or a list for multiple lines/instruments |
| `instrument` | List of forward models, lines, spectral responses and optional corrections |
| `model` | Network type, size and coordinate encoding |
| `training` | Run budget, optimizer, validation and execution settings |
| `stokes_loss` | Profile continuum normalization and MSE loss |
| `normalization` | Linear or asinh scaling of the Stokes components |
| `weights` | I/Q/U/V loss weights or schedules |
| `logging` | W&B mode and short run name; project is always `spinnme` |

## Dataset and instrument settings

Every dataset supplies `type`, `data_path` and `instrument_id`. Supported public observation adapters are `hmi`, `hinode`, `phi-hrt`, `phi-fdt` and generic `fits`. Reader-specific axes, wavelengths, masks and calibration options are explained under [data](data.md) and the instrument guides.

| Common option | Meaning |
| --- | --- |
| `sample_idx` | Select one acquisition; validation defaults to the middle selected acquisition if omitted |
| `time_range` | Inclusive UTC model-epoch range, using the dataset’s `time_reference` and `time_scale` |
| `exclude_samples`, `exclude_times` | Exclude selected indices or UTC model epochs, for example to create a holdout |
| `n_samples`, `first_n_samples` | Select evenly spaced acquisitions or the first N |
| `mu_limit`, `pixel_mask` | Restrict the spatial samples used |
| `stokes_normalization` | Divide observed intensity and polarization by this positive scale |
| `loss_weight` | Relative per-profile loss weight for this dataset |
| `polarization_filter` | Optional polarized-profile selection; see the [oversampling recipe](training_options.md#oversample-polarized-profiles) |

`instrument_id` connects a dataset to its forward model. Acquisitions sharing an ID must use the same line center, wavelength count and Stokes normalization; their wavelength positions may differ. Distinct instrument IDs may have different spectral layouts. `line_id` identifies the atomic line and controls whether line atmosphere parameters are shared.

## Training and model settings

Use [training options](training_options.md) for budgets, learning rates, optimizers, batching, model size, normalization and sampling recipes. Magnetic-field and velocity outputs are linear by default in all bundled run configurations. Optional `model.b_sinh_scale` and `model.v_sinh_scale` enable sinh output scaling in gauss and km/s; omit them or set them to `null` to keep outputs linear.

`data.ref_time` sets the time origin; leave it `null` to use the earliest selected observation. `seconds_per_dt`, `Rs_per_ds` and `gauss_per_dB` define the time, length and field scales. Keep these settings consistent when comparing runs or resuming training.

The model selects its source-function parameterization from `stokes_loss.continuum_normalization`: `ratio` for normalized profiles (the `b0` and `b1` coefficients sum to one) and `independent` for absolute-intensity fits. Normalized spectra do not determine the absolute source-function scale. It is normally unnecessary to set `model.source_function` yourself.

See [training](training.md#save-export-and-resume) for saved files and resume requirements, and the [CLI reference](cli.md) for all command arguments.
