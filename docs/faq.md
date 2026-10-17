# Beginner FAQ

## What does SPINNME do?

It fits a model to calibrated polarized solar spectra and estimates magnetic-field, Doppler-velocity and atmosphere maps. An **inversion** works backward from measured light to parameters that could produce it. Start with the [offline example](circular.md) to check the workflow, then the [HMI download-to-export quickstart](quickstart.md).

## What are Stokes I, Q, U and V?

They describe intensity (**I**), two components of linear polarization (**Q/U**) and circular polarization (**V**). Each is measured at several wavelengths. SPINNME fits all four; polarization contains information about the magnetic field. Use the calibrated component order and signs for your instrument.

## Can I run without installing anything on my computer?

Use the [Colab notebooks](colab.md). They install SPINNME in the online runtime and guide you through a small inversion, fit checks and exports. Start with the generated circular pattern, or choose HMI, Hinode, PHI or custom FITS. Download the results before the runtime ends.

## Can I use any FITS file or an existing magnetogram?

Inversion needs calibrated **I/Q/U/V spectra**, their wavelength sampling and correct solar observing geometry. A continuum image, photograph or pipeline magnetic map is insufficient. A pipeline map can serve as an [evaluation reference](evaluation.md). The separate [crop command](cropping.md) works on ordinary FITS images too.

**FITS** stores scientific arrays and metadata. **WCS** maps array pixels to physical sky coordinates. Correct-looking images can still have incorrect WCS; see [input requirements](data.md).

## Do I need to train a neural network myself?

`spinnme run` fits a new network to your selected observation or collection. You can use the bundled YAML settings without writing ML code. This release provides configurations, rather than a pretrained model for arbitrary observations. A saved `.pme` contains your fitted model; [how the fit works](method.md) explains the representation.

## What does Milne–Eddington mean? What does spherical mean?

Milne–Eddington is a simplified atmosphere used to synthesize polarized spectral lines. The fitted maps describe effective line-forming conditions, rather than a height-resolved atmosphere. Spherical means the fit uses solar-surface geometry and viewing directions across the curved Sun. See [model assumptions](method.md#what-the-fit-represents).

## Do I need a GPU, JSOC registration or internet access?

A CPU can run the small [offline example](circular.md); its synthetic-data dependency must first be installed. Training automatically detects an available GPU accelerator, otherwise using CPU. Snapshot exports prefer CUDA when available, otherwise CPU. You do not need to set a device; `--device cpu` is an optional override. Runtime and memory depend on the data, spectral response, batch size and model.

HMI observation and calibration downloads require internet access and a registered JSOC email. Already-prepared local observations can be inverted offline. JSOC registration is unnecessary for synthetic data, local exports or other archives.

## Where are my results, and how do I view them?

`--run_path ./runs/hmi` writes under the directory where you ran the command. After training, export a scalar preview:

```bash
spinnme export ./runs/hmi/inversion.pme --all-observations \
  --parameters Blos --format jpg --output ./products/hmi-preview
```

JPG is a quick look. Use FITS or NPZ for analysis. The default `Brtp` selection writes radial, southward and longitudinal magnetic components; [exports](exports.md) lists the keys, units and signs.

## Which files should I keep? Can I resume?

Keep `inversion.pme` to export the fitted model on its saved grids without the original training files. Keep `last.ckpt`, `resolved.yaml`, inputs and calibration to [resume compatible training](training.md#save-export-and-resume). A `.pme` is an inference snapshot; it does not contain the optimizer state needed to resume. Choose a new run directory for a different scientific setup.

## Is a completed example ready for scientific use?

Completion means the requested computation finished. Five synthetic steps or 1,000 HMI steps demonstrate the workflow. Check fits by Stokes component and spectral line, compare residuals with noise, and test held-out observations where available. A low loss does not provide error bars or prove a unique solution. Follow the [fit-quality guide](evaluation.md#check-the-stokes-fit-before-comparing-maps).

## Does a finer exported grid or cadence add information?

It samples the continuous fitted model more densely. It adds no measurements or guaranteed resolving power. One acquisition cannot establish time evolution; a scanning raster samples different positions at different times. See [observation types](method.md#one-acquisition-a-raster-or-a-time-series) and [Python exports](python_exports.md).

## Why can my field differ from a pipeline inversion?

Check calibration, spectral response, resolution, timing and conventions first. The magnetic azimuth has an unresolved 180° ambiguity; signed transverse and spherical components can select different branches. A single viewing direction also does not independently constrain three velocity components. Compare ambiguity-safe quantities and read the [interpretation limits](method.md#interpretation).

## What should I change for my first custom run?

Copy the matching instrument preset, verify its data/axes/wavelengths/response, choose a small region or time selection, and check a short fit before expanding it. Put changes in one YAML per run; pass values for its `<<KEY>>` placeholders as `--KEY VALUE`. Use [training options](training_options.md), [custom FITS](custom.md) and [troubleshooting](troubleshooting.md) for the next specific task.
