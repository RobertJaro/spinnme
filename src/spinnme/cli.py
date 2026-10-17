"""Unified, lazily imported command line interface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import __version__


def _config_options(parser):
    parser.add_argument("--config", required=True, help="YAML path or bundled name (spinnme config list)")
    parser.add_argument("--input", help="Files, directory, or quoted glob; binds the first and additional unset training inputs, plus validation")
    parser.add_argument("--run_path", "--run-path", "--output", "--out", dest="output", help="Overwrite <<run_path>> / path")
    parser.add_argument("--work_path", "--work-path", "--work-directory", dest="work_path", help="Overwrite <<work_path>> / work_path")
    parser.add_argument("--instrument", help="YAML instrument definition or bundled hmi/hinode/phi_hrt/phi_fdt name")
    parser.add_argument("--reload", action="store_true", help="Rebuild prepared caches from the current inputs")


def _selection_options(parser):
    parser.add_argument("snapshot", help="Versioned SPINNME .pme file")
    parser.add_argument("--observation", help="Saved grid ID or reference solar FITS image")
    parser.add_argument("--instrument", dest="instrument_id", help="Instrument ID for observer-dependent quantities")
    parser.add_argument("--parameters", nargs="+", help="Explicit parameter keys, e.g. Brtp Blos Vdop vmac damping; default Brtp")
    parser.add_argument("--time", help="Model evaluation epoch in UTC; retains selected grid geometry")
    parser.add_argument("--device", default=None, help="Optional cpu/cuda/cuda:N override; default auto prefers CUDA, otherwise CPU")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--mu-limit", type=float, default=0.)
    parser.add_argument('--shape', nargs=2, type=int, metavar=('NY', 'NX'), help='Output map dimensions')
    parser.add_argument('--bounds', nargs=4, type=float, metavar=('XMIN', 'YMIN', 'XMAX', 'YMAX'),
                        help='Helioprojective subframe in arcsec')


def build_parser():
    parser = argparse.ArgumentParser(prog="spinnme", description="Spherical neural Milne–Eddington inversions of solar Stokes FITS data.")
    parser.add_argument("--version", action="version", version=f"spinnme {__version__}")
    parser.add_argument("--debug", action="store_true", help="Show Python tracebacks on command failure")
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Prepare and train a finite spherical inversion")
    _config_options(run)
    run.add_argument("--device", choices=["cpu", "cuda", "auto"],
                     help="Optional training override; default detects available accelerators")
    run.add_argument("--resume", action="store_true", help="Resume last.ckpt explicitly")
    run.add_argument("--profile", action="store_true")
    run.add_argument("--export", choices=["fits", "npz", "jpg"], help="Export all saved native grids after training")
    prep = commands.add_parser("prep", help="Validate inputs and create reusable training caches")
    _config_options(prep)
    export = commands.add_parser("export", help="Export snapshot maps through the shared loader")
    _selection_options(export)
    export.add_argument("--format", choices=["fits", "npz", "jpg"], default="fits")
    export.add_argument("--output", "--out", dest="output", required=True)
    export.add_argument("--all-observations", action="store_true")
    export.add_argument("--overwrite", action="store_true")
    evaluate = commands.add_parser("evaluate", help="Compute summary statistics or reference residual metrics")
    _selection_options(evaluate)
    evaluate.add_argument("--reference", help="NPZ export or scalar FITS map for comparison")
    evaluate.add_argument("--output", "--out", dest="output", required=True, help="Output JSON or directory")
    config = commands.add_parser("config", help="List, show, copy, or validate configuration templates")
    actions = config.add_subparsers(dest="action", required=True)
    actions.add_parser("list")
    for action in ("show", "copy", "validate", "instrument"):
        sub = actions.add_parser(action)
        sub.add_argument("source")
        sub.add_argument("--output")
        sub.add_argument("--overwrite", action="store_true", help="Replace an existing output configuration")
        if action != "instrument":
            sub.add_argument("--set", dest="overrides", action="append", default=[])
    inspect = commands.add_parser("inspect", help="Inspect snapshot grids or FITS metadata")
    inspect.add_argument("path")
    download = commands.add_parser("download", help="Download archive products with a checksum manifest")
    download.add_argument("source", choices=["hmi", "hmi-transmission", "phi", "hinode"])
    download.add_argument("--output", "--out", dest="output", required=True)
    download.add_argument("--input", nargs="+", help="Downloaded Stokes directories/files/globs for hmi-transmission")
    download.add_argument("--overwrite", action="store_true", help="Regenerate hmi-transmission profile files")
    download.add_argument("--start")
    download.add_argument("--end")
    download.add_argument("--email")
    download.add_argument("--product")
    download.add_argument("--segments", nargs="+", help="HMI segment names, such as field inclination azimuth")
    download.add_argument("--cadence", help="Archive sampling interval, e.g. 90s or 720s; inferred from HMI product if omitted")
    download.add_argument("--telescope", choices=["hrt", "fdt"], default="hrt")
    download.add_argument("--time-scale", choices=["utc", "tai"], default="utc")
    download.add_argument("--url", action="append", default=[], help="Explicit public Hinode FITS URL; repeat for a raster")
    crop = commands.add_parser("crop", help="Crop FITS images and cubes while preserving spatial WCS and aligned extensions")
    crop.add_argument("--input", nargs="+", required=True, help="FITS files, directories or quoted globs")
    crop.add_argument("--output", required=True, help="New or empty directory for cropped FITS files")
    crop.add_argument("--spatial-axes", nargs=2, type=int, default=[1, 2], metavar=("X", "Y"),
                      help="One-based FITS spatial axes in x,y order (default: 1 2)")
    crop.add_argument("--hdu", help="Science image HDU index or name; default is the first image")
    crop.add_argument("--unit", default="arcsec", help="Unit of --bounds in the input WCS (default: arcsec)")
    region = crop.add_mutually_exclusive_group(required=True)
    region.add_argument("--pixels", nargs=4, type=int, metavar=("X0", "Y0", "X1", "Y1"),
                        help="Zero-based input-pixel bounds; upper bounds excluded")
    region.add_argument("--bounds", nargs=4, type=float, metavar=("XMIN", "YMIN", "XMAX", "YMAX"),
                        help="World-coordinate bounds in --unit in each input WCS")
    synthetic = commands.add_parser("synthetic", help="Create circular-pattern solar FITS data and optional cadence/noise acquisitions")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument('--config', help='Dataset YAML with generation/acquisition settings; writes dataset.json and acquisitions.tsv')
    synthetic.add_argument("--size", type=int, help='Square image size for the simple example (default: 32)')
    synthetic.add_argument("--frames", type=int, help='Frame count for the simple example (default: 2)')
    return parser


def _resolved(args):
    from .configuration import resolve_config
    overrides = []
    if getattr(args, "device", None) is not None:
        overrides.extend([f"training.accelerator={'gpu' if args.device == 'cuda' else args.device}", "training.devices=1"])
    return resolve_config(args.config, overrides=overrides, input_path=args.input, output=args.output,
                          work_path=args.work_path, instrument=args.instrument, bindings=getattr(args, "bindings", None))


def _result(snapshot, args, observation=None):
    return snapshot.evaluate(observation=observation if observation is not None else args.observation,
                             parameters=args.parameters, instrument_id=args.instrument_id, time=args.time,
                             batch_size=args.batch_size, mu_limit=args.mu_limit,
                             shape=args.shape, bounds=args.bounds)


def _run(args):
    from .training.runner import run
    config = _resolved(args)
    output = Path(config["path"])
    if (output / "inversion.pme").exists() and not args.resume:
        raise FileExistsError(f"{output}/inversion.pme exists; choose a new output or --resume.")
    run(config, resume=args.resume, profile=args.profile, reload=args.reload)
    if args.export:
        from . import load
        device = args.device if args.device in {"cpu", "cuda"} else None
        with load(output / "inversion.pme", device=device) as snapshot:
            for observation in snapshot.observations:
                snapshot.evaluate(observation["id"]).write(output / "exports" / observation["id"],
                    format=args.export, overwrite=args.resume)
    print(output / "inversion.pme")


def _prep(args):
    """Optional cache-only preprocessing of the selected run YAML."""
    from .configuration import config_hash
    config = _resolved(args)
    from .data.cache import load_or_prepare_data_module
    module = load_or_prepare_data_module(config['data'], config['work_path'], reload=args.reload)
    output = Path(config["path"])
    observations = module.get_observations()
    from .snapshots import encode
    report = {"config_hash": config_hash(config), "observations": encode(observations),
              "training_datasets": len(module.train_datasets), "cache_directory": config["work_path"],
              "cache_reused": module.cache_reused, "cache_manifest": str(module.cache_manifest)}
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "preparation.json"
    report_path.write_text(json.dumps(report, indent=2))
    print(report_path)


def _export(args):
    from . import load
    with load(args.snapshot, device=args.device) as snapshot:
        if args.all_observations and args.observation:
            raise ValueError("Select --observation or --all-observations.")
        observations = [row["id"] for row in snapshot.observations] if args.all_observations else [args.observation]
        if not observations:
            raise ValueError("This snapshot has no saved observation grids; supply --observation with a reference FITS.")
        for observation in observations:
            output = Path(args.output) / observation if args.all_observations else Path(args.output)
            paths = _result(snapshot, args, observation).write(output, format=args.format, overwrite=args.overwrite)
            for path in paths:
                print(path)


def _evaluate(args):
    from . import load
    from .evaluation import evaluate_result, read_reference, write_metrics
    with load(args.snapshot, device=args.device) as snapshot:
        result = _result(snapshot, args)
        reference = read_reference(args.reference, result) if args.reference else None
        metrics = evaluate_result(result, reference)
        for path in write_metrics(metrics, args.output):
            print(path)


def _config(args):
    from .configuration import dump_config, list_configs, resolve_config, write_config
    if args.action == "list":
        print("\n".join(list_configs()))
        return
    if args.output and Path(args.output).exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; choose a new output or --overwrite.")
    if args.action == 'instrument':
        from .instruments import load_instrument_definition
        definition = load_instrument_definition(args.source)
        if args.output:
            print(write_config(definition, args.output))
        else:
            print(dump_config(definition), end='')
        return
    configuration = resolve_config(args.source, overrides=args.overrides, bindings=getattr(args, "bindings", None), require_input=False)
    if args.output:
        print(write_config(configuration, args.output))
    elif args.action == "validate":
        print("Configuration is valid; dataset paths and instrument metadata are checked during prep.")
    else:
        print(dump_config(configuration), end="")


def _inspect(args):
    path = Path(args.path)
    if path.suffix.lower() == ".pme":
        from . import load
        from .snapshots import encode
        with load(path, device="cpu") as snapshot:
            print(json.dumps(encode({"instruments": snapshot.instrument_ids, "observations": snapshot.observations,
                                     "config_hash": snapshot.metadata["config_hash"], "conventions": snapshot.metadata["conventions"]}), indent=2))
    else:
        from astropy.io import fits
        with fits.open(path) as hdus:
            print(json.dumps([{"hdu": i, "name": h.name, "shape": None if h.data is None else list(h.data.shape),
                               "header": dict(h.header)} for i, h in enumerate(hdus)], default=str, indent=2))


def _download(args):
    if args.source == 'hmi-transmission':
        if not args.input or not args.email:
            raise ValueError('download hmi-transmission requires --input and --email.')
        if args.segments:
            raise ValueError('--segments is supported for HMI data downloads only.')
        from .data.hmi_transmission import prepare_hmi_transmission_directory
        print(prepare_hmi_transmission_directory(args.input, args.output, email=args.email,
                                                 overwrite=args.overwrite))
        return
    if args.input or args.overwrite:
        raise ValueError('--input and --overwrite are only supported for download hmi-transmission.')
    from .data.download import download
    if args.segments and args.source != "hmi":
        raise ValueError("--segments is supported for HMI downloads only.")
    path = download(args.source, output=args.output, start=args.start, end=args.end, email=args.email,
                    product=args.product, telescope=args.telescope, time_scale=args.time_scale, urls=args.url,
                    segments=args.segments,
                    cadence=args.cadence or ("90s" if args.product and "90s" in args.product else "720s"))
    print(path)


def _crop(args):
    from .data.crop import crop_fits
    hdu = args.hdu
    if hdu is not None:
        try:
            hdu = int(hdu)
        except ValueError:
            pass
    for path in crop_fits(args.input, args.output, pixels=args.pixels, bounds=args.bounds,
                          spatial_axes=args.spatial_axes, hdu=hdu, unit=args.unit):
        print(path)


def _synthetic(args):
    if args.config:
        if args.size is not None or args.frames is not None:
            raise ValueError('Use --config or the simple --size/--frames options, not both.')
        import yaml
        from .data.synthetic_sequence import generate_dataset
        try:
            specification = yaml.safe_load(Path(args.config).expanduser().read_text())
        except yaml.YAMLError as error:
            raise ValueError(f'Invalid synthetic dataset YAML: {error}') from error
        manifest = generate_dataset(specification, args.output)
        print(manifest)
        print(manifest.parent / 'acquisitions.tsv')
    else:
        from .data.circular import create_circular_dataset
        print(create_circular_dataset(args.output, size=args.size if args.size is not None else 32,
                                      frames=args.frames if args.frames is not None else 2))


def main(argv=None):
    parser = build_parser()
    args, extra = parser.parse_known_args(argv)
    functions = {"run": _run, "prep": _prep, "export": _export, "evaluate": _evaluate, "config": _config,
                 "inspect": _inspect, "download": _download, "crop": _crop, "synthetic": _synthetic}
    try:
        if extra:
            source = getattr(args, 'config', None) or getattr(args, 'source', None)
            if source is None or args.command not in ('run', 'prep', 'config'):
                parser.error('unrecognized arguments: '+' '.join(extra))
            from .configuration import read_config, placeholder_keys
            keys = placeholder_keys(read_config(source))
            overwrite_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
            for key in sorted(keys):
                overwrite_parser.add_argument('--'+key)
            import yaml
            def typed(value):
                try:
                    return yaml.safe_load(value)
                except yaml.YAMLError:
                    return value
            args.bindings = {key: typed(value) for key, value in vars(overwrite_parser.parse_args(extra)).items() if value is not None}
        functions[args.command](args)
    except (ValueError, KeyError, OSError, ImportError, RuntimeError) as error:
        if args.debug:
            raise
        parser.exit(2, f"spinnme: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
