"""Command line adapters for synthetic generation, experiments and trials, pretrained weights, the preview of
segmentation input recipes and the workflow."""
import argparse
from pathlib import Path
import sys


def _parser():
    """The ``starfinder`` argument parser with every command group (the docs' command tables read it)."""
    parser = argparse.ArgumentParser(prog='starfinder')
    groups = parser.add_subparsers(dest='group', required=True)
    synthetic = groups.add_parser('synthetic', help='Processed-image synthetic generation')
    generate = synthetic.add_subparsers(dest='command', required=True).add_parser('generate')
    generate.add_argument('--mode', choices=['e2e', 'registration'], required=True)
    generate.add_argument('--preset', required=True, help='tiny, small, medium, large, tissue or thick_medium')
    generate.add_argument('--output', type=Path, required=True, help='New directory; existing paths are rejected')
    generate.add_argument('--seed', type=int, required=True)
    generate.add_argument('--no-noise', action='store_true', help='Disable Poisson and read noise')
    generate.add_argument('--dtype', choices=['uint8', 'uint16'], default='uint16',
                          help='Image dtype (default uint16); uint8 scales intensities by 1/16')
    generate.add_argument('--owner', required=True)
    experiment = groups.add_parser('experiment', help='Expand an experiment into trial configurations, evaluate '
                                   'saved workflow, checkpoint and trial outputs, or report a saved evaluation')
    experiment_commands = experiment.add_subparsers(dest='command', required=True)
    expand = experiment_commands.add_parser(
        'expand', help='Write one trial configuration per method, condition, seed and input, and manifest.json',
        description='Resolve every trial (defaults, input SHA-256) before writing; no processing runs. Exit 0 '
                    'with the manifest written, 2 for an invalid experiment or an output folder that is not new or '
                    'empty.')
    expand.add_argument('--config', type=Path, required=True,
                        help='Experiment mapping (YAML or JSON): methods, conditions, seeds, inputs, optional metadata')
    expand.add_argument('--output', type=Path, required=True,
                        help='New or empty folder for trial-NNNNNN.json and manifest.json; a nonempty one is refused')
    evaluate_saved = experiment_commands.add_parser(
        'evaluate', help='Compute selected starfinder.evaluation metrics from saved sources, without processing',
        description='Read workflow outputs, checkpoints and trial outputs, verify every consumed file against its '
                    'SHA-256 and write evaluation.json, values.csv and metrics/. Exit 0 when every source is verified '
                    'and every metric computed, 1 when a source failed or a metric was skipped, unavailable or '
                    'failed (the evaluation is still written), 2 for invalid sources or metrics files or an output '
                    'folder that is not new or empty, or lies inside a source.')
    evaluate_saved.add_argument('--sources', type=Path, required=True,
                                help='Sources mapping (YAML or JSON): workflow, checkpoint, trial and files sources')
    evaluate_saved.add_argument('--metrics', type=Path, required=True,
                                help='Metrics mapping (YAML or JSON): name, metric, function, parameters, arguments')
    evaluate_saved.add_argument('--output', type=Path, required=True,
                                help='New or empty folder outside every source')
    report_saved = experiment_commands.add_parser(
        'report', help='Render a saved evaluation as a review report, without processing',
        description='Read evaluation.json, values.csv and metrics/ of experiment evaluate (they have no recorded '
                    'SHA-256; values.csv must equal evaluation.json) and the saved records and figure files they '
                    'name, each checked against its recorded SHA-256, and write report.html, report.json, tables/ '
                    'and figures/. Exit 0 with the report written, 2 for a saved file or revision that does not '
                    'check, an invalid protocol, or an output folder that is not new or empty or lies inside the '
                    'evaluation or a source (nothing is written).')
    report_saved.add_argument('--evaluation', type=Path, required=True,
                              help='The folder experiment evaluate wrote')
    report_saved.add_argument('--output', type=Path, required=True,
                              help='New or empty folder outside the evaluation and every source')
    report_saved.add_argument('--protocol', type=Path,
                              help='Protocol mapping (YAML or JSON) of supplied text: purposes, conditions, cut-offs, '
                                   'rows, cards, verdicts, findings and figures')
    trial = groups.add_parser('trial', help='Run one method on one input in this process')
    trial_run = trial.add_subparsers(dest='command', required=True).add_parser(
        'run', help='Run one trial configuration and write its outputs and record.json',
        description='One method in this process, with no scheduler and no retry. Exit 0 when the trial succeeded, '
                    '1 when the method or an input failed (record.json says failed), 2 for an unreadable '
                    'configuration or an output folder that is not empty.')
    trial_run.add_argument('--config', type=Path, required=True,
                           help='Trial mapping (YAML or JSON), such as a trial-NNNNNN.json of experiment expand; its '
                                'root_output_path must be new or empty')
    weights = groups.add_parser('weights', help='Fetch, list and verify the known pretrained weights and '
                                'segmentation models')
    weight_commands = weights.add_subparsers(dest='command', required=True)
    fetch = weight_commands.add_parser('fetch', help='Download, verify and install one known model (uses the network)')
    fetch.add_argument('method')
    fetch.add_argument('model')
    listing = weight_commands.add_parser('list', help='Print the known-weights table and the local state')
    verify = weight_commands.add_parser('verify', help='Re-hash the local copies (all, or one method and model)')
    verify.add_argument('method', nargs='?')
    verify.add_argument('model', nargs='?')
    for command in (fetch, listing, verify):
        command.add_argument('--dir', type=Path, help='Weights cache; default STARFINDER_WEIGHTS_DIR, '
                             'else $XDG_CACHE_HOME/starfinder/weights (~/.cache/starfinder/weights)')
    segment = groups.add_parser('segment', help='Preview segmentation input recipes before a workflow run')
    preview = segment.add_subparsers(dest='command', required=True).add_parser(
        'preview', help='Segment a few FOVs with a few input recipes and draw them (writes only under --output)',
        description='One run of the workflow configuration, its recipe replaced by each of --recipes, on each of '
                    '--fovs: overlay.png, sizes.png, sizes.csv, summary.csv and preview.json under --output. It reads '
                    'the files the workflow would read and runs no preprocessing, registration, detection or '
                    'assignment. Exit 0 with the preview written, 2 for an invalid configuration, run, recipe or '
                    'source, or an output folder that is not absent or empty.')
    preview.add_argument('config', type=Path, help='The workflow configuration (YAML), with its roots')
    preview.add_argument('--run', required=True,
                         help='A run of segmentation.runs (with --factor-yx, of export.segmentation.runs when it has it)')
    preview.add_argument('--fovs', required=True, help='At most 8 FOV ids, comma-separated')
    preview.add_argument('--recipes', required=True,
                         help='At most 8 recipe names of segmentation.input_recipes or of --recipe-file, comma-separated')
    preview.add_argument('--output', type=Path, required=True, help='Absent or empty folder for the preview')
    preview.add_argument('--recipe-file', type=Path,
                         help='YAML mapping of recipe names to recipes, in the layout of segmentation.input_recipes')
    preview.add_argument('--factor-yx', type=int,
                         help='Preview at the pixel size of the route after fusion with fusion factor K (one view per '
                              'FOV, no blend)')
    preview.add_argument('--crop-um', help='Y,X,H,W in micrometres: the window the panels show')
    preview.add_argument('--device', choices=['cpu', 'cuda'], default='cpu', help='Device of the method (default cpu)')
    preview.add_argument('--save-inputs', action='store_true',
                         help='Also write each model input as inputs/<fov>/<recipe>.ome.tif')
    from starfinder.workflow._cli import add_parser
    add_parser(groups)
    return parser


def main(argv=None):
    """Run a subcommand; usage/input errors exit 2, stage failures exit 1 (the workflow commands also exit 3 for a
    refused verified resume)."""
    from starfinder.workflow._cli import run as run_workflow
    parser = _parser()
    args = parser.parse_args(argv)
    if args.group == 'workflow':
        # §2.13: the task commands map their own outcomes to exit 0, 1, 2 and 3.
        return run_workflow(args, list(sys.argv[1:] if argv is None else argv)[1:])
    try:
        if args.group == 'weights':
            return _weights(args)
        if args.group == 'segment':
            return _segment_preview(args)
        if args.group in ('experiment', 'trial'):
            return _experiment(args)
        from starfinder.synthetic import BENCHMARK_PRESETS
        from starfinder.synthetic._storage import (MANIFEST_SCHEMA_VERSION, _reference, _write, _write_dataset,
                                                   _write_registration_pairs)
        if not args.owner.strip():
            raise ValueError('owner must be nonempty')
        if args.preset not in BENCHMARK_PRESETS:
            raise ValueError(f'unknown preset {args.preset!r}; choose from {list(BENCHMARK_PRESETS)}')
        args.output.mkdir(parents=True, exist_ok=False)
        options = dict(seed=args.seed, dtype=args.dtype, noise=not args.no_noise)
        if args.mode == 'e2e':
            _write_dataset(args.preset, args.output, **options)
        else:
            _write_registration_pairs(args.preset, args.output, **options)
        _write(args.output / 'manifest.json', {'schema_version': MANIFEST_SCHEMA_VERSION,
            'owner': args.owner, 'command': vars(args) | {'output': str(args.output)},
            'artifacts': [_reference(args.output, p) for p in sorted(args.output.rglob('*')) if p.is_file()],
            'backup_status': 'unverified', 'retention': 'owner decision; retain through handoff'})
        print(args.output)
        return 0
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


def _experiment(args):
    """starfinder experiment expand|evaluate|report and starfinder trial run, over the private benchmark entries."""
    # §2.14
    import yaml
    try:
        if args.group == 'trial':
            from starfinder.benchmark._trial import _main, _read
            status = _main(args.config)
            print((args.config.resolve().parent / _read(args.config)['root_output_path']).resolve() / 'record.json')
            return status
        if args.command == 'expand':
            from starfinder.benchmark._experiment import expand_experiment
            from starfinder._records import file_sha256
            if args.output.exists() and (not args.output.is_dir() or any(args.output.iterdir())):
                raise FileExistsError(f'the expansion output must be a new or empty folder: {args.output}')
            manifest = expand_experiment(args.config, args.output)
            # The record as written: its format and each trial file's SHA-256.
            if manifest.get('format') != 'starfinder.benchmark.experiment/1' or any(
                    'sha256:' + file_sha256(args.output / entry['path']) != entry['sha256']
                    for entry in manifest['trials']):
                raise ValueError(f'the expansion in {args.output} does not match its manifest')
            print(args.output.resolve() / 'manifest.json')
            return 0
        if args.command == 'report':
            from starfinder.benchmark._report import render
            print(render(args.evaluation, args.output, args.protocol) / 'report.html')
            return 0
        from starfinder.benchmark._evaluate import evaluate
        path, failed = evaluate(args.sources, args.metrics, args.output)
        print(path / 'evaluation.json')
        return int(failed)
    except yaml.YAMLError as exc:
        raise ValueError(f'{args.config}: not YAML or JSON: {exc}') from None


def _segment_preview(args):
    """starfinder segment preview: preview_recipes with the command's options; prints the output folder."""
    # W-891
    from starfinder.segmentation import preview_recipes

    def names(value):
        return [item.strip() for item in value.split(',') if item.strip()]
    crop = None if args.crop_um is None else tuple(float(v) for v in args.crop_um.split(','))
    preview_recipes(args.config, run=args.run, fovs=names(args.fovs), recipes=names(args.recipes), output=args.output,
                    recipe_file=args.recipe_file, factor_yx=args.factor_yx, crop_um=crop, device=args.device,
                    save_inputs=args.save_inputs)
    print(args.output.resolve())
    return 0


def _weights(args):
    """starfinder weights fetch|list|verify; verify exits 1 when a local copy is missing or changed.

    The known detector models (spot_finding.KNOWN_MODELS) and the known segmentation
    models (segmentation.KNOWN_MODELS) share the cache and the commands. Both stages
    name their members alike, so each is used through its module.
    """
    # §2.7, §2.9
    from starfinder import segmentation, spot_finding
    from starfinder._weights import RECORD_NAME, model_folder, weights_directory
    from starfinder.segmentation._models import _fetch_model
    from starfinder.spot_finding._weights import listed_files
    if args.command == 'fetch':
        if (args.method, args.model) in segmentation.KNOWN_MODELS:
            print(_fetch_model(args.method, args.model, directory=args.dir))
        else:
            print(spot_finding.fetch_model(args.method, args.model, directory=args.dir))
        return 0
    root = weights_directory(args.dir)
    if args.command == 'list':
        print(f'weights directory: {root}')
        for (method, model), entry in spot_finding.KNOWN_MODELS.items():
            folder = model_folder(method, model, args.dir)
            state = ('fetched' if (folder / RECORD_NAME).is_file() else 'incomplete' if folder.exists()
                     else 'not fetched')
            print(f'{method}\t{model}\t{entry.dimensionality}\t{entry.bytes} bytes\tsha256 {entry.sha256}\t'
                  f'{entry.revision}\t{state}')
        for (method, model), entry in segmentation.KNOWN_MODELS.items():
            folder = root / method / model
            present = all((folder / item.path).is_file() for item in entry.files)
            state = ('fetched' if (folder / RECORD_NAME).is_file() else 'present' if present
                     else 'incomplete' if folder.exists() else 'not fetched')
            print(f'{method}\t{model}\t{entry.dimensionality}\t{entry.bytes} bytes\tsha256 {entry.sha256}\t'
                  f'{entry.url}\t{state}')
        return 0
    if (args.method is None) != (args.model is None):
        raise ValueError('verify takes a method and a model, or neither')
    if args.method is not None:
        keys = [(args.method, args.model)]
    else:
        keys = ([key for key in spot_finding.KNOWN_MODELS if model_folder(*key, args.dir).exists()]
                + [key for key in segmentation.KNOWN_MODELS if (root / key[0] / key[1]).exists()])
    if not keys:
        print(f'no local weights in {root}')
    failed = False
    for method, model in keys:
        try:
            if (method, model) in segmentation.KNOWN_MODELS:
                segmentation.resolve_model(method, model=model, directory=args.dir)
                folder = root / method / model
            else:
                # Every file spot_finding.KNOWN_MODELS lists for the model is re-hashed, as every detection does.
                folder = spot_finding.resolve_model(method, model, directory=args.dir,
                                                    extracted=listed_files(method, model))
            print(f'{method}\t{model}\tverified\t{folder}')
        except (spot_finding.MissingModelError, spot_finding.ModelHashMismatchError,
                segmentation.MissingModelError, segmentation.ModelHashMismatchError) as error:
            failed = True
            print(f'{method}\t{model}\tfailed\t{error}')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
