"""Command line adapters for synthetic generation, benchmark lifecycle and pretrained weights."""
import argparse
from pathlib import Path
import sys


def main(argv=None):
    """Run a subcommand; usage/input errors exit 2, stage failures exit 1."""
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
    benchmark = groups.add_parser('benchmark', help='Run, reevaluate, or report saved trials')
    commands = benchmark.add_subparsers(dest='command', required=True)
    run = commands.add_parser('run')
    run.add_argument('--config', type=Path, required=True, help='JSON: schema_version, cases, repetitions, optional provenance')
    run.add_argument('--input-root', type=Path, required=True)
    run.add_argument('--output-root', type=Path, required=True)
    run.add_argument('--owner', required=True)
    run.add_argument('--run-id')
    run.add_argument('--resume', action='store_true', help='Verify identity and skip completed trials; no overwrite')
    evaluate = commands.add_parser('evaluate')
    evaluate.add_argument('--run-dir', type=Path, required=True)
    report = commands.add_parser('report')
    report.add_argument('--evaluation-dir', type=Path, required=True)
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
    args = parser.parse_args(argv)
    try:
        if args.group == 'weights':
            return _weights(args)
        from starfinder.benchmark._storage import _read, _write, _reference, SCHEMA_VERSION
        if args.group == 'synthetic':
            from starfinder.synthetic import BENCHMARK_PRESETS
            from starfinder.benchmark._synthetic_io import _write_dataset, _write_registration_pairs
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
            _write(args.output / 'manifest.json', {'schema_version': SCHEMA_VERSION,
                'owner': args.owner, 'command': vars(args) | {'output': str(args.output)},
                'artifacts': [_reference(args.output, p) for p in sorted(args.output.rglob('*')) if p.is_file()],
                'backup_status': 'unverified', 'retention': 'owner decision; retain through handoff'})
            print(args.output)
            return 0
        from starfinder.benchmark import BenchmarkCase, run_benchmark, evaluate_benchmark, report_benchmark
        if args.command == 'run':
            config = _read(args.config)
            if config.get('schema_version') != SCHEMA_VERSION or set(config) - {'schema_version', 'cases', 'repetitions', 'provenance'}:
                raise ValueError('unsupported benchmark configuration schema or fields')
            path = run_benchmark([BenchmarkCase.from_dict(c) for c in config['cases']],
                input_root=args.input_root, output_root=args.output_root, owner=args.owner,
                repetitions=config['repetitions'], provenance=config.get('provenance'),
                run_id=args.run_id, resume=args.resume)
            from starfinder.benchmark._lifecycle import _load_run, _trials
            root, manifest = _load_run(path)
            failed = any(t.status['processing'] == 'failed' for t in _trials(root, manifest))
        elif args.command == 'evaluate':
            path = evaluate_benchmark(args.run_dir)
            failed = any(t['status']['evaluation'] in ('failed', 'skipped') for t in _read(path / 'results.json'))
        else:
            path = report_benchmark(args.evaluation_dir)
            failed = False
        print(path)
        return int(failed)
    except (OSError, ValueError, TypeError, KeyError) as exc:
        parser.error(str(exc))


def _weights(args):
    """starfinder weights fetch|list|verify; verify exits 1 when a local copy is missing or changed.

    The known detector weights (KNOWN_WEIGHTS, §2.7) and the known segmentation models
    (KNOWN_MODELS, §2.9) share the cache and the commands.
    """
    from starfinder.segmentation import KNOWN_MODELS, MissingModelError, ModelHashMismatchError, resolve_model
    from starfinder.segmentation._models import RECORD_NAME as MODEL_RECORD, _fetch_model
    from starfinder.spot_finding import KNOWN_WEIGHTS, MissingWeightsError, WeightsHashMismatchError
    from starfinder.spot_finding import fetch_weights, resolve_weights
    from starfinder.spot_finding._weights import RECORD_NAME, listed_files, model_folder, weights_directory
    if args.command == 'fetch':
        if (args.method, args.model) in KNOWN_MODELS:
            print(_fetch_model(args.method, args.model, directory=args.dir))
        else:
            print(fetch_weights(args.method, args.model, directory=args.dir))
        return 0
    root = weights_directory(args.dir)
    if args.command == 'list':
        print(f'weights directory: {root}')
        for (method, model), entry in KNOWN_WEIGHTS.items():
            folder = model_folder(method, model, args.dir)
            state = ('fetched' if (folder / RECORD_NAME).is_file() else 'incomplete' if folder.exists()
                     else 'not fetched')
            print(f'{method}\t{model}\t{entry.dimensionality}\t{entry.bytes} bytes\tsha256 {entry.sha256}\t'
                  f'{entry.revision}\t{state}')
        for (method, model), entry in KNOWN_MODELS.items():
            folder = root / method / model
            present = all((folder / item.path).is_file() for item in entry.files)
            state = ('fetched' if (folder / MODEL_RECORD).is_file() else 'present' if present
                     else 'incomplete' if folder.exists() else 'not fetched')
            print(f'{method}\t{model}\t{entry.dimensionality}\t{entry.bytes} bytes\tsha256 {entry.sha256}\t'
                  f'{entry.url}\t{state}')
        return 0
    if (args.method is None) != (args.model is None):
        raise ValueError('verify takes a method and a model, or neither')
    if args.method is not None:
        keys = [(args.method, args.model)]
    else:
        keys = ([key for key in KNOWN_WEIGHTS if model_folder(*key, args.dir).exists()]
                + [key for key in KNOWN_MODELS if (root / key[0] / key[1]).exists()])
    if not keys:
        print(f'no local weights in {root}')
    failed = False
    for method, model in keys:
        try:
            if (method, model) in KNOWN_MODELS:
                resolve_model(method, model=model, directory=args.dir)
                folder = root / method / model
            else:
                # Every file KNOWN_WEIGHTS lists for the model is re-hashed, as every detection does.
                folder = resolve_weights(method, model, directory=args.dir, extracted=listed_files(method, model))
            print(f'{method}\t{model}\tverified\t{folder}')
        except (MissingWeightsError, WeightsHashMismatchError, MissingModelError, ModelHashMismatchError) as error:
            failed = True
            print(f'{method}\t{model}\tfailed\t{error}')
    return int(failed)


if __name__ == '__main__':
    sys.exit(main())
