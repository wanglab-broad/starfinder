"""Command line adapters for synthetic generation and benchmark lifecycle."""
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
    generate.add_argument('--preset', required=True)
    generate.add_argument('--output', type=Path, required=True, help='New directory; existing paths are rejected')
    generate.add_argument('--seed', type=int, required=True)
    generate.add_argument('--no-noise', action='store_true')
    generate.add_argument('--dtype', choices=['uint8', 'uint16'], default='uint8', help='E2E image dtype; registration currently requires uint8')
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
    args = parser.parse_args(argv)
    try:
        from starfinder.benchmark._storage import _read, _write, _reference, SCHEMA_VERSION
        if args.group == 'synthetic':
            from starfinder.synthetic import get_preset_config, generate_dataset, generate_registration_pairs
            from starfinder.benchmark._synthetic_io import _write_dataset, _write_registration_pairs
            if not args.owner.strip():
                raise ValueError('owner must be nonempty')
            config = get_preset_config(args.preset)
            if args.mode == 'registration' and args.dtype != 'uint8':
                raise ValueError('registration generation supports uint8 only')
            args.output.mkdir(parents=True, exist_ok=False)
            if args.mode == 'e2e':
                config.seed, config.add_noise, config.dtype = args.seed, not args.no_noise, args.dtype
                _write_dataset(generate_dataset(config=config, preset=args.preset), args.output, annotations=False)
            else:
                _write_registration_pairs(generate_registration_pairs(presets=[args.preset], seed=args.seed,
                    add_noise=not args.no_noise), args.output, inspections=False)
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


if __name__ == '__main__':
    sys.exit(main())
