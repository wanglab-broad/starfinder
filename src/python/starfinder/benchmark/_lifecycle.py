"""Unique, append-only benchmark runs and saved-only evaluation/reporting."""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import csv
import os
import platform
import re
import subprocess
import sys
import time
import tracemalloc
import uuid

from ._records import BenchmarkCase, BenchmarkTrialResult, _json
from ._storage import (_write, _read, _identity, _digest, _inside, _reference, _verified,
                       _array, _save_array, _save_transform, SCHEMA_VERSION)
from ._adapters import _validate, _process, _evaluate


def _id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value):
        raise ValueError('IDs must be nonempty safe filename components')
    return value


def _unique():
    return datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ-') + uuid.uuid4().hex


def _error(exc):
    return {'type': type(exc).__name__, 'message': str(exc)}


def _git():
    def command(*args):
        result = subprocess.run(['git', *args], capture_output=True, text=True)
        return result.stdout.strip() if result.returncode == 0 else None
    return {'commit': command('rev-parse', 'HEAD'), 'dirty_status': command('status', '--porcelain'),
            'diff_sha256': _identity(command('diff', 'HEAD'))}


def _load_run(run_dir):
    root = Path(run_dir).resolve()
    manifest = _read(root / 'manifest.json')
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported benchmark manifest schema')
    if manifest['identity_sha256'] != _identity(manifest['identity']):
        raise ValueError('manifest identity checksum mismatch')
    return root, manifest


def _trials(root, manifest):
    trials = []
    for slot in manifest['trials']:
        path = _inside(root, slot['record'])
        if not path.is_file():
            raise FileNotFoundError(f'trial not complete: {path}; resume processing explicitly')
        expected = _read(path.with_suffix('.checksum.json'))
        _verified(root, expected)
        trial = BenchmarkTrialResult.from_dict(_read(path))
        if (trial.run_id, trial.case_id, trial.trial_id) != (manifest['run_id'], slot['case_id'], slot['trial_id']):
            raise ValueError('trial identity mismatch')
        for ref in trial.artifacts.values():
            _verified(root, ref)
        trials.append(trial)
    return trials


def run_benchmark(cases, *, input_root, output_root, owner, repetitions=1,
                  run_id=None, resume=False, provenance=None):
    """Process explicit cases into a unique run directory and return its Path.

    Resume requires the original run_id and identical schema, config, roots,
    owner, provenance and input hashes. Completed records (including failures)
    are verified and skipped. An interrupted directory without a record blocks
    resume; start a new run to retry. No overwrite option or implicit warmup.
    """
    cases = list(cases)
    if not cases or len({c.case_id for c in cases}) != len(cases):
        raise ValueError('cases must be nonempty with unique case IDs')
    if type(repetitions) is not int or repetitions < 1 or not isinstance(owner, str) or not owner.strip():
        raise ValueError('positive repetitions and explicit owner required')
    inputs = Path(input_root).resolve(strict=True)
    outputs = Path(output_root).resolve()
    if outputs == inputs or outputs.is_relative_to(inputs) or inputs.is_relative_to(outputs):
        raise ValueError('input and output roots must be disjoint')
    references = {}
    for case in cases:
        _id(case.case_id)
        _validate(case)
        references[case.case_id] = {}
        for category in ('inputs', 'truth', 'artifacts'):
            references[case.case_id][category] = {}
            for name, filename in getattr(case, category).items():
                _id(name)
                path = _inside(inputs, filename)
                if not path.is_file():
                    raise FileNotFoundError(f'required input missing: {path}')
                references[case.case_id][category][name] = _reference(inputs, path)
    identity = {'cases': [c.to_dict() for c in cases], 'input_root': str(inputs),
        'output_root': str(outputs), 'owner': owner, 'repetitions': repetitions,
        'inputs': references, 'provenance': provenance or {}}
    identity = _json(identity)
    if resume and run_id is None:
        raise ValueError('resume requires run_id')
    run_id = _id(run_id or _unique())
    root = outputs / run_id
    slots = [{'case_id': c.case_id, 'trial_id': f'{i:04d}',
              'record': f'{c.case_id}/{i:04d}/trial.json'}
             for c in cases for i in range(repetitions)]
    if resume:
        _, manifest = _load_run(root)
        if manifest['identity_sha256'] != _identity(identity) or manifest['identity'] != identity:
            raise ValueError('resume config/input/ownership identity mismatch')
    else:
        root.mkdir(parents=True, exist_ok=False)
        manifest = {'schema_version': SCHEMA_VERSION, 'run_id': run_id,
            'created_at': datetime.now(timezone.utc).isoformat(), 'identity': identity,
            'identity_sha256': _identity(identity), 'trials': slots,
            'code': _git(), 'host': platform.node(), 'python': sys.version,
            'environment': {k: os.environ.get(k) for k in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS',
                'MKL_NUM_THREADS', 'ITK_GLOBAL_DEFAULT_NUMBER_OF_THREADS', 'NUMEXPR_NUM_THREADS', 'CUDA_VISIBLE_DEVICES')},
            'cpu_affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None,
            'retention': 'owner decision; retain through thesis/project handoff', 'backup_status': 'unverified'}
        _write(root / 'manifest.json', manifest)
    case_map = {c.case_id: c for c in cases}
    for slot in slots:
        path = root / slot['record']
        if path.exists():
            # Validate only completed slots; processing is never re-invoked.
            _trials(root, {**manifest, 'trials': [slot]})
            continue
        directory = path.parent
        directory.mkdir(parents=True, exist_ok=False)
        case = case_map[slot['case_id']]
        trial = BenchmarkTrialResult(run_id, case.case_id, slot['trial_id'],
            case.config['registration']['method'] if case.task == 'registration' else 'pipeline', None,
            {'processing': 'failed', 'evaluation': 'not_run'}, {}, case.config, {}, {},
            provenance={'case': case.to_dict(), 'source_references': references[case.case_id]})
        # Include loading, estimator attempts, application and persistence in scope.
        if tracemalloc.is_tracing():
            raise RuntimeError('benchmark requires ownership of tracemalloc; stop existing tracing first')
        tracemalloc.start()
        wall, cpu = time.perf_counter(), time.process_time()
        try:
            arrays = {}
            for name, ref in references[case.case_id]['inputs'].items():
                source = _verified(inputs, ref)
                if case.task == 'pipeline':
                    dest = directory / (name + source.suffix)
                    with source.open('rb') as src, dest.open('xb') as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    trial.artifacts[name] = _reference(root, dest)
                else:
                    arrays[name] = _array(source)
                    trial.artifacts[name] = _save_array(root, directory, name, arrays[name])
            # Copy truth and explicitly supplied artifacts byte-for-byte into the run.
            for category in ('truth', 'artifacts'):
                for name, ref in references[case.case_id][category].items():
                    source = _verified(inputs, ref)
                    dest = directory / (category + '-' + name + source.suffix)
                    with source.open('rb') as src, dest.open('xb') as dst:
                        import shutil
                        shutil.copyfileobj(src, dst)
                    trial.artifacts[category + ':' + name] = _reference(root, dest)
            if case.task == 'pipeline':
                from ._pipeline import _process_pipeline
                _process_pipeline(case, root, directory, trial)
            else:
                registered, result = _process(case, arrays['reference'], arrays['moving'], trial.attempts)
                trial.actual_method = result.diagnostics.method
                trial.effective_configs = {**case.config, 'registration': asdict(result.diagnostics.effective_config),
                    'application': asdict(result.application_config)}
                trial.artifacts['registered'] = _save_array(root, directory, 'registered', registered)
                _save_transform(root, directory, 'transform', result, trial.artifacts)
                trial.status['processing'] = 'fallback_success' if len(trial.attempts) > 1 else 'success'
        except MemoryError:
            raise
        except Exception as exc:
            trial.errors['processing'] = _error(exc)
        finally:
            elapsed, cpu_elapsed = time.perf_counter() - wall, time.process_time() - cpu
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            scope = 'single trial: input loading, attempts, application, artifact persistence; excludes evaluation'
            trial.resources = {
                'wall': {'value': elapsed, 'unit': 'second', 'source': 'time.perf_counter', 'scope': scope},
                'cpu': {'value': cpu_elapsed, 'unit': 'second', 'source': 'time.process_time', 'scope': scope},
                'python_peak_allocation': {'value': peak, 'unit': 'byte', 'source': 'tracemalloc', 'scope': scope},
                'process_peak_rss': {'value': None, 'unit': 'byte', 'source': None,
                    'scope': 'not measured; use an external per-process monitor such as GNU time'},
            }
        _write(path, trial.to_dict())
        _write(path.with_suffix('.checksum.json'), _reference(root, path))
    return root


def evaluate_benchmark(run_dir):
    """Evaluate checksummed saved artifacts, returning a new evaluation directory.

    Processing failures become skipped evaluations. Invalid metrics become
    evaluation failures; undefined metric results retain null/reasons. Missing
    or corrupt required artifacts raise before any evaluation output is made.
    """
    root, manifest = _load_run(run_dir)
    trials = _trials(root, manifest)
    for trial in trials:
        if trial.status['processing'] in ('success', 'fallback_success'):
            case = BenchmarkCase.from_dict(trial.provenance['case'])
            required = (('pipeline', 'spots', 'reads') if case.task == 'pipeline' else
                ('reference', 'moving', 'registered', 'transform', *('truth:' + k for k in case.truth)))
            for name in required:
                if name not in trial.artifacts:
                    raise FileNotFoundError(f'required artifact reference missing: {name}')
    directory = root / 'evaluations' / _unique()
    directory.mkdir(parents=True, exist_ok=False)
    results = []
    for trial in trials:
        trial.provenance['evaluation_source'] = {'run_manifest_sha256': _digest(root / 'manifest.json'),
            'trial_sha256': _digest(root / trial.case_id / trial.trial_id / 'trial.json')}
        if trial.status['processing'] not in ('success', 'fallback_success'):
            trial.status['evaluation'] = 'skipped'
            trial.errors['evaluation'] = {'type': 'ProcessingFailed', 'message': 'processing did not succeed'}
        else:
            try:
                case = BenchmarkCase.from_dict(trial.provenance['case'])
                if case.task == 'pipeline':
                    from ._pipeline import _evaluate_pipeline
                    trial.metrics = _evaluate_pipeline(root, trial)
                else:
                    arrays = {k: _array(_verified(root, trial.artifacts[k])) for k in ('reference', 'moving', 'registered')}
                    transform = _read(_verified(root, trial.artifacts['transform']))
                    truth = {k: _read(_verified(root, trial.artifacts['truth:' + k])) for k in case.truth}
                    trial.metrics = _evaluate(case, arrays, transform, truth)
                trial.status['evaluation'] = 'undefined' if any(x['status'] != 'ok' for x in trial.metrics.values()) else 'success'
            except Exception as exc:
                trial.status['evaluation'] = 'failed'
                trial.errors['evaluation'] = _error(exc)
        results.append(trial.to_dict())
    _write(directory / 'results.json', results)
    _write(directory / 'manifest.json', {'schema_version': SCHEMA_VERSION, 'run_id': manifest['run_id'],
        'run_manifest_sha256': _digest(root / 'manifest.json'),
        'results': _reference(directory, directory / 'results.json')})
    return directory


def report_benchmark(evaluation_dir):
    """Report only saved evaluation records into a new JSON/CSV directory.

    Every trial remains visible, including failures, fallback results and null
    metrics. No implicit ranking or aggregation across incompatible methods.
    """
    evaluation = Path(evaluation_dir).resolve()
    manifest = _read(evaluation / 'manifest.json')
    if manifest['schema_version'] != SCHEMA_VERSION:
        raise ValueError('unsupported evaluation schema')
    records = _read(_verified(evaluation, manifest['results']))
    trials = [BenchmarkTrialResult.from_dict(x) for x in records]
    directory = evaluation / 'reports' / _unique()
    directory.mkdir(parents=True, exist_ok=False)
    _write(directory / 'trials.json', records)
    with (directory / 'summary.csv').open('x', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['case_id', 'trial_id', 'requested_method', 'actual_method', 'processing', 'evaluation', 'metric', 'value', 'unit', 'metric_status', 'reason'])
        for trial in trials:
            prefix = [trial.case_id, trial.trial_id, trial.requested_method, trial.actual_method,
                      trial.status['processing'], trial.status['evaluation']]
            if not trial.metrics:
                writer.writerow(prefix + ['', '', '', '', str(trial.errors)])
            for group, metric in trial.metrics.items():
                for key, value in metric['values'].items():
                    writer.writerow(prefix + [group + '.' + key, value, metric['units'][key], metric['status'], metric['reasons'].get(key, '')])
    _write(directory / 'manifest.json', {'schema_version': SCHEMA_VERSION, 'run_id': manifest['run_id'],
        'evaluation_manifest_sha256': _digest(evaluation / 'manifest.json'),
        'outputs': [_reference(directory, directory / name) for name in ('trials.json', 'summary.csv')]})
    return directory
