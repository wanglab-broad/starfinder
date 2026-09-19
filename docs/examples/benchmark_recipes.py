"""Run source-derived profiles on deterministic bounded fixtures only.

Usage from src/python: uv run python ../../docs/examples/benchmark_recipes.py
/new/external/directory (on one line). No scientific datasets or historical
scripts are executed. Every new image is at most 16x32x32, four channels/rounds.
"""
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import tifffile

from starfinder.benchmark import run_benchmark, evaluate_benchmark, report_benchmark

REPO = Path(__file__).resolve().parents[2]


def load_helper(name):
    spec = importlib.util.spec_from_file_location(name, REPO/'benchmarks'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(destination):
    root = Path(destination).resolve()
    root.mkdir(parents=True, exist_ok=False)
    inputs = root/'inputs'
    inputs.mkdir()
    recipes = load_helper('recipes')
    image = np.zeros((16, 32, 32), dtype=np.float32)
    for z in (3, 8, 12):
        for y in (6, 16, 25):
            for x in (6, 16, 25):
                image[z, y, x] = 1000 + z*y + x
    image = gaussian_filter(image, .65)
    np.save(inputs/'reference.npy', image)
    np.save(inputs/'moving.npy', np.roll(image, (0, 1, -1), axis=(0, 1, 2)))
    cases = [recipes.registration_case(name, case_id=name, reference='reference.npy', moving='moving.npy',
        evaluation={'ncc': True}, reference_metadata={'frame_id':'ref'}, moving_metadata={'frame_id':'mov'})
        for name in ('translation', 'py_demons', 'py_diffeo', 'tps-small', 'cpd-small')]
    sources = {}
    for i in range(1, 5):
        sources[f'round{i}'] = {}
        for j in range(4):
            array = np.zeros((8, 16, 16), dtype=np.uint8)
            array[3, 7, 8] = 200 if j == 0 else 20
            array[5, 10, 11] = 150 if j == 1 else 10
            name = f'r{i}-ch{j:02}.tif'
            tifffile.imwrite(inputs/name, array, metadata={'axes':'ZYX'}, photometric='minisblack')
            sources[f'round{i}'][f'ch{j:02}'] = name
    (inputs/'codebook.csv').write_text('gene,barcode\ngeneA,CCCCC\ngeneB,CACAC\n')
    for name in ('large-batch', 'large-streaming', 'large-global-local'):
        case = recipes.pipeline_case(name, case_id=name, fov_id='FOV_001', sources=sources,
                                     codebook='codebook.csv', n_rounds=4)
        # Same sample identity permits parity comparison across execution modes.
        case.config['workflow']['sample_id'] = 'fixture'
        cases.append(case)
    config = {'schema_version':1, 'repetitions':1, 'cases':[c.to_dict() for c in cases],
        'provenance':{'kind':'deterministic adapter fixture; not scientific evidence','seed':None}}
    (root/'cases.json').write_text(json.dumps(config,indent=2)+'\n')
    run = run_benchmark(cases, input_root=inputs, output_root=root/'runs', owner='Jiahao',
                        provenance=config['provenance'])
    evaluation = evaluate_benchmark(run)
    records = json.loads((evaluation/'results.json').read_text())
    for result in records:
        assert result['status'] == {'processing':'success','evaluation':'success'}, result
    pd.testing.assert_frame_equal(pd.read_csv(run/'large-batch/0000/reads.csv'),
                                  pd.read_csv(run/'large-streaming/0000/reads.csv'))
    report = report_benchmark(evaluation)
    historical = root/'saved-fixture.csv'
    historical.write_text('dataset,pair_type,backend,ncc,status\nfixture,shift,python,1,success\n')
    saved_report = load_helper('report_saved').report_saved(historical, output_dir=root/'saved-report',
        keys=['dataset','pair_type','backend'], variant='fixture', timing_scope='not measured', memory_scope='not measured')
    summary = {'run':str(run),'evaluation':str(evaluation),'report':str(report),
               'saved_report':str(saved_report),'trials':len(records),'status':'passed'}
    (root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main(sys.argv[1])
