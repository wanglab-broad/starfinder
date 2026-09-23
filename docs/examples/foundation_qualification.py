"""Bounded W-171 probes; protocol is frozen before MATLAB execution."""
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

from artifact_contracts import signal_example
from starfinder.barcode import (Codebook, NeighborhoodSumConfig, WtaDecoderConfig,
    decode_barcodes, encode_bases, extract_intensities, filter_reads)
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult, save_candidate_checkpoint, load_candidate_checkpoint
from starfinder.registration import TranslationConfig, estimate_transform, apply_transform
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult, find_spots

REPO = Path(__file__).resolve().parents[2]
CHANNELS = ('ch02', 'ch00', 'ch03', 'ch01')
ROUNDS = ('round10', 'round2')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def source_identity():
    paths = [Path(__file__), REPO/'docs/foundation-qualification.md',
             REPO/'workflow/rules/common.smk', REPO/'workflow/scripts/qualify_foundation.m']
    paths += sorted((REPO/'src/matlab').glob('*.m'))
    paths += sorted((REPO/'src/python/starfinder').rglob('*.py'))
    return {str(p): digest(p) for p in paths}


def fixture(depth):
    centers = np.array([[depth//2, 2, 3], [depth//2, 6, 7], [0, 0, 0], [depth//2, 4, 5]])
    first = np.zeros((depth, 9, 11, 4), dtype=np.uint16)
    second = np.zeros_like(first)
    first[tuple(centers[0])+(1,)] = 7
    first[tuple(centers[1])+(0,)] = 11
    second[tuple(centers[0])+(0,)] = 9
    second[tuple(centers[1])+(1,)] = 13
    extraction = first.copy()
    extraction[tuple(centers[3])+(0,)] = 5
    extraction[tuple(centers[3])+(1,)] = 5
    second[tuple(centers[3])+(0,)] = 5
    second[tuple(centers[3])+(1,)] = 5
    reference = first.sum(axis=-1, dtype=np.float64)
    moving = np.roll(reference, (1, -1), axis=(1, 2))
    return centers, first, second, extraction, reference, moving


def prepare(root):
    root.mkdir(parents=True, exist_ok=False)
    for depth in (5, 1):
        centers, first, second, extraction, reference, moving = fixture(depth)
        savemat(root/f'z{depth}-input.mat', dict(first=first.transpose(1,2,0,3),
            second=second.transpose(1,2,0,3), extract_first=extraction.transpose(1,2,0,3),
            reference=reference.transpose(1,2,0), moving=moving.transpose(1,2,0),
            centers_xyz=centers[:,::-1]+1))
    config = dict(starfinder_path=str(REPO), directory=str(root))
    write_json(root/'matlab-config.json', config)
    documents = root/'fixture/sample/output/documents'
    documents.mkdir(parents=True)
    (documents/'sample-annotation.csv').write_text('sample_id,fov_start,fov_end\nsample,1,1\n')
    common_config = dict(starfinder_path=str(REPO), root_input_path=str(root/'fixture'),
        root_output_path=str(root/'fixture'), dataset_id='sample', sample_id='sample', output_id='output',
        config_path=str(root/'unused.yaml'),
        n_rounds=2, n_fovs=1, fov_id_pattern='FOV_{i}', subset_list=[], subset_range=False,
        subset_random=False, rules={}, backend='matlab')
    # Use the actual Snakemake include/helper, with a bounded qualification entry point.
    argument = "'" + str(root/'matlab-config.json').replace("'", "''") + "'"
    snake = f'''config.update({common_config!r})
include: {str(REPO/'workflow/rules/common.smk')!r}
rule qualify:
    default_target: True
    output: {str(root/'z5-result.mat')!r}, {str(root/'z1-result.mat')!r}
    threads: 1
    run:
        run_matlab_scripts({argument!r}, 'qualify_foundation')
'''
    (root/'Snakefile').write_text(snake)
    write_json(root/'frozen.json', dict(fixture='foundation-backend-v1', seed=None,
        no_rng_reason='Literal isolated impulses and explicit integer shifts',
        protocol_sha256=digest(REPO/'docs/foundation-qualification.md'), sources=source_identity(),
        inputs={str(p): digest(p) for p in root.rglob('*') if p.is_file()},
        commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
        patch_sha256=hashlib.sha256(subprocess.check_output(['git','diff','--binary','HEAD'],cwd=REPO)).hexdigest()))
    print(f'Frozen inputs: {root}/frozen.json; run Snakemake, then compare.')


def match(actual, expected):
    """Independent all-pairs bijection, deliberately unrelated to detector order."""
    actual, expected = np.asarray(actual), np.asarray(expected)
    assert actual.shape == expected.shape
    distances = np.linalg.norm(actual[:,None,:] - expected[None,:,:], axis=2)
    hits = distances <= 1e-9
    assert np.all(hits.sum(axis=0) == 1) and np.all(hits.sum(axis=1) == 1)
    return hits.argmax(axis=0)


def compare(root):
    frozen = json.loads((root/'frozen.json').read_text())
    assert source_identity() == frozen['sources'], 'Source changed after protocol freeze'
    for p, sha in frozen['inputs'].items():
        assert digest(p) == sha, f'Frozen input changed: {p}'
    rows = []
    for depth in (5, 1):
        centers, first, second, extraction, reference, moving = fixture(depth)
        result = loadmat(root/f'z{depth}-result.mat')
        matlab = json.loads((root/f'z{depth}-result.json').read_text())
        assert matlab['runtime']['release'] == '2023b' and matlab['runtime']['threads'] == 1
        metadata = ImageMetadata('foundation/reference')
        detected = find_spots(first, config=LocalMaximaConfig('adaptive',0.1),
                              metadata=metadata, spot_namespace=f'python/z{depth}')
        py_match = match(detected.spots[['z','y','x']], centers[:2])
        np.testing.assert_array_equal(detected.spots.channel.to_numpy()[py_match], [1,0])
        if depth > 1:
            ml_match = match(result['detection_xyz'][:,::-1]-1, centers[:2])
            np.testing.assert_array_equal(result['detection_channel'].ravel()[ml_match]-1, [1,0])
        registration = estimate_transform(reference, moving, config=TranslationConfig(),
            reference_metadata=metadata, moving_metadata=ImageMetadata('foundation/moving'))
        assert registration.transform.correction_zyx == (0,-1,1)
        np.testing.assert_array_equal(result['shifts_yxz'].ravel()[[2,0,1]], [0,-1,1])
        registered = result['registered']
        if depth == 1: registered = registered[:,:,None]
        np.testing.assert_allclose(registered.transpose(2,0,1), reference, rtol=0, atol=1e-12)
        np.testing.assert_allclose(apply_transform(moving, registration.transform, config=registration.application_config), reference, rtol=0, atol=1e-12)
        table = pd.DataFrame(centers.astype(float), columns=['z','y','x'])
        table['spot_id'] = pd.Series(['A','B','zero','tie'],dtype='string')
        spots = SpotFindingResult(table, metadata, f'probe/z{depth}', LocalMaximaConfig(), {})
        rounds = {ROUNDS[0]:ImageLoadResult(extraction,metadata,CHANNELS,()),
                  ROUNDS[1]:ImageLoadResult(second,metadata,CHANNELS,())}
        book = Codebook(pd.DataFrame({'gene_id':['gene-A','gene-B'],'color_sequence':['12','21']}),
                        ROUNDS, CHANNELS, {'1':1,'2':0,'3':3,'4':2})
        expected = np.array([[[0,9],[7,0],[0,0],[0,0]], [[11,0],[0,13],[0,0],[0,0]],
                             np.zeros((4,2)), [[5,5],[5,5],[0,0],[0,0]]], dtype=float)
        expected_scores = np.log1p(1e-6 / np.array([[7,9],[11,13]]))
        calls = np.asarray(matlab['calls'])
        for radius in (0,1):
            extracted = extract_intensities(rounds, spots, config=NeighborhoodSumConfig((radius,)*3))
            np.testing.assert_array_equal(extracted.values, expected)
            decoded = decode_barcodes(extracted, book, config=WtaDecoderConfig())
            assert decoded.table.call_status.tolist() == ['assigned','assigned','no_signal','ambiguous']
            np.testing.assert_allclose(decoded.table.wta_l2_nll.iloc[:2], expected_scores.sum(axis=1), atol=1e-12, rtol=0)
            np.testing.assert_array_equal(calls[:,:,radius], [['1','2'],['2','1'],['M','M'],['M','M']])
            np.testing.assert_allclose(result['scores'][:2,:,radius], expected_scores, atol=1e-12, rtol=0)
            assert np.isinf(result['scores'][2:,:,radius]).all()
            assert filter_reads(decoded).accepted.gene_id.tolist() == ['gene-A','gene-B']
        actual = pd.read_csv(root/f'z{depth}-filtered.csv')
        order = match(actual[['z','y','x']].to_numpy()-1, centers[:2])
        assert actual.gene.to_numpy()[order].tolist() == ['gene-A','gene-B']
        assert matlab['encoded'] == ['12','21']
        assert [encode_bases(v) for v in ('AAC','ACC')] == ['12','21']
        rows.append(dict(depth=depth, shape=list(first.shape), status='passed',
            purpose='Compare literal registration, isolated detection, trace calls and selected filtering.',
            detection='passed' if depth>1 else 'unexecuted MATLAB volumetric detector at Z=1',
            matched_centers=centers[:2].tolist(), correction_zyx=[0,-1,1],
            expected_scores=expected_scores.tolist(), matlab_scores=result['scores'][:2,:,0].tolist(),
            max_matlab_image_error=float(np.abs(registered.transpose(2,0,1)-reference).max()),
            runtime=matlab['runtime'], python_population=['assigned','assigned','no_signal','ambiguous'],
            matlab_calls=calls.tolist(), filtering_population='A/B uniquely called only'))
    write_json(root/'comparison.json',dict(protocol_sha256=frozen['protocol_sha256'], conditions=rows,
        input_sha256=frozen['inputs'], output_sha256={str(p):digest(p) for p in root.glob('z*-result.*')},
        limitations=['No general algorithm parity; MATLAB Z=1 detection not qualified.',
                     'MATLAB raw NCR trace is not exposed; only Python trace and MATLAB calls/scores compared.',
                     'MATLAB filtering excludes zero/tie reads; Python retains their rejection states.']))
    print('Actual Python/MATLAB bounded comparisons passed.')


def storage(root):
    import h5py
    import pyarrow as pa
    import pyarrow.parquet as pq
    root.mkdir(parents=True, exist_ok=False)
    records = []
    z,y,x = np.indices((9,48,48))
    ramp = 100*z+10*y+x
    for dtype in ('uint16','float32','float64'):
        image = ramp.astype(dtype)[...,None]
        for name, chunks, codec in [('contiguous',None,None),('default',(8,48,48,1),'gzip'),('small-chunks',(3,16,16,1),'gzip')]:
            path = root/f'{dtype}-{name}.h5'
            started = perf_counter()
            kwargs = dict(chunks=chunks, compression=codec)
            if codec: kwargs.update(compression_opts=4,shuffle=True)
            with h5py.File(path,'w') as f: f.create_dataset('image',data=image,**kwargs)
            write_seconds = perf_counter()-started
            started = perf_counter()
            with h5py.File(path) as f:
                loaded = f['image'][:]
                np.testing.assert_array_equal(loaded,image,strict=True)
                region = f['image'][1:2,1:3,1:3,:]
            read_seconds = perf_counter()-started
            np.testing.assert_array_equal(region,image[1:2,1:3,1:3,:],strict=True)
            records.append(dict(kind='image',variant=f'{dtype}-{name}',shape=list(image.shape),dtype=dtype,
                chunks=chunks,compression=codec,raw_bytes=image.nbytes,size_bytes=path.stat().st_size,
                write_seconds=write_seconds,read_seconds=read_seconds,sha256=digest(path),status='passed'))
    spots, signal, book, *_ = signal_example(3)
    # Repeat existing literal values with distinct identities, no new scientific population.
    table = pd.concat([spots.spots]*128, ignore_index=True)
    table['spot_id'] = pd.Series([f'cost-{i}' for i in range(256)],dtype='string')
    spots = replace(spots,spots=table)
    signal = replace(signal,values=np.tile(signal.values,(128,1,1)),
        valid=np.tile(signal.valid,(128,1)),spot_ids=tuple(table.spot_id))
    source_hash = hashlib.sha256(signal.values.tobytes()).hexdigest()
    costs = []
    expected_qc = filter_reads(decode_barcodes(signal,book,config=WtaDecoderConfig()))
    for enabled in (True,False):
        start = perf_counter()
        saved = save_candidate_checkpoint(root/f'saving-{enabled}',spots,signal,codebook=book,
            enabled=enabled,dataset_id='foundation-storage-v1',sample_id='sample',FOV='literal',
            run_id=f'cost-{enabled}',config={'seed':None},code={
                'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip(),
                'source_sha256':source_identity()})
        seconds = perf_counter()-start
        if enabled:
            start = perf_counter()
            reloaded = load_candidate_checkpoint(saved.path,sha256=digest(saved.path))
            read_seconds = perf_counter()-start
            np.testing.assert_array_equal(reloaded.intensities.values,signal.values,strict=True)
            pd.testing.assert_frame_equal(filter_reads(decode_barcodes(reloaded.intensities,
                reloaded.codebook,config=WtaDecoderConfig())).table,expected_qc.table,check_exact=True)
        else:
            read_seconds = None
            assert saved.path is None and saved.size_bytes == 0 and saved.reason == 'candidates_signals_disabled'
            assert not (root/f'saving-{enabled}').exists()
        costs.append(dict(enabled=enabled,size_bytes=saved.size_bytes,write_seconds=seconds,
                          read_seconds=read_seconds,reason=saved.reason,counts=expected_qc.counts))
    canonical = pq.read_table(root/'saving-True/signals.parquet')
    for dtype, codec, row_group, partitions in [('float64','zstd',65536,1),('float64',None,65536,1),
            ('float64','zstd',64,1),('float64','zstd',65536,4),('float32','zstd',65536,1)]:
        schema = canonical.schema.set(canonical.schema.get_field_index('value'),pa.field('value',getattr(pa,dtype)()))
        current = canonical.cast(schema)
        name = f'{dtype}-{codec}-rg{row_group}-p{partitions}'
        folder = root/name; folder.mkdir()
        start = perf_counter()
        for part in range(partitions):
            pq.write_table(current.slice(part*len(current)//partitions,len(current)//partitions),
                           folder/f'{part}.parquet',compression=codec,row_group_size=row_group)
        write_seconds = perf_counter()-start
        start = perf_counter()
        restored = pa.concat_tables([pq.read_table(p) for p in sorted(folder.glob('*.parquet'))])
        read_seconds = perf_counter()-start
        assert restored.equals(current)
        np.testing.assert_array_equal(restored['value'].to_numpy(),canonical['value'].to_numpy())
        records.append(dict(kind='signal-layout-probe',variant=name,rows=len(current),dtype=dtype,
            compression=codec,row_group_size=row_group,partitions=partitions,
            size_bytes=sum(p.stat().st_size for p in folder.iterdir()),write_seconds=write_seconds,
            read_seconds=read_seconds,status='passed',schema=str(current.schema),
            sha256={p.name:digest(p) for p in folder.iterdir()}))
    assert hashlib.sha256(signal.values.tobytes()).hexdigest() == source_hash
    write_json(root/'storage.json',dict(fixture='foundation-storage-v1',seed=None,
        no_rng_reason='Literal ramp and repeated artifact-contract-v1 traces',records=records,saving=costs,
        decision='Retain persistent candidate/signal saving and explicit disable override for traceability; no canonical format changes.',
        limitations=['Single small-fixture observations on this filesystem; no production timing/size extrapolation.',
                     'Alternative layouts are cost diagnostics, not loadable v1 checkpoints.',
                     'Float32 preserves these small integers only; canonical float64 remains required.']))
    print('Bounded storage probes and saving override passed.')


if __name__ == '__main__':
    operation, directory = sys.argv[1:]
    {'prepare':prepare,'compare':compare,'storage':storage}[operation](Path(directory).resolve())
