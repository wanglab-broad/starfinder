"""Snakemake adapters of the workflow rules, which §2.13 (workflow orchestration) replaces.

The runners that take the ``snakemake`` object of a rule script, the rule input lists that the
rule files build their inputs from, and the writers of the MATLAB-shaped per-FOV files. They were
moved unchanged out of ``starfinder.dataset.workflow``, which keeps the configuration translation
they call (WorkflowConfig, from_workflow_config and the block translators). Only ``workflow/``
and the tests import this module (docs/workflows.md, "What §2.13 replaces").
"""
import json
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

from starfinder import _layout
from starfinder.dataset._declared import reference_settings
from starfinder.dataset.types import ChannelInfo
from starfinder.dataset.workflow import (_BIGSTITCHER, _LEGACY_SEGMENTATION_RUN, _NUCLEI_REFERENCE_CHANNEL,
    _after_fusion, _assembly_block, _declared_metadata, _export_config, _export_dataset, _flag, _label_namespace,
    _legacy_molecules, _molecule_table, _nuclei_checkpoints, _nuclei_registration, _output_root, _reads_assignment, _report_assignment, _report_block,
    _rule_checkpoints, _rule_parameters, _sample_fovs, _sample_table, _segmentation, _stitching_block,
    _workflow_dataset, from_workflow_config)
from starfinder.preprocessing import ProjectionConfig
from starfinder.registration import (ExternalReference, RegistrationRecipe, RegistrationSignalConfig,
    RegistrationStep, TranslationConfig)


def _run_workflow(snakemake, rule):
    """Thin common runner used by the five Snakemake script adapters; each run writes the FOV's run.json.

    The run record is fovs/<fov>/run.json, or fovs/<fov>/subtile_<n>/run.json
    for the resident subtile rules; it holds the counts, fractions, read
    summary, round roles, registration attempts and transforms that the
    MATLAB-shaped logs held before. The rules that are not resident also write
    the reference image (FOV.save_reference_image, the file reference_image
    names), except with reference_image_import, whose file they never write.
    """
    from .dataset.fov import FOV
    adapted = from_workflow_config(snakemake.config, rule)
    dataset = adapted.dataset
    fov_id = snakemake.wildcards.fovID
    resident = rule in ('lrsf_single_fov_subtile', 'deep_rsf_subtile')
    if adapted.pipeline.decoding and dataset.readout_mode == 'direct':
        # In direct mode the rule's codebook input is the panel CSV.
        dataset.load_direct_panel(Path(snakemake.input[1]))
    elif adapted.pipeline.decoding:
        dataset.load_codebook(Path(snakemake.input[1]), encoding=adapted.encoding, layout=adapted.layout)
    fov = FOV.from_subtile(Path(snakemake.input[2]), dataset, fov_id) if resident else dataset.fov(fov_id)
    fov.run(adapted.pipeline, execution=adapted.execution, checkpoints=_rule_checkpoints(snakemake.config, rule))
    if resident:
        from starfinder.readout.spots import _export_legacy_spots
        number = snakemake.wildcards.n_subtile
        # The legacy route: stitch_subtile reads the one-based subtile tables.
        _export_legacy_spots(fov.spot_result, fov.filtering_result,
                             Path(snakemake.input[2]).parent / f'subtile_goodSpots_{number}.csv')
    else:
        if dataset.reference_image_import is None:
            fov.save_reference_image(projection=adapted.reference_projection, reference_image=adapted.reference_image,
                                     reference_channel=adapted.reference_channel)
        if rule == 'rsf_single_fov':
            fov.export_spots()
        else:
            fov.save_subtiles()


def _run_nuclei_registration(snakemake):
    """Python counterpart of nuclei_registration.m for the Python backend, through FOV.prepare_morphology.

    The entry loads the reference round's ch04 image as the reference stain
    and each additional round with its channel_order, rotates them by
    rotate_angle, and registers each round by one translation step on its
    shared stain (the channel _nuclei_registration matches) against the stain
    as an ExternalReference labelled ``<ref_round>:ch04``, transferring the
    transform to all its channels. A round whose shared stain has another
    pattern than the rounds before it is registered after the entry by the
    same public calls (load_images with the angle, register), in the
    configured order. Writes fovs/<fov>/nuclei_registration.json
    (FOV.save_registration_record: the registered rounds and their attempts)
    and each channel's image images/<round>/<channel label>/<fov>.tif, named
    by the unique channel label (ChannelInfo.channel) where MATLAB uses the
    descriptive name (ZYX, or the Z maximum YX with maximum_projection),
    keeping the dtype; with the
    Python-only rules.nuclei_registration.parameters.checkpoints, also the
    saved form of each prepared image (other_rounds/<name>/,
    docs/checkpoints.md), which FOV.load_registered_round reads. Unlike
    MATLAB, it does not min-max stretch the other rounds.
    """
    from dataclasses import asdict
    import numpy as np
    import tifffile
    from starfinder.io import save_volume
    from starfinder.preprocessing import project_image
    from .dataset.config import MorphologyConfig
    from .dataset.types import REFERENCE_STAIN
    config = snakemake.config
    dataset, stains, folders = _nuclei_registration(config)
    checkpoints = _nuclei_checkpoints(config)
    if not dataset.reference_stains:
        # nuclei_registration.m reads the reference round's ch04 file with or without dapi_round. The rule reads
        # no sequencing channel, so a ch04 that a configuration without dapi_round lists among the sequencing
        # colours is left out of the rule's Dataset, where it is the reference stain.
        sequencing = dataset.channel_info(dataset.rounds.sequencing_rounds[0])
        dataset = replace(dataset, channel_order=tuple(c for c in sequencing if c.channel != _NUCLEI_REFERENCE_CHANNEL),
                          other_channel_order={name: dataset.channel_info(name) for name in dataset.other_channel_order},
                          reference_stains=(ChannelInfo(_NUCLEI_REFERENCE_CHANNEL, config['ref_channel']),))
    fov = dataset.fov(snakemake.wildcards.fovID)
    angle, ref = config.get('rotate_angle'), dataset.rounds.reference_round
    names = list(dataset.rounds.other_rounds)

    def recipe(name):
        signal = RegistrationSignalConfig('channel', _NUCLEI_REFERENCE_CHANNEL, stains[name])
        return RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=signal)

    # The leading rounds that share the first round's stain pattern go through the entry in one call.
    leading = next((i for i, name in enumerate(names) if stains[name] != stains[names[0]]), len(names))
    fov.prepare_morphology(MorphologyConfig(rotation_degrees=angle, recipe=recipe(names[0]),
                                            rounds=tuple(names[:leading])), checkpoints=checkpoints)
    reference = ExternalReference(np.ascontiguousarray(fov.images[REFERENCE_STAIN][..., 0]),
                                  fov.metadata[REFERENCE_STAIN], f'{ref}:{_NUCLEI_REFERENCE_CHANNEL}')
    for name in names[leading:]:
        fov.load_images(rounds=[name], rotation_degrees=angle)
        fov.register(recipe(name), rounds=[name], reference=reference, checkpoints=checkpoints)
    fov.save_registration_record()
    projection = ProjectionConfig() if config.get('maximum_projection', False) else None
    for name in names:
        # One folder per channel label, so channels that share a descriptive name write separate files.
        for c, folder in enumerate(folders[name]):
            path = _layout.image_file(dataset.output_root, fov.fov_id, name, folder)
            volume, metadata = fov.images[name][..., c], fov.metadata[name]
            if projection is None:
                save_volume(volume, path, metadata=metadata)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                tifffile.imwrite(path, project_image(volume, config=projection)[0], photometric='minisblack',
                                 metadata={'axes': 'YX',
                                           'starfinder_metadata': asdict(metadata.projected(method=projection.method))})


def _read_stack(path):
    """A legacy TIFF as ZYX (a YX file gains a singleton Z) and whether the file was YX."""
    import tifffile
    image = tifffile.imread(path)
    return (image[None] if image.ndim == 2 else image), image.ndim == 2


def _run_nuclei_amplicon_overlay(snakemake):
    """create_nuclei_amplicon_overlay: composite_nuclei_amplicon of the DAPI and reference merged images.

    The composite of create_nuclei_amplicon_overlay.py bit for bit; with the
    rule's maximum_projection the output is its Z maximum (YX). YX inputs are
    read as one plane and give a YX output. Written as the script wrote it.
    """
    import tifffile
    from starfinder.preprocessing import ProjectionConfig, project_image
    from starfinder.segmentation import composite_nuclei_amplicon
    params = _rule_parameters(snakemake.config, 'create_nuclei_amplicon_overlay')
    projection = _flag(params, 'maximum_projection', 'create_nuclei_amplicon_overlay')
    nuclear, plane = _read_stack(snakemake.input['dapi_img'])
    amplicon, _ = _read_stack(snakemake.input['amplicon_img'])
    composite, _ = composite_nuclei_amplicon(nuclear, amplicon)
    if projection:
        composite = project_image(composite, config=ProjectionConfig())
    tifffile.imwrite(snakemake.output[0], composite[0] if projection or plane else composite)


def _run_enhance_dapi_with_flamingo(snakemake):
    """enhance_dapi_with_flamingo: enhance_with_flamingo of the DAPI and Flamingo images, as the script wrote it."""
    import tifffile
    from starfinder.segmentation import enhance_with_flamingo
    nuclear, plane = _read_stack(snakemake.input['dapi_img'])
    flamingo, _ = _read_stack(snakemake.input['flamingo_img'])
    enhanced, _ = enhance_with_flamingo(nuclear, flamingo)
    tifffile.imwrite(snakemake.output[0], enhanced[0] if plane else enhanced)


def _run_stardist_segmentation(snakemake):
    """stardist_segmentation: the translated run (_segmentation) on the rule's one input file.

    The input is read with load_volume; its grid is reference_grid_from_file,
    whose metadata is the file's own or, for a file without one, declared from
    voxel_size_z and voxel_size_xy (the projection of it for a YX file with a
    translated projection). The labels are segment's uint32 result after the
    translated operations, written to the rule's output in the file's
    dimensionality (zlib), with the run record beside it as {fovID}.json. There
    is no foreground gate: an image without objects gives an all-zero image
    (outcome empty).
    """
    import tifffile
    from starfinder.io import load_volume
    from starfinder._records import jsonable
    from starfinder.segmentation import SegmentationInput, reference_grid_from_file, segment
    from starfinder._records import file_sha256
    from starfinder.segmentation._plan import _expand
    config, fov_id = snakemake.config, snakemake.wildcards.fovID
    legacy = _segmentation(config)
    path = Path(snakemake.input[0])
    loaded = load_volume(path)
    declared = loaded.diagnostics['metadata_source'] != 'stored'
    with tifffile.TiffFile(path) as tif:
        plane = len(tif.series[0].shape) == 2
    metadata = None
    if declared:
        metadata = _declared_metadata(config, fov_id)
        if plane and legacy.projection is not None:
            metadata = metadata.projected(method=legacy.projection.method)
    grid = reference_grid_from_file(path, metadata=metadata)
    source = {'path': str(path), 'segmentation_input_folder': legacy.record['segmentation_input_folder']}
    result = segment(SegmentationInput(loaded.image[..., None], grid, (legacy.role,), (source,)),
                     config=legacy.config, target=legacy.target, device=legacy.device,
                     label_namespace=_label_namespace(config, fov_id, _LEGACY_SEGMENTATION_RUN))
    for operation in legacy.operations:
        result = _expand(result, operation)
    output = Path(snakemake.output[0])
    output.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(output, result.labels[0] if plane else result.labels, compression='zlib',
                     photometric='minisblack')
    record = dict(result.record, run=_LEGACY_SEGMENTATION_RUN,
                  input=dict(result.record['input'], path=str(path), file_sha256=file_sha256(path),
                             projection=None if legacy.projection is None else asdict(legacy.projection)),
                  workflow=dict(legacy.record, metadata_source='declared' if declared else 'stored',
                                output=str(output), output_sha256=file_sha256(output)))
    output.with_suffix('.json').write_text(json.dumps(jsonable(record), indent=2, default=str))


def _fov_sample(annotation, config, fov_id):
    """(sample, FOV number) of fov_id: the sample-annotation row whose fov_start..fov_end holds it."""
    import pandas as pd
    table = pd.read_csv(annotation)
    for row in table.itertuples():
        for i in range(int(row.fov_start), int(row.fov_end) + 1):
            if config['fov_id_pattern'].format(i=i) == fov_id:
                return row.sample_id, i
    raise ValueError(f'{fov_id} is in no sample of {annotation}')


def _tile_record(path, number):
    """The FOV's row of output/tile_config_<sample>.csv (offsets x, y, z and the start/end_*_norm box)."""
    import pandas as pd
    table = pd.read_csv(path, index_col=0)
    rows = table[table['id'] == int(number)]
    if rows.empty:
        raise ValueError(f'{path} has no row with id {number}')
    return rows.iloc[0]


def _codebook_genes(path, mode):
    """The codebook's genes in first-appearance order: the gene (gene_id) column, or a direct panel's gene_id."""
    import csv
    with open(path, newline='', encoding='utf-8-sig') as handle:
        rows = [row for row in csv.reader(handle) if row]
    header = [v.strip() for v in rows[0]] if rows else []
    if mode == 'direct':
        if header != ['round', 'channel', 'gene_id']:
            raise ValueError(f'{path}: the direct panel header must be round,channel,gene_id')
        column = 2
    elif header and header[0] in ('gene', 'gene_id', 'entry_id'):
        column = header.index('gene_id') if 'gene_id' in header else 0
    else:
        column, header = 0, None
    body = rows[1:] if header else rows
    return list(dict.fromkeys(row[column].strip() for row in body))


def _assignment_genes(genes_csv, config):
    """The gene list of the count matrix: documents/genes.csv, checked against the codebook's genes.

    The codebook is the sequencing rules' input genes.csv (a direct panel in
    readout_mode direct). A gene on one list only raises ValueError naming it.
    The order is that of documents/genes.csv, the legacy matrix's columns.
    """
    import pandas as pd
    codebook = Path(config['root_input_path']) / config['dataset_id'] / config['sample_id'] / 'genes.csv'
    listed = list(dict.fromkeys(pd.read_csv(genes_csv, header=None, dtype=str)[0].str.strip()))
    known = _codebook_genes(codebook, config.get('readout_mode', 'multiplexed'))
    only_listed, only_known = [g for g in listed if g not in known], [g for g in known if g not in listed]
    if only_listed or only_known:
        raise ValueError(f'{genes_csv} and the codebook {codebook} list different genes: only in genes.csv '
                         f'{only_listed}, only in the codebook {only_known}')
    return tuple(listed)


def _legacy_cells(path, config, fov_id, target):
    """(molecule grid, cell run) of a legacy label file, imported on a grid declared from the configuration.

    A ZYX file is on the grid of its shape; a YX file is a plane on the
    projection of the img_z × Y × X grid. The metadata is _declared_metadata.
    """
    import tifffile
    from starfinder.segmentation import ReferenceGrid, import_labels
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(n) for n in tif.series[0].shape)
    metadata = _declared_metadata(config, fov_id)
    if len(shape) == 2:
        if 'img_z' not in config:
            raise ValueError(f'{path} is a YX label image; img_z gives the Z size of the molecule grid')
        grid = ReferenceGrid((int(config['img_z']), *shape), metadata, 'declared')
        label_grid = grid.projected(method='max') if grid.shape_zyx[0] > 1 else grid
    else:
        grid = label_grid = ReferenceGrid(shape, metadata, 'declared')
    cells = import_labels(path, grid=label_grid, target=target,
                          label_namespace=_label_namespace(config, fov_id, _LEGACY_SEGMENTATION_RUN))
    return grid, cells


def _assembly_cells(path, output_root, config, fov_id, target):
    """(molecule grid, cell run) of a legacy label file with an assembly block: on the reference grid.

    The molecule grid is that of the FOV's reference image (the file
    reference_image_import or reference_image names, default
    fovs/{fovID}/reference.tif), the grid
    Dataset.export gives assemble (starfinder.dataset._declared.reference_grid:
    the file's stored metadata, or the grid voxel_size_xy and voxel_size_z
    declare for a file that stores none), so the saved assignment passes
    assemble's grid check; a ZYX label file must have its shape, and a YX one
    is a plane on its Z maximum projection.
    """
    import tifffile
    from starfinder.image import IncompatibleGeometryError
    from starfinder.segmentation import import_labels
    from .dataset._declared import reference_grid
    grid = reference_grid(output_root, fov_id, **reference_settings(config))
    with tifffile.TiffFile(path) as tif:
        shape = tuple(int(n) for n in tif.series[0].shape)
    if len(shape) == 2 and shape == tuple(grid.shape_zyx[1:]):
        label_grid = grid.projected(method='max') if grid.shape_zyx[0] > 1 else grid
    elif shape == tuple(grid.shape_zyx):
        label_grid = grid
    else:
        raise IncompatibleGeometryError(f'{path} has shape {shape}; the reference grid of FOV {fov_id!r} '
                                        f'({reference_settings(config)["reference_image"]}) is {grid.shape_zyx}')
    cells = import_labels(path, grid=label_grid, target=target,
                          label_namespace=_label_namespace(config, fov_id, _LEGACY_SEGMENTATION_RUN))
    return grid, cells


def _in_box(x, y, tile):
    """Whether integer positions lie in the tile's start/end_x_norm, start/end_y_norm box (legacy range test)."""
    return ((x >= tile['start_x_norm']) & (x < tile['end_x_norm']) & (y >= tile['start_y_norm'])
            & (y < tile['end_y_norm']))


def _cell_obs(result, rows, *, sample, fov_id, tile):
    """The raw.h5ad obs of the kept cells ``rows``: the legacy columns, then the cell table's.

    Legacy meaning, computed on the territories assign samples: volume is
    expanded_size_voxels (size_voxels without expansion), fov_x/y/z the
    truncated (expanded_)centroid, seg_label the cell_id, global_* fov_* plus
    the tile offsets; plane cells have no fov_z and global_z. Without a tile
    (an assembly block, option W1) there are no global_* columns.
    """
    import numpy as np
    import pandas as pd
    expanded = result.territories is not None
    prefix = 'expanded_' if expanded else ''
    plane = result.cell_labels.shape[0] == 1
    n = len(rows)
    columns = {'sample': np.array([sample] * n, dtype=object), 'fov_id': np.array([fov_id] * n, dtype=object),
               'volume': rows[f'{prefix}size_voxels'].to_numpy(np.float64)}
    axes = ('x', 'y') if plane else ('x', 'y', 'z')
    for axis in axes:
        columns[f'fov_{axis}'] = rows[f'{prefix}centroid_{axis}'].to_numpy(np.float64).astype(np.int64)
    columns['seg_label'] = rows.cell_id.to_numpy(np.int64)
    for axis in axes if tile is not None else ():
        columns[f'global_{axis}'] = columns[f'fov_{axis}'] + np.int64(tile[axis])
    columns.update(size_voxels=rows.size_voxels.to_numpy(np.int64),
                   expanded_size_voxels=pd.array(rows.expanded_size_voxels, dtype='Int64'),
                   size_physical=rows.size_physical.to_numpy(np.float64))
    for axis in ('z', 'y', 'x'):
        columns[f'centroid_{axis}'] = rows[f'centroid_{axis}'].to_numpy(np.float64)
    columns.update(n_molecules=rows.n_molecules.to_numpy(np.int64), n_nuclei=pd.array(rows.n_nuclei, dtype='Int64'))
    for name in ('correspondence', 'correspondence_flags', 'compartments'):
        # Categorical, as anndata stores strings; an all-null column (no nuclei) has no categories.
        columns[name] = pd.Categorical(np.array([None if pd.isna(v) else str(v) for v in rows[name]], dtype=object))
    return pd.DataFrame(columns, index=pd.Index([str(i) for i in range(n)], dtype=object))


def _write_raw_h5ad(result, path, *, sample, fov_id, tile):
    """expr/{fovID}/raw.h5ad from an AssignmentResult: the kept cells inside the tile box.

    X is the float64 whole-cell counts in result.genes order and obs the
    legacy and cell-table columns (_cell_obs); the legacy overlap filter keeps
    the cells whose fov_x and fov_y lie in the tile box. With tile None (an
    assembly block, option W1) every kept cell is written and obs has no
    global_* columns. With nuclei, the
    float64 layers nucleus and cytoplasm hold the compartment counts, 0.0 for
    a measured zero and NaN for every gene of a cell whose compartments are not
    available. uns["assignment"] is the record as JSON text. Returns the
    seg_label values of the cells written.
    """
    import numpy as np
    import pandas as pd
    from starfinder.assignment import write_h5ad
    from starfinder._records import jsonable
    _, rows = result.matrix()
    # Object strings throughout: anndata 0.12 writes no pandas Arrow string arrays, which pandas 3 infers.
    with pd.option_context('future.infer_string', False):
        obs = _cell_obs(result, rows, sample=sample, fov_id=fov_id, tile=tile)
        keep = (_in_box(obs.fov_x.to_numpy(), obs.fov_y.to_numpy(), tile) if tile is not None
                else np.ones(len(obs), dtype=bool))
        obs = obs[keep]
        obs.index = pd.Index([str(i) for i in range(len(obs))], dtype=object)
        write_h5ad(path, obs, rows[['cell_id']][keep].reset_index(drop=True), result.counts, result.genes,
                   layers=result.nucleus_labels is not None,
                   uns={'assignment': json.dumps(jsonable(result.record), default=str)})
    return obs.seg_label.to_numpy()


def _write_reads_assignment(result, reads_csv, path, *, tile, cells, legacy):
    """expr/{fovID}/reads_assignment.csv: the molecule table's rows with the assignment columns.

    The columns are the molecule table's, then global_x/y/z and seg_label (the
    cell at the sampled voxel or 0), then the assignment columns
    assignment_status, cell_id, in_expansion, original_cell_id, nucleus_id
    and compartment. A molecules.csv (``legacy`` False) is zero-based and holds
    the identities: its rows get their assignment by
    ``(spot_namespace, spot_id)``. The legacy goodSpots CSV (``legacy`` True)
    keeps the columns of reads_assignment.py (zero-based x, y, z after the
    subtraction of one), its row i is the molecule ``csv:<i>``, and spot_id is
    added before the assignment columns. The legacy overlap filter keeps the
    molecules assigned to the written cells, then the
    unassigned and excluded_cell molecules whose sampled voxel lies in the tile
    box. Every outside_grid molecule follows them, in input order, unfiltered
    (it has no sampled voxel, and the script wrapped or raised on it): seg_label
    0, null cell columns and coordinates computed as for the other rows. With
    tile None (an assembly block, option W1) every molecule is written in input
    order, without global_* columns.
    """
    import numpy as np
    import pandas as pd
    columns = ('assignment_status', 'cell_id', 'in_expansion', 'original_cell_id', 'nucleus_id', 'compartment')
    if legacy:
        reads = pd.read_csv(reads_csv)
        for axis in ('x', 'y', 'z'):
            reads[axis] = reads[axis] - 1
        molecules = result.molecules
        columns = ('spot_id',) + columns
    else:
        from starfinder._records import _read_table
        from starfinder.readout.spots import MOLECULE_DTYPES
        reads = _read_table(reads_csv, MOLECULE_DTYPES)
        keys = ['spot_namespace', 'spot_id']
        order = pd.DataFrame({k: result.molecules[k].astype(object) for k in keys}).reset_index(names='row')
        rows = reads[keys].astype(object).merge(order, on=keys, how='left', validate='one_to_one')['row']
        if rows.isna().any():
            raise ValueError(f'{reads_csv}: molecules without an assignment')
        molecules = result.molecules.iloc[rows.to_numpy(np.int64)].reset_index(drop=True)
    if tile is not None:
        for axis in ('x', 'y', 'z'):
            reads[f'global_{axis}'] = reads[axis] + tile[axis]
    reads['seg_label'] = molecules.cell_id.astype('Int64').fillna(0).to_numpy(np.int64)
    for column in columns:
        reads[column] = molecules[column].to_numpy(dtype=object, na_value=None)
    if tile is None:
        reads.to_csv(path, index=False)
        return
    status = molecules.assignment_status.to_numpy(dtype=object)
    assigned = (status == 'assigned') & np.isin(reads.seg_label.to_numpy(), cells)
    x = molecules.voxel_x.astype('Int64').fillna(-1).to_numpy(np.int64)
    y = molecules.voxel_y.astype('Int64').fillna(-1).to_numpy(np.int64)
    off_grid = status == 'outside_grid'
    background = (status != 'assigned') & ~off_grid & (x >= 0) & (y >= 0) & _in_box(x, y, tile)
    pd.concat([reads[assigned], reads[background], reads[off_grid]]).to_csv(path, index=False)


def _assignment_ratio(counts):
    """The ratio line of log.txt: assigned / all molecules of the record's counts, with both counts."""
    if not counts['molecules']:
        return 'assignment ratio: none (no molecule)'
    return (f"assignment ratio: {counts['assigned'] / counts['molecules']:.2%} "
            f"({counts['assigned']} of {counts['molecules']} molecules assigned)")


def _write_assignment_figures(result, image_path, folder, *, ownership=None):
    """expr/{fovID}/assignment.png and, for a ZYX label image, assignment_single_layer.png of an assignment.

    plot_assignment over the DAPI image ``image_path`` (left out when it is
    None or its Y, X are not those of the labels): the Z-maximum view and the
    middle plane. reads_assignment draws them without ``ownership``;
    export_sample draws them again with each FOV's FOVOwnership.
    """
    import matplotlib.pyplot as plt
    import tifffile
    from starfinder.assignment import plot_assignment
    image = None if image_path is None else tifffile.imread(image_path)
    image = image if image is not None and image.shape[-2:] == result.cell_labels.shape[1:] else None
    figures = {'assignment.png': {}}
    n_z = result.cell_labels.shape[0]
    if n_z > 1:  # a ZYX label image; a YX one is a plane with the Z-maximum view only
        layer_image = image if image is None or image.ndim == 2 or image.shape[0] in (1, n_z) else None
        figures['assignment_single_layer.png'] = {'view': 'single_layer', 'image': layer_image}
    Path(folder).mkdir(parents=True, exist_ok=True)
    for name, options in figures.items():
        figure = plot_assignment(result, **{'image': image, **options}, ownership=ownership)
        figure.savefig(Path(folder) / name)
        plt.close(figure)


def _run_reads_assignment(snakemake):
    """reads_assignment: assign_molecules on the legacy label file, written as raw.h5ad and reads_assignment.csv.

    Inputs (as the rule declares them): the sample annotation, the DAPI image
    (diagnostic plot only), images/stardist_segmentation/{fovID}.tif (imported
    with the inferred target on the declared grid, _legacy_cells), the FOV's
    molecule table, which the configuration's producer names (_molecule_table:
    fovs/{fovID}/molecules.csv of the Python rsf_single_fov, read with
    load_molecules and its identities kept; on the legacy route, MATLAB's
    rsf_single_fov and stitch_subtile, the one-based goodSpots CSV read with
    import_legacy_molecules_csv, integer and float coordinates),
    documents/genes.csv (checked against the codebook, _assignment_genes) and
    the tile configuration, read for the sample of the FOV and applied outside
    the package: global coordinates and the overlap filter, as the script does
    (§2.10 boundary); outside_grid molecules are kept in reads_assignment.csv.
    Also writes expr/{fovID}/assignment.png (plot_assignment, Z-maximum view
    over the DAPI image), assignment_single_layer.png (its middle plane) for a
    ZYX label image only, and log.txt with the assignment ratio.

    With an assembly block (option W1 of docs/assembly-contract.md, "Workflow
    configuration") no tile configuration is read: the molecule grid is that
    of the FOV's reference image (reference_image), as Dataset.export reads it
    (_assembly_cells), the per-FOV files are written without the overlap
    filter and without global_* columns, and the result is saved as the FOV's
    assignment folder fovs/{fovID}/assignment/<assembly.assignment>/
    (its record named so), which the export_sample rule assembles.
    """
    from starfinder.assignment import assign_molecules, import_legacy_molecules_csv, load_molecules
    config, fov_id = snakemake.config, snakemake.wildcards.fovID
    assignment, target, translation = _reads_assignment(config)
    block = _assembly_block(config) if 'assembly' in config else None
    sample, number = _fov_sample(snakemake.input[0], config, fov_id)
    output_root = Path(config['root_output_path']) / config['dataset_id'] / config['output_id']
    tile = None if block is not None else _tile_record(_layout.tile_config(output_root, sample), number)
    genes = _assignment_genes(snakemake.input[4], config)
    if block is None:
        grid, cells = _legacy_cells(Path(snakemake.input[2]), config, fov_id, target)
    else:
        grid, cells = _assembly_cells(Path(snakemake.input[2]), output_root, config, fov_id, target)
    legacy = _legacy_molecules(config)
    reader = import_legacy_molecules_csv if legacy else load_molecules
    molecules = reader(snakemake.input[3], spot_namespace=_label_namespace(config, fov_id), genes=genes)
    result = assign_molecules(molecules, cells, grid=grid, config=assignment)
    name = 'reads_assignment' if block is None or block.assignment is None else block.assignment
    result = replace(result, record=dict(result.record, name=name, workflow=translation))
    if block is not None and block.assignment is not None:
        from starfinder.assignment import summarize_assignment
        from starfinder.assignment._persist import assignment_directory, write_assignment
        result = replace(result, record=dict(result.record, counts=summarize_assignment(result)))
        result = write_assignment(assignment_directory(_layout.fov_dir(output_root, fov_id), name), result,
                                  metadata=cells.grid.metadata)
    h5ad, csv_path = Path(snakemake.output[0]), Path(snakemake.output[1])
    written = _write_raw_h5ad(result, h5ad, sample=sample, fov_id=fov_id, tile=tile)
    _write_reads_assignment(result, snakemake.input[3], csv_path, tile=tile, cells=written, legacy=legacy)
    _write_assignment_figures(result, snakemake.input[1], h5ad.parent)
    counts = result.record['counts']
    kept = (f'{len(written)} cells inside the tile box' if block is None else
            f'{len(written)} cells written without the tile filter (export_sample assembles the sample)')
    (h5ad.parent / 'log.txt').write_text(
        f"{counts['assigned']} of {counts['molecules']} molecules assigned to {counts['cells_kept']} cells; "
        f"{counts['unassigned']} unassigned, {counts['excluded_cell']} in excluded cells, "
        f"{counts['outside_grid']} outside the grid; {kept}\n"
        f"{_assignment_ratio(counts)}\n")


def _stitching_position_inputs(config, output_root, sample):
    """The position file the placement rules read, as a list of rule inputs (stitching-py.smk).

    With source stage_file, positions.stage_file or else the sample's
    <OUTPUT>/documents/maf/<sample>.maf, the expression Dataset.stitch's
    table_from_positions uses, so that create_sample_maf is scheduled when the
    file is absent; with source table, the supplied CSV; a regular grid reads no
    file and adds none. A path is passed as the configuration gives it, as the
    package opens it.
    """
    positions = (config.get('stitching') or {}).get('positions') or {}
    source = positions.get('source')
    if source == 'stage_file':
        return [str(positions.get('stage_file') or _layout.stage_file(output_root, sample))]
    if source == 'table' and positions.get('table'):
        return [str(positions['table'])]
    return []


def _reference_image_inputs(config, output_root, fov_ids):
    """The FOVs' reference images as rule inputs: the file reference_image_import names, else reference_image's.

    The default is fovs/{fov_id}/reference.tif (_layout.reference_image_path).
    The rules that read a FOV's reference image (the placement rules,
    reads_assignment with an assembly block, create_nuclei_amplicon_overlay
    and stitch_subtile of the Python backend) declare the file the adapters
    read; a FOV ID '{fovID}' gives the rule's wildcard pattern.
    """
    settings = reference_settings(config)
    return [str(_layout.reference_image_path(output_root, fov, settings['reference_image'],
                                             settings['reference_image_import'])) for fov in fov_ids]


def _reference_image_outputs(config, output_root, fov_ids):
    """The reference images the Python processing rules write: the files reference_image names, as rule outputs.

    None ([]) with reference_image_import: the processing rules never write an
    imported reference image, which every reader reads where it is.
    """
    settings = reference_settings(config)
    if settings['reference_image_import'] is not None:
        return []
    return [str(_layout.reference_image_path(output_root, fov, settings['reference_image'])) for fov in fov_ids]


def _run_record_outputs(output_root, fov_ids, subtiles=None):
    """The FOVs' run records as rule outputs: fovs/<fov>/run.json, or fovs/<fov>/subtile_<n>/run.json per subtile.

    A FOV ID '{fovID}' and a subtile '{n_subtile}' give the rule's wildcard
    patterns.
    """
    if subtiles is None:
        return [str(_layout.run_record(output_root, fov)) for fov in fov_ids]
    return [str(_layout.run_record(output_root, fov, n)) for fov in fov_ids for n in subtiles]


def _assignment_record_outputs(output_root, fov_ids, name):
    """The FOVs' saved assignment records, fovs/<fov>/assignment/<name>/assignment.json; '{fovID}' for the pattern."""
    return [str(_layout.assignment_dir(_layout.fov_dir(output_root, fov), name) / _layout.ASSIGNMENT_RECORD)
            for fov in fov_ids]


def _shared_image_inputs(config, output_root):
    """The inputs of the shared segmentation rules for the Python backend, by input name.

    ``amplicon_img`` of create_nuclei_amplicon_overlay is the FOV's reference
    image (_reference_image_inputs); ``dapi_img`` and ``flamingo_img`` of
    enhance_dapi_with_flamingo are the images that nuclei_registration writes
    for the channels of the additional round ``flamingo`` named ``DAPI`` and
    ``Flamingo``, in the folders of their channel labels. Those two are input
    functions, so that a configuration without such a round fails only when
    the rule is scheduled. The MATLAB backend keeps its own paths
    (common.smk).
    """
    def channel_image(name):
        def image(wildcards):
            channels = [c for c in _workflow_dataset(config).channel_info('flamingo') if c.name == name]
            if len(channels) != 1:
                raise ValueError(f"enhance_dapi_with_flamingo reads the channel named {name!r} of the additional "
                                 f"round 'flamingo'; found {len(channels)}")
            return str(_layout.image_file(output_root, wildcards.fovID, 'flamingo', channels[0].channel))
        return image

    return {'amplicon_img': _reference_image_inputs(config, output_root, ['{fovID}'])[0],
            'dapi_img': channel_image('DAPI'), 'flamingo_img': channel_image('Flamingo')}


def _bigstitcher_folder(block, sample):
    """<bigstitcher_save_dir>/<sample>/: BigStitcher's save folder, with the tiles in its tiles/."""
    if block.method != _BIGSTITCHER:
        raise ValueError(f'the BigStitcher rules run with method bigstitcher; method is {block.method!r}')
    return block.bigstitcher_save_dir / sample


def _write_tile_config(result, path, config):
    """The legacy writer: output/tile_config_{sample}.csv from the import's raw translations and view map.

    legacy_tile_config_view (T3 to T7 of docs/stitching-baseline.md) with the
    configuration's img_col and img_row, written as create_tile_config.py
    writes it (frame.to_csv()); it reads no status, so it is the same for every
    fallback.
    """
    from starfinder.stitching import legacy_tile_config_view
    frame = legacy_tile_config_view(result, img_col=config['img_col'], img_row=config['img_row'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frame.to_csv())


def _read_view_map(path):
    import pandas as pd
    return pd.read_csv(path, dtype={'fov_id': str}, keep_default_na=False)


def _run_stitching(snakemake):
    """stitch_sample: one sample's placement in <OUTPUT>/stitching/{sample}/ (its placement.json is output 0).

    Input 0 is the sample annotation, which gives the sample's FOVs. With an
    in-process method, Dataset.stitch with the block's positions, config,
    input, review and fallback, saved to the output's folder; with a stage
    file the adapter builds the table itself (_sample_table), so that its
    PositionIDs are named by fov_id_pattern.format(i=p). With bigstitcher,
    the import of the engine rule's dataset.xml (input 1) with the view map
    (input 2) and the macro (input 3) of the macro rule, on the table of the
    positions and the FOVs' reference grids (of the file reference_image_import
    or reference_image names, default fovs/{fovID}/reference.tif, as Dataset.stitch reads
    them: the stored metadata, or the grid voxel_size_xy and voxel_size_z
    declare for a file that stores none); the placement is saved there and
    the legacy writer writes output/tile_config_{sample}.csv (output 1) from
    the record.
    """
    from starfinder.stitching import import_bigstitcher, save_placement
    from .dataset._declared import reference_grid
    config, sample = snakemake.config, snakemake.wildcards.sample
    block = _stitching_block(config)
    dataset = _workflow_dataset(config)
    fov_ids = _sample_fovs(snakemake.input[0], config, sample)
    folder = Path(snakemake.output[0]).parent
    if block.method != _BIGSTITCHER:
        table = block.positions
        if table.get('source') == 'stage_file':
            # Named by the workflow's rule here: Dataset.stitch would format the brace pattern with %.
            grids = {fov: reference_grid(dataset.output_root, fov, **reference_settings(config)) for fov in fov_ids}
            table = _sample_table(block, dataset, sample, fov_ids, grids, config)
        dataset.stitch(sample, fov_ids, table=table, config=block.config, input=block.input,
                       review=block.review, fallback=block.fallback, output_dir=folder)
        return
    grids = {fov: reference_grid(dataset.output_root, fov, **reference_settings(config)) for fov in fov_ids}
    table = _sample_table(block, dataset, sample, fov_ids, grids, config)
    result = import_bigstitcher(snakemake.input[1], table=table, view_map=_read_view_map(snakemake.input[2]),
                                grids=grids, config=block.config, review=block.review, fallback=block.fallback,
                                sample=sample, macro=Path(snakemake.input[3]).read_text())
    save_placement(result, folder)
    _write_tile_config(result, snakemake.output[1], config)


def _run_bigstitcher_macro(snakemake):
    """bigstitcher_macro_sample: the tiles, the macro (output 0) and the view map (output 1) of one sample.

    export_bigstitcher on the block's input images of the sample's FOVs (read
    as Dataset.stitch reads them) and its positions table, with the tiles in
    <bigstitcher_save_dir>/<sample>/tiles/ and that folder as BigStitcher's save
    folder.
    """
    from starfinder.stitching import export_bigstitcher
    from .dataset._stitching import read_inputs
    config, sample = snakemake.config, snakemake.wildcards.sample
    block = _stitching_block(config)
    folder = _bigstitcher_folder(block, sample)
    dataset = _workflow_dataset(config)
    fov_ids = _sample_fovs(snakemake.input[0], config, sample)
    loaded = read_inputs(dataset.output_root, fov_ids, block.input, reference=reference_settings(config))
    table = _sample_table(block, dataset, sample, fov_ids, {fov: item.grid for fov, item in loaded.items()}, config)
    text, view_map = export_bigstitcher(table, [loaded[fov] for fov in table.fov_ids], config=block.config,
                                        tile_dir=folder / 'tiles', save_dir=folder)
    Path(snakemake.output[0]).write_text(text)
    view_map.to_csv(snakemake.output[1], index=False)


def _check_bigstitcher_engine(dataset_xml, view_map, returncode=None):
    """The engine rule's success check: dataset.xml exists and holds a Stitching Transform for every view.

    Fiji exits with status 0 when the macro fails, so the check, not the exit
    status, decides (docs/stitching-contract.md, "BigStitcher"). A view is
    mapped to its registration through the ViewSetup tile attribute, as the
    import maps it. Raises RuntimeError naming the path (no file, or not
    readable XML) or the views without a Stitching Transform, with Fiji's exit
    status when given.
    """
    from xml.etree import ElementTree
    from starfinder.stitching._bigstitcher import STITCHING_TRANSFORM, _children, _setup_tiles, _text
    status = '' if returncode is None else f' (Fiji exited with status {returncode})'
    path = Path(dataset_xml)
    if not path.is_file():
        raise RuntimeError(f'the BigStitcher engine task wrote no dataset.xml at {path}{status}')
    try:
        root = ElementTree.parse(path).getroot()
        tiles = _setup_tiles(root, path)
    except (ElementTree.ParseError, ValueError) as error:
        raise RuntimeError(f'{path} is not a dataset.xml the import can read{status}: {error}') from None
    placed = set()
    for registration in _children(root, 'ViewRegistration'):
        setup = registration.get('setup')
        names = {_text(transform, 'Name') for transform in _children(registration, 'ViewTransform')}
        if setup is not None and setup.isdigit() and int(setup) in tiles and STITCHING_TRANSFORM in names:
            placed.add(tiles[int(setup)])
    missing = [int(view) for view in view_map['view'] if int(view) not in placed]
    if missing:
        raise RuntimeError(f"{path} holds no {STITCHING_TRANSFORM!r} for views {', '.join(map(str, missing))}; the "
                           f'BigStitcher engine task failed{status}')


def _run_bigstitcher_engine(snakemake):
    """bigstitcher_engine: Fiji on the macro (input 0), then the success check against the view map (input 1).

    fiji_path is read here, when the rule runs, not when the rule file is
    parsed; the command is that of the legacy run_BigStitcher_macro rule, with
    the BigStitcher folder as Java's temporary folder. The output dataset.xml
    (output 0) is removed first, so the check reads this run's file.
    """
    import subprocess
    fiji = snakemake.config.get('fiji_path')
    if not fiji:
        raise ValueError('the BigStitcher engine rule needs fiji_path (the Fiji executable); the key is read when the '
                         'rule runs')
    macro, output = Path(snakemake.input[0]), Path(snakemake.output[0])
    output.unlink(missing_ok=True)
    completed = subprocess.run([str(fiji), f'-Djava.io.tmpdir={output.parent}', '--ij2', '--headless', '--run',
                                str(macro)], check=False)
    _check_bigstitcher_engine(output, _read_view_map(snakemake.input[1]), completed.returncode)


def _export_sample_files(config, output_root, fov_ids, sample='{sample}'):
    """(inputs, output) of the export_sample rule for one sample (export.smk builds the rule from them).

    The inputs are the sample's placement stitching/<sample>/placement.json,
    then by route (docs/sample-segmentation-contract.md, "The entry"). On the
    route per FOV, the per-FOV assignment outputs of reads_assignment
    (raw.h5ad, reads_assignment.csv and the saved folder's assignment.json),
    or with assembly.assignment null the molecule tables (on either route;
    _molecule_table: fovs/{fovID}/molecules.csv, or the legacy goodSpots CSV
    of stitch_subtile). On the route after fusion, no reads_assignment output:
    the molecule tables, the reference images (_reference_image_inputs, whose
    grids place the molecules) and, with fuse on, the source images of
    export.images that are files of the workflow (reference_merged, the
    reference image, and each <name>=file:<pattern>); the saved form of
    reference_stain under fovs/<fov>/other_rounds/ is no declared output of
    any rule, so it is read and not declared, as by stitch_sample. The output, export.json,
    is the one declared file: it lists every other file of export/<sample>/.
    """
    block = _assembly_block(config)
    export = _export_config(config)[0] if block.assignment is not None else None
    root = Path(output_root)
    inputs = [str(_layout.stitching_dir(root, sample) / 'placement.json')]
    if block.assignment is None:
        inputs += [str(_molecule_table(config, root, fov)) for fov in fov_ids]
    elif _after_fusion(export, block):
        from .dataset._export import source_path
        inputs += [str(_molecule_table(config, root, fov)) for fov in fov_ids]
        inputs += _reference_image_inputs(config, root, fov_ids)
        # The files Dataset.export's fuse part reads, by the same expression (source_path).
        settings = reference_settings(config)
        files = SimpleNamespace(output_root=root, reference_image=settings['reference_image'],
                                reference_image_import=settings['reference_image_import'])
        for entry in export.fusion.images if export.fuse else ():
            if entry != 'reference_stain':
                inputs += [p for p in (str(source_path(files, fov, entry)) for fov in fov_ids) if p not in inputs]
    else:
        inputs += [str(_layout.expr_dir(root, fov) / 'raw.h5ad') for fov in fov_ids]
        inputs += [str(_layout.expr_dir(root, fov) / 'reads_assignment.csv') for fov in fov_ids]
        inputs += _assignment_record_outputs(root, fov_ids, block.assignment)
    return inputs, str(_layout.export_dir(root, sample) / _layout.EXPORT_FILE)


def _run_export(snakemake):
    """export_sample: Dataset.export of one sample into <OUTPUT>/export/{sample}/ (its export.json is output 0).

    Input 0 is the sample's saved placement, stitching/{sample}/placement.json;
    the other inputs are the per-FOV files export reads, by route
    (_export_sample_files: the assignment outputs of reads_assignment on the
    route per FOV; the molecule tables for molecules only and, with the
    reference and source images, on the route after fusion, where the
    codebook gives the genes, as for molecules only). The parts and their
    settings are those of the export block and the assembly block
    (_export_config); the assignment name is the assembly block's. A part
    that cannot run (an extra not installed, an input missing) is recorded
    not_run in export.json and does not fail the rule.

    After the export on the route per FOV, with an assignment name and an
    assembly in the folder (assemble written or reused), it draws
    expr/{fovID}/assignment.png and, for a ZYX label image,
    assignment_single_layer.png again for each FOV of the sample that has a
    saved assignment (its folder's assignment.json, an input of the rule, and
    an input of the assembly), with the FOV's ownership
    (_write_ownership_figures). They are not declared outputs: reads_assignment
    writes them first, without ownership. On the route after fusion no figure
    is drawn: the sample's cells come from no per-FOV assignment.
    """
    config, sample = snakemake.config, snakemake.wildcards.sample
    export, block = _export_config(config)
    dataset = _export_dataset(config, block, export)
    placement = Path(snakemake.input[0]).parent
    record = dataset.export(sample, placement=placement, assignment=block.assignment, config=export,
                            output_dir=Path(snakemake.output[0]).parent)
    if block.assignment is not None and not _after_fusion(export, block) \
            and record.parts.get('assemble') in ('written', 'reused'):
        output_root = Path(config['root_output_path']) / config['dataset_id'] / config['output_id']
        _write_ownership_figures(output_root, block.assignment, placement, record.folder, snakemake.input)


def _write_ownership_figures(output_root, name, placement_dir, export_dir, inputs):
    """The two assignment figures of every assembled FOV whose saved assignment ``name`` is one of ``inputs``.

    The assembly is read from export/<sample>/ (load_assembly), the placement
    from stitching/<sample>/, each FOV's assignment from
    fovs/{fovID}/assignment/<name>/ and its DAPI image from
    images/DAPI/{fovID}.tif, the reads_assignment input (not an input of
    export_sample: without the file the figures have no image).
    """
    import os
    from starfinder.assembly import fov_ownership, load_assembly
    from starfinder.assignment._persist import RECORD_FILE, assignment_directory, read_assignment
    from starfinder.stitching import load_placement
    declared = {os.path.abspath(p) for p in inputs}
    assembly, placement = load_assembly(export_dir), load_placement(placement_dir)
    fovs = assembly.fovs
    for fov_id in fovs['fov_id'][fovs['kind'].eq('assignment').fillna(False).to_numpy(dtype=bool)]:
        folder = assignment_directory(_layout.fov_dir(output_root, fov_id), name)
        if os.path.abspath(folder / RECORD_FILE) not in declared:
            continue
        result = read_assignment(folder, name)
        image = _layout.image_file(output_root, fov_id, 'DAPI')
        _write_assignment_figures(result, image if image.is_file() else None, _layout.expr_dir(output_root, fov_id),
                                  ownership=fov_ownership(assembly, fov_id, placement=placement))


def _report_sample_files(config, output_root, fov_ids, sample='{sample}'):
    """(inputs, output) of the report_sample rule for one sample (report.smk builds the rule from them).

    The input is the last per-sample output the configuration has (WR1): the
    sample's export/<sample>/export.json with an assembly block (export_sample,
    which an export block configures); else stitching/<sample>/placement.json
    with a stitching block; else the FOVs' fovs/{fovID}/run.json. The
    output is report/<sample>/report.json, the record that lists every other
    file of the report's folder.
    """
    root = Path(output_root)
    if 'assembly' in config:
        inputs = [str(_layout.export_dir(root, sample) / _layout.EXPORT_FILE)]
    elif 'stitching' in config:
        inputs = [str(_layout.stitching_dir(root, sample) / 'placement.json')]
    else:
        inputs = [str(_layout.run_record(root, fov)) for fov in fov_ids]
    return inputs, str(_layout.report_dir(root, sample) / 'report.json')


def _report_dataset_files(output_root, samples):
    """(inputs, output) of the report_dataset rule: every sample's report/<sample>/report.json, and
    report/samples.csv (beside it Dataset.compare_samples writes report/index.html)."""
    root = _layout.report_root(output_root)
    return [str(root / sample / 'report.json') for sample in samples], str(root / 'samples.csv')


def _run_report_sample(snakemake):
    """report_sample: Dataset.report of one sample into <OUTPUT>/report/{sample}/ (its report.json is output 0).

    The FOVs are the sample's of the sample annotation
    (<OUTPUT>/documents/sample-annotation.csv, as common.smk reads them), the
    assignment name that of the assembly block (_report_assignment) and the
    configuration the report block's (_report_block).
    """
    config, sample = snakemake.config, snakemake.wildcards.sample
    report = _report_block(config)
    dataset = _workflow_dataset(config)
    fov_ids = _sample_fovs(_layout.sample_annotation(_output_root(config)), config, sample)
    dataset.report(sample, fov_ids, assignment=_report_assignment(config), config=report,
                   output_dir=Path(snakemake.output[0]).parent)


def _run_report_dataset(snakemake):
    """report_dataset: Dataset.compare_samples of every sample whose report is an input, in the inputs' order, into
    <OUTPUT>/report/ (index.html and samples.csv, output 0)."""
    config = snakemake.config
    _report_block(config)
    samples = [Path(path).parent.name for path in snakemake.input]
    _workflow_dataset(config).compare_samples(samples, report_root=Path(snakemake.output[0]).parent)
