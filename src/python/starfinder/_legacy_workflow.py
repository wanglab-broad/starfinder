"""The adapters of the frozen MATLAB workflow's shared rules (D7; design/specs/orchestration-engines.md, "The MATLAB
legacy separation").

The runners that take the ``snakemake`` object of the shared rules' scripts (``workflow/scripts/``
``create_nuclei_amplicon_overlay.py``, ``enhance_dapi_with_flamingo.py``, ``stardist_segmentation.py`` and
``reads_assignment.py``), the input list ``segmentation.smk`` read for the Python backend, and the writers of the
MATLAB-shaped per-FOV files, moved unchanged from ``starfinder/_snakemake.py`` with the helpers they call that the
``starfinder workflow`` commands do not keep (``_assembly_cells``, ``_reference_image_inputs``). They read the
configuration translation of ``starfinder.dataset.workflow``, which stays there for ``from_workflow_config``. Only those
four scripts and the tests that pin the legacy outputs (``test_assignment_workflow.py``,
``test_workflow_assembly_golden.py``) import this module; the Python workflow (``workflow/Snakefile``,
``workflow/rules/tasks.smk``) and ``starfinder.workflow`` never do (check C8).
"""
# §2.13
import json
from dataclasses import asdict, replace
from pathlib import Path

from starfinder import _layout
from starfinder.dataset._declared import reference_settings
from starfinder.dataset.workflow import (_LEGACY_SEGMENTATION_RUN, _assembly_block, _declared_metadata, _flag,
    _label_namespace, _legacy_molecules, _membership, _reads_assignment, _reference_genes, _rule_parameters,
    _segmentation, _workflow_dataset)


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


def _fov_sample(membership, config, fov_id):
    """(sample, tile-configuration id) of fov_id, from the rule's membership input.

    The Python backend reads fovs.csv (_membership, restricted by
    selected_fovs): the FOV's sample, and its position_id, or without that
    column its 1-based place among the sample's FOVs (the id
    legacy_tile_config_view gives a FOV of a table without PositionIDs). The
    MATLAB backend reads its sample annotation: the row whose
    fov_start..fov_end holds the FOV, the id its FOV number.
    """
    import pandas as pd
    if config.get('backend', 'matlab') == 'python':
        members = _membership(config, membership)
        for sample, fovs in members.samples.items():
            if fov_id in fovs:
                return sample, (fovs.index(fov_id) + 1 if members.position_ids is None
                                else members.position_ids[fov_id])
        raise ValueError(f'{fov_id} is in no sample of {membership}')
    annotation = membership
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
    """<folder>/assignment.png and, for a ZYX label image, assignment_single_layer.png of an assignment.

    plot_assignment over the DAPI image ``image_path`` (left out when it is
    None or its Y, X are not those of the labels): the Z-maximum view and the
    middle plane. reads_assignment draws them without ``ownership``, into
    expr/{fovID}/ without a saved assignment, or into fovs/{fovID}/previews/
    with assembly.previews on; export_sample draws them with each FOV's
    FOVOwnership into export/{sample}/previews/{fovID}/, with assembly.previews
    on.
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

    Inputs (as the rule declares them): the FOV membership (fovs.csv for the
    Python backend, MATLAB's sample annotation; _fov_sample), the DAPI image
    (diagnostic plot only), images/stardist_segmentation/{fovID}.tif (imported
    with the inferred target on the declared grid, _legacy_cells), the FOV's
    molecule table, which the configuration's producer names (_molecule_table:
    fovs/{fovID}/molecules.csv of the Python rsf_single_fov, read with
    load_molecules and its identities kept; on the legacy route, MATLAB's
    rsf_single_fov and stitch_subtile, the one-based goodSpots CSV read with
    import_legacy_molecules_csv, integer and float coordinates), the codebook
    (the direct panel in readout_mode direct) of the sequencing rules, whose
    loaded genes, in its order, are the count matrix's columns
    (_reference_genes: Codebook.genes or DirectPanel.genes), and
    the tile configuration, read for the sample of the FOV and applied outside
    the package: global coordinates and the overlap filter, as the script does
    (at the boundary with the sample assembly); outside_grid molecules are kept in
    reads_assignment.csv.
    Also writes expr/{fovID}/assignment.png (plot_assignment, Z-maximum view
    over the DAPI image), assignment_single_layer.png (its middle plane) for a
    ZYX label image only, and log.txt with the assignment ratio.

    With an assembly block (option W1 of design/specs/assembly.md, "Workflow
    configuration") no tile configuration is read: the molecule grid is that
    of the FOV's reference image (reference_image), as Dataset.export reads it
    (_assembly_cells). With assembly.assignment null (molecules only) the
    per-FOV files above are written without the overlap filter and without
    global_* columns. With an assignment name the result is saved as the FOV's
    assignment folder fovs/{fovID}/assignment/<assembly.assignment>/ (its
    record named so), which the export_sample rule assembles, and nothing else
    is written: no raw.h5ad, reads_assignment.csv or log.txt (the record
    holds the counts), and the two figures only with assembly.previews on,
    into fovs/{fovID}/previews/ (_reads_assignment_outputs).
    """
    # §2.10 boundary
    from starfinder.assignment import assign_molecules, import_legacy_molecules_csv, load_molecules
    config, fov_id = snakemake.config, snakemake.wildcards.fovID
    assignment, target, translation = _reads_assignment(config)
    block = _assembly_block(config) if 'assembly' in config else None
    sample, number = _fov_sample(snakemake.input[0], config, fov_id)
    output_root = Path(config['root_output_path']) / config['dataset_id'] / config['output_id']
    tile = None if block is not None else _tile_record(_layout.tile_config(output_root, sample), number)
    genes = _reference_genes(config, Path(snakemake.input[4]))
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
        if block.previews:
            _write_assignment_figures(result, snakemake.input[1], _layout.fov_previews_dir(output_root, fov_id))
        return
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


def _reference_image_inputs(config, output_root, fov_ids):
    """The FOVs' reference images as rule inputs: the file reference_image_import names, else reference_image's.

    The default is fovs/{fov_id}/reference.tif (_layout.reference_image_path).
    The rules that read a FOV's reference image (the placement rules,
    reads_assignment with an assembly block, create_nuclei_amplicon_overlay
    and stitch_subtile of the Python backend) declare the file the adapters
    read; a FOV ID '{fovID}' gives the rule's wildcard pattern. One path per
    FOV ID: a rule input that names one file takes the list's element
    (common.smk's REFERENCE_IMAGE_INPUT), so that the adapter gets one path.
    """
    settings = reference_settings(config)
    return [str(_layout.reference_image_path(output_root, fov, settings['reference_image'],
                                             settings['reference_image_import'])) for fov in fov_ids]


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
