"""Sample/section SpatialData export, using one shared prepared raster store.

Optional exporter dependencies are imported only on use. The frozen environment
in ``docs/export-viewer-requirements.txt`` is the supported serialization profile.
"""
from dataclasses import asdict, dataclass
import hashlib
import html
import json
from pathlib import Path
import shutil
import tempfile

import numpy as np
import pandas as pd

from starfinder.io.molecules import MoleculeIndex, load_final_checkpoint
from starfinder.raster import RasterResult

__all__ = ['SavedFile', 'SampleExportConfig', 'export_sample', 'open_sample_viewer',
           'lookup_export_trace']


@dataclass(frozen=True)
class SavedFile:
    """An existing source file pinned by SHA-256; directories are not accepted."""

    path: Path
    sha256: str

    def verify(self):
        """Verify bytes and return a JSON source descriptor."""
        path = Path(self.path).absolute()
        actual = _hash(path)
        if actual != self.sha256:
            raise ValueError(f'source hash mismatch: {path}')
        return dict(path=str(path), sha256=actual, size_bytes=path.stat().st_size)


@dataclass(frozen=True)
class SampleExportConfig:
    """Named common frame and population semantics; no inferred assembly.

    ``source_mappings`` is keyed by final artifact ID. Each record requires
    ``source_frame_id``, ``direction='source_to_target'``, ``matrix_zyx`` and
    ``provenance_sha256``. Matrices map saved FOV voxel centers to the common
    physical or index frame, not to the downsampled export grid.
    """

    sample_id: str
    target_frame_id: str
    source_mappings: dict
    section_id: str | None = None
    expression_meaning: str | None = None
    assignment_provenance: dict | None = None
    complete_cell_relation: bool = False
    molecules_absent_reason: str | None = None


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: x.tolist() if isinstance(x, np.ndarray) else
        x.item() if isinstance(x, np.generic) else str(x)), encoding='utf-8')


def _keys(table, columns, name):
    if not set(columns) <= set(table):
        raise ValueError(f'{name}: missing key columns')
    if table[list(columns)].isna().any().any() or table.duplicated(list(columns)).any():
        raise ValueError(f'{name}: null or duplicate keys')
    for column in columns:
        if column.endswith('namespace') or column in ('cell_id', 'spot_id'):
            if any(not isinstance(v, str) or not v for v in table[column]):
                raise ValueError(f'{name}: nonempty string identities required')
    return list(table[list(columns)].itertuples(index=False, name=None))


def _matrix(record, frame):
    if record.get('source_frame_id') != frame or record.get('direction') != 'source_to_target':
        raise ValueError('missing or contradictory source frame mapping')
    digest = record.get('provenance_sha256', '')
    if len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
        raise ValueError('mapping provenance SHA-256 required')
    if not record.get('provenance_path'):
        raise ValueError('saved mapping provenance path required')
    SavedFile(Path(record['provenance_path']), digest).verify()
    matrix = np.asarray(record.get('matrix_zyx'), dtype=float)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all() or not np.array_equal(matrix[3], [0, 0, 0, 1]):
        raise ValueError('finite affine source-to-target matrix required')
    if abs(np.linalg.det(matrix[:3, :3])) < 1e-15:
        raise ValueError('singular source-to-target matrix')
    return matrix


def _molecules(index, config):
    records, sources, counts = [], [], []
    if index is None:
        if not config.molecules_absent_reason or config.source_mappings:
            raise ValueError('absent molecules require a reason and no mappings')
        return pd.DataFrame(columns=['spot_namespace', 'spot_id', 'z', 'y', 'x',
                                     'sample_z', 'sample_y', 'sample_x']), sources, counts
    if not isinstance(index, MoleculeIndex) or (index.sample_id, index.section_id) != (config.sample_id, config.section_id):
        raise ValueError('molecule sample/section identity mismatch')
    if set(config.source_mappings) != {r.artifact_id for r in index.sources}:
        raise ValueError('exact source mapping coverage required')
    for ref in index.sources:
        saved = load_final_checkpoint(ref.path, sha256=ref.sha256)
        if (saved.artifact['artifact_id'], saved.artifact['run_id']) != (ref.artifact_id, ref.run_id):
            raise ValueError('molecule source identity mismatch')
        if (saved.artifact['sample_id'], saved.artifact['dataset_id']) != (index.sample_id, index.dataset_id):
            raise ValueError('molecule dataset/sample mismatch')
        matrix = _matrix(config.source_mappings[ref.artifact_id], saved.pre_qc.spots.metadata.frame_id)
        table = saved.molecule_table()
        xyz = table[['z', 'y', 'x']].to_numpy(dtype=float)
        if not np.isfinite(xyz).all():
            raise ValueError('nonfinite source points')
        mapped = np.c_[xyz, np.ones(len(xyz))] @ matrix.T
        if not np.allclose((mapped @ np.linalg.inv(matrix).T)[:, :3], xyz, atol=1e-9, rtol=0):
            raise ValueError('point affine inverse mismatch')
        for i, axis in enumerate('zyx'):
            table['sample_' + axis] = mapped[:, i]
        table['final_artifact_id'] = ref.artifact_id
        table['final_source_path'] = str(ref.path)
        table['final_source_sha256'] = ref.sha256
        records.append(table)
        sources.append(dict(path=str(ref.path), sha256=ref.sha256, artifact_id=ref.artifact_id,
                            run_id=ref.run_id, mapping=config.source_mappings[ref.artifact_id]))
        counts.append(dict(artifact_id=ref.artifact_id, final=len(table),
                           pre_qc=len(saved.filtering.table),
                           rejected=len(saved.filtering.table)-len(table),
                           qc_status_counts=saved.filtering.table['accepted'].value_counts().to_dict()
                           if 'accepted' in saved.filtering.table else {}))
    result = pd.concat(records, ignore_index=True)
    _keys(result, ('spot_namespace', 'spot_id'), 'molecules')
    return result, sources, counts


def _geometry(level, physical):
    if physical:
        return np.asarray(level.metadata.spacing_zyx), np.asarray(level.metadata.origin_zyx)
    return np.diag(level.index_to_source_zyx)[:3], level.index_to_source_zyx[:3, 3]


def export_sample(directory: str | Path, rasters: RasterResult, *, config: SampleExportConfig,
                  raster_sources: tuple[SavedFile, ...], cell_map: SavedFile,
                  molecules: MoleculeIndex | None = None, assignments: SavedFile | None = None,
                  expression: SavedFile | None = None) -> Path:
    """Write one new sample export directory, refusing overwrite or partial publication.

    ``rasters`` is the W-169 result from explicitly selected saved sources.
    ``raster_sources`` must cover every image source path and the saved aligned
    mask; caller-owned assembly precedes preparation. ``cell_map`` is Parquet
    with mask_namespace/local_label/cell_namespace/cell_id/instance_id columns.
    ``assignments`` is Parquet with molecule keys, nullable cell keys, explicit
    assignment_status and assignment_reason. Missing rows become unavailable.
    ``expression`` is a saved H5AD with explicit cell keys in obs; its X, var,
    layers and metadata are copied, never recomputed. Table-only rows remain
    accessible through the retained original H5AD and unmatched-cell sidecar.
    """
    import anndata as ad
    import dask.array as da
    from datatree import DataTree
    import spatialdata as sd
    from spatialdata.models import Image3DModel, Labels3DModel, PointsModel, TableModel
    from spatialdata.transformations import Affine, Identity
    import zarr

    destination = Path(directory).absolute()
    if destination.exists():
        raise FileExistsError(destination)
    if not isinstance(rasters, RasterResult) or not rasters.levels:
        raise ValueError('nonempty prepared RasterResult required')
    if not config.sample_id or not config.target_frame_id or config.target_frame_id != rasters.source_metadata.frame_id:
        raise ValueError('explicit matching sample/target frame required')
    source_records = [s.verify() for s in raster_sources]
    if not source_records:
        raise ValueError('saved raster sources required')
    paths = {r['path'] for r in source_records}
    for image in rasters.levels[0].images.values():
        if not image.source_paths or not all(str(Path(p).absolute()) in paths for p in image.source_paths):
            raise ValueError('image source paths require pinned raster_sources')
    if not set(rasters.levels[0].images) <= {'reference', 'stain'} or 'reference' not in rasters.levels[0].images:
        raise ValueError('reference required; only reference/stain roles supported')
    map_ref = cell_map.verify()
    cells = pd.read_parquet(cell_map.path)
    cell_keys = _keys(cells, ('cell_namespace', 'cell_id'), 'cells')
    _keys(cells, ('mask_namespace', 'local_label'), 'mask map')
    _keys(cells, ('instance_id',), 'instance map')
    for column in ('local_label', 'instance_id'):
        if cells[column].dtype.kind not in 'ui' or (cells[column] <= 0).any():
            raise ValueError('positive integer label IDs required')
    if (cells.instance_id > np.iinfo(np.uint32).max).any():
        raise ValueError('label IDs exceed uint32')
    for level in rasters.levels:
        if set(np.unique(level.labels)) - {0} != set(cells.instance_id):
            raise ValueError('cell map must exactly cover every raster level')
    points, molecular_sources, counts = _molecules(molecules, config)
    molecule_keys = _keys(points, ('spot_namespace', 'spot_id'), 'molecules')
    if {'cell_namespace', 'cell_id', 'assignment_status', 'assignment_reason'} & set(points):
        raise ValueError('ambiguous preexisting cell assignment columns')
    original = None
    expression_ref = None
    expression_keys = []
    if expression is not None:
        expression_ref = expression.verify()
        original = ad.read_h5ad(expression.path)
        expression_keys = _keys(original.obs, ('cell_namespace', 'cell_id'), 'expression')
    known_cells = set(cell_keys) | set(expression_keys)
    assignment_ref = None
    if assignments is not None:
        assignment_ref = assignments.verify()
        if not config.assignment_provenance or not config.assignment_provenance.get('method') or not config.assignment_provenance.get('config_sha256'):
            raise ValueError('upstream assignment method/config provenance required')
        links = pd.read_parquet(assignments.path)
        link_keys = _keys(links, ('spot_namespace', 'spot_id'), 'assignments')
        if not set(link_keys) <= set(molecule_keys):
            raise ValueError('dangling molecule assignment')
        required = ['cell_namespace', 'cell_id', 'assignment_status', 'assignment_reason']
        if not set(required) <= set(links):
            raise ValueError('assignment status/reason and cell keys required')
        for row in links.itertuples():
            nulls = (pd.isna(row.cell_namespace), pd.isna(row.cell_id))
            if nulls[0] != nulls[1]:
                raise ValueError('partial cell key')
            if row.assignment_status == 'assigned':
                if nulls[0] or (row.cell_namespace, row.cell_id) not in known_cells:
                    raise ValueError('dangling cell assignment')
            elif row.assignment_status not in ('unassigned', 'unavailable') or not nulls[0] or pd.isna(row.assignment_reason) or not row.assignment_reason:
                raise ValueError('invalid assignment missingness')
        points = points.merge(links[['spot_namespace', 'spot_id'] + required],
                              on=['spot_namespace', 'spot_id'], how='left', validate='one_to_one', sort=False)
    else:
        for key in ('cell_namespace', 'cell_id', 'assignment_status', 'assignment_reason'):
            points[key] = pd.Series([None]*len(points), dtype='string')
    absent = points.assignment_status.isna()
    points.loc[absent, 'assignment_status'] = 'unavailable'
    points.loc[absent, 'assignment_reason'] = 'assignment_not_supplied'
    for column in ('cell_namespace', 'cell_id', 'assignment_status', 'assignment_reason'):
        points[column] = points[column].astype('string')
    measured = None
    unmatched = pd.DataFrame(columns=['cell_namespace', 'cell_id', 'reason'])
    cells = cells.copy()
    cells['expression_status'] = 'unavailable'
    if config.complete_cell_relation and expression is None and len(cells):
        raise ValueError('complete cell relation requires expression')
    if expression is not None:
        if not config.expression_meaning:
            raise ValueError('expression X meaning required')
        if not original.obs_names.is_unique or not original.var_names.is_unique:
            raise ValueError('unique expression observation/gene identities required')
        matched = [i for i, key in enumerate(expression_keys) if key in cell_keys]
        missing = [key for key in expression_keys if key not in cell_keys]
        if config.complete_cell_relation and (missing or set(expression_keys) != set(cell_keys)):
            raise ValueError('incomplete declared mask/expression relation')
        unmatched = pd.DataFrame(missing, columns=['cell_namespace', 'cell_id'])
        unmatched['reason'] = 'no_mask_instance'
        if {'region', 'instance_id'} & set(original.obs):
            raise ValueError('expression has reserved region/instance_id columns')
        measured = original[matched].copy()
        measured.obs['region'] = pd.Categorical(['cells'] * len(matched))
        id_map = dict(zip(cell_keys, cells.instance_id))
        measured.obs['instance_id'] = np.array([int(id_map[expression_keys[i]]) for i in matched], dtype=np.uint32)
        if len(measured):
            measured = TableModel.parse(measured, region='cells', region_key='region', instance_key='instance_id')
        cells.loc[[k in expression_keys for k in cell_keys], 'expression_status'] = 'measured'
    physical = rasters.config.coordinate_space == 'physical'
    base_scale, base_origin = _geometry(rasters.levels[0], physical)
    affine = np.eye(4); affine[:3, :3] = np.diag(base_scale); affine[:3, 3] = base_origin
    transform = {config.target_frame_id: Affine(affine, input_axes=('z','y','x'), output_axes=('z','y','x'))}
    def element(role, name=None):
        nodes = {}
        for i, level in enumerate(rasters.levels):
            is_image = role == 'images'
            array = np.moveaxis(level.images[name].image, -1, 0) if is_image else level.labels.astype('uint32')
            chunks = tuple(min(n, c) for n, c in zip(array.shape, (1,8,32,32) if is_image else (8,32,32)))
            kwargs = dict(dims=('c','z','y','x') if is_image else ('z','y','x'), transformations=transform)
            if is_image:
                kwargs['c_coords'] = list(level.images[name].channel_labels)
            nodes[f'scale{i}'] = (Image3DModel if is_image else Labels3DModel).parse(da.from_array(array, chunks=chunks), **kwargs)
        return next(iter(nodes.values())) if len(nodes) == 1 else DataTree.from_dict(nodes)
    images = {name: element('images', name) for name in rasters.levels[0].images}
    point_elements = {}
    if len(points):
        point_elements['molecules'] = PointsModel.parse(points.rename(columns={a:'source_'+a for a in 'zyx'}), coordinates={a:'sample_'+a for a in 'zyx'},
                                                       transformations={config.target_frame_id: Identity()})
    data = sd.SpatialData(images=images, labels={'cells':element('labels')}, points=point_elements,
                         tables={} if measured is None or not len(measured) else {'cell_expression':measured})
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix='.sample-export-', dir=destination.parent))
    try:
        store = staging/'sample.zarr'
        data.write(store, consolidate_metadata=False)
        groups = []
        for role, names in [('images', list(images)), ('labels', ['cells'])]:
            for name in names:
                group = zarr.open_group(str(store/role/name), mode='a')
                multiscales = group.attrs['multiscales']
                for axis in multiscales[0]['axes']:
                    if axis['type'] == 'space' and physical:
                        axis['unit'] = rasters.source_metadata.spatial_unit
                for i, dataset in enumerate(multiscales[0]['datasets']):
                    scale, origin = _geometry(rasters.levels[i], physical)
                    # NGFF datasets carry absolute calibrated geometry for Fiji.
                    # SpatialData's top-level transform retains base-to-sample;
                    # our explicit-level adapter uses these dataset maps directly.
                    if role == 'images':
                        scale = np.r_[1, scale]; origin = np.r_[0, origin]
                    dataset['coordinateTransformations'] = [dict(type='scale', scale=scale.tolist()),
                                                            dict(type='translation', translation=origin.tolist())]
                    array = group[dataset['path']]
                    metadata = json.loads((store/role/name/dataset['path']/'.zarray').read_text())
                    expected_codec = dict(id='blosc', cname='lz4', clevel=5, shuffle=1, blocksize=0)
                    expected_chunks = [min(n, c) for n, c in zip(array.shape, (1,8,32,32) if role=='images' else (8,32,32))]
                    if metadata['chunks'] != expected_chunks:
                        raise ValueError('unsupported writer chunk layout')
                    if not (metadata['compressor'] == expected_codec and metadata['order']=='C' and
                            metadata['dimension_separator']=='/' and metadata['zarr_format']==2 and metadata['fill_value']==0):
                        raise ValueError('unsupported writer codec/layout')
                    expected = np.moveaxis(rasters.levels[i].images[name].image, -1, 0) if role=='images' else rasters.levels[i].labels
                    if not np.array_equal(array[:], expected):
                        raise ValueError('writer changed prepared raster values')
                group.attrs['multiscales'] = multiscales
                groups.append(dict(path=f'sample.zarr/{role}/{name}', axes=[a['name'] for a in multiscales[0]['axes']],
                                   levels=multiscales[0]['datasets'], unit=rasters.source_metadata.spatial_unit))
        points.to_parquet(staging/'molecules.parquet', index=False)
        cells.to_parquet(staging/'cell-map.parquet', index=False)
        unmatched.to_parquet(staging/'unmatched-cells.parquet', index=False)
        manifest = dict(schema_name='starfinder.sample_export', schema_version=1, config=asdict(config),
            raster_sources=source_records, cell_map_source=map_ref, molecular_sources=molecular_sources,
            assignment_source=assignment_ref, expression_source=expression_ref, molecule_counts=counts,
            final_count=len(points), assignment_counts=points.assignment_status.value_counts().to_dict(),
            cell_count=len(cells), table_only_count=len(unmatched), expression_available=expression is not None,
            requested_raster_config=asdict(rasters.config), raster_diagnostics=rasters.diagnostics,
            source_metadata=asdict(rasters.source_metadata),
            levels=[dict(factors_zyx=l.factors_zyx, metadata=asdict(l.metadata), index_to_source_zyx=l.index_to_source_zyx,
                images={k:dict(channels=v.channel_labels, sources=[str(p) for p in v.source_paths], diagnostics=v.diagnostics)
                        for k,v in l.images.items()}) for l in rasters.levels],
            viewer_qualification='unqualified: Fiji N5 group reopening remains blocked',
            limitations=['Use open_sample_viewer for explicit levels; default napari multiscale center translations are incorrect.',
                         'Automatic zoom level switching unsupported; Fiji has no molecular/expression integration.',
                         'Pinned Fiji N5 Viewer group parsing is not qualified; opening attempts failed.'])
        _json(staging/'open-in-fiji.json', dict(groups=groups, instruction='Open each group in Fiji N5 Viewer; molecular tables are excluded.'))
        (staging/'README.html').write_text('<!doctype html><html lang="en"><meta charset="utf-8"><title>Sample export</title>'
            '<h1>Sample export</h1><p>Use starfinder.sample_export.open_sample_viewer for calibrated explicit-level napari viewing.</p>'
            '<p>In pinned Fiji, open these image/mask groups with N5 Viewer:</p><pre>' + html.escape('\n'.join(str(destination/g['path']) for g in groups))+
            '</pre><p>Retain original H5AD and checkpoint sources. See export-manifest.json for their hashes and missingness.</p></html>')
        manifest['outputs'] = [dict(path=str(p.relative_to(staging)), sha256=_hash(p), size_bytes=p.stat().st_size)
                               for p in sorted(staging.rglob('*')) if p.is_file()]
        manifest['payload_bytes'] = sum(r['size_bytes'] for r in manifest['outputs'])
        _json(staging/'export-manifest.json', manifest)
        # Recheck source bytes before publication. Checkpoint readers validate their components.
        for source in (*raster_sources, cell_map, assignments, expression):
            if source is not None:
                source.verify()
        for source in molecular_sources:
            load_final_checkpoint(source['path'], sha256=source['sha256'])
        staging.rename(destination)
    except BaseException:
        shutil.rmtree(staging)
        raise
    return destination/'export-manifest.json'


def _read_manifest(root):
    manifest = json.loads((root/'export-manifest.json').read_text())
    if manifest.get('schema_name') != 'starfinder.sample_export' or manifest.get('schema_version') != 1:
        raise ValueError('unsupported export manifest')
    for record in manifest['outputs']:
        path = (root/record['path']).resolve()
        if not path.is_relative_to(root.resolve()) or _hash(path) != record['sha256']:
            raise ValueError('export component path/hash mismatch')
    return manifest


def lookup_export_trace(directory: str | Path, *, spot_namespace: str, spot_id: str) -> dict:
    """Resolve a retained final identity through its verified original checkpoint."""
    root = Path(directory)
    manifest = _read_manifest(root)
    rows = pd.read_parquet(root/'molecules.parquet')
    rows = rows[(rows.spot_namespace == spot_namespace) & (rows.spot_id == spot_id)]
    if len(rows) != 1:
        raise ValueError('missing or ambiguous exported molecule')
    row = rows.iloc[0]
    source = next(s for s in manifest['molecular_sources'] if s['artifact_id'] == row.final_artifact_id)
    saved = load_final_checkpoint(source['path'], sha256=source['sha256'])
    return saved.pre_qc.source_trace(spot_namespace=spot_namespace, spot_id=spot_id)


def open_sample_viewer(directory: str | Path, *, level: int = 0):
    """Open explicit stored levels using public napari transforms; return Viewer.

    Each level/channel is selectable in the layer list. Only the requested level
    is initially visible. Points remain in the common frame. Cell-key and
    expression metadata are attached to labels; use the retained SpatialData
    table for expression selection. Automatic zoom switching is unsupported.
    Call ``napari.run()`` for an interactive event loop. The store is read-only.
    """
    import dask.array as da
    import napari
    import spatialdata as sd
    import zarr
    root = Path(directory)
    manifest = _read_manifest(root)
    if not 0 <= level < len(manifest['levels']):
        raise ValueError('stored level out of range')
    data = sd.read_zarr(root/'sample.zarr')
    viewer = napari.Viewer()
    viewer.window.resize(1600, 900)
    from napari_spatialdata import Interactive
    from qtpy.QtWidgets import QPlainTextEdit
    interactive = Interactive(data, headless=True)
    details = QPlainTextEdit()
    details.setReadOnly(True)
    details.setPlainText('Select a label ID to inspect its cell/expression row, or a molecule to resolve its source trace.')
    viewer.window.add_dock_widget(details, name='Cell and molecule source', area='right')
    # Keep the SpatialData UI alive while explicit-level layers use public transforms.
    viewer.layers.events.inserted.connect(lambda event: interactive)
    cells = pd.read_parquet(root/'cell-map.parquet')
    def show_cell(layer):
        selected = cells[cells.instance_id == layer.selected_label]
        record = {'instance_id': int(layer.selected_label), 'cells': selected.to_dict('records')}
        table = data.tables.get('cell_expression')
        if table is not None:
            rows = table[table.obs.instance_id == layer.selected_label]
            values = rows.X.toarray() if hasattr(rows.X, 'toarray') else np.asarray(rows.X)
            record['expression'] = values.tolist()
            record['genes'] = rows.var_names.tolist()
        layer.metadata['selected_cell'] = record
        details.setPlainText(json.dumps(record, indent=2))
    for role, names in [('images', list(data.images)), ('labels', list(data.labels))]:
        for name in names:
            group = zarr.open_group(str(root/'sample.zarr'/role/name), mode='r')
            for i, dataset in enumerate(group.attrs['multiscales'][0]['datasets']):
                transforms = dataset['coordinateTransformations']
                scale, translate = transforms[0]['scale'][-3:], transforms[1]['translation'][-3:]
                array = da.from_zarr(group[dataset['path']])
                kwargs = dict(scale=scale, translate=translate, visible=i==level,
                              metadata={'spatialdata':data, 'level':i, 'frame':manifest['config']['target_frame_id']})
                if role == 'images':
                    channels = [c['label'] for c in group.attrs['omero']['channels']]
                    for channel, label in enumerate(channels):
                        viewer.add_image(array[channel], name=f'{name}/{label}/level-{i}', **kwargs)
                else:
                    kwargs['metadata']['cell_map'] = cells
                    label_layer = viewer.add_labels(array, name=f'cells/level-{i}', **kwargs)
                    label_layer.events.selected_label.connect(lambda event, layer=label_layer: show_cell(layer))
    points = pd.read_parquet(root/'molecules.parquet')
    if len(points):
        point_layer = viewer.add_points(points[['sample_z','sample_y','sample_x']].to_numpy(), name='molecules/final',
                          features=points, size=1, metadata={'spatialdata':data, 'export_path':str(root)})
        def show_molecule(*event):
            selected = []
            for i in point_layer.selected_data:
                row = points.iloc[i]
                trace = lookup_export_trace(root, spot_namespace=row.spot_namespace, spot_id=row.spot_id)
                selected.append(dict(spot_namespace=row.spot_namespace, spot_id=row.spot_id,
                                     assignment_status=row.assignment_status, trace=trace))
            point_layer.metadata['selected_molecules'] = selected
            details.setPlainText(json.dumps(selected, indent=2, default=str))
        point_layer.selected_data.events.items_changed.connect(show_molecule)
    return viewer
