"""Bounded literal export integration with independent geometry/population checks."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult, checkpoint_reference, load_final_checkpoint, MoleculeIndex
from starfinder.raster import RasterConfig, prepare_rasters
from starfinder.sample_export import SavedFile, SampleExportConfig, export_sample, lookup_export_trace
from test.test_molecular_checkpoints import create


def pin(path):
    return SavedFile(path, hashlib.sha256(path.read_bytes()).hexdigest())


def export_fixture(root, *, case='E7', empty=False, unknown=False, omitted=False):
    root.mkdir(parents=True, exist_ok=True)
    cases = json.loads((Path(__file__).resolve().parents[3]/'docs/examples/sample_export_cases.json').read_text())
    cases = {c['id']:c for c in cases['cases']}
    label_case = 'E1' if case == 'E7' else case
    labels = np.array(cases[label_case]['labels'], dtype=np.uint32)
    if empty:
        labels[:] = 0
    z,y,x = np.indices(labels.shape)
    image = np.stack([100*z+10*y+x,100*z+10*y+x+1000],axis=-1).astype('float32')
    np.save(root/'image.npy', image); np.save(root/'labels.npy', labels)
    metadata = ImageMetadata('sample-frame') if unknown else ImageMetadata('sample-frame',(2,1,1),(10,20,30),tuple(map(tuple,np.eye(3))),'micrometer')
    images = {'reference':ImageLoadResult(image, metadata, ('reference','stain'),(root/'image.npy',))}
    factors = (1,1,1) if case=='E7' else tuple(cases[case]['requested'])
    prepared = prepare_rasters(images, labels, metadata=metadata, declared_ids=[int(v) for v in np.unique(labels) if v],
        config=RasterConfig(factors,coarser_levels=2 if case=='E7' else 0,coordinate_space='index' if unknown else 'physical'))
    cell_rows = [dict(mask_namespace='mask',local_label=int(v),cell_namespace='cell',cell_id=str(v),instance_id=int(v)) for v in np.unique(labels) if v]
    cells = pd.DataFrame(cell_rows,columns=['mask_namespace','local_label','cell_namespace','cell_id','instance_id'])
    cells = cells.astype({'local_label':'int64','instance_id':'uint32'})
    cells.to_parquet(root/'cells.parquet',index=False)
    refs=[];mappings={}
    for name in ['A','B']:
        source=create(root/name, namespace=name,FOV=name,omitted=omitted,variant='empty' if empty else 'normal')
        ref=checkpoint_reference(source[5]);refs.append(ref)
        # Explicit common-frame translation; preserved source coordinates are independent.
        matrix=np.eye(4);matrix[:3,3]=[10,20,30] if not unknown else [0,0,0]
        provenance=root/(name+'-mapping.json')
        provenance.write_text(json.dumps(dict(matrix_zyx=matrix.tolist(),source_frame_id=source[0].metadata.frame_id)))
        mappings[ref.artifact_id]=dict(source_frame_id=source[0].metadata.frame_id,direction='source_to_target',matrix_zyx=matrix.tolist(),provenance_path=str(provenance),provenance_sha256=pin(provenance).sha256)
    index=MoleculeIndex('artifact-contract-v1','sample',None,tuple(refs))
    points=index.read_table()
    links=points[['spot_namespace','spot_id']].copy()
    links['cell_namespace']=pd.Series(['cell',None] if len(links) else [],dtype='string')
    links['cell_id']=pd.Series(['1',None] if len(links) else [],dtype='string')
    links['assignment_status']=pd.Series(['assigned','unassigned'] if len(links) else [],dtype='string')
    links['assignment_reason']=pd.Series(['upstream','outside_mask'] if len(links) else [],dtype='string')
    links.to_parquet(root/'assignments.parquet',index=False)
    import anndata as ad
    obs=pd.DataFrame({'cell_namespace':['cell']*len(cells),'cell_id':cells.cell_id.tolist()},index=[f'cell-{i}' for i in range(len(cells))])
    a=ad.AnnData(np.array([[2,0],[0,0]],dtype=np.int32)[:len(cells)],obs=obs,var=pd.DataFrame(index=['gene-A','gene-B']))
    a.uns['meaning']='supplied counts';a.write_h5ad(root/'expression.h5ad')
    config=SampleExportConfig('sample','sample-frame',mappings,expression_meaning='supplied measured counts',assignment_provenance={'method':'literal','config_sha256':'2'*64})
    kwargs=dict(config=config,raster_sources=(pin(root/'image.npy'),pin(root/'labels.npy')),cell_map=pin(root/'cells.parquet'),
                molecules=index,assignments=pin(root/'assignments.parquet'),expression=pin(root/'expression.h5ad'))
    return prepared,kwargs


def test_export_geometry_populations_links(tmp_path):
    sd=pytest.importorskip('spatialdata')
    import zarr
    prepared,kwargs=export_fixture(tmp_path/'inputs')
    destination=tmp_path/'export'
    export_sample(destination,prepared,**kwargs)
    manifest=json.loads((destination/'export-manifest.json').read_text())
    read=sd.read_zarr(destination/'sample.zarr')
    np.testing.assert_array_equal(read.tables['cell_expression'].X,[[2,0],[0,0]])
    assert read.tables['cell_expression'].var_names.tolist()==['gene-A','gene-B']
    assert read.tables['cell_expression'].obs.instance_id.tolist()==[1,2]
    p=pd.read_parquet(destination/'molecules.parquet')
    assert list(zip(p.spot_namespace,p.spot_id))==[('A','A'),('B','A')]
    np.testing.assert_array_equal(p[['sample_z','sample_y','sample_x']],p[['z','y','x']].to_numpy()+[10,20,30])
    assert p.assignment_status.tolist()==['assigned','unassigned']
    assert p.cell_id.isna().tolist()==[False,True]
    stored_points=read.points['molecules'].compute()
    np.testing.assert_array_equal(stored_points[['source_z','source_y','source_x']],p[['z','y','x']])
    np.testing.assert_array_equal(stored_points[['z','y','x']],p[['sample_z','sample_y','sample_x']])
    trace=lookup_export_trace(destination,spot_namespace='A',spot_id='A')
    np.testing.assert_array_equal(trace['values'],[[0,9],[7,0],[0,0],[0,0]])
    assert manifest['molecule_counts'][0]['rejected']==1
    for role,name in [('images','reference'),('labels','cells')]:
        group=zarr.open_group(str(destination/'sample.zarr'/role/name),mode='r')
        dims=group.attrs['multiscales'][0]['datasets']
        assert dims[1]['coordinateTransformations'][-1]['translation'][-3:]==[10,20.5,30.5]
        assert dims[1]['coordinateTransformations'][0]['scale'][-3:]==[2,2,2]
        expected=np.array([[[1,2]]],dtype=np.uint32) if role=='labels' else np.array([[[[5.5,7.5]]],[[[1005.5,1007.5]]]],dtype=np.float32)
        np.testing.assert_array_equal(group['1'][:],expected)
    with pytest.raises(FileExistsError):export_sample(destination,prepared,**kwargs)


@pytest.mark.parametrize('case',['E1','E2','E4','E5','E9'])
def test_required_raster_cases(tmp_path,case):
    pytest.importorskip('spatialdata')
    import zarr
    prepared,kwargs=export_fixture(tmp_path/'inputs',case=case,empty=case=='E9',unknown=case=='E9')
    export_sample(tmp_path/'export',prepared,**kwargs)
    group=zarr.open_group(str(tmp_path/'export/sample.zarr/labels/cells'),mode='r')
    np.testing.assert_array_equal(group['0'][:],prepared.levels[0].labels)
    if case=='E9':
        assert not any('unit' in a for a in group.attrs['multiscales'][0]['axes'])
        assert pd.read_parquet(tmp_path/'export/molecules.parquet').empty


def test_missing_links_and_omitted_trace(tmp_path):
    pytest.importorskip('spatialdata')
    prepared,kwargs=export_fixture(tmp_path/'inputs',omitted=True)
    kwargs['assignments']=None
    export_sample(tmp_path/'export',prepared,**kwargs)
    points=pd.read_parquet(tmp_path/'export/molecules.parquet')
    assert points.assignment_status.tolist()==['unavailable','unavailable']
    assert lookup_export_trace(tmp_path/'export',spot_namespace='A',spot_id='A')==dict(available=False,reason='candidates_signals_disabled')


@pytest.mark.parametrize('bad',['hash','matrix','duplicate','dangling'])
def test_reject_invalid_sources_and_links(tmp_path,bad):
    pytest.importorskip('spatialdata')
    prepared,kwargs=export_fixture(tmp_path/'inputs')
    if bad=='hash':kwargs['cell_map']=SavedFile(kwargs['cell_map'].path,'0'*64)
    elif bad=='matrix':
        key=next(iter(kwargs['config'].source_mappings))
        kwargs['config'].source_mappings[key]['matrix_zyx']=np.zeros((4,4)).tolist()
    else:
        path=kwargs['assignments'].path
        links=pd.read_parquet(path)
        if bad=='duplicate':links=pd.concat([links,links.iloc[:1]])
        else:links.loc[0,'cell_id']='absent'
        links.to_parquet(path,index=False);kwargs['assignments']=pin(path)
    with pytest.raises(ValueError):export_sample(tmp_path/'export',prepared,**kwargs)
    assert not (tmp_path/'export').exists()


def test_mask_only_table_only_and_complete_relation(tmp_path):
    sd=pytest.importorskip('spatialdata')
    import anndata as ad
    prepared,kwargs=export_fixture(tmp_path/'inputs')
    path=kwargs['expression'].path
    a=ad.read_h5ad(path)
    a.obs.loc['cell-1','cell_id']='table-only'
    a.write_h5ad(path);kwargs['expression']=pin(path)
    export_sample(tmp_path/'export',prepared,**kwargs)
    cells=pd.read_parquet(tmp_path/'export/cell-map.parquet')
    assert cells.expression_status.tolist()==['measured','unavailable']
    unmatched=pd.read_parquet(tmp_path/'export/unmatched-cells.parquet')
    assert unmatched.cell_id.tolist()==['table-only']
    assert sd.read_zarr(tmp_path/'export/sample.zarr').tables['cell_expression'].shape==(1,2)
    kwargs['config']=replace(kwargs['config'],complete_cell_relation=True)
    with pytest.raises(ValueError,match='incomplete'):export_sample(tmp_path/'bad',prepared,**kwargs)


def test_regional_access_reads_fewer_chunks(tmp_path):
    pytest.importorskip('spatialdata')
    import zarr
    metadata=ImageMetadata('index-frame')
    z,y,x=np.indices((9,48,48))
    image=(100*z+10*y+x).astype(np.float32)[...,None]
    labels=np.zeros((9,48,48),dtype=np.uint32)
    np.save(tmp_path/'image.npy',image);np.save(tmp_path/'labels.npy',labels)
    cells=pd.DataFrame({'mask_namespace':pd.Series(dtype='string'),'local_label':pd.Series(dtype='int64'),
        'cell_namespace':pd.Series(dtype='string'),'cell_id':pd.Series(dtype='string'),'instance_id':pd.Series(dtype='uint32')})
    cells.to_parquet(tmp_path/'cells.parquet',index=False)
    prepared=prepare_rasters({'reference':ImageLoadResult(image,metadata,('ramp',),(tmp_path/'image.npy',))},
        labels,metadata=metadata,declared_ids=(),config=RasterConfig(coordinate_space='index'))
    export_sample(tmp_path/'export',prepared,config=SampleExportConfig('sample','index-frame',{},molecules_absent_reason='raster-only diagnostic'),
        raster_sources=(pin(tmp_path/'image.npy'),pin(tmp_path/'labels.npy')),cell_map=pin(tmp_path/'cells.parquet'))
    class CountingStore(zarr.storage.DirectoryStore):
        def __init__(self,path):super().__init__(path);self.reads=[]
        def __getitem__(self,key):
            if key[0].isdigit():self.reads.append(key)
            return super().__getitem__(key)
    store=CountingStore(str(tmp_path/'export/sample.zarr/images/reference/0'))
    array=zarr.open_array(store,mode='r')
    regional=array[:,:1,:4,:4];regional_keys=list(store.reads);store.reads.clear()
    full=array[:];full_keys=list(store.reads)
    np.testing.assert_array_equal(regional,full[:,:1,:4,:4])
    assert len(regional_keys)==1 and len(full_keys)==8
    (tmp_path/'regional-read.json').write_text(json.dumps({'regional_keys':regional_keys,'full_keys':full_keys,'equal':True},indent=2))


def test_separate_stain_role_and_component_tampering(tmp_path):
    pytest.importorskip('spatialdata')
    import zarr
    prepared,kwargs=export_fixture(tmp_path/'inputs',case='E1')
    levels=[]
    for level in prepared.levels:
        source=level.images['reference']
        stain=ImageLoadResult(source.image[...,1:],source.metadata,('selected-stain',),source.source_paths,{'selection':'channel 1'})
        levels.append(replace(level,images={**level.images,'stain':stain}))
    prepared=replace(prepared,levels=tuple(levels))
    export_sample(tmp_path/'export',prepared,**kwargs)
    stain=zarr.open_group(str(tmp_path/'export/sample.zarr/images/stain'),mode='r')
    np.testing.assert_array_equal(stain['0'][:],[[[[1005.5,1007.5]]]])
    assert stain.attrs['omero']['channels'][0]['label']=='selected-stain'
    file=tmp_path/'export/molecules.parquet'
    file.write_bytes(file.read_bytes()+b'tamper')
    with pytest.raises(ValueError,match='hash'):lookup_export_trace(tmp_path/'export',spot_namespace='A',spot_id='A')


def literal_link_fixture(root):
    """Frozen W-168 A7/B7/A8 link oracle, independent of expression recounts."""
    from dataclasses import replace
    from starfinder.barcode import decode_barcodes, filter_reads, WtaDecoderConfig, ReadFilterConfig
    from starfinder.io import save_candidate_checkpoint, save_decoded_checkpoint, save_final_checkpoint
    from test.test_candidate_checkpoints import oracle
    import anndata as ad
    prepared,kwargs=export_fixture(root,unknown=True)
    refs=[];mappings={}
    for namespace,ids,coordinates in [('A',('7','8'),[[0,0,0],[0,1,1]]),('B',('7',),[[0,0,2]])]:
        spots,signals,book,*_=oracle.signal_example(1)
        table=spots.spots.iloc[:len(ids)].copy()
        table['spot_id']=pd.Series(ids,index=table.index,dtype='string')
        table[['z','y','x']]=np.array(coordinates,dtype=float)
        spots=replace(spots,spot_namespace=namespace,spots=table)
        values=np.repeat(signals.values[:1],len(ids),axis=0)
        signals=replace(signals,spot_namespace=namespace,spot_ids=ids,values=values,valid=np.ones((len(ids),2),dtype=bool))
        context=dict(dataset_id='artifact-contract-v1',sample_id='sample',FOV=namespace,run_id='literal-'+namespace,
                     config={'fixture':'sample-export-contract-v1 links'},code={'commit':'literal-input-v1'})
        source_root=root/('literal-'+namespace)
        source=save_candidate_checkpoint(source_root/'candidates',spots,signals,codebook=book,**context)
        decoded=decode_barcodes(signals,book,config=WtaDecoderConfig())
        filtered=filter_reads(decoded,config=ReadFilterConfig())
        saved=save_decoded_checkpoint(source_root/'decoded',spots,decoded,book,candidate_source=checkpoint_reference(source.path),**context)
        final=save_final_checkpoint(source_root/'final',filtered,decoded_source=checkpoint_reference(saved),config={},code=context['code'])
        ref=checkpoint_reference(final);refs.append(ref)
        provenance=root/(namespace+'-literal-map.json');provenance.write_text(json.dumps({'matrix_zyx':np.eye(4).tolist()}))
        mappings[ref.artifact_id]=dict(source_frame_id=spots.metadata.frame_id,direction='source_to_target',matrix_zyx=np.eye(4).tolist(),provenance_path=str(provenance),provenance_sha256=pin(provenance).sha256)
    kwargs['molecules']=MoleculeIndex('artifact-contract-v1','sample',None,tuple(refs))
    kwargs['config']=replace(kwargs['config'],source_mappings=mappings)
    cells=pd.DataFrame({'mask_namespace':['A','B'],'local_label':[1,1],'cell_namespace':['A','B'],'cell_id':['1','1'],'instance_id':np.array([1,2],dtype=np.uint32)})
    cells.to_parquet(root/'literal-cells.parquet',index=False);kwargs['cell_map']=pin(root/'literal-cells.parquet')
    links=pd.DataFrame({'spot_namespace':['B','A','A'],'spot_id':['7','8','7'],'cell_namespace':['B',None,'A'],'cell_id':['1',None,'1'],
                        'assignment_status':['assigned','unassigned','assigned'],'assignment_reason':['supplied','upstream_unassigned','supplied']})
    links.to_parquet(root/'literal-assignments.parquet',index=False);kwargs['assignments']=pin(root/'literal-assignments.parquet')
    expression=ad.read_h5ad(kwargs['expression'].path);expression.obs['cell_namespace']=['A','B'];expression.obs['cell_id']=['1','1']
    expression.write_h5ad(root/'literal-expression.h5ad');kwargs['expression']=pin(root/'literal-expression.h5ad')
    return prepared,kwargs


def test_frozen_three_molecule_link_oracle(tmp_path):
    sd=pytest.importorskip('spatialdata')
    prepared,kwargs=literal_link_fixture(tmp_path/'inputs')
    export_sample(tmp_path/'export',prepared,**kwargs)
    points=pd.read_parquet(tmp_path/'export/molecules.parquet').set_index(['spot_namespace','spot_id'])
    assert set(points.index)=={('A','7'),('B','7'),('A','8')}
    for key,coord in [(('A','7'),[0,0,0]),(('B','7'),[0,0,2]),(('A','8'),[0,1,1])]:
        np.testing.assert_array_equal(points.loc[key,['sample_z','sample_y','sample_x']].to_numpy(float),coord)
    assert points.loc[('A','8'),'assignment_status']=='unassigned'
    assert pd.isna(points.loc[('A','8'),'cell_id'])
    assert points.loc[('B','7'),'cell_namespace']=='B'
    table=sd.read_zarr(tmp_path/'export/sample.zarr').tables['cell_expression']
    np.testing.assert_array_equal(table.X,[[2,0],[0,0]])
    assert table.obs.instance_id.tolist()==[1,2]
    assert lookup_export_trace(tmp_path/'export',spot_namespace='A',spot_id='8')['available']
