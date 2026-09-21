"""Offline qualification packet, rendered solely from pinned saved evidence."""
from collections import defaultdict
from datetime import datetime, timezone
import html
import json
import os
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from starfinder.io import load_image_checkpoint, load_candidate_checkpoint, load_final_checkpoint
from starfinder.provenance import read_run
from starfinder.reporting import embed_figure, render_table, write_run_summary
from saved_synthetic import checksum, read_json, verify_files, write_json
from saved_synthetic_report import decoding_rows, inspection_figure


def details(title, value):
    return '<details><summary>'+html.escape(title)+'</summary><pre>'+html.escape(json.dumps(value, indent=2))+'</pre></details>'


def trajectory(context):
    """Milestone-colored dependency nodes, separate actual intervals and coverage."""
    nodes = context['issues']
    ids = [n['id'] for n in nodes]
    colors = {'foundation': '#236b9a', 'synthetic': '#257d62', 'human': '#9a6231'}
    fig, ax = plt.subplots(figsize=(15, 8), layout='constrained')
    # Node positions follow actual first execution, while arrows encode prerequisites.
    positions = {identity: (i % 6, -(i//6)*2) for i, identity in enumerate(ids)}
    edges = []
    for node in nodes:
        for source in node['dependencies']:
            if source in positions:
                a, b = positions[source], positions[node['id']]
                ax.annotate('', xy=b, xytext=a, arrowprops=dict(arrowstyle='->',
                    color='#738396', alpha=.65, shrinkA=24, shrinkB=24,
                    connectionstyle='arc3,rad=.12', lw=1))
                edges.append(dict(source=source, target=node['id'], relation='prerequisite'))
    for node in nodes:
        x, y = positions[node['id']]
        ax.text(x, y, node['id']+'\n'+node['topic'], ha='center', va='center', color='white',
                fontsize=9, bbox=dict(boxstyle='round,pad=.55', fc=colors[node['group']], ec='white'))
    ax.set_xlim(-.6, 5.6); ax.set_ylim(-5, .8); ax.axis('off')
    ax.set_title('Issue dependencies · nodes arranged by first recorded execution / gate placement')
    ax.legend(handles=[Patch(facecolor=c, label=n) for n,c in colors.items()], loc='lower center', ncol=3)
    output = embed_figure(fig, 'Milestone and topic colored issue nodes with prerequisite arrows')
    fig, ax = plt.subplots(figsize=(15, 3.4), layout='constrained')
    executed = [n for n in nodes if n['id'] != 'W-174']
    for row, group in enumerate((executed[:10], executed[10:])):
        for index, node in enumerate(group):
            ax.text(index, -row, node['id'], ha='center', va='center', color='white',
                    bbox=dict(boxstyle='round,pad=.5',fc=colors[node['group']],ec='white'),fontsize=10)
            if index:
                ax.annotate('',xy=(index-.3,-row),xytext=(index-1+.3,-row),
                            arrowprops=dict(arrowstyle='->',color='#dc6d41',linestyle='--'))
    ax.set(xlim=(-.6,9.6),ylim=(-1.8,.8),title='Actual issue order · dashed arrows · batch 2 follows explicit W-173 approval')
    ax.axis('off')
    output += embed_figure(fig,'Actual execution order from batch 1 through review repair and approved batch 2')
    output += '<p>Solid arrows are prerequisites, not elapsed work. Actual sequential execution is listed below; human gates are brown. Related-to links are in the accessible table and do not imply prerequisites.</p>'
    output += render_table([dict(issue=n['id'], topic=n['topic'], milestone=n['milestone'],
        state=n['status'], prerequisites=', '.join(n['dependencies']), related=', '.join(n['related'])) for n in nodes])
    phases = {'Implement':'#3477ad', 'Validate':'#30986c', 'Review':'#9861b0',
              'Repair':'#dc6d41', 'Finalize':'#aa9131', 'Human wait':'#ddd0bf', 'Setup pause':'#aaaaaa'}
    intervals = context['intervals']
    for title, selected in [('Batch 1 and review repairs', [x for x in intervals if x['issue'] not in ['W-'+str(i) for i in range(161,168)]]),
                             ('Batch 2 actual execution', [x for x in intervals if x['issue'] in ['W-'+str(i) for i in range(161,168)]])]:
        rows = list(dict.fromkeys(x['issue'] for x in selected))
        fig, ax = plt.subplots(figsize=(15, max(3, .48*len(rows)+1.5)), layout='constrained')
        for record in selected:
            start = mdates.date2num(datetime.fromisoformat(record['start']))
            end = mdates.date2num(datetime.fromisoformat(record['end']))
            y = rows.index(record['issue'])
            ax.barh(y, end-start, left=start, height=.65, color=phases[record['phase']])
            if record.get('exit_code') not in (0, None):
                ax.plot(end,y,'x',color='red')
        ax.set_yticks(range(len(rows)), rows); ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
        ax.set(title=title, xlabel='21 September 2026 · UTC; red × = failed invocation/check')
        ax.grid(axis='x', alpha=.2)
        ax.legend(handles=[Patch(facecolor=c,label=n) for n,c in phases.items()], ncol=4, fontsize=8)
        output += embed_figure(fig, title+' intervals, phases and failures')
    output += '<p>W-173 waiting spans packet delivery to explicit approval and overlaps repairs; it is not compute time. W-174 is pending with no completed wait duration. Unannotated gaps have unknown causes. W-167 ends at this evidence snapshot, not worker completion. Checks inside implementation overlap that interval and must not be added to it.</p>'
    output += '<h3>Unique-session cumulative usage coverage</h3>'+render_table(context['usage_rows'])
    output += '<p>Latest observed cumulative counter counted once per session. Cached input is a subset of input; reasoning is a subset of output. Missing coverage is unknown, never zero. W-175 repair usage, active W-167 usage and outer coordination are unavailable here. Requested models/efforts do not prove serving identity. No monetary inference.</p>'
    output += '<details><summary>Accessible actual-order / attempt / phase table</summary>'+render_table(pd.DataFrame([
        {k:v for k,v in row.items() if k not in ('command','session')} for row in intervals]).fillna('unavailable'))+'</details>'
    return output


def gallery(presets, qualification):
    rows = {c['case']:c for c in qualification['cases']}
    result = []
    for condition in ['clean','brightness','axial_width','lateral_width','elongation','placement',
                      'dropout','weakening','trend','loss','gain','mixing','baseline','gradient','regions',
                      'texture','dependent_noise','independent_noise','translation','local','combined']:
        root = presets/('small-'+condition)
        saved = load_image_checkpoint(root/'prepared')
        histories = pd.read_parquet(root/'round-truth.parquet')
        layer = saved.layers[1]
        fig, axes = plt.subplots(1,4,figsize=(14,3.4),layout='constrained')
        for c, ax in enumerate(axes):
            view = ax.imshow(layer.loaded.image[4,:,:,c], origin='lower', cmap='viridis', vmin=-1, vmax=16)
            ax.set(title=layer.loaded.channel_labels[c], xlabel='X index', ylabel='Y index')
            for row in histories[histories.round_label == 'round2'].itertuples():
                ax.plot(row.x,row.y,'+',color='#ff8555')
                ax.annotate(row.amplicon_id,(row.x,row.y),xytext=(3,5),textcoords='offset points',color='white',fontsize=8)
        fig.colorbar(view, ax=axes, label='Stored intensity: fixed −1 to 16', shrink=.8)
        fig.suptitle(condition+' · round2 · XY z=4 · full truth marked even without emission')
        result.append('<h3 id="case-'+condition+'">'+condition+'</h3>'+embed_figure(fig, condition+' saved channel-specific images and truth'))
        result.append('<p>Independent full-grid audit: pass; max absolute error '+f"{rows['small-'+condition]['max_absolute_error']:.3g}"+'.</p>')
    result.append('<h3>Combined geometry in Z=1 and wider 3D field</h3>')
    for size in ('z1','wide'):
        root = presets/(size+'-combined')
        layer = load_image_checkpoint(root/'prepared').layers[1]
        image = layer.loaded.image
        fig, axes=plt.subplots(1,4,figsize=(14,3.4),layout='constrained')
        for c,ax in enumerate(axes):
            ax.imshow(image[image.shape[0]//2,:,:,c],vmin=-1,vmax=16,cmap='viridis',origin='lower')
            ax.set(title=size+' / '+layer.loaded.channel_labels[c],xlabel='X index',ylabel='Y index')
        result.append(embed_figure(fig,size+' combined channel slices; singleton Z remains sampled 3D'))
    return ''.join(result)


def representative(presets):
    fig, axes = plt.subplots(2, 2, figsize=(9, 7), layout='constrained')
    for i, condition in enumerate(('clean', 'combined')):
        root = presets/('small-'+condition)
        layer = load_image_checkpoint(root/'prepared').layers[1]
        truth = pd.read_parquet(root/'round-truth.parquet')
        for c, ax in enumerate(axes[i]):
            view = ax.imshow(layer.loaded.image[4,:,:,c], vmin=-1,vmax=16,cmap='viridis',origin='lower')
            ax.set(title=condition+' / '+layer.loaded.channel_labels[c],xlabel='X index',ylabel='Y index')
            for row in truth[truth.round_label=='round2'].itertuples():
                ax.plot(row.x,row.y,'+',color='#ff8555')
                ax.annotate(row.amplicon_id,(row.x,row.y),xytext=(3,5),textcoords='offset points',color='white',fontsize=8)
    fig.colorbar(view,ax=axes,label='Stored intensity · fixed −1 to 16',shrink=.7)
    fig.suptitle('Representative saved outputs · round2, XY z=4 · two of four channels')
    return embed_figure(fig,'Clean and combined saved outputs with explicit gt-A/gt-B truth overlays')


def render(root, destination):
    context = read_json(root/'review-context.json')
    for path, digest in context['pinned_inputs'].items():
        assert checksum(path) == digest, path
    presets, processing = Path(context['presets']), Path(context['processing'])
    qualification = read_json(root/'qualification.json')
    manifest = read_json(presets/'manifest.json')
    assert qualification['preset_manifest_sha256'] == checksum(presets/'manifest.json')
    assert len(qualification['cases']) == 63 and all(c['status']=='pass' for c in qualification['cases'])
    for name,digest in manifest['files'].items():
        assert checksum(presets/name) == digest
    delivery = read_json(processing/'saved/delivery.json')
    verify_files(processing/'saved', delivery)
    for name,digest in read_json(processing/'inputs.json')['sources'].items():
        assert checksum(processing/name) == digest
    repeat = [read_json(root/f'repeat-{seed}.json') for seed in (1,999)]
    assert repeat[0]['cases'] == repeat[1]['cases']
    assert repeat[0]['process'] != repeat[1]['process']
    assert all(r['preset_manifest_sha256'] == checksum(presets/'manifest.json') for r in repeat)
    if destination.exists():
        raise FileExistsError('Preserve previous review packets')
    css = 'body{font:16px/1.55 system-ui;color:#203047;max-width:1400px;margin:32px auto;padding:0 24px}h1{font-size:2.4rem}h2{margin-top:48px;border-bottom:2px solid #deebf4}h3{margin-top:30px}table{border-collapse:collapse;display:block;overflow:auto;margin:18px 0;font-size:.9rem}td,th{padding:8px 12px;border:1px solid #d8e1e8;text-align:left}th,nav{background:#eef4f8}img{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;background:#f3f6f8;padding:16px}nav{padding:16px}nav a{margin-right:18px;display:inline-block}a{color:#14598d}details{margin:16px 0}summary{cursor:pointer;font-weight:600}.lead{font-size:1.15rem}'
    sections=[]
    def section(identity,title,body):
        sections.append((identity,title,body))
    section('outcome','Outcome and requested decision',
        '<p class="lead">All 63 controlled development presets pass independent image/truth checks and exact cross-process reproduction. Clean 3D and Z=1 processing retain explicit truth correspondence, decoded calls and source traces. The deliberate TPS failure remains failed with partial extraction and unavailable final counts.</p>'
        '<p><strong>Jiahao:</strong> after controller checks and revision pinning, approve this identified packet/revision or request changes in W-174. W-174 stays open. This implementation snapshot is uncommitted; full controller gates are pending. Technical qualification does not approve another batch.</p>'
        '<p>These are bounded software-development fixtures, not calibrated D04, biological RNA truth, method-accuracy evidence or historical v1/v2 qualification. Source '+html.escape(context['code']['commit'][:12])+' plus the source snapshot in the appendix.</p>'
        +render_table(context['acceptance'])+representative(presets))
    section('settings','Fixed settings and controlled comparisons',
        '<p>ZYXC float32 images; float64 NCR truth; ZYX voxel-index coordinates; physical calibration unknown. Rounds round10 → round2 → round1; channels ch02 → ch00 → ch03 → ch01. Mapping 1→1, 2→0, 3→3, 4→2. gt-A=123/gene-A and gt-B=214/gene-B. Root seed 42, development scene controlled-development-v1.</p>'
        +render_table(context['parameters'])
        +render_table([dict(condition=c['name'], factors=', '.join(c['factors']) or 'none', shape=c['shape'], seconds=c['seconds'], bytes=c['bytes']) for c in manifest['cases'] if c['name'].startswith('small-')])
        +'<p>Preset timings above are inherited W-166 creation measurements, not W-167 processing timings. Sizes: Z=1 (1×32×32), small (9×32×32), wide (9×48×48). Z=1 samples the 3D kernel without projection or renormalization; axial-width-only changes are image-invariant there. Only the named factor changes; namespaces differ intentionally. Different sizes do not promise identical overlapping noise samples.</p>')
    section('gallery','Saved single-factor and combined gallery',
        '<p>All panels use the same −1 to 16 display scale; values outside it saturate only in this display, never in saved float images. Crosses are full independent truth, not detections. The middle round exposes dropout, weakening and fractional motion. Combined enables weakening, trend, gain, mixing, all backgrounds, both noises, translation/local geometry; it leaves dropout/loss and appearance/placement off.</p>'
        +gallery(presets, qualification))
    histories=[]
    for condition in ('clean','dropout','weakening','trend','loss','combined'):
        case=presets/('small-'+condition)
        table=pd.read_parquet(case/'round-truth.parquet')
        with np.load(case/'signals.npz') as values:
            for i, identity in enumerate(('gt-A','gt-B')):
                for r,label in enumerate(('round10','round2','round1')):
                    row=table[(table.amplicon_id==identity)&(table.round_label==label)].iloc[0]
                    histories.append(dict(condition=condition,truth=identity,round=label,
                        intended=values['intended'][i,:,r].tolist(),pre_mix=values['pre_mix'][i,:,r].tolist(),
                        realized=values['realized'][i,:,r].tolist(),dropped=bool(row.dropped),weakened=bool(row.weakened),
                        lost=bool(row.lost),emitting=bool(row.emitting),position=[row.z,row.y,row.x]))
    section('histories','Signal histories, geometry and complete populations',
        '<p>Intended channels persist through loss. Dropout recovers; persistent loss does not. Mixing uses destination rows/source columns after gain. Geometry maps q → q+d(q)+t in reference coordinates; molecules keep their kernel shape while background uses the inverse map. Baseline is added in the destination frame, then dependent and independent residuals.</p>'
        +render_table(histories)
        +'<p>All presets retain 2 formed objects and 6 history rows. The separate edge probe retains 4 objects/12 histories: coincident overlap-A/B both contribute, edge x=−0.5 contributes 7.059975220676764 at x=0 despite its out-of-frame center, far x=−20 has no sampled support. Middle-round dropout and last-round persistent loss keep every row; N=0 retains typed empty truth and (0,4,3) arrays. These tests are independent of detection and never invent an eligible benchmark denominator.</p>'
        +render_table([{k:v for k,v in row.items() if k != 'config_sha256'} for row in qualification['cases']])
        +details('Saved combined forward maps and inverse diagnostics',read_json(presets/'small-combined/synthetic.json')['extensions']['starfinder.synthetic']['transforms']))
    processing_parts=[]
    for case in delivery['cases']:
        folder=processing/'saved'/case['root']
        candidates=load_candidate_checkpoint(processing/'saved'/case['candidates'])
        formed=pd.read_parquet(folder/'formed.parquet')
        final=load_final_checkpoint(folder/'run/final-accepted')
        rows,identities,signals=decoding_rows(candidates,formed,final.pre_qc.decoded.table,final.filtering.table,case['settings']['filtering'])
        for identity in candidates.intensities.spot_ids:
            assert final.pre_qc.source_trace(spot_namespace=candidates.spots.spot_namespace,spot_id=identity)['available']
        processing_parts += [f'<h3 id="decoding-{case["root"]}">{case["root"]}: saved decoding and QC</h3>',render_table(rows),
            '<p>Observed colors → nucleotide barcode uses saved start C: 214 → CAAT, 123 → CCAG. Gene assignment is separate. WTA is exact matching, with no correction/distance metric; endpoint filtering is disabled. spot-1 ↔ gt-B and spot-2 ↔ gt-A are explicit coordinate correspondences, never assumed row order.</p>',
            render_table(signals),details('Full truth/detector namespaces',identities)]
        images=load_image_checkpoint(processing/'saved'/case['registered'])
        processing_parts.append(inspection_figure(images.layers[0].loaded,formed,0,case['root']+' clean reference'))
        summary=root/(case['root']+'-summary.html')
        write_run_summary(folder/'run',summary,title=case['root']+' saved successful run')
        processing_parts.append('<p><a href="'+summary.name+'">Complete W-165 static run summary</a></p>')
    failure=read_run(processing/'failed')
    assert failure['status']=='failed'
    assert failure['extensions']['starfinder.provenance']['final_state']['counts'] is None
    write_run_summary(processing/'failed',root/'failed-summary.html',title='Intentional TPS failure — saved partial output')
    section('processing','Saved processing, decoding, filtering and source traces',''.join(processing_parts)+
        '<h3>Partial / failed diagnostic</h3><p>TPS with only two landmarks raises InsufficientLandmarksError. The run remains failed; candidates/signals are partial, decoded/final stages unavailable, final counts null rather than zero. This expected failure is separate from successful clean runs. <a href="failed-summary.html">Inspect the complete saved failure summary</a>.</p>'
        +details('Saved failed-run stage state',failure['extensions']['starfinder.provenance']['stage_state'])
        +details('Saved failure diagnostics',failure['failures']))
    section('qualification','Independent checks, isolation and reproducibility',
        '<p>The full-grid oracle uses published constants, independent descriptor bytes/draws and 55-step scalar bisection for the inverse background map. It imports no production renderer, geometry or observation implementation. Every saved preset is checked. Exact regeneration is a separate persistence/repeatability check, not the analytic oracle.</p>'
        +render_table(pd.DataFrame([{k:v for k,v in row.items() if k not in ('command','log')} for row in context['checks']]).fillna('unavailable'))
        +'<p>Float64 combined image oracle: absolute tolerance 2e−10. Float32 saved arrays: max(1e−6, |oracle|×eps32/2+2e−10), relative tolerance zero. Literal binary-representable signal arithmetic is exact; coordinates 1e−12; inverse residual 2e−10. Saved byte/dtype/config checks use exact equality. Noise amplitudes may respond to changed upstream intensity, while standardized streams and unrelated latents remain fixed.</p>'
        +'<p>Calibration/evaluation descriptors differ but generation is rejected. No evaluation scene was used for development. Cross-version NumPy portability, calibrated ranges, original RNA abundance, benchmark matching/eligibility, historical v1/v2/large shifts and empirical add_noise/background_std semantics remain outside this qualification.</p>')
    section('w93','Criterion-mapped W-93 evidence and limits',render_table(context['w93'])+
        '<p>W-93 and W-57 remain open. No complete scientific Stage 1/2 checkbox is promoted by this child delivery.</p>')
    section('inspection','Offline access and HDF5 / TIFF / Fiji inspection',
        '<p>Open review.html directly on GP099-29C, or copy this single HTML file and open it in any browser with networking disabled. All essential figures/tables are embedded; no kernel or rerun is needed. Companion links need the inspection bundle or benchmark mount.</p>'
        '<p>Canonical image data use custom HDF5 ZYXC layout, one dataset per round. Existing value-preserving inspection TIFFs use ImageJ ZCYX hyperstacks, with channel/round labels retained. In Fiji, use the supplied inspect_saved_synthetic_fiji.py recipe and clear its absent-calibration micrometre default to pixel indices. See <a href="'+html.escape(os.path.relpath(processing/'saved/fiji-import.html',destination.parent),quote=True)+'">the preserved import recipe</a>. No new OME exporter or MATLAB validation is implied.</p>'
        +render_table(context['viewers'])
        +'<p>Copied images/tables are portable inspection inputs; some provenance/source references retain original private paths and need the benchmark mount. Headless application opening is distinct from GUI interaction and reviewer-machine access. Those remain unverified. Preserve previous W-160/W-175 packets. Jiahao owns retention through thesis/publication; backup and public reproducibility are unverified.</p>')
    section('trajectory','Issue trajectory, attempts, human waits and usage',trajectory(context))
    section('appendix','Exact identities and reproduction appendix',
        '<p>The report hash is stored in report-identity.json and the delivery manifest; the manifest hash is in manifest.sha256 and Linear, avoiding self-reference. Source revision plus uncommitted source snapshot identify this implementation phase. Controller validation, local commit pinning and W-174 human decision remain subsequent steps.</p>'
        +details('Independent oracle evidence and configuration hashes',qualification)
        +details('Two fresh-process records, different PYTHONHASHSEED',repeat)
        +details('Code, source hashes, commands, sessions and pinned input identities',context))
    document='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Batch 2 · independent synthetic qualification</title><style>'+css+'</style></head><body><h1>Batch 2 · independent synthetic qualification</h1>'
    document+='<nav aria-label="Contents">'+''.join('<a href="#'+i+'">'+t+'</a>' for i,t,_ in sections)+'</nav>'
    document+=''.join('<section id="'+i+'"><h2>'+t+'</h2>'+b+'</section>' for i,t,b in sections)+'</body></html>'
    destination.write_text(document)
    write_json(root/'report-identity.json',dict(report_sha256=checksum(destination),context_sha256=checksum(root/'review-context.json'),
        qualification_sha256=checksum(root/'qualification.json'),code=context['code']))


if __name__=='__main__':
    render(Path(sys.argv[1]),Path(sys.argv[2]))
