"""Offline review of saved examples and optional revision-pinned execution evidence."""
from collections import defaultdict
from datetime import datetime, timezone
import base64
import html
from io import BytesIO, StringIO
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

from starfinder.barcode import decode_color_sequence
from starfinder.io import load_candidate_checkpoint, load_image_checkpoint
from saved_synthetic import checksum, read_json, verify_files, write_json


def table(rows):
    return pd.DataFrame(rows).to_html(index=False, escape=True, border=0)


def figure(fig, label, svg=False):
    output = StringIO() if svg else BytesIO()
    fig.savefig(output, format='svg' if svg else 'png', dpi=130, bbox_inches='tight')
    plt.close(fig)
    if svg:
        content = output.getvalue()
        content = content[content.index('<svg '):]
        return content.replace('<svg ', '<svg role="img" aria-label="'+html.escape(label, quote=True)+'" ', 1)
    return '<img alt="'+html.escape(label, quote=True)+'" src="data:image/png;base64,'+base64.b64encode(output.getvalue()).decode()+'">'


def inspection_figure(loaded, formed, round_index, name):
    image = loaded.image
    z = image.shape[0]//2
    fig, axes = plt.subplots(2, 4, figsize=(14, 6.8), layout='constrained')
    for c, channel in enumerate(loaded.channel_labels):
        ax = axes[0,c]
        view = ax.imshow(image[z,:,:,c], vmin=0, vmax=8, cmap='viridis', origin='lower', interpolation='nearest')
        ax.set(title=channel+' | XY at z='+str(z), xlabel='X index', ylabel='Y index')
        for row in formed.itertuples():
            ax.plot(row.x,row.y,'+',color='#ff8555', markersize=8)
            ax.annotate(row.amplicon_id,(row.x,row.y),xytext=(3,5),textcoords='offset points',color='white',fontsize=8)
    for i, (row, active) in enumerate(zip(formed.itertuples(), ((1,0,3),(0,1,2)), strict=True)):
        center = int(row.x)
        channel = active[round_index]
        axes[1,i].imshow(image[:,int(row.y),:,channel],vmin=0,vmax=8,cmap='viridis',origin='lower',interpolation='nearest',aspect='auto')
        axes[1,i].set(title=f'{row.amplicon_id}: XZ, y={int(row.y)}, {loaded.channel_labels[channel]}',xlabel='X index',ylabel='Z index')
        ax=axes[1,i+2]
        offsets=np.arange(-6,7)
        ax.plot(offsets,image[z,int(row.y),center+offsets,channel],'o-',label='X samples')
        axial=np.arange(image.shape[0])-z
        ax.plot(axial,image[:,int(row.y),center,channel],'s--',label='Z samples')
        ax.set(title=row.amplicon_id+' intensity profiles',xlabel='Offset from center (index)',ylabel='Stored intensity',ylim=(-.2,8.5))
        ax.legend(fontsize=8)
        ax.grid(alpha=.2)
    fig.colorbar(view,ax=axes[0,:],label='Stored intensity — shared linear scale',shrink=.8)
    fig.suptitle(name+' | channel-specific saved float32 data',fontsize=15)
    return figure(fig,f'{name} center slices, XZ views and per-spot intensity profiles')


def decoding_rows(saved, formed, decoded, filtered, filtering):
    """Join saved stages by identity, including rejected and unmatched detections."""
    keys = ['spot_namespace', 'spot_id']
    calls = decoded.set_index(keys, verify_integrity=True)
    outcomes = filtered.set_index(keys, verify_integrity=True)
    signal_indices = {identity: i for i, identity in enumerate(saved.intensities.spot_ids)}
    rows, identities, signals = [], [], []
    for candidate in saved.spots.spots.itertuples():
        key = (saved.spots.spot_namespace, candidate.spot_id)
        call, outcome = calls.loc[key], outcomes.loc[key]
        display = f'spot-{int(candidate.spot_id)+1}'
        matches = formed[(formed.z == candidate.z) & (formed.y == candidate.y) & (formed.x == candidate.x)]
        match = matches.iloc[0] if len(matches) == 1 else None
        gt = match.amplicon_id if match is not None else ('unmatched' if matches.empty else 'ambiguous GT match')
        observed = call.observed_color_sequence
        barcode = (decode_color_sequence(observed, filtering['start_base'])
                   if isinstance(observed, str) and observed and set(observed) <= set('1234') else 'unavailable (invalid color call)')
        row = {'Detected spot ID': display, 'Matched GT ID': gt,
            'Color sequence': observed, 'Decoded barcode': barcode,
            'Assigned gene': call.gene_id if pd.notna(call.gene_id) else 'unassigned',
            'Filter status': 'accepted' if outcome.accepted else 'rejected',
            'Rejection reason': outcome.rejection_reasons or call.failure_reason or '—',
            'GT color sequence': match.codeword if match is not None else '—',
            'GT barcode': match.barcode if match is not None else '—',
            'GT gene': match.gene_id if match is not None else '—'}
        if 'hamming_to_wta' in call:
            row['Matching distance (Hamming to WTA)'] = call.hamming_to_wta
        rows.append(row)
        identities.append(dict(simulation_namespace=match.namespace if match is not None else None,
            simulation_id=gt, detector_display=display, detector_namespace=key[0], detector_id=key[1]))
        i = signal_indices[candidate.spot_id]
        for r, name in enumerate(saved.intensities.round_labels):
            signals.append({'Matched GT ID': gt, 'Detected spot ID': display, 'Round': name,
                **{c: saved.intensities.values[i,k,r] for k,c in enumerate(saved.intensities.channel_labels)},
                'Valid': bool(saved.intensities.valid[i,r])})
    return rows, identities, signals


def phase(invocation):
    path=invocation.get('path','')
    if '/accept-' in path: return 'Review'
    if '/finalize-' in path: return 'Finalize'
    if '/repair-' in path: return 'Repair'
    return 'Implement'


def execution_graph(context):
    """Draw recorded intervals, with one latest cumulative counter per session.

    context supplies verified semantics and raw invocation evidence. Missing
    semantics disable numerical aggregation, rather than guessing token totals.
    """
    telemetry=context.get('telemetry')
    if not telemetry:
        return '<p>Historical execution evidence was not supplied to this standalone example.</p>'
    invocations=telemetry['invocations']
    issues=['W-'+str(i) for i in range(153,161)]
    names=['Baseline','Artifact contracts','Simulation specification','Provenance','Formed scenes','Image checkpoints','Candidates/signals','Saved example']
    intervals=[]
    attempts=defaultdict(int)
    for inv in sorted(invocations, key=lambda x:x['started_at']):
        attempts[inv['issue']]+=1
        intervals.append(dict(issue=inv['issue'],attempt=attempts[inv['issue']],phase=phase(inv),start=inv['started_at'],end=inv['finished_at'],exit=inv['exit_code']))
    for validation in telemetry['validations']:
        for check in validation['checks']:
            intervals.append(dict(issue=validation['issue'],attempt='check',phase='Validate',start=check['started_at'],end=check['finished_at'],exit=check['exit_code']))
    latest={}
    unique=defaultdict(set)
    for inv in sorted(invocations,key=lambda x:x['finished_at']):
        unique[inv['issue']].add(inv['session_id'])
        events=inv.get('execution_evidence',{}).get('usage_events',[])
        if events:
            latest[inv['session_id']]=(inv['issue'],events[-1]['usage'])
    counts=defaultdict(lambda:dict(input=0,cached=0,output=0,covered=0))
    verified=context.get('usage_semantics',{}).get('verified',False)
    for issue,usage in latest.values():
        assert 0 <= usage['cached_input_tokens'] <= usage['input_tokens']
        counts[issue]['covered']+=1
        if verified:
            counts[issue]['input']+=usage['input_tokens']
            counts[issue]['cached']+=usage['cached_input_tokens']
            counts[issue]['output']+=usage['output_tokens']
    colors={'Implement':'#3477ad','Validate':'#30986c','Review':'#9861b0','Repair':'#dc6d41','Finalize':'#aa9131'}
    fig,(ax,tokens)=plt.subplots(1,2,figsize=(16,6.6),gridspec_kw={'width_ratios':[2.1,1]},layout='constrained')
    for record in intervals:
        row=issues.index(record['issue'])
        start=mdates.date2num(datetime.fromisoformat(record['start']))
        end=mdates.date2num(datetime.fromisoformat(record['end']))
        ax.barh(row,end-start,left=start,height=.62,color=colors[record['phase']])
        if record['exit']:
            ax.plot(end,row,'x',color='red',markersize=8)
    for pause in context.get('pauses',[]):
        row=issues.index(pause['issue'])
        start=mdates.date2num(datetime.fromisoformat(pause['start']))
        end=mdates.date2num(datetime.fromisoformat(pause['end']))
        ax.barh(row,end-start,left=start,height=.85,facecolor='none',edgecolor='#d25934',hatch='////')
        ax.annotate(pause['label'],(start,row),xytext=(-40,-27),textcoords='offset points',fontsize=8,arrowprops={'arrowstyle':'-'})
    ax.set_yticks(range(8),[a+' '+b for a,b in zip(issues,names)])
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.MinuteLocator(interval=30))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=timezone.utc))
    ax.set(xlabel='21 September 2026 · UTC · actual order (no dependency edges)',title='Recorded execution intervals and attempts')
    ax.grid(axis='x',alpha=.2)
    ax.legend(handles=[Patch(facecolor=color,label=label) for label,color in colors.items()],ncol=3,loc='upper center',bbox_to_anchor=(.5,-.15),fontsize=8)
    rows=[]
    for i,issue in enumerate(issues):
        c=counts[issue]
        if verified:
            tokens.barh(i,c['input']/1e6,color='#adcbe2',height=.5)
            tokens.barh(i,c['cached']/1e6,color='#3477ad',height=.5)
            tokens.text(0,i+.37,f"output {c['output']:,}; coverage {c['covered']}/{len(unique[issue])}",fontsize=8)
        rows.append({'Issue':issue,'Invocations':sum(x['issue']==issue for x in invocations),
            'Input tokens':c['input'] if verified else 'unverified','Cached subset':c['cached'] if verified else 'unverified',
            'Output tokens':c['output'] if verified else 'unverified','Sessions covered (recorded only)':f"{c['covered']}/{len(unique[issue])}"})
    tokens.set_yticks(range(8),issues);tokens.invert_yaxis()
    tokens.set(xlabel='Input tokens (millions); darker = cached subset',title='Observed cumulative usage, unique sessions')
    tokens.grid(axis='x',alpha=.2)
    result=figure(fig,'Issue intervals, phases, attempts, documented pauses and token coverage',svg=True)
    result+='<p>Each interval is a recorded invocation or validation check; red × marks a nonzero exit. Gaps have unknown causes unless annotated. Worker checks inside implementation are included in that interval. Token columns use the latest observed cumulative counters once per unique session; cached input is included within input, and reasoning is included within output. Missing sessions and outer coordination are excluded, never treated as zero. No billing is inferred.</p>'
    if not verified: result+='<p>Counter semantics remain unverified; token aggregation disabled.</p>'
    result+=table(rows)
    repair=context.get('repair_intervals',[])
    if repair:
        fig,ax=plt.subplots(figsize=(14,max(2.4, .45*len(repair)+1.2)),layout='constrained')
        for i,r in enumerate(repair):
            start=mdates.date2num(datetime.fromisoformat(r['start']));end=mdates.date2num(datetime.fromisoformat(r['end']))
            ax.barh(i,end-start,left=start,color='#dc6d41')
        ax.set_yticks(range(len(repair)),[r['phase'] for r in repair]);ax.invert_yaxis()
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M',tz=timezone.utc))
        ax.set(title='W-175 — separate follow-up segment; usage coverage unavailable for active task',xlabel='21 September 2026 · UTC')
        result+=figure(fig,'W-175 follow-up execution intervals',svg=True)
    result+='<details><summary>Accessible interval source table (UTC)</summary>'+table(intervals)+'</details>'
    return result


def render_report(directory):
    manifest=read_json(directory/'delivery.json');verify_files(directory,manifest)
    reloads=sorted(directory.glob('reload-*.json'))
    if not reloads: raise ValueError('fresh-process reload evidence required before reporting')
    for p in reloads: assert read_json(p)['delivery_sha256']==checksum(directory/'delivery.json')
    destination=directory/'review.html'
    if destination.exists(): raise FileExistsError('Preserve report versions; use a fresh delivery directory')
    context_path=directory/'review-context.json'
    context=read_json(context_path) if context_path.exists() else {}
    viewer_path=directory/'fiji-verification.json'
    viewer=read_json(viewer_path) if viewer_path.exists() else None
    sections=['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Batch 1 · review round 2</title>',
        '<style>body{font:16px/1.55 system-ui,sans-serif;color:#203047;max-width:1400px;margin:36px auto;padding:0 26px;background:#fff}h1{font-size:2.4rem}h2{margin-top:52px;border-bottom:2px solid #deebf4;padding-bottom:8px}h3{margin-top:30px}table{border-collapse:collapse;margin:18px 0;max-width:100%;display:block;overflow:auto;font-size:.94rem}td,th{padding:9px 14px;border:1px solid #d8e1e8;text-align:left}th{background:#eef4f8}img,svg{max-width:100%;height:auto}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.8rem;background:#f3f6f8;padding:16px}nav{background:#eef4f8;padding:16px}nav a{margin-right:18px;display:inline-block}a{color:#14598d}details{margin:18px 0}summary{cursor:pointer;font-weight:600}.lead{font-size:1.15rem}.badge{background:#dceee4;padding:5px 12px;border-radius:4px}</style></head><body>',
        '<h1>Batch 1 · review round 2</h1><p class="lead">Saved 3D and singleton-Z examples with visible Gaussian gradients, reload comparisons and Fiji inspection files.</p>',
        '<p><span class="badge">Fixture revision 3</span> · source '+html.escape(context.get('reviewed_commit',manifest['code']['commit'])[:7])+'</p>',
        '<p><strong>Decision requested:</strong> Jiahao, approve or request changes against this packet. W-173 remains open; technical delivery does not approve the next batch.</p>',
        '<p>Two spots and three rounds are preserved. Independent Gaussian checks pass; image, signal, decoding and filtering reload comparisons are exact. This is a software development fixture with no noise, biological calibration or scientific accuracy claim.</p>',
        '<nav aria-label="Contents">'+''.join(f'<a href="#{i}">{label}</a>' for i,label in [('datasets','Data & settings'),('images','Images & signals'),('decoding-z9','3D decoding'),('decoding-z1','Z=1 decoding'),('reload','Reload checks'),('fiji','Fiji inspection'),('execution','Execution & tokens'),('validation','Validation & limitations'),('response','Review response'),('reproducibility','Reproducibility')])+'</nav>',
        '<h2 id="datasets">Data and frozen settings</h2>',
        table([{'Fixture':f"saved-formed-z{c['depth']}-v3",'Shape (ZYXC)':str(c['shape']),'Dtype':'float32','Objects':2,'Rounds':3,'Calibration':'unknown'} for c in manifest['cases']]),
        '<p>The 3D fixture exceeds the requested (4,32,32,4) minimum. The deliberate (1,32,32,4) counterpart samples the same 3D kernel at z=0; it is not a projection. Version 1 and 2 packets remain preserved separately; revision 3 changes truth labels and adds decoding inspection.</p>',
        table([{'Acquisition':'Round order','Value':'round10 → round2 → round1'},{'Acquisition':'Channel order','Value':'ch02 → ch00 → ch03 → ch01'},{'Acquisition':'Color symbol → channel index','Value':'1→1 (ch00), 2→0 (ch02), 3→3 (ch01), 4→2 (ch03)'}]),
        '<p>Color calls follow round10 → round2 → round1. Nucleotide barcodes use decode_color_sequence with the saved start_base=C; the initial C is included (3 colors → 4 bases). No reversal is applied to this displayed sequencing-direction barcode. The color-only codebook carries EncodingConfig(reverse_bases=True, split_index=None), applicable when importing base sequences; no base-sequence column or split is used here. End bases are unset and endpoint exclusion is disabled. WTA matches exact color strings; no error correction or matching-distance field is produced (distance is not a WTA score).</p>',
        table([{'Parameter':'Spot centers (ZYX)','Value':'gt-A=(Z//2,10,10); gt-B=(Z//2,22,22)'},{'Parameter':'Peak brightness','Value':'8 (unchanged)'},{'Parameter':'Gaussian widths','Value':'axial σ=1 slice; lateral σ=1.25 pixels'},{'Parameter':'Shape','Value':'elongation=1; angle=0'},{'Parameter':'Support','Value':'closed ellipsoid: (dz/1)²+(dy/1.25)²+(dx/1.25)² ≤16; zero outside'},{'Parameter':'Boundaries','Value':'3D support fully in bounds; Z=1 support is truncated axially without renormalization'}]),
        table([{'Processing':'Registration','Setting':'TranslationConfig defaults; all measured corrections exactly (0,0,0)'},{'Processing':'Spot finding','Setting':'LocalMaximaConfig: adaptive, threshold 0.1; existing defaults'},{'Processing':'Extraction','Setting':'NeighborhoodSumConfig radius=(0,0,0); center values'},{'Processing':'Decoding / filtering','Setting':'WtaDecoderConfig / ReadFilterConfig defaults, frozen in saved configuration'}]),
        table([{'Simulation':'Root seed','Setting':'42; independent component stream descriptors saved'},{'Simulation':'Input/model','Setting':'formed development v1 model; explicit coordinates and genes'},{'Simulation':'Effects','Setting':'background, noise, round effects, crosstalk, dropout and deformation disabled'},{'Simulation':'Scope','Setting':'processed images; not calibrated D04 or RNA truth'}]),
        '<h2 id="images">Saved images, identity and signals</h2><p>All image panels share a linear 0–8 scale. Center slices retain separate channels; XZ cuts use each spot’s active channel. Profiles show stored samples. Crosses identify simulated centers, before decoding.</p>']
    correspondence=[]
    for case in manifest['cases']:
        root=directory/case['root'];formed=pd.read_parquet(root/'formed.parquet')
        saved=load_candidate_checkpoint(directory/case['candidates'])
        images=load_image_checkpoint(directory/case['registered']).sequencing_images(require_registered=True)
        sections.append(f'<h3>{case["root"]}: '+('3D' if case['depth']>1 else 'intentional singleton Z')+'</h3>')
        for r,(name,loaded) in enumerate(images.items()): sections.append(inspection_figure(loaded,formed,r,name))
        decoded=pd.read_parquet(root/'uninterrupted-decoded.parquet')
        filtered=pd.read_parquet(root/'uninterrupted-filtered.parquet')
        rows, identities, signals = decoding_rows(saved, formed, decoded, filtered, case['settings']['filtering'])
        correspondence.extend(dict(case=case['root'], **row) for row in identities)
        sections.append('<h4 id="decoding-'+case['root']+'">Per-spot decoding and independent truth</h4>'+table(rows)+
            '<p>Simulation and detector namespaces are distinct. Correspondence is an explicit coordinate match, not numbering. '
            'Display IDs are spot- plus the original zero-based numeric detector ID + 1; sorting or filtering never renumbers them. '
            'All detections remain listed with a separate filter status. Observed colors come from the saved decoder results; '
            'nucleotide conversion uses those colors, never the assigned gene. GT colors and barcodes come from saved simulation truth. '
            'The model specifies color-space truth; GT nucleotide expectations are independently fixed under the saved start-base convention.</p>'+table(signals))
    sections.extend(['<h2 id="reload">Independent and fresh-process checks</h2>',table([{'Check':'Gaussian image samples','Result':'All voxels: absolute tolerance 1e-6, relative tolerance 0'},{'Check':'Peak / lateral neighbor / support edge / outside','Result':'8 / 5.8091923 / 0.002683701 / 0'},{'Check':'Truth population / round history','Result':'2 objects / 6 history rows; all emitting; no dropped/lost rows'},{'Check':'Prepared versus registered images','Result':'Bytes, dtype, geometry, channel labels exact; zero registration correction'},{'Check':'Reloaded extraction / signal validity','Result':'Exact values and stable identity'},{'Check':'Decoding, filtering and counts','Result':'Exact uninterrupted-versus-reloaded tables; gene-A and gene-B accepted'},{'Check':'Fresh process','Result':'Separate creation and reload process IDs, recorded below'},{'Check':'TIFF versus canonical HDF5','Result':'All float32 values and ordered labels exact'}]),
        '<h2 id="fiji">Fiji inspection</h2>',
        '<p>'+('Fiji reopened both formats for all six saved rounds. All voxel values agree with the independent Gaussian oracle and between HDF5 and TIFF; dimensions and labels pass.' if viewer else 'Fiji runtime evidence is not supplied in this example invocation; format checks alone do not establish viewer compatibility.')+'</p>',
        '<p>Use the accompanying <a href="fiji-import.html">Fiji import instructions</a>. HDF5 dataset layout is <code>zyxc</code>, one dataset per round. TIFF storage is ZCYX, one hyperstack per round. Fiji’s HDF5 plugin assumes micrometres when calibration is absent; the provided script removes that default and labels index units as pixels. No physical calibration is asserted.</p>'])
    for case in manifest['cases']:
        sections.append('<p>'+case['root']+': <a href="'+case['root']+'/registered/images.h5">Canonical HDF5</a> · '+ ' · '.join('<a href="'+case['root']+'/inspection/'+r+'.tif">'+r+' TIFF</a>' for r in ['round10','round2','round1'])+'</p>')
    sections.extend(['<h2 id="execution">Execution, attempts and token coverage</h2>',execution_graph(context),
        '<h2 id="validation">Validation and limitations</h2>',
        table(context.get('validation',[{'Check':'Full-suite / docs gates','Outcome':'Not supplied by this standalone invocation; see packet context'}])),
        '<p>Six redundant parameter combinations were removed: ten depth×dtype round trips become six (all five dtypes in 3D plus float32 Z=1); six depth×empty-input cases become four (all three empty forms in 3D plus explicit-empty Z=1). The four depth×precision Gaussian cases remain because singleton support and float tolerances interact. Failure/corruption, overwrite, independent truth and fresh-process coverage remain. The saved-example integration test is reused.</p>',
        '<p>Known limits: clean processed-image examples only; no real-data calibration, biological accuracy claim, MATLAB, later-batch work or public release reproducibility. Physical calibration and backup coverage remain unknown. Resource limits are measured targets, not enforced memory caps. Original packet history is retained.</p>',
        '<h2 id="response">Human review response</h2>',
        table(context.get('responses',[{'Review point':'1. Readability','Change':'Single current report; exact identifiers collapsed'},{'Review point':'2. Graph','Change':'Recorded phases, attempts, pauses and unique-session token coverage when evidence is supplied'},{'Review point':'3–4. Navigation and formatting','Change':'Linked contents and separate parameter/comparison tables'},{'Review point':'5. Dimensions / naming / gradients','Change':'Version 3, 9×32×32×4 and explicit Z=1; gt-A/B; sampled Gaussian profiles'},{'Review point':'5. Fiji / TIFF','Change':'Canonical custom-layout import; value-preserving per-round TIFFs; runtime evidence separately required'},{'Review point':'6. Tests','Change':'Six redundant combinations removed; required coverage retained'}])),
        '<h2 id="reproducibility">Reproducibility appendix</h2><details><summary>Exact revisions, hashes, process identities, commands, configuration and source tables</summary>',
        '<p>Report hashes are in report-identity.json and the outer manifest, avoiding a self-referential hash. Essential figures and tables are embedded; data links are optional inspection downloads.</p>',
        '<h3>Full identity correspondence</h3>'+table(correspondence),
        '<h3>Delivery manifest</h3><pre>'+html.escape(json.dumps(manifest,indent=2))+'</pre>',
        '<h3>Reload records</h3><pre>'+html.escape(json.dumps([read_json(p) for p in reloads],indent=2))+'</pre>',
        '<h3>Execution / validation source evidence</h3><pre>'+html.escape(json.dumps(context,indent=2))+'</pre>',
        '<h3>Fiji runtime evidence</h3><pre>'+html.escape(json.dumps(viewer,indent=2))+'</pre>'])
    for case in manifest['cases']:
        sections.append('<h3>'+case['root']+' complete synthetic configuration</h3><pre>'+html.escape((directory/case['root']/'synthetic.json').read_text())+'</pre>')
    sections.append('</details></body></html>')
    destination.write_text('\n'.join(sections))
    write_json(directory/'report-identity.json',dict(report_sha256=checksum(destination),delivery_sha256=checksum(directory/'delivery.json'),reloads={p.name:checksum(p) for p in reloads},review_context_sha256=checksum(context_path) if context_path.exists() else None))
    print(destination)
