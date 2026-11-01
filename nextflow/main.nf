
include { SUMMARIZE as SUMMARIZE_0 } from './modules/summarize'
include { SUMMARIZE as SUMMARIZE_1 } from './modules/summarize'
include { SUMMARIZE as SUMMARIZE_2 } from './modules/summarize'
include { MERGE as MERGE_0 } from './modules/merge'
include { MERGE as MERGE_1 } from './modules/merge'
include { MERGE as MERGE_2 } from './modules/merge'
include { FOV } from './modules/fov'
include { TILE_GLOBAL } from './modules/tile_global'
include { TILE } from './modules/tile'
include { TILE_MERGE } from './modules/tile_merge'
include { MORPHOLOGY } from './modules/morphology'
include { SEGMENT as SEGMENT_0 } from './modules/segment'
include { SEGMENT_GPU as SEGMENT_0_GPU } from './modules/segment'
include { SEGMENT as SEGMENT_1 } from './modules/segment'
include { SEGMENT_GPU as SEGMENT_1_GPU } from './modules/segment'
include { SEGMENT as SEGMENT_2 } from './modules/segment'
include { SEGMENT_GPU as SEGMENT_2_GPU } from './modules/segment'
include { ASSIGN } from './modules/assign'
include { BIGSTITCHER_IMPORT } from './modules/bigstitcher_import'
include { STITCH } from './modules/stitch'
include { EXPORT ; EXPORT_GPU } from './modules/export'
include { REPORT } from './modules/report'
include { COMPARE } from './modules/compare'

process PLAN {
    cache false
    input:
    val config_path
    output:
    path 'plan.json'
    script:
    """
    starfinder workflow plan --config ${TaskCommands.quote(config_path)} > plan.json
    """
    stub:
    """
    starfinder workflow plan --config ${TaskCommands.quote(config_path)} > plan.json
    """
}

process VERIFY_BEFORE {
    executor 'local'
    cache false
    input:
    val manifest_path
    output:
    tuple val(manifest_path), path('tokens.tsv')
    script:
    """
    starfinder workflow verify --manifest ${TaskCommands.quote(manifest_path)} > tokens.tsv
    """
    stub:
    """
    starfinder workflow verify --manifest ${TaskCommands.quote(manifest_path)} > tokens.tsv
    """
}

process VERIFY_AFTER {
    executor 'local'
    cache false
    input:
    val manifest_path
    path records
    output:
    path 'verified.tsv'
    script:
    """
    starfinder workflow verify --manifest ${TaskCommands.quote(manifest_path)} --final > verified.tsv
    """
    stub:
    """
    starfinder workflow verify --manifest ${TaskCommands.quote(manifest_path)} --final > verified.tsv
    """
}

// Refuse unsupported graphs before fan-out; depth computation is engine ordering only.
def checkedTasks(manifest) {
    def allowed = ['summarize','merge','fov','tile_global','tile','tile_merge','morphology',
                   'segment','assign','bigstitcher_import','stitch','export','report','compare']
    def positions = [:]
    manifest.tasks.each { t ->
        if (t.kind in ['bigstitcher_engine', 'bigstitcher_macro'])
            error "D3: Nextflow runs no Fiji (${t.id}); supply stitching.dataset_xml and its view_map, or use Snakemake"
        if (!(t.kind in allowed)) error "Nextflow has no implementation for kind ${t.kind}: ${t.id}"
        def upstream = t.needs.collect { positions[it] }
        if (upstream.any { it == null }) error "Task ${t.id} has an unknown or unordered need"
        def depth = 0
        def chain = [t.id]
        if (t.kind == 'segment') {
            def parents = upstream.findAll { it.kind == 'segment' }
            depth = parents ? parents.collect { it.depth }.max() + 1 : 0
            if (parents) chain = parents.max { it.depth }.chain + [t.id]
            if (depth >= 3) error "Nextflow seed chain deeper than three at ${t.id}: ${chain.join(' -> ')}"
        }
        if (t.kind in ['summarize','merge']) {
            def parents = upstream.findAll { it.kind in ['summarize','merge'] }
            depth = parents ? parents.collect { it.depth + (t.kind == 'summarize' ? 1 : 0) }.max() : 0
            if (depth >= 3) error "Nextflow refuses more than three supplied steps at ${t.id}"
        }
        t.depth = depth
        positions[t.id] = [kind:t.kind, depth:depth, chain:chain]
    }
    return manifest.tasks
}

def ready(tasks, done) {
    tasks.filter { id, t -> t.needs }
        .flatMap { id, t -> t.needs.collect { need -> tuple(need, id, t) } }
        .combine(done, by: 0)
        .map { need, id, t, record -> tuple(groupKey(id, t.needs.size()), t, record) }
        .groupTuple()
        .map { key, ts, records -> tuple(key.toString(), ts[0], records.sort { a, b -> a.name <=> b.name }) }
        .mix(tasks.filter { id, t -> !t.needs }.map { id, t -> tuple(id, t, []) })
        .map { id, t, records -> tuple(id, t.fingerprint, t.token, t.manifest_path, records) }
}

workflow {
    if (!params.config) error 'Give --config <workflow.yaml>'
    if (params.require_container && !params.container) error 'The sge profile requires --container <shared-drive CPU SIF> for PLAN and VERIFY as well as tasks'
    PLAN(file(params.config).toAbsolutePath().toString())
    manifest_path = PLAN.out.map { p ->
        def summary = new groovy.json.JsonSlurper().parse(p.toFile())
        checkedTasks(new groovy.json.JsonSlurper().parse(new File(summary.manifest)))
        summary.manifest
    }
    VERIFY_BEFORE(manifest_path)
    tasks = VERIFY_BEFORE.out.flatMap { mpath, tokens ->
        def manifest = new groovy.json.JsonSlurper().parse(new File(mpath))
        def by_id = tokens.readLines().collectEntries { line ->
            def columns = line.split('\t'); [(columns[0]): columns[1]]
        }
        checkedTasks(manifest).collect { t ->
            t.token = by_id[t.id]; t.manifest_path = mpath
            tuple(t.id, t)
        }
    }
    done = Channel.empty()
    SUMMARIZE_0(ready(tasks.filter { id, t -> t.kind == 'summarize' && t.depth == 0 }, done))
    done = done.mix(SUMMARIZE_0.out.done)
    MERGE_0(ready(tasks.filter { id, t -> t.kind == 'merge' && t.depth == 0 }, done))
    done = done.mix(MERGE_0.out.done)
    SUMMARIZE_1(ready(tasks.filter { id, t -> t.kind == 'summarize' && t.depth == 1 }, done))
    done = done.mix(SUMMARIZE_1.out.done)
    MERGE_1(ready(tasks.filter { id, t -> t.kind == 'merge' && t.depth == 1 }, done))
    done = done.mix(MERGE_1.out.done)
    SUMMARIZE_2(ready(tasks.filter { id, t -> t.kind == 'summarize' && t.depth == 2 }, done))
    done = done.mix(SUMMARIZE_2.out.done)
    MERGE_2(ready(tasks.filter { id, t -> t.kind == 'merge' && t.depth == 2 }, done))
    done = done.mix(MERGE_2.out.done)
    FOV(ready(tasks.filter { id, t -> t.kind == 'fov' }, done))
    done = done.mix(FOV.out.done)
    TILE_GLOBAL(ready(tasks.filter { id, t -> t.kind == 'tile_global' }, done))
    done = done.mix(TILE_GLOBAL.out.done)
    TILE(ready(tasks.filter { id, t -> t.kind == 'tile' }, done))
    done = done.mix(TILE.out.done)
    TILE_MERGE(ready(tasks.filter { id, t -> t.kind == 'tile_merge' }, done))
    done = done.mix(TILE_MERGE.out.done)
    MORPHOLOGY(ready(tasks.filter { id, t -> t.kind == 'morphology' }, done))
    done = done.mix(MORPHOLOGY.out.done)
    route_SEGMENT_0 = tasks.filter { id, t -> t.kind == 'segment' && t.depth == 0 }.branch { id, t ->
        gpu: t.execution.gpu
        cpu: true
    }
    SEGMENT_0(ready(route_SEGMENT_0.cpu, done))
    done = done.mix(SEGMENT_0.out.done)
    SEGMENT_0_GPU(ready(route_SEGMENT_0.gpu, done))
    done = done.mix(SEGMENT_0_GPU.out.done)
    route_SEGMENT_1 = tasks.filter { id, t -> t.kind == 'segment' && t.depth == 1 }.branch { id, t ->
        gpu: t.execution.gpu
        cpu: true
    }
    SEGMENT_1(ready(route_SEGMENT_1.cpu, done))
    done = done.mix(SEGMENT_1.out.done)
    SEGMENT_1_GPU(ready(route_SEGMENT_1.gpu, done))
    done = done.mix(SEGMENT_1_GPU.out.done)
    route_SEGMENT_2 = tasks.filter { id, t -> t.kind == 'segment' && t.depth == 2 }.branch { id, t ->
        gpu: t.execution.gpu
        cpu: true
    }
    SEGMENT_2(ready(route_SEGMENT_2.cpu, done))
    done = done.mix(SEGMENT_2.out.done)
    SEGMENT_2_GPU(ready(route_SEGMENT_2.gpu, done))
    done = done.mix(SEGMENT_2_GPU.out.done)
    ASSIGN(ready(tasks.filter { id, t -> t.kind == 'assign' }, done))
    done = done.mix(ASSIGN.out.done)
    BIGSTITCHER_IMPORT(ready(tasks.filter { id, t -> t.kind == 'bigstitcher_import' }, done))
    done = done.mix(BIGSTITCHER_IMPORT.out.done)
    STITCH(ready(tasks.filter { id, t -> t.kind == 'stitch' }, done))
    done = done.mix(STITCH.out.done)
    route_EXPORT = tasks.filter { id, t -> t.kind == 'export' }.branch { id, t ->
        gpu: t.execution.gpu
        cpu: true
    }
    EXPORT(ready(route_EXPORT.cpu, done))
    done = done.mix(EXPORT.out.done)
    EXPORT_GPU(ready(route_EXPORT.gpu, done))
    done = done.mix(EXPORT_GPU.out.done)
    REPORT(ready(tasks.filter { id, t -> t.kind == 'report' }, done))
    done = done.mix(REPORT.out.done)
    COMPARE(ready(tasks.filter { id, t -> t.kind == 'compare' }, done))
    done = done.mix(COMPARE.out.done)
    VERIFY_AFTER(manifest_path, done.map { id, record -> record }.collect().ifEmpty([]))
}
