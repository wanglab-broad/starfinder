process MERGE {
    tag "${task_id}"
    label 'cpu'
    input:
    tuple val(task_id), val(fingerprint), val(token), val(manifest_path), path(needs_records)
    output:
    tuple val(task_id), path("${task_id.replace('/', '.')}.json"), emit: done
    script:
    TaskCommands.run(TaskCommands.entry(manifest_path, task_id), manifest_path, token, nextflow.version, task.attempt, task.container)
    stub:
    """
    ${TaskCommands.quote(params.stub_runner)} ${TaskCommands.quote(manifest_path)} ${TaskCommands.quote(task_id)} ${TaskCommands.quote(token)} ${task.attempt}
    """
}
