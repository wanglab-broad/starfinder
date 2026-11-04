import groovy.json.JsonSlurper

/** Rendering only: scientific work and completion rules belong to the commands. */
class TaskCommands {
    static String quote(Object value) { "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    static Map entry(String manifest, String id) {
        new JsonSlurper().parse(new File(manifest)).tasks.find { it.id == id } as Map
    }
    static String run(Map t, String manifest, String token, Object version, Object attempt, Object image) {
        def output = new JsonSlurper().parse(new File(manifest)).roots.output
        def command = t.command.collect { quote(it) }.join(' ')
        // fingerprint is a val input even when the rendered command is unchanged.
        // The job id is the scheduler's $JOB_ID, which the sge profile passes into the container. Without it, the task's
        // hash: task.hash renders as null in a script: block at a task's first attempt, so the shell reads it from the
        // work folder, which Nextflow names <first two hex digits>/<the rest> (W-805).
        """task_hash="\$(basename "\$(dirname "\$PWD")")\$(basename "\$PWD")"
        ${command} --engine nextflow --engine-version ${quote(version)} \
          --job-id "\${JOB_ID:-\$task_hash}" --attempt ${attempt} --image ${quote(image ?: 'none')} \
          --cache-token ${quote(token)}
        cp ${quote(output + '/' + t.record)} ${quote(t.id.replace('/', '.') + '.json')}
        """
    }
}
