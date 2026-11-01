import groovy.json.JsonSlurper

/** Rendering only: scientific work and completion rules belong to the commands. */
class TaskCommands {
    static String quote(Object value) { "'" + value.toString().replace("'", "'\"'\"'") + "'" }
    static Map entry(String manifest, String id) {
        new JsonSlurper().parse(new File(manifest)).tasks.find { it.id == id } as Map
    }
    static String run(Map t, String manifest, String token, Object version, Object attempt, Object image, Object job) {
        def output = new JsonSlurper().parse(new File(manifest)).roots.output
        def command = t.command.collect { quote(it) }.join(' ')
        // fingerprint is a val input even when the rendered command is unchanged.
        """${command} --engine nextflow --engine-version ${quote(version)} \
          --job-id "\${JOB_ID:-${job}}" --attempt ${attempt} --image ${quote(image ?: 'none')} \
          --cache-token ${quote(token)}
        cp ${quote(output + '/' + t.record)} ${quote(t.id.replace('/', '.') + '.json')}
        """
    }
}
