"""
    The Python workflow over the contract (design/specs/orchestration-engines.md, "The Snakefile over the manifest"):
    the plan runs once, in the main process, and one rule per rule group of its manifest runs each task's command.

    The plan (starfinder workflow plan --config <file>) validates and resolves the configuration and writes
    <output root>/workflow/manifest.json; a refusal (exit 2) stops Snakemake before the graph is built, with the plan's
    message. A job's own parse (a cluster job, where workflow.is_main_process is false) reads the manifest that the
    main process wrote (plan --locate). Each rule takes from the manifest: its outputs (the group's templates, temp()
    for its temporary outputs, directory() for a folder), the task record as one more output and the task log as its
    log; per job, the task's declared inputs and the records of the tasks it needs, its command and fingerprint as
    params (Snakemake's params trigger reruns a task whose fingerprint changed), and its execution settings as threads
    and resources, memory and runtime times the attempt. The shell runs the command with the execution context; a task
    with a tool_command (the BigStitcher engine task, D3) removes its outputs, runs the tool and then the command, which
    checks the tool's result, whatever the tool's exit status; without the tool's path (execution.fiji unset) the job
    fails at once naming the key. merge and compare are local rules. Nothing here imports the package or
    reads the configuration; no default, translation or result-affecting computation lives here.

    Snakemake requires every output and log of a rule to hold the same wildcards. A group's record and log name every
    identity field, its outputs may not (fovs/{fov}/... has no {sample}); the group's tasks are then split by the values
    of the fields its outputs lack, one rule per value, with those values written into the record and log templates.
    Rules are named <kind>_<group index>, and <kind>_<group index>_<part> when a group is split.

    A deconvolved copy outside the output root (deconvolution.copy_root beside or an absolute path) is shared by every
    output root that plans it, and its psf and deconvolve tasks verify and reuse it (design/specs/deconvolution.md,
    "Reuse and staleness"). Its files are therefore neither outputs nor inputs of any rule here: Snakemake removes a
    job's existing outputs before it runs the job (snakemake/jobs.py:797-809 in 9.27.1), touches them after it
    succeeds (dag.py:848-853) and removes them after it fails (jobs.py:955-958, called at 1322-1323); each would
    change a copy that another root's records name, and update() keeps them from the first only. The tasks stay
    ordered by the records of the tasks they need; the commands check the copy's files (the deconvolve task before
    reuse, a consumer before it reads them).
"""

import json
import os
import re
import shlex
import subprocess
import time

import snakemake as _engine
from snakemake.exceptions import WorkflowError

# One version date for the run (W-914): a modified checkout's version takes its .d<date> part from SOURCE_DATE_EPOCH,
# else from the clock, and the plan and every local job inherit this process's environment, so the run records one
# version, also across 00:00 UTC (starfinder._records.fix_version_date). A value the caller set is kept. A UGER job runs
# in the image, whose version is the build's.
os.environ.setdefault("SOURCE_DATE_EPOCH", str(int(time.time())))

# The plan reads one configuration file: several files or --config overrides would run a configuration that the
# plan, and so the manifest, never saw (the contract's changed statement of workflow-configuration.md:864-866).
if len(workflow.configfiles) != 1 or workflow.config_settings.config:
    raise WorkflowError(
        "the Python workflow runs the plan of one configuration file: give exactly one --configfile and no --config "
        "override; edit the file instead (the plan records it and its hash)")
CONFIGFILE = str(workflow.configfiles[0])


def _starfinder_plan(*arguments):
    """The plan's printed summary (JSON); its refusal stops Snakemake with the plan's message."""
    command = ["starfinder", "workflow", "plan", "--config", CONFIGFILE, *arguments]
    done = subprocess.run(command, capture_output=True, text=True)
    if done.returncode:
        raise WorkflowError(f"{shlex.join(command)} exited with status {done.returncode}:\n{done.stderr.strip()}")
    return json.loads(done.stdout)


MANIFEST_PATH = _starfinder_plan() if workflow.is_main_process else _starfinder_plan("--locate")
with open(MANIFEST_PATH["manifest"]) as _handle:
    MANIFEST = json.load(_handle)
ROOTS = MANIFEST["roots"]
# The execution context of every task command (the contract's context arguments), braces doubled for the shell's
# formatting; the job id is the scheduler's on UGER, else Snakemake's job number.
CONTEXT = (f"--engine snakemake --engine-version {shlex.quote(_engine.__version__)}".replace("{", "{{")
           .replace("}", "}}")
           + ' --job-id "${{JOB_ID:-{jobid}}}" --attempt {resources.attempt} --image "${{STARFINDER_IMAGE:-none}}"')
LOCAL_KINDS = ("merge", "compare")


def _path(declared):
    """A declared path or template: input:<path> under the input root, an absolute path as it is, any other under
    the output root; a folder's trailing slash removed."""
    if declared.startswith("input:"):
        path = os.path.join(ROOTS["input"], declared[len("input:"):])
    elif os.path.isabs(declared):
        path = declared
    else:
        path = os.path.join(ROOTS["output"], declared)
    return path.rstrip("/") if declared.endswith("/") else path


def _fill(template, values):
    """A template with the given wildcards written as their (brace-escaped) values."""
    for name, value in values.items():
        escaped = str(value).replace("{", "{{").replace("}", "}}")
        template = re.sub(r"(?<!\{)\{" + re.escape(name) + r"\}(?!\})", lambda _: escaped, template)
    return template


def _named(template):
    """The wildcard names of a template (a doubled brace is a literal one)."""
    return set(re.findall(r"(?<!\{)\{(\w+)\}(?!\})", template))


def _job(tasks, names):
    """The task of a job: looked up by the values of the rule's wildcards."""
    return lambda wildcards: tasks[tuple(wildcards[name] for name in names)]


# The folder of a deconvolved copy outside the output root (the manifest names it absolute), or None.
_COPY = (MANIFEST.get("deconvolution") or {}).get("copy")
SHARED_COPY = _path(_COPY).rstrip("/") + "/" if _COPY and os.path.isabs(_COPY) else None


def _shared(path):
    """Whether a declared path or template lies in the shared copy (a template's braces are doubled)."""
    if SHARED_COPY is None:
        return False
    return _path(path).startswith(SHARED_COPY) or _path(path).startswith(
        SHARED_COPY.replace("{", "{{").replace("}", "}}"))


def _inputs(job, by_id):
    return lambda wildcards: [_path(i) for i in job(wildcards)["inputs"] if not _shared(i)] + \
        [_path(by_id[need]["record"]) for need in job(wildcards)["needs"]]


def _tool(task):
    """The shell commands that run before a task's command: none, or for a task with a tool_command (the BigStitcher
    engine task, D3) its declared outputs removed, then the tool with its {<tool name>} placeholder written as the
    tool's path, whose failure is reported and left to the command to judge. Without the path (execution.<tool name>
    unset; planning without it is valid) the job stops there with exit status 2 and a message naming the key, before
    the || echo and the command, so that no empty tool runs.

    A job's own value (a param), not the rule's: Snakemake evaluates a rule's shell expression again when the job
    runs, when a loop variable such as the group's would hold the last group's value."""
    if "tool_command" not in task:
        return ""
    tool = task["execution"]["tool"]
    removed = "rm -f " + shlex.join(_path(o) for o in task["outputs"]) + "; "
    if tool["path"] is None:
        message = (f"task {task['id']} runs {tool['name']}, but execution.{tool['name']} is not set: set "
                   f"execution.{tool['name']} to the {tool['name']} executable in the configuration and plan again")
        return removed + f"{{ echo {shlex.quote(message)} >&2; exit 2; }}; "
    line = shlex.join(tool["path"] if part == "{" + tool["name"] + "}" else part for part in task["tool_command"])
    return removed + line + ' || echo "the tool exited with status $?" >&2; '


def _rules():
    """(rule name, group, {wildcard values: task}, wildcard names, fixed values) for each rule."""
    by_group = {}
    for task in MANIFEST["tasks"]:
        by_group.setdefault(task["group"], []).append(task)
    rules = []
    for index, group in enumerate(MANIFEST["groups"]):
        templates = [*group["outputs"], group["record"], group["log"]]
        shared = set(group["wildcards"]).intersection(*(_named(t) for t in templates))
        names = [w for w in group["wildcards"] if w in shared]
        fixed = [w for w in group["wildcards"] if w not in shared]
        parts = {}
        for task in by_group.get(group["name"], []):
            key = tuple(task["identity"][w] for w in fixed)
            parts.setdefault(key, {})[tuple(task["identity"][w] for w in names)] = task
        for part, (key, tasks) in enumerate(sorted(parts.items())):
            name = f"{group['kind']}_{index}" + (f"_{part}" if len(parts) > 1 else "")
            rules.append((name, group, tasks, names, dict(zip(fixed, key))))
    return rules


RULES = _rules()
BY_ID = {task["id"]: task for task in MANIFEST["tasks"]}
# The targets of the dispatcher's rule all: every task's record and non-temporary declared outputs, so a missing
# output reruns its producer even when its record remains. Removed temporary outputs stay exempt after completion;
# an explicit snakemake <record> target still selects only that task and its upstream. A shared copy's files are not
# targets (above).
_GROUPS = {group["name"]: group for group in MANIFEST["groups"]}
TASK_RECORDS = [_path(task["record"]) for task in MANIFEST["tasks"]] + [
    _path(output) for task in MANIFEST["tasks"] for output in task["outputs"]
    if output not in {_fill(template, task["identity"]) for template in _GROUPS[task["group"]]["temporary"]}
    and not _shared(output)]


for _name, _group, _tasks, _names, _fixed in RULES:
    _job_of = _job(_tasks, _names)
    _outputs = []
    for _template in _group["outputs"]:
        if _shared(_template):
            continue
        _output = _path(_fill(_template, _fixed))
        if _template in _group["temporary"]:
            _output = temp(_output)
        elif _template.endswith("/"):
            _output = directory(_output)
        _outputs.append(_output)
    _constraints = {w: "|".join(re.escape(k[i]) for k in sorted(_tasks)) for i, w in enumerate(_names)}
    _gpu = any(task["execution"]["gpu"] for task in _tasks.values())

    rule:
        name:
            _name
        localrule: _group["kind"] in LOCAL_KINDS
        wildcard_constraints:
            **_constraints
        input:
            _inputs(_job_of, BY_ID)
        output:
            *_outputs,
            record=_path(_fill(_group["record"], _fixed)),
        log:
            _path(_fill(_group["log"], _fixed)),
        params:
            command=lambda wildcards, job=_job_of: shlex.join(job(wildcards)["command"]),
            fingerprint=lambda wildcards, job=_job_of: job(wildcards)["fingerprint"],
            tool=lambda wildcards, job=_job_of: _tool(job(wildcards)),
            # Not in the shell (tool holds the removal): kept because Snakemake's params trigger compares the set of a
            # job's param values, so that a run completed before this rule changed does not rerun every task.
            outputs=lambda wildcards, job=_job_of: shlex.join(_path(o) for o in job(wildcards)["outputs"]),
        threads: lambda wildcards, job=_job_of: job(wildcards)["execution"]["threads"]
        resources:
            mem_mb=lambda wildcards, attempt, job=_job_of: job(wildcards)["execution"]["memory_mb"] * attempt,
            runtime=lambda wildcards, attempt, job=_job_of: job(wildcards)["execution"]["runtime_min"] * attempt,
            attempt=lambda wildcards, attempt: attempt,
            **({"gpu": lambda wildcards, job=_job_of: int(job(wildcards)["execution"]["gpu"])} if _gpu else {}),
        shell:
            "{params.tool}{params.command} " + CONTEXT
