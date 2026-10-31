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
    with a tool_command (the BigStitcher engine task, D3) runs the tool first and the command, which checks the tool's
    result, whatever the tool's exit status. merge and compare are local rules. Nothing here imports the package or
    reads the configuration; no default, translation or result-affecting computation lives here.

    Snakemake requires every output and log of a rule to hold the same wildcards. A group's record and log name every
    identity field, its outputs may not (fovs/{fov}/... has no {sample}); the group's tasks are then split by the values
    of the fields its outputs lack, one rule per value, with those values written into the record and log templates.
    Rules are named <kind>_<group index>, and <kind>_<group index>_<part> when a group is split.
"""

import json
import os
import re
import shlex
import subprocess

import snakemake as _engine
from snakemake.exceptions import WorkflowError

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


def _inputs(job, by_id):
    return lambda wildcards: [_path(i) for i in job(wildcards)["inputs"]] + \
        [_path(by_id[need]["record"]) for need in job(wildcards)["needs"]]


def _tool(task):
    """A task's tool command with its {<tool name>} placeholder written as the tool's path."""
    tool = task["execution"]["tool"]
    return shlex.join(tool["path"] if part == "{" + tool["name"] + "}" else part for part in task["tool_command"])


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
# The targets of the dispatcher's rule all: every task's record, so that snakemake <record> runs one task and its
# upstream.
TASK_RECORDS = [_path(task["record"]) for task in MANIFEST["tasks"]]


for _name, _group, _tasks, _names, _fixed in RULES:
    _job_of = _job(_tasks, _names)
    _outputs = []
    for _template in _group["outputs"]:
        _output = _path(_fill(_template, _fixed))
        if _template in _group["temporary"]:
            _output = temp(_output)
        elif _template.endswith("/"):
            _output = directory(_output)
        _outputs.append(_output)
    _constraints = {w: "|".join(re.escape(k[i]) for k in sorted(_tasks)) for i, w in enumerate(_names)}
    _tooled = any("tool_command" in task for task in _tasks.values())
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
            tool=lambda wildcards, job=_job_of: _tool(job(wildcards)) if "tool_command" in job(wildcards) else "",
            outputs=lambda wildcards, job=_job_of: shlex.join(_path(o) for o in job(wildcards)["outputs"]),
        threads: lambda wildcards, job=_job_of: job(wildcards)["execution"]["threads"]
        resources:
            mem_mb=lambda wildcards, attempt, job=_job_of: job(wildcards)["execution"]["memory_mb"] * attempt,
            runtime=lambda wildcards, attempt, job=_job_of: job(wildcards)["execution"]["runtime_min"] * attempt,
            attempt=lambda wildcards, attempt: attempt,
            **({"gpu": lambda wildcards, job=_job_of: int(job(wildcards)["execution"]["gpu"])} if _gpu else {}),
        shell:
            ('rm -f {params.outputs}; {params.tool} || echo "the tool exited with status $?" >&2; '
             if _tooled else "") + "{params.command} " + CONTEXT
