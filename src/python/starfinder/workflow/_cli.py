"""The ``starfinder workflow`` command group (design/specs/orchestration.md, "The CLI").

``plan`` and ``verify``, and one command per API entry, each taking the common arguments ``--manifest`` and
``--task`` and the execution context ``--engine``, ``--engine-version``, ``--job-id``, ``--attempt``, ``--image`` and
``--cache-token``, which are recorded and never interpreted. Exit status 0 succeeded, 1 failed, 2 refused before any
work, 3 a refused verified resume.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from starfinder.workflow._commands import FUNCTIONS
from starfinder.workflow._shape import COMMANDS

ENGINES = ("snakemake", "nextflow", "manual")


def add_parser(groups):
    """Add the ``workflow`` group and its commands to the console script's subparsers."""
    workflow = groups.add_parser("workflow", help="Plan a run and run its tasks (the manifest's commands)")
    commands = workflow.add_subparsers(dest="command", required=True)
    plan = commands.add_parser("plan", help="Validate a configuration and write the resolved configuration and the "
                               "manifest")
    plan.add_argument("--config", type=Path, required=True, help="The configuration file (YAML, the Python shape)")
    plan.add_argument("--output-root", type=Path, help="The output root; default "
                      "<root_output_path>/<dataset_id>/<output_id>")
    plan.add_argument("--locate", action="store_true", help="Print the path of the manifest the plan writes, without "
                      "planning (an engine's parse in a cluster job)")
    verify =commands.add_parser("verify", help="Apply the completion and staleness rule from the output root")
    verify.add_argument("--manifest", type=Path, required=True)
    verify.add_argument("--task", help="One task id; default every task")
    verify.add_argument("--hash", action="store_true", help="Compare each output's SHA-256, not only its size")
    verify.add_argument("--final", action="store_true", help="Exit 1 when a task is not complete")
    for kind in sorted(FUNCTIONS, key=COMMANDS.get):
        task = commands.add_parser(COMMANDS[kind], help=f"Run one task of kind {kind}")
        task.add_argument("--manifest", type=Path, required=True, help="The plan's workflow/manifest.json")
        task.add_argument("--task", required=True, help="The task id in the manifest")
        task.add_argument("--engine", choices=ENGINES, default="manual")
        task.add_argument("--engine-version")
        task.add_argument("--job-id")
        task.add_argument("--attempt", type=int, default=1)
        task.add_argument("--image", help="The image reference by digest, or none")
        task.add_argument("--cache-token", help="The engine's cache token, kept as the record's workflow.token")
    return workflow


def run(args, argv=None):
    """Run the parsed ``starfinder workflow`` command; return its exit status."""
    if args.command == "plan":
        from starfinder.workflow._plan import PlanRefused, locate, plan
        try:
            if args.locate:
                print(json.dumps({"manifest": str(locate(args.config, args.output_root))}))
                return 0
            manifest = plan(args.config, args.output_root)
        except PlanRefused as error:
            print(f"plan refused: {error}", file=sys.stderr)
            return 2
        kinds = {}
        for task in manifest["tasks"]:
            kinds[task["kind"]] = kinds.get(task["kind"], 0) + 1
        print(json.dumps({"manifest": str(Path(manifest["roots"]["output"]) / "workflow" / "manifest.json"),
                          "tasks": len(manifest["tasks"]), "kinds": kinds, "export_route": manifest["export_route"]}))
        return 0
    if args.command == "verify":
        from starfinder.workflow._task import TaskRefused
        from starfinder.workflow._verify import verify
        try:
            result = verify(args.manifest, task=args.task, hash=args.hash, final=args.final)
        except (TaskRefused, ValueError) as error:
            print(f"verify refused: {error}", file=sys.stderr)
            return 2
        for task_id, entry in result["tasks"].items():
            state = "complete" if entry["complete"] else "not complete: " + "; ".join(entry["problems"])
            print(f"{task_id}\t{entry['token']}\t{state}")
        return 0 if result["passed"] else 1
    from starfinder.workflow._task import Context, execute
    kind = next(k for k in FUNCTIONS if COMMANDS[k] == args.command)
    function = FUNCTIONS[kind]
    context = Context(args.engine, args.engine_version, args.job_id, args.attempt,
                      None if args.image in (None, "none") else args.image, args.cache_token)
    prepare = {"fov": _prepare_fov, "tile_global": _prepare_tile_global}.get(kind)
    return execute(args.command, args.manifest, args.task, function, context,
                   argv=["starfinder", "workflow", *(argv or [])], prepare=prepare)


def _prepare_fov(run):
    """The fov task's verified resume: the earlier run.json and its kept stages, or nothing (exit 3 on a mismatch)."""
    from starfinder.workflow import _resume
    resume = _resume.prepare(run)
    if resume is None:
        return ()
    run.resume = resume
    return _resume.kept_outputs(run, *resume)


def _prepare_tile_global(run):
    """The global task's verified resume: every declared output kept when its run.json matches and the chunked
    checkpoint holds every round with its recorded SHA-256; otherwise a run from scratch (exit 3 on a mismatch)."""
    from starfinder.workflow import _resume
    resume = _resume.prepare_global(run)
    if resume is None:
        return ()
    run.resume = resume
    return [declared for declared, _ in run.outputs()]
