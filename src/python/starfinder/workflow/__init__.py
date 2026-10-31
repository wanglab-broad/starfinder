"""Workflow orchestration: the plan, the manifest and the task commands of ``starfinder workflow`` (§2.13).

The engine-neutral task layer of design/specs/orchestration.md. ``plan`` validates a configuration in the Python
shape (one block per step and an execution section), resolves its defaults and writes
``<output root>/workflow/resolved.json`` and ``workflow/manifest.json``: every task of the run with its identity,
dependencies, declared inputs and outputs, record, log, command, execution settings, the blocks its kind reads and
its fingerprint. Engines (Snakemake, Nextflow) or a person run each task's command, ``starfinder workflow <command>
--manifest <manifest> --task <id>``; ``verify`` applies the completion and staleness rule from the output root.
"""
from starfinder.workflow._plan import PlanRefused, plan
from starfinder.workflow._shape import TASK_BLOCKS
from starfinder.workflow._task import ResumeRefused, TaskRefused
from starfinder.workflow._verify import verify

__all__ = ["PlanRefused", "ResumeRefused", "TASK_BLOCKS", "TaskRefused", "plan", "verify"]
