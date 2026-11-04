"""Generate the experiment commands' reference tables from the package (docs/benchmark/commands.md).

``experiment-commands`` renders ``starfinder experiment expand``, ``starfinder experiment evaluate``, ``starfinder
experiment report`` and ``starfinder trial run`` with their arguments from the package's own parser
(``starfinder.__main__._parser``). ``evaluation-sources`` renders the source kinds of ``experiment evaluate`` with the
keys each accepts, and ``evaluation-arguments`` the keys of a metric argument, both from
``starfinder.benchmark._evaluate``. Their content describes, one ``name: description`` line each, every kind or key
the code has; a missing or extra line fails the build, so the tables cannot drift from the code.
``report-protocol`` renders the keys of the report's protocol with their meaning from ``starfinder.benchmark._report``.
"""
from __future__ import annotations

import argparse

from docutils.parsers.rst import Directive

from method_tables import _table
from workflow_config import _described

COMMANDS = (("experiment", "expand"), ("experiment", "evaluate"), ("experiment", "report"), ("trial", "run"))


def _choices(parser):
    return next(a for a in parser._actions if isinstance(a, argparse._SubParsersAction)).choices


def command_rows():
    """(rows of the commands, rows of their arguments), from the package's parser."""
    from starfinder.__main__ import _parser
    root = _parser()
    commands, arguments = [], []
    for group, name in COMMANDS:
        parser = _choices(_choices(root)[group])[name]
        commands.append([f"starfinder {group} {name}", parser.description or ""])
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction):
                continue
            arguments.append([", ".join(action.option_strings), f"{group} {name}",
                              "required" if action.required else "optional", action.help or ""])
    return commands, arguments


class ExperimentCommands(Directive):
    """The experiment and trial commands and their arguments, from the package's parser."""

    def run(self):
        try:
            commands, arguments = command_rows()
        except (ImportError, StopIteration, KeyError) as error:
            raise self.error(f"experiment-commands: {error}")
        return [_table(["Command", "What it does and its exit status"], commands, literal_columns={0}),
                _table(["Argument", "Of", "Required", "Description"], arguments, literal_columns={0, 1})]


class EvaluationSources(Directive):
    """One row per source kind: the keys it accepts (besides name and kind) and the content's description."""

    has_content = True

    def run(self):
        from starfinder.benchmark import _evaluate
        try:
            described = _described(self.content, list(_evaluate.SOURCE_KINDS), "source kind")
        except ValueError as error:
            raise self.error(f"evaluation-sources: {error}")
        rows = [[kind, ", ".join(sorted(_evaluate._KINDS[kind].keys)), described[kind]]
                for kind in _evaluate.SOURCE_KINDS]
        return [_table(["Kind", "Keys", "What it reads"], rows, literal_columns={0, 1})]


class EvaluationArguments(Directive):
    """One row per key of a metric argument, with the content's description."""

    has_content = True

    def run(self):
        from starfinder.benchmark import _evaluate
        keys = sorted(_evaluate._ARGUMENT_KEYS)
        try:
            described = _described(self.content, keys, "argument key")
        except ValueError as error:
            raise self.error(f"evaluation-arguments: {error}")
        return [_table(["Key", "Meaning"], [[key, described[key]] for key in keys], literal_columns={0})]


class ReportProtocol(Directive):
    """One row per key of the report protocol, with its meaning, from the package."""

    def run(self):
        from starfinder.benchmark import _report
        return [_table(["Key", "What it holds"], [[k, v] for k, v in _report.PROTOCOL_KEYS.items()],
                       literal_columns={0})]


def setup(app):
    app.add_directive("experiment-commands", ExperimentCommands)
    app.add_directive("evaluation-sources", EvaluationSources)
    app.add_directive("evaluation-arguments", EvaluationArguments)
    app.add_directive("report-protocol", ReportProtocol)
    return {"parallel_read_safe": True, "parallel_write_safe": True}
