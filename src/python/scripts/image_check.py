#!/usr/bin/env python3
"""The checks that run inside a built learned image (W-878).

Two checks, both with the standard library only, so that the image's own Python runs them:

* ``pip-check <variant>`` reads the output of ``uv pip check`` on standard input. The image has no ``uv``; the host's
  binary is mounted into the container. The ``cpu`` image must report no incompatibility. The ``learned`` and
  ``learned-cuda`` images may report exactly one line, :data:`ACCEPTED_LINE`: ``uv.lock`` holds only
  ``opencv-python-headless`` (W-813), which provides the ``cv2`` module that ``piscis`` imports, so ``piscis``'s
  requirement for the distribution ``opencv-python`` is never met by name. The line is matched by its exact text;
  any other line fails the check.
* ``imports`` imports ``cv2``, ``cellpose``, ``piscis`` and ``stardist`` together on the CPU and prints their versions.

Usage, with the image already loaded (``docs/workflow/broad-uger.md``, "Checking a built image")::

    uv pip check --python /opt/starfinder/bin/python 2>&1 | python image_check.py pip-check learned
    CUDA_VISIBLE_DEVICES= python image_check.py imports

The exit status is 0 when the check passes and 1 when it does not.
"""

from __future__ import annotations

import argparse
import os
import re
import sys

#: The one ``uv pip check`` line a learned image accepts, by exact text (Jiahao, 2026-11-18, W-870 finding F8).
ACCEPTED_LINE = "The package `piscis` requires `opencv-python`, but it's not installed"
#: The images whose lock holds ``piscis``, so that they carry :data:`ACCEPTED_LINE`.
ACCEPTING_VARIANTS = ("learned", "learned-cuda")
VARIANTS = ("cpu",) + ACCEPTING_VARIANTS

_STATUS = re.compile(r"^(Using Python \S+ environment at: \S+|Checked \d+ packages? in \S+)$")
_CLEAN = "All installed packages are compatible"
_FOUND = re.compile(r"^Found (\d+) incompatibilit(?:y|ies)$")
_FINDING = "The package `"


def pip_check_problems(output, variant):
    """List what a ``uv pip check`` output holds that the image's check does not accept.

    Parameters
    ----------
    output : str
        The standard output and error of ``uv pip check`` in the image, joined.
    variant : {"cpu", "learned", "learned-cuda"}
        The image's target. Only ``learned`` and ``learned-cuda`` accept :data:`ACCEPTED_LINE`.

    Returns
    -------
    list of str
        One message per line or condition that fails the check; empty when the output is accepted. The output must hold
        a ``Checked N packages`` line (which a failed ``uv`` run would not print) and then either the compatible line
        or a ``Found N incompatibility`` line followed by N findings, all of them the accepted line, with the accepted
        line at most once.

    Raises
    ------
    ValueError
        If ``variant`` is not one of the three targets.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    accepted = ACCEPTED_LINE if variant in ACCEPTING_VARIANTS else None
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    problems = []
    checked = False
    found = None
    findings = []
    clean = False
    for line in lines:
        if _STATUS.match(line):
            checked = checked or line.startswith("Checked")
        elif line == _CLEAN:
            clean = True
        elif _FOUND.match(line):
            found = int(_FOUND.match(line).group(1))
        elif line.startswith(_FINDING):
            findings.append(line)
        else:
            problems.append(f"unexpected line: {line}")
    if not checked:
        problems.append("no 'Checked N packages' line: uv pip check did not run to its end")
    for line in findings:
        if line != accepted:
            problems.append(f"finding not accepted: {line}")
    if findings.count(ACCEPTED_LINE) > 1 and accepted:
        problems.append(f"the accepted line appears {findings.count(ACCEPTED_LINE)} times, not once")
    if found is not None and found != len(findings):
        problems.append(f"'Found {found}' does not match the {len(findings)} findings listed")
    if clean and (found or findings):
        problems.append("compatible line together with findings")
    if not clean and found is None:
        problems.append("neither the compatible line nor a 'Found N incompatibility' line")
    if found is not None and found == 0:
        problems.append("'Found 0 incompatibilities' is not a uv output")
    return problems


def import_versions():
    """Import ``cv2``, ``cellpose``, ``piscis`` and ``stardist`` together and return their versions.

    Returns
    -------
    dict of str to str
        The installed version of each module's distribution (``opencv-python-headless`` for ``cv2``).

    Raises
    ------
    AssertionError
        If ``CUDA_VISIBLE_DEVICES`` is not set to the empty string, or if TensorFlow (for StarDist) or PyTorch (for
        Cellpose) reports a GPU, so that the import is on the CPU.
    ImportError
        If a module does not import.
    importlib.metadata.PackageNotFoundError
        If a module's distribution is not installed (``opencv-python-headless`` for ``cv2``).
    """
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "", "run with CUDA_VISIBLE_DEVICES= (the CPU check)"
    import importlib.metadata

    import cellpose  # noqa: F401
    import cv2  # noqa: F401
    import piscis  # noqa: F401
    import stardist  # noqa: F401
    import tensorflow
    import torch

    assert tensorflow.config.list_physical_devices("GPU") == [], "TensorFlow sees a GPU"
    assert not torch.cuda.is_available(), "PyTorch sees a GPU"
    distributions = {"cv2": "opencv-python-headless", "cellpose": "cellpose", "piscis": "piscis", "stardist": "stardist"}
    return {name: importlib.metadata.version(distribution) for name, distribution in distributions.items()}


def main(argv=None):
    """Run one check from the command line.

    Parameters
    ----------
    argv : list of str, optional
        The arguments; ``sys.argv[1:]`` by default.

    Returns
    -------
    int
        0 when the check passes, 1 when it does not.
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    pip = commands.add_parser("pip-check", help="filter the uv pip check output on standard input")
    pip.add_argument("variant", choices=VARIANTS)
    commands.add_parser("imports", help="import cv2, cellpose, piscis and stardist on the CPU")
    args = parser.parse_args(argv)
    if args.command == "pip-check":
        output = sys.stdin.read()
        print(output, end="" if output.endswith("\n") else "\n")
        problems = pip_check_problems(output, args.variant)
        for problem in problems:
            print(f"uv pip check: {problem}")
        if problems:
            return 1
        if ACCEPTED_LINE in output:
            print(f"accepted by exact text: {ACCEPTED_LINE}")
        return 0
    for name, version in import_versions().items():
        print(name, version)
    print("imports ok")
    return 0


if __name__ == "__main__":
    sys.exit(main())
