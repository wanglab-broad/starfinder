"""MATLAB invocation for Snakemake rules.

``common.smk`` imports this standard-library-only module so the MATLAB backend
does not depend on the STARfinder Python package. Two top-level config keys
select the launcher:

``matlab_launcher``
    ``path`` (default) runs ``STARFINDER_MATLAB_EXECUTABLE`` when set, otherwise
    ``matlab`` found on ``PATH``. ``broad`` sources the Broad ``useuse`` script,
    loads ``use Matlab`` and then runs ``matlab`` in the same shell.
``matlab_single_thread``
    ``false`` (default) leaves MATLAB multithreaded; ``true`` adds
    ``-singleCompThread``.

MATLAB runs in ``-batch`` mode, so an error in the entry point exits non-zero
and ``subprocess.CalledProcessError`` propagates to Snakemake.
"""

import os
import re
import shlex
import shutil
import subprocess
from pathlib import Path

LAUNCHERS = ("path", "broad")
DEFAULT_LAUNCHER = "path"
EXECUTABLE_VARIABLE = "STARFINDER_MATLAB_EXECUTABLE"
BROAD_USEUSE = "/broad/software/scripts/useuse"
_ENTRY_POINT = re.compile(r"[A-Za-z][A-Za-z0-9_]*")


def matlab_command(config, param_string, matlab_script_name, environ=None):
    """Return the argv list that runs ``matlab_script_name(param_string)``.

    ``param_string`` is the existing trusted MATLAB argument expression built by
    the rule. Raises ``ValueError`` for an invalid entry point or config value
    and ``FileNotFoundError`` when the selected launcher is unavailable.
    """
    if not _ENTRY_POINT.fullmatch(matlab_script_name):
        raise ValueError(f"MATLAB entry point must be a function name, got {matlab_script_name!r}")
    launcher = config.get("matlab_launcher", DEFAULT_LAUNCHER)
    if launcher not in LAUNCHERS:
        raise ValueError(f"Unknown matlab_launcher {launcher!r}. Valid options: 'path', 'broad'")
    single_thread = config.get("matlab_single_thread", False)
    if not isinstance(single_thread, bool):
        raise ValueError(f"matlab_single_thread must be true or false, got {single_thread!r}")

    script_path = str(Path(config["starfinder_path"]) / "workflow" / "scripts").replace("'", "''")
    expression = f"addpath('{script_path}'); {matlab_script_name}({param_string});"
    arguments = ["-singleCompThread"] if single_thread else []
    arguments += ["-batch", expression]

    if launcher == "broad":
        if not Path(BROAD_USEUSE).is_file():
            raise FileNotFoundError(
                f"matlab_launcher 'broad' needs {BROAD_USEUSE}; use matlab_launcher 'path' off the Broad cluster"
            )
        # A fresh shell does not inherit the jobscript's 'use Matlab', so load it here.
        setup = f"source {BROAD_USEUSE} && use Matlab && exec "
        return ["/bin/bash", "-c", setup + shlex.join(["matlab", *arguments])]

    environ = os.environ if environ is None else environ
    selected = environ.get(EXECUTABLE_VARIABLE)
    executable = shutil.which(selected or "matlab")
    if executable is None:
        if selected:
            raise FileNotFoundError(f"{EXECUTABLE_VARIABLE} is not an executable: {selected}")
        raise FileNotFoundError(
            f"MATLAB not found: set {EXECUTABLE_VARIABLE} or add matlab to PATH, "
            "or use matlab_launcher 'broad' on the Broad cluster"
        )
    return [executable, *arguments]


def run_matlab_scripts(config, param_string, matlab_script_name):
    """Print the full MATLAB command line to the Snakemake log, then run it."""
    command = matlab_command(config, param_string, matlab_script_name)
    print(shlex.join(command), flush=True)
    subprocess.run(command, check=True)
