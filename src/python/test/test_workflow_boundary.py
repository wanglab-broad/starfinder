"""C8 of design/validation/orchestration.md: the boundary between the engines, the package and the legacy module.

No ``.smk`` or ``.nf`` file of the Python workflow imports ``starfinder``; the Python branch of ``workflow/Snakefile``
and ``rules/tasks.smk`` read no configuration key but ``backend``; and no package module of the Python workflow
imports the legacy module ``starfinder._legacy_workflow`` (D7, design/specs/orchestration-engines.md, "The MATLAB
legacy separation"), by the static import graph of the package, function-level imports included. The files are read
as text and tokenized with Python's tokenizer (the Snakemake keywords ``rule``, ``include`` and ``input:`` tokenize
as names and operators; comments and strings are left out). Nextflow's ``nextflow/`` (task group 5) is checked when it
exists.
"""
import ast
import io
import re
import tokenize
from pathlib import Path

import pytest
import yaml

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = ROOT / "src" / "python" / "starfinder"
LEGACY = "starfinder._legacy_workflow"
#: The Python workflow's Snakemake files: the dispatcher and the rules over the manifest (Nextflow's: nextflow_files).
PYTHON_SMK = (ROOT / "workflow" / "Snakefile", ROOT / "workflow" / "rules" / "tasks.smk")
#: The only files that import the legacy module: the four adapter scripts of the frozen workflow's shared rules and
#: the tests that pin the legacy outputs (their import line changed with W-750; their ids stay).
LEGACY_IMPORTERS = {
    "workflow/scripts/create_nuclei_amplicon_overlay.py", "workflow/scripts/enhance_dapi_with_flamingo.py",
    "workflow/scripts/stardist_segmentation.py", "workflow/scripts/reads_assignment.py",
    "src/python/test/test_assignment_workflow.py", "src/python/test/test_workflow_assembly_golden.py",
    "src/python/test/test_segmentation_golden.py", "src/python/test/test_segmentation_workflow.py",
    "src/python/test/test_stitching_assembly_outputs.py", "src/python/test/test_fov_membership.py",
    "src/python/test/test_stitching_workflow.py", "src/python/test/test_workflow_declared_grids.py"}
#: The old Python workflow's files that W-750 removed (the engine record, "The MATLAB legacy separation").
REMOVED = ("rules/registration-py.smk", "rules/spot-finding-py.smk", "rules/stitching-py.smk", "rules/export.smk",
           "rules/report.smk", *(f"scripts/{name}.py" for name in (
               "rsf_single_fov", "gr_single_fov_subtile", "lrsf_single_fov_subtile", "deep_create_subtile",
               "deep_rsf_subtile", "nuclei_registration", "bigstitcher_macro_sample", "bigstitcher_engine",
               "stitch_sample", "export_sample", "report_sample", "report_dataset", "create_segmentation_preview")))


def tokens(path, strings=False):
    """The code tokens of a file: no comment, newline or indentation token, and no string unless ``strings``."""
    skip = {tokenize.COMMENT, tokenize.NL, tokenize.NEWLINE, tokenize.INDENT, tokenize.DEDENT, tokenize.ENDMARKER}
    if not strings:
        skip.add(tokenize.STRING)
    return [t for t in tokenize.generate_tokens(io.StringIO(path.read_text()).readline) if t.type not in skip]


def package_imports(code):
    """The modules a token stream imports by an import statement, ``importlib.import_module`` or ``__import__``."""
    found = []
    for i, token in enumerate(code):
        if token.string in ("import", "from") and i + 1 < len(code):
            j, name = i + 1, ""
            while j < len(code) and (code[j].type == tokenize.NAME or code[j].string == ".") and \
                    code[j].string not in ("import", "as"):
                name += code[j].string
                j += 1
            found.append(name)
        if token.string in ("import_module", "__import__"):
            found.append(token.string)
    return found


def configuration_reads(code):
    """Each read of the ``config`` object, from a token stream with its strings: the key of ``config.get("<key>")``
    or ``config["<key>"]``, and ``None`` for any other use of the name (an attribute, a call argument, an iteration,
    a key that is not a literal)."""
    reads = []
    for i, token in enumerate(code):
        if token.type != tokenize.NAME or token.string != "config" or (i and code[i - 1].string == "."):
            continue
        following = code[i + 1:i + 5]
        if [t.string for t in following[:3]] == [".", "get", "("]:
            key = following[3] if len(following) > 3 else None
        elif following[:1] and following[0].string == "[":
            key = following[1] if len(following) > 1 else None
        else:
            key = None
        reads.append(ast.literal_eval(key.string) if key is not None and key.type == tokenize.STRING else None)
    return reads


def nextflow_files():
    folder = ROOT / "nextflow"
    return sorted(folder.rglob("*.nf")) if folder.is_dir() else []


def test_no_engine_file_of_the_python_workflow_imports_the_package():
    for path in (*PYTHON_SMK, *nextflow_files()):
        if path.suffix == ".nf":
            # Groovy has no Python import; the package is reached only through the starfinder command.
            assert "import starfinder" not in path.read_text() and "from starfinder" not in path.read_text(), path
            continue
        imported = package_imports(tokens(path))
        assert not [m for m in imported if m == "starfinder" or m.startswith("starfinder.")], (path, imported)
        assert "import_module" not in imported and "__import__" not in imported, path
        # No path to the package's source is put on sys.path either (the Python rules of W-737's base did).
        assert "sys.path" not in "".join(t.string for t in tokens(path)), path


def test_the_python_branch_reads_no_configuration_key_but_backend(tmp_path):
    snakefile, tasks = PYTHON_SMK
    assert configuration_reads(tokens(snakefile, strings=True)) == ["backend"]
    # tasks.smk reads no configuration key: it passes the one --configfile to the plan (workflow.configfiles).
    assert configuration_reads(tokens(tasks, strings=True)) == []
    # The reader itself: a read of another key, or another use of the object, is reported; a comment, a string and
    # another object's attribute are not reads.
    probe = tmp_path / "probe.smk"
    probe.write_text('# config["comment"]\nA = config["rules"]\nB = config.get("x", 1)\nC = f(config)\n'
                     'D = "config"\nE = workflow.config_settings.config\nrule all:\n    input: config.get(KEY)\n')
    assert configuration_reads(tokens(probe, strings=True)) == ["rules", "x", None, None]


def module_name(path):
    relative = path.relative_to(PACKAGE.parent).with_suffix("")
    parts = list(relative.parts)
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def import_graph():
    """{module: the starfinder modules it imports} of every module of the package, from its AST (every import
    statement, at any depth; ``from X import Y`` adds X.Y when it is a module)."""
    modules = {module_name(p): p for p in PACKAGE.rglob("*.py")}
    graph = {}
    for name, path in modules.items():
        package = name if path.name == "__init__.py" else name.rpartition(".")[0]
        edges = set()
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                edges |= {alias.name for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    anchor = package.split(".")[:len(package.split(".")) - node.level + 1]
                    base = ".".join(anchor + ([base] if base else []))
                edges.add(base)
                edges |= {f"{base}.{alias.name}" for alias in node.names}
        graph[name] = {e for e in edges if e in modules}
    return graph


def reachable(graph, roots):
    seen, stack = set(), list(roots)
    while stack:
        name = stack.pop()
        if name in seen:
            continue
        seen.add(name)
        # Importing a module imports its parent packages first.
        stack += [p for p in (name.rsplit(".", k)[0] for k in range(1, name.count(".") + 1)) if p in graph]
        stack += graph.get(name, ())
    return seen


def test_no_module_of_the_python_workflow_imports_the_legacy_module():
    graph = import_graph()
    assert LEGACY in graph and "starfinder._snakemake" not in graph
    roots = [name for name in graph if name == "starfinder.workflow" or name.startswith("starfinder.workflow.")]
    assert "starfinder.workflow._commands" in roots and "starfinder.workflow._plan" in roots
    assert LEGACY not in reachable(graph, roots)
    # No package module imports it at all: only the adapter scripts and the pinning tests do.
    assert [name for name, edges in graph.items() if LEGACY in edges] == []
    import re
    statement = re.compile(r"^\s*(from\s+starfinder\._legacy_workflow\s+import|from\s+starfinder\s+import\s+.*"
                           r"\b_legacy_workflow\b|import\s+starfinder\._legacy_workflow\b)", re.M)
    importers = set()
    for path in [*(ROOT / "workflow").rglob("*.py"), *(ROOT / "workflow").rglob("*.smk"),
                 *(ROOT / "src" / "python").rglob("*.py")]:
        if statement.search(path.read_text()):
            importers.add(path.relative_to(ROOT).as_posix())
    assert importers == LEGACY_IMPORTERS


def test_the_python_workflow_includes_no_file_of_the_old_python_workflow():
    """The dispatcher includes rules/legacy.smk and rules/tasks.smk only, tasks.smk includes no file, and legacy.smk
    the MATLAB rule set's; the old Python workflow's rule files and adapters are gone and no workflow file names
    them (W-750; it held the Python half of the old tests of the legacy configurations' rule set)."""
    import re
    snakefile, tasks = (path.read_text() for path in PYTHON_SMK)
    assert re.findall(r'include: "([^"]+)"', snakefile) == ["rules/legacy.smk", "rules/tasks.smk"]
    assert re.findall(r"include:", tasks) == []
    legacy = (ROOT / "workflow" / "rules" / "legacy.smk").read_text()
    assert re.findall(r'include: "([^"]+)"', legacy) == ["common.smk", "registration.smk", "spot-finding.smk",
                                                         "segmentation.smk", "stitching.smk", "reads-assignment.smk",
                                                         "utils.smk"]
    assert [name for name in REMOVED if (ROOT / "workflow" / name).exists()] == []
    texts = {path: path.read_text() for path in (ROOT / "workflow").rglob("*") if path.is_file()
             and path.suffix in (".smk", ".py", "") and path.name != "__pycache__"}
    for path, text in texts.items():
        assert not [name for name in REMOVED if Path(name).name in text], path
    rules = re.compile(r"^\s*rule (\w+):", re.M)
    named = {rule for text in texts.values() for rule in rules.findall(text)}
    assert not named & {"report_sample", "report_dataset", "export_sample", "stitch_sample", "bigstitcher_engine",
                        "bigstitcher_macro_sample"}


def test_nextflow_calls_the_package_only_through_commands():
    """C8's Nextflow half: the only literal package commands are planning and verification."""
    import re
    for path in nextflow_files():
        source = path.read_text()
        commands = re.findall(r"\bstarfinder\s+([^\n]+)", source)
        assert all(re.match(r"workflow\s+(plan|verify)\b", command) for command in commands), path
        assert not re.search(r"(?:python|python3)\s+(?:-m|-c)", source), path
        if path.parent.name == "modules":
            assert "TaskCommands.run(" in source, path
    helper = ROOT / "nextflow/lib/TaskCommands.groovy"
    assert helper.is_file()
    assert "t.command.collect" in helper.read_text()
    assert "import starfinder" not in helper.read_text()


def test_no_workflow_or_job_env_uses_the_runner_context():
    doc = yaml.safe_load((ROOT / ".github" / "workflows" / "tests.yml").read_text())
    scopes = [doc.get("env", {}), *(job.get("env", {}) for job in doc["jobs"].values())]
    assert not [v for scope in scopes for v in scope.values() if re.search(r"\$\{\{[^}]*\brunner\.", str(v))]
