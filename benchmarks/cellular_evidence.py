"""The cellular checks GN23 and the ``slow`` and acquired cases of GN24 as an evidence script (W-536).

These checks of docs/synthetic-cellular-scene-validation.md ran in the controller's ``pytest-slow-synthetic``
selection until W-536 moved them to the Chapter II evidence project: GN23 compares the ``slow`` context presets and
their acquired-pixel variants with W-393's target ranges (TR1 to TR12), and GN24 generates each ``slow`` preset and
each acquired-pixel variant in its own process within RG1 (one CPU, 16 GiB and 3,600 s per step). They read no real
data. The bodies are the tests of ``src/python/test/test_synthetic_cellular_presets.py`` at revision
``ORIGIN_REVISION``, unchanged in what they assert; the helpers stay in ``src/python/test/`` and are imported from
there. GN7 and GN5 on the ``slow`` presets and GN24's routine cases (the 4 GiB bound) stay in the suite.

    cd src/python
    uv run python ../../benchmarks/cellular_evidence.py --output <run dir>/cellular
    uv run python ../../benchmarks/cellular_evidence.py --list

Checks, in the order the tests ran (``--check`` selects some):

* ``GN23-slow``: each ``slow`` context preset, generated in this process, in its target ranges;
* ``GN23-acquired``: the 2D tissue and culture ``slow`` presets at the acquired pixel, TR4 and TR5;
* ``GN24-slow`` and ``GN24-acquired``: each generated in its own process under /usr/bin/time -v, within RG1.

The printed lines are those of the tests (``GN23 ROW`` and ``GN24 ROW`` JSON lines). ``--output`` (new or empty)
receives ``results.json`` (each case's outcome and wall time, the presets' seeds and the revisions), ``gn23_rows.jsonl``
and ``gn24_rows.jsonl``. Exit status 0 when every selected case passed, 1 when one failed. Do not run with
``python -O``, which drops the assertions.
"""
import argparse
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
# The revision whose test suite last held these checks as tests (W-536's starting revision).
ORIGIN_REVISION = "0b9fa8b8e840697408557c81b2d6e426b858e209"
ORIGIN_MODULE = "src/python/test/test_synthetic_cellular_presets.py"
CHECKS = ("GN23-slow", "GN23-acquired", "GN24-slow", "GN24-acquired")
ORIGIN_TESTS = {"GN23-slow": "test_gn23_slow_preset_in_its_target_ranges",
                "GN23-acquired": "test_gn23_acquired_variant_edge_and_bright_regions",
                "GN24-slow": "test_gn24_generation_cost_in_its_own_process",
                "GN24-acquired": "test_gn24_generation_cost_in_its_own_process"}
# context_slow_acquired: the 2D tissue and culture slow presets at the pixel W-393 measured TR4 and TR5 on.
ACQUIRED = {"context-tissue2d-slow-v1": (0.346255, 0.094635, 0.094635),
            "context-culture-slow-v1": (0.346255, 0.194398, 0.194398)}
CONTEXT = {"dense3d": ("DAPI", False), "tissue2d": ("PI", True), "culture": ("DAPI", True)}
# Compared quantities that GN23 reports with their value, range and count and does not assert: TR7 of dense 3D
# tissue, a known limit of the scene (docs/synthetic-cellular-scene-validation.md, the amendment of 2026-10-15 after
# the TR table).
KNOWN_LIMITS = {("dense3d", "within-nucleus CV")}


def parse(argv):
    parser = argparse.ArgumentParser(
        description="Run GN23 and the slow and acquired cases of GN24 of docs/synthetic-cellular-scene-validation.md, "
                    "moved out of the test suite by W-536. Reads no real data; RG1 allows one CPU, 16 GiB and "
                    "3,600 s per step.",
        epilog="Run from src/python: uv run python ../../benchmarks/cellular_evidence.py --output <run dir>/cellular")
    parser.add_argument("--output", type=Path, help="new or empty directory for results.json and the row tables "
                                                     "(outside the repository)")
    parser.add_argument("--check", action="append", choices=CHECKS,
                        help="run only this check (repeatable); default: all, in the order listed")
    parser.add_argument("--list", action="store_true", help="import the helpers and list the cases with their seeds")
    args = parser.parse_args(argv)
    if not args.list and args.output is None:
        parser.error("--output is required unless --list is given")
    return args


def _helpers():
    """The test helpers, imported from src/python/test (the package ``test`` of this checkout, not the standard
    library's)."""
    if str(PACKAGE) not in sys.path[:1]:
        sys.path.insert(0, str(PACKAGE))
    from test import cellular_measures as measures
    from test import test_synthetic_cellular_presets as presets
    return measures, presets


def _revision():
    spec = importlib.util.spec_from_file_location("benchmark_revision", Path(__file__).resolve().parent / "revision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.revision()


def context_of(name):
    return name.split("-")[1]


def acquired(name):
    """A context_slow_acquired variant: the preset with spacing_zyx replaced, every other value as the preset."""
    from starfinder.synthetic import context_scene_preset
    book, config = context_scene_preset(name)
    acquisition = replace(config.sample.acquisition, spacing_zyx=ACQUIRED[name])
    return book, replace(config, sample=replace(config.sample, acquisition=acquisition))


def generate(name, variant=False):
    from starfinder.synthetic import context_scene_preset, generate_context_scene
    book, config = acquired(name) if variant else context_scene_preset(name)
    return config, generate_context_scene(book, config=config)


def cases(presets):
    return ([("GN23-slow", n) for n in presets.SLOW] + [("GN23-acquired", n) for n in ACQUIRED]
            + [("GN24-slow", n) for n in presets.SLOW] + [("GN24-acquired", n) for n in ACQUIRED])


class Evidence:
    def __init__(self, measures, presets):
        self.measures, self.presets = measures, presets
        self.gn23_rows, self.gn24_rows = [], []

    # ---------------------------------------------------------------- GN23: the four cases in their target ranges

    def check_targets(self, label, name, scene, quantities):
        m = self.measures
        context = context_of(name)
        stain, projection = CONTEXT[context]
        rows = m.scene_targets(scene, context, stain, projection, quantities)
        findings, adjustable = [], []
        for row in rows:
            row = dict(row, scene=label, outcome=m.outcome(row))
            known = (context, row["quantity"]) in KNOWN_LIMITS and not row["outcome"].startswith("reported")
            if known:
                row["outcome"] += "; reported, not asserted (known limit, amendment of 2026-10-15)"
            print("GN23 ROW " + json.dumps(row, sort_keys=True))
            self.gn23_rows.append(row)
            if row["outcome"] == "outside the range" and not known:
                # The bright-region size may be adjusted by the page's rule within its stated conversion; every
                # other quantity outside its range is a finding returned to Jiahao.
                (adjustable if row["quantity"].startswith("bright region size") else findings).append(row)
        for row in adjustable:
            print(f"GN23 {label}: the bright-region size {row['value']:.4g} lies outside {row['low']}–{row['high']} on "
                  f"{row['objects']} regions (adjustable within the conversion of 'Presets')")
        assert not findings, "GN23 findings (TR rows outside their range where V12 compares them): " + "; ".join(
            f"{r['row']} {r['quantity']} {r['value']:.4g} not in [{r['low']}, {r['high']}] on {r['objects']} "
            f"{r['kind']}" for r in findings)
        return rows

    def gn23_slow(self, name):
        m = self.measures
        _, scene = generate(name)
        context = context_of(name)
        quantities = [q for q, (_, _, ranges) in m.TARGETS.items() if context in ranges
                      and (context == "dense3d" or q not in m.EDGE_AND_REGIONS)]
        rows = self.check_targets(name, name, scene, quantities)
        nuclei = len(scene.sample.nuclei)
        print(f"GN23 {name}: {nuclei} nuclei, {len(scene.sample.cells)} cells, {len(scene.sample.molecules)} "
              f"molecules; compared {sum(r['objects'] >= m.MINIMUM and r['value'] is not None for r in rows)} of "
              f"{len(rows)} quantities")

    def gn23_acquired(self, name):
        config, scene = generate(name, variant=True)
        assert tuple(config.sample.acquisition.spacing_zyx) == ACQUIRED[name]
        label = f"{name}@acquired"
        rows = self.check_targets(label, name, scene, list(self.measures.EDGE_AND_REGIONS))
        print(f"GN23 {label}: {len(scene.sample.nuclei)} nuclei; {[(r['quantity'], r['objects']) for r in rows]}")

    # ---------------------------------------------------------------- GN24: generation cost, one process per scene

    def gn24(self, name, spacing, tier):
        p = self.presets
        row = dict(p.generate_in_own_process(name, spacing), scene=name + ("@acquired" if spacing else ""), tier=tier)
        print("GN24 ROW " + json.dumps(row, sort_keys=True))
        self.gn24_rows.append(row)
        assert row["wall_s"] <= p.BOUND_S and row["peak_rss_kib"] <= p.BOUND_KIB, row

    def gn24_slow(self, name):
        self.gn24(name, None, "slow")

    def gn24_acquired(self, name):
        self.gn24(name, ACQUIRED[name], "acquired")

    RUN = {"GN23-slow": gn23_slow, "GN23-acquired": gn23_acquired, "GN24-slow": gn24_slow,
           "GN24-acquired": gn24_acquired}


def identities(presets):
    from starfinder.synthetic import CONTEXT_PRESETS
    return {name: dict(seed=CONTEXT_PRESETS[name]["seed"], acquired_spacing_zyx=list(ACQUIRED.get(name, ())) or None)
            for name in presets.SLOW}


def case_id(check, name):
    return f"{name}@acquired" if check == "GN24-acquired" else name


def main(argv=None):
    args = parse(argv)
    measures, presets = _helpers()
    selected = [c for c in cases(presets) if args.check is None or c[0] in args.check]
    if args.list:
        print(json.dumps(dict(presets=identities(presets), cases=[f"{c}[{n}]" for c, n in selected],
                              origin=dict(revision=ORIGIN_REVISION, module=ORIGIN_MODULE, tests=ORIGIN_TESTS)),
                         indent=1))
        return 0
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        sys.exit(f"--output {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(measures, presets)
    results = []
    for check, name in selected:
        start = time.perf_counter()
        try:
            Evidence.RUN[check](evidence, name)
            outcome, message = "passed", ""
        except AssertionError as error:
            outcome, message = "failed", str(error)
        except Exception as error:  # a case that raises is recorded and the next one runs
            outcome, message = "error", f"{type(error).__name__}: {error}"
        seconds = time.perf_counter() - start
        print(f"{check}[{name}] {outcome.upper()} in {seconds:.2f} s" + (f": {message}" if message else ""),
              flush=True)
        results.append(dict(check=check, case=name,
                            origin_test=f"{ORIGIN_MODULE}::{ORIGIN_TESTS[check]}[{case_id(check, name)}]",
                            outcome=outcome, message=message, wall_s=seconds))
    for name, rows in (("gn23_rows.jsonl", evidence.gn23_rows), ("gn24_rows.jsonl", evidence.gn24_rows)):
        if rows:
            (output / name).write_text("".join(json.dumps(r, sort_keys=True) + "\n" for r in rows), encoding="utf-8")
    (output / "results.json").write_text(json.dumps(dict(
        script="benchmarks/cellular_evidence.py", software=_revision(), python=sys.version.split()[0],
        origin=dict(revision=ORIGIN_REVISION, module=ORIGIN_MODULE), presets=identities(presets),
        bound=dict(rg1_seconds=presets.BOUND_S, rg1_rss_kib=presets.BOUND_KIB), cases=results),
        indent=1, default=float) + "\n", encoding="utf-8")
    print(f"{len(results)} cases: " + ", ".join(f"{sum(r['outcome'] == o for r in results)} {o}"
                                                 for o in ("passed", "failed", "error")))
    return 1 if any(r["outcome"] != "passed" for r in results) else 0


if __name__ == "__main__":
    sys.exit(main())
