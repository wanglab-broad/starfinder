"""The real-bank appearance checks GA23, GA24 (``slow`` part), GA26 and GA27 as an evidence script (W-536).

These checks of design/validation/synthetic-real-appearance-scene.md ran in the controller's ``pytest-slow-synthetic``
selection until W-536 moved them to the Chapter II evidence project. They read the three real banks, rebuild them from
real images, and allow up to 3,600 s and 16 GiB per step (RG1), so they are evidence workloads rather than software
checks. The bodies are the tests of ``src/python/test/test_synthetic_appearance_presets.py`` at revision
``ORIGIN_REVISION``, unchanged in what they assert; the helpers stay in ``src/python/test/`` and are imported from
there. The routine part of GA24, GA8 to GA18 and the closed-form bank checks stay in the suite.

    cd src/python
    STARFINDER_APPEARANCE_BANKS=<bank directory> uv run python ../../benchmarks/appearance_evidence.py \\
        --output <run dir>/appearance
    uv run python ../../benchmarks/appearance_evidence.py --list

Checks, in the order the tests ran (``--check`` selects some; GA23 measures the scenes GA24 generated, so selecting
GA23 also generates them):

* ``GA24-slow``: each ``slow`` preset generated in its own process under /usr/bin/time -v, within RG1;
* ``GA24-acquired``: the 2D tissue and culture ``slow`` presets at the acquired pixel, within RG1;
* ``GA24-build``: each real bank rebuilt from its build configuration in its own process, within RG1;
* ``GA26``: the real banks against their records, W-392's inventory and W-513's counts;
* ``GA27``: a rebuild gives the recorded digest (the rebuild of ``GA24-build``);
* ``GA23-slow`` and ``GA23-acquired``: each row of "Target definitions" on the saved scenes.

Each step runs on one CPU with one numerical thread; the scenes are saved under ``--output``, which must be new or
empty, and the rebuilt bank files are deleted once measured. The printed lines are those of the tests (``GA23 CSV
BEGIN``/``END`` blocks, ``GA23 ROW`` JSON lines). ``--output`` also receives ``results.json`` (each case's outcome
and wall time, the bank identities, the seeds and the revisions) and ``ga23_targets.csv``. Exit status 0 when every
selected case passed, 1 when one failed, 2 when none failed but one was skipped (a missing bank, inventory or
/usr/bin/time, each named). Do not run with ``python -O``, which drops the assertions.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
PACKAGE = ROOT / "src" / "python"
# The revision whose test suite last held these checks as tests (W-536's starting revision).
ORIGIN_REVISION = "0b9fa8b8e840697408557c81b2d6e426b858e209"
ORIGIN_MODULE = "src/python/test/test_synthetic_appearance_presets.py"
CHECKS = ("GA24-slow", "GA24-acquired", "GA24-build", "GA26", "GA27", "GA23-slow", "GA23-acquired")
# The test each check was, at ORIGIN_REVISION.
ORIGIN_TESTS = {"GA24-slow": "test_ga24_slow_preset_generation_within_rg1",
                "GA24-acquired": "test_ga24_acquired_pixel_variant_within_rg1",
                "GA24-build": "test_ga24_bank_build_within_rg1", "GA26": "test_ga26_real_banks",
                "GA27": "test_ga27_a_rebuild_gives_the_recorded_digest",
                "GA23-slow": "test_ga23_slow_preset_in_its_target_ranges",
                "GA23-acquired": "test_ga23_acquired_variant_edge_and_bright_regions"}
# W-392's inventory of the real inputs, with the SHA-256 of each file (GA26).
W392_INVENTORY = Path("/home/unix/jiahao/wanglab/jiahao/test/starfinder_benchmark/runs/W-392/20261009T0323Z-inputs/"
                      "inventory")
# SRC2's held-out image of each bank, never a source.
HELD_OUT = {"ln-sample1-v1": "Position048", "tissue2d-v1": "crop_y18432_x0", "culture3d-v1": "Position397"}
# SEL1's counts per source image, W-513 tables/bank_selection_<context>.csv (design/specs/synthetic/real-appearance.md,
# "Selection of entries"): labels, left out by rule 1 (by the Z end alone, in LN where that rule applies), by rule 2,
# by rule 3, by rule 4, kept; and the plateau.
W513 = {
    "ln-sample1-v1": {"Position049": (2475, 1018, 993, 146, 0, 110, 1201, 45.0),
                      "Position057": (2405, 856, 838, 156, 0, 102, 1291, 39.0),
                      "Position058": (1207, 565, 558, 66, 0, 35, 541, 50.0)},
    "tissue2d-v1": {"crop_y20480_x2048": (291, 23, 0, 28, 0, 69, 171, 80.0),
                    "crop_y20480_x4096": (242, 34, 0, 22, 0, 53, 133, 92.5)},
    "culture3d-v1": {"Position398": (54, 23, 0, 4, 0, 0, 27, 48.0), "Position407": (56, 17, 0, 4, 0, 0, 35, 46.0),
                     "Position408": (57, 22, 0, 4, 2, 0, 29, 45.0)},
}
COUNTS = ("labels", "left_out_border", "left_out_border_z_alone", "left_out_size_after_border",
          "left_out_nucleus_after_size", "left_out_squashed_after_rest", "kept")
ENTRIES = {"ln-sample1-v1": 3033, "tissue2d-v1": 304, "culture3d-v1": 91}
# O8's g, the median plateau of each bank's source images.
PLATEAU = {"ln-sample1-v1": 45.0, "tissue2d-v1": 86.25, "culture3d-v1": 46.0}
# The objects of each slow preset at the prototype's density (W-513 tables/resource_estimates.csv; estimates), against
# which the bank size is reported (BS2).
ESTIMATED_OBJECTS = {"ln-sample1-v1": 738, "tissue2d-v1": 337, "culture3d-v1": 32}
GUARD = 1.5
# The rows of "Target definitions" each scene reports: TR4 and TR5 of 2D tissue and culture on app_slow_acquired.
GA23_ROWS = {"dense3d": {"TR1", "TR2", "TR2v", "TR3", "TR4", "TR5", "TR6", "TR7", "TR8", "TR8r", "TR8v", "TR9", "TR10",
                         "TR11"},
             "tissue2d": {"TR1", "TR2", "TR2v", "TR3", "TR6", "TR7", "TR8", "TR8r", "TR8v", "TR9", "TR10", "TR12"},
             "culture": {"TR1", "TR2", "TR2v", "TR3", "TR6", "TR7", "TR8", "TR8r", "TR8v", "TR9", "TR10"}}


def parse(argv):
    parser = argparse.ArgumentParser(
        description="Run the real-bank appearance checks GA23, GA24 (slow presets, acquired variants, bank builds), "
                    "GA26 and GA27 of design/validation/synthetic-real-appearance-scene.md, moved out of the test suite by "
                    "W-536. Reads the real banks through STARFINDER_APPEARANCE_BANKS; RG1 allows one CPU, 16 GiB and "
                    "3,600 s per step.",
        epilog="Run from src/python: STARFINDER_APPEARANCE_BANKS=<bank directory> uv run python "
               "../../benchmarks/appearance_evidence.py --output <run dir>/appearance")
    parser.add_argument("--output", type=Path, help="new or empty directory for the scenes, results.json and "
                                                     "ga23_targets.csv (outside the repository)")
    parser.add_argument("--check", action="append", choices=CHECKS,
                        help="run only this check (repeatable); default: all, in the order listed")
    parser.add_argument("--list", action="store_true",
                        help="import the helpers and list the cases with their banks and seeds; reads no bank")
    args = parser.parse_args(argv)
    if not args.list and args.output is None:
        parser.error("--output is required unless --list is given")
    return args


def _helpers():
    """The test helpers, imported from src/python/test (the package ``test`` of this checkout, not the standard
    library's)."""
    if str(PACKAGE) not in sys.path[:1]:
        sys.path.insert(0, str(PACKAGE))
    from test import cellular_appearance_measures as measures
    from test import cellular_appearance_scenes as scenes
    return measures, scenes


def _revision():
    spec = importlib.util.spec_from_file_location("benchmark_revision", Path(__file__).resolve().parent / "revision.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.revision()


def cases(scenes):
    """[(check, case)] in the order the tests ran."""
    banks = tuple(scenes.SLOW_BANKS.values())
    return ([("GA24-slow", n) for n in scenes.SLOW_PRESETS] + [("GA24-acquired", n) for n in scenes.ACQUIRED_SPACING]
            + [("GA24-build", b) for b in banks] + [("GA26", b) for b in banks] + [("GA27", b) for b in banks]
            + [("GA23-slow", n) for n in scenes.SLOW_PRESETS] + [("GA23-acquired", n) for n in scenes.ACQUIRED_SPACING])


class Evidence:
    """The checks, with the session caches of the tests' ``app_slow`` and ``rebuilt`` fixtures under ``output``."""

    def __init__(self, output, measures, scenes):
        self.output, self.measures, self.scenes = output, measures, scenes
        self.generated, self.rebuilt_banks, self.ga23_rows = {}, {}, []

    # ``app_slow`` and ``app_slow_acquired``: a slow preset generated once in its own process under /usr/bin/time -v.
    def app_slow(self, name, acquired=False):
        s = self.scenes
        if (name, acquired) not in self.generated:
            s.real_bank(s.SLOW_BANKS[name])
            directory = self.output / "scenes" / (f"{name}-acquired" if acquired else name)
            self.generated[name, acquired] = s.generate_measured(
                name, directory, s.ACQUIRED_SPACING[name] if acquired else None)
        return self.generated[name, acquired]

    # GA27's rebuild of each real bank, once, in its own process under /usr/bin/time -v.
    def rebuilt(self, name):
        s = self.scenes
        if name not in self.rebuilt_banks:
            self.rebuilt_banks[name] = (s.real_bank(name), s.rebuild_measured(s.real_bank(name),
                                                                              self.output / "rebuild" / name))
        return self.rebuilt_banks[name]

    def within_rg1(self, result, what):
        assert result.exit == 0, f"{what}: exit {result.exit}\n{result.output}"
        assert result.wall_s <= self.scenes.RG1_SECONDS and result.peak_rss_kib <= self.scenes.RG1_RSS_KIB, (
            f"{what}: {result.wall_s} s and {result.peak_rss_kib} KiB against RG1")

    def report_slow(self, name, result, label):
        from starfinder.synthetic import APPEARANCE_PRESETS
        summary = result.summary
        density = summary["density"]
        assert summary["bank"]["sha256"] == APPEARANCE_PRESETS[name]["bank_sha256"]
        assert density["mode"] == "visible" and density["target"] == APPEARANCE_PRESETS[name]["target"]
        assert density["stopped_by"] == "target" or density["reached"] >= density["floor"] * density["target"]
        print(f"GA24 {label}: process {result.wall_s:.2f} s (generation {summary['generation_s']:.2f} s), peak RSS "
              f"{result.peak_rss_kib} KiB; {summary['placed']} objects in {summary['proposals']} proposals, visible "
              f"share {density['reached']:.4f} against {density['target']:.4f} ({density['stopped_by']}); bank cycles "
              f"{summary['bank']['cycles']}")

    # ------------------------------------------------------------------------------------------------ GA24

    def ga24_slow(self, name):
        result = self.app_slow(name)
        self.within_rg1(result, name)
        self.report_slow(name, result, name)

    def ga24_acquired(self, name):
        import pytest
        spacing = self.scenes.ACQUIRED_SPACING[name]
        result = self.app_slow(name, acquired=True)
        self.within_rg1(result, f"{name} at the acquired pixel")
        assert result.summary["spacing_zyx"] == pytest.approx(list(spacing), abs=1e-12)
        self.report_slow(name, result, f"{name} at {spacing[1]} µm")

    def ga24_build(self, name):
        _, result = self.rebuilt(name)
        self.within_rg1(result, f"build of {name}")
        print(f"GA24 build {name}: process {result.wall_s:.2f} s (build {result.summary['build_s']:.2f} s), peak RSS "
              f"{result.peak_rss_kib} KiB, file {result.summary['file']['size']} bytes")

    # ------------------------------------------------------------------------------------------------ GA26 and GA27

    @staticmethod
    def inventory():
        """W-392's inventory: path -> SHA-256; a skip naming the file where it is missing."""
        import pytest
        out = {}
        for context in ("ln", "culture", "tissue2d"):
            path = W392_INVENTORY / f"{context}.json"
            if not path.is_file():
                pytest.skip(f"no file {path}: GA26 compares the build records' inputs with W-392's inventory")
            out.update({f["path"]: f["sha256"] for f in json.loads(path.read_text(encoding="utf-8"))["files"]})
        return out

    @staticmethod
    def sha256(path):
        import hashlib
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 22), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def crops_meet(a, b):
        if a is None or b is None:
            return True
        return a[0] < b[0] + b[2] and b[0] < a[0] + a[2] and a[1] < b[1] + b[3] and b[1] < a[1] + a[3]

    def ga26(self, name):
        import pytest
        from starfinder.synthetic import BankReference, load_appearance_bank
        bank = self.scenes.real_bank(name)
        record = bank.record
        # The file is the one its preset names, and its arrays give the recorded digest.
        assert self.sha256(bank.path) == bank.sha256 == record["file"]["sha256"]
        loaded = load_appearance_bank(BankReference(name=name, sha256=bank.sha256))
        assert loaded.digest == record["digest"] and len(loaded) == ENTRIES[name] == record["counts"]["entries"]
        # Every input is inventoried by W-392 with an equal SHA-256; the configuration is the recorded one.
        known = self.inventory()
        files = [*record["inputs"], *(i for h in record["held_out"] for i in h["inputs"])]
        assert files and all(f["path"] in known and known[f["path"]] == f["sha256"] for f in files), [
            f["path"] for f in files if known.get(f["path"]) != f["sha256"]]
        assert [s["image"] for s in bank.config["sources"]] == [s["image"] for s in record["sources"]]
        # The held-out image is never a source: not among the images, and no source reads its pixels.
        assert [h["image"] for h in record["held_out"]] == [HELD_OUT[name]]
        images = [i["image"] for i in record["images"]]
        assert HELD_OUT[name] not in images and images == list(W513[name])
        for held in record["held_out"]:
            for f in held["inputs"]:
                for source in record["inputs"]:
                    if source["sha256"] == f["sha256"]:
                        assert not self.crops_meet(held.get("crop_yx"), source["crop_yx"]), (held["image"],
                                                                                              source["image"])
        # SEL1's counts per image equal W-513's, and the plateaus; BR4's guard on every source.
        for image in record["images"]:
            *counts, plateau = W513[name][image["image"]]
            assert tuple(image[k] for k in COUNTS) == tuple(counts), image["image"]
            assert image["plateau"] == pytest.approx(plateau, abs=1e-9)
            assert 1 / GUARD <= image["plateau_over_median"] <= GUARD
            assert 1 / GUARD <= image["visible_over_plateau"] <= GUARD
        assert record["plateau_median"] == pytest.approx(PLATEAU[name], abs=1e-9)
        assert record["counts"]["kept"] == sum(v[6] for v in W513[name].values()) == ENTRIES[name]
        estimate = ESTIMATED_OBJECTS[name]
        print(f"GA26 {name}: {len(loaded)} entries from {images} (held out {HELD_OUT[name]}), "
              f"{record['counts']['patches']} patches, g {record['plateau_median']}; plateau over the median "
              f"{[round(i['plateau_over_median'], 4) for i in record['images']]}, visible median over the plateau "
              f"{[round(i['visible_over_plateau'], 4) for i in record['images']]}; bank size against about {estimate} "
              f"placements (estimate): {len(loaded) / estimate:.2f} times, reuse without replacement "
              f"{max(0, estimate - len(loaded)) / estimate:.3f}")

    def ga27(self, name):
        bank, result = self.rebuilt(name)
        assert result.exit == 0, result.output
        assert result.summary["digest"] == bank.record["digest"]
        assert result.summary["counts"] == bank.record["counts"]
        print(f"GA27 {name}: rebuilt digest {result.summary['digest']} equals the record's; the rebuilt file's SHA-256 "
              f"{result.summary['file']['sha256']} (the stored {bank.sha256}; a zip archive's bytes may carry a time)")

    # ------------------------------------------------------------------------------------------------ GA23 (W-520)
    # Each row of "Target definitions" on the scenes GA24 generated and saved, read back and measured by
    # ``cellular_appearance_measures.py``. A row the page asserts and V12 allows, outside its range, is a finding.
    # Under the amendment of 2026-10-18 TR2v of dense 3D and TR8r of dense 3D and 2D tissue are reported beside their
    # range, not asserted.

    def measure_targets(self, name, result, acquired):
        from starfinder.synthetic import APPEARANCE_PRESETS, BankReference, load_appearance_bank
        m = self.measures
        assert result.exit == 0, result.output
        context = m.context_of(name)
        label = f"{name}@acquired" if acquired else name
        scene = m.load_saved_scene(result.directory / "scene", reference_round=context == "tissue2d" and not acquired)
        assert scene.record["stream_scheme"]["seed"] == 42 and scene.record["bank"]["sha256"] == APPEARANCE_PRESETS[
            name]["bank_sha256"]
        bank = load_appearance_bank(BankReference(name=self.scenes.SLOW_BANKS[name],
                                                  sha256=APPEARANCE_PRESETS[name]["bank_sha256"]))
        rows = m.appearance_targets(name, label, scene, bank, acquired=acquired)
        print(f"GA23 CSV BEGIN {label}\n{m.csv_block(rows)}\nGA23 CSV END {label}")
        for row in rows:
            print("GA23 ROW " + json.dumps(row, sort_keys=True, default=float))
        self.ga23_rows += rows
        return rows

    def assert_no_findings(self, rows):
        found = self.measures.findings(rows)
        assert not found, "GA23 findings (rows the page asserts outside their range where V12 compares them): " + (
            "; ".join(f"{r['preset']} {r['row']} {r['quantity']} {r['value']:.4g} not in [{r['low']}, {r['high']}] "
                      f"on {r['objects']} {r['kind']}" for r in found))

    def ga23_slow(self, name):
        m = self.measures
        rows = self.measure_targets(name, self.app_slow(name), acquired=False)
        context = m.context_of(name)
        assert {r["row"] for r in rows} == GA23_ROWS[context]
        if context == "dense3d":
            # TR7 of appearance-dense3d-slow-v1 is asserted against 0.25–0.41 (AM1; GN23 keeps reporting the
            # closed-form preset's): V12 allows it, so its outcome is a comparison, not a report.
            (cv,) = [r for r in rows if r["quantity"] == "within-nucleus CV"]
            assert (cv["low"], cv["high"]) == (0.25, 0.41) and cv["outcome"] in ("in range", "outside the range"), cv
        # The amended rows: reported with their value and range, under the amendment of 2026-10-18.
        amended = [r for r in rows if r["row"] in m.AMENDED.get(context, set())
                   and not r["quantity"].startswith("label share")]
        assert len(amended) == len(m.AMENDED.get(context, set()))
        assert all(r["outcome"] == f"reported: {m.AMENDMENT}" and r["low"] is not None and r["value"] is not None
                   for r in amended), amended
        compared = [r for r in rows if not r["outcome"].startswith("reported")]
        placed = next(r["objects"] for r in rows if r["row"] == "TR2v")
        print(f"GA23 {name}: {placed} objects placed; compared {len(compared)} of {len(rows)} quantities "
              f"({sum(r['outcome'] == 'in range' for r in compared)} in range)")
        self.assert_no_findings(rows)

    def ga23_acquired(self, name):
        rows = self.measure_targets(name, self.app_slow(name, acquired=True), acquired=True)
        assert {r["row"] for r in rows} == {"TR4", "TR5"}
        assert all(r["outcome"].startswith("reported") for r in rows)
        print(f"GA23 {name}@acquired: {[(r['quantity'], r['objects']) for r in rows]}")

    RUN = {"GA24-slow": ga24_slow, "GA24-acquired": ga24_acquired, "GA24-build": ga24_build, "GA26": ga26,
           "GA27": ga27, "GA23-slow": ga23_slow, "GA23-acquired": ga23_acquired}


def identities(scenes):
    """The banks by name and SHA-256 and the seed of each slow preset, as the presets name them."""
    from starfinder.synthetic import APPEARANCE_PRESET_VERSION, APPEARANCE_PRESETS
    return {name: dict(version=APPEARANCE_PRESET_VERSION, bank=scenes.SLOW_BANKS[name],
                       bank_sha256=APPEARANCE_PRESETS[name]["bank_sha256"], seed=APPEARANCE_PRESETS[name]["seed"],
                       acquired_spacing_zyx=list(scenes.ACQUIRED_SPACING.get(name, ())) or None)
            for name in scenes.SLOW_PRESETS}


def main(argv=None):
    args = parse(argv)
    measures, scenes = _helpers()
    selected = [c for c in cases(scenes) if args.check is None or c[0] in args.check]
    if args.list:
        print(json.dumps(dict(presets=identities(scenes), cases=[f"{c}[{n}]" for c, n in selected],
                              origin=dict(revision=ORIGIN_REVISION, module=ORIGIN_MODULE, tests=ORIGIN_TESTS)),
                         indent=1))
        return 0
    import pytest
    output = args.output.resolve()
    if output.exists() and any(output.iterdir()):
        sys.exit(f"--output {output} is not empty")
    output.mkdir(parents=True, exist_ok=True)
    evidence = Evidence(output, measures, scenes)
    results = []
    for check, name in selected:
        start = time.perf_counter()
        try:
            Evidence.RUN[check](evidence, name)
            outcome, message = "passed", ""
        except AssertionError as error:
            outcome, message = "failed", str(error)
        except pytest.skip.Exception as skip:
            outcome, message = "skipped", str(skip.msg)
        except Exception as error:  # a case that raises is recorded and the next one runs
            outcome, message = "error", f"{type(error).__name__}: {error}"
        seconds = time.perf_counter() - start
        print(f"{check}[{name}] {outcome.upper()} in {seconds:.2f} s" + (f": {message}" if message else ""),
              flush=True)
        results.append(dict(check=check, case=name, origin_test=f"{ORIGIN_MODULE}::{ORIGIN_TESTS[check]}[{name}]",
                            outcome=outcome, message=message, wall_s=seconds))
    measured = {f"{n}{'@acquired' if a else ''}": dict(wall_s=r.wall_s, peak_rss_kib=r.peak_rss_kib, exit=r.exit)
                for (n, a), r in evidence.generated.items()}
    measured.update({f"build {n}": dict(wall_s=r.wall_s, peak_rss_kib=r.peak_rss_kib, exit=r.exit)
                     for n, (_, r) in evidence.rebuilt_banks.items()})
    if evidence.ga23_rows:
        (output / "ga23_targets.csv").write_text(measures.csv_block(evidence.ga23_rows) + "\n", encoding="utf-8")
    (output / "results.json").write_text(json.dumps(dict(
        script="benchmarks/appearance_evidence.py", software=_revision(), python=sys.version.split()[0],
        origin=dict(revision=ORIGIN_REVISION, module=ORIGIN_MODULE), presets=identities(scenes),
        bound=dict(rg1_seconds=scenes.RG1_SECONDS, rg1_rss_kib=scenes.RG1_RSS_KIB), steps=measured, cases=results),
        indent=1, default=float) + "\n", encoding="utf-8")
    outcomes = {r["outcome"] for r in results}
    print(f"{len(results)} cases: " + ", ".join(f"{sum(r['outcome'] == o for r in results)} {o}"
                                                 for o in ("passed", "failed", "error", "skipped")))
    return 1 if outcomes & {"failed", "error"} else 2 if "skipped" in outcomes else 0


if __name__ == "__main__":
    sys.exit(main())
