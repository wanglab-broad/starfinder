"""The table "Implemented checks" of docs/synthetic-cellular-scene-validation.md, and GN25.

``test_every_gn_row_names_collected_tests`` keeps the table true: it reads the rows GN1 to GN25 of the table
"Generator checks" and checks that "Implemented checks" has exactly those rows, in that order; that each names at
least one test pytest collects; that none of them is an expected failure or skipped by a mark; that each carries the
subsystem marker ``synthetic`` and a kind marker the GN row names; and that the tier of the line matches the tests'
``slow`` markers. ``test_gn25_existing_tests_unchanged`` is GN25: the files the row names, with their content at
7d6c417 (the base revision of the §2.12 implementation issues; the same at 943bb6b except the golden test, which W-394
added after it), pinned by SHA-256 (line endings read as LF).
"""
import ast
import hashlib
from pathlib import Path
import re

import pytest

pytestmark = [pytest.mark.synthetic, pytest.mark.contract]

TEST = Path(__file__).resolve().parent
DESIGN = TEST.parents[2] / "docs" / "synthetic-cellular-scene-validation.md"
SECTION = "## Implemented checks\n"
TIERS = {"routine": (False,), "slow": (True,), "routine and slow": (False, True)}
KINDS = {"contract", "validation", "golden"}

# GN25: test/test_synthetic_sample_scene.py, test/test_synthetic_sample_golden.py, test/report_scene.py and
# test/test_report_chain.py; the tests docs/synthetic-sample-scene.md names in "Opt-in, and what stays as it is" (the
# files with the synthetic marker, the tests that consume existing presets' output with their helpers and the pinned
# R6 tables; the session fixture small_dataset of conftest.py, pinned below by its source); and those named by
# docs/stitching-assembly-validation.md and docs/sample-segmentation-validation.md that generate a sample preset
# (directly, or through test/sample_segmentation_fixtures.py).
PINS = {
    "test_synthetic_sample_scene.py":
        "2abe07ab33ba9460e7b90ba0bf4aee17d33a42ba4f8875bc8f40ba54ac00c438",
    "test_synthetic_sample_golden.py":
        "b490f716c2844fdce89d7fe800c57fd2e040f0407096c71442c0e9e6745c65d1",
    "report_scene.py":
        "6faf88d709944d38f82be7f05c5b0e50c5c5f78b3a74f88581048f21f7e9c683",
    "test_report_chain.py":
        "ef2e0e2c99c7c29904317517c095ea1881148573509aff8989e1721888c1aed4",
    "test_synthetic_contracts.py":
        "090a3025e881d992223c2085ace5d4c91538ded4b4c58ec9378fecd645d577db",
    "test_formed_scene.py":
        "a5c3448080a8f371f9e86eeabe2ad2522d49877c2efe97e28b23ad7dea9ade16",
    "test_formed_geometry.py":
        "41b2ef75ed9efce64d610f80d1c0b1a0fcac2fd740e2fcb373b0301d9ee996ca",
    "test_formed_streams.py":
        "eeaf322607542885aa79aa839356e5fcdbd02910dc4c821d3416076bb08d7127",
    "test_development_fixtures.py":
        "4a929af4febeb11dc7cac96497fd2ce4d0b28c64fd994035c6f6ceba56fac028",
    "test_development_presets.py":
        "a307dec2024936cead25b69aaa4cbaa4343c786077baf48841af1c969f0636d7",
    "test_calibrated_presets.py":
        "a1fe6435370d4174aff6ef68eb78948ed0ea4cb1b7fe21bbd4ee216c0779bed9",
    "test_background_noise.py":
        "8563c46ad68a0efc642075fa6e5e466bb79bfc497c550fd7bb382bdc97760bb6",
    "test_readout_effects.py":
        "bdaf5f02d5bf209e703d07aa194ec90b5437490bd09e33bfb6fbd5d428b1079f",
    "test_benchmark_synthetic.py":
        "725799fd5e04bef3f462d8120fa3bfbc858dc524c013ac4e6c0f0a62713b7711",
    "test_readout_validation.py":
        "18ebb6d5382ef716438763d524840ff8e2d40d60377dc5604d288c99b9f4a537",
    "test_spot_finding_validation.py":
        "c439ed0dbd30a336be81f25e69c77729bf84eaf98a46465ff8f08d08d5e4c16b",
    "test_spot_finding_rounds.py":
        "68ef93f352a845b533c6fb73792152a6d1a22755338cfa358ee3b7517ccd1292",
    "spot_finding_scenes.py":
        "a0811b27c3fff22a8c4ac5262f198ac2b1f3b4b2a1ef3de3ea5aefed2e9919f1",
    "spot_finding_fixtures.py":
        "c75cbe2f0410ce22610095b3ee1bab618970f04b1d1e7027f733e2f6b1fa2bcc",
    "test_assignment.py":
        "4fb6acea3b98891de6395cd95ee584b26ca13d695e84bc40e7048d6c7b1fc298",
    "test_preprocessing_synthetic_evaluation.py":
        "d7797b6c27fecadb49afcfe174fd0540c871f45702ca689dfc7254ad0757ff9f",
    "test_preprocessing_calibrated_evaluation.py":
        "d6320e7cfd2db5ded6281a8c5e4549a9952c45a77733dc2b0f2692f7ee725182",
    "test_image_statistics.py":
        "a28a4d472481f332a3723a82daee0601319283ea4fbfb9ee11fd14d54861a1d0",
    "test_segmentation_labels.py":
        "bd6530877ebf47b84fecb1e48512823afa998bff2f07ba6a945ae17e5bca1de9",
    "test_segmentation_segment.py":
        "7ced52bc98ed8f026664e04a690374954cd9a775f50ffead26f5e87d77667bc5",
    "test_dataset_channels.py":
        "96e9064440307bd07a1c2275f6a3e03de67c4de8481bff8136ccb05d0d70773a",
    "test_stitching_assembly_scene.py":
        "077090f7b962b8c14aa0e9ae8c4a31e2dd05ef21dde77303505f59003c185c57",
    "test_stitching_multiview.py":
        "a817d28f647f07ddf4e984a53e11826193f301977194b8c5a68a595aa2ec6fee",
    "test_stitching_placement.py":
        "fcfadc631da5684c8a96f02a4f91dab7b015ef6e69ce96a0cdc9ab7ca853e620",
    "test_stitching_positions.py":
        "2cdfdfcdad0ad894a4e0f77878acc52d81d5fbd564ddad59539df505c0b6c0c9",
    "test_stitching_supplied.py":
        "c9898098cc0b8bc1466c277f05eec379407fffc12bdaa19f0b5ecaa1e1fd4b3d",
    "test_stitching_validation.py":
        "7ad81798af4bf97bf60542ad73150c65d7f703b3e9bf3610c365ffbd1ddc5295",
    "sample_segmentation_fixtures.py":
        "c8170c5132ebafd94fcbfb34eb28ce4a723755acc5a41d7a0520eff571355804",
    "test_sample_segmentation_compare.py":
        "a76379f83419f79376e79ee5c49450349ffe4103be3dde33d5068dccdad7cfb0",
    "test_sample_segmentation_readers.py":
        "ab3ccf43e45e7da68844a6b1cdec33a99cc264fd5a9b2508802969d9ac6101a3",
    "test_sample_segmentation_route.py":
        "6bf8202f69928790d17889bdf076a7509c059529cf4eed9fd2291c60ce0f5f6c",
    "test_sample_segmentation_segment.py":
        "949659c1de30b0789aeb83098837084dbc3d02b3a857801e14cb48769a787542",
    "test_sample_segmentation_workflow.py":
        "33350146fda2f4eef4e4215ec65d01dd79a66b4806c535a4dea168257b23fa08",
    "data/readout_r6_cal_141c093/background_only_codebook_aware.csv":
        "eb1e7343b81d2f82f02c3e5c2bfcd56141b4168243a254a16511792d36e6b667",
    "data/readout_r6_cal_141c093/background_only_wta.csv":
        "e8df0e2c37ca995ef8f046e0099935835c253d5f5c2d3a55b4f66bed53d052b9",
    "data/readout_r6_cal_141c093/combined_codebook_aware.csv":
        "36deaa2cdccc7c0cc2eab0fdd686f677d4dbd6de1fdfccb3eebdfc06bbd47f51",
    "data/readout_r6_cal_141c093/combined_wta.csv":
        "7bef145328a088c2e5f0512904c719789a3b55db32f10bc3615ec559945e44ff",
    "data/readout_r6_cal_141c093/gain_codebook_aware.csv":
        "1155f9f88e3281f008380901437cec1c7684bf89d92100ce72964c2e843305b5",
    "data/readout_r6_cal_141c093/gain_wta.csv":
        "d7374eb5ea9ea8e737f99c1bce681a019f147ee77bc4391a813651f3f4af6f50",
    "data/readout_r6_cal_141c093/mixing_codebook_aware.csv":
        "6a01339ae27a0d24ea9d69e730112f6141c9edf6264c4c3093ee0e01e7bbeb0c",
    "data/readout_r6_cal_141c093/mixing_wta.csv":
        "902a0d15f12e0e6fb1f57c3ffa9e5127f75f2eac3898501b68fb03490bb0a0f3",
    "data/readout_r6_cal_141c093/noise_codebook_aware.csv":
        "5c8242aaebffe7806af68969fee99b16ee9164a22ba10256225ed7b5f869b399",
    "data/readout_r6_cal_141c093/noise_wta.csv":
        "2dec97abe38b4439136f4fcbf4b6222347fbe66ddaace805aa0680ef3f7aefcd",
    "data/readout_r6_cal_141c093/round_effect_only_codebook_aware.csv":
        "9e7ffab01630429cc7b5011f7e6c55afd7930917fd5708b55848294bfc5795a1",
    "data/readout_r6_cal_141c093/round_effect_only_wta.csv":
        "c80606b85831185482d0a67cd93bccc704289c37dde174d4bb32bc928bf16baa",
    "data/readout_r6_cal_141c093/trend_codebook_aware.csv":
        "69e7d7ff2896ed8f43c6368f629c3d675578f65afdf4d2489a7acdac3464f05c",
    "data/readout_r6_cal_141c093/trend_wta.csv":
        "1a4417d3a1259e1dfb1d41021da7421dbade948481eae284ebcaad7b92bcdf0e",
    "data/readout_r6_cal_141c093/weakening_codebook_aware.csv":
        "22db7fa294ddd0162b6f30e16471c239bba8d6035d3eb7b986def5a1843e6611",
    "data/readout_r6_cal_141c093/weakening_wta.csv":
        "2335b7b4a74690742b7758f4a1f91dc02c455c778c16d4a41453d0443f16a00d",
}
R6 = "data/readout_r6_cal_141c093"
# The session fixture small_dataset of conftest.py and the two constants it uses, by the SHA-256 of their source.
CONFTEST_PINS = {
    "SMALL_DATASET_ARGS": "be99aa27ae579c9610ef2872afa9f27805476486b24511750571b3ef545bdc29",
    "SMALL_DATASET_GENERATOR": "4150f82a632b679e028fd12fefa75b5dd978943b618f0f0391b0e01daec2473f",
    "small_dataset": "58876bd4e06b27bd30d5f3e23c89149617814a0bb364a3ac52222167517cec32",
}


def digest(data):
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


def conftest_sources():
    """{name: source} of the top-level assignments and functions of conftest.py named in CONFTEST_PINS, a function
    with its decorators."""
    text = (TEST / "conftest.py").read_text(encoding="utf-8").replace("\r\n", "\n")
    lines, found = text.splitlines(), {}
    for node in ast.parse(text).body:
        if isinstance(node, ast.FunctionDef):
            name, start = node.name, (node.decorator_list[0] if node.decorator_list else node).lineno
        elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            name, start = node.targets[0].id, node.lineno
        else:
            continue
        if name in CONFTEST_PINS:
            found[name] = "\n".join(lines[start - 1:node.end_lineno])
    return found


def test_gn25_existing_tests_unchanged():
    changed = [name for name, pin in PINS.items() if digest((TEST / name).read_bytes()) != pin]
    sources = conftest_sources()
    assert sorted(sources) == sorted(CONFTEST_PINS)
    changed += [f"conftest.py::{name}" for name, pin in CONFTEST_PINS.items()
                if hashlib.sha256(sources[name].encode("utf-8")).hexdigest() != pin]
    assert not changed, f"GN25: files that must stay as at 7d6c417 were edited: {changed}"
    held = sorted(p.relative_to(TEST).as_posix() for p in (TEST / R6).rglob("*") if p.is_file())
    assert held == sorted(name for name in PINS if name.startswith(R6 + "/"))


# --- The table "Implemented checks" -----------------------------------------------------------------------------------

def gn_rows():
    """{row: the backticked markers of its "Tier and markers" cell} of the table "Generator checks", in page order."""
    text = DESIGN.read_text(encoding="utf-8").split("## Generator checks\n", 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in text.splitlines():
        match = re.match(r"\| (GN\d+) \|", line)
        if match:
            cells = [cell.strip() for cell in line.strip().strip("|").split(" | ")]
            assert len(cells) == 7 and match.group(1) not in rows, line[:40]
            rows[match.group(1)] = set(re.findall(r"`([a-z_]+)`", cells[6]))
    return rows


def implemented_checks():
    """{row: (the backticked names of its "Module and tests" cell, its tier)}, in table order."""
    section = DESIGN.read_text(encoding="utf-8").split(SECTION, 1)[1].split("\n## ", 1)[0]
    rows = {}
    for line in section.splitlines():
        match = re.match(r"\| (GN\d+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \| ([^|]+) \|$", line)
        if match:
            assert match.group(1) not in rows, match.group(1)
            rows[match.group(1)] = (re.findall(r"`([^`]+)`", match.group(2)), match.group(4).strip())
    return rows


def named_tests(names):
    """(module, test name) of a row, each test under the module named before it."""
    module, named = None, []
    for name in names:
        if name.endswith(".py"):
            module = name
        elif name.startswith("test_"):
            assert module is not None, name
            named.append((module, name))
    return named


def collected(session, module):
    """The items pytest collects from one test module."""
    items, nodes = [], [pytest.Module.from_parent(session, path=TEST / module)]
    while nodes:
        node = nodes.pop()
        for child in node.collect():
            (items if isinstance(child, pytest.Item) else nodes).append(child)
    return items


def test_every_gn_row_names_collected_tests(request):
    expected, rows = gn_rows(), implemented_checks()
    assert list(expected) == [f"GN{k}" for k in range(1, 26)]
    assert list(rows) == list(expected)
    modules = sorted({module for names, _ in rows.values() for module, _ in named_tests(names)})
    items_of = {module: collected(request.session, module) for module in modules}
    for row, (names, tier) in rows.items():
        named = named_tests(names)
        assert named and tier in TIERS, row
        kinds = expected[row] & KINDS
        items = []
        for module, name in named:
            found = [item for item in items_of[module] if item.originalname == name]
            assert found, (row, module, name)
            items += found
        for item in items:
            marks = {mark.name for mark in item.iter_markers()}
            assert not marks & {"xfail", "skip", "skipif"}, item.nodeid
            assert "synthetic" in marks and len(marks & KINDS) == 1, item.nodeid
            # The kind marker is one the GN row names (GN25's "their own tiers and markers" names none).
            assert not kinds or marks & kinds, (row, item.nodeid)
        slow = {item.get_closest_marker("slow") is not None for item in items}
        assert tuple(sorted(slow)) == TIERS[tier], (row, tier)
