"""Compare four Python stages with committed MATLAB reference outputs on a literal fixture (W-212).

design/validation/matlab-stage-comparison.md fixes the fixture, every mapping and every tolerance;
this tool implements that page. Run from src/python:

    uv run python scripts/matlab_stage_comparison.py prepare <dir>
    STARFINDER_MATLAB_EXECUTABLE=<matlab> uv run snakemake --snakefile <dir>/Snakefile --cores 1 --directory <dir>
    uv run python scripts/matlab_stage_comparison.py export-reference <dir>
    uv run python scripts/matlab_stage_comparison.py compare <dir>

prepare writes the fixture as MATLAB inputs and a Snakefile that runs workflow/scripts/compare_matlab_stages.m
through run_matlab_scripts (workflow/rules/common.smk, matlab_launcher.py); export-reference turns the MATLAB
outputs into the reference files under src/python/test/data/matlab_reference/ and writes nothing else in the
repository; compare runs the comparison against those files. test/test_matlab_stage_comparison.py runs the same
comparison without MATLAB.
"""
import argparse
import hashlib
import json
import subprocess
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.readout import (
    Codebook,
    NeighborhoodSumConfig,
    WtaDecoderConfig,
    bases_to_colors,
    decode_reads,
    extract_intensities,
    filter_reads,
)
from starfinder.registration import TranslationConfig, apply_transform, estimate_transform
from starfinder.spot_finding import LocalMaximaConfig, SpotFindingResult, SpotFindingWarning, find_spots

REPO = Path(__file__).resolve().parents[3]
PROTOCOL = Path("design/validation/matlab-stage-comparison.md")
REFERENCE = REPO / "src/python/test/data/matlab_reference"
ENTRY = "compare_matlab_stages"
FIXTURE_ID = "matlab-stage-comparison-v1"
SCHEMA = "starfinder.matlab_stage_reference/1"
DEPTHS = (5, 1)
RADII = (0, 1)
ROUNDS = ("round10", "round2")
CHANNELS = ("ch02", "ch00", "ch03", "ch01")
COLOR_TO_CHANNEL = {"1": 1, "2": 0, "3": 3, "4": 2}
CODEBOOK = {"gene-A": "12", "gene-B": "21"}
BASES = {"gene-A": "AAC", "gene-B": "ACC"}
SPOT_IDS = ("A", "B", "zero", "tie")
# The literal neighbourhood sums (spot, channel, round) at both radii: the boxes of radius 1 hold one peak each.
SUMS = np.array([[[0, 9], [7, 0], [0, 0], [0, 0]], [[11, 0], [0, 13], [0, 0], [0, 0]],
                 [[0, 0], [0, 0], [0, 0], [0, 0]], [[5, 5], [5, 5], [0, 0], [0, 0]]], dtype=np.float64)
# Per-round WTA score -log(a / (a + 1e-6)) = log1p(1e-6 / a) of the winning amplitudes of A and B.
ROUND_SCORES = np.log1p(1e-6 / np.array([[7.0, 9.0], [11.0, 13.0]]))
CORRECTION_ZYX = (0.0, -1.0, 1.0)
DISTANCE = 1e-9
ATOL = 1e-12
# The MATLAB functions the entry script calls; export-reference refuses outputs if one changed after prepare.
MATLAB_SOURCES = tuple(f"src/matlab/{name}.m" for name in (
    "SpotFindingMax3D", "DFTRegister3D", "DFTApply3D", "DFTRegister2D", "DFTApply2D", "ExtractFromLocation",
    "EncodeBases", "DecodeCS", "FilterReads")) + ("workflow/scripts/compare_matlab_stages.m",)
SOURCES = MATLAB_SOURCES + ("workflow/rules/matlab_launcher.py", "workflow/rules/common.smk",
                            "src/python/scripts/matlab_stage_comparison.py", str(PROTOCOL))
RAW_OUTPUTS = ("matlab-command.json", "matlab-runtime.json", "matlab-ver.txt") + tuple(
    f"z{d}-matlab.mat" for d in DEPTHS) + tuple(f"z{d}-r{r}-filtered.csv" for d in DEPTHS for r in RADII)
TOOLBOXES = ("Image Processing Toolbox", "Statistics and Machine Learning Toolbox")


@dataclass(frozen=True)
class Mapping:
    """MATLAB to Python conventions (design record, "Mappings"); a broken field makes the comparison fail.

    point_axes: the MATLAB (x, y, z) columns that give Python (z, y, x); index_base: subtracted from MATLAB
    points; image_axes: the MATLAB (row, column, Z) axes that give Python ZYX; shift_order: the positions of
    MATLAB's [row, column, z] correction that give the ZYX correction; detection_base: subtracted from MATLAB
    detection channels; extraction_channels: MATLAB extraction input channel m is Python channel
    extraction_channels[m - 1] - 1.
    """
    point_axes: tuple = (2, 1, 0)
    index_base: int = 1
    image_axes: tuple = (2, 0, 1)
    shift_order: tuple = (2, 0, 1)
    detection_base: int = 1
    extraction_channels: tuple = (2, 1, 4, 3)


MAPPING = Mapping()


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=1, allow_nan=False) + "\n")


def fixture(depth):
    """The literal arrays of one depth: centers (zero-based ZYX of A, B, zero, tie) and the ZYX(C) images."""
    centers = np.array([[depth // 2, 2, 3], [depth // 2, 6, 7], [0, 0, 0], [depth // 2, 4, 5]], dtype=np.int64)
    first = np.zeros((depth, 9, 11, 4), dtype=np.uint16)
    second = np.zeros_like(first)
    first[tuple(centers[0]) + (1,)] = 7
    first[tuple(centers[1]) + (0,)] = 11
    second[tuple(centers[0]) + (0,)] = 9
    second[tuple(centers[1]) + (1,)] = 13
    extract_first = first.copy()
    for image in (extract_first, second):
        image[tuple(centers[3]) + (0,)] = 5
        image[tuple(centers[3]) + (1,)] = 5
    reference = first.sum(axis=-1, dtype=np.float64)
    moving = np.roll(reference, (1, -1), axis=(1, 2))
    return dict(centers=centers, first=first, second=second, extract_first=extract_first, reference=reference,
                moving=moving)


def fixture_sha256():
    """SHA-256 over each array's depth, name, dtype, shape and C-order bytes, in a fixed order."""
    h = hashlib.sha256(FIXTURE_ID.encode())
    for depth in DEPTHS:
        for name, array in sorted(fixture(depth).items()):
            h.update(f"|{depth}|{name}|{array.dtype.str}|{array.shape}|".encode())
            h.update(np.ascontiguousarray(array).tobytes())
    return h.hexdigest()


def matlab_invocation(root):
    """The trusted MATLAB argument expression and entry point that run_matlab_scripts receives."""
    return "'" + str(Path(root) / "matlab-config.json").replace("'", "''") + "'", ENTRY


def snakemake_config(root):
    """The configuration the Snakefile sets before including common.smk: a one-FOV stub and the path launcher."""
    root = Path(root)
    return dict(starfinder_path=str(REPO), root_input_path=str(root / "fixture"),
                root_output_path=str(root / "fixture"), dataset_id="sample", sample_id="sample", output_id="output",
                config_path=str(root / "unused.yaml"), n_rounds=2, n_fovs=1, fov_id_pattern="FOV_{i}",
                subset_list=[], rules={}, backend="matlab", matlab_launcher="path", matlab_single_thread=True)


def snakefile_text(root):
    root = Path(root)
    param, entry = matlab_invocation(root)
    outputs = ", ".join(repr(str(root / name)) for name in RAW_OUTPUTS)
    return f'''import json
import sys
# common.smk imports matlab_launcher from the workflow's rules folder.
sys.path.insert(0, {str(REPO / "workflow/rules")!r})
config.update({snakemake_config(root)!r})
include: {str(REPO / "workflow/rules/common.smk")!r}

rule compare_matlab_stages:
    default_target: True
    output: {outputs}
    threads: 1
    run:
        with open({str(root / "matlab-command.json")!r}, "w") as handle:
            json.dump(matlab_launcher.matlab_command(config, {param!r}, {entry!r}), handle, indent=1)
        run_matlab_scripts({param!r}, {entry!r})
'''


def _git(*args):
    try:
        return subprocess.run(["git", *args], cwd=REPO, check=True, capture_output=True).stdout
    except (OSError, subprocess.CalledProcessError):
        return None


def prepare(root):
    """Write the MATLAB inputs (YXZC, one-based XYZ centers), the entry config, the Snakefile and frozen.json."""
    from scipy.io import savemat

    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    for depth in DEPTHS:
        f = fixture(depth)
        savemat(root / f"z{depth}-input.mat", dict(
            first=f["first"].transpose(1, 2, 0, 3), second=f["second"].transpose(1, 2, 0, 3),
            extract_first=f["extract_first"].transpose(1, 2, 0, 3), reference=f["reference"].transpose(1, 2, 0),
            moving=f["moving"].transpose(1, 2, 0), centers_xyz=(f["centers"][:, ::-1] + 1).astype(np.float64)))
    write_json(root / "matlab-config.json", dict(starfinder_path=str(REPO), directory=str(root)))
    documents = root / "fixture/sample/output/documents"
    documents.mkdir(parents=True)
    (documents / "sample-annotation.csv").write_text("sample_id,fov_start,fov_end\nsample,1,1\n")
    (root / "Snakefile").write_text(snakefile_text(root))
    commit, patch = _git("rev-parse", "HEAD"), _git("diff", "--binary", "HEAD")
    write_json(root / "frozen.json", dict(
        fixture=FIXTURE_ID, seed=None, no_rng_reason="literal isolated peaks and an explicit integer shift",
        fixture_sha256=fixture_sha256(), protocol=dict(path=str(PROTOCOL), sha256=digest(REPO / PROTOCOL)),
        sources={p: digest(REPO / p) for p in SOURCES},
        inputs={p.name: digest(p) for p in sorted(root.iterdir()) if p.is_file()},
        commit=commit.decode().strip() if commit else None,
        patch_sha256=hashlib.sha256(patch).hexdigest() if patch is not None else None,
        snakemake_config=snakemake_config(root)))
    print(f"Prepared {root}; run the Snakefile with STARFINDER_MATLAB_EXECUTABLE set, then export-reference.")


def _matlab_strings(cells):
    """A MATLAB cell array of character vectors (scipy.io.loadmat) as a list of lists of str."""
    return [[str(c[0]) if c.size else "" for c in row] for row in cells]


def _scores(values):
    return [["Inf" if np.isposinf(v) else float(v) for v in row] for row in np.asarray(values, dtype=float)]


def export_reference(root, reference=REFERENCE):
    """Check the frozen hashes and the MATLAB runtime, then write provenance.json, z5.json and z1.json."""
    from scipy.io import loadmat

    root, reference = Path(root), Path(reference)
    frozen = json.loads((root / "frozen.json").read_text())
    missing = [name for name in RAW_OUTPUTS if not (root / name).is_file()]
    assert not missing, f"MATLAB outputs missing: {missing}"
    for name, sha in frozen["inputs"].items():
        assert digest(root / name) == sha, f"input changed after prepare: {name}"
    assert frozen["fixture_sha256"] == fixture_sha256(), "the fixture changed after prepare"
    for path in MATLAB_SOURCES:
        assert digest(REPO / path) == frozen["sources"][path], f"MATLAB source changed after prepare: {path}"
    runtime = json.loads((root / "matlab-runtime.json").read_text())
    command = json.loads((root / "matlab-command.json").read_text())
    toolboxes = runtime["toolboxes"] if isinstance(runtime["toolboxes"], list) else [runtime["toolboxes"]]
    names = [t["Name"] for t in toolboxes]
    assert runtime["release"] == "2023b" and runtime["threads"] == 1, runtime["release"]
    assert all(runtime["licenses"].values()), runtime["licenses"]
    assert all(name in names for name in TOOLBOXES), names
    assert command[1:3] == ["-singleCompThread", "-batch"] and f"{ENTRY}(" in command[-1], command
    reference.mkdir(parents=True, exist_ok=True)
    raw = {name: digest(root / name) for name in RAW_OUTPUTS}
    for depth in DEPTHS:
        m = loadmat(root / f"z{depth}-matlab.mat")
        detection = None
        if depth > 1:
            detection = dict(centroid_xyz=m["detection_xyz"].tolist(),
                             channel=m["detection_channel"].ravel().astype(int).tolist())
        registered = np.asarray(m["registered"], dtype=float)
        extraction, filtering = [], []
        for radius in RADII:
            extraction.append(dict(radius=radius, calls=_matlab_strings(m[f"calls_r{radius}"]),
                                   scores=_scores(m[f"scores_r{radius}"])))
            table = pd.read_csv(root / f"z{depth}-r{radius}-filtered.csv", dtype={"color_seq": str})
            filtering.append(dict(radius=radius, rows=[dict(x=float(r.x), y=float(r.y), z=float(r.z),
                                                            color_seq=r.color_seq, gene=r.gene)
                                                       for r in table.itertuples()]))
        write_json(reference / f"z{depth}.json", dict(
            schema=SCHEMA, fixture=FIXTURE_ID, fixture_sha256=frozen["fixture_sha256"], depth=depth,
            conventions="MATLAB: one-based (x, y, z); images (row, column, Z); extraction winners in the input "
                        "channel order [2 1 4 3]; Inf scores as the string Inf",
            detection=detection,
            detection_note=None if depth > 1 else "not run: SpotFindingMax3D is volumetric (regionprops3)",
            registration=dict(shifts=m["shifts"].ravel().tolist(), registered_shape=list(registered.shape),
                              registered=registered.tolist()),
            extraction=dict(channel_order=list(MAPPING.extraction_channels), radii=extraction),
            filtering=dict(end_base=["AC"], radii=filtering),
            encoding=dict(zip(("AAC", "ACC"), _matlab_strings(m["encoded"])[0])),
            raw_sha256={f"z{depth}-matlab.mat": raw[f"z{depth}-matlab.mat"]}))
    write_json(reference / "provenance.json", dict(
        schema=SCHEMA, fixture=FIXTURE_ID, fixture_sha256=frozen["fixture_sha256"], protocol=frozen["protocol"],
        matlab=dict(release=runtime["release"], version=runtime["version"], matlabroot=runtime["matlabroot"],
                    computer=runtime["computer"], threads=runtime["threads"], licenses=runtime["licenses"],
                    toolboxes=[dict(name=t["Name"], version=t["Version"], release=t["Release"]) for t in toolboxes],
                    date=runtime["date"]),
        launcher=dict(module="workflow/rules/matlab_launcher.py",
                      helper="run_matlab_scripts (workflow/rules/common.smk)",
                      matlab_launcher=frozen["snakemake_config"]["matlab_launcher"],
                      matlab_single_thread=frozen["snakemake_config"]["matlab_single_thread"], entry=ENTRY,
                      command=command),
        sources=frozen["sources"], commit=frozen["commit"], patch_sha256=frozen["patch_sha256"], raw_sha256=raw,
        regeneration=["python scripts/matlab_stage_comparison.py prepare <dir>",
                      "STARFINDER_MATLAB_EXECUTABLE=<matlab> snakemake --snakefile <dir>/Snakefile --cores 1 "
                      "--directory <dir>",
                      "python scripts/matlab_stage_comparison.py export-reference <dir>"]))
    print(f"Wrote {reference}/provenance.json, z5.json and z1.json.")


def load_reference(reference=REFERENCE):
    return {name: json.loads((Path(reference) / f"{name}.json").read_text()) for name in ("provenance", "z5", "z1")}


def python_stages(depth):
    """Run the four Python stages on the fixture of one depth."""
    f = fixture(depth)
    metadata = ImageMetadata("stage-comparison/reference")
    with warnings.catch_warnings():
        # The literal channels are almost all zero, so the noise record warns that its MAD is 0; the
        # adaptive threshold does not use it.
        warnings.simplefilter("ignore", SpotFindingWarning)
        detected = find_spots(f["first"], config=LocalMaximaConfig("adaptive", 0.1), metadata=metadata,
                              spot_namespace=f"python/z{depth}")
    registration = estimate_transform(f["reference"], f["moving"], config=TranslationConfig(),
                                      reference_metadata=metadata,
                                      moving_metadata=ImageMetadata("stage-comparison/moving"))
    registered = apply_transform(f["moving"], registration.transform, config=registration.application_config)
    table = pd.DataFrame(f["centers"].astype(np.float64), columns=["z", "y", "x"])
    table["spot_id"] = pd.Series(SPOT_IDS, dtype="string")
    spots = SpotFindingResult(table, metadata, f"supplied/z{depth}", LocalMaximaConfig(), {})
    rounds = {ROUNDS[0]: ImageLoadResult(f["extract_first"], metadata, CHANNELS, ()),
              ROUNDS[1]: ImageLoadResult(f["second"], metadata, CHANNELS, ())}
    codebook = Codebook(pd.DataFrame({"gene_id": list(CODEBOOK), "color_sequence": list(CODEBOOK.values())}),
                        ROUNDS, CHANNELS, COLOR_TO_CHANNEL)
    readout = {}
    for radius in RADII:
        extracted = extract_intensities(rounds, spots, config=NeighborhoodSumConfig((radius,) * 3))
        decoded = decode_reads(extracted, codebook, config=WtaDecoderConfig())
        readout[radius] = dict(values=extracted.values, decoded=decoded.table, accepted=filter_reads(decoded).accepted)
    return dict(depth=depth, centers=f["centers"], reference=f["reference"], detection=detected.spots,
                correction_zyx=tuple(-d + 0.0 for d in registration.transform.displacement_zyx),
                registered=registered, readout=readout,
                encoded={bases: bases_to_colors(bases) for bases in BASES.values()})


def match(actual, expected, stage):
    """Indices of the actual rows matched one-to-one to the expected points within DISTANCE, in expected order."""
    actual, expected = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    assert actual.shape == expected.shape, f"{stage}: {len(actual)} points, expected {len(expected)}"
    hits = np.linalg.norm(actual[:, None, :] - expected[None, :, :], axis=2) <= DISTANCE
    assert (hits.sum(axis=0) == 1).all() and (hits.sum(axis=1) == 1).all(), \
        f"{stage}: no one-to-one match within {DISTANCE} of {expected.tolist()}: {actual.tolist()}"
    return hits.argmax(axis=0)


def _points(rows, mapping):
    return np.asarray(rows, dtype=float).reshape(-1, 3)[:, list(mapping.point_axes)] - mapping.index_base


def _colors(calls, mapping):
    """MATLAB winners (one-based, in the reordered input) as a Python color sequence; M stays M."""
    channel_to_color = {c: s for s, c in COLOR_TO_CHANNEL.items()}
    return "".join(c if c == "M" else channel_to_color[mapping.extraction_channels[int(c) - 1] - 1] for c in calls)


def compare_depth(python, reference, mapping=MAPPING):
    """Apply checks D1-F2 of the design record to one depth; AssertionError names the failing stage."""
    depth, centers, literal = python["depth"], python["centers"][:2], python["reference"]
    assert reference["depth"] == depth and reference["fixture_sha256"] == fixture_sha256(), "fixture differs"
    row = dict(depth=depth)

    # D1, D2: detection.
    spots = python["detection"]
    order = match(spots[["z", "y", "x"]], centers, "detection (Python)")
    assert spots.channel.to_numpy()[order].tolist() == [1, 0], "detection (Python): channel"
    if depth > 1:
        found = reference["detection"]
        order = match(_points(found["centroid_xyz"], mapping), centers, "detection (MATLAB)")
        channels = np.asarray(found["channel"]) - mapping.detection_base
        assert channels[order].tolist() == [1, 0], f"detection (MATLAB): channel {channels.tolist()}"
        row["detection"] = "compared"
    else:
        assert reference["detection"] is None
        row["detection"] = "not compared: the MATLAB detector is volumetric (regionprops3)"

    # R1, R2: registration.
    registration = reference["registration"]
    shifts = list(registration["shifts"]) + [0.0] * (3 - len(registration["shifts"]))
    correction = tuple(float(shifts[i]) for i in mapping.shift_order)
    assert correction == CORRECTION_ZYX, f"registration (MATLAB): correction {correction}"
    assert python["correction_zyx"] == CORRECTION_ZYX, f"registration (Python): correction {python['correction_zyx']}"
    image = np.asarray(registration["registered"], dtype=float)
    image = (image[:, :, None] if image.ndim == 2 else image).transpose(mapping.image_axes)
    assert image.shape == literal.shape, f"registration (MATLAB): shape {image.shape}"
    for name, actual in (("MATLAB", image), ("Python", python["registered"])):
        np.testing.assert_allclose(actual, literal, rtol=0, atol=ATOL, err_msg=f"registration ({name})")
    np.testing.assert_allclose(image, python["registered"], rtol=0, atol=ATOL, err_msg="registration (MATLAB/Python)")
    row["registration"] = dict(correction_zyx=list(correction),
                               max_abs_error=float(np.abs(image - literal).max()))

    # E1-E3: extraction and color calling; F1: filtering.
    radii = {r["radius"]: r for r in reference["extraction"]["radii"]}
    filtered = {r["radius"]: r["rows"] for r in reference["filtering"]["radii"]}
    row["readout"] = []
    for radius, result in python["readout"].items():
        np.testing.assert_array_equal(result["values"], SUMS, err_msg="extraction (Python): sums")
        decoded = result["decoded"]
        assert decoded.call_status.tolist() == ["assigned", "assigned", "no_signal", "ambiguous"], \
            "extraction (Python): statuses"
        calls = [_colors(seq, mapping) for seq in radii[radius]["calls"]]
        assert calls[:2] == list(CODEBOOK.values()), f"extraction (MATLAB): calls {calls[:2]}"
        assert calls[2:] == ["MM", "MM"], f"extraction (MATLAB): zero/tie calls {calls[2:]}"
        assert decoded.observed_color_sequence.tolist()[:2] == calls[:2], "extraction: Python and MATLAB calls"
        scores = radii[radius]["scores"]
        assert all(s == "Inf" for r in scores[2:] for s in r), "extraction (MATLAB): zero/tie scores"
        matlab = np.asarray(scores[:2], dtype=float)
        np.testing.assert_allclose(matlab, ROUND_SCORES, rtol=0, atol=ATOL, err_msg="extraction (MATLAB): scores")
        summed = matlab.sum(axis=1)
        for name, actual in (("MATLAB", summed), ("Python", decoded.wta_l2_nll.to_numpy()[:2])):
            np.testing.assert_allclose(actual, ROUND_SCORES.sum(axis=1), rtol=0, atol=ATOL,
                                       err_msg=f"extraction ({name}): summed score")
        np.testing.assert_allclose(decoded.wta_l2_nll.to_numpy()[:2], summed, rtol=0, atol=ATOL,
                                   err_msg="extraction (MATLAB/Python): summed score")
        accepted = result["accepted"]
        assert dict(zip(accepted.spot_id, accepted.gene_id)) == {"A": "gene-A", "B": "gene-B"}, \
            "filtering (Python): accepted reads"
        rows = filtered[radius]
        order = match(_points([[r["x"], r["y"], r["z"]] for r in rows], mapping), centers, "filtering (MATLAB)")
        genes = [rows[i]["gene"] for i in order]
        assert genes == list(CODEBOOK), f"filtering (MATLAB): genes {genes}"
        row["readout"].append(dict(radius=radius, calls=calls, matlab_summed_score=summed.tolist(),
                                   python_summed_score=decoded.wta_l2_nll.to_numpy()[:2].tolist(), genes=genes))

    # F2: encoding.
    assert reference["encoding"] == {"AAC": "12", "ACC": "21"} == python["encoded"], "encoding"
    return row


def compare(root, reference=REFERENCE):
    """Compare the Python stages with the reference files and write <root>/comparison.json."""
    root, loaded = Path(root), load_reference(reference)
    assert loaded["provenance"]["fixture_sha256"] == fixture_sha256(), "the references belong to another fixture"
    rows = [compare_depth(python_stages(depth), loaded[f"z{depth}"]) for depth in DEPTHS]
    write_json(root / "comparison.json", dict(
        fixture=FIXTURE_ID, fixture_sha256=fixture_sha256(), protocol=loaded["provenance"]["protocol"],
        reference={p.name: digest(p) for p in sorted(Path(reference).glob("*.json"))}, status="passed",
        conditions=rows,
        limitations=["No general algorithm equivalence is claimed; Z=1 detection is not compared.",
                     "MATLAB exposes calls and scores only, not the raw neighbourhood sums.",
                     "MATLAB filtering receives A and B only; Python keeps the zero and tie reads as rejected."]))
    print(f"All checks passed; wrote {root}/comparison.json.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("command", choices=("prepare", "export-reference", "compare"))
    parser.add_argument("directory", type=Path, help="the external run directory")
    parser.add_argument("--reference", type=Path, default=REFERENCE, help="the reference folder")
    args = parser.parse_args(argv)
    directory = args.directory.resolve()
    if args.command == "prepare":
        prepare(directory)
    elif args.command == "export-reference":
        export_reference(directory, args.reference)
    else:
        compare(directory, args.reference)


if __name__ == "__main__":
    main()
