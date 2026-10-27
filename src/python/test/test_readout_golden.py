"""Golden baseline for the current extraction, decoding and read filtering (W-279, §2.8).

Pins, as explicit expected values, on one small seeded fixture (12×48×48 voxels, four
uint16 channels, four rounds) and ten hand-placed candidates:

(a) the extracted neighborhood-sum tensor and its ``valid`` mask;
(b) the decoding table of both decoders (WTA and codebook-aware, at their defaults);
(c) the filtering table of each decoding table with the default filter, with one score
    bound and with the end-base check;
(d) the ``pre_qc`` table after ``FOV.run``, written as a CSV checkpoint and reloaded,
    together with the written ``candidates.csv`` and ``pre_qc.csv``, each read back with
    the dtypes of its header;
(e) the same results when the pipeline and the codebook come from the legacy YAML keys
    (``reads_extraction``, ``reads_filtration``, ``load_codebook``) through
    ``from_workflow_config``, for the cases those keys can express (WTA only).

The expected values are the files under ``data/expected/test_readout_golden/`` (its
README names the revision and the command that wrote them; W-573 replaced the SHA-256
digests pinned before, which the expected values reproduce exactly). They compare as
docs/contributing.md ("Expected values and tolerances") states: the computed float64
columns of ``COMPUTED_COLUMNS`` within ``FLOAT64``, every other column (strings,
statuses, booleans, integers and the float columns that hold integers: coordinates,
neighborhood sums of uint16 pixels, ``hamming_to_wta``, ``qc_rounds``) and the
extracted tensor exactly. The fixture inputs and the codebook stay pinned by digest
(integer and discrete data).

The fixture holds a round whose two brightest channels tie exactly, a round with zero
signal in every channel, a read one color substitution away from a codeword, a read two
substitutions from every codeword, and a spot next to the y=0 face whose extraction box
is clipped. The codebook is a gene,barcode CSV of 5-base barcodes with the two-base
encoding and reversed bases, one segment. The first digests were produced with the
locked project environment (NumPy 2.2.6, pandas 3.0.0) and are bit-identical over
repeated single-thread runs; see docs/readout-baseline.md.

Every extraction, decoding, filtering, encoding, detection, pipeline and checkpoint
configuration is built by ``readout_config``, which holds the only imports of those
config types. The
§2.8 work may change that helper's body only; every pinned value stays, except where
docs/readout-algorithms.md ("Engineering validation design (task group 6)") names an edit.
"""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from starfinder.readout import decode_reads, extract_intensities, filter_reads, load_codebook
from starfinder.dataset import Dataset, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult

from .coordination_helpers import digest, table_digest
from .expected_values import FLOAT64, read_arrays, read_csv, read_table, table_differences

pytestmark = [pytest.mark.readout, pytest.mark.golden]

SHAPE_ZYX = (12, 48, 48)
CHANNELS = ("ch00", "ch01", "ch02", "ch03")
ROUNDS = ("round1", "round2", "round3", "round4")
SEED = 20261002
NAMESPACE = json.dumps(["golden", "sample", "FOV_001", None])
METADATA = ImageMetadata("golden/FOV_001")
AMPLITUDE = 800.0
# A region at least this large around a planted feature is overwritten, so every
# extraction radius up to (3, 4, 4) sees the same tie or the same zeros.
FEATURE_RADIUS_ZYX = (3, 4, 4)

# Codebook color sequences (round order, colors 1-4) and gene IDs; the CSV stores the
# 5-base barcodes that encode to them with reversed bases.
CODEWORDS = {"1234": "GeneA", "2143": "GeneB", "3412": "GeneC", "4321": "GeneD",
             "2222": "GeneE", "3131": "GeneF", "4242": "GeneG", "1313": "GeneH"}
# spot_id -> (z, y, x, planted color sequence, note). The planted sequence gives the
# brightest channel of each round; the notes name the feature.
SPOTS = {
    "0": (6.0, 12.0, 12.0, "1234", "clean"),
    "1": (6.0, 12.0, 36.0, "2143", "clean"),
    "2": (5.0, 24.0, 24.0, "3412", "clean"),
    "3": (6.0, 1.0, 24.0, "4321", "near the y=0 face: box clipped to y 0-3"),
    "4": (6.0, 36.0, 12.0, "2222", "round2: ch01 and ch02 tie exactly"),
    "5": (6.0, 36.0, 36.0, "3131", "round4: zero in every channel"),
    "6": (4.0, 24.0, 8.0, "1234", "round3: ch03 slightly above ch02 (observed 1244)"),
    "7": (8.0, 24.0, 40.0, "2424", "two substitutions from every codeword"),
    "8": (4.0, 40.0, 24.0, "4242", "clean"),
    "9": (8.0, 8.0, 24.0, "1313", "clean"),
}
TIED_SPOT, TIED_ROUND = "4", 1
ZERO_SPOT, ZERO_ROUND = "5", 3
SUBSTITUTION_SPOT, SUBSTITUTION_ROUND = "6", 2
# The one score bound of (c), an upper bound on the decoder's score, and the changed bound
# of the negative control. Each accepts a different nonempty subset of the assigned reads.
WTA_BOUND, WTA_CHANGED_BOUND = 0.069, 0.06  # wta_l2_nll
CODEBOOK_AWARE_BOUND, CODEBOOK_AWARE_CHANGED_BOUND = 1.105, 1.05  # probability_nll


def readout_config(kind, *, radius=(1, 2, 2), decoder="wta", score_bound=None, end_bases=None,
                   directory=None, **decoder_options):
    """The only place that builds every configuration of this test.

    kind "extraction": NeighborhoodSumConfig(radius); "decoding": the decoder config
    ("wta" or "codebook_aware", with decoder_options for the codebook-aware gates);
    "filtering": ReadFilterConfig with an optional upper score bound on the decoder's
    score (wta_l2_nll or probability_nll) and optional end_bases (start base C);
    "encoding": the codebook EncodingConfig (two-base, reversed, one segment);
    "detection": the LocalMaximaConfig recorded on the hand-built candidates;
    "scoring": the shared read-QC score ReadScoreConfig (§2.8);
    "pipeline": a PipelineConfig of extraction, decoding, scoring and filtering, as the
    workflow adapter scores whenever it decodes;
    "checkpoints": the CheckpointConfig of the CSV candidates and pre_qc stages under
    directory;
    "yaml": the WorkflowConfig that from_workflow_config translates from the legacy
    keys (rule rsf_single_fov, raw loading and spot finding off). The legacy keys
    always select WtaDecoderConfig(diagnostics=True), so "yaml" requires decoder "wta".
    """
    from starfinder._registry import config_type_for
    from starfinder.readout import DECODING_METHODS, ENCODINGS, NeighborhoodSumConfig, ReadFilterConfig, ReadScoreConfig
    from starfinder.dataset import CheckpointConfig, PipelineConfig
    from starfinder.spot_finding import LocalMaximaConfig

    score = {"wta": "wta_l2_nll", "codebook_aware": "probability_nll"}[decoder]
    bounds = {} if score_bound is None else {score: (None, score_bound)}
    if kind == "yaml":
        if decoder != "wta" or decoder_options:
            raise ValueError("the legacy keys select WtaDecoderConfig(diagnostics=True) only")
        filtration = {"run": True, "score_bounds": {k: list(v) for k, v in bounds.items()}}
        if end_bases is not None:
            filtration["end_base"] = end_bases
        workflow = {"n_rounds": len(ROUNDS), "ref_round": "round1", "dataset_id": "golden",
                    "sample_id": "sample", "output_id": "out",
                    "root_input_path": "unused-input", "root_output_path": "unused-output",
                    "seq_channel_order": list(CHANNELS),
                    "rules": {"rsf_single_fov": {"parameters": {
                        "load_raw_images": {"run": False},
                        "load_codebook": {"run": True},
                        "reads_extraction": {"run": True, "voxel_size": list(radius)},
                        "reads_filtration": filtration}}}}
        return from_workflow_config(workflow, "rsf_single_fov")
    if kind == "extraction":
        return NeighborhoodSumConfig(tuple(radius))
    if kind == "decoding":
        return config_type_for(DECODING_METHODS, decoder, "decoding method")(**decoder_options)
    if kind == "filtering":
        return ReadFilterConfig(score_bounds=bounds, end_bases=end_bases)
    if kind == "encoding":
        return config_type_for(ENCODINGS, "two_base", "encoding")(reverse_bases=True, split_index=None)
    if kind == "detection":
        return LocalMaximaConfig()
    if kind == "scoring":
        return ReadScoreConfig()
    if kind == "pipeline":
        return PipelineConfig(extraction=readout_config("extraction", radius=radius),
                              decoding=readout_config("decoding", decoder=decoder, **decoder_options),
                              filtering=readout_config("filtering", decoder=decoder, score_bound=score_bound,
                                                       end_bases=end_bases),
                              scoring=readout_config("scoring"))
    if kind == "checkpoints":
        return CheckpointConfig(stages=("candidates", "pre_qc"), directory=directory)
    raise ValueError(f"unknown kind {kind!r}")


def _gaussian(center):
    z, y, x = np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE_ZYX), indexing="ij")
    cz, cy, cx = center
    return np.exp(-((z - cz) / 1.0) ** 2 / 2 - ((y - cy) / 1.2) ** 2 / 2 - ((x - cx) / 1.2) ** 2 / 2)


def _region(center):
    return tuple(slice(max(int(c) - r, 0), int(c) + r + 1) for c, r in zip(center, FEATURE_RADIUS_ZYX))


def fixture_rounds():
    """uint16 ZYXC image per round; every value comes from SEED.

    Each round has a background of 20 grey levels with noise sd 3 in every channel.
    Each spot adds a Gaussian of amplitude AMPLITUDE (sd 1.0 in Z, 1.2 in Y and X) to
    the channel of its planted color in that round, and 5 % of it to the next channel
    (crosstalk). Spot 6 puts 0.85 × AMPLITUDE in ch02 and AMPLITUDE in ch03 in round 3.
    Then spot 4's ch02 region in round 2 is overwritten with its ch01 region (exact
    tie), and spot 5's region in round 4 is set to 0 in every channel.
    """
    rng = np.random.RandomState(SEED)
    rounds = {}
    for r, label in enumerate(ROUNDS):
        image = 20.0 + rng.normal(0.0, 3.0, SHAPE_ZYX + (len(CHANNELS),))
        for spot_id, (z, y, x, colors, _) in SPOTS.items():
            blob = _gaussian((z, y, x))
            channel = int(colors[r]) - 1
            if spot_id == SUBSTITUTION_SPOT and r == SUBSTITUTION_ROUND:
                image[..., 2] += 0.85 * AMPLITUDE * blob
                image[..., 3] += AMPLITUDE * blob
                continue
            image[..., channel] += AMPLITUDE * blob
            image[..., (channel + 1) % len(CHANNELS)] += 0.05 * AMPLITUDE * blob
        image = np.clip(np.rint(image), 0, 65535).astype(np.uint16)
        if r == TIED_ROUND:
            region = _region(SPOTS[TIED_SPOT][:3])
            image[region + (2,)] = image[region + (1,)]
        if r == ZERO_ROUND:
            image[_region(SPOTS[ZERO_SPOT][:3])] = 0
        rounds[label] = image
    return rounds


def candidates():
    """The hand-placed candidates as a SpotFindingResult (no detection runs in this test)."""
    spots = pd.DataFrame({"spot_id": pd.array(list(SPOTS), dtype="string"),
                          "z": [v[0] for v in SPOTS.values()], "y": [v[1] for v in SPOTS.values()],
                          "x": [v[2] for v in SPOTS.values()]})
    return SpotFindingResult(spots, METADATA, NAMESPACE, readout_config("detection"),
                             {"channel_labels": list(CHANNELS)})


def write_codebook(directory):
    """genes.csv-style gene,barcode file of the 5-base barcodes of CODEWORDS (start base C)."""
    from starfinder.readout import colors_to_bases
    path = directory / "codebook.csv"
    rows = [f"{gene},{colors_to_bases(colors, 'C')[::-1]}" for colors, gene in CODEWORDS.items()]
    path.write_text("\n".join(rows) + "\n")
    return path


def golden_codebook(directory):
    return load_codebook(write_codebook(directory), round_labels=ROUNDS, channel_labels=CHANNELS,
                         encoding=readout_config("encoding"))


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_entry(table):
    """The table without the entry_id column that §2.8 adds (docs/readout-contract.md, option E1)."""
    return table.drop(columns="entry_id")


# §2.8 (option C1): the score columns that scoring adds to pre_qc and filtering.
SCORE_COLUMNS = ["qc_score", "qc_ambiguity_max", "qc_signal_to_background", "qc_rounds", "qc_reason"]


def without_score(table):
    return table.drop(columns=SCORE_COLUMNS)


def extract(radius=(1, 2, 2)):
    loaded = {label: ImageLoadResult(image, METADATA, CHANNELS, (), {})
              for label, image in fixture_rounds().items()}
    return extract_intensities(loaded, candidates(), config=readout_config("extraction", radius=radius))


def decode(directory, decoder, **options):
    return decode_reads(extract(), golden_codebook(directory),
                           config=readout_config("decoding", decoder=decoder, **options))


FILTERS = {"default": {}, "bound": {"bounded": True}, "end_bases": {"end_bases": "CC"}}


def filter_options(decoder, name, bound=None):
    options = dict(FILTERS[name])
    if options.pop("bounded", False):
        options["score_bound"] = bound if bound is not None else (
            WTA_BOUND if decoder == "wta" else CODEBOOK_AWARE_BOUND)
    return options


def golden_dataset(root):
    return Dataset(root, root / "out", "golden", "sample", "out",
                   rounds=RoundState(sequencing_rounds=list(ROUNDS), reference_round="round1"),
                   channel_order=list(CHANNELS))


def run_and_reload(root, dataset, pipeline, execution=None):
    """FOV.run on resident rounds and the hand-built candidates, with CSV candidates and
    pre_qc checkpoints; the extracted tensor, the reloaded pre_qc table, the filtering table
    and the written candidates.csv and pre_qc.csv read back with the dtypes of their headers."""
    checkpoints = readout_config("checkpoints", directory=root / "checkpoints")
    fov = dataset.fov("FOV_001")
    fov.images = fixture_rounds()
    fov.metadata = {label: METADATA for label in ROUNDS}
    fov.spot_result = candidates()
    if execution is None:
        fov.run(pipeline, checkpoints=checkpoints)
    else:
        fov.run(pipeline, execution=execution, checkpoints=checkpoints)
    reloaded = dataset.fov("FOV_001").load_checkpoint("pre_qc", checkpoints=checkpoints)
    pd.testing.assert_frame_equal(reloaded.decoding_result.table, fov.decoding_result.table)
    assert reloaded.decoding_result.config == fov.decoding_result.config
    pd.testing.assert_frame_equal(reloaded.scoring_result.table, fov.scoring_result.table)
    directory = root / "checkpoints" / "FOV_001" / "checkpoints"

    def written(stage):
        return read_csv(directory / f"{stage}.csv", json.loads((directory / f"{stage}.json").read_text())["dtypes"])

    return {"values": fov.intensity_result.values, "valid": fov.intensity_result.valid,
            "pre_qc": reloaded.scoring_result.table, "filtering": fov.filtering_result.table,
            "candidates.csv": written("candidates"), "pre_qc.csv": written("pre_qc")}


# The explicit expected values (data/expected/test_readout_golden/README.md).
EXPECTED = Path(__file__).parent / "data" / "expected" / "test_readout_golden"
# The computed float64 columns: decoder and read-QC scores, and the background and noise estimates
# (bg_*, noise_*) of candidates.csv. They compare within FLOAT64 (docs/contributing.md); every other
# column compares exactly.
COMPUTED_COLUMNS = ("wta_l2_nll", "probability_nll", "score_delta", "geomean_probability", "min_round_margin",
                    "corrected_round_margin", "mean_total_intensity", "qc_score", "qc_ambiguity_max",
                    "qc_signal_to_background")
COMPUTED_PREFIXES = ("bg_", "noise_")


def differences(actual, expected):
    """table_differences with the comparison of each column above."""
    close = [c for c in expected if c in COMPUTED_COLUMNS or str(c).startswith(COMPUTED_PREFIXES)]
    return table_differences(actual, expected, close=close, tolerance=FLOAT64)


def assert_matches(actual, expected, name=""):
    found = differences(actual, expected)
    assert not found, f"{name}: " + "; ".join(found)


def expected_extraction(radius):
    """(values, valid) of the extracted tensor: sums of uint16 pixels and a mask, compared exactly."""
    arrays = read_arrays(EXPECTED / "extraction.npz")
    key = "_".join(map(str, radius))
    return arrays[f"values_{key}"], arrays[f"valid_{key}"]


def expected_pre_qc(decoder):
    """The scored reads of FOV.run (pre_qc); without the score columns, the decoding table."""
    return read_table(EXPECTED / f"pre_qc-{decoder}.csv")


def expected_decoding(decoder):
    return without_score(expected_pre_qc(decoder))


def expected_filtering(decoder, name):
    """The filtering table of the scored reads of FOV.run; without the score columns, that of the
    decoding table."""
    return read_table(EXPECTED / f"filtering-{decoder}-{name}.csv")


def expected_candidates():
    """candidates.csv as FOV.run writes it, with the background columns (the same for both decoders)."""
    return read_table(EXPECTED / "candidates.csv")


def assert_extraction_matches(values, valid, radius=(1, 2, 2)):
    expected_values, expected_valid = expected_extraction(radius)
    np.testing.assert_array_equal(values, expected_values, strict=True)
    np.testing.assert_array_equal(valid, expected_valid, strict=True)


def assert_run_matches(outputs, decoder, name):
    assert_extraction_matches(outputs["values"], outputs["valid"])
    assert_matches(outputs["pre_qc"], expected_pre_qc(decoder), "pre_qc")
    assert_matches(outputs["filtering"], expected_filtering(decoder, name), "filtering")
    assert_matches(outputs["candidates.csv"], expected_candidates(), "candidates.csv")
    assert_matches(outputs["pre_qc.csv"], expected_pre_qc(decoder), "pre_qc.csv")


PINNED_INPUT = {
    "round1": "a639da573a740379e5ffc89bd8c3ff3837402f664116a7e0ee141a35f0ff6e06",
    "round2": "abe3b5e1e8dd43528e8f25cd39f008993620944b897363d38c7a8bd86301198c",
    "round3": "e5070fa8b009cd1dbe570ad86d75a7e43b3da43124a21d348f0c909127296fee",
    "round4": "5c2c313146fe7970c367f2ea273cabc75e41eb276bef6be0b4569d0ccdf95175",
}
PINNED_CODEBOOK = "8517168607ca65dfa61db5d4a764de68da588534f0f68573d932540d981aded4"
# §2.8 (option E1): the codebook gains entry_id. PINNED_CODEBOOK is the digest of the table
# without it; this pins the extended table.
PINNED_CODEBOOK_ENTRIES = "d18e0560cae1aeac5d3aa20912a9f632ad46cc262798fcf9bfe3b0621cfb48a8"
# decoder -> spot_id -> (call_status, failure_reason)
PINNED_STATUS = {
    "wta": {
        "0": ("assigned", ""),
        "1": ("assigned", ""),
        "2": ("assigned", ""),
        "3": ("assigned", ""),
        "4": ("ambiguous", "tied_channels"),
        "5": ("no_signal", "zero_signal_round"),
        "6": ("unmatched", "not_in_codebook"),
        "7": ("unmatched", "not_in_codebook"),
        "8": ("assigned", ""),
        "9": ("assigned", ""),
    },
    "codebook_aware": {
        "0": ("assigned", ""),
        "1": ("assigned", ""),
        "2": ("assigned", ""),
        "3": ("assigned", ""),
        "4": ("assigned", ""),
        "5": ("no_signal", "zero_signal_round"),
        "6": ("assigned", ""),
        "7": ("unmatched", "no_candidate"),
        "8": ("assigned", ""),
        "9": ("assigned", ""),
    },
}
# (decoder, filter) -> accepted count
ACCEPTED = {("wta", "default"): 6, ("wta", "bound"): 3, ("wta", "end_bases"): 6,
            ("codebook_aware", "default"): 8, ("codebook_aware", "bound"): 5,
            ("codebook_aware", "end_bases"): 8}


def test_fixture_inputs_are_pinned():
    assert {label: digest(image) for label, image in fixture_rounds().items()} == PINNED_INPUT


def test_codebook_from_the_barcode_csv_is_pinned(tmp_path):
    book = golden_codebook(tmp_path)
    assert dict(zip(book.table.color_sequence, book.table.gene_id)) == CODEWORDS
    assert table_digest(without_entry(book.table)) == PINNED_CODEBOOK
    assert table_digest(book.table) == PINNED_CODEBOOK_ENTRIES


def test_fixture_has_the_listed_features():
    values = extract().values
    index = {spot_id: i for i, spot_id in enumerate(SPOTS)}
    tied = values[index[TIED_SPOT], :, TIED_ROUND]
    assert tied[1] == tied[2] == tied.max()
    assert values[index[ZERO_SPOT], :, ZERO_ROUND].sum() == 0
    near = values[index["6"], :, SUBSTITUTION_ROUND]
    assert near.argmax() == 3 and near[2] > 0.7 * near[3]


@pytest.mark.parametrize("radius", [(1, 2, 2), (0, 1, 1)])
def test_extracted_tensor_is_pinned(radius):
    result = extract(radius)
    assert result.values.shape == (len(SPOTS), len(CHANNELS), len(ROUNDS))
    assert_extraction_matches(result.values, result.valid, radius)


def test_valid_is_always_true_at_the_border_and_in_a_zero_round():
    # Legacy behavior that §2.8 changes (docs/readout-baseline.md): extraction never
    # marks a measurement unavailable, even where the box is clipped by a face.
    assert extract().valid.all()


@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_decoding_tables_are_pinned(tmp_path, decoder):
    table = decode(tmp_path, decoder).table
    assert dict(zip(table.spot_id, zip(table.call_status, table.failure_reason))) == PINNED_STATUS[decoder]
    assert_matches(table, expected_decoding(decoder), decoder)


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_filtering_tables_are_pinned(tmp_path, decoder, name):
    filtered = filter_reads(decode(tmp_path, decoder),
                            config=readout_config("filtering", decoder=decoder, **filter_options(decoder, name)))
    assert filtered.counts["accepted"] == ACCEPTED[(decoder, name)]
    assert_matches(filtered.table, without_score(expected_filtering(decoder, name)), f"{decoder} {name}")


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_pre_qc_after_run_save_and_reload_is_pinned(tmp_path, decoder, name):
    dataset = golden_dataset(tmp_path)
    dataset.codebook = golden_codebook(tmp_path)
    outputs = run_and_reload(tmp_path, dataset,
                             readout_config("pipeline", decoder=decoder, **filter_options(decoder, name)))
    assert_run_matches(outputs, decoder, name)


@pytest.mark.parametrize("name", list(FILTERS))
def test_legacy_yaml_keys_give_the_same_digests(tmp_path, name):
    adapted = readout_config("yaml", **filter_options("wta", name))
    dataset = adapted.dataset
    dataset.load_codebook(write_codebook(tmp_path), split_index=adapted.split_index)
    assert adapted.split_index is None
    outputs = run_and_reload(tmp_path, dataset, adapted.pipeline, adapted.execution)
    assert_run_matches(outputs, "wta", name)


# The negative controls compare the changed output with the expected values and assert that it lies
# outside the comparison of each column (docs/contributing.md).
def test_a_changed_neighborhood_radius_changes_the_tensor_digest():
    changed = extract((1, 3, 3)).values
    # Sums of uint16 pixels compare exactly.
    assert not np.array_equal(changed, expected_extraction((1, 2, 2))[0])


@pytest.mark.parametrize("gate", [{"allow_rescue": False}, {"max_corrected_round_margin": 0.01},
                                  {"min_geomean_probability": 0.95}], ids=lambda g: next(iter(g)))
def test_a_changed_decoder_gate_changes_the_decoding_digest(tmp_path, gate):
    changed = decode(tmp_path, "codebook_aware", **gate).table
    assert differences(without_entry(changed), without_entry(expected_decoding("codebook_aware")))


@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_a_changed_score_bound_changes_the_filtering_digest(tmp_path, decoder):
    bound = {"wta": WTA_CHANGED_BOUND, "codebook_aware": CODEBOOK_AWARE_CHANGED_BOUND}[decoder]
    filtered = filter_reads(decode(tmp_path, decoder), config=readout_config(
        "filtering", decoder=decoder, **filter_options(decoder, "bound", bound)))
    expected = without_score(expected_filtering(decoder, "bound"))
    assert differences(without_entry(filtered.table), without_entry(expected))


def test_candidates_with_a_round_column_raise_on_decoding(tmp_path):
    # Legacy behavior that §2.8 changes (docs/readout-contract.md, "Direct readout"):
    # a multi-round candidate set cannot be decoded until a readout mode exists.
    dataset = golden_dataset(tmp_path)
    dataset.codebook = golden_codebook(tmp_path)
    fov = dataset.fov("FOV_001")
    spots = candidates().spots.assign(round=pd.array(["round1"] * len(SPOTS), dtype="string"))
    fov.spot_result = SpotFindingResult(spots, METADATA, NAMESPACE, readout_config("detection"), {})
    fov.intensity_result = extract()
    with pytest.raises(ValueError, match="needs a readout mode"):
        fov.decode_reads(config=readout_config("decoding"))
