"""Golden baseline for the current extraction, decoding and read filtering (W-279, §2.8).

Pins, with exact SHA-256 digests, on one small seeded fixture (12×48×48 voxels, four
uint16 channels, four rounds) and ten hand-placed candidates:

(a) the extracted neighborhood-sum tensor and its ``valid`` mask;
(b) the decoding table of both decoders (WTA and codebook-aware, at their defaults);
(c) the filtering table of each decoding table with the default filter, with one score
    bound and with the end-base check;
(d) the ``pre_qc`` table after ``FOV.run``, written as a CSV checkpoint and reloaded,
    together with the written ``candidates.csv`` and ``pre_qc.csv``;
(e) the same digests when the pipeline and the codebook come from the legacy YAML keys
    (``reads_extraction``, ``reads_filtration``, ``load_codebook``) through
    ``from_workflow_config``, for the cases those keys can express (WTA only).

The fixture holds a round whose two brightest channels tie exactly, a round with zero
signal in every channel, a read one color substitution away from a codeword, a read two
substitutions from every codeword, and a spot next to the y=0 face whose extraction box
is clipped. The codebook is a gene,barcode CSV of 5-base barcodes with the two-base
encoding and reversed bases, one segment. The digests were produced with the locked
project environment (NumPy 2.2.6, pandas 3.0.0) and are bit-identical over repeated
single-thread runs; see docs/readout-baseline.md.

Every extraction, decoding, filtering, encoding, detection, pipeline and checkpoint
configuration is built by ``readout_config``, which holds the only imports of those
config types. The
§2.8 work may change that helper's body only; every pinned digest stays, except where
docs/readout-contract.md names an edit.
"""
import hashlib
import json

import numpy as np
import pandas as pd
import pytest

from starfinder.barcode import decode_barcodes, extract_intensities, filter_reads, load_codebook
from starfinder.dataset import Dataset, RoundState
from starfinder.dataset.workflow import from_workflow_config
from starfinder.image import ImageMetadata
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import SpotFindingResult

pytestmark = [pytest.mark.barcode, pytest.mark.golden]

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
# of the digest-change test. Each accepts a different nonempty subset of the assigned reads.
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
    from starfinder.barcode import DECODING_METHODS, ENCODINGS, NeighborhoodSumConfig, ReadFilterConfig, ReadScoreConfig
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
    from starfinder.barcode import decode_color_sequence
    path = directory / "codebook.csv"
    rows = [f"{gene},{decode_color_sequence(colors, 'C')[::-1]}" for colors, gene in CODEWORDS.items()]
    path.write_text("\n".join(rows) + "\n")
    return path


def golden_codebook(directory):
    return load_codebook(write_codebook(directory), round_labels=ROUNDS, channel_labels=CHANNELS,
                         encoding=readout_config("encoding"))


def digest(array):
    """SHA-256 over dtype, shape and C-order bytes."""
    array = np.ascontiguousarray(array)
    h = hashlib.sha256(f"{array.dtype.str}|{array.shape}|".encode())
    h.update(array.tobytes())
    return h.hexdigest()


def table_digest(table):
    """SHA-256 over the column names and dtypes and the CSV text (floats in %.17g)."""
    header = json.dumps([[str(c), str(t)] for c, t in table.dtypes.items()])
    text = table.to_csv(index=False, float_format="%.17g", na_rep="<NA>")
    return hashlib.sha256((header + "\n" + text).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def without_entry(table):
    """The table without the entry_id column that §2.8 adds (docs/readout-contract.md, option E1)."""
    return table.drop(columns="entry_id")


# §2.8 (option C1): the score columns that scoring adds to pre_qc and filtering, and the
# background columns that extraction adds to candidates.csv.
SCORE_COLUMNS = ["qc_score", "qc_ambiguity_max", "qc_signal_to_background", "qc_rounds", "qc_reason"]


def without_score(table):
    return table.drop(columns=SCORE_COLUMNS)


def without_background(table):
    prefixes = ("bg_", "noise_", "bgvox_", "boxvox_")
    return table.drop(columns=[c for c in table.columns if c.startswith(prefixes)])


def extract(radius=(1, 2, 2)):
    loaded = {label: ImageLoadResult(image, METADATA, CHANNELS, (), {})
              for label, image in fixture_rounds().items()}
    return extract_intensities(loaded, candidates(), config=readout_config("extraction", radius=radius))


def decode(directory, decoder, **options):
    return decode_barcodes(extract(), golden_codebook(directory),
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
    pre_qc checkpoints; digests of the written files and of the reloaded pre_qc table."""
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
    reads = reloaded.scoring_result.table
    directory = root / "checkpoints" / "FOV_001"
    # The pre_qc table without entry_id (and without the score columns), the pre_qc table without
    # the score columns and the candidates table without the background columns, written by the
    # same writer.
    from starfinder.io._checkpoint import _read_table, _write_table
    for name, table in (("without_entry", without_entry(without_score(reads))), ("without_score", without_score(reads))):
        (root / name).mkdir()
        _write_table(table, root / name, "pre_qc", "csv")
    written = _read_table(directory / "candidates.csv", json.loads((directory / "candidates.json").read_text())["dtypes"])
    (root / "without_background").mkdir()
    _write_table(without_background(written), root / "without_background", "candidates", "csv")
    return {"values": digest(fov.intensity_result.values), "valid": digest(fov.intensity_result.valid),
            "pre_qc": table_digest(without_entry(without_score(reads))),
            "pre_qc_entries": table_digest(without_score(reads)),
            "pre_qc_scored": table_digest(reads),
            "filtering": table_digest(without_entry(without_score(fov.filtering_result.table))),
            "filtering_entries": table_digest(without_score(fov.filtering_result.table)),
            "filtering_scored": table_digest(fov.filtering_result.table),
            "candidates_csv": file_digest(root / "without_background" / "candidates.csv"),
            "candidates_csv_background": file_digest(directory / "candidates.csv"),
            "pre_qc_csv": file_digest(root / "without_entry" / "pre_qc.csv"),
            "pre_qc_csv_entries": file_digest(root / "without_score" / "pre_qc.csv"),
            "pre_qc_csv_scored": file_digest(directory / "pre_qc.csv")}


PINNED_INPUT = {
    "round1": "a639da573a740379e5ffc89bd8c3ff3837402f664116a7e0ee141a35f0ff6e06",
    "round2": "abe3b5e1e8dd43528e8f25cd39f008993620944b897363d38c7a8bd86301198c",
    "round3": "e5070fa8b009cd1dbe570ad86d75a7e43b3da43124a21d348f0c909127296fee",
    "round4": "5c2c313146fe7970c367f2ea273cabc75e41eb276bef6be0b4569d0ccdf95175",
}
PINNED_CODEBOOK = "8517168607ca65dfa61db5d4a764de68da588534f0f68573d932540d981aded4"
# §2.8 (option E1): the codebook, decoding, filtering and pre_qc tables gain entry_id. The
# PINNED_* digests above and below are of the tables without it; these pin the extended tables.
PINNED_CODEBOOK_ENTRIES = "d18e0560cae1aeac5d3aa20912a9f632ad46cc262798fcf9bfe3b0621cfb48a8"
# radius -> (digest of values, digest of valid)
PINNED_EXTRACTION = {
    (1, 2, 2): (
        "4971a273134cf4c7edcb7d8a7eefbf3732fcd645ac2080a03df65b671f43cd2e",
        "854cd4794fb5dff12ea72c2954c67babb6910fd393c1dfefe629b90fa53b996a"),
    (0, 1, 1): (
        "4b4b2732e29941acbeac328d7d8d6d60fb2ccca451d7ddf458c2bf3e1e2710f2",
        "854cd4794fb5dff12ea72c2954c67babb6910fd393c1dfefe629b90fa53b996a"),
}
PINNED_DECODING = {
    "wta": "0e6176f9cef92b1fcbf8ec87a9501fbf5f3e57aca6d8224cef2eaa3e46173f4e",
    "codebook_aware": "6bd0be00e0cd99326f5c44da4309bb6fcaa3bcf63bc4750023d7f782a4ed8b44",
}
PINNED_DECODING_ENTRIES = {
    "wta": "1843f42012f33eb1577a8bd7fcdc3978a63f6a29a36c96ac4841e01998cb8d5d",
    "codebook_aware": "25ce6e994b95afd1797c73b991c41782e7b9792d0ba39e56f68e39755b44531e",
}
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
# (decoder, filter) -> (accepted count, table digest)
PINNED_FILTERING = {
    ("wta", "default"): (
        6, "bc707599c1d4819c43b830281d25092c8eac7544bdf6403c5d9be65985c1802d"),
    ("wta", "bound"): (
        3, "1cf7b647ba16e4bc3b7ab2ca7a2b959e9aa43213034eb9e3a67f41990cbdff45"),
    ("wta", "end_bases"): (
        6, "05f2e087a9f84d73312aede07f4a1763e1c4dd579d928de35f07a9809a6ae038"),
    ("codebook_aware", "default"): (
        8, "0c2ff86561ae310f2a03c76f86d3b47a53bbf3e7a2e4b56fb779c62601830ac5"),
    ("codebook_aware", "bound"): (
        5, "0c35110c989a13b7b90bd7421895d794eba7e0f6d885e62617f3322a6e100ff5"),
    ("codebook_aware", "end_bases"): (
        8, "bd80466ad744468f8407907e4f7ea2c9e4bee2d54742e4bd7875a7ea88a4dca1"),
}
PINNED_FILTERING_ENTRIES = {
    ("wta", "default"): "4bf3a8b42bc19acd5a8ba529d8cd89832b9c6ee56afe1356a882e9cc08caabd0",
    ("wta", "bound"): "5b7eb75850e3e81434906211cbf9315f74f1e4c31f000548a118827962e377a0",
    ("wta", "end_bases"): "f0fccaa1d7a7fae92be3fb89bf62eabd87e9338f75595cfca216804643fc250c",
    ("codebook_aware", "default"): "217817ec748f3d0b05ef4942ab5addce75842f9ddf26759ea645b86ddc14ce1c",
    ("codebook_aware", "bound"): "b015c4f41158abef80c844007cef549248ee9a41acc06de882ba35bcf60f6326",
    ("codebook_aware", "end_bases"): "9e0e168bd9f0dab0bd8db2ce5d9eef5c26403068b58413478cc157b670c21607",
}
# decoder -> (sha256 of candidates.csv without the background columns, sha256 of pre_qc.csv without
# entry_id and the score columns, each written by the checkpoint writer) written by FOV.run
PINNED_RUN_FILES = {
    "wta": (
        "285e412fecc9c50060f375ecbda9d9c8b2c6c554c1dcf3126cc2162383fa8ee8",
        "a48343b3c570cbc749987dda6189e3a910ac9899a5efc80b76ac09fc515d3a9d"),
    "codebook_aware": (
        "285e412fecc9c50060f375ecbda9d9c8b2c6c554c1dcf3126cc2162383fa8ee8",
        "b80e7beef0da52ffeb166833992ddedd57ab599c21a8965e650db22d9b2d47a9"),
}
# decoder -> sha256 of the pre_qc.csv (with entry_id, without the score columns, written by the
# checkpoint writer) written by FOV.run
PINNED_PRE_QC_CSV_ENTRIES = {
    "wta": "29c269cfe57a6d39d671f76c388c3ba3ea0c586f755bc8d405577e93c75449ce",
    "codebook_aware": "13041598a0b3df5a406481186641d1f822074223c23375fcc174dc3f6f28dc64",
}
# §2.8 (option C1): candidates.csv gains the background columns and pre_qc and the filtering
# tables gain the score columns. The pins above are of the tables without them; these pin the
# extended tables and files written by FOV.run.
PINNED_CANDIDATES_CSV_BACKGROUND = "8feb7c6c11a41bfaf5f6b9af24bee72f1584ff46539d571928a6144a8dd09dcc"
# decoder -> (digest of the scored pre_qc table, sha256 of the scored pre_qc.csv)
PINNED_PRE_QC_SCORED = {
    "wta": ("ef206100aec530fa6dc814386c3b58b022e541d6d052041767622a5fe13f1088",
            "338e0976f483009f09fbad7ac1f620f11f8e4a7c5e09d4e74f58b3681af0b760"),
    "codebook_aware": ("37f16de3b4801b532543f11499dd8cfb5176f5cf2fde338ba349d1fcc2f4891b",
                       "92516655cdbcc913d4fd56bef0e1175500ffab78a780783cd4fbbc772826cd4c"),
}
# (decoder, filter) -> digest of the filtering table of the scored reads after FOV.run
PINNED_FILTERING_SCORED = {
    ("wta", "default"): "4f979dc0ce24efb871e5d3889b81470799d5d140625eb182afce6f72e99501f6",
    ("wta", "bound"): "58cada2233abf4d6f0ddcb67c2b34d4147ec7ae33a0706bbe117e7c05dd5b500",
    ("wta", "end_bases"): "9d5764981750684ff355eb319b8b322c7984cfb5165bb71c48207217fa85a9d3",
    ("codebook_aware", "default"): "b5d31351e6cef812410c591d073cbcc372935bebdd9c5709df6efefc49e884b4",
    ("codebook_aware", "bound"): "40dba301c24a6c7a943179017029f245615749b602cc906f62d2ecba36439dd4",
    ("codebook_aware", "end_bases"): "75d02f3b610385062d1e4b2e0b39af0b670d6aebfd619252ad3788c17c2433d8",
}


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
    assert (digest(result.values), digest(result.valid)) == PINNED_EXTRACTION[radius]


def test_valid_is_always_true_at_the_border_and_in_a_zero_round():
    # Legacy behavior that §2.8 changes (docs/readout-baseline.md): extraction never
    # marks a measurement unavailable, even where the box is clipped by a face.
    assert extract().valid.all()


@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_decoding_tables_are_pinned(tmp_path, decoder):
    table = decode(tmp_path, decoder).table
    assert dict(zip(table.spot_id, zip(table.call_status, table.failure_reason))) == PINNED_STATUS[decoder]
    assert table_digest(without_entry(table)) == PINNED_DECODING[decoder]
    assert table_digest(table) == PINNED_DECODING_ENTRIES[decoder]


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_filtering_tables_are_pinned(tmp_path, decoder, name):
    filtered = filter_reads(decode(tmp_path, decoder),
                            config=readout_config("filtering", decoder=decoder, **filter_options(decoder, name)))
    assert (filtered.counts["accepted"], table_digest(without_entry(filtered.table))) == PINNED_FILTERING[(decoder, name)]
    assert table_digest(filtered.table) == PINNED_FILTERING_ENTRIES[(decoder, name)]


@pytest.mark.parametrize("name", list(FILTERS))
@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_pre_qc_after_run_save_and_reload_is_pinned(tmp_path, decoder, name):
    dataset = golden_dataset(tmp_path)
    dataset.codebook = golden_codebook(tmp_path)
    digests = run_and_reload(tmp_path, dataset,
                             readout_config("pipeline", decoder=decoder, **filter_options(decoder, name)))
    assert (digests["values"], digests["valid"]) == PINNED_EXTRACTION[(1, 2, 2)]
    assert digests["pre_qc"] == PINNED_DECODING[decoder]
    assert digests["filtering"] == PINNED_FILTERING[(decoder, name)][1]
    assert (digests["candidates_csv"], digests["pre_qc_csv"]) == PINNED_RUN_FILES[decoder]
    assert digests["pre_qc_entries"] == PINNED_DECODING_ENTRIES[decoder]
    assert digests["filtering_entries"] == PINNED_FILTERING_ENTRIES[(decoder, name)]
    assert digests["pre_qc_csv_entries"] == PINNED_PRE_QC_CSV_ENTRIES[decoder]
    assert digests["candidates_csv_background"] == PINNED_CANDIDATES_CSV_BACKGROUND
    assert (digests["pre_qc_scored"], digests["pre_qc_csv_scored"]) == PINNED_PRE_QC_SCORED[decoder]
    assert digests["filtering_scored"] == PINNED_FILTERING_SCORED[(decoder, name)]


@pytest.mark.parametrize("name", list(FILTERS))
def test_legacy_yaml_keys_give_the_same_digests(tmp_path, name):
    adapted = readout_config("yaml", **filter_options("wta", name))
    dataset = adapted.dataset
    dataset.load_codebook(write_codebook(tmp_path), split_index=adapted.split_index)
    assert adapted.split_index is None
    digests = run_and_reload(tmp_path, dataset, adapted.pipeline, adapted.execution)
    assert (digests["values"], digests["valid"]) == PINNED_EXTRACTION[(1, 2, 2)]
    assert digests["pre_qc"] == PINNED_DECODING["wta"]
    assert digests["filtering"] == PINNED_FILTERING[("wta", name)][1]
    assert (digests["candidates_csv"], digests["pre_qc_csv"]) == PINNED_RUN_FILES["wta"]
    assert digests["pre_qc_entries"] == PINNED_DECODING_ENTRIES["wta"]
    assert digests["filtering_entries"] == PINNED_FILTERING_ENTRIES[("wta", name)]
    assert digests["pre_qc_csv_entries"] == PINNED_PRE_QC_CSV_ENTRIES["wta"]
    assert digests["candidates_csv_background"] == PINNED_CANDIDATES_CSV_BACKGROUND
    assert (digests["pre_qc_scored"], digests["pre_qc_csv_scored"]) == PINNED_PRE_QC_SCORED["wta"]
    assert digests["filtering_scored"] == PINNED_FILTERING_SCORED[("wta", name)]


def test_a_changed_neighborhood_radius_changes_the_tensor_digest():
    assert digest(extract((1, 3, 3)).values) != PINNED_EXTRACTION[(1, 2, 2)][0]


@pytest.mark.parametrize("gate", [{"allow_rescue": False}, {"max_corrected_round_margin": 0.01},
                                  {"min_geomean_probability": 0.95}], ids=lambda g: next(iter(g)))
def test_a_changed_decoder_gate_changes_the_decoding_digest(tmp_path, gate):
    assert table_digest(without_entry(decode(tmp_path, "codebook_aware", **gate).table)) != PINNED_DECODING["codebook_aware"]


@pytest.mark.parametrize("decoder", ["wta", "codebook_aware"])
def test_a_changed_score_bound_changes_the_filtering_digest(tmp_path, decoder):
    bound = {"wta": WTA_CHANGED_BOUND, "codebook_aware": CODEBOOK_AWARE_CHANGED_BOUND}[decoder]
    filtered = filter_reads(decode(tmp_path, decoder), config=readout_config(
        "filtering", decoder=decoder, **filter_options(decoder, "bound", bound)))
    assert table_digest(without_entry(filtered.table)) != PINNED_FILTERING[(decoder, "bound")][1]


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
        fov.decode_barcodes(config=readout_config("decoding"))
