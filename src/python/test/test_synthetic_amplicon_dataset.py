"""The public in-memory writer ``save_amplicon_dataset`` (W-559; W-526, NM-22 and SM-20, choice B).

It writes what the private streaming writer of ``starfinder synthetic generate --mode e2e`` writes for the same
dataset, in the round-first layout that ``Dataset`` reads: the same pixel arrays, the same truth files (apart from
the generator path, which both record as ``generate_amplicon_dataset``) and the same codebook entries.
"""
from dataclasses import replace
import json

import numpy as np
import pytest
import tifffile

from starfinder.benchmark._synthetic_io import _write_dataset
from starfinder.readout import EncodingConfig, load_codebook
from starfinder.synthetic import (benchmark_scene_preset, generate_amplicon_dataset, generate_registration_pair,
                                  save_amplicon_dataset)

pytestmark = pytest.mark.synthetic

TRUTH_CSV = ("formed.csv", "round_truth.csv", "scene_truth.csv")
TRUTH_JSON = ("ground_truth.json", "generation.json")


def _tiny(**kwargs):
    codebook, config = benchmark_scene_preset("tiny")
    return generate_amplicon_dataset(codebook, replace(config, seed=42), fov_ids=("FOV_001", "FOV_002"),
                                     preset="tiny", **kwargs)


def _entries(path, book):
    loaded = load_codebook(path, round_labels=book.round_labels, channel_labels=book.channel_labels,
                           encoding=EncodingConfig(reverse_bases=True))
    return loaded.table[["entry_id", "gene_id", "color_sequence"]].values.tolist()


def test_the_public_and_the_streaming_writer_write_the_same_dataset(tmp_path):
    streamed = tmp_path / "streamed"
    _write_dataset("tiny", streamed, seed=42)
    dataset = _tiny()
    written = save_amplicon_dataset(dataset, tmp_path / "public")
    assert written == tmp_path / "public"
    book = dataset.codebook
    images = {f"{label}/{fov}/{channel}.tif" for fov in ("FOV_001", "FOV_002") for label in book.round_labels
              for channel in book.channel_labels}
    files = {p.relative_to(written).as_posix() for p in written.rglob("*") if p.is_file()}
    assert files == images | {"codebook.csv", *TRUTH_CSV, *TRUTH_JSON}
    for name in sorted(images):
        label, fov, channel = name.split("/")
        expected = tifffile.imread(streamed / fov / label / channel)
        actual = tifffile.imread(written / name)
        assert actual.dtype == expected.dtype and actual.shape == expected.shape == tuple(dataset.rounds[fov][
            label].shape[:3])
        np.testing.assert_array_equal(actual, expected)
    for name in TRUTH_CSV:
        assert (written / name).read_text() == (streamed / name).read_text(), name
    for name in TRUTH_JSON:
        public, stream = (json.loads((root / name).read_text()) for root in (written, streamed))
        public.pop("generator", None), stream.pop("generator", None)
        assert public == stream, name
    assert (written / "codebook.csv").read_text().splitlines()[0] == "entry_id,gene_id,color_sequence"
    public_entries, streamed_entries = _entries(written / "codebook.csv", book), _entries(streamed / "codebook.csv",
                                                                                         book)
    assert [row[1:] for row in public_entries] == [row[1:] for row in streamed_entries]
    assert [row[1:] for row in public_entries] == book.table[["gene_id", "color_sequence"]].values.tolist()
    # The entry IDs follow the gene,barcode codebook (the barcode as written), so decoded tables are equal too.
    assert public_entries == streamed_entries


def test_a_streamed_dataset_raises_value_error(tmp_path):
    dataset = _tiny(on_round=lambda fov, label, image, metadata: None)
    with pytest.raises(ValueError, match="on_round"):
        save_amplicon_dataset(dataset, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_a_nonempty_directory_raises_file_exists_error(tmp_path):
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "keep.txt").write_text("kept\n")
    with pytest.raises(FileExistsError):
        save_amplicon_dataset(_tiny(), tmp_path / "out")
    assert [p.name for p in (tmp_path / "out").iterdir()] == ["keep.txt"]


def test_other_inputs_are_rejected(tmp_path):
    with pytest.raises(TypeError):
        save_amplicon_dataset({"rounds": {}}, tmp_path / "out")
    with pytest.raises(ValueError, match="generate_amplicon_dataset"):
        save_amplicon_dataset(generate_registration_pair("tiny"), tmp_path / "out")
