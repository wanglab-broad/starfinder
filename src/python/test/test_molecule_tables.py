"""The molecule and read tables of W-562 (OM-04, OM-18; docs/migration.md, "Molecule and read tables").

``FOV.export_spots`` writes ``fovs/<fov_id>/molecules.csv`` (the accepted reads with their identities and zero-based
ZYX, ``entry_id`` and ``qc_score``) and, on request, ``reads.csv`` (every read joined to its detection, each column
name once); ``run.json`` of the FOV's latest checkpointed run gains ``outputs.<slot>``; ``assignment.load_molecules``
reads ``molecules.csv`` back; the workflow chooses the molecule table from the configuration. Fixtures: the bounded
4×12×14 coordination FOV (two spots) and the direct2 FOV of test_readout_direct.py.
"""
import hashlib
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from starfinder.assignment import import_legacy_molecules_csv, load_molecules, molecule_table
from starfinder.dataset import CheckpointConfig
from starfinder.dataset.workflow import _legacy_molecules, _molecule_table
from starfinder.readout import ReadScoreConfig, export_spots
from starfinder.readout.spots import MOLECULE_DTYPES

from . import hash_oracles
from .coordination_helpers import write_molecules_csv
from .test_coordination_contract import complete, dataset, resident
from .test_readout_direct import direct_dataset, direct_fov, pipeline

pytestmark = [pytest.mark.readout, pytest.mark.dataset, pytest.mark.contract]

MOLECULE_COLUMNS = ["spot_namespace", "spot_id", "z", "y", "x", "gene_id", "entry_id", "qc_score"]


def accepted_spots(fov):
    """The detection rows of the accepted reads, in detection order, with the filtering columns."""
    accepted = fov.filtering_result.accepted
    return fov.spot_result.spots.merge(accepted, on="spot_id", validate="one_to_one", sort=False)


def test_molecules_csv_is_typed_zero_based_and_keyed_with_unscored_reads(tmp_path):
    fov = resident(dataset(tmp_path)).run(complete())
    path = fov.export_spots()
    assert path == tmp_path / "out" / "fovs" / "FOV" / "molecules.csv"
    text = path.read_text().splitlines()
    assert text[0] == ",".join(MOLECULE_COLUMNS)
    table = hash_oracles.read_table(path, MOLECULE_DTYPES)
    assert {c: str(t) for c, t in table.dtypes.items()} == MOLECULE_DTYPES
    expected = accepted_spots(fov)
    assert len(table) == len(expected) == 2
    assert table.spot_id.tolist() == expected.spot_id.tolist()
    assert set(table.spot_namespace) == {fov.spot_result.spot_namespace}
    # Zero-based ZYX: exactly the in-memory detection coordinates.
    np.testing.assert_array_equal(table[["z", "y", "x"]].to_numpy(), expected[["z", "y", "x"]].to_numpy())
    assert table.gene_id.tolist() == expected.gene_id.tolist()
    assert table.entry_id.tolist() == expected.entry_id.tolist()
    # An unscored run has no qc_score: <NA> in the file, missing in the table.
    assert table.qc_score.isna().all() and all(line.endswith(",<NA>") for line in text[1:])


def test_molecules_csv_holds_the_qc_score_of_scored_reads(tmp_path):
    fov = resident(dataset(tmp_path)).run(complete(scoring=ReadScoreConfig()))
    table = hash_oracles.read_table(fov.export_spots(), MOLECULE_DTYPES)
    expected = accepted_spots(fov)
    assert expected.qc_score.notna().all()
    np.testing.assert_allclose(table.qc_score, expected.qc_score, rtol=0, atol=1e-9)


def test_reads_csv_holds_every_read_once_with_unique_columns(tmp_path):
    fov = resident(dataset(tmp_path)).run(complete())
    decoded = fov.decoding_result.table.copy()
    decoded.loc[decoded.index[0], ["call_status", "gene_id"]] = ["unmatched", pd.NA]
    fov.decoding_result = replace(fov.decoding_result, table=decoded)
    path = fov.export_spots("reads")
    assert path == tmp_path / "out" / "fovs" / "FOV" / "reads.csv"
    header = path.read_text().splitlines()[0].split(",")
    assert len(header) == len(set(header))
    assert header[:5] == ["spot_namespace", "spot_id", "z", "y", "x"]
    assert set(header) == set(fov.spot_result.spots.columns) | set(decoded.columns)
    table = pd.read_csv(path, dtype={"spot_id": str}, na_values=["<NA>"], keep_default_na=False)
    spots = fov.spot_result.spots
    assert table.spot_id.tolist() == spots.spot_id.tolist()
    np.testing.assert_array_equal(table[["z", "y", "x"]].to_numpy(), spots[["z", "y", "x"]].to_numpy())
    assert table.gene_id.isna().tolist() == [True, False] and table.call_status.tolist()[0] == "unmatched"
    # columns selects and orders; an unknown column raises.
    selected = fov.export_spots("reads", ["spot_id", "x", "gene_id"], path=tmp_path / "selected.csv")
    assert selected.read_text().splitlines()[0] == "spot_id,x,gene_id"
    with pytest.raises(ValueError, match="gene"):
        fov.export_spots("reads", ["gene"], path=tmp_path / "bad.csv")
    with pytest.raises(ValueError, match="molecules or reads"):
        fov.export_spots("goodSpots")
    with pytest.raises(ValueError, match=r"\.csv"):
        fov.export_spots(path=tmp_path / "molecules.txt")


def test_direct_readout_keeps_one_round_and_one_channel(tmp_path):
    fov = direct_fov(direct_dataset(tmp_path)).run(pipeline())
    labels = fov.decoding_result.channel_labels
    path = fov.export_spots("reads")
    header = path.read_text().splitlines()[0].split(",")
    assert len(header) == len(set(header)) and header.count("round") == header.count("channel") == 1
    assert not [c for c in header if c.endswith(("_x", "_y"))]
    table = pd.read_csv(path, dtype={"spot_id": str, "channel": str}, na_values=["<NA>"], keep_default_na=False)
    spots = fov.spot_result.spots
    # The reads' copy: the round, and the channel label of the detection's channel index.
    assert table["round"].tolist() == spots["round"].tolist()
    assert table.channel.tolist() == [labels[i] for i in spots.channel]
    molecules = hash_oracles.read_table(fov.export_spots(), MOLECULE_DTYPES)
    assert molecules.gene_id.tolist() == ["Gfap", "Gad1", "Sst", "Aqp4"]
    expected = fov.filtering_result.accepted.set_index("spot_id")
    assert molecules.entry_id.tolist() == expected.loc[molecules.spot_id, "entry_id"].tolist()
    assert molecules.qc_score.isna().all()


@pytest.mark.parametrize("column", ["round", "channel"])
def test_a_shared_column_that_disagrees_raises(tmp_path, column):
    fov = direct_fov(direct_dataset(tmp_path)).run(pipeline())
    table = fov.decoding_result.table.copy()
    other = {"round": "round2" if table["round"].iloc[0] == "round1" else "round1",
             "channel": next(c for c in fov.decoding_result.channel_labels if c != table.channel.iloc[0])}
    table.loc[table.index[0], column] = other[column]
    reads = replace(fov.decoding_result, table=table)
    with pytest.raises(ValueError, match=f"disagree on '{column}'"):
        export_spots(fov.spot_result, reads, tmp_path / "reads.csv")


def run_record(fov):
    return json.loads((fov.paths.fov_dir / "run.json").read_text())


def test_run_json_records_the_exported_tables(tmp_path):
    fov = resident(dataset(tmp_path)).run(complete(), checkpoints=CheckpointConfig(stages=()))
    before = run_record(fov)
    assert before["outputs"] == {}
    path = fov.export_spots()
    record = run_record(fov)
    assert record["format_version"] == 2
    assert {k: v for k, v in record.items() if k != "outputs"} == {k: v for k, v in before.items() if k != "outputs"}
    entry = record["outputs"]["molecules"]
    assert set(entry) == {"path", "sha256", "molecules_sha256", "dtypes", "rows"}
    assert entry["path"] == "molecules.csv"
    assert (fov.paths.fov_dir / entry["path"]).resolve() == path.resolve()
    assert entry["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    loaded = load_molecules(path, genes=("gene",))
    joined = molecule_table(fov.spot_result, fov.filtering_result, genes=("gene",))
    assert entry["molecules_sha256"] == loaded.sha256 == joined.sha256
    assert entry["dtypes"] == MOLECULE_DTYPES and entry["rows"] == len(loaded) == 2
    reads = fov.export_spots("reads")
    entry = run_record(fov)["outputs"]["reads"]
    assert entry["path"] == "reads.csv" and entry["molecules_sha256"] is None
    assert entry["sha256"] == hashlib.sha256(reads.read_bytes()).hexdigest() and entry["rows"] == 2
    assert run_record(fov)["outputs"]["molecules"]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    # A later run without checkpoints has no record: export_spots records nothing, the earlier run.json stays.
    saved = (fov.paths.fov_dir / "run.json").read_bytes()
    fov.run(complete())
    fov.export_spots()
    assert (fov.paths.fov_dir / "run.json").read_bytes() == saved


def test_without_a_checkpointed_run_nothing_is_recorded(tmp_path):
    fov = resident(dataset(tmp_path)).run(complete())
    fov.export_spots()
    assert not (tmp_path / "out" / "fovs" / "FOV" / "run.json").exists()
    assert not (tmp_path / "out" / "fovs" / "FOV" / "checkpoints").exists()


def test_load_molecules_keeps_the_identities_and_checks_the_file(tmp_path):
    namespace = '["data","sample","FOV",null]'
    path = write_molecules_csv(tmp_path / "molecules.csv", namespace, [(1, 2, 3), (4.5, 5, 6)], ["A", "B"],
                               ["s7", "s3"])
    table = load_molecules(path, genes=("A", "B"))
    assert table.population == "final" and table.spot_namespace == namespace
    assert table.table.spot_id.tolist() == ["s7", "s3"]
    assert table.table[["z", "y", "x"]].to_numpy().tolist() == [[1.0, 2.0, 3.0], [4.5, 5.0, 6.0]]
    assert table.source == {"spot_namespace": namespace, "path": str(path),
                            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    with pytest.raises(ValueError, match="spot_namespace"):
        load_molecules(path, genes=("A", "B"), spot_namespace='["data","sample","OTHER",null]')
    empty = write_molecules_csv(tmp_path / "empty.csv", namespace, [], [])
    with pytest.raises(ValueError, match="pass spot_namespace"):
        load_molecules(empty, genes=("A",))
    assert len(load_molecules(empty, genes=("A",), spot_namespace=namespace)) == 0
    legacy = tmp_path / "goodSpots.csv"
    legacy.write_text("x,y,z,gene\n1,2,3,A\n")
    with pytest.raises(ValueError, match="is not a molecules.csv"):
        load_molecules(legacy, genes=("A",))
    with pytest.raises(FileNotFoundError):
        load_molecules(tmp_path / "missing.csv", genes=("A",))
    # The legacy reader stays for one-based files.
    assert import_legacy_molecules_csv(legacy, spot_namespace=namespace, genes=("A",)).table.spot_id.tolist() == \
        ["csv:0"]


@pytest.mark.stitching
def test_dataset_export_reads_the_molecule_table_of_its_route(tmp_path):
    """Dataset.export(assignment=None) reads fovs/<fov>/molecules.csv with load_molecules and keeps the identities;
    on the workflow's legacy route (the dataset's _legacy_molecules, set from the configuration) it reads the
    one-based signal/<fov>_goodSpots.csv with the legacy reader, whose identities are csv:<row>."""
    from starfinder.dataset._export import assembly_inputs

    from .fusion_fixtures import export_folder
    folder = export_folder(tmp_path)
    inputs = assembly_inputs(folder.dataset, folder.placement, None)
    assert [i.fov_id for i in inputs] == list(folder.placement.fovs.fov_id)
    for item in inputs:
        expected = folder.results[item.fov_id].molecules
        assert item.molecules.table.spot_id.tolist() == expected.spot_id.tolist()
        assert not item.molecules.table.spot_id.str.startswith("csv:").any()
        assert item.molecules.source["path"] == str(folder.root / "fovs" / item.fov_id / "molecules.csv")
        np.testing.assert_array_equal(item.molecules.table[["z", "y", "x"]].to_numpy(),
                                      expected[["z", "y", "x"]].to_numpy(np.float64))
        legacy = folder.root / "signal" / f"{item.fov_id}_goodSpots.csv"
        legacy.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"x": expected.x + 1, "y": expected.y + 1, "z": expected.z + 1,
                      "gene": expected.gene_id.astype(str)}).to_csv(legacy, index=False)
    for item in assembly_inputs(replace(folder.dataset, _legacy_molecules=True), folder.placement, None):
        expected = folder.results[item.fov_id].molecules
        assert item.molecules.table.spot_id.tolist() == [f"csv:{i}" for i in range(len(expected))]
        assert item.molecules.source["coordinates"] == "one-based"
        np.testing.assert_allclose(item.molecules.table[["z", "y", "x"]].to_numpy(),
                                   expected[["z", "y", "x"]].to_numpy(np.float64), rtol=0, atol=1e-9)


@pytest.mark.workflow
@pytest.mark.parametrize("config, legacy", [
    ({"backend": "python"}, False),
    ({"backend": "python", "workflow_mode": "direct"}, False),
    ({"backend": "python", "rules": {"rsf_single_fov": {"run": True}, "gr_single_fov_subtile": {"run": False}}}, False),
    ({"backend": "matlab"}, True),
    ({}, True),
    ({"backend": "python", "workflow_mode": "subtile"}, True),
    ({"backend": "python", "workflow_mode": "deep"}, True),
    ({"backend": "python", "rules": {"gr_single_fov_subtile": {"run": True}}}, True),
    ({"backend": "python", "rules": {"deep_create_subtile": {"run": True}}}, True),
])
def test_the_workflow_chooses_the_molecule_table_from_the_configuration(tmp_path, config, legacy):
    assert _legacy_molecules(config) is legacy
    path = _molecule_table(config, tmp_path, "{fovID}")
    assert path == (tmp_path / "signal" / "{fovID}_goodSpots.csv" if legacy else
                    tmp_path / "fovs" / "{fovID}" / "molecules.csv")
