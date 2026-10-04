"""Codebook entries separate from genes (W-292, D5; docs/readout-contract.md, "Codebook entries and genes").

Check R4 of docs/readout-algorithms.md on the ``entries`` fixture: the W-278 ``cal`` scene
of seed 103 (condition ``mixing``: calibrated_scene_preset("clean") with the preset's own
crosstalk switched on, as W-278 built it) with its 16-entry codebook relabeled to 6 genes
(entries e1-e16, genes g1-g6, 2 to 3 entries each). Per-entry decoding rows equal the run
with unique genes, each gene's count in summarize_reads is the sum of its entries' counts, and a repeated
color_sequence, entry_id or base_sequence raises naming both source rows.
"""
import warnings
from dataclasses import replace

import pandas as pd
import pytest

from starfinder.barcode import (Codebook, CodebookAwareDecoderConfig, NeighborhoodSumConfig, WtaDecoderConfig,
                                decode_barcodes, extract_intensities, load_codebook, summarize_reads)
from starfinder.io import ImageLoadResult
from starfinder.spot_finding import LocalMaximaConfig, find_spots
from starfinder.synthetic import calibrated_scene_preset, generate_formed_scene

pytestmark = pytest.mark.barcode

SEED = 103
# entry -> gene: six genes of 2 to 3 entries each.
GENE_OF = {f"e{k}": f"g{g}" for k, g in zip(range(1, 17), [1, 1, 1, 2, 2, 2, 3, 3, 4, 4, 4, 5, 5, 6, 6, 6])}


@pytest.fixture(scope="module")
def cal_103():
    """Codebook, intensities and decoder configs of the W-278 cal scene (seed 103, mixing)."""
    book, config = calibrated_scene_preset("clean", seed=SEED)
    config = replace(config, readout=replace(config.readout, mixing_enabled=True))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        scene = generate_formed_scene(book, config=config)
        labels = list(scene.round_labels)
        spots = find_spots(scene.rounds[labels[0]], config=LocalMaximaConfig(), metadata=scene.metadata,
                           spot_namespace="cal-103")
    rounds = {label: ImageLoadResult(scene.rounds[label], scene.metadata, book.channel_labels, ())
              for label in labels}
    return book, extract_intensities(rounds, spots, config=NeighborhoodSumConfig((1, 2, 2)))


def relabel(book, genes):
    """The codebook's entries as e1-e16 with the given gene of each entry."""
    entries = [f"e{k}" for k in range(1, len(book.table) + 1)]
    table = pd.DataFrame({"entry_id": entries, "gene_id": [genes[e] for e in entries],
                          "color_sequence": book.table.color_sequence.astype(str).tolist()})
    return Codebook(table, book.round_labels, book.channel_labels, book.color_to_channel, book.encoding)


@pytest.mark.validation
@pytest.mark.parametrize("config", [WtaDecoderConfig(), CodebookAwareDecoderConfig()], ids=["wta", "codebook_aware"])
def test_r4_entries_sharing_a_gene_decode_as_with_unique_genes(cal_103, config):
    book, intensities = cal_103
    assert len(book.table) == 16
    unique = relabel(book, {e: e for e in GENE_OF})
    shared = relabel(book, GENE_OF)
    assert (shared.n_entries, shared.n_genes, unique.n_genes) == (16, 6, 16)
    with pytest.raises(ValueError, match="one entry per gene"):
        shared.gene_to_seq
    by_entry = decode_barcodes(intensities, unique, config=config).table
    shared_reads = decode_barcodes(intensities, shared, config=config)
    by_gene = shared_reads.table
    assert by_entry.call_status.eq("assigned").sum() > 0
    # Per-entry rows: status, entry, sequences and scores equal; the gene is the entry's gene.
    gene_columns = [c for c in ("gene_id", "gene_wta") if c in by_entry]
    pd.testing.assert_frame_equal(by_gene.drop(columns=gene_columns), by_entry.drop(columns=gene_columns),
                                  check_exact=True)
    expected = by_entry.entry_id.map(GENE_OF, na_action="ignore").astype("string")
    pd.testing.assert_series_equal(by_gene.gene_id, expected, check_names=False)
    assert by_gene.entry_id.notna().eq(by_gene.call_status.eq("assigned")).all()
    # summarize_reads per gene: each gene's count is the sum of its entries' counts.
    summary = summarize_reads(shared_reads)
    per_entry, per_gene = summary["entries"], summary["genes"]
    assert sum(per_gene.values()) == sum(per_entry.values()) == summary["call_status"]["assigned"]
    assert set(per_gene) <= set(GENE_OF.values()) and set(per_entry) <= set(GENE_OF)
    for gene in set(GENE_OF.values()):
        assert per_gene.get(gene, 0) == sum(per_entry.get(e, 0) for e, g in GENE_OF.items() if g == gene)
    # The run with unique genes reports the same per-entry counts.
    assert summarize_reads(decode_barcodes(intensities, unique, config=config))["entries"] == per_entry


@pytest.mark.validation
@pytest.mark.parametrize("column, rows, message", [
    ("color_sequence", "entry_id,gene_id,color_sequence\ne1,g1,1234\ne2,g1,4321\ne3,g2,1234\n",
     "row 4: encoded sequence collision '1234' with row 2"),
    ("entry_id", "entry_id,gene_id,color_sequence\ne1,g1,1234\ne2,g1,4321\ne1,g2,2222\n",
     "row 4: repeated entry_id 'e1' of row 2"),
    ("base_sequence", "entry_id,gene_id,color_sequence,base_sequence\ne1,g1,4422,CACGC\ne2,g1,2222,CACAC\n"
                      "e3,g2,4422,CACGC\n", "row 4: repeated base_sequence 'CACGC' of row 2"),
], ids=["color_sequence", "entry_id", "base_sequence"])
def test_r4_repeated_identifiers_raise_naming_both_rows(tmp_path, column, rows, message):
    path = tmp_path / "entries.csv"
    path.write_text(rows)
    with pytest.raises(ValueError, match=message):
        load_codebook(path, round_labels=("r1", "r2", "r3", "r4"), channel_labels=("a", "b", "c", "d"))
    # The same table built directly names its own rows (1-based, without the header).
    table = pd.read_csv(path, dtype=str)
    with pytest.raises(ValueError, match=message.replace("row 4", "codebook row 3").replace("row 2", "row 1")):
        Codebook(table, ("r1", "r2", "r3", "r4"), ("a", "b", "c", "d"))


@pytest.mark.contract
def test_entry_lookups_and_canonical_files(tmp_path):
    path = tmp_path / "entries.csv"
    path.write_text("entry_id,gene_id,color_sequence\nGfap_probe1,Gfap,1234\nGfap_probe2,Gfap,4321\n"
                    "Mbp_probe1,Mbp,2222\n")
    book = load_codebook(path, round_labels=("r1", "r2", "r3", "r4"), channel_labels=("a", "b", "c", "d"))
    assert book.seq_to_entry == {"1234": "Gfap_probe1", "4321": "Gfap_probe2", "2222": "Mbp_probe1"}
    assert book.entry_to_seq == {"Gfap_probe1": "1234", "Gfap_probe2": "4321", "Mbp_probe1": "2222"}
    assert book.seq_to_gene == {"1234": "Gfap", "4321": "Gfap", "2222": "Mbp"}
    assert (book.genes, book.n_genes, book.n_entries) == (["Gfap", "Mbp"], 2, 3)
    assert repr(book) == ("Codebook: 3 entries of 2 genes × 4 rounds, channels a, b, c, d; "
                          "encoding two_base, one segment of 4 colors")
    # Without entry_id, entry_id is the color_sequence.
    path.write_text("gene_id,color_sequence\nGfap,1234\n")
    book = load_codebook(path, round_labels=("r1", "r2", "r3", "r4"), channel_labels=("a", "b", "c", "d"))
    assert book.table.entry_id.tolist() == ["1234"] and book.gene_to_seq == {"Gfap": "1234"}
