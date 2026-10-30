"""Decoding does no per-read codebook work: lookups built once per call, no codebook scan per read (W-722)."""

import pickle

import numpy as np
import pytest
from starfinder.readout import (
    Codebook,
    CodebookAwareDecoderConfig,
    WtaDecoderConfig,
    decode_reads,
)
from starfinder.readout import _codebook_aware

from .readout_cases import codebook, intensity, tensor

pytestmark = pytest.mark.readout

LOOKUPS = ("seq_to_gene", "gene_to_seq", "seq_to_entry", "entry_to_seq")
CONFIGS = {
    "wta": WtaDecoderConfig(diagnostics=False),
    "wta_diagnostics": WtaDecoderConfig(diagnostics=True),
    "codebook_aware": CodebookAwareDecoderConfig(diagnostics=False),
    "codebook_aware_diagnostics": CodebookAwareDecoderConfig(diagnostics=True),
    "codebook_aware_h2_diagnostics": CodebookAwareDecoderConfig(diagnostics=True, max_hamming=2),
}


def _scan(wta_seq, one_error_index, seq_to_gene, unknown_chars="MN", max_hamming=1):
    """_candidate_sequences as of 56c35b8: every codebook sequence compared with the read."""
    if wta_seq in seq_to_gene:
        return [wta_seq]
    unknowns = sum(char in unknown_chars for char in wta_seq)
    if unknowns == 0 and max_hamming == 1:
        return one_error_index.get(wta_seq, [])
    return sorted(
        seq for seq in seq_to_gene
        if len(seq) == len(wta_seq)
        and sum(o not in unknown_chars and o != c for o, c in zip(wta_seq, seq)) + unknowns <= max_hamming
    )


def _book(n_entries=60, rounds=5, seed=7):
    rng = np.random.default_rng(seed)
    codes = rng.choice(4**rounds, n_entries, replace=False)
    seqs = ["".join(str(1 + (c // 4**k) % 4) for k in range(rounds)) for c in codes]
    return {s: f"G{i}" for i, s in enumerate(seqs)}


def _reads(mapping, n, seed=3):
    """n reads: exact entries, entries with one round changed and entries with one round tied (M)."""
    rng = np.random.default_rng(seed)
    seqs = list(mapping)
    picked = [seqs[i] for i in rng.integers(0, len(seqs), n)]
    values = tensor(picked)
    for i in range(n):
        j = int(rng.integers(0, values.shape[2]))
        own = int(picked[i][j]) - 1
        other = (own + 1 + int(rng.integers(0, 3))) % 4
        if i % 3 == 1:
            values[i, :, j] = 1.0
            values[i, other, j] = 100.0
        elif i % 3 == 2:
            values[i, other, j] = 100.0
    return intensity(values)


class _Counted:
    """Replaces the lookup properties of Codebook with ones that count each build."""

    def __init__(self, monkeypatch):
        self.counts = dict.fromkeys(LOOKUPS, 0)
        for name in LOOKUPS:
            original = Codebook.__dict__[name].fget
            monkeypatch.setattr(Codebook, name, property(self._counting(name, original)))

    def _counting(self, name, original):
        def get(book):
            self.counts[name] += 1
            return original(book)
        return get


@pytest.mark.parametrize("path", list(CONFIGS))
def test_each_lookup_is_built_a_constant_number_of_times_per_decode(monkeypatch, path):
    mapping = _book()
    book = codebook(mapping)
    counted = _Counted(monkeypatch)
    builds = []
    for n in (12, 48):
        reads = _reads(mapping, n)
        counted.counts = dict.fromkeys(LOOKUPS, 0)
        decode_reads(reads, book, config=CONFIGS[path])
        builds.append(dict(counted.counts))
    assert builds[0] == builds[1]
    assert all(count <= 2 for count in builds[0].values()), builds[0]


@pytest.mark.parametrize("name", LOOKUPS)
def test_a_returned_lookup_can_be_mutated_without_changing_the_codebook(name):
    book = codebook({"1234": "A", "4321": "B"})
    before = (repr(book), book.table.copy(), getattr(book, name), pickle.dumps(book))
    returned = getattr(book, name)
    assert type(returned) is dict
    key = next(iter(returned))
    returned[key] = "changed"
    returned["extra"] = "added"
    assert getattr(book, name) == before[2]
    assert getattr(book, name) is not returned
    assert repr(book) == before[0]
    assert book.table.equals(before[1])
    assert pickle.dumps(book) == before[3]


class _NoScan(dict):
    """A sequence lookup that fails when iterated: membership is allowed, a scan is not."""

    def __iter__(self):
        raise AssertionError("the codebook was scanned")

    def keys(self):
        raise AssertionError("the codebook was scanned")

    def items(self):
        raise AssertionError("the codebook was scanned")


def test_an_unmatched_read_at_max_hamming_0_scans_no_codebook():
    mapping = _NoScan(_book())
    for read in ("11111", "1111M", "MN111"):
        assert read not in mapping
        assert _codebook_aware._candidate_sequences(read, {}, mapping, max_hamming=0) == []
    exact = next(iter(_book()))
    assert _codebook_aware._candidate_sequences(exact, {}, mapping, max_hamming=0) == [exact]


@pytest.mark.parametrize("path", ["wta_diagnostics", "codebook_aware_diagnostics",
                                  "codebook_aware_h2_diagnostics"])
def test_decoding_compares_no_read_with_every_codebook_sequence(monkeypatch, path):
    # 1,024 entries of 4**6: each read's nearby sequences (at most 1 + 18 + 15 * 9 = 154) are fewer.
    mapping = _book(n_entries=1024, rounds=6)
    calls = []
    original = _codebook_aware._known_hamming
    monkeypatch.setattr(_codebook_aware, "_known_hamming",
                        lambda *args: calls.append(args) or original(*args))
    result = decode_reads(_reads(mapping, 60), codebook(mapping), config=CONFIGS[path])
    assert calls == []
    assert (result.table.call_status != "assigned").any()


@pytest.mark.parametrize("max_hamming", [0, 1, 2, 3])
@pytest.mark.parametrize("n_entries", [6, 300])
def test_the_candidates_equal_those_of_the_codebook_scan(max_hamming, n_entries):
    mapping = _book(n_entries=n_entries, rounds=6, seed=11)
    index = _codebook_aware._build_one_error_index(mapping, 4, 6)
    rng = np.random.default_rng(5)
    seqs = list(mapping)
    reads = list(seqs[:5])
    for _ in range(300):
        read = list(seqs[int(rng.integers(0, len(seqs)))])
        for k in rng.choice(6, int(rng.integers(0, 4)), replace=False):
            read[k] = "1234MN"[int(rng.integers(0, 6))]
        reads.append("".join(read))
    unknown = [r for r in reads if "M" in r or "N" in r]
    assert len(unknown) > 50 and any(r.count("M") + r.count("N") >= 2 for r in unknown)
    for read in reads:
        expected = _scan(read, index, mapping, max_hamming=max_hamming)
        assert _codebook_aware._candidate_sequences(read, index, mapping, max_hamming=max_hamming,
                                                    labels="1234") == expected, read
        assert _codebook_aware._candidate_sequences(read, index, mapping, max_hamming=max_hamming) == expected


def test_the_candidate_rows_equal_those_of_the_candidate_table():
    mapping = _book(n_entries=300, rounds=6, seed=11)
    probs = _codebook_aware._channel_probabilities(_reads(mapping, 3).values)[1]
    seqs = list(mapping)[:40]
    table = _codebook_aware._score_candidates(probs, seqs)
    assert _codebook_aware._candidate_scores(probs, seqs) == list(table.itertuples(index=False, name=None))
    assert _codebook_aware._candidate_scores(probs, []) == []
    with pytest.raises(ValueError, match="has length"):
        _codebook_aware._candidate_scores(probs, ["123"])
