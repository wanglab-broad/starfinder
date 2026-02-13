"""Tests for dataset type definitions."""

import pytest

from starfinder.dataset.types import (
    Codebook,
    CropWindow,
    LayerState,
    SubtileConfig,
)


class TestLayerState:
    """Tests for LayerState dataclass."""

    def test_all_layers_order(self):
        ls = LayerState(seq=["r1", "r2"], other=["p1"], ref="r1")
        assert ls.all_layers == ["r1", "r2", "p1"]

    def test_to_register_excludes_ref(self):
        ls = LayerState(seq=["r1", "r2", "r3"], ref="r1")
        assert ls.to_register == ["r2", "r3"]

    def test_validate_ref_not_in_layers(self):
        ls = LayerState(seq=["r1"], ref="missing")
        with pytest.raises(ValueError, match="not found"):
            ls.validate()

    def test_validate_overlap(self):
        ls = LayerState(seq=["r1"], other=["r1"], ref="r1")
        with pytest.raises(ValueError, match="both seq and other"):
            ls.validate()

    def test_validate_passes(self):
        ls = LayerState(seq=["r1", "r2"], other=["p1"], ref="r1")
        ls.validate()  # should not raise


class TestCodebook:
    """Tests for Codebook dataclass."""

    def test_from_csv(self, mini_dataset):
        cb = Codebook.from_csv(mini_dataset / "codebook.csv")
        assert cb.n_genes == 8
        assert "GeneA" in cb.genes
        assert len(cb.seq_to_gene) == 8

    def test_genes_sorted(self):
        cb = Codebook(
            gene_to_seq={"B": "11", "A": "22"},
            seq_to_gene={"11": "B", "22": "A"},
        )
        assert cb.genes == ["A", "B"]


class TestCropWindow:
    """Tests for CropWindow dataclass."""

    def test_to_slice(self):
        w = CropWindow(y_start=10, y_end=50, x_start=20, x_end=60)
        sy, sx = w.to_slice()
        assert sy == slice(10, 50)
        assert sx == slice(20, 60)

    def test_frozen(self):
        w = CropWindow(0, 10, 0, 10)
        with pytest.raises(AttributeError):
            w.y_start = 5


class TestSubtileConfig:
    """Tests for SubtileConfig dataclass."""

    def test_compute_windows_2x2(self):
        cfg = SubtileConfig(sqrt_pieces=2, overlap_ratio=0.1)
        cfg.compute_windows(100, 100)
        assert cfg.n_subtiles == 4

        # First tile (top-left): no overlap on outer edges
        w0 = cfg.windows[0]
        assert w0.y_start == 0
        assert w0.x_start == 0
        # Extended inward by overlap_half
        assert w0.y_end > 50
        assert w0.x_end > 50

        # Last tile (bottom-right): clamped to image boundary
        w3 = cfg.windows[3]
        assert w3.y_end == 100
        assert w3.x_end == 100

    def test_compute_windows_1x1(self):
        cfg = SubtileConfig(sqrt_pieces=1, overlap_ratio=0.0)
        cfg.compute_windows(256, 256)
        assert cfg.n_subtiles == 1
        w = cfg.windows[0]
        assert (w.y_start, w.y_end, w.x_start, w.x_end) == (0, 256, 0, 256)

    def test_n_subtiles_before_compute(self):
        cfg = SubtileConfig(sqrt_pieces=3)
        assert cfg.n_subtiles == 0
