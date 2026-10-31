"""Tests for dataset type definitions."""

import pytest

from starfinder.dataset.types import (
    RoundState,
)

pytestmark = pytest.mark.dataset


class TestLayerState:
    """Tests for RoundState dataclass."""

    def test_all_layers_order(self):
        ls = RoundState(sequencing_rounds=["r1", "r2"], other_rounds=["p1"], reference_round="r1")
        assert ls.all_rounds == ["r1", "r2", "p1"]

    def test_to_register_excludes_ref(self):
        ls = RoundState(sequencing_rounds=["r1", "r2", "r3"], reference_round="r1")
        assert ls.moving_rounds == ["r2", "r3"]

    def test_validate_ref_not_in_layers(self):
        ls = RoundState(sequencing_rounds=["r1"], reference_round="missing")
        with pytest.raises(ValueError, match="not found"):
            ls.validate()

    def test_validate_overlap(self):
        ls = RoundState(sequencing_rounds=["r1"], other_rounds=["r1"], reference_round="r1")
        with pytest.raises(ValueError, match="unique"):
            ls.validate()

    def test_validate_passes(self):
        ls = RoundState(sequencing_rounds=["r1", "r2"], other_rounds=["p1"], reference_round="r1")
        ls.validate()  # should not raise
