"""The member renames of W-557 (W-526, NC-01, NC-02, NM-16, NM-17 and NM-18): the old names are gone, with no
alias, and the new ones are defined and exported."""
import pytest

import starfinder
from starfinder import readout, segmentation, spot_finding
from starfinder.dataset import FOV

pytestmark = [pytest.mark.readout, pytest.mark.spot_finding, pytest.mark.contract]

READOUT = {"decode_barcodes": "decode_reads", "encode_bases": "bases_to_colors",
           "decode_color_sequence": "colors_to_bases", "BarcodeDecodingResult": "ReadDecodingResult",
           "assign_direct": "decode_direct", "DirectAssignmentConfig": "DirectDecoderConfig"}
SPOT_FINDING = {"resolve_weights": "resolve_model", "fetch_weights": "fetch_model", "KNOWN_WEIGHTS": "KNOWN_MODELS",
                "KnownWeights": "KnownModel", "WeightsFile": "ModelFile", "MissingWeightsError": "MissingModelError",
                "WeightsHashMismatchError": "ModelHashMismatchError"}
CASES = ([(readout, old, new) for old, new in READOUT.items()]
         + [(spot_finding, old, new) for old, new in SPOT_FINDING.items()]
         + [(starfinder, "decode_barcodes", "decode_reads")])


@pytest.mark.parametrize(("module", "old", "new"), CASES, ids=lambda v: getattr(v, "__name__", v))
def test_old_member_names_are_gone_and_the_new_ones_exported(module, old, new):
    assert not hasattr(module, old) and old not in module.__all__
    assert hasattr(module, new) and new in module.__all__


@pytest.mark.parametrize("name", ["decode_reads", "bases_to_colors", "colors_to_bases", "ReadDecodingResult",
                                  "decode_direct", "DirectDecoderConfig"])
def test_readout_members_carry_their_new_name(name):
    assert getattr(readout, name).__name__ == name


@pytest.mark.parametrize("name", ["resolve_model", "fetch_model", "KnownModel", "ModelFile", "MissingModelError",
                                  "ModelHashMismatchError"])
def test_spot_finding_members_carry_their_new_name(name):
    assert getattr(spot_finding, name).__name__ == name


@pytest.mark.parametrize("name", ["KNOWN_MODELS", "resolve_model", "KnownModel", "ModelFile", "MissingModelError",
                                  "ModelHashMismatchError"])
def test_spot_finding_and_segmentation_share_the_names_not_the_definitions(name):
    # NM-18: the two stages keep their own tables, types, resolvers and errors.
    assert getattr(spot_finding, name) is not getattr(segmentation, name)


def test_fov_decodes_with_decode_reads():
    assert not hasattr(FOV, "decode_barcodes")
    assert callable(FOV.decode_reads) and starfinder.decode_reads is readout.decode_reads
