"""The routing of FOV.register by the rounds' category (W-560, NC-05: the other-round entry merged into ``register``).

One row per line of the routing table: the default (the sequencing rounds other than the reference),
explicit sequencing rounds with ``source``, other rounds with ``reference`` and ``checkpoints``, both
categories in one call, an empty sequence, and a parameter of the other route. Every refusal is a
ValueError raised before any estimator call, with nothing recorded and no image changed. The FOV is
``other_round_fov`` of test_registration_other_rounds.py (8x32x32 uint16 rounds from continuous
textures); round2 is replaced by circular shifts of round1, so that its translation is known.
"""
import re

import numpy as np
import pytest

from starfinder.dataset import CheckpointConfig
from starfinder.registration import RegistrationRecipe, RegistrationStep, TranslationConfig

from .test_registration_other_rounds import TRANSLATION, external, other_round_fov, spy  # noqa: F401

pytestmark = pytest.mark.registration

# The sequencing route registers round2 on the maximum over its channels (round2 has no ch00, TRANSLATION's stain).
MAX = RegistrationRecipe((RegistrationStep(TranslationConfig()),))
SEQUENCING_SHIFT = (0, 1, -1)
SOURCE_SHIFT = (0, 2, 1)


def sequencing_fov(tmp_path, shift=SEQUENCING_SHIFT):
    """other_round_fov with round2 = round1 shifted circularly by shift (moved(q) = round1(q - shift))."""
    fov, _, _ = other_round_fov(tmp_path)
    fov.images["round2"] = np.ascontiguousarray(np.roll(fov.images["round1"], shift, axis=(0, 1, 2)))
    return fov


def untouched(fov, before):
    """Nothing registered, recorded or resampled."""
    assert fov.registration_attempts == {} and fov.registration_results == {} and fov.registration_chains == {}
    assert "recipe" not in fov.registration_record and "rounds" not in fov.registration_record
    assert all(np.array_equal(fov.images[name], image) for name, image in before.items())


def test_the_default_registers_the_sequencing_rounds_on_the_sequencing_route(tmp_path, spy):
    fov = sequencing_fov(tmp_path)
    morph = fov.images["morph"].copy()
    assert fov.register(MAX) is fov
    assert len(spy) == 1
    assert list(fov.registration_results) == list(fov.registration_chains) == ["round2"]
    assert np.allclose(fov.registration_results["round2"][0].transform.displacement_zyx, SEQUENCING_SHIFT, atol=1e-9)
    # The sequencing route keeps one recipe per FOV, and no per-round entry.
    assert fov.registration_record["recipe"]["steps"] == ["translation"] and "rounds" not in fov.registration_record
    assert np.array_equal(fov.images["morph"], morph)


def test_explicit_sequencing_rounds_register_their_source_snapshot(tmp_path, spy):
    fov = sequencing_fov(tmp_path, shift=(0, 0, 0))
    shifted = np.ascontiguousarray(np.roll(fov.images["round1"], SOURCE_SHIFT, axis=(0, 1, 2)))
    fov.snapshots = {"round1": {"bg": fov.images["round1"].copy()}, "round2": {"bg": shifted}}
    fov.register(MAX, rounds=["round2"], source="bg")
    assert len(spy) == 1 and list(fov.registration_results) == ["round2"]
    chain = fov.registration_chains["round2"]
    assert np.allclose(chain.transforms[0].displacement_zyx, SOURCE_SHIFT, atol=1e-9)
    # The same chain as registering an image equal to the snapshot without a source.
    plain = sequencing_fov(tmp_path, shift=SOURCE_SHIFT).register(MAX, rounds=["round2"])
    assert chain.transforms == plain.registration_chains["round2"].transforms
    assert fov.registration_record["recipe"] == plain.registration_record["recipe"]
    assert "rounds" not in fov.registration_record


def test_other_rounds_register_to_a_reference_with_checkpoints_on_the_other_round_route(tmp_path, spy):
    fov, _, _ = other_round_fov(tmp_path)
    sequencing = fov.images["round2"].copy()
    checkpoints = CheckpointConfig(directory=tmp_path / "ck")
    fov.register(TRANSLATION, rounds=["morph"], reference=external(fov), checkpoints=checkpoints)
    assert len(spy) == 1 and list(fov.registration_results) == ["morph"]
    assert fov.registration_results["morph"][0].transform.displacement_zyx == (0.0, 3.0, -2.0)
    # The other-round route keeps one recipe per round, with its reference, and no FOV recipe.
    entry = fov.registration_record["rounds"]["morph"]
    assert entry["reference"] == "ref_round:ch04" and entry["recipe"]["steps"] == ["translation"]
    assert "recipe" not in fov.registration_record
    saved = tmp_path / "ck" / "FOV" / "other_rounds" / "morph"
    assert sorted(p.name for p in saved.iterdir()) == ["image.ome.tif", "registration.json"]
    assert np.array_equal(fov.images["round2"], sequencing)


def test_the_step_name_of_both_routes_is_register(tmp_path, caplog):
    fov = sequencing_fov(tmp_path)
    with caplog.at_level("INFO", logger="starfinder"):
        fov.register(MAX, rounds=["round2"]).register(TRANSLATION, rounds=["morph"])
    started = [r.getMessage() for r in caplog.records if "Starting" in r.getMessage()]
    assert started == ["[FOV] Starting register", "[FOV] Starting register"]


@pytest.mark.parametrize("names, mixed", [
    (["round2", "morph"], (["round2"], ["morph"])),
    (["morph", "round2"], (["round2"], ["morph"])),
    (["round2", "atlas"], (["round2"], ["atlas"])),   # atlas: a loaded round the dataset does not configure
])
def test_sequencing_and_other_rounds_in_one_call_are_refused_before_estimation(tmp_path, spy, names, mixed):
    fov = sequencing_fov(tmp_path)
    fov.images["atlas"], fov.metadata["atlas"] = fov.images["morph"].copy(), fov.metadata["morph"]
    before = {name: fov.images[name].copy() for name in fov.images}
    with pytest.raises(ValueError, match=re.escape(f"rounds {mixed[0]} are sequencing rounds and rounds {mixed[1]} "
                                                   "are not; register the sequencing rounds and the other rounds")):
        fov.register(MAX, rounds=names)
    assert spy == []
    untouched(fov, before)


@pytest.mark.parametrize("rounds", [[], ()])
def test_an_empty_sequence_is_refused_before_estimation(tmp_path, spy, rounds):
    fov = sequencing_fov(tmp_path)
    before = {name: fov.images[name].copy() for name in fov.images}
    with pytest.raises(ValueError, match="rounds must be a nonempty sequence of round names"):
        fov.register(MAX, rounds=rounds)
    assert spy == []
    untouched(fov, before)


@pytest.mark.parametrize("rounds, kwargs, message", [
    (None, dict(reference="round1"), r"reference must be None for the sequencing rounds \['round2'\]"),
    (["round2"], dict(reference="round1"), r"reference must be None for the sequencing rounds \['round2'\]"),
    (["round2"], dict(checkpoints="ck"), r"checkpoints must be None for the sequencing rounds \['round2'\]"),
    (["round2"], dict(reference="round1", checkpoints="ck"),
     r"reference and checkpoints must be None for the sequencing rounds \['round2'\]"),
    (["morph"], dict(source="bg"), r"source must be None for the other rounds \['morph'\]"),
])
def test_a_parameter_of_the_other_route_is_refused_before_estimation(tmp_path, spy, rounds, kwargs, message):
    fov = sequencing_fov(tmp_path)
    fov.snapshots = {name: {"bg": fov.images[name].copy()} for name in fov.images}
    if "checkpoints" in kwargs:
        kwargs = dict(kwargs, checkpoints=CheckpointConfig(directory=tmp_path / kwargs["checkpoints"]))
    before = {name: fov.images[name].copy() for name in fov.images}
    with pytest.raises(ValueError, match=message):
        fov.register(MAX if rounds != ["morph"] else TRANSLATION, rounds=rounds, **kwargs)
    assert spy == []
    untouched(fov, before)
    assert not (tmp_path / "ck").exists()
