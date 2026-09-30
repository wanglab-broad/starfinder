"""Other-round and external-reference registration from a shared stain (registration task group 5).

Three uint16 rounds of 8x32x32 are generated in session from continuous
textures (sums of Gaussian blobs), so every image is known analytically: the
reference round1 with the stain in ch04, the sequencing round2, and the other
round morph with the stain in ch00 and two further channels. morph is the
truth displaced by d = (0, 3, -2): morph(q) = truth(q - d), so the pull map is
p + d and the translation correction is -d = (0, -3, 2). The local fixture adds
a Gaussian displacement of magnitude 2 voxels to that pull map.
"""
import hashlib
import json
import runpy
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import tifffile

from starfinder.dataset import Dataset, ExternalReference, RegistrationRecipe, RegistrationStep, RoundState
from starfinder.evaluation.registration import _valid_overlap, evaluate_displacement_field
from starfinder.image import ImageMetadata, IncompatibleGeometryError
from starfinder.io import save_volume
from starfinder.registration import (DemonsConfig, RegistrationSignalConfig, TranslationConfig, WarpConfig,
                                     apply_transform)
from starfinder import registration

ROOT = Path(__file__).resolve().parents[3]
SHAPE = (8, 32, 32)
SHIFT = np.array([0.0, 3.0, -2.0])
SEQUENCING = ("ch01", "ch02", "ch03", "ch04")
OTHER = ("ch00", "ch01", "ch02")
STAIN = RegistrationSignalConfig("channel", reference_channel="ch04", moving_channel="ch00")
TRANSLATION = RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=STAIN)
# The valid overlap of the pull map p + (0, 3, -2): y + 3 <= 31 and x - 2 >= 0.
OVERLAP = (slice(None), slice(0, 29), slice(2, 32))
# The local displacement: magnitude 2 voxels along (0, 1, 1)/sqrt(2), scale 8 voxels.
BUMP_CENTER, BUMP_DIRECTION = np.array([3.5, 12.4, 18.6]), np.array([0.0, 1.0, 1.0]) / np.sqrt(2)


@pytest.fixture(autouse=True, scope="module")
def one_thread():
    """SimpleITK at one thread, as the project contract requires."""
    sitk = pytest.importorskip("SimpleITK")
    sitk.ProcessObject.SetGlobalDefaultNumberOfThreads(1)


def texture(seed, points):
    """A continuous texture of 400 Gaussian blobs evaluated at ZYX points (3, ...), rounded for uint16."""
    rng = np.random.default_rng([257, seed])
    centers = rng.uniform([-2, -6, -6], [10, 38, 38], (400, 3))
    amplitudes = rng.uniform(500, 3000, 400)
    out = np.full(points.shape[1:], 100.0)
    for center, amplitude in zip(centers, amplitudes):
        r2 = ((points[0] - center[0]) / 1.5) ** 2 + ((points[1] - center[1]) / 2) ** 2 + ((points[2] - center[2]) / 2) ** 2
        out += amplitude * np.exp(-r2 / 2)
    return np.rint(out).astype(np.uint16)


def grid():
    return np.stack(np.meshgrid(*(np.arange(n, dtype=np.float64) for n in SHAPE), indexing="ij"))


def bump(points):
    """The local displacement at ZYX points (3, ...)."""
    r2 = sum((points[i] - BUMP_CENTER[i]) ** 2 for i in range(3)) / (2 * 8.0 ** 2)
    return 2.0 * np.exp(-r2)[None] * BUMP_DIRECTION.reshape(3, 1, 1, 1)


def rounds(local=False):
    """(reference round1, sequencing round2, other round morph, morph truth, truth pull field).

    Stain texture 0 is ch04 of round1 and ch00 of morph; textures 4 and 5 are
    morph's ch01 and ch02. morph(q) = truth(Phi^-1(q)) with Phi(p) = p + d
    (+ the bump when local); the inverse is found by fixed-point iteration.
    """
    p = grid()
    shift = SHIFT.reshape(3, 1, 1, 1)
    source = p - shift
    if local:
        for _ in range(60):
            source = p - shift - bump(source)
    reference = np.stack([texture(s, p) for s in (1, 2, 3, 0)], axis=-1)
    sequencing = np.stack([texture(s, p) for s in (6, 7, 8, 9)], axis=-1)
    morph = np.stack([texture(s, source) for s in (0, 4, 5)], axis=-1)
    truth = np.stack([texture(s, p) for s in (0, 4, 5)], axis=-1)
    field = shift + (bump(p) if local else 0)
    return reference, sequencing, morph, truth, np.moveaxis(np.broadcast_to(field, p.shape), 0, -1)


def other_round_fov(tmp_path, local=False):
    reference, sequencing, morph, truth, field = rounds(local)
    dataset = Dataset(tmp_path, tmp_path / "output", "dataset", "sample", "output",
                      RoundState(["round1", "round2"], ["morph"], "round1"), SEQUENCING,
                      other_channel_order={"morph": OTHER})
    fov = dataset.fov("FOV")
    fov.images = {"round1": reference, "round2": sequencing, "morph": morph}
    fov.metadata = {name: ImageMetadata(f"FOV/{name}") for name in fov.images}
    return fov, truth, field


def external(fov):
    return ExternalReference(image=fov.images["round1"][..., 3], metadata=fov.metadata["round1"], label="ref_round:ch04")


@pytest.fixture
def spy(monkeypatch):
    """Counts estimate_transform calls made through the registration module."""
    calls = []
    original = registration.estimate_transform

    def counting(*args, **kwargs):
        calls.append(kwargs.get("config"))
        return original(*args, **kwargs)
    monkeypatch.setattr(registration, "estimate_transform", counting)
    return calls


def test_a_shared_stain_registers_an_other_round_and_transfers_the_transform(tmp_path):
    fov, truth, _ = other_round_fov(tmp_path)
    before = {name: fov.images[name].copy() for name in ("round1", "round2")}
    fov.register_rounds(TRANSLATION, rounds=["morph"])
    assert fov.registration_results["morph"][0].transform.correction_zyx == (0.0, -3.0, 2.0)
    registered = fov.images["morph"]
    assert registered.dtype == np.uint16 and registered.shape == SHAPE + (3,)
    for c in range(3):
        assert np.array_equal(registered[OVERLAP + (c,)], truth[OVERLAP + (c,)])
    for name, image in before.items():
        assert np.array_equal(fov.images[name], image)
    assert list(fov.registration_results) == ["morph"]


def test_an_external_reference_gives_the_same_result(tmp_path):
    by_round, _, _ = other_round_fov(tmp_path)
    by_round.register_rounds(TRANSLATION, rounds=["morph"])
    fov, _, _ = other_round_fov(tmp_path)
    reference = external(fov)
    fov.register_rounds(TRANSLATION, rounds=["morph"], reference=reference)
    expected, actual = by_round.registration_chains["morph"], fov.registration_chains["morph"]
    assert [t.correction_zyx for t in actual.transforms] == [t.correction_zyx for t in expected.transforms]
    assert actual.transforms == expected.transforms
    assert np.array_equal(actual.pull_field().displacement_zyx, expected.pull_field().displacement_zyx)
    assert np.array_equal(fov.images["morph"], by_round.images["morph"])
    digest = hashlib.sha256(np.ascontiguousarray(reference.image).tobytes()).hexdigest()
    estimation = fov.registration_attempts["morph"][0]
    assert estimation["reference"] == "ref_round:ch04" and estimation["reference_sha256"] == digest
    assert fov.registration_record["rounds"]["morph"]["reference_sha256"] == digest


@pytest.mark.parametrize("signal, names", [
    (RegistrationSignalConfig("channel", "ch04", "ch09"), ["morph"]),      # unknown label in morph
    (RegistrationSignalConfig("channel", "ch04", "ch00"), ["morph", "round2"]),  # ch00 is not a round2 label
    (RegistrationSignalConfig("channel", "ch04", 3), ["morph"]),           # index outside morph's 3 channels
])
def test_label_errors_are_raised_before_estimation(tmp_path, spy, signal, names):
    fov, _, _ = other_round_fov(tmp_path)
    moving = {name: fov.images[name].copy() for name in names}
    with pytest.raises(ValueError) as error:
        fov.register_rounds(RegistrationRecipe((RegistrationStep(TranslationConfig()),), signal=signal), rounds=names)
    assert type(error.value) is ValueError
    assert spy == [] and fov.registration_attempts == {} and fov.registration_results == {}
    assert all(np.array_equal(fov.images[name], image) for name, image in moving.items())


@pytest.mark.parametrize("image, metadata", [
    (np.zeros((8, 32, 31), dtype=np.uint16), None),                                  # different shape
    (None, ImageMetadata("atlas", spacing_zyx=(2.0, 0.5, 0.5))),                      # different spacing
])
def test_grid_errors_are_raised_before_estimation(tmp_path, spy, image, metadata):
    fov, _, _ = other_round_fov(tmp_path)
    reference = ExternalReference(fov.images["round1"][..., 3] if image is None else image,
                                  metadata or fov.metadata["round1"], "atlas")
    with pytest.raises(IncompatibleGeometryError):
        fov.register_rounds(TRANSLATION, rounds=["morph"], reference=reference)
    assert spy == [] and fov.registration_attempts == {}


def test_local_and_composed_recipes_work_for_other_rounds(tmp_path, spy):
    fov, _, truth = other_round_fov(tmp_path, local=True)
    original = fov.images["morph"].copy()
    recipe = RegistrationRecipe((RegistrationStep(TranslationConfig()), RegistrationStep(DemonsConfig())), signal=STAIN)
    fov.register_rounds(recipe, rounds=["morph"])
    assert [type(config) for config in spy] == [TranslationConfig, DemonsConfig]
    chain = fov.registration_chains["morph"]
    u = chain.pull_field().displacement_zyx
    errors = evaluate_displacement_field(u, truth, mask=_valid_overlap(u, SHAPE)).values
    assert errors["median_error"] <= 0.5 and errors["p95_error"] <= 1.0, errors
    once = apply_transform(original, chain, config=WarpConfig(backend="scipy"))
    for c in range(3):
        assert np.array_equal(fov.images["morph"][..., c], once[..., c])
    attempts = fov.registration_attempts["morph"]
    assert [(a["record"], a.get("actual_method"), a.get("backend")) for a in attempts] == [
        ("estimation", "translation", "scipy_fft"), ("estimation", "demons", "simpleitk"), ("application", None, None)]
    assert all(a["reference"] == "round1" and a["reference_sha256"] is None for a in attempts[:2])


@pytest.mark.parametrize("use_external", [False, True])
def test_records_name_the_reference(tmp_path, use_external):
    fov, _, _ = other_round_fov(tmp_path)
    reference = external(fov) if use_external else None
    fov.register_rounds(TRANSLATION, rounds=["morph"], reference=reference)
    estimation, application = fov.registration_attempts["morph"]
    digest = hashlib.sha256(np.ascontiguousarray(fov.images["round1"][..., 3]).tobytes()).hexdigest()
    assert {k: estimation[k] for k in ("record", "step", "attempt", "requested_method", "actual_method", "fallback",
                                       "backend", "reference", "reference_sha256", "outcome")} == dict(
        record="estimation", step=0, attempt=0, requested_method="translation", actual_method="translation",
        fallback=False, backend="scipy_fft", reference="ref_round:ch04" if use_external else "round1",
        reference_sha256=digest if use_external else None, outcome="succeeded")
    assert application["record"] == "application" and application["outcome"] == "succeeded"
    assert fov.registration_record["rounds"]["morph"]["recipe"]["steps"] == ["translation"]
    log = fov.save_processing_log("nr")
    assert log == tmp_path / "output" / "log" / "FOV_nr.txt"
    assert list(json.loads(log.read_text())["registration_attempts"]) == ["morph"]
    shifts = pd.read_csv(tmp_path / "output" / "log" / "gr_shifts" / "FOV_nr.txt")
    assert list(shifts.columns) == ["fov_id", "round", "row", "col", "z"]
    assert shifts.to_dict("records") == [dict(fov_id="FOV", round="morph", row=3, col=-2, z=0)]


def rule_block(path):
    """Input and output lines of the nuclei_registration rule of a Snakemake file."""
    text = path.read_text()
    block = text[text.index("rule nuclei_registration:"):]
    block = block[:block.index("    resources:")]
    return block


@pytest.mark.parametrize("maximum_projection", [False, True])
def test_the_python_workflow_rule_mirrors_the_matlab_outputs(tmp_path, maximum_projection):
    reference, sequencing, morph, truth, _ = rounds()
    config = dict(starfinder_path=str(ROOT), root_input_path=str(tmp_path / "input"), dataset_id="dataset",
                  sample_id="sample", root_output_path=str(tmp_path / "output"), output_id="output",
                  fov_id_pattern="FOV%d", n_rounds=2, ref_round="round1", seq_channel_order=list(SEQUENCING),
                  ref_channel="DAPI", rotate_angle=90, maximum_projection=maximum_projection, backend="python",
                  additional_round=[dict(round_name="morph", channel_order=[
                      dict(wavelength=405, channel="ch00", name="DAPI"), dict(wavelength=488, channel="ch01", name="Nissl"),
                      dict(wavelength=647, channel="ch02", name="GFAP")])],
                  rules=dict(nuclei_registration=dict(run=True)))
    # Inputs are written unrotated; the rule rotates every image by rotate_angle = 90 degrees.
    source = tmp_path / "input" / "dataset" / "sample"
    for name, image, labels in (("round1", reference, SEQUENCING), ("round2", sequencing, SEQUENCING),
                                ("morph", morph, OTHER)):
        for c, label in enumerate(labels):
            save_volume(np.rot90(image[..., c], k=-1, axes=(1, 2)), source / name / "FOV" / f"FOV_{label}.tif")
    snakemake = SimpleNamespace(config=config, wildcards=SimpleNamespace(fovID="FOV"),
                                input=[str(tmp_path / "config.json"), str(source / "morph" / "FOV")],
                                output=[str(tmp_path / "output" / "dataset" / "output" / "log" / "FOV_nr.txt"),
                                        str(tmp_path / "output" / "dataset" / "output" / "log" / "gr_shifts" / "FOV_nr.txt")])
    runpy.run_path(str(ROOT / "workflow" / "scripts" / "nuclei_registration.py"), init_globals={"snakemake": snakemake})

    output = tmp_path / "output" / "dataset" / "output"
    names = sorted(p.relative_to(output).as_posix() for p in output.rglob("*") if p.is_file())
    assert names == ["images/morph/DAPI/FOV.tif", "images/morph/GFAP/FOV.tif", "images/morph/Nissl/FOV.tif",
                     "log/FOV_nr.txt", "log/gr_shifts/FOV_nr.txt"]
    assert all(Path(path).is_file() for path in snakemake.output)
    shifts = pd.read_csv(output / "log" / "gr_shifts" / "FOV_nr.txt")
    assert shifts.to_dict("records") == [dict(fov_id="FOV", round="morph", row=3, col=-2, z=0)]
    for c, folder in enumerate(("DAPI", "Nissl", "GFAP")):
        saved = tifffile.imread(output / "images" / "morph" / folder / "FOV.tif")
        assert saved.dtype == np.uint16
        if maximum_projection:
            assert saved.shape == SHAPE[1:]
        else:
            assert np.array_equal(saved[OVERLAP], truth[OVERLAP + (c,)])
    # The Python rule declares the MATLAB rule's inputs and outputs and runs the Python script.
    python_rule, matlab_rule = (ROOT / "workflow" / "rules" / "registration-py.smk",
                                ROOT / "workflow" / "rules" / "registration.smk")
    assert rule_block(python_rule) == rule_block(matlab_rule)
    assert '"../scripts/nuclei_registration.py"' in python_rule.read_text()
    assert "matlab_script_name = 'nuclei_registration'" in matlab_rule.read_text()
