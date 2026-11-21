"""The line filter of the learned image's ``uv pip check`` (W-878): the known piscis line passes, any other fails."""

import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.workflow, pytest.mark.contract]

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "image_check.py"
HEADER = "Using Python 3.12.16 environment at: /opt/starfinder\nChecked 243 packages in 159ms\n"
KNOWN = "The package `piscis` requires `opencv-python`, but it's not installed"
OTHER = "The package `cellpose` requires `torch>=2`, but `1.13` is installed"


@pytest.fixture(scope="module")
def check():
    sys.path.insert(0, str(SCRIPT.parent))
    try:
        import image_check
    finally:
        sys.path.remove(str(SCRIPT.parent))
    return image_check


def found(*lines):
    count = len(lines)
    return HEADER + f"Found {count} incompatibilit{'y' if count == 1 else 'ies'}\n" + "\n".join(lines) + "\n"


@pytest.mark.parametrize("variant", ["learned", "learned-cuda"])
def test_the_known_line_alone_is_accepted(check, variant):
    assert check.pip_check_problems(found(KNOWN), variant) == []


@pytest.mark.parametrize("variant", ["cpu", "learned", "learned-cuda"])
def test_a_clean_check_is_accepted(check, variant):
    assert check.pip_check_problems(HEADER + "All installed packages are compatible\n", variant) == []


def test_the_exact_line_in_the_text_of_uv_0_9_28_is_the_accepted_one(check):
    # The text of W-870's failed attempt 1 (runs/W-870/20261118T0309Z-tissue2d/scripts/image/w860-diff).
    assert check.ACCEPTED_LINE == KNOWN


@pytest.mark.parametrize("variant", ["learned", "learned-cuda"])
def test_another_line_fails(check, variant):
    problems = check.pip_check_problems(found(OTHER), variant)
    assert problems == [f"finding not accepted: {OTHER}"]


@pytest.mark.parametrize("variant", ["learned", "learned-cuda"])
def test_another_line_beside_the_known_one_fails(check, variant):
    problems = check.pip_check_problems(found(KNOWN, OTHER), variant)
    assert problems == [f"finding not accepted: {OTHER}"]


def test_the_known_line_twice_fails(check):
    problems = check.pip_check_problems(found(KNOWN, KNOWN), "learned")
    assert problems == ["the accepted line appears 2 times, not once"]


@pytest.mark.parametrize(
    "line",
    [
        KNOWN.replace("piscis", "cellpose"),
        KNOWN.replace("`opencv-python`", "`opencv-python-headless`"),
        KNOWN.replace("The package", "the package"),
        KNOWN + ".",
    ],
)
def test_a_line_that_differs_in_text_fails(check, line):
    assert check.pip_check_problems(found(line), "learned")


@pytest.mark.parametrize(
    "line",
    [KNOWN + " ", KNOWN + "\t", " " + KNOWN, "\t" + KNOWN, KNOWN + "\x0c", KNOWN + "\x0b"],
    ids=["trailing space", "trailing tab", "leading space", "leading tab", "form feed", "vertical tab"],
)
def test_the_known_text_with_whitespace_added_is_another_line(check, line):
    assert check.pip_check_problems(found(line), "learned")


@pytest.mark.parametrize("blank", [" ", "\t", "   \t "], ids=["space", "tab", "mixed"])
def test_a_whitespace_only_line_among_the_findings_fails(check, blank):
    text = HEADER + "Found 1 incompatibility\n" + blank + "\n" + KNOWN + "\n"
    assert check.pip_check_problems(text, "learned") == [f"unexpected line: {blank}"]


def test_the_known_line_ending_in_a_carriage_return_and_line_feed_passes(check):
    text = found(KNOWN).replace("\n", "\r\n")
    assert check.pip_check_problems(text, "learned") == []


def test_empty_lines_are_ignored(check):
    text = "\n" + found(KNOWN).replace("\n", "\n\n")
    assert check.pip_check_problems(text, "learned") == []


def test_the_cpu_image_accepts_no_finding(check):
    assert check.pip_check_problems(found(KNOWN), "cpu") == [f"finding not accepted: {KNOWN}"]


def test_a_line_that_is_not_a_finding_fails(check):
    text = found(KNOWN) + "warning: something else\n"
    assert check.pip_check_problems(text, "learned") == ["unexpected line: warning: something else"]


def test_a_count_that_does_not_match_the_findings_fails(check):
    text = HEADER + "Found 2 incompatibilities\n" + KNOWN + "\n"
    assert check.pip_check_problems(text, "learned") == ["'Found 2' does not match the 1 findings listed"]


@pytest.mark.parametrize("text", ["", "\n", "error: uv not found\n"])
def test_output_of_a_check_that_did_not_run_fails(check, text):
    assert check.pip_check_problems(text, "learned")


def test_an_unknown_variant_raises(check):
    with pytest.raises(ValueError, match="variant must be one of"):
        check.pip_check_problems(found(KNOWN), "gpu")


def run(variant, text):
    return subprocess.run(
        [sys.executable, "-B", str(SCRIPT), "pip-check", variant], input=text, capture_output=True, text=True
    )


def test_the_command_exits_0_on_the_known_line_and_1_on_another():
    accepted = run("learned", found(KNOWN))
    assert accepted.returncode == 0
    assert f"accepted by exact text: {KNOWN}" in accepted.stdout
    refused = run("learned", found(KNOWN, OTHER))
    assert refused.returncode == 1
    assert f"uv pip check: finding not accepted: {OTHER}" in refused.stdout
    assert run("cpu", found(KNOWN)).returncode == 1


def test_the_import_check_refuses_to_run_with_a_visible_device(monkeypatch, check):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(AssertionError, match="CUDA_VISIBLE_DEVICES"):
        check.import_versions()
