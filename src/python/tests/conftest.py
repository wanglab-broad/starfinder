"""Pytest fixtures for STARfinder tests."""

import json
from pathlib import Path

import pytest

# Pre-generated fixtures path (relative to repo root)
REPO_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def mini_dataset() -> Path:
    """Path to pre-generated mini synthetic dataset (1 FOV, fast tests)."""
    path = FIXTURES_DIR / "mini"
    if not path.exists():
        pytest.skip(
            "Mini synthetic dataset not found. Run: "
            "uv run python -m starfinder.testing --preset mini --output tests/fixtures/synthetic/mini"
        )
    return path


@pytest.fixture(scope="session")
def standard_dataset() -> Path:
    """Path to pre-generated standard synthetic dataset (4 FOVs)."""
    path = FIXTURES_DIR / "standard"
    if not path.exists():
        pytest.skip(
            "Standard synthetic dataset not found. Run: "
            "uv run python -m starfinder.testing --preset standard --output tests/fixtures/synthetic/standard"
        )
    return path


@pytest.fixture(scope="session")
def mini_ground_truth(mini_dataset: Path) -> dict:
    """Load ground truth metadata for mini dataset."""
    with open(mini_dataset / "ground_truth.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def standard_ground_truth(standard_dataset: Path) -> dict:
    """Load ground truth metadata for standard dataset."""
    with open(standard_dataset / "ground_truth.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def mini_codebook(mini_dataset: Path) -> dict[str, str]:
    """Load codebook for mini dataset as gene->barcode dict."""
    import csv

    codebook = {}
    with open(mini_dataset / "codebook.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codebook[row["gene"]] = row["barcode"]
    return codebook
