"""Pytest fixtures for STARfinder tests."""

import json
from pathlib import Path

import pytest

# Pre-generated fixtures path (at repo root /tests/fixtures/)
REPO_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def mini_dataset() -> Path:
    """Path to pre-generated mini synthetic dataset (1 FOV, fast tests)."""
    path = FIXTURES_DIR / "mini"
    if not path.exists():
        pytest.skip(
            "Mini synthetic dataset not found. Run: "
            "uv run python -m starfinder.testdata --preset mini --output tests/fixtures/synthetic/mini"
        )
    return path


@pytest.fixture(scope="session")
def standard_dataset() -> Path:
    """Path to pre-generated standard synthetic dataset (4 FOVs)."""
    path = FIXTURES_DIR / "standard"
    if not path.exists():
        pytest.skip(
            "Standard synthetic dataset not found. Run: "
            "uv run python -m starfinder.testdata --preset standard --output tests/fixtures/synthetic/standard"
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


@pytest.fixture(scope="session")
def e2e_result(mini_dataset: Path, mini_ground_truth: dict, tmp_path_factory):
    """Run full pipeline on mini dataset, return (fov, dataset, ground_truth).

    Session-scoped: runs once, shared across all e2e tests.
    Pipeline: load → enhance → global_reg → spot_find →
    reads_extract → reads_filter → save_signal.
    """
    from starfinder.dataset import STARMapDataset
    from starfinder.dataset.types import LayerState

    tmp_path = tmp_path_factory.mktemp("e2e")

    # Restructure: mini/{fov}/{round}/ → tmp/{round}/{fov}/
    fov_dir = mini_dataset / "FOV_001"
    for round_dir in fov_dir.iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / "FOV_001"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)

    ds = STARMapDataset(
        input_root=tmp_path,
        output_root=tmp_path / "output",
        dataset_id="test",
        sample_id="mini",
        output_id="out",
        layers=LayerState(
            seq=["round1", "round2", "round3", "round4"],
            ref="round1",
        ),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
        fov_pattern="FOV_%03d",
    )
    ds.load_codebook(mini_dataset / "codebook.csv")

    fov = ds.fov("FOV_001")
    (
        fov.load_raw_images()
        .enhance_contrast(snr_threshold=5.0)
        .global_registration()
        .spot_finding()
        .reads_extraction()
        .reads_filtration()
    )
    fov.save_signal(slot="goodSpots")

    return fov, ds, mini_ground_truth
