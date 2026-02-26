"""Pytest fixtures for STARfinder tests."""

import json
from pathlib import Path

import pytest

# Pre-generated fixtures path (at repo root /tests/fixtures/)
REPO_ROOT = Path(__file__).parent.parent.parent.parent
FIXTURES_DIR = REPO_ROOT / "tests" / "fixtures" / "synthetic"


@pytest.fixture(scope="session")
def small_dataset() -> Path:
    """Path to pre-generated small synthetic dataset (2 FOVs, unit tests)."""
    path = FIXTURES_DIR / "small"
    if not path.exists():
        pytest.skip(
            "Small synthetic dataset not found. Run: "
            "uv run python -m starfinder.benchmark --preset small --output tests/fixtures/synthetic/small"
        )
    return path


@pytest.fixture(scope="session")
def medium_dataset() -> Path:
    """Path to pre-generated medium synthetic dataset (2 FOVs)."""
    path = FIXTURES_DIR / "medium"
    if not path.exists():
        pytest.skip(
            "Medium synthetic dataset not found. Run: "
            "uv run python -m starfinder.benchmark --preset medium --output tests/fixtures/synthetic/medium"
        )
    return path


@pytest.fixture(scope="session")
def small_ground_truth(small_dataset: Path) -> dict:
    """Load ground truth metadata for small dataset."""
    with open(small_dataset / "ground_truth.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def medium_ground_truth(medium_dataset: Path) -> dict:
    """Load ground truth metadata for medium dataset."""
    with open(medium_dataset / "ground_truth.json") as f:
        return json.load(f)


@pytest.fixture(scope="session")
def small_codebook(small_dataset: Path) -> dict[str, str]:
    """Load codebook for small dataset as gene->barcode dict."""
    import csv

    codebook = {}
    with open(small_dataset / "codebook.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            codebook[row["gene"]] = row["barcode"]
    return codebook


@pytest.fixture(scope="session")
def e2e_result(small_dataset: Path, small_ground_truth: dict, tmp_path_factory):
    """Run full pipeline on small dataset, return (fov, dataset, ground_truth).

    Session-scoped: runs once, shared across all e2e tests.
    Pipeline: load → enhance → global_reg → spot_find →
    reads_extract → reads_filter → save_signal.
    """
    from starfinder.dataset import STARMapDataset
    from starfinder.dataset.types import LayerState

    tmp_path = tmp_path_factory.mktemp("e2e")

    # Restructure: small/{fov}/{round}/ → tmp/{round}/{fov}/
    fov_dir = small_dataset / "FOV_001"
    for round_dir in fov_dir.iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / "FOV_001"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)

    ds = STARMapDataset(
        input_root=tmp_path,
        output_root=tmp_path / "output",
        dataset_id="test",
        sample_id="small",
        output_id="out",
        layers=LayerState(
            seq=["round1", "round2", "round3", "round4"],
            ref="round1",
        ),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
        fov_pattern="FOV_%03d",
    )
    ds.load_codebook(small_dataset / "codebook.csv")

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

    return fov, ds, small_ground_truth
