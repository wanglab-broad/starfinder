from starfinder.preprocessing import MinMaxNormalizationConfig
from starfinder.dataset import RegistrationStep
from starfinder.registration import TranslationConfig
"""Pytest fixtures for STARfinder tests."""

from starfinder.spot_finding import LocalMaximaConfig
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
            "uv run starfinder synthetic generate --mode e2e --owner Jiahao --preset small --seed 42 --output tests/fixtures/synthetic/small"
        )
    return path




@pytest.fixture(scope="session")
def small_ground_truth(small_dataset: Path) -> dict:
    """Load ground truth metadata for small dataset."""
    with open(small_dataset / "ground_truth.json") as f:
        return json.load(f)






@pytest.fixture(scope="session")
def e2e_result(small_dataset: Path, small_ground_truth: dict, tmp_path_factory):
    """Run full pipeline on small dataset, return (fov, dataset, ground_truth).

    Session-scoped: runs once, shared across all e2e tests.
    Pipeline: load → enhance → global_reg → spot_find →
    reads_extract → reads_filter → save_spots.
    """
    from starfinder.dataset import Dataset
    from starfinder.dataset.types import RoundState

    tmp_path = tmp_path_factory.mktemp("e2e")

    # Restructure: small/{fov}/{round}/ → tmp/{round}/{fov}/
    fov_dir = small_dataset / "FOV_001"
    for round_dir in fov_dir.iterdir():
        if round_dir.is_dir():
            target = tmp_path / round_dir.name / "FOV_001"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.symlink_to(round_dir)

    ds = Dataset(
        input_root=tmp_path,
        output_root=tmp_path / "output",
        dataset_id="test",
        sample_id="small",
        output_id="out",
        rounds=RoundState(
            sequencing_rounds=["round1", "round2", "round3", "round4"],
            reference_round="round1",
        ),
        channel_order=["ch00", "ch01", "ch02", "ch03"],
        fov_pattern="FOV_%03d",
    )
    ds.load_codebook(small_dataset / "codebook.csv")

    fov = ds.fov("FOV_001")
    (
        fov.load_images()
        .normalize_intensity(config=MinMaxNormalizationConfig('uint8', (0, 255), snr_threshold=5.0))
        .register(RegistrationStep(TranslationConfig())).find_spots(config=LocalMaximaConfig())
        .extract_intensities()
        .decode_barcodes().filter_reads()
    )
    fov.save_spots(slot="goodSpots")
    fov.save_processing_log()

    return fov, ds, small_ground_truth
