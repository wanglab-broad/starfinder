"""Keep the independent W-154 specification examples executable."""

from pathlib import Path
import runpy


def test_artifact_contract_examples():
    example = Path(__file__).resolve().parents[3] / "docs/examples/artifact_contracts.py"
    runpy.run_path(str(example), run_name="__main__")
