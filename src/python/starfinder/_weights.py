"""The shared pretrained-weights cache: its root, a model's folder in it and the record file name.

Spot finding (spot_finding.KNOWN_MODELS) and segmentation (segmentation.KNOWN_MODELS) keep their models
in this one cache as ``<root>/<method>/<model>/``, each folder next to a
``starfinder-weights.json`` record once fetched. Each stage keeps its own
artifact table, errors and record content; the file SHA-256 is
:func:`starfinder._records.file_sha256`. Nothing here uses the network: the
download and extraction are in ``_weights_fetch``, which only the fetch
functions import.
"""
import os
from pathlib import Path

RECORD_NAME = "starfinder-weights.json"


def weights_directory(directory=None) -> Path:
    """Root of the weights cache, as an absolute path.

    directory when given; otherwise STARFINDER_WEIGHTS_DIR when set, else
    $XDG_CACHE_HOME/starfinder/weights (default ~/.cache/starfinder/weights).
    A relative or ~ root is expanded and resolved here, so no library ever
    receives a relative path (Piscis would prefix one with its own model
    cache). A model lives in <root>/<method>/<model>/.
    """
    if directory is not None:
        root = Path(directory)
    elif os.environ.get("STARFINDER_WEIGHTS_DIR"):
        root = Path(os.environ["STARFINDER_WEIGHTS_DIR"])
    else:
        root = Path(os.environ.get("XDG_CACHE_HOME") or str(Path.home() / ".cache")) / "starfinder" / "weights"
    return root.expanduser().resolve()


def model_folder(method: str, model: str, directory=None) -> Path:
    """The folder of one model in the cache (it may not exist)."""
    return weights_directory(directory) / method / model
