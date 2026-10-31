"""The chunked checkpoint of tiled execution: the registered rounds of one FOV in one HDF5 file, read by window.

design/specs/orchestration-tiled.md, "The chunked checkpoint": one file per FOV, one dataset per pipeline round
(``rounds/<round>``, ZYXC, uncompressed, chunks of (Z, 128, 128, C)) and one per extraction-source snapshot
(``snapshots/<round>/<name>``), each with its ImageMetadata; the file carries the identity, the round order and the
channel labels. The global task writes each round once, as it is produced; a tile task reads its window of each.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from starfinder._records import atomic_path, tuples
from starfinder.image import ImageMetadata
from starfinder.io.tiff import ImageLoadResult

#: The chunk edge in Y and X, fixed by the format (the tiled record's choice A); Z and C are whole in a chunk.
CHUNK_EDGE = 128
#: The format of the file's ``format`` attribute.
CHUNKED_FORMAT = "starfinder.chunked-checkpoint/1"


def array_sha256(image) -> str:
    """``sha256:<hex>`` of an array's bytes in C order (the chunked checkpoint's hash of a round)."""
    data = np.ascontiguousarray(image)
    return "sha256:" + hashlib.sha256(memoryview(data).cast("B")).hexdigest()


def _dataset(handle, round_name, snapshot):
    name = f"rounds/{round_name}" if snapshot is None else f"snapshots/{round_name}/{snapshot}"
    if name not in handle:
        raise KeyError(f"{handle.filename} holds no {name!r}")
    return handle[name]


class ChunkedCheckpointWriter:
    """Write the registered rounds of one FOV into one chunked HDF5 file, one round at a time.

    Parameters
    ----------
    path : str or pathlib.Path
        The file to write; it is written under a temporary sibling name and renamed into place by ``close``.
    identity : dict
        The FOV's identity (``dataset_id``, ``sample_id``, ``fov_id``), stored as the file's ``identity``.
    channel_labels : sequence of str
        The channel labels of every round, in the image's channel order.
    compression : str or None
        None (the default) writes uncompressed chunks; ``"gzip"`` writes gzip level 1 (for a run short of scratch).

    Notes
    -----
    Use as a context manager: a body that raises removes the temporary file and writes nothing. Each round is a
    ZYXC dataset with chunks (Z, c, c, C), ``c`` the smaller of ``CHUNK_EDGE`` and the image's Y and X sizes, in the
    dtype it was produced in, with its ImageMetadata (JSON) and the SHA-256 of its array bytes as attributes.
    """

    def __init__(self, path, *, identity, channel_labels, compression=None):
        if compression not in (None, "gzip"):
            raise ValueError("compression must be None or 'gzip'")
        self.path = Path(path)
        self.identity = dict(identity)
        self.channel_labels = tuple(channel_labels)
        self.compression = compression
        self.rounds = {}
        self._tmp = None
        self._handle = None

    def __enter__(self):
        import h5py
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._tmp = atomic_path(self.path)
        self._handle = h5py.File(self._tmp, "w")
        self._handle.attrs["format"] = CHUNKED_FORMAT
        self._handle.attrs["identity"] = json.dumps(self.identity, sort_keys=True)
        self._handle.attrs["channel_labels"] = json.dumps(list(self.channel_labels))
        self._handle.attrs["chunk_edge"] = CHUNK_EDGE
        return self

    def write_round(self, round_name, image, metadata, snapshots=None) -> dict:
        """Write one round's image and its snapshots; return the round's entry of ``record``.

        Parameters
        ----------
        round_name : str
            The pipeline round.
        image : numpy.ndarray
            The round's ZYXC detection image after the post-registration steps.
        metadata : ImageMetadata
            The round's geometry.
        snapshots : dict or None
            Snapshot name to ZYXC image (the extraction source), in the round's coordinates; None writes none.

        Returns
        -------
        dict
            ``shape_zyxc``, ``dtype``, ``chunk_zyxc``, ``sha256`` and ``snapshots`` (name to SHA-256) of the round.
        """
        if round_name in self.rounds:
            raise ValueError(f"round {round_name!r} is written already")
        image = np.asarray(image)
        if image.ndim != 4 or image.shape[3] != len(self.channel_labels):
            raise ValueError(f"round {round_name!r}: a ZYXC image with {len(self.channel_labels)} channels is "
                             f"required, not shape {image.shape}")
        entry = {**self._write(f"rounds/{round_name}", image, metadata), "snapshots": {}}
        for name, snapshot in (snapshots or {}).items():
            entry["snapshots"][name] = self._write(f"snapshots/{round_name}/{name}", np.asarray(snapshot),
                                                   metadata)["sha256"]
        self.rounds[round_name] = entry
        self._handle.attrs["rounds"] = json.dumps(list(self.rounds))
        return entry

    def _write(self, name, image, metadata):
        z, y, x, c = image.shape
        chunk = (z, min(CHUNK_EDGE, y), min(CHUNK_EDGE, x), c)
        options = {} if self.compression is None else {"compression": "gzip", "compression_opts": 1}
        data = self._handle.create_dataset(name, data=image, chunks=chunk, **options)
        digest = array_sha256(image)
        data.attrs["metadata"] = json.dumps(asdict(metadata))
        data.attrs["sha256"] = digest
        return {"shape_zyxc": list(image.shape), "dtype": str(image.dtype), "chunk_zyxc": list(chunk),
                "sha256": digest}

    def close(self) -> dict:
        """Close the file and rename it into place (the context manager calls it when its body succeeds).

        Returns
        -------
        dict
            ``record``: the written rounds by name.
        """
        self._handle.close()
        self._handle = None
        self._tmp.replace(self.path)
        self._tmp = None
        return self.record

    @property
    def record(self) -> dict:
        """The written rounds by name, each as ``write_round`` returned it."""
        return dict(self.rounds)

    def __exit__(self, kind, error, trace):
        if kind is None:
            self.close()
            return False
        if self._handle is not None:
            self._handle.close()
        if self._tmp is not None:
            self._tmp.unlink(missing_ok=True)
        return False


def read_chunked_record(path) -> dict:
    """The header of a chunked checkpoint: its format, identity, channel labels, rounds and each round's entry.

    Parameters
    ----------
    path : str or pathlib.Path
        The chunked checkpoint (``fovs/<fov>/tiled/registered.h5``).

    Returns
    -------
    dict
        ``format``, ``identity``, ``channel_labels``, ``chunk_edge``, ``rounds`` (in write order) and ``entries``
        (per round: ``shape_zyxc``, ``dtype``, ``chunk_zyxc``, ``sha256``, ``metadata`` and ``snapshots``, snapshot
        name to SHA-256), read from the attributes without reading a pixel.
    """
    import h5py
    with h5py.File(path, "r") as handle:
        if handle.attrs.get("format") != CHUNKED_FORMAT:
            raise ValueError(f"{path} is not a chunked checkpoint of format {CHUNKED_FORMAT}")
        rounds = json.loads(handle.attrs.get("rounds", "[]"))
        entries = {}
        for name in rounds:
            data = handle[f"rounds/{name}"]
            snapshots = handle.get(f"snapshots/{name}")
            entries[name] = {"shape_zyxc": list(data.shape), "dtype": str(data.dtype), "chunk_zyxc": list(data.chunks),
                             "sha256": data.attrs["sha256"], "metadata": json.loads(data.attrs["metadata"]),
                             "snapshots": {} if snapshots is None else
                             {key: snapshots[key].attrs["sha256"] for key in snapshots}}
        return {"format": CHUNKED_FORMAT, "identity": json.loads(handle.attrs["identity"]),
                "channel_labels": json.loads(handle.attrs["channel_labels"]),
                "chunk_edge": int(handle.attrs["chunk_edge"]), "rounds": rounds, "entries": entries}


def chunked_round_sha256(path, round_name, snapshot=None) -> str:
    """The SHA-256 of one stored array's bytes, computed again from the file (not the recorded attribute).

    Parameters
    ----------
    path : str or pathlib.Path
        The chunked checkpoint.
    round_name : str
        The pipeline round.
    snapshot : str or None
        A snapshot of the round; None is the round's detection image.

    Returns
    -------
    str
        ``sha256:<hex>`` of the array's bytes in C order.
    """
    import h5py
    with h5py.File(path, "r") as handle:
        return array_sha256(_dataset(handle, round_name, snapshot)[...])


def read_window(path, round_name, window_yx, *, snapshot=None, frame_id=None) -> ImageLoadResult:
    """One round's window of a chunked checkpoint: the ZYXC crop and its geometry.

    Parameters
    ----------
    path : str or pathlib.Path
        The chunked checkpoint.
    round_name : str
        The pipeline round.
    window_yx : sequence of int
        ``(y0, y1, x0, x1)``, half-open, in the FOV's Y and X indices; Z and the channels are whole.
    snapshot : str or None
        A snapshot of the round (the extraction source); None reads the round's detection image.
    frame_id : str or None
        The frame id of the window's geometry; None is the round's frame id with ``/window:<y0>,<x0>``.

    Returns
    -------
    ImageLoadResult
        The window (only the chunks it overlaps are read), its ImageMetadata (the round's, cropped at the window's
        start, ``ImageMetadata.cropped``), the file's channel labels, the file as source and the window in
        ``diagnostics``.
    """
    import h5py
    y0, y1, x0, x1 = (int(v) for v in window_yx)
    with h5py.File(path, "r") as handle:
        data = _dataset(handle, round_name, snapshot)
        _, height, width, _ = data.shape
        if not (0 <= y0 < y1 <= height and 0 <= x0 < x1 <= width):
            raise ValueError(f"window {(y0, y1, x0, x1)} does not lie in the round's Y and X ({height}, {width})")
        image = data[:, y0:y1, x0:x1, :]
        source = ImageMetadata(**tuples(json.loads(data.attrs["metadata"])))
        labels = tuple(json.loads(handle.attrs["channel_labels"]))
    metadata = source.cropped((0, y0, x0), frame_id=frame_id or f"{source.frame_id}/window:{y0},{x0}")
    return ImageLoadResult(image, metadata, labels, (Path(path),), {"window_yx": [y0, y1, x0, x1]})
