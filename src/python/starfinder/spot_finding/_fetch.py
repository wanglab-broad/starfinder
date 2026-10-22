"""Explicit weights download (starfinder weights fetch); never imported by a detection."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile

from starfinder._records import file_sha256
from starfinder._weights import RECORD_NAME

from ._errors import WeightsHashMismatchError
from ._weights import _listed, listed_files, resolve_weights


def _record(entry, folder):
    files = {p.relative_to(folder).as_posix(): file_sha256(p)
             for p in sorted(folder.rglob("*")) if p.is_file() and p.name != RECORD_NAME}
    return {"method": entry.method, "model": entry.model, "url": entry.url, "revision": entry.revision,
            "sha256": entry.sha256, "bytes": entry.bytes, "md5": entry.md5,
            "fetched_at": datetime.now(timezone.utc).isoformat(), "files": files}


def _checked(entry):
    """The files of entry that detection and verify check: every file KNOWN_WEIGHTS lists for the model."""
    return _listed(entry, listed_files(entry.method, entry.model))


def _stage(entry, staging):
    """Download entry into staging, verify the download and every listed file in it; return the model folder."""
    from starfinder._weights_fetch import _download, _extract
    download = staging / "download"
    size, sha256, md5 = _download(entry.url, download)
    if size != entry.bytes or sha256 != entry.sha256:
        raise WeightsHashMismatchError(
            f"{entry.url}: downloaded {size} bytes with SHA-256 {sha256}, expected {entry.bytes} bytes with "
            f"SHA-256 {entry.sha256}")
    if entry.md5 is not None and md5 != entry.md5:
        raise WeightsHashMismatchError(f"{entry.url}: MD5 {md5}, expected the library's {entry.md5}")
    if entry.archive:
        extracted = _extract(download, staging / "extracted")
    else:
        extracted = staging / "extracted"
        extracted.mkdir()
        os.replace(download, extracted / entry.files[0].path)
    for item in _checked(entry):
        path = extracted / item.path
        if not path.is_file():
            raise WeightsHashMismatchError(f"{entry.url}: the download has no {item.path}")
        actual = file_sha256(path)
        if actual != item.sha256:
            raise WeightsHashMismatchError(f"{entry.url}: {item.path} has SHA-256 {actual}, expected {item.sha256}")
    return extracted


def fetch(entry, root):
    """Install entry under root/<method>/<model>/ after verifying the download; see fetch_weights.

    An existing folder is checked file by file against every listed file. A
    present file that differs raises WeightsHashMismatchError and a folder
    without a record raises FileExistsError; neither is changed. When every
    present file verifies, the missing listed files, if any, are restored from
    one verified download and the record is rewritten. A present file is never
    overwritten.
    """
    parent = Path(root) / entry.method
    folder = parent / entry.model
    missing = []
    if folder.exists():
        # A verified copy is kept and a changed one is left for the user to inspect.
        for item in _checked(entry):
            path = folder / item.path
            if not path.is_file():
                missing.append(item)
                continue
            actual = file_sha256(path)
            if actual != item.sha256:
                raise WeightsHashMismatchError(
                    f"{entry.method} model {entry.model!r}: {path} has SHA-256 {actual}, expected {item.sha256}; "
                    f"fetch never overwrites a present file, so remove {folder} and fetch again")
        if not (folder / RECORD_NAME).is_file():
            raise FileExistsError(f"{folder} exists without {RECORD_NAME}; remove it and fetch again")
        if not missing:
            return folder
    parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{entry.model}.", suffix=".tmp", dir=parent))
    try:
        extracted = _stage(entry, staging)
        if missing:
            # Restore only the missing listed files; the present ones verified above stay untouched.
            for item in missing:
                (folder / item.path).parent.mkdir(parents=True, exist_ok=True)
                os.replace(extracted / item.path, folder / item.path)
            record = _record(entry, folder)
            (staging / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
            os.replace(staging / RECORD_NAME, folder / RECORD_NAME)
        else:
            record = _record(entry, extracted)
            (extracted / RECORD_NAME).write_text(json.dumps(record, indent=2) + "\n")
            os.replace(extracted, folder)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return resolve_weights(entry.method, entry.model, directory=root,
                           extracted=listed_files(entry.method, entry.model))
