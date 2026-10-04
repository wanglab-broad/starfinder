"""Source revision record of a benchmark run: the commit and a checksum of uncommitted changes (W-288).

The one definition behind the ``_revision()`` of image_statistics.py and
preprocessing_synthetic.py, which load this file by its location so that it works
both when they run as scripts and when they are loaded by file location.
"""
import hashlib
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
UNCOMMITTED_DIFF = ("sha256 of `git diff HEAD --binary` followed by each untracked path in sorted order: its name, a "
                    "NUL byte and the file's bytes; an untracked directory entry (such as a nested Git repository) "
                    "contributes its name and the NUL byte only, not the files inside it")


def revision(root=ROOT):
    """HEAD, dirty state, the uncommitted-diff checksum and the untracked paths of the checkout at root."""
    root = Path(root)
    try:
        head = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        dirty = bool(subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True,
                                    text=True).stdout.strip())
        diff = subprocess.run(["git", "-C", str(root), "diff", "HEAD", "--binary"], capture_output=True).stdout
        listing = subprocess.run(["git", "-C", str(root), "ls-files", "-z", "--others", "--exclude-standard"],
                                 capture_output=True).stdout
    except OSError:
        return {"revision": None, "dirty": None}
    untracked = sorted(os.fsdecode(name) for name in listing.split(b"\0") if name)
    digest = hashlib.sha256(diff)
    for name in untracked:
        path = root / name
        digest.update(os.fsencode(name) + b"\0" + (b"" if path.is_dir() else path.read_bytes()))
    return {"revision": head, "dirty": dirty, "uncommitted_diff_sha256": digest.hexdigest() if dirty else None,
            "uncommitted_diff": UNCOMMITTED_DIFF, "untracked_files": untracked}
