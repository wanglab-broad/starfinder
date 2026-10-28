"""The record ``export.json`` of the export folder.

design/specs/sample-output.md ("When a part cannot run", "How the folder is kept equal to its record").
"""
from __future__ import annotations

import hashlib
import json
import resource
import shutil
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from starfinder._layout import EXPORT_FILE

FORMAT_VERSION = 1
STAGE = "export"
#: The parts in the order Dataset.export calls them, with the section each writes. ``segment`` is the part of the
#: route after fusion (docs/sample-segmentation-contract.md, "The entry"); on the route per FOV it is ``off``.
PARTS = ("assemble", "fuse", "segment", "fuse_labels", "export_spatialdata")
SECTIONS = {"assemble": "assembly", "fuse": "fusion", "segment": "segmentation", "fuse_labels": "labels",
            "export_spatialdata": "spatialdata"}
STATUSES = ("written", "reused", "off", "empty", "not_run", "failed")
#: Statuses of a part that list no file.
NO_FILES = ("off", "not_run", "empty")
#: The reason of the entry ``Dataset.export`` records for each part it will run before the first part starts, so
#: that a record left by an interrupted export is not complete; the part replaces it with its own section.
PENDING_REASON = "export_not_finished"
VIEWER_QUALIFICATION = "deferred"
#: The SHA-256 of a section that another section records when it reads it: recording section →
#: {field: the section it hashes}. Written by ``fuse_images`` (``_fuse.py``), ``fuse_labels`` (``_labels.py``) and
#: ``export_spatialdata`` (``_spatialdata.py``); a field holding None records nothing.
RECORDED_SECTION_HASHES = {"fusion": {"assembly_sha256": "assembly"},
                           "labels": {"assembly_sha256": "assembly"},
                           "spatialdata": {"assembly_sha256": "assembly", "fusion_sha256": "fusion",
                                           "labels_sha256": "labels"}}


class ExportIncompleteWarning(UserWarning):
    """A part of the export could not run (an extra not installed or an input file missing); the record says which."""


def section_sha256(section) -> str:
    """SHA-256 of a section's canonical JSON: sorted keys, no spaces, UTF-8."""
    text = json.dumps(section, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def verify_section_hashes(record: Mapping, where="export.json") -> None:
    """Every section SHA-256 the record holds (:data:`RECORDED_SECTION_HASHES`) equals that section's current one.

    A recording section that is absent (its part off or not run) records
    nothing. Raises ValueError naming the changed section and the field that
    recorded its hash.
    """
    for recorder, fields in RECORDED_SECTION_HASHES.items():
        section = record.get(recorder)
        if not isinstance(section, Mapping):
            continue
        for field, name in fields.items():
            recorded = section.get(field)
            if recorded is None:
                continue
            current = section_sha256(record.get(name))
            if recorded != current:
                state = "is absent" if name not in record else f"has SHA-256 {current}"
                raise ValueError(f"{where} fails its hashes: the {name} section {state}, but {recorder}.{field} "
                                 f"recorded {recorded} when the {recorder} section read it")


def as_json(value):
    """The value as it reads back from export.json."""
    from starfinder._records import json_record
    return json_record(value)


def read_export(folder) -> dict:
    """The parsed ``export.json`` of a folder, or an empty dict when there is none."""
    path = Path(folder) / EXPORT_FILE
    if not path.exists():
        return {}
    record = json.loads(path.read_text())
    if not isinstance(record, dict):
        raise ValueError(f"{path} is not a JSON object")
    return record


def part_status(record: Mapping, part: str) -> str | None:
    """A part's status: ``parts[part]``, else its section's ``status``, else ``written`` for a saved assembly."""
    status = (record.get("parts") or {}).get(part)
    if status is not None:
        return status
    section = record.get(SECTIONS[part])
    if not isinstance(section, Mapping):
        return None
    if "status" in section:
        return section["status"]
    return "written" if part == "assemble" and section.get("stage") == "assembly" else None


def pending_section() -> dict:
    """The entry of a part that ``Dataset.export`` will run and that has not written its own section yet."""
    return {"status": "not_run", "reason": PENDING_REASON}


def is_pending(record: Mapping, part: str) -> bool:
    """Whether a part's section is still the entry :func:`pending_section` (the export stopped before it ended)."""
    section = record.get(SECTIONS[part])
    return isinstance(section, Mapping) and section.get("status") == "not_run" \
        and section.get("reason") == PENDING_REASON


def is_complete(record: Mapping) -> bool:
    """``complete``: no part of the record is ``not_run`` or ``failed``."""
    return all(part_status(record, part) not in ("not_run", "failed") for part in PARTS)


def write_export(folder, record: dict, *, sample: str | None = None) -> dict:
    """Write ``export.json`` atomically with its top-level fields recomputed; return the record written."""
    from starfinder._records import write_json
    record = dict(record)
    sample = sample or record.get("sample")
    record.update({"format_version": FORMAT_VERSION, "stage": STAGE, "sample": sample,
                   "frame_id": f"sample:{sample}", "viewer_qualification": VIEWER_QUALIFICATION})
    record["parts"] = {part: part_status(record, part) for part in PARTS if part_status(record, part) is not None}
    record["complete"] = is_complete(record)
    order = ["format_version", "stage", "sample", "frame_id", "complete", "parts", "viewer_qualification"]
    record = {key: record[key] for key in order} | {k: v for k, v in record.items() if k not in order}
    write_json(record, Path(folder) / EXPORT_FILE)
    return as_json(record)


def update_section(folder, sample: str, part: str, section: Mapping, *, status: str | None = None,
                   fill: bool = True) -> dict:
    """Replace one part's section of ``export.json`` (created when absent), leaving the other sections unchanged.

    With ``fill`` a part that has no status yet (a record created by a part
    called alone) is recorded ``off``; ``Dataset.export`` passes ``fill=False``
    for the parts it has not run yet.
    """
    record = read_export(folder)
    if record.get("sample") not in (None, sample):
        raise ValueError(f"{Path(folder) / EXPORT_FILE} records sample {record.get('sample')!r}, not {sample!r}")
    record[SECTIONS[part]] = as_json(section)
    parts = dict(record.get("parts") or {})
    parts[part] = status or section.get("status")
    record["parts"] = parts
    if fill:
        for other in PARTS:
            if part_status(record, other) is None:
                record[SECTIONS[other]] = {"status": "off"}
                parts[other] = "off"
    return write_export(folder, record, sample=sample)


def listed_files(record: Mapping, part: str) -> list[str]:
    """The paths, relative to the folder, that a record lists for a part (files and folders)."""
    section = record.get(SECTIONS[part])
    if not isinstance(section, Mapping):
        return []
    if part == "assemble":
        return [entry["path"] for entry in (section.get("files") or {}).values() if isinstance(entry, Mapping)]
    paths = []
    for entry in section.get("outputs") or []:
        if isinstance(entry, Mapping) and entry.get("path"):
            paths.append(entry["path"])
    for entry in (section.get("files") or {}).values() if isinstance(section.get("files"), Mapping) else ():
        if isinstance(entry, Mapping) and entry.get("path"):
            paths.append(entry["path"])
    return paths


def listed_file_hashes(record: Mapping) -> dict[str, str]:
    """Every file a record lists, relative path → SHA-256 (the files inside a listed folder included)."""
    hashes = {}
    for part in PARTS:
        section = record.get(SECTIONS[part])
        if not isinstance(section, Mapping):
            continue
        if part == "assemble":
            for entry in (section.get("files") or {}).values():
                hashes[entry["path"]] = entry["sha256"]
            continue
        for entry in section.get("outputs") or []:
            if entry.get("files"):
                for name, digest in entry["files"].items():
                    hashes[f"{entry['path']}/{name}"] = digest
            else:
                hashes[entry["path"]] = entry["sha256"]
    return hashes


def remove_listed(folder, paths) -> None:
    """Remove the listed files or folders of an earlier record; a path the record does not list is never touched."""
    folder = Path(folder).resolve()
    for relative in paths:
        path = (folder / relative).resolve()
        if folder not in path.parents:
            raise ValueError(f"export.json lists {relative!r}, which is outside the export folder {folder}")
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()


def directory_files(path) -> dict[str, str]:
    """Every file under a folder, relative POSIX path → SHA-256, sorted."""
    from starfinder._records import file_sha256
    path = Path(path)
    return {p.relative_to(path).as_posix(): file_sha256(p) for p in sorted(path.rglob("*")) if p.is_file()}


class Timer:
    """Wall and CPU time of a part, and the process's peak RSS (KiB) at its end."""

    def __init__(self):
        self.wall, self.cpu = time.perf_counter(), time.process_time()

    def record(self) -> dict:
        return {"wall_s": time.perf_counter() - self.wall, "cpu_s": time.process_time() - self.cpu,
                "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)}


@dataclass(frozen=True)
class ExportRecord:
    """The record ``export.json`` of an export folder, as ``Dataset.export`` wrote it.

    ``folder`` is the export folder; ``record`` the parsed ``export.json``:
    ``format_version``, ``stage``, ``sample``, ``frame_id``, ``complete``,
    ``parts`` (each part's status: ``written``, ``reused``, ``off``,
    ``empty``, ``not_run`` or ``failed``), ``viewer_qualification`` and one
    section per part (``assembly``, ``fusion``, ``segmentation``, ``labels``,
    ``spatialdata``).
    """

    folder: Path
    record: Mapping[str, Any]

    @property
    def complete(self) -> bool:
        """No requested part is ``not_run`` or ``failed``."""
        return bool(self.record.get("complete"))

    @property
    def parts(self) -> dict[str, str]:
        """Each part's status."""
        return dict(self.record.get("parts") or {})

    @property
    def paths(self) -> tuple[Path, ...]:
        """Every file and folder the record lists, as paths inside ``folder``."""
        return tuple(Path(self.folder) / p for part in PARTS for p in listed_files(self.record, part))
