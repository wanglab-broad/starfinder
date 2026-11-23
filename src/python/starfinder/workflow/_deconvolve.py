"""The commands of the deconvolution stage: ``psf``, ``deconvolve``, the wisdom store and a consumer's check.

design/specs/deconvolution.md. The ``psf`` task writes the copy's PSFs (generated with psfmodels, or supplied files),
``psf/psf.json`` and ``copy.json``, or finds them verified and writes nothing. A ``deconvolve`` task runs deconwolf on
each channel of its round in a private temporary folder and writes the copy's ``<image dir>/<label>.tif`` files with
the workflow's image codec and its record ``records/<round name>/<fov>/deconvolution.json``, or reuses a verified round.
``starfinder deconvolve wisdom`` adds one key's FFTW wisdom file to a store, which tasks only read. A consumer checks
the copy's files against their round's record before it reads them and names the copy in its ``run.json``.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import resource
import shutil
import socket
import subprocess
import tempfile
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from starfinder._records import atomic_file, file_sha256
from starfinder.workflow._deconvolution import (COPY_FILE, COPY_FORMAT, PSF_DIR, PSF_FILE, PSF_FORMAT, RECORD_FILE,
    RECORD_FORMAT, TILE_PADDING, copy_fields, copy_id, psf_file_path, record_dir)
from starfinder.workflow._shape import canonical

logger = logging.getLogger("starfinder.workflow")

#: The one deconwolf version whose kept offset is known (5 untiled, 0 tiled; design/specs/deconvolution.md).
DECONWOLF_VERSION = "0.4.5"
OFFSET = 5
#: An untiled output whose minimum is below this has not kept the offset.
UNTILED_MINIMUM = 4.5
#: deconwolf v0.4.5 refuses an image whose maximum is below this ("ERROR: The image has too low intensity, can not
#: continue!", exit 1; ``src/dw.c`` 2522–2532). A channel whose raw values are all 0 skips deconwolf (W-933).
DECONWOLF_FLOOR = 1.0
SKIP_REASON = f"raw values all 0; deconwolf v{DECONWOLF_VERSION} refuses a maximum below 1"
WISDOM_FORMAT = "starfinder.deconwolf-wisdom/1"
WISDOM_MANIFEST = "manifest.json"
#: psfmodels' defaults that a generated PSF uses and records (W-855): coverslip, working distance, oversampling.
PSFMODELS_DEFAULTS = {"tg": 170, "tg0": 170, "ng": 1.515, "ng0": 1.515, "ti0": 150, "oversample_factor": 3}


def _now():
    return datetime.now(timezone.utc).isoformat()


def image_sha256(image):
    """SHA-256 of an array's C-order bytes (the decoded identity, W-909's ID1)."""
    return hashlib.sha256(np.ascontiguousarray(image).tobytes()).hexdigest()


def _write_json(path, data):
    with atomic_file(path) as tmp:
        tmp.write_text(json.dumps(data, indent=1) + "\n")


def wisdom_name(threads):
    """The file name of deconwolf v0.4.5's FFTW wisdom for ``threads`` (``src/fft.c``, ``get_swf_file_name``)."""
    return f"fftw_wisdom_float_inplace_threads_{threads}.dat"


# --- deconwolf ----------------------------------------------------------------------------------------------------

def deconwolf_binary(execution):
    """The deconwolf executable: ``execution.deconwolf``, else ``dw`` on PATH; ValueError when there is none."""
    given = execution.get("deconwolf")
    path = given or shutil.which("dw")
    if path is None or not Path(path).is_file():
        raise ValueError(f"execution.deconwolf {given!r}: no deconwolf executable" if given else
                         "no deconwolf executable: dw is not on PATH; set execution.deconwolf")
    return str(Path(path).absolute())


def deconwolf_identity(binary):
    """``dw --version`` (the version, the FFT backend line and the text), whether ``--help`` lists ``--float``, and
    the binary's SHA-256. A version other than 0.4.5 raises ValueError: its kept offset is not established."""
    text = subprocess.run([binary, "--version"], capture_output=True, text=True).stdout
    version = re.search(r"deconwolf:\s*'?([0-9][^'\s]*)", text)
    version = version.group(1) if version else None
    if version != DECONWOLF_VERSION:
        raise ValueError(f"deconwolf {binary} reports version {version!r}: the stage removes the kept offset by "
                         f"v{DECONWOLF_VERSION}'s rule (5 untiled, 0 tiled) only; establish the rule for another "
                         "version first")
    backend = re.search(r"FFT Backend:\s*'([^']*)'", text)
    help_text = subprocess.run([binary, "--help"], capture_output=True, text=True)
    return {"version": version, "version_text": text.strip(), "path": binary, "sha256": file_sha256(binary),
            "fft_backend": backend.group(1) if backend else None,
            "float_output": "--float" in help_text.stdout + help_text.stderr}


def cpu_identity():
    """The CPU's model name and the SHA-256 of its sorted flags (``/proc/cpuinfo``), and their hash."""
    model, flags = None, []
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            key, _, value = line.partition(":")
            if key.strip() == "model name" and model is None:
                model = value.strip()
            elif key.strip() == "flags" and not flags:
                flags = sorted(value.split())
    except OSError:
        pass
    flags_sha256 = hashlib.sha256(" ".join(flags).encode()).hexdigest()
    return {"model": model, "flags_sha256": flags_sha256,
            "sha256": hashlib.sha256(canonical({"model name": model, "flags": flags})).hexdigest()}


def wisdom_fields(image_shape, psf_shape, border_quality, tile_size, threads, cpu, identity):
    """The fields of a wisdom key (design/specs/deconvolution.md, "FFTW wisdom"): the job shape (the image plus the PSF
    less one per axis), the PSF's shape, the two settings that change the transform sizes, the threads, the CPU, the
    FFT backend and deconwolf's version and binary."""
    return {"job_shape_zyx": [int(i) + int(p) - 1 for i, p in zip(image_shape, psf_shape)],
            "psf_shape_zyx": [int(p) for p in psf_shape], "border_quality": int(border_quality),
            "tile_size": tile_size, "threads": int(threads), "cpu": cpu,
            "fftw": identity["fft_backend"], "deconwolf": {"version": identity["version"], "sha256": identity["sha256"]}}


def wisdom_key(fields):
    """The key: the SHA-256 of the fields' canonical JSON."""
    return hashlib.sha256(canonical(fields)).hexdigest()


def _store_manifest(store):
    path = Path(store) / WISDOM_MANIFEST
    if not path.is_file():
        return {"format": WISDOM_FORMAT, "keys": {}}
    return json.loads(path.read_text())


def find_wisdom(store, key):
    """The store's file for ``key`` checked against the manifest: (path, SHA-256), or None when the store has no such
    key. A file whose SHA-256 differs from the manifest raises ValueError naming it (a damaged store)."""
    entry = _store_manifest(store)["keys"].get(key)
    if entry is None:
        return None
    path = Path(store) / entry["file"]
    found = file_sha256(path) if path.is_file() else None
    if found != entry["sha256"]:
        raise ValueError(f"the wisdom store's file {path} has SHA-256 {found}; its manifest names {entry['sha256']}: "
                         "the store is damaged")
    return path, entry["sha256"]


@contextmanager
def exclusive(path, timeout, what):
    """A lock file made with O_EXCL, waited on up to ``timeout`` seconds and removed at the end."""
    path = Path(path)
    start = time.monotonic()
    while True:
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError:
            if time.monotonic() - start > timeout:
                raise TimeoutError(f"{what}: the lock {path} is held for more than {timeout:.0f} s") from None
            time.sleep(0.1)
    try:
        os.write(fd, f"{socket.gethostname()} {os.getpid()}\n".encode())
        os.close(fd)
        yield
    finally:
        path.unlink(missing_ok=True)


def _private_home(root):
    """A private HOME with ``.config/deconwolf/`` under ``root``, deconwolf's working directory too."""
    home = Path(root) / "home"
    (home / ".config" / "deconwolf").mkdir(parents=True)
    return home


def run_deconwolf(argv, home, threads):
    """Run deconwolf with HOME and the working directory the private folder and OMP_NUM_THREADS ``threads``; (exit
    status, its output, wall seconds, CPU seconds)."""
    env = dict(os.environ, HOME=str(home), OMP_NUM_THREADS=str(threads))
    before, start = resource.getrusage(resource.RUSAGE_CHILDREN), time.perf_counter()
    done = subprocess.run(argv, cwd=home, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    after = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = (after.ru_utime - before.ru_utime) + (after.ru_stime - before.ru_stime)
    return done.returncode, done.stdout, time.perf_counter() - start, cpu


# --- the PSF --------------------------------------------------------------------------------------------------------

def centre_psf(psf):
    """The PSF moved by whole voxels so that its peak is at the centre index: ``np.roll``, the slices the roll wrapped
    (in each axis it shifts) set to 0, then normalised to sum 1 (W-855 amendment 2).

    Returns (centred float64 array, record): the shift (centre less peak), the peak before and after and its offset
    from the centre before, and the share of the input's sum the zeroed slices held.
    """
    psf = np.asarray(psf, dtype=np.float64)
    centre = [n // 2 for n in psf.shape]
    peak = [int(v) for v in np.unravel_index(int(np.argmax(psf)), psf.shape)]
    shift = [c - p for c, p in zip(centre, peak)]
    centred = np.roll(psf, shift, axis=(0, 1, 2))
    mask = np.zeros(psf.shape, dtype=bool)
    for axis, s in enumerate(shift):
        index = [slice(None)] * 3
        if s > 0:
            index[axis] = slice(0, s)
        elif s < 0:
            index[axis] = slice(psf.shape[axis] + s, psf.shape[axis])
        else:
            continue
        mask[tuple(index)] = True
    dropped = float(centred[mask].sum() / psf.sum()) if mask.any() else 0.0
    centred[mask] = 0.0
    centred /= centred.sum()
    after = [int(v) for v in np.unravel_index(int(np.argmax(centred)), centred.shape)]
    return centred, {"shift_zyx": shift, "peak_before": peak, "peak_offset_before": [p - c for p, c in zip(peak, centre)],
                     "peak_after": after, "centre": centre, "sum_dropped": dropped}


def fwhm(profile, step):
    """Full width at half maximum of a profile through its peak, by linear interpolation, in units of ``step``."""
    p = np.asarray(profile, dtype=np.float64)
    i = int(np.argmax(p))
    half = p[i] / 2
    left = right = i
    while left > 0 and p[left - 1] > half:
        left -= 1
    while right < len(p) - 1 and p[right + 1] > half:
        right += 1
    lx = left - (p[left] - half) / (p[left] - p[left - 1]) if left > 0 else float(left)
    rx = right + (p[right] - half) / (p[right] - p[right + 1]) if right < len(p) - 1 else float(right)
    return float((rx - lx) * step)


def psf_parameters(psf, label, dxy, dz):
    """The arguments of ``psfmodels.confocal_psf`` for a label: the block's optics, the label's wavelengths (µm), the
    pinhole in Airy units (radius / (0.61 × emission / NA)) and the dataset's spacing."""
    channel = psf["channels"][label]
    emission = float(channel["emission_nm"]) / 1000
    nz, ny, nx = psf["shape_zyx"]
    return {"z": int(nz), "nx": int(nx), "pinhole_au": float(psf["pinhole_radius_nm"]) / 1000 / (0.61 * emission
                                                                                                 / float(psf["na"])),
            "dxy": float(dxy), "dz": float(dz), "pz": float(psf["depth_um"]), "NA": float(psf["na"]),
            "ex_wvl": float(channel["excitation_nm"]) / 1000, "em_wvl": emission, "ns": float(psf["sample_ri"]),
            "ni": float(psf["immersion_ri"]), "ni0": float(psf["immersion_ri"]), "model": "vectorial"}


def generate_psf(parameters):
    """psfmodels' confocal PSF (``confocal_psf``) of the parameters, normalised to sum 1 and stored as float32, then
    centred (``centre_psf``) and returned as float32 with the centring record, as W-855 made its PSFs."""
    try:
        import psfmodels
    except ImportError:
        raise ValueError("deconvolution.psf.source generate needs psfmodels: install the deconvolution extra "
                         "(uv sync --extra deconvolution), or supply PSF files (psf.source file)") from None
    psf = psfmodels.confocal_psf(**parameters).astype(np.float64)
    psf = (psf / psf.sum()).astype(np.float32)
    centred, record = centre_psf(psf)
    return centred.astype(np.float32), record, psfmodels.__version__


def read_supplied_psf(path):
    """A supplied PSF file as float64 ZYX: refused when not real, not three-dimensional or negative; renormalised to
    sum 1 when it is not (returned with that flag and its sum)."""
    import tifffile
    data = tifffile.imread(path)
    if not (np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, np.integer)):
        raise ValueError(f"the PSF file {path} has dtype {data.dtype}, not a real type")
    if data.ndim != 3:
        raise ValueError(f"the PSF file {path} has {data.ndim} axes, not ZYX")
    data = data.astype(np.float64)
    if not np.isfinite(data).all() or (data < 0).any():
        raise ValueError(f"the PSF file {path} has a negative or non-finite value")
    total = float(data.sum())
    renormalised = abs(total - 1) > 1e-6
    return (data / total if renormalised else data), renormalised, total


def _psf_record(label, block, r):
    """The record of one label's PSF as this run would make it (without the written file's hashes)."""
    psf = block["psf"]
    if psf["source"] == "file":
        path = psf_file_path(r, psf["files"][label])
        return {"source": "file", "path": str(path), "file_sha256": file_sha256(path),
                "declared_spacing": dict(psf["voxel_size_um"])}
    return {"source": "generate", "function": "psfmodels.confocal_psf",
            "parameters": psf_parameters(psf, label, r["voxel_size_xy"], r["voxel_size_z"]),
            "defaults": dict(PSFMODELS_DEFAULTS)}


def _compared(record):
    """The fields of a label's PSF record that a later run compares (the parameters, or the supplied file)."""
    keys = ("source", "parameters") if record.get("source") == "generate" else ("source", "file_sha256",
                                                                                "declared_spacing")
    return {k: record.get(k) for k in keys}


def check_copy_psf(run):
    """The psf task's check of an existing copy (exit 2 on a mismatch, nothing written): True when ``copy.json``
    exists and its fields, the PSFs' recorded parameters and every PSF file's SHA-256 equal this run's; False when
    the copy has no ``copy.json``."""
    from starfinder.workflow._task import TaskRefused
    block, entry = run.configuration["deconvolution"], run.task["deconvolution"]
    r = run.configuration.plain()
    copy = run.path(entry["copy"])
    path = copy / COPY_FILE
    fields = copy_fields(r, entry["labels"])
    if copy_id(fields) != entry["copy_id"]:
        raise TaskRefused(f"the copy id of this configuration is {copy_id(fields)}, the plan's {entry['copy_id']}: a "
                          "supplied PSF file changed after the plan; plan the run again")
    if not path.is_file():
        return False
    try:
        recorded = json.loads(path.read_text())
        psf_record = json.loads((copy / PSF_DIR / PSF_FILE).read_text())
    except (OSError, ValueError) as error:
        raise TaskRefused(f"the copy {copy} cannot be read: {error}") from None
    if canonical(recorded.get("fields")) != canonical(fields) or recorded.get("copy_id") != entry["copy_id"]:
        raise TaskRefused(f"{path}: its fields differ from this run's; the copy is damaged or edited")
    for label in entry["labels"]:
        stored = (psf_record.get("labels") or {}).get(label) or {}
        expected = _compared(_psf_record(label, block, r))
        for key, value in expected.items():
            if canonical(stored.get(key)) != canonical(value):
                raise TaskRefused(f"{copy / PSF_DIR / PSF_FILE}: labels.{label}.{key} is {stored.get(key)!r}, this run "
                                  f"uses {value!r}; the copy is damaged or edited")
        file = copy / PSF_DIR / f"{label}.tif"
        digest = file_sha256(file) if file.is_file() else None
        if digest != (stored.get("copy") or {}).get("file_sha256"):
            raise TaskRefused(f"{file}: SHA-256 {digest}, {PSF_FILE} records "
                              f"{(stored.get('copy') or {}).get('file_sha256')}; the copy's PSF is damaged")
    if file_sha256(copy / PSF_DIR / PSF_FILE) != (recorded.get("psf") or {}).get("sha256"):
        raise TaskRefused(f"{copy / PSF_DIR / PSF_FILE}: its SHA-256 differs from the one {path} records")
    return True


def prepare_psf(run):
    """The psf command's preparation: a verified copy is reused (its PSFs written once); a damaged one refused."""
    run.reuse = check_copy_psf(run)
    return ()


def psf(run):
    """The psf task: each label's PSF, generated (psfmodels, centred) or supplied (checked), written float32 into
    ``<copy>/psf/<label>.tif``, then ``psf/psf.json`` and ``copy.json`` last; nothing when the copy is verified. Each
    file is marked written (``run.wrote``) just before it is written, so that a failure removes only those."""
    import tifffile
    entry = run.task["deconvolution"]
    copy = run.path(entry["copy"])
    if getattr(run, "reuse", False):
        run.record_extra["reused"] = True
        logger.info(f"the copy {copy} holds its verified PSFs; nothing written")
        return
    run.record_extra["reused"] = False
    block = run.configuration["deconvolution"]
    r = run.configuration.plain()
    labels, records = {}, {}
    for label in entry["labels"]:
        record = _psf_record(label, block, r)
        if record["source"] == "generate":
            array, centring, version = generate_psf(record["parameters"])
            record.update(psfmodels_version=version, centring=centring, normalisation="sum 1")
            peak = centring["peak_after"]
            record["fwhm_axial_um"] = fwhm(array[:, peak[1], peak[2]], record["parameters"]["dz"])
            record["fwhm_lateral_um"] = fwhm(array[peak[0], peak[1], :], record["parameters"]["dxy"])
        else:
            data, renormalised, total = read_supplied_psf(record["path"])
            record.update(sha256=image_sha256(tifffile.imread(record["path"])), renormalised=renormalised, sum=total)
            array = data.astype(np.float32)
        record["shape_zyx"] = list(array.shape)
        labels[label] = array
        records[label] = record
    for label, array in labels.items():
        target = copy / PSF_DIR / f"{label}.tif"
        run.wrote(target)
        with atomic_file(target) as tmp:
            tifffile.imwrite(tmp, array, photometric="minisblack", metadata={"axes": "ZYX"})
        records[label]["copy"] = {"path": f"{PSF_DIR}/{label}.tif", "file_sha256": file_sha256(target),
                                  "sha256": image_sha256(array)}
    from starfinder.workflow._plan import package_identity
    # No time: the copy's PSF records are the same whichever engine or run writes them (copy.json holds their hash).
    run.wrote(copy / PSF_DIR / PSF_FILE, copy / COPY_FILE)
    _write_json(copy / PSF_DIR / PSF_FILE, {"format": PSF_FORMAT, "copy_id": entry["copy_id"], "labels": records,
                                            "code": package_identity()})
    fields = copy_fields(r, entry["labels"])
    _write_json(copy / COPY_FILE, {"format": COPY_FORMAT, "copy_id": copy_id(fields), "fields": fields,
                                   "psf": {"path": f"{PSF_DIR}/{PSF_FILE}",
                                           "sha256": file_sha256(copy / PSF_DIR / PSF_FILE)}})
    logger.info(f"wrote the PSFs of {', '.join(entry['labels'])} and copy.json into {copy}")


# --- the deconvolve task --------------------------------------------------------------------------------------------

def _scale(block, name, label):
    scale = block["scale"]
    return float(scale[name][label] if isinstance(scale, dict) else scale)


def _reusable(run, record_path, inputs, psfs, copy):
    """Why the round's existing record cannot be reused, or None when it can (design/specs/deconvolution.md,
    "Across runs: verified reuse")."""
    entry = run.task["deconvolution"]
    try:
        record = json.loads(record_path.read_text())
    except (OSError, ValueError):
        return "no readable record"
    if record.get("round") != entry["round"] or record.get("copy_id") != entry["copy_id"]:
        return "the record's round or copy id differs"
    channels = {c.get("label"): c for c in record.get("channels") or []}
    if sorted(channels) != sorted(entry["labels"]):
        return "the record's labels differ"
    for label in entry["labels"]:
        channel = channels[label]
        if channel["input"]["file_sha256"] != inputs[label][1]:
            return f"the raw file of {label} changed"
        if channel["psf"]["file_sha256"] != psfs[label][1]:
            return f"the PSF of {label} changed"
        output = copy / channel["output"]["path"]
        if not output.is_file() or file_sha256(output) != channel["output"]["file_sha256"]:
            return f"the copy's file of {label} is missing or changed"
    if run.configuration["execution"].get("code_version") == "rerun":
        from starfinder.workflow._plan import package_identity
        code = {k: v for k, v in package_identity().items() if k != "version_source"}
        if {k: v for k, v in (record.get("code") or {}).items() if k != "version_source"} != code:
            return "the package's code identity differs (code_version: rerun)"
        binary = deconwolf_binary(run.configuration["execution"])
        if file_sha256(binary) != (record.get("deconwolf") or {}).get("sha256"):
            return "the deconwolf binary differs (code_version: rerun)"
    return None


def deconvolve(run):
    """The deconvolve task of one (FOV, round name): under the lock of its record folder, reuse the verified round or
    deconvolve each channel, then write the copy's files and the round's record last.

    A failure removes only what this attempt took over (``run.wrote``): nothing when it reused the round, timed out on
    the lock or failed before it decided to redo the round; on a redo, its round's label files, logs and record, which
    it takes over under the lock when it removes the old record (an unrecorded file of the round is stale for every
    reader), never the other round's files in a shared image folder."""
    entry = run.task["deconvolution"]
    copy = run.path(entry["copy"])
    fov, name = run.identity["fov"], entry["round"]
    records = record_dir(copy, name, fov)
    records.mkdir(parents=True, exist_ok=True)
    images = run.manifest["inputs"]["images"][fov][name]
    timeout = run.task["execution"]["runtime_min"] * 60 * run.context.attempt
    with exclusive(records / ".lock", timeout, f"task {run.id}"):
        inputs = {label: (run.root / images[label]["path"], file_sha256(run.root / images[label]["path"]))
                  for label in entry["labels"]}
        psfs = {label: (copy / PSF_DIR / f"{label}.tif", file_sha256(copy / PSF_DIR / f"{label}.tif"))
                for label in entry["labels"]}
        record_path = records / RECORD_FILE
        reason = _reusable(run, record_path, inputs, psfs, copy) if record_path.is_file() else "no record"
        if reason is None:
            run.record_extra["reused"] = True
            logger.info(f"reused the verified round {name} of FOV {fov} in {copy}")
            return
        run.record_extra["reused"] = False
        logger.info(f"deconvolving round {name} of FOV {fov} into {copy} ({reason})")
        run.wrote()
        record_path.unlink(missing_ok=True)
        _deconvolve_round(run, entry, copy, records, inputs, psfs)


def _skipped(path):
    """(whether the raw channel at ``path`` is all 0, its maximum): a channel of zeros skips deconwolf (W-933)."""
    from starfinder.io import load_volume
    image = load_volume(path).image
    if not image.any():
        return True, 0.0
    return False, float(image.max())


def _deconvolve_round(run, entry, copy, records, inputs, psfs):
    """Deconvolve the round's channels and write the copy's files, the logs and the round's record
    (design/specs/deconvolution.md, "The scaling", "Records").

    The task first reads every channel and takes its maximum, before it calls deconwolf. A channel whose raw values
    are all 0 is not passed to deconwolf, untiled or tiled, since v0.4.5 refuses a maximum below 1 (``src/dw.c``): its
    values after the offset are 0, the exact deconvolution of zeros. Its record says ``deconwolf_skipped: true`` with
    ``skip_reason`` and ``max_after_offset`` 0, and its deconwolf fields (``arguments``, ``iterations_done``,
    ``float_min``, ``float_max``, ``kept_offset``, ``wall_s``, ``cpu_s``) are null; its log, a declared output, holds
    one line naming the skip. It stores zeros and takes part in the round's maximum M as 0. A channel that is not all 0
    but whose maximum is below 1 (a float input) fails the task before deconwolf is called, naming the round, the
    channel and its maximum. The planning is decided at the first channel passed to deconwolf; a round with none
    records ``{mode: none}`` and looks up no wisdom. Then each other channel is deconvolved to float32 in the private
    temporary folder, M is taken over the round, and each channel's file is written with the round's factor (uint8) or
    its scale (uint16)."""
    from starfinder.io import load_volume, save_volume
    from starfinder.workflow._commands import image_compression
    from starfinder.workflow._plan import package_identity
    import tifffile
    started_at = _now()
    block, execution = run.configuration["deconvolution"], run.configuration["execution"]
    name, labels = entry["round"], entry["labels"]
    skipped = {}
    for label in labels:
        skipped[label], peak = _skipped(inputs[label][0])
        if not skipped[label] and peak < DECONWOLF_FLOOR:
            raise ValueError(f"round {name}, channel {label}: its raw maximum is {peak}, below deconwolf "
                             f"v{DECONWOLF_VERSION}'s floor of {DECONWOLF_FLOOR:g} (it refuses an image whose maximum "
                             "is below 1, src/dw.c), and the channel is not all 0; it is not passed to deconwolf")
    gpu = bool(run.task["execution"]["gpu"])
    threads = 1 if gpu else int(run.task["execution"]["threads"])
    tile = block["tile_size"]
    kept = 0 if tile else OFFSET
    binary = deconwolf_binary(execution)
    identity = deconwolf_identity(binary)
    codec = image_compression(run)
    tmp = Path(tempfile.mkdtemp(prefix="starfinder-deconvolve-"))
    try:
        home = _private_home(tmp)
        channels, metadata, outputs, logs = [], {}, {}, {}
        planning = None
        for label in labels:
            path, digest = inputs[label]
            loaded = load_volume(path)
            image = loaded.image
            metadata[label] = loaded.metadata if loaded.diagnostics["metadata_source"] == "stored" else None
            described = {"path": str(path), "file_sha256": digest, "dtype": str(image.dtype),
                         "shape": list(image.shape), "sha256": image_sha256(image)}
            if skipped[label]:
                channels.append({
                    "label": label, "input": described,
                    "psf": {"path": f"{PSF_DIR}/{label}.tif", "file_sha256": psfs[label][1]},
                    "deconwolf_skipped": True, "skip_reason": SKIP_REASON, "iterations_done": None,
                    "float_min": None, "float_max": None, "kept_offset": None, "max_after_offset": 0.0,
                    "clipped_low": 0, "arguments": None, "wall_s": None, "cpu_s": None})
                logs[label] = tmp / f"{label}.dw.log"
                logs[label].write_text(f"deconwolf skipped for {label} of round {name}: {SKIP_REASON}\n")
                logger.info(f"{label} of round {name}: {SKIP_REASON}; deconwolf skipped, the copy stores zeros")
                continue
            if planning is None:
                planning = _planning(execution, gpu, image.shape, psfs[label][0], block, threads, identity, home)
            source = tmp / f"{label}.input.tif"
            tifffile.imwrite(source, image.astype(np.float32), photometric="minisblack")
            out = tmp / f"{label}.dw.tif"
            argv = [binary, "--iter", str(block["iterations"]), "--threads", str(threads), "--offset", str(OFFSET)]
            if block["border_quality"] != 2:
                argv += ["--bq", str(block["border_quality"])]
            if gpu:
                argv.append("--gpu")
            if tile:
                argv += ["--tilesize", str(tile)]
            if planning["mode"] == "estimate":
                argv.append("--noplan")
            if identity["float_output"]:
                argv.append("--float")
            argv += ["--out", str(out), str(source), str(psfs[label][0])]
            status, output, wall, cpu = run_deconwolf(argv, home, threads)
            own_log = Path(f"{out}.log.txt")
            text = output + (b"\n--- " + own_log.name.encode() + b"\n" + own_log.read_bytes() if own_log.is_file()
                             else b"")
            logs[label] = tmp / f"{label}.dw.log"
            logs[label].write_bytes(text)
            source.unlink()
            if status != 0:
                logger.error(text.decode(errors="replace")[-4000:])
                raise RuntimeError(f"deconwolf exited with status {status} on {label} of round {name}")
            decoded = text.decode(errors="replace")
            y = _float_output(out, identity, decoded, label)
            lowest, highest = float(y.min()), float(y.max())
            if not tile and lowest < UNTILED_MINIMUM:
                raise ValueError(f"deconwolf's untiled output of {label} has minimum {lowest}, below "
                                 f"{UNTILED_MINIMUM}: the kept offset rule of v{DECONWOLF_VERSION} (an offset of "
                                 f"{OFFSET} kept untiled) does not hold")
            values = y - kept
            done = [int(n) for n in re.findall(r"Iteration\s+(\d+)", decoded)]
            channels.append({
                "label": label, "input": described,
                "psf": {"path": f"{PSF_DIR}/{label}.tif", "file_sha256": psfs[label][1]}, "deconwolf_skipped": False,
                "iterations_done": max(done) if done else None, "float_min": lowest, "float_max": highest,
                "kept_offset": kept, "max_after_offset": float(max(values.max(), 0.0)),
                "clipped_low": int((values < 0).sum()), "arguments": argv, "wall_s": wall, "cpu_s": cpu})
            outputs[label] = out
        # The second pass: one channel in memory at a time, from the float outputs kept in the temporary folder.
        storage = {"dtype": block["dtype"]}
        if block["dtype"] == "uint8":
            round_max = max(c["max_after_offset"] for c in channels)
            factor = 255.0 / round_max if round_max > 0 else 1.0
            storage.update(stretch_factor=factor, round_max=round_max)
        saturated = 0
        for channel in channels:
            label = channel["label"]
            if channel["deconwolf_skipped"]:
                values = np.zeros(channel["input"]["shape"], dtype=np.float64)
            else:
                values = np.clip(_float_output(outputs[label], identity, logs[label].read_text(errors="replace"),
                                               label) - kept, 0, None)
            if block["dtype"] == "uint8":
                scaled = np.rint(values * storage["stretch_factor"])
                stored = np.clip(scaled, 0, 255).astype(np.uint8)
                channel["clipped_high"] = int((scaled > 255).sum())
            else:
                scale = _scale(block, name, label)
                scaled = np.rint(values * scale)
                stored = np.clip(scaled, 0, 65535).astype(np.uint16)
                channel["scale"] = scale
                channel["clipped_high"] = int((scaled > 65535).sum())
                saturated += channel["clipped_high"]
            target = copy / entry["image_dir"] / f"{label}.tif"
            with atomic_file(target) as tmp_file:
                save_volume(stored, tmp_file, compress=codec == "zlib", metadata=metadata[label])
            channel["output"] = {"path": f"{entry['image_dir']}/{label}.tif", "file_sha256": file_sha256(target),
                                 "sha256": image_sha256(stored)}
        if saturated:
            logger.warning(f"{saturated} voxels of round {name} saturated at 65535 (dtype uint16): lower "
                           "deconvolution.scale (s x the largest expected value must stay within 65535)")
        for channel in channels:
            target = records / f"{channel['label']}.dw.log"
            with atomic_file(target) as tmp_file:
                shutil.copyfile(logs[channel["label"]], tmp_file)
            channel["log"] = {"path": f"{channel['label']}.dw.log", "sha256": file_sha256(target)}
        if planning is None:
            planning = {"mode": "none", "reason": "no channel passed to deconwolf"}
        first = next((logs[c["label"]] for c in channels if not c["deconwolf_skipped"]), None)
        device = ({"kind": "gpu", "opencl_device": _opencl_device(first.read_text(errors="replace")) if first else None,
                   "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")} if gpu else
                  {"kind": "cpu", "cpu_model": cpu_identity()["model"], "threads": threads})
        record = {
            "format": RECORD_FORMAT, "copy_id": entry["copy_id"], "fov": run.identity["fov"], "round": name,
            "labels": labels, "image_dir": entry["image_dir"], "deconwolf": identity, "device": device,
            "options": {"iterations": block["iterations"], "method": block["method"],
                        "border_quality": block["border_quality"], "tile_size": tile,
                        "tile_padding": TILE_PADDING if tile else None, "offset": OFFSET, "background": "auto"},
            "storage": storage, "planning": planning, "image_compression": codec, "channels": channels,
            "code": package_identity(), "started_at": started_at, "ended_at": _now(), "host": socket.gethostname(),
            "job_id": run.context.job_id}
        _write_json(records / RECORD_FILE, record)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def _float_output(path, identity, log, label):
    """deconwolf's output as float64: the float32 file of ``--float``, or without that option its uint16 output divided
    by the scale its log names (the specification's fallback)."""
    import tifffile
    y = tifffile.imread(path).astype(np.float64)
    if identity["float_output"]:
        return y
    scaling = re.findall(r"scaling:\s*([0-9.eE+-]+)", log)
    if not scaling:
        raise ValueError(f"deconwolf wrote no 'scaling:' line for {label}, which its uint16 output needs")
    return y / float(scaling[-1])


def _opencl_device(text):
    found = re.search(r"(?:CL_DEVICE_NAME|Device name|device)\s*[:=]\s*(.+)", text)
    return found.group(1).strip() if found else None


def _planning(execution, gpu, shape, psf_path, block, threads, identity, home):
    """The task's planning: ``gpu``; ``measure`` with the store's file for its key copied into the private
    ``.config/deconwolf/``; or ``estimate`` (``--noplan``) with the reason."""
    if gpu:
        return {"mode": "gpu"}
    store = execution.get("deconwolf_wisdom")
    if store is None:
        return {"mode": "estimate", "reason": "no store"}
    import tifffile
    with tifffile.TiffFile(psf_path) as tif:
        psf_shape = tif.series[0].shape
    fields = wisdom_fields(shape, psf_shape, block["border_quality"], block["tile_size"], threads, cpu_identity()["sha256"],
                           identity)
    key = wisdom_key(fields)
    found = find_wisdom(store, key)
    if found is None:
        return {"mode": "estimate", "reason": "no matching key", "wisdom_key": key}
    shutil.copyfile(found[0], home / ".config" / "deconwolf" / wisdom_name(threads))
    return {"mode": "measure", "wisdom_key": key, "wisdom_sha256": found[1], "store": str(store)}


# --- a consumer's check -------------------------------------------------------------------------------------------

def consumer_files(run, name):
    """The copy's image files that a consumer declares for the deconvolved round ``name`` of its FOV: the ``.tif``
    outputs of the round's ``deconvolve`` task among its needs that are among its declared inputs, as paths."""
    inputs = set(run.task["inputs"])
    for need in run.task["needs"]:
        task = run.tasks.get(need)
        if (task is not None and task["kind"] == "deconvolve" and task["deconvolution"]["round"] == name
                and task["identity"]["fov"] == run.identity["fov"]):
            return [Path(run.path(o)).absolute() for o in task["outputs"] if o.endswith(".tif") and o in inputs]
    return []


def check_consumer(run):
    """A consumer's check of the copy it reads (design/specs/deconvolution.md, "A consumer checks the copy"): each
    deconvolved round's record exists and names every copy file the consumer declares for that round, and each file
    the record names has its SHA-256; ValueError naming the round and the file (and both hashes on a mismatch).
    Returns the ``run.json`` entry ``deconvolution``."""
    stage = run.manifest["deconvolution"]
    copy = Path(run.path(stage["copy"])).absolute()
    entry = {"copy": str(copy), "copy_id": stage["copy_id"], "rounds": {}}
    for name, folder in sorted(run.task["deconvolved"].items()):
        path = record_dir(copy, name, run.identity["fov"]) / RECORD_FILE
        if not path.is_file():
            raise ValueError(f"the copy {copy} has no record of round {name} of FOV {run.identity['fov']} ({path})")
        record = json.loads(path.read_text())
        named = {copy / channel["output"]["path"]: channel["output"]["file_sha256"]
                 for channel in record.get("channels") or []}
        for file in consumer_files(run, name):
            if file not in named:
                raise ValueError(f"the record {path} of round {name} of FOV {run.identity['fov']} names no file "
                                 f"{file}, which the task reads: the copy is stale or damaged")
        for file, recorded in named.items():
            digest = file_sha256(file) if file.is_file() else None
            if digest != recorded:
                raise ValueError(f"the copy's file {file} has SHA-256 {digest}; its record {path} names "
                                 f"{recorded}: the copy is stale or damaged")
        entry["rounds"][name] = {"record": str(path), "record_sha256": file_sha256(path)}
    return entry


def set_deconvolution(path, entry):
    """Add the ``deconvolution`` entry to a written ``run.json`` (an additive key, like ``spot_finding_input``)."""
    from starfinder import _records
    data = json.loads(Path(path).read_text())
    data["deconvolution"] = entry
    _records.write_json(data, path)


def run_copy_id(manifest):
    """The copy id a run's saved spot-finding set must match: the plan's, when its block deconvolves sequencing rounds
    (``rounds`` null or naming one), else None."""
    stage = manifest.get("deconvolution")
    return stage["copy_id"] if stage and stage.get("sequencing") else None


# --- the wisdom store's command -----------------------------------------------------------------------------------

def add_wisdom(store, shape, psf_shape, threads, binary=None, border_quality=2, tile_size=None, timeout=7200):
    """Plan FFTW wisdom for one key on this node and add it to the store (``starfinder deconvolve wisdom``).

    Under the store's lock, a key the manifest already holds is refused (FileExistsError, nothing written); otherwise
    deconwolf runs on a synthetic volume of ``shape`` with a PSF of ``psf_shape`` and planning on, in a private HOME,
    and its wisdom file is copied to ``<store>/<key>/`` and listed in the manifest, which is rewritten atomically.
    Returns the manifest entry.

    The volume is ``1 + 254 u`` in float32, u uniform on [0, 1) with seed 0: values from 1 to 255, the range of an
    8-bit image. deconwolf v0.4.5 refuses an image whose maximum is below 1 ("ERROR: The image has too low intensity,
    can not continue!", exit 1; ``src/dw.c``) or that holds a negative value, so the volume uniform on [0, 1) that
    this command wrote before made no wisdom (W-925's RD3, W-929). With 1 as the lower bound every voxel passes, so
    the check holds at any shape. FFTW's plans depend on the sizes, the threads and the options, not on the values, so
    the key, its fields and the store are those of the specification.
    """
    import tifffile
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    binary = deconwolf_binary({"deconwolf": binary})
    identity = deconwolf_identity(binary)
    fields = wisdom_fields(shape, psf_shape, border_quality, tile_size, threads, cpu_identity()["sha256"], identity)
    key = wisdom_key(fields)
    with exclusive(store / ".lock", timeout, "starfinder deconvolve wisdom"):
        manifest = _store_manifest(store)
        if key in manifest["keys"]:
            raise FileExistsError(f"the wisdom store {store} already holds the key {key}")
        tmp = Path(tempfile.mkdtemp(prefix="starfinder-wisdom-"))
        try:
            home = _private_home(tmp)
            rng = np.random.default_rng(0)
            volume = np.float32(1) + np.float32(254) * rng.random(tuple(shape), dtype=np.float32)
            tifffile.imwrite(home / "volume.tif", volume)
            grid = np.indices(tuple(psf_shape)).astype(np.float64)
            centre = np.array([(n - 1) / 2 for n in psf_shape]).reshape(3, 1, 1, 1)
            kernel = np.exp(-((grid - centre) ** 2).sum(axis=0) / 2)
            tifffile.imwrite(home / "psf.tif", (kernel / kernel.sum()).astype(np.float32))
            argv = [binary, "--iter", "1", "--threads", str(threads), "--offset", str(OFFSET)]
            if border_quality != 2:
                argv += ["--bq", str(border_quality)]
            if tile_size:
                argv += ["--tilesize", str(tile_size)]
            argv += ["--float", "--out", str(home / "out.tif"), str(home / "volume.tif"), str(home / "psf.tif")]
            status, output, wall, cpu = run_deconwolf(argv, home, threads)
            made = home / ".config" / "deconwolf" / wisdom_name(threads)
            if status != 0 or not made.is_file():
                raise RuntimeError(f"deconwolf exited with status {status} and wrote no {made.name}: "
                                   f"{output.decode(errors='replace')[-2000:]}")
            target = store / key / made.name
            with atomic_file(target) as tmp_file:
                shutil.copyfile(made, tmp_file)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        entry = {"fields": fields, "file": f"{key}/{made.name}", "bytes": target.stat().st_size,
                 "sha256": file_sha256(target), "made_at": _now(), "host": socket.gethostname(),
                 "job_id": os.environ.get("JOB_ID"), "command": argv, "wall_s": wall, "cpu_s": cpu}
        manifest["keys"][key] = entry
        _write_json(store / WISDOM_MANIFEST, manifest)
    return dict(entry, key=key)


def add_parser(groups):
    """The ``starfinder deconvolve`` group: ``wisdom``."""
    group = groups.add_parser("deconvolve", help="The deconvolution stage's FFTW wisdom store")
    commands = group.add_subparsers(dest="command", required=True)
    wisdom = commands.add_parser(
        "wisdom", help="Plan FFTW wisdom for one key on this node and add it to a store",
        description="Plan deconwolf's FFTs (FFTW_MEASURE) for one key (job shape, PSF shape, border quality, tile "
                    "size, threads, this CPU, the FFT backend and deconwolf) on a synthetic volume with values from 1 "
                    "to 255 (deconwolf refuses an image whose maximum is below 1), and add the file "
                    "and its manifest entry to the store under its lock. Exit 0 with the key added, 1 when the store "
                    "already holds the key or deconwolf fails (nothing written), 2 for invalid arguments.")
    wisdom.add_argument("--store", type=Path, required=True, help="The store's folder (execution.deconwolf_wisdom)")
    wisdom.add_argument("--shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"),
                        help="The image shape of the volumes the tasks deconvolve")
    wisdom.add_argument("--psf-shape", type=int, nargs=3, required=True, metavar=("Z", "Y", "X"),
                        help="The PSF's shape (deconvolution.psf.shape_zyx)")
    wisdom.add_argument("--threads", type=int, required=True, help="The tasks' threads")
    wisdom.add_argument("--deconwolf", help="The dw executable; default dw on PATH")
    wisdom.add_argument("--border-quality", type=int, default=2, help="deconvolution.border_quality (default 2)")
    wisdom.add_argument("--tile-size", type=int, help="deconvolution.tile_size (default untiled)")
    return group


def run_command(args):
    """Run ``starfinder deconvolve wisdom``; the exit status."""
    import sys
    try:
        entry = add_wisdom(args.store, args.shape, args.psf_shape, args.threads, binary=args.deconwolf,
                           border_quality=args.border_quality, tile_size=args.tile_size)
    except (FileExistsError, RuntimeError, ValueError, TimeoutError) as error:
        print(f"deconvolve wisdom: {error}", file=sys.stderr)
        return 1
    print(json.dumps({"key": entry["key"], "file": entry["file"], "sha256": entry["sha256"]}))
    return 0
