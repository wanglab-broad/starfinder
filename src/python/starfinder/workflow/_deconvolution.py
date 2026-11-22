"""The deconvolution stage in the plan: the block's defaults and refusals, the copy and its id, and the tasks.

design/specs/deconvolution.md. A present ``deconvolution`` block makes the plan add one ``psf`` task per run and one
``deconvolve`` task per (FOV, deconvolved round) that a consumer (``summarize``, ``fov``, ``tile_global``,
``morphology``) reads. Each consumer then declares the copy's files of its deconvolved rounds in place of the raw
files, needs the ``deconvolve`` tasks that make them and names their folders in its entry (``deconvolved``). A run
without the block plans what it planned before: nothing here runs.
"""
from __future__ import annotations

import hashlib
import warnings
from pathlib import Path, PurePosixPath

from starfinder import _layout
from starfinder.workflow._shape import canonical

#: The record formats of the stage: a round's record, the copy's record and the PSFs' record.
RECORD_FORMAT = "starfinder.deconvolution/1"
COPY_FORMAT = "starfinder.deconvolution.copy/1"
PSF_FORMAT = "starfinder.deconvolution.psf/1"
#: The round name of the reference stains (a deconvolved round in the reference round's folder).
REFERENCE_STAIN = "reference_stain"
#: The block's defaults, written into the resolved block (design/specs/deconvolution.md, "The configuration"); the
#: optics have none. ``scale`` has its default 64 with ``dtype: uint16`` only.
DEFAULTS = {"rounds": None, "copy_root": "output", "method": "shb", "border_quality": 2, "tile_size": None,
            "dtype": "uint8"}
PSF_DEFAULTS = {"source": "generate", "files": None, "voxel_size_um": None, "depth_um": 0.0,
                "shape_zyx": [21, 33, 33]}
DEFAULT_SCALE = 64
#: The optics a generated PSF needs besides each label's wavelengths.
OPTICS = ("na", "immersion_ri", "sample_ri", "pinhole_radius_nm")
DTYPES = ("uint8", "uint16")
#: deconwolf's padding of a tile on each side (``--tilepad``, 20 by default in v0.4.5).
TILE_PADDING = 20
#: A declared PSF spacing may differ from the dataset's by at most this fraction of the dataset's value.
SPACING_TOLERANCE = 1e-6
#: The folders of the copy: the PSFs and the rounds' records; ``copy.json`` at its top.
PSF_DIR = "psf"
RECORDS_DIR = "records"
COPY_FILE = "copy.json"
PSF_FILE = "psf.json"
RECORD_FILE = "deconvolution.json"


def _refuse(message):
    from starfinder.workflow._plan import PlanRefused
    raise PlanRefused(message)


def resolve_block(block, r):
    """The resolved ``deconvolution`` block: its defaults written out, ``scale`` resolved by ``dtype``; the refusals
    that need no input discovery (design/specs/deconvolution.md, "Plan refusals")."""
    if r.get("backend") == "matlab":
        _refuse("deconvolution: backend: matlab refuses the deconvolution block; the stage runs in the Python workflow "
                "only (MT1)")
    d = {**DEFAULTS, **dict(block or {})}
    iterations = d.get("iterations")
    if iterations is None:
        _refuse("deconvolution.iterations is required and has no default: choose it per dataset (\"Preparing a "
                "dataset\", the development-position rule)")
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations < 1:
        _refuse(f"deconvolution.iterations must be a positive integer; got {iterations!r}")
    if d["dtype"] not in DTYPES:
        _refuse(f"deconvolution.dtype must be uint8 or uint16; got {d['dtype']!r}")
    if d["dtype"] == "uint8":
        if d.get("scale") is not None:
            _refuse("deconvolution.scale is read with dtype: uint16 only; with uint8 each (FOV, round) has its own "
                    "stretch factor 255 / its maximum: remove scale or set dtype: uint16")
        d.pop("scale", None)
    else:
        scale = DEFAULT_SCALE if d.get("scale") is None else d["scale"]
        values = [v for labels in scale.values() for v in (labels or {}).values()] if isinstance(scale, dict) \
            else [scale]
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                _refuse(f"deconvolution.scale must be a positive number or {{<round>: {{<label>: number}}}} of "
                        f"positive numbers; got {value!r}")
        d["scale"] = scale
    copy_root = d["copy_root"]
    if not isinstance(copy_root, str) or not (copy_root in ("output", "beside") or Path(copy_root).is_absolute()):
        _refuse(f"deconvolution.copy_root must be output, beside or an absolute path; got {copy_root!r}")
    psf = {**PSF_DEFAULTS, **dict(d.get("psf") or {})}
    psf["shape_zyx"] = list(psf["shape_zyx"])
    d["psf"] = psf
    for key in ("voxel_size_xy", "voxel_size_z"):
        if r.get(key) is None:
            _refuse(f"deconvolution: {key} is required with a deconvolution block (the copy id and the PSF's spacing "
                    "read it)")
    if psf["source"] == "generate":
        missing = [f"deconvolution.psf.{key}" for key in OPTICS if psf.get(key) is None]
        if missing:
            _refuse(f"{', '.join(missing)}: a generated PSF needs the microscope's optics, which have no default")
    elif psf["source"] == "file":
        declared = psf.get("voxel_size_um")
        if not isinstance(declared, dict) or declared.get("xy") is None or declared.get("z") is None:
            _refuse("deconvolution.psf.voxel_size_um: psf.source file needs the files' declared spacing {xy, z} "
                    "in micrometres")
        for key, given in (("xy", r["voxel_size_xy"]), ("z", r["voxel_size_z"])):
            if abs(float(declared[key]) - float(given)) > SPACING_TOLERANCE * abs(float(given)):
                _refuse(f"deconvolution.psf.voxel_size_um.{key} {declared[key]} differs from voxel_size_{key} {given} "
                        f"by more than 1 part in 10^6: a PSF sampled on another grid than the images deconvolves "
                        "them wrongly")
    else:
        _refuse(f"deconvolution.psf.source must be generate or file; got {psf['source']!r}")
    return d


def _labels(entry):
    return entry if isinstance(entry, str) else entry["channel"]


def round_labels(r):
    """{round name: labels} of every round the stage may deconvolve: the sequencing rounds, the additional rounds and,
    with reference stains, ``reference_stain``."""
    from starfinder.workflow._plan import sequencing_rounds
    rounds = {name: [_labels(c) for c in r.get("seq_channel_order") or []] for name in sequencing_rounds(r)}
    rounds.update({e["round_name"]: [_labels(c) for c in e.get("channel_order") or []]
                   for e in r.get("additional_round") or []})
    if r.get("reference_stains"):
        rounds[REFERENCE_STAIN] = [_labels(c) for c in r["reference_stains"]]
    return rounds


def folder_round(r, name):
    """The round whose image folder holds a deconvolved round: the reference round's for ``reference_stain``."""
    return r["ref_round"] if name == REFERENCE_STAIN else name


def read_rounds(r):
    """The rounds the consumers read from the input root, per FOV (the same for every FOV), in the plan's order."""
    from starfinder.workflow._plan import sequencing_rounds, spot_finding_input, two_pass
    read = []
    p = r.get("processing")
    if p is not None and (spot_finding_input(r) is None or two_pass(r) or p.get("mode") == "tiled"):
        read += sequencing_rounds(r)
    if "morphology" in r:
        read += ([REFERENCE_STAIN] if r.get("reference_stains") else []) + list(r["morphology"]["rounds"])
    return list(dict.fromkeys(read))


def deconvolved_rounds(r):
    """The deconvolved rounds of a plan: ``rounds`` (null: the sequencing rounds) that a consumer reads. A listed round
    that names no round of the configuration, or that no consumer reads, is refused."""
    from starfinder.workflow._plan import sequencing_rounds
    d, labels, read = r["deconvolution"], round_labels(r), read_rounds(r)
    if d["rounds"] is None:
        return [name for name in sequencing_rounds(r) if name in read]
    if not isinstance(d["rounds"], list) or not d["rounds"]:
        _refuse(f"deconvolution.rounds must be null or a list of round names; got {d['rounds']!r}")
    unknown = [name for name in d["rounds"] if name not in labels]
    if unknown:
        _refuse(f"deconvolution.rounds names {', '.join(map(repr, unknown))}, which is no round of this configuration "
                f"(rounds {', '.join(labels)})")
    unread = [name for name in d["rounds"] if name not in read]
    if unread:
        _refuse(f"deconvolution.rounds names {', '.join(map(repr, unread))}, which no task of this plan reads "
                "(summarize, fov, tile_global and morphology read the rounds)")
    return list(dict.fromkeys(d["rounds"]))


def check_labels(r, rounds):
    """Refuse two labels that could be confused in one image folder of the copy: equal labels, or a label whose pattern
    ``*<label>*.tif`` matches another label's file ``<label>.tif`` (``io.load_round``)."""
    labels = round_labels(r)
    folders = {}
    for name in rounds:
        for label in labels[name]:
            folders.setdefault(folder_round(r, name), []).append((name, label))
    for folder, entries in folders.items():
        for i, (name, label) in enumerate(entries):
            for other, other_label in entries[i + 1:]:
                if label == other_label or label in other_label or other_label in label:
                    _refuse(f"deconvolution.rounds: the labels {label!r} ({name}) and {other_label!r} ({other}) are "
                            f"written into one image folder of the copy ({folder}), where the pattern *<label>*.tif "
                            "of one matches the other's file; rename a label or leave a round undeconvolved")


def configured_rounds(r):
    """The rounds the block names (null: the sequencing rounds), whatever the consumers of this plan read."""
    from starfinder.workflow._plan import sequencing_rounds
    return list(dict.fromkeys(r["deconvolution"]["rounds"] or sequencing_rounds(r)))


def psf_labels(r, rounds):
    """The labels of ``rounds``, in order. The copy holds the PSF of every label of the configured rounds
    (``configured_rounds``), so that its id and PSFs do not depend on which rounds a plan's consumers read."""
    labels = round_labels(r)
    return list(dict.fromkeys(label for name in rounds for label in labels[name]))


def psf_file_path(r, value):
    """A supplied PSF file: absolute, or relative to the input root."""
    from starfinder.workflow._plan import input_root
    path = Path(value)
    return path if path.is_absolute() else input_root(r) / path


def check_psf(r, rounds):
    """The PSF's refusals that need the labels of the configured rounds: the wavelengths of each label (generate), or a
    file per label that exists (file); and a tile size that holds the PSF and deconwolf's padding."""
    d, psf, labels = r["deconvolution"], r["deconvolution"]["psf"], psf_labels(r, configured_rounds(r))
    if psf["source"] == "generate" and psf["shape_zyx"][1] != psf["shape_zyx"][2]:
        _refuse(f"deconvolution.psf.shape_zyx {psf['shape_zyx']}: a generated PSF is square in Y and X (psfmodels' nx)")
    if psf["source"] == "generate":
        channels = psf.get("channels") or {}
        missing = [label for label in labels if not isinstance(channels.get(label), dict)
                   or channels[label].get("excitation_nm") is None or channels[label].get("emission_nm") is None]
        if missing:
            _refuse(f"deconvolution.psf.channels: psf.source generate needs excitation_nm and emission_nm of the "
                    f"labels {', '.join(map(repr, missing))}, which a deconvolved round reads")
    else:
        files = psf.get("files") or {}
        missing = [label for label in labels if files.get(label) is None]
        if missing:
            _refuse(f"deconvolution.psf.files: psf.source file needs a file for the labels "
                    f"{', '.join(map(repr, missing))}, which a deconvolved round reads")
        absent = [str(psf_file_path(r, files[label])) for label in labels if not psf_file_path(r, files[label]).is_file()]
        if absent:
            _refuse(f"deconvolution.psf.files: the files {', '.join(absent)} do not exist")
    if d["tile_size"] is not None:
        if psf["source"] == "file":
            # A supplied PSF's size is its file's (the header alone is read).
            import tifffile
            shapes = []
            for label in labels:
                with tifffile.TiffFile(psf_file_path(r, psf["files"][label])) as tif:
                    shapes.append(tif.series[0].shape)
            lateral = max(max(shape[-2:]) for shape in shapes)
        else:
            lateral = max(psf["shape_zyx"][1:])
        minimum = lateral + 2 * TILE_PADDING
        if isinstance(d["tile_size"], bool) or not isinstance(d["tile_size"], int) or d["tile_size"] < minimum:
            _refuse(f"deconvolution.tile_size {d['tile_size']!r} is below the PSF's lateral size {lateral} plus "
                    f"deconwolf's padding of {TILE_PADDING} on each side ({minimum})")
    if d["dtype"] == "uint16" and isinstance(d["scale"], dict):
        all_labels = round_labels(r)
        missing = [f"{name}/{label}" for name in rounds for label in all_labels[name]
                   if label not in (d["scale"].get(name) or {})]
        if missing:
            _refuse(f"deconvolution.scale: the mapping misses the deconvolved (round, label) {', '.join(missing)}")


def copy_fields(r, labels):
    """The fields the copy id hashes (design/specs/deconvolution.md, "Terms"): the dataset, sample and layout, the
    spacing, the resolved block less ``copy_root`` and the SHA-256 of each supplied PSF file of ``labels`` (null with
    generate)."""
    from starfinder._records import file_sha256
    block = {k: v for k, v in r["deconvolution"].items() if k != "copy_root"}
    psf = r["deconvolution"]["psf"]
    files = None
    if psf["source"] == "file":
        files = {label: file_sha256(psf_file_path(r, psf["files"][label])) for label in labels}
    return {"dataset_id": r["dataset_id"], "sample_id": r["sample_id"], "image_dir_template": r["image_dir_template"],
            "voxel_size_xy": r["voxel_size_xy"], "voxel_size_z": r["voxel_size_z"], "deconvolution": block,
            "psf_files": files}


def copy_id(fields):
    """The first 12 hexadecimal digits of the SHA-256 of the fields' canonical JSON."""
    return hashlib.sha256(canonical(fields)).hexdigest()[:12]


def copy_dir(r, out, identifier):
    """The copy folder by ``copy_root``: under the output root, beside the dataset's folder, or under a given path."""
    root = r["deconvolution"]["copy_root"]
    name = f"dw-{identifier}"
    if root == "output":
        return Path(out) / "deconvolved" / r["sample_id"] / name
    if root == "beside":
        return Path(r["root_input_path"]) / f"{r['dataset_id']}-dw" / r["sample_id"] / name
    return Path(root) / r["dataset_id"] / r["sample_id"] / name


def image_dir(r, name, fov):
    """The image folder of a deconvolved round inside the copy (POSIX, relative to the copy)."""
    return PurePosixPath(_layout.check_image_dir_template(r["image_dir_template"]).format(
        round=folder_round(r, name), fov=fov)).as_posix()


def record_dir(copy, name, fov):
    """``<copy>/records/<round name>/<fov>``: the folder only the task of (round name, FOV) writes."""
    return Path(copy) / RECORDS_DIR / name / fov


def _threshold_modes(value):
    """Every ``threshold_mode`` a spot-finding block names, at any depth."""
    if isinstance(value, dict):
        found = [value["threshold_mode"]] if isinstance(value.get("threshold_mode"), str) else []
        return found + [m for v in value.values() for m in _threshold_modes(v)]
    if isinstance(value, list):
        return [m for v in value for m in _threshold_modes(v)]
    return []


def intensity_warnings(r, rounds):
    """The plan's warnings for a uint8 copy whose rounds feed what compares intensities across FOVs: a two-pass
    statistic (``fit: supplied``), or a ``global`` or ``absolute`` spot threshold (one warning each)."""
    from starfinder.workflow._plan import sequencing_rounds, supplied_steps
    if r["deconvolution"]["dtype"] != "uint8" or not set(rounds) & set(sequencing_rounds(r)):
        return []
    found = []
    if supplied_steps(r):
        found.append("a preprocessing step with fit: supplied (the two-pass statistics)")
    modes = _threshold_modes((r.get("processing") or {}).get("spot_finding"))
    for mode in ("global", "absolute"):
        if mode in modes:
            found.append(f"spot finding's {mode} threshold")
    return [f"deconvolution: the copy is uint8 with one stretch factor per (FOV, round), so {what} compares each FOV "
            "at its own factor; set dtype: uint16 for one fixed scale across FOVs" for what in found]


def check_stage(r):
    """The stage's refusals that read the configuration alone, before input discovery: the rounds, the labels of the
    copy's image folders and the PSF; the deconvolved rounds."""
    rounds = deconvolved_rounds(r)
    check_labels(r, rounds)
    check_psf(r, rounds)
    return rounds


class Stage:
    """The stage of one plan: the deconvolved rounds, the copy and its id, and the tasks it adds to the graph."""

    def __init__(self, r, graph):
        self.r, self.graph = r, graph
        self.rounds = check_stage(r)
        self.labels = round_labels(r)
        self.psf_labels = psf_labels(r, configured_rounds(r))
        self.fields = copy_fields(r, self.psf_labels)
        self.copy_id = copy_id(self.fields)
        self.copy = copy_dir(r, graph.out, self.copy_id)
        self.tasks = {}
        for message in intensity_warnings(r, self.rounds):
            warnings.warn(message, UserWarning, stacklevel=4)

    def rel(self, path):
        return self.graph.rel(path)

    def psf_outputs(self):
        folder = self.copy / PSF_DIR
        return [folder / f"{label}.tif" for label in self.psf_labels] + [folder / PSF_FILE,
                                                                                         self.copy / COPY_FILE]

    def manifest_entry(self):
        """The manifest's ``deconvolution``: the copy (relative to the output root, or absolute), its id and rounds."""
        from starfinder.workflow._plan import sequencing_rounds
        configured = self.r["deconvolution"]["rounds"] or sequencing_rounds(self.r)
        return {"copy": self.rel(self.copy), "copy_id": self.copy_id, "rounds": list(self.rounds),
                "sequencing": any(name in sequencing_rounds(self.r) for name in configured)}

    def add_tasks(self, samples, reads):
        """The psf task and one deconvolve task per (FOV, deconvolved round) that a consumer reads (``reads``: the
        rounds the consumers read)."""
        g, r = self.graph, self.r
        psf = r["deconvolution"]["psf"]
        inputs = []
        if psf["source"] == "file":
            inputs = [g.inputs.add_file(f"psf.{label}", psf_file_path(r, psf["files"][label]),
                                        f"deconvolution.psf.files.{label}") for label in self.psf_labels]
        entry = {"copy": self.rel(self.copy), "copy_id": self.copy_id, "labels": self.psf_labels}
        self.psf_task = g.add("psf", {}, [], inputs, self.psf_outputs(), extra={"deconvolution": entry})
        gpu = bool((r["execution"]["tasks"].get("deconvolve") or {}).get("gpu"))
        psf_json = self.rel(self.copy / PSF_DIR / PSF_FILE)
        for sample, members in samples.items():
            for fov in members:
                for name in self.rounds:
                    if name not in reads or name not in g.inputs.images[fov]:
                        continue
                    labels = self.labels[name]
                    folder = image_dir(r, name, fov)
                    records = record_dir(self.copy, name, fov)
                    outputs = [self.copy / folder / f"{label}.tif" for label in labels]
                    outputs += [records / RECORD_FILE] + [records / f"{label}.dw.log" for label in labels]
                    inputs = [f"input:{c['path']}" for c in g.inputs.images[fov][name].values()]
                    inputs += [self.rel(self.copy / PSF_DIR / f"{label}.tif") for label in labels] + [psf_json]
                    entry = {"copy": self.rel(self.copy), "copy_id": self.copy_id, "round": name,
                             "folder_round": folder_round(r, name), "labels": labels, "image_dir": folder}
                    self.tasks[(fov, name)] = g.add(
                        "deconvolve", {"sample": sample, "fov": fov, "round": name}, [self.psf_task], inputs, outputs,
                        group=f"deconvolve.{name}", gpu=gpu, extra={"deconvolution": entry})

    def images(self, fov, names):
        """(declared inputs, needs, ``deconvolved``) of a consumer that reads ``names`` of a FOV: the copy's label
        files and the round's record for a deconvolved round, the raw files otherwise."""
        g = self.graph
        inputs, needs, folders = [], [], {}
        for name in names:
            task = self.tasks.get((fov, name))
            if task is None:
                inputs += g.inputs.image_inputs(fov, [name])
                continue
            entry = g.by_id[task]
            inputs += [o for o in entry["outputs"] if o.endswith(".tif") or o.endswith(RECORD_FILE)]
            needs.append(task)
            folders[name] = self.rel(self.copy / image_dir(self.r, name, fov))
        return inputs, needs, folders
