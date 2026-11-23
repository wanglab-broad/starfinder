"""The fixtures of design/validation/deconvolution.md ("Fixtures"): the stand-in ``dw``, ``input_root`` and
``input_root_stored``, the wisdom store and the PSF arrays, with the helpers that plan and run the stage.

``stand_in_dw`` is a Python script installed as ``dw`` in a folder and named by ``execution.deconwolf``. It accepts
the options of deconwolf v0.4.5 that the stage passes, writes the float32 output ``y = x + 5`` untiled and ``y = x``
tiled (plus ``shift`` when given), a log with ``scaling:`` and ``Iteration`` lines beside it, and, unless ``--noplan``
or ``--gpu`` is given, appends to ``fftw_wisdom_float_inplace_threads_<n>.dat`` where v0.4.5 would write it. Like
v0.4.5, it refuses an input whose maximum is below ``MIN_INTENSITY`` (1) with v0.4.5's message and exit status 1,
writing nothing. Each call
appends its arguments, ``HOME``, ``OMP_NUM_THREADS``, its working directory and the wisdom files it found to a call
log. ``input_root`` is 2 FOVs × 2 rounds × 2 labels of 8 × 32 × 32 uint8 with sparse Gaussian spots and the reference
stain ``dapi`` in the reference round's folder; ``stored=True`` writes each channel with ``save_volume`` and one
``ImageMetadata`` per round folder.
"""
from __future__ import annotations

import hashlib
import json
import stat
import sys
from pathlib import Path

import numpy as np
import tifffile

DATASET_ID = "dw-tiny"
SAMPLE = "sample"
FOVS = ("Position001", "Position002")
ROUNDS = ("round1", "round2")
LABELS = ("ch00", "ch01")
STAIN = "dapi"
#: The additional round that the morphology block registers by its shared stain (the configuration names it only
#: with a morphology block).
OTHER = "other"
OTHER_LABELS = (STAIN, "ch02")
SHAPE = (8, 32, 32)
VOXEL_XY, VOXEL_Z = 0.1, 0.3
#: deconwolf v0.4.5 refuses an input image whose maximum is below 1 (``src/dw.c``: "ERROR: The image has too low
#: intensity, can not continue!", exit 1); the stand-in does the same unless asked not to (W-929).
MIN_INTENSITY = 1.0

STAND_IN = r'''#!{python}
"""A stand-in for deconwolf v0.4.5 (design/validation/deconvolution.md, fixture stand_in_dw)."""
import json, os, sys, time
from pathlib import Path

OPTIONS = {options}
args = sys.argv[1:]
if args == ["--version"]:
    print(f"deconwolf: '{{OPTIONS['version']}}'\nBUILD_DATE: 'stand-in'\nFFT Backend: 'fftw-3.3.8-sse2-avx'\n"
          "OpenMP: YES\nOpenCL: YES\nVkFFT: YES")
    sys.exit(0)
if args == ["--help"]:
    listed = ["--iter", "--threads", "--gpu", "--tilesize", "--noplan", "--offset", "--out", "--bq"]
    print("usage: dw [<options>] image.tif psf.tif\n" + "\n".join(listed + ([] if OPTIONS["no_float"] else ["--float"])))
    sys.exit(0)
values, flags, positional = {{}}, set(), []
i = 0
while i < len(args):
    if args[i] in ("--iter", "--threads", "--tilesize", "--offset", "--out", "--bq"):
        values[args[i]] = args[i + 1]
        i += 2
    elif args[i] in ("--gpu", "--noplan", "--float"):
        flags.add(args[i])
        i += 1
    elif args[i].startswith("--"):
        print(f"unknown option {{args[i]}}", file=sys.stderr)
        sys.exit(2)
    else:
        positional.append(args[i])
        i += 1
home = Path(os.environ.get("HOME", ""))
folder = home / ".config" / "deconwolf" if (home / ".config").is_dir() else Path.cwd()
found = sorted(str(p) for p in folder.glob("fftw_wisdom_*.dat"))
call = {{"argv": args, "HOME": os.environ.get("HOME"), "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
        "cwd": os.getcwd(), "wisdom_found": found,
        "wisdom_found_sha256": {{p: __import__("hashlib").sha256(Path(p).read_bytes()).hexdigest() for p in found}}}}
log = Path(OPTIONS["call_log"])
with open(log, "a") as handle:
    handle.write(json.dumps(call) + "\n")
image, psf = positional
import numpy as np
import tifffile
if OPTIONS["fail_on"] and OPTIONS["fail_on"] in Path(image).name:
    print("stand-in: failing as asked", file=sys.stderr)
    sys.exit(3)
time.sleep(OPTIONS["delay"])
x = tifffile.imread(image).astype(np.float32)
if OPTIONS["min_intensity"] is not None and float(x.max()) < OPTIONS["min_intensity"]:
    print("ERROR: The image has too low intensity, can not continue!", file=sys.stderr)
    sys.exit(1)
tiled = "--tilesize" in values
y = x + np.float32((0 if tiled or OPTIONS["min_zero"] else 5) + OPTIONS["shift"])
out = Path(values["--out"])
if "--float" in flags:
    tifffile.imwrite(out, y.astype(np.float32))
    scaling = 1.0
else:
    scaling = 65535.0 / float(y.max()) if y.max() > 0 else 1.0
    tifffile.imwrite(out, np.clip(np.rint(y * scaling), 0, 65535).astype(np.uint16))
lines = [f"Iteration {{k:3d}}/{{int(values['--iter']):3d}}, Idiv=0.0" for k in range(1, int(values["--iter"]) + 1)]
lines += [f"scaling: {{scaling}}"]
if "--gpu" in flags:
    lines.append("Device name: stand-in OpenCL device")
Path(str(out) + ".log.txt").write_text("\n".join(lines) + "\n")
print("stand-in deconwolf done")
if "--noplan" not in flags and "--gpu" not in flags:
    with open(folder / f"fftw_wisdom_float_inplace_threads_{{values['--threads']}}.dat", "a") as handle:
        handle.write("stand-in wisdom\n")
'''


def stand_in(folder, *, version="0.4.5", fail_on=None, min_zero=False, shift=0, no_float=False, delay=0.0,
             name="dw", min_intensity=MIN_INTENSITY):
    """Install the stand-in ``dw`` in ``folder``; (its path, its call log). ``min_intensity=None`` accepts any
    input."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    call_log = folder / f"{name}.calls.jsonl"
    options = {"version": version, "fail_on": fail_on, "min_zero": min_zero, "shift": shift, "no_float": no_float,
               "delay": delay, "call_log": str(call_log), "min_intensity": min_intensity}
    path = folder / name
    path.write_text(STAND_IN.format(python=sys.executable, options=repr(options)))
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path, call_log


def calls(call_log):
    """The stand-in's calls, one dict per call (none when it was never called)."""
    path = Path(call_log)
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def spots(seed, shape=SHAPE, n=6, peak=200):
    """Sparse Gaussian spots of sigma 1 voxel in uint8, from ``seed``."""
    rng = np.random.default_rng(seed)
    image = np.zeros(shape, dtype=np.float64)
    grid = np.indices(shape).astype(np.float64)
    for _ in range(n):
        centre = [rng.uniform(1, s - 2) for s in shape]
        image += peak * rng.uniform(0.5, 1.0) * np.exp(-sum((g - c) ** 2 for g, c in zip(grid, centre)) / 2)
    return np.clip(np.rint(image), 0, 255).astype(np.uint8)


def input_root(work, *, stored=False):
    """Write ``input_root`` (or ``input_root_stored``) under ``work/input``; the input root (dataset/sample)."""
    from starfinder.image import ImageMetadata
    from starfinder.io import save_volume
    root = Path(work) / "input" / DATASET_ID / SAMPLE
    root.mkdir(parents=True, exist_ok=True)
    (root / "fovs.csv").write_text("sample,fov_id\n" + "".join(f"{SAMPLE},{fov}\n" for fov in FOVS))
    (root / "genes.csv").write_text("gene,barcode\nGeneA,CC\nGeneB,CA\n")
    seed = 0
    for name in (*ROUNDS, OTHER):
        for fov in FOVS:
            folder = root / name / fov
            folder.mkdir(parents=True, exist_ok=True)
            labels = OTHER_LABELS if name == OTHER else list(LABELS) + ([STAIN] if name == ROUNDS[0] else [])
            metadata = ImageMetadata(f"raw:{name}/{fov}", spacing_zyx=(VOXEL_Z, VOXEL_XY, VOXEL_XY),
                                     origin_zyx=(1.0, 2.0, 3.0), spatial_unit="micrometer")
            for label in labels:
                seed += 1
                image = spots(seed)
                path = folder / f"{fov}_{label}.tif"
                if stored:
                    save_volume(image, path, metadata=metadata)
                else:
                    tifffile.imwrite(path, image)
    return root


def configuration(work, output_id, *, deconvolution=None, dw=None, processing=True, morphology=False, **extra):
    """A configuration of ``input_root``: the reference round ``round1``, the stain ``dapi``, a processing block that
    registers and finds spots (no decoding), the morphology block when asked, and the deconvolution block when given
    (with ``execution.deconwolf`` the stand-in)."""
    work = Path(work)
    config = {"backend": "python", "root_input_path": str(work / "input"), "dataset_id": DATASET_ID,
              "sample_id": SAMPLE, "root_output_path": str(work / "output"), "output_id": output_id,
              "n_rounds": len(ROUNDS), "ref_round": ROUNDS[0], "ref_channel": "DAPI",
              "seq_channel_order": list(LABELS), "reference_stains": [{"channel": STAIN, "name": "DAPI"}],
              "rotate_angle": 0, "voxel_size_xy": VOXEL_XY, "voxel_size_z": VOXEL_Z,
              "execution": {"tasks": {"fov": {"threads": 1}}}}
    if processing:
        config["processing"] = {"registration": {"steps": [{"method": "translation"}]},
                                "spot_finding": {"method": "local_maxima", "exclude_border": False}}
    if morphology:
        config["morphology"] = {}
        config["additional_round"] = [{"round_name": OTHER, "channel_order": [
            {"channel": STAIN, "name": "DAPI"}, {"channel": "ch02", "name": "Other"}]}]
    if deconvolution is not None:
        config["deconvolution"] = deconvolution
    if dw is not None:
        config["execution"]["deconwolf"] = str(dw)
    for key, value in extra.items():
        if key == "execution":
            config["execution"] = {**config["execution"], **value}
        else:
            config[key] = value
    return config


def block(**keys):
    """A deconvolution block with W-855's optics for the labels of ``input_root`` (iterations 3)."""
    optics = {"na": 1.30, "immersion_ri": 1.518, "sample_ri": 1.338, "pinhole_radius_nm": 307,
              "shape_zyx": [5, 9, 9],
              "channels": {"ch00": {"excitation_nm": 499, "emission_nm": 529},
                           "ch01": {"excitation_nm": 598, "emission_nm": 634},
                           "dapi": {"excitation_nm": 405, "emission_nm": 461},
                           "ch02": {"excitation_nm": 557, "emission_nm": 577}}}
    out = {"iterations": 3, "psf": dict(optics, **keys.pop("psf", {}))}
    out.update(keys)
    return out


def psf_small(shift=(0, 0, 0)):
    """``psf_small``: a 7 × 9 × 9 float32 Gaussian with full support, peak at (3, 4, 4), rolled by ``shift``."""
    z, y, x = np.indices((7, 9, 9)).astype(np.float64)
    g = np.exp(-(z - 3) ** 2 / 2 - ((y - 4) ** 2 + (x - 4) ** 2) / (2 * 1.5 ** 2))
    return np.roll((g / g.sum()).astype(np.float32), shift, axis=(0, 1, 2))


def psf_core(shift=(0, 0, 0)):
    """``psf_core``: psf_small's Gaussian on the core z 2–4, y and x 2–6, exactly 0 elsewhere, rolled by ``shift``."""
    z, y, x = np.indices((7, 9, 9)).astype(np.float64)
    g = np.exp(-(z - 3) ** 2 / 2 - ((y - 4) ** 2 + (x - 4) ** 2) / (2 * 1.5 ** 2))
    core = np.zeros_like(g)
    core[2:5, 2:7, 2:7] = g[2:5, 2:7, 2:7]
    return np.roll((core / core.sum()).astype(np.float32), shift, axis=(0, 1, 2))


def write_psf_files(folder, labels=(*LABELS, STAIN, "ch02"), array=None):
    """``psf_small`` (or ``array``) as one float32 ZYX file per label; {label: path}."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    out = {}
    for label in labels:
        path = folder / f"psf_{label}.tif"
        tifffile.imwrite(path, psf_small() if array is None else array, metadata={"axes": "ZYX"})
        out[label] = str(path)
    return out


def file_block(files, **keys):
    """A deconvolution block with ``psf.source: file`` and the files' declared spacing equal to the dataset's."""
    return {"iterations": 3, "psf": {"source": "file", "files": files,
                                     "voxel_size_um": {"xy": VOXEL_XY, "z": VOXEL_Z}}, **keys}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tree(folder):
    """{relative path: (SHA-256, bytes, mtime_ns)} of every file under ``folder``."""
    folder = Path(folder)
    return {p.relative_to(folder).as_posix(): (sha256(p), p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(folder.rglob("*")) if p.is_file()}


def write_config(work, name, config):
    import yaml
    path = Path(work) / "configs" / f"{name}.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path


def plan(work, name, config):
    """Plan ``config``; (manifest, manifest path)."""
    from starfinder.workflow import plan as plan_
    manifest = plan_(write_config(work, name, config))
    return manifest, Path(manifest["roots"]["output"]) / "workflow" / "manifest.json"


def run(manifest_path, task_id, *extra):
    """Run one task's command in this process; its exit status."""
    from starfinder.__main__ import main
    from starfinder.workflow._shape import COMMANDS
    kind = next(t["kind"] for t in json.loads(Path(manifest_path).read_text())["tasks"] if t["id"] == task_id)
    return main(["workflow", COMMANDS[kind], "--manifest", str(manifest_path), "--task", task_id, *extra])


def run_kinds(manifest, manifest_path, *kinds):
    """Run the tasks of ``kinds`` (all when none) in the manifest's order; {task id: exit status}."""
    return {t["id"]: run(manifest_path, t["id"]) for t in manifest["tasks"] if not kinds or t["kind"] in kinds}


def copy_folder(manifest):
    """The copy folder of a planned run."""
    stage = manifest["deconvolution"]
    path = Path(stage["copy"])
    return path if path.is_absolute() else Path(manifest["roots"]["output"]) / path
