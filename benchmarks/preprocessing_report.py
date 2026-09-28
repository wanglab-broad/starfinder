"""Render the §2.5 inspection report (task group 6, W-234) from saved task-group-5 outputs.

Reads the W-233 evaluation directory written by preprocessing_synthetic.py
(manifest.json, tables/ and curves/) and writes one standalone HTML file with
inline figures and styles. It supports human review; it is not scientific
acceptance and gives no recommended defaults.

Before rendering it checks that
  * every file listed in the manifest has the recorded size and sha256;
  * the manifest's revision matches this checkout: the recorded commit is an
    ancestor of HEAD, a dirty run's uncommitted-diff checksum equals the diff
    from that commit to a later commit, and the evaluation sources are unchanged
    since that commit.

Every table value is copied from a manifest-listed file named in the report.
No table value is recomputed. The before/after panels need images, which the
evaluation does not save, so the renderer regenerates the displayed scene from
its recorded configuration and preprocesses it with the recorded recipe, then
requires the scene and output sha256 to equal the manifest's records. No
detection, sweep or evaluation is rerun.

    uv run python ../../benchmarks/preprocessing_report.py --evaluation <W-233 run>/evaluation \
        --output <run dir>/report.html
"""
import argparse
import base64
from contextlib import contextmanager
import hashlib
import html
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def _load_harness():
    spec = importlib.util.spec_from_file_location("preprocessing_synthetic", HERE / "preprocessing_synthetic.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


harness = _load_harness()

SCHEMA = "starfinder.benchmark.preprocessing_report/1"
#: Paths whose content determines the saved outputs; they must not change after the evaluated revision.
EVALUATION_SOURCES = ("benchmarks/preprocessing_synthetic.py", "src/python/starfinder")
PANEL_DTYPE = "uint8"
PANEL_SEED = harness.EVAL_SEEDS[0]
#: Display-only vertical stretch of XZ views (the presets record no voxel spacing).
XZ_STRETCH = 2


class RevisionMismatch(RuntimeError):
    """The manifest's source revision does not match this checkout."""


class SourceMismatch(RuntimeError):
    """A saved output differs from its manifest record."""


# --- Report text -----------------------------------------------------------------------

#: One sentence per displayed condition.
CONDITIONS = {
    "clean": "Puncta alone, with unit readout, no background and no noise; it is the harm check for every method.",
    "background_only": "The combined preset's background (baseline, X gradient, one tissue region and texture blobs) "
                       "without readout effects or noise.",
    "round_effect_only": "The combined preset's readout effects (weakening, round trend, gain and channel mixing) "
                         "without background or noise.",
    "combined": "Readout effects, background and dependent plus independent noise together, without geometry; it is "
                "the combined condition of every comparison.",
    "baseline": "A constant offset per channel and round, the targeted condition of scalar background subtraction.",
    "gradient": "A linear background ramp along X and Z (gradient_slopes_zyx=(2, 0, 2)), the Z-varying background "
                "targeted by the 3D and XY background methods.",
    "regions": "One Gaussian tissue region of height 3 and sigma (2, 5, 5) voxels in ZYX, weighted per channel, as "
               "background with extent along Z.",
    "texture": "Two texture blobs weighted per channel; they are dimmer than the puncta, so they only approximate "
               "bright outliers.",
    "gain": "A readout gain of 0.5 in every channel and round, which dims every punctum by the same factor.",
    "trend": "A round trend (trend_base=0.5) that changes punctum brightness from round to round.",
    "gain_baseline": "The gain condition together with the baseline offset, a target of the pre-normalization "
                     "extraction source.",
    "gain_texture": "The gain condition together with the texture blobs, the second target of the "
                    "pre-normalization extraction source.",
    "combined_geometry": "The combined condition plus the preset's translation and local deformation, processed "
                         "with global translation registration.",
    "mf_density": "Three FOVs with the combined appearance and 80, 20 and 2 amplicons (dense, sparse and "
                  "near-empty), the targeted set of sample-level fitting.",
    "mf_gain_drift": "Three FOVs of 80 amplicons with the combined appearance and readout gain times 1.0, 0.75 and "
                     "0.5, the secondary sample-level target.",
    "mf_density_clean": "The density set with the clean appearance, the harm check of sample-level fitting.",
}

#: Method and mode sections, in report order. Text follows docs/preprocessing-algorithms.md and
#: docs/preprocessing-baseline.md. The panel is the isolated comparison on the first targeted
#: condition unless stated.
METHODS = {
    "scalar_background": dict(
        title="Scalar background subtraction",
        problem="A constant additive offset per channel and round biases the channel ratios used for colour "
                "calling, so a channel with a higher offset looks partly on where it should be off.",
        cause="The baseline term: camera offset, export offset or near-uniform nonspecific background.",
        why_small="Post-Huygens exports may already have their background subtracted, and the low percentile of "
                  "percentile normalization already subtracts an offset.",
        panel=dict(condition="baseline", before="none", after="scalar")),
    "background_3d": dict(
        title="3D background subtraction",
        problem="Spatially varying background that also changes along Z; slice-wise XY methods can jump between "
                "slices and ignore Z sampling.",
        cause="Thick-tissue autofluorescence and out-of-focus light from neighbouring planes.",
        why_small="Deconvolution already removes most out-of-focus light, and if the background varies mainly in "
                  "XY the 3D and XY methods give similar results; the cost is also high.",
        panel=dict(condition="gradient", before="none", after="bg3d")),
    "percentile_normalization": dict(
        title="Percentile normalization",
        problem="Channel and round gain differences, as for min–max, but robust to bright outliers and preserving "
                "the input dtype.",
        cause="The gain term, plus bright outliers.",
        why_small="Without bright outliers it differs from min–max mainly in dtype and in the p_low offset.",
        panel=dict(condition="gain", before="none", after="pct")),
    "extraction_source": dict(
        title="Pre-normalization extraction source (mode)",
        problem="Normalization changes the intensities used for colour calling; extracting from the "
                "background-corrected image before normalization keeps the linear channel ratios.",
        cause="Nonlinearity and saturation introduced by the intensity step.",
        why_small="Intensities extracted before normalization are not corrected for channel gain, and whether that "
                  "helps depends on how the decoder handles channel scale.",
        panel=dict(condition="gain_baseline", before="r2_scalar", after="r2_scalar_xsrc", snapshot=True),
        panel_note="The panels compare the image each arm extracts from: the detection image of r2_scalar and the "
                   "bg_corrected snapshot that r2_scalar_xsrc reads. Detection images are identical between the arms."),
    "sample_level_fitting": dict(
        title="Sample-level fitting (mode)",
        problem="Statistics fitted per FOV depend on its content: in a near-empty FOV the high percentile falls in "
                "the noise, so normalization stretches noise over the full range.",
        cause="FOV content: tissue density and near-empty fields at tissue edges.",
        why_small="When FOVs have similar content, per-FOV and sample-level statistics coincide.",
        panel=dict(condition="mf_density", before="r2_scalar", after="r2_scalar_sample", fov_role="near_empty"),
        panel_note="The panels show the near-empty FOV of the density set, fitted per FOV (r2_scalar) and with "
                   "sample-level statistics merged over the three FOVs (r2_scalar_sample)."),
    "min_max_normalization": dict(
        title="Min–max normalization (baseline method)",
        problem="Channel and round gain differences.",
        cause="The gain term.",
        why_small="The range is set by the single brightest voxel, the output is forced to uint8 and the background "
                  "is compressed into few grey levels.",
        panel=dict(condition="gain", before="none", after="minmax")),
    "histogram_matching": dict(
        title="Histogram matching (baseline method)",
        problem="Channel intensity distributions that differ in shape.",
        cause="The gain term and channel-specific background.",
        why_small="It assumes every channel should share one intensity distribution, which holds only when barcode "
                  "bases are balanced across channels; the development codebook is not balanced.",
        panel=dict(condition="gain", before="none", after="hist")),
    "reconstruction": dict(
        title="XY morphological reconstruction (baseline method)",
        problem="Spatial background in XY.",
        cause="The spatial background term.",
        why_small="It zeroes most voxels, so MAD and the noise threshold can reach 0.",
        panel=dict(condition="regions", before="none", after="recon")),
    "white_tophat": dict(
        title="XY white top-hat (baseline method)",
        problem="Spatial background in XY.",
        cause="The spatial background term.",
        why_small="It works slice by slice with radii in pixels, ignoring Z sampling, and no agreed recipe contains "
                  "it, so it has only the isolated comparison.",
        panel=dict(condition="regions", before="none", after="tophat")),
}

#: Recipes shown on the combined preset: arm, label.
COMBINED_RECIPES = (("none", "no preprocessing"), ("r1", "recipe 1: min–max → histogram matching"),
                    ("r1_recon", "recipe 1 with reconstruction"),
                    ("r2_scalar", "recipe 2: scalar background → percentile"),
                    ("r2_3d", "recipe 2: 3D background → percentile"))

LIMITATIONS = (
    "Synthetic presets only: every value comes from uncalibrated §2.12 development presets, and no real data were "
    "used.",
    "Uncalibrated backgrounds: the baseline, gradient, region and texture amplitudes are development choices, not "
    "measured tissue backgrounds, and the texture blobs are dimmer than puncta, so there is no bright-outlier "
    "fixture.",
    "uint8 export scaling unconfirmed: the float presets were scaled uniformly (×12 for uint8, ×192 for uint16) "
    "before the cast; whether catalog uint8 exports have this dynamic range is not confirmed.",
    "No recommended default: the pipeline default remains recipe 1 and no recipe is recommended; E13 decides on "
    "real data.",
    "The 2-percentage-point low-benefit threshold is provisional and may be revised; a flag informs review and does "
    "not remove a method.",
    "Scenes are 16×64×64 voxels (reduction 1 of the batch guidance) with a development codebook of two genes and "
    "three rounds.",
)


# --- Identity checks ------------------------------------------------------------------------

def _git(repo, *args, binary=False):
    result = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, check=False)
    if result.returncode != 0:
        raise RevisionMismatch(f"git {' '.join(args)} failed: {result.stderr.decode(errors='replace').strip()}")
    return result.stdout if binary else result.stdout.decode().strip()


def check_revision(software, repo=ROOT):
    """Resolve the manifest's source identity to a commit of this checkout, or raise RevisionMismatch.

    A clean run's revision is used as is. A dirty run records the sha256 of
    ``git diff HEAD --binary`` at run time; the evaluated source is the first
    commit after the recorded revision (on the ancestry path to HEAD) whose diff
    from it has that checksum. The evaluation sources must be unchanged from that
    commit to the working tree.
    """
    revision = software.get("revision")
    if not revision:
        raise RevisionMismatch("the manifest records no source revision")
    _git(repo, "cat-file", "-e", f"{revision}^{{commit}}")
    head = _git(repo, "rev-parse", "HEAD")
    if subprocess.run(["git", "-C", str(repo), "merge-base", "--is-ancestor", revision, head]).returncode != 0:
        raise RevisionMismatch(f"manifest revision {revision} is not an ancestor of HEAD {head}")
    if not software.get("dirty"):
        evaluated, how = revision, "the run's tree was clean at the manifest revision"
    else:
        if software.get("untracked_files"):
            raise RevisionMismatch("the run had untracked files, which no commit can reproduce")
        recorded = software.get("uncommitted_diff_sha256")
        evaluated = None
        for commit in _git(repo, "rev-list", "--reverse", "--ancestry-path", f"{revision}..{head}").split():
            if hashlib.sha256(_git(repo, "diff", revision, commit, "--binary", binary=True)).hexdigest() == recorded:
                evaluated = commit
                break
        if evaluated is None:
            raise RevisionMismatch(f"no commit after {revision} reproduces the recorded uncommitted diff {recorded}")
        how = (f"sha256 of git diff {revision[:7]} {evaluated[:7]} --binary equals the manifest's "
               "uncommitted_diff_sha256")
    changed = _git(repo, "diff", "--name-only", evaluated, "--", *EVALUATION_SOURCES).split()
    changed += _git(repo, "ls-files", "--others", "--exclude-standard", "--", *EVALUATION_SOURCES).split()
    if changed:
        raise RevisionMismatch(f"evaluation sources changed since {evaluated}: {', '.join(sorted(changed))}")
    return dict(manifest_revision=revision, dirty=bool(software.get("dirty")),
                uncommitted_diff_sha256=software.get("uncommitted_diff_sha256"), evaluated_revision=evaluated,
                match=how, sources_unchanged=list(EVALUATION_SOURCES))


def verify_sources(evaluation, manifest):
    """Check every manifest-listed file's size and sha256; return path -> record."""
    records = {}
    for entry in manifest["files"]:
        path = Path(evaluation) / entry["path"]
        if not path.is_file():
            raise SourceMismatch(f"{path} is missing")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if len(data) != entry["bytes"] or digest != entry["sha256"]:
            raise SourceMismatch(f"{entry['path']} differs from the manifest (sha256 {digest})")
        records[entry["path"]] = dict(entry, path=entry["path"])
    return records


class Sources:
    """Verified outputs; tables are read only through here, so each table names a checked file."""

    def __init__(self, evaluation, records):
        self.evaluation, self.records, self.used, self._cache = Path(evaluation), records, set(), {}

    def table(self, relative):
        if relative not in self.records:
            raise SourceMismatch(f"{relative} is not listed in the manifest")
        self.used.add(relative)
        if relative not in self._cache:
            self._cache[relative] = pd.read_csv(self.evaluation / relative)
        return self._cache[relative]

    def data(self, relative):
        if relative not in self.records:
            raise SourceMismatch(f"{relative} is not listed in the manifest")
        self.used.add(relative)
        return (self.evaluation / relative).read_bytes()


# --- Panels -------------------------------------------------------------------------------------

@contextmanager
def _pyplot():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    yield plt
    plt.close("all")


def _png(figure):
    buffer = io.BytesIO()
    figure.savefig(buffer, dpi=80, format="png")
    return buffer.getvalue()


def _digest(images, labels):
    digest = hashlib.sha256()
    for name in labels:
        digest.update(np.ascontiguousarray(images[name]).tobytes())
    return digest.hexdigest()


class Scenes:
    """Regenerate displayed scenes and outputs; every array must match its manifest checksum."""

    def __init__(self, manifest, workdir):
        self.manifest, self.workdir = manifest, workdir
        p = manifest["parameters"]
        self.shape, self.count = tuple(p["shape_zyx"]), p["amplicons_per_fov"]
        self.runs = {(r.get("condition") or r.get("multi_fov"), r["dtype"], r["seed"], r["arm"], r.get("fov_id")):
                     r["output_sha256"] for r in manifest["runs"]}
        self.scenes = {(s["condition"], s["dtype"], s["seed"], None): s["image_sha256"] for s in manifest["scenes"]}
        self.scenes.update({(s["multi_fov"], s["dtype"], s["seed"], s["fov_id"]): s["image_sha256"]
                            for s in manifest["multi_fov_scenes"]})
        self.verified = []

    def _check_scene(self, key, scene):
        if scene.provenance["image_sha256"] != self.scenes.get(key):
            raise SourceMismatch(f"regenerated scene {key} differs from the manifest")

    def _check_output(self, key, images, labels, what):
        digest = _digest(images, labels)
        if digest != self.runs.get(key):
            raise SourceMismatch(f"regenerated {what} {key} differs from the manifest output_sha256")
        self.verified.append(dict(key="/".join(str(k) for k in key if k is not None), what=what, sha256=digest))

    def primary(self, condition, arms, dtype=PANEL_DTYPE, seed=PANEL_SEED, snapshot_of=None):
        """Scene data and the reference-round output of each arm (verified); snapshot_of: arm -> checked arm."""
        book, config = harness.scene_config(condition, dtype=dtype, seed=seed, shape=self.shape, count=self.count)
        scene, background, signal, truth = harness.generate(book, config)
        self._check_scene((condition, dtype, seed, None), scene)
        recipes = harness.arms(harness.background_radius(config))
        ref, outputs = book.round_labels[0], {}
        for arm in arms:
            fov = harness.preprocess(harness.make_fov(scene, book, config.FOV_id, self.workdir), recipes[arm], False)
            self._check_output((condition, dtype, seed, arm, None), fov.images, book.round_labels, "output")
            outputs[arm] = fov.images[ref]
            if snapshot_of and arm in snapshot_of:
                snapshots = {r: s[harness.SNAPSHOT] for r, s in fov.snapshots.items()}
                self._check_output((condition, dtype, seed, snapshot_of[arm], None), snapshots, book.round_labels,
                                   f"{arm} {harness.SNAPSHOT} snapshot (equal to the {snapshot_of[arm]} arm)")
                outputs[arm + ":snapshot"] = snapshots[ref]
        return dict(book=book, scene=scene, truth=truth, background=background[ref], signal=signal[ref],
                    outputs=outputs, fov_id=config.FOV_id, key=dict(condition=condition, dtype=dtype, seed=seed))

    def multi_fov(self, name, role, arms, dtype=PANEL_DTYPE, seed=PANEL_SEED):
        """One FOV of a multi-FOV set; supplied arms fit their statistics over the whole set first."""
        spec = harness.MULTI_FOV[name]
        fovs = {}
        for k, (fov_role, n, gain) in enumerate(zip(harness.FOV_ROLES[name], spec["counts"], spec["gains"])):
            fov_id = f"Position{k + 1:03d}"
            book, config = harness.scene_config(spec["base"], dtype=dtype, seed=seed, shape=self.shape,
                count=min(n, self.count), fov_id=fov_id, gain=gain,
                scene_key=f"controlled-development-v1/w233/{name}/{fov_id}")
            fovs[fov_id] = (*harness.generate(book, config), fov_role, config)
            self._check_scene((name, dtype, seed, fov_id), fovs[fov_id][0])
        radius = harness.background_radius(config)
        target = next(f for f, v in fovs.items() if v[4] == role)
        scene, background, signal, truth, _role, config = fovs[target]
        ref, outputs = book.round_labels[0], {}
        for arm in arms:
            recipe_name, fit = harness.MULTI_FOV_ARMS[arm]
            recipe = harness.arms(radius, fit, Path(self.workdir) / f"supplied-{name}-{arm}.json")[recipe_name]
            if fit == "supplied":
                harness.fit_supplied(recipe, fovs, book, self.workdir)
            fov = harness.preprocess(harness.make_fov(scene, book, target, self.workdir), recipe, False)
            self._check_output((name, dtype, seed, arm, target), fov.images, book.round_labels, "output")
            outputs[arm] = fov.images[ref]
        return dict(book=book, scene=scene, truth=truth, background=background[ref], signal=signal[ref],
                    outputs=outputs, fov_id=target, key=dict(condition=name, dtype=dtype, seed=seed, fov_id=target))


def dim_punctum(truth, signal, book, margin=(1, 6, 6)):
    """The truth punctum with the lowest noise-free peak on its reference-round channel, away from the edges."""
    shape, best = signal.shape[:3], None
    for row in truth.itertuples():
        center = tuple(int(round(v)) for v in (row.z, row.y, row.x))
        channel = book.color_to_channel[row.color_sequence[0]]
        inside = all(m <= c < n - m for c, n, m in zip(center, shape, margin))
        peak = float(signal[center + (channel,)])
        rank = (not inside, peak <= 0, peak)
        if best is None or rank < best[0]:
            best = (rank, dict(center=center, channel=channel, peak=peak, amplicon_id=str(row.amplicon_id)))
    if best is None:
        return dict(center=tuple(n // 2 for n in shape), channel=0, peak=None, amplicon_id=None)
    return best[1]


def _range(volume):
    low, high = float(volume.min()), float(np.percentile(volume, 99.9))
    return low, high if high > low else low + 1.0


def _truth_marks(truth, book, channel, axis, value):
    """(x, other) of truth centres on the channel within one voxel of the displayed plane."""
    marks = []
    for row in truth.itertuples():
        if book.color_to_channel[row.color_sequence[0]] != channel:
            continue
        position = dict(z=row.z, y=row.y, x=row.x)
        if abs(position[axis] - value) <= 1:
            marks.append((row.x, row.y if axis == "z" else row.z))
    return marks


def _show(axis, image, low, high, marks, dim, title, xz=False):
    axis.imshow(image, cmap="gray", vmin=low, vmax=high, interpolation="nearest",
                aspect=XZ_STRETCH if xz else 1, origin="upper")
    if marks:
        xs, ys = zip(*marks)
        axis.scatter(xs, ys, s=60, facecolors="none", edgecolors="cyan", linewidths=1)
    axis.scatter([dim[0]], [dim[1]], s=110, marker="s", facecolors="none", edgecolors="red", linewidths=1.5)
    axis.set_title(title, fontsize=8)
    axis.set_xlabel("x (voxel)", fontsize=7)
    axis.set_ylabel("z (voxel)" if xz else "y (voxel)", fontsize=7)
    axis.tick_params(labelsize=6)


def method_figure(data, before, after, labels, cutoffs):
    """Z slice and XZ view before/after, per-channel histograms and a dim-punctum line profile; PNG bytes.

    cutoffs maps "before"/"after" to per-channel noise cutoffs at threshold_value=5.
    Returns (png, facts) where facts states the slice, channel and display ranges.
    """
    book, truth = data["book"], data["truth"]
    images = dict(before=data["outputs"][before], after=data["outputs"][after])
    dim = dim_punctum(truth, data["signal"], book)
    (z0, y0, x0), c = dim["center"], dim["channel"]
    channels = book.channel_labels
    ranges = {side: _range(images[side][..., c]) for side in images}
    with _pyplot() as plt:
        figure = plt.figure(figsize=(13, 10.5), layout="constrained")
        grid = figure.add_gridspec(3, 4, height_ratios=(1.35, 1, 0.9))
        for k, side in enumerate(("before", "after")):
            volume = images[side][..., c]
            low, high = ranges[side]
            _show(figure.add_subplot(grid[0, k]), volume[z0], low, high, _truth_marks(truth, book, c, "z", z0),
                  (x0, y0), f"{side}: {labels[side]}\nZ slice z={z0}, {channels[c]}; display [{low:g}, {high:g}]")
            _show(figure.add_subplot(grid[0, 2 + k]), volume[:, y0, :], low, high,
                  _truth_marks(truth, book, c, "y", y0), (x0, z0),
                  f"{side}: XZ view y={y0}, {channels[c]}\ndisplay [{low:g}, {high:g}], Z drawn ×{XZ_STRETCH}",
                  xz=True)
        top = 256 if images["before"].dtype == np.uint8 and images["after"].dtype == np.uint8 else None
        for k, label in enumerate(channels):
            axis = figure.add_subplot(grid[1, k])
            for side, colour in (("before", "0.45"), ("after", "tab:orange")):
                values = images[side][..., k].ravel()
                bins = np.arange(0, top + 1) if top else 128
                axis.hist(values, bins=bins, histtype="step", color=colour, label=side, log=True)
                axis.axvline(cutoffs[side][k], color=colour, linestyle="--", linewidth=1)
            axis.set_title(f"{label}: noise cutoff (dashed)\nbefore {cutoffs['before'][k]:.4g}, "
                           f"after {cutoffs['after'][k]:.4g}", fontsize=8)
            axis.set_xlabel("intensity", fontsize=7)
            axis.set_ylabel("voxels (log)", fontsize=7)
            axis.tick_params(labelsize=6)
            if k == 0:
                axis.legend(fontsize=7)
        axis = figure.add_subplot(grid[2, :])
        for side, colour in (("before", "0.3"), ("after", "tab:orange")):
            axis.plot(images[side][z0, y0, :, c], color=colour, marker=".", label=f"{side}: {labels[side]}")
            axis.axhline(cutoffs[side][c], color=colour, linestyle="--", linewidth=1,
                         label=f"{side} noise cutoff {cutoffs[side][c]:.4g}")
        if before == "none":
            axis.plot(data["background"][z0, y0, :, c], color="tab:blue", linestyle=":",
                      label="background truth (input units)")
        axis.axvline(x0, color="red", linestyle=":", linewidth=1, label=f"dim punctum x={x0}")
        axis.set_title(f"Line profile along X at z={z0}, y={y0}, {channels[c]} through the dim punctum "
                       f"{dim['amplicon_id']} (noise-free peak {dim['peak']:.4g})", fontsize=9)
        axis.set_xlabel("x (voxel)", fontsize=8)
        axis.set_ylabel("intensity", fontsize=8)
        axis.legend(fontsize=7, ncol=3)
        png = _png(figure)
    return png, dict(z=z0, y=y0, x=x0, channel=channels[c], amplicon_id=dim["amplicon_id"], ranges=ranges)


def combined_figure(data, recipes):
    """Z slice and XZ view of each recipe on the combined preset; PNG bytes and facts."""
    book, truth = data["book"], data["truth"]
    dim = dim_punctum(truth, data["signal"], book)
    (z0, y0, x0), c = dim["center"], dim["channel"]
    ranges = {}
    with _pyplot() as plt:
        figure, axes = plt.subplots(2, len(recipes), figsize=(3.1 * len(recipes), 6.2), layout="constrained",
                                    height_ratios=(1.6, 1))
        for k, (arm, label) in enumerate(recipes):
            volume = data["outputs"][arm][..., c]
            low, high = ranges[arm] = _range(volume)
            _show(axes[0, k], volume[z0], low, high, _truth_marks(truth, book, c, "z", z0), (x0, y0),
                  f"{arm}: {label}\nz={z0}, {book.channel_labels[c]}; display [{low:g}, {high:g}]")
            _show(axes[1, k], volume[:, y0, :], low, high, _truth_marks(truth, book, c, "y", y0), (x0, z0),
                  f"{arm}: XZ y={y0}; display [{low:g}, {high:g}]", xz=True)
        png = _png(figure)
    return png, dict(z=z0, y=y0, x=x0, channel=book.channel_labels[c], ranges=ranges)


def panel_cutoffs(diagnostics, key, arm, book):
    """Per-channel noise cutoffs of an arm's reference round, read from tables/diagnostics.csv."""
    ref = book.round_labels[0]
    rows = diagnostics[(diagnostics["dtype"] == key["dtype"]) & (diagnostics["seed"] == key["seed"])
                       & (diagnostics["arm"] == arm) & (diagnostics["round"] == ref)]
    if "fov_id" in key:
        rows = rows[(rows["multi_fov"] == key["condition"]) & (rows["fov_id"] == key["fov_id"])]
    else:
        rows = rows[(rows["condition"] == key["condition"]) & rows["multi_fov"].isna()]
    values = rows.set_index("channel").noise_threshold
    if set(values.index) != set(book.channel_labels) or values.index.duplicated().any():
        raise SourceMismatch(f"diagnostics.csv has no unique cutoffs for {key} {arm}")
    return [float(values[label]) for label in book.channel_labels]


def _check_cutoffs(image, cutoffs, what):
    expected = harness._noise_thresholds(image, harness.DEFAULT_THRESHOLD)
    if not np.allclose(expected, cutoffs, rtol=1e-9, atol=1e-9):
        raise SourceMismatch(f"{what}: diagnostics.csv cutoffs {cutoffs} differ from the image's {expected}")


# --- HTML ------------------------------------------------------------------------------------------

CSS = """
body { font-family: system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif; margin: 24px auto; max-width: 1340px;
       color: #1d1d1f; line-height: 1.45; font-size: 14px; }
h1 { font-size: 24px; } h2 { font-size: 20px; border-bottom: 2px solid #ccc; padding-bottom: 4px; margin-top: 36px; }
h3 { font-size: 15px; margin-top: 18px; }
code, pre { font-family: ui-monospace, 'SFMono-Regular', Menlo, Consolas, monospace; font-size: 12px; }
code { background: #f1f1f4; padding: 0 3px; border-radius: 3px; word-break: break-all; }
pre { background: #f6f6f8; border: 1px solid #ddd; padding: 8px 10px; white-space: pre-wrap; word-break: break-all; }
table { border-collapse: collapse; margin: 6px 0 4px; font-size: 11.5px; }
th, td { border: 1px solid #ccc; padding: 2px 6px; text-align: left; vertical-align: top; }
th { background: #eef0f4; } td.num { text-align: right; font-variant-numeric: tabular-nums; }
td code { white-space: nowrap; word-break: normal; } span.range { color: #666; font-size: 10.5px; white-space: nowrap; }
.caption { font-size: 12px; color: #444; margin: 2px 0 14px; }
.flag-yes { background: #fde8e8; } .flag-no { background: #e8f6ea; }
.note { background: #fff8e1; border-left: 4px solid #e0a800; padding: 6px 10px; }
.grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
.grid figure { margin: 0; } .grid img { width: 100%; }
figure { margin: 8px 0; } figure img { max-width: 100%; border: 1px solid #ddd; }
figcaption { font-size: 12px; color: #444; }
dl.items dt { font-weight: 600; } dl.items dd { margin: 0 0 6px 16px; }
"""


def esc(value):
    return html.escape(str(value))


def code(value):
    return f"<code>{esc(value)}</code>"


def fmt(value, digits=4):
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "–"
    if isinstance(value, (bool, np.bool_)):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.{digits}g}"
    return str(value)


def spread(record, stem, digits=3):
    """'mean [min, max]' from a record's <stem>_mean/_min/_max columns."""
    mean, low, high = (record.get(f"{stem}_{s}") for s in ("mean", "min", "max"))
    if mean is None or (isinstance(mean, float) and np.isnan(mean)):
        return "–"
    return f"{fmt(mean, digits)}<br><span class=\"range\">[{fmt(low, digits)}, {fmt(high, digits)}]</span>"


def html_table(rows, columns, source, sources, note=""):
    """rows: records; columns: (header, function(record) -> str, numeric). Names the source file."""
    head = "".join(f"<th>{header}</th>" for header, _f, _n in columns)
    body = []
    for record in rows:
        cells = []
        for _header, function, numeric in columns:
            value = function(record)
            classes = ["num"] if numeric else []
            if value in ("flagged",):
                classes.append("flag-yes")
            elif value in ("not flagged",):
                classes.append("flag-no")
            cells.append(f"<td class=\"{' '.join(classes)}\">{value}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    digest = sources.records[source]["sha256"]
    caption = f"Source: {code(source)} (sha256 {code(digest[:16])}…, full checksum in the source list)."
    return (f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"
            f"<p class=\"caption\">{caption} {note}</p>")


def image(png, alt):
    return f"<img alt=\"{esc(alt)}\" src=\"data:image/png;base64,{base64.b64encode(png).decode()}\">"


def _records(frame):
    return frame.astype(object).where(frame.notna(), None).to_dict("records")


def flags_section(method, sources):
    flags = sources.table("tables/low_benefit_flags.csv")
    part = flags[flags.method == method]
    rows = _records(part)
    columns = [
        ("comparison", lambda r: code(r["comparison"]), False),
        ("dtype", lambda r: code(r["dtype"]), False),
        ("level", lambda r: esc(r["level"]), False),
        ("condition", lambda r: esc(r["condition"]), False),
        ("low-benefit flag", lambda r: "flagged" if r["low_benefit"] else "not flagged", False),
        ("clean harm", lambda r: fmt(r["clean_harm"]), False),
        ("max-F1 Δ", lambda r: fmt(r["max_f1_delta"]), True),
        ("max-F1 seed range", lambda r: fmt(r["max_f1_seed_range"]), True),
        ("correct-decode Δ", lambda r: fmt(r["correct_fraction_delta"]), True),
        ("correct-decode seed range", lambda r: fmt(r["correct_fraction_seed_range"]), True),
        ("threshold (provisional)",
         lambda r: f"{fmt(r['threshold_points'])} ({'provisional' if r['provisional'] else 'final'})", True),
        ("reason", lambda r: esc(r["reason"] or ""), False),
    ]
    return (f"<h3 id=\"{method}-flag\">Low-benefit flag (provisional threshold of 2 percentage points)</h3>"
            "<p>A comparison is flagged when no targeted condition improves max-F1 or the correct-decode fraction "
            "by at least max(held-out seed range, 0.02), or when the clean condition worsens by more than its seed "
            "range. The 0.02 threshold is <strong>provisional</strong>; the flag informs review and does not remove "
            "a method.</p>" + html_table(rows, columns, "tables/low_benefit_flags.csv", sources))


def metrics_section(method, sources):
    comparisons = sources.table("tables/comparisons.csv")
    rows = _records(comparisons[comparisons.method == method])
    arms = lambda r: f"{code(r['before'])} → {code(r['after'])}"  # noqa: E731
    key = [("comparison: before → after", lambda r: f"{code(r['comparison'])}:<br>{arms(r)}", False),
           ("condition (role)", lambda r: f"{code(r['condition'])} ({esc(r['role'])})", False),
           ("dtype", lambda r: code(r["dtype"]), False)]
    downstream = key + [
        ("max-F1 before", lambda r: spread(r, "max_f1_before"), True),
        ("max-F1 after", lambda r: spread(r, "max_f1_after"), True),
        ("max-F1 Δ", lambda r: spread(r, "max_f1_delta"), True),
        ("AUPRC Δ", lambda r: spread(r, "auprc_delta"), True),
        ("correct-decode before → after (dev-selected threshold)",
         lambda r: f"{fmt(r['correct_fraction_before_mean'])} → {fmt(r['correct_fraction_after_mean'])}", True),
        ("correct-decode Δ", lambda r: spread(r, "correct_fraction_delta"), True),
        ("F1 at 5 Δ", lambda r: spread(r, "f1_t5_delta"), True),
        ("wrong-gene reads Δ", lambda r: spread(r, "reads_wrong_gene_delta"), True),
        ("false-detection reads Δ", lambda r: spread(r, "reads_false_detection_delta"), True),
    ]
    direct_metrics = [("bg_rmse", "background RMSE"), ("bg_bias", "background bias"),
                      ("contrast_median", "puncta contrast"), ("intensity_cv", "true-spot intensity CV"),
                      ("clipped_fraction", "clipped fraction"), ("saturated_fraction", "saturated fraction"),
                      ("color_call_agreement", "colour-call agreement"),
                      ("per_fov_correct_fraction_range", "per-FOV correct-decode range")]
    present = [(stem, label) for stem, label in direct_metrics
               if f"{stem}_before_mean" in comparisons and any(r.get(f"{stem}_before_mean") is not None
                                                               or r.get(f"{stem}_after_mean") is not None
                                                               for r in rows)]
    before_after = lambda stem: (lambda r: f"{fmt(r.get(f'{stem}_before_mean'))} → "  # noqa: E731
                                           f"{fmt(r.get(f'{stem}_after_mean'))}")
    direct = key + [(label, before_after(stem), True) for stem, label in present]
    mad = key + [(label, before_after(stem), True) for stem, label in
                 (("zero_fraction", "zero fraction"), ("median", "median"), ("mad", "MAD"),
                  ("noise_threshold", "noise cutoff at 5"), ("mad_zero", "fraction of channels with MAD 0"))]
    return (
        f"<h3 id=\"{method}-metrics\">Downstream metrics</h3><p>Mean [min, max] across the held-out seeds 100, "
        "101 and 102. Max-F1 and AUPRC use the threshold grid; reads use the threshold selected on development "
        "seeds, and “at 5” uses <code>threshold_value=5.0</code>.</p>"
        + html_table(rows, downstream, "tables/comparisons.csv", sources)
        + f"<h3 id=\"{method}-direct\">Direct signal checks</h3><p>Before → after means across held-out seeds, "
          "reference round, averaged over channels within each seed. Background error is measured at the stage "
          "named by <code>bg_stage</code> in <code>tables/direct_metrics.csv</code>.</p>"
        + html_table(rows, direct, "tables/comparisons.csv", sources)
        + f"<h3 id=\"{method}-mad\">MAD diagnostics</h3><p>Reference-round per-channel diagnostics, averaged over "
          "channels within each seed, then over held-out seeds (before → after).</p>"
        + html_table(rows, mad, "tables/comparisons.csv", sources))


def method_section(method, spec, figures, sources):
    panel = spec["panel"]
    figure = figures[method]
    facts = figure["facts"]
    ranges = "; ".join(f"{side} [{fmt(lo)}, {fmt(hi)}]" for side, (lo, hi) in facts["ranges"].items())
    where = f"FOV {code(figure['fov_id'])}, " if "fov_role" in panel else ""
    caption = (f"{code(panel['before'])} (before) and {code(panel['after'])} (after) on {code(panel['condition'])}, "
               f"{code(PANEL_DTYPE)}, held-out seed {PANEL_SEED}, reference round, {where}channel "
               f"{code(facts['channel'])}. Display ranges (linear, grey levels): {ranges}. Cyan circles are "
               f"ground-truth puncta on this channel within one voxel of the plane; the red square is the dim "
               f"punctum {code(facts['amplicon_id'])} of the line profile. Histograms show all voxels of each "
               f"channel with the noise cutoff (<code>threshold_value=5</code>) from "
               f"<code>tables/diagnostics.csv</code>.")
    note = f"<p class=\"note\">{esc(spec['panel_note'])}</p>" if spec.get("panel_note") else ""
    harm, combined = (("mf_density_clean", "mf_density") if method == "sample_level_fitting"
                      else ("clean", "combined"))
    targeted = sorted({c for entry in _comparison_specs() if entry["method"] == method for c in entry["targeted"]},
                      key=list(CONDITIONS).index)
    intro = "".join(f"<li>{code(c)}: {esc(CONDITIONS[c])}</li>" for c in targeted)
    return (f"<section id=\"{method}\"><h2>{esc(spec['title'])}</h2>"
            f"<dl class=\"items\"><dt>Problem</dt><dd>{esc(spec['problem'])}</dd>"
            f"<dt>Cause</dt><dd>{esc(spec['cause'])}</dd>"
            f"<dt>Targeted condition</dt><dd><ul>{intro}</ul>Harm check: {code(harm)}; "
            f"combined condition: {code(combined)}.</dd>"
            f"<dt>Why the benefit may be small</dt><dd>{esc(spec['why_small'])}</dd></dl>"
            f"<h3>Before/after panels</h3>{note}<figure>{image(figure['png'], spec['title'])}"
            f"<figcaption>{caption}</figcaption></figure>"
            + flags_section(method, sources) + metrics_section(method, sources) + "</section>")


def _comparison_specs():
    return [dict(method=m, comparison=c, before=b, after=a, targeted=list(t))
            for m, c, b, a, t in harness.COMPARISONS + harness.SAMPLE_COMPARISONS]


def operating_points_section(sources):
    points = sources.table("tables/operating_points.csv")
    recipes = [arm for arm in harness.PRIMARY_ARMS if arm == "none" or arm.startswith(("r1", "r2"))]
    conditions = list(harness.PRIMARY) + ["combined_geometry"]
    parts = []
    for dtype in harness.DTYPES:
        part = points[(points.dtype == dtype) & points.arm.isin(recipes) & points.condition.isin(conditions)]
        part = part.assign(_c=part.condition.map(conditions.index), _a=part.arm.map(recipes.index)).sort_values(
            ["_c", "_a", "operating_point"], ascending=[True, True, False])
        columns = [("condition", lambda r: code(r["condition"]), False), ("arm", lambda r: code(r["arm"]), False),
                   ("operating point", lambda r: esc(r["operating_point"]), False),
                   ("threshold_value", lambda r: fmt(r["threshold"]), True),
                   ("precision", lambda r: spread(r, "precision"), True),
                   ("recall", lambda r: spread(r, "recall"), True), ("F1", lambda r: spread(r, "f1"), True),
                   ("correct-decode fraction", lambda r: spread(r, "correct_fraction"), True),
                   ("wrong-gene reads", lambda r: spread(r, "reads_wrong_gene", 3), True),
                   ("false-detection reads", lambda r: spread(r, "reads_false_detection", 3), True),
                   ("localization error (voxel)", lambda r: spread(r, "localization_error", 3), True)]
        parts.append(f"<section id=\"operating-points-{dtype}\"><h2>Operating points, {dtype}</h2>"
                     f"<p>Each recipe on the primary conditions and the coupled case, at the threshold with the "
                     f"highest mean F1 on development seeds (<code>dev_max_f1</code>) and at "
                     f"<code>threshold_value=5.0</code> (<code>default</code>); mean [min, max] over held-out "
                     f"seeds. Other conditions and arms are in the same file.</p>"
                     + html_table(_records(part), columns, "tables/operating_points.csv", sources) + "</section>")
    return "".join(parts)


def pr_section(manifest, sources):
    cells = []
    for condition in manifest["conditions"]:
        for dtype in harness.DTYPES:
            relative = f"curves/pr_{condition}_{dtype}.png"
            if relative not in sources.records:
                continue
            cells.append(f"<figure>{image(sources.data(relative), relative)}<figcaption>{code(relative)}: "
                         f"{esc(CONDITIONS[condition])}</figcaption></figure>")
    return ("<section id=\"pr-curves\"><h2>Precision–recall curves</h2><p>Mean precision and recall over held-out "
            "seeds at each <code>threshold_value</code> in {2, 3, 4, 5, 6, 8, 10, 12, 15}, one line per arm, as "
            "saved by the evaluation. Per-seed values are in <code>curves/pr_curves.csv</code>; max-F1 and AUPRC "
            "per seed are in <code>tables/recipe_per_seed.csv</code>.</p>"
            f"<div class=\"grid\">{''.join(cells)}</div></section>")


def multi_fov_section(sources):
    spread_table = sources.table("tables/multi_fov_spread.csv")
    part = spread_table[spread_table.operating_point == "dev_max_f1"]
    order = list(harness.MULTI_FOV_ARMS)
    part = part.assign(_m=part.multi_fov.map(list(harness.MULTI_FOV).index), _a=part.arm.map(order.index),
                       _d=part.dtype.map(list(harness.DTYPES).index))
    part = part.sort_values(["_m", "_d", "_a"])
    roles = lambda r: " / ".join(fmt(r.get(f"correct_fraction_{role}_mean"), 3)  # noqa: E731
                                 for role in harness.FOV_ROLES[r["multi_fov"]])
    columns = [("set", lambda r: code(r["multi_fov"]), False), ("dtype", lambda r: code(r["dtype"]), False),
               ("arm", lambda r: code(r["arm"]), False),
               ("correct-decode per FOV role (mean)", roles, True),
               ("correct-decode range across FOVs", lambda r: spread(r, "correct_fraction_fov_range", 3), True),
               ("decoding accuracy range across FOVs", lambda r: spread(r, "decoding_accuracy_fov_range", 3), True),
               ("false-positive rate range across FOVs", lambda r: spread(r, "false_positive_rate_fov_range", 3),
                True),
               ("F1 range across FOVs", lambda r: spread(r, "f1_fov_range", 3), True)]
    intro = "".join(f"<li>{code(n)}: {esc(CONDITIONS[n])} Roles: {esc(', '.join(harness.FOV_ROLES[n]))}.</li>"
                    for n in harness.MULTI_FOV)
    return ("<section id=\"multi-fov\"><h2>Multi-FOV comparison</h2><ul>" + intro + "</ul>"
            "<p>Per-FOV (<code>fit=\"fov\"</code>) against sample-level (<code>fit=\"supplied\"</code>, arms ending "
            "in <code>_sample</code>) fitting at the pooled development-selected threshold. Ranges are the spread "
            "across the three FOVs, as mean [min, max] over held-out seeds; per-role values are held-out means in "
            "role order.</p>" + html_table(_records(part), columns, "tables/multi_fov_spread.csv", sources)
            + "</section>")


def mad_zero_section(manifest, sources):
    zero = pd.DataFrame(manifest["mad_zero_reference_round"])
    grouped = zero.groupby(["condition", "dtype"]).arm.apply(lambda a: ", ".join(sorted(a))).reset_index()
    columns = [("condition", lambda r: code(r["condition"]), False), ("dtype", lambda r: code(r["dtype"]), False),
               ("arms with a reference-round channel at MAD 0", lambda r: esc(r["arm"]), False)]
    rows = _records(grouped)
    body = html_table(rows, columns, "manifest.json", sources,
                      note="From the manifest's <code>mad_zero_reference_round</code> list.")
    return ("<section id=\"mad-zero\"><h2>MAD diagnostics: where MAD reaches 0</h2><p>When MAD is 0 the noise "
            "cutoff equals the median, so the <code>threshold_value</code> sweep does not change detections. The "
            "noise-free presets reach MAD 0 in every arm; in the noisy conditions only the reconstruction arms "
            "do.</p>" + body + "</section>")


def sources_section(manifest_path, manifest_sha, sources):
    rows = [dict(path="manifest.json", bytes=manifest_path.stat().st_size, sha256=manifest_sha, used=True)]
    rows += [dict(r, used=path in sources.used) for path, r in sorted(sources.records.items())
             if path != "manifest.json"]
    head = "<tr><th>file</th><th>bytes</th><th>sha256</th><th>verified</th><th>used in this report</th></tr>"
    body = "".join(f"<tr><td>{code(r['path'])}</td><td class=\"num\">{r['bytes']}</td><td>{code(r['sha256'])}</td>"
                   f"<td>{'manifest itself' if r['path'] == 'manifest.json' else 'size and sha256 match manifest'}"
                   f"</td><td>{fmt(r['used'])}</td></tr>" for r in rows)
    return (f"<section id=\"sources\"><h2>Sources and checksums</h2><p>All files are under "
            f"{code(sources.evaluation)}. Every table in this report is copied from one of these files and "
            f"rounded for display only (mean [min, max] to 3 significant digits, other values to 4); the "
            f"renderer read each file only after checking its size and sha256 against the manifest.</p>"
            f"<table><thead>{head}</thead><tbody>{body}</tbody></table></section>")


def panel_checks_section(verified):
    rows = "".join(f"<tr><td>{code(v['key'])}</td><td>{esc(v['what'])}</td><td>{code(v['sha256'])}</td></tr>"
                   for v in verified)
    return ("<section id=\"panel-checks\"><h2>Panel image checks</h2><p>The evaluation saves checksums, not "
            "images. The renderer regenerated each displayed scene from its recorded configuration and "
            "preprocessed it with the recorded recipe; every scene's image checksums and every output below equal "
            "the manifest's <code>scenes</code>, <code>multi_fov_scenes</code> and <code>runs</code> records "
            "(sha256 over the rounds in codebook order). Histogram cutoffs come from "
            "<code>tables/diagnostics.csv</code> and equal the cutoffs computed from these images.</p>"
            "<table><thead><tr><th>condition/dtype/seed/arm[/FOV]</th><th>array</th><th>sha256 (matches manifest)"
            f"</th></tr></thead><tbody>{rows}</tbody></table></section>")


def render(evaluation, output, *, repo=ROOT, command=None):
    """Check the saved outputs and write the report; returns a summary dict."""
    evaluation, output = Path(evaluation).resolve(), Path(output)
    manifest_path = evaluation / "manifest.json"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    manifest = json.loads(manifest_bytes)
    if manifest.get("schema") != harness.SCHEMA:
        raise SourceMismatch(f"unexpected manifest schema {manifest.get('schema')}")
    identity = check_revision(manifest["software"], repo)
    records = verify_sources(evaluation, manifest)
    records["manifest.json"] = dict(path="manifest.json", bytes=len(manifest_bytes), sha256=manifest_sha)
    sources = Sources(evaluation, records)
    sources.used.add("manifest.json")
    rendering = harness._revision()
    diagnostics = sources.table("tables/diagnostics.csv")
    figures = {}
    with tempfile.TemporaryDirectory(prefix="w234-") as workdir:
        scenes = Scenes(manifest, workdir)
        for method, spec in METHODS.items():
            panel = spec["panel"]
            before, after = panel["before"], panel["after"]
            if "fov_role" in panel:
                data = scenes.multi_fov(panel["condition"], panel["fov_role"], (before, after))
            elif panel.get("snapshot"):
                data = scenes.primary(panel["condition"], (before, after), snapshot_of={after: "scalar"})
                after = after + ":snapshot"
            else:
                data = scenes.primary(panel["condition"], (before, after))
            cutoffs = {}
            for side, arm, record_arm in (("before", before, panel["before"]),
                                          ("after", after, "scalar" if panel.get("snapshot") else panel["after"])):
                cutoffs[side] = panel_cutoffs(diagnostics, data["key"], record_arm, data["book"])
                _check_cutoffs(data["outputs"][arm], cutoffs[side], f"{method} {side}")
            labels = dict(before=panel["before"], after=panel["after"] + (" (bg_corrected snapshot)"
                                                                          if panel.get("snapshot") else ""))
            png, facts = method_figure(data, before, after, labels, cutoffs)
            figures[method] = dict(png=png, facts=facts, fov_id=data["fov_id"])
        combined = scenes.primary("combined", [arm for arm, _ in COMBINED_RECIPES])
        combined_png, combined_facts = combined_figure(combined, COMBINED_RECIPES)
        verified = scenes.verified
    software = manifest["software"]
    run_command = " ".join(["uv run python"] + [a if " " not in a else json.dumps(a) for a in software["command"]])
    render_command = command or ("uv run python ../../benchmarks/preprocessing_report.py "
                                 f"--evaluation {evaluation} --output {output.resolve()}")
    time_v = manifest.get("time_v", {})
    rendering_state = (f"HEAD {code(rendering['revision'])}, " + (
        f"with uncommitted changes (sha256 of <code>git diff HEAD --binary</code> plus untracked files: "
        f"{code(rendering['uncommitted_diff_sha256'])})" if rendering["dirty"] else "clean tree"))
    ranges = "; ".join(f"{code(a)} [{fmt(lo)}, {fmt(hi)}]" for a, (lo, hi) in combined_facts["ranges"].items())
    toc = "".join(f"<li><a href=\"#{m}\">{esc(s['title'])}</a></li>" for m, s in METHODS.items())
    body = f"""
<h1>§2.5 preprocessing: inspection report (task group 6)</h1>
<p class="note"><strong>Development evidence for human review.</strong> Rendered from the saved task-group-5
(W-233) outputs on uncalibrated synthetic presets. It is not scientific acceptance, gives no recommended default,
and its low-benefit threshold is provisional.</p>
<section id="identity"><h2>Identity and checks</h2>
<table><tbody>
<tr><th>Code revision (rendering)</th><td>{rendering_state}</td></tr>
<tr><th>Evaluated code revision</th><td>{code(identity['evaluated_revision'])}</td></tr>
<tr><th>Manifest revision</th><td>{code(identity['manifest_revision'])}, dirty={fmt(identity['dirty'])},
 <code>uncommitted_diff_sha256</code> {code(identity['uncommitted_diff_sha256'])}</td></tr>
<tr><th>Revision check</th><td>Passed: the manifest revision is an ancestor of HEAD; {esc(identity['match'])};
 {', '.join(code(s) for s in identity['sources_unchanged'])} are unchanged from
 {code(identity['evaluated_revision'][:7])} to the rendering tree.</td></tr>
<tr><th>Task-group-5 manifest</th><td>{code(manifest_path)}</td></tr>
<tr><th>Manifest sha256</th><td>{code(manifest_sha)}</td></tr>
<tr><th>Saved outputs</th><td>{len(sources.records) - 1} files listed in the manifest; every size and sha256
 verified before rendering (see <a href="#sources">Sources and checksums</a>).</td></tr>
<tr><th>Evaluation run</th><td>wall time {fmt(time_v.get('wall_seconds'))} s, maximum RSS
 {fmt(time_v.get('max_rss_kib'))} KiB, exit status {fmt(time_v.get('exit_status'))}; reductions:
 {esc('; '.join(manifest.get('reductions', [])))}</td></tr>
<tr><th>Report schema</th><td>{code(SCHEMA)}</td></tr>
</tbody></table>
<p class="caption">Source: {code('manifest.json')} (<code>software</code>, <code>time_v</code>,
<code>reductions</code>) and the git history of this checkout.</p>
<h3>Commands</h3>
<p>Evaluation (task group 5, from <code>src/python</code>, as recorded in the manifest):</p>
<pre><code>{esc(run_command)}</code></pre>
<p>This report (from <code>src/python</code>):</p>
<pre><code>{esc(render_command)}</code></pre>
</section>
<section id="limitations"><h2>Limitations</h2><ul>
{''.join(f'<li>{esc(item)}</li>' for item in LIMITATIONS)}
{''.join(f'<li>Fixture gap: {esc(item)}</li>' for item in manifest.get('fixture_gaps', []))}
</ul></section>
<section id="conditions"><h2>Conditions</h2>
<p>Each condition starts from <code>development_scene_preset("clean")</code> and takes the named groups of
<code>FormedSceneConfig</code> fields from other development presets; scenes are
{code('×'.join(map(str, manifest['parameters']['shape_zyx'])))} voxels (ZYX) with
{manifest['parameters']['amplicons_per_fov']} amplicons, in <code>uint8</code> and <code>uint16</code>, with
development seeds {{0, 1, 2}} and held-out seeds {{100, 101, 102}}.</p>
<ul>{''.join(f'<li>{code(c)}: {esc(t)}</li>' for c, t in CONDITIONS.items())}</ul>
<h3>Reading the panels</h3>
<p>Each method section shows the isolated comparison on its first targeted condition in <code>{PANEL_DTYPE}</code>,
held-out seed {PANEL_SEED}, reference round. The Z slice and the XZ view pass through the dimmest noise-free
punctum away from the edges, on that punctum's channel; before and after each state their linear display range.
Histograms show every voxel per channel on a log scale with the noise cutoff at <code>threshold_value=5</code>.
XZ views are drawn with Z stretched ×{XZ_STRETCH} because the presets record no voxel spacing.</p>
<ol>{toc}<li><a href="#combined-recipes">Recipes on the combined preset</a></li>
<li><a href="#pr-curves">Precision–recall curves</a></li>
<li><a href="#operating-points-uint8">Operating points</a></li>
<li><a href="#mad-zero">MAD diagnostics</a></li><li><a href="#multi-fov">Multi-FOV comparison</a></li>
<li><a href="#panel-checks">Panel image checks</a></li><li><a href="#sources">Sources and checksums</a></li></ol>
</section>
{''.join(method_section(m, s, figures, sources) for m, s in METHODS.items())}
<section id="combined-recipes"><h2>Recipes on the combined preset</h2>
<p>{esc(CONDITIONS['combined'])} Each recipe's detection image, <code>{PANEL_DTYPE}</code>, held-out seed
{PANEL_SEED}, reference round, channel {code(combined_facts['channel'])}, Z slice z={combined_facts['z']} and XZ view
y={combined_facts['y']}.</p>
<figure>{image(combined_png, 'recipes on the combined preset')}<figcaption>Display ranges (linear): {ranges}.
Cyan circles are ground-truth puncta on this channel within one voxel of the plane; the red square is the dim
punctum. The <code>_xsrc</code> variants have the same detection images as their recipes.</figcaption></figure>
</section>
{pr_section(manifest, sources)}
{operating_points_section(sources)}
{mad_zero_section(manifest, sources)}
{multi_fov_section(sources)}
{panel_checks_section(verified)}
{sources_section(manifest_path, manifest_sha, sources)}
"""
    document = (f"<!DOCTYPE html>\n<html lang=\"en\"><head><meta charset=\"utf-8\">"
                f"<title>§2.5 preprocessing inspection report</title><style>{CSS}</style></head>"
                f"<body>{body}</body></html>\n")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(document.encode())  # one full-file write (network mounts)
    return dict(output=str(output), bytes=len(document.encode()), manifest_sha256=manifest_sha,
                identity=identity, rendering=rendering, verified_arrays=len(verified),
                sources_used=sorted(sources.used))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--evaluation", type=Path, required=True, help="W-233 evaluation directory (manifest.json)")
    parser.add_argument("--output", type=Path, required=True, help="report HTML file, outside Git")
    args = parser.parse_args()
    command = " ".join(["uv run python", os.path.relpath(Path(sys.argv[0]).resolve(), Path.cwd())]
                       + [f"--evaluation {args.evaluation}", f"--output {args.output}"])
    summary = render(args.evaluation, args.output, command=command)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
